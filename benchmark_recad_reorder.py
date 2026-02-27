#!/usr/bin/env python3
"""
Rec-AD Style Graph-Based Community Detection Reordering for DLRM Embedding Tables.

Implements the Rec-AD approach: build a co-occurrence graph of cold embedding indices
that appear together in the same batch, run Louvain community detection to find clusters,
then place community members contiguously in H.265 video frames.

Compares against the existing batch-affinity reordering method.

Pipeline:
  Phase 1: Load existing profiling/hot-cold data
  Phase 2: Build co-occurrence graph + Louvain community detection per table
  Phase 3: Re-encode H.265 per-frame files with new reordering
  Phase 4: Run inference benchmarks (same configs as codec_ondemand_benchmark.py)
  Phase 5: Compare results side-by-side
"""

import os, sys, time, json, gc, subprocess
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms.community import louvain_communities
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import psutil

# C++ extension
try:
    import compressed_emb as _C
    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import av

# ============================================================
# CONFIGURATION (same as codec_ondemand_benchmark.py)
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
DATA_FILE = os.path.expanduser("~/input/train.txt")
PROCESSED_DATA = os.path.expanduser("~/input/kaggleAdDisplayChallenge_processed.npz")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 2048
EMB_DIM = 16
HOT_COVERAGE = 0.80
LARGE_TABLE_THRESHOLD = 50000
H265_CRF = 0  # lossless

RESOLUTIONS = {
    '480p':  (640, 480),
    '1080p': (1920, 1080),
    '4K':    (3840, 2160),
}

TILE_H = 4
TILE_W = 4

RESULTS_DIR = "results"
PROFILING_DIR = os.path.join(RESULTS_DIR, "profiling")
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR_ORIG = os.path.join(RESULTS_DIR, "reorder")       # existing batch-affinity
REORDER_DIR_RECAD = os.path.join(RESULTS_DIR, "reorder_recad") # new Rec-AD
ONDEMAND_DIR_ORIG = os.path.join(RESULTS_DIR, "ondemand")      # existing encoded frames
ONDEMAND_DIR_RECAD = os.path.join(RESULTS_DIR, "ondemand_recad")
LOG_FILE = "logs/recad_benchmark.log"

for d in [REORDER_DIR_RECAD, ONDEMAND_DIR_RECAD, "logs"]:
    os.makedirs(d, exist_ok=True)

log_fh = open(LOG_FILE, 'w')

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()

def get_rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def drop_caches():
    try:
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except Exception:
        pass


# ============================================================
# MODEL + DATA LOADING (reuse from codec_ondemand_benchmark)
# ============================================================
def create_args():
    class Args: pass
    a = Args()
    a.arch_sparse_feature_size = ARCH_SPARSE_FEATURE_SIZE
    a.arch_mlp_bot = ARCH_MLP_BOT; a.arch_mlp_top = ARCH_MLP_TOP
    a.arch_interaction_op = "dot"; a.arch_interaction_itself = False
    a.data_generation = "dataset"; a.data_set = "kaggle"
    a.raw_data_file = DATA_FILE; a.processed_data_file = PROCESSED_DATA
    a.loss_function = "bce"; a.max_ind_range = -1
    a.test_mini_batch_size = TEST_BATCH_SIZE; a.test_num_workers = 0; a.num_workers = 0
    a.mlperf_logging = False; a.memory_map = False; a.data_randomize = "total"
    a.data_trace_enable_padding = False; a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10; a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 128; a.round_targets = True
    a.mlperf_bin_loader = False; a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False
    return a

def load_model_and_data():
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net
    args = create_args()
    log("Loading dataset...")
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    ln_emb = np.array(train_data.counts)
    m_spa = args.arch_sparse_feature_size
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + args.arch_mlp_top, dtype=int, sep="-")
    dlrm = DLRM_Net(m_spa, ln_emb, ln_bot, ln_top,
                     arch_interaction_op="dot", arch_interaction_itself=False,
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")
    log("Loading model checkpoint...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    log(f"Model loaded. {len(ln_emb)} tables, emb_dim={m_spa}")
    return dlrm, test_ld, train_ld, ln_emb


# ============================================================
# QUANTIZATION (same as codec_ondemand_benchmark.py)
# ============================================================
def quantize_table(w):
    """Global quantization: single scale/zero-point for entire table."""
    mn = w.min().item()
    mx = w.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


# ============================================================
# TILING + H.265 ENCODING (same as codec_ondemand_benchmark.py)
# ============================================================
def rows_to_tiled_frame(emb_rows, width, height):
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    tiles = emb_rows[:rows_per_frame].reshape(tiles_per_col, tiles_per_row, TILE_H, TILE_W)
    frame = tiles.transpose(0, 2, 1, 3).reshape(height, width)
    return frame

def tiled_frame_to_rows(frame, width, height):
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    grid = frame.reshape(tiles_per_col, TILE_H, tiles_per_row, TILE_W)
    rows = grid.transpose(0, 2, 1, 3).reshape(rows_per_frame, EMB_DIM)
    return rows

def encode_h265_perframe(q_np, width, height, crf=0, output_dir=None, table_id=0):
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    num_rows = q_np.shape[0]
    num_frames = max(1, (num_rows + rows_per_frame - 1) // rows_per_frame)

    padded_rows = num_frames * rows_per_frame
    padded = np.zeros((padded_rows, EMB_DIM), dtype=np.uint8)
    padded[:num_rows] = q_np

    frame_dir = os.path.join(output_dir, f'table_{table_id}')
    os.makedirs(frame_dir, exist_ok=True)

    t0 = time.time()
    total_compressed = 0

    for i in range(num_frames):
        frame_rows = padded[i * rows_per_frame:(i + 1) * rows_per_frame]
        frame_2d = rows_to_tiled_frame(frame_rows, width, height)
        frame_path = os.path.join(frame_dir, f'frame_{i:05d}.h265')

        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', f'{width}x{height}',
            '-r', '1',
            '-i', 'pipe:0',
            '-c:v', 'libx265',
            '-preset', 'ultrafast',
            '-pix_fmt', 'gray',
            '-x265-params',
            f'keyint=1:min-keyint=1:{"lossless=1" if crf == 0 else f"crf={crf}"}:log-level=error',
            '-f', 'matroska',
            frame_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.stdin.write(frame_2d.tobytes())
        proc.stdin.close()
        proc.wait()

        total_compressed += os.path.getsize(frame_path)

    encode_time = time.time() - t0
    raw_bytes = num_rows * EMB_DIM
    ratio = raw_bytes / total_compressed if total_compressed > 0 else 0

    log(f"    H.265 encode ({width}x{height}): {num_frames} frames, "
        f"{raw_bytes/1024/1024:.1f}MB -> {total_compressed/1024/1024:.1f}MB "
        f"({ratio:.2f}x), {encode_time:.1f}s")

    return num_frames, frame_dir, total_compressed, encode_time, rows_per_frame


# ============================================================
# REC-AD: CO-OCCURRENCE GRAPH + LOUVAIN COMMUNITY DETECTION
# ============================================================
def build_cooccurrence_reorder(batch_access_log, cold_indices_np, ln_emb_t, table_id,
                                min_cooccurrence=1, resolution=1.0):
    """
    Build a co-occurrence graph of cold indices and find communities via Louvain.

    Rec-AD approach: indices co-occurring in the same batch get graph edges,
    then community detection groups tightly co-accessed indices together.

    For very large/sparse graphs, falls back to scipy connected components
    + batch-affinity within each component (faster than full Louvain).

    Returns:
        reorder: np.int64 array — permutation of cold sequential indices
    """
    n_cold = len(cold_indices_np)
    num_batches = len(batch_access_log)

    # Step 1: Map original indices to cold sequential positions
    cold_to_seq = np.full(ln_emb_t, -1, dtype=np.int64)
    cold_to_seq[cold_indices_np] = np.arange(n_cold, dtype=np.int64)

    # Step 2: Build batch membership lists (only cold indices)
    log(f"    Building batch membership lists...")
    cold_freq = np.zeros(n_cold, dtype=np.int64)
    cold_first_batch = np.full(n_cold, -1, dtype=np.int64)
    batch_cold_seqs = []
    num_sample_batches = min(num_batches, 5000)

    for bi in range(num_sample_batches):
        accessed = batch_access_log[bi].get(table_id, set())
        seqs = []
        for idx in accessed:
            if idx < ln_emb_t:
                seq = cold_to_seq[idx]
                if seq >= 0:
                    cold_freq[seq] += 1
                    if cold_first_batch[seq] == -1:
                        cold_first_batch[seq] = bi
                    seqs.append(seq)
        batch_cold_seqs.append(np.array(seqs, dtype=np.int64))

    active_mask = cold_freq > 0
    n_active = int(active_mask.sum())
    log(f"    {n_active:,} active cold indices (of {n_cold:,})")

    if n_active == 0:
        return np.arange(n_cold, dtype=np.int64)

    # Step 3: Build sparse co-occurrence matrix
    log(f"    Building sparse co-occurrence matrix...")
    t0 = time.time()

    active_seqs = np.where(active_mask)[0]
    seq_to_compact = np.full(n_cold, -1, dtype=np.int64)
    seq_to_compact[active_seqs] = np.arange(n_active, dtype=np.int64)

    # Build COO data for X (active_compact_idx, batch_idx)
    row_list = []
    col_list = []
    for bi in range(num_sample_batches):
        seqs = batch_cold_seqs[bi]
        if len(seqs) == 0:
            continue
        compact = seq_to_compact[seqs]
        valid = compact >= 0
        compact = compact[valid]
        if len(compact) == 0:
            continue
        row_list.append(compact)
        col_list.append(np.full(len(compact), bi, dtype=np.int64))

    if not row_list:
        return np.arange(n_cold, dtype=np.int64)

    rows = np.concatenate(row_list)
    cols = np.concatenate(col_list)
    data = np.ones(len(rows), dtype=np.float32)

    X = sp.csr_matrix((data, (rows, cols)), shape=(n_active, num_sample_batches))
    avg_batches_per_idx = X.nnz / n_active
    log(f"    X matrix: {X.shape}, nnz={X.nnz:,}, "
        f"avg batches/idx={avg_batches_per_idx:.1f} ({time.time()-t0:.1f}s)")

    # Compute co-occurrence: C = X @ X.T
    t0 = time.time()
    C = X @ X.T
    C.setdiag(0)
    C.eliminate_zeros()
    log(f"    Co-occurrence matrix: nnz={C.nnz:,} ({time.time()-t0:.1f}s)")

    # Threshold
    if min_cooccurrence > 1:
        mask = C.data >= min_cooccurrence
        C.data[~mask] = 0
        C.eliminate_zeros()
        log(f"    After threshold (>={min_cooccurrence}): nnz={C.nnz:,}")

    n_edges = C.nnz // 2  # symmetric → unique edges

    # Step 4: Choose community detection strategy based on graph size
    active_freq = cold_freq[active_seqs]
    active_first_batch = cold_first_batch[active_seqs]

    MAX_EDGES_FOR_LOUVAIN = 20_000_000  # 20M edges max for networkx Louvain

    if n_edges == 0:
        # No co-occurrence structure — fall back to batch-affinity sort
        log(f"    No co-occurrence edges — falling back to batch-affinity sort")
        sort_key = active_first_batch.astype(np.float64) * 1e12 - active_freq.astype(np.float64)
        compact_order = np.argsort(sort_key)

    elif n_edges > MAX_EDGES_FOR_LOUVAIN:
        # Graph too large for networkx Louvain — use scipy connected components
        # then batch-affinity within each component
        log(f"    Graph too large for Louvain ({n_edges:,} edges > {MAX_EDGES_FOR_LOUVAIN:,})")
        log(f"    Using scipy connected components + batch-affinity within components...")
        t0 = time.time()

        # Make symmetric (already is, but ensure CSC for connected_components)
        n_components, labels = sp.csgraph.connected_components(C, directed=False)
        log(f"    Found {n_components:,} connected components ({time.time()-t0:.1f}s)")

        # Log component size distribution
        comp_sizes = np.bincount(labels)
        top_sizes = sorted(comp_sizes, reverse=True)[:10]
        log(f"    Top 10 component sizes: {top_sizes}")

        # If few large components, try Louvain on the largest ones individually
        large_comps = np.where(comp_sizes >= 100)[0]
        community_id = np.full(n_active, -1, dtype=np.int64)
        next_cid = 0

        if len(large_comps) > 0 and len(large_comps) <= 50:
            log(f"    Running Louvain on {len(large_comps)} large components...")
            for comp_idx in large_comps:
                comp_nodes = np.where(labels == comp_idx)[0]
                if len(comp_nodes) < 3:
                    for n in comp_nodes:
                        community_id[n] = next_cid
                    next_cid += 1
                    continue

                # Extract subgraph
                sub_C = C[comp_nodes][:, comp_nodes]
                sub_coo = sub_C.tocoo()
                mask = sub_coo.row < sub_coo.col
                if mask.sum() > MAX_EDGES_FOR_LOUVAIN:
                    # Still too large, just use component as one community
                    for n in comp_nodes:
                        community_id[n] = next_cid
                    next_cid += 1
                    continue

                G_sub = nx.Graph()
                G_sub.add_nodes_from(range(len(comp_nodes)))
                edges = list(zip(sub_coo.row[mask].tolist(), sub_coo.col[mask].tolist(),
                                sub_coo.data[mask].tolist()))
                G_sub.add_weighted_edges_from(edges)

                sub_communities = louvain_communities(G_sub, weight='weight',
                                                      resolution=resolution, seed=42)
                for members in sub_communities:
                    for m in members:
                        community_id[comp_nodes[m]] = next_cid
                    next_cid += 1

        # Assign remaining (small components) by component ID
        for i in range(n_active):
            if community_id[i] == -1:
                community_id[i] = next_cid + labels[i]
        next_cid += n_components

        log(f"    Total communities: {len(np.unique(community_id)):,}")

        sort_key = community_id.astype(np.float64) * 1e12 - active_freq.astype(np.float64)
        compact_order = np.argsort(sort_key)

    else:
        # Graph is manageable — full Louvain
        log(f"    Building networkx graph ({n_edges:,} edges)...")
        t0 = time.time()

        G = nx.Graph()
        G.add_nodes_from(range(n_active))

        C_coo = C.tocoo()
        mask = C_coo.row < C_coo.col
        edges = list(zip(C_coo.row[mask].tolist(), C_coo.col[mask].tolist(),
                         C_coo.data[mask].tolist()))
        G.add_weighted_edges_from(edges)
        log(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        log(f"    Running Louvain (resolution={resolution})...")
        communities = louvain_communities(G, weight='weight', resolution=resolution, seed=42)
        n_communities = len(communities)
        log(f"    Found {n_communities} communities ({time.time()-t0:.1f}s)")

        sizes = sorted([len(c) for c in communities], reverse=True)
        if len(sizes) > 10:
            log(f"    Top 10 community sizes: {sizes[:10]}")
        else:
            log(f"    Community sizes: {sizes}")

        community_id = np.full(n_active, -1, dtype=np.int64)
        for cid, members in enumerate(communities):
            for m in members:
                community_id[m] = cid

        sort_key = community_id.astype(np.float64) * 1e12 - active_freq.astype(np.float64)
        compact_order = np.argsort(sort_key)

    # Step 5: Build final reorder permutation over ALL cold indices
    reorder = np.empty(n_cold, dtype=np.int64)
    reorder[:n_active] = active_seqs[compact_order]
    inactive_seqs = np.where(~active_mask)[0]
    reorder[n_active:] = inactive_seqs

    log(f"    Reorder complete: {n_active:,} active + {len(inactive_seqs):,} inactive")
    return reorder


# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("REC-AD STYLE COMMUNITY DETECTION REORDERING BENCHMARK")
    log("Co-occurrence graph + Louvain vs Batch-affinity sort")
    log("=" * 70)
    log(f"RSS at start: {get_rss_mb():.0f}MB")

    # ---- Phase 1: Load model, data, profiling ----
    log("\nPhase 1 — Loading model, data, profiling results")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")
    log(f"C++ extension: {'ENABLED' if HAS_CPP_EXT else 'DISABLED'}")

    total_emb_mb = sum(state_dict[k].numel() * 4 for k in emb_keys) / 1024 / 1024
    log(f"Total embedding memory: {total_emb_mb:.1f}MB")

    # Load hot/cold data
    is_hot = {}
    hot_indices = {}
    cold_indices = {}
    for t in large_tables:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]
        cold_indices[t] = torch.where(~is_hot[t])[0]

    # Load batch access log
    bal_path = os.path.join(PROFILING_DIR, 'batch_access_log.npz')
    _bal = np.load(bal_path, allow_pickle=True)
    batch_access_log = _bal['batch_access_log'].tolist()
    log(f"Batch access log: {len(batch_access_log)} batches")

    # ---- Phase 2: Rec-AD reordering ----
    log("\n" + "=" * 70)
    log("Phase 2 — Rec-AD Co-Occurrence Reordering")
    log("=" * 70)

    recad_done = os.path.join(REORDER_DIR_RECAD, '.done')
    if os.path.exists(recad_done):
        log("Rec-AD reordering SKIPPED — already done")
    else:
        for t in large_tables:
            n_cold = len(cold_indices[t])
            if n_cold == 0:
                continue
            log(f"\n  Table {t}: {n_cold:,} cold embeddings")

            cold_idx_np = cold_indices[t].numpy()

            # Run Rec-AD community detection reordering
            reorder = build_cooccurrence_reorder(
                batch_access_log, cold_idx_np, ln_emb[t], t,
                min_cooccurrence=2, resolution=1.0)

            # Build reordered mapping: orig_idx -> new cold position
            o2c_reordered = torch.full((ln_emb[t],), -1, dtype=torch.long)
            orig_indices_reordered = cold_idx_np[reorder]
            new_positions = torch.arange(len(reorder), dtype=torch.long)
            o2c_reordered[torch.from_numpy(orig_indices_reordered)] = new_positions

            # Save artifacts (same format as original)
            np.save(os.path.join(REORDER_DIR_RECAD, f'cold_order_{t}.npy'), reorder)
            torch.save(o2c_reordered, os.path.join(REORDER_DIR_RECAD, f'orig_to_cold_reordered_{t}.pt'))
            with open(os.path.join(REORDER_DIR_RECAD, f'num_cold_{t}.txt'), 'w') as f:
                f.write(str(n_cold))

            # Also save as .npy for mmap access
            np.save(os.path.join(REORDER_DIR_RECAD, f'orig_to_cold_reordered_{t}.npy'),
                    o2c_reordered.numpy())

            log(f"  Saved reorder artifacts for table {t}")
            del reorder, o2c_reordered
            gc.collect()

        with open(recad_done, 'w') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
        log("\nPhase 2 complete — Rec-AD reordering done")

    # ---- Phase 3: H.265 encoding with Rec-AD reordering ----
    log("\n" + "=" * 70)
    log("Phase 3 — H.265 Encoding with Rec-AD Reordering")
    log("=" * 70)

    cold_quant_scale = {}
    cold_quant_zp = {}
    cold_num_rows = {}

    for t in large_tables:
        n_cold_path = os.path.join(REORDER_DIR_RECAD, f'num_cold_{t}.txt')
        with open(n_cold_path) as f:
            cold_num_rows[t] = int(f.read().strip())

    for res_name, (width, height) in RESOLUTIONS.items():
        res_dir = os.path.join(ONDEMAND_DIR_RECAD, res_name)
        done_marker = os.path.join(res_dir, '.done')

        if os.path.exists(done_marker):
            log(f"\nEncoding for {res_name} SKIPPED — already done")
            for t in large_tables:
                if t not in cold_quant_scale:
                    meta_path = os.path.join(res_dir, f'table_{t}', 'meta.json')
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            meta = json.load(f)
                        cold_quant_scale[t] = meta['quant_scale']
                        cold_quant_zp[t] = meta['quant_zp']
            continue

        log(f"\n{'='*70}")
        log(f"ENCODING: {res_name} ({width}x{height}) with Rec-AD reorder")
        log(f"{'='*70}")

        os.makedirs(res_dir, exist_ok=True)
        pixels_per_frame = width * height
        rows_per_frame = pixels_per_frame // EMB_DIM

        for t in large_tables:
            n_cold = cold_num_rows[t]
            if n_cold == 0:
                continue

            log(f"\n  Table {t}: {n_cold:,} cold embeddings")

            cold_order = np.load(os.path.join(REORDER_DIR_RECAD, f'cold_order_{t}.npy'))
            cold_w = state_dict[emb_keys[t]][cold_indices[t]]
            reordered_w = cold_w[cold_order]
            q, s, zp = quantize_table(reordered_w)
            cold_quant_scale[t] = s
            cold_quant_zp[t] = zp

            num_frames, frame_dir, compressed_bytes, enc_time, rpf = \
                encode_h265_perframe(q.numpy(), width, height, crf=H265_CRF,
                                     output_dir=res_dir, table_id=t)

            # Save metadata
            raw_bytes = n_cold * EMB_DIM
            meta = {
                'num_frames': num_frames,
                'rows_per_frame': rpf,
                'width': width, 'height': height,
                'n_cold': n_cold,
                'compressed_bytes': compressed_bytes,
                'raw_bytes': raw_bytes,
                'quant_scale': s,
                'quant_zp': zp,
            }
            meta_path = os.path.join(frame_dir, 'meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f)

            del cold_w, reordered_w, q
            gc.collect()

        with open(done_marker, 'w') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
        log(f"  Encoding for {res_name} complete")

    # ---- Phase 4: Inference benchmarks ----
    log("\n" + "=" * 70)
    log("Phase 4 — Inference Benchmarks")
    log("=" * 70)

    # Import classes from codec_ondemand_benchmark (top-level code just creates dirs + opens log)
    import codec_ondemand_benchmark as cob
    OnDemandFrameDecoder = cob.OnDemandFrameDecoder
    OnDemandPrefetchCache = cob.OnDemandPrefetchCache
    GlobalFrameCache = cob.GlobalFrameCache
    CompressedEmbeddingBag = cob.CompressedEmbeddingBag
    InMemoryFrameDecoder = cob.InMemoryFrameDecoder
    MarkovPredictor = cob.MarkovPredictor

    # Pre-cache test batches
    log("Pre-caching test batches...")
    test_batches = []
    for j, (X, lS_o, lS_i, T) in enumerate(test_ld):
        test_batches.append((X, lS_o, lS_i, T))
    log(f"  {len(test_batches)} test batches cached")

    # Store original embedding modules for restoration
    original_emb_modules = {}
    for t in large_tables:
        original_emb_modules[t] = dlrm.emb_l[t]

    def restore_weights():
        for t_idx in large_tables:
            dlrm.emb_l[t_idx] = original_emb_modules[t_idx]
            k = emb_keys[t_idx]
            with torch.no_grad():
                dlrm.emb_l[t_idx].weight = nn.Parameter(
                    state_dict[k].clone(), requires_grad=False)
        gc.collect()

    # ---- Baseline run ----
    log("\nRunning BASELINE (no compression)...")
    restore_weights()
    drop_caches()

    baseline_scores = np.empty(len(test_batches) * TEST_BATCH_SIZE + TEST_BATCH_SIZE, dtype=np.float32)
    baseline_targets = np.empty_like(baseline_scores)
    sample_idx = 0
    blats = []
    t0 = time.time()
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            z_np = Z.detach().cpu().numpy().ravel()
            t_np = T.detach().cpu().numpy().ravel()
            bs = z_np.shape[0]
            baseline_scores[sample_idx:sample_idx+bs] = z_np
            baseline_targets[sample_idx:sample_idx+bs] = t_np
            sample_idx += bs
    baseline_time = time.time() - t0
    baseline_auc = roc_auc_score(baseline_targets[:sample_idx], baseline_scores[:sample_idx])
    baseline_lat = np.mean(blats) * 1000
    log(f"  Baseline: AUC={baseline_auc:.6f}, Time={baseline_time:.2f}s, "
        f"Mean lat={baseline_lat:.2f}ms, RSS={get_rss_mb():.0f}MB")

    # ---- Run compressed configs for BOTH reorder methods ----
    all_results = {'baseline': {
        'auc': baseline_auc, 'total_time': baseline_time,
        'mean_lat_ms': baseline_lat, 'method': 'none',
    }}

    def run_inference_with_reorder(reorder_dir, ondemand_dir, method_name,
                                    res_name, cache_capacity, tag,
                                    use_bitmap=False, full_cpp=True):
        """Run inference benchmark with a specific reorder method."""
        width, height = RESOLUTIONS[res_name]
        pixels_per_frame = width * height
        rows_per_frame = pixels_per_frame // EMB_DIM
        res_dir = os.path.join(ondemand_dir, res_name)

        log(f"\n--- {tag} [{method_name}]: {res_name}, cache={cache_capacity}, "
            f"bitmap={use_bitmap}, full_cpp={full_cpp} ---")

        restore_weights()

        # Load reorder mappings from this method's directory
        o2c_map = {}
        for t in large_tables:
            fp = os.path.join(reorder_dir, f'orig_to_cold_reordered_{t}.pt')
            o2c_map[t] = torch.load(fp, map_location='cpu', weights_only=True)

        # Load quant params
        local_quant_scale = {}
        local_quant_zp = {}
        for t in large_tables:
            meta_path = os.path.join(res_dir, f'table_{t}', 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                local_quant_scale[t] = meta['quant_scale']
                local_quant_zp[t] = meta['quant_zp']

        # Create global cache
        cap = 9999 if full_cpp else cache_capacity
        global_cache = GlobalFrameCache(capacity=cap, store_uint8=True)

        # Build CompressedEmbeddingBag for each large table
        caches = {}
        total_compressed_bytes = 0
        total_hot_mb = 0
        total_mapping_mb = 0

        for t_idx in large_tables:
            n_cold = cold_num_rows.get(t_idx, 0)
            if n_cold == 0:
                continue
            frame_dir = os.path.join(res_dir, f'table_{t_idx}')
            if not os.path.exists(frame_dir):
                log(f"  WARNING: No frame dir for table {t_idx}")
                continue

            frame_files = sorted([f for f in os.listdir(frame_dir)
                                  if f.startswith('frame_') and f.endswith('.h265')])
            comp_bytes = sum(os.path.getsize(os.path.join(frame_dir, ff))
                             for ff in frame_files)
            total_compressed_bytes += comp_bytes

            num_frames_t = len(frame_files)
            cache_obj = OnDemandPrefetchCache(
                frame_dir=frame_dir,
                rows_per_frame=rows_per_frame,
                emb_dim=EMB_DIM,
                num_cold_rows=n_cold,
                width=width, height=height,
                quant_scale=local_quant_scale[t_idx],
                quant_zp=local_quant_zp[t_idx],
                cache_capacity=cache_capacity,
                predictor=None,
                num_prefetch_workers=2,
                global_cache=global_cache,
                table_id=t_idx,
            )
            caches[t_idx] = cache_obj

            # Build compact hot embedding
            w = state_dict[emb_keys[t_idx]]
            h_idx = hot_indices[t_idx]
            hot_weight = w[h_idx].clone()
            total_hot_mb += hot_weight.numel() / 1024 / 1024  # uint8

            orig_to_hot = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
            orig_to_hot[h_idx] = torch.arange(len(h_idx))

            comp_emb = CompressedEmbeddingBag(
                hot_weight=hot_weight,
                is_hot=is_hot[t_idx],
                orig_to_hot=orig_to_hot,
                orig_to_cold_reordered=o2c_map[t_idx],
                cold_cache=cache_obj,
                num_embeddings=ln_emb[t_idx],
                embedding_dim=EMB_DIM,
                quantize_hot=True,
            )
            total_mapping_mb += ln_emb[t_idx] * 4 / 1024 / 1024
            dlrm.emb_l[t_idx] = comp_emb

        # Free state_dict entries for large tables
        freed_state = {}
        for t_idx in large_tables:
            k = emb_keys[t_idx]
            if k in state_dict:
                freed_state[k] = state_dict.pop(k)
        for t_idx in large_tables:
            if t_idx in original_emb_modules:
                original_emb_modules[t_idx].weight = nn.Parameter(
                    torch.zeros(1, EMB_DIM), requires_grad=False)
        gc.collect()

        total_compressed_mb = total_compressed_bytes / 1024 / 1024
        rss_after_setup = get_rss_mb()

        # Register tables in C++ for fast_forward
        compressed_table_ids = set(caches.keys())
        num_tabs = len(dlrm.emb_l)

        if HAS_CPP_EXT and compressed_table_ids:
            table_kinds = []
            weights = []
            mappings = []
            scales = []
            zero_points = []
            _cold_caches = {}
            _empty_psw = {}
            _empty_set = set()

            for k in range(num_tabs):
                E = dlrm.emb_l[k]
                if k in compressed_table_ids and isinstance(E, CompressedEmbeddingBag):
                    if E.quantize_hot:
                        table_kinds.append(2)
                        weights.append(E.hot_weight_q8)
                        mappings.append(E.mapping)
                        scales.append(float(E.hot_scale))
                        zero_points.append(int(E.hot_zp))
                    else:
                        table_kinds.append(1)
                        weights.append(E.hot_weight)
                        mappings.append(E.mapping)
                        scales.append(0.0)
                        zero_points.append(0)
                    _cold_caches[k] = E.cold_cache
                    _empty_psw[k] = E._empty_psw
                else:
                    table_kinds.append(0)
                    weights.append(E.weight)
                    mappings.append(torch.empty(0, dtype=torch.int32))
                    scales.append(0.0)
                    zero_points.append(0)

            use_hash = False
            _C.register_tables(table_kinds, weights, mappings, scales, zero_points,
                               use_hash_table=use_hash, use_bitmap=use_bitmap)

            if use_bitmap:
                _cold_reordered_mmap = {}
                for k in _cold_caches:
                    E = dlrm.emb_l[k]
                    mmap_path = os.path.join(reorder_dir, f'orig_to_cold_reordered_{k}.npy')
                    if not os.path.exists(mmap_path):
                        pt_path = os.path.join(reorder_dir, f'orig_to_cold_reordered_{k}.pt')
                        o2c = torch.load(pt_path, map_location='cpu', weights_only=True)
                        np.save(mmap_path, o2c.numpy())
                        del o2c
                    _cold_reordered_mmap[k] = np.load(mmap_path, mmap_mode='r')
                    if hasattr(E, 'mapping'):
                        del E.mapping
                del mappings
                gc.collect()

            # Warmup cache (1 batch)
            log(f"  Warming up cache (1 batch)...")
            with torch.no_grad():
                X, lS_o, lS_i, T = test_batches[0]
                Z = dlrm(X, lS_o, lS_i)

            # Full C++ mode: register cold frames
            _full_cpp_active = False
            _cold_frame_mb = 0.0
            if full_cpp and global_cache is not None:
                log(f"  Scanning all batches for cold frame coverage...")
                t_scan0 = time.time()
                needed_frames = {t_idx: set() for t_idx in caches}
                for X_s, lS_o_s, lS_i_s, T_s in test_batches:
                    for t_idx in caches:
                        indices = lS_i_s[t_idx]
                        cold_mask = ~is_hot[t_idx][indices]
                        if cold_mask.any():
                            cold_orig = indices[cold_mask]
                            cold_mapped = o2c_map[t_idx][cold_orig]
                            valid = cold_mapped >= 0
                            if valid.any():
                                fids = (cold_mapped[valid] // rows_per_frame).unique().tolist()
                                needed_frames[t_idx].update(fids)
                total_needed = sum(len(v) for v in needed_frames.values())
                log(f"  Found {total_needed} unique cold frames ({time.time()-t_scan0:.1f}s)")

                for t_idx in caches:
                    for fid in sorted(needed_frames[t_idx]):
                        cached = global_cache.get(t_idx, fid)
                        if cached is None:
                            frame_data = caches[t_idx]._decode_raw(fid)
                            global_cache.put(t_idx, fid, frame_data)

                log(f"  Registering cold frames in C++...")
                total_cold_frame_mb = 0
                for t_idx in caches:
                    table_frames = {}
                    for (tid, fid), data in global_cache.cache.items():
                        if tid == t_idx:
                            table_frames[fid] = data
                    if not table_frames:
                        continue
                    sorted_fids = sorted(table_frames.keys())
                    padded_frames = []
                    for fid in sorted_fids:
                        frame = table_frames[fid]
                        if frame.shape[0] < rows_per_frame:
                            pad = np.zeros((rows_per_frame - frame.shape[0], EMB_DIM), dtype=np.uint8)
                            frame = np.concatenate([frame, pad], axis=0)
                        padded_frames.append(frame)
                    all_data = np.concatenate(padded_frames, axis=0)
                    frame_data_tensor = torch.from_numpy(all_data.copy())
                    frame_ids_tensor = torch.tensor(sorted_fids, dtype=torch.long)

                    if use_bitmap:
                        mmap_path = os.path.join(reorder_dir, f'orig_to_cold_reordered_{t_idx}.npy')
                        cold_mapping_tensor = torch.from_numpy(
                            np.load(mmap_path).copy()).int()
                        _C.register_cold_frames_for_table(
                            t_idx, frame_ids_tensor, frame_data_tensor,
                            float(local_quant_scale[t_idx]),
                            float(local_quant_zp[t_idx]),
                            rows_per_frame,
                            cold_mapping_tensor)
                    else:
                        _C.register_cold_frames_for_table(
                            t_idx, frame_ids_tensor, frame_data_tensor,
                            float(local_quant_scale[t_idx]),
                            float(local_quant_zp[t_idx]),
                            rows_per_frame,
                            torch.empty(0, dtype=torch.long))
                    total_cold_frame_mb += all_data.nbytes / 1024 / 1024

                _full_cpp_active = True
                _cold_frame_mb = total_cold_frame_mb
                log(f"  Cold frames registered: {total_cold_frame_mb:.1f}MB uint8")

                def _full_cpp_apply_emb(lS_o, lS_i, emb_l, v_W_l):
                    start_time = time.time()
                    if isinstance(lS_i, (list, tuple)):
                        lS_i_2d = torch.stack(lS_i)
                    elif lS_i.dim() == 2:
                        lS_i_2d = lS_i
                    else:
                        lS_i_2d = lS_i.view(num_tabs, -1)
                    if isinstance(lS_o, (list, tuple)):
                        lS_o_2d = torch.stack(lS_o)
                    elif lS_o.dim() == 2:
                        lS_o_2d = lS_o
                    else:
                        lS_o_2d = lS_o.view(num_tabs, -1)
                    results = _C.fast_forward(lS_i_2d, lS_o_2d)
                    dlrm.time_look_up += time.time() - start_time
                    return results[-1]
                dlrm.apply_emb = _full_cpp_apply_emb
                log(f"  Full C++ apply_emb enabled")

        # Run inference
        drop_caches()
        gc.collect()

        num_test_batches = len(test_batches)
        max_samples = num_test_batches * TEST_BATCH_SIZE + TEST_BATCH_SIZE
        all_scores = np.empty(max_samples, dtype=np.float32)
        all_targets_arr = np.empty(max_samples, dtype=np.float32)
        sample_idx = 0
        blats_local = []
        t0 = time.time()

        with torch.no_grad():
            for batch_idx in range(num_test_batches):
                X, lS_o, lS_i, T = test_batches[batch_idx]
                bt0 = time.time()
                Z = dlrm(X, lS_o, lS_i)
                blats_local.append(time.time() - bt0)

                z_np = Z.detach().cpu().numpy().ravel()
                t_np = T.detach().cpu().numpy().ravel()
                bs = z_np.shape[0]
                all_scores[sample_idx:sample_idx+bs] = z_np
                all_targets_arr[sample_idx:sample_idx+bs] = t_np
                sample_idx += bs

                if batch_idx % 500 == 0:
                    log(f"    Batch {batch_idx}: lat={blats_local[-1]*1000:.1f}ms, RSS={get_rss_mb():.0f}MB")

        total_time = time.time() - t0
        scores_arr = all_scores[:sample_idx]
        targets_arr = all_targets_arr[:sample_idx]
        auc = roc_auc_score(targets_arr, scores_arr)
        rss = get_rss_mb()

        total_hits = sum(c.stats['cache_hits'] for c in caches.values())
        total_misses = sum(c.stats['cache_misses'] for c in caches.values())
        total_demand = sum(c.stats['demand_decomps'] for c in caches.values())
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0

        # Memory accounting
        effective_mapping_mb = total_mapping_mb
        if use_bitmap:
            bm_mb = 0
            for t_idx in caches:
                n_rows = ln_emb[t_idx]
                n_words = (n_rows + 63) // 64
                bm_mb += (n_words * 8 + (n_words + 1) * 4) / 1024 / 1024
            effective_mapping_mb = bm_mb
            if _full_cpp_active:
                cold_map_mb = sum(ln_emb[t] * 4 for t in caches) / 1024 / 1024
                effective_mapping_mb += cold_map_mb

        lru_mb = _cold_frame_mb if _full_cpp_active else 0
        total_mem_mb = total_hot_mb + effective_mapping_mb + lru_mb
        compression_ratio = total_emb_mb / total_mem_mb if total_mem_mb > 0 else 0

        log(f"\n  RESULT [{method_name}] {tag}: AUC={auc:.6f}, Time={total_time:.2f}s, "
            f"Mean lat={np.mean(blats_local)*1000:.2f}ms, "
            f"P99={np.percentile(blats_local, 99)*1000:.2f}ms, "
            f"Hit rate={hit_rate:.1%}, Memory={total_mem_mb:.0f}MB ({compression_ratio:.1f}x)")

        result = {
            'auc': auc, 'total_time': total_time,
            'mean_lat_ms': float(np.mean(blats_local) * 1000),
            'p99_lat_ms': float(np.percentile(blats_local, 99) * 1000),
            'rss_mb': rss,
            'hot_mb': total_hot_mb,
            'mapping_mb': effective_mapping_mb,
            'compressed_cold_mb': total_compressed_mb,
            'lru_mb': lru_mb,
            'total_mem_mb': total_mem_mb,
            'hit_rate': hit_rate,
            'demand_decomps': total_demand,
            'compression_ratio': compression_ratio,
            'method': method_name,
            'resolution': res_name,
            'cache_capacity': cache_capacity,
            'use_bitmap': use_bitmap,
        }

        # Restore state_dict entries
        for k, v in freed_state.items():
            state_dict[k] = v
        restore_weights()

        return result

    # ---- Run configs for both methods ----
    torch.set_num_threads(40)
    log(f"Thread count: 40")

    configs = [
        ('1080p', 32, '1080p_fullcpp', False),
        ('1080p', 32, '1080p_bitmap_fullcpp', True),
    ]

    for res_name, cap, tag, use_bm in configs:
        # Original batch-affinity
        key = f'orig_{tag}'
        all_results[key] = run_inference_with_reorder(
            REORDER_DIR_ORIG, ONDEMAND_DIR_ORIG, 'batch-affinity',
            res_name, cap, tag, use_bitmap=use_bm)

        # Rec-AD community detection
        key = f'recad_{tag}'
        all_results[key] = run_inference_with_reorder(
            REORDER_DIR_RECAD, ONDEMAND_DIR_RECAD, 'recad-louvain',
            res_name, cap, tag, use_bitmap=use_bm)

    # ---- Phase 5: Save results and comparison ----
    log("\n" + "=" * 70)
    log("Phase 5 — Results Comparison")
    log("=" * 70)

    # Save JSON
    results_json = os.path.join(RESULTS_DIR, 'recad_results.json')
    serializable = {}
    for k, v in all_results.items():
        sv = {kk: vv for kk, vv in v.items()
              if isinstance(vv, (int, float, str, bool, type(None)))}
        serializable[k] = sv
    with open(results_json, 'w') as f:
        json.dump(serializable, f, indent=2)
    log(f"  Saved: {results_json}")

    # Generate comparison table
    summary = []
    summary.append("# Rec-AD vs Batch-Affinity Reordering Comparison\n")
    summary.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Baseline AUC: {baseline_auc:.6f}, Time: {baseline_time:.2f}s\n")

    header = (f"{'Config':<36} {'Method':<16} {'AUC':>8} {'Time(s)':>7} "
              f"{'MeanLat':>8} {'P99Lat':>8} {'Hit%':>6} {'TotalMB':>8} {'Reduc':>6}")
    summary.append(header)
    summary.append("-" * len(header))

    for key in sorted(all_results.keys()):
        if key == 'baseline':
            continue
        r = all_results[key]
        auc = r.get('auc', 0)
        t = r.get('total_time', 0)
        mlat = r.get('mean_lat_ms', 0)
        p99 = r.get('p99_lat_ms', 0)
        hit = r.get('hit_rate', 0)
        tmem = r.get('total_mem_mb', 0)
        reduc = r.get('compression_ratio', 0)
        method = r.get('method', '')
        line = (f"{key:<36} {method:<16} {auc:>8.6f} {t:>7.2f} "
                f"{mlat:>7.2f}ms {p99:>7.2f}ms {hit:>5.1%} {tmem:>7.0f}MB {reduc:>5.1f}x")
        summary.append(line)

    # Add delta analysis
    summary.append("\n\n## Delta Analysis (Rec-AD vs Batch-Affinity)\n")
    for res_name, cap, tag, use_bm in configs:
        orig_key = f'orig_{tag}'
        recad_key = f'recad_{tag}'
        if orig_key in all_results and recad_key in all_results:
            o = all_results[orig_key]
            r = all_results[recad_key]
            summary.append(f"### {tag}")
            summary.append(f"  AUC delta:     {r['auc'] - o['auc']:+.6f}")
            summary.append(f"  Time delta:    {r['total_time'] - o['total_time']:+.2f}s "
                         f"({(r['total_time']/o['total_time'] - 1)*100:+.1f}%)")
            summary.append(f"  Mean lat delta: {r['mean_lat_ms'] - o['mean_lat_ms']:+.2f}ms")
            summary.append(f"  Hit rate delta: {r['hit_rate'] - o['hit_rate']:+.3f}")
            summary.append("")

    summary_text = "\n".join(summary)
    summary_path = os.path.join(RESULTS_DIR, 'recad_results.md')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    log(f"\n{summary_text}")
    log(f"\n  Saved: {summary_path}")

    log(f"\n{'='*70}")
    log("REC-AD BENCHMARK COMPLETE")
    log(f"{'='*70}")
    log(f"Final RSS: {get_rss_mb():.0f}MB")


if __name__ == '__main__':
    main()
