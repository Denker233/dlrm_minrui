#!/usr/bin/env python3
"""
Deep Dive: Reordering Algorithms for H.265 Embedding Compression

Compares 4 cold row orderings under lossy H.265 (CRF=18):
  1. Natural (original row ID order)
  2. Random
  3. Frequency sort (descending access frequency)
  4. Batch-affinity (first_batch, -frequency)

For each ordering, measures:
  - Per-row MSE by frequency bucket (error steering quality)
  - AUC after lossy decode (end-to-end quality)
  - Compression ratio (on cold rows)
  - Frame access pattern (unique frames per batch, total frames accessed)

Dataset: Kaggle (D=16) — faster iteration
"""

import os, sys, time, json, gc, types
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import av

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
DATA_FILE = os.path.expanduser("~/input/train.txt")
PROCESSED_DATA = os.path.expanduser("~/input/kaggleAdDisplayChallenge_processed.npz")
ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 2048
EMB_DIM = 16
LARGE_TABLE_THRESHOLD = 50000
HOT_FRAC = 0.043  # top 4.3% by access frequency
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
TILE_H = 4
TILE_W = 4
RPF = (FRAME_WIDTH * FRAME_HEIGHT) // (TILE_H * TILE_W)  # 129600

RESULTS_DIR = "results/reorder_deepdive"
LOG_FILE = "logs/reorder_deepdive.log"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_fh = open(LOG_FILE, 'w')
def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


# ============================================================
# UTILITIES
# ============================================================
def quantize_table_uint8(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = torch.clamp(torch.round(w / s + zp), 0, 255).to(torch.uint8)
    return q, s, zp

def dequantize_uint8(q, s, zp):
    return (q.float() - zp) * s

def pack_frame_tiled(rows, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Pack embedding rows into 1920x1080 frame using 4x4 tiling."""
    tpr = width // TILE_W
    tpc = height // TILE_H
    rpf = tpr * tpc
    n = rows.shape[0]
    if n < rpf:
        pad = np.zeros((rpf - n, EMB_DIM), dtype=np.uint8)
        rows = np.concatenate([rows, pad], axis=0)
    tiles = rows[:rpf].reshape(tpc, tpr, TILE_H, TILE_W)
    frame = tiles.transpose(0, 2, 1, 3).reshape(height, width)
    return frame

def unpack_frame_tiled(frame, n_rows, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Unpack 1920x1080 frame back to embedding rows."""
    tpr = width // TILE_W
    tpc = height // TILE_H
    rpf = tpr * tpc
    grid = frame.reshape(tpc, TILE_H, tpr, TILE_W)
    rows = grid.transpose(0, 2, 1, 3).reshape(rpf, EMB_DIM)
    return rows[:n_rows]

def encode_frame_h265(frame_2d, crf, lossless=False):
    """Encode single frame with H.265, return compressed size and decoded frame."""
    height, width = frame_2d.shape
    output = av.open('/dev/null', 'w', format='null')
    stream = output.add_stream('libx265', rate=1)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'gray'
    if lossless:
        stream.options = {'x265-params': 'lossless=1', 'preset': 'ultrafast'}
    else:
        stream.options = {'crf': str(crf), 'preset': 'ultrafast',
                         'x265-params': 'keyint=1:min-keyint=1'}

    # Encode to bytes
    import io
    buf = io.BytesIO()
    container = av.open(buf, 'w', format='hevc')
    s = container.add_stream('libx265', rate=1)
    s.width = width; s.height = height; s.pix_fmt = 'gray'
    if lossless:
        s.options = {'x265-params': 'lossless=1', 'preset': 'ultrafast'}
    else:
        s.options = {'crf': str(crf), 'preset': 'ultrafast',
                    'x265-params': 'keyint=1:min-keyint=1'}

    frame = av.VideoFrame.from_ndarray(frame_2d, format='gray')
    for pkt in s.encode(frame):
        container.mux(pkt)
    for pkt in s.encode():
        container.mux(pkt)
    container.close()

    comp_bytes = buf.tell()

    # Decode
    buf.seek(0)
    dec_container = av.open(buf, format='hevc')
    for dec_frame in dec_container.decode(video=0):
        decoded = dec_frame.to_ndarray(format='gray')
        break
    dec_container.close()

    return comp_bytes, decoded


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
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    log(f"Model loaded. {len(ln_emb)} tables, emb_dim={m_spa}")
    return dlrm, test_ld, ln_emb


def run_auc(dlrm, test_ld):
    scores, targets = [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_ld:
            Z = dlrm(X, lS_o, lS_i)
            scores.append(Z.detach().cpu().numpy().ravel())
            targets.append(T.detach().cpu().numpy().ravel())
    return roc_auc_score(np.concatenate(targets), np.concatenate(scores))


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def main():
    log("=" * 70)
    log("DEEP DIVE: Reordering Algorithms for H.265 Embedding Compression")
    log("=" * 70)

    dlrm, test_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = {int(k.split('.')[1]): k for k in state_dict if 'emb_l' in k and 'weight' in k}
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Large tables: {large_tables}")

    # Baseline AUC
    baseline_auc = run_auc(dlrm, test_ld)
    log(f"Baseline AUC: {baseline_auc:.6f}")

    # ========================================
    # Phase 1: Profile access frequencies
    # ========================================
    log("\n--- Phase 1: Profiling access frequencies ---")
    freq = {}
    first_batch = {}  # batch index where row first appears

    # Cache test batches
    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    n_batches = len(test_batches)
    log(f"  {n_batches} test batches")

    for t in large_tables:
        N = ln_emb[t]
        f = torch.zeros(N, dtype=torch.long)
        fb = torch.full((N,), n_batches, dtype=torch.long)
        for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
            idx = lS_i[t].flatten()
            # Frequency
            uniq, counts = idx.unique(return_counts=True)
            f[uniq] += counts
            # First batch
            first_unseen = fb[uniq] > bi
            fb[uniq[first_unseen]] = bi
        freq[t] = f
        first_batch[t] = fb
        active = (f > 0).sum().item()
        log(f"  Table {t}: {N:,} rows, {active:,} active ({100*active/N:.1f}%)")

    # ========================================
    # Phase 2: Build hot/cold split
    # ========================================
    log("\n--- Phase 2: Hot/cold split ---")
    hot_indices = {}
    cold_indices = {}
    is_hot = {}

    for t in large_tables:
        N = ln_emb[t]
        n_hot = max(1, int(N * HOT_FRAC))
        sorted_idx = freq[t].argsort(descending=True)
        hot_idx = sorted_idx[:n_hot]
        cold_idx = sorted_idx[n_hot:]
        hot_mask = torch.zeros(N, dtype=torch.bool)
        hot_mask[hot_idx] = True
        hot_indices[t] = hot_idx
        cold_indices[t] = cold_idx
        is_hot[t] = hot_mask
        log(f"  Table {t}: {n_hot:,} hot, {N - n_hot:,} cold")

    # ========================================
    # Phase 3: Build orderings
    # ========================================
    log("\n--- Phase 3: Building 4 cold row orderings ---")

    orderings = {}

    for t in large_tables:
        cold_idx = cold_indices[t]  # indices into original table
        n_cold = len(cold_idx)
        cold_freq = freq[t][cold_idx]
        cold_fb = first_batch[t][cold_idx]

        # 1. Natural: cold rows in original row ID order
        natural_order = torch.arange(n_cold)
        # cold_idx is already sorted by freq (descending), so natural != cold_idx order
        # We need cold_idx sorted by original row ID
        _, natural_order = cold_idx.sort()

        # 2. Random
        random_order = torch.randperm(n_cold)

        # 3. Frequency sort (descending frequency of cold rows)
        freq_order = cold_freq.argsort(descending=True)

        # 4. Batch-affinity: sort by (first_batch, -frequency)
        sort_key = cold_fb.float() * 1e12 - cold_freq.float()
        ba_order = sort_key.argsort()

        orderings[t] = {
            'natural': natural_order,
            'random': random_order,
            'frequency': freq_order,
            'batch_affinity': ba_order,
        }
        log(f"  Table {t}: 4 orderings built for {n_cold:,} cold rows")

    # ========================================
    # Phase 4: Co-access analysis
    # ========================================
    log("\n--- Phase 4: Co-access analysis ---")
    # For each ordering, count how many unique frames are accessed per batch
    # Also count total unique frames across all batches

    frame_analysis = {}
    for order_name in ['natural', 'random', 'frequency', 'batch_affinity']:
        log(f"\n  Ordering: {order_name}")

        # Build cold_row -> frame_id mapping for each table under this ordering
        cold_frame_map = {}  # table -> numpy array: cold_seq_idx -> frame_id
        for t in large_tables:
            n_cold = len(cold_indices[t])
            perm = orderings[t][order_name]
            # perm[new_pos] = old_cold_seq
            # We need: for each old_cold_seq, what frame does it go to?
            inv_perm = torch.argsort(perm)  # old_cold_seq -> new_pos
            frame_ids = (inv_perm.numpy() // RPF).astype(np.int32)
            n_frames = (n_cold + RPF - 1) // RPF
            cold_frame_map[t] = frame_ids  # indexed by cold_seq_idx
            log(f"    Table {t}: {n_cold:,} cold rows → {n_frames} frames")

        # Build orig_row -> cold_seq_idx mapping
        orig_to_cold_seq = {}
        for t in large_tables:
            o2c = torch.full((ln_emb[t],), -1, dtype=torch.long)
            o2c[cold_indices[t]] = torch.arange(len(cold_indices[t]))
            orig_to_cold_seq[t] = o2c.numpy()

        # Count frames accessed per batch
        frames_per_batch = []
        all_accessed_frames = {t: set() for t in large_tables}
        cold_lookups_per_batch = []

        for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
            batch_frames = set()
            batch_cold_count = 0
            for t in large_tables:
                idx = lS_i[t].flatten().numpy()
                # Find cold rows
                cold_mask = ~is_hot[t].numpy()[idx]
                cold_orig_idx = idx[cold_mask]
                if len(cold_orig_idx) == 0:
                    continue
                # Map to cold seq idx
                cold_seq = orig_to_cold_seq[t][cold_orig_idx]
                valid = cold_seq >= 0
                cold_seq_valid = cold_seq[valid]
                if len(cold_seq_valid) == 0:
                    continue
                # Map to frame IDs
                fids = cold_frame_map[t][cold_seq_valid]
                unique_fids = set(fids.tolist())
                for fid in unique_fids:
                    batch_frames.add((t, fid))
                    all_accessed_frames[t].add(fid)
                batch_cold_count += len(cold_seq_valid)

            frames_per_batch.append(len(batch_frames))
            cold_lookups_per_batch.append(batch_cold_count)

        total_accessed = sum(len(v) for v in all_accessed_frames.values())
        total_possible = sum((len(cold_indices[t]) + RPF - 1) // RPF for t in large_tables)

        frame_analysis[order_name] = {
            'frames_per_batch_mean': float(np.mean(frames_per_batch)),
            'frames_per_batch_p50': float(np.median(frames_per_batch)),
            'frames_per_batch_p99': float(np.percentile(frames_per_batch, 99)),
            'total_accessed_frames': total_accessed,
            'total_possible_frames': total_possible,
            'frame_access_ratio': round(total_accessed / total_possible, 4),
            'cold_lookups_per_batch_mean': float(np.mean(cold_lookups_per_batch)),
            'per_table_accessed': {str(t): len(all_accessed_frames[t]) for t in large_tables},
            'per_table_total': {str(t): (len(cold_indices[t]) + RPF - 1) // RPF for t in large_tables},
        }

        log(f"    Frames/batch: mean={np.mean(frames_per_batch):.1f}, "
            f"p50={np.median(frames_per_batch):.0f}, p99={np.percentile(frames_per_batch, 99):.0f}")
        log(f"    Total accessed: {total_accessed}/{total_possible} frames "
            f"({100*total_accessed/total_possible:.1f}%)")
        log(f"    Cold lookups/batch: {np.mean(cold_lookups_per_batch):.0f}")

    # ========================================
    # Phase 5: Error steering analysis (per-row MSE by frequency bucket)
    # ========================================
    log("\n--- Phase 5: Error steering under CRF=18 ---")

    crf_values = [0, 18]  # both lossless and lossy
    bucket_names = ['top_1pct', '1_10pct', '10_50pct', '50_100pct']
    error_results = {}

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        N = w.shape[0]
        cold_idx = cold_indices[t]
        n_cold = len(cold_idx)
        cold_w = w[cold_idx]
        q_cold, s, zp = quantize_table_uint8(cold_w)

        # Frequency buckets for cold rows
        cold_freq_vals = freq[t][cold_idx]
        freq_rank = cold_freq_vals.argsort(descending=True)
        bucket = torch.full((n_cold,), 3, dtype=torch.long)
        n1 = max(1, int(n_cold * 0.01))
        n10 = max(1, int(n_cold * 0.10))
        n50 = max(1, int(n_cold * 0.50))
        bucket[freq_rank[:n1]] = 0
        bucket[freq_rank[n1:n10]] = 1
        bucket[freq_rank[n10:n50]] = 2

        table_results = {}

        for order_name in ['natural', 'random', 'frequency', 'batch_affinity']:
            perm = orderings[t][order_name]
            inv_perm = torch.argsort(perm)
            q_ordered = q_cold[perm]

            for crf in crf_values:
                lossless = (crf == 0)
                total_comp = 0
                recon_rows = []

                n_frames = (n_cold + RPF - 1) // RPF
                for fid in range(n_frames):
                    start = fid * RPF
                    end = min(start + RPF, n_cold)
                    frame_rows = q_ordered[start:end].numpy()
                    frame = pack_frame_tiled(frame_rows)
                    comp_size, decoded = encode_frame_h265(frame, crf, lossless=lossless)
                    total_comp += comp_size
                    recon = unpack_frame_tiled(decoded, end - start)
                    recon_rows.append(torch.from_numpy(recon.copy()))

                recon_ordered = torch.cat(recon_rows, dim=0)

                # Per-row MSE in reordered space
                per_row_mse_ordered = ((q_ordered.float() - recon_ordered.float()) ** 2).mean(dim=1)

                # Map back to original cold-row order
                per_row_mse = per_row_mse_ordered[inv_perm]

                # Compute per-bucket MSE
                bucket_mse = {}
                for bi, bname in enumerate(bucket_names):
                    mask = bucket == bi
                    if mask.any():
                        bucket_mse[bname] = round(per_row_mse[mask].mean().item(), 4)

                overall_mse = round(per_row_mse.mean().item(), 4)
                ratio = n_cold * EMB_DIM / max(1, total_comp)

                key = f"{order_name}_crf{crf}"
                table_results[key] = {
                    'bucket_mse': bucket_mse,
                    'overall_mse': overall_mse,
                    'comp_bytes': total_comp,
                    'ratio': round(ratio, 2),
                }

                log(f"  Table {t}, {order_name}, CRF={crf}: "
                    f"MSE top1%={bucket_mse.get('top_1pct', '?')}, "
                    f"overall={overall_mse}, ratio={ratio:.1f}x")

        error_results[str(t)] = table_results

    # ========================================
    # Phase 6: AUC under lossy compression for each ordering
    # ========================================
    log("\n--- Phase 6: AUC under lossy compression (CRF=18) ---")
    from dlrm_s_pytorch import DLRM_Net
    orig_apply = DLRM_Net.apply_emb

    auc_results = {}

    for order_name in ['natural', 'random', 'frequency', 'batch_affinity']:
        log(f"\n  Testing ordering: {order_name}")

        # Apply H.265 CRF=18 compression to cold rows of each large table
        for t in large_tables:
            w_orig = state_dict[emb_keys[t]]
            N = w_orig.shape[0]
            cold_idx = cold_indices[t]
            hot_idx = hot_indices[t]
            n_cold = len(cold_idx)

            cold_w = w_orig[cold_idx]
            q_cold, s, zp = quantize_table_uint8(cold_w)

            perm = orderings[t][order_name]
            inv_perm = torch.argsort(perm)
            q_ordered = q_cold[perm]

            # Encode/decode
            recon_rows = []
            n_frames = (n_cold + RPF - 1) // RPF
            for fid in range(n_frames):
                start = fid * RPF
                end = min(start + RPF, n_cold)
                frame_rows = q_ordered[start:end].numpy()
                frame = pack_frame_tiled(frame_rows)
                _, decoded = encode_frame_h265(frame, 18, lossless=False)
                recon = unpack_frame_tiled(decoded, end - start)
                recon_rows.append(torch.from_numpy(recon.copy()))

            recon_ordered = torch.cat(recon_rows, dim=0)
            recon_cold = recon_ordered[inv_perm]

            # Dequantize back to fp32
            cold_fp32 = dequantize_uint8(recon_cold, s, zp)

            # Rebuild full weight: hot rows unchanged, cold rows lossy-decoded
            w_new = w_orig.clone()
            w_new[cold_idx] = cold_fp32
            dlrm.emb_l[t].weight.data = w_new

        # Run AUC
        auc = run_auc(dlrm, test_ld)
        delta = auc - baseline_auc
        auc_results[order_name] = {
            'auc': round(auc, 6),
            'delta': round(delta, 6),
            'delta_pct': round(abs(delta) * 100, 4),
        }
        log(f"    AUC={auc:.6f} (delta={delta:+.6f}, loss={abs(delta)*100:.4f}%)")

        # Restore
        for t in large_tables:
            dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

    # Also test: frequency sort applied only to cold rows (matching our paper system)
    log(f"\n  Testing: hot-cold split + frequency sort (paper system)")
    for t in large_tables:
        w_orig = state_dict[emb_keys[t]]
        hot_idx = hot_indices[t]
        cold_idx = cold_indices[t]
        n_cold = len(cold_idx)

        cold_w = w_orig[cold_idx]
        q_cold, s, zp = quantize_table_uint8(cold_w)
        perm = orderings[t]['frequency']
        inv_perm = torch.argsort(perm)
        q_ordered = q_cold[perm]

        recon_rows = []
        n_frames = (n_cold + RPF - 1) // RPF
        for fid in range(n_frames):
            start = fid * RPF
            end = min(start + RPF, n_cold)
            frame_rows = q_ordered[start:end].numpy()
            frame = pack_frame_tiled(frame_rows)
            _, decoded = encode_frame_h265(frame, 18, lossless=False)
            recon = unpack_frame_tiled(decoded, end - start)
            recon_rows.append(torch.from_numpy(recon.copy()))

        recon_ordered = torch.cat(recon_rows, dim=0)
        recon_cold = recon_ordered[inv_perm]
        cold_fp32 = dequantize_uint8(recon_cold, s, zp)

        # Hot rows UNCHANGED (fp32 original), cold rows lossy-decoded
        w_new = w_orig.clone()
        w_new[cold_idx] = cold_fp32
        dlrm.emb_l[t].weight.data = w_new

    auc = run_auc(dlrm, test_ld)
    delta = auc - baseline_auc
    auc_results['freq_hot_preserved'] = {
        'auc': round(auc, 6),
        'delta': round(delta, 6),
        'delta_pct': round(abs(delta) * 100, 4),
    }
    log(f"    AUC={auc:.6f} (delta={delta:+.6f}, loss={abs(delta)*100:.4f}%)")
    for t in large_tables:
        dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

    # And batch-affinity with hot preserved
    log(f"\n  Testing: hot-cold split + batch-affinity (alternative system)")
    for t in large_tables:
        w_orig = state_dict[emb_keys[t]]
        cold_idx = cold_indices[t]
        n_cold = len(cold_idx)

        cold_w = w_orig[cold_idx]
        q_cold, s, zp = quantize_table_uint8(cold_w)
        perm = orderings[t]['batch_affinity']
        inv_perm = torch.argsort(perm)
        q_ordered = q_cold[perm]

        recon_rows = []
        n_frames = (n_cold + RPF - 1) // RPF
        for fid in range(n_frames):
            start = fid * RPF
            end = min(start + RPF, n_cold)
            frame_rows = q_ordered[start:end].numpy()
            frame = pack_frame_tiled(frame_rows)
            _, decoded = encode_frame_h265(frame, 18, lossless=False)
            recon = unpack_frame_tiled(decoded, end - start)
            recon_rows.append(torch.from_numpy(recon.copy()))

        recon_ordered = torch.cat(recon_rows, dim=0)
        recon_cold = recon_ordered[inv_perm]
        cold_fp32 = dequantize_uint8(recon_cold, s, zp)

        w_new = w_orig.clone()
        w_new[cold_idx] = cold_fp32
        dlrm.emb_l[t].weight.data = w_new

    auc = run_auc(dlrm, test_ld)
    delta = auc - baseline_auc
    auc_results['ba_hot_preserved'] = {
        'auc': round(auc, 6),
        'delta': round(delta, 6),
        'delta_pct': round(abs(delta) * 100, 4),
    }
    log(f"    AUC={auc:.6f} (delta={delta:+.6f}, loss={abs(delta)*100:.4f}%)")
    for t in large_tables:
        dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

    # ========================================
    # Phase 7: Hybrid ordering — batch-affinity WITHIN frequency bands
    # ========================================
    log("\n--- Phase 7: Hybrid orderings ---")

    # Hybrid 1: Frequency bands + batch-affinity within bands
    # Split cold rows into bands by frequency, then batch-affinity within each band
    hybrid_auc_results = {}

    for n_bands in [4, 8, 16]:
        log(f"\n  Hybrid: {n_bands} frequency bands + batch-affinity within")
        for t in large_tables:
            w_orig = state_dict[emb_keys[t]]
            cold_idx = cold_indices[t]
            n_cold = len(cold_idx)
            cold_w = w_orig[cold_idx]
            q_cold, s, zp = quantize_table_uint8(cold_w)

            cold_freq_vals = freq[t][cold_idx]
            cold_fb = first_batch[t][cold_idx]

            # Assign bands by frequency quantile
            freq_rank = cold_freq_vals.argsort(descending=True)
            band_size = max(1, n_cold // n_bands)
            bands = torch.zeros(n_cold, dtype=torch.long)
            for b in range(n_bands):
                start = b * band_size
                end = min(start + band_size, n_cold) if b < n_bands - 1 else n_cold
                bands[freq_rank[start:end]] = b

            # Within each band, sort by batch-affinity
            sort_key = bands.float() * 1e15 + cold_fb.float() * 1e12 - cold_freq_vals.float()
            hybrid_perm = sort_key.argsort()
            inv_perm = torch.argsort(hybrid_perm)
            q_ordered = q_cold[hybrid_perm]

            recon_rows = []
            n_frames = (n_cold + RPF - 1) // RPF
            for fid in range(n_frames):
                start = fid * RPF
                end = min(start + RPF, n_cold)
                frame_rows = q_ordered[start:end].numpy()
                frame = pack_frame_tiled(frame_rows)
                _, decoded = encode_frame_h265(frame, 18, lossless=False)
                recon = unpack_frame_tiled(decoded, end - start)
                recon_rows.append(torch.from_numpy(recon.copy()))

            recon_ordered = torch.cat(recon_rows, dim=0)
            recon_cold = recon_ordered[inv_perm]
            cold_fp32 = dequantize_uint8(recon_cold, s, zp)

            w_new = w_orig.clone()
            w_new[cold_idx] = cold_fp32
            dlrm.emb_l[t].weight.data = w_new

        auc = run_auc(dlrm, test_ld)
        delta = auc - baseline_auc
        hybrid_auc_results[f'hybrid_{n_bands}bands'] = {
            'auc': round(auc, 6),
            'delta': round(delta, 6),
            'delta_pct': round(abs(delta) * 100, 4),
        }
        log(f"    AUC={auc:.6f} (delta={delta:+.6f}, loss={abs(delta)*100:.4f}%)")
        for t in large_tables:
            dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

    # ========================================
    # RESULTS SUMMARY
    # ========================================
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)

    all_results = {
        'baseline_auc': baseline_auc,
        'frame_analysis': frame_analysis,
        'error_steering': error_results,
        'auc_results': auc_results,
        'hybrid_auc_results': hybrid_auc_results,
    }

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"Saved: {json_path}")

    # Print summary tables
    log("\n--- AUC Comparison (CRF=18, cold rows only) ---")
    log(f"{'Ordering':<30} {'AUC':>10} {'Delta':>10} {'Loss%':>8}")
    log("-" * 60)
    for name in ['natural', 'random', 'frequency', 'batch_affinity',
                 'freq_hot_preserved', 'ba_hot_preserved']:
        r = auc_results[name]
        log(f"{name:<30} {r['auc']:>10.6f} {r['delta']:>+10.6f} {r['delta_pct']:>7.4f}%")

    log("\n--- Hybrid Orderings ---")
    for name, r in hybrid_auc_results.items():
        log(f"{name:<30} {r['auc']:>10.6f} {r['delta']:>+10.6f} {r['delta_pct']:>7.4f}%")

    log("\n--- Frame Access Pattern ---")
    log(f"{'Ordering':<20} {'Frames/batch':>14} {'Total accessed':>16} {'Ratio':>8}")
    log("-" * 60)
    for name in ['natural', 'random', 'frequency', 'batch_affinity']:
        fa = frame_analysis[name]
        log(f"{name:<20} {fa['frames_per_batch_mean']:>13.1f} "
            f"{fa['total_accessed_frames']:>15d} "
            f"{fa['frame_access_ratio']:>7.1%}")

    # Aggregate error steering across all tables
    log("\n--- Error Steering (CRF=18, averaged across large tables) ---")
    log(f"{'Ordering':<20} {'Top 1%':>8} {'1-10%':>8} {'10-50%':>8} {'50-100%':>8} {'Overall':>8}")
    log("-" * 62)
    for order_name in ['natural', 'random', 'frequency', 'batch_affinity']:
        bucket_sums = {b: 0.0 for b in bucket_names}
        overall_sum = 0.0
        n_tables = 0
        for t_str, t_data in error_results.items():
            key = f"{order_name}_crf18"
            if key in t_data:
                n_tables += 1
                for b in bucket_names:
                    bucket_sums[b] += t_data[key]['bucket_mse'].get(b, 0)
                overall_sum += t_data[key]['overall_mse']
        if n_tables > 0:
            log(f"{order_name:<20} "
                f"{bucket_sums['top_1pct']/n_tables:>8.3f} "
                f"{bucket_sums['1_10pct']/n_tables:>8.3f} "
                f"{bucket_sums['10_50pct']/n_tables:>8.3f} "
                f"{bucket_sums['50_100pct']/n_tables:>8.3f} "
                f"{overall_sum/n_tables:>8.3f}")

    # Generate markdown summary
    md_lines = ["# Reordering Deep Dive Results\n"]
    md_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"Dataset: Kaggle (D=16), Baseline AUC={baseline_auc:.6f}\n")

    md_lines.append("## AUC under CRF=18 (cold rows only)\n")
    md_lines.append("| Ordering | AUC | AUC Loss (%) |")
    md_lines.append("|----------|-----|-------------|")
    for name in ['natural', 'random', 'frequency', 'batch_affinity',
                 'freq_hot_preserved', 'ba_hot_preserved']:
        r = auc_results[name]
        md_lines.append(f"| {name} | {r['auc']:.6f} | {r['delta_pct']:.4f}% |")
    for name, r in hybrid_auc_results.items():
        md_lines.append(f"| {name} | {r['auc']:.6f} | {r['delta_pct']:.4f}% |")

    md_lines.append("\n## Frame Access Pattern\n")
    md_lines.append("| Ordering | Frames/batch (mean) | Total frames accessed | % of all frames |")
    md_lines.append("|----------|--------------------|-----------------------|-----------------|")
    for name in ['natural', 'random', 'frequency', 'batch_affinity']:
        fa = frame_analysis[name]
        md_lines.append(f"| {name} | {fa['frames_per_batch_mean']:.1f} | "
                       f"{fa['total_accessed_frames']} | {fa['frame_access_ratio']:.1%} |")

    md_lines.append("\n## Error Steering (CRF=18, averaged across large tables)\n")
    md_lines.append("| Ordering | Top 1% MSE | 1-10% MSE | 10-50% MSE | 50-100% MSE | Overall MSE |")
    md_lines.append("|----------|-----------|----------|-----------|------------|------------|")
    for order_name in ['natural', 'random', 'frequency', 'batch_affinity']:
        bucket_sums = {b: 0.0 for b in bucket_names}
        overall_sum = 0.0
        n_tables = 0
        for t_str, t_data in error_results.items():
            key = f"{order_name}_crf18"
            if key in t_data:
                n_tables += 1
                for b in bucket_names:
                    bucket_sums[b] += t_data[key]['bucket_mse'].get(b, 0)
                overall_sum += t_data[key]['overall_mse']
        if n_tables > 0:
            md_lines.append(
                f"| {order_name} | {bucket_sums['top_1pct']/n_tables:.3f} | "
                f"{bucket_sums['1_10pct']/n_tables:.3f} | "
                f"{bucket_sums['10_50pct']/n_tables:.3f} | "
                f"{bucket_sums['50_100pct']/n_tables:.3f} | "
                f"{overall_sum/n_tables:.3f} |")

    md_path = os.path.join(RESULTS_DIR, 'summary.md')
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines))
    log(f"\nSaved: {md_path}")

    log(f"\n{'='*70}")
    log("DEEP DIVE COMPLETE")
    log(f"{'='*70}")


if __name__ == '__main__':
    main()
