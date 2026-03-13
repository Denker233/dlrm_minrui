#!/usr/bin/env python3
"""CRF sweep + cache simulation + memory breakdown.

For each dataset (Kaggle, Terabyte):
1. Profile access frequencies, frequency-sort cold rows
2. Simulate LRU cache behavior: which frames accessed per batch, hit rate
3. Compute memory breakdown at each CRF level + cache size combo
4. (Optional) Encode at CRF levels, decode, measure AUC

Key question: does higher CRF + reordering reduce runtime memory?
"""

import sys, os, time, json, math
import numpy as np
import torch
from collections import OrderedDict

# --- Logging ---
LOG_FILE = "logs/crf_cache_memory.log"
os.makedirs("logs", exist_ok=True)
os.makedirs("results/crf_cache_memory", exist_ok=True)
_log_f = open(LOG_FILE, "w")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_f.write(line + "\n"); _log_f.flush()

# --- Constants ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
LARGE_TABLE_THRESHOLD = 50000
HOT_FRACTION = 0.043  # top 4.3% of rows by frequency

# --- Load C++ extension ---
HAS_CPP = False
try:
    import compressed_emb as _C
    HAS_CPP = True
    log("C++ extension loaded")
except ImportError:
    log("WARNING: C++ extension not available, tiling won't work")

# --- Model loading (reuse pattern from experiment_mlsys_baselines.py) ---
def load_dataset(dataset_name):
    """Load model, test data, and state dict for given dataset."""
    log(f"Loading {dataset_name} dataset...")

    os.environ['CRITEO_DAYS'] = '4'  # we have 4 days of terabyte data
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net

    class Args: pass
    a = Args()
    if dataset_name == 'kaggle':
        a.arch_sparse_feature_size = 16
        a.arch_mlp_bot = "13-512-256-64-16"; a.arch_mlp_top = "512-256-1"
        a.raw_data_file = os.path.expanduser("~/input/train.txt")
        a.processed_data_file = os.path.expanduser("~/input/kaggleAdDisplayChallenge_processed.npz")
        a.data_set = "kaggle"; a.max_ind_range = -1
        model_path = "./models/dlrm_kaggle_correct.pt"
    else:
        a.arch_sparse_feature_size = 64
        a.arch_mlp_bot = "13-512-256-64"; a.arch_mlp_top = "512-512-256-1"
        a.raw_data_file = os.path.expanduser("~/input/terabyte/day")
        a.processed_data_file = os.path.expanduser("~/input/terabyte/terabyte_processed.npz")
        a.memory_map = False  # force npz path
        a.data_set = "terabyte"; a.max_ind_range = 10000000
        model_path = "./models/dlrm_terabyte_4day.pt"

    a.arch_interaction_op = "dot"; a.arch_interaction_itself = False
    a.data_generation = "dataset"; a.loss_function = "bce"
    a.test_mini_batch_size = 2048
    a.test_num_workers = 0; a.num_workers = 0
    a.mlperf_logging = False; a.memory_map = False; a.data_randomize = "total"
    a.data_trace_enable_padding = False; a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10; a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 2048; a.round_targets = True
    a.mlperf_bin_loader = False; a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False

    D = a.arch_sparse_feature_size
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(a)
    ln_emb = np.array(train_data.counts)

    ln_bot = np.fromstring(a.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + a.arch_mlp_top, dtype=int, sep="-")

    dlrm = DLRM_Net(
        m_spa=D, ln_emb=ln_emb, ln_bot=ln_bot, ln_top=ln_top,
        arch_interaction_op="dot", sigmoid_bot=-1, sigmoid_top=ln_top.size - 2
    )

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    dlrm.load_state_dict(state_dict)
    dlrm.eval()
    log(f"  {len(ln_emb)} tables, D={D}")

    return dlrm, test_ld, ln_emb, state_dict, D

def quantize_table_uint8(w):
    """Global min/max quantization to uint8."""
    mn, mx = w.min().item(), w.max().item()
    if mx == mn:
        return torch.zeros_like(w, dtype=torch.uint8), 1.0, 0
    s = (mx - mn) / 255.0
    zp = round(-mn / s)
    zp = max(0, min(255, zp))
    q = ((w / s) + zp).round().clamp(0, 255).to(torch.uint8)
    return q, s, zp

def dequantize_uint8(q, s, zp):
    return (q.float() - zp) * s

def encode_frame_h265(frame_np, crf):
    """Encode a (H,W) uint8 frame via ffmpeg, return compressed_size."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix='.h265', delete=True) as tmp:
        h, w = frame_np.shape
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{w}x{h}', '-i', 'pipe:0',
               '-c:v', 'libx265', '-preset', 'medium',
               '-x265-params', f'qp={crf}:log-level=0' if crf == 0 else f'log-level=0',
               '-crf', str(crf), tmp.name]
        proc = subprocess.run(cmd, input=frame_np.tobytes(),
                             capture_output=True, timeout=120)
        comp_size = os.path.getsize(tmp.name)

        # Decode back
        decoded = None
        if HAS_CPP and hasattr(_C, 'decode_h265_frame_from_file'):
            decoded = _C.decode_h265_frame_from_file(tmp.name).numpy()
        else:
            import av
            container = av.open(tmp.name)
            for frame in container.decode(video=0):
                decoded = frame.to_ndarray(format='gray')
                break
            container.close()
        return comp_size, decoded

def run_auc(dlrm, test_ld):
    """Run inference, return AUC."""
    from sklearn.metrics import roc_auc_score
    all_scores, all_labels = [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_ld:
            Z = dlrm(X, lS_o, lS_i)
            all_scores.append(Z.detach().cpu().numpy().flatten())
            all_labels.append(T.numpy().flatten())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return roc_auc_score(labels, scores)


def profile_and_simulate(dataset_name, dlrm, test_ld, ln_emb, state_dict, D):
    """
    Phase 1: Profile access patterns, frequency-sort, simulate cache.
    Phase 2: CRF sweep with actual AUC measurement.
    """
    large_tables = [i for i, n in enumerate(ln_emb) if n >= LARGE_TABLE_THRESHOLD]
    log(f"\nLarge tables: {large_tables}")

    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                       key=lambda x: int(x.split('.')[1]))

    # ====================================================================
    # PHASE 1: Profile access frequencies
    # ====================================================================
    log("\nPhase 1: Profiling access frequencies...")
    freq = {}
    for X, lS_o, lS_i, T in test_ld:
        for t in large_tables:
            if isinstance(lS_i, (list, tuple)):
                idx = lS_i[t].flatten()
            elif lS_i.dim() == 2:
                idx = lS_i[t].flatten()
            else:
                idx = lS_i.flatten()
            if t not in freq:
                freq[t] = torch.zeros(ln_emb[t], dtype=torch.long)
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))

    # Determine hot/cold split and frequency sorting
    if D == 16:
        rpf = (FRAME_WIDTH // 4) * (FRAME_HEIGHT // 4)  # 129600 for 4x4 tiling
        tiling = "4x4"
    else:
        rpf = (FRAME_WIDTH // D) * FRAME_HEIGHT  # 32400 for D=64 flat
        tiling = "flat"
    log(f"  Layout: {tiling}, rows_per_frame={rpf}")

    table_info = {}
    for t in large_tables:
        n = int(ln_emb[t])
        n_hot = max(1, int(n * HOT_FRACTION))
        sorted_idx = freq[t].argsort(descending=True)
        hot_idx = sorted_idx[:n_hot]
        cold_idx = sorted_idx[n_hot:]
        n_cold = len(cold_idx)
        n_frames = (n_cold + rpf - 1) // rpf

        # Build mapping: for each cold row (in sorted order), which frame?
        # Use numpy array for fast vectorized lookup (-1 = hot/unmapped)
        cold_frame_map_arr = np.full(n, -1, dtype=np.int32)
        cold_idx_np = cold_idx.numpy()
        frame_ids = np.arange(n_cold, dtype=np.int32) // rpf
        cold_frame_map_arr[cold_idx_np] = frame_ids
        cold_frame_map = cold_frame_map_arr  # numpy array indexed by row_id

        n_accessed = int((freq[t] > 0).sum())
        n_hot_accessed = int(freq[t][hot_idx].gt(0).sum())
        n_cold_accessed = int(freq[t][cold_idx].gt(0).sum())

        table_info[t] = {
            'n_rows': n, 'n_hot': n_hot, 'n_cold': n_cold,
            'n_frames': n_frames, 'cold_frame_map': cold_frame_map,
            'n_accessed': n_accessed, 'n_hot_accessed': n_hot_accessed,
            'n_cold_accessed': n_cold_accessed,
            'hot_idx': hot_idx, 'cold_idx': cold_idx, 'sorted_idx': sorted_idx,
        }

        # Which frames contain accessed cold rows?
        accessed_cold = cold_idx_np[freq[t].numpy()[cold_idx_np] > 0]
        accessed_frames = set(cold_frame_map[accessed_cold].tolist())
        table_info[t]['accessed_frames'] = accessed_frames

        log(f"  Table {t}: {n:,} rows, {n_hot:,} hot ({n_hot_accessed} accessed), "
            f"{n_cold:,} cold ({n_cold_accessed} accessed), {n_frames} frames, "
            f"{len(accessed_frames)} frames with accessed rows")

    # ====================================================================
    # PHASE 2: Simulate LRU cache per-batch (vectorized, per-row hits)
    # ====================================================================
    log("\nPhase 2: Simulating LRU cache behavior...")

    # Collect per-batch frame accesses: both unique frames and row counts per frame
    batch_frame_data = []  # list of dicts: {(table_id, frame_id): n_rows}
    n_batches = 0
    for X, lS_o, lS_i, T in test_ld:
        frame_row_counts = {}  # (table, frame) -> count of cold row lookups
        for t in large_tables:
            if isinstance(lS_i, (list, tuple)):
                idx = lS_i[t].flatten().numpy()
            elif lS_i.dim() == 2:
                idx = lS_i[t].flatten().numpy()
            else:
                idx = lS_i.flatten().numpy()

            cfm = table_info[t]['cold_frame_map']  # numpy array
            # Vectorized: look up frame for each index
            idx_clamped = np.clip(idx, 0, len(cfm) - 1)
            frames = cfm[idx_clamped]
            # frames == -1 means hot row (not cold)
            cold_mask = frames >= 0
            cold_frames = frames[cold_mask]
            # Count rows per frame
            for fid in cold_frames:
                key = (t, int(fid))
                frame_row_counts[key] = frame_row_counts.get(key, 0) + 1

        batch_frame_data.append(frame_row_counts)
        n_batches += 1
        if n_batches % 1000 == 0:
            log(f"    Profiled {n_batches} batches...")

    log(f"  {n_batches} batches profiled")

    # Simulate LRU cache at different sizes
    # Count both frame-level and row-level hit rates
    cache_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    cache_results = {}

    # Total unique frames
    all_unique = set()
    for bfd in batch_frame_data:
        all_unique.update(bfd.keys())
    log(f"  Total unique frames accessed: {len(all_unique)}")

    for cs in cache_sizes:
        cache = OrderedDict()
        frame_hits = 0
        frame_misses = 0
        row_hits = 0
        row_misses = 0

        for frame_row_counts in batch_frame_data:
            for key, n_rows in frame_row_counts.items():
                if key in cache:
                    cache.move_to_end(key)
                    frame_hits += 1
                    row_hits += n_rows
                else:
                    frame_misses += 1
                    row_misses += n_rows
                    cache[key] = True
                    while len(cache) > cs:
                        cache.popitem(last=False)

        total_frames_acc = frame_hits + frame_misses
        total_rows_acc = row_hits + row_misses
        frame_hit_rate = frame_hits / max(total_frames_acc, 1) * 100
        row_hit_rate = row_hits / max(total_rows_acc, 1) * 100
        cache_results[cs] = {
            'cache_size': cs,
            'frame_accesses': total_frames_acc,
            'frame_hits': frame_hits,
            'frame_hit_rate': frame_hit_rate,
            'row_accesses': total_rows_acc,
            'row_hits': row_hits,
            'row_hit_rate': row_hit_rate,
            'unique_frames': len(all_unique),
        }
        log(f"  Cache={cs:3d}: frame_hit={frame_hit_rate:.1f}%, row_hit={row_hit_rate:.1f}%, "
            f"frame_misses={frame_misses}, row_accesses={total_rows_acc:,}")

    # Count per-table unique frames
    per_table_unique = {}
    for t in large_tables:
        per_table_unique[t] = len(table_info[t]['accessed_frames'])
        log(f"    Table {t}: {per_table_unique[t]} unique frames / {table_info[t]['n_frames']} total")

    # ====================================================================
    # PHASE 3: Memory breakdown at each CRF level
    # ====================================================================
    log("\nPhase 3: Computing memory breakdown...")

    # Quantize each table to get compressed sizes at different CRFs
    # We'll compute analytically for the memory table, and encode a few frames for actual measurement

    # First, compute table sizes
    total_fp32_bytes = sum(int(ln_emb[t]) * D * 4 for t in large_tables)
    total_uint8_bytes = sum(table_info[t]['n_cold'] * D for t in large_tables)
    total_hot_bytes = sum(table_info[t]['n_hot'] * D * 4 for t in large_tables)
    total_mapping_bytes = sum(int(ln_emb[t]) * 4 for t in large_tables)  # int32 per row
    bitmap_bytes = sum(((int(ln_emb[t]) + 63) // 64 * 8 + ((int(ln_emb[t]) + 63) // 64 + 1) * 4)
                       for t in large_tables)

    frame_bytes = rpf * D  # bytes per decoded uint8 frame
    n_unique_frames = len(all_unique)
    # For cache configs, use row_hit_rate
    for cs_key in cache_results:
        cache_results[cs_key]['hit_rate'] = cache_results[cs_key]['row_hit_rate']
    total_frames = sum(table_info[t]['n_frames'] for t in large_tables)

    log(f"\n  Total fp32 (large tables): {total_fp32_bytes / 1024**2:.1f} MB")
    log(f"  Total cold uint8: {total_uint8_bytes / 1024**2:.1f} MB")
    log(f"  Total hot fp32: {total_hot_bytes / 1024**2:.1f} MB")
    log(f"  Mapping (int32/row): {total_mapping_bytes / 1024**2:.1f} MB")
    log(f"  Bitmap overhead: {bitmap_bytes / 1024**2:.2f} MB")
    log(f"  Per decoded frame: {frame_bytes / 1024**2:.2f} MB")
    log(f"  Total frames (all tables): {total_frames}")
    log(f"  Unique frames accessed: {n_unique_frames}")

    # Encode sample frames to get compression ratios at each CRF
    log("\nPhase 3b: Encoding sample frames for compression ratios...")
    crfs = [0, 10, 18, 23, 28]
    crf_compressed_sizes = {}

    # Helper to pack a chunk of uint8 rows into a frame
    def pack_frame(chunk, D, rpf):
        if D == 16 and HAS_CPP:
            if chunk.shape[0] < rpf:
                chunk_padded = torch.cat([chunk, torch.zeros(rpf - chunk.shape[0], D, dtype=torch.uint8)])
            else:
                chunk_padded = chunk
            return _C.tile_rows_to_frame(chunk_padded, FRAME_WIDTH, FRAME_HEIGHT).numpy()
        else:
            rpr = FRAME_WIDTH // D
            n_rows = chunk.shape[0]
            H_used = (n_rows + rpr - 1) // rpr
            frame_np = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
            flat_data = chunk.numpy().flatten()
            n_fill = H_used * rpr * D
            if len(flat_data) < n_fill:
                flat_data = np.pad(flat_data, (0, n_fill - len(flat_data)))
            frame_np[:H_used] = flat_data[:n_fill].reshape(H_used, rpr * D)
            return frame_np

    # For speed, encode sample frames (first 3 + last 1 per table) and extrapolate
    MAX_SAMPLE_FRAMES = 4

    for crf in crfs:
        total_comp = 0
        total_uint8_est = 0
        for t in large_tables:
            w = state_dict[emb_keys[t]]
            q, s, zp = quantize_table_uint8(w)
            q_cold_sorted = q[table_info[t]['cold_idx']]

            n_cold = len(table_info[t]['cold_idx'])
            n_frames_t = table_info[t]['n_frames']

            # Sample frames: first few + last one (different data distributions)
            if n_frames_t <= MAX_SAMPLE_FRAMES + 1:
                sample_fis = list(range(n_frames_t))
            else:
                sample_fis = list(range(min(3, n_frames_t))) + [n_frames_t - 1]

            sample_comp = 0
            sample_uint8 = 0
            for fi in sample_fis:
                start = fi * rpf
                end = min(start + rpf, n_cold)
                chunk = q_cold_sorted[start:end]
                frame_np = pack_frame(chunk, D, rpf)
                comp_size, _ = encode_frame_h265(frame_np, crf)
                sample_comp += comp_size
                sample_uint8 += (end - start) * D

            # Extrapolate to all frames
            if sample_uint8 > 0:
                ratio_sample = sample_comp / sample_uint8
                t_comp = int(n_cold * D * ratio_sample)
            else:
                t_comp = n_cold * D

            total_comp += t_comp
            log(f"    CRF={crf}, Table {t}: {n_frames_t} frames (sampled {len(sample_fis)}), "
                f"{n_cold * D / 1024:.0f}KB uint8 → {t_comp / 1024:.0f}KB "
                f"({n_cold * D / max(t_comp, 1):.1f}x)")

        crf_compressed_sizes[crf] = total_comp
        ratio_uint8 = total_uint8_bytes / max(total_comp, 1)
        ratio_fp32 = total_fp32_bytes / max(total_comp, 1)
        log(f"  CRF={crf}: total={total_comp / 1024**2:.1f}MB, "
            f"uint8 ratio={ratio_uint8:.1f}x, fp32 ratio={ratio_fp32:.1f}x")

    # ====================================================================
    # PHASE 4: Generate comprehensive memory table
    # ====================================================================
    log("\n" + "=" * 70)
    log("COMPREHENSIVE MEMORY BREAKDOWN")
    log("=" * 70)

    mapping_mb = (total_mapping_bytes + bitmap_bytes) / 1024**2
    hot_mb = total_hot_bytes / 1024**2

    configs = []

    # Baseline
    baseline_mb = total_fp32_bytes / 1024**2
    configs.append({
        'name': 'Baseline (fp32)',
        'hot_mb': baseline_mb, 'compressed_mb': 0,
        'decoded_mb': 0, 'mapping_mb': 0,
        'total_mb': baseline_mb,
        'storage_mb': baseline_mb,
        'crf': '-', 'cache': '-', 'auc_delta': 0,
    })

    # For each CRF × cache size combination
    for crf in crfs:
        comp_mb = crf_compressed_sizes[crf] / 1024**2

        # Config: pre-decode all accessed frames (no cache limit)
        decoded_all_mb = n_unique_frames * frame_bytes / 1024**2
        configs.append({
            'name': f'CRF={crf}, pre-decode',
            'hot_mb': hot_mb, 'compressed_mb': comp_mb,
            'decoded_mb': decoded_all_mb, 'mapping_mb': mapping_mb,
            'total_mb': hot_mb + comp_mb + decoded_all_mb + mapping_mb,
            'storage_mb': hot_mb + comp_mb,
            'crf': crf, 'cache': 'all', 'auc_delta': '-',
        })

        # Config: cache=4
        for cs in [4, 8, 16]:
            decoded_cache_mb = cs * frame_bytes / 1024**2
            configs.append({
                'name': f'CRF={crf}, cache={cs}',
                'hot_mb': hot_mb, 'compressed_mb': comp_mb,
                'decoded_mb': decoded_cache_mb, 'mapping_mb': mapping_mb,
                'total_mb': hot_mb + comp_mb + decoded_cache_mb + mapping_mb,
                'storage_mb': hot_mb + comp_mb,
                'crf': crf, 'cache': cs,
                'hit_rate': cache_results[cs]['hit_rate'],
                'auc_delta': '-',
            })

    # Print table
    log(f"\n{'Config':<30} {'Hot':>7} {'Comp':>7} {'Cache':>7} {'Map':>7} {'Total':>7} {'Storage':>7} {'HitRate':>8}")
    log("-" * 97)
    for c in configs:
        hr = f"{c.get('hit_rate', '-'):>7.1f}%" if isinstance(c.get('hit_rate'), float) else f"{'—':>8}"
        log(f"{c['name']:<30} {c['hot_mb']:>6.1f}M {c['compressed_mb']:>6.1f}M "
            f"{c['decoded_mb']:>6.1f}M {c['mapping_mb']:>6.1f}M "
            f"{c['total_mb']:>6.1f}M {c['storage_mb']:>6.1f}M {hr}")

    # ====================================================================
    # PHASE 5: AUC at key CRF levels (encode all, decode, dequant, measure)
    # ====================================================================
    log("\n" + "=" * 70)
    log("PHASE 5: AUC measurement at key CRF levels")
    log("=" * 70)

    baseline_auc = run_auc(dlrm, test_ld)
    log(f"  Baseline AUC: {baseline_auc:.6f}")

    auc_results = {'baseline': baseline_auc}

    for crf in crfs:
        log(f"\n  --- CRF={crf} ---")
        # Encode + decode all large tables
        reconstructed = {}
        for t in large_tables:
            w = state_dict[emb_keys[t]]
            q, s, zp = quantize_table_uint8(w)
            sorted_idx = table_info[t]['sorted_idx']
            q_sorted = q[sorted_idx]
            inv_perm = torch.empty_like(sorted_idx)
            inv_perm[sorted_idx] = torch.arange(len(sorted_idx))

            n = w.shape[0]
            n_frames_t = (n + rpf - 1) // rpf
            all_recon = []

            for fi in range(n_frames_t):
                start = fi * rpf
                end = min(start + rpf, n)
                chunk = q_sorted[start:end]
                actual = min(rpf, end - start)

                frame_np = pack_frame(chunk, D, rpf)
                comp_size, decoded_np = encode_frame_h265(frame_np, crf)

                if D == 16 and HAS_CPP:
                    decoded_t = torch.from_numpy(decoded_np)
                    rows_back = _C.untile_frame_to_rows(decoded_t, rpf)
                else:
                    rpr = FRAME_WIDTH // D
                    H_used = (actual + rpr - 1) // rpr
                    data = decoded_np[:H_used].reshape(-1, D)
                    rows_back = torch.from_numpy(data)

                all_recon.append(rows_back[:actual])

                if fi % 50 == 0 and fi > 0:
                    log(f"    Table {t}: encoded {fi}/{n_frames_t} frames")

            recon_sorted = torch.cat(all_recon, dim=0)
            recon_orig = recon_sorted[inv_perm]
            recon_fp32 = dequantize_uint8(recon_orig, s, zp)
            reconstructed[t] = recon_fp32
            log(f"    Table {t}: done ({n_frames_t} frames)")

        # Replace weights, measure AUC
        orig = {}
        for t in large_tables:
            orig[t] = dlrm.emb_l[t].weight.data.clone()
            dlrm.emb_l[t].weight.data = reconstructed[t]

        auc = run_auc(dlrm, test_ld)
        delta = auc - baseline_auc
        auc_results[crf] = {'auc': auc, 'delta': delta}
        log(f"  CRF={crf}: AUC={auc:.6f}, delta={delta:+.6f} ({delta*100:+.4f}%)")

        # Restore
        for t in large_tables:
            dlrm.emb_l[t].weight.data = orig[t]

    # ====================================================================
    # FINAL: Save results
    # ====================================================================
    results = {
        'dataset': dataset_name,
        'D': D,
        'tiling': tiling,
        'rpf': rpf,
        'large_tables': large_tables,
        'table_info': {str(t): {
            'n_rows': table_info[t]['n_rows'],
            'n_hot': table_info[t]['n_hot'],
            'n_cold': table_info[t]['n_cold'],
            'n_frames': table_info[t]['n_frames'],
            'accessed_frames': len(table_info[t]['accessed_frames']),
            'n_accessed': table_info[t]['n_accessed'],
        } for t in large_tables},
        'cache_simulation': {str(cs): cache_results[cs] for cs in cache_sizes},
        'unique_frames': n_unique_frames,
        'total_frames': total_frames,
        'memory': {
            'total_fp32_mb': total_fp32_bytes / 1024**2,
            'total_cold_uint8_mb': total_uint8_bytes / 1024**2,
            'hot_fp32_mb': total_hot_bytes / 1024**2,
            'mapping_mb': mapping_mb,
            'frame_bytes': frame_bytes,
        },
        'crf_compressed_mb': {str(crf): crf_compressed_sizes[crf] / 1024**2 for crf in crfs},
        'auc': auc_results,
        'configs': configs,
    }

    out_path = f"results/crf_cache_memory/{dataset_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {out_path}")

    return results


def main():
    log("=" * 70)
    log("CRF SWEEP + CACHE SIMULATION + MEMORY BREAKDOWN")
    log("=" * 70)

    all_results = {}

    # Run Kaggle first (faster)
    dataset = sys.argv[1] if len(sys.argv) > 1 else "both"

    if dataset in ("kaggle", "both"):
        dlrm, test_ld, ln_emb, state_dict, D = load_dataset("kaggle")
        all_results['kaggle'] = profile_and_simulate("kaggle", dlrm, test_ld, ln_emb, state_dict, D)
        del dlrm, test_ld  # free memory

    if dataset in ("terabyte", "both"):
        dlrm, test_ld, ln_emb, state_dict, D = load_dataset("terabyte")
        all_results['terabyte'] = profile_and_simulate("terabyte", dlrm, test_ld, ln_emb, state_dict, D)

    log("\n" + "=" * 70)
    log("ALL DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()
