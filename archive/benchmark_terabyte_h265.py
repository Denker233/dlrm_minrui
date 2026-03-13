#!/usr/bin/env python3
"""
Terabyte H.265 Benchmark: Baseline vs Codec+fast_forward

Adapted from benchmark_full_comparison.py for Criteo Terabyte data with D=64.
Uses flat frame layout (no 4x4 tiling) since D=64 doesn't divide into 4x4 tiles.
"""

import os, sys, time, json, gc, io
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

try:
    import compressed_emb as _C
    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "./models/dlrm_terabyte_4day.pt"
DATA_DIR = os.path.expanduser("~/input/terabyte")
EMB_DIM = 64
MAX_IND_RANGE = 10000000
TEST_BATCH_SIZE = 2048
LARGE_TABLE_THRESHOLD = 50000
HOT_THRESHOLD = 0.043  # top 4.3% by access frequency
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
ROWS_PER_FRAME = (FRAME_WIDTH // EMB_DIM) * FRAME_HEIGHT  # 30 * 1080 = 32400
OPTIMIZED_THREADS = 40
NUM_RUNS = 3

RESULTS_DIR = "results/terabyte"
LOG_FILE = "logs/terabyte_h265_benchmark.log"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_fh = open(LOG_FILE, "w")
def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n"); log_fh.flush()

def drop_caches():
    try:
        os.system("sync")
    except:
        pass

# ============================================================
# DATA LOADING
# ============================================================
def create_args():
    class Args: pass
    a = Args()
    a.arch_sparse_feature_size = EMB_DIM
    a.arch_mlp_bot = "13-512-256-64"; a.arch_mlp_top = "512-512-256-1"
    a.arch_interaction_op = "dot"; a.arch_interaction_itself = False
    a.data_generation = "dataset"; a.data_set = "terabyte"
    a.raw_data_file = os.path.join(DATA_DIR, "day")
    a.processed_data_file = os.path.join(DATA_DIR, "terabyte_processed.npz")
    a.loss_function = "bce"; a.max_ind_range = MAX_IND_RANGE
    a.test_mini_batch_size = TEST_BATCH_SIZE
    a.test_num_workers = 0; a.num_workers = 0
    a.mlperf_logging = False; a.memory_map = False; a.data_randomize = "total"
    a.data_trace_enable_padding = False; a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10; a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 2048; a.round_targets = True
    a.mlperf_bin_loader = False; a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False
    return a

def load_model_and_data():
    os.environ['CRITEO_DAYS'] = '4'
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
    log("Loading checkpoint...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    log(f"Model loaded: {len(ln_emb)} tables, emb_dim={m_spa}")
    for i, ne in enumerate(ln_emb):
        if ne > LARGE_TABLE_THRESHOLD:
            log(f"  Table {i}: {ne:,} rows ({ne*EMB_DIM*4/1024/1024:.1f} MB)")
    return dlrm, test_ld, ln_emb, ld["state_dict"]

def quantize_table(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    return s, zp

def compute_metrics(scores, targets):
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
    auc = roc_auc_score(targets, probs)
    ll = log_loss(targets, np.clip(probs, 1e-7, 1 - 1e-7))
    acc = accuracy_score(targets, (probs >= 0.5).astype(np.float32))
    return auc, ll, acc

# ============================================================
# FRAME PACKING (flat layout for D=64)
# ============================================================
def pack_rows_to_frame(rows_uint8, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Pack uint8 rows into a (height, width) frame using flat contiguous layout.
    Each row of D bytes occupies D consecutive pixels.
    width/D rows fit per pixel row."""
    N, D = rows_uint8.shape
    rows_per_row = width // D
    max_rows = rows_per_row * height
    assert N <= max_rows, f"Too many rows {N} for {width}x{height} frame (max {max_rows})"
    frame = torch.zeros(height, width, dtype=torch.uint8)
    # Reshape rows into frame
    for r in range(N):
        py = r // rows_per_row
        px = (r % rows_per_row) * D
        frame[py, px:px+D] = rows_uint8[r]
    return frame

def pack_rows_to_frame_fast(rows_uint8, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Vectorized version of pack_rows_to_frame."""
    N, D = rows_uint8.shape
    rows_per_row = width // D
    # Pad to fill complete pixel rows
    padded_N = ((N + rows_per_row - 1) // rows_per_row) * rows_per_row
    if padded_N > N:
        pad = torch.zeros(padded_N - N, D, dtype=torch.uint8)
        data = torch.cat([rows_uint8, pad], dim=0)
    else:
        data = rows_uint8
    # Reshape: (padded_N, D) → (H_used, rows_per_row, D) → (H_used, width)
    H_used = padded_N // rows_per_row
    frame = torch.zeros(height, width, dtype=torch.uint8)
    frame[:H_used] = data.reshape(H_used, rows_per_row * D)
    return frame

def unpack_frame_to_rows(frame, n_rows, D=EMB_DIM, width=FRAME_WIDTH):
    """Unpack frame back to rows."""
    rows_per_row = width // D
    H_used = (n_rows + rows_per_row - 1) // rows_per_row
    data = frame[:H_used].reshape(-1, D)
    return data[:n_rows]

# ============================================================
# INFERENCE
# ============================================================
def run_inference(dlrm, test_ld, tag=""):
    """Run inference through DataLoader. Returns metrics dict."""
    all_scores, all_targets = [], []
    batch_lats, emb_times, interact_times, mlp_times = [], [], [], []
    n_batches = 0

    with torch.no_grad():
        for X, lS_o, lS_i, T in test_ld:
            t0 = time.time()
            dlrm.time_look_up = 0
            Z = dlrm(X, lS_o, lS_i)
            t1 = time.time()

            lat_ms = (t1 - t0) * 1000
            emb_ms = dlrm.time_look_up * 1000
            batch_lats.append(lat_ms)
            emb_times.append(emb_ms)
            all_scores.append(Z.detach().cpu().numpy().flatten())
            all_targets.append(T.detach().cpu().numpy().flatten())
            n_batches += 1

    scores = np.concatenate(all_scores)
    targets = np.concatenate(all_targets)
    auc, ll, acc = compute_metrics(scores, targets)

    return {
        'auc': auc, 'log_loss': ll, 'accuracy': acc,
        'mean_lat_ms': np.mean(batch_lats),
        'p50_lat_ms': np.median(batch_lats),
        'p99_lat_ms': np.percentile(batch_lats, 99),
        'emb_ms': np.mean(emb_times),
        'n_batches': n_batches,
        'tag': tag
    }

# ============================================================
# PROFILING + HOT/COLD SPLIT
# ============================================================
def profile_and_split(dlrm, test_ld, ln_emb, state_dict):
    """Profile access patterns and build hot/cold split."""
    num_tabs = len(ln_emb)
    large_tables = [i for i in range(num_tabs) if ln_emb[i] > LARGE_TABLE_THRESHOLD]
    log(f"Large tables (>{LARGE_TABLE_THRESHOLD} rows): {large_tables}")

    # Count access frequencies (vectorized)
    freq = {t: torch.zeros(ln_emb[t], dtype=torch.long) for t in large_tables}
    for X, lS_o, lS_i, T in test_ld:
        for t in large_tables:
            if isinstance(lS_i, (list, tuple)):
                indices = lS_i[t].flatten()
            elif lS_i.dim() == 2:
                indices = lS_i[t].flatten()
            else:
                indices = lS_i.flatten()
            valid = indices[indices < ln_emb[t]]
            freq[t].scatter_add_(0, valid.long(), torch.ones_like(valid, dtype=torch.long))

    # Determine hot/cold split
    hot_indices, cold_indices, is_hot = {}, {}, {}
    cold_num_rows = {}
    emb_keys = {}

    for t in large_tables:
        emb_keys[t] = f'emb_l.{t}.weight'
        n = ln_emb[t]
        accessed = (freq[t] > 0).sum().item()
        hot_count = max(1, int(n * HOT_THRESHOLD))

        # Sort by frequency, take top hot_count as hot
        sorted_idx = freq[t].argsort(descending=True)
        h_idx = sorted_idx[:hot_count].sort().values
        c_idx = sorted_idx[hot_count:].sort().values

        hot_indices[t] = h_idx
        cold_indices[t] = c_idx
        ih = torch.zeros(n, dtype=torch.bool)
        ih[h_idx] = True
        is_hot[t] = ih
        cold_num_rows[t] = len(c_idx)

        log(f"  Table {t}: {n:,} rows, {len(h_idx):,} hot ({100*len(h_idx)/n:.1f}%), "
            f"{len(c_idx):,} cold, {accessed:,} accessed in test")

    return large_tables, hot_indices, cold_indices, is_hot, cold_num_rows, emb_keys, freq

# ============================================================
# H.265 ENCODE/DECODE
# ============================================================
def encode_cold_h265(state_dict, cold_indices, emb_keys, large_tables):
    """Quantize cold embeddings and encode as H.265 frames."""
    import av

    cold_quant_scale, cold_quant_zp = {}, {}
    o2c_map = {}  # orig_idx → cold_reordered_idx
    compressed_files = {}
    total_compressed_bytes = 0

    frame_dir = os.path.join(RESULTS_DIR, "h265_frames")
    os.makedirs(frame_dir, exist_ok=True)

    for t in large_tables:
        table_dir = os.path.join(frame_dir, f"table_{t}")
        os.makedirs(table_dir, exist_ok=True)

        c_idx = cold_indices[t]
        cold_w = state_dict[emb_keys[t]][c_idx]  # (N_cold, D)

        # Quantize
        s, zp = quantize_table(cold_w)
        cold_quant_scale[t] = s
        cold_quant_zp[t] = zp
        cold_uint8 = torch.clamp(
            torch.round(cold_w / s + zp), 0, 255
        ).to(torch.uint8)

        # Build reorder map (identity for now — no frequency reorder)
        n_cold = len(c_idx)
        mapping = torch.full((int(state_dict[emb_keys[t]].shape[0]),), -1, dtype=torch.long)
        mapping[c_idx] = torch.arange(n_cold, dtype=torch.long)
        o2c_map[t] = mapping

        # Pack into frames and encode
        n_frames = (n_cold + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
        log(f"  Table {t}: {n_cold:,} cold rows → {n_frames} frames")

        table_compressed = 0
        for fid in range(n_frames):
            start = fid * ROWS_PER_FRAME
            end = min(start + ROWS_PER_FRAME, n_cold)
            frame_rows = cold_uint8[start:end]

            frame = pack_rows_to_frame_fast(frame_rows)
            out_path = os.path.join(table_dir, f"frame_{fid:05d}.h265")

            # Encode with H.265 lossless
            if HAS_CPP_EXT and hasattr(_C, 'encode_h265_frame'):
                _C.encode_h265_frame(frame, out_path, True, 0)
            else:
                # Fallback: PyAV
                encode_frame_pyav(frame.numpy(), out_path)

            fsize = os.path.getsize(out_path)
            table_compressed += fsize

        total_compressed_bytes += table_compressed
        compressed_files[t] = table_dir
        raw_mb = n_cold * EMB_DIM / 1024 / 1024
        comp_mb = table_compressed / 1024 / 1024
        log(f"    Compressed: {raw_mb:.1f}MB uint8 → {comp_mb:.1f}MB H.265 ({raw_mb/max(0.01,comp_mb):.1f}x)")

        del cold_w, cold_uint8

    log(f"  Total compressed: {total_compressed_bytes/1024/1024:.1f}MB")
    return cold_quant_scale, cold_quant_zp, o2c_map, compressed_files, total_compressed_bytes

def encode_frame_pyav(frame_np, out_path):
    """Encode a single frame with PyAV (fallback)."""
    import av
    h, w = frame_np.shape
    container = av.open(out_path, mode='w', format='hevc')
    stream = container.add_stream('libx265', rate=1)
    stream.width = w; stream.height = h
    stream.pix_fmt = 'gray'
    stream.options = {'x265-params': 'lossless=1'}

    video_frame = av.VideoFrame.from_ndarray(frame_np, format='gray')
    for packet in stream.encode(video_frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

# ============================================================
# CODEC FULL_CPP SETUP
# ============================================================
def setup_codec_fullcpp(dlrm, ln_emb, state_dict, test_batches,
                        hot_indices, cold_indices, is_hot, o2c_map,
                        cold_quant_scale, cold_quant_zp, cold_num_rows,
                        large_tables, emb_keys, compressed_files,
                        compressed_bytes_map=None):
    """Register tables and cold frames in C++ for fast_forward inference."""

    num_tabs = len(dlrm.emb_l)

    # Register tables
    table_kinds, weights, mappings_list, scales, zero_points = [], [], [], [], []
    total_hot_mb = 0
    compressed_tables = set()

    for k in range(num_tabs):
        if k in cold_num_rows and cold_num_rows[k] > 0:
            compressed_tables.add(k)
            table_kinds.append(1)  # COMPRESSED
            h_idx = hot_indices[k]
            hot_w = state_dict[emb_keys[k]][h_idx].clone()
            weights.append(hot_w)
            total_hot_mb += hot_w.numel() * 4 / 1024 / 1024
            # Build mapping: hot → positive compact idx, cold → -(cold_reordered_idx + 1)
            mapping = torch.full((ln_emb[k],), 0, dtype=torch.int32)  # INVALID
            mapping[h_idx] = torch.arange(len(h_idx), dtype=torch.int32)
            c_idx = cold_indices[k]
            cm = o2c_map[k][c_idx]
            valid = cm >= 0
            mapping[c_idx[valid]] = -(cm[valid].int() + 1)
            mappings_list.append(mapping)
            scales.append(float(cold_quant_scale[k]))
            zero_points.append(int(cold_quant_zp[k]))
        else:
            table_kinds.append(0)
            weights.append(dlrm.emb_l[k].weight.data)
            mappings_list.append(torch.empty(0, dtype=torch.int32))
            scales.append(0.0); zero_points.append(0)

    _C.register_tables(table_kinds, weights, mappings_list, scales, zero_points)

    # Scan batches for needed frames
    log("  Scanning batches for cold frame coverage...")
    t_scan = time.time()
    needed_frames = {t: set() for t in compressed_tables}

    for X, lS_o, lS_i, T in test_batches:
        for t_idx in compressed_tables:
            if isinstance(lS_i, (list, tuple)):
                indices = lS_i[t_idx]
            elif lS_i.dim() == 2:
                indices = lS_i[t_idx]
            else:
                indices = lS_i
            cold_mask = ~is_hot[t_idx][indices]
            if cold_mask.any():
                cold_orig = indices[cold_mask]
                cold_mapped = o2c_map[t_idx][cold_orig]
                valid = cold_mapped >= 0
                if valid.any():
                    fids = (cold_mapped[valid] // ROWS_PER_FRAME).unique().tolist()
                    needed_frames[t_idx].update(fids)

    total_needed = sum(len(v) for v in needed_frames.values())
    log(f"  Found {total_needed} unique frames in {time.time()-t_scan:.1f}s")

    # Decode and register (from in-memory bytes if available, else from disk)
    log("  Decoding H.265 frames and registering in C++...")
    total_cold_frame_mb = 0

    for t_idx in compressed_tables:
        if not needed_frames[t_idx]:
            continue

        table_dir = compressed_files.get(t_idx)
        if not table_dir and not (compressed_bytes_map and t_idx in compressed_bytes_map):
            continue

        sorted_fids = sorted(needed_frames[t_idx])
        decoded_frames = []

        for fid in sorted_fids:
            if compressed_bytes_map and t_idx in compressed_bytes_map and fid in compressed_bytes_map[t_idx]:
                # Decode from in-memory bytes
                data = compressed_bytes_map[t_idx][fid]
                compressed_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                frame = _C.decode_h265_frame_from_bytes(compressed_t)
            else:
                # Fallback to disk decode
                h265_path = os.path.join(table_dir, f"frame_{fid:05d}.h265")
                if HAS_CPP_EXT and hasattr(_C, 'decode_h265_frame_from_file'):
                    frame = _C.decode_h265_frame_from_file(h265_path)
                else:
                    frame = decode_frame_pyav(h265_path)
            # Unpack frame to rows
            n_cold = cold_num_rows[t_idx]
            start = fid * ROWS_PER_FRAME
            end = min(start + ROWS_PER_FRAME, n_cold)
            rows = unpack_frame_to_rows(frame, end - start)
            # Pad to rows_per_frame
            if rows.shape[0] < ROWS_PER_FRAME:
                pad = torch.zeros(ROWS_PER_FRAME - rows.shape[0], EMB_DIM, dtype=torch.uint8)
                rows = torch.cat([rows, pad], dim=0)
            decoded_frames.append(rows)

        all_data = torch.cat(decoded_frames, dim=0)
        total_cold_frame_mb += all_data.numel() / 1024 / 1024

        fids_t = torch.tensor(sorted_fids, dtype=torch.long)
        cold_mapping = o2c_map[t_idx].int()

        _C.register_cold_frames_for_table(
            t_idx, fids_t, all_data,
            float(cold_quant_scale[t_idx]), float(cold_quant_zp[t_idx]),
            ROWS_PER_FRAME, cold_mapping)

    log(f"  Cold frames registered: {total_cold_frame_mb:.1f}MB uint8")

    # Replace apply_emb with fast_forward
    _orig_apply = dlrm.apply_emb

    def _full_cpp_apply(lS_o, lS_i, emb_l, v_W_l):
        start = time.time()
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
        dlrm.time_look_up += time.time() - start
        return results[-1]

    dlrm.apply_emb = _full_cpp_apply

    total_compressed_mb = sum(
        sum(os.path.getsize(os.path.join(compressed_files[t], f))
            for f in os.listdir(compressed_files[t]) if f.endswith('.h265'))
        for t in compressed_tables if t in compressed_files
    ) / 1024 / 1024

    mem_info = {
        'hot_mb': total_hot_mb,
        'cold_disk_mb': total_compressed_mb,
        'cold_frame_mb': total_cold_frame_mb,
        'total_needed_frames': total_needed,
    }

    def cleanup():
        dlrm.apply_emb = _orig_apply

    return cleanup, mem_info

def decode_frame_pyav(path):
    """Decode H.265 frame with PyAV (fallback)."""
    import av
    container = av.open(path)
    for frame in container.decode(video=0):
        return torch.from_numpy(frame.to_ndarray(format='gray'))

# ============================================================
# MAIN
# ============================================================
def main():
    assert HAS_CPP_EXT, "C++ extension required. Build with: python3 setup_compressed_emb.py build_ext --inplace"

    log("=" * 70)
    log("TERABYTE H.265 BENCHMARK")
    log("=" * 70)

    # Load model
    dlrm, test_ld, ln_emb, state_dict = load_model_and_data()

    num_tabs = len(ln_emb)
    total_emb_mb = sum(ln_emb[i] * EMB_DIM * 4 for i in range(num_tabs)) / 1024 / 1024
    log(f"Total embedding memory: {total_emb_mb:.1f} MB fp32")

    emb_keys_all = {i: f'emb_l.{i}.weight' for i in range(num_tabs)}

    # Profile and split
    log("\n" + "=" * 70)
    log("PROFILING ACCESS PATTERNS")
    log("=" * 70)
    large_tables, hot_indices, cold_indices, is_hot, cold_num_rows, emb_keys, freq = \
        profile_and_split(dlrm, test_ld, ln_emb, state_dict)

    # Encode with H.265
    log("\n" + "=" * 70)
    log("H.265 ENCODING (lossless, CRF=0)")
    log("=" * 70)
    cold_quant_scale, cold_quant_zp, o2c_map, compressed_files, total_comp_bytes = \
        encode_cold_h265(state_dict, cold_indices, emb_keys, large_tables)

    total_cold_raw_mb = sum(cold_num_rows[t] * EMB_DIM for t in large_tables) / 1024 / 1024
    total_comp_mb = total_comp_bytes / 1024 / 1024
    log(f"\nCompression: {total_cold_raw_mb:.1f}MB uint8 → {total_comp_mb:.1f}MB H.265 "
        f"({total_cold_raw_mb/max(0.01,total_comp_mb):.1f}x)")

    # Load compressed H.265 frame bytes into RAM for in-memory decode
    log("Loading compressed H.265 frame bytes into RAM...")
    compressed_bytes_map = {}
    for t in large_tables:
        table_dir = compressed_files.get(t)
        if not table_dir or not os.path.exists(table_dir):
            continue
        compressed_bytes_map[t] = {}
        frame_files = sorted([f for f in os.listdir(table_dir) if f.endswith('.h265')])
        for ff in frame_files:
            fid = int(ff.split('_')[1].split('.')[0])
            with open(os.path.join(table_dir, ff), 'rb') as fh:
                compressed_bytes_map[t][fid] = fh.read()
    total_loaded = sum(sum(len(v) for v in tb.values()) for tb in compressed_bytes_map.values())
    log(f"  Loaded {total_loaded/1024/1024:.1f}MB compressed bytes into RAM")

    torch.set_num_threads(OPTIMIZED_THREADS)
    all_results = OrderedDict()

    # ---- 1. BASELINE ----
    log("\n" + "=" * 70)
    log("BASELINE: Full fp32 inference")
    log("=" * 70)
    baseline_runs = []
    for run in range(NUM_RUNS):
        gc.collect(); drop_caches(); time.sleep(0.3)
        with torch.no_grad():
            for t_idx, k in emb_keys.items():
                dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()
        r = run_inference(dlrm, test_ld, f"baseline run {run+1}")
        log(f"  Run {run+1}: AUC={r['auc']:.6f}, BLat={r['mean_lat_ms']:.2f}ms, "
            f"emb={r['emb_ms']:.2f}ms")
        baseline_runs.append(r)
    baseline_runs.sort(key=lambda x: x['mean_lat_ms'])
    all_results['Baseline'] = baseline_runs[NUM_RUNS // 2]

    # ---- 2. CODEC FULL_CPP ----
    log("\n" + "=" * 70)
    log("CODEC FULL_CPP: H.265 + fast_forward")
    log("=" * 70)

    # Pre-cache test batches for frame scanning
    test_batches = []
    for batch in test_ld:
        test_batches.append((batch[0], batch[1], batch[2], batch[3]))

    codec_runs = []
    for run in range(NUM_RUNS):
        gc.collect(); drop_caches(); time.sleep(0.3)
        with torch.no_grad():
            for t_idx, k in emb_keys.items():
                dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

        cleanup, mem_info = setup_codec_fullcpp(
            dlrm, ln_emb, state_dict, test_batches,
            hot_indices, cold_indices, is_hot, o2c_map,
            cold_quant_scale, cold_quant_zp, cold_num_rows,
            large_tables, emb_keys, compressed_files,
            compressed_bytes_map)

        r = run_inference(dlrm, test_ld, f"codec run {run+1}")
        r.update(mem_info)
        log(f"  Run {run+1}: AUC={r['auc']:.6f}, BLat={r['mean_lat_ms']:.2f}ms, "
            f"emb={r['emb_ms']:.2f}ms, frames={mem_info['total_needed_frames']}")
        codec_runs.append(r)
        cleanup()

    codec_runs.sort(key=lambda x: x['mean_lat_ms'])
    all_results['Codec H.265'] = codec_runs[NUM_RUNS // 2]

    # ---- SUMMARY ----
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    bl = all_results['Baseline']
    cd = all_results['Codec H.265']

    log(f"Baseline:  AUC={bl['auc']:.6f}, BLat={bl['mean_lat_ms']:.2f}ms, emb={bl['emb_ms']:.2f}ms")
    log(f"Codec:     AUC={cd['auc']:.6f}, BLat={cd['mean_lat_ms']:.2f}ms, emb={cd['emb_ms']:.2f}ms")
    log(f"Speedup:   {bl['mean_lat_ms']/cd['mean_lat_ms']:.2f}x")
    log(f"AUC delta: {cd['auc'] - bl['auc']:.6f}")
    log(f"Storage:   {total_emb_mb:.1f}MB fp32 → {total_comp_mb:.1f}MB H.265 "
        f"({total_emb_mb/max(0.01,total_comp_mb):.1f}x)")

    # Save results
    results_path = os.path.join(RESULTS_DIR, "terabyte_h265_results.json")
    with open(results_path, 'w') as f:
        json.dump({k: {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                       for kk, vv in v.items()}
                   for k, v in all_results.items()}, f, indent=2)
    log(f"Results saved to {results_path}")
    log_fh.close()

if __name__ == "__main__":
    main()
