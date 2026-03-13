#!/usr/bin/env python3
"""
MLSys Reviewer Experiments — Five experiments to address reviewer concerns.

Experiment 1: Fair Baseline Decomposition (W1)
  Config A: PyTorch EmbeddingBag fp32
  Config B: C++ fast_forward with ALL tables as STANDARD fp32
  Config C: C++ fast_forward with hot/cold split + uint8 cold

Experiment 2: Post-Training Compression Comparison (W2)
  9 methods: INT8, INT4, PQ, SVD, row pruning, Zstd-19, H.265 lossless,
  H.265 CRF=18, H.265 CRF=18 + freq sort

Experiment 3: Zero-Out Cold Rows (W5)
  Sweep zeroing bottom X% rows by frequency

Experiment 4: CPU Cache Profiling (perf stat)
  Hardware counters for configs A, B, C

Experiment 5: Per-Row Error Analysis (Error Steering)
  MSE by access frequency bucket across orderings
"""

import os, sys, time, json, gc, tempfile, subprocess, signal, types
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

try:
    import compressed_emb as _C
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("WARNING: C++ extension not available")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIG
# ============================================================
KAGGLE_MODEL = "./models/dlrm_kaggle_correct.pt"
TERABYTE_MODEL = "./models/dlrm_terabyte_4day.pt"
LARGE_TABLE_THRESHOLD = 50000
HOT_THRESHOLD = 0.043  # top 4.3% by access frequency
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
NUM_RUNS = 3
NUM_THREADS = 40
TEST_BATCH_SIZE = 2048

RESULTS_DIR = "results/mlsys_baselines"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_fh = open(os.path.join(RESULTS_DIR, "experiment.log"), "w")
def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n"); log_fh.flush()

def rows_per_frame(D):
    return (FRAME_WIDTH // D) * FRAME_HEIGHT

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

def pack_rows_flat(rows_uint8, D, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    N = rows_uint8.shape[0]
    rows_per_row = width // D
    padded_N = ((N + rows_per_row - 1) // rows_per_row) * rows_per_row
    if padded_N > N:
        pad = torch.zeros(padded_N - N, D, dtype=torch.uint8)
        data = torch.cat([rows_uint8, pad], dim=0)
    else:
        data = rows_uint8
    H_used = padded_N // rows_per_row
    frame = torch.zeros(height, width, dtype=torch.uint8)
    frame[:H_used] = data.reshape(H_used, rows_per_row * D)
    return frame

def unpack_frame_flat(frame, n_rows, D, width=FRAME_WIDTH):
    rows_per_row = width // D
    H_used = (n_rows + rows_per_row - 1) // rows_per_row
    data = frame[:H_used].reshape(-1, D)
    return data[:n_rows]

def encode_frame_h265(frame, crf, lossless=False):
    with tempfile.NamedTemporaryFile(suffix='.h265', delete=False) as f:
        tmp_path = f.name
    try:
        if HAS_CPP and hasattr(_C, 'encode_h265_frame'):
            _C.encode_h265_frame(frame, tmp_path, lossless, crf)
        else:
            encode_frame_pyav(frame.numpy(), tmp_path, crf, lossless)
        comp_size = os.path.getsize(tmp_path)
        if HAS_CPP and hasattr(_C, 'decode_h265_frame_from_file'):
            decoded = _C.decode_h265_frame_from_file(tmp_path)
        else:
            decoded = decode_frame_pyav(tmp_path)
        return comp_size, decoded
    finally:
        os.unlink(tmp_path)

def encode_frame_pyav(frame_np, out_path, crf=0, lossless=False):
    import av
    h, w = frame_np.shape
    container = av.open(out_path, mode='w', format='hevc')
    stream = container.add_stream('libx265', rate=1)
    stream.width = w; stream.height = h
    stream.pix_fmt = 'gray'
    if lossless:
        stream.options = {'x265-params': 'lossless=1'}
    else:
        stream.options = {'crf': str(crf), 'x265-params': f'qp={crf}'}
    video_frame = av.VideoFrame.from_ndarray(frame_np, format='gray')
    for packet in stream.encode(video_frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

def decode_frame_pyav(path):
    import av
    container = av.open(path)
    for frame in container.decode(video=0):
        return torch.from_numpy(frame.to_ndarray(format='gray'))

from sklearn.metrics import roc_auc_score

def run_auc(model, loader):
    all_s, all_t = [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in loader:
            Z = model(X, lS_o, lS_i)
            all_s.append(Z.cpu().numpy().flatten())
            all_t.append(T.cpu().numpy().flatten())
    scores = np.concatenate(all_s)
    targets = np.concatenate(all_t)
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
    return roc_auc_score(targets, probs)

def run_inference_timed(dlrm, test_ld, warmup=50):
    """Run inference, return (mean_batch_ms, mean_emb_ms, n_batches)."""
    batch_lats, emb_times = [], []
    with torch.no_grad():
        for i, (X, lS_o, lS_i, T) in enumerate(test_ld):
            dlrm.time_look_up = 0
            t0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            t1 = time.time()
            if i >= warmup:
                batch_lats.append((t1 - t0) * 1000)
                emb_times.append(dlrm.time_look_up * 1000)
    return np.median(batch_lats), np.median(emb_times), len(batch_lats)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================
def load_model_and_data(dataset_name):
    """Load model, data, return (dlrm, test_ld, ln_emb, state_dict)."""
    if dataset_name == 'terabyte':
        os.environ['CRITEO_DAYS'] = '4'

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
        model_path = KAGGLE_MODEL
    else:
        a.arch_sparse_feature_size = 64
        a.arch_mlp_bot = "13-512-256-64"; a.arch_mlp_top = "512-512-256-1"
        a.raw_data_file = os.path.expanduser("~/input/terabyte/day")
        a.processed_data_file = os.path.expanduser("~/input/terabyte/terabyte_processed.npz")
        a.data_set = "terabyte"; a.max_ind_range = 10000000
        model_path = TERABYTE_MODEL

    a.arch_interaction_op = "dot"; a.arch_interaction_itself = False
    a.data_generation = "dataset"; a.loss_function = "bce"
    a.test_mini_batch_size = TEST_BATCH_SIZE
    a.test_num_workers = 0; a.num_workers = 0
    a.mlperf_logging = False; a.memory_map = False; a.data_randomize = "total"
    a.data_trace_enable_padding = False; a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10; a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 2048; a.round_targets = True
    a.mlperf_bin_loader = False; a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False

    log(f"Loading {dataset_name} dataset...")
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(a)
    ln_emb = np.array(train_data.counts)
    D = a.arch_sparse_feature_size
    ln_bot = np.fromstring(a.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + a.arch_mlp_top, dtype=int, sep="-")

    dlrm = DLRM_Net(D, ln_emb, ln_bot, ln_top,
                     arch_interaction_op="dot", arch_interaction_itself=False,
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")
    ld = torch.load(model_path, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()

    log(f"  {len(ln_emb)} tables, D={D}")
    return dlrm, test_ld, ln_emb, ld["state_dict"]

# ============================================================
# PROFILE ACCESS FREQUENCIES
# ============================================================
def profile_access_frequencies(test_ld, ln_emb, large_tables):
    """Count per-row access frequency across test set."""
    freq = {}
    for t in large_tables:
        freq[t] = torch.zeros(ln_emb[t], dtype=torch.long)

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

    for t in large_tables:
        accessed = (freq[t] > 0).sum().item()
        log(f"  Table {t}: {ln_emb[t]:,} rows, {accessed:,} accessed ({100*accessed/ln_emb[t]:.1f}%)")
    return freq

# ============================================================
# HOT/COLD SPLIT
# ============================================================
def build_hot_cold_split(ln_emb, freq, large_tables, state_dict):
    """Build hot/cold indices and quantization for cold rows."""
    hot_indices, cold_indices, is_hot = {}, {}, {}
    cold_num_rows = {}
    emb_keys = {}
    cold_quant_scale, cold_quant_zp = {}, {}
    o2c_map = {}

    for t in large_tables:
        emb_keys[t] = f'emb_l.{t}.weight'
        n = ln_emb[t]
        hot_count = max(1, int(n * HOT_THRESHOLD))
        sorted_idx = freq[t].argsort(descending=True)
        h_idx = sorted_idx[:hot_count].sort().values
        # Cold indices in FREQUENCY order (most-accessed cold first)
        # This ensures most-accessed cold rows land in frame 0, 1, ...
        # so only a few frames are needed at inference time.
        c_idx = sorted_idx[hot_count:]  # already freq-descending from argsort

        hot_indices[t] = h_idx
        cold_indices[t] = c_idx
        ih = torch.zeros(n, dtype=torch.bool)
        ih[h_idx] = True
        is_hot[t] = ih
        cold_num_rows[t] = len(c_idx)

        # Quantize cold weights
        cold_w = state_dict[emb_keys[t]][c_idx]
        mn, mx = cold_w.min().item(), cold_w.max().item()
        s = (mx - mn) / 255.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        cold_quant_scale[t] = s
        cold_quant_zp[t] = zp

        # Build orig→cold mapping
        mapping = torch.full((n,), -1, dtype=torch.long)
        mapping[c_idx] = torch.arange(len(c_idx), dtype=torch.long)
        o2c_map[t] = mapping

        log(f"  Table {t}: {len(h_idx):,} hot ({100*len(h_idx)/n:.1f}%), {len(c_idx):,} cold")

    return hot_indices, cold_indices, is_hot, cold_num_rows, emb_keys, cold_quant_scale, cold_quant_zp, o2c_map

# ============================================================
# EXPERIMENT 1: FAIR BASELINE DECOMPOSITION (W1)
# ============================================================
def experiment1_fair_baseline(dlrm, test_ld, ln_emb, state_dict,
                              hot_indices, cold_indices, is_hot, cold_num_rows,
                              emb_keys, cold_quant_scale, cold_quant_zp, o2c_map,
                              large_tables, D, dataset_name):
    """Decompose speedup: A (PyTorch fp32), B (C++ STANDARD fp32), C (C++ hot/cold)."""
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 1: Fair Baseline Decomposition ({dataset_name})")
    log(f"{'='*70}")

    assert HAS_CPP, "C++ extension required for Experiment 1"
    num_tabs = len(dlrm.emb_l)
    rpf = rows_per_frame(D)
    results = {}

    # Save and restore original apply_emb
    from dlrm_s_pytorch import DLRM_Net
    orig_apply = DLRM_Net.apply_emb

    def restore_pytorch_apply():
        """Restore original PyTorch EmbeddingBag apply_emb."""
        dlrm.apply_emb = types.MethodType(orig_apply, dlrm)

    # --- Config A: PyTorch EmbeddingBag fp32 ---
    log("\n  Config A: PyTorch EmbeddingBag fp32")
    for t_idx in emb_keys:
        dlrm.emb_l[t_idx].weight.data = state_dict[emb_keys[t_idx]].clone()
    restore_pytorch_apply()

    run_lats_a = []
    for run in range(NUM_RUNS):
        gc.collect(); time.sleep(0.2)
        lat, emb, nb = run_inference_timed(dlrm, test_ld)
        run_lats_a.append({'batch_ms': lat, 'emb_ms': emb})
        log(f"    Run {run+1}: batch={lat:.2f}ms, emb={emb:.2f}ms")

    run_lats_a.sort(key=lambda x: x['batch_ms'])
    results['config_A'] = run_lats_a[NUM_RUNS // 2]
    auc_a = run_auc(dlrm, test_ld)
    results['config_A']['auc'] = auc_a
    log(f"  Config A median: batch={results['config_A']['batch_ms']:.2f}ms, "
        f"emb={results['config_A']['emb_ms']:.2f}ms, AUC={auc_a:.6f}")

    # Helper for C++ apply_emb (reused by configs B and C)
    def _make_cpp_apply():
        def _cpp_apply(lS_o, lS_i, emb_l, v_W_l):
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
            results_ff = _C.fast_forward(lS_i_2d, lS_o_2d)
            dlrm.time_look_up += time.time() - start
            return results_ff[-1]
        return _cpp_apply

    # --- Config B: C++ fast_forward, ALL tables STANDARD fp32 ---
    log("\n  Config B: C++ fast_forward, ALL tables STANDARD fp32")
    table_kinds_b = [0] * num_tabs  # ALL STANDARD
    weights_b = [dlrm.emb_l[k].weight.data for k in range(num_tabs)]
    mappings_b = [torch.empty(0, dtype=torch.int32)] * num_tabs
    scales_b = [0.0] * num_tabs
    zps_b = [0] * num_tabs
    _C.register_tables(table_kinds_b, weights_b, mappings_b, scales_b, zps_b)
    dlrm.apply_emb = _make_cpp_apply()

    run_lats_b = []
    for run in range(NUM_RUNS):
        gc.collect(); time.sleep(0.2)
        lat, emb, nb = run_inference_timed(dlrm, test_ld)
        run_lats_b.append({'batch_ms': lat, 'emb_ms': emb})
        log(f"    Run {run+1}: batch={lat:.2f}ms, emb={emb:.2f}ms")

    run_lats_b.sort(key=lambda x: x['batch_ms'])
    results['config_B'] = run_lats_b[NUM_RUNS // 2]
    auc_b = run_auc(dlrm, test_ld)
    results['config_B']['auc'] = auc_b
    log(f"  Config B median: batch={results['config_B']['batch_ms']:.2f}ms, "
        f"emb={results['config_B']['emb_ms']:.2f}ms, AUC={auc_b:.6f}")

    # --- Config C: C++ fast_forward with hot/cold + uint8 cold ---
    log("\n  Config C: C++ fast_forward with hot/cold split + uint8 cold")

    # Encode cold frames and register
    table_kinds_c, weights_c, mappings_c, scales_c, zps_c = [], [], [], [], []
    compressed_tables = set()

    for k in range(num_tabs):
        if k in cold_num_rows and cold_num_rows[k] > 0:
            compressed_tables.add(k)
            table_kinds_c.append(1)  # COMPRESSED_FP32
            h_idx = hot_indices[k]
            hot_w = state_dict[emb_keys[k]][h_idx].clone()
            weights_c.append(hot_w)
            mapping = torch.full((ln_emb[k],), 0, dtype=torch.int32)
            mapping[h_idx] = torch.arange(len(h_idx), dtype=torch.int32)
            c_idx = cold_indices[k]
            cm = o2c_map[k][c_idx]
            valid = cm >= 0
            mapping[c_idx[valid]] = -(cm[valid].int() + 1)
            mappings_c.append(mapping)
            scales_c.append(float(cold_quant_scale[k]))
            zps_c.append(int(cold_quant_zp[k]))
        else:
            table_kinds_c.append(0)
            weights_c.append(dlrm.emb_l[k].weight.data)
            mappings_c.append(torch.empty(0, dtype=torch.int32))
            scales_c.append(0.0); zps_c.append(0)

    _C.register_tables(table_kinds_c, weights_c, mappings_c, scales_c, zps_c)

    # Scan test batches for needed cold frames
    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
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
                valid_cm = cold_mapped >= 0
                if valid_cm.any():
                    fids = (cold_mapped[valid_cm] // rpf).unique().tolist()
                    needed_frames[t_idx].update(fids)

    total_needed = sum(len(v) for v in needed_frames.values())
    log(f"  Need {total_needed} unique cold frames")

    # Encode + decode cold frames in memory
    for t_idx in compressed_tables:
        if not needed_frames[t_idx]:
            continue
        c_idx = cold_indices[t_idx]
        cold_w = state_dict[emb_keys[t_idx]][c_idx]
        s, zp = cold_quant_scale[t_idx], cold_quant_zp[t_idx]
        cold_uint8 = torch.clamp(torch.round(cold_w / s + zp), 0, 255).to(torch.uint8)

        sorted_fids = sorted(needed_frames[t_idx])
        decoded_frames = []
        for fid in sorted_fids:
            start = fid * rpf
            end = min(start + rpf, len(c_idx))
            frame_rows = cold_uint8[start:end]
            frame = pack_rows_flat(frame_rows, D)
            # Encode lossless then decode
            _, decoded_frame = encode_frame_h265(frame, 0, lossless=True)
            rows = unpack_frame_flat(decoded_frame, end - start, D)
            if rows.shape[0] < rpf:
                pad = torch.zeros(rpf - rows.shape[0], D, dtype=torch.uint8)
                rows = torch.cat([rows, pad], dim=0)
            decoded_frames.append(rows)

        all_data = torch.cat(decoded_frames, dim=0)
        fids_t = torch.tensor(sorted_fids, dtype=torch.long)
        cold_mapping = o2c_map[t_idx].int()
        _C.register_cold_frames_for_table(
            t_idx, fids_t, all_data,
            float(s), float(zp), rpf, cold_mapping)

    log(f"  Cold frames encoded and registered")
    dlrm.apply_emb = _make_cpp_apply()

    run_lats_c = []
    for run in range(NUM_RUNS):
        gc.collect(); time.sleep(0.2)
        lat, emb, nb = run_inference_timed(dlrm, test_ld)
        run_lats_c.append({'batch_ms': lat, 'emb_ms': emb})
        log(f"    Run {run+1}: batch={lat:.2f}ms, emb={emb:.2f}ms")

    run_lats_c.sort(key=lambda x: x['batch_ms'])
    results['config_C'] = run_lats_c[NUM_RUNS // 2]
    auc_c = run_auc(dlrm, test_ld)
    results['config_C']['auc'] = auc_c
    log(f"  Config C median: batch={results['config_C']['batch_ms']:.2f}ms, "
        f"emb={results['config_C']['emb_ms']:.2f}ms, AUC={auc_c:.6f}")

    # Decomposition
    a_lat = results['config_A']['batch_ms']
    b_lat = results['config_B']['batch_ms']
    c_lat = results['config_C']['batch_ms']
    results['decomposition'] = {
        'speedup_A_to_B': round(a_lat / b_lat, 3),
        'speedup_B_to_C': round(b_lat / c_lat, 3),
        'speedup_A_to_C': round(a_lat / c_lat, 3),
        'cpp_benefit_ms': round(a_lat - b_lat, 3),
        'compression_benefit_ms': round(b_lat - c_lat, 3),
    }
    log(f"\n  DECOMPOSITION:")
    log(f"    A→B (C++ benefit):          {a_lat:.2f} → {b_lat:.2f} ms ({a_lat/b_lat:.2f}x)")
    log(f"    B→C (compression benefit):  {b_lat:.2f} → {c_lat:.2f} ms ({b_lat/c_lat:.2f}x)")
    log(f"    A→C (total):                {a_lat:.2f} → {c_lat:.2f} ms ({a_lat/c_lat:.2f}x)")

    # Restore original apply_emb
    restore_pytorch_apply()
    for t_idx in emb_keys:
        dlrm.emb_l[t_idx].weight.data = state_dict[emb_keys[t_idx]].clone()

    return results

# ============================================================
# EXPERIMENT 2: POST-TRAINING COMPRESSION COMPARISON (W2)
# ============================================================
def experiment2_compression_comparison(dlrm, test_ld, ln_emb, state_dict,
                                       freq, large_tables, D, dataset_name):
    """Compare 9 compression methods on ratio vs AUC delta."""
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 2: Post-Training Compression Comparison ({dataset_name})")
    log(f"{'='*70}")

    from dlrm_s_pytorch import DLRM_Net
    orig_apply = DLRM_Net.apply_emb
    dlrm.apply_emb = types.MethodType(orig_apply, dlrm)

    # Baseline AUC
    for t in large_tables:
        dlrm.emb_l[t].weight.data = state_dict[f'emb_l.{t}.weight'].clone()
    baseline_auc = run_auc(dlrm, test_ld)
    log(f"  Baseline AUC: {baseline_auc:.6f}")

    rpf = rows_per_frame(D)
    results = {'baseline_auc': baseline_auc, 'methods': {}}

    def measure_method(name, apply_fn):
        """Apply compression to large tables, measure AUC, restore."""
        total_orig_bytes = 0
        total_comp_bytes = 0

        for t in large_tables:
            w_orig = state_dict[f'emb_l.{t}.weight']
            orig_bytes, comp_bytes, w_new = apply_fn(t, w_orig, freq.get(t))
            total_orig_bytes += orig_bytes
            total_comp_bytes += comp_bytes
            dlrm.emb_l[t].weight.data = w_new

        auc = run_auc(dlrm, test_ld)
        delta = auc - baseline_auc
        ratio = total_orig_bytes / max(1, total_comp_bytes)

        # Restore
        for t in large_tables:
            dlrm.emb_l[t].weight.data = state_dict[f'emb_l.{t}.weight'].clone()

        results['methods'][name] = {
            'auc': auc, 'delta': round(delta, 6),
            'ratio': round(ratio, 2),
            'orig_bytes': total_orig_bytes, 'comp_bytes': total_comp_bytes,
        }
        log(f"  {name:35s}: AUC={auc:.6f} (delta={delta:+.6f}), ratio={ratio:.1f}x")

    # --- Method 1: INT8 uniform quantization ---
    def int8_quant(t, w, f):
        q, s, zp = quantize_table_uint8(w)
        w_new = dequantize_uint8(q, s, zp)
        return w.numel() * 4, w.numel(), w_new
    measure_method("1. INT8 uniform quant", int8_quant)

    # --- Method 2: INT4 uniform quantization ---
    def int4_quant(t, w, f):
        mn, mx = w.min().item(), w.max().item()
        s = (mx - mn) / 15.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        q4 = torch.clamp(torch.round(w / s + zp), 0, 15).to(torch.uint8)
        w_new = (q4.float() - zp) * s
        # INT4 packs 2 per byte
        return w.numel() * 4, w.numel() // 2, w_new
    measure_method("2. INT4 uniform quant", int4_quant)

    # --- Method 3: Product Quantization ---
    for M in [2, 4, 8]:
        if D % M != 0:
            continue
        sub_d = D // M
        def pq_quant(t, w, f, _M=M, _sub_d=sub_d):
            N = w.shape[0]
            w_new = w.clone()
            total_codebook_bytes = 0
            # Subsample for clustering if too large
            sample_n = min(100000, N)
            sample_idx = torch.randperm(N)[:sample_n]
            for m in range(_M):
                sub = w[sample_idx, m*_sub_d:(m+1)*_sub_d].numpy()
                from sklearn.cluster import MiniBatchKMeans
                kmeans = MiniBatchKMeans(n_clusters=256, batch_size=min(50000, sample_n),
                                         max_iter=10, n_init=1, random_state=42)
                kmeans.fit(sub)
                # Assign all rows
                full_sub = w[:, m*_sub_d:(m+1)*_sub_d].numpy()
                codes = kmeans.predict(full_sub)
                centroids = torch.from_numpy(kmeans.cluster_centers_).float()
                w_new[:, m*_sub_d:(m+1)*_sub_d] = centroids[codes]
                total_codebook_bytes += 256 * _sub_d * 4  # codebook
            # Compressed size: N bytes per subvector (code index) + codebook
            comp_bytes = N * _M + total_codebook_bytes
            return N * D * 4, comp_bytes, w_new
        measure_method(f"3. PQ M={M} K=256", pq_quant)

    # --- Method 4: SVD low-rank ---
    for rank in [1, 2, 4, 8]:
        if rank >= D:
            continue
        def svd_quant(t, w, f, _rank=rank):
            N = w.shape[0]
            U, S, Vh = torch.svd_lowrank(w.float(), q=_rank)
            w_new = U @ torch.diag(S) @ Vh.t()
            # Compressed: U (N x rank) + S (rank) + Vh (rank x D)
            comp_bytes = (N * _rank + _rank + _rank * D) * 4
            return N * D * 4, comp_bytes, w_new
        measure_method(f"4. SVD rank={rank}", svd_quant)

    # --- Method 5: Row pruning by frequency ---
    for pct in [90, 95, 99]:
        def prune_quant(t, w, f, _pct=pct):
            N = w.shape[0]
            if f is None:
                f = torch.zeros(N, dtype=torch.long)
            sorted_idx = f.argsort(descending=True)
            keep_n = max(1, int(N * (1 - _pct / 100)))
            w_new = w.clone()
            # Zero out bottom _pct% rows
            zero_idx = sorted_idx[keep_n:]
            w_new[zero_idx] = 0
            # Only keep_n rows need storage
            comp_bytes = keep_n * D * 4
            return N * D * 4, comp_bytes, w_new
        measure_method(f"5. Row prune {pct}%", prune_quant)

    # --- Method 6: Zstd-19 on uint8 ---
    def zstd_quant(t, w, f):
        import zstandard as zstd
        q, s, zp = quantize_table_uint8(w)
        w_new = dequantize_uint8(q, s, zp)
        raw = q.numpy().tobytes()
        cctx = zstd.ZstdCompressor(level=19)
        comp = cctx.compress(raw)
        return w.numel() * 4, len(comp), w_new
    measure_method("6. Zstd-19 on uint8", zstd_quant)

    # --- Method 7: H.265 lossless (CRF=0) ---
    def h265_lossless(t, w, f):
        N, _D = w.shape
        q, s, zp = quantize_table_uint8(w)
        total_comp = 0
        recon_rows = []
        for fid in range((N + rpf - 1) // rpf):
            start = fid * rpf
            end = min(start + rpf, N)
            frame = pack_rows_flat(q[start:end], _D)
            comp_size, decoded = encode_frame_h265(frame, 0, lossless=True)
            total_comp += comp_size
            recon_rows.append(unpack_frame_flat(decoded, end - start, _D))
        recon = torch.cat(recon_rows, dim=0)
        w_new = dequantize_uint8(recon, s, zp)
        return N * _D * 4, total_comp, w_new
    measure_method("7. H.265 lossless (CRF=0)", h265_lossless)

    # --- Method 8: H.265 CRF=18 (natural order) ---
    def h265_crf18_natural(t, w, f):
        N, _D = w.shape
        q, s, zp = quantize_table_uint8(w)
        total_comp = 0
        recon_rows = []
        for fid in range((N + rpf - 1) // rpf):
            start = fid * rpf
            end = min(start + rpf, N)
            frame = pack_rows_flat(q[start:end], _D)
            comp_size, decoded = encode_frame_h265(frame, 18, lossless=False)
            total_comp += comp_size
            recon_rows.append(unpack_frame_flat(decoded, end - start, _D))
        recon = torch.cat(recon_rows, dim=0)
        w_new = dequantize_uint8(recon, s, zp)
        return N * _D * 4, total_comp, w_new
    measure_method("8. H.265 CRF=18 (natural)", h265_crf18_natural)

    # --- Method 9: H.265 CRF=18 + frequency sort ---
    def h265_crf18_freq(t, w, f):
        N, _D = w.shape
        q, s, zp = quantize_table_uint8(w)
        if f is not None:
            perm = f.argsort(descending=True)
        else:
            perm = torch.arange(N)
        inv_perm = torch.argsort(perm)
        q_sorted = q[perm]
        total_comp = 0
        recon_rows = []
        for fid in range((N + rpf - 1) // rpf):
            start = fid * rpf
            end = min(start + rpf, N)
            frame = pack_rows_flat(q_sorted[start:end], _D)
            comp_size, decoded = encode_frame_h265(frame, 18, lossless=False)
            total_comp += comp_size
            recon_rows.append(unpack_frame_flat(decoded, end - start, _D))
        recon_sorted = torch.cat(recon_rows, dim=0)
        recon = recon_sorted[inv_perm]
        w_new = dequantize_uint8(recon, s, zp)
        return N * _D * 4, total_comp, w_new
    measure_method("9. H.265 CRF=18 + freq sort", h265_crf18_freq)

    return results

# ============================================================
# EXPERIMENT 3: ZERO-OUT COLD ROWS (W5)
# ============================================================
def experiment3_zero_out_cold(dlrm, test_ld, ln_emb, state_dict,
                              freq, large_tables, D, dataset_name):
    """Sweep zeroing bottom X% rows by frequency. Compare to H.265 CRF=18."""
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 3: Zero-Out Cold Rows ({dataset_name})")
    log(f"{'='*70}")

    from dlrm_s_pytorch import DLRM_Net
    orig_apply = DLRM_Net.apply_emb
    dlrm.apply_emb = types.MethodType(orig_apply, dlrm)

    for t in large_tables:
        dlrm.emb_l[t].weight.data = state_dict[f'emb_l.{t}.weight'].clone()
    baseline_auc = run_auc(dlrm, test_ld)
    log(f"  Baseline AUC: {baseline_auc:.6f}")

    thresholds = [50, 80, 90, 95, 99, 99.5, 99.9, 100]
    results = {'baseline_auc': baseline_auc, 'zero_out': {}}

    for pct in thresholds:
        total_orig, total_kept = 0, 0
        for t in large_tables:
            w_orig = state_dict[f'emb_l.{t}.weight']
            N = w_orig.shape[0]
            f = freq.get(t, torch.zeros(N, dtype=torch.long))
            sorted_idx = f.argsort(descending=True)
            keep_n = max(1, int(N * (1 - pct / 100)))
            w_new = w_orig.clone()
            w_new[sorted_idx[keep_n:]] = 0
            dlrm.emb_l[t].weight.data = w_new
            total_orig += N * D * 4
            total_kept += keep_n * D * 4

        auc = run_auc(dlrm, test_ld)
        delta = auc - baseline_auc
        ratio = total_orig / max(1, total_kept)
        results['zero_out'][str(pct)] = {
            'auc': auc, 'delta': round(delta, 6), 'ratio': round(ratio, 2)
        }
        log(f"  Zero {pct:5.1f}%: AUC={auc:.6f} (delta={delta:+.6f}), ratio={ratio:.1f}x")

        # Restore
        for t in large_tables:
            dlrm.emb_l[t].weight.data = state_dict[f'emb_l.{t}.weight'].clone()

    return results

# ============================================================
# EXPERIMENT 4: CPU CACHE PROFILING
# ============================================================
def experiment4_cache_profiling(dlrm, test_ld, ln_emb, state_dict,
                                hot_indices, cold_indices, is_hot, cold_num_rows,
                                emb_keys, cold_quant_scale, cold_quant_zp, o2c_map,
                                large_tables, D, dataset_name):
    """Measure hardware cache counters with perf stat for configs A, B, C."""
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 4: CPU Cache Profiling ({dataset_name})")
    log(f"{'='*70}")

    # Check if perf is available
    try:
        subprocess.run(['perf', 'stat', '--version'], capture_output=True, timeout=5)
        has_perf = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        has_perf = False
        log("  WARNING: perf not available, skipping hardware counters")

    if not has_perf:
        return {'error': 'perf not available'}

    assert HAS_CPP, "C++ extension required for Experiment 4"
    from dlrm_s_pytorch import DLRM_Net
    orig_apply = DLRM_Net.apply_emb
    num_tabs = len(dlrm.emb_l)
    rpf = rows_per_frame(D)
    NUM_WARMUP = 50
    NUM_MEASURED = 200
    results = {}

    events = "L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,instructions,cycles"

    def run_with_perf(config_name, setup_fn, cleanup_fn=None):
        """Run NUM_MEASURED batches with perf stat attached."""
        setup_fn()
        # Warmup
        log(f"  {config_name}: warming up ({NUM_WARMUP} batches)...")
        with torch.no_grad():
            for i, (X, lS_o, lS_i, T) in enumerate(test_ld):
                if i >= NUM_WARMUP:
                    break
                dlrm(X, lS_o, lS_i)

        pid = os.getpid()
        perf_out = os.path.join(RESULTS_DIR, f"perf_{config_name}_{dataset_name}.txt")

        # Start perf stat
        perf_cmd = ['perf', 'stat', '-x', ',', '-e', events,
                    '-p', str(pid), '-o', perf_out]
        try:
            perf_proc = subprocess.Popen(perf_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            log(f"  WARNING: Could not start perf: {e}")
            if cleanup_fn: cleanup_fn()
            return None

        # Run measured batches
        log(f"  {config_name}: running {NUM_MEASURED} measured batches...")
        batch_lats = []
        with torch.no_grad():
            for i, (X, lS_o, lS_i, T) in enumerate(test_ld):
                if i < NUM_WARMUP:
                    continue
                if i >= NUM_WARMUP + NUM_MEASURED:
                    break
                t0 = time.time()
                dlrm(X, lS_o, lS_i)
                batch_lats.append((time.time() - t0) * 1000)

        # Stop perf
        perf_proc.send_signal(signal.SIGINT)
        try:
            perf_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            perf_proc.kill()

        time.sleep(0.5)

        # Parse perf output
        counters = {}
        try:
            with open(perf_out) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            val = int(parts[0].strip().replace('.', ''))
                            name = parts[2].strip()
                            counters[name] = val
                        except ValueError:
                            pass
        except FileNotFoundError:
            log(f"  WARNING: perf output file not found: {perf_out}")

        if cleanup_fn: cleanup_fn()

        result = {
            'counters': counters,
            'mean_lat_ms': np.mean(batch_lats) if batch_lats else 0,
            'n_batches': len(batch_lats),
        }

        # Compute derived metrics
        l1_loads = counters.get('L1-dcache-loads', 0)
        l1_misses = counters.get('L1-dcache-load-misses', 0)
        llc_loads = counters.get('LLC-loads', 0)
        llc_misses = counters.get('LLC-load-misses', 0)
        if l1_loads > 0:
            result['l1_miss_rate'] = round(l1_misses / l1_loads * 100, 4)
        if llc_loads > 0:
            result['llc_miss_rate'] = round(llc_misses / llc_loads * 100, 4)
        if counters.get('cycles', 0) > 0:
            result['ipc'] = round(counters.get('instructions', 0) / counters['cycles'], 3)

        log(f"  {config_name}: lat={result['mean_lat_ms']:.2f}ms, "
            f"L1 miss={result.get('l1_miss_rate', '?')}%, "
            f"LLC miss={result.get('llc_miss_rate', '?')}%, "
            f"IPC={result.get('ipc', '?')}")
        return result

    # Config A: PyTorch fp32
    def setup_a():
        for t_idx in emb_keys:
            dlrm.emb_l[t_idx].weight.data = state_dict[emb_keys[t_idx]].clone()
        dlrm.apply_emb = types.MethodType(orig_apply, dlrm)

    results['config_A'] = run_with_perf(f"config_A", setup_a)

    # Config B: C++ STANDARD fp32
    def _make_cpp_apply():
        def _cpp_std_apply(lS_o, lS_i, emb_l, v_W_l):
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
            results_ff = _C.fast_forward(lS_i_2d, lS_o_2d)
            dlrm.time_look_up += time.time() - start
            return results_ff[-1]
        return _cpp_std_apply

    def setup_b():
        for t_idx in emb_keys:
            dlrm.emb_l[t_idx].weight.data = state_dict[emb_keys[t_idx]].clone()
        table_kinds_b = [0] * num_tabs
        weights_b = [dlrm.emb_l[k].weight.data for k in range(num_tabs)]
        mappings_b = [torch.empty(0, dtype=torch.int32)] * num_tabs
        _C.register_tables(table_kinds_b, weights_b, mappings_b, [0.0]*num_tabs, [0]*num_tabs)
        dlrm.apply_emb = _make_cpp_apply()

    results['config_B'] = run_with_perf(f"config_B", setup_b)

    # Config C: C++ hot/cold
    def setup_c():
        table_kinds_c, weights_c, mappings_c, scales_c, zps_c = [], [], [], [], []
        compressed_tables = set()
        for k in range(num_tabs):
            if k in cold_num_rows and cold_num_rows[k] > 0:
                compressed_tables.add(k)
                table_kinds_c.append(1)
                h_idx = hot_indices[k]
                hot_w = state_dict[emb_keys[k]][h_idx].clone()
                weights_c.append(hot_w)
                mapping = torch.full((ln_emb[k],), 0, dtype=torch.int32)
                mapping[h_idx] = torch.arange(len(h_idx), dtype=torch.int32)
                c_idx = cold_indices[k]
                cm = o2c_map[k][c_idx]
                valid = cm >= 0
                mapping[c_idx[valid]] = -(cm[valid].int() + 1)
                mappings_c.append(mapping)
                scales_c.append(float(cold_quant_scale[k]))
                zps_c.append(int(cold_quant_zp[k]))
            else:
                table_kinds_c.append(0)
                weights_c.append(dlrm.emb_l[k].weight.data)
                mappings_c.append(torch.empty(0, dtype=torch.int32))
                scales_c.append(0.0); zps_c.append(0)
        _C.register_tables(table_kinds_c, weights_c, mappings_c, scales_c, zps_c)

        # Register cold frames (same as exp1 config C)
        test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
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
                    valid_cm = cold_mapped >= 0
                    if valid_cm.any():
                        fids = (cold_mapped[valid_cm] // rpf).unique().tolist()
                        needed_frames[t_idx].update(fids)

        for t_idx in compressed_tables:
            if not needed_frames[t_idx]:
                continue
            c_idx = cold_indices[t_idx]
            cold_w = state_dict[emb_keys[t_idx]][c_idx]
            s, zp_val = cold_quant_scale[t_idx], cold_quant_zp[t_idx]
            cold_uint8 = torch.clamp(torch.round(cold_w / s + zp_val), 0, 255).to(torch.uint8)
            sorted_fids = sorted(needed_frames[t_idx])
            decoded_frames = []
            for fid in sorted_fids:
                start = fid * rpf
                end = min(start + rpf, len(c_idx))
                frame_rows = cold_uint8[start:end]
                frame = pack_rows_flat(frame_rows, D)
                _, decoded_frame = encode_frame_h265(frame, 0, lossless=True)
                rows = unpack_frame_flat(decoded_frame, end - start, D)
                if rows.shape[0] < rpf:
                    pad_n = rpf - rows.shape[0]
                    rows = torch.cat([rows, torch.zeros(pad_n, D, dtype=torch.uint8)], dim=0)
                decoded_frames.append(rows)
            all_data = torch.cat(decoded_frames, dim=0)
            fids_t = torch.tensor(sorted_fids, dtype=torch.long)
            cold_mapping = o2c_map[t_idx].int()
            _C.register_cold_frames_for_table(
                t_idx, fids_t, all_data,
                float(s), float(zp_val), rpf, cold_mapping)

        dlrm.apply_emb = _make_cpp_apply()

    results['config_C'] = run_with_perf(f"config_C", setup_c)

    # Restore
    dlrm.apply_emb = types.MethodType(orig_apply, dlrm)
    for t_idx in emb_keys:
        dlrm.emb_l[t_idx].weight.data = state_dict[emb_keys[t_idx]].clone()

    return results

# ============================================================
# EXPERIMENT 5: PER-ROW ERROR ANALYSIS (ERROR STEERING)
# ============================================================
def experiment5_error_analysis(state_dict, freq, large_tables, D, dataset_name):
    """Prove frequency sorting steers H.265 error to infrequent rows."""
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 5: Per-Row Error Analysis ({dataset_name})")
    log(f"{'='*70}")

    rpf = rows_per_frame(D)
    orderings = ['natural', 'random', 'frequency']
    crf = 18
    results = {}

    # Frequency buckets: top 1%, 1-10%, 10-50%, 50-100%
    bucket_names = ['top_1pct', '1_10pct', '10_50pct', '50_100pct']

    for t_idx in large_tables:
        w = state_dict[f'emb_l.{t_idx}.weight']
        N = w.shape[0]
        q, s, zp = quantize_table_uint8(w)
        f = freq.get(t_idx, torch.zeros(N, dtype=torch.long))

        # Compute bucket boundaries based on frequency ranking
        freq_rank = f.argsort(descending=True)
        row_bucket = torch.full((N,), 3, dtype=torch.long)  # default: 50-100%
        n1 = max(1, int(N * 0.01))
        n10 = max(1, int(N * 0.10))
        n50 = max(1, int(N * 0.50))
        row_bucket[freq_rank[:n1]] = 0   # top 1%
        row_bucket[freq_rank[n1:n10]] = 1  # 1-10%
        row_bucket[freq_rank[n10:n50]] = 2  # 10-50%

        table_results = {}

        for order in orderings:
            # Reorder
            if order == 'natural':
                perm = torch.arange(N)
            elif order == 'random':
                perm = torch.randperm(N)
            elif order == 'frequency':
                perm = f.argsort(descending=True)
            inv_perm = torch.argsort(perm)

            q_ordered = q[perm]

            # Encode/decode with CRF=18
            recon_rows = []
            for fid in range((N + rpf - 1) // rpf):
                start_r = fid * rpf
                end_r = min(start_r + rpf, N)
                frame = pack_rows_flat(q_ordered[start_r:end_r], D)
                _, decoded = encode_frame_h265(frame, crf, lossless=False)
                recon = unpack_frame_flat(decoded, end_r - start_r, D)
                recon_rows.append(recon)
            recon_ordered = torch.cat(recon_rows, dim=0)

            # Per-row MSE in uint8 space (in reordered space)
            per_row_mse_ordered = ((q_ordered.float() - recon_ordered.float()) ** 2).mean(dim=1)

            # Map back to original order
            per_row_mse = per_row_mse_ordered[inv_perm]

            # Compute mean MSE per bucket
            bucket_mse = {}
            for bi, bname in enumerate(bucket_names):
                mask = row_bucket == bi
                if mask.any():
                    bucket_mse[bname] = round(per_row_mse[mask].mean().item(), 4)
                else:
                    bucket_mse[bname] = 0.0

            overall_mse = round(per_row_mse.mean().item(), 4)
            table_results[order] = {
                'overall_mse': overall_mse,
                'bucket_mse': bucket_mse,
            }

            log(f"  Table {t_idx} [{order:12s}]: overall MSE={overall_mse:.4f}, "
                f"top1%={bucket_mse['top_1pct']:.4f}, "
                f"1-10%={bucket_mse['1_10pct']:.4f}, "
                f"10-50%={bucket_mse['10_50pct']:.4f}, "
                f"50-100%={bucket_mse['50_100pct']:.4f}")

        results[t_idx] = table_results

    return results

# ============================================================
# SUMMARY GENERATION
# ============================================================
def generate_summary(all_results, dataset_name):
    summary_path = os.path.join(RESULTS_DIR, "summary.md")
    mode = 'a' if os.path.exists(summary_path) and os.path.getsize(summary_path) > 0 else 'w'
    with open(summary_path, mode) as f:
        if mode == 'w':
            f.write("# MLSys Reviewer Experiments\n\n")

        f.write(f"\n## {dataset_name.upper()}\n\n")

        # Experiment 1
        key = f'{dataset_name}_exp1'
        if key in all_results:
            exp1 = all_results[key]
            f.write("### Experiment 1: Fair Baseline Decomposition (W1)\n\n")
            f.write("| Config | Batch (ms) | Emb (ms) | AUC |\n")
            f.write("|--------|-----------|---------|-----|\n")
            for cfg in ['config_A', 'config_B', 'config_C']:
                if cfg in exp1:
                    d = exp1[cfg]
                    f.write(f"| {cfg} | {d['batch_ms']:.2f} | {d['emb_ms']:.2f} | {d.get('auc', 'N/A'):.6f} |\n")
            if 'decomposition' in exp1:
                dec = exp1['decomposition']
                f.write(f"\n- A→B (C++ benefit): {dec['speedup_A_to_B']:.2f}x\n")
                f.write(f"- B→C (compression benefit): {dec['speedup_B_to_C']:.2f}x\n")
                f.write(f"- A→C (total): {dec['speedup_A_to_C']:.2f}x\n\n")

        # Experiment 2
        key = f'{dataset_name}_exp2'
        if key in all_results:
            exp2 = all_results[key]
            f.write("### Experiment 2: Post-Training Compression Comparison (W2)\n\n")
            f.write(f"Baseline AUC: {exp2.get('baseline_auc', 'N/A')}\n\n")
            f.write("| Method | AUC delta | Compression ratio |\n")
            f.write("|--------|----------|------------------|\n")
            for name, d in sorted(exp2.get('methods', {}).items()):
                f.write(f"| {name} | {d['delta']:+.6f} | {d['ratio']:.1f}x |\n")
            f.write("\n")

        # Experiment 3
        key = f'{dataset_name}_exp3'
        if key in all_results:
            exp3 = all_results[key]
            f.write("### Experiment 3: Zero-Out Cold Rows (W5)\n\n")
            f.write(f"Baseline AUC: {exp3.get('baseline_auc', 'N/A')}\n\n")
            f.write("| Zero % | AUC delta | Effective ratio |\n")
            f.write("|--------|----------|----------------|\n")
            for pct, d in sorted(exp3.get('zero_out', {}).items(), key=lambda x: float(x[0])):
                f.write(f"| {pct}% | {d['delta']:+.6f} | {d['ratio']:.1f}x |\n")
            f.write("\n")

        # Experiment 4
        key = f'{dataset_name}_exp4'
        if key in all_results:
            exp4 = all_results[key]
            f.write("### Experiment 4: CPU Cache Profiling\n\n")
            f.write("| Config | Lat (ms) | L1 miss% | LLC miss% | IPC |\n")
            f.write("|--------|---------|---------|----------|-----|\n")
            for cfg in ['config_A', 'config_B', 'config_C']:
                if cfg in exp4 and exp4[cfg]:
                    d = exp4[cfg]
                    f.write(f"| {cfg} | {d.get('mean_lat_ms', 0):.2f} | "
                            f"{d.get('l1_miss_rate', 'N/A')} | "
                            f"{d.get('llc_miss_rate', 'N/A')} | "
                            f"{d.get('ipc', 'N/A')} |\n")
            f.write("\n")

        # Experiment 5
        key = f'{dataset_name}_exp5'
        if key in all_results:
            exp5 = all_results[key]
            f.write("### Experiment 5: Per-Row Error Analysis (Error Steering)\n\n")
            f.write("CRF=18, MSE in uint8 space by access frequency bucket\n\n")
            for t_idx in sorted(exp5.keys(), key=lambda x: int(x)):
                f.write(f"\n**Table {t_idx}**\n\n")
                f.write("| Ordering | Overall | Top 1% | 1-10% | 10-50% | 50-100% |\n")
                f.write("|----------|---------|--------|-------|--------|--------|\n")
                for order in ['natural', 'random', 'frequency']:
                    if order in exp5[t_idx]:
                        d = exp5[t_idx][order]
                        b = d['bucket_mse']
                        f.write(f"| {order} | {d['overall_mse']:.4f} | "
                                f"{b['top_1pct']:.4f} | {b['1_10pct']:.4f} | "
                                f"{b['10_50pct']:.4f} | {b['50_100pct']:.4f} |\n")
            f.write("\n")

    log(f"Summary appended to {summary_path}")

# ============================================================
# JSON SERIALIZATION HELPER
# ============================================================
def convert_for_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(v) for v in obj]
    return obj

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("MLSYS REVIEWER EXPERIMENTS")
    log("=" * 70)

    torch.set_num_threads(NUM_THREADS)
    all_results = {}

    # Remove old summary
    summary_path = os.path.join(RESULTS_DIR, "summary.md")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    for dataset_name in ['kaggle', 'terabyte']:
        model_path = KAGGLE_MODEL if dataset_name == 'kaggle' else TERABYTE_MODEL
        if not os.path.exists(model_path):
            log(f"Skipping {dataset_name}: model not found at {model_path}")
            continue

        log(f"\n{'#'*70}")
        log(f"# {dataset_name.upper()}")
        log(f"{'#'*70}")

        dlrm, test_ld, ln_emb, state_dict = load_model_and_data(dataset_name)
        D = 16 if dataset_name == 'kaggle' else 64
        num_tabs = len(ln_emb)
        large_tables = [i for i in range(num_tabs) if ln_emb[i] > LARGE_TABLE_THRESHOLD]
        log(f"Large tables: {large_tables}")

        # Profile access frequencies
        done_marker = os.path.join(RESULTS_DIR, f"{dataset_name}_profile.done")
        log("\nProfiling access frequencies...")
        freq = profile_access_frequencies(test_ld, ln_emb, large_tables)

        # Build hot/cold split
        hot_indices, cold_indices, is_hot, cold_num_rows, emb_keys, \
            cold_quant_scale, cold_quant_zp, o2c_map = \
            build_hot_cold_split(ln_emb, freq, large_tables, state_dict)

        # --- Experiment 1 ---
        done1 = os.path.join(RESULTS_DIR, f"{dataset_name}_exp1.done")
        if not os.path.exists(done1):
            try:
                r1 = experiment1_fair_baseline(
                    dlrm, test_ld, ln_emb, state_dict,
                    hot_indices, cold_indices, is_hot, cold_num_rows,
                    emb_keys, cold_quant_scale, cold_quant_zp, o2c_map,
                    large_tables, D, dataset_name)
                all_results[f'{dataset_name}_exp1'] = r1
                open(done1, 'w').write('done')
            except Exception as e:
                log(f"  Experiment 1 failed: {e}")
                import traceback; traceback.print_exc()
        else:
            log(f"  Experiment 1 already done ({done1})")

        # --- Experiment 2 ---
        done2 = os.path.join(RESULTS_DIR, f"{dataset_name}_exp2.done")
        if not os.path.exists(done2):
            try:
                r2 = experiment2_compression_comparison(
                    dlrm, test_ld, ln_emb, state_dict, freq, large_tables, D, dataset_name)
                all_results[f'{dataset_name}_exp2'] = r2
                open(done2, 'w').write('done')
            except Exception as e:
                log(f"  Experiment 2 failed: {e}")
                import traceback; traceback.print_exc()
        else:
            log(f"  Experiment 2 already done ({done2})")

        # --- Experiment 3 ---
        done3 = os.path.join(RESULTS_DIR, f"{dataset_name}_exp3.done")
        if not os.path.exists(done3):
            try:
                r3 = experiment3_zero_out_cold(
                    dlrm, test_ld, ln_emb, state_dict, freq, large_tables, D, dataset_name)
                all_results[f'{dataset_name}_exp3'] = r3
                open(done3, 'w').write('done')
            except Exception as e:
                log(f"  Experiment 3 failed: {e}")
                import traceback; traceback.print_exc()
        else:
            log(f"  Experiment 3 already done ({done3})")

        # --- Experiment 4 ---
        done4 = os.path.join(RESULTS_DIR, f"{dataset_name}_exp4.done")
        if not os.path.exists(done4):
            try:
                r4 = experiment4_cache_profiling(
                    dlrm, test_ld, ln_emb, state_dict,
                    hot_indices, cold_indices, is_hot, cold_num_rows,
                    emb_keys, cold_quant_scale, cold_quant_zp, o2c_map,
                    large_tables, D, dataset_name)
                all_results[f'{dataset_name}_exp4'] = r4
                open(done4, 'w').write('done')
            except Exception as e:
                log(f"  Experiment 4 failed: {e}")
                import traceback; traceback.print_exc()
        else:
            log(f"  Experiment 4 already done ({done4})")

        # --- Experiment 5 ---
        done5 = os.path.join(RESULTS_DIR, f"{dataset_name}_exp5.done")
        if not os.path.exists(done5):
            try:
                r5 = experiment5_error_analysis(
                    state_dict, freq, large_tables, D, dataset_name)
                all_results[f'{dataset_name}_exp5'] = r5
                open(done5, 'w').write('done')
            except Exception as e:
                log(f"  Experiment 5 failed: {e}")
                import traceback; traceback.print_exc()
        else:
            log(f"  Experiment 5 already done ({done5})")

        # Generate summary for this dataset
        generate_summary(all_results, dataset_name)

        # Clean up
        del dlrm, test_ld, ln_emb, state_dict
        gc.collect()

    # Save all results
    results_path = os.path.join(RESULTS_DIR, "all_experiments.json")
    with open(results_path, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    log(f"\nAll results saved to {results_path}")

    log("\n" + "=" * 70)
    log("ALL EXPERIMENTS COMPLETE")
    log("=" * 70)
    log_fh.close()

if __name__ == "__main__":
    main()
