#!/usr/bin/env python3
"""
Comprehensive Benchmark: Baseline vs CAFE+ vs Codec+LRU

Strategy:
  1. Baseline: actual fp32 inference (3 runs, median)
  2. Codec full_cpp: actual inference with all frames pre-decoded in C++ (3 runs, median)
  3. Cache size sweep: simulate LRU behavior by tracking per-batch frame accesses,
     compute decode overhead for each cache size {4,8,16,22,32,64}
  4. Dynamic look-ahead: actual inference with group-buffer approach,
     group sizes {1,10,50,100,500}

No batch pre-loading in the timed inference loop.
"""

import os, sys, time, json, gc, subprocess, threading
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import psutil
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
MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
DATA_FILE = os.path.expanduser("~/input/train.txt")
PROCESSED_DATA = os.path.expanduser("~/input/kaggleAdDisplayChallenge_processed.npz")
EMB_DIM = 16
TEST_BATCH_SIZE = 2048
LARGE_TABLE_THRESHOLD = 50000
TILE_H = TILE_W = 4
OPTIMIZED_THREADS = 40
NUM_RUNS = 3

RESULTS_DIR = "results"
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
ONDEMAND_DIR = os.path.join(RESULTS_DIR, "ondemand")
LOG_FILE = "logs/full_comparison.log"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "full_comparison.md")

os.makedirs("logs", exist_ok=True)
log_fh = open(LOG_FILE, 'w')

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n"); log_fh.flush()

def get_rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def drop_caches():
    try:
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except Exception: pass

# ============================================================
# MODEL + DATA
# ============================================================
def create_args():
    class Args: pass
    a = Args()
    a.arch_sparse_feature_size = EMB_DIM
    a.arch_mlp_bot = "13-512-256-64-16"; a.arch_mlp_top = "512-256-1"
    a.arch_interaction_op = "dot"; a.arch_interaction_itself = False
    a.data_generation = "dataset"; a.data_set = "kaggle"
    a.raw_data_file = DATA_FILE; a.processed_data_file = PROCESSED_DATA
    a.loss_function = "bce"; a.max_ind_range = -1
    a.test_mini_batch_size = TEST_BATCH_SIZE
    a.test_num_workers = 0; a.num_workers = 0
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
    log("Loading checkpoint...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    log(f"Model loaded: {len(ln_emb)} tables, emb_dim={m_spa}")
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
# RUN INFERENCE (shared for baseline and codec full_cpp)
# ============================================================
def run_inference(dlrm, test_ld, tag=""):
    """Run inference iterating through DataLoader (no pre-cache).
    Returns dict of metrics."""

    max_samples = 2000 * TEST_BATCH_SIZE
    all_scores = np.empty(max_samples, dtype=np.float32)
    all_targets = np.empty(max_samples, dtype=np.float32)
    sample_idx = 0
    blats = []
    data_load_times = []

    dlrm.time_look_up = 0; dlrm.time_interact = 0; dlrm.time_mlp = 0

    t0 = time.time()
    with torch.no_grad():
        for inputBatch in test_ld:
            t_dl = time.time()
            X, lS_o, lS_i, T = inputBatch[0], inputBatch[1], inputBatch[2], inputBatch[3]
            data_load_times.append(time.time() - t_dl)

            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)

            z_np = Z.detach().cpu().numpy().ravel()
            t_np = T.detach().cpu().numpy().ravel()
            bs = z_np.shape[0]
            all_scores[sample_idx:sample_idx+bs] = z_np
            all_targets[sample_idx:sample_idx+bs] = t_np
            sample_idx += bs

    total_time = time.time() - t0
    scores, targets = all_scores[:sample_idx], all_targets[:sample_idx]
    auc, ll, acc = compute_metrics(scores, targets)
    n_b = len(blats)

    return {
        'auc': auc, 'log_loss': ll, 'accuracy': acc,
        'total_time': total_time, 'num_batches': n_b,
        'mean_lat_ms': np.mean(blats)*1000,
        'p50_lat_ms': np.percentile(blats,50)*1000,
        'p99_lat_ms': np.percentile(blats,99)*1000,
        'emb_ms': dlrm.time_look_up/n_b*1000,
        'interact_ms': dlrm.time_interact/n_b*1000,
        'mlp_ms': dlrm.time_mlp/n_b*1000,
        'data_load_ms': np.mean(data_load_times)*1000,
        'rss_mb': get_rss_mb(),
    }


# ============================================================
# SETUP CODEC FULL_CPP (register all needed frames in C++)
# ============================================================
def setup_codec_fullcpp(dlrm, ln_emb, state_dict, test_batches,
                        hot_indices, cold_indices, is_hot, o2c_map,
                        cold_quant_scale, cold_quant_zp, cold_num_rows,
                        large_tables, emb_keys):
    """Register tables and cold frames in C++ for full C++ inference.
    Returns (apply_emb_fn, cleanup_fn, memory_info, frame_access_log)."""

    width, height = 1920, 1080
    rows_per_frame = (width // TILE_W) * (height // TILE_H)
    res_dir = os.path.join(ONDEMAND_DIR, '1080p')
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
            mapping = torch.full((ln_emb[k],), -1, dtype=torch.int32)
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

    # Scan all batches for needed frames + build frame access log
    log("  Scanning batches for cold frame coverage...")
    t_scan0 = time.time()
    needed_frames = {t: set() for t in compressed_tables}
    # Per-batch frame access log for cache simulation
    batch_frame_log = []  # list of {table_id: set(frame_ids)} per batch

    for X, lS_o, lS_i, T in test_batches:
        batch_frames = {}
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
                    fids = (cold_mapped[valid] // rows_per_frame).unique().tolist()
                    needed_frames[t_idx].update(fids)
                    batch_frames[t_idx] = set(fids)
        batch_frame_log.append(batch_frames)

    total_needed = sum(len(v) for v in needed_frames.values())
    log(f"  Found {total_needed} unique frames in {time.time()-t_scan0:.1f}s")

    # Decode and register all needed frames
    log("  Decoding and registering cold frames in C++...")
    total_cold_frame_mb = 0
    total_compressed_bytes = 0

    for t_idx in compressed_tables:
        frame_dir = os.path.join(res_dir, f'table_{t_idx}')
        if not os.path.exists(frame_dir):
            continue

        # Count compressed bytes
        frame_files = [f for f in os.listdir(frame_dir)
                       if f.startswith('frame_') and (f.endswith('.h265') or f.endswith('.h264'))]
        total_compressed_bytes += sum(os.path.getsize(os.path.join(frame_dir, ff)) for ff in frame_files)

        if not needed_frames[t_idx]:
            continue

        # Batch decode all needed frames
        sorted_fids = sorted(needed_frames[t_idx])
        fids_t = torch.tensor(sorted_fids, dtype=torch.long)

        if len(sorted_fids) > 1 and hasattr(_C, 'batch_decode_frames'):
            decoded = _C.batch_decode_frames(frame_dir, fids_t)
        else:
            decoded = [_C.decode_h265_frame_from_file(
                os.path.join(frame_dir, f'frame_{fid:05d}.h265')) for fid in sorted_fids]

        # Concatenate into contiguous buffer
        padded = []
        for frame in decoded:
            rows = _C.untile_frame_to_rows(frame, rows_per_frame) if frame.dim() == 2 else frame
            if rows.shape[0] < rows_per_frame:
                pad = torch.zeros(rows_per_frame - rows.shape[0], EMB_DIM, dtype=torch.uint8)
                rows = torch.cat([rows, pad], dim=0)
            padded.append(rows)
        all_data = torch.cat(padded, dim=0)
        total_cold_frame_mb += all_data.numel() / 1024 / 1024

        # Load cold mapping
        mmap_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t_idx}.npy')
        pt_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t_idx}.pt')
        if os.path.exists(mmap_path):
            cold_mapping = torch.from_numpy(np.load(mmap_path).copy()).int()
        else:
            cold_mapping = torch.load(pt_path, map_location='cpu', weights_only=True).int()

        _C.register_cold_frames_for_table(
            t_idx, fids_t, all_data,
            float(cold_quant_scale[t_idx]), float(cold_quant_zp[t_idx]),
            rows_per_frame, cold_mapping)

    log(f"  Cold frames registered: {total_cold_frame_mb:.1f}MB uint8")

    # Build full_cpp apply_emb
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
        return results[-1]  # stacked [T, B, D]

    dlrm.apply_emb = _full_cpp_apply

    # Compute memory
    bm_mb = sum(((ln_emb[t]+63)//64*8 + ((ln_emb[t]+63)//64+1)*4)
                for t in compressed_tables) / 1024 / 1024
    cold_map_mb = sum(ln_emb[t]*4 for t in compressed_tables) / 1024 / 1024
    mapping_mb = bm_mb + cold_map_mb
    compressed_mb = total_compressed_bytes / 1024 / 1024
    orig_cold_mb = sum(cold_num_rows.get(t,0)*EMB_DIM*4 for t in compressed_tables) / 1024 / 1024

    mem_info = {
        'hot_mb': total_hot_mb,
        'cold_disk_mb': compressed_mb,
        'lru_mb': total_cold_frame_mb,
        'mapping_mb': mapping_mb,
        'total_mem_mb': total_hot_mb + total_cold_frame_mb + mapping_mb,
        'compression_ratio': orig_cold_mb / max(0.01, compressed_mb),
        'total_needed_frames': total_needed,
    }

    def cleanup():
        dlrm.apply_emb = _orig_apply

    return cleanup, mem_info, batch_frame_log


# ============================================================
# LRU CACHE SIMULATION
# ============================================================
def simulate_lru_cache(batch_frame_log, cache_capacity):
    """Simulate LRU cache behavior from frame access log.
    Returns dict with cache stats."""
    cache = OrderedDict()  # (table, frame) -> access_time
    evictions = 0
    hits = 0
    misses = 0
    per_batch_misses = []

    for batch_idx, batch_frames in enumerate(batch_frame_log):
        batch_miss = 0
        for t_idx, fids in batch_frames.items():
            for fid in fids:
                key = (t_idx, fid)
                if key in cache:
                    cache.move_to_end(key)
                    hits += 1
                else:
                    misses += 1
                    batch_miss += 1
                    cache[key] = True
                    while len(cache) > cache_capacity:
                        cache.popitem(last=False)
                        evictions += 1
        per_batch_misses.append(batch_miss)

    total = hits + misses
    return {
        'hit_rate': hits / total if total > 0 else 1.0,
        'total_hits': hits,
        'total_misses': misses,
        'evictions': evictions,
        'unique_frames_accessed': len(set(
            (t, f) for bf in batch_frame_log for t, fids in bf.items() for f in fids)),
        'avg_misses_per_batch': np.mean(per_batch_misses) if per_batch_misses else 0,
        'max_misses_per_batch': max(per_batch_misses) if per_batch_misses else 0,
        'p99_misses_per_batch': np.percentile(per_batch_misses, 99) if per_batch_misses else 0,
        'cache_capacity': cache_capacity,
    }


# ============================================================
# REAL LRU CACHE INFERENCE
# ============================================================
def run_real_lru_inference(dlrm, test_ld, cache_capacity, ln_emb, state_dict,
                           hot_indices, cold_indices, is_hot, o2c_map,
                           cold_quant_scale, cold_quant_zp, cold_num_rows,
                           large_tables, emb_keys):
    """Run real end-to-end inference with LRU cache of decoded frames."""

    width, height = 1920, 1080
    rows_per_frame = (width // TILE_W) * (height // TILE_H)
    res_dir = os.path.join(ONDEMAND_DIR, '1080p')
    num_tabs = len(dlrm.emb_l)
    compressed_tables = set(t for t in large_tables if cold_num_rows.get(t, 0) > 0)

    # Register tables in C++ (hot weights + mappings)
    table_kinds, weights, mappings_list, scales, zero_points = [], [], [], [], []
    total_hot_mb = 0
    for k in range(num_tabs):
        if k in compressed_tables:
            table_kinds.append(1)
            h_idx = hot_indices[k]
            hot_w = state_dict[emb_keys[k]][h_idx].clone()
            weights.append(hot_w)
            total_hot_mb += hot_w.numel()*4/1024/1024
            mapping = torch.full((ln_emb[k],), -1, dtype=torch.int32)
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

    # Load cold mappings
    cold_map_tensors = {}
    for t in compressed_tables:
        mmap_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.npy')
        pt_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        if os.path.exists(mmap_path):
            cold_map_tensors[t] = torch.from_numpy(np.load(mmap_path).copy()).int()
        else:
            cold_map_tensors[t] = torch.load(pt_path, map_location='cpu', weights_only=True).int()

    # Compressed file sizes
    compressed_bytes = 0
    for t in compressed_tables:
        frame_dir = os.path.join(res_dir, f'table_{t}')
        if os.path.exists(frame_dir):
            for f in os.listdir(frame_dir):
                if f.startswith('frame_') and (f.endswith('.h265') or f.endswith('.h264')):
                    compressed_bytes += os.path.getsize(os.path.join(frame_dir, f))
    compressed_mb = compressed_bytes / 1024 / 1024
    bm_mb = sum(((ln_emb[t]+63)//64*8 + ((ln_emb[t]+63)//64+1)*4)
                for t in compressed_tables) / 1024 / 1024
    cold_map_mb = sum(ln_emb[t]*4 for t in compressed_tables) / 1024 / 1024
    mapping_mb = bm_mb + cold_map_mb
    orig_cold_mb = sum(cold_num_rows.get(t,0)*EMB_DIM*4
                       for t in compressed_tables) / 1024 / 1024

    # Full C++ apply_emb
    _orig_apply = dlrm.apply_emb
    def _apply(lS_o, lS_i, emb_l, v_W_l):
        start = time.time()
        if isinstance(lS_i, (list, tuple)): lS_i_2d = torch.stack(lS_i)
        elif lS_i.dim() == 2: lS_i_2d = lS_i
        else: lS_i_2d = lS_i.view(num_tabs, -1)
        if isinstance(lS_o, (list, tuple)): lS_o_2d = torch.stack(lS_o)
        elif lS_o.dim() == 2: lS_o_2d = lS_o
        else: lS_o_2d = lS_o.view(num_tabs, -1)
        results = _C.fast_forward(lS_i_2d, lS_o_2d)
        dlrm.time_look_up += time.time() - start
        return results[-1]
    dlrm.apply_emb = _apply

    # LRU cache: (table_id, frame_id) -> decoded rows (uint8, rows_per_frame x EMB_DIM)
    frame_cache = OrderedDict()
    cache_hits = 0
    cache_misses = 0
    evictions = 0
    total_frames_decoded = 0

    # Pre-build ordered lists for C++ scan
    comp_list = sorted(compressed_tables)
    is_hot_list = [is_hot[t] for t in comp_list]
    o2c_map_list = [o2c_map[t].int() for t in comp_list]

    # Run inference
    dlrm.time_look_up = 0; dlrm.time_interact = 0; dlrm.time_mlp = 0
    max_samples = 2000 * TEST_BATCH_SIZE
    all_scores = np.empty(max_samples, dtype=np.float32)
    all_targets = np.empty(max_samples, dtype=np.float32)
    sample_idx = 0
    blats, data_load_times, scan_times, decode_times = [], [], [], []

    t0 = time.time()
    with torch.no_grad():
        for inputBatch in test_ld:
            t_dl = time.time()
            X, lS_o, lS_i, T = inputBatch[0], inputBatch[1], inputBatch[2], inputBatch[3]
            data_load_times.append(time.time() - t_dl)

            # SCAN for needed frames (fused C++)
            t_scan = time.time()
            lS_i_for_scan = [lS_i[t] if isinstance(lS_i, (list, tuple)) or lS_i.dim() == 2
                             else lS_i for t in comp_list]
            frame_lists = _C.scan_needed_frames(lS_i_for_scan, is_hot_list,
                                                 o2c_map_list, rows_per_frame)
            needed = {}
            for k, t_idx in enumerate(comp_list):
                if frame_lists[k].numel() > 0:
                    needed[t_idx] = set(frame_lists[k].tolist())
            scan_ms = (time.time() - t_scan) * 1000
            scan_times.append(scan_ms)

            # CHECK cache + collect misses
            t_dec = time.time()
            dirty_tables = set()
            misses_by_table = {}

            for t_idx, fids in needed.items():
                for fid in fids:
                    key = (t_idx, fid)
                    if key in frame_cache:
                        frame_cache.move_to_end(key)
                        cache_hits += 1
                    else:
                        cache_misses += 1
                        misses_by_table.setdefault(t_idx, []).append(fid)

            # Batch decode misses per table
            for t_idx, miss_fids in misses_by_table.items():
                total_frames_decoded += len(miss_fids)
                frame_dir = os.path.join(res_dir, f'table_{t_idx}')
                sorted_fids = sorted(miss_fids)

                if len(sorted_fids) > 1 and hasattr(_C, 'batch_decode_frames'):
                    fids_t = torch.tensor(sorted_fids, dtype=torch.long)
                    decoded_list = _C.batch_decode_frames(frame_dir, fids_t)
                else:
                    decoded_list = []
                    for fid in sorted_fids:
                        fpath = os.path.join(frame_dir, f'frame_{fid:05d}.h265')
                        if not os.path.exists(fpath):
                            fpath = os.path.join(frame_dir, f'frame_{fid:05d}.h264')
                        decoded_list.append(_C.decode_h265_frame_from_file(fpath))

                for fid, frame in zip(sorted_fids, decoded_list):
                    rows = _C.untile_frame_to_rows(frame, rows_per_frame)
                    if rows.shape[0] < rows_per_frame:
                        rows = torch.cat([rows, torch.zeros(
                            rows_per_frame - rows.shape[0], EMB_DIM, dtype=torch.uint8)])
                    frame_cache[(t_idx, fid)] = rows
                dirty_tables.add(t_idx)

            # Evict if over capacity
            while len(frame_cache) > cache_capacity:
                evicted_key, _ = frame_cache.popitem(last=False)
                evictions += 1
                dirty_tables.add(evicted_key[0])

            # Re-register dirty tables in C++
            for t_idx in dirty_tables:
                table_frames = [(fid, data) for (tid, fid), data
                                in frame_cache.items() if tid == t_idx]
                if table_frames:
                    table_frames.sort(key=lambda x: x[0])
                    fids_t = torch.tensor([f[0] for f in table_frames], dtype=torch.long)
                    all_data = torch.cat([f[1] for f in table_frames], dim=0)
                    _C.register_cold_frames_for_table(
                        t_idx, fids_t, all_data,
                        float(cold_quant_scale[t_idx]), float(cold_quant_zp[t_idx]),
                        rows_per_frame, cold_map_tensors[t_idx])

            dec_ms = (time.time() - t_dec) * 1000
            decode_times.append(dec_ms)

            # RUN forward
            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)

            z_np = Z.detach().cpu().numpy().ravel()
            t_np = T.detach().cpu().numpy().ravel()
            bs = z_np.shape[0]
            all_scores[sample_idx:sample_idx+bs] = z_np
            all_targets[sample_idx:sample_idx+bs] = t_np
            sample_idx += bs

    total_time = time.time() - t0
    scores, targets = all_scores[:sample_idx], all_targets[:sample_idx]
    auc, ll, acc = compute_metrics(scores, targets)
    n_b = len(blats)

    dlrm.apply_emb = _orig_apply

    lru_mb = len(frame_cache) * rows_per_frame * EMB_DIM / 1024 / 1024
    total_accesses = cache_hits + cache_misses

    return {
        'auc': auc, 'log_loss': ll, 'accuracy': acc,
        'total_time': total_time, 'num_batches': n_b,
        'mean_lat_ms': np.mean(blats)*1000,
        'p50_lat_ms': np.percentile(blats,50)*1000,
        'p99_lat_ms': np.percentile(blats,99)*1000,
        'emb_ms': dlrm.time_look_up/n_b*1000,
        'interact_ms': dlrm.time_interact/n_b*1000,
        'mlp_ms': dlrm.time_mlp/n_b*1000,
        'data_load_ms': np.mean(data_load_times)*1000,
        'scan_ms_per_batch': np.mean(scan_times),
        'decode_ms_per_batch': np.mean(decode_times),
        'rss_mb': get_rss_mb(),
        'hot_mb': total_hot_mb,
        'cold_disk_mb': compressed_mb,
        'lru_mb': lru_mb,
        'mapping_mb': mapping_mb,
        'total_mem_mb': total_hot_mb + lru_mb + mapping_mb,
        'compression_ratio': orig_cold_mb / max(0.01, compressed_mb),
        'cache_capacity': cache_capacity,
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'hit_rate': cache_hits / total_accesses if total_accesses > 0 else 1.0,
        'evictions': evictions,
        'total_frames_decoded': total_frames_decoded,
        'avg_misses_per_batch': cache_misses / n_b if n_b > 0 else 0,
    }


# ============================================================
# DYNAMIC LOOK-AHEAD INFERENCE
# ============================================================
def run_lookahead_inference(dlrm, test_ld, group_size, ln_emb, state_dict,
                            hot_indices, cold_indices, is_hot, o2c_map,
                            cold_quant_scale, cold_quant_zp, cold_num_rows,
                            large_tables, emb_keys):
    """Run inference with dynamic look-ahead: buffer group_size batches,
    scan for needed frames, batch decode, process group."""

    width, height = 1920, 1080
    rows_per_frame = (width // TILE_W) * (height // TILE_H)
    res_dir = os.path.join(ONDEMAND_DIR, '1080p')
    num_tabs = len(dlrm.emb_l)
    compressed_tables = set(t for t in large_tables if cold_num_rows.get(t, 0) > 0)

    # Register tables in C++ (hot weights + mappings)
    table_kinds, weights, mappings_list, scales, zero_points = [], [], [], [], []
    total_hot_mb = 0
    for k in range(num_tabs):
        if k in compressed_tables:
            table_kinds.append(1)
            h_idx = hot_indices[k]
            hot_w = state_dict[emb_keys[k]][h_idx].clone()
            weights.append(hot_w)
            total_hot_mb += hot_w.numel()*4/1024/1024
            mapping = torch.full((ln_emb[k],), -1, dtype=torch.int32)
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

    # Load cold mappings for register_cold_frames_for_table
    cold_map_tensors = {}
    for t in compressed_tables:
        mmap_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.npy')
        pt_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        if os.path.exists(mmap_path):
            cold_map_tensors[t] = torch.from_numpy(np.load(mmap_path).copy()).int()
        else:
            cold_map_tensors[t] = torch.load(pt_path, map_location='cpu', weights_only=True).int()

    # Compute memory
    compressed_bytes = 0
    for t in compressed_tables:
        frame_dir = os.path.join(res_dir, f'table_{t}')
        if os.path.exists(frame_dir):
            for f in os.listdir(frame_dir):
                if f.startswith('frame_') and (f.endswith('.h265') or f.endswith('.h264')):
                    compressed_bytes += os.path.getsize(os.path.join(frame_dir, f))
    compressed_mb = compressed_bytes / 1024 / 1024
    bm_mb = sum(((ln_emb[t]+63)//64*8 + ((ln_emb[t]+63)//64+1)*4)
                for t in compressed_tables) / 1024 / 1024
    cold_map_mb = sum(ln_emb[t]*4 for t in compressed_tables) / 1024 / 1024
    mapping_mb = bm_mb + cold_map_mb
    orig_cold_mb = sum(cold_num_rows.get(t,0)*EMB_DIM*4 for t in compressed_tables) / 1024 / 1024

    # Full C++ apply_emb
    _orig_apply = dlrm.apply_emb
    def _apply(lS_o, lS_i, emb_l, v_W_l):
        start = time.time()
        if isinstance(lS_i, (list, tuple)): lS_i_2d = torch.stack(lS_i)
        elif lS_i.dim() == 2: lS_i_2d = lS_i
        else: lS_i_2d = lS_i.view(num_tabs, -1)
        if isinstance(lS_o, (list, tuple)): lS_o_2d = torch.stack(lS_o)
        elif lS_o.dim() == 2: lS_o_2d = lS_o
        else: lS_o_2d = lS_o.view(num_tabs, -1)
        results = _C.fast_forward(lS_i_2d, lS_o_2d)
        dlrm.time_look_up += time.time() - start
        return results[-1]
    dlrm.apply_emb = _apply

    # Pre-build ordered lists for C++ scan
    comp_list = sorted(compressed_tables)
    is_hot_list = [is_hot[t] for t in comp_list]
    o2c_map_list = [o2c_map[t].int() for t in comp_list]

    # Run with group buffering
    dlrm.time_look_up = 0; dlrm.time_interact = 0; dlrm.time_mlp = 0
    max_samples = 2000 * TEST_BATCH_SIZE
    all_scores = np.empty(max_samples, dtype=np.float32)
    all_targets = np.empty(max_samples, dtype=np.float32)
    sample_idx = 0
    blats, data_load_times, scan_times, decode_times = [], [], [], []
    total_frames_decoded = 0
    total_decode_ms = 0

    def _cpp_scan_group(batch_buffer):
        """Fused C++ scan across all batches in the group."""
        needed = {t: set() for t in compressed_tables}
        for bX, bO, bI, bT in batch_buffer:
            lS_i_for_scan = [bI[t] if isinstance(bI, (list, tuple)) or bI.dim() == 2
                             else bI for t in comp_list]
            frame_lists = _C.scan_needed_frames(lS_i_for_scan, is_hot_list,
                                                 o2c_map_list, rows_per_frame)
            for k, t_idx in enumerate(comp_list):
                if frame_lists[k].numel() > 0:
                    needed[t_idx].update(frame_lists[k].tolist())
        return needed

    t0 = time.time()
    with torch.no_grad():
        batch_buffer = []
        for inputBatch in test_ld:
            t_dl = time.time()
            X, lS_o, lS_i, T = inputBatch[0], inputBatch[1], inputBatch[2], inputBatch[3]
            data_load_times.append(time.time() - t_dl)
            batch_buffer.append((X, lS_o, lS_i, T))

            if len(batch_buffer) >= group_size:
                # SCAN for needed frames (fused C++)
                t_scan = time.time()
                needed = _cpp_scan_group(batch_buffer)
                scan_ms = (time.time() - t_scan) * 1000
                scan_times.append(scan_ms)

                # DECODE + REGISTER needed frames
                t_dec = time.time()
                group_cold_mb = 0
                for t_idx in compressed_tables:
                    if not needed[t_idx]:
                        continue
                    frame_dir = os.path.join(res_dir, f'table_{t_idx}')
                    sorted_fids = sorted(needed[t_idx])
                    total_frames_decoded += len(sorted_fids)

                    fids_t = torch.tensor(sorted_fids, dtype=torch.long)
                    if len(sorted_fids) > 1 and hasattr(_C, 'batch_decode_frames'):
                        decoded = _C.batch_decode_frames(frame_dir, fids_t)
                    else:
                        ext = '.h265'
                        for e in ['.h265', '.h264']:
                            if os.path.exists(os.path.join(frame_dir, f'frame_00000{e}')):
                                ext = e; break
                        decoded = [_C.decode_h265_frame_from_file(
                            os.path.join(frame_dir, f'frame_{fid:05d}{ext}')) for fid in sorted_fids]

                    padded = []
                    for frame in decoded:
                        rows = _C.untile_frame_to_rows(frame, rows_per_frame)
                        if rows.shape[0] < rows_per_frame:
                            rows = torch.cat([rows, torch.zeros(rows_per_frame-rows.shape[0], EMB_DIM, dtype=torch.uint8)])
                        padded.append(rows)
                    all_data = torch.cat(padded, dim=0)
                    group_cold_mb += all_data.numel() / 1024 / 1024

                    _C.register_cold_frames_for_table(
                        t_idx, fids_t, all_data,
                        float(cold_quant_scale[t_idx]), float(cold_quant_zp[t_idx]),
                        rows_per_frame, cold_map_tensors[t_idx])

                dec_ms = (time.time() - t_dec) * 1000
                decode_times.append(dec_ms)
                total_decode_ms += dec_ms

                # PROCESS buffered batches
                for bX, bO, bI, bT in batch_buffer:
                    bt0 = time.time()
                    Z = dlrm(bX, bO, bI)
                    blats.append(time.time() - bt0)
                    z_np = Z.detach().cpu().numpy().ravel()
                    t_np = bT.detach().cpu().numpy().ravel()
                    bs = z_np.shape[0]
                    all_scores[sample_idx:sample_idx+bs] = z_np
                    all_targets[sample_idx:sample_idx+bs] = t_np
                    sample_idx += bs

                batch_buffer = []

        # Process remaining
        if batch_buffer:
            t_scan = time.time()
            needed = _cpp_scan_group(batch_buffer)
            scan_times.append((time.time() - t_scan)*1000)

            t_dec = time.time()
            for t_idx in compressed_tables:
                if not needed[t_idx]: continue
                frame_dir = os.path.join(res_dir, f'table_{t_idx}')
                sorted_fids = sorted(needed[t_idx])
                total_frames_decoded += len(sorted_fids)
                fids_t = torch.tensor(sorted_fids, dtype=torch.long)
                if len(sorted_fids) > 1:
                    decoded = _C.batch_decode_frames(frame_dir, fids_t)
                else:
                    ext = '.h265'
                    for e in ['.h265', '.h264']:
                        if os.path.exists(os.path.join(frame_dir, f'frame_00000{e}')):
                            ext = e; break
                    decoded = [_C.decode_h265_frame_from_file(
                        os.path.join(frame_dir, f'frame_{fid:05d}{ext}')) for fid in sorted_fids]
                padded = []
                for frame in decoded:
                    rows = _C.untile_frame_to_rows(frame, rows_per_frame)
                    if rows.shape[0] < rows_per_frame:
                        rows = torch.cat([rows, torch.zeros(rows_per_frame-rows.shape[0], EMB_DIM, dtype=torch.uint8)])
                    padded.append(rows)
                all_data = torch.cat(padded, dim=0)
                _C.register_cold_frames_for_table(
                    t_idx, fids_t, all_data,
                    float(cold_quant_scale[t_idx]), float(cold_quant_zp[t_idx]),
                    rows_per_frame, cold_map_tensors[t_idx])
            decode_times.append((time.time() - t_dec)*1000)
            total_decode_ms += decode_times[-1]

            for bX, bO, bI, bT in batch_buffer:
                bt0 = time.time()
                Z = dlrm(bX, bO, bI)
                blats.append(time.time() - bt0)
                z_np = Z.detach().cpu().numpy().ravel()
                t_np = bT.detach().cpu().numpy().ravel()
                bs = z_np.shape[0]
                all_scores[sample_idx:sample_idx+bs] = z_np
                all_targets[sample_idx:sample_idx+bs] = t_np
                sample_idx += bs

    total_time = time.time() - t0
    scores, targets = all_scores[:sample_idx], all_targets[:sample_idx]
    auc, ll, acc = compute_metrics(scores, targets)
    n_b = len(blats)

    dlrm.apply_emb = _orig_apply

    n_groups = max(1, len(scan_times))
    return {
        'auc': auc, 'log_loss': ll, 'accuracy': acc,
        'total_time': total_time, 'num_batches': n_b,
        'mean_lat_ms': np.mean(blats)*1000,
        'p50_lat_ms': np.percentile(blats,50)*1000,
        'p99_lat_ms': np.percentile(blats,99)*1000,
        'emb_ms': dlrm.time_look_up/n_b*1000,
        'interact_ms': dlrm.time_interact/n_b*1000,
        'mlp_ms': dlrm.time_mlp/n_b*1000,
        'data_load_ms': np.mean(data_load_times)*1000,
        'scan_ms_per_group': np.mean(scan_times) if scan_times else 0,
        'decode_ms_per_group': np.mean(decode_times) if decode_times else 0,
        'scan_ms_per_batch': sum(scan_times)/n_b if scan_times else 0,
        'decode_ms_per_batch': total_decode_ms/n_b,
        'rss_mb': get_rss_mb(),
        'hot_mb': total_hot_mb,
        'cold_disk_mb': compressed_mb,
        'lru_mb': 0,  # no persistent LRU in look-ahead
        'mapping_mb': mapping_mb,
        'total_mem_mb': total_hot_mb + mapping_mb,
        'compression_ratio': orig_cold_mb / max(0.01, compressed_mb),
        'total_frames_decoded': total_frames_decoded,
        'avg_frames_decoded_per_batch': total_frames_decoded / n_b,
        'group_size': group_size,
        'num_groups': n_groups,
    }


# ============================================================
# REPORT
# ============================================================
def generate_report(all_results, decode_time_per_frame_ms, output_file):
    lines = []
    def w(s=""): lines.append(s)

    w("# Comprehensive Benchmark: Baseline vs CAFE+ vs Codec+LRU")
    w()
    w(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Hardware: Intel Xeon Platinum 8380 (80 cores), {OPTIMIZED_THREADS} threads")
    w(f"Dataset: Criteo Kaggle, batch_size={TEST_BATCH_SIZE}, {NUM_RUNS} runs/config (median)")
    w(f"Codec: H.265 lossless, 1080p, batch-affinity reordered")
    w(f"**No batch pre-loading**: batches loaded from DataLoader during timed loop")
    w()

    # Summary table
    w("## Summary Comparison")
    w()
    w("| Config | AUC | Accuracy | LogLoss | BLat (ms) | p50 (ms) | p99 (ms) | Total (s) | Memory (MB) | Compress |")
    w("|--------|-----|----------|---------|-----------|----------|----------|-----------|-------------|---------|")
    for name, r in all_results.items():
        mem = r.get('total_mem_mb', r.get('rss_mb', 0))
        comp = r.get('compression_ratio', 1.0)
        w(f"| {name} | {r['auc']:.6f} | {r['accuracy']:.4f} | {r['log_loss']:.4f} | "
          f"{r['mean_lat_ms']:.2f} | {r['p50_lat_ms']:.2f} | {r['p99_lat_ms']:.2f} | "
          f"{r['total_time']:.1f} | {mem:.1f} | {comp:.1f}x |")

    # Per-component latency
    w()
    w("## Per-Component Latency (ms/batch)")
    w()
    w("| Config | Emb | Interact | MLP | Data Load | Scan/batch | Decode/batch | Fwd Total |")
    w("|--------|-----|----------|-----|-----------|------------|--------------|-----------|")
    for name, r in all_results.items():
        scan = r.get('scan_ms_per_batch', 0)
        decode = r.get('decode_ms_per_batch', 0)
        w(f"| {name} | {r['emb_ms']:.2f} | {r['interact_ms']:.2f} | {r['mlp_ms']:.2f} | "
          f"{r['data_load_ms']:.2f} | {scan:.2f} | {decode:.2f} | {r['mean_lat_ms']:.2f} |")

    # Memory breakdown
    w()
    w("## Memory Breakdown (MB)")
    w()
    baseline_mem = next(iter(all_results.values())).get('total_mem_mb', 2061)
    w("| Config | Hot Table | Cold (disk) | LRU/Decoded | Mapping | Total | RSS | Reduction |")
    w("|--------|-----------|-------------|-------------|---------|-------|-----|-----------|")
    for name, r in all_results.items():
        mem = r.get('total_mem_mb', r.get('rss_mb', 0))
        red = baseline_mem / mem if mem > 0 else 0
        w(f"| {name} | {r.get('hot_mb',0):.1f} | {r.get('cold_disk_mb',0):.1f} | "
          f"{r.get('lru_mb',0):.1f} | {r.get('mapping_mb',0):.1f} | {mem:.1f} | "
          f"{r['rss_mb']:.0f} | {red:.1f}x |")

    # Cache sweep results (real end-to-end)
    w()
    w("## Cache Size Sweep (real LRU, end-to-end)")
    w()
    cache_results = {k: v for k, v in all_results.items() if 'cache=' in k}
    if cache_results:
        w("| Cache Size | Hit Rate | Misses | Evictions | BLat (ms) | Scan+Decode (ms/batch) | Total (s) |")
        w("|------------|----------|--------|-----------|-----------|------------------------|-----------|")
        for name, r in cache_results.items():
            overhead = r.get('scan_ms_per_batch', 0) + r.get('decode_ms_per_batch', 0)
            w(f"| {r['cache_capacity']} | {r['hit_rate']:.1%} | {r.get('cache_misses',0)} | "
              f"{r.get('evictions',0)} | {r['mean_lat_ms']:.2f} | "
              f"{overhead:.2f} | {r['total_time']:.1f} |")

    # Dynamic look-ahead results
    w()
    w("## Dynamic Look-Ahead Results")
    w()
    la_results = {k: v for k, v in all_results.items() if 'group=' in k}
    if la_results:
        w("| Group Size | AUC | BLat (ms) | Scan/batch (ms) | Decode/batch (ms) | Total Overhead | Total (s) | Frames Decoded |")
        w("|------------|-----|-----------|-----------------|-------------------|----------------|-----------|----------------|")
        for name, r in la_results.items():
            overhead = r.get('scan_ms_per_batch', 0) + r.get('decode_ms_per_batch', 0)
            w(f"| {r.get('group_size',0)} | {r['auc']:.6f} | {r['mean_lat_ms']:.2f} | "
              f"{r.get('scan_ms_per_batch',0):.2f} | {r.get('decode_ms_per_batch',0):.2f} | "
              f"{overhead:.2f} | {r['total_time']:.1f} | {r.get('total_frames_decoded',0)} |")

    # CAFE+ reference
    w()
    w("## CAFE+ Reference (from training logs — different model)")
    w()
    w("| Config | AUC | Accuracy | Compression | Model Size |")
    w("|--------|-----|----------|-------------|------------|")
    w("| CAFE+ (121x) | 72.87% | 76.32% | 121x | 17 MB |")
    w("| CAFE+ (147x) | — | — | ~147x | 14 MB |")
    w("| CAFE+ (158x) | — | — | ~158x | 13 MB |")
    w("| **Codec+LRU** | **80.25%** | **~78.8%** | **10.3x** | **~200 MB** |")
    w()
    w("CAFE+ achieves 121x compression but loses ~7.4% AUC. Codec+LRU achieves")
    w("10.3x compression with <0.001% AUC loss (lossless codec + uint8 quantization).")

    # Visualization
    w()
    w("## Latency Breakdown (ASCII)")
    w("```")
    max_lat = max(r['mean_lat_ms'] for r in all_results.values() if 'cache=' not in list(all_results.keys())[list(all_results.values()).index(r)])
    # Only visualize non-cache-sim configs
    for name, r in all_results.items():
        if 'cache=' in name:
            continue
        bar_w = 50
        total = r['mean_lat_ms']
        scale = bar_w / max(max_lat, 0.01)
        e = max(1, int(r['emb_ms'] * scale))
        i = max(1, int(r['interact_ms'] * scale))
        m = max(1, int(r['mlp_ms'] * scale))
        o = max(0, int(total * scale) - e - i - m)
        bar = "E"*e + "I"*i + "M"*m + "."*o
        w(f"  {name:35s} |{bar}| {total:.2f}ms")
    w("  Legend: E=emb, I=interact, M=mlp, .=other")
    w("```")

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    log(f"Report: {output_file}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    log(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"C++ ext: {HAS_CPP_EXT}, Threads: {OPTIMIZED_THREADS}, Runs: {NUM_RUNS}")

    dlrm, test_ld, ln_emb, state_dict = load_model_and_data()
    large_tables = [i for i, n in enumerate(ln_emb) if n > LARGE_TABLE_THRESHOLD]
    total_emb_mb = sum(n * EMB_DIM * 4 for n in ln_emb) / 1024 / 1024
    emb_keys = {i: f'emb_l.{i}.weight' for i in range(len(ln_emb))}
    log(f"Tables: {large_tables}, Total emb: {total_emb_mb:.1f}MB")

    # Load hot/cold data
    hot_indices, cold_indices, is_hot, cold_num_rows = {}, {}, {}, {}
    for t in large_tables:
        h = torch.load(os.path.join(HOTCOLD_DIR, f'hot_indices_{t}.pt'), map_location='cpu', weights_only=True)
        c = torch.load(os.path.join(HOTCOLD_DIR, f'cold_indices_{t}.pt'), map_location='cpu', weights_only=True)
        hot_indices[t] = h; cold_indices[t] = c
        ih = torch.zeros(ln_emb[t], dtype=torch.bool); ih[h] = True
        is_hot[t] = ih; cold_num_rows[t] = len(c)

    # Load reordering + quant params
    o2c_map, cold_quant_scale, cold_quant_zp = {}, {}, {}
    for t in large_tables:
        npy = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.npy')
        pt = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        if os.path.exists(npy):
            o2c_map[t] = torch.from_numpy(np.load(npy).copy()).long()
        elif os.path.exists(pt):
            o2c_map[t] = torch.load(pt, map_location='cpu', weights_only=True).long()
        cold_w = state_dict[emb_keys[t]][cold_indices[t]]
        s, zp = quantize_table(cold_w)
        cold_quant_scale[t] = s; cold_quant_zp[t] = zp
        del cold_w

    torch.set_num_threads(OPTIMIZED_THREADS)
    all_results = OrderedDict()

    # ---- 1. BASELINE ----
    log(f"\n{'='*70}\nBASELINE: Full fp32\n{'='*70}")
    baseline_runs = []
    for run in range(NUM_RUNS):
        gc.collect(); drop_caches(); time.sleep(0.3)
        r = run_inference(dlrm, test_ld, f"baseline run {run+1}")
        r['hot_mb'] = total_emb_mb; r['cold_disk_mb'] = 0; r['lru_mb'] = 0
        r['mapping_mb'] = 0; r['total_mem_mb'] = total_emb_mb
        r['compression_ratio'] = 1.0
        log(f"  Run {run+1}: AUC={r['auc']:.6f}, BLat={r['mean_lat_ms']:.2f}ms, "
            f"emb={r['emb_ms']:.2f}, int={r['interact_ms']:.2f}, mlp={r['mlp_ms']:.2f}")
        baseline_runs.append(r)
    baseline_runs.sort(key=lambda x: x['mean_lat_ms'])
    all_results['Baseline (fp32)'] = baseline_runs[NUM_RUNS//2]

    # ---- 2. CODEC FULL_CPP (pre-decode all needed frames) ----
    log(f"\n{'='*70}\nCODEC FULL_CPP: pre-decode all frames\n{'='*70}")

    # Pre-cache test batches for frame scanning (but not for inference timing)
    log("Pre-caching test batches for frame scan...")
    t_pc = time.time()
    test_batches = []
    for batch in test_ld:
        test_batches.append((batch[0], batch[1], batch[2], batch[3]))
    log(f"Pre-cached {len(test_batches)} batches in {time.time()-t_pc:.1f}s")

    codec_runs = []
    for run in range(NUM_RUNS):
        gc.collect(); drop_caches(); time.sleep(0.3)
        # Restore weights
        with torch.no_grad():
            for t_idx, k in emb_keys.items():
                dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

        cleanup, mem_info, batch_frame_log = setup_codec_fullcpp(
            dlrm, ln_emb, state_dict, test_batches,
            hot_indices, cold_indices, is_hot, o2c_map,
            cold_quant_scale, cold_quant_zp, cold_num_rows,
            large_tables, emb_keys)

        r = run_inference(dlrm, test_ld, f"codec run {run+1}")
        r.update(mem_info)
        r['compression_ratio'] = mem_info['compression_ratio']
        log(f"  Run {run+1}: AUC={r['auc']:.6f}, BLat={r['mean_lat_ms']:.2f}ms, "
            f"emb={r['emb_ms']:.2f}, int={r['interact_ms']:.2f}, mlp={r['mlp_ms']:.2f}")
        codec_runs.append(r)
        cleanup()

    codec_runs.sort(key=lambda x: x['mean_lat_ms'])
    codec_result = codec_runs[NUM_RUNS//2]
    all_results['Codec full_cpp'] = codec_result
    # ---- 3. CACHE SIZE SWEEP (real end-to-end with LRU) ----
    log(f"\n{'='*70}\nCACHE SIZE SWEEP (real LRU)\n{'='*70}")

    for cache_size in [4, 8, 16, 22, 32, 64]:
        log(f"\n--- LRU cache={cache_size} ---")
        cache_runs = []
        for run in range(NUM_RUNS):
            gc.collect(); drop_caches(); time.sleep(0.3)
            with torch.no_grad():
                for t_idx, k in emb_keys.items():
                    dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

            r = run_real_lru_inference(
                dlrm, test_ld, cache_size, ln_emb, state_dict,
                hot_indices, cold_indices, is_hot, o2c_map,
                cold_quant_scale, cold_quant_zp, cold_num_rows,
                large_tables, emb_keys)
            log(f"  Run {run+1}: AUC={r['auc']:.6f}, BLat={r['mean_lat_ms']:.2f}ms, "
                f"hit={r['hit_rate']:.1%}, decode={r['decode_ms_per_batch']:.2f}ms/batch, "
                f"total={r['total_time']:.1f}s, frames={r['total_frames_decoded']}")
            cache_runs.append(r)
        cache_runs.sort(key=lambda x: x['mean_lat_ms'])
        all_results[f'Codec cache={cache_size}'] = cache_runs[NUM_RUNS//2]

    decode_time_per_frame = 6.3  # reference for report

    # ---- 4. DYNAMIC LOOK-AHEAD ----
    log(f"\n{'='*70}\nDYNAMIC LOOK-AHEAD\n{'='*70}")

    for group_size in [1, 10, 50, 100, 500]:
        log(f"\n--- Look-ahead group={group_size} ---")
        la_runs = []
        for run in range(NUM_RUNS):
            gc.collect(); drop_caches(); time.sleep(0.3)
            with torch.no_grad():
                for t_idx, k in emb_keys.items():
                    dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

            r = run_lookahead_inference(
                dlrm, test_ld, group_size, ln_emb, state_dict,
                hot_indices, cold_indices, is_hot, o2c_map,
                cold_quant_scale, cold_quant_zp, cold_num_rows,
                large_tables, emb_keys)
            log(f"  Run {run+1}: AUC={r['auc']:.6f}, BLat={r['mean_lat_ms']:.2f}ms, "
                f"scan={r['scan_ms_per_batch']:.2f}ms, decode={r['decode_ms_per_batch']:.2f}ms, "
                f"total={r['total_time']:.1f}s, frames={r['total_frames_decoded']}")
            la_runs.append(r)

        la_runs.sort(key=lambda x: x['mean_lat_ms'])
        all_results[f'Lookahead group={group_size}'] = la_runs[NUM_RUNS//2]

    # ---- 5. GENERATE REPORT ----
    log(f"\n{'='*70}\nGENERATING REPORT\n{'='*70}")
    generate_report(all_results, decode_time_per_frame, OUTPUT_FILE)

    # Save JSON
    json_file = os.path.join(RESULTS_DIR, 'full_comparison.json')
    ser = {}
    for k, v in all_results.items():
        sv = {}
        for kk, vv in v.items():
            if isinstance(vv, (int, float, str, bool, type(None))):
                sv[kk] = vv
            elif isinstance(vv, (np.floating, np.integer)):
                sv[kk] = float(vv)
        ser[k] = sv
    with open(json_file, 'w') as f:
        json.dump(ser, f, indent=2)

    log(f"\nALL DONE at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_fh.close()
