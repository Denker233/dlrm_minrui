#!/usr/bin/env python3
"""Batch size sweep: Baseline vs Codec cache=16 vs Lookahead g=500.
Smaller batch sizes reduce data loading dominance to expose forward pass speedup."""

import os, sys, time, gc, subprocess, json, io
import numpy as np
import torch
from collections import OrderedDict
import psutil
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C

MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
PROCESSED_DATA = os.path.expanduser("~/input/kaggleAdDisplayChallenge_processed.npz")
DATA_FILE = os.path.expanduser("~/input/train.txt")
EMB_DIM = 16
TILE_H = TILE_W = 4
THREADS = 40
HOTCOLD_DIR = "results/hotcold"
REORDER_DIR = "results/reorder"
ONDEMAND_DIR = "results/ondemand"
NUM_RUNS = 3

def decode_frame_from_bytes_cpp(data):
    """Decode H.265 frame from in-memory bytes using C++ extension."""
    compressed_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return _C.decode_h265_frame_from_bytes(compressed_t)


def batch_decode_frames_from_bytes(compressed_bytes_table, fids):
    """Decode multiple frames from in-memory bytes."""
    return [decode_frame_from_bytes_cpp(compressed_bytes_table[fid]) for fid in fids]


def load_compressed_bytes_from_disk(compressed_tables, res_dir):
    """Load all H.265 compressed frame bytes from disk into RAM."""
    compressed_bytes_map = {}
    for t_idx in compressed_tables:
        frame_dir = os.path.join(res_dir, f'table_{t_idx}')
        if not os.path.exists(frame_dir):
            continue
        compressed_bytes_map[t_idx] = {}
        frame_files = sorted([f for f in os.listdir(frame_dir)
                              if f.startswith('frame_') and (f.endswith('.h265') or f.endswith('.h264'))])
        for ff in frame_files:
            fid = int(ff.split('_')[1].split('.')[0])
            with open(os.path.join(frame_dir, ff), 'rb') as fh:
                compressed_bytes_map[t_idx][fid] = fh.read()
    return compressed_bytes_map


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def drop_caches():
    try:
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except Exception: pass

def compute_metrics(scores, targets):
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
    auc = roc_auc_score(targets, probs)
    return auc

def quantize_table(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    return s, zp

def make_dataloader(batch_size):
    """Create a test DataLoader with specified batch size."""
    import dlrm_data_pytorch as dp
    class Args:
        pass
    a = Args()
    a.arch_sparse_feature_size = EMB_DIM
    a.arch_mlp_bot = "13-512-256-64-16"; a.arch_mlp_top = "512-256-1"
    a.arch_interaction_op = "dot"; a.arch_interaction_itself = False
    a.data_generation = "dataset"; a.data_set = "kaggle"
    a.raw_data_file = DATA_FILE; a.processed_data_file = PROCESSED_DATA
    a.loss_function = "bce"; a.max_ind_range = -1
    a.test_mini_batch_size = batch_size
    a.test_num_workers = 0; a.num_workers = 0
    a.mlperf_logging = False; a.memory_map = False; a.data_randomize = "total"
    a.data_trace_enable_padding = False; a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10; a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 128; a.round_targets = True
    a.mlperf_bin_loader = False; a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(a)
    return test_ld, np.array(train_data.counts)

# ============================================================
# BASELINE INFERENCE
# ============================================================
def run_baseline(dlrm, test_ld):
    dlrm.time_look_up = 0; dlrm.time_interact = 0; dlrm.time_mlp = 0
    blats = []
    n_b = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in test_ld:
            X, lS_o, lS_i, T = batch[0], batch[1], batch[2], batch[3]
            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            n_b += 1
    total = time.time() - t0
    return {
        'blat': np.mean(blats)*1000,
        'emb': dlrm.time_look_up/n_b*1000,
        'interact': dlrm.time_interact/n_b*1000,
        'mlp': dlrm.time_mlp/n_b*1000,
        'total': total,
        'n_batches': n_b,
    }

# ============================================================
# CODEC CACHE=16 INFERENCE
# ============================================================
def run_cache16(dlrm, test_ld, ln_emb, state_dict,
                hot_indices, cold_indices, is_hot, o2c_map,
                cold_quant_scale, cold_quant_zp, cold_num_rows,
                large_tables, emb_keys, compressed_bytes_map=None):
    width, height = 1920, 1080
    rows_per_frame = (width // TILE_W) * (height // TILE_H)
    res_dir = os.path.join(ONDEMAND_DIR, '1080p')
    num_tabs = len(dlrm.emb_l)
    compressed_tables = sorted(t for t in large_tables if cold_num_rows.get(t, 0) > 0)

    # Register tables
    table_kinds, weights, mappings_list, scales, zero_points = [], [], [], [], []
    for k in range(num_tabs):
        if k in compressed_tables:
            table_kinds.append(1)
            h_idx = hot_indices[k]
            hot_w = state_dict[emb_keys[k]][h_idx].clone()
            weights.append(hot_w)
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

    cold_map_tensors = {}
    for t in compressed_tables:
        mmap_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.npy')
        pt_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        if os.path.exists(mmap_path):
            cold_map_tensors[t] = torch.from_numpy(np.load(mmap_path).copy()).int()
        else:
            cold_map_tensors[t] = torch.load(pt_path, map_location='cpu', weights_only=True).int()

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

    # Pre-build C++ scan inputs
    comp_list = sorted(compressed_tables)
    is_hot_list = [is_hot[t] for t in comp_list]
    o2c_map_list = [o2c_map[t].int() for t in comp_list]

    frame_cache = OrderedDict()
    cache_capacity = 16
    blats, scan_times, decode_times = [], [], []

    dlrm.time_look_up = 0; dlrm.time_interact = 0; dlrm.time_mlp = 0
    n_b = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in test_ld:
            X, lS_o, lS_i, T = batch[0], batch[1], batch[2], batch[3]

            # C++ fused scan
            t_scan = time.time()
            lS_i_for_scan = [lS_i[t] if isinstance(lS_i, (list, tuple)) or lS_i.dim() == 2
                             else lS_i for t in comp_list]
            frame_lists = _C.scan_needed_frames(lS_i_for_scan, is_hot_list,
                                                 o2c_map_list, rows_per_frame)
            needed = {}
            for k, t_idx in enumerate(comp_list):
                if frame_lists[k].numel() > 0:
                    needed[t_idx] = set(frame_lists[k].tolist())
            scan_times.append((time.time() - t_scan) * 1000)

            # Cache check + decode
            t_dec = time.time()
            dirty_tables = set()
            for t_idx, fids in needed.items():
                for fid in fids:
                    key = (t_idx, fid)
                    if key in frame_cache:
                        frame_cache.move_to_end(key)
                    else:
                        if compressed_bytes_map and t_idx in compressed_bytes_map and fid in compressed_bytes_map[t_idx]:
                            frame = decode_frame_from_bytes_cpp(compressed_bytes_map[t_idx][fid])
                        else:
                            frame_dir = os.path.join(res_dir, f'table_{t_idx}')
                            fpath = os.path.join(frame_dir, f'frame_{fid:05d}.h265')
                            if not os.path.exists(fpath):
                                fpath = os.path.join(frame_dir, f'frame_{fid:05d}.h264')
                            frame = _C.decode_h265_frame_from_file(fpath)
                        rows = _C.untile_frame_to_rows(frame, rows_per_frame)
                        if rows.shape[0] < rows_per_frame:
                            rows = torch.cat([rows, torch.zeros(
                                rows_per_frame - rows.shape[0], EMB_DIM, dtype=torch.uint8)])
                        frame_cache[(t_idx, fid)] = rows
                        dirty_tables.add(t_idx)

            while len(frame_cache) > cache_capacity:
                evicted_key, _ = frame_cache.popitem(last=False)
                dirty_tables.add(evicted_key[0])

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
            decode_times.append((time.time() - t_dec) * 1000)

            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            n_b += 1

    total = time.time() - t0
    dlrm.apply_emb = _orig_apply

    return {
        'blat': np.mean(blats)*1000,
        'emb': dlrm.time_look_up/n_b*1000,
        'interact': dlrm.time_interact/n_b*1000,
        'mlp': dlrm.time_mlp/n_b*1000,
        'scan': np.mean(scan_times),
        'decode': np.mean(decode_times),
        'total': total,
        'n_batches': n_b,
    }

# ============================================================
# LOOKAHEAD g=500 INFERENCE
# ============================================================
def run_lookahead500(dlrm, test_ld, ln_emb, state_dict,
                     hot_indices, cold_indices, is_hot, o2c_map,
                     cold_quant_scale, cold_quant_zp, cold_num_rows,
                     large_tables, emb_keys, compressed_bytes_map=None):
    width, height = 1920, 1080
    rows_per_frame = (width // TILE_W) * (height // TILE_H)
    res_dir = os.path.join(ONDEMAND_DIR, '1080p')
    num_tabs = len(dlrm.emb_l)
    compressed_tables = sorted(t for t in large_tables if cold_num_rows.get(t, 0) > 0)
    group_size = 500

    # Register tables
    table_kinds, weights, mappings_list, scales, zero_points = [], [], [], [], []
    for k in range(num_tabs):
        if k in compressed_tables:
            table_kinds.append(1)
            h_idx = hot_indices[k]
            hot_w = state_dict[emb_keys[k]][h_idx].clone()
            weights.append(hot_w)
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

    cold_map_tensors = {}
    for t in compressed_tables:
        mmap_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.npy')
        pt_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        if os.path.exists(mmap_path):
            cold_map_tensors[t] = torch.from_numpy(np.load(mmap_path).copy()).int()
        else:
            cold_map_tensors[t] = torch.load(pt_path, map_location='cpu', weights_only=True).int()

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

    comp_list = sorted(compressed_tables)
    is_hot_list = [is_hot[t] for t in comp_list]
    o2c_map_list = [o2c_map[t].int() for t in comp_list]

    blats, scan_times, decode_times = [], [], []
    dlrm.time_look_up = 0; dlrm.time_interact = 0; dlrm.time_mlp = 0
    n_b = 0

    def _cpp_scan_group(batch_buffer):
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

    def process_group(batch_buffer):
        nonlocal n_b
        t_scan = time.time()
        needed = _cpp_scan_group(batch_buffer)
        scan_ms = (time.time() - t_scan) * 1000
        scan_times.append(scan_ms)

        t_dec = time.time()
        for t_idx in compressed_tables:
            if not needed[t_idx]: continue
            sorted_fids = sorted(needed[t_idx])
            fids_t = torch.tensor(sorted_fids, dtype=torch.long)
            if compressed_bytes_map and t_idx in compressed_bytes_map:
                decoded = batch_decode_frames_from_bytes(
                    compressed_bytes_map[t_idx], sorted_fids)
            else:
                frame_dir = os.path.join(res_dir, f'table_{t_idx}')
                if len(sorted_fids) > 1:
                    decoded = _C.batch_decode_frames(frame_dir, fids_t)
                else:
                    fpath = os.path.join(frame_dir, f'frame_{sorted_fids[0]:05d}.h265')
                    if not os.path.exists(fpath):
                        fpath = os.path.join(frame_dir, f'frame_{sorted_fids[0]:05d}.h264')
                    decoded = [_C.decode_h265_frame_from_file(fpath)]
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
        decode_times.append((time.time() - t_dec) * 1000)

        for bX, bO, bI, bT in batch_buffer:
            bt0 = time.time()
            Z = dlrm(bX, bO, bI)
            blats.append(time.time() - bt0)
            n_b += 1

    t0 = time.time()
    with torch.no_grad():
        batch_buffer = []
        for batch in test_ld:
            X, lS_o, lS_i, T = batch[0], batch[1], batch[2], batch[3]
            batch_buffer.append((X, lS_o, lS_i, T))
            if len(batch_buffer) >= group_size:
                process_group(batch_buffer)
                batch_buffer = []
        if batch_buffer:
            process_group(batch_buffer)

    total = time.time() - t0
    dlrm.apply_emb = _orig_apply

    return {
        'blat': np.mean(blats)*1000,
        'emb': dlrm.time_look_up/n_b*1000,
        'interact': dlrm.time_interact/n_b*1000,
        'mlp': dlrm.time_mlp/n_b*1000,
        'scan': np.mean(scan_times) / max(1, group_size),  # per batch
        'decode': np.mean(decode_times) / max(1, group_size),  # per batch
        'total': total,
        'n_batches': n_b,
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    torch.set_num_threads(THREADS)

    # Load model once (reuse existing infrastructure)
    log("Loading model...")
    from benchmark_full_comparison import load_model_and_data
    dlrm, _, ln_emb, state_dict = load_model_and_data()
    log(f"Model loaded: {len(ln_emb)} tables")

    large_tables = [i for i, n in enumerate(ln_emb) if n > 50000]
    emb_keys = {i: f'emb_l.{i}.weight' for i in range(len(ln_emb))}

    # Load hot/cold
    hot_indices, cold_indices, is_hot, cold_num_rows = {}, {}, {}, {}
    o2c_map, cold_quant_scale, cold_quant_zp = {}, {}, {}
    for t in large_tables:
        h = torch.load(os.path.join(HOTCOLD_DIR, f'hot_indices_{t}.pt'), map_location='cpu', weights_only=True)
        c = torch.load(os.path.join(HOTCOLD_DIR, f'cold_indices_{t}.pt'), map_location='cpu', weights_only=True)
        hot_indices[t] = h; cold_indices[t] = c
        ih = torch.zeros(ln_emb[t], dtype=torch.bool); ih[h] = True
        is_hot[t] = ih; cold_num_rows[t] = len(c)
        npy = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.npy')
        pt = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        if os.path.exists(npy):
            o2c_map[t] = torch.from_numpy(np.load(npy).copy()).long()
        elif os.path.exists(pt):
            o2c_map[t] = torch.load(pt, map_location='cpu', weights_only=True).long()
        cold_w = state_dict[emb_keys[t]][c]
        s, zp = quantize_table(cold_w)
        cold_quant_scale[t] = s; cold_quant_zp[t] = zp
        del cold_w

    # Load compressed H.265 frame bytes into RAM for in-memory decode
    res_dir = os.path.join(ONDEMAND_DIR, '1080p')
    compressed_tables_all = sorted(t for t in large_tables if cold_num_rows.get(t, 0) > 0)
    log("Loading compressed H.265 frame bytes into RAM...")
    compressed_bytes_map = load_compressed_bytes_from_disk(compressed_tables_all, res_dir)
    total_loaded = sum(sum(len(v) for v in tb.values()) for tb in compressed_bytes_map.values())
    log(f"  Loaded {total_loaded/1024/1024:.1f}MB compressed bytes")

    # Batch size sweep
    batch_sizes = [256, 512, 1024, 2048, 4096]
    all_results = {}

    for bs in batch_sizes:
        log(f"\n{'='*60}")
        log(f"BATCH SIZE = {bs}")
        log(f"{'='*60}")

        log(f"Creating DataLoader (bs={bs})...")
        test_ld, _ = make_dataloader(bs)

        # --- Baseline ---
        log(f"  Running baseline...")
        base_runs = []
        for run in range(NUM_RUNS):
            gc.collect(); drop_caches()
            with torch.no_grad():
                for t_idx, k in emb_keys.items():
                    dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()
            r = run_baseline(dlrm, test_ld)
            log(f"    Run {run+1}: BLat={r['blat']:.2f}ms, emb={r['emb']:.2f}, "
                f"int={r['interact']:.2f}, mlp={r['mlp']:.2f}, total={r['total']:.1f}s")
            base_runs.append(r)
        base_runs.sort(key=lambda x: x['blat'])
        base = base_runs[NUM_RUNS//2]

        # --- Cache=16 ---
        log(f"  Running codec cache=16...")
        cache_runs = []
        for run in range(NUM_RUNS):
            gc.collect(); drop_caches()
            with torch.no_grad():
                for t_idx, k in emb_keys.items():
                    dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()
            r = run_cache16(dlrm, test_ld, ln_emb, state_dict,
                           hot_indices, cold_indices, is_hot, o2c_map,
                           cold_quant_scale, cold_quant_zp, cold_num_rows,
                           large_tables, emb_keys, compressed_bytes_map)
            log(f"    Run {run+1}: BLat={r['blat']:.2f}ms, emb={r['emb']:.2f}, "
                f"int={r['interact']:.2f}, mlp={r['mlp']:.2f}, "
                f"scan={r['scan']:.2f}, dec={r['decode']:.2f}, total={r['total']:.1f}s")
            cache_runs.append(r)
        cache_runs.sort(key=lambda x: x['blat'])
        cache = cache_runs[NUM_RUNS//2]

        # --- Lookahead g=500 ---
        log(f"  Running lookahead g=500...")
        la_runs = []
        for run in range(NUM_RUNS):
            gc.collect(); drop_caches()
            with torch.no_grad():
                for t_idx, k in emb_keys.items():
                    dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()
            r = run_lookahead500(dlrm, test_ld, ln_emb, state_dict,
                                hot_indices, cold_indices, is_hot, o2c_map,
                                cold_quant_scale, cold_quant_zp, cold_num_rows,
                                large_tables, emb_keys, compressed_bytes_map)
            log(f"    Run {run+1}: BLat={r['blat']:.2f}ms, emb={r['emb']:.2f}, "
                f"int={r['interact']:.2f}, mlp={r['mlp']:.2f}, "
                f"scan={r['scan']:.3f}, dec={r['decode']:.3f}, total={r['total']:.1f}s")
            la_runs.append(r)
        la_runs.sort(key=lambda x: x['blat'])
        la = la_runs[NUM_RUNS//2]

        all_results[bs] = {'baseline': base, 'cache16': cache, 'lookahead500': la}

    # Print summary
    log(f"\n{'='*80}")
    log("SUMMARY: Batch Size Sweep")
    log(f"{'='*80}")

    print(f"\n## Forward Latency (ms/batch)")
    print(f"| Batch Size | Baseline | Cache=16 | LA g=500 | Cache Speedup | LA Speedup |")
    print(f"|------------|----------|----------|----------|---------------|------------|")
    for bs in batch_sizes:
        b = all_results[bs]['baseline']
        c = all_results[bs]['cache16']
        l = all_results[bs]['lookahead500']
        c_total = c['blat'] + c['scan'] + c['decode']
        l_total = l['blat'] + l['scan'] + l['decode']
        print(f"| {bs} | {b['blat']:.2f} | {c_total:.2f} | {l_total:.2f} | "
              f"{b['blat']/c_total:.2f}x | {b['blat']/l_total:.2f}x |")

    print(f"\n## Wall Clock Total (s)")
    print(f"| Batch Size | Baseline | Cache=16 | LA g=500 | Cache Speedup | LA Speedup |")
    print(f"|------------|----------|----------|----------|---------------|------------|")
    for bs in batch_sizes:
        b = all_results[bs]['baseline']
        c = all_results[bs]['cache16']
        l = all_results[bs]['lookahead500']
        print(f"| {bs} | {b['total']:.1f} | {c['total']:.1f} | {l['total']:.1f} | "
              f"{b['total']/c['total']:.2f}x | {b['total']/l['total']:.2f}x |")

    print(f"\n## Per-Component (ms/batch)")
    print(f"| Batch Size | Config | Emb | Interact | MLP | Scan | Decode |")
    print(f"|------------|--------|-----|----------|-----|------|--------|")
    for bs in batch_sizes:
        b = all_results[bs]['baseline']
        c = all_results[bs]['cache16']
        l = all_results[bs]['lookahead500']
        print(f"| {bs} | Baseline | {b['emb']:.2f} | {b['interact']:.2f} | {b['mlp']:.2f} | — | — |")
        print(f"| {bs} | Cache=16 | {c['emb']:.2f} | {c['interact']:.2f} | {c['mlp']:.2f} | {c['scan']:.2f} | {c['decode']:.2f} |")
        print(f"| {bs} | LA g=500 | {l['emb']:.2f} | {l['interact']:.2f} | {l['mlp']:.2f} | {l['scan']:.3f} | {l['decode']:.3f} |")

    log("\nDONE")
