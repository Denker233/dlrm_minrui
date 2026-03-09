#!/usr/bin/env python3
"""
Benchmark: C++ fused scan+gather vs Python scan+gather for cold embeddings.

Tests three configurations:
1. Python scan + Python gather (original)
2. C++ scan + Python gather (partial optimization)
3. C++ scan + C++ gather (fully fused)
"""

import os, sys, time, json
import numpy as np
import torch
from collections import OrderedDict
import zstandard as zstd

sys.path.insert(0, '.')

RESULTS_DIR = "results"
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "codec_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMB_DIM = 16
ROWS_PER_FRAME = 129600
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
CACHE_SIZE = 16

MODEL_PATH = "./models/dlrm_kaggle_correct.pt"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class FrameCache:
    """Simple LRU frame cache for decoded Zstd frames."""
    def __init__(self, max_size=CACHE_SIZE):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, fid):
        if fid in self.cache:
            self.cache.move_to_end(fid)
            self.hits += 1
            return self.cache[fid]
        return None

    def put(self, fid, data):
        self.misses += 1
        self.cache[fid] = data
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def ensure(self, fid, decode_fn):
        result = self.get(fid)
        if result is None:
            result = decode_fn(fid)
            self.put(fid, result)
        return result


class TableStorage:
    """Per-table cold storage with Zstd compression."""
    def __init__(self, t_idx, cold_idx, weight, scale, zp, level=19):
        self.t_idx = t_idx
        self.scale = scale
        self.zp = zp
        self.emb_dim = weight.shape[1]
        self.n_total = weight.shape[0]

        quant = ((weight[cold_idx] / scale).round() + zp).clamp(0, 255).to(torch.uint8).numpy()

        order_path = os.path.join(REORDER_DIR, f"cold_order_{t_idx}.npy")
        if os.path.exists(order_path):
            order = np.load(order_path)
            quant = quant[order]
            cold_idx = cold_idx[torch.from_numpy(order.astype(np.int64))]

        self.cold_idx = cold_idx
        self.n_cold = len(cold_idx)

        self.orig_to_cold = torch.full((self.n_total,), -1, dtype=torch.int32)
        self.orig_to_cold[self.cold_idx] = torch.arange(self.n_cold, dtype=torch.int32)
        self.is_hot = torch.ones(self.n_total, dtype=torch.bool)
        self.is_hot[self.cold_idx] = False

        self.n_frames = (self.n_cold + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
        cctx = zstd.ZstdCompressor(level=level)
        self.compressed_frames = []
        self.frame_n_rows = []
        for i in range(self.n_frames):
            start = i * ROWS_PER_FRAME
            end = min(start + ROWS_PER_FRAME, self.n_cold)
            self.compressed_frames.append(cctx.compress(quant[start:end].tobytes()))
            self.frame_n_rows.append(end - start)

        self.dctx = zstd.ZstdDecompressor()

    def decode_frame(self, fid):
        raw = self.dctx.decompress(self.compressed_frames[fid])
        n_rows = self.frame_n_rows[fid]
        uint8_data = np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, self.emb_dim)
        fp32_data = (uint8_data.astype(np.float32) - self.zp) * self.scale
        return torch.from_numpy(fp32_data).contiguous()


def python_scan_gather(lS_i, storages, caches, comp_tables):
    """Pure Python scan + gather."""
    writes = {}  # t_idx -> (orig_indices, gathered_embeddings)
    for t_idx in comp_tables:
        storage = storages[t_idx]
        cache = caches[t_idx]

        if isinstance(lS_i, (list, tuple)):
            indices = lS_i[t_idx]
        elif lS_i.dim() == 2:
            indices = lS_i[t_idx]
        else:
            indices = lS_i

        cold_mask = ~storage.is_hot[indices]
        if not cold_mask.any():
            continue

        cold_orig = indices[cold_mask]
        cold_reordered = storage.orig_to_cold[cold_orig].long()
        frame_ids = (cold_reordered // ROWS_PER_FRAME).unique().tolist()

        for fid in frame_ids:
            cache.ensure(fid, storage.decode_frame)

        frame_ids_all = cold_reordered // ROWS_PER_FRAME
        rows_in_frame = cold_reordered % ROWS_PER_FRAME
        gathered = torch.zeros(len(cold_orig), storage.emb_dim)
        for fid in frame_ids:
            fmask = (frame_ids_all == fid)
            if fmask.any():
                gathered[fmask] = cache.cache[fid][rows_in_frame[fmask]]

        writes[t_idx] = (cold_orig, gathered)
    return writes


def cpp_scan_python_gather(lS_i, storages, caches, comp_tables, _C,
                           is_hot_list, o2c_map_list):
    """C++ scan + Python gather."""
    lS_i_for_scan = []
    for t_idx in comp_tables:
        if isinstance(lS_i, (list, tuple)):
            lS_i_for_scan.append(lS_i[t_idx].long())
        elif lS_i.dim() == 2:
            lS_i_for_scan.append(lS_i[t_idx].long())
        else:
            lS_i_for_scan.append(lS_i.long())

    frame_lists = _C.scan_needed_frames(
        lS_i_for_scan, is_hot_list, o2c_map_list, ROWS_PER_FRAME
    )

    writes = {}
    for k, t_idx in enumerate(comp_tables):
        storage = storages[t_idx]
        cache = caches[t_idx]
        needed = frame_lists[k]
        if needed.numel() == 0:
            continue

        frame_ids = needed.tolist()
        for fid in frame_ids:
            cache.ensure(fid, storage.decode_frame)

        indices = lS_i_for_scan[k]
        cold_mask = ~storage.is_hot[indices]
        if not cold_mask.any():
            continue

        cold_orig = indices[cold_mask]
        cold_reordered = storage.orig_to_cold[cold_orig].long()
        frame_ids_all = cold_reordered // ROWS_PER_FRAME
        rows_in_frame = cold_reordered % ROWS_PER_FRAME
        gathered = torch.zeros(len(cold_orig), storage.emb_dim)
        for fid in frame_ids:
            fmask = (frame_ids_all == fid)
            if fmask.any():
                gathered[fmask] = cache.cache[fid][rows_in_frame[fmask]]

        writes[t_idx] = (cold_orig, gathered)
    return writes


def cpp_scan_cpp_gather(lS_i, storages, caches, comp_tables, _C,
                        is_hot_list, o2c_map_list):
    """C++ scan + C++ gather."""
    lS_i_for_scan = []
    for t_idx in comp_tables:
        if isinstance(lS_i, (list, tuple)):
            lS_i_for_scan.append(lS_i[t_idx].long())
        elif lS_i.dim() == 2:
            lS_i_for_scan.append(lS_i[t_idx].long())
        else:
            lS_i_for_scan.append(lS_i.long())

    # C++ scan
    frame_lists = _C.scan_needed_frames(
        lS_i_for_scan, is_hot_list, o2c_map_list, ROWS_PER_FRAME
    )

    # Ensure frames are in cache, collect cached frame tensors for C++ gather
    cached_frames_list = []
    frame_offsets_list = []
    for k, t_idx in enumerate(comp_tables):
        storage = storages[t_idx]
        cache = caches[t_idx]
        needed = frame_lists[k]

        if needed.numel() > 0:
            for fid in needed.tolist():
                cache.ensure(fid, storage.decode_frame)

        # Build contiguous list of all cached frames (by frame ID order)
        # We need to pass frames that cover all possible frame IDs for this table
        min_fid = 0
        frame_tensors = []
        for fid in range(storage.n_frames):
            if fid in cache.cache:
                frame_tensors.append(cache.cache[fid])
            else:
                # Placeholder - won't be accessed
                frame_tensors.append(torch.empty(0, dtype=torch.float32))

        cached_frames_list.append(frame_tensors)
        frame_offsets_list.append(min_fid)

    # C++ gather
    results = _C.gather_cold_embeddings(
        lS_i_for_scan, is_hot_list, o2c_map_list,
        cached_frames_list, frame_offsets_list,
        ROWS_PER_FRAME, EMB_DIM
    )

    writes = {}
    for k, t_idx in enumerate(comp_tables):
        gathered = results[2*k]
        orig_indices = results[2*k+1]
        if gathered.size(0) > 0:
            writes[t_idx] = (orig_indices, gathered)
    return writes


def main():
    import compressed_emb as _C

    log("Loading model...")
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']

    log("Building cold storage...")
    storages = {}
    for t_idx in LARGE_TABLES:
        cold_idx_path = os.path.join(HOTCOLD_DIR, f"cold_indices_{t_idx}.pt")
        if not os.path.exists(cold_idx_path):
            continue
        cold_idx = torch.load(cold_idx_path, weights_only=False)
        if cold_idx.numel() == 0:
            continue
        weight = sd[f'emb_l.{t_idx}.weight']
        mn, mx = weight[cold_idx].min().item(), weight[cold_idx].max().item()
        scale = (mx - mn) / 255.0
        if scale == 0: scale = 1.0
        zp = round(-mn / scale)
        storages[t_idx] = TableStorage(t_idx, cold_idx, weight, scale, zp, level=19)
        log(f"  Table {t_idx}: {storages[t_idx].n_cold:,} cold, {storages[t_idx].n_frames} frames")

    comp_tables = sorted(storages.keys())
    is_hot_list = [storages[t].is_hot for t in comp_tables]
    o2c_map_list = [storages[t].orig_to_cold for t in comp_tables]

    # Load test data
    log("Loading test data...")
    from benchmark_full_comparison import load_model_and_data
    _, test_ld, _, _ = load_model_and_data()

    # Warm up caches (first pass fills them)
    log("Warming up caches...")
    caches = {t: FrameCache(CACHE_SIZE) for t in comp_tables}
    n_warm = 50
    for i, batch in enumerate(test_ld):
        if i >= n_warm:
            break
        lS_i = batch[2]
        python_scan_gather(lS_i, storages, caches, comp_tables)

    # Benchmark A: Python scan + Python gather
    log("\nBenchmarking Python scan + Python gather...")
    caches_a = {t: FrameCache(CACHE_SIZE) for t in comp_tables}
    times_a = []
    for batch in test_ld:
        lS_i = batch[2]
        t0 = time.time()
        python_scan_gather(lS_i, storages, caches_a, comp_tables)
        times_a.append(time.time() - t0)

    # Benchmark B: C++ scan + Python gather
    log("Benchmarking C++ scan + Python gather...")
    caches_b = {t: FrameCache(CACHE_SIZE) for t in comp_tables}
    times_b = []
    for batch in test_ld:
        lS_i = batch[2]
        t0 = time.time()
        cpp_scan_python_gather(lS_i, storages, caches_b, comp_tables, _C,
                               is_hot_list, o2c_map_list)
        times_b.append(time.time() - t0)

    # Benchmark C: C++ scan + C++ gather
    log("Benchmarking C++ scan + C++ gather...")
    caches_c = {t: FrameCache(CACHE_SIZE) for t in comp_tables}
    times_c = []
    for batch in test_ld:
        lS_i = batch[2]
        t0 = time.time()
        cpp_scan_cpp_gather(lS_i, storages, caches_c, comp_tables, _C,
                            is_hot_list, o2c_map_list)
        times_c.append(time.time() - t0)

    # Verify correctness: compare outputs for last batch
    log("\nVerifying correctness...")
    lS_i = batch[2]
    cache_v = {t: FrameCache(CACHE_SIZE) for t in comp_tables}
    # Warm up
    for i, b in enumerate(test_ld):
        if i >= 50:
            break
        python_scan_gather(b[2], storages, cache_v, comp_tables)

    py_result = python_scan_gather(lS_i, storages, cache_v, comp_tables)
    cpp_result = cpp_scan_cpp_gather(lS_i, storages, cache_v, comp_tables, _C,
                                      is_hot_list, o2c_map_list)

    all_match = True
    for t_idx in comp_tables:
        if t_idx in py_result and t_idx in cpp_result:
            py_orig, py_gathered = py_result[t_idx]
            cpp_orig, cpp_gathered = cpp_result[t_idx]
            if not torch.equal(py_orig, cpp_orig):
                log(f"  Table {t_idx}: orig indices MISMATCH")
                all_match = False
            elif not torch.allclose(py_gathered, cpp_gathered, atol=1e-6):
                max_diff = (py_gathered - cpp_gathered).abs().max().item()
                log(f"  Table {t_idx}: gathered MISMATCH (max_diff={max_diff})")
                all_match = False

    log(f"  Correctness: {'PASS' if all_match else 'FAIL'}")

    # Results
    log(f"\n{'='*60}")
    log("SCAN+GATHER BENCHMARK RESULTS")
    log(f"{'='*60}")
    log(f"  Batches: {len(times_a)}")

    for label, times in [
        ("Python scan + Python gather", times_a),
        ("C++ scan + Python gather", times_b),
        ("C++ scan + C++ gather", times_c),
    ]:
        mean_ms = np.mean(times) * 1000
        p50_ms = np.percentile(times, 50) * 1000
        p99_ms = np.percentile(times, 99) * 1000
        log(f"\n  {label}:")
        log(f"    Mean: {mean_ms:.3f} ms")
        log(f"    p50:  {p50_ms:.3f} ms")
        log(f"    p99:  {p99_ms:.3f} ms")

    a_mean = np.mean(times_a) * 1000
    b_mean = np.mean(times_b) * 1000
    c_mean = np.mean(times_c) * 1000
    log(f"\n  Speedup (C++ scan only): {a_mean / b_mean:.2f}x")
    log(f"  Speedup (C++ scan+gather): {a_mean / c_mean:.2f}x")

    results = {
        'n_batches': len(times_a),
        'python_python': {
            'mean_ms': a_mean,
            'p50_ms': np.percentile(times_a, 50) * 1000,
            'p99_ms': np.percentile(times_a, 99) * 1000,
        },
        'cpp_python': {
            'mean_ms': b_mean,
            'p50_ms': np.percentile(times_b, 50) * 1000,
            'p99_ms': np.percentile(times_b, 99) * 1000,
        },
        'cpp_cpp': {
            'mean_ms': c_mean,
            'p50_ms': np.percentile(times_c, 50) * 1000,
            'p99_ms': np.percentile(times_c, 99) * 1000,
        },
        'speedup_scan_only': a_mean / b_mean,
        'speedup_full': a_mean / c_mean,
        'correctness': all_match,
    }
    json_path = os.path.join(OUTPUT_DIR, 'scan_gather_benchmark.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
