#!/usr/bin/env python3
"""
A/B test: Python scan vs C++ scan_needed_frames for batch index scanning.

Measures the per-batch overhead of determining which cold frames are needed.
The Python path uses torch tensor ops (masking, integer division, unique).
The C++ path uses a single fused call via compressed_emb.scan_needed_frames.
"""

import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, '.')

RESULTS_DIR = "results"
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "codec_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROWS_PER_FRAME = 129600
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]

MODEL_PATH = "./models/dlrm_kaggle_correct.pt"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def python_scan(lS_i, comp_tables, is_hot_list, o2c_map_list):
    """Python-only scan: for each compressed table, find needed frame IDs."""
    needed = {}
    for k, t_idx in enumerate(comp_tables):
        if isinstance(lS_i, list) or isinstance(lS_i, tuple):
            indices = lS_i[t_idx]
        elif lS_i.dim() == 2:
            indices = lS_i[t_idx]
        else:
            indices = lS_i

        cold_mask = ~is_hot_list[k][indices]
        if not cold_mask.any():
            continue
        cold_orig = indices[cold_mask]
        cold_reordered = o2c_map_list[k][cold_orig].long()
        frame_ids = (cold_reordered // ROWS_PER_FRAME).unique()
        needed[t_idx] = frame_ids
    return needed


def cpp_scan(lS_i, comp_tables, is_hot_list, o2c_map_list, _C):
    """C++ fused scan: single call for all tables."""
    lS_i_for_scan = []
    for t_idx in comp_tables:
        if isinstance(lS_i, list) or isinstance(lS_i, tuple):
            lS_i_for_scan.append(lS_i[t_idx].long())
        elif lS_i.dim() == 2:
            lS_i_for_scan.append(lS_i[t_idx].long())
        else:
            lS_i_for_scan.append(lS_i.long())

    frame_lists = _C.scan_needed_frames(
        lS_i_for_scan, is_hot_list, o2c_map_list, ROWS_PER_FRAME
    )

    needed = {}
    for k, t_idx in enumerate(comp_tables):
        if frame_lists[k].numel() > 0:
            needed[t_idx] = frame_lists[k]
    return needed


def main():
    log("Loading C++ extension...")
    try:
        import compressed_emb as _C
        log("  compressed_emb loaded OK")
    except ImportError:
        log("ERROR: compressed_emb not available. Build it first.")
        return

    log("Loading model state dict...")
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']

    # Build is_hot and o2c_map for each compressed table
    comp_tables = []
    is_hot_list = []
    o2c_map_list = []

    for t_idx in LARGE_TABLES:
        cold_idx_path = os.path.join(HOTCOLD_DIR, f"cold_indices_{t_idx}.pt")
        if not os.path.exists(cold_idx_path):
            continue
        cold_idx = torch.load(cold_idx_path, weights_only=False)
        if cold_idx.numel() == 0:
            continue

        key = f'emb_l.{t_idx}.weight'
        n_rows = sd[key].shape[0]

        # Apply reordering
        order_path = os.path.join(REORDER_DIR, f"cold_order_{t_idx}.npy")
        if os.path.exists(order_path):
            order = np.load(order_path)
            cold_idx = cold_idx[torch.from_numpy(order.astype(np.int64))]

        # Build is_hot mask
        is_hot = torch.ones(n_rows, dtype=torch.bool)
        is_hot[cold_idx] = False

        # Build orig_to_cold map
        orig_to_cold = torch.full((n_rows,), -1, dtype=torch.int32)
        orig_to_cold[cold_idx] = torch.arange(len(cold_idx), dtype=torch.int32)

        comp_tables.append(t_idx)
        is_hot_list.append(is_hot)
        o2c_map_list.append(orig_to_cold)

    log(f"  {len(comp_tables)} compressed tables: {comp_tables}")

    # Load test data using the same loader as benchmark_full_comparison
    log("Loading test data...")
    from benchmark_full_comparison import load_model_and_data
    _, test_ld, _, _ = load_model_and_data()

    # Warm up: run 10 batches through both paths
    log("Warming up...")
    n_warmup = 10
    for i, inputBatch in enumerate(test_ld):
        if i >= n_warmup:
            break
        lS_i = inputBatch[2]
        python_scan(lS_i, comp_tables, is_hot_list, o2c_map_list)
        cpp_scan(lS_i, comp_tables, is_hot_list, o2c_map_list, _C)

    # Benchmark: run all batches with both paths
    log("Benchmarking...")
    py_times = []
    cpp_times = []
    n_batches = 0
    mismatches = 0

    for inputBatch in test_ld:
        lS_i = inputBatch[2]

        # Python scan
        t0 = time.time()
        py_needed = python_scan(lS_i, comp_tables, is_hot_list, o2c_map_list)
        py_times.append(time.time() - t0)

        # C++ scan
        t0 = time.time()
        cpp_needed = cpp_scan(lS_i, comp_tables, is_hot_list, o2c_map_list, _C)
        cpp_times.append(time.time() - t0)

        # Verify correctness
        for t_idx in comp_tables:
            py_frames = sorted(py_needed.get(t_idx, torch.tensor([], dtype=torch.long)).tolist())
            cpp_frames = sorted(cpp_needed.get(t_idx, torch.tensor([], dtype=torch.long)).tolist())
            if py_frames != cpp_frames:
                mismatches += 1
                if mismatches <= 3:
                    log(f"  MISMATCH batch {n_batches} table {t_idx}: "
                        f"py={py_frames} cpp={cpp_frames}")

        n_batches += 1

    py_mean = np.mean(py_times) * 1000
    py_p50 = np.percentile(py_times, 50) * 1000
    py_p99 = np.percentile(py_times, 99) * 1000
    cpp_mean = np.mean(cpp_times) * 1000
    cpp_p50 = np.percentile(cpp_times, 50) * 1000
    cpp_p99 = np.percentile(cpp_times, 99) * 1000

    log(f"\n{'='*60}")
    log("SCAN BENCHMARK RESULTS")
    log(f"{'='*60}")
    log(f"  Batches: {n_batches}")
    log(f"  Mismatches: {mismatches}")
    log(f"")
    log(f"  Python scan:")
    log(f"    Mean: {py_mean:.3f} ms")
    log(f"    p50:  {py_p50:.3f} ms")
    log(f"    p99:  {py_p99:.3f} ms")
    log(f"")
    log(f"  C++ scan:")
    log(f"    Mean: {cpp_mean:.3f} ms")
    log(f"    p50:  {cpp_p50:.3f} ms")
    log(f"    p99:  {cpp_p99:.3f} ms")
    log(f"")
    log(f"  Speedup: {py_mean / cpp_mean:.2f}x (mean), {py_p50 / cpp_p50:.2f}x (p50)")

    # Save results
    results = {
        'n_batches': n_batches,
        'mismatches': mismatches,
        'python': {'mean_ms': py_mean, 'p50_ms': py_p50, 'p99_ms': py_p99},
        'cpp': {'mean_ms': cpp_mean, 'p50_ms': cpp_p50, 'p99_ms': cpp_p99},
        'speedup_mean': py_mean / cpp_mean,
        'speedup_p50': py_p50 / cpp_p50,
    }
    json_path = os.path.join(OUTPUT_DIR, 'scan_ab_test.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
