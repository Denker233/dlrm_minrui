#!/usr/bin/env python3
"""A/B benchmark: Python scan vs C++ fused scan for needed frames."""

import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import compressed_emb as _C

# --- Config ---
REORDER_DIR = 'results/reorder'
HOTCOLD_DIR = 'results/hotcold'
TILE_W, TILE_H = 4, 4
width, height = 1920, 1080
rows_per_frame = (width // TILE_W) * (height // TILE_H)
EMB_DIM = 16
TEST_BATCH_SIZE = 2048

# --- Load model and data using existing infrastructure ---
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_full_comparison import load_model_and_data, HOTCOLD_DIR, REORDER_DIR

dlrm, test_ld, ln_emb, state_dict = load_model_and_data()
num_tabs = len(ln_emb)
large_tables = [i for i in range(num_tabs) if ln_emb[i] > 50000]

# --- Load hot/cold data ---
print("Loading hot/cold data...")
hot_indices, cold_indices, is_hot, o2c_map = {}, {}, {}, {}
cold_num_rows = {}

for t in large_tables:
    hi = torch.load(os.path.join(HOTCOLD_DIR, f'hot_indices_{t}.pt'), map_location='cpu', weights_only=True)
    ci = torch.load(os.path.join(HOTCOLD_DIR, f'cold_indices_{t}.pt'), map_location='cpu', weights_only=True)
    hot_indices[t] = hi
    cold_indices[t] = ci
    cold_num_rows[t] = len(ci)
    ih = torch.ones(ln_emb[t], dtype=torch.bool)
    ih[ci] = False
    is_hot[t] = ih

    mmap_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.npy')
    pt_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
    if os.path.exists(mmap_path):
        o2c_map[t] = torch.from_numpy(np.load(mmap_path).copy()).int()
    else:
        o2c_map[t] = torch.load(pt_path, map_location='cpu', weights_only=True).int()

compressed_tables = sorted(t for t in large_tables if cold_num_rows.get(t, 0) > 0)
print(f"Compressed tables: {compressed_tables}")

# Collect batches
print("Collecting batches...")
batches = []
for i, batch in enumerate(test_ld):
    if i >= 500:
        break
    batches.append(batch)
print(f"Collected {len(batches)} batches")

# --- Pre-build C++ scan inputs ---
is_hot_list = [is_hot[t] for t in compressed_tables]
o2c_map_list = [o2c_map[t].int() for t in compressed_tables]

# --- Warmup ---
print("\nWarming up...")
for batch in batches[:10]:
    lS_i = batch[2]
    # Python scan
    for t_idx in compressed_tables:
        indices = lS_i[t_idx] if isinstance(lS_i, (list, tuple)) or lS_i.dim() == 2 else lS_i
        cold_mask = ~is_hot[t_idx][indices]
        if cold_mask.any():
            cm = o2c_map[t_idx][indices[cold_mask]]
            valid = cm >= 0
            if valid.any():
                fids = (cm[valid] // rows_per_frame).unique().tolist()
    # C++ scan
    lS_i_for_scan = [lS_i[t] if isinstance(lS_i, (list, tuple)) or lS_i.dim() == 2
                     else lS_i for t in compressed_tables]
    _C.scan_needed_frames(lS_i_for_scan, is_hot_list, o2c_map_list, rows_per_frame)

# --- Benchmark Python scan ---
print("\n=== Python scan ===")
py_times = []
py_results = []
for batch in batches:
    lS_i = batch[2]
    t_scan = time.time()
    needed = {}
    for t_idx in compressed_tables:
        indices = lS_i[t_idx] if isinstance(lS_i, (list, tuple)) or lS_i.dim() == 2 else lS_i
        cold_mask = ~is_hot[t_idx][indices]
        if cold_mask.any():
            cm = o2c_map[t_idx][indices[cold_mask]]
            valid = cm >= 0
            if valid.any():
                fids = (cm[valid] // rows_per_frame).unique().tolist()
                needed[t_idx] = set(fids)
    scan_ms = (time.time() - t_scan) * 1000
    py_times.append(scan_ms)
    py_results.append(needed)

py_times = np.array(py_times)
print(f"  Mean:   {py_times.mean():.3f} ms")
print(f"  Median: {np.median(py_times):.3f} ms")
print(f"  p5:     {np.percentile(py_times, 5):.3f} ms")
print(f"  p95:    {np.percentile(py_times, 95):.3f} ms")

# --- Benchmark C++ scan ---
print("\n=== C++ fused scan ===")
cpp_times = []
cpp_results = []
for batch in batches:
    lS_i = batch[2]
    t_scan = time.time()
    lS_i_for_scan = [lS_i[t] if isinstance(lS_i, (list, tuple)) or lS_i.dim() == 2
                     else lS_i for t in compressed_tables]
    frame_lists = _C.scan_needed_frames(lS_i_for_scan, is_hot_list, o2c_map_list, rows_per_frame)
    needed = {}
    for k, t_idx in enumerate(compressed_tables):
        if frame_lists[k].numel() > 0:
            needed[t_idx] = set(frame_lists[k].tolist())
    scan_ms = (time.time() - t_scan) * 1000
    cpp_times.append(scan_ms)
    cpp_results.append(needed)

cpp_times = np.array(cpp_times)
print(f"  Mean:   {cpp_times.mean():.3f} ms")
print(f"  Median: {np.median(cpp_times):.3f} ms")
print(f"  p5:     {np.percentile(cpp_times, 5):.3f} ms")
print(f"  p95:    {np.percentile(cpp_times, 95):.3f} ms")

# --- Verify correctness ---
print("\n=== Correctness check ===")
mismatches = 0
for i in range(len(batches)):
    if py_results[i] != cpp_results[i]:
        mismatches += 1
        if mismatches <= 3:
            print(f"  Mismatch at batch {i}:")
            print(f"    Python: {py_results[i]}")
            print(f"    C++:    {cpp_results[i]}")
if mismatches == 0:
    print("  All batches match! Correctness verified.")
else:
    print(f"  {mismatches}/{len(batches)} mismatches!")

# --- Summary ---
speedup = py_times.mean() / cpp_times.mean()
print(f"\n=== Summary ===")
print(f"  Python scan: {py_times.mean():.3f} ms/batch (median {np.median(py_times):.3f})")
print(f"  C++ scan:    {cpp_times.mean():.3f} ms/batch (median {np.median(cpp_times):.3f})")
print(f"  Speedup:     {speedup:.1f}x")
print(f"  Saving:      {py_times.mean() - cpp_times.mean():.3f} ms/batch")
