#!/usr/bin/env python3
"""Test decode pipeline: overlap H.265 decode with inference using C++ threads."""

import os, sys, time, gc
import numpy as np
import torch
import concurrent.futures
import threading

os.environ['CRITEO_DAYS'] = '0'
sys.path.insert(0, '.')

# Load C++ extension
LD_LIB = os.path.join(os.path.dirname(torch.__file__), 'lib')
if LD_LIB not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = LD_LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')
import compressed_emb as _C

from codec_ondemand_benchmark import load_model_and_data, HOTCOLD_DIR, REORDER_DIR

print("Loading model and data...")
dlrm, test_ld, train_ld, ln_emb = load_model_and_data()

# Load hot/cold masks and o2c maps
large_tables = [t for t in range(len(ln_emb)) if ln_emb[t] >= 50000]
is_hot = {}
o2c_map = {}
for t in large_tables:
    is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                           map_location='cpu', weights_only=True)
    o2c_map[t] = torch.load(os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt'),
                            map_location='cpu', weights_only=True)

EMB_DIM = 16
rows_per_frame = (1920 * 1080) // EMB_DIM  # 129600

# CRF=18 frame directories
FRAME_BASE = 'results/ondemand/1080p_crf18'

# Pre-load test batches
print("Loading test batches...")
test_batches = []
for j, (X, lS_o, lS_i, T) in enumerate(test_ld):
    if j >= 200:
        break
    test_batches.append((X, lS_o, lS_i, T))
print(f"Loaded {len(test_batches)} batches")

# Pre-compute frame paths for each batch
def get_batch_frame_paths(lS_i):
    """Get list of frame paths for a batch's cold accesses."""
    paths = []
    for t_idx in large_tables:
        indices = lS_i[t_idx]
        cold_mask = ~is_hot[t_idx][indices]
        if not cold_mask.any():
            continue
        cold_orig = indices[cold_mask]
        cold_mapped = o2c_map[t_idx][cold_orig]
        valid = cold_mapped >= 0
        if not valid.any():
            continue
        fids = (cold_mapped[valid] // rows_per_frame).unique().tolist()
        frame_dir = os.path.join(FRAME_BASE, f'table_{t_idx}')
        for fid in fids:
            fpath = os.path.join(frame_dir, f'frame_{fid:05d}.h265')
            if os.path.exists(fpath):
                paths.append(fpath)
    return paths

print("Pre-computing frame paths per batch...")
batch_frame_paths = []
for X, lS_o, lS_i, T in test_batches:
    paths = get_batch_frame_paths(lS_i)
    batch_frame_paths.append(paths)
avg_frames = np.mean([len(p) for p in batch_frame_paths])
print(f"Average {avg_frames:.1f} frames/batch")

# Warmup
print("\nWarming up...")
torch.set_num_threads(40)
with torch.no_grad():
    for i in range(5):
        X, lS_o, lS_i, T = test_batches[i]
        dlrm(X, lS_o, lS_i)
# Warmup decode (C++ path)
if batch_frame_paths[0]:
    _C.batch_decode_file_paths(batch_frame_paths[0], 1, 20)

NUM_BATCHES = 100  # measure over this many batches

# ========== Test 1: Inference only (baseline, no decode) ==========
print("\n" + "="*60)
print("TEST 1: Inference only (no overlapped decode)")
print("="*60)

for emb_threads, mlp_threads in [(40, 40), (10, 40)]:
    lats = []
    with torch.no_grad():
        for i in range(NUM_BATCHES):
            X, lS_o, lS_i, T = test_batches[i]
            torch.set_num_threads(emb_threads)
            t0 = time.time()
            ly = dlrm.apply_emb(lS_o, lS_i, dlrm.emb_l, dlrm.v_W_l)
            torch.set_num_threads(mlp_threads)
            x = dlrm.apply_mlp(X, dlrm.bot_l)
            z = dlrm.interact_features(x, ly)
            p = dlrm.apply_mlp(z, dlrm.top_l)
            lats.append(time.time() - t0)
    arr = np.array(lats) * 1000
    print(f"  emb={emb_threads}T mlp={mlp_threads}T: mean={arr.mean():.2f}ms  "
          f"p50={np.median(arr):.2f}ms  p99={np.percentile(arr,99):.2f}ms")

# ========== Test 2: Decode only (C++ batch, baseline decode speed) ==========
print("\n" + "="*60)
print("TEST 2: Decode only — C++ batch_decode_file_paths (no inference)")
print("="*60)

for max_par in [20, 10, 5]:
    for tpd in [1, 2]:
        lats = []
        for i in range(min(NUM_BATCHES, 30)):
            paths = batch_frame_paths[i]
            if not paths:
                continue
            t0 = time.time()
            _C.batch_decode_file_paths(paths, tpd, max_par)
            lats.append(time.time() - t0)
        arr = np.array(lats) * 1000
        print(f"  max_par={max_par} tpd={tpd} ({avg_frames:.0f} frames): "
              f"mean={arr.mean():.2f}ms  p50={np.median(arr):.2f}ms")

# ========== Test 3: Pipeline (overlapped decode + inference) ==========
print("\n" + "="*60)
print("TEST 3: Pipelined C++ decode + inference")
print("="*60)

# Use a single background thread to call the C++ batch decode (GIL released)
for max_par, tpd in [(20, 1), (10, 1), (5, 1), (20, 2), (10, 2)]:
    emb_threads = 10
    mlp_threads = 40

    batch_lats = []
    infer_lats = []

    with torch.no_grad():
        for i in range(NUM_BATCHES):
            batch_t0 = time.time()

            # Launch async C++ decode for next batch via a single Python thread
            decode_future = None
            if i + 1 < len(test_batches) and batch_frame_paths[i + 1]:
                next_paths = batch_frame_paths[i + 1]
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                decode_future = executor.submit(
                    _C.batch_decode_file_paths, next_paths, tpd, max_par)

            # Run inference on current batch
            X, lS_o, lS_i, T = test_batches[i]
            torch.set_num_threads(emb_threads)
            infer_t0 = time.time()
            ly = dlrm.apply_emb(lS_o, lS_i, dlrm.emb_l, dlrm.v_W_l)
            torch.set_num_threads(mlp_threads)
            x = dlrm.apply_mlp(X, dlrm.bot_l)
            z = dlrm.interact_features(x, ly)
            p = dlrm.apply_mlp(z, dlrm.top_l)
            infer_t1 = time.time()
            infer_lats.append(infer_t1 - infer_t0)

            # Wait for decode to finish
            if decode_future is not None:
                decode_future.result()
                executor.shutdown(wait=False)

            batch_lats.append(time.time() - batch_t0)

    b_arr = np.array(batch_lats) * 1000
    i_arr = np.array(infer_lats) * 1000
    print(f"  max_par={max_par} tpd={tpd} emb={emb_threads}T mlp={mlp_threads}T:")
    print(f"    Inference: mean={i_arr.mean():.2f}ms  p50={np.median(i_arr):.2f}ms")
    print(f"    Batch total: mean={b_arr.mean():.2f}ms  p50={np.median(b_arr):.2f}ms")
    print(f"    Decode hidden: {(1 - (b_arr.mean() - i_arr.mean()) / b_arr.mean()) * 100:.0f}%")

# ========== Test 4: Interference test ==========
print("\n" + "="*60)
print("TEST 4: Inference slowdown from concurrent C++ decode")
print("="*60)

# Inference only
torch.set_num_threads(10)
infer_only = []
with torch.no_grad():
    for i in range(NUM_BATCHES):
        X, lS_o, lS_i, T = test_batches[i]
        torch.set_num_threads(10)
        t0 = time.time()
        ly = dlrm.apply_emb(lS_o, lS_i, dlrm.emb_l, dlrm.v_W_l)
        torch.set_num_threads(40)
        x = dlrm.apply_mlp(X, dlrm.bot_l)
        z = dlrm.interact_features(x, ly)
        p = dlrm.apply_mlp(z, dlrm.top_l)
        infer_only.append(time.time() - t0)

# Inference + concurrent C++ decode (various configs)
for max_par, tpd in [(20, 1), (10, 1), (5, 1)]:
    infer_with_decode = []
    with torch.no_grad():
        for i in range(NUM_BATCHES):
            X, lS_o, lS_i, T = test_batches[i]
            paths = batch_frame_paths[i]

            # Start C++ decode in background
            decode_future = None
            if paths:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                decode_future = executor.submit(
                    _C.batch_decode_file_paths, paths, tpd, max_par)

            torch.set_num_threads(10)
            t0 = time.time()
            ly = dlrm.apply_emb(lS_o, lS_i, dlrm.emb_l, dlrm.v_W_l)
            torch.set_num_threads(40)
            x = dlrm.apply_mlp(X, dlrm.bot_l)
            z = dlrm.interact_features(x, ly)
            p = dlrm.apply_mlp(z, dlrm.top_l)
            infer_with_decode.append(time.time() - t0)

            if decode_future is not None:
                decode_future.result()
                executor.shutdown(wait=False)

    a = np.array(infer_only) * 1000
    b = np.array(infer_with_decode) * 1000
    slowdown = (b.mean() - a.mean()) / a.mean() * 100
    print(f"  Inference only:                    mean={a.mean():.2f}ms  p50={np.median(a):.2f}ms")
    print(f"  Inference + C++ decode (par={max_par} tpd={tpd}): "
          f"mean={b.mean():.2f}ms  p50={np.median(b):.2f}ms  slowdown={slowdown:+.1f}%")

torch.set_num_threads(40)
print("\nDone.")
