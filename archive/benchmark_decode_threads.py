#!/usr/bin/env python3
"""Benchmark decode with different thread counts to understand H.264 vs H.265 scaling."""

import os, sys, time, torch, tempfile, shutil
import compressed_emb as _C

def time_fn(fn, warmup=3, repeat=15):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)  # ms
    times.sort()
    return times[len(times) // 2]

# Create a realistic frame from real data
sd = torch.load("./models/dlrm_kaggle_correct.pt", map_location='cpu', weights_only=False)['state_dict']
weight = sd['emb_l.2.weight']
cold_idx = torch.load("results/hotcold/cold_indices_2.pt", map_location='cpu', weights_only=True)

W, H = 1920, 1080
frame_data = _C.fused_gather_quantize_tile(weight, cold_idx[:129600], W, H)
frame = frame_data[0]

tmpdir = tempfile.mkdtemp(prefix="thread_bench_")

# Encode with both codecs
for codec in ['h264', 'h265']:
    ext = '.h264' if codec == 'h264' else '.h265'
    _C.encode_frame_codec(frame, os.path.join(tmpdir, f"test{ext}"), codec, True, 0)

print(f"Frame: {W}x{H}, H.264: {os.path.getsize(os.path.join(tmpdir, 'test.h264')):,}B, "
      f"H.265: {os.path.getsize(os.path.join(tmpdir, 'test.h265')):,}B")
print()

# Test decode with different thread counts
# We need to modify the C++ code to accept thread_count parameter,
# but we can control it via torch threads and test the current behavior
print(f"Current torch threads: {torch.get_num_threads()}")
print()

# The decode function uses thread_count=0 (auto) with FF_THREAD_SLICE
# Let's measure at different OMP/torch thread settings
for codec in ['h264', 'h265']:
    ext = '.h264' if codec == 'h264' else '.h265'
    fpath = os.path.join(tmpdir, f"test{ext}")

    t = time_fn(lambda p=fpath: _C.decode_h265_frame_from_file(p))
    print(f"{codec.upper()} decode (thread_count=0 auto): {t:.2f} ms")

# Also test decode from bytes (no file I/O)
print()
for codec in ['h264', 'h265']:
    ext = '.h264' if codec == 'h264' else '.h265'
    fpath = os.path.join(tmpdir, f"test{ext}")
    with open(fpath, 'rb') as f:
        raw = f.read()
    compressed = torch.frombuffer(bytearray(raw), dtype=torch.uint8)

    t = time_fn(lambda c=compressed: _C.decode_h265_frame_from_bytes(c))
    print(f"{codec.upper()} decode from memory: {t:.2f} ms")

# Batch decode comparison
print()
for codec in ['h264', 'h265']:
    ext = '.h264' if codec == 'h264' else '.h265'
    fpath = os.path.join(tmpdir, f"test{ext}")
    # Create multiple copies for batch decode
    for i in range(10):
        src = fpath
        dst = os.path.join(tmpdir, f"frame_{i:05d}{ext}")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    for n_frames in [1, 2, 5, 10]:
        frame_ids = torch.arange(n_frames, dtype=torch.int64)
        t = time_fn(lambda fi=frame_ids: _C.batch_decode_frames(tmpdir, fi), warmup=2, repeat=10)
        print(f"{codec.upper()} batch decode {n_frames} frames: {t:.2f} ms ({t/n_frames:.2f} ms/frame)")
    print()

shutil.rmtree(tmpdir)
