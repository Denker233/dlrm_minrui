#!/usr/bin/env python3
"""
AV1 decode throughput benchmark.
Encode embedding frames as AV1, decode with dav1d (AVX-512), compare with H.265.
"""
import os, sys, time, json, subprocess, tempfile, shutil
import numpy as np
import torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
import compressed_emb as _C
from codec_ondemand_benchmark import quantize_table, REORDER_DIR

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
EMB_DIM = 16
WIDTH, HEIGHT = 1920, 1080
RPF = (WIDTH * HEIGHT) // EMB_DIM

sd = torch.load('models/dlrm_kaggle_correct.pt', map_location='cpu', weights_only=False)['state_dict']
ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k], key=lambda x: int(x.split('.')[1]))

tmpdir = tempfile.mkdtemp(prefix='av1_bench_')
print(f"Temp: {tmpdir}")
torch.set_num_threads(1)

# ================================================================
# Step 1: Encode frames as both H.265 and AV1
# ================================================================
print("=" * 70)
print("STEP 1: Encoding frames (H.265 and AV1)")
print("=" * 70)

# Use first 20 frames from table 11 (our standard benchmark set)
cold_order = np.load(f'{REORDER_DIR}/cold_order_11.npy')
q, s, zp = quantize_table(sd[ek[11]][cold_order])
q_t = q[:RPF * 20]  # 20 frames worth
tiled_frames = _C.fused_quantize_tile_multiframe(q_t, WIDTH, HEIGHT)

h265_files = []
av1_files = []
av1_q30_files = []

for fi in range(min(20, len(tiled_frames))):
    raw = tiled_frames[fi].numpy().tobytes()

    # H.265 CRF=30 (our standard)
    h265_path = os.path.join(tmpdir, f'frame_{fi:03d}.h265')
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
           '-s', f'{WIDTH}x{HEIGHT}', '-r', '1', '-i', 'pipe:0',
           '-c:v', 'libx265', '-preset', 'medium', '-pix_fmt', 'gray',
           '-x265-params', 'keyint=1:min-keyint=1:crf=30:log-level=error:no-deblock=1:no-sao=1',
           '-f', 'matroska', h265_path]
    subprocess.run(cmd, input=raw, capture_output=True, timeout=60)
    h265_files.append(h265_path)

    # AV1 with libaom (intra-only via -g 1, quality similar to CRF=30)
    # cpu-used=8 for fastest encode (we only care about decode speed)
    av1_path = os.path.join(tmpdir, f'frame_{fi:03d}.av1.mkv')
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
           '-s', f'{WIDTH}x{HEIGHT}', '-r', '1', '-i', 'pipe:0',
           '-c:v', 'libaom-av1', '-pix_fmt', 'gray',
           '-cpu-used', '8', '-crf', '30', '-g', '1',
           '-f', 'matroska', av1_path]
    r = subprocess.run(cmd, input=raw, capture_output=True, timeout=300)
    if os.path.exists(av1_path) and os.path.getsize(av1_path) > 0:
        av1_files.append(av1_path)
    else:
        print(f"  Frame {fi} AV1 encode failed: {r.stderr.decode()[-200:]}")
        break

    # Skip higher quality for now — focus on decode speed

    if fi % 5 == 0:
        h265_sz = os.path.getsize(h265_path) / 1024
        av1_sz = os.path.getsize(av1_path) / 1024 if os.path.exists(av1_path) else 0
        print(f"  Frame {fi}: H.265={h265_sz:.1f}KB, AV1={av1_sz:.1f}KB")

n_frames = min(len(h265_files), len(av1_files))
print(f"\nEncoded {n_frames} frames in both formats")

if n_frames == 0:
    print("ERROR: No AV1 frames encoded. Check libaom-av1 availability.")
    shutil.rmtree(tmpdir)
    sys.exit(1)

# Compression comparison
h265_total = sum(os.path.getsize(f) for f in h265_files[:n_frames])
av1_total = sum(os.path.getsize(f) for f in av1_files[:n_frames])
raw_total = n_frames * WIDTH * HEIGHT
print(f"H.265: {h265_total/1024:.0f}KB ({raw_total/h265_total:.0f}x)")
print(f"AV1:   {av1_total/1024:.0f}KB ({raw_total/av1_total:.0f}x)")

# ================================================================
# Step 2: Decode throughput benchmark
# ================================================================
print(f"\n{'='*70}")
print("STEP 2: Decode throughput (H.265 vs AV1)")
print(f"{'='*70}")

decoded_mb = n_frames * WIDTH * HEIGHT / 1024 / 1024

def bench_ffmpeg_decode(files, decoder_name, n_reps=5):
    """Decode using FFmpeg's Python API (via our C++ extension or subprocess)."""
    # Warmup
    for f in files[:3]:
        cmd = ['ffmpeg', '-y', '-i', f, '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
        subprocess.run(cmd, capture_output=True, timeout=30)

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        for f in files:
            cmd = ['ffmpeg', '-y', '-i', f, '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
            r = subprocess.run(cmd, capture_output=True, timeout=30)
        times.append(time.perf_counter() - t0)

    med = np.median(times) * 1000
    mn = min(times) * 1000
    tp = decoded_mb / (med / 1000)
    return med, mn, tp

def bench_c_decode(files, n_reps=5):
    """Decode H.265 using our C++ batch_decode_fast (pool-accelerated)."""
    # Warmup
    _C.batch_decode_fast(files, 2, len(files), True, False, False)

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _C.batch_decode_fast(files, 2, len(files), True, False, False)
        times.append(time.perf_counter() - t0)

    med = np.median(times) * 1000
    mn = min(times) * 1000
    tp = decoded_mb / (med / 1000)
    return med, mn, tp

# H.265 via C++ pool (our optimized path)
print(f"\n--- H.265 via C++ pool (batch_decode_fast) ---")
med, mn, tp = bench_c_decode(h265_files[:n_frames])
print(f"  {n_frames} frames: {med:.1f}ms (min {mn:.1f}ms), {tp:.0f} MB/s = {tp/1024:.2f} GB/s")

# H.265 via FFmpeg subprocess
print(f"\n--- H.265 via FFmpeg subprocess ---")
med, mn, tp = bench_ffmpeg_decode(h265_files[:n_frames], "libx265")
print(f"  {n_frames} frames: {med:.0f}ms (min {mn:.0f}ms), {tp:.0f} MB/s = {tp/1024:.2f} GB/s")

# AV1 via FFmpeg subprocess (uses libdav1d with AVX-512)
print(f"\n--- AV1 via FFmpeg/dav1d subprocess ---")
med, mn, tp = bench_ffmpeg_decode(av1_files[:n_frames], "libdav1d")
print(f"  {n_frames} frames: {med:.0f}ms (min {mn:.0f}ms), {tp:.0f} MB/s = {tp/1024:.2f} GB/s")

# ================================================================
# Step 3: Thread scaling for AV1
# ================================================================
print(f"\n{'='*70}")
print("STEP 3: AV1 decode with thread control")
print(f"{'='*70}")

for threads in [1, 4, 8, 16]:
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for f in av1_files[:n_frames]:
            cmd = ['ffmpeg', '-y', '-threads', str(threads),
                   '-i', f, '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
            subprocess.run(cmd, capture_output=True, timeout=30)
        times.append(time.perf_counter() - t0)
    med = np.median(times) * 1000
    tp = decoded_mb / (med / 1000)
    print(f"  AV1 {threads}T: {med:.0f}ms, {tp:.0f} MB/s = {tp/1024:.2f} GB/s")

# Also test parallel decode (multiple frames at once)
print(f"\n--- Parallel AV1 decode (all {n_frames} frames simultaneously) ---")
import concurrent.futures

for max_workers in [4, 8, 16, 20]:
    def decode_one(f):
        cmd = ['ffmpeg', '-y', '-threads', '1', '-i', f,
               '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
        return subprocess.run(cmd, capture_output=True, timeout=30)

    # Warmup
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(decode_one, av1_files[:n_frames]))

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(decode_one, av1_files[:n_frames]))
        times.append(time.perf_counter() - t0)

    med = np.median(times) * 1000
    tp = decoded_mb / (med / 1000)
    print(f"  AV1 1T×{max_workers}par: {med:.0f}ms, {tp:.0f} MB/s = {tp/1024:.2f} GB/s")

# ================================================================
# Summary
# ================================================================
print(f"\n{'='*70}")
print("SUMMARY: H.265 vs AV1 decode throughput")
print(f"{'='*70}")
print(f"  Frames: {n_frames}, resolution: {WIDTH}×{HEIGHT}, decoded: {decoded_mb:.0f} MB")
print(f"  H.265 compressed: {h265_total/1024:.0f} KB ({raw_total/h265_total:.0f}x)")
print(f"  AV1 compressed:   {av1_total/1024:.0f} KB ({raw_total/av1_total:.0f}x)")
print(f"  CPU: Xeon Platinum 8380 (AVX-512)")
print(f"  dav1d version: 0.9.2 (has {2363} AVX-512 instructions)")

shutil.rmtree(tmpdir)
print("\nDone.")
