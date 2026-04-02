#!/usr/bin/env python3
"""
Benchmark: H.265 and Zstd decode throughput vs thread count.
Find the bottleneck: is it codec, mutex, memory bandwidth, or OS scheduling?
"""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import compressed_emb as _C

ONDEMAND_DIR = 'results/ondemand'
FRAME_DIR_H265 = os.path.join(ONDEMAND_DIR, '1080p_crf30_nofilter')
FRAME_DIR_ZSTD = os.path.join(ONDEMAND_DIR, '1080p_zstd3')

def find_frames(base_dir, ext):
    """Find all frame files across tables."""
    frames = []
    for t_dir in sorted(os.listdir(base_dir)):
        t_path = os.path.join(base_dir, t_dir)
        if not os.path.isdir(t_path) or not t_dir.startswith('table_'):
            continue
        for f in sorted(os.listdir(t_path)):
            if f.endswith(ext):
                frames.append(os.path.join(t_path, f))
    return frames

def benchmark_batch_decode(paths, n_frames, max_parallel, n_reps=5, codec='h265'):
    """Benchmark batch decode with specific parallelism."""
    subset = paths[:n_frames]
    # Warmup
    # Both H.265 and Zstd go through batch_decode_fast (auto-detects .zst)
    _C.batch_decode_fast(subset, 2, max_parallel, True, False, False)

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        results = _C.batch_decode_fast(subset, 2, max_parallel, True, False, False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    med = sorted(times)[len(times)//2]
    mn = min(times)
    decoded_mb = n_frames * 1920 * 1080 / 1024 / 1024
    return med, mn, decoded_mb

def main():
    torch.set_num_threads(1)  # Don't let torch interfere

    h265_frames = find_frames(FRAME_DIR_H265, '.h265')
    print(f"Found {len(h265_frames)} H.265 frames")

    # Check for Zstd frames
    zstd_frames = find_frames(FRAME_DIR_ZSTD, '.zst') if os.path.exists(FRAME_DIR_ZSTD) else []
    print(f"Found {len(zstd_frames)} Zstd frames")

    # Get file sizes
    h265_sizes = [os.path.getsize(f) for f in h265_frames[:20]]
    print(f"H.265 frame sizes (first 20): min={min(h265_sizes)/1024:.1f}KB, "
          f"max={max(h265_sizes)/1024:.1f}KB, avg={sum(h265_sizes)/len(h265_sizes)/1024:.1f}KB")

    n_frames = 20  # Standard benchmark: 20 frames
    n_reps = 7

    # ================================================================
    # Test 1: H.265 decode with varying max_parallel (thread count)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"H.265 DECODE: {n_frames} frames, varying thread count")
    print(f"{'='*70}")
    print(f"{'Threads':>8s} {'Median':>8s} {'Min':>8s} {'Per-frame':>10s} {'Throughput':>12s}")

    thread_counts = [1, 2, 4, 8, 10, 15, 20, 30, 40, 60, 80]
    h265_results = {}

    for tc in thread_counts:
        if tc > len(h265_frames):
            continue
        med, mn, decoded_mb = benchmark_batch_decode(
            h265_frames, n_frames, tc, n_reps, 'h265')
        per_frame = med / n_frames
        throughput = decoded_mb / med * 1000  # GB/s
        print(f"{tc:>8d} {med:>7.1f}ms {mn:>7.1f}ms {per_frame:>9.2f}ms {throughput:>10.1f} GB/s")
        h265_results[tc] = {'median_ms': med, 'min_ms': mn, 'per_frame_ms': per_frame, 'gbps': throughput}

    # ================================================================
    # Test 2: Zstd decode with varying thread count
    # ================================================================
    if zstd_frames:
        print(f"\n{'='*70}")
        print(f"ZSTD DECODE: {n_frames} frames, varying thread count")
        print(f"{'='*70}")
        print(f"{'Threads':>8s} {'Median':>8s} {'Min':>8s} {'Per-frame':>10s} {'Throughput':>12s}")

        zstd_results = {}
        for tc in thread_counts:
            if tc > len(zstd_frames):
                continue
            med, mn, decoded_mb = benchmark_batch_decode(
                zstd_frames, n_frames, tc, n_reps, 'zstd')
            per_frame = med / n_frames
            throughput = decoded_mb / med * 1000
            print(f"{tc:>8d} {med:>7.1f}ms {mn:>7.1f}ms {per_frame:>9.2f}ms {throughput:>10.1f} GB/s")
            zstd_results[tc] = {'median_ms': med, 'min_ms': mn, 'per_frame_ms': per_frame, 'gbps': throughput}

    # ================================================================
    # Test 3: Single-frame decode latency breakdown
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SINGLE FRAME DECODE LATENCY (serial, no parallelism)")
    print(f"{'='*70}")

    # Decode 1 frame at a time, measure each
    frame_times = []
    for i in range(min(20, len(h265_frames))):
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _C.batch_decode_fast([h265_frames[i]], 2, 1, True, False, False)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        med = sorted(times)[len(times)//2]
        frame_times.append(med)
        sz = os.path.getsize(h265_frames[i]) / 1024
        print(f"  Frame {i:2d}: {med:.2f}ms  ({sz:.1f}KB) {h265_frames[i].split('/')[-2]}/{h265_frames[i].split('/')[-1]}")

    print(f"\n  Avg single-frame: {sum(frame_times)/len(frame_times):.2f}ms")
    print(f"  Sum of 20 serial: {sum(frame_times):.1f}ms")

    # ================================================================
    # Test 4: Scaling analysis — is it mutex contention?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SCALING ANALYSIS")
    print(f"{'='*70}")

    serial_sum = sum(frame_times)
    print(f"  Serial sum (20 frames, 1 at a time): {serial_sum:.1f}ms")
    print(f"  Perfect scaling would give: {serial_sum/80:.2f}ms (80 cores)")
    print()

    for tc in [1, 5, 10, 20, 40, 80]:
        if tc in h265_results:
            r = h265_results[tc]
            speedup = serial_sum / r['median_ms']
            efficiency = speedup / tc * 100
            print(f"  {tc:>3d} threads: {r['median_ms']:>7.1f}ms  "
                  f"speedup={speedup:.1f}x  efficiency={efficiency:.0f}%")

    # ================================================================
    # Test 5: Vary number of frames (check if it's per-frame or fixed overhead)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"FRAME COUNT SCALING (40 threads)")
    print(f"{'='*70}")

    for nf in [1, 2, 5, 10, 20, 40, 60]:
        if nf > len(h265_frames):
            break
        med, mn, decoded_mb = benchmark_batch_decode(
            h265_frames, nf, 40, n_reps, 'h265')
        per_frame = med / nf
        throughput = decoded_mb / med * 1000
        print(f"  {nf:>3d} frames: {med:>7.1f}ms  per_frame={per_frame:.2f}ms  "
              f"throughput={throughput:.1f} GB/s")

    # Save results
    results = {
        'h265_thread_scaling': h265_results,
        'single_frame_times': frame_times,
        'serial_sum_20f': serial_sum,
    }
    with open('results/decode_thread_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/decode_thread_benchmark.json")


if __name__ == '__main__':
    main()
