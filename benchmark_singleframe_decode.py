#!/usr/bin/env python3
"""
Benchmark decode speed for actual single-frame-per-table H.265 files.
These are LARGE frames (e.g., 3840×40400 for table 2 = 155MB decoded).
Test: threads per frame, parallel frames, combinations.
"""
import os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
import compressed_emb as _C

SF_DIR = 'results/ondemand/single_frame/h265'
TABLES = [2, 3, 9, 11, 15, 20, 23, 25]

SF_DIMS = {
    2: (3840, 40400), 3: (1920, 17568), 9: (1920, 744), 11: (3840, 33304),
    15: (1920, 43556), 20: (1920, 56200), 23: (1920, 2284), 25: (1920, 1140),
}

def bench(fn, n_reps=5, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times), min(times)

def main():
    torch.set_num_threads(1)
    ncpu = os.cpu_count()
    print(f"CPU cores: {ncpu}")

    # Collect frame paths and sizes
    paths = []
    for t in TABLES:
        f = os.path.join(SF_DIR, f'table_{t}', 'frame_00000.h265')
        if os.path.exists(f):
            w, h = SF_DIMS[t]
            decoded_mb = w * h / 1024 / 1024
            comp_kb = os.path.getsize(f) / 1024
            paths.append(f)
            print(f"  table {t}: {w}×{h} = {decoded_mb:.1f}MB decoded, {comp_kb:.1f}KB compressed")

    total_decoded_mb = sum(SF_DIMS[t][0] * SF_DIMS[t][1] / 1024 / 1024 for t in TABLES)
    print(f"  Total decoded: {total_decoded_mb:.0f}MB across {len(paths)} frames")

    # ================================================================
    # Test 1: Single large frame, varying threads per frame
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 1: Single frame decode (table 2: 3840×40400 = 148MB)")
    print(f"{'='*70}")
    print(f"{'Threads':>8} {'Median':>10} {'Min':>10} {'Throughput':>12}")

    big_frame = os.path.join(SF_DIR, 'table_2', 'frame_00000.h265')
    big_decoded_mb = SF_DIMS[2][0] * SF_DIMS[2][1] / 1024 / 1024

    for tpf in [1, 2, 4, 8, 16, 32]:
        med, mn = bench(
            lambda t=tpf: _C.decode_h265_frame_from_file(big_frame, t, True, False, False),
            n_reps=3, warmup=1)
        tp = big_decoded_mb / (med / 1000)  # MB/s
        print(f"{tpf:>8} {med:>9.1f}ms {mn:>9.1f}ms {tp:>10.0f} MB/s")

    # ================================================================
    # Test 2: All 8 frames, varying parallel count (1 thread per frame)
    # Using batch_decode_file_paths (non-pool, supports thread count per frame)
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 2: All 8 single-frame files, varying parallelism")
    print(f"{'='*70}")

    # 2a: batch_decode_file_paths with varying threads_per_frame and max_parallel
    print(f"\n--- batch_decode_file_paths (non-pool, supports multi-thread per frame) ---")
    print(f"{'Config':>25} {'Total T':>8} {'Median':>10} {'Min':>10} {'Throughput':>12}")

    configs = [
        # (threads_per_frame, max_parallel, label)
        (1, 1, "1T × 1 par (serial)"),
        (1, 2, "1T × 2 par"),
        (1, 4, "1T × 4 par"),
        (1, 8, "1T × 8 par"),
        (2, 4, "2T × 4 par"),
        (2, 8, "2T × 8 par"),
        (4, 2, "4T × 2 par"),
        (4, 4, "4T × 4 par"),
        (4, 8, "4T × 8 par"),
        (8, 1, "8T × 1 par"),
        (8, 2, "8T × 2 par"),
        (8, 4, "8T × 4 par"),
        (8, 8, "8T × 8 par"),
        (16, 2, "16T × 2 par"),
        (16, 4, "16T × 4 par"),
        (16, 8, "16T × 8 par"),
        (32, 2, "32T × 2 par"),
        (32, 4, "32T × 4 par"),
    ]

    best_config = None
    best_time = float('inf')

    for tpf, mp, label in configs:
        total_t = tpf * min(mp, 8)
        med, mn = bench(
            lambda t=tpf, m=mp: _C.batch_decode_file_paths(paths, t, m, True, False, False),
            n_reps=3, warmup=1)
        tp = total_decoded_mb / (med / 1000)
        print(f"{label:>25} {total_t:>7}T {med:>9.1f}ms {mn:>9.1f}ms {tp:>10.0f} MB/s")
        if med < best_time:
            best_time = med
            best_config = label

    # 2b: batch_decode_fast (pool, 1 thread per frame)
    print(f"\n--- batch_decode_fast (pool, 1 thread per frame) ---")
    for mp in [1, 2, 4, 8]:
        med, mn = bench(
            lambda m=mp: _C.batch_decode_fast(paths, 2, m, True, False, False),
            n_reps=3, warmup=1)
        tp = total_decoded_mb / (med / 1000)
        label = f"pool × {mp} par"
        print(f"{label:>25} {min(mp,8):>7}T {med:>9.1f}ms {mn:>9.1f}ms {tp:>10.0f} MB/s")
        if med < best_time:
            best_time = med
            best_config = label

    # ================================================================
    # Test 3: Individual frame decode times
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 3: Individual frame decode (1 thread, serial)")
    print(f"{'='*70}")
    total_serial = 0
    for t in TABLES:
        f = os.path.join(SF_DIR, f'table_{t}', 'frame_00000.h265')
        w, h = SF_DIMS[t]
        decoded_mb = w * h / 1024 / 1024
        med, mn = bench(
            lambda path=f: _C.decode_h265_frame_from_file(path, 1, True, False, False),
            n_reps=3, warmup=1)
        tp = decoded_mb / (med / 1000)
        print(f"  table {t:>2} ({w}×{h:>5}, {decoded_mb:>6.1f}MB): {med:>7.1f}ms  ({tp:.0f} MB/s)")
        total_serial += med

    print(f"\n  Serial sum: {total_serial:.0f}ms")
    print(f"  Perfect 8-way parallel: {total_serial/8:.0f}ms")
    print(f"  Best achieved: {best_time:.0f}ms ({best_config})")
    print(f"  Parallel efficiency: {total_serial/8/best_time*100:.0f}%")

    # ================================================================
    # Test 4: Compare with 1080p multi-frame decode
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 4: Compare single-frame vs 1080p decode")
    print(f"{'='*70}")

    # Find 1080p frames (first 20)
    dir_1080p = 'results/ondemand/1080p_crf30_nofilter'
    frames_1080p = []
    for td in sorted(os.listdir(dir_1080p)):
        tp = os.path.join(dir_1080p, td)
        if not os.path.isdir(tp) or not td.startswith('table_'):
            continue
        for ff in sorted(os.listdir(tp)):
            if ff.endswith('.h265'):
                frames_1080p.append(os.path.join(tp, ff))
    frames_1080p_20 = frames_1080p[:20]

    # 1080p pool decode
    med_1080p, mn_1080p = bench(
        lambda: _C.batch_decode_fast(frames_1080p_20, 2, 20, True, False, False),
        n_reps=5, warmup=2)
    decoded_1080p_mb = 20 * 1920 * 1080 / 1024 / 1024
    tp_1080p = decoded_1080p_mb / (med_1080p / 1000)

    print(f"  1080p (20 frames, pool×20): {med_1080p:.1f}ms, {decoded_1080p_mb:.0f}MB, {tp_1080p:.0f} MB/s")
    print(f"  Single-frame (8 frames, best): {best_time:.1f}ms, {total_decoded_mb:.0f}MB, "
          f"{total_decoded_mb/(best_time/1000):.0f} MB/s")
    print(f"\n  1080p is {best_time/med_1080p:.1f}x {'slower' if best_time > med_1080p else 'faster'} "
          f"for {total_decoded_mb/decoded_1080p_mb:.1f}x more data")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    sf_gbps = total_decoded_mb / 1024 / (best_time / 1000)
    mp_gbps = decoded_1080p_mb / 1024 / (med_1080p / 1000)
    print(f"  Single-frame best:  {best_time:.0f}ms  ({sf_gbps:.2f} GB/s)  [{best_config}]")
    print(f"  1080p pool×20:      {med_1080p:.0f}ms  ({mp_gbps:.2f} GB/s)")
    print(f"  Serial sum (8 frames): {total_serial:.0f}ms")

    # For loading time recalculation
    print(f"\n  Corrected loading projections (100GB model, SSD 5 GB/s):")
    uint8_100gb = 25  # GB
    t_ssd_uncomp = 100 / 5
    for label, gbps in [("Single-frame best", sf_gbps), ("1080p pool", mp_gbps)]:
        t_decode = uint8_100gb / gbps
        t_total = t_decode  # SSD read is negligible for compressed
        print(f"    {label}: decode {t_decode:.1f}s (read raw: {t_ssd_uncomp:.1f}s, speedup: {t_ssd_uncomp/t_total:.1f}x)")


if __name__ == '__main__':
    main()
