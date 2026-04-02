#!/usr/bin/env python3
"""
Benchmark: Find optimal thread configuration for H.265 decode.
Tests: threads_per_frame × frames_in_parallel combinations.
Also compares pool (batch_decode_fast) vs non-pool (batch_decode_file_paths).
"""
import os, sys, time, json, itertools
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import compressed_emb as _C

ONDEMAND_DIR = 'results/ondemand'

def find_frames(base_dir, ext='.h265'):
    frames = []
    for t_dir in sorted(os.listdir(base_dir)):
        t_path = os.path.join(base_dir, t_dir)
        if not os.path.isdir(t_path) or not t_dir.startswith('table_'):
            continue
        for f in sorted(os.listdir(t_path)):
            if f.endswith(ext):
                frames.append(os.path.join(t_path, f))
    return frames

def find_first_frame_per_table(base_dir, ext='.h265'):
    """Get just frame_00000 from each table — simulates 'one frame per table'."""
    frames = []
    for t_dir in sorted(os.listdir(base_dir)):
        t_path = os.path.join(base_dir, t_dir)
        if not os.path.isdir(t_path) or not t_dir.startswith('table_'):
            continue
        f0 = os.path.join(t_path, f'frame_00000{ext}')
        if os.path.exists(f0):
            frames.append(f0)
    return frames

def bench(fn, n_reps=7, warmup=2):
    """Run fn n_reps times, return median ms."""
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

    h265_dir = os.path.join(ONDEMAND_DIR, '1080p_crf30_nofilter')
    all_frames = find_frames(h265_dir)
    one_per_table = find_first_frame_per_table(h265_dir)

    # Also collect frames accessed during inference (first ~20)
    accessed_frames = all_frames[:20]  # typical inference working set

    print(f"All frames: {len(all_frames)}")
    print(f"One per table: {len(one_per_table)}")
    print(f"Accessed (first 20): {len(accessed_frames)}")

    sizes = {f: os.path.getsize(f) for f in all_frames[:30]}
    avg_sz = np.mean(list(sizes.values()))
    print(f"Avg frame size: {avg_sz/1024:.1f} KB")

    results = {}

    # ================================================================
    # Test 1: Single frame, vary threads_per_frame (1,2,4,8,16)
    # Using non-pool decode (decode_h265_frame_from_file) to test thread scaling
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 1: Single frame decode, varying threads per frame")
    print(f"{'='*70}")
    print(f"{'Threads':>8s} {'Median':>8s} {'Min':>8s}")

    test_frame = one_per_table[0]
    print(f"Frame: {test_frame} ({os.path.getsize(test_frame)/1024:.1f} KB)")

    for tpf in [1, 2, 4, 8, 16]:
        med, mn = bench(
            lambda t=tpf: _C.decode_h265_frame_from_file(test_frame, t, True, False, False))
        print(f"{tpf:>8d} {med:>7.2f}ms {mn:>7.2f}ms")
        results[f'single_frame_tpf{tpf}'] = {'median': med, 'min': mn}

    # ================================================================
    # Test 2: 8 frames (one per table), vary config
    # batch_decode_file_paths(paths, num_threads_per_decode, max_parallel)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"TEST 2: {len(one_per_table)} frames (one per table), varying config")
    print(f"{'='*70}")
    print(f"{'Config':>25s} {'Threads':>8s} {'Median':>8s} {'Min':>8s} {'Per-frm':>8s}")

    nf = len(one_per_table)
    configs_8f = [
        # (threads_per_frame, max_parallel, label)
        (1, 1, "1×1 (serial)"),
        (1, 2, "1×2"),
        (1, 4, "1×4"),
        (1, 8, f"1×{nf} (all par)"),
        (2, 4, "2×4"),
        (2, 8, f"2×{nf}"),
        (4, 2, "4×2"),
        (4, 4, "4×4"),
        (4, 8, f"4×{nf}"),
        (8, 1, "8×1"),
        (8, 2, "8×2"),
        (8, 4, "8×4"),
        (8, 8, f"8×{nf}"),
        (16, 1, "16×1"),
        (16, 2, "16×2"),
        (16, 4, "16×4"),
    ]
    for tpf, mp, label in configs_8f:
        total_threads = tpf * min(mp, nf)
        med, mn = bench(
            lambda t=tpf, m=mp: _C.batch_decode_file_paths(
                one_per_table, t, m, True, False, False))
        pf = med / nf
        print(f"{label:>25s} {total_threads:>6d}T {med:>7.1f}ms {mn:>7.1f}ms {pf:>7.2f}ms")
        results[f'8f_tpf{tpf}_par{mp}'] = {'median': med, 'min': mn, 'total_threads': total_threads}

    # ================================================================
    # Test 3: 8 frames (one per table), pool-accelerated (batch_decode_fast)
    # Pool uses 1 thread per frame, varies max_parallel
    # ================================================================
    print(f"\n{'='*70}")
    print(f"TEST 3: {len(one_per_table)} frames (one per table), POOL decode")
    print(f"{'='*70}")
    print(f"{'max_parallel':>12s} {'Median':>8s} {'Min':>8s} {'Per-frm':>8s}")

    for mp in [1, 2, 4, 8]:
        med, mn = bench(
            lambda m=mp: _C.batch_decode_fast(one_per_table, 2, m, True, False, False))
        pf = med / nf
        print(f"{mp:>12d} {med:>7.1f}ms {mn:>7.1f}ms {pf:>7.2f}ms")
        results[f'8f_pool_par{mp}'] = {'median': med, 'min': mn}

    # ================================================================
    # Test 4: 20 frames (inference working set), all configs
    # ================================================================
    print(f"\n{'='*70}")
    print(f"TEST 4: {len(accessed_frames)} frames (inference working set)")
    print(f"{'='*70}")
    print(f"{'Config':>30s} {'Threads':>8s} {'Median':>8s} {'Min':>8s} {'Per-frm':>8s}")

    nf20 = len(accessed_frames)
    configs_20f = [
        # Non-pool: (tpf, max_parallel, label)
        (1, 5, "npool 1×5"),
        (1, 10, "npool 1×10"),
        (1, 20, "npool 1×20"),
        (1, 40, "npool 1×40"),
        (2, 5, "npool 2×5"),
        (2, 10, "npool 2×10"),
        (4, 5, "npool 4×5"),
        (4, 10, "npool 4×10"),
        (8, 5, "npool 8×5"),
        (8, 10, "npool 8×10"),
    ]
    for tpf, mp, label in configs_20f:
        total_threads = tpf * min(mp, nf20)
        med, mn = bench(
            lambda t=tpf, m=mp: _C.batch_decode_file_paths(
                accessed_frames, t, m, True, False, False))
        pf = med / nf20
        print(f"{label:>30s} {total_threads:>6d}T {med:>7.1f}ms {mn:>7.1f}ms {pf:>7.2f}ms")
        results[f'20f_npool_tpf{tpf}_par{mp}'] = {'median': med, 'min': mn, 'total_threads': total_threads}

    # Pool decode for 20 frames
    print()
    for mp in [5, 10, 20, 40, 80]:
        med, mn = bench(
            lambda m=mp: _C.batch_decode_fast(accessed_frames, 2, m, True, False, False))
        pf = med / nf20
        print(f"{'pool 1×'+str(mp):>30s} {min(mp,nf20):>6d}T {med:>7.1f}ms {mn:>7.1f}ms {pf:>7.2f}ms")
        results[f'20f_pool_par{mp}'] = {'median': med, 'min': mn}

    # ================================================================
    # Test 5: Zstd comparison (same frame sets)
    # ================================================================
    zstd_dir = os.path.join(ONDEMAND_DIR, '1080p_zstd3')
    if os.path.exists(zstd_dir):
        zstd_8f = find_first_frame_per_table(zstd_dir, '.zst')
        zstd_20f = find_frames(zstd_dir, '.zst')[:20]

        print(f"\n{'='*70}")
        print(f"TEST 5: Zstd comparison")
        print(f"{'='*70}")

        if zstd_8f:
            for mp in [1, 4, 8]:
                med, mn = bench(
                    lambda m=mp: _C.batch_decode_fast(zstd_8f, 2, m, True, False, False))
                print(f"  Zstd 8f pool par={mp}: {med:.1f}ms (min {mn:.1f}ms)")

        if zstd_20f:
            for mp in [10, 20, 40]:
                med, mn = bench(
                    lambda m=mp: _C.batch_decode_fast(zstd_20f, 2, m, True, False, False))
                print(f"  Zstd 20f pool par={mp}: {med:.1f}ms (min {mn:.1f}ms)")

    # ================================================================
    # Summary: Find best config
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: Best configs")
    print(f"{'='*70}")

    # Find best for 8 frames
    best_8f = min(
        [(k, v) for k, v in results.items() if k.startswith('8f_')],
        key=lambda x: x[1]['median'])
    print(f"  Best 8 frames:  {best_8f[0]} → {best_8f[1]['median']:.1f}ms")

    # Find best for 20 frames
    best_20f = min(
        [(k, v) for k, v in results.items() if k.startswith('20f_')],
        key=lambda x: x[1]['median'])
    print(f"  Best 20 frames: {best_20f[0]} → {best_20f[1]['median']:.1f}ms")

    with open('results/thread_config_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/thread_config_benchmark.json")


if __name__ == '__main__':
    main()
