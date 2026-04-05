#!/usr/bin/env python3
"""
Thread sweep for H.265 decode on the two target configs.
Tests 1-32 threads per frame to find the optimal point.

Machine: 2x Xeon 8380, 80 cores / 160 threads.
"""
import os, sys, time, json
import numpy as np

os.environ.setdefault('CRITEO_DAYS', '4')
import torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')

torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if torch_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import compressed_emb as _C

# Configs to test (use frames already encoded by v2 experiment)
CONFIGS = {
    'crf30_med_nodeblock': '/tmp/h265_decode_opts_v2/crf30_med_nodeblock',
    'crf18_uf_nofilter': None,  # will use the ondemand dir
}

# Also test the real encoded frames from the AUC test
ONDEMAND_CONFIGS = {
    'crf18_uf_nofilter_real': 'results/ondemand/1080p_crf18_uf_nofilter/table_2',
    'crf30_nofilter_real': 'results/ondemand/1080p_crf30_nofilter/table_2',
}

THREAD_COUNTS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
REPEAT = 7
NUM_FRAMES = 20


def benchmark_single(path, num_threads, skip_loop=True):
    """Benchmark single frame decode."""
    times = []
    for _ in range(REPEAT + 2):
        t0 = time.perf_counter()
        _C.decode_h265_frame_from_file(path, num_threads, skip_loop)
        times.append(time.perf_counter() - t0)
    return np.median(times[2:]) * 1000


def benchmark_batch(paths, tpd, max_par, skip_loop=True):
    """Benchmark batch decode."""
    times = []
    for _ in range(REPEAT + 2):
        t0 = time.perf_counter()
        _C.batch_decode_file_paths(paths, tpd, max_par, skip_loop)
        times.append(time.perf_counter() - t0)
    return np.median(times[2:]) * 1000


def find_frames(config_dir):
    """Find frame files in directory."""
    if not os.path.isdir(config_dir):
        return []
    frames = []
    for ext in ['.h265', '.mkv']:
        candidates = sorted([os.path.join(config_dir, f)
                            for f in os.listdir(config_dir)
                            if f.startswith('frame_') and f.endswith(ext)])
        if candidates:
            frames = candidates
            break
    return frames[:NUM_FRAMES]


def main():
    print("=" * 80)
    print("H.265 Decode Thread Sweep")
    print(f"Machine: 80 cores / 160 threads (2x Xeon 8380)")
    print(f"Thread counts: {THREAD_COUNTS}")
    print("=" * 80)

    all_results = {}

    # Test v2 experiment frames
    for name, config_dir in CONFIGS.items():
        if config_dir is None or not os.path.isdir(config_dir):
            print(f"\n  SKIP {name}: directory not found")
            continue

        frames = find_frames(config_dir)
        if not frames:
            print(f"\n  SKIP {name}: no frames found")
            continue

        print(f"\n=== {name} ({len(frames)} frames) ===")
        print(f"  Frame: {os.path.basename(frames[0])}, "
              f"{os.path.getsize(frames[0])} bytes")

        results = {}
        print(f"\n  {'Threads':>8s}  {'Single':>8s}  {'Speedup':>8s}  "
              f"{'Batch20':>8s}  {'Effective':>10s}")
        print(f"  {'-'*50}")

        base_time = None
        for nt in THREAD_COUNTS:
            single = benchmark_single(frames[0], nt, skip_loop=True)
            # Batch: use 1 thread per decode, max parallelism = 20
            # Also test: nt threads per decode, fewer parallel
            batch_1t = benchmark_batch(frames, 1, NUM_FRAMES, skip_loop=True)
            batch_nt = benchmark_batch(frames, nt, max(1, NUM_FRAMES // max(1, nt//2)),
                                       skip_loop=True)

            if base_time is None:
                base_time = single
            speedup = base_time / single

            results[nt] = {
                'single_ms': single,
                'speedup_vs_1t': speedup,
                'batch_1t_par20_ms': batch_1t,
                'batch_nt_ms': batch_nt,
            }
            print(f"  {nt:>8d}  {single:>7.2f}m  {speedup:>7.2f}x  "
                  f"{batch_1t:>7.2f}m  {single/nt:>8.2f}m/core")

        all_results[name] = results

    # Test real ondemand frames
    for name, config_dir in ONDEMAND_CONFIGS.items():
        frames = find_frames(config_dir)
        if not frames:
            print(f"\n  SKIP {name}: no frames in {config_dir}")
            continue

        print(f"\n=== {name} ({len(frames)} frames) ===")
        print(f"  Frame: {os.path.basename(frames[0])}, "
              f"{os.path.getsize(frames[0])} bytes")

        results = {}
        print(f"\n  {'Threads':>8s}  {'Single':>8s}  {'Speedup':>8s}  "
              f"{'Batch20':>8s}  {'Per-core':>10s}")
        print(f"  {'-'*50}")

        base_time = None
        for nt in THREAD_COUNTS:
            single = benchmark_single(frames[0], nt, skip_loop=True)
            batch = benchmark_batch(frames, 1, NUM_FRAMES, skip_loop=True)

            if base_time is None:
                base_time = single
            speedup = base_time / single

            results[nt] = {
                'single_ms': single,
                'speedup_vs_1t': speedup,
                'batch_1t_par20_ms': batch,
            }
            print(f"  {nt:>8d}  {single:>7.2f}m  {speedup:>7.2f}x  "
                  f"{batch:>7.2f}m  {single/nt:>8.2f}m/core")

        all_results[name] = results

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Optimal thread count per config")
    print(f"{'='*80}")
    for name, results in all_results.items():
        best_nt = min(results, key=lambda nt: results[nt]['single_ms'])
        best_ms = results[best_nt]['single_ms']
        t1_ms = results.get(1, {}).get('single_ms', 999)
        print(f"  {name}: best={best_nt}T @ {best_ms:.2f}ms "
              f"(1T={t1_ms:.2f}ms, {t1_ms/best_ms:.2f}x speedup)")

    # Save
    out_file = 'results/thread_sweep_results.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
