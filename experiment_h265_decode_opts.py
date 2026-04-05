#!/usr/bin/env python3
"""
Experiment: H.265 decode optimization options.

Tests 4 optimization axes (option 4 excluded per user request):
  1. Higher CRF (18 vs 30 vs 40)
  2. Encoder preset (ultrafast vs superfast vs medium)
  3. Skip loop filter on decode (AVDISCARD_ALL)
  5. H.265 tiles/WPP for intra-frame parallel decode

Uses real cold embedding data from table 2 (largest, 9.7M cold rows).
"""

import os, sys, time, json, subprocess, shutil, struct
import numpy as np

os.environ.setdefault('CRITEO_DAYS', '4')

import torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')

# Load C++ extension
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if torch_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import compressed_emb as _C

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_FRAME_DIR = "results/ondemand/1080p_crf18/table_2"
OUTPUT_BASE = "/tmp/h265_decode_opts"
WIDTH, HEIGHT = 1920, 1080
NUM_TEST_FRAMES = 20   # decode 20 frames in batch (matches real workload)
BATCH_TPD = 4           # threads per decode
BATCH_PAR = 20          # max parallel decodes
REPEAT = 5              # repeat measurements

# Encoding configs to test
ENCODE_CONFIGS = [
    # (name, crf, preset, extra_x265_params)
    ("crf18_medium",     18, "medium",    ""),
    ("crf18_ultrafast",  18, "ultrafast", ""),
    ("crf18_superfast",  18, "superfast", ""),
    ("crf30_medium",     30, "medium",    ""),
    ("crf30_ultrafast",  30, "ultrafast", ""),
    ("crf40_ultrafast",  40, "ultrafast", ""),
    # Tiles: 2 columns (each 960 wide) for intra-frame parallel decode
    ("crf18_uf_tiles2",  18, "ultrafast", "frame-threads=1:tiles=2"),
    ("crf18_uf_tiles4",  18, "ultrafast", "frame-threads=1:tiles=4"),
    # WPP (wavefront parallel processing) - default for x265 but ensure it's on
    ("crf18_uf_wpp",     18, "ultrafast", "wpp=1"),
    # Combined: higher CRF + tiles
    ("crf30_uf_tiles2",  30, "ultrafast", "frame-threads=1:tiles=2"),
    ("crf30_uf_tiles4",  30, "ultrafast", "frame-threads=1:tiles=4"),
]


def get_raw_frame_data():
    """Load one real CRF=18 frame, decode it to raw pixels for re-encoding."""
    frame_path = os.path.join(BASE_FRAME_DIR, "frame_00000.h265")
    # Decode to get raw pixels
    raw = _C.decode_h265_frame_from_file(frame_path, 4, False)
    # raw is [H, W] uint8
    return raw.numpy()


def encode_frame_ffmpeg(raw_np, output_path, crf, preset, extra_x265):
    """Encode a single raw grayscale frame using ffmpeg subprocess."""
    h, w = raw_np.shape
    x265_params = f"keyint=1:min-keyint=1:crf={crf}:log-level=error"
    if extra_x265:
        x265_params += ":" + extra_x265

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'gray', '-s', f'{w}x{h}',
        '-r', '1', '-i', 'pipe:0',
        '-c:v', 'libx265', '-preset', preset,
        '-pix_fmt', 'gray',
        '-x265-params', x265_params,
        '-f', 'hevc',  # raw HEVC bitstream (no container overhead)
        output_path,
    ]
    proc = subprocess.run(cmd, input=raw_np.tobytes(), capture_output=True, timeout=30)
    if proc.returncode != 0:
        # Try matroska container as fallback
        cmd[-2] = 'matroska'
        cmd[-1] = output_path
        proc = subprocess.run(cmd, input=raw_np.tobytes(), capture_output=True, timeout=30)
        if proc.returncode != 0:
            print(f"  ENCODE FAILED: {proc.stderr.decode()[-200:]}")
            return False
    return True


def encode_all_configs(raw_np):
    """Encode the raw frame with all configs, creating 20 copies of each."""
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    results = {}

    for name, crf, preset, extra in ENCODE_CONFIGS:
        config_dir = os.path.join(OUTPUT_BASE, name)
        os.makedirs(config_dir, exist_ok=True)

        # Encode one frame
        single_path = os.path.join(config_dir, "frame_00000.h265")
        t0 = time.perf_counter()
        ok = encode_frame_ffmpeg(raw_np, single_path, crf, preset, extra)
        enc_time = time.perf_counter() - t0

        if not ok or not os.path.exists(single_path):
            print(f"  SKIP {name}: encode failed")
            continue

        fsize = os.path.getsize(single_path)
        print(f"  {name}: {fsize} bytes, encode {enc_time*1000:.1f}ms")

        # Copy to create 20 frames (simulates real workload)
        for i in range(1, NUM_TEST_FRAMES):
            dst = os.path.join(config_dir, f"frame_{i:05d}.h265")
            shutil.copy2(single_path, dst)

        results[name] = {
            'crf': crf, 'preset': preset, 'extra': extra,
            'frame_bytes': fsize, 'encode_ms': enc_time * 1000,
            'dir': config_dir,
        }

    return results


def benchmark_single_frame(config_dir, skip_loop_filter=False):
    """Benchmark single frame decode with 1T and 4T."""
    path = os.path.join(config_dir, "frame_00000.h265")
    if not os.path.exists(path):
        return None, None

    times_1t = []
    times_4t = []
    for _ in range(REPEAT + 1):
        t0 = time.perf_counter()
        _C.decode_h265_frame_from_file(path, 1, skip_loop_filter)
        t1 = time.perf_counter()
        _C.decode_h265_frame_from_file(path, 4, skip_loop_filter)
        t2 = time.perf_counter()
        times_1t.append(t1 - t0)
        times_4t.append(t2 - t1)

    # Drop first (warmup)
    return np.median(times_1t[1:]) * 1000, np.median(times_4t[1:]) * 1000


def benchmark_batch_decode(config_dir, skip_loop_filter=False):
    """Benchmark batch decode of 20 frames with optimal threading."""
    paths = [os.path.join(config_dir, f"frame_{i:05d}.h265")
             for i in range(NUM_TEST_FRAMES)]

    # Verify all files exist
    for p in paths:
        if not os.path.exists(p):
            return None

    times = []
    for _ in range(REPEAT + 1):
        t0 = time.perf_counter()
        frames = _C.batch_decode_file_paths(paths, BATCH_TPD, BATCH_PAR, skip_loop_filter)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Drop first (warmup)
    return np.median(times[1:]) * 1000


def benchmark_real_crf18():
    """Benchmark the real CRF=18 frames (not re-encoded copies)."""
    paths = [os.path.join(BASE_FRAME_DIR, f"frame_{i:05d}.h265")
             for i in range(NUM_TEST_FRAMES)]

    existing = [p for p in paths if os.path.exists(p)]
    if len(existing) < NUM_TEST_FRAMES:
        print(f"  Only {len(existing)}/{NUM_TEST_FRAMES} real CRF=18 frames exist")
        paths = existing

    # Single frame
    path0 = paths[0]
    times_1t, times_4t = [], []
    for _ in range(REPEAT + 1):
        t0 = time.perf_counter()
        _C.decode_h265_frame_from_file(path0, 1, False)
        t1 = time.perf_counter()
        _C.decode_h265_frame_from_file(path0, 4, False)
        t2 = time.perf_counter()
        times_1t.append(t1 - t0)
        times_4t.append(t2 - t1)

    single_1t = np.median(times_1t[1:]) * 1000
    single_4t = np.median(times_4t[1:]) * 1000

    # Batch
    batch_times_normal = []
    batch_times_skiploop = []
    for _ in range(REPEAT + 1):
        t0 = time.perf_counter()
        _C.batch_decode_file_paths(paths, BATCH_TPD, BATCH_PAR, False)
        t1 = time.perf_counter()
        _C.batch_decode_file_paths(paths, BATCH_TPD, BATCH_PAR, True)
        t2 = time.perf_counter()
        batch_times_normal.append(t1 - t0)
        batch_times_skiploop.append(t2 - t1)

    batch_normal = np.median(batch_times_normal[1:]) * 1000
    batch_skiploop = np.median(batch_times_skiploop[1:]) * 1000

    return single_1t, single_4t, batch_normal, batch_skiploop


def benchmark_zstd_baseline():
    """Benchmark Zstd decode for comparison."""
    zstd_dir = "results/ondemand/1080p_zstd3/table_2"
    if not os.path.isdir(zstd_dir):
        return None

    paths = []
    for i in range(NUM_TEST_FRAMES):
        p = os.path.join(zstd_dir, f"frame_{i:05d}.zst")
        if os.path.exists(p):
            paths.append(p)
    if len(paths) < NUM_TEST_FRAMES:
        print(f"  Only {len(paths)}/{NUM_TEST_FRAMES} Zstd frames exist")
        if not paths:
            return None

    # Batch decode
    times = []
    for _ in range(REPEAT + 1):
        t0 = time.perf_counter()
        _C.batch_decode_file_paths(paths, 1, 20, False)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times[1:]) * 1000


def measure_quality(config_dir, raw_ref):
    """Measure decode quality vs raw reference."""
    path = os.path.join(config_dir, "frame_00000.h265")
    if not os.path.exists(path):
        return None, None

    decoded = _C.decode_h265_frame_from_file(path, 4, False).numpy()
    diff = decoded.astype(np.float32) - raw_ref.astype(np.float32)
    mse = np.mean(diff ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    return mse, psnr


def main():
    print("=" * 70)
    print("H.265 Decode Optimization Benchmark")
    print("=" * 70)
    print(f"Frames: {NUM_TEST_FRAMES}, Batch: tpd={BATCH_TPD} par={BATCH_PAR}")
    print()

    # Step 1: Get raw frame data
    print("[1] Loading raw frame from real CRF=18 data...")
    raw_np = get_raw_frame_data()
    print(f"    Frame shape: {raw_np.shape}, dtype: {raw_np.dtype}")
    raw_bytes = raw_np.nbytes
    print(f"    Raw size: {raw_bytes:,} bytes ({raw_bytes/1024:.1f} KB)")
    print()

    # Step 2: Encode all configs
    print("[2] Encoding frames with different configs...")
    configs = encode_all_configs(raw_np)
    print()

    # Step 3: Benchmark real CRF=18 frames
    print("[3] Benchmarking REAL CRF=18 frames (original, not re-encoded)...")
    r = benchmark_real_crf18()
    if r:
        s1t, s4t, batch_n, batch_sl = r
        print(f"    Single frame: 1T={s1t:.2f}ms, 4T={s4t:.2f}ms")
        print(f"    Batch {NUM_TEST_FRAMES}f: normal={batch_n:.1f}ms, skip_loop={batch_sl:.1f}ms")
        real_crf18_batch = batch_n
        real_crf18_skip = batch_sl
    else:
        real_crf18_batch = real_crf18_skip = None
    print()

    # Step 4: Benchmark each config (normal + skip_loop_filter)
    print("[4] Benchmarking each encode config...")
    print()
    all_results = {}

    for name, info in configs.items():
        print(f"  --- {name} ---")
        config_dir = info['dir']

        # Single frame decode
        s1t, s4t = benchmark_single_frame(config_dir, skip_loop_filter=False)
        s1t_sl, s4t_sl = benchmark_single_frame(config_dir, skip_loop_filter=True)

        # Batch decode
        batch_normal = benchmark_batch_decode(config_dir, skip_loop_filter=False)
        batch_skiploop = benchmark_batch_decode(config_dir, skip_loop_filter=True)

        # Quality
        mse, psnr = measure_quality(config_dir, raw_np)

        print(f"    Single: 1T={s1t:.2f}ms, 4T={s4t:.2f}ms | skip_loop: 1T={s1t_sl:.2f}ms, 4T={s4t_sl:.2f}ms")
        print(f"    Batch {NUM_TEST_FRAMES}f: normal={batch_normal:.1f}ms, skip_loop={batch_skiploop:.1f}ms")
        if mse is not None:
            print(f"    Quality: MSE={mse:.2f}, PSNR={psnr:.1f}dB")
        print(f"    Compression: {raw_bytes/info['frame_bytes']:.1f}x ({info['frame_bytes']} bytes)")
        print()

        all_results[name] = {
            **info,
            'single_1t_ms': s1t,
            'single_4t_ms': s4t,
            'single_1t_skiploop_ms': s1t_sl,
            'single_4t_skiploop_ms': s4t_sl,
            'batch_normal_ms': batch_normal,
            'batch_skiploop_ms': batch_skiploop,
            'mse': mse,
            'psnr': psnr,
            'compression_ratio': raw_bytes / info['frame_bytes'],
        }

    # Step 5: Zstd baseline
    print("[5] Zstd baseline...")
    zstd_batch = benchmark_zstd_baseline()
    if zstd_batch:
        print(f"    Zstd batch {NUM_TEST_FRAMES}f: {zstd_batch:.1f}ms")
    else:
        print("    Zstd frames not found, skipping")
    print()

    # Step 6: Summary table
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Config':<25s} {'Bytes':>7s} {'Ratio':>6s} {'1T':>7s} {'4T':>7s} "
          f"{'Batch':>7s} {'Skip':>7s} {'PSNR':>7s}")
    print("-" * 70)

    if real_crf18_batch:
        print(f"{'REAL_crf18 (original)':<25s} {'N/A':>7s} {'N/A':>6s} "
              f"{s1t:>6.1f}m {s4t:>6.1f}m "
              f"{real_crf18_batch:>6.1f}m {real_crf18_skip:>6.1f}m {'lossyR':>7s}")

    for name in sorted(all_results.keys()):
        r = all_results[name]
        psnr_str = f"{r['psnr']:.1f}" if r['psnr'] != float('inf') else "inf"
        print(f"{name:<25s} {r['frame_bytes']:>7d} {r['compression_ratio']:>5.1f}x "
              f"{r['single_1t_ms']:>6.1f}m {r['single_4t_ms']:>6.1f}m "
              f"{r['batch_normal_ms']:>6.1f}m {r['batch_skiploop_ms']:>6.1f}m "
              f"{psnr_str:>7s}")

    if zstd_batch:
        print(f"{'Zstd-3 (lossless)':<25s} {'~105K':>7s} {'~19x':>6s} "
              f"{'0.2':>6s}m {'N/A':>6s}m "
              f"{zstd_batch:>6.1f}m {'N/A':>7s} {'inf':>7s}")

    print("-" * 70)
    print()

    # Step 7: Analysis
    print("KEY FINDINGS:")
    if all_results:
        # Best batch decode
        best_batch = min(all_results.items(), key=lambda x: x[1]['batch_normal_ms'])
        best_skip = min(all_results.items(), key=lambda x: x[1]['batch_skiploop_ms'])
        print(f"  Best batch decode (normal):     {best_batch[0]} = {best_batch[1]['batch_normal_ms']:.1f}ms")
        print(f"  Best batch decode (skip_loop):  {best_skip[0]} = {best_skip[1]['batch_skiploop_ms']:.1f}ms")

        if real_crf18_batch:
            improvement = (real_crf18_batch - best_skip[1]['batch_skiploop_ms']) / real_crf18_batch * 100
            print(f"  Improvement vs real CRF=18:     {improvement:.1f}%")

        if zstd_batch:
            ratio = best_skip[1]['batch_skiploop_ms'] / zstd_batch
            print(f"  Best H.265 / Zstd ratio:        {ratio:.1f}x slower")

    # Save results
    out_file = "results/h265_decode_opts_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    save_data = {
        'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'dir'}
                        for k, v in all_results.items()},
        'real_crf18_batch_ms': real_crf18_batch,
        'real_crf18_skiploop_ms': real_crf18_skip,
        'zstd_batch_ms': zstd_batch,
        'config': {
            'num_frames': NUM_TEST_FRAMES,
            'batch_tpd': BATCH_TPD,
            'batch_par': BATCH_PAR,
            'width': WIDTH, 'height': HEIGHT,
        }
    }
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
