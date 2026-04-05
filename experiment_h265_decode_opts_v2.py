#!/usr/bin/env python3
"""
Experiment: H.265 decode optimization v2 — find config for pipeline-viable decode.

Target: single-frame decode < 2.3ms (= inference time per batch), so we can
pipeline decode of next-batch frames during current-batch inference.

Current best (v1): CRF=30 medium skip_loop = 5.13ms (4T). Need ~2.4x improvement.

New axes tested:
  1. Presets: ultrafast, medium, slow, slower (does the trend continue?)
  2. CRF: 30, 40, 45, 50, 51 (max — smaller bitstream = faster decode)
  3. Encoder params: no-deblock:no-sao (remove filters from bitstream)
  4. Decoder flags: skip_loop, skip_idct, FLAG2_FAST, all combos
  5. Medium + tiles (not tested in v1)
  6. Thread sweep: 1, 2, 4 per frame
"""

import os, sys, time, json, subprocess, shutil
import numpy as np

os.environ.setdefault('CRITEO_DAYS', '4')

import torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')

torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if torch_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import compressed_emb as _C

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_FRAME_DIR = "results/ondemand/1080p_crf18/table_2"
OUTPUT_BASE = "/tmp/h265_decode_opts_v2"
WIDTH, HEIGHT = 1920, 1080
NUM_TEST_FRAMES = 20
REPEAT = 7  # more repeats for stability

# Encoding configs: (name, crf, preset, extra_x265_params)
ENCODE_CONFIGS = [
    # --- Preset sweep at CRF=30 (best CRF from v1) ---
    ("crf30_ultrafast",    30, "ultrafast", ""),
    ("crf30_superfast",    30, "superfast", ""),
    ("crf30_medium",       30, "medium",    ""),
    ("crf30_slow",         30, "slow",      ""),
    ("crf30_slower",       30, "slower",    ""),

    # --- CRF sweep with medium preset (best preset from v1) ---
    ("crf40_medium",       40, "medium",    ""),
    ("crf45_medium",       45, "medium",    ""),
    ("crf50_medium",       50, "medium",    ""),
    ("crf51_medium",       51, "medium",    ""),

    # --- Encoder-side filter removal (less work for decoder) ---
    ("crf30_med_nodeblock",    30, "medium", "no-deblock=1:no-sao=1"),
    ("crf40_med_nodeblock",    40, "medium", "no-deblock=1:no-sao=1"),
    ("crf51_med_nodeblock",    51, "medium", "no-deblock=1:no-sao=1"),
    ("crf30_uf_nodeblock",     30, "ultrafast", "no-deblock=1:no-sao=1"),
    ("crf51_uf_nodeblock",     51, "ultrafast", "no-deblock=1:no-sao=1"),

    # --- Medium + tiles (not tested in v1) ---
    ("crf30_med_tiles2",   30, "medium", "frame-threads=1:tiles=2"),
    ("crf30_med_tiles4",   30, "medium", "frame-threads=1:tiles=4"),
    ("crf51_med_tiles4",   51, "medium", "frame-threads=1:tiles=4"),

    # --- High CRF + ultrafast (baseline for high CRF) ---
    ("crf45_ultrafast",    45, "ultrafast", ""),
    ("crf50_ultrafast",    50, "ultrafast", ""),
    ("crf51_ultrafast",    51, "ultrafast", ""),

    # --- Kitchen sink: max CRF + no filters + medium ---
    ("crf51_med_nodeblock_tiles4", 51, "medium", "no-deblock=1:no-sao=1:frame-threads=1:tiles=4"),
]


def get_raw_frame_data():
    """Load one real CRF=18 frame, decode to raw pixels for re-encoding."""
    frame_path = os.path.join(BASE_FRAME_DIR, "frame_00000.h265")
    raw = _C.decode_h265_frame_from_file(frame_path, 4, False)
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
        '-f', 'hevc',
        output_path,
    ]
    proc = subprocess.run(cmd, input=raw_np.tobytes(), capture_output=True, timeout=60)
    if proc.returncode != 0:
        # fallback to matroska container
        cmd[-2] = 'matroska'
        cmd[-1] = output_path.replace('.h265', '.mkv')
        proc = subprocess.run(cmd, input=raw_np.tobytes(), capture_output=True, timeout=60)
        if proc.returncode != 0:
            print(f"  ENCODE FAILED: {proc.stderr.decode()[-300:]}")
            return False, output_path
        return True, output_path.replace('.h265', '.mkv')
    return True, output_path


def encode_all_configs(raw_np):
    """Encode the raw frame with all configs, creating NUM_TEST_FRAMES copies."""
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    results = {}

    for name, crf, preset, extra in ENCODE_CONFIGS:
        config_dir = os.path.join(OUTPUT_BASE, name)
        os.makedirs(config_dir, exist_ok=True)

        single_path = os.path.join(config_dir, "frame_00000.h265")
        t0 = time.perf_counter()
        ok, actual_path = encode_frame_ffmpeg(raw_np, single_path, crf, preset, extra)
        enc_time = time.perf_counter() - t0

        if not ok or not os.path.exists(actual_path):
            print(f"  SKIP {name}: encode failed")
            continue

        fsize = os.path.getsize(actual_path)
        ext = os.path.splitext(actual_path)[1]
        print(f"  {name}: {fsize} bytes, encode {enc_time*1000:.1f}ms")

        # Copy to create test frames
        for i in range(1, NUM_TEST_FRAMES):
            dst = os.path.join(config_dir, f"frame_{i:05d}{ext}")
            shutil.copy2(actual_path, dst)

        results[name] = {
            'crf': crf, 'preset': preset, 'extra': extra,
            'frame_bytes': fsize, 'encode_ms': enc_time * 1000,
            'dir': config_dir, 'ext': ext,
        }

    return results


def benchmark_single_frame(config_dir, ext='.h265', skip_loop=False, skip_idct=False, fast_decode=False):
    """Benchmark single frame decode at different thread counts."""
    path = os.path.join(config_dir, f"frame_00000{ext}")
    if not os.path.exists(path):
        return {}

    thread_counts = [1, 2, 4]
    result = {}

    for nt in thread_counts:
        times = []
        for _ in range(REPEAT + 2):
            t0 = time.perf_counter()
            _C.decode_h265_frame_from_file(path, nt, skip_loop, skip_idct, fast_decode)
            times.append(time.perf_counter() - t0)

        # Drop first 2 warmups
        result[f'{nt}t'] = np.median(times[2:]) * 1000

    return result


def benchmark_batch_decode(config_dir, ext='.h265', skip_loop=False, skip_idct=False, fast_decode=False):
    """Benchmark batch decode of NUM_TEST_FRAMES frames."""
    paths = [os.path.join(config_dir, f"frame_{i:05d}{ext}") for i in range(NUM_TEST_FRAMES)]
    if not all(os.path.exists(p) for p in paths):
        return None

    # Use tpd=1, par=20 for batch (each frame single-threaded, max parallelism)
    times = []
    for _ in range(REPEAT + 2):
        t0 = time.perf_counter()
        _C.batch_decode_file_paths(paths, 1, 20, skip_loop, skip_idct, fast_decode)
        times.append(time.perf_counter() - t0)

    return np.median(times[2:]) * 1000


def measure_quality(config_dir, raw_ref, ext='.h265'):
    """Measure decode quality vs raw reference."""
    path = os.path.join(config_dir, f"frame_00000{ext}")
    if not os.path.exists(path):
        return None, None

    decoded = _C.decode_h265_frame_from_file(path, 4, False).numpy()
    diff = decoded.astype(np.float32) - raw_ref.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    return mse, psnr


# Decoder flag combinations to test
DECODER_FLAGS = [
    ("none",         False, False, False),
    ("skip_loop",    True,  False, False),
    ("skip_idct",    False, True,  False),
    ("fast",         False, False, True),
    ("loop+idct",    True,  True,  False),
    ("loop+fast",    True,  False, True),
    ("all_flags",    True,  True,  True),
]


def main():
    print("=" * 80)
    print("H.265 Decode Optimization v2 — Target: <2.3ms single-frame for pipelining")
    print("=" * 80)
    print()

    # Step 1: Get raw frame data
    print("[1] Loading raw frame from real CRF=18 data...")
    raw_np = get_raw_frame_data()
    raw_bytes = raw_np.nbytes
    print(f"    Frame: {raw_np.shape}, {raw_bytes:,} bytes ({raw_bytes/1024:.1f} KB)")
    print()

    # Step 2: Encode all configs
    print("[2] Encoding frames with all configs...")
    configs = encode_all_configs(raw_np)
    print(f"    {len(configs)} configs encoded successfully")
    print()

    # Step 3: Benchmark each config × decoder flags
    print("[3] Benchmarking decode speed...")
    print()

    all_results = {}

    for name, info in configs.items():
        config_dir = info['dir']
        ext = info['ext']
        print(f"  === {name} ({info['frame_bytes']} bytes, {raw_bytes/info['frame_bytes']:.0f}x) ===")

        # Quality (decode without skips)
        mse, psnr = measure_quality(config_dir, raw_np, ext)

        config_results = {
            'crf': info['crf'], 'preset': info['preset'], 'extra': info['extra'],
            'frame_bytes': info['frame_bytes'], 'encode_ms': info['encode_ms'],
            'compression_ratio': raw_bytes / info['frame_bytes'],
            'mse': mse, 'psnr': psnr,
            'decoder_configs': {},
        }

        for flag_name, sl, si, fd in DECODER_FLAGS:
            single = benchmark_single_frame(config_dir, ext, sl, si, fd)
            batch = benchmark_batch_decode(config_dir, ext, sl, si, fd)

            config_results['decoder_configs'][flag_name] = {
                'skip_loop': sl, 'skip_idct': si, 'fast_decode': fd,
                'single_ms': single,
                'batch_20f_ms': batch,
            }

            best_single = min(single.values()) if single else 999
            print(f"    {flag_name:15s}: 1T={single.get('1t',0):5.2f}  2T={single.get('2t',0):5.2f}  "
                  f"4T={single.get('4t',0):5.2f}  batch={batch:6.2f}ms" if batch else
                  f"    {flag_name:15s}: 1T={single.get('1t',0):5.2f}  2T={single.get('2t',0):5.2f}  "
                  f"4T={single.get('4t',0):5.2f}  batch=N/A")

        all_results[name] = config_results
        print()

    # Step 4: Zstd baseline
    print("[4] Zstd baseline...")
    zstd_dir = "results/ondemand/1080p_zstd3/table_2"
    zstd_batch = None
    if os.path.isdir(zstd_dir):
        paths = [os.path.join(zstd_dir, f"frame_{i:05d}.zst") for i in range(NUM_TEST_FRAMES)]
        existing = [p for p in paths if os.path.exists(p)]
        if len(existing) >= NUM_TEST_FRAMES:
            times = []
            for _ in range(REPEAT + 2):
                t0 = time.perf_counter()
                _C.batch_decode_file_paths(existing, 1, 20, False)
                times.append(time.perf_counter() - t0)
            zstd_batch = np.median(times[2:]) * 1000
            print(f"    Zstd-3 batch {NUM_TEST_FRAMES}f: {zstd_batch:.2f}ms")
    if zstd_batch is None:
        print("    Zstd frames not found")
    print()

    # Step 5: Summary — find best configs
    print("=" * 80)
    print("TOP RESULTS BY SINGLE-FRAME DECODE TIME (4T)")
    print("=" * 80)
    print(f"{'Config':<35s} {'Flags':<15s} {'1T':>6s} {'2T':>6s} {'4T':>6s} "
          f"{'Batch':>7s} {'Ratio':>6s} {'PSNR':>7s}")
    print("-" * 95)

    # Collect all (config, flag_combo, time) triples
    rankings = []
    for name, r in all_results.items():
        for flag_name, dc in r['decoder_configs'].items():
            t4 = dc['single_ms'].get('4t', 999)
            rankings.append((name, flag_name, dc, r))

    # Sort by 4T single-frame time
    rankings.sort(key=lambda x: x[2]['single_ms'].get('4t', 999))

    for name, flag_name, dc, r in rankings[:25]:
        s = dc['single_ms']
        psnr_str = f"{r['psnr']:.1f}" if r['psnr'] and r['psnr'] != float('inf') else "inf"
        batch_str = f"{dc['batch_20f_ms']:.2f}" if dc['batch_20f_ms'] else "N/A"
        print(f"{name:<35s} {flag_name:<15s} {s.get('1t',0):5.2f}m {s.get('2t',0):5.2f}m "
              f"{s.get('4t',0):5.2f}m {batch_str:>7s} {r['compression_ratio']:>5.0f}x {psnr_str:>7s}")

    print("-" * 95)
    if zstd_batch:
        print(f"{'Zstd-3 (lossless reference)':<35s} {'N/A':<15s} {'':>6s} {'':>6s} {'':>6s} "
              f"{zstd_batch:>6.2f}m {'~19':>5s}x {'inf':>7s}")
    print(f"\nTarget: single-frame 4T < 2.3ms for pipeline viability")

    # Best result
    if rankings:
        best = rankings[0]
        best_4t = best[2]['single_ms'].get('4t', 999)
        print(f"\nBEST: {best[0]} + {best[1]} → {best_4t:.2f}ms (4T)")
        if best_4t < 2.3:
            print("  ✓ PIPELINE VIABLE!")
        else:
            print(f"  Still {best_4t/2.3:.1f}x above target")

    # Save results
    out_file = "results/h265_decode_opts_v2_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    save_data = {
        'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'dir'}
                        for k, v in all_results.items()},
        'zstd_batch_ms': zstd_batch,
        'config': {
            'num_frames': NUM_TEST_FRAMES,
            'width': WIDTH, 'height': HEIGHT,
            'repeats': REPEAT,
        }
    }
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
