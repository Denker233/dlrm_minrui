#!/usr/bin/env python3
"""
Benchmark: H.265 decode speed optimizations at fixed 1920x1080 resolution.

Tests:
1. Lossy CRF values (0=lossless, 18, 28, 35) — smaller bitstream → faster decode
2. H.265 with slices (1,2,4,8) — parallel slice decode
3. H.264 lossless — simpler decoder
4. FFV1 lossless — simplest entropy coding
5. H.265 decoder thread_count variants

Also measures quality impact (MSE, max error) for lossy configs.
"""

import os, sys, time, tempfile, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C

EMB_DIM = 16
TILE_H = TILE_W = 4
WIDTH, HEIGHT = 1920, 1080
ROWS_PER_FRAME = (WIDTH // TILE_W) * (HEIGHT // TILE_H)  # 129600
HOTCOLD_DIR = "results/hotcold"
REORDER_DIR = "results/reorder"

WARMUP = 3
ITERS = 20


def log(msg):
    print(msg, flush=True)


def load_cold_frame_data(table_id=2):
    """Load real cold embedding data and prepare a tiled frame."""
    from benchmark_full_comparison import load_model_and_data
    log("Loading model and data...")
    dlrm, test_ld, ln_emb, state_dict = load_model_and_data()

    cold_indices = torch.load(os.path.join(HOTCOLD_DIR, f'cold_indices_{table_id}.pt'),
                               map_location='cpu', weights_only=True)
    o2c_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{table_id}.npy')
    if os.path.exists(o2c_path):
        o2c = torch.from_numpy(np.load(o2c_path).copy()).long()
    else:
        o2c = torch.load(os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{table_id}.pt'),
                          map_location='cpu', weights_only=True).long()

    emb_key = f'emb_l.{table_id}.weight'
    full_emb = state_dict[emb_key]
    cold_emb = full_emb[cold_indices].float()

    # Quantize
    vmin, vmax = cold_emb.min(), cold_emb.max()
    scale = (vmax - vmin) / 255.0
    zp = vmin
    cold_uint8 = ((cold_emb - zp) / scale).clamp(0, 255).to(torch.uint8)

    # Reorder
    max_cold_idx = o2c[cold_indices].max().item() + 1
    reordered = torch.zeros(max_cold_idx, EMB_DIM, dtype=torch.uint8)
    for i in range(cold_indices.shape[0]):
        orig_idx = cold_indices[i].item()
        cold_pos = o2c[orig_idx].item()
        if cold_pos >= 0:
            reordered[cold_pos] = cold_uint8[i]

    # Take first frame's worth of data
    frame_data = reordered[:ROWS_PER_FRAME]
    if frame_data.shape[0] < ROWS_PER_FRAME:
        pad = torch.zeros(ROWS_PER_FRAME - frame_data.shape[0], EMB_DIM, dtype=torch.uint8)
        frame_data = torch.cat([frame_data, pad])

    # Tile into frame
    tiled = _C.tile_rows_to_frame(frame_data, WIDTH, HEIGHT)

    log(f"Prepared frame: {WIDTH}x{HEIGHT}, {ROWS_PER_FRAME} rows, data range [{frame_data.min()}, {frame_data.max()}]")
    return tiled, frame_data, float(scale), float(zp)


def encode_and_measure(tiled_frame, original_rows, scale, zp, config_name,
                       codec="h265", lossless=True, crf=0, extra_params=""):
    """Encode frame with given config, measure decode time and quality."""
    tmpdir = tempfile.mkdtemp(prefix=f'decbench_{config_name}_')
    fpath = os.path.join(tmpdir, f'frame.h265')
    if codec == "h264":
        fpath = os.path.join(tmpdir, f'frame.h264')
    elif codec == "ffv1":
        fpath = os.path.join(tmpdir, f'frame.mkv')

    try:
        # Encode
        t_enc0 = time.perf_counter()
        _C.encode_frame_with_params(tiled_frame, fpath, codec, lossless, crf, extra_params)
        enc_ms = (time.perf_counter() - t_enc0) * 1000

        file_size = os.path.getsize(fpath)
        compression_ratio = (ROWS_PER_FRAME * EMB_DIM) / max(1, file_size)

        # Decode timing
        decode_times = []
        decoded = None
        for i in range(WARMUP + ITERS):
            t0 = time.perf_counter()
            decoded = _C.decode_h265_frame_from_file(fpath)
            dt = (time.perf_counter() - t0) * 1000
            if i >= WARMUP:
                decode_times.append(dt)

        avg_dec = np.mean(decode_times)
        p50_dec = np.percentile(decode_times, 50)
        p99_dec = np.percentile(decode_times, 99)
        min_dec = np.min(decode_times)

        # Quality: untile and compare to original
        decoded_rows = _C.untile_frame_to_rows(decoded, ROWS_PER_FRAME)
        # Dequantize both
        orig_fp32 = original_rows.float() * scale + zp
        dec_fp32 = decoded_rows.float() * scale + zp

        diff = (orig_fp32 - dec_fp32).abs()
        mse = (diff ** 2).mean().item()
        max_err = diff.max().item()
        mean_err = diff.mean().item()

        return {
            'config': config_name,
            'codec': codec,
            'lossless': lossless,
            'crf': crf,
            'extra_params': extra_params,
            'file_size_kb': file_size / 1024,
            'compression_ratio': compression_ratio,
            'encode_ms': enc_ms,
            'avg_decode_ms': avg_dec,
            'p50_decode_ms': p50_dec,
            'p99_decode_ms': p99_dec,
            'min_decode_ms': min_dec,
            'mse': mse,
            'max_err': max_err,
            'mean_err': mean_err,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    tiled_frame, original_rows, scale, zp = load_cold_frame_data()

    results = []

    # ============================================================
    # TEST 1: H.265 CRF sweep (lossy vs lossless)
    # ============================================================
    log(f"\n{'='*70}")
    log("TEST 1: H.265 CRF sweep (lossless vs lossy)")
    log(f"{'='*70}")

    for crf_val in [0, 10, 18, 23, 28, 35]:
        ll = (crf_val == 0)
        name = f"h265_crf{crf_val}" if not ll else "h265_lossless"
        log(f"\n--- {name} ---")
        r = encode_and_measure(tiled_frame, original_rows, scale, zp,
                               name, codec="h265", lossless=ll, crf=crf_val)
        results.append(r)
        log(f"  Decode: avg={r['avg_decode_ms']:.2f}ms, p50={r['p50_decode_ms']:.2f}ms, "
            f"file={r['file_size_kb']:.0f}KB, ratio={r['compression_ratio']:.1f}x, "
            f"MSE={r['mse']:.4f}, max_err={r['max_err']:.4f}")

    # ============================================================
    # TEST 2: H.265 lossless with slices for parallel decode
    # ============================================================
    log(f"\n{'='*70}")
    log("TEST 2: H.265 lossless with different slice counts")
    log(f"{'='*70}")

    for num_slices in [1, 2, 4, 8, 16]:
        name = f"h265_lossless_slices{num_slices}"
        log(f"\n--- {name} ---")
        r = encode_and_measure(tiled_frame, original_rows, scale, zp,
                               name, codec="h265", lossless=True, crf=0,
                               extra_params=f"slices={num_slices}")
        results.append(r)
        log(f"  Decode: avg={r['avg_decode_ms']:.2f}ms, p50={r['p50_decode_ms']:.2f}ms, "
            f"file={r['file_size_kb']:.0f}KB, ratio={r['compression_ratio']:.1f}x")

    # ============================================================
    # TEST 3: H.265 lossy (CRF=18) with slices
    # ============================================================
    log(f"\n{'='*70}")
    log("TEST 3: H.265 CRF=18 with different slice counts")
    log(f"{'='*70}")

    for num_slices in [1, 2, 4, 8]:
        name = f"h265_crf18_slices{num_slices}"
        log(f"\n--- {name} ---")
        r = encode_and_measure(tiled_frame, original_rows, scale, zp,
                               name, codec="h265", lossless=False, crf=18,
                               extra_params=f"slices={num_slices}")
        results.append(r)
        log(f"  Decode: avg={r['avg_decode_ms']:.2f}ms, p50={r['p50_decode_ms']:.2f}ms, "
            f"file={r['file_size_kb']:.0f}KB, MSE={r['mse']:.4f}, max_err={r['max_err']:.4f}")

    # ============================================================
    # TEST 4: H.264 (lossless and lossy)
    # ============================================================
    log(f"\n{'='*70}")
    log("TEST 4: H.264 (lossless and lossy)")
    log(f"{'='*70}")

    for crf_val in [0, 18, 28]:
        ll = (crf_val == 0)
        name = f"h264_crf{crf_val}" if not ll else "h264_lossless"
        log(f"\n--- {name} ---")
        r = encode_and_measure(tiled_frame, original_rows, scale, zp,
                               name, codec="h264", lossless=ll, crf=crf_val)
        results.append(r)
        log(f"  Decode: avg={r['avg_decode_ms']:.2f}ms, p50={r['p50_decode_ms']:.2f}ms, "
            f"file={r['file_size_kb']:.0f}KB, ratio={r['compression_ratio']:.1f}x, "
            f"MSE={r['mse']:.4f}, max_err={r['max_err']:.4f}")

    # ============================================================
    # TEST 5: FFV1 lossless
    # ============================================================
    log(f"\n{'='*70}")
    log("TEST 5: FFV1 lossless")
    log(f"{'='*70}")

    name = "ffv1_lossless"
    log(f"\n--- {name} ---")
    r = encode_and_measure(tiled_frame, original_rows, scale, zp,
                           name, codec="ffv1", lossless=True)
    results.append(r)
    log(f"  Decode: avg={r['avg_decode_ms']:.2f}ms, p50={r['p50_decode_ms']:.2f}ms, "
        f"file={r['file_size_kb']:.0f}KB, ratio={r['compression_ratio']:.1f}x")

    # ============================================================
    # TEST 6: H.265 with WPP on/off
    # ============================================================
    log(f"\n{'='*70}")
    log("TEST 6: H.265 lossless WPP on/off")
    log(f"{'='*70}")

    for wpp in [0, 1]:
        name = f"h265_lossless_wpp{wpp}"
        log(f"\n--- {name} ---")
        r = encode_and_measure(tiled_frame, original_rows, scale, zp,
                               name, codec="h265", lossless=True, crf=0,
                               extra_params=f"wpp={wpp}")
        results.append(r)
        log(f"  Decode: avg={r['avg_decode_ms']:.2f}ms, p50={r['p50_decode_ms']:.2f}ms, "
            f"file={r['file_size_kb']:.0f}KB")

    # ============================================================
    # SUMMARY
    # ============================================================
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")
    log(f"{'Config':<30s} | {'Decode ms':>10s} | {'Size KB':>8s} | {'Ratio':>6s} | "
        f"{'MSE':>8s} | {'MaxErr':>8s} | {'Encode ms':>10s}")
    log("-" * 100)

    # Sort by decode time
    results.sort(key=lambda r: r['avg_decode_ms'])
    for r in results:
        log(f"  {r['config']:<28s} | {r['avg_decode_ms']:>8.2f}ms | "
            f"{r['file_size_kb']:>6.0f}KB | {r['compression_ratio']:>5.1f}x | "
            f"{r['mse']:>8.4f} | {r['max_err']:>8.4f} | {r['encode_ms']:>8.0f}ms")

    # Best lossless
    lossless_results = [r for r in results if r['mse'] == 0.0]
    if lossless_results:
        best_ll = min(lossless_results, key=lambda r: r['avg_decode_ms'])
        log(f"\nBest lossless: {best_ll['config']} @ {best_ll['avg_decode_ms']:.2f}ms "
            f"({best_ll['compression_ratio']:.1f}x compression)")

    # Best with MSE < 0.01
    low_err = [r for r in results if r['mse'] < 0.01]
    if low_err:
        best_le = min(low_err, key=lambda r: r['avg_decode_ms'])
        log(f"Best low-error (MSE<0.01): {best_le['config']} @ {best_le['avg_decode_ms']:.2f}ms "
            f"(MSE={best_le['mse']:.6f})")

    baseline = next((r for r in results if r['config'] == 'h265_lossless'), None)
    if baseline:
        log(f"\nSpeedup vs baseline (h265_lossless @ {baseline['avg_decode_ms']:.2f}ms):")
        for r in results:
            speedup = baseline['avg_decode_ms'] / r['avg_decode_ms']
            log(f"  {r['config']:<28s}: {speedup:.2f}x")


if __name__ == '__main__':
    main()
