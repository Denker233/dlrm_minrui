#!/usr/bin/env python3
"""Quick benchmark for remaining decode configs: H.264, FFV1, WPP."""

import os, sys, time, tempfile, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C

EMB_DIM = 16
TILE_H = TILE_W = 4
WIDTH, HEIGHT = 1920, 1080
ROWS_PER_FRAME = (WIDTH // TILE_W) * (HEIGHT // TILE_H)
HOTCOLD_DIR = "results/hotcold"
REORDER_DIR = "results/reorder"
WARMUP = 3
ITERS = 20


def log(msg):
    print(msg, flush=True)


def load_cold_frame():
    from benchmark_full_comparison import load_model_and_data
    log("Loading model and data...")
    dlrm, test_ld, ln_emb, state_dict = load_model_and_data()

    table_id = 2
    cold_indices = torch.load(os.path.join(HOTCOLD_DIR, f'cold_indices_{table_id}.pt'),
                               map_location='cpu', weights_only=True)
    o2c_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{table_id}.npy')
    o2c = torch.from_numpy(np.load(o2c_path).copy()).long()

    full_emb = state_dict[f'emb_l.{table_id}.weight']
    cold_emb = full_emb[cold_indices].float()
    vmin, vmax = cold_emb.min(), cold_emb.max()
    scale = (vmax - vmin) / 255.0
    zp = vmin
    cold_uint8 = ((cold_emb - zp) / scale).clamp(0, 255).to(torch.uint8)

    max_cold_idx = o2c[cold_indices].max().item() + 1
    reordered = torch.zeros(max_cold_idx, EMB_DIM, dtype=torch.uint8)
    for i in range(cold_indices.shape[0]):
        orig_idx = cold_indices[i].item()
        cold_pos = o2c[orig_idx].item()
        if cold_pos >= 0:
            reordered[cold_pos] = cold_uint8[i]

    frame_data = reordered[:ROWS_PER_FRAME]
    if frame_data.shape[0] < ROWS_PER_FRAME:
        frame_data = torch.cat([frame_data, torch.zeros(ROWS_PER_FRAME - frame_data.shape[0], EMB_DIM, dtype=torch.uint8)])

    tiled = _C.tile_rows_to_frame(frame_data, WIDTH, HEIGHT)
    return tiled, frame_data, float(scale), float(zp)


def bench(tiled, original, scale, zp, name, codec, lossless, crf=0, extra_params=""):
    tmpdir = tempfile.mkdtemp(prefix=f'decbench_')
    ext = {'h265': '.h265', 'h264': '.h264', 'ffv1': '.mkv'}.get(codec, '.h265')
    fpath = os.path.join(tmpdir, f'frame{ext}')

    try:
        t0 = time.perf_counter()
        _C.encode_frame_with_params(tiled, fpath, codec, lossless, crf, extra_params)
        enc_ms = (time.perf_counter() - t0) * 1000
        fsize = os.path.getsize(fpath)
        ratio = (ROWS_PER_FRAME * EMB_DIM) / max(1, fsize)

        times = []
        for i in range(WARMUP + ITERS):
            t0 = time.perf_counter()
            dec = _C.decode_h265_frame_from_file(fpath)
            dt = (time.perf_counter() - t0) * 1000
            if i >= WARMUP:
                times.append(dt)

        avg = np.mean(times)
        p50 = np.percentile(times, 50)

        # Quality check
        dec_rows = _C.untile_frame_to_rows(dec, ROWS_PER_FRAME)
        diff = (original.float() * scale + zp) - (dec_rows.float() * scale + zp)
        mse = (diff ** 2).mean().item()
        max_err = diff.abs().max().item()

        log(f"  {name:<30s} | avg={avg:.2f}ms p50={p50:.2f}ms | {fsize/1024:.0f}KB ({ratio:.1f}x) | "
            f"MSE={mse:.6f} maxerr={max_err:.4f} | enc={enc_ms:.0f}ms")
        return {'config': name, 'avg_decode_ms': avg, 'file_kb': fsize/1024,
                'ratio': ratio, 'mse': mse, 'max_err': max_err}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    tiled, orig, scale, zp = load_cold_frame()

    log(f"\n{'='*70}")
    log("DECODE SPEED BENCHMARK (1920x1080, same frame)")
    log(f"{'='*70}\n")

    results = []

    # Baseline
    results.append(bench(tiled, orig, scale, zp, "h265_lossless", "h265", True))

    # CRF sweep
    for crf in [10, 18, 23, 28]:
        results.append(bench(tiled, orig, scale, zp, f"h265_crf{crf}", "h265", False, crf))

    # H.264
    results.append(bench(tiled, orig, scale, zp, "h264_lossless", "h264", True))
    for crf in [18, 28]:
        results.append(bench(tiled, orig, scale, zp, f"h264_crf{crf}", "h264", False, crf))

    # FFV1
    results.append(bench(tiled, orig, scale, zp, "ffv1_lossless", "ffv1", True))

    # WPP
    results.append(bench(tiled, orig, scale, zp, "h265_lossless_wpp0", "h265", True, 0, "wpp=0"))
    results.append(bench(tiled, orig, scale, zp, "h265_lossless_wpp1", "h265", True, 0, "wpp=1"))

    # H.265 CRF=18 + WPP
    results.append(bench(tiled, orig, scale, zp, "h265_crf18_wpp1", "h265", False, 18, "wpp=1"))

    # Also test on existing real files from the production encoding
    log(f"\n--- Real production .h265 files (existing on disk) ---")
    ondemand_dir = "results/ondemand/1080p/table_2"
    if os.path.exists(ondemand_dir):
        for fname in sorted(os.listdir(ondemand_dir)):
            if not fname.endswith('.h265'):
                continue
            fpath = os.path.join(ondemand_dir, fname)
            fsize = os.path.getsize(fpath)
            times = []
            for i in range(WARMUP + ITERS):
                t0 = time.perf_counter()
                _C.decode_h265_frame_from_file(fpath)
                dt = (time.perf_counter() - t0) * 1000
                if i >= WARMUP:
                    times.append(dt)
            avg = np.mean(times)
            log(f"  {fname}: avg={avg:.2f}ms, {fsize/1024:.0f}KB")

    # Summary
    log(f"\n{'='*70}")
    log("SORTED BY DECODE TIME")
    log(f"{'='*70}")
    results.sort(key=lambda r: r['avg_decode_ms'])
    baseline_ms = next(r['avg_decode_ms'] for r in results if r['config'] == 'h265_lossless')
    for r in results:
        speedup = baseline_ms / r['avg_decode_ms']
        log(f"  {r['config']:<30s} {r['avg_decode_ms']:>6.2f}ms ({speedup:.2f}x) | "
            f"{r['file_kb']:>6.0f}KB | MSE={r['mse']:.6f}")


if __name__ == '__main__':
    main()
