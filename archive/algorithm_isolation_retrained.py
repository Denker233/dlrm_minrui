#!/usr/bin/env python3
"""
Algorithm Isolation on Retrained Model
Replicates the early Phase 3 isolation tests with the retrained model.

Tests (single-frame per table):
1. FFV1 (entropy only, no spatial prediction)
2. H.265 lossless (CRF 0) — intra pred + transform + CABAC
3. H.265 CRF 23 — full pipeline with lossy quant
4. H.265 CRF 23, no SAO
5. H.265 CRF 23, no deblock
6. H.265 CRF 23, no SAO + no deblock
7. H.265 CRF 23, no intra smoothing
8. H.264 CRF 23

Tests (multi-frame, all tables in one video):
9.  I-frames only (keyint=1) — no inter-frame prediction
10. With inter-frame (keyint=250)
11. With inter-frame (keyint=999)
"""

import os
import sys
import time
import json
import tempfile
import subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dlrm_s_pytorch import tile_embeddings, untile_embeddings

MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
RESULTS_DIR = os.path.expanduser("~/experiment-control")

# Frame size for multi-frame encoding (all tables padded to this)
MULTI_FRAME_W = 512
MULTI_FRAME_H = 1024

# Single-frame encoding params (same as Phase 2A)
MIN_WIDTH = 64
MIN_HEIGHT = 64
MAX_DIM = 16384
TILING_THRESHOLD = 50000
TILE_SIZE = 4


def quantize_table(weights):
    """Quantize FP32 weights to UINT8."""
    w_min = weights.min().item()
    w_max = weights.max().item()
    scale = (w_max - w_min) / 255.0
    if scale == 0:
        scale = 1.0
    zp = round(-w_min / scale)
    quantized = ((weights / scale).round() + zp).clamp(0, 255).to(torch.uint8)
    return quantized, scale, zp


def prepare_single_frame(pixels_np, num_emb, emb_dim):
    """Prepare a single-frame raw buffer with tiling/padding (same as Phase 2A)."""
    width = emb_dim
    height = num_emb
    tiled = False

    if num_emb > TILING_THRESHOLD:
        data_flat = pixels_np.reshape(-1)
        image, grid_size, tiles_per_emb = tile_embeddings(
            data_flat, emb_dim, num_emb, TILE_SIZE)
        width = image.shape[1]
        height = image.shape[0]
        raw_data = image.tobytes()
        tiled = True
    else:
        raw_data = pixels_np.tobytes()

    total_pixels = width * height
    if height > MAX_DIM or width < MIN_WIDTH or height < MIN_HEIGHT:
        min_required = MIN_WIDTH * MIN_HEIGHT
        if total_pixels < min_required:
            width = MIN_WIDTH
            height = MIN_HEIGHT
        elif height > MAX_DIM:
            width = (total_pixels + MAX_DIM - 1) // MAX_DIM
            height = MAX_DIM
            if width < MIN_WIDTH:
                width = MIN_WIDTH
                height = (total_pixels + width - 1) // width
        elif width < MIN_WIDTH:
            width = MIN_WIDTH
            height = (total_pixels + width - 1) // width
            if height < MIN_HEIGHT:
                height = MIN_HEIGHT
        elif height < MIN_HEIGHT:
            height = MIN_HEIGHT
            width = (total_pixels + height - 1) // height
            if width < MIN_WIDTH:
                width = MIN_WIDTH
                height = MIN_HEIGHT

        padded_pixels = width * height
        if padded_pixels > len(raw_data):
            padded = bytearray(padded_pixels)
            padded[:len(raw_data)] = raw_data
            raw_data = bytes(padded)

    return raw_data, width, height


def encode_single_frame(raw_data, width, height, codec_args, container='mp4'):
    """Encode a single frame with given codec args. Returns (compressed_bytes, time)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_file = os.path.join(tmpdir, 'pixels.raw')
        ext = 'mkv' if container == 'mkv' else 'mp4'
        video_file = os.path.join(tmpdir, f'out.{ext}')
        with open(raw_file, 'wb') as f:
            f.write(raw_data)

        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{width}x{height}', '-r', '1', '-i', raw_file
               ] + codec_args + ['-frames:v', '1', video_file]
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elapsed = time.time() - t0
        if result.returncode != 0:
            raise RuntimeError(f"Encode failed: {result.stderr[:500]}")
        with open(video_file, 'rb') as f:
            compressed = f.read()
    return compressed, elapsed


def encode_multiframe(all_raw_frames, frame_w, frame_h, num_frames, codec_args, container='mp4'):
    """Encode multiple frames as one video. Returns (compressed_bytes, compress_time, decompress_time)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_file = os.path.join(tmpdir, 'all_frames.raw')
        ext = 'mkv' if container == 'mkv' else 'mp4'
        video_file = os.path.join(tmpdir, f'out.{ext}')
        decoded_file = os.path.join(tmpdir, 'decoded.raw')

        with open(raw_file, 'wb') as f:
            f.write(all_raw_frames)

        # Encode
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{frame_w}x{frame_h}', '-r', '30',
               '-i', raw_file] + codec_args + [video_file]
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ct = time.time() - t0
        if result.returncode != 0:
            raise RuntimeError(f"Encode failed: {result.stderr[:500]}")
        with open(video_file, 'rb') as f:
            compressed = f.read()

        # Decode
        t0 = time.time()
        result = subprocess.run(['ffmpeg', '-y', '-i', video_file, '-pix_fmt', 'gray',
                                 '-f', 'rawvideo', decoded_file],
                                capture_output=True, text=True, check=False)
        dt = time.time() - t0

    return compressed, ct, dt


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 80)
    print("ALGORITHM ISOLATION ON RETRAINED MODEL")
    print("Replicating early Phase 3 tests with retrained weights")
    print("=" * 80)

    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]
    num_tables = len(emb_keys)

    total_fp32 = sum(state_dict[k].numel() * 4 for k in emb_keys)
    total_uint8 = sum(state_dict[k].numel() for k in emb_keys)

    print(f"  Model: {MODEL_PATH}")
    print(f"  Tables: {num_tables}")
    print(f"  Total FP32: {total_fp32:,} bytes ({total_fp32/1024/1024:.2f} MB)")
    print(f"  Total UINT8: {total_uint8:,} bytes ({total_uint8/1024/1024:.2f} MB)")

    # ================================================================
    # SINGLE-FRAME TESTS (each table encoded independently)
    # ================================================================
    single_frame_configs = [
        ("Full H.265 (CRF 23)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1']),

        ("Entropy only (FFV1)", 'mkv',
         ['-c:v', 'ffv1']),

        ("Lossless H.265 (CRF 0)", 'mp4',
         ['-c:v', 'libx265', '-x265-params', 'lossless=1:log-level=error',
          '-preset', 'ultrafast']),

        ("No SAO (CRF 23)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1:sao=0']),

        ("No Deblock (CRF 23)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1:deblock=0,0']),

        ("No SAO + No Deblock (CRF 23)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1:sao=0:deblock=0,0']),

        ("No Intra Smoothing (CRF 23)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1:strong-intra-smoothing=0']),

        ("No Transform Skip (CRF 23)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1:tskip=0']),

        ("H.264 (CRF 23)", 'mp4',
         ['-c:v', 'libx264', '-crf', '23', '-preset', 'ultrafast']),
    ]

    single_results = []

    for config_name, container, codec_args in single_frame_configs:
        print(f"\n{'='*60}")
        print(f"SINGLE-FRAME: {config_name}")
        print(f"{'='*60}")

        total_compressed = 0
        total_ct = 0.0
        total_dt = 0.0
        failed = False

        for t in range(num_tables):
            w = state_dict[emb_keys[t]]
            num_emb, emb_dim = w.shape
            quantized, scale, zp = quantize_table(w)
            pixels = quantized.numpy()
            raw_data, width, height = prepare_single_frame(pixels, num_emb, emb_dim)

            try:
                compressed, ct = encode_single_frame(raw_data, width, height, codec_args, container)
                total_compressed += len(compressed)
                total_ct += ct

                if t % 5 == 0:
                    table_fp32 = w.numel() * 4
                    print(f"  Table {t:2d}: {w.shape} -> {len(compressed):,} bytes "
                          f"({table_fp32/len(compressed):.1f}x)")
            except Exception as e:
                print(f"  Table {t} FAILED: {e}")
                total_compressed += w.numel()  # fallback to uint8 size
                failed = True

        ratio_uint8 = total_uint8 / total_compressed if total_compressed > 0 else 0
        ratio_fp32 = total_fp32 / total_compressed if total_compressed > 0 else 0
        bits_per_val = (total_compressed * 8) / total_uint8

        result = {
            'name': config_name,
            'compressed_bytes': total_compressed,
            'compressed_mb': total_compressed / 1024 / 1024,
            'ratio_vs_uint8': ratio_uint8,
            'ratio_vs_fp32': ratio_fp32,
            'bits_per_value': bits_per_val,
            'compress_time': total_ct,
            'failed': failed,
        }
        single_results.append(result)

        print(f"  TOTAL: {total_compressed:,} bytes ({total_compressed/1024/1024:.2f} MB), "
              f"{ratio_uint8:.2f}x vs uint8, {ratio_fp32:.2f}x vs fp32, "
              f"{bits_per_val:.4f} bits/val, ct={total_ct:.2f}s")

    # ================================================================
    # MULTI-FRAME TESTS (all tables in one video)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"PREPARING MULTI-FRAME DATA")
    print(f"Frame size: {MULTI_FRAME_W}x{MULTI_FRAME_H} = {MULTI_FRAME_W*MULTI_FRAME_H} pixels/frame")
    print(f"{'='*60}")

    frame_size = MULTI_FRAME_W * MULTI_FRAME_H
    all_frames_data = bytearray()
    total_frames = 0

    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        quantized, scale, zp = quantize_table(w)
        pixels = quantized.numpy().reshape(-1)
        num_pixels = len(pixels)

        # Pad to multiple of frame_size
        n_frames = (num_pixels + frame_size - 1) // frame_size
        padded_size = n_frames * frame_size
        if padded_size > num_pixels:
            pixels = np.concatenate([pixels, np.zeros(padded_size - num_pixels, dtype=np.uint8)])

        all_frames_data.extend(pixels.tobytes())
        total_frames += n_frames
        if t % 5 == 0:
            print(f"  Table {t}: {w.shape} -> {n_frames} frames")

    all_frames_bytes = bytes(all_frames_data)
    print(f"  Total frames: {total_frames}")
    print(f"  Total raw size: {len(all_frames_bytes):,} bytes ({len(all_frames_bytes)/1024/1024:.2f} MB)")

    multi_configs = [
        ("Multi-frame: I-frames only (keyint=1)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1:keyint=1']),

        ("Multi-frame: with inter-frame (keyint=250)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1:keyint=250']),

        ("Multi-frame: with inter-frame (keyint=999)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1:keyint=999']),

        ("Multi-frame: I-only lossless (keyint=1, CRF 0)", 'mp4',
         ['-c:v', 'libx265', '-preset', 'ultrafast',
          '-x265-params', 'lossless=1:log-level=error:keyint=1']),

        ("Multi-frame: inter lossless (keyint=250, CRF 0)", 'mp4',
         ['-c:v', 'libx265', '-preset', 'ultrafast',
          '-x265-params', 'lossless=1:log-level=error:keyint=250']),
    ]

    multi_results = []

    for config_name, container, codec_args in multi_configs:
        print(f"\n{'='*60}")
        print(f"MULTI-FRAME: {config_name}")
        print(f"{'='*60}")

        try:
            compressed, ct, dt = encode_multiframe(
                all_frames_bytes, MULTI_FRAME_W, MULTI_FRAME_H,
                total_frames, codec_args, container)

            ratio_uint8 = total_uint8 / len(compressed)
            ratio_fp32 = total_fp32 / len(compressed)
            bits_per_val = (len(compressed) * 8) / total_uint8

            result = {
                'name': config_name,
                'compressed_bytes': len(compressed),
                'compressed_mb': len(compressed) / 1024 / 1024,
                'ratio_vs_uint8': ratio_uint8,
                'ratio_vs_fp32': ratio_fp32,
                'bits_per_value': bits_per_val,
                'compress_time': ct,
                'decompress_time': dt,
                'num_frames': total_frames,
                'failed': False,
            }
            multi_results.append(result)

            print(f"  RESULT: {len(compressed):,} bytes ({len(compressed)/1024/1024:.2f} MB), "
                  f"{ratio_uint8:.2f}x vs uint8, {ratio_fp32:.2f}x vs fp32, "
                  f"ct={ct:.2f}s, dt={dt:.2f}s")

        except Exception as e:
            print(f"  FAILED: {e}")
            multi_results.append({
                'name': config_name, 'failed': True, 'error': str(e)
            })

    # ================================================================
    # WRITE RESULTS
    # ================================================================
    print(f"\n{'='*60}")
    print("WRITING RESULTS")
    print(f"{'='*60}")

    out_path = os.path.join(RESULTS_DIR, "algorithm_isolation_retrained.md")
    with open(out_path, 'w') as f:
        f.write("# Algorithm Isolation: Retrained Model\n\n")
        f.write("## Reference Sizes\n\n")
        f.write(f"- **Raw float32:** {total_fp32:,} bytes ({total_fp32/1024/1024:.2f} MB)\n")
        f.write(f"- **Raw uint8 (after INT8 quantization):** {total_uint8:,} bytes ({total_uint8/1024/1024:.2f} MB)\n\n")

        # Single-frame table
        f.write("## Per-Algorithm Results (Single-Frame Encoding)\n\n")
        f.write("Each embedding table is encoded as a single grayscale video frame.\n\n")
        f.write("| Algorithm Config | File Size (bytes) | File Size (MB) | Comp Ratio vs uint8 | "
                "Comp Ratio vs float32 | Bits/Value | Compress Time (s) |\n")
        f.write("|-----------------|-------------------|---------------|--------------------|-"
                "----------------------|-----------|-------------------|\n")
        for r in single_results:
            if not r.get('failed'):
                f.write(f"| {r['name']} | {r['compressed_bytes']:,} | {r['compressed_mb']:.2f} | "
                        f"{r['ratio_vs_uint8']:.2f}x | {r['ratio_vs_fp32']:.2f}x | "
                        f"{r['bits_per_value']:.4f} | {r['compress_time']:.2f} |\n")

        # Multi-frame table
        f.write(f"\n## Multi-Frame Encoding Results (Inter-Frame Prediction Test)\n\n")
        f.write(f"All 26 embedding tables encoded as consecutive frames in a single video "
                f"({total_frames} frames of {MULTI_FRAME_W}x{MULTI_FRAME_H}).\n\n")
        f.write("| Algorithm Config | File Size (bytes) | File Size (MB) | Comp Ratio vs uint8 | "
                "Comp Ratio vs float32 | Bits/Value | Compress Time (s) | Decompress Time (s) |\n")
        f.write("|-----------------|-------------------|---------------|--------------------|-"
                "----------------------|-----------|-------------------|--------------------|\n")
        for r in multi_results:
            if not r.get('failed'):
                f.write(f"| {r['name']} | {r['compressed_bytes']:,} | {r['compressed_mb']:.2f} | "
                        f"{r['ratio_vs_uint8']:.2f}x | {r['ratio_vs_fp32']:.2f}x | "
                        f"{r['bits_per_value']:.4f} | {r['compress_time']:.2f} | "
                        f"{r.get('decompress_time', 0):.2f} |\n")

        # Inter-frame savings
        i_only_lossy = next((r for r in multi_results if 'I-frames only' in r['name'] and not r.get('failed')), None)
        inter_lossy = next((r for r in multi_results if 'keyint=250)' in r['name'] and 'CRF 0' not in r['name'] and not r.get('failed')), None)
        i_only_lossless = next((r for r in multi_results if 'I-only lossless' in r['name'] and not r.get('failed')), None)
        inter_lossless = next((r for r in multi_results if 'inter lossless' in r['name'] and not r.get('failed')), None)

        if i_only_lossy and inter_lossy:
            savings = i_only_lossy['compressed_bytes'] - inter_lossy['compressed_bytes']
            pct = savings / i_only_lossy['compressed_bytes'] * 100
            f.write(f"\n**Inter-frame savings (CRF 23):** {savings:,} bytes ({savings/1024/1024:.2f} MB) — "
                    f"inter-frame prediction reduces multi-frame encoding size by **{pct:.1f}%**.\n")

        if i_only_lossless and inter_lossless:
            savings = i_only_lossless['compressed_bytes'] - inter_lossless['compressed_bytes']
            pct = savings / i_only_lossless['compressed_bytes'] * 100
            f.write(f"\n**Inter-frame savings (lossless):** {savings:,} bytes ({savings/1024/1024:.2f} MB) — "
                    f"inter-frame prediction reduces lossless encoding by **{pct:.1f}%**.\n")

        # Algorithm contribution analysis
        f.write("\n## Algorithm Contribution Analysis\n\n")

        ffv1_r = next((r for r in single_results if 'FFV1' in r['name'] and not r.get('failed')), None)
        h265_lossless_r = next((r for r in single_results if 'Lossless H.265' in r['name'] and not r.get('failed')), None)
        h265_crf23_r = next((r for r in single_results if r['name'] == 'Full H.265 (CRF 23)' and not r.get('failed')), None)

        if ffv1_r and h265_lossless_r and h265_crf23_r:
            f.write("### Single-Frame Pipeline Breakdown (uint8 → H.265 CRF 23)\n\n")
            total_savings = total_uint8 - h265_crf23_r['compressed_bytes']

            entropy_savings = total_uint8 - ffv1_r['compressed_bytes']
            intra_savings = ffv1_r['compressed_bytes'] - h265_lossless_r['compressed_bytes']
            lossy_savings = h265_lossless_r['compressed_bytes'] - h265_crf23_r['compressed_bytes']

            f.write("| Component | Bytes Saved | MB Saved | % of Total Savings |\n")
            f.write("|-----------|-------------|---------|-------------------|\n")
            f.write(f"| Entropy coding (FFV1 baseline) | {entropy_savings:,} | "
                    f"{entropy_savings/1024/1024:.2f} | {entropy_savings/total_savings*100:.1f}% |\n")
            f.write(f"| Intra prediction (H.265 lossless vs FFV1) | {intra_savings:,} | "
                    f"{intra_savings/1024/1024:.2f} | {intra_savings/total_savings*100:.1f}% |\n")
            f.write(f"| Lossy quantization (CRF 23 vs lossless) | {lossy_savings:,} | "
                    f"{lossy_savings/1024/1024:.2f} | {lossy_savings/total_savings*100:.1f}% |\n")
            f.write(f"| **Total** | **{total_savings:,}** | **{total_savings/1024/1024:.2f}** | **100%** |\n")

        # SAO/deblock analysis
        no_sao_r = next((r for r in single_results if 'No SAO (CRF 23)' == r['name'] and not r.get('failed')), None)
        no_deblock_r = next((r for r in single_results if 'No Deblock (CRF 23)' == r['name'] and not r.get('failed')), None)
        no_both_r = next((r for r in single_results if 'No SAO + No Deblock' in r['name'] and not r.get('failed')), None)
        no_smooth_r = next((r for r in single_results if 'No Intra Smoothing' in r['name'] and not r.get('failed')), None)
        no_tskip_r = next((r for r in single_results if 'No Transform Skip' in r['name'] and not r.get('failed')), None)
        h264_r = next((r for r in single_results if 'H.264' in r['name'] and not r.get('failed')), None)

        if h265_crf23_r:
            f.write("\n### Flag Toggle Comparison (vs Full H.265 CRF 23)\n\n")
            f.write("| Config | Size (bytes) | Diff vs Full | Notes |\n")
            f.write("|--------|-------------|-------------|-------|\n")
            base = h265_crf23_r['compressed_bytes']
            f.write(f"| Full H.265 CRF 23 | {base:,} | — | baseline |\n")
            for name, r in [("No SAO", no_sao_r), ("No Deblock", no_deblock_r),
                            ("No SAO+Deblock", no_both_r), ("No Intra Smooth", no_smooth_r),
                            ("No Transform Skip", no_tskip_r), ("H.264", h264_r)]:
                if r:
                    diff = r['compressed_bytes'] - base
                    f.write(f"| {name} | {r['compressed_bytes']:,} | {diff:+,} | "
                            f"{'same' if abs(diff) < 100 else f'{diff/base*100:+.2f}%'} |\n")

        # Inter-frame contribution
        if i_only_lossy and inter_lossy:
            f.write("\n### Inter-Frame Contribution\n\n")
            f.write("| Component | Bytes Saved | MB Saved |\n")
            f.write("|-----------|-------------|----------|\n")
            savings = i_only_lossy['compressed_bytes'] - inter_lossy['compressed_bytes']
            f.write(f"| Inter-frame prediction (I-only → with P/B frames, CRF 23) | "
                    f"{savings:,} | {savings/1024/1024:.2f} |\n")
            f.write(f"| Reduction percentage | | "
                    f"{savings/i_only_lossy['compressed_bytes']*100:.1f}% of I-frame-only size |\n")

        if i_only_lossless and inter_lossless:
            savings = i_only_lossless['compressed_bytes'] - inter_lossless['compressed_bytes']
            f.write(f"| Inter-frame prediction (I-only → with P/B frames, lossless) | "
                    f"{savings:,} | {savings/1024/1024:.2f} |\n")
            f.write(f"| Reduction percentage | | "
                    f"{savings/i_only_lossless['compressed_bytes']*100:.1f}% of I-frame-only size |\n")

    print(f"\nResults written to: {out_path}")

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "algorithm_isolation_retrained.json")
    with open(json_path, 'w') as f:
        json.dump({
            'total_fp32': total_fp32,
            'total_uint8': total_uint8,
            'single_frame': single_results,
            'multi_frame': multi_results,
        }, f, indent=2, default=str)
    print(f"JSON saved to: {json_path}")

    print("\n" + "=" * 80)
    print("ALGORITHM ISOLATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
