#!/usr/bin/env python3
"""
Profile each step of the H.265 codec pipeline for DLRM embeddings.
Breaks down: reorder, quantize, pad, tile, encode, decode, untile, dequantize.
"""

import os, sys, time, json, subprocess, io, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import av

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
EMB_DIM = 16
TILE_H = 4
TILE_W = 4
H265_CRF = 0  # lossless

RESOLUTIONS = {
    '1080p': (1920, 1080),
    '4K':    (3840, 2160),
}

HOTCOLD_DIR = "results/hotcold"
REORDER_DIR = "results/reorder"
ONDEMAND_DIR = "results/ondemand"

# ============================================================
# TILING FUNCTIONS
# ============================================================
def rows_to_tiled_frame(emb_rows, width, height):
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    tiles = emb_rows[:rows_per_frame].reshape(tiles_per_col, tiles_per_row, TILE_H, TILE_W)
    frame = tiles.transpose(0, 2, 1, 3).reshape(height, width)
    return frame

def tiled_frame_to_rows(frame, width, height):
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    grid = frame.reshape(tiles_per_col, TILE_H, tiles_per_row, TILE_W)
    rows = grid.transpose(0, 2, 1, 3).reshape(rows_per_frame, EMB_DIM)
    return rows


def profile_table(table_id, state_dict, emb_keys, res_name):
    """Profile the full encode/decode pipeline for one table at one resolution."""
    width, height = RESOLUTIONS[res_name]
    pixels_per_frame = width * height
    rows_per_frame = pixels_per_frame // EMB_DIM

    print(f"\n{'='*70}")
    print(f"TABLE {table_id} — {res_name} ({width}x{height}), {rows_per_frame} rows/frame")
    print(f"{'='*70}")

    # ---- Step 0: Load raw data ----
    t0 = time.time()
    cold_indices = torch.where(
        ~torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{table_id}.pt'),
                     map_location='cpu', weights_only=True)
    )[0]
    cold_w = state_dict[emb_keys[table_id]][cold_indices]  # (n_cold, 16) fp32
    cold_order = np.load(os.path.join(REORDER_DIR, f'cold_order_{table_id}.npy'))
    t_load = time.time() - t0

    n_cold = len(cold_indices)
    num_frames = max(1, (n_cold + rows_per_frame - 1) // rows_per_frame)
    raw_fp32_mb = n_cold * EMB_DIM * 4 / 1024 / 1024
    raw_uint8_mb = n_cold * EMB_DIM / 1024 / 1024

    print(f"  Cold rows: {n_cold:,}, Frames: {num_frames}")
    print(f"  Raw size: {raw_fp32_mb:.1f}MB fp32, {raw_uint8_mb:.1f}MB uint8")
    print()

    # ---- Step 1: Reorder (batch-affinity) ----
    t0 = time.time()
    reordered_w = cold_w[cold_order]
    t_reorder = time.time() - t0

    # ---- Step 2: Global quantization (fp32 -> uint8) ----
    t0 = time.time()
    mn = reordered_w.min().item()
    mx = reordered_w.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((reordered_w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    t_quantize = time.time() - t0

    q_np = q.numpy()

    # ---- Step 3: Pad to fill complete frames ----
    t0 = time.time()
    padded_rows = num_frames * rows_per_frame
    padded = np.zeros((padded_rows, EMB_DIM), dtype=np.uint8)
    padded[:n_cold] = q_np
    t_pad = time.time() - t0

    # ---- Step 4: Tile all frames (rows -> 4x4 spatial) ----
    t0 = time.time()
    tiled_frames = []
    for i in range(num_frames):
        frame_rows = padded[i * rows_per_frame:(i + 1) * rows_per_frame]
        frame_2d = rows_to_tiled_frame(frame_rows, width, height)
        tiled_frames.append(frame_2d)
    t_tile = time.time() - t0

    # ---- Step 5: H.265 encode all frames (ffmpeg subprocess) ----
    t0 = time.time()
    compressed_data = []
    for i, frame_2d in enumerate(tiled_frames):
        with tempfile.NamedTemporaryFile(suffix='.h265', delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', f'{width}x{height}',
            '-r', '1',
            '-i', 'pipe:0',
            '-c:v', 'libx265',
            '-preset', 'ultrafast',
            '-pix_fmt', 'gray',
            '-x265-params',
            f'keyint=1:min-keyint=1:{"lossless=1" if H265_CRF == 0 else f"crf={H265_CRF}"}:log-level=error',
            '-f', 'matroska',
            tmp_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.stdin.write(frame_2d.tobytes())
        proc.stdin.close()
        proc.wait()
        with open(tmp_path, 'rb') as f:
            compressed_data.append(f.read())
        os.unlink(tmp_path)
    t_encode = time.time() - t0

    compressed_mb = sum(len(d) for d in compressed_data) / 1024 / 1024

    # ---- Step 6: H.265 decode all frames (PyAV from memory) ----
    t0 = time.time()
    decoded_frames = []
    for data in compressed_data:
        container = av.open(io.BytesIO(data))
        frame = next(container.decode(video=0))
        arr = frame.to_ndarray(format='gray')  # (height, width) uint8
        container.close()
        decoded_frames.append(arr)
    t_decode = time.time() - t0

    # ---- Step 7: Untile all frames (4x4 spatial -> rows) ----
    t0 = time.time()
    untiled_rows = []
    for arr in decoded_frames:
        rows = tiled_frame_to_rows(arr, width, height)
        untiled_rows.append(rows)
    t_untile = time.time() - t0

    # ---- Step 8: Dequantize (uint8 -> fp32) ----
    t0 = time.time()
    for i, rows in enumerate(untiled_rows):
        row_start = i * rows_per_frame
        row_end = min(row_start + rows_per_frame, n_cold)
        actual = row_end - row_start
        fp32 = (rows[:actual].astype(np.float32) - zp) * s
    t_dequant = time.time() - t0

    # ---- Also profile per-frame times ----
    # Single frame encode
    t0 = time.time()
    for _ in range(3):
        with tempfile.NamedTemporaryFile(suffix='.h265', delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
            '-s', f'{width}x{height}', '-r', '1', '-i', 'pipe:0',
            '-c:v', 'libx265', '-preset', 'ultrafast', '-pix_fmt', 'gray',
            '-x265-params',
            f'keyint=1:min-keyint=1:{"lossless=1" if H265_CRF == 0 else f"crf={H265_CRF}"}:log-level=error',
            '-f', 'matroska', tmp_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.stdin.write(tiled_frames[0].tobytes())
        proc.stdin.close()
        proc.wait()
        os.unlink(tmp_path)
    t_enc_single = (time.time() - t0) / 3

    # Single frame decode
    t0 = time.time()
    for _ in range(10):
        container = av.open(io.BytesIO(compressed_data[0]))
        frame = next(container.decode(video=0))
        arr = frame.to_ndarray(format='gray')
        container.close()
    t_dec_single = (time.time() - t0) / 10

    # Single frame tile
    t0 = time.time()
    for _ in range(100):
        rows_to_tiled_frame(padded[:rows_per_frame], width, height)
    t_tile_single = (time.time() - t0) / 100

    # Single frame untile
    t0 = time.time()
    for _ in range(100):
        tiled_frame_to_rows(decoded_frames[0], width, height)
    t_untile_single = (time.time() - t0) / 100

    # Single frame dequant
    sample_rows = untiled_rows[0][:min(rows_per_frame, n_cold)]
    t0 = time.time()
    for _ in range(100):
        _ = (sample_rows.astype(np.float32) - zp) * s
    t_dequant_single = (time.time() - t0) / 100

    # ---- Print results ----
    total_encode_pipeline = t_reorder + t_quantize + t_pad + t_tile + t_encode
    total_decode_pipeline = t_decode + t_untile + t_dequant

    print(f"  ENCODE PIPELINE ({num_frames} frames total):")
    print(f"  {'Step':<30} {'Total':>10} {'Per Frame':>12} {'% of Encode':>12}")
    print(f"  {'-'*66}")
    for label, total, per_frame in [
        ("1. Reorder (batch-affinity)", t_reorder, t_reorder / num_frames),
        ("2. Quantize (fp32->uint8)", t_quantize, t_quantize / num_frames),
        ("3. Pad to frame size", t_pad, t_pad / num_frames),
        ("4. Tile (rows->4x4 spatial)", t_tile, t_tile_single),
        ("5. H.265 encode (ffmpeg)", t_encode, t_enc_single),
    ]:
        pct = total / total_encode_pipeline * 100
        if total >= 1:
            print(f"  {label:<30} {total:>9.3f}s {per_frame*1000:>10.2f}ms {pct:>10.1f}%")
        else:
            print(f"  {label:<30} {total*1000:>8.1f}ms {per_frame*1000:>10.2f}ms {pct:>10.1f}%")
    print(f"  {'TOTAL ENCODE':<30} {total_encode_pipeline:>9.3f}s")

    print()
    print(f"  DECODE PIPELINE ({num_frames} frames total):")
    print(f"  {'Step':<30} {'Total':>10} {'Per Frame':>12} {'% of Decode':>12}")
    print(f"  {'-'*66}")
    for label, total, per_frame in [
        ("6. H.265 decode (PyAV)", t_decode, t_dec_single),
        ("7. Untile (4x4->rows)", t_untile, t_untile_single),
        ("8. Dequantize (uint8->fp32)", t_dequant, t_dequant_single),
    ]:
        pct = total / total_decode_pipeline * 100
        if total >= 1:
            print(f"  {label:<30} {total:>9.3f}s {per_frame*1000:>10.2f}ms {pct:>10.1f}%")
        else:
            print(f"  {label:<30} {total*1000:>8.1f}ms {per_frame*1000:>10.2f}ms {pct:>10.1f}%")
    print(f"  {'TOTAL DECODE':<30} {total_decode_pipeline*1000:>8.1f}ms")

    print()
    print(f"  SIZES:")
    print(f"    fp32 original:  {raw_fp32_mb:>8.1f} MB")
    print(f"    uint8 quantized:{raw_uint8_mb:>8.1f} MB")
    print(f"    H.265 compressed:{compressed_mb:>7.1f} MB")
    print(f"    Compression: {raw_fp32_mb/compressed_mb:.1f}x fp32, "
          f"{raw_uint8_mb/compressed_mb:.1f}x uint8")

    print()
    print(f"  PER-FRAME DECODE LATENCY BREAKDOWN:")
    dec_total = t_dec_single + t_untile_single + t_dequant_single
    print(f"    H.265 decode:  {t_dec_single*1000:>8.2f}ms  ({t_dec_single/dec_total*100:>5.1f}%)")
    print(f"    Untile:        {t_untile_single*1000:>8.2f}ms  ({t_untile_single/dec_total*100:>5.1f}%)")
    print(f"    Dequantize:    {t_dequant_single*1000:>8.2f}ms  ({t_dequant_single/dec_total*100:>5.1f}%)")
    print(f"    TOTAL:         {dec_total*1000:>8.2f}ms")
    print(f"    Rows decoded:  {min(rows_per_frame, n_cold):,} rows in {dec_total*1000:.2f}ms "
          f"= {min(rows_per_frame, n_cold)/dec_total/1e6:.1f}M rows/s")

    return {
        'table_id': table_id, 'res': res_name,
        'n_cold': n_cold, 'num_frames': num_frames,
        't_reorder': t_reorder, 't_quantize': t_quantize,
        't_pad': t_pad, 't_tile': t_tile, 't_encode': t_encode,
        't_decode': t_decode, 't_untile': t_untile, 't_dequant': t_dequant,
        't_enc_single': t_enc_single, 't_dec_single': t_dec_single,
        't_tile_single': t_tile_single, 't_untile_single': t_untile_single,
        't_dequant_single': t_dequant_single,
        'compressed_mb': compressed_mb,
    }


def main():
    print("Loading model checkpoint...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = ld['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    print(f"Loaded {len(emb_keys)} embedding tables")

    # Profile the largest table (table 2: 9.6M cold rows) and a medium one (table 3: 2.2M)
    large_tables = [2, 3]

    for res_name in ['4K']:
        for t in large_tables:
            profile_table(t, state_dict, emb_keys, res_name)


if __name__ == '__main__':
    main()
