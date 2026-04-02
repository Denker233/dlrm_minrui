#!/usr/bin/env python3
"""
Compare "one frame per table" (custom resolution) vs current "1080p multi-frame" approach
for H.265 and Zstd compression of DLRM cold embedding tables.

Approach:
- For each table, create a single large frame with width=1920 and variable height
  that fits all cold rows. For very large tables, use wider frames (3840, 7680, etc.)
- Encode with H.265 (CRF=30 medium no-deblock no-sao) and Zstd-3
- Compare compressed size, PSNR, max error vs 1080p multi-frame baseline
"""

import os, sys, time, json, subprocess, math, tempfile, struct
import numpy as np

os.chdir('/home/cc/expr/dlrm_minrui')
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')

import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import compressed_emb as _C
import zstandard as zstd

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
REORDER_DIR = "results/reorder"
ONDEMAND_DIR = "results/ondemand"
OUTPUT_DIR = "results/ondemand/single_frame"
EMB_DIM = 16
TILE_W = 4
TILE_H = 4

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]

# H.265 encoding parameters (matching 1080p_crf30_nofilter)
H265_CRF = 30
H265_PRESET = 'medium'
H265_EXTRA = 'no-deblock=1:no-sao=1'

# Zstd level
ZSTD_LEVEL = 3

# x265 max supported height depends on level. Level 5.1 supports up to 8192x4320.
# For very tall frames we may need --level-idc 8.5 (supports up to 16K).
# FFmpeg 4.4's x265 should handle large resolutions with appropriate flags.
MAX_HEIGHT_STANDARD = 8192  # Above this, try wider frame

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Helper: quantize embedding table (matching codec_ondemand_benchmark.py)
# ============================================================
def quantize_table(w):
    """Global quantization: single scale/zero-point for entire table."""
    mn = w.min().item()
    mx = w.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp

# ============================================================
# Helper: rows_to_tiled_frame using 4x4 tiles (matching existing approach)
# ============================================================
def rows_to_tiled_frame(emb_rows, width, height):
    """
    Convert embedding rows (N, 16) into a tiled 2D frame (height, width).
    Each embedding becomes a 4x4 tile. Uses C++ extension.
    """
    if isinstance(emb_rows, np.ndarray):
        emb_rows = torch.from_numpy(emb_rows)
    rows_per_frame = (width // TILE_W) * (height // TILE_H)
    if emb_rows.shape[0] < rows_per_frame:
        padded = torch.zeros(rows_per_frame, EMB_DIM, dtype=torch.uint8)
        padded[:emb_rows.shape[0]] = emb_rows
        emb_rows = padded
    frame = _C.tile_rows_to_frame(emb_rows, width, height)
    return frame.numpy()

def untile_frame(frame_np, width, height):
    """Convert tiled 2D frame back to embedding rows."""
    frame_t = torch.from_numpy(frame_np) if isinstance(frame_np, np.ndarray) else frame_np
    rows_per_frame = (width // TILE_W) * (height // TILE_H)
    rows = _C.untile_frame_to_rows(frame_t, rows_per_frame)
    return rows.numpy()

# ============================================================
# Choose frame dimensions for single-frame-per-table
# ============================================================
def compute_single_frame_dims(n_cold_rows, emb_dim=16):
    """
    Compute (width, height) for a single frame that fits all cold rows.

    With 4x4 tiling: each row occupies a 4x4 tile.
    tiles_per_row = width / 4, tiles_per_col = height / 4
    rows_per_frame = tiles_per_row * tiles_per_col = (width * height) / 16

    We need rows_per_frame >= n_cold_rows.
    Start with width=1920, compute height. If too tall, widen.
    Both width and height must be multiples of 4 (for tiling) and even.
    """
    for base_width in [1920, 3840, 7680, 15360]:
        tiles_per_row = base_width // TILE_W  # e.g., 480 for 1920
        needed_tile_rows = math.ceil(n_cold_rows / tiles_per_row)
        height = needed_tile_rows * TILE_H
        # Ensure even
        if height % 2 != 0:
            height += TILE_H  # add one more tile row (keeps multiple of 4)

        rows_per_frame = tiles_per_row * (height // TILE_H)

        if height <= 65536:  # x265 theoretical max with --level-idc 8.5
            return base_width, height, rows_per_frame

    # Fallback: shouldn't reach here for Kaggle tables
    raise ValueError(f"Cannot fit {n_cold_rows} rows in single frame")

# ============================================================
# Encode H.265 single frame
# ============================================================
def encode_h265_single_frame(q_np, width, height, output_path, crf=30, preset='medium', extra=''):
    """Encode a 2D uint8 frame as a single H.265 file.
    Uses a temp file for raw input to avoid pipe buffer issues with large frames."""
    frame_2d = rows_to_tiled_frame(q_np, width, height)

    # Write raw frame to temp file (avoids pipe buffer overflow for large frames)
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(frame_2d.tobytes())

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'gray',
        '-s', f'{width}x{height}',
        '-r', '1',
        '-i', tmp_path,
        '-c:v', 'libx265',
        '-preset', preset,
        '-pix_fmt', 'gray',
        '-x265-params',
        f'keyint=1:min-keyint=1:crf={crf}:log-level=error'
        + (f':{extra}' if extra else ''),
        '-f', 'matroska',
        output_path,
    ]

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, stderr = proc.communicate()
    encode_time = time.time() - t0

    os.unlink(tmp_path)

    if proc.returncode != 0:
        print(f"  WARNING: ffmpeg returned {proc.returncode}")
        print(f"  stderr: {stderr.decode()[:500]}")
        return 0, encode_time

    compressed_size = os.path.getsize(output_path)
    return compressed_size, encode_time

# ============================================================
# Decode H.265 single frame
# ============================================================
def decode_h265_single_frame(h265_path, width, height):
    """Decode an H.265 file back to a 2D frame."""
    cmd = [
        'ffmpeg', '-y',
        '-i', h265_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'gray',
        'pipe:1',
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_data, _ = proc.communicate()
    decode_time = time.time() - t0

    expected = width * height
    if len(raw_data) != expected:
        print(f"  WARNING: decoded {len(raw_data)} bytes, expected {expected}")
        # Pad or truncate
        if len(raw_data) < expected:
            raw_data = raw_data + b'\x00' * (expected - len(raw_data))
        else:
            raw_data = raw_data[:expected]

    frame = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width)
    return frame, decode_time

# ============================================================
# Encode/decode Zstd single frame (flat, no tiling)
# ============================================================
def encode_zstd_single(q_np, output_path, level=3):
    """Compress uint8 data with Zstd."""
    raw_bytes = q_np.tobytes()
    cctx = zstd.ZstdCompressor(level=level)
    t0 = time.time()
    compressed = cctx.compress(raw_bytes)
    encode_time = time.time() - t0

    with open(output_path, 'wb') as f:
        f.write(compressed)

    return len(compressed), encode_time

def decode_zstd_single(zstd_path, expected_size):
    """Decompress Zstd data."""
    dctx = zstd.ZstdDecompressor()
    t0 = time.time()
    with open(zstd_path, 'rb') as f:
        raw = dctx.decompress(f.read(), max_output_size=expected_size + 1024)
    decode_time = time.time() - t0
    return np.frombuffer(raw, dtype=np.uint8), decode_time

# ============================================================
# Compute quality metrics
# ============================================================
def compute_metrics(original_uint8, decoded_uint8, n_rows):
    """Compute PSNR, max error, mean error between original and decoded uint8 data."""
    orig = original_uint8[:n_rows].flatten().astype(np.float64)
    dec = decoded_uint8[:n_rows].flatten().astype(np.float64)

    diff = orig - dec
    mse = np.mean(diff ** 2)
    max_err = np.max(np.abs(diff))
    mean_err = np.mean(np.abs(diff))

    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * math.log10(255.0**2 / mse)

    return psnr, max_err, mean_err

# ============================================================
# Decode existing 1080p multi-frame H.265 for quality comparison
# ============================================================
def decode_1080p_h265(table_id, meta, ondemand_dir):
    """Decode all 1080p frames and reconstruct uint8 rows."""
    frame_dir = os.path.join(ondemand_dir, '1080p_crf30_nofilter', f'table_{table_id}')
    width, height = meta['width'], meta['height']
    rpf = meta['rows_per_frame']
    n_cold = meta['n_cold']
    num_frames = meta['num_frames']

    all_rows = []
    for i in range(num_frames):
        fpath = os.path.join(frame_dir, f'frame_{i:05d}.h265')
        frame, _ = decode_h265_single_frame(fpath, width, height)
        rows = untile_frame(frame, width, height)
        all_rows.append(rows)

    combined = np.concatenate(all_rows, axis=0)
    return combined[:n_cold]

# ============================================================
# Main experiment
# ============================================================
def main():
    print("=" * 80)
    print("EXPERIMENT: Single Frame Per Table vs 1080p Multi-Frame")
    print("=" * 80)

    # Load model to get cold embeddings
    print("\nLoading model...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = ld['state_dict'] if 'state_dict' in ld else ld

    # Find embedding keys
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda k: int(k.split('.')[1]))
    print(f"Found {len(emb_keys)} embedding tables")

    # Load existing 1080p baseline metadata
    baseline_h265 = {}
    baseline_zstd = {}
    for t in TABLES:
        with open(os.path.join(ONDEMAND_DIR, '1080p_crf30_nofilter', f'table_{t}', 'meta.json')) as f:
            baseline_h265[t] = json.load(f)
        with open(os.path.join(ONDEMAND_DIR, '1080p_zstd3', f'table_{t}', 'meta.json')) as f:
            baseline_zstd[t] = json.load(f)

    # Results storage
    results = {
        'single_frame_h265': {},
        'single_frame_zstd': {},
        'baseline_1080p_h265': {},
        'baseline_1080p_zstd': {},
    }

    print(f"\n{'='*80}")
    print(f"{'Table':>6} | {'n_cold':>12} | {'Config':>20} | {'Resolution':>14} | {'#Frm':>5} | "
          f"{'Compressed':>12} | {'Ratio':>8} | {'PSNR':>8} | {'MaxErr':>6} | {'MeanErr':>8} | {'EncTime':>8}")
    print(f"{'-'*6}-+-{'-'*12}-+-{'-'*20}-+-{'-'*14}-+-{'-'*5}-+-"
          f"{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")

    for t in TABLES:
        print(f"\n--- Table {t} ---")
        n_cold = int(open(os.path.join(REORDER_DIR, f'num_cold_{t}.txt')).read().strip())
        raw_bytes = n_cold * EMB_DIM

        # Load cold order and get quantized uint8 data
        cold_order = np.load(os.path.join(REORDER_DIR, f'cold_order_{t}.npy'))
        reordered_w = state_dict[emb_keys[t]][cold_order]
        q_uint8, scale, zp = quantize_table(reordered_w)
        q_np = q_uint8.numpy()

        # Compute single-frame dimensions
        sf_width, sf_height, sf_rpf = compute_single_frame_dims(n_cold)

        # --- 1. Single-frame H.265 ---
        sf_h265_dir = os.path.join(OUTPUT_DIR, 'h265', f'table_{t}')
        os.makedirs(sf_h265_dir, exist_ok=True)
        sf_h265_path = os.path.join(sf_h265_dir, 'frame_00000.h265')

        print(f"  Encoding single-frame H.265 ({sf_width}x{sf_height})...")
        sf_h265_size, sf_h265_enc_time = encode_h265_single_frame(
            q_np, sf_width, sf_height, sf_h265_path,
            crf=H265_CRF, preset=H265_PRESET, extra=H265_EXTRA)

        if sf_h265_size > 0:
            # Decode and measure quality
            decoded_frame, sf_h265_dec_time = decode_h265_single_frame(
                sf_h265_path, sf_width, sf_height)
            decoded_rows = untile_frame(decoded_frame, sf_width, sf_height)
            sf_h265_psnr, sf_h265_maxerr, sf_h265_meanerr = compute_metrics(
                q_np, decoded_rows, n_cold)
            sf_h265_ratio = raw_bytes / sf_h265_size
        else:
            sf_h265_psnr = sf_h265_maxerr = sf_h265_meanerr = 0
            sf_h265_ratio = 0
            sf_h265_dec_time = 0

        results['single_frame_h265'][t] = {
            'n_cold': n_cold, 'raw_bytes': raw_bytes,
            'width': sf_width, 'height': sf_height, 'rpf': sf_rpf,
            'num_frames': 1, 'compressed_bytes': sf_h265_size,
            'ratio': sf_h265_ratio, 'psnr': sf_h265_psnr,
            'max_err': sf_h265_maxerr, 'mean_err': sf_h265_meanerr,
            'encode_time': sf_h265_enc_time, 'decode_time': sf_h265_dec_time,
        }

        print(f"{'':>6} | {n_cold:>12,} | {'SF H.265':>20} | {sf_width}x{sf_height:>8} | {'1':>5} | "
              f"{sf_h265_size:>12,} | {sf_h265_ratio:>7.1f}x | {sf_h265_psnr:>7.1f} | "
              f"{sf_h265_maxerr:>5.0f} | {sf_h265_meanerr:>7.3f} | {sf_h265_enc_time:>7.1f}s")

        # --- 2. Single-frame Zstd ---
        sf_zstd_dir = os.path.join(OUTPUT_DIR, 'zstd3', f'table_{t}')
        os.makedirs(sf_zstd_dir, exist_ok=True)
        sf_zstd_path = os.path.join(sf_zstd_dir, 'frame_00000.zst')

        print(f"  Encoding single-frame Zstd-3 (flat)...")
        sf_zstd_size, sf_zstd_enc_time = encode_zstd_single(q_np, sf_zstd_path, level=ZSTD_LEVEL)

        # Zstd is lossless, but verify
        decoded_zstd, sf_zstd_dec_time = decode_zstd_single(sf_zstd_path, n_cold * EMB_DIM)
        decoded_zstd_rows = decoded_zstd.reshape(-1, EMB_DIM)
        sf_zstd_psnr, sf_zstd_maxerr, sf_zstd_meanerr = compute_metrics(
            q_np, decoded_zstd_rows, n_cold)
        sf_zstd_ratio = raw_bytes / sf_zstd_size

        results['single_frame_zstd'][t] = {
            'n_cold': n_cold, 'raw_bytes': raw_bytes,
            'num_frames': 1, 'compressed_bytes': sf_zstd_size,
            'ratio': sf_zstd_ratio, 'psnr': sf_zstd_psnr,
            'max_err': sf_zstd_maxerr, 'mean_err': sf_zstd_meanerr,
            'encode_time': sf_zstd_enc_time, 'decode_time': sf_zstd_dec_time,
        }

        print(f"{'':>6} | {n_cold:>12,} | {'SF Zstd-3':>20} | {'flat':>14} | {'1':>5} | "
              f"{sf_zstd_size:>12,} | {sf_zstd_ratio:>7.1f}x | {sf_zstd_psnr:>7.1f} | "
              f"{sf_zstd_maxerr:>5.0f} | {sf_zstd_meanerr:>7.3f} | {sf_zstd_enc_time:>7.1f}s")

        # --- 3. 1080p baseline H.265 (from meta.json) ---
        bl_h265 = baseline_h265[t]
        bl_h265_ratio = bl_h265['raw_bytes'] / bl_h265['compressed_bytes']

        # Decode 1080p multi-frame for quality comparison
        print(f"  Decoding 1080p multi-frame H.265 ({bl_h265['num_frames']} frames) for quality check...")
        decoded_1080p = decode_1080p_h265(t, bl_h265, ONDEMAND_DIR)
        bl_h265_psnr, bl_h265_maxerr, bl_h265_meanerr = compute_metrics(
            q_np, decoded_1080p, n_cold)

        results['baseline_1080p_h265'][t] = {
            'n_cold': n_cold, 'raw_bytes': bl_h265['raw_bytes'],
            'width': 1920, 'height': 1080,
            'num_frames': bl_h265['num_frames'],
            'compressed_bytes': bl_h265['compressed_bytes'],
            'ratio': bl_h265_ratio, 'psnr': bl_h265_psnr,
            'max_err': bl_h265_maxerr, 'mean_err': bl_h265_meanerr,
        }

        print(f"{'':>6} | {n_cold:>12,} | {'1080p H.265':>20} | {'1920x1080':>14} | "
              f"{bl_h265['num_frames']:>5} | {bl_h265['compressed_bytes']:>12,} | "
              f"{bl_h265_ratio:>7.1f}x | {bl_h265_psnr:>7.1f} | "
              f"{bl_h265_maxerr:>5.0f} | {bl_h265_meanerr:>7.3f} | {'N/A':>8}")

        # --- 4. 1080p baseline Zstd (from meta.json, lossless so skip decode) ---
        bl_zstd = baseline_zstd[t]
        bl_zstd_ratio = bl_zstd['raw_bytes'] / bl_zstd['compressed_bytes']

        results['baseline_1080p_zstd'][t] = {
            'n_cold': n_cold, 'raw_bytes': bl_zstd['raw_bytes'],
            'width': 1920, 'height': 1080,
            'num_frames': bl_zstd['num_frames'],
            'compressed_bytes': bl_zstd['compressed_bytes'],
            'ratio': bl_zstd_ratio, 'psnr': float('inf'),
            'max_err': 0, 'mean_err': 0,
        }

        print(f"{'':>6} | {n_cold:>12,} | {'1080p Zstd-3':>20} | {'1920x1080':>14} | "
              f"{bl_zstd['num_frames']:>5} | {bl_zstd['compressed_bytes']:>12,} | "
              f"{bl_zstd_ratio:>7.1f}x | {'inf':>8} | {'0':>6} | {'0.000':>8} | {'N/A':>8}")

        del reordered_w, q_uint8, q_np
        import gc; gc.collect()

    # ============================================================
    # Summary table
    # ============================================================
    print("\n" + "=" * 100)
    print("SUMMARY: Per-Table Comparison")
    print("=" * 100)

    # Header
    print(f"\n{'Table':>6} | {'n_cold':>12} | {'Codec':>8} | {'1080p Size':>12} | {'1080p Ratio':>12} | "
          f"{'SF Size':>12} | {'SF Ratio':>12} | {'Size Change':>12} | {'1080p PSNR':>10} | {'SF PSNR':>10}")
    print("-" * 130)

    total_1080p_h265 = 0
    total_sf_h265 = 0
    total_1080p_zstd = 0
    total_sf_zstd = 0
    total_raw = 0

    for t in TABLES:
        n_cold = results['single_frame_h265'][t]['n_cold']
        raw = results['single_frame_h265'][t]['raw_bytes']
        total_raw += raw

        # H.265
        b1 = results['baseline_1080p_h265'][t]['compressed_bytes']
        s1 = results['single_frame_h265'][t]['compressed_bytes']
        r1_bl = results['baseline_1080p_h265'][t]['ratio']
        r1_sf = results['single_frame_h265'][t]['ratio']
        p1_bl = results['baseline_1080p_h265'][t]['psnr']
        p1_sf = results['single_frame_h265'][t]['psnr']
        change_h265 = ((s1 - b1) / b1 * 100) if b1 > 0 else 0
        total_1080p_h265 += b1
        total_sf_h265 += s1

        sf_dims = f"{results['single_frame_h265'][t]['width']}x{results['single_frame_h265'][t]['height']}"

        p1_bl_str = f"{p1_bl:.1f}" if p1_bl != float('inf') else "inf"
        p1_sf_str = f"{p1_sf:.1f}" if p1_sf != float('inf') else "inf"

        print(f"{t:>6} | {n_cold:>12,} | {'H.265':>8} | {b1:>12,} | {r1_bl:>11.1f}x | "
              f"{s1:>12,} | {r1_sf:>11.1f}x | {change_h265:>+11.1f}% | {p1_bl_str:>10} | {p1_sf_str:>10}  [{sf_dims}]")

        # Zstd
        b2 = results['baseline_1080p_zstd'][t]['compressed_bytes']
        s2 = results['single_frame_zstd'][t]['compressed_bytes']
        r2_bl = results['baseline_1080p_zstd'][t]['ratio']
        r2_sf = results['single_frame_zstd'][t]['ratio']
        change_zstd = ((s2 - b2) / b2 * 100) if b2 > 0 else 0
        total_1080p_zstd += b2
        total_sf_zstd += s2

        print(f"{'':>6} | {'':>12} | {'Zstd-3':>8} | {b2:>12,} | {r2_bl:>11.1f}x | "
              f"{s2:>12,} | {r2_sf:>11.1f}x | {change_zstd:>+11.1f}% | {'inf':>10} | {'inf':>10}")

    print("-" * 130)

    # Totals
    h265_bl_ratio = total_raw / total_1080p_h265 if total_1080p_h265 > 0 else 0
    h265_sf_ratio = total_raw / total_sf_h265 if total_sf_h265 > 0 else 0
    h265_change = ((total_sf_h265 - total_1080p_h265) / total_1080p_h265 * 100) if total_1080p_h265 > 0 else 0

    zstd_bl_ratio = total_raw / total_1080p_zstd if total_1080p_zstd > 0 else 0
    zstd_sf_ratio = total_raw / total_sf_zstd if total_sf_zstd > 0 else 0
    zstd_change = ((total_sf_zstd - total_1080p_zstd) / total_1080p_zstd * 100) if total_1080p_zstd > 0 else 0

    print(f"{'TOTAL':>6} | {'':>12} | {'H.265':>8} | {total_1080p_h265:>12,} | {h265_bl_ratio:>11.1f}x | "
          f"{total_sf_h265:>12,} | {h265_sf_ratio:>11.1f}x | {h265_change:>+11.1f}% |")
    print(f"{'':>6} | {'':>12} | {'Zstd-3':>8} | {total_1080p_zstd:>12,} | {zstd_bl_ratio:>11.1f}x | "
          f"{total_sf_zstd:>12,} | {zstd_sf_ratio:>11.1f}x | {zstd_change:>+11.1f}% |")

    print(f"\nRaw uint8 total: {total_raw:,} bytes ({total_raw/1024/1024:.1f} MB)")
    print(f"Raw fp32 total:  {total_raw*4:,} bytes ({total_raw*4/1024/1024:.1f} MB)")

    # Final summary
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    print(f"\nH.265 (CRF={H265_CRF}, {H265_PRESET}, {H265_EXTRA}):")
    print(f"  1080p multi-frame: {total_1080p_h265:,} bytes ({total_1080p_h265/1024:.1f} KB), {h265_bl_ratio:.1f}x compression")
    print(f"  Single frame/table: {total_sf_h265:,} bytes ({total_sf_h265/1024:.1f} KB), {h265_sf_ratio:.1f}x compression")
    print(f"  Change: {h265_change:+.1f}% ({'larger' if h265_change > 0 else 'smaller'})")

    print(f"\nZstd-3 (lossless):")
    print(f"  1080p multi-frame: {total_1080p_zstd:,} bytes ({total_1080p_zstd/1024/1024:.1f} MB), {zstd_bl_ratio:.1f}x compression")
    print(f"  Single frame/table: {total_sf_zstd:,} bytes ({total_sf_zstd/1024/1024:.1f} MB), {zstd_sf_ratio:.1f}x compression")
    print(f"  Change: {zstd_change:+.1f}% ({'larger' if zstd_change > 0 else 'smaller'})")

    # Save results
    results_file = os.path.join(OUTPUT_DIR, 'comparison_results.json')
    # Convert inf to string for JSON
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [sanitize(x) for x in obj]
        return obj

    with open(results_file, 'w') as f:
        json.dump(sanitize(results), f, indent=2)
    print(f"\nResults saved to {results_file}")

if __name__ == '__main__':
    main()
