#!/usr/bin/env python3
"""
Experiment: Finding optimal CRF / codec combination for sorted H.265 embeddings.

Goal: Get both good compression AND good AUC with frequency-sorted data.

Configs tested:
  A. Sorted CRF=30 (existing baseline)            — 319 KB, AUC 0.802107
  B. Unsorted CRF=30 (existing baseline)           — 360 KB, AUC 0.802355
  C. Sorted CRF=18                                 — less lossy
  D. Sorted CRF=0 (lossless H.265)                 — no quality loss
  E. Sorted 2-frame split: top 10% CRF=0, rest CRF=30
  F. Sorted Zstd-3 (lossless)                      — compression only
  G. Unsorted Zstd-3 (lossless)                     — compression only

For C, D, E: measure compressed size + PSNR + AUC inference.
For F, G: measure compressed size only (lossless → AUC = uint8 baseline).
Also run uint8-only baseline (lossless decode) for AUC reference.
"""

import os, sys, time, json, gc, subprocess, tempfile, math
import numpy as np

os.chdir('/home/cc/expr/dlrm_minrui')
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')

import torch
import torch.nn as nn
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import compressed_emb as _C
from sklearn.metrics import roc_auc_score

from codec_ondemand_benchmark import (
    load_model_and_data, CompressedEmbeddingBag,
    HOTCOLD_DIR, REORDER_DIR, ONDEMAND_DIR,
    EMB_DIM, MODEL_PATH, TILE_H, TILE_W, TEST_BATCH_SIZE,
)

# ============================================================
# Configuration
# ============================================================
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
H265_PRESET = 'medium'
H265_EXTRA = 'no-deblock=1:no-sao=1'
SPLIT_TOP_FRACTION = 0.10  # Top 10% of cold rows get lossless encoding

OUTPUT_DIR = 'results/crf_quality_sweep'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Single-frame dimensions (from freq_sort_ablation)
SINGLE_FRAME_DIMS = {
    2:  (3840, 40400),
    3:  (1920, 17568),
    9:  (1920, 744),
    11: (3840, 33304),
    15: (1920, 43556),
    20: (1920, 56200),
    23: (1920, 2284),
    25: (1920, 1140),
}

# Existing baseline results (from freq_sort_ablation/full_results.json)
EXISTING_RESULTS = {
    'A_sorted_crf30': {
        'total_compressed': 319361,
        'auc': 0.8021072220774818,
        'baseline_auc': 0.8024972457390324,
    },
    'B_unsorted_crf30': {
        'total_compressed': 360443,
        'auc': 0.8023545863118972,
        'baseline_auc': 0.8024972457390324,
    },
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ============================================================
# Helpers
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


def rows_to_tiled_frame(emb_rows, width, height):
    """Convert embedding rows (N, 16) into a tiled 2D frame (height, width)."""
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


def encode_h265_single_frame(q_np, width, height, output_path, crf=30):
    """Encode uint8 rows as a single H.265 frame with specified CRF."""
    frame_2d = rows_to_tiled_frame(q_np, width, height)

    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(frame_2d.tobytes())

    x265_params = f'keyint=1:min-keyint=1:crf={crf}:log-level=error'
    if crf == 0:
        # Lossless mode
        x265_params = f'keyint=1:min-keyint=1:lossless=1:log-level=error'
    if H265_EXTRA:
        x265_params += f':{H265_EXTRA}'

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'gray',
        '-s', f'{width}x{height}',
        '-r', '1',
        '-i', tmp_path,
        '-c:v', 'libx265',
        '-preset', H265_PRESET,
        '-pix_fmt', 'gray',
        '-x265-params', x265_params,
        '-f', 'matroska',
        output_path,
    ]

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, stderr = proc.communicate()
    encode_time = time.time() - t0

    os.unlink(tmp_path)

    if proc.returncode != 0:
        log(f"  WARNING: ffmpeg returned {proc.returncode}")
        log(f"  stderr: {stderr.decode()[:500]}")
        return 0, encode_time

    compressed_size = os.path.getsize(output_path)
    return compressed_size, encode_time


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
        log(f"  WARNING: decoded {len(raw_data)} bytes, expected {expected}")
        if len(raw_data) < expected:
            raw_data = raw_data + b'\x00' * (expected - len(raw_data))
        else:
            raw_data = raw_data[:expected]

    frame = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width)
    return frame, decode_time


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


def compute_frame_dims_for_nrows(n_rows):
    """Compute appropriate single-frame dimensions for a given number of rows.
    Uses 4x4 tiling. Width chosen to be 1920 or 3840 depending on size."""
    # Each row = 16 values = 4x4 tile
    # Rows per width: width // TILE_W
    # Rows per frame = (width // TILE_W) * (height // TILE_H)
    # Try width=1920 first
    for width in [1920, 3840]:
        cols = width // TILE_W  # tiles per row
        rows_needed = math.ceil(n_rows / cols)
        height = rows_needed * TILE_H
        # Round up height to be even (H.265 requires even)
        if height % 2 != 0:
            height += 1
        rpf = cols * (height // TILE_H)
        if rpf >= n_rows:
            return width, height
    # Fallback: use 3840 width
    width = 3840
    cols = width // TILE_W
    rows_needed = math.ceil(n_rows / cols)
    height = rows_needed * TILE_H
    if height % 2 != 0:
        height += 1
    return width, height


# ============================================================
# Phase 1: Compression experiments
# ============================================================
def run_compression_phase():
    log("=" * 80)
    log("PHASE 1: Compression & Quality Measurements")
    log("=" * 80)

    # Load model weights
    log("Loading model weights...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = ld['state_dict'] if 'state_dict' in ld else ld
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda k: int(k.split('.')[1]))

    # Load hot/cold masks
    is_hot = {}
    for t in LARGE_TABLES:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)

    results = {
        'C_sorted_crf18': {},
        'D_sorted_crf0': {},
        'E_sorted_split': {},
        'F_sorted_zstd3': {},
        'G_unsorted_zstd3': {},
    }

    # We need quant params and decoded data for AUC phase too
    quant_params = {}

    for t in LARGE_TABLES:
        log(f"\n{'='*60}")
        log(f"Table {t}")
        log(f"{'='*60}")

        w = state_dict[emb_keys[t]]
        n_total = w.shape[0]
        is_hot_t = is_hot[t][:n_total]

        # Cold indices
        cold_indices_natural = torch.where(~is_hot_t)[0]
        n_cold = len(cold_indices_natural)
        raw_bytes_uint8 = n_cold * EMB_DIM
        raw_bytes_fp32 = n_cold * EMB_DIM * 4

        # Load frequency-sorted cold order
        cold_order = np.load(os.path.join(REORDER_DIR, f'cold_order_{t}.npy'))

        # Frame dims for full table
        sf_width, sf_height = SINGLE_FRAME_DIMS[t]
        rpf = (sf_width // TILE_W) * (sf_height // TILE_H)

        log(f"  n_cold={n_cold:,}, frame={sf_width}x{sf_height}, rpf={rpf:,}")

        # Quantize sorted cold rows
        w_sorted = w[cold_order]
        q_sorted, s, zp = quantize_table(w_sorted)
        q_sorted_np = q_sorted.numpy()
        quant_params[t] = (s, zp)

        # Quantize unsorted cold rows (for Zstd comparison)
        w_unsorted = w[cold_indices_natural]
        q_unsorted, s_u, zp_u = quantize_table(w_unsorted)
        q_unsorted_np = q_unsorted.numpy()

        # ============================================================
        # Config C: Sorted CRF=18
        # ============================================================
        log(f"  [C] Sorted CRF=18...")
        c_dir = os.path.join(OUTPUT_DIR, 'C_sorted_crf18', f'table_{t}')
        os.makedirs(c_dir, exist_ok=True)
        c_path = os.path.join(c_dir, 'frame_00000.h265')

        c_size, c_enc = encode_h265_single_frame(q_sorted_np, sf_width, sf_height, c_path, crf=18)
        c_frame, c_dec = decode_h265_single_frame(c_path, sf_width, sf_height)
        c_rows = untile_frame(c_frame, sf_width, sf_height)
        c_psnr, c_maxerr, c_meanerr = compute_metrics(q_sorted_np, c_rows, n_cold)

        results['C_sorted_crf18'][str(t)] = {
            'n_cold': n_cold, 'raw_bytes': raw_bytes_uint8,
            'compressed_bytes': c_size,
            'ratio_uint8': raw_bytes_uint8 / c_size if c_size > 0 else 0,
            'ratio_fp32': raw_bytes_fp32 / c_size if c_size > 0 else 0,
            'psnr': c_psnr if not math.isinf(c_psnr) else 'inf',
            'max_err': float(c_maxerr), 'mean_err': float(c_meanerr),
            'encode_time': c_enc, 'decode_time': c_dec,
            'quant_scale': s, 'quant_zp': zp,
        }
        log(f"  [C] Size={c_size:>10,} B, Ratio={raw_bytes_uint8/c_size if c_size>0 else 0:.1f}x uint8, "
            f"PSNR={c_psnr:.1f}, MaxErr={c_maxerr:.0f}, MeanErr={c_meanerr:.4f}")

        # ============================================================
        # Config D: Sorted CRF=0 (lossless)
        # ============================================================
        log(f"  [D] Sorted CRF=0 (lossless)...")
        d_dir = os.path.join(OUTPUT_DIR, 'D_sorted_crf0', f'table_{t}')
        os.makedirs(d_dir, exist_ok=True)
        d_path = os.path.join(d_dir, 'frame_00000.h265')

        d_size, d_enc = encode_h265_single_frame(q_sorted_np, sf_width, sf_height, d_path, crf=0)
        d_frame, d_dec = decode_h265_single_frame(d_path, sf_width, sf_height)
        d_rows = untile_frame(d_frame, sf_width, sf_height)
        d_psnr, d_maxerr, d_meanerr = compute_metrics(q_sorted_np, d_rows, n_cold)

        results['D_sorted_crf0'][str(t)] = {
            'n_cold': n_cold, 'raw_bytes': raw_bytes_uint8,
            'compressed_bytes': d_size,
            'ratio_uint8': raw_bytes_uint8 / d_size if d_size > 0 else 0,
            'ratio_fp32': raw_bytes_fp32 / d_size if d_size > 0 else 0,
            'psnr': d_psnr if not math.isinf(d_psnr) else 'inf',
            'max_err': float(d_maxerr), 'mean_err': float(d_meanerr),
            'encode_time': d_enc, 'decode_time': d_dec,
            'quant_scale': s, 'quant_zp': zp,
        }
        log(f"  [D] Size={d_size:>10,} B, Ratio={raw_bytes_uint8/d_size if d_size>0 else 0:.1f}x uint8, "
            f"PSNR={'inf' if math.isinf(d_psnr) else f'{d_psnr:.1f}'}, MaxErr={d_maxerr:.0f}, MeanErr={d_meanerr:.4f}")

        # ============================================================
        # Config E: Sorted 2-frame split (top 10% CRF=0, rest CRF=30)
        # ============================================================
        log(f"  [E] Sorted 2-frame split (top 10% lossless, rest CRF=30)...")
        n_top = max(1, int(n_cold * SPLIT_TOP_FRACTION))
        n_bottom = n_cold - n_top

        e_dir = os.path.join(OUTPUT_DIR, 'E_sorted_split', f'table_{t}')
        os.makedirs(e_dir, exist_ok=True)

        # Top frame: first n_top rows (most accessed) at CRF=0
        q_top = q_sorted_np[:n_top]
        top_width, top_height = compute_frame_dims_for_nrows(n_top)
        top_rpf = (top_width // TILE_W) * (top_height // TILE_H)
        e_top_path = os.path.join(e_dir, 'frame_top.h265')
        e_top_size, e_top_enc = encode_h265_single_frame(q_top, top_width, top_height, e_top_path, crf=0)

        # Bottom frame: remaining rows at CRF=30
        q_bottom = q_sorted_np[n_top:n_cold]
        bottom_width, bottom_height = compute_frame_dims_for_nrows(n_bottom)
        bottom_rpf = (bottom_width // TILE_W) * (bottom_height // TILE_H)
        e_bottom_path = os.path.join(e_dir, 'frame_bottom.h265')
        e_bottom_size, e_bottom_enc = encode_h265_single_frame(q_bottom, bottom_width, bottom_height, e_bottom_path, crf=30)

        e_total_size = e_top_size + e_bottom_size

        # Decode both and merge for quality metrics
        e_top_frame, _ = decode_h265_single_frame(e_top_path, top_width, top_height)
        e_top_rows = untile_frame(e_top_frame, top_width, top_height)

        e_bottom_frame, _ = decode_h265_single_frame(e_bottom_path, bottom_width, bottom_height)
        e_bottom_rows = untile_frame(e_bottom_frame, bottom_width, bottom_height)

        # Merge: top rows + bottom rows
        e_merged = np.zeros((n_cold, EMB_DIM), dtype=np.uint8)
        e_merged[:n_top] = e_top_rows[:n_top]
        e_merged[n_top:] = e_bottom_rows[:n_bottom]

        e_psnr, e_maxerr, e_meanerr = compute_metrics(q_sorted_np[:n_cold], e_merged, n_cold)

        # Also measure quality on just the top 10% (should be lossless)
        e_top_psnr, e_top_maxerr, e_top_meanerr = compute_metrics(q_top, e_top_rows, n_top)
        # And bottom 90%
        e_bot_psnr, e_bot_maxerr, e_bot_meanerr = compute_metrics(
            q_sorted_np[n_top:n_cold], e_bottom_rows, n_bottom)

        results['E_sorted_split'][str(t)] = {
            'n_cold': n_cold, 'raw_bytes': raw_bytes_uint8,
            'n_top': n_top, 'n_bottom': n_bottom,
            'top_frame': f'{top_width}x{top_height}', 'bottom_frame': f'{bottom_width}x{bottom_height}',
            'top_compressed_bytes': e_top_size,
            'bottom_compressed_bytes': e_bottom_size,
            'total_compressed_bytes': e_total_size,
            'ratio_uint8': raw_bytes_uint8 / e_total_size if e_total_size > 0 else 0,
            'ratio_fp32': raw_bytes_fp32 / e_total_size if e_total_size > 0 else 0,
            'psnr_overall': e_psnr if not math.isinf(e_psnr) else 'inf',
            'max_err_overall': float(e_maxerr), 'mean_err_overall': float(e_meanerr),
            'psnr_top10': e_top_psnr if not math.isinf(e_top_psnr) else 'inf',
            'max_err_top10': float(e_top_maxerr), 'mean_err_top10': float(e_top_meanerr),
            'psnr_bottom90': e_bot_psnr if not math.isinf(e_bot_psnr) else 'inf',
            'max_err_bottom90': float(e_bot_maxerr), 'mean_err_bottom90': float(e_bot_meanerr),
            'encode_time_top': e_top_enc, 'encode_time_bottom': e_bottom_enc,
            'quant_scale': s, 'quant_zp': zp,
        }
        log(f"  [E] Top {n_top:,} rows: {e_top_size:,} B (CRF=0), "
            f"PSNR={'inf' if math.isinf(e_top_psnr) else f'{e_top_psnr:.1f}'}")
        log(f"  [E] Bottom {n_bottom:,} rows: {e_bottom_size:,} B (CRF=30), "
            f"PSNR={e_bot_psnr:.1f}")
        log(f"  [E] Total={e_total_size:>10,} B, Ratio={raw_bytes_uint8/e_total_size if e_total_size>0 else 0:.1f}x, "
            f"Overall PSNR={e_psnr:.1f}")

        # ============================================================
        # Config F: Sorted Zstd-3 (lossless)
        # ============================================================
        log(f"  [F] Sorted Zstd-3 (lossless)...")
        q_sorted_flat = torch.from_numpy(q_sorted_np[:n_cold].reshape(-1))
        f_compressed = _C.zstd_compress_frame(q_sorted_flat, 3)
        f_size = f_compressed.numel()

        results['F_sorted_zstd3'][str(t)] = {
            'n_cold': n_cold, 'raw_bytes': raw_bytes_uint8,
            'compressed_bytes': f_size,
            'ratio_uint8': raw_bytes_uint8 / f_size if f_size > 0 else 0,
            'ratio_fp32': raw_bytes_fp32 / f_size if f_size > 0 else 0,
            'psnr': 'inf', 'max_err': 0.0, 'mean_err': 0.0,
            'quant_scale': s, 'quant_zp': zp,
        }
        log(f"  [F] Size={f_size:>10,} B, Ratio={raw_bytes_uint8/f_size:.1f}x uint8 (lossless)")

        # ============================================================
        # Config G: Unsorted Zstd-3 (lossless)
        # ============================================================
        log(f"  [G] Unsorted Zstd-3 (lossless)...")
        q_unsorted_flat = torch.from_numpy(q_unsorted_np[:n_cold].reshape(-1))
        g_compressed = _C.zstd_compress_frame(q_unsorted_flat, 3)
        g_size = g_compressed.numel()

        results['G_unsorted_zstd3'][str(t)] = {
            'n_cold': n_cold, 'raw_bytes': raw_bytes_uint8,
            'compressed_bytes': g_size,
            'ratio_uint8': raw_bytes_uint8 / g_size if g_size > 0 else 0,
            'ratio_fp32': raw_bytes_fp32 / g_size if g_size > 0 else 0,
            'psnr': 'inf', 'max_err': 0.0, 'mean_err': 0.0,
            'quant_scale': s_u, 'quant_zp': zp_u,
        }
        log(f"  [G] Size={g_size:>10,} B, Ratio={raw_bytes_uint8/g_size:.1f}x uint8 (lossless)")

        # Sorting improvement for Zstd
        if g_size > 0:
            zstd_improvement = (g_size - f_size) / g_size * 100
            log(f"  >> Sorting impact on Zstd: {zstd_improvement:+.1f}%")

        del w_sorted, w_unsorted, q_sorted, q_unsorted
        gc.collect()

    # Save compression results
    comp_path = os.path.join(OUTPUT_DIR, 'compression_results.json')
    with open(comp_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nCompression results saved to {comp_path}")

    return results, quant_params


# ============================================================
# Phase 2: AUC experiments for configs C, D, E + uint8 baseline
# ============================================================
def run_auc_phase(comp_results, quant_params):
    log("\n" + "=" * 80)
    log("PHASE 2: AUC Inference")
    log("=" * 80)

    # Load model and data
    log("Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))

    # Pre-cache test batches
    log("Pre-caching test batches...")
    test_batches = []
    for X, lS_o, lS_i, T in test_ld:
        test_batches.append((X, lS_o, lS_i, T))
    log(f"Pre-cached {len(test_batches)} batches")

    # Load hot/cold data
    is_hot = {}
    hot_indices = {}
    for t in LARGE_TABLES:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]

    cold_num_rows = {}
    for t in LARGE_TABLES:
        with open(os.path.join(REORDER_DIR, f'num_cold_{t}.txt')) as f:
            cold_num_rows[t] = int(f.read().strip())

    # Load orig_to_cold_reordered for sorted configs
    orig_to_cold_reordered = {}
    for t in LARGE_TABLES:
        fp = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        orig_to_cold_reordered[t] = torch.load(fp, map_location='cpu', weights_only=True)

    num_tabs = len(dlrm.emb_l)
    orig_apply_emb = dlrm.apply_emb

    # ---- Helper: run inference ----
    def run_inference(tag):
        log(f"  Running inference for {tag}...")
        num_test_batches = len(test_batches)
        max_samples = num_test_batches * TEST_BATCH_SIZE + TEST_BATCH_SIZE
        all_scores = np.empty(max_samples, dtype=np.float32)
        all_targets = np.empty(max_samples, dtype=np.float32)
        sample_idx = 0

        t0 = time.time()
        with torch.no_grad():
            for batch_idx in range(num_test_batches):
                X, lS_o, lS_i, T = test_batches[batch_idx]
                Z = dlrm(X, lS_o, lS_i)
                z_np = Z.detach().cpu().numpy().ravel()
                t_np = T.detach().cpu().numpy().ravel()
                bs = z_np.shape[0]
                all_scores[sample_idx:sample_idx+bs] = z_np
                all_targets[sample_idx:sample_idx+bs] = t_np
                sample_idx += bs
                if batch_idx % 500 == 0:
                    log(f"    Batch {batch_idx}/{num_test_batches}")

        total_time = time.time() - t0
        auc = roc_auc_score(all_targets[:sample_idx], all_scores[:sample_idx])
        log(f"  {tag}: AUC = {auc:.6f}, Time = {total_time:.2f}s")
        return auc, total_time

    # ---- Helper: setup compressed inference and run ----
    def setup_compressed_and_run(tag, decode_fn, o2c_map_fn, quant_fn):
        log(f"\n{'='*60}")
        log(f"Config: {tag}")
        log(f"{'='*60}")

        # Restore original weights
        with torch.no_grad():
            for k in emb_keys:
                t_idx = int(k.split('.')[1])
                dlrm.emb_l[t_idx] = nn.EmbeddingBag(
                    ln_emb[t_idx], EMB_DIM, mode='sum', sparse=False)
                dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

        # Build CompressedEmbeddingBag for each large table
        for t_idx in LARGE_TABLES:
            n_cold = cold_num_rows.get(t_idx, 0)
            if n_cold == 0:
                continue

            w = state_dict[emb_keys[t_idx]]
            h_idx = hot_indices[t_idx]
            hot_weight = w[h_idx].clone()

            orig_to_hot = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
            orig_to_hot[h_idx] = torch.arange(len(h_idx))

            o2c = o2c_map_fn(t_idx)

            comp_emb = CompressedEmbeddingBag(
                hot_weight=hot_weight,
                is_hot=is_hot[t_idx],
                orig_to_hot=orig_to_hot,
                orig_to_cold_reordered=o2c,
                cold_cache=None,
                num_embeddings=ln_emb[t_idx],
                embedding_dim=EMB_DIM,
                quantize_hot=False,
            )
            dlrm.emb_l[t_idx] = comp_emb

        # Register tables in C++
        table_kinds = []
        weights = []
        mappings = []
        scales = []
        zero_points = []

        for k_idx in range(num_tabs):
            E = dlrm.emb_l[k_idx]
            if k_idx in cold_num_rows and isinstance(E, CompressedEmbeddingBag):
                table_kinds.append(1)  # COMPRESSED_FP32
                weights.append(E.hot_weight)
                mappings.append(E.mapping)
                scales.append(0.0)
                zero_points.append(0)
            else:
                table_kinds.append(0)  # STANDARD
                weights.append(E.weight)
                mappings.append(torch.empty(0, dtype=torch.int32))
                scales.append(0.0)
                zero_points.append(0)

        _C.register_tables(table_kinds, weights, mappings, scales, zero_points,
                           use_hash_table=False, use_bitmap=True)
        log("  Tables registered in C++ (bitmap mode)")

        # Decode and register cold frames
        log("  Decoding cold frames...")
        total_cold_mb = 0

        for t_idx in LARGE_TABLES:
            n_cold = cold_num_rows.get(t_idx, 0)
            if n_cold == 0:
                continue

            decoded_uint8 = decode_fn(t_idx)
            valid_data = decoded_uint8[:n_cold]
            if isinstance(valid_data, np.ndarray):
                valid_data = torch.from_numpy(valid_data)
            valid_cold_ranks = torch.arange(n_cold, dtype=torch.long)

            scale, zp = quant_fn(t_idx)
            _C.register_cold_sparse_flat(
                t_idx, valid_data, valid_cold_ranks,
                float(scale), float(zp), n_cold)

            cold_mb = valid_data.nbytes / 1024 / 1024
            total_cold_mb += cold_mb
            log(f"    Table {t_idx}: registered {valid_data.shape[0]:,} cold rows ({cold_mb:.1f}MB)")

            del decoded_uint8, valid_data
            gc.collect()

        log(f"  Total cold memory: {total_cold_mb:.1f}MB")

        # Setup fast_forward
        def _full_cpp_apply_emb(lS_o, lS_i, emb_l, v_W_l):
            if isinstance(lS_i, (list, tuple)):
                lS_i_2d = torch.stack(lS_i)
            elif lS_i.dim() == 2:
                lS_i_2d = lS_i
            else:
                lS_i_2d = lS_i.view(num_tabs, -1)
            if isinstance(lS_o, (list, tuple)):
                lS_o_2d = torch.stack(lS_o)
            elif lS_o.dim() == 2:
                lS_o_2d = lS_o
            else:
                lS_o_2d = lS_o.view(num_tabs, -1)
            results = _C.fast_forward(lS_i_2d, lS_o_2d)
            return results[-1]

        dlrm.apply_emb = _full_cpp_apply_emb
        auc, total_time = run_inference(tag)
        return {'auc': float(auc), 'time': float(total_time), 'cold_mb': float(total_cold_mb)}

    # ---- fp32 Baseline ----
    log(f"\n{'='*60}")
    log("Baseline: Full fp32")
    log(f"{'='*60}")
    with torch.no_grad():
        for k in emb_keys:
            t_idx = int(k.split('.')[1])
            dlrm.emb_l[t_idx] = nn.EmbeddingBag(
                ln_emb[t_idx], EMB_DIM, mode='sum', sparse=False)
            dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()
    dlrm.apply_emb = orig_apply_emb
    baseline_auc, baseline_time = run_inference("Baseline fp32")

    auc_results = {
        'baseline': {'auc': float(baseline_auc), 'time': float(baseline_time)}
    }

    # ---- Config C: Sorted CRF=18 ----
    def decode_C(t_idx):
        path = os.path.join(OUTPUT_DIR, 'C_sorted_crf18', f'table_{t_idx}', 'frame_00000.h265')
        w, h = SINGLE_FRAME_DIMS[t_idx]
        frame, _ = decode_h265_single_frame(path, w, h)
        return untile_frame(frame, w, h)

    def o2c_sorted(t_idx):
        return orig_to_cold_reordered[t_idx]

    def quant_sorted(t_idx):
        return quant_params[t_idx]

    dlrm.apply_emb = orig_apply_emb
    gc.collect()
    auc_results['C_sorted_crf18'] = setup_compressed_and_run(
        'C: Sorted CRF=18', decode_C, o2c_sorted, quant_sorted)

    # ---- Config D: Sorted CRF=0 (lossless) ----
    def decode_D(t_idx):
        path = os.path.join(OUTPUT_DIR, 'D_sorted_crf0', f'table_{t_idx}', 'frame_00000.h265')
        w, h = SINGLE_FRAME_DIMS[t_idx]
        frame, _ = decode_h265_single_frame(path, w, h)
        return untile_frame(frame, w, h)

    dlrm.apply_emb = orig_apply_emb
    gc.collect()
    auc_results['D_sorted_crf0'] = setup_compressed_and_run(
        'D: Sorted CRF=0 (lossless)', decode_D, o2c_sorted, quant_sorted)

    # ---- Config E: Sorted 2-frame split ----
    def decode_E(t_idx):
        """Decode both frames and merge."""
        e_dir = os.path.join(OUTPUT_DIR, 'E_sorted_split', f'table_{t_idx}')
        n_cold = cold_num_rows[t_idx]
        n_top = max(1, int(n_cold * SPLIT_TOP_FRACTION))
        n_bottom = n_cold - n_top

        # Decode top frame
        top_path = os.path.join(e_dir, 'frame_top.h265')
        top_info = comp_results['E_sorted_split'][str(t_idx)]
        tw, th = [int(x) for x in top_info['top_frame'].split('x')]
        top_frame, _ = decode_h265_single_frame(top_path, tw, th)
        top_rows = untile_frame(top_frame, tw, th)

        # Decode bottom frame
        bottom_path = os.path.join(e_dir, 'frame_bottom.h265')
        bw, bh = [int(x) for x in top_info['bottom_frame'].split('x')]
        bottom_frame, _ = decode_h265_single_frame(bottom_path, bw, bh)
        bottom_rows = untile_frame(bottom_frame, bw, bh)

        # Merge
        merged = np.zeros((n_cold, EMB_DIM), dtype=np.uint8)
        merged[:n_top] = top_rows[:n_top]
        merged[n_top:] = bottom_rows[:n_bottom]
        return merged

    dlrm.apply_emb = orig_apply_emb
    gc.collect()
    auc_results['E_sorted_split'] = setup_compressed_and_run(
        'E: Sorted split (top10% CRF=0, rest CRF=30)', decode_E, o2c_sorted, quant_sorted)

    # ---- uint8 baseline: lossless quantization only (use config D since it's lossless H.265) ----
    # Config D is lossless H.265, so decoded = original uint8. This gives us the
    # "uint8 quantization only" AUC (no codec error).
    # We already have this from config D. But let's verify by also testing with raw uint8.
    log(f"\n{'='*60}")
    log("uint8-only baseline (quantize + dequantize, no codec)")
    log(f"{'='*60}")

    def decode_uint8_only(t_idx):
        """Just quantize and return the uint8 data directly (no codec)."""
        cold_order = np.load(os.path.join(REORDER_DIR, f'cold_order_{t_idx}.npy'))
        w = state_dict[emb_keys[t_idx]]
        w_sorted = w[cold_order]
        q, _, _ = quantize_table(w_sorted)
        return q.numpy()

    dlrm.apply_emb = orig_apply_emb
    gc.collect()
    auc_results['uint8_only'] = setup_compressed_and_run(
        'uint8 quantization only (no codec)', decode_uint8_only, o2c_sorted, quant_sorted)

    return auc_results


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 80)
    log("EXPERIMENT: CRF Quality Sweep for Sorted H.265 Embeddings")
    log(f"Configs: C(CRF=18), D(CRF=0), E(split 10%/90%), F(Zstd sorted), G(Zstd unsorted)")
    log(f"H.265: preset={H265_PRESET}, {H265_EXTRA}")
    log("=" * 80)

    # Phase 1: Compression
    comp_results, quant_params = run_compression_phase()

    # Phase 2: AUC
    auc_results = run_auc_phase(comp_results, quant_params)

    # ============================================================
    # Final Summary
    # ============================================================
    log("\n" + "=" * 80)
    log("FINAL RESULTS: Compression Comparison")
    log("=" * 80)

    baseline_auc = auc_results['baseline']['auc']

    # Compute totals for each config
    configs = {
        'A_sorted_crf30': {'label': 'A: Sorted CRF=30', 'total_bytes': 319361, 'auc': 0.8021072220774818},
        'B_unsorted_crf30': {'label': 'B: Unsorted CRF=30', 'total_bytes': 360443, 'auc': 0.8023545863118972},
    }

    # Add new configs from our results
    for cfg_key in ['C_sorted_crf18', 'D_sorted_crf0', 'F_sorted_zstd3', 'G_unsorted_zstd3']:
        total = sum(comp_results[cfg_key][str(t)]['compressed_bytes'] for t in LARGE_TABLES)
        configs[cfg_key] = {'label': cfg_key.replace('_', ' ').replace('sorted', 'Sorted').replace('unsorted', 'Unsorted'),
                            'total_bytes': total}

    # Config E total
    e_total = sum(comp_results['E_sorted_split'][str(t)]['total_compressed_bytes'] for t in LARGE_TABLES)
    configs['E_sorted_split'] = {'label': 'E: Split (10% CRF=0 / 90% CRF=30)', 'total_bytes': e_total}

    # Add AUC for configs that have it
    for key in ['C_sorted_crf18', 'D_sorted_crf0', 'E_sorted_split']:
        if key in auc_results:
            configs[key]['auc'] = auc_results[key]['auc']
    if 'uint8_only' in auc_results:
        configs['uint8_only'] = {
            'label': 'uint8 quant only (no codec)',
            'total_bytes': 0,  # no compression
            'auc': auc_results['uint8_only']['auc'],
        }

    # Zstd is lossless, so AUC = uint8 baseline
    if 'uint8_only' in auc_results:
        for key in ['F_sorted_zstd3', 'G_unsorted_zstd3']:
            configs[key]['auc'] = auc_results['uint8_only']['auc']

    total_raw_uint8 = sum(comp_results['C_sorted_crf18'][str(t)]['raw_bytes'] for t in LARGE_TABLES)
    total_raw_fp32 = total_raw_uint8 * 4

    log(f"\n{'Config':<42} | {'Size':>10} | {'Ratio(u8)':>10} | {'Ratio(f32)':>10} | "
        f"{'AUC':>10} | {'AUC Loss':>12} | {'Loss %':>8}")
    log("-" * 120)

    # Sort by total bytes for nice display (but show uint8-only first)
    display_order = ['uint8_only', 'A_sorted_crf30', 'B_unsorted_crf30',
                     'C_sorted_crf18', 'D_sorted_crf0', 'E_sorted_split',
                     'F_sorted_zstd3', 'G_unsorted_zstd3']

    for key in display_order:
        if key not in configs:
            continue
        cfg = configs[key]
        total_bytes = cfg['total_bytes']
        if total_bytes > 0:
            ratio_u8 = total_raw_uint8 / total_bytes
            ratio_f32 = total_raw_fp32 / total_bytes
            size_str = f"{total_bytes/1024:.1f} KB"
            ratio_u8_str = f"{ratio_u8:.0f}x"
            ratio_f32_str = f"{ratio_f32:.0f}x"
        else:
            size_str = "N/A"
            ratio_u8_str = "N/A"
            ratio_f32_str = "N/A"

        if 'auc' in cfg:
            auc_val = cfg['auc']
            auc_loss = auc_val - baseline_auc
            loss_pct = auc_loss / baseline_auc * 100
            auc_str = f"{auc_val:.6f}"
            loss_str = f"{auc_loss:+.6f}"
            pct_str = f"{loss_pct:+.4f}%"
        else:
            auc_str = "N/A"
            loss_str = "N/A"
            pct_str = "N/A"

        log(f"{cfg['label']:<42} | {size_str:>10} | {ratio_u8_str:>10} | {ratio_f32_str:>10} | "
            f"{auc_str:>10} | {loss_str:>12} | {pct_str:>8}")

    log("-" * 120)
    log(f"Baseline fp32 AUC: {baseline_auc:.6f}")
    log(f"Total raw uint8: {total_raw_uint8:,} bytes ({total_raw_uint8/1024/1024:.1f} MB)")
    log(f"Total raw fp32:  {total_raw_fp32:,} bytes ({total_raw_fp32/1024/1024:.1f} MB)")

    # ============================================================
    # Per-table PSNR comparison
    # ============================================================
    log(f"\n{'='*80}")
    log("Per-table PSNR comparison")
    log(f"{'='*80}")
    log(f"{'Table':>6} | {'CRF=30':>10} | {'CRF=18':>10} | {'CRF=0':>10} | {'Split Overall':>14} | "
        f"{'Split Top10%':>13} | {'Split Bot90%':>13}")
    log("-" * 90)

    # Load CRF=30 PSNR from existing results
    with open('results/freq_sort_ablation/full_results.json') as f:
        old_results = json.load(f)

    for t in LARGE_TABLES:
        ts = str(t)
        crf30_psnr = old_results['compression']['sorted'][ts]['psnr']
        crf18_psnr = comp_results['C_sorted_crf18'][ts]['psnr']
        crf0_psnr = comp_results['D_sorted_crf0'][ts]['psnr']
        split_psnr = comp_results['E_sorted_split'][ts]['psnr_overall']
        split_top = comp_results['E_sorted_split'][ts]['psnr_top10']
        split_bot = comp_results['E_sorted_split'][ts]['psnr_bottom90']

        def fmt_psnr(p):
            return 'inf' if p == 'inf' or (isinstance(p, float) and math.isinf(p)) else f'{p:.1f}'

        log(f"{t:>6} | {fmt_psnr(crf30_psnr):>10} | {fmt_psnr(crf18_psnr):>10} | "
            f"{fmt_psnr(crf0_psnr):>10} | {fmt_psnr(split_psnr):>14} | "
            f"{fmt_psnr(split_top):>13} | {fmt_psnr(split_bot):>13}")

    # ============================================================
    # Zstd sorting impact
    # ============================================================
    log(f"\n{'='*80}")
    log("Zstd-3 sorting impact per table")
    log(f"{'='*80}")
    log(f"{'Table':>6} | {'Sorted':>12} | {'Unsorted':>12} | {'Improvement':>12}")
    log("-" * 50)

    total_f = 0
    total_g = 0
    for t in LARGE_TABLES:
        ts = str(t)
        f_size = comp_results['F_sorted_zstd3'][ts]['compressed_bytes']
        g_size = comp_results['G_unsorted_zstd3'][ts]['compressed_bytes']
        total_f += f_size
        total_g += g_size
        improvement = (g_size - f_size) / g_size * 100 if g_size > 0 else 0
        log(f"{t:>6} | {f_size:>10,} B | {g_size:>10,} B | {improvement:>+10.1f}%")

    log("-" * 50)
    total_imp = (total_g - total_f) / total_g * 100 if total_g > 0 else 0
    log(f"{'TOTAL':>6} | {total_f:>10,} B | {total_g:>10,} B | {total_imp:>+10.1f}%")

    # Save all results
    all_results = {
        'compression': comp_results,
        'auc': auc_results,
        'config': {
            'h265_preset': H265_PRESET,
            'h265_extra': H265_EXTRA,
            'split_top_fraction': SPLIT_TOP_FRACTION,
            'emb_dim': EMB_DIM,
            'tile_w': TILE_W, 'tile_h': TILE_H,
            'tables': LARGE_TABLES,
            'frame_dims': {str(k): list(v) for k, v in SINGLE_FRAME_DIMS.items()},
        },
        'summary': {
            'baseline_auc': float(baseline_auc),
            'total_raw_uint8': total_raw_uint8,
            'total_raw_fp32': total_raw_fp32,
        },
    }

    # Add per-config summary
    for key, cfg in configs.items():
        all_results['summary'][key] = {
            'total_compressed': cfg['total_bytes'],
            'auc': cfg.get('auc', None),
            'auc_loss': cfg.get('auc', baseline_auc) - baseline_auc if 'auc' in cfg else None,
        }

    results_path = os.path.join(OUTPUT_DIR, 'full_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nFull results saved to {results_path}")


if __name__ == '__main__':
    main()
