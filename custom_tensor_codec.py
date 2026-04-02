#!/usr/bin/env python3
"""
Custom Tensor Codec: DCT + Quantize + Zstd
Designed to fill the gap between Zstd (fast decode, low compression) and
H.265 (slow decode, high compression) for embedding table compression.

Key insight: H.265's compression comes from DCT+quantization (32x on top of uint8).
The slow parts (CABAC, intra prediction) contribute little compression but dominate decode.
We keep DCT+quantization and replace the rest with Zstd.
"""
import os, sys, time, json, struct, gc
import numpy as np
import zstandard as zstd
from scipy.fft import dctn, idctn
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

from codec_ondemand_benchmark import (
    load_model_and_data, EMB_DIM, MODEL_PATH,
    HOTCOLD_DIR, REORDER_DIR, quantize_table,
)

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
BLOCK = 8  # 8×8 DCT blocks

# ================================================================
# CODEC: Encode
# ================================================================
def tc_encode(uint8_data, step_size=16, zstd_level=3):
    """
    Encode uint8 embedding rows with DCT + quantize + Zstd.

    Args:
        uint8_data: (N, D) uint8 numpy array
        step_size: quantization step (higher = more compression, more loss)
        zstd_level: Zstd compression level

    Returns:
        compressed_bytes, metadata dict
    """
    n_rows, dim = uint8_data.shape

    # Flatten and pad to multiple of BLOCK*BLOCK
    flat = uint8_data.astype(np.float32).ravel()
    block_size = BLOCK * BLOCK
    n_total = len(flat)
    n_padded = ((n_total + block_size - 1) // block_size) * block_size
    if n_padded > n_total:
        flat = np.concatenate([flat, np.zeros(n_padded - n_total, dtype=np.float32)])

    n_blocks = n_padded // block_size
    blocks = flat.reshape(n_blocks, BLOCK, BLOCK)

    # DCT transform (vectorized over all blocks)
    dct_coeffs = dctn(blocks, axes=(-2, -1), type=2, norm='ortho')

    # Quantize: divide by step_size, round to int16
    quantized = np.round(dct_coeffs / step_size).astype(np.int16)

    # Zstd compress the quantized int16 array
    cctx = zstd.ZstdCompressor(level=zstd_level)
    compressed = cctx.compress(quantized.tobytes())

    meta = {
        'n_rows': n_rows, 'dim': dim,
        'n_blocks': n_blocks, 'n_padded': n_padded,
        'step_size': step_size, 'block_size': BLOCK,
    }

    # Pack: 4 bytes header (n_blocks) + compressed data
    header = struct.pack('<III', n_rows, dim, step_size)
    return header + compressed, meta


def tc_decode(compressed_bytes):
    """
    Decode: Zstd decompress + dequantize + inverse DCT.
    Returns (N, D) uint8 numpy array.
    """
    # Unpack header
    n_rows, dim, step_size = struct.unpack('<III', compressed_bytes[:12])
    payload = compressed_bytes[12:]

    block_size = BLOCK * BLOCK
    n_total = n_rows * dim
    n_padded = ((n_total + block_size - 1) // block_size) * block_size
    n_blocks = n_padded // block_size

    # Zstd decompress
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(payload)
    quantized = np.frombuffer(raw, dtype=np.int16).reshape(n_blocks, BLOCK, BLOCK)

    # Dequantize
    dct_coeffs = quantized.astype(np.float32) * step_size

    # Inverse DCT (vectorized)
    reconstructed = idctn(dct_coeffs, axes=(-2, -1), type=2, norm='ortho')

    # Flatten, truncate, clip, convert to uint8
    flat = reconstructed.ravel()[:n_total]
    result = np.clip(np.round(flat), 0, 255).astype(np.uint8)

    return result.reshape(n_rows, dim)


def tc_decode_timed(compressed_bytes):
    """Decode with timing breakdown."""
    n_rows, dim, step_size = struct.unpack('<III', compressed_bytes[:12])
    payload = compressed_bytes[12:]

    block_size = BLOCK * BLOCK
    n_total = n_rows * dim
    n_padded = ((n_total + block_size - 1) // block_size) * block_size
    n_blocks = n_padded // block_size

    t0 = time.perf_counter()
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(payload)
    quantized = np.frombuffer(raw, dtype=np.int16).reshape(n_blocks, BLOCK, BLOCK)
    t_zstd = time.perf_counter() - t0

    t0 = time.perf_counter()
    dct_coeffs = quantized.astype(np.float32) * step_size
    t_dequant = time.perf_counter() - t0

    t0 = time.perf_counter()
    reconstructed = idctn(dct_coeffs, axes=(-2, -1), type=2, norm='ortho')
    t_idct = time.perf_counter() - t0

    t0 = time.perf_counter()
    flat = reconstructed.ravel()[:n_total]
    result = np.clip(np.round(flat), 0, 255).astype(np.uint8).reshape(n_rows, dim)
    t_clip = time.perf_counter() - t0

    return result, {'zstd_ms': t_zstd*1000, 'dequant_ms': t_dequant*1000,
                    'idct_ms': t_idct*1000, 'clip_ms': t_clip*1000,
                    'total_ms': (t_zstd+t_dequant+t_idct+t_clip)*1000}


# ================================================================
# JPEG-style quantization matrix (optional, better quality/compression tradeoff)
# ================================================================
def jpeg_quant_matrix(quality=50):
    """Standard JPEG luminance quantization matrix scaled by quality."""
    base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68,109,103, 77],
        [24, 35, 55, 64, 81,104,113, 92],
        [49, 64, 78, 87,103,121,120,101],
        [72, 92, 95, 98,112,100,103, 99],
    ], dtype=np.float32)
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality
    qmat = np.floor((base * scale + 50) / 100).clip(1, 255)
    return qmat


def tc_encode_jpeg(uint8_data, quality=50, zstd_level=3):
    """Encode with JPEG-style quantization matrix (better high-freq suppression)."""
    n_rows, dim = uint8_data.shape
    flat = uint8_data.astype(np.float32).ravel()
    block_size = BLOCK * BLOCK
    n_total = len(flat)
    n_padded = ((n_total + block_size - 1) // block_size) * block_size
    if n_padded > n_total:
        flat = np.concatenate([flat, np.zeros(n_padded - n_total, dtype=np.float32)])

    n_blocks = n_padded // block_size
    blocks = flat.reshape(n_blocks, BLOCK, BLOCK)
    dct_coeffs = dctn(blocks, axes=(-2, -1), type=2, norm='ortho')

    qmat = jpeg_quant_matrix(quality)
    quantized = np.round(dct_coeffs / qmat[np.newaxis, :, :]).astype(np.int16)

    cctx = zstd.ZstdCompressor(level=zstd_level)
    compressed = cctx.compress(quantized.tobytes())

    header = struct.pack('<IIi', n_rows, dim, quality)
    return header + compressed, {'n_blocks': n_blocks, 'quality': quality}


def tc_decode_jpeg(compressed_bytes):
    """Decode JPEG-style quantized data."""
    n_rows, dim, quality = struct.unpack('<IIi', compressed_bytes[:12])
    payload = compressed_bytes[12:]

    block_size = BLOCK * BLOCK
    n_total = n_rows * dim
    n_padded = ((n_total + block_size - 1) // block_size) * block_size
    n_blocks = n_padded // block_size

    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(payload)
    quantized = np.frombuffer(raw, dtype=np.int16).reshape(n_blocks, BLOCK, BLOCK)

    qmat = jpeg_quant_matrix(quality)
    dct_coeffs = quantized.astype(np.float32) * qmat[np.newaxis, :, :]
    reconstructed = idctn(dct_coeffs, axes=(-2, -1), type=2, norm='ortho')

    flat = reconstructed.ravel()[:n_total]
    return np.clip(np.round(flat), 0, 255).astype(np.uint8).reshape(n_rows, dim)


# ================================================================
# Main experiment
# ================================================================
def main():
    print("=" * 70)
    print("CUSTOM TENSOR CODEC: DCT + Quantize + Zstd")
    print("=" * 70)

    # Load model and data
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    torch.set_num_threads(32)

    # Load cold data (frequency sorted)
    cold_data, cold_scale, cold_zp, cold_n = {}, {}, {}, {}
    for t in TABLES:
        cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
        w = sd[ek[t]][cold_order]
        q, s, zp = quantize_table(w)
        cold_data[t] = q.numpy()
        cold_scale[t] = s
        cold_zp[t] = zp
        cold_n[t] = len(cold_order)

    total_uint8 = sum(cold_data[t].size for t in TABLES)
    total_fp32 = total_uint8 * 4
    print(f"Total cold: {total_fp32/1024/1024:.0f}MB fp32, {total_uint8/1024/1024:.0f}MB uint8")

    # Baseline AUC
    is_hot = {}
    for t in TABLES:
        is_hot[t] = torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True)

    baseline_auc = None
    def compute_auc(cold_decoded_dict):
        """Compute AUC with decoded cold rows."""
        nonlocal baseline_auc
        with torch.no_grad():
            for k in ek:
                t_idx = int(k.split('.')[1])
                dlrm.emb_l[t_idx] = nn.EmbeddingBag(int(ln_emb[t_idx]), EMB_DIM,
                                                      mode='sum', sparse=True)
                w = sd[k].clone()
                if t_idx in cold_decoded_dict:
                    cold_order = np.load(f'{REORDER_DIR}/cold_order_{t_idx}.npy')
                    decoded = cold_decoded_dict[t_idx]
                    s, zp = cold_scale[t_idx], cold_zp[t_idx]
                    decoded_fp32 = (torch.from_numpy(decoded).float() - zp) * s
                    n = min(len(cold_order), decoded_fp32.shape[0])
                    w[cold_order[:n]] = decoded_fp32[:n]
                dlrm.emb_l[t_idx].weight.data = w

        scores, targets = [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                Z = dlrm(X, o, i)
                scores.append(Z.numpy().ravel())
                targets.append(T.numpy().ravel())
        return roc_auc_score(np.concatenate(targets), np.concatenate(scores))

    print("\nBaseline AUC (fp32)...")
    baseline_auc = compute_auc({})
    print(f"  Baseline: {baseline_auc:.6f}")

    print(f"\nUint8 AUC (no codec)...")
    auc_uint8 = compute_auc({t: cold_data[t] for t in TABLES})
    print(f"  Uint8: {auc_uint8:.6f} (loss={baseline_auc-auc_uint8:.6f})")

    # ================================================================
    # Part 1: Uniform step_size sweep
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 1: Uniform DCT quantization step sweep")
    print(f"{'='*70}")

    step_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print(f"\n{'Step':>5} {'Size':>10} {'R(uint8)':>10} {'R(fp32)':>10} "
          f"{'MaxErr':>7} {'MeanErr':>8} {'AUC':>10} {'Loss%':>8} "
          f"{'Decode':>10}")
    print("-" * 95)

    results_uniform = {}

    for step in step_sizes:
        total_comp = 0
        all_decoded = {}
        max_err_all = 0
        mean_err_sum = 0
        decode_ms_total = 0

        for t in TABLES:
            comp_bytes, meta = tc_encode(cold_data[t], step_size=step)
            total_comp += len(comp_bytes)

            # Decode with timing
            decoded, timing = tc_decode_timed(comp_bytes)
            all_decoded[t] = decoded
            decode_ms_total += timing['total_ms']

            # Error
            err = np.abs(decoded.astype(np.int16) - cold_data[t].astype(np.int16))
            max_err_all = max(max_err_all, err.max())
            mean_err_sum += err.mean() * cold_data[t].size

        mean_err = mean_err_sum / total_uint8
        ratio_uint8 = total_uint8 / total_comp
        ratio_fp32 = total_fp32 / total_comp

        auc = compute_auc(all_decoded)
        auc_loss = (baseline_auc - auc) * 100

        print(f"{step:>5} {total_comp/1024:>9.0f}KB {ratio_uint8:>9.0f}x {ratio_fp32:>9.0f}x "
              f"{max_err_all:>6.0f} {mean_err:>8.3f} {auc:>10.6f} {auc_loss:>+7.4f}% "
              f"{decode_ms_total:>8.0f}ms")

        results_uniform[step] = {
            'compressed_bytes': total_comp,
            'ratio_uint8': float(ratio_uint8),
            'ratio_fp32': float(ratio_fp32),
            'max_err': float(max_err_all),
            'mean_err': float(mean_err),
            'auc': float(auc),
            'auc_loss_pct': float(auc_loss),
            'decode_ms': float(decode_ms_total),
        }
        del all_decoded; gc.collect()

    # ================================================================
    # Part 2: JPEG-style quantization matrix sweep
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: JPEG-style quantization matrix sweep")
    print(f"{'='*70}")

    qualities = [95, 80, 50, 30, 10, 5, 1]

    print(f"\n{'Qual':>5} {'Size':>10} {'R(uint8)':>10} {'R(fp32)':>10} "
          f"{'MaxErr':>7} {'MeanErr':>8} {'AUC':>10} {'Loss%':>8}")
    print("-" * 80)

    results_jpeg = {}

    for q in qualities:
        total_comp = 0
        all_decoded = {}
        max_err_all = 0
        mean_err_sum = 0

        for t in TABLES:
            comp_bytes, meta = tc_encode_jpeg(cold_data[t], quality=q)
            total_comp += len(comp_bytes)
            decoded = tc_decode_jpeg(comp_bytes)
            all_decoded[t] = decoded

            err = np.abs(decoded.astype(np.int16) - cold_data[t].astype(np.int16))
            max_err_all = max(max_err_all, err.max())
            mean_err_sum += err.mean() * cold_data[t].size

        mean_err = mean_err_sum / total_uint8
        ratio_uint8 = total_uint8 / total_comp
        ratio_fp32 = total_fp32 / total_comp

        auc = compute_auc(all_decoded)
        auc_loss = (baseline_auc - auc) * 100

        print(f"{q:>5} {total_comp/1024:>9.0f}KB {ratio_uint8:>9.0f}x {ratio_fp32:>9.0f}x "
              f"{max_err_all:>6.0f} {mean_err:>8.3f} {auc:>10.6f} {auc_loss:>+7.4f}%")

        results_jpeg[q] = {
            'compressed_bytes': total_comp,
            'ratio_uint8': float(ratio_uint8),
            'ratio_fp32': float(ratio_fp32),
            'max_err': float(max_err_all),
            'mean_err': float(mean_err),
            'auc': float(auc),
            'auc_loss_pct': float(auc_loss),
        }
        del all_decoded; gc.collect()

    # ================================================================
    # Part 3: Decode speed comparison
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: Decode speed comparison (all 8 tables, ~492MB uint8)")
    print(f"{'='*70}")

    # Custom codec decode speed (step=16 and step=32)
    for step in [1, 8, 16, 32, 64]:
        encoded = {}
        for t in TABLES:
            comp, _ = tc_encode(cold_data[t], step_size=step)
            encoded[t] = comp

        # Warmup
        for t in TABLES:
            tc_decode(encoded[t])

        # Timed decode
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            for t in TABLES:
                tc_decode(encoded[t])
            times.append((time.perf_counter() - t0) * 1000)

        med = np.median(times)
        tp = total_uint8 / 1024 / 1024 / (med / 1000)  # MB/s
        comp_total = sum(len(v) for v in encoded.values())
        print(f"  Custom (step={step:>3}): {med:>7.0f}ms ({tp:>6.0f} MB/s) "
              f"  comp={comp_total/1024:.0f}KB ({total_fp32/comp_total:.0f}x)")

    # Plain Zstd (no DCT)
    zstd_encoded = {}
    cctx = zstd.ZstdCompressor(level=3)
    for t in TABLES:
        zstd_encoded[t] = cctx.compress(cold_data[t].tobytes())

    dctx_z = zstd.ZstdDecompressor()
    for t in TABLES:
        dctx_z.decompress(zstd_encoded[t])

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for t in TABLES:
            raw = dctx_z.decompress(zstd_encoded[t])
            np.frombuffer(raw, dtype=np.uint8)
        times.append((time.perf_counter() - t0) * 1000)
    med = np.median(times)
    tp = total_uint8 / 1024 / 1024 / (med / 1000)
    comp_total = sum(len(v) for v in zstd_encoded.values())
    print(f"  Zstd-3 (no DCT):   {med:>7.0f}ms ({tp:>6.0f} MB/s) "
          f"  comp={comp_total/1024:.0f}KB ({total_fp32/comp_total:.0f}x)")

    # ================================================================
    # Summary: Pareto comparison
    # ================================================================
    print(f"\n{'='*70}")
    print("PARETO COMPARISON: Custom Codec vs H.265 vs Zstd")
    print(f"{'='*70}")

    # H.265 numbers from previous experiments
    h265_points = {
        'H265 CRF=0': {'ratio_fp32': 27, 'auc_loss_pct': -0.0001},
        'H265 CRF=18': {'ratio_fp32': 232, 'auc_loss_pct': -0.0025},
        'H265 CRF=25': {'ratio_fp32': 2650, 'auc_loss_pct': -0.0070},
        'H265 CRF=30': {'ratio_fp32': 6466, 'auc_loss_pct': -0.0129},
        'H265 CRF=35': {'ratio_fp32': 9502, 'auc_loss_pct': -0.0208},
        'H265 CRF=51': {'ratio_fp32': 12437, 'auc_loss_pct': -0.0372},
    }

    print(f"\n{'Method':<25} {'Ratio(fp32)':>12} {'AUC Loss':>10} {'Decode':>12}")
    print("-" * 65)

    # Zstd baseline
    print(f"{'Zstd-3 (lossless)':<25} {'27x':>12} {'-0.049%':>10} {'~100ms':>12}")

    # Custom codec points
    for step, r in sorted(results_uniform.items()):
        label = f"Custom step={step}"
        print(f"{label:<25} {r['ratio_fp32']:>11.0f}x {r['auc_loss_pct']:>+9.4f}% "
              f"{r['decode_ms']:>10.0f}ms")

    print()
    # H.265 points
    for label, r in h265_points.items():
        decode = "72ms (1080p)" if 'CRF=30' in label else "—"
        print(f"{label:<25} {r['ratio_fp32']:>11.0f}x {r['auc_loss_pct']:>+9.4f}% "
              f"{decode:>12}")

    # CAFE+ reference
    print(f"{'CAFE+ (retrain)':<25} {'~10000x':>12} {'~-0.75%':>10} {'0ms':>12}")

    # Save results
    os.makedirs('results/custom_codec', exist_ok=True)
    all_results = {
        'uniform': results_uniform,
        'jpeg_style': {str(k): v for k, v in results_jpeg.items()},
        'baseline_auc': float(baseline_auc),
        'uint8_auc': float(auc_uint8),
        'total_uint8_bytes': total_uint8,
        'total_fp32_bytes': total_fp32,
    }
    with open('results/custom_codec/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/custom_codec/results.json")


if __name__ == '__main__':
    main()
