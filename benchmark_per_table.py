#!/usr/bin/env python3
"""
Per-table compression analysis: understand why compression ratios vary.

For each of the 8 large tables, measure:
- Table size (rows, bytes)
- Data characteristics (entropy, value distribution, unique values)
- Compression ratio with each compressor
- How table properties correlate with compressibility
"""

import os, sys, time, json
import numpy as np
import torch
import zstandard as zstd
import lz4.frame

sys.path.insert(0, '.')

RESULTS_DIR = "results"
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "codec_comparison")

MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
EMB_DIM = 16
ROWS_PER_FRAME = 129600
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def byte_entropy(data_bytes):
    """Compute Shannon entropy of byte distribution."""
    counts = np.bincount(np.frombuffer(data_bytes, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def main():
    log("Loading model state dict...")
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']

    try:
        import compressed_emb as _C
        has_h265 = True
    except ImportError:
        has_h265 = False

    results = {}

    for t_idx in LARGE_TABLES:
        log(f"\n{'='*60}")
        log(f"TABLE {t_idx}")
        log(f"{'='*60}")

        cold_idx_path = os.path.join(HOTCOLD_DIR, f"cold_indices_{t_idx}.pt")
        if not os.path.exists(cold_idx_path):
            continue
        cold_idx = torch.load(cold_idx_path, weights_only=False)
        if cold_idx.numel() == 0:
            continue

        key = f'emb_l.{t_idx}.weight'
        weight = sd[key]
        n_total = weight.shape[0]
        cold_weight = weight[cold_idx]
        n_cold = len(cold_idx)

        # Quantize
        mn, mx = cold_weight.min().item(), cold_weight.max().item()
        scale = (mx - mn) / 255.0
        if scale == 0: scale = 1.0
        zp = round(-mn / scale)
        quant = ((cold_weight / scale).round() + zp).clamp(0, 255).to(torch.uint8).numpy()

        # Apply reordering
        order_path = os.path.join(REORDER_DIR, f"cold_order_{t_idx}.npy")
        if os.path.exists(order_path):
            order = np.load(order_path)
            quant = quant[order]

        raw_bytes = quant.tobytes()

        # Data characteristics
        entropy = byte_entropy(raw_bytes)
        unique_vals = len(np.unique(quant))
        mean_val = np.mean(quant.astype(np.float32))
        std_val = np.std(quant.astype(np.float32))
        zero_frac = np.mean(quant == zp)  # fraction near zero_point

        # Row-to-row similarity (mean absolute difference between consecutive rows)
        row_diff = np.mean(np.abs(quant[1:].astype(np.float32) - quant[:-1].astype(np.float32)))

        log(f"  Rows: {n_cold:,} cold / {n_total:,} total ({100*n_cold/n_total:.1f}% cold)")
        log(f"  Raw: {len(raw_bytes)/1e6:.1f}MB")
        log(f"  Entropy: {entropy:.2f} bits/byte (max=8.0)")
        log(f"  Unique values: {unique_vals}/256")
        log(f"  Mean: {mean_val:.1f}, Std: {std_val:.1f}, ZP: {zp}")
        log(f"  Zero-fraction: {zero_frac:.3f}")
        log(f"  Row-to-row diff: {row_diff:.2f}")

        # Compress with each method
        compressor_results = {}
        for comp_name, compress_fn in [
            ('LZ4', lambda d: lz4.frame.compress(d)),
            ('Zstd-3', lambda d: zstd.ZstdCompressor(level=3).compress(d)),
            ('Zstd-19', lambda d: zstd.ZstdCompressor(level=19).compress(d)),
        ]:
            # Compress in frames
            n_frames = (n_cold + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
            total_compressed = 0
            for i in range(n_frames):
                start = i * ROWS_PER_FRAME
                end = min(start + ROWS_PER_FRAME, n_cold)
                chunk = quant[start:end].tobytes()
                compressed = compress_fn(chunk)
                total_compressed += len(compressed)

            ratio = len(raw_bytes) / total_compressed
            compressor_results[comp_name] = {
                'compressed_bytes': total_compressed,
                'ratio': ratio,
            }
            log(f"  {comp_name:<8s}: {total_compressed/1e6:.1f}MB, {ratio:.1f}x")

        # H.265
        if has_h265:
            n_frames = (n_cold + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
            total_h265 = 0
            for i in range(n_frames):
                start = i * ROWS_PER_FRAME
                end = min(start + ROWS_PER_FRAME, n_cold)
                chunk = quant[start:end]
                if len(chunk) < ROWS_PER_FRAME:
                    padded = np.zeros((ROWS_PER_FRAME, EMB_DIM), dtype=np.uint8)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                frame_t = _C.tile_rows_to_frame(torch.from_numpy(chunk), 1920, 1080)
                tmp_path = f"/tmp/_pertable_{t_idx}_{i}.h265"
                _C.encode_h265_frame(frame_t, tmp_path, True)
                total_h265 += os.path.getsize(tmp_path)
                os.remove(tmp_path)

            ratio = len(raw_bytes) / total_h265
            compressor_results['H.265'] = {
                'compressed_bytes': total_h265,
                'ratio': ratio,
            }
            log(f"  {'H.265':<8s}: {total_h265/1e6:.1f}MB, {ratio:.1f}x")

        results[f'table_{t_idx}'] = {
            'table_idx': t_idx,
            'n_total': n_total,
            'n_cold': n_cold,
            'cold_fraction': n_cold / n_total,
            'raw_mb': len(raw_bytes) / 1e6,
            'entropy': entropy,
            'unique_values': unique_vals,
            'mean': float(mean_val),
            'std': float(std_val),
            'zero_point': zp,
            'zero_fraction': float(zero_frac),
            'row_diff': float(row_diff),
            'scale': scale,
            'compressors': compressor_results,
        }

    # ---- Summary ----
    log(f"\n{'='*60}")
    log("PER-TABLE SUMMARY")
    log(f"{'='*60}\n")

    log(f"  {'Table':>6s} | {'Rows':>10s} | {'MB':>6s} | {'Entropy':>8s} | "
        f"{'RowDiff':>8s} | {'LZ4':>6s} | {'Zstd-19':>8s} | {'H.265':>7s}")
    log(f"  {'-'*75}")

    for t_idx in LARGE_TABLES:
        key = f'table_{t_idx}'
        if key not in results:
            continue
        r = results[key]
        lz4_r = r['compressors'].get('LZ4', {}).get('ratio', 0)
        zstd_r = r['compressors'].get('Zstd-19', {}).get('ratio', 0)
        h265_r = r['compressors'].get('H.265', {}).get('ratio', 0)
        log(f"  {t_idx:>6d} | {r['n_cold']:>10,d} | {r['raw_mb']:>5.1f} | "
            f"{r['entropy']:>7.2f}b | {r['row_diff']:>7.2f} | "
            f"{lz4_r:>5.1f}x | {zstd_r:>6.1f}x | {h265_r:>5.1f}x")

    log(f"\n  Correlation: entropy vs Zstd-19 ratio:")
    entropies = []
    ratios = []
    for key, r in results.items():
        entropies.append(r['entropy'])
        ratios.append(r['compressors'].get('Zstd-19', {}).get('ratio', 0))
    corr = np.corrcoef(entropies, ratios)[0, 1]
    log(f"    r = {corr:.3f} (lower entropy → better compression)")

    # Save
    json_path = os.path.join(OUTPUT_DIR, 'per_table_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
