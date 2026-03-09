#!/usr/bin/env python3
"""
Measure the compression ratio improvement from frequency-based reordering
for each compressor. This demonstrates that reordering is the key contribution
and benefits ALL compressors equally.

For each compressor, measures:
- Compression ratio WITHOUT reordering (random/original order)
- Compression ratio WITH frequency-based reordering
- The improvement factor
"""

import os, sys, time, json
import numpy as np
import torch
import zstandard as zstd
import lz4.frame
import snappy

sys.path.insert(0, '.')

RESULTS_DIR = "results"
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "codec_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
EMB_DIM = 16
ROWS_PER_FRAME = 129600
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compress_data(data_bytes, comp_name):
    """Compress bytes with the given compressor."""
    if comp_name == 'LZ4':
        return lz4.frame.compress(data_bytes)
    elif comp_name == 'Snappy':
        return snappy.compress(data_bytes)
    elif comp_name.startswith('Zstd'):
        level = int(comp_name.split('-')[1])
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data_bytes)
    else:
        raise ValueError(f"Unknown compressor: {comp_name}")


def decompress_data(data_bytes, comp_name):
    if comp_name == 'LZ4':
        return lz4.frame.decompress(data_bytes)
    elif comp_name == 'Snappy':
        return snappy.decompress(data_bytes)
    elif comp_name.startswith('Zstd'):
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data_bytes)


def main():
    log("Loading model state dict...")
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']

    compressor_names = ['LZ4', 'Snappy', 'Zstd-3', 'Zstd-9', 'Zstd-19']

    # Try to add H.265 if the extension is available
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
            log(f"  Skipping: no cold indices")
            continue

        cold_idx = torch.load(cold_idx_path, weights_only=False)
        if cold_idx.numel() == 0:
            continue

        key = f'emb_l.{t_idx}.weight'
        weight = sd[key]
        cold_weight = weight[cold_idx]

        # Quantize
        mn, mx = cold_weight.min().item(), cold_weight.max().item()
        s = (mx - mn) / 255.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        quant_original = ((cold_weight / s).round() + zp).clamp(0, 255).to(torch.uint8).numpy()

        # Reordered version (frequency-sorted)
        order_path = os.path.join(REORDER_DIR, f"cold_order_{t_idx}.npy")
        if os.path.exists(order_path):
            order = np.load(order_path)
            quant_reordered = quant_original[order]
        else:
            log(f"  Warning: no reorder found, using original order")
            quant_reordered = quant_original

        # Random shuffled version (to show benefit vs random)
        rng = np.random.RandomState(42)
        random_perm = rng.permutation(len(quant_original))
        quant_random = quant_original[random_perm]

        n_rows = quant_original.shape[0]
        raw_bytes = n_rows * EMB_DIM

        log(f"  {n_rows:,} cold rows, {raw_bytes/1e6:.1f}MB uint8")

        # Split into frames for consistency
        n_frames = (n_rows + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME

        for comp_name in compressor_names:
            total_random = 0
            total_natural = 0
            total_reordered = 0

            for i in range(n_frames):
                start = i * ROWS_PER_FRAME
                end = min(start + ROWS_PER_FRAME, n_rows)

                chunk_rand = quant_random[start:end].tobytes()
                chunk_nat = quant_original[start:end].tobytes()
                chunk_reord = quant_reordered[start:end].tobytes()

                total_random += len(compress_data(chunk_rand, comp_name))
                total_natural += len(compress_data(chunk_nat, comp_name))
                total_reordered += len(compress_data(chunk_reord, comp_name))

            ratio_rand = raw_bytes / total_random
            ratio_nat = raw_bytes / total_natural
            ratio_reord = raw_bytes / total_reordered
            imp_vs_random = ratio_reord / ratio_rand
            imp_vs_natural = ratio_reord / ratio_nat

            result_key = f"table_{t_idx}_{comp_name}"
            results[result_key] = {
                'table': t_idx,
                'compressor': comp_name,
                'n_rows': n_rows,
                'raw_bytes': raw_bytes,
                'random_compressed': total_random,
                'natural_compressed': total_natural,
                'reordered_compressed': total_reordered,
                'ratio_random': ratio_rand,
                'ratio_natural': ratio_nat,
                'ratio_reordered': ratio_reord,
                'improvement_vs_random': imp_vs_random,
                'improvement_vs_natural': imp_vs_natural,
            }

            log(f"  {comp_name:<10s}: random={ratio_rand:.1f}x, natural={ratio_nat:.1f}x, "
                f"reordered={ratio_reord:.1f}x (vs_rand={imp_vs_random:.2f}x)")

        # H.265 comparison (encode full frames)
        if has_h265:
            total_random_h265 = 0
            total_natural_h265 = 0
            total_reordered_h265 = 0

            for i in range(n_frames):
                start = i * ROWS_PER_FRAME
                end = min(start + ROWS_PER_FRAME, n_rows)

                for label, quant in [('random', quant_random),
                                      ('natural', quant_original),
                                      ('reordered', quant_reordered)]:
                    chunk = quant[start:end]
                    if len(chunk) < ROWS_PER_FRAME:
                        padded = np.zeros((ROWS_PER_FRAME, EMB_DIM), dtype=np.uint8)
                        padded[:len(chunk)] = chunk
                        chunk = padded

                    # tile_rows_to_frame expects uint8 tensor
                    frame_t = _C.tile_rows_to_frame(
                        torch.from_numpy(chunk), 1920, 1080
                    )
                    tmp_path = f"/tmp/_reorder_test_{t_idx}_{i}_{label}.h265"
                    _C.encode_h265_frame(frame_t, tmp_path, True)
                    sz = os.path.getsize(tmp_path)
                    if label == 'random':
                        total_random_h265 += sz
                    elif label == 'natural':
                        total_natural_h265 += sz
                    else:
                        total_reordered_h265 += sz
                    os.remove(tmp_path)

            ratio_rand = raw_bytes / total_random_h265
            ratio_nat = raw_bytes / total_natural_h265
            ratio_reord = raw_bytes / total_reordered_h265
            imp_vs_random = ratio_reord / ratio_rand

            results[f"table_{t_idx}_H.265"] = {
                'table': t_idx,
                'compressor': 'H.265',
                'n_rows': n_rows,
                'raw_bytes': raw_bytes,
                'random_compressed': total_random_h265,
                'natural_compressed': total_natural_h265,
                'reordered_compressed': total_reordered_h265,
                'ratio_random': ratio_rand,
                'ratio_natural': ratio_nat,
                'ratio_reordered': ratio_reord,
                'improvement_vs_random': imp_vs_random,
                'improvement_vs_natural': ratio_reord / ratio_nat,
            }
            log(f"  {'H.265':<10s}: random={ratio_rand:.1f}x, natural={ratio_nat:.1f}x, "
                f"reordered={ratio_reord:.1f}x (vs_rand={imp_vs_random:.2f}x)")

    # ---- Summary across all tables ----
    log(f"\n{'='*60}")
    log("SUMMARY: AVERAGE IMPROVEMENT FROM REORDERING")
    log(f"{'='*60}\n")

    all_compressors = compressor_names + (['H.265'] if has_h265 else [])
    log(f"  {'Compressor':<12s} | {'Random':>8s} | {'Natural':>9s} | {'Reordered':>10s} | {'vs Random':>10s}")
    log(f"  {'-'*65}")

    summary = {}
    for comp_name in all_compressors:
        rand_ratios = []
        nat_ratios = []
        reord_ratios = []
        improvements = []
        for key, val in results.items():
            if val['compressor'] == comp_name:
                rand_ratios.append(val['ratio_random'])
                nat_ratios.append(val['ratio_natural'])
                reord_ratios.append(val['ratio_reordered'])
                improvements.append(val['improvement_vs_random'])
        if rand_ratios:
            avg_rand = np.mean(rand_ratios)
            avg_nat = np.mean(nat_ratios)
            avg_reord = np.mean(reord_ratios)
            avg_imp = np.mean(improvements)
            log(f"  {comp_name:<12s} | {avg_rand:>6.1f}x | {avg_nat:>7.1f}x | "
                f"{avg_reord:>8.1f}x | {avg_imp:>8.2f}x")
            summary[comp_name] = {
                'avg_ratio_random': float(avg_rand),
                'avg_ratio_natural': float(avg_nat),
                'avg_ratio_reordered': float(avg_reord),
                'avg_improvement_vs_random': float(avg_imp),
            }

    # Save
    all_data = {'per_table': results, 'summary': summary}
    json_path = os.path.join(OUTPUT_DIR, 'reorder_benefit.json')
    with open(json_path, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    log(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
