#!/usr/bin/env python3
"""
Test AV1-inspired improvements to DC block-mean approach.

1. Delta-code DC means along frequency-sorted order
2. Adaptive precision (fp16 semi-hot, uint8 deep-cold)
3. Prediction between tables (cross-table DC prediction)

Baseline: current DC approach (block mean, Zstd on means)
Metric: compression ratio + AUC for each variant.
"""
import os, sys, time, json, struct
import numpy as np
import torch
import zstandard as zstd
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
from sklearn.metrics import roc_auc_score
from codec_ondemand_benchmark import load_model_and_data, quantize_table, REORDER_DIR

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
EMB_DIM = 16
BLOCK_SIZE = 16  # dc16 mode: 1 mean per 16 rows, broadcast to all dims


def run_auc(dlrm, test_ld):
    all_s, all_l = [], []
    with torch.no_grad():
        for X, o, i, T in test_ld:
            Z = dlrm(X, o, i)
            all_s.append(Z.cpu().numpy().flatten())
            all_l.append(T.numpy().flatten())
    return roc_auc_score(np.concatenate(all_l), np.concatenate(all_s))


def compute_dc_means(cold_fp32, block_size=BLOCK_SIZE):
    """Compute block means. Returns (n_blocks,) float32 array."""
    n_rows = cold_fp32.shape[0]
    n_blocks = (n_rows + block_size - 1) // block_size
    # Pad to full blocks
    if n_rows % block_size != 0:
        padded = torch.zeros(n_blocks * block_size, EMB_DIM)
        padded[:n_rows] = cold_fp32
        cold_fp32 = padded
    # Reshape into blocks and compute mean across rows AND dims
    blocks = cold_fp32.reshape(n_blocks, block_size, EMB_DIM)
    means = blocks.mean(dim=(1, 2))  # (n_blocks,) — single mean per block
    return means


def reconstruct_from_dc(means, cold_order, orig_w, block_size=BLOCK_SIZE):
    """Reconstruct table: cold rows = block mean broadcast, hot rows = original."""
    n_cold = len(cold_order)
    n_blocks = len(means)
    recon = orig_w.clone()
    for bi in range(n_blocks):
        start = bi * block_size
        end = min(start + block_size, n_cold)
        row_indices = cold_order[start:end]
        recon[row_indices] = means[bi].item()
    return recon


def compress_zstd(data_bytes, level=3):
    """Zstd compress, return compressed bytes."""
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data_bytes)


def decompress_zstd(comp_bytes):
    """Zstd decompress."""
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(comp_bytes)


# ============================================================
# Compression strategies
# ============================================================

def strategy_baseline(means):
    """Baseline: raw fp32 means + Zstd-3."""
    raw = means.numpy().astype(np.float32).tobytes()
    comp = compress_zstd(raw, level=3)
    # Decompress to verify
    dec = np.frombuffer(decompress_zstd(comp), dtype=np.float32)
    return comp, torch.from_numpy(dec.copy()), 'baseline_fp32_zstd3'


def strategy_baseline_uint8(means):
    """Baseline: quantize means to uint8 + Zstd-3."""
    mn, mx = means.min().item(), means.max().item()
    s = (mx - mn) / 255.0 if mx != mn else 1.0
    zp = round(-mn / s) if s != 0 else 0
    zp = max(0, min(255, zp))
    q = ((means / s) + zp).round().clamp(0, 255).to(torch.uint8)
    raw = q.numpy().tobytes()
    comp = compress_zstd(raw, level=3)
    # Decompress + dequant
    dec_q = np.frombuffer(decompress_zstd(comp), dtype=np.uint8)
    dec = (torch.from_numpy(dec_q.copy()).float() - zp) * s
    # Store scale+zp as 8-byte header
    header = struct.pack('ff', s, float(zp))
    return header + comp, dec, 'baseline_uint8_zstd3'


# --- IDEA 1: Delta-code means ---

def strategy_delta_fp32(means):
    """Delta-code fp32 means + Zstd-3."""
    deltas = torch.zeros_like(means)
    deltas[0] = means[0]
    deltas[1:] = means[1:] - means[:-1]
    raw = deltas.numpy().astype(np.float32).tobytes()
    comp = compress_zstd(raw, level=3)
    # Decompress + cumsum
    dec_deltas = np.frombuffer(decompress_zstd(comp), dtype=np.float32)
    dec = torch.from_numpy(np.cumsum(dec_deltas).astype(np.float32).copy())
    return comp, dec, 'delta_fp32_zstd3'


def strategy_delta_uint8(means):
    """Quantize means to uint8, delta-code, Zstd-3."""
    mn, mx = means.min().item(), means.max().item()
    s = (mx - mn) / 255.0 if mx != mn else 1.0
    zp = round(-mn / s) if s != 0 else 0
    zp = max(0, min(255, zp))
    q = ((means / s) + zp).round().clamp(0, 255).to(torch.int16)
    deltas = torch.zeros_like(q)
    deltas[0] = q[0]
    deltas[1:] = q[1:] - q[:-1]  # int16 deltas (can be negative)
    # Pack as int8 if range fits, else int16
    d_np = deltas.numpy()
    if d_np.min() >= -128 and d_np.max() <= 127:
        raw = d_np.astype(np.int8).tobytes()
        dtype_flag = b'\x01'  # int8
    else:
        raw = d_np.astype(np.int16).tobytes()
        dtype_flag = b'\x02'  # int16
    comp = compress_zstd(raw, level=3)
    # Decompress + cumsum + dequant
    dec_raw = decompress_zstd(comp)
    if dtype_flag == b'\x01':
        dec_d = np.frombuffer(dec_raw, dtype=np.int8).astype(np.int64)
    else:
        dec_d = np.frombuffer(dec_raw, dtype=np.int16).astype(np.int64)
    dec_q = np.cumsum(dec_d).astype(np.float32)
    dec = (torch.from_numpy(dec_q.copy()) - zp) * s
    header = struct.pack('ff', s, float(zp)) + dtype_flag
    return header + comp, dec, 'delta_uint8_zstd3'


def strategy_delta2_uint8(means):
    """Double-delta (acceleration) on uint8 means + Zstd-3."""
    mn, mx = means.min().item(), means.max().item()
    s = (mx - mn) / 255.0 if mx != mn else 1.0
    zp = round(-mn / s) if s != 0 else 0
    zp = max(0, min(255, zp))
    q = ((means / s) + zp).round().clamp(0, 255).to(torch.int32)
    # First delta
    d1 = torch.zeros_like(q); d1[0] = q[0]; d1[1:] = q[1:] - q[:-1]
    # Second delta
    d2 = torch.zeros_like(d1); d2[0] = d1[0]; d2[1:] = d1[1:] - d1[:-1]
    raw = d2.numpy().astype(np.int8).tobytes()
    comp = compress_zstd(raw, level=3)
    # Decompress + double cumsum
    dec_d2 = np.frombuffer(decompress_zstd(comp), dtype=np.int8).astype(np.int64)
    dec_d1 = np.cumsum(dec_d2)
    dec_q = np.cumsum(dec_d1).astype(np.float32)
    dec = (torch.from_numpy(dec_q.copy()) - zp) * s
    header = struct.pack('ff', s, float(zp))
    return header + comp, dec, 'delta2_uint8_zstd3'


# --- IDEA 2: Adaptive precision ---

def strategy_adaptive_precision(means, hot_frac=0.20):
    """Top 20% cold means in fp16, bottom 80% in uint8, both Zstd-3."""
    n = len(means)
    n_semi = int(n * hot_frac)
    semi_hot = means[:n_semi]   # first rows = highest freq cold
    deep_cold = means[n_semi:]

    # Semi-hot: fp16
    semi_raw = semi_hot.numpy().astype(np.float16).tobytes()
    semi_comp = compress_zstd(semi_raw, level=3)

    # Deep-cold: uint8
    mn, mx = deep_cold.min().item(), deep_cold.max().item()
    s = (mx - mn) / 255.0 if mx != mn else 1.0
    zp = round(-mn / s) if s != 0 else 0
    zp = max(0, min(255, zp))
    q = ((deep_cold / s) + zp).round().clamp(0, 255).to(torch.uint8)
    deep_raw = q.numpy().tobytes()
    deep_comp = compress_zstd(deep_raw, level=3)

    total_comp = struct.pack('IIff', n_semi, n - n_semi, s, float(zp)) + semi_comp + deep_comp
    total_bytes = len(total_comp)

    # Reconstruct
    dec_semi = torch.from_numpy(
        np.frombuffer(decompress_zstd(semi_comp), dtype=np.float16).astype(np.float32).copy())
    dec_deep_q = np.frombuffer(decompress_zstd(deep_comp), dtype=np.uint8)
    dec_deep = (torch.from_numpy(dec_deep_q.copy()).float() - zp) * s
    dec = torch.cat([dec_semi, dec_deep])

    return total_comp, dec, f'adaptive_fp16top{int(hot_frac*100)}_uint8rest'


def strategy_adaptive_fp32_top(means, hot_frac=0.05):
    """Top 5% cold means in fp32, rest in uint8. Max quality for semi-hot."""
    n = len(means)
    n_semi = int(n * hot_frac)
    semi_hot = means[:n_semi]
    deep_cold = means[n_semi:]

    semi_comp = compress_zstd(semi_hot.numpy().astype(np.float32).tobytes(), level=3)

    mn, mx = deep_cold.min().item(), deep_cold.max().item()
    s = (mx - mn) / 255.0 if mx != mn else 1.0
    zp = round(-mn / s) if s != 0 else 0
    zp = max(0, min(255, zp))
    q = ((deep_cold / s) + zp).round().clamp(0, 255).to(torch.uint8)
    deep_comp = compress_zstd(q.numpy().tobytes(), level=3)

    total_comp = struct.pack('IIff', n_semi, n - n_semi, s, float(zp)) + semi_comp + deep_comp

    dec_semi = torch.from_numpy(
        np.frombuffer(decompress_zstd(semi_comp), dtype=np.float32).copy())
    dec_deep_q = np.frombuffer(decompress_zstd(deep_comp), dtype=np.uint8)
    dec_deep = (torch.from_numpy(dec_deep_q.copy()).float() - zp) * s
    dec = torch.cat([dec_semi, dec_deep])

    return total_comp, dec, f'adaptive_fp32top{int(hot_frac*100)}_uint8rest'


# --- IDEA 3: Cross-table prediction ---
# Predict table B's DC from table A's DC (requires correlated tables)
# We'll test: predict each table from the global mean of all tables

def compute_all_table_means(table_data):
    """Compute DC means for all tables, return dict."""
    all_means = {}
    for t in TABLES:
        td = table_data[t]
        all_means[t] = compute_dc_means(td['cold_fp32'])
    return all_means


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("DC BLOCK-MEAN: AV1-inspired improvements")
    print("=" * 70)

    dlrm, test_ld, _, ln_emb = load_model_and_data()
    state_dict = dlrm.state_dict()
    ek = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))

    print("Baseline AUC...")
    baseline_auc = run_auc(dlrm, test_ld)
    print(f"  Baseline: {baseline_auc:.6f}")

    # Load table data
    table_data = {}
    total_cold_fp32_bytes = 0
    for t in TABLES:
        w = state_dict[ek[t]].clone()
        co = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
        cold_fp32 = w[co]
        table_data[t] = {
            'w': w, 'co': co, 'cold_fp32': cold_fp32,
            'n_cold': len(co),
        }
        total_cold_fp32_bytes += len(co) * EMB_DIM * 4
        print(f"  table {t}: {len(co)} cold rows, {len(co)*EMB_DIM*4/1024/1024:.1f} MB fp32")

    total_cold_fp32_mb = total_cold_fp32_bytes / 1024**2
    print(f"\nTotal cold fp32: {total_cold_fp32_mb:.1f} MB")

    # Compute DC means
    all_means = {}
    total_means = 0
    for t in TABLES:
        means = compute_dc_means(table_data[t]['cold_fp32'])
        all_means[t] = means
        total_means += len(means)
        print(f"  table {t}: {len(means)} DC means "
              f"(range [{means.min():.4f}, {means.max():.4f}], std={means.std():.4f})")

    print(f"\nTotal DC means: {total_means} ({total_means*4/1024:.1f} KB raw fp32)")

    # ============================================================
    # Test each strategy
    # ============================================================
    strategies = [
        strategy_baseline,
        strategy_baseline_uint8,
        strategy_delta_fp32,
        strategy_delta_uint8,
        strategy_delta2_uint8,
        lambda m: strategy_adaptive_precision(m, 0.20),
        lambda m: strategy_adaptive_precision(m, 0.10),
        lambda m: strategy_adaptive_fp32_top(m, 0.05),
        lambda m: strategy_adaptive_fp32_top(m, 0.10),
    ]

    strategy_names = [
        'baseline_fp32_zstd3',
        'baseline_uint8_zstd3',
        'delta_fp32_zstd3',
        'delta_uint8_zstd3',
        'delta2_uint8_zstd3',
        'adaptive_fp16top20_uint8',
        'adaptive_fp16top10_uint8',
        'adaptive_fp32top5_uint8',
        'adaptive_fp32top10_uint8',
    ]

    results = []

    for si, strat_fn in enumerate(strategies):
        total_comp_bytes = 0
        reconstructed = {}

        for t in TABLES:
            comp_data, dec_means, name = strat_fn(all_means[t])
            total_comp_bytes += len(comp_data)

            # Reconstruct table from decoded means
            recon = reconstruct_from_dc(dec_means, table_data[t]['co'],
                                        table_data[t]['w'])
            reconstructed[t] = recon

        # Measure AUC
        for t in TABLES:
            dlrm.emb_l[t].weight.data = reconstructed[t]
        auc = run_auc(dlrm, test_ld)
        delta = auc - baseline_auc
        for t in TABLES:
            dlrm.emb_l[t].weight.data = table_data[t]['w']

        comp_kb = total_comp_bytes / 1024
        ratio_fp32 = total_cold_fp32_bytes / total_comp_bytes
        ratio_means = (total_means * 4) / total_comp_bytes

        result = {
            'name': strategy_names[si],
            'comp_bytes': total_comp_bytes,
            'comp_kb': comp_kb,
            'ratio_fp32': ratio_fp32,
            'ratio_means': ratio_means,
            'auc': auc,
            'delta': delta,
        }
        results.append(result)
        print(f"\n  {strategy_names[si]:>35}: {comp_kb:>8.1f} KB, "
              f"{ratio_fp32:>8.0f}x fp32, AUC {delta*100:+.4f}%")

    # ============================================================
    # Idea 3: Cross-table prediction
    # ============================================================
    print(f"\n{'='*70}")
    print("IDEA 3: Cross-table prediction")
    print(f"{'='*70}")

    # Compute correlation between tables' DC means
    # Use the first min_len means from each table
    min_len = min(len(all_means[t]) for t in TABLES)
    stacked = torch.stack([all_means[t][:min_len] for t in TABLES])  # (8, min_len)
    corr = torch.corrcoef(stacked)
    print(f"\nDC mean correlation matrix (first {min_len} blocks):")
    print(f"{'':>8}", end='')
    for t in TABLES:
        print(f"  t{t:>2}", end='')
    print()
    for i, t1 in enumerate(TABLES):
        print(f"  t{t1:>2}  ", end='')
        for j, t2 in enumerate(TABLES):
            c = corr[i, j].item()
            print(f" {c:>5.2f}", end='')
        print()

    # Strategy: predict each table from the one most correlated with it
    # For each table, find best predictor (highest |corr|, not itself)
    print(f"\nCross-table residual coding:")
    total_residual_bytes = 0
    reconstructed_xt = {}

    for i, t in enumerate(TABLES):
        means_t = all_means[t]
        # Find best predictor
        best_j, best_corr = -1, 0
        for j, t2 in enumerate(TABLES):
            if t2 == t:
                continue
            c = abs(corr[i, j].item())
            if c > best_corr:
                best_corr = c
                best_j = j
        pred_t = TABLES[best_j]
        pred_means = all_means[pred_t]

        # Align lengths
        n = min(len(means_t), len(pred_means))
        # Linear prediction: means_t ≈ a * pred_means + b
        x = pred_means[:n].numpy()
        y = means_t[:n].numpy()
        if best_corr > 0.1 and len(x) > 10:
            # Least squares fit
            A = np.column_stack([x, np.ones(n)])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b = coeffs
            pred = a * pred_means[:len(means_t)].numpy() + b
            residual = means_t.numpy() - pred[:len(means_t)]
        else:
            residual = means_t.numpy()
            a, b = 0, 0

        # Compress residual with Zstd
        res_comp = compress_zstd(residual.astype(np.float32).tobytes(), level=3)
        # Also compress original for comparison
        orig_comp = compress_zstd(means_t.numpy().astype(np.float32).tobytes(), level=3)

        savings = 1 - len(res_comp) / len(orig_comp)
        print(f"  table {t}: pred=t{pred_t} (corr={best_corr:.3f}), "
              f"a={a:.3f} b={b:.4f}, "
              f"orig={len(orig_comp)}B res={len(res_comp)}B ({savings*100:+.1f}%)")

        total_residual_bytes += len(res_comp)

        # Reconstruct from residual
        dec_res = np.frombuffer(decompress_zstd(res_comp), dtype=np.float32)
        if best_corr > 0.1:
            pred_full = a * pred_means[:len(means_t)].numpy() + b
            dec_means = torch.from_numpy((dec_res + pred_full[:len(means_t)]).astype(np.float32).copy())
        else:
            dec_means = torch.from_numpy(dec_res.copy())

        recon = reconstruct_from_dc(dec_means, table_data[t]['co'], table_data[t]['w'])
        reconstructed_xt[t] = recon

    # Measure cross-table AUC
    for t in TABLES:
        dlrm.emb_l[t].weight.data = reconstructed_xt[t]
    auc_xt = run_auc(dlrm, test_ld)
    delta_xt = auc_xt - baseline_auc
    for t in TABLES:
        dlrm.emb_l[t].weight.data = table_data[t]['w']

    comp_kb_xt = total_residual_bytes / 1024
    ratio_xt = total_cold_fp32_bytes / total_residual_bytes
    print(f"\n  Cross-table residual: {comp_kb_xt:.1f} KB, "
          f"{ratio_xt:.0f}x fp32, AUC {delta_xt*100:+.4f}%")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY (cold fp32 = {total_cold_fp32_mb:.1f} MB)")
    print(f"{'='*70}")
    print(f"{'Strategy':>40} {'KB':>10} {'fp32x':>10} {'AUC delta':>10}")
    for r in results:
        print(f"{r['name']:>40} {r['comp_kb']:>10.1f} {r['ratio_fp32']:>10.0f} {r['delta']*100:>+10.4f}%")
    print(f"{'cross_table_residual':>40} {comp_kb_xt:>10.1f} {ratio_xt:>10.0f} {delta_xt*100:>+10.4f}%")

    # Save results
    out = {
        'baseline_auc': baseline_auc,
        'total_cold_fp32_mb': total_cold_fp32_mb,
        'strategies': results,
        'cross_table': {
            'comp_kb': comp_kb_xt, 'ratio_fp32': ratio_xt,
            'auc': auc_xt, 'delta': delta_xt,
        },
    }
    with open('results/dc_av1_ideas.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to results/dc_av1_ideas.json")


if __name__ == '__main__':
    main()
