#!/usr/bin/env python3
"""
Compare single-scalar DC vs per-dimension DC block-mean.
Both with PCA sort, 4-bit quantization, real AUC.

Single-scalar: each block of 16 rows → 1 scalar (mean of all 256 values)
Per-dimension: each block of 16 rows → 16 values (mean per dimension)
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
from codec_ondemand_benchmark import load_model_and_data, MODEL_PATH

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
EMB_DIM = 16
OUT_DIR = 'results/intelligent_agent'
os.makedirs(OUT_DIR, exist_ok=True)

HOT_FRACTIONS = [0.005, 0.01, 0.02, 0.043]
BLOCK_SIZE = 16


def pca_sort_order(cold_w):
    sample_size = min(50000, len(cold_w))
    pca = PCA(n_components=1, random_state=42)
    if sample_size < len(cold_w):
        idx = np.random.RandomState(42).choice(len(cold_w), sample_size, replace=False)
        pca.fit(cold_w[idx])
    else:
        pca.fit(cold_w)
    return np.argsort(pca.transform(cold_w).ravel())


def apply_dc(weight, freq, hf, mode='scalar'):
    """
    mode='scalar': 1 scalar per block (original 189x approach)
    mode='perdim': 16 values per block (per-dimension means)
    """
    w = weight.numpy().copy()
    n_rows, dim = w.shape
    n_hot = max(1, int(n_rows * hf))
    n_cold = n_rows - n_hot

    if n_cold == 0:
        return weight.clone()

    freq_np = freq.numpy() if isinstance(freq, torch.Tensor) else freq
    sorted_by_freq = np.argsort(-freq_np)
    cold_idx = sorted_by_freq[n_hot:]
    cold_w = w[cold_idx]

    # PCA sort
    order = pca_sort_order(cold_w)
    cold_w_sorted = cold_w[order]

    # Block
    n_blocks = (n_cold + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded = np.zeros((n_blocks * BLOCK_SIZE, dim))
    padded[:n_cold] = cold_w_sorted
    blocks = padded.reshape(n_blocks, BLOCK_SIZE, dim)

    if mode == 'scalar':
        # Single scalar: mean over both rows AND dimensions
        means = blocks.mean(axis=(1, 2))  # shape (n_blocks,)
        # 4-bit quantize
        mn, mx = means.min(), means.max()
        s = (mx - mn) / 15.0 if mx != mn else 1.0
        q = np.clip(np.round((means - mn) / s), 0, 15)
        deq = q * s + mn
        # Reconstruct: broadcast scalar to all rows and dims
        reconstructed = np.repeat(deq[:, None, None], BLOCK_SIZE, axis=1)
        reconstructed = np.repeat(reconstructed, dim, axis=2)
        reconstructed = reconstructed.reshape(-1, dim)[:n_cold]
    else:
        # Per-dimension: mean over rows only, keep dims separate
        means = blocks.mean(axis=1)  # shape (n_blocks, dim)
        # 4-bit quantize per dimension
        mn, mx = means.min(), means.max()
        s = (mx - mn) / 15.0 if mx != mn else 1.0
        q = np.clip(np.round((means - mn) / s), 0, 15)
        deq = q * s + mn
        # Reconstruct: broadcast to all rows in block
        reconstructed = np.repeat(deq, BLOCK_SIZE, axis=0)[:n_cold]

    # Un-sort
    unsort = np.argsort(order)
    result = weight.clone()
    result[cold_idx] = torch.from_numpy(reconstructed[unsort]).float()
    return result


def main():
    print("=" * 70)
    print("SINGLE-SCALAR DC vs PER-DIMENSION DC (Real AUC)")
    print("=" * 70)

    print("\n[1] Loading...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    torch.set_num_threads(32)

    # Profile freq
    print("  Profiling...")
    freq_counts = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, lS_o, lS_i, T in test_batches:
        for t in TABLES:
            idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
            freq_counts[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))

    # Baseline
    print("\n[2] Baseline...")
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        dlrm.emb_l[t].weight.data = sd[k].clone()
    scores, targets = [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            Z = dlrm(X, lS_o, lS_i)
            scores.append(Z.detach().numpy().ravel())
            targets.append(T.numpy().ravel())
    baseline_auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"  Baseline AUC = {baseline_auc:.6f}")

    # Compute fp32 totals for ratio
    total_fp32 = sum(sd[k].numel() * 4 for k in ek)
    small_tables = [i for i in range(26) if i not in TABLES]
    small_uint8 = sum(sd[ek[t]].shape[0] * EMB_DIM * 1 for t in small_tables)

    # Sweep
    print(f"\n[3] Evaluating {len(HOT_FRACTIONS)} hf × 2 modes = {len(HOT_FRACTIONS)*2} configs...")
    results = []

    for hf in HOT_FRACTIONS:
        for mode in ['scalar', 'perdim']:
            t0 = time.time()
            # Apply to all large tables
            for k in ek:
                t = int(k.split('.')[1])
                dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
                if t in TABLES:
                    dlrm.emb_l[t].weight.data = apply_dc(sd[k], freq_counts[t], hf, mode)
                else:
                    dlrm.emb_l[t].weight.data = sd[k].clone()

            scores, targets = [], []
            with torch.no_grad():
                for X, lS_o, lS_i, T in test_batches:
                    Z = dlrm(X, lS_o, lS_i)
                    scores.append(Z.detach().numpy().ravel())
                    targets.append(T.numpy().ravel())
            auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))

            # Compute ratio
            hot_bytes = 0
            cold_dc_bytes = 0
            bitmap_bytes = 0
            for t in TABLES:
                n = sd[ek[t]].shape[0]
                n_hot = max(1, int(n * hf))
                n_cold = n - n_hot
                n_blocks = (n_cold + BLOCK_SIZE - 1) // BLOCK_SIZE
                hot_bytes += n_hot * EMB_DIM * 1
                if mode == 'scalar':
                    cold_dc_bytes += n_blocks * 4 / 8  # 4 bits per block
                else:
                    cold_dc_bytes += n_blocks * EMB_DIM * 4 / 8  # 4 bits × 16 dims
                bitmap_bytes += n // 8

            compressed = hot_bytes + cold_dc_bytes + bitmap_bytes + small_uint8
            ratio = total_fp32 / compressed

            elapsed = time.time() - t0
            loss_pct = (auc - baseline_auc) / baseline_auc * 100
            results.append({
                'hf': hf, 'mode': mode, 'auc': auc, 'loss_pct': loss_pct,
                'ratio': ratio, 'compressed_mb': compressed / 1024 / 1024,
                'hot_mb': hot_bytes / 1024 / 1024,
                'cold_dc_mb': cold_dc_bytes / 1024 / 1024,
                'bitmap_mb': bitmap_bytes / 1024 / 1024,
            })

            print(f"  hf={hf}, {mode:>7}: AUC={auc:.6f} ({loss_pct:+.4f}%), "
                  f"{compressed/1024/1024:.1f} MB, {ratio:.0f}x [{elapsed:.1f}s]")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Single-Scalar DC vs Per-Dimension DC (PCA sort, 4-bit)")
    print(f"{'='*70}")
    print(f"Baseline AUC: {baseline_auc:.6f}")
    print(f"\n{'HF':>6} {'Mode':>8} {'AUC':>10} {'Loss%':>10} {'Size':>8} {'Ratio':>8} {'Hot':>7} {'DC':>7} {'Map':>7}")
    print(f"{'-'*76}")
    for r in sorted(results, key=lambda x: (x['hf'], x['mode'])):
        print(f"{r['hf']:>6.1%} {r['mode']:>8} {r['auc']:>10.6f} {r['loss_pct']:>+9.4f}% "
              f"{r['compressed_mb']:>7.1f}MB {r['ratio']:>7.0f}x "
              f"{r['hot_mb']:>6.1f}MB {r['cold_dc_mb']:>6.1f}MB {r['bitmap_mb']:>6.1f}MB")

    # Save
    output = {
        'baseline_auc': baseline_auc,
        'total_fp32_mb': total_fp32 / 1024 / 1024,
        'results': [{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                      for k, v in r.items()} for r in results],
    }
    with open(f'{OUT_DIR}/scalar_vs_perdim.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUT_DIR}/scalar_vs_perdim.json")


if __name__ == '__main__':
    main()
