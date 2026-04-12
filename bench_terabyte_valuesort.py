#!/usr/bin/env python3
"""
Value-sort DC on Terabyte (D=64) — cross-dataset validation.
Reuses load_terabyte() from bench_terabyte_dc.py.
"""
import os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
os.environ['CRITEO_DAYS'] = '4'
from sklearn.metrics import roc_auc_score
from bench_terabyte_dc import load_terabyte

LARGE_THRESHOLD = 50000
EMB_DIM = 64
BS = 16
HOT_FRACTIONS = [0.043, 0.02, 0.01, 0.005]
MODEL_PATH = 'models/dlrm_terabyte_4day.pt'

def run_auc(dlrm, test_ld):
    all_s, all_l = [], []
    with torch.no_grad():
        for X, o, i, T in test_ld:
            Z = dlrm(X, o, i)
            all_s.append(Z.cpu().numpy().flatten())
            all_l.append(T.numpy().flatten())
    return roc_auc_score(np.concatenate(all_l), np.concatenate(all_s))

def apply_dc(w_orig, cold_idx, sort_order='freq', nbits=8):
    w = w_orig.clone()
    cold_w = w[cold_idx]
    nc = len(cold_idx)
    D = cold_w.shape[1]

    if sort_order == 'value':
        row_means = cold_w.mean(dim=1)
        order = torch.argsort(row_means)
        cold_w = cold_w[order]
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(nc)
    else:
        inv_order = None

    nb = (nc + BS - 1) // BS
    padded = torch.zeros(nb * BS, D)
    padded[:nc] = cold_w
    means = padded.reshape(nb, BS, D).mean(dim=(1, 2))

    n_levels = 2 ** nbits
    mn, mx = means.min().item(), means.max().item()
    if mx != mn:
        s = (mx - mn) / (n_levels - 1)
        q = ((means - mn) / s).round().clamp(0, n_levels - 1)
        means = q * s + mn

    recon_cold = means.unsqueeze(1).unsqueeze(2).expand(nb, BS, D).reshape(-1, D)[:nc]
    if inv_order is not None:
        recon_cold = recon_cold[inv_order]
    w[cold_idx] = recon_cold
    return w

def main():
    print("=" * 70)
    print("TERABYTE VALUE-SORT DC: cross-dataset validation (D=64)")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_terabyte()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))

    LARGE_TABLES = [i for i, n in enumerate(ln_emb) if n >= LARGE_THRESHOLD]
    print(f"Large tables: {LARGE_TABLES}")

    baseline_auc = run_auc(dlrm, test_ld)
    print(f"Baseline AUC: {baseline_auc:.6f}")

    # Profile
    print("Profiling access frequencies...")
    freq = {}
    bc = 0
    for X, lS_o, lS_i, T in test_ld:
        for t in LARGE_TABLES:
            idx = lS_i[t].flatten() if isinstance(lS_i, (list, tuple)) else lS_i[t].flatten()
            if t not in freq:
                freq[t] = torch.zeros(int(ln_emb[t]), dtype=torch.long)
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
        bc += 1
        if bc % 200 == 0:
            print(f"  {bc} batches...")
    print(f"  done: {bc} batches")

    orig_w = {t: sd[ek[t]].clone() for t in LARGE_TABLES}
    total_fp32_mb = sum(sd[ek[i]].shape[0] * EMB_DIM * 4 for i in range(len(ln_emb))) / 1024**2
    small_uint8_mb = sum(sd[ek[i]].shape[0] * EMB_DIM * 1 for i in range(len(ln_emb))
                         if i not in LARGE_TABLES) / 1024**2

    print(f"Total fp32: {total_fp32_mb:.1f} MB, Small uint8: {small_uint8_mb:.1f} MB")
    print(f"\n{'hf':>6} {'method':>16} {'ratio':>8} {'AUC':>10} {'delta%':>10}")
    print("-" * 60)

    results = {'baseline_auc': baseline_auc, 'total_fp32_mb': total_fp32_mb, 'configs': []}

    for hf in HOT_FRACTIONS:
        cold_idx = {}
        for t in LARGE_TABLES:
            n = int(ln_emb[t])
            n_hot = max(1, int(n * hf))
            sorted_idx = freq[t].argsort(descending=True)
            cold_idx[t] = sorted_idx[n_hot:]

        hot_mb = sum((int(ln_emb[t]) - len(cold_idx[t])) * EMB_DIM * 1
                     for t in LARGE_TABLES) / 1024**2
        bitmap_mb = sum(int(ln_emb[t]) for t in LARGE_TABLES) / 8 / 1024**2

        for method in ['zero', 'dc_freq', 'dc_value']:
            bits_list = [8] if method == 'zero' else [8, 4]
            for nbits in bits_list:
                for t in LARGE_TABLES:
                    if method == 'zero':
                        w = orig_w[t].clone()
                        w[cold_idx[t]] = 0.0
                        dlrm.emb_l[t].weight.data = w
                    else:
                        so = 'value' if method == 'dc_value' else 'freq'
                        w = apply_dc(orig_w[t], cold_idx[t], so, nbits)
                        dlrm.emb_l[t].weight.data = w

                auc = run_auc(dlrm, test_ld)
                delta = auc - baseline_auc
                for t in LARGE_TABLES:
                    dlrm.emb_l[t].weight.data = orig_w[t].clone()

                if method == 'zero':
                    cold_mb = 0
                else:
                    nb = sum((len(cold_idx[t]) + BS - 1) // BS for t in LARGE_TABLES)
                    cold_mb = nb * nbits / 8 / 1024**2

                total_mb = hot_mb + cold_mb + small_uint8_mb + bitmap_mb
                ratio = total_fp32_mb / total_mb
                label = method if method == 'zero' else f"{method}_{nbits}b"

                print(f"{hf:>6.1%} {label:>16} {ratio:>8.0f}x {auc:>10.6f} {delta*100:>+10.4f}")
                results['configs'].append({
                    'method': label, 'hf': hf, 'nbits': nbits,
                    'ratio': ratio, 'auc': auc, 'delta': delta,
                })
        print()

    out_path = 'results/terabyte_valuesort.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
