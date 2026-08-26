#!/usr/bin/env python3
"""Terabyte D=64 sweep on the properly-trained 4-day checkpoint (AUC 0.795188).

Adds to bench_terabyte_valuesort.py:
  - whole-table INT8 / INT4 quantization baselines (the standard post-training
    compression everyone compares against)
  - PCA-sort (README reports it beats value-sort by 17-26% on Kaggle)
  - a bounded AUC evaluation (MAX_AUC_BATCHES) so the sweep is hours not days

env: MAX_AUC_BATCHES (default 20000 batches x 1024 = 20.5M samples; 0 = full set)
"""
import os, sys, time, json
import numpy as np, torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
os.environ.setdefault('CRITEO_DAYS', '4')
from sklearn.metrics import roc_auc_score
from bench_terabyte_dc import load_terabyte

LARGE_THRESHOLD = 50000
EMB_DIM   = 64
BS        = 16                      # rows per DC block
HOT_FRACTIONS = [0.043, 0.02, 0.01, 0.005]
MODEL_PATH = 'models/dlrm_terabyte_4day.pt'
MAX_AUC_BATCHES = int(os.environ.get('MAX_AUC_BATCHES', 20000))

def run_auc(dlrm, test_ld):
    s, l, n = [], [], 0
    with torch.no_grad():
        for X, o, i, T in test_ld:
            s.append(dlrm(X, o, i).numpy().flatten()); l.append(T.numpy().flatten())
            n += 1
            if MAX_AUC_BATCHES and n >= MAX_AUC_BATCHES: break
    return roc_auc_score(np.concatenate(l), np.concatenate(s))

def quant_dequant(w, nbits):
    """Whole-table symmetric-range quantization -> the INT8 / INT4 baseline."""
    levels = (1 << nbits) - 1
    mn, mx = w.min(), w.max()
    s = (mx - mn) / levels
    if s == 0: s = torch.tensor(1.0)
    q = ((w - mn) / s).round().clamp(0, levels)
    return q * s + mn

def sort_key(w_cold, how):
    if how == 'value': return w_cold.mean(dim=1)
    if how == 'pca':
        x = w_cold - w_cold.mean(dim=0, keepdim=True)
        # first principal component score, via power iteration (cheap, no full SVD)
        v = torch.randn(x.shape[1], generator=torch.Generator().manual_seed(0))
        v /= v.norm()
        for _ in range(20):
            v = x.T @ (x @ v); v /= (v.norm() + 1e-12)
        return x @ v
    return None                                   # 'freq' = already freq-ordered

def apply_dc(w_orig, cold_idx, how, nbits):
    w = w_orig.clone()
    cold = w[cold_idx]
    if how != 'freq':
        order = torch.argsort(sort_key(cold, how))
        cold = cold[order]
    n = cold.shape[0]; nb = (n + BS - 1) // BS
    pad = nb * BS - n
    if pad: cold = torch.cat([cold, torch.zeros(pad, cold.shape[1])], 0)
    means = cold.view(nb, BS, -1).mean(dim=(1, 2))            # scalar DC per block
    lo, hi = means.min(), means.max()
    levels = (1 << nbits) - 1
    s = (hi - lo) / levels
    if s == 0: s = torch.tensor(1.0)
    means_q = (((means - lo) / s).round().clamp(0, levels)) * s + lo
    recon = means_q.repeat_interleave(BS).unsqueeze(1).expand(-1, cold.shape[1]).contiguous()
    recon = recon[:n]
    if how != 'freq':
        inv = torch.empty_like(order); inv[order] = torch.arange(n)
        recon = recon[inv]
    w[cold_idx] = recon
    return w

def main():
    t0 = time.time()
    dlrm, test_ld, train_ld, ln_emb = load_terabyte()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k], key=lambda x: int(x.split('.')[1]))
    dlrm.load_state_dict(sd)
    LARGE = [i for i, n in enumerate(ln_emb) if n >= LARGE_THRESHOLD]
    print(f"large tables: {LARGE}", flush=True)

    base = run_auc(dlrm, test_ld)
    print(f"[{time.time()-t0:6.0f}s] BASELINE AUC {base:.6f}  "
          f"({MAX_AUC_BATCHES or 'all'} batches)", flush=True)

    orig = {t: sd[ek[t]].clone() for t in LARGE}
    total_fp32_mb = sum(sd[ek[i]].shape[0] * EMB_DIM * 4 for i in range(len(ln_emb))) / 1024**2
    small_u8_mb   = sum(sd[ek[i]].shape[0] * EMB_DIM * 1 for i in range(len(ln_emb))
                        if i not in LARGE) / 1024**2
    res = {'baseline_auc': base, 'total_fp32_mb': total_fp32_mb,
           'auc_batches': MAX_AUC_BATCHES, 'configs': []}

    def record(label, hf, mb, auc):
        r = {'label': label, 'hot_frac': hf, 'mb': mb, 'ratio': total_fp32_mb / mb,
             'auc': auc, 'delta_pct': (auc - base) * 100}
        res['configs'].append(r)
        print(f"  {label:<26} {hf if hf else '-':>6} {mb:>8.1f}MB "
              f"{total_fp32_mb/mb:>7.0f}x {auc:.6f} {(auc-base)*100:>+8.4f}%", flush=True)
        json.dump(res, open('results/terabyte_full_sweep.json', 'w'), indent=1)

    # ---- whole-table quantization baselines (no hot/cold split) ----
    print("\n=== quantization baselines ===", flush=True)
    for nbits in (8, 4):
        for t in LARGE: dlrm.emb_l[t].weight.data = quant_dequant(orig[t], nbits)
        mb = total_fp32_mb * nbits / 32.0
        record(f'INT{nbits} whole-table', None, mb, run_auc(dlrm, test_ld))
        for t in LARGE: dlrm.emb_l[t].weight.data = orig[t].clone()

    # ---- profile access frequency for the hot/cold split ----
    print("\nprofiling access frequency...", flush=True)
    freq, bc = {}, 0
    for X, o, i, T in test_ld:
        for t in LARGE:
            idx = i[t].flatten()
            if t not in freq: freq[t] = torch.zeros(int(ln_emb[t]), dtype=torch.long)
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
        bc += 1
        if bc >= 2000: break
    print(f"  profiled {bc} batches", flush=True)

    # ---- DC sweep ----
    print("\n=== DC block-mean ===", flush=True)
    for hf in HOT_FRACTIONS:
        cold = {}
        for t in LARGE:
            n = int(ln_emb[t]); nh = max(1, int(n * hf))
            cold[t] = freq[t].argsort(descending=True)[nh:]
        hot_mb  = sum((int(ln_emb[t]) - len(cold[t])) * EMB_DIM * 1 for t in LARGE) / 1024**2
        bmap_mb = sum(int(ln_emb[t]) for t in LARGE) / 8 / 1024**2
        for how in ('zero', 'freq', 'value', 'pca'):
            for nbits in ((8,) if how == 'zero' else (4,)):
                for t in LARGE:
                    if how == 'zero':
                        w = orig[t].clone(); w[cold[t]] = 0.0
                    else:
                        w = apply_dc(orig[t], cold[t], how, nbits)
                    dlrm.emb_l[t].weight.data = w
                dc_mb = 0.0 if how == 'zero' else sum(
                    (len(cold[t]) + BS - 1) // BS for t in LARGE) * nbits / 8 / 1024**2
                mb = hot_mb + bmap_mb + small_u8_mb + dc_mb
                record(f'{how}' + ('' if how == 'zero' else f'-sort {nbits}bit'), hf, mb,
                       run_auc(dlrm, test_ld))
                for t in LARGE: dlrm.emb_l[t].weight.data = orig[t].clone()
    print(f"\ndone in {(time.time()-t0)/60:.0f} min -> results/terabyte_full_sweep.json")

main()
