#!/usr/bin/env python3
"""Why does PCA-sort beat value-sort on Kaggle (D=16) but not Terabyte (D=64)?

Scalar DC replaces a block B of rows with one number mu_B = mean of all its values.
The squared error decomposes exactly:

  sum_{r in B} sum_d (w[r,d] - mu_B)^2
    = sum_r sum_d (w[r,d] - mu_r)^2      <- WITHIN-ROW scatter, independent of ordering
    + D * sum_r (mu_r - mu_B)^2          <- BETWEEN-ROW scatter, the only ordering-dependent term

So ordering can only reduce the between-row term, which depends on the row means
alone -- and sorting by row mean is exactly optimal for that. PCA-sort orders by the
first principal component instead, which can only match or lose. This measures both
terms on real tables from both datasets.
"""
import sys, torch, numpy as np
torch.set_grad_enabled(False)
BS = 16
SAMPLE = 300_000

def pc1_score(x):
    xc = x - x.mean(0, keepdim=True)
    g = torch.Generator().manual_seed(0)
    v = torch.randn(xc.shape[1], generator=g); v /= v.norm()
    for _ in range(50):
        v = xc.T @ (xc @ v); v /= (v.norm() + 1e-12)
    return xc @ v, v

def dc_terms(rows):
    """total, within-row, between-row squared error for contiguous blocks of BS"""
    n, D = rows.shape
    nb = n // BS
    r = rows[:nb*BS].reshape(nb, BS, D)
    mu_row   = r.mean(dim=2, keepdim=True)          # (nb, BS, 1)
    mu_block = r.mean(dim=(1,2), keepdim=True)      # (nb, 1, 1)
    within  = ((r - mu_row) ** 2).sum().item()
    between = (D * (mu_row.squeeze(2) - mu_block.squeeze(2)) ** 2).sum().item()
    total   = ((r - mu_block) ** 2).sum().item()
    return total, within, between

def analyse(name, path, dim, ntab=4):
    sd = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    keys = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                  key=lambda x: int(x.split('.')[1]))
    big = sorted(keys, key=lambda k: -sd[k].shape[0])[:ntab]
    print(f"\n{'='*78}\n{name}  (D={dim})\n{'='*78}")
    print(f"{'table':>18}{'rows':>10}{'corr(PC1,mean)':>16}{'within%':>10}"
          f"{'between: value':>16}{'pca':>12}{'random':>12}")
    for k in big:
        w = sd[k]
        n = w.shape[0]
        if n > SAMPLE:
            g = torch.Generator().manual_seed(1)
            sel = torch.randperm(n, generator=g)[:SAMPLE]
            w = w[sel]
        mu = w.mean(dim=1)
        pc, v = pc1_score(w)
        c = float(np.corrcoef(pc.numpy(), mu.numpy())[0, 1])
        res = {}
        for how, key in (('value', mu), ('pca', pc), ('random', torch.randn(w.shape[0], generator=torch.Generator().manual_seed(2)))):
            o = torch.argsort(key)
            res[how] = dc_terms(w[o])
        tot, within, _ = res['value']
        print(f"{k:>18}{w.shape[0]:>10}{c:>+16.4f}{100*within/tot:>9.1f}%"
              f"{res['value'][2]:>16.4g}{res['pca'][2]:>12.4g}{res['random'][2]:>12.4g}")
        print(f"{'':>18}{'':>10}{'':>16}{'':>10}"
              f"  total MSE: value {res['value'][0]:.4g}  pca {res['pca'][0]:.4g}  "
              f"random {res['random'][0]:.4g}")
    # how aligned is PC1 with the all-ones direction?
    w = sd[big[0]]
    if w.shape[0] > SAMPLE:
        g = torch.Generator().manual_seed(1)
        w = w[torch.randperm(w.shape[0], generator=g)[:SAMPLE]]
    _, v = pc1_score(w)
    ones = torch.ones(v.shape[0]) / (v.shape[0] ** 0.5)
    print(f"\n  |cos(PC1, all-ones)| on largest table: {abs(float(v @ ones)):.4f}"
          f"   (1.0 => PCA-sort IS value-sort)")

analyse("Criteo Terabyte", "models/dlrm_terabyte_4day.pt", 64)
analyse("Criteo Kaggle",   "models/dlrm_kaggle_correct.pt", 16)
