#!/usr/bin/env python3
"""What is worth storing per cold row at D=64?

Scalar block DC stores ONE number per 16 rows. Keeping a per-ROW mean instead removes
the between-row term. But the decomposition showed within-row scatter is ~99.7% of the
error at D=64, so this should buy almost nothing for 16x the storage. Per-DIMENSION
means are the only variant that attacks the within-row term. Measured here.
"""
import torch, numpy as np
torch.set_grad_enabled(False)
BS = 16; SAMPLE = 300_000

def mse(a, b): return float(((a - b) ** 2).mean())

sd = torch.load('models/dlrm_terabyte_4day.pt', map_location='cpu', weights_only=False)['state_dict']
keys = sorted([k for k in sd if 'emb_l' in k and 'weight' in k], key=lambda x: int(x.split('.')[1]))
big  = sorted(keys, key=lambda k: -sd[k].shape[0])[:3]

print(f"{'variant':<34}{'bytes/row':>11}{'MSE':>12}{'vs scalar':>11}{'% err left':>12}")
print("-" * 80)
for k in big:
    w = sd[k]
    if w.shape[0] > SAMPLE:
        g = torch.Generator().manual_seed(1)
        w = w[torch.randperm(w.shape[0], generator=g)[:SAMPLE]]
    D = w.shape[1]
    w = w[torch.argsort(w.mean(dim=1))]              # value-sort, the optimal ordering
    n = (w.shape[0] // BS) * BS
    w = w[:n]; r = w.reshape(-1, BS, D)
    base = mse(w, torch.zeros_like(w))               # error if cold rows were zeroed

    variants = []
    # 1. scalar block mean (current method)
    m = r.mean(dim=(1, 2), keepdim=True).expand_as(r).reshape(n, D)
    variants.append(("scalar block mean (current)", 4/8/BS, mse(w, m)))
    # 2. scalar per-ROW mean
    m = w.mean(dim=1, keepdim=True).expand(-1, D)
    variants.append(("scalar per-ROW mean", 4/8, mse(w, m)))
    # 3. scalar per-row mean at 8 bit
    variants.append(("scalar per-ROW mean (8-bit)", 1.0, mse(w, m)))
    # 4. per-DIM block mean
    m = r.mean(dim=1, keepdim=True).expand_as(r).reshape(n, D)
    variants.append(("per-DIM block mean", D*4/8/BS, mse(w, m)))
    # 5. per-DIM block mean, block of 64
    B2 = 64
    n2 = (w.shape[0] // B2) * B2
    r2 = w[:n2].reshape(-1, B2, D)
    m2 = r2.mean(dim=1, keepdim=True).expand_as(r2).reshape(n2, D)
    variants.append((f"per-DIM block mean (BS={B2})", D*4/8/B2, mse(w[:n2], m2)))
    # 6. per-dim block mean + per-row scalar offset
    m = r.mean(dim=1, keepdim=True).expand_as(r).reshape(n, D)
    off = (w - m).mean(dim=1, keepdim=True)
    variants.append(("per-DIM block + per-row offset", D*4/8/BS + 4/8, mse(w, m + off)))
    # 7. full uint8 row (INT8, no hot/cold split) for reference
    mn = w.min(); s = (w.max() - mn) / 255.0
    q = ((w - mn) / s).round().clamp(0, 255) * s + mn
    variants.append(("uint8 row (INT8 reference)", D*1.0, mse(w, q)))

    print(f"[{k}]  n={n:,} D={D}   zeroing MSE = {base:.3e}")
    sc = variants[0][2]
    for name, bpr, e in variants:
        print(f"  {name:<32}{bpr:>11.3f}{e:>12.3e}{sc/e:>10.2f}x{100*e/base:>11.1f}%")
    print()
