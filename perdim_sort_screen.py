#!/usr/bin/env python3
"""Screen candidate orderings for PER-DIM block means by their exact objective:
total within-block per-dimension variance (what per-dim reconstruction minimizes).

Candidates (all O(N log N), feasible at 10M rows):
  none      original id order
  value     sort by row mean (current method; optimal for SCALAR only)
  norm      sort by L2 norm
  pc1       sort by 1st principal component score
  pc1|pc2   lexicographic: coarse PC1 buckets, PC2 within  (2-level projection)
  zorder    Morton order on quantized (PC1, PC2)
Measured on the 3 largest tables of the 24-day model, cold = all but top 1% by norm
(proxy; ordering ranking is what matters, block=256, D=64).
"""
import torch, numpy as np, time, json
torch.set_grad_enabled(False)
BLOCK, D = 256, 64
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

sd = torch.load('models/gpu24/dlrm_terabyte_24day.pt', map_location='cpu',
                weights_only=False)['state_dict']
keys = sorted([k for k in sd if 'emb_l' in k and 'weight' in k], key=lambda x:int(x.split('.')[1]))
big = sorted(keys, key=lambda k: -sd[k].shape[0])[:3]

def pcs(x, k=2):
    xc = x - x.mean(0, keepdim=True)
    g = torch.Generator().manual_seed(0)
    vs = []
    for i in range(k):
        v = torch.randn(x.shape[1], generator=g); v /= v.norm()
        for _ in range(30):
            v = xc.T @ (xc @ v)
            for u in vs: v -= (v@u)*u
            v /= (v.norm()+1e-12)
        vs.append(v)
    return [xc @ v for v in vs]

def obj(rows, order):
    r = rows[order]
    n = (r.shape[0] // BLOCK) * BLOCK
    b = r[:n].view(-1, BLOCK, D)
    return float(((b - b.mean(dim=1, keepdim=True))**2).sum())

def morton(a, b, bits=16):
    """interleave bits of two quantized keys"""
    z = torch.zeros_like(a, dtype=torch.int64)
    for i in range(bits):
        z |= ((a >> i) & 1) << (2*i+1)
        z |= ((b >> i) & 1) << (2*i)
    return z

out = {}
for k in big:
    w = sd[k]
    n = w.shape[0]
    g = torch.Generator().manual_seed(1)
    sel = torch.randperm(n, generator=g)[:2_000_000]     # 2M-row sample per table
    w = w[sel].float()
    nrm = w.norm(dim=1)
    hot = torch.topk(nrm, int(0.01*len(nrm))).indices
    mask = torch.ones(len(nrm), dtype=torch.bool); mask[hot] = False
    cold = w[mask]
    p1, p2 = pcs(cold)
    q = lambda x: ((x - x.min())/(x.max()-x.min()+1e-12)*65535).long()
    cands = {
        'none':    torch.arange(cold.shape[0]),
        'value':   torch.argsort(cold.mean(dim=1)),
        'norm':    torch.argsort(cold.norm(dim=1)),
        'pc1':     torch.argsort(p1),
        'pc1|pc2': torch.argsort(q(p1)//4096 * 100_000_000 + q(p2)),
        'zorder':  torch.argsort(morton(q(p1), q(p2))),
    }
    res = {nm: obj(cold, o) for nm, o in cands.items()}
    base = res['value']
    out[k] = res
    log(f"{k} (cold {cold.shape[0]:,} rows) — within-block per-dim variance, value=1.0:")
    for nm, v in sorted(res.items(), key=lambda x: x[1]):
        log(f"    {nm:<8} {v/base:6.4f}")
json.dump(out, open('results/perdim_sort_screen.json','w'), indent=1)
log("done")
