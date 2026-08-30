#!/usr/bin/env python3
"""Scalar vs per-dim DC on Avazu_x1 (third dataset, non-Criteo, D=16, no dense).

Mirrors the Terabyte/Kaggle methodology: hot/cold by test-set access frequency on
tables >=50k rows, value-sort cold, 4-bit means, AUC on the full test split.
Variants: zero, scalar bs16, perdim bs{16,64,256}; hot fractions 4.3/2/1/0.5%.
"""
import os, sys, time, json, numpy as np, torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui'); os.chdir('/home/cc/expr/dlrm_minrui')
from sklearn.metrics import roc_auc_score
from dlrm_s_pytorch import DLRM_Net
torch.set_num_threads(24); torch.set_grad_enabled(False)
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

z = np.load('/mnt/nvme1/avazu/avazu_x1.npz')
Xte, yte, counts = z['X_cat_test'], z['y_test'], z['counts']
ck = torch.load('models/dlrm_avazu_x1.pt', map_location='cpu', weights_only=False)
D = ck['D']; ntab = len(counts)
ln_emb = np.array(counts); ln_bot = np.array([1,64,D])
num_int = (ntab+1)*ntab//2 + D
dlrm = DLRM_Net(D, ln_emb, ln_bot, np.array([num_int,512,256,1]),
                arch_interaction_op="dot", arch_interaction_itself=False,
                sigmoid_bot=-1, sigmoid_top=2, loss_function="bce")
dlrm.load_state_dict(ck['state_dict']); dlrm.eval()
W = [dlrm.emb_l[t].weight.data for t in range(ntab)]
orig = [w.clone() for w in W]
LARGE = [t for t in range(ntab) if counts[t] >= 50000]
total_mb = sum(int(c)*D*4 for c in counts)/2**20
small_mb = sum(int(counts[t])*D*4 for t in range(ntab) if t not in LARGE)/2**20
log(f"tables={ntab} large={LARGE} (cards {[int(counts[t]) for t in LARGE]}) fp32={total_mb:.1f} MB")

TB = 16384
dte = torch.zeros(TB,1); offe = torch.arange(TB)
ncov = len(yte)//TB*TB
def auc():
    s=[]
    for j in range(0, ncov, TB):
        xc = torch.from_numpy(Xte[j:j+TB].astype(np.int64))
        s.append(dlrm(dte, [offe]*ntab, [xc[:,t] for t in range(ntab)]).numpy().ravel())
    return roc_auc_score(yte[:ncov], np.concatenate(s))
# NOTE arg order: dlrm(dense, lS_o, lS_i)
base = auc()
log(f"BASELINE AUC {base:.6f} on {ncov:,} test rows (ckpt recorded {ck.get('test_auc', float('nan')):.6f})")

# frequency from the test split (same protocol as Criteo runs)
freq = {t: np.bincount(Xte[:,t], minlength=int(counts[t])) for t in LARGE}

def apply_dc(t, cold_idx, mode, block, nbits=4):
    w = orig[t].clone(); cold = w[cold_idx]
    order = torch.argsort(cold.mean(dim=1)); cs = cold[order]
    n = cs.shape[0]; nb=(n+block-1)//block; pad=nb*block-n
    if pad: cs = torch.cat([cs, torch.zeros(pad, D)],0)
    r = cs.view(nb, block, D)
    means = r.mean(dim=(1,2)) if mode=='scalar' else r.mean(dim=1)
    lo,hi = means.min(), means.max(); lv=(1<<nbits)-1
    s=(hi-lo)/lv
    if s==0: s=torch.tensor(1.0)
    mq = (((means-lo)/s).round().clamp(0,lv))*s+lo
    rec = (mq.repeat_interleave(block).unsqueeze(1).expand(-1,D) if mode=='scalar'
           else mq.repeat_interleave(block, dim=0))[:n].contiguous()
    inv = torch.empty_like(order); inv[order]=torch.arange(n)
    w[cold_idx] = rec[inv]
    return w

results=[dict(config='fp32', mem_mb=total_mb, auc=base)]
for hf in (0.043, 0.02, 0.01, 0.005):
    cold={}
    for t in LARGE:
        nh = max(1, int(counts[t]*hf))
        order = torch.from_numpy(np.argsort(-freq[t]).copy())
        cold[t] = order[nh:]
    hot_mb  = sum((int(counts[t])-len(cold[t]))*D for t in LARGE)/2**20   # uint8 hot
    bmap_mb = sum(int(counts[t]) for t in LARGE)/8/2**20
    for mode, block in [('zero',16),('scalar',16),('perdim',16),('perdim',64),('perdim',256)]:
        for t in LARGE:
            if mode=='zero':
                w=orig[t].clone(); w[cold[t]]=0.0
            else:
                w=apply_dc(t, cold[t], mode, block)
            W[t].copy_(w)
        nblk = sum((len(cold[t])+block-1)//block for t in LARGE)
        dc_mb = 0 if mode=='zero' else nblk*(0.5 if mode=='scalar' else D*0.5)/2**20
        mb = hot_mb+bmap_mb+small_mb+dc_mb
        a = auc()
        results.append(dict(config=f'{mode} bs{block}' if mode!='zero' else 'zero',
                            hot_frac=hf, mem_mb=mb, auc=a))
        json.dump(results, open('results/avazu_dc_sweep.json','w'), indent=1)
        log(f"hf={hf:.3f} {mode:>6} bs{block:<4} {mb:7.2f}MB ({total_mb/mb:5.1f}x) "
            f"AUC {a:.6f} ({(a-base)*100:+.4f}%)")
        for t in LARGE: W[t].copy_(orig[t])
log("done -> results/avazu_dc_sweep.json")
