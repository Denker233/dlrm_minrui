#!/usr/bin/env python3
"""DC compression loss vs training progress.

Evaluates each periodic checkpoint (250k..2M + final) with:
  fp32 baseline AUC, scalar block=16 @1% hot, per-dim block=256 @1% hot.
Data + frequency profile load once; per checkpoint only the state_dict swaps.
"""
import os, sys, glob, time, json, re, ctypes
import numpy as np, torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui'); os.chdir('/home/cc/expr/dlrm_minrui')
os.environ.setdefault('CRITEO_DAYS', '24')
from sklearn.metrics import roc_auc_score
import bench_unified_terabyte as U
from bench_perdim_terabyte import build

BATCH, D = U.BATCH, U.EMB_DIM
HF = 0.01
AUC_BATCHES = int(os.environ.get('AUC_BATCHES', 2000))
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.0f}s] {m}", flush=True)
def parr(p): return (ctypes.c_void_p*len(p))(*[ctypes.c_void_p(x) for x in p])
def farr(v): return (ctypes.c_float*len(v))(*v)
def iarr(v): return (ctypes.c_int*len(v))(*v)

lib = ctypes.CDLL('./libembfwd.so')
log("loading data (once) ...")
dlrm, test_ld, ln_emb = U.load_tb()
ntab = len(ln_emb)
LARGE = set(i for i,n in enumerate(ln_emb) if n >= U.LARGE_THRESH)
torch.set_grad_enabled(False)
outs = [torch.empty((BATCH, D), dtype=torch.float32) for _ in range(ntab)]
outp = parr([o.data_ptr() for o in outs])
is_large = iarr([1 if t in LARGE else 0 for t in range(ntab)])

log("profiling access frequency (once) ...")
freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in LARGE}
n=0
for X,o,i,T in test_ld:
    for t in LARGE:
        freq[t].scatter_add_(0, i[t].flatten().long(), torch.ones(i[t].numel(), dtype=torch.long))
    n+=1
    if n>=500: break
hot_idx, cold_idx, hot_pos = {}, {}, {}
for t in LARGE:
    nrow=int(ln_emb[t]); nh=max(1,int(nrow*HF))
    order=freq[t].argsort(descending=True)
    hot_idx[t], cold_idx[t] = order[:nh], order[nh:]
    hp=torch.full((nrow,),-1,dtype=torch.long); hp[hot_idx[t]]=torch.arange(nh)
    hot_pos[t]=hp.contiguous()
hpp = parr([hot_pos[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])

def head(i):
    cols=[i[t].contiguous().to(torch.int64) for t in range(ntab)]
    return cols, parr([c.data_ptr() for c in cols])

def auc_of(fwd):
    s,l,n=[],[],0
    for X,o,i,T in test_ld:
        if X.shape[0]!=BATCH: continue
        s.append(fwd(X,o,i).numpy().flatten()); l.append(T.numpy().flatten()); n+=1
        if n>=AUC_BATCHES: break
    return roc_auc_score(np.concatenate(l), np.concatenate(s))

ckpts = sorted(glob.glob('models/gpu24/ckpt_it*.pt'),
               key=lambda p: int(re.search(r'it(\d+)', p).group(1)))
ckpts.append('models/gpu24/dlrm_terabyte_24day.pt')
results=[]
for ck in ckpts:
    it = int(re.search(r'it(\d+)', ck).group(1)) if 'ckpt_it' in ck else 2047023
    log(f"=== {os.path.basename(ck)} (it {it:,}) ===")
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    sd = sd['state_dict'] if 'state_dict' in sd else sd
    dlrm.load_state_dict(sd); dlrm.eval(); del sd
    W = [dlrm.emb_l[t].weight.data.contiguous() for t in range(ntab)]
    Wp = parr([w.data_ptr() for w in W])
    def f_fp32(X,o,i):
        cols, ip = head(i)
        lib.fwd_fp32_all(ip, Wp, ntab, BATCH, D, outp)
        x = dlrm.apply_mlp(X, dlrm.bot_l)
        return dlrm.apply_mlp(dlrm.interact_features(x, outs), dlrm.top_l)
    base = auc_of(f_fp32)
    row = dict(iteration=it, ckpt=os.path.basename(ck), baseline_auc=base)
    for mode, block, fn in [('scalar',16,lib.fwd_dc_all), ('perdim',256,lib.fwd_dcpd_all)]:
        hu,hss,hmm,cr,dq,dss,dll = {},{},{},{},{},{},{}
        for t in LARGE:
            a,b,c,order,dd,e,f2 = build(W[t], cold_idx[t], hot_idx[t], mode, block)
            hu[t],hss[t],hmm[t],dq[t],dss[t],dll[t]=a,b,c,dd.contiguous(),e,f2
            nc=len(cold_idx[t])
            rank=torch.empty(nc,dtype=torch.long); rank[order]=torch.arange(nc)
            full=torch.zeros(int(ln_emb[t]),dtype=torch.long); full[cold_idx[t]]=rank
            cr[t]=full.contiguous()
        hup=parr([hu[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
        crp=parr([cr[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
        dqp=parr([dq[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
        hsa=farr([float(hss[t]) if t in LARGE else 0. for t in range(ntab)])
        hma=farr([float(hmm[t]) if t in LARGE else 0. for t in range(ntab)])
        dsa=farr([float(dss[t]) if t in LARGE else 0. for t in range(ntab)])
        dla=farr([float(dll[t]) if t in LARGE else 0. for t in range(ntab)])
        def f_dc(X,o,i,fn=fn,block=block):
            cols, ip = head(i)
            fn(ip, hpp, hup, hsa, hma, crp, dqp, dsa, dla, Wp, is_large,
               block, ntab, BATCH, D, outp)
            x = dlrm.apply_mlp(X, dlrm.bot_l)
            return dlrm.apply_mlp(dlrm.interact_features(x, outs), dlrm.top_l)
        a = auc_of(f_dc)
        row[f'{mode}_auc'] = a
        row[f'{mode}_delta_pct'] = (a-base)*100
    results.append(row)
    json.dump(results, open('results/dc_vs_training_progress.json','w'), indent=1)
    log(f"  base {base:.6f}  scalar {row['scalar_delta_pct']:+.4f}%  perdim256 {row['perdim_delta_pct']:+.4f}%")
log("done -> results/dc_vs_training_progress.json")
