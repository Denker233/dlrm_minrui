#!/usr/bin/env python3
"""Does per-dim DC need value-sort? Ablation on the 24-day model: perdim bs256 @1%,
value-sorted vs unsorted (original frequency order), AUC on 4.1M day-23 samples."""
import os, sys, time, json, ctypes
import numpy as np, torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui'); os.chdir('/home/cc/expr/dlrm_minrui')
os.environ.setdefault('CRITEO_DAYS','24')
from sklearn.metrics import roc_auc_score
import bench_unified_terabyte as U
BATCH, D, BLOCK, HF = U.BATCH, U.EMB_DIM, 256, 0.01
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
def parr(p): return (ctypes.c_void_p*len(p))(*[ctypes.c_void_p(x) for x in p])
def farr(v): return (ctypes.c_float*len(v))(*v)
def iarr(v): return (ctypes.c_int*len(v))(*v)
lib = ctypes.CDLL('./libembfwd.so')
dlrm, test_ld, ln_emb = U.load_tb()
sd = torch.load('models/gpu24/dlrm_terabyte_24day.pt', map_location='cpu', weights_only=False)['state_dict']
dlrm.load_state_dict(sd); dlrm.eval(); torch.set_grad_enabled(False)
ntab=len(ln_emb); LARGE=set(t for t,n in enumerate(ln_emb) if n>=U.LARGE_THRESH)
W=[dlrm.emb_l[t].weight.data.contiguous() for t in range(ntab)]
Wp=parr([w.data_ptr() for w in W]); is_large=iarr([1 if t in LARGE else 0 for t in range(ntab)])
outs=[torch.empty((BATCH,D),dtype=torch.float32) for _ in range(ntab)]
outp=parr([o.data_ptr() for o in outs])
freq={t: torch.zeros(int(ln_emb[t]),dtype=torch.long) for t in LARGE}
n=0
for X,o,i,T in test_ld:
    for t in LARGE: freq[t].scatter_add_(0, i[t].flatten().long(), torch.ones(i[t].numel(),dtype=torch.long))
    n+=1
    if n>=500: break
hot_idx,cold_idx,hot_pos={},{},{}
for t in LARGE:
    nrow=int(ln_emb[t]); nh=max(1,int(nrow*HF))
    order=freq[t].argsort(descending=True)
    hot_idx[t],cold_idx[t]=order[:nh],order[nh:]
    hp=torch.full((nrow,),-1,dtype=torch.long); hp[hot_idx[t]]=torch.arange(nh); hot_pos[t]=hp.contiguous()
hpp=parr([hot_pos[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
def head(i):
    cols=[i[t].contiguous().to(torch.int64) for t in range(ntab)]
    return cols, parr([c.data_ptr() for c in cols])
def auc_of(fwd):
    s,l,n=[],[],0
    for X,o,i,T in test_ld:
        if X.shape[0]!=BATCH: continue
        s.append(fwd(X,o,i).numpy().flatten()); l.append(T.numpy().flatten()); n+=1
        if n>=2000: break
    return roc_auc_score(np.concatenate(l),np.concatenate(s))
def build_pd(w, cold, hot, sort):
    hw=w[hot]; hmn=hw.min(); hs=(hw.max()-hmn)/255.0
    if hs==0: hs=torch.tensor(1.0)
    hu=((hw-hmn)/hs).round().clamp(0,255).to(torch.uint8).contiguous()
    c=w[cold]
    order = torch.argsort(c.mean(dim=1)) if sort else torch.arange(c.shape[0])
    cs=c[order]; n=cs.shape[0]; nb=(n+BLOCK-1)//BLOCK; pad=nb*BLOCK-n
    if pad: cs=torch.cat([cs,torch.zeros(pad,D)],0)
    means=cs.view(nb,BLOCK,D).mean(dim=1)
    lo,hi=means.min(),means.max(); s2=(hi-lo)/15
    if s2==0: s2=torch.tensor(1.0)
    q=((means-lo)/s2).round().clamp(0,15).to(torch.uint8)
    packed=((q[:,0::2]<<4)|(q[:,1::2])).contiguous()
    return hu,hs,hmn,order,packed,s2,lo
results={}
for sort in (True, False):
    hu,hss,hmm,cr,dq,dss,dll={},{},{},{},{},{},{}
    for t in LARGE:
        a,b,c,order,d,e,f2=build_pd(W[t],cold_idx[t],hot_idx[t],sort)
        hu[t],hss[t],hmm[t],dq[t],dss[t],dll[t]=a,b,c,d,e,f2
        nc=len(cold_idx[t]); rank=torch.empty(nc,dtype=torch.long); rank[order]=torch.arange(nc)
        full=torch.zeros(int(ln_emb[t]),dtype=torch.long); full[cold_idx[t]]=rank; cr[t]=full.contiguous()
    hup=parr([hu[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
    crp=parr([cr[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
    dqp=parr([dq[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
    hsa=farr([float(hss[t]) if t in LARGE else 0. for t in range(ntab)])
    hma=farr([float(hmm[t]) if t in LARGE else 0. for t in range(ntab)])
    dsa=farr([float(dss[t]) if t in LARGE else 0. for t in range(ntab)])
    dla=farr([float(dll[t]) if t in LARGE else 0. for t in range(ntab)])
    def f(X,o,i):
        cols,ip=head(i)
        lib.fwd_dcpd_all(ip,hpp,hup,hsa,hma,crp,dqp,dsa,dla,Wp,is_large,BLOCK,ntab,BATCH,D,outp)
        x=dlrm.apply_mlp(X,dlrm.bot_l)
        return dlrm.apply_mlp(dlrm.interact_features(x,outs),dlrm.top_l)
    a=auc_of(f); results['value-sorted' if sort else 'unsorted']=a
    log(f"perdim bs256 @1% {'value-sorted' if sort else 'unsorted (id/freq order)':<28} AUC {a:.6f}")
json.dump(results, open('results/perdim_order_ablation.json','w'), indent=1)
log("done")
