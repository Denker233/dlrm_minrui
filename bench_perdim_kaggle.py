#!/usr/bin/env python3
"""Per-dimension vs scalar DC block means on Criteo Kaggle (D=16).

Cross-dataset check of the Terabyte finding. The error decomposition predicts a
SMALLER per-dim win here: the ordering-dependent (between-row) term is 2.8% of the
error at D=16 versus 0.3% at D=64, so scalar means already capture relatively more.
Same C-fused gathers (libembfwd.so) as the Terabyte sweep.
"""
import os, sys, time, json, ctypes
import numpy as np, torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui'); os.chdir('/home/cc/expr/dlrm_minrui')
from sklearn.metrics import roc_auc_score
from bench_kaggle_dc import load_kaggle, MODEL_PATH

D = 16
LARGE_THRESH = 50000
HOT_FRACTIONS = [float(x) for x in os.environ.get('HOT_FRACTIONS','0.043,0.02,0.01,0.005').split(',')]
VARIANTS = [('scalar',16), ('perdim',16), ('perdim',64), ('perdim',256)]
AUC_BATCHES    = int(os.environ.get('AUC_BATCHES', 0))     # 0 = full test set
TIMING_BATCHES = int(os.environ.get('TIMING_BATCHES', 200))
MAXB = 32768
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.0f}s] {m}", flush=True)
def parr(p): return (ctypes.c_void_p*len(p))(*[ctypes.c_void_p(x) for x in p])
def farr(v): return (ctypes.c_float*len(v))(*v)
def iarr(v): return (ctypes.c_int*len(v))(*v)

def build(w, cold_idx, hot_idx, mode, block, nbits=4):
    hw = w[hot_idx]; hmn = hw.min(); hs = (hw.max()-hmn)/255.0
    if hs == 0: hs = torch.tensor(1.0)
    hot_u8 = ((hw-hmn)/hs).round().clamp(0,255).to(torch.uint8).contiguous()
    cold = w[cold_idx]
    order = torch.argsort(cold.mean(dim=1))              # value-sort (proved optimal)
    cs = cold[order]
    n = cs.shape[0]; nb = (n+block-1)//block
    pad = nb*block - n
    if pad: cs = torch.cat([cs, torch.zeros(pad, D)], 0)
    r = cs.view(nb, block, D)
    means = r.mean(dim=(1,2)) if mode=='scalar' else r.mean(dim=1)
    lo, hi = means.min(), means.max(); lv = (1<<nbits)-1
    ds = (hi-lo)/lv
    if ds == 0: ds = torch.tensor(1.0)
    q = ((means-lo)/ds).round().clamp(0,lv).to(torch.uint8)
    packed = q.contiguous() if mode=='scalar' else ((q[:,0::2]<<4)|q[:,1::2]).contiguous()
    return hot_u8, hs, hmn, order, packed, ds, lo

def main():
    lib = ctypes.CDLL('./libembfwd.so')
    log("loading Kaggle model + data ...")
    dlrm, test_ld, train_ld, ln_emb = load_kaggle()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    sd = sd['state_dict'] if 'state_dict' in sd else sd
    ntab = len(ln_emb)
    LARGE = set(i for i,n in enumerate(ln_emb) if n >= LARGE_THRESH)
    W = [dlrm.emb_l[t].weight.data.contiguous() for t in range(ntab)]
    Wp = parr([w.data_ptr() for w in W]); is_large = iarr([1 if t in LARGE else 0 for t in range(ntab)])
    total_mb = sum(int(ln_emb[i])*D*4 for i in range(ntab))/1024**2
    small_u8 = sum(int(ln_emb[i])*D for i in range(ntab) if i not in LARGE)/1024**2
    outs = [torch.empty((MAXB, D), dtype=torch.float32) for _ in range(ntab)]
    outp = parr([o.data_ptr() for o in outs])
    torch.set_grad_enabled(False)
    log(f"tables={ntab} large={sorted(LARGE)} fp32={total_mb:.0f} MB")
    results = []

    def head(i):
        cols=[i[t].contiguous().to(torch.int64) for t in range(ntab)]
        return cols, parr([c.data_ptr() for c in cols])

    def measure(name, hf, mb, fwd):
        it = iter(test_ld); times=[]
        for _ in range(10):
            try: X,o,i,T = next(it)
            except StopIteration: break
            fwd(X,o,i,X.shape[0])
        while len(times) < TIMING_BATCHES:
            try: X,o,i,T = next(it)
            except StopIteration: break
            t=time.perf_counter(); fwd(X,o,i,X.shape[0]); times.append((time.perf_counter()-t)*1000)
        times=np.array(times); s,l,n=[],[],0
        for X,o,i,T in test_ld:
            s.append(fwd(X,o,i,X.shape[0]).numpy().flatten()); l.append(T.numpy().flatten()); n+=1
            if AUC_BATCHES and n>=AUC_BATCHES: break
        auc=roc_auc_score(np.concatenate(l), np.concatenate(s))
        results.append(dict(config=name, hot_frac=hf, mem_mb=mb, auc=auc,
            mean_ms=float(times.mean()), p99_ms=float(np.percentile(times,99)),
            auc_samples=int(sum(len(x) for x in l))))
        json.dump(results, open('results/perdim_kaggle.json','w'), indent=1)
        log(f"{name:<30} hf={str(hf):>6} {mb:>8.2f}MB AUC {auc:.6f} fwd {times.mean():6.2f} p99 {np.percentile(times,99):6.2f}")

    def mk_fp32():
        def f(X,o,i,bs):
            cols, ip = head(i)
            lib.fwd_fp32_all(ip, Wp, ntab, bs, D, outp)
            x = dlrm.apply_mlp(X, dlrm.bot_l)
            return dlrm.apply_mlp(dlrm.interact_features(x, [t[:bs] for t in outs]), dlrm.top_l)
        return f
    measure('fp32 (C fused)', None, total_mb, mk_fp32())

    log("profiling access frequency ...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in LARGE}
    n=0
    for X,o,i,T in test_ld:
        for t in LARGE:
            freq[t].scatter_add_(0, i[t].flatten().long(), torch.ones(i[t].numel(), dtype=torch.long))
        n+=1
        if n>=500: break

    for hf in HOT_FRACTIONS:
        hot_idx, cold_idx, hot_pos = {},{},{}
        for t in LARGE:
            nrow=int(ln_emb[t]); nh=max(1,int(nrow*hf))
            order=freq[t].argsort(descending=True)
            hot_idx[t], cold_idx[t] = order[:nh], order[nh:]
            hp=torch.full((nrow,),-1,dtype=torch.long); hp[hot_idx[t]]=torch.arange(nh)
            hot_pos[t]=hp.contiguous()
        hot_mb = sum(len(hot_idx[t])*D for t in LARGE)/1024**2
        bmap_mb= sum(int(ln_emb[t]) for t in LARGE)/8/1024**2
        for mode, block in VARIANTS:
            hu,hss,hmm,cr,dq,dss,dll = {},{},{},{},{},{},{}
            nblk=0
            for t in LARGE:
                a,b,c,order,d,e,f2 = build(W[t], cold_idx[t], hot_idx[t], mode, block)
                hu[t],hss[t],hmm[t],dq[t],dss[t],dll[t]=a,b,c,d.contiguous(),e,f2
                nc=len(cold_idx[t]); nblk += (nc+block-1)//block
                rank=torch.empty(nc,dtype=torch.long); rank[order]=torch.arange(nc)
                full=torch.zeros(int(ln_emb[t]),dtype=torch.long); full[cold_idx[t]]=rank
                cr[t]=full.contiguous()
            dc_mb = nblk*(0.5 if mode=='scalar' else D*0.5)/1024**2
            hp=parr([hot_pos[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
            hup=parr([hu[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
            crp=parr([cr[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
            dqp=parr([dq[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
            hsa=farr([float(hss[t]) if t in LARGE else 0. for t in range(ntab)])
            hma=farr([float(hmm[t]) if t in LARGE else 0. for t in range(ntab)])
            dsa=farr([float(dss[t]) if t in LARGE else 0. for t in range(ntab)])
            dla=farr([float(dll[t]) if t in LARGE else 0. for t in range(ntab)])
            fn = lib.fwd_dc_all if mode=='scalar' else lib.fwd_dcpd_all
            def mk(fn=fn, block=block):
                def f(X,o,i,bs):
                    cols, ip = head(i)
                    fn(ip, hp, hup, hsa, hma, crp, dqp, dsa, dla, Wp, is_large,
                       block, ntab, bs, D, outp)
                    x = dlrm.apply_mlp(X, dlrm.bot_l)
                    return dlrm.apply_mlp(dlrm.interact_features(x, [t[:bs] for t in outs]), dlrm.top_l)
                return f
            bpr = (0.5 if mode=='scalar' else D*0.5)/block
            measure(f'{mode} block={block} 4bit ({bpr:.3f} B/row)', hf,
                    hot_mb+bmap_mb+small_u8+dc_mb, mk())
    log("done -> results/perdim_kaggle.json")

main()
