#!/usr/bin/env python3
"""C-fused counterpart to bench_unified_terabyte.py.

Same model, same data, same batch size, same forward structure -- but the
embedding gather for fp32 / DC / SSD all run through fused C (libembfwd.so)
instead of PyTorch tensor ops.  fp32 is fused too, so the baseline is not
handicapped relative to the compressed paths.

env: AUC_BATCHES (2000), TIMING_BATCHES (200), HOT_FRACTIONS, SSD_THREADS, SSD_QD
"""
import os, sys, time, json, ctypes
import numpy as np, torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui'); os.chdir('/home/cc/expr/dlrm_minrui')
os.environ.setdefault('CRITEO_DAYS', '4')
from sklearn.metrics import roc_auc_score
import bench_unified_terabyte as U     # reuse load_tb / build_dc / sort_key / constants

BATCH, D, BS_BLOCK = U.BATCH, U.EMB_DIM, U.BS_BLOCK
ROW_BYTES = U.ROW_BYTES
AUC_BATCHES    = int(os.environ.get('AUC_BATCHES', 2000))
TIMING_BATCHES = int(os.environ.get('TIMING_BATCHES', 200))
HOT_FRACTIONS  = [float(x) for x in os.environ.get('HOT_FRACTIONS','0.043,0.02,0.01,0.005').split(',')]
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.0f}s] {m}", flush=True)

P  = lambda t, c: t.data_ptr()
def parr(ptrs):        # build a C array of pointers
    a = (ctypes.c_void_p * len(ptrs))(*[ctypes.c_void_p(p) for p in ptrs]); return a
def farr(v): return (ctypes.c_float * len(v))(*v)
def iarr(v): return (ctypes.c_int * len(v))(*v)
def larr(v): return (ctypes.c_long * len(v))(*v)

def main():
    lib = ctypes.CDLL('./libembfwd.so')
    log("loading model + data ...")
    dlrm, test_ld, ln_emb = U.load_tb()
    sd = torch.load(U.MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    dlrm.load_state_dict(sd); dlrm.eval(); torch.set_grad_enabled(False)
    ntab = len(ln_emb)
    LARGE = set(i for i, n in enumerate(ln_emb) if n >= U.LARGE_THRESH)
    W = [dlrm.emb_l[t].weight.data.contiguous() for t in range(ntab)]
    Wp = parr([w.data_ptr() for w in W])
    is_large = iarr([1 if t in LARGE else 0 for t in range(ntab)])
    total_mb = sum(int(ln_emb[i]) * D * 4 for i in range(ntab)) / 1024**2
    small_u8 = sum(int(ln_emb[i]) * D for i in range(ntab) if i not in LARGE) / 1024**2
    log(f"tables={ntab} large={sorted(LARGE)} fp32={total_mb:.0f} MB")

    outs = [torch.empty((BATCH, D), dtype=torch.float32) for _ in range(ntab)]
    outp = parr([o.data_ptr() for o in outs])
    results = []

    def run_head(idx):
        """build the C index pointer array for this batch"""
        cols = [idx[t].contiguous().to(torch.int64) for t in range(ntab)]
        return cols, parr([c.data_ptr() for c in cols])

    def mk_fp32():
        def f(X, o, i):
            cols, ip = run_head(i)
            lib.fwd_fp32_all(ip, Wp, ntab, BATCH, D, outp)
            x = dlrm.apply_mlp(X, dlrm.bot_l)
            z = dlrm.interact_features(x, outs)
            return dlrm.apply_mlp(z, dlrm.top_l)
        return f

    def mk_dc(hot_pos, hot_u8, hs, hmn, cold_rank, dcq, ds, dlo):
        hp = parr([hot_pos[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
        hu = parr([hot_u8[t].data_ptr()  if t in LARGE else 0 for t in range(ntab)])
        cr = parr([cold_rank[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
        dq = parr([dcq[t].data_ptr()     if t in LARGE else 0 for t in range(ntab)])
        hsa = farr([float(hs[t])  if t in LARGE else 0.0 for t in range(ntab)])
        hma = farr([float(hmn[t]) if t in LARGE else 0.0 for t in range(ntab)])
        dsa = farr([float(ds[t])  if t in LARGE else 0.0 for t in range(ntab)])
        dla = farr([float(dlo[t]) if t in LARGE else 0.0 for t in range(ntab)])
        def f(X, o, i):
            cols, ip = run_head(i)
            lib.fwd_dc_all(ip, hp, hu, hsa, hma, cr, dq, dsa, dla, Wp, is_large,
                           BS_BLOCK, ntab, BATCH, D, outp)
            x = dlrm.apply_mlp(X, dlrm.bot_l)
            z = dlrm.interact_features(x, outs)
            return dlrm.apply_mlp(z, dlrm.top_l)
        return f

    def mk_ssd(hot_pos, hot_u8, hs, hmn, base):
        hp = parr([hot_pos[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
        hu = parr([hot_u8[t].data_ptr()  if t in LARGE else 0 for t in range(ntab)])
        hsa = farr([float(hs[t])  if t in LARGE else 0.0 for t in range(ntab)])
        hma = farr([float(hmn[t]) if t in LARGE else 0.0 for t in range(ntab)])
        ba  = larr([int(base.get(t, 0)) for t in range(ntab)])
        def f(X, o, i):
            cols, ip = run_head(i)
            lib.fwd_ssd_all(ip, hp, hu, hsa, hma, ba, Wp, is_large, ntab, BATCH, D, outp)
            x = dlrm.apply_mlp(X, dlrm.bot_l)
            z = dlrm.interact_features(x, outs)
            return dlrm.apply_mlp(z, dlrm.top_l)
        return f

    def measure(name, hf, mb, fwd):
        it = iter(test_ld)
        nb = 0
        for _ in range(20):
            X, o, i, T = next(it)
            if X.shape[0] != BATCH: continue
            fwd(X, o, i)
        times = []
        while len(times) < TIMING_BATCHES:
            X, o, i, T = next(it)
            if X.shape[0] != BATCH: continue
            t = time.perf_counter(); fwd(X, o, i); times.append((time.perf_counter()-t)*1000)
        times = np.array(times)
        s, l, n = [], [], 0
        for X, o, i, T in test_ld:
            if X.shape[0] != BATCH: continue
            s.append(fwd(X, o, i).numpy().flatten()); l.append(T.numpy().flatten())
            n += 1
            if n >= AUC_BATCHES: break
        auc = roc_auc_score(np.concatenate(l), np.concatenate(s))
        r = dict(config=name, hot_frac=hf, mem_mb=mb, auc=auc, mean_ms=float(times.mean()),
                 p50_ms=float(np.percentile(times,50)), p99_ms=float(np.percentile(times,99)),
                 timing_batches=TIMING_BATCHES, auc_samples=n*BATCH, impl='C-fused')
        results.append(r); json.dump(results, open('results/cpp_terabyte.json','w'), indent=1)
        log(f"{name:<24} hf={str(hf):>6} {mb:>8.1f}MB AUC {auc:.6f} "
            f"fwd mean {times.mean():6.2f} p50 {np.percentile(times,50):6.2f} p99 {np.percentile(times,99):6.2f}")

    measure('fp32 (C fused)', None, total_mb, mk_fp32())

    Q, qs, qm = {}, {}, {}
    for t in LARGE:
        w = W[t]; mn = w.min(); sc = (w.max() - mn) / 255.0
        if sc == 0: sc = torch.tensor(1.0)
        Q[t] = ((w - mn) / sc).round().clamp(0, 255).to(torch.uint8).contiguous()
        qs[t], qm[t] = sc, mn
    qp  = parr([Q[t].data_ptr() if t in LARGE else 0 for t in range(ntab)])
    qsa = farr([float(qs[t]) if t in LARGE else 0.0 for t in range(ntab)])
    qma = farr([float(qm[t]) if t in LARGE else 0.0 for t in range(ntab)])
    def mk_int8():
        def f(X, o, i):
            cols, ip = run_head(i)
            lib.fwd_int8_all(ip, qp, qsa, qma, Wp, is_large, ntab, BATCH, D, outp)
            x = dlrm.apply_mlp(X, dlrm.bot_l)
            z = dlrm.interact_features(x, outs)
            return dlrm.apply_mlp(z, dlrm.top_l)
        return f
    measure('INT8 (C fused)', None,
            small_u8 + sum(int(ln_emb[t])*D for t in LARGE)/1024**2, mk_int8())

    log("profiling access frequency ...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in LARGE}
    n = 0
    for X, o, i, T in test_ld:
        for t in LARGE:
            freq[t].scatter_add_(0, i[t].flatten().long(), torch.ones(i[t].numel(), dtype=torch.long))
        n += 1
        if n >= 500: break

    base, off = {}, 0
    for t in sorted(LARGE):
        base[t] = off; off += int(ln_emb[t]) * ROW_BYTES
    lib.emb_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    log(f"emb_open rc={lib.emb_open(U.COLD1.encode(), U.COLD2.encode(), int(os.environ.get('SSD_THREADS',8)), int(os.environ.get('SSD_QD',128)), ROW_BYTES)}")

    for hf in HOT_FRACTIONS:
        hot_pos, cold_rank, hot_idx, cold_idx = {}, {}, {}, {}
        for t in LARGE:
            nrow = int(ln_emb[t]); nh = max(1, int(nrow*hf))
            order = freq[t].argsort(descending=True)
            hot_idx[t], cold_idx[t] = order[:nh], order[nh:]
            hp = torch.full((nrow,), -1, dtype=torch.long); hp[hot_idx[t]] = torch.arange(nh)
            hot_pos[t] = hp.contiguous()
        hot_mb  = sum(len(hot_idx[t])*D for t in LARGE)/1024**2
        bmap_mb = sum(int(ln_emb[t]) for t in LARGE)/8/1024**2
        nblk    = sum((len(cold_idx[t])+BS_BLOCK-1)//BS_BLOCK for t in LARGE)
        dc_mb   = nblk*4/8/1024**2

        hu, hss, hmm, cr, dq, dss, dll = {}, {}, {}, {}, {}, {}, {}
        for t in LARGE:
            a,b,c,order,d,e,f2 = U.build_dc(W[t], cold_idx[t], hot_idx[t], 'value')
            hu[t],hss[t],hmm[t],dq[t],dss[t],dll[t] = a.contiguous(),b,c,d.contiguous(),e,f2
            nc = len(cold_idx[t])
            rank = torch.empty(nc, dtype=torch.long); rank[order] = torch.arange(nc)
            full = torch.zeros(int(ln_emb[t]), dtype=torch.long); full[cold_idx[t]] = rank
            cr[t] = full.contiguous()
        measure('DC value-sort (C fused)', hf, hot_mb+bmap_mb+small_u8+dc_mb,
                mk_dc(hot_pos, hu, hss, hmm, cr, dq, dss, dll))
        measure('SSD cold (C fused)', hf, hot_mb+bmap_mb+small_u8,
                mk_ssd(hot_pos, hu, hss, hmm, base))

    lib.emb_close()
    log("done -> results/cpp_terabyte.json")

main()
