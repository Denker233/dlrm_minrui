#!/usr/bin/env python3
"""DC inference on the GPU (A100): latency vs fp32/INT8 + GPU memory footprint.

Mirrors the CPU experiment's configs (bench_perdim_terabyte.py):
  fp32                       all 26 tables fp32 CUDA
  INT8                       uint8 + per-table (scale, min), dequant on gather
  DC scalar  block=16 4-bit  value-sorted (hot u8, cold scalar block means)
  DC per-dim block=256 4-bit value-sorted (block means = 64 nibbles / 32 B)
Hot fractions 4.3/2/1/0.5% (LARGE tables only; small tables stay fp32).

Measurement: 20 warmup + 200 timed batches, torch.cuda.synchronize() around
every wall timestamp; CUDA events for the post-H2D region and the gather stage.
AUC anchors (CPU 24-day run, 2000 batches) must reproduce within ~3e-4.

env: AUC_BATCHES (2000), TIMING_BATCHES (200), TB_MODEL, CRITEO_DAYS
Output: results/gpu_inference.json
"""
import os, sys, time, json
import numpy as np, torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui'); os.chdir('/home/cc/expr/dlrm_minrui')
os.environ.setdefault('CRITEO_DAYS', '24')
from sklearn.metrics import roc_auc_score
import bench_unified_terabyte as U
from bench_perdim_terabyte import build   # __main__-guarded, import is safe

BATCH, D = U.BATCH, U.EMB_DIM
AUC_BATCHES    = int(os.environ.get('AUC_BATCHES', 2000))
TIMING_BATCHES = int(os.environ.get('TIMING_BATCHES', 200))
WARMUP         = 20
HOT_FRACTIONS  = [0.043, 0.02, 0.01, 0.005]
DEV = torch.device('cuda')
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.0f}s] {m}", flush=True)

ANCHORS = {  # config-key -> CPU 24-day AUC @2000 batches
    ('fp32', None): 0.798717,
    ('int8', None): 0.798712,
    ('dc_scalar_bs16', 0.01): 0.797366,
    ('dc_perdim_bs256', 0.01): 0.798121,
    ('dc_perdim_bs256', 0.005): 0.797800,
}
TOL = 3e-4


def main():
    log("loading model + data ...")
    dlrm, test_ld, ln_emb = U.load_tb()
    sd = torch.load(U.MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    dlrm.load_state_dict(sd); dlrm.eval(); torch.set_grad_enabled(False)
    ntab = len(ln_emb)
    LARGE = sorted(i for i, n in enumerate(ln_emb) if n >= U.LARGE_THRESH)
    LSET = set(LARGE)
    W = [dlrm.emb_l[t].weight.data.contiguous() for t in range(ntab)]

    # MLPs to GPU once (shared by every config; counted separately in footprint)
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    mlp_base = torch.cuda.memory_allocated()
    dlrm.bot_l.to(DEV); dlrm.top_l.to(DEV)
    torch.cuda.synchronize()
    mlp_mb = (torch.cuda.memory_allocated() - mlp_base) / 2**20
    log(f"MLP params on GPU: {mlp_mb:.1f} MB")

    # small tables stay fp32 in every config
    small_gpu = {t: W[t].to(DEV) for t in range(ntab) if t not in LSET}
    small_mb = sum(w.numel() * 4 for w in small_gpu.values()) / 2**20

    # frequency profile — same as CPU code (500 test batches, scatter_add)
    log("profiling access frequency (500 batches) ...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in LARGE}
    n = 0
    for X, o, i, T in test_ld:
        for t in LARGE:
            freq[t].scatter_add_(0, i[t].flatten().long(),
                                 torch.ones(i[t].numel(), dtype=torch.long))
        n += 1
        if n >= 500: break

    results = []

    def gather_small(i_gpu, out):
        for t, w in small_gpu.items():
            out[t] = w[i_gpu[t]]

    def measure(name, hf, storage_mb, idx_mb, fwd):
        """fwd(X_gpu, i_gpu, ev) -> scores tensor on GPU.
        ev = (gather_start, gather_end) CUDA events fwd must record around the
        embedding-gather stage."""
        it = iter(test_ld)
        # -- warmup
        nb = 0
        while nb < WARMUP:
            X, o, i, T = next(it)
            if X.shape[0] != BATCH: continue
            ev = (torch.cuda.Event(True), torch.cuda.Event(True))
            fwd(X.to(DEV), i.to(DEV), ev)
            nb += 1
        torch.cuda.synchronize()
        # -- timing
        e2e, gpu_ms, gat_ms = [], [], []
        while len(e2e) < TIMING_BATCHES:
            X, o, i, T = next(it)
            if X.shape[0] != BATCH: continue
            g0, g1 = torch.cuda.Event(True), torch.cuda.Event(True)
            c0, c1 = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            Xg = X.to(DEV); ig = i.to(DEV)          # H2D
            c0.record()
            z = fwd(Xg, ig, (g0, g1))
            c1.record()
            torch.cuda.synchronize()
            e2e.append((time.perf_counter() - t0) * 1000)
            gpu_ms.append(c0.elapsed_time(c1))
            gat_ms.append(g0.elapsed_time(g1))
        # -- AUC
        s, l, n = [], [], 0
        for X, o, i, T in test_ld:
            if X.shape[0] != BATCH: continue
            ev = (torch.cuda.Event(True), torch.cuda.Event(True))
            z = fwd(X.to(DEV), i.to(DEV), ev)
            s.append(z.float().cpu().numpy().flatten()); l.append(T.numpy().flatten())
            n += 1
            if n >= AUC_BATCHES: break
        auc = roc_auc_score(np.concatenate(l), np.concatenate(s))
        stat = lambda a: dict(mean=float(np.mean(a)), p50=float(np.percentile(a, 50)),
                              p99=float(np.percentile(a, 99)))
        total_mb = storage_mb + idx_mb + small_mb + mlp_mb
        r = dict(config=name, hot_frac=hf,
                 storage_mb=round(storage_mb, 2), idx_mb=round(idx_mb, 2),
                 small_fp32_mb=round(small_mb, 2), mlp_mb=round(mlp_mb, 2),
                 total_mb=round(total_mb, 2),
                 models_in_80gb=int(80 * 1024 // total_mb),
                 auc=auc, auc_samples=n * BATCH,
                 e2e_ms=stat(e2e), gpu_ms=stat(gpu_ms), gather_ms=stat(gat_ms),
                 timing_batches=TIMING_BATCHES)
        key = (name.split(' @')[0], hf)
        if key in ANCHORS:
            d = auc - ANCHORS[key]
            r['anchor'] = ANCHORS[key]; r['anchor_delta'] = float(d)
            r['anchor_ok'] = bool(abs(d) <= TOL)
            log(f"  ANCHOR {key}: got {auc:.6f} want {ANCHORS[key]:.6f} "
                f"delta {d:+.6f} -> {'OK' if r['anchor_ok'] else 'FAIL'}")
        results.append(r)
        os.makedirs('results', exist_ok=True)
        json.dump(results, open('results/gpu_inference.json', 'w'), indent=1)
        log(f"{name:<26} hf={str(hf):>6} tot {total_mb:8.1f}MB AUC {auc:.6f} "
            f"e2e {r['e2e_ms']['mean']:.3f} gpu {r['gpu_ms']['mean']:.3f} "
            f"gather {r['gather_ms']['mean']:.3f} (p99 e2e {r['e2e_ms']['p99']:.3f})")

    def head_tail(Xg, outs, ev_pair):
        x = dlrm.apply_mlp(Xg, dlrm.bot_l)
        z = dlrm.interact_features(x, outs)
        return dlrm.apply_mlp(z, dlrm.top_l)

    # ---------------- fp32 ----------------
    torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated()
    big_fp32 = {t: W[t].to(DEV) for t in LARGE}
    torch.cuda.synchronize()
    fp32_mb = (torch.cuda.memory_allocated() - m0) / 2**20

    def fwd_fp32(Xg, ig, ev):
        outs = [None] * ntab
        ev[0].record()
        gather_small(ig, outs)
        for t in LARGE:
            outs[t] = big_fp32[t][ig[t]]
        ev[1].record()
        return head_tail(Xg, outs, ev)
    measure('fp32', None, fp32_mb, 0.0, fwd_fp32)

    # ---------------- INT8 ----------------
    Q, qs, qm = {}, {}, {}
    for t in LARGE:
        w = W[t]; mn = w.min(); sc = (w.max() - mn) / 255.0
        if sc == 0: sc = torch.tensor(1.0)
        Q[t] = ((w - mn) / sc).round().clamp(0, 255).to(torch.uint8)
        qs[t], qm[t] = float(sc), float(mn)
    torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated()
    Qg = {t: Q[t].to(DEV) for t in LARGE}
    torch.cuda.synchronize()
    int8_mb = (torch.cuda.memory_allocated() - m0) / 2**20

    def fwd_int8(Xg, ig, ev):
        outs = [None] * ntab
        ev[0].record()
        gather_small(ig, outs)
        for t in LARGE:
            outs[t] = Qg[t][ig[t]].float() * qs[t] + qm[t]
        ev[1].record()
        return head_tail(Xg, outs, ev)
    measure('int8', None, int8_mb, 0.0, fwd_int8)
    del big_fp32; torch.cuda.empty_cache()

    # ---------------- DC variants ----------------
    for hf in HOT_FRACTIONS:
        hot_idx, cold_idx, hot_pos32 = {}, {}, {}
        for t in LARGE:
            nrow = int(ln_emb[t]); nh = max(1, int(nrow * hf))
            order = freq[t].argsort(descending=True)
            hot_idx[t], cold_idx[t] = order[:nh], order[nh:]
            hp = torch.full((nrow,), -1, dtype=torch.int32)
            hp[hot_idx[t]] = torch.arange(nh, dtype=torch.int32)
            hot_pos32[t] = hp
        for mode, block in (('scalar', 16), ('perdim', 256)):
            hu, hss, hmm, dq, cr32 = {}, {}, {}, {}, {}
            dss, dll = {}, {}
            for t in LARGE:
                a, b, c, order, d, e, f2 = build(W[t], cold_idx[t], hot_idx[t], mode, block)
                hu[t], hss[t], hmm[t] = a, float(b), float(c)
                dq[t], dss[t], dll[t] = d.contiguous(), float(e), float(f2)
                nc = len(cold_idx[t])
                rank = torch.empty(nc, dtype=torch.int32)
                rank[order] = torch.arange(nc, dtype=torch.int32)
                full = torch.zeros(int(ln_emb[t]), dtype=torch.int32)
                full[cold_idx[t]] = rank
                cr32[t] = full
            # ---- fuse all large tables into single concatenated tensors so the
            # whole DC gather is ~15 kernel launches instead of ~10 x 21.
            # Branchless: compute hot and cold values for every index, select
            # with torch.where (no data-dependent shapes).
            nL = len(LARGE)
            nh_l   = [len(hot_idx[t]) for t in LARGE]
            nblk_l = [(len(cold_idx[t]) + block - 1) // block for t in LARGE]
            row_off  = np.cumsum([0] + [int(ln_emb[t]) for t in LARGE])[:-1]
            hot_base = np.cumsum([0] + nh_l)[:-1]
            blk_base = np.cumsum([0] + nblk_l)[:-1]
            hp_cat_l = []
            for k, t in enumerate(LARGE):
                hp = hot_pos32[t].clone()
                m = hp >= 0
                hp[m] += int(hot_base[k])
                hp_cat_l.append(hp)
            torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated()
            hu_cat = torch.cat([hu[t] for t in LARGE], 0).to(DEV)          # (sum_nh, D) u8
            dq_cat = torch.cat([dq[t] for t in LARGE], 0).to(DEV)          # (sum_nb,) or (sum_nb, D/2)
            torch.cuda.synchronize()
            storage_mb = (torch.cuda.memory_allocated() - m0) / 2**20
            torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated()
            hp_cat = torch.cat(hp_cat_l, 0).to(DEV)                        # (sum_nrow,) i32
            cr_cat = torch.cat([cr32[t] for t in LARGE], 0).to(DEV)        # (sum_nrow,) i32 local ranks
            torch.cuda.synchronize()
            idx_mb = (torch.cuda.memory_allocated() - m0) / 2**20
            large_rows = torch.tensor(LARGE, dtype=torch.long, device=DEV)
            row_off_g  = torch.tensor(row_off,  dtype=torch.long, device=DEV).view(nL, 1)
            blk_base_g = torch.tensor(blk_base, dtype=torch.long, device=DEV).view(nL, 1)
            hs_v  = torch.tensor([hss[t] for t in LARGE], device=DEV).view(nL, 1, 1)
            hmn_v = torch.tensor([hmm[t] for t in LARGE], device=DEV).view(nL, 1, 1)
            ds_v  = torch.tensor([dss[t] for t in LARGE], device=DEV).view(nL, 1, 1)
            dlo_v = torch.tensor([dll[t] for t in LARGE], device=DEV).view(nL, 1, 1)

            def fwd_dc(Xg, ig, ev, mode=mode, block=block,
                       hu_cat=hu_cat, dq_cat=dq_cat, hp_cat=hp_cat, cr_cat=cr_cat,
                       large_rows=large_rows, row_off_g=row_off_g, blk_base_g=blk_base_g,
                       hs_v=hs_v, hmn_v=hmn_v, ds_v=ds_v, dlo_v=dlo_v, nL=nL):
                outs = [None] * ntab
                ev[0].record()
                gather_small(ig, outs)
                gid  = ig[large_rows] + row_off_g                     # (nL, B) global row ids
                pos  = hp_cat[gid]                                    # (nL, B) i32
                hotm = pos >= 0
                hv = hu_cat[pos.clamp(min=0).long()].float() * hs_v + hmn_v   # (nL, B, D)
                blk = cr_cat[gid].long() // block + blk_base_g        # (nL, B); hot rows -> rank 0, discarded
                if mode == 'scalar':
                    cv = (dq_cat[blk].float() * ds_v.view(nL, 1) + dlo_v.view(nL, 1)).unsqueeze(2).expand(-1, -1, D)
                else:
                    b = dq_cat[blk]                                   # (nL, B, D/2) u8
                    vals = torch.empty((nL, b.shape[1], D), dtype=torch.float32, device=DEV)
                    vals[:, :, 0::2] = (b >> 4).float()
                    vals[:, :, 1::2] = (b & 15).float()
                    cv = vals * ds_v + dlo_v
                out = torch.where(hotm.unsqueeze(2), hv, cv)          # (nL, B, D)
                for k, t in enumerate(LARGE):
                    outs[t] = out[k]
                ev[1].record()
                return head_tail(Xg, outs, ev)
            measure(f'dc_{mode}_bs{block} @{hf}', hf, storage_mb, idx_mb, fwd_dc)
            del hu_cat, dq_cat, hp_cat, cr_cat; torch.cuda.empty_cache()

    bad = [r for r in results if r.get('anchor_ok') is False]
    log(f"done -> results/gpu_inference.json  anchors: "
        f"{'ALL OK' if not bad else f'{len(bad)} FAILED'}")
    if bad: sys.exit(3)

if __name__ == '__main__':
    main()
