#!/usr/bin/env python3
"""Unified Terabyte comparison. Every number measured on THIS machine, one run.

For each configuration we report FULL FORWARD time per batch (bottom MLP +
embedding lookup + interaction + top MLP, including any disk I/O) as
mean/p50/p99, AND the AUC -- through the *same* code path, so the only thing
that differs between configs is where embedding rows come from.

Configs:
  fp32            all embeddings fp32 in DRAM (baseline)
  INT8 / INT4     whole-table quantize-dequantize, no hot/cold split
  zero            hot rows kept fp32, cold rows zeroed
  freq/value/pca  DC block-mean 4-bit, cold rows sorted by that key
  ssd             hot rows in DRAM, cold rows read from NVMe per batch via
                  optimised io_uring (O_DIRECT 512B, IOPOLL, registered files,
                  N threads NUMA-pinned, mirrored across both drives)

env: AUC_BATCHES (default 10000 x 2048 = 20.5M samples), TIMING_BATCHES (100),
     SSD_THREADS (8), SSD_QD (128)
"""
import os, sys, time, json, ctypes
import numpy as np, torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
os.environ.setdefault('CRITEO_DAYS', '4')
from sklearn.metrics import roc_auc_score

BATCH        = 2048
EMB_DIM      = 64
ROW_BYTES    = EMB_DIM * 4          # 256 B fp32 row; 2 rows per 512 B sector
BS_BLOCK     = 16                   # rows per DC block
LARGE_THRESH = 50000
HOT_FRACTIONS= [float(x) for x in os.environ.get('HOT_FRACTIONS','0.043,0.02,0.01,0.005').split(',')]
MODEL_PATH   = 'models/dlrm_terabyte_4day.pt'
COLD1        = '/mnt/nvme1/tb_cold.bin'
COLD2        = '/mnt/nvme0/tb_cold.bin'
AUC_BATCHES    = int(os.environ.get('AUC_BATCHES', 10000))
TIMING_BATCHES = int(os.environ.get('TIMING_BATCHES', 100))
SSD_THREADS  = int(os.environ.get('SSD_THREADS', 8))
SSD_QD       = int(os.environ.get('SSD_QD', 128))
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.0f}s] {m}", flush=True)

# ---------------------------------------------------------------- data + model
def load_tb():
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net
    from codec_ondemand_benchmark import create_args
    args = create_args()
    args.arch_sparse_feature_size = EMB_DIM
    args.arch_mlp_bot = "13-512-256-64"
    args.data_set = "terabyte"
    args.raw_data_file = "/home/cc/input/terabyte/day"
    args.processed_data_file = "/home/cc/input/terabyte/terabyte_processed.npz"
    args.memory_map = True
    args.mini_batch_size = BATCH
    args.test_mini_batch_size = BATCH          # time at the batch size the paper quotes
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    ln_emb = np.array(train_data.counts)
    ln_bot = np.array([13, 512, 256, 64]); ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    num_int = (num_fea * (num_fea - 1)) // 2 + ln_bot[-1]
    ln_top = np.fromstring(str(num_int) + "-512-512-256-1", dtype=int, sep="-")
    dlrm = DLRM_Net(EMB_DIM, ln_emb, ln_bot, ln_top, arch_interaction_op="dot",
                    arch_interaction_itself=False, sigmoid_bot=-1,
                    sigmoid_top=ln_top.size - 2, loss_function="bce")
    return dlrm, test_ld, ln_emb

# ---------------------------------------------------------------- forward paths
def fwd_src(dlrm, dense, idx, src, ntab):
    x  = dlrm.apply_mlp(dense, dlrm.bot_l)
    ly = [src.gather(t, idx[t]) for t in range(ntab)]
    z  = dlrm.interact_features(x, ly)
    return dlrm.apply_mlp(z, dlrm.top_l)

class SrcSSD:
    """hot rows uint8 in DRAM (same store as DC); cold rows read exact fp32 from NVMe.
    One io_uring call per batch covering every cold row across all tables."""
    def __init__(self, lib, W, large, hot_pos, hot_u8, hs, hmn, base):
        self.lib, self.W, self.large = lib, W, large
        self.hot_pos, self.hot_u8, self.hs, self.hmn, self.base = hot_pos, hot_u8, hs, hmn, base
        self.buf = None
    def forward(self, dlrm, dense, idx, ntab):
        plan, offs = [], []
        for t in range(ntab):
            if t in self.large:
                pos = self.hot_pos[t][idx[t]]
                cold = pos < 0
                plan.append((t, pos, cold))
                nc = int(cold.sum())
                if nc:
                    offs.append(self.base[t] + idx[t][cold].numpy().astype(np.int64) * ROW_BYTES)
            else:
                plan.append((t, None, None))
        if offs:
            alloff = np.ascontiguousarray(np.concatenate(offs))
            n = len(alloff)
            if self.buf is None or self.buf.size < n * EMB_DIM:
                self.buf = np.empty(n * EMB_DIM, dtype=np.float32)
            self.lib.cold_read(alloff.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
                               ctypes.c_int(n),
                               self.buf.ctypes.data_as(ctypes.POINTER(ctypes.c_char)))
        x = dlrm.apply_mlp(dense, dlrm.bot_l)
        ly, cur = [], 0
        for t, pos, cold in plan:
            if pos is None:
                ly.append(self.W[t][idx[t]]); continue
            out = torch.zeros((idx[t].shape[0], EMB_DIM), dtype=torch.float32)
            hotm = pos >= 0
            if bool(hotm.any()):
                out[hotm] = self.hot_u8[t][pos[hotm]].to(torch.float32) * self.hs[t] + self.hmn[t]
            nc = int(cold.sum())
            if nc:
                rows = self.buf[cur*EMB_DIM:(cur+nc)*EMB_DIM].reshape(nc, EMB_DIM)
                out[cold] = torch.from_numpy(rows.copy())
                cur += nc
            ly.append(out)
        z = dlrm.interact_features(x, ly)
        return dlrm.apply_mlp(z, dlrm.top_l)

# ---------------------------------------------------------------- real storage
class SrcFP32:
    """baseline: fp32 rows in DRAM"""
    def __init__(self, W): self.W = W
    def gather(self, t, idx): return self.W[t][idx]

class SrcQuant:
    """whole-table INT8/INT4: uint8 (or nibble-packed) storage, dequant on gather"""
    def __init__(self, W, large, nbits):
        self.nbits, self.large, self.W = nbits, large, W
        self.q, self.s, self.mn = {}, {}, {}
        lv = (1 << nbits) - 1
        for t in large:
            w = W[t]; mn = w.min(); s = (w.max() - mn) / lv
            if s == 0: s = torch.tensor(1.0)
            q = ((w - mn) / s).round().clamp(0, lv).to(torch.uint8)
            if nbits == 4:                      # pack two values per byte
                q = (q[:, 0::2] << 4) | q[:, 1::2]
            self.q[t], self.s[t], self.mn[t] = q.contiguous(), s, mn
    def gather(self, t, idx):
        if t not in self.large: return self.W[t][idx]
        g = self.q[t][idx]
        if self.nbits == 4:
            hi = (g >> 4).to(torch.float32); lo = (g & 0xF).to(torch.float32)
            out = torch.stack([hi, lo], dim=2).view(g.shape[0], -1)
        else:
            out = g.to(torch.float32)
        return out * self.s[t] + self.mn[t]

class SrcDC:
    """hot rows uint8 + cold rows as 4-bit block means; dequant on gather"""
    def __init__(self, W, large, hot_pos, hot_u8, hs, hmn, cold_rank, dcq, ds, dlo, zero_mode):
        self.W, self.large, self.zero = W, large, zero_mode
        self.hot_pos, self.hot_u8, self.hs, self.hmn = hot_pos, hot_u8, hs, hmn
        self.cold_rank, self.dcq, self.ds, self.dlo = cold_rank, dcq, ds, dlo
    def gather(self, t, idx):
        if t not in self.large: return self.W[t][idx]
        pos = self.hot_pos[t][idx]
        hotm = pos >= 0
        out = torch.zeros((idx.shape[0], EMB_DIM), dtype=torch.float32)
        if bool(hotm.any()):
            out[hotm] = self.hot_u8[t][pos[hotm]].to(torch.float32) * self.hs[t] + self.hmn[t]
        if not self.zero:
            coldm = ~hotm
            if bool(coldm.any()):
                blk = self.cold_rank[t][idx[coldm]] // BS_BLOCK
                v = self.dcq[t][blk].to(torch.float32) * self.ds[t] + self.dlo[t]
                out[coldm] = v.unsqueeze(1).expand(-1, EMB_DIM)
        return out

def build_dc(w, cold_idx, hot_idx, how, nbits=4):
    """returns the real compressed arrays for one table"""
    hw = w[hot_idx]
    hmn = hw.min(); hs = (hw.max() - hmn) / 255.0
    if hs == 0: hs = torch.tensor(1.0)
    hot_u8 = ((hw - hmn) / hs).round().clamp(0, 255).to(torch.uint8).contiguous()
    cold = w[cold_idx]
    order = torch.arange(cold.shape[0]) if how == 'freq' else torch.argsort(sort_key(cold, how))
    cs = cold[order]
    n = cs.shape[0]; nb = (n + BS_BLOCK - 1) // BS_BLOCK
    pad = nb * BS_BLOCK - n
    if pad: cs = torch.cat([cs, torch.zeros(pad, cs.shape[1])], 0)
    means = cs.view(nb, BS_BLOCK, -1).mean(dim=(1, 2))
    lo, hi = means.min(), means.max(); lv = (1 << nbits) - 1
    ds = (hi - lo) / lv
    if ds == 0: ds = torch.tensor(1.0)
    dcq = ((means - lo) / ds).round().clamp(0, lv).to(torch.uint8).contiguous()
    return hot_u8, hs, hmn, order, dcq, ds, lo

# ---------------------------------------------------------------- compressors
def quant_dequant(w, nbits):
    lv = (1 << nbits) - 1
    mn, mx = w.min(), w.max()
    s = (mx - mn) / lv
    if s == 0: s = torch.tensor(1.0)
    return ((w - mn) / s).round().clamp(0, lv) * s + mn

def sort_key(cold, how):
    if how == 'value': return cold.mean(dim=1)
    x = cold - cold.mean(dim=0, keepdim=True)
    g = torch.Generator().manual_seed(0)
    v = torch.randn(x.shape[1], generator=g); v /= v.norm()
    for _ in range(20):
        v = x.T @ (x @ v); v /= (v.norm() + 1e-12)
    return x @ v

def apply_dc(w_orig, cold_idx, how, nbits=4):
    w = w_orig.clone(); cold = w[cold_idx]
    if how != 'freq':
        order = torch.argsort(sort_key(cold, how)); cold = cold[order]
    n = cold.shape[0]; nb = (n + BS_BLOCK - 1) // BS_BLOCK
    pad = nb * BS_BLOCK - n
    if pad: cold = torch.cat([cold, torch.zeros(pad, cold.shape[1])], 0)
    means = cold.view(nb, BS_BLOCK, -1).mean(dim=(1, 2))
    lo, hi = means.min(), means.max(); lv = (1 << nbits) - 1
    s = (hi - lo) / lv
    if s == 0: s = torch.tensor(1.0)
    mq = (((means - lo) / s).round().clamp(0, lv)) * s + lo
    rec = mq.repeat_interleave(BS_BLOCK).unsqueeze(1).expand(-1, cold.shape[1]).contiguous()[:n]
    if how != 'freq':
        inv = torch.empty_like(order); inv[order] = torch.arange(n); rec = rec[inv]
    w[cold_idx] = rec
    return w

# ---------------------------------------------------------------- measurement
def measure(name, hf, mb, run_fwd, test_ld, results):
    it = iter(test_ld)
    for _ in range(20):                      # warmup
        X, o, i, T = next(it); run_fwd(X, o, i)
    times = []
    for _ in range(TIMING_BATCHES):
        X, o, i, T = next(it)
        t = time.perf_counter(); run_fwd(X, o, i); times.append((time.perf_counter()-t)*1000)
    times = np.array(times)
    s, l, n = [], [], 0
    with torch.no_grad():
        for X, o, i, T in test_ld:
            s.append(run_fwd(X, o, i).numpy().flatten()); l.append(T.numpy().flatten())
            n += 1
            if n >= AUC_BATCHES: break
    auc = roc_auc_score(np.concatenate(l), np.concatenate(s))
    r = dict(config=name, hot_frac=hf, mem_mb=mb, auc=auc,
             mean_ms=float(times.mean()), p50_ms=float(np.percentile(times, 50)),
             p99_ms=float(np.percentile(times, 99)), timing_batches=TIMING_BATCHES,
             auc_samples=n * BATCH)
    results.append(r)
    json.dump(results, open('results/unified_terabyte.json', 'w'), indent=1)
    log(f"{name:<22} hf={str(hf):>6} {mb:>9.1f}MB  AUC {auc:.6f}  "
        f"fwd mean {times.mean():6.2f} p50 {np.percentile(times,50):6.2f} "
        f"p99 {np.percentile(times,99):6.2f} ms")
    return r

def main():
    log("loading model + data ...")
    dlrm, test_ld, ln_emb = load_tb()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    dlrm.load_state_dict(sd); dlrm.eval()
    ntab = len(ln_emb)
    LARGE = set(i for i, n in enumerate(ln_emb) if n >= LARGE_THRESH)
    W = [dlrm.emb_l[t].weight.data for t in range(ntab)]
    total_mb = sum(int(ln_emb[i]) * EMB_DIM * 4 for i in range(ntab)) / 1024**2
    small_u8 = sum(int(ln_emb[i]) * EMB_DIM * 1 for i in range(ntab) if i not in LARGE) / 1024**2
    log(f"tables={ntab} large={sorted(LARGE)} fp32={total_mb:.0f} MB")
    results = []
    torch.set_grad_enabled(False)

    F = lambda src: (lambda X, o, i: fwd_src(dlrm, X, i, src, ntab))

    measure('fp32', None, total_mb, F(SrcFP32(W)), test_ld, results)
    for nb in (8, 4):
        src = SrcQuant(W, LARGE, nb)
        measure(f'INT{nb}', None, small_u8 + sum(int(ln_emb[t]) * EMB_DIM * nb / 8
                for t in LARGE) / 1024**2, F(src), test_ld, results)
        del src

    log("profiling access frequency ...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in LARGE}
    n = 0
    for X, o, i, T in test_ld:
        for t in LARGE:
            freq[t].scatter_add_(0, i[t].flatten().long(), torch.ones(i[t].numel(), dtype=torch.long))
        n += 1
        if n >= 500: break
    log(f"  profiled {n} batches")

    base, off = {}, 0
    for t in sorted(LARGE):
        base[t] = off; off += int(ln_emb[t]) * ROW_BYTES
    if not os.path.exists(COLD1) or os.path.getsize(COLD1) != off:
        log(f"writing cold file {off/2**30:.1f} GiB ...")
        with open(COLD1, 'wb') as f:
            for t in sorted(LARGE): f.write(W[t].numpy().astype(np.float32).tobytes())
        os.system(f"cp {COLD1} {COLD2}; sync")
        log("cold file written + mirrored")
    lib = ctypes.CDLL('./libcoldread.so')
    lib.cold_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    log(f"cold_open rc={lib.cold_open(COLD1.encode(), COLD2.encode(), SSD_THREADS, SSD_QD, ROW_BYTES)}")

    for hf in HOT_FRACTIONS:
        hot_pos, cold_rank, hot_idx, cold_idx = {}, {}, {}, {}
        for t in LARGE:
            nrow = int(ln_emb[t]); nh = max(1, int(nrow * hf))
            order = freq[t].argsort(descending=True)
            hot_idx[t], cold_idx[t] = order[:nh], order[nh:]
            hp = torch.full((nrow,), -1, dtype=torch.long); hp[hot_idx[t]] = torch.arange(nh)
            hot_pos[t] = hp
        hot_mb  = sum(len(hot_idx[t]) * EMB_DIM for t in LARGE) / 1024**2
        bmap_mb = sum(int(ln_emb[t]) for t in LARGE) / 8 / 1024**2
        nblk    = sum((len(cold_idx[t]) + BS_BLOCK - 1)//BS_BLOCK for t in LARGE)
        dc_mb   = nblk * 4 / 8 / 1024**2

        hot_u8 = hs = hmn = None
        for how in ('zero', 'freq', 'value', 'pca'):
            hu, hss, hmm, cr, dq, dss, dll = {}, {}, {}, {}, {}, {}, {}
            for t in LARGE:
                a, b, c, order, d, e, f2 = build_dc(W[t], cold_idx[t], hot_idx[t], 
                                                    'freq' if how == 'zero' else how)
                hu[t], hss[t], hmm[t], dq[t], dss[t], dll[t] = a, b, c, d, e, f2
                nc = len(cold_idx[t])
                rank = torch.empty(nc, dtype=torch.long); rank[order] = torch.arange(nc)
                full = torch.zeros(int(ln_emb[t]), dtype=torch.long); full[cold_idx[t]] = rank
                cr[t] = full
            if hot_u8 is None: hot_u8, hs, hmn = hu, hss, hmm
            src = SrcDC(W, LARGE, hot_pos, hu, hss, hmm, cr, dq, dss, dll, how == 'zero')
            mb = hot_mb + bmap_mb + small_u8 + (0 if how == 'zero' else dc_mb)
            measure('zero (hot uint8)' if how == 'zero' else f'DC {how}-sort 4bit',
                    hf, mb, F(src), test_ld, results)
            del src, hu, hss, hmm, cr, dq, dss, dll

        ssrc = SrcSSD(lib, W, LARGE, hot_pos, hot_u8, hs, hmn, base)
        measure('SSD cold (io_uring)', hf, hot_mb + bmap_mb + small_u8,
                lambda X, o, i: ssrc.forward(dlrm, X, i, ntab), test_ld, results)
        del ssrc

    lib.cold_close()
    log("done -> results/unified_terabyte.json")

if __name__ == '__main__':
    main()
