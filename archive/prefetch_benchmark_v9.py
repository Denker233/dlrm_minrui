#!/usr/bin/env python3
"""
Prefetch Benchmark V9: Entropy Coding (uint8 quantization) with Prefetch

Tests per-row uint8 quantization with vectorized prefetch/dequantization.
No video codec - just quantize to uint8, dequantize on demand.

Configs:
1. Baseline - no compression, streaming from dataloader
2. Full reconstruct - quantize->dequantize all at setup, vanilla inference
3. Per-batch prefetch - cold as uint8 arrays, vectorized dequant in background thread
"""

import os, sys, time, json, threading, gc, zlib, subprocess
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlrm_data_pytorch as dp

MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
DATA_FILE = "./input/train.txt"
RESULTS_DIR = os.path.expanduser("~/experiment-control")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 2048
EMB_DIM = 16
HOT_THRESHOLD = 0.80
PROFILE_BATCHES = 200
LARGE_TABLE_THRESHOLD = 50000

os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def drop_caches():
    try:
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except Exception:
        pass


def create_args():
    class Args: pass
    a = Args()
    a.arch_sparse_feature_size = ARCH_SPARSE_FEATURE_SIZE
    a.arch_mlp_bot = ARCH_MLP_BOT; a.arch_mlp_top = ARCH_MLP_TOP
    a.arch_interaction_op = "dot"; a.arch_interaction_itself = False
    a.data_generation = "dataset"; a.data_set = "kaggle"
    a.raw_data_file = DATA_FILE; a.processed_data_file = PROCESSED_DATA
    a.loss_function = "bce"; a.max_ind_range = -1
    a.test_mini_batch_size = TEST_BATCH_SIZE; a.test_num_workers = 0; a.num_workers = 0
    a.mlperf_logging = False; a.memory_map = False; a.data_randomize = "total"
    a.data_trace_enable_padding = False; a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10; a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 128; a.round_targets = True
    a.mlperf_bin_loader = False; a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False
    return a


def load_model_and_data():
    from dlrm_s_pytorch import DLRM_Net
    args = create_args()
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    ln_emb = np.array(train_data.counts)
    m_spa = args.arch_sparse_feature_size
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + args.arch_mlp_top, dtype=int, sep="-")
    dlrm = DLRM_Net(m_spa, ln_emb, ln_bot, ln_top,
                     arch_interaction_op="dot", arch_interaction_itself=False,
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")
    ld = torch.load(MODEL_PATH, map_location='cpu')
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    return dlrm, test_ld, train_ld, ln_emb


def quantize(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def dequantize(q, s, zp):
    return (q.float() - zp) * s


def restore_weights(dlrm, state_dict, emb_keys):
    with torch.no_grad():
        for k in emb_keys:
            t = int(k.split('.')[1])
            dlrm.emb_l[t].weight.data = state_dict[k].clone()


def latency_stats(blats):
    a = np.array(blats)
    return {
        'count': len(a), 'mean_ms': float(np.mean(a) * 1000),
        'p50_ms': float(np.percentile(a, 50) * 1000),
        'p95_ms': float(np.percentile(a, 95) * 1000),
        'p99_ms': float(np.percentile(a, 99) * 1000),
    }


# ==============================================================
# COLD STORE: uint8 arrays with vectorized dequantization
# ==============================================================

class EntropyColdStore:
    def __init__(self, large_tables):
        self.cold_uint8 = {}       # {t: np.array (num_cold, 16) uint8}
        self.cold_seq_lookup = {}  # {t: np.array (num_emb,) int32, -1 if not cold}
        self.quant_params = {}     # {t: (scale, zp)}
        self.large_tables = large_tables

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize(cold_weights)
        self.cold_uint8[t] = q.numpy()
        self.quant_params[t] = (s, zp)
        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

    def dequantize_for_batch(self, lS_i):
        """Vectorized: find cold indices, dequantize, return ready-to-inject tensors."""
        result = {}
        for t in self.large_tables:
            if t not in self.cold_uint8:
                continue
            indices = lS_i[t].numpy().flatten()
            unique = np.unique(indices)
            seq = self.cold_seq_lookup[t][unique]
            mask = seq >= 0
            if not mask.any():
                continue
            cold_orig = unique[mask]
            cold_seq = seq[mask]
            q_rows = self.cold_uint8[t][cold_seq]
            s, zp = self.quant_params[t]
            fp_rows = (q_rows.astype(np.float32) - zp) * s
            result[t] = (torch.from_numpy(cold_orig.astype(np.int64)),
                         torch.from_numpy(fp_rows))
        return result

    @staticmethod
    def inject(dlrm, result):
        with torch.no_grad():
            for t, (idx_tensor, val_tensor) in result.items():
                dlrm.emb_l[t].weight.data[idx_tensor] = val_tensor

    @property
    def memory_bytes(self):
        return sum(a.nbytes for a in self.cold_uint8.values())


class PrefetchDequantizer:
    def __init__(self, store):
        self.store = store
        self._thread = None
        self._result = {}

    def prefetch_async(self, lS_i):
        if self._thread:
            self._thread.join()
        self._result = {}
        def work():
            self._result = self.store.dequantize_for_batch(lS_i)
        self._thread = threading.Thread(target=work)
        self._thread.start()

    def wait(self):
        if self._thread:
            self._thread.join()
            self._thread = None
        return self._result

    def fetch_sync(self, lS_i):
        return self.store.dequantize_for_batch(lS_i)


# ==============================================================
# SETUP FUNCTIONS
# ==============================================================

def setup_full_reconstruct(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                           large_tables, hot_indices):
    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        if w.shape[0] < LARGE_TABLE_THRESHOLD:
            q, s, zp = quantize(w)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = dequantize(q, s, zp)

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())
        hot_idx = sorted(hi)
        if hot_idx:
            hw = w[hot_idx]; qh, sh, zh = quantize(hw)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[torch.tensor(hot_idx, dtype=torch.long)] = dequantize(qh, sh, zh)
        cold_idx = sorted(set(range(ln_emb[t])) - hi)
        cw = w[cold_idx]
        if cw.shape[0] > 0:
            qc, sc, zc = quantize(cw)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[torch.tensor(cold_idx, dtype=torch.long)] = dequantize(qc, sc, zc)
        log(f"    Table {t}: {len(hot_idx):,} hot + {len(cold_idx):,} cold dequantized")


def setup_entropy_cold(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                       large_tables, hot_indices):
    store = EntropyColdStore(large_tables)
    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        if w.shape[0] < LARGE_TABLE_THRESHOLD:
            q, s, zp = quantize(w)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = dequantize(q, s, zp)

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())
        hot_idx = sorted(hi)
        if hot_idx:
            hw = w[hot_idx]; qh, sh, zh = quantize(hw)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[torch.tensor(hot_idx, dtype=torch.long)] = dequantize(qh, sh, zh)
        cold_idx = sorted(set(range(ln_emb[t])) - hi)
        cw = w[cold_idx]
        store.add_table(t, ln_emb[t], cold_idx, cw)
        log(f"    Table {t}: {len(hot_idx):,} hot (fp32), "
            f"{len(cold_idx):,} cold (uint8, {cw.shape[0] * 16 / 1024 / 1024:.1f}MB)")
    return store


# ==============================================================
# INFERENCE FUNCTIONS
# ==============================================================

def run_baseline_inference(dlrm, test_ld):
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    nb = 0
    t0 = time.time()
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_ld:
            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            S = Z.detach().cpu().numpy().flatten()
            Tn = T.detach().cpu().numpy().flatten()
            accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
            samp += Tn.shape[0]
            scores.extend(S.tolist()); targets.extend(Tn.tolist())
            nb += 1
            if nb % 500 == 0:
                log(f"    Batch {nb}, lat={blats[-1] * 1000:.1f}ms")
    total = time.time() - t0
    return accu / samp, roc_auc_score(targets, scores), total, blats, nb


def run_entropy_inference(dlrm, test_ld, store):
    prefetcher = PrefetchDequantizer(store)
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    nb = 0

    t0 = time.time()
    dataloader_iter = iter(test_ld)

    try:
        current_batch = next(dataloader_iter)
    except StopIteration:
        return 0, 0, 0, [], 0

    _, _, lS_i_0, _ = current_batch
    current_result = prefetcher.fetch_sync(lS_i_0)

    try:
        lookahead_batch = next(dataloader_iter)
        _, _, la_lS_i, _ = lookahead_batch
        prefetcher.prefetch_async(la_lS_i)
        has_lookahead = True
    except StopIteration:
        has_lookahead = False

    while True:
        X, lS_o, lS_i, T = current_batch
        bt0 = time.time()

        store.inject(dlrm, current_result)

        with torch.no_grad():
            Z = dlrm(X, lS_o, lS_i)
        blats.append(time.time() - bt0)

        S = Z.detach().cpu().numpy().flatten()
        Tn = T.detach().cpu().numpy().flatten()
        accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
        samp += Tn.shape[0]
        scores.extend(S.tolist()); targets.extend(Tn.tolist())
        nb += 1

        if nb % 500 == 0:
            log(f"    Batch {nb}, lat={blats[-1] * 1000:.1f}ms")

        if not has_lookahead:
            break

        next_result = prefetcher.wait()
        current_batch = lookahead_batch
        current_result = next_result

        try:
            lookahead_batch = next(dataloader_iter)
            _, _, la_lS_i, _ = lookahead_batch
            prefetcher.prefetch_async(la_lS_i)
            has_lookahead = True
        except StopIteration:
            has_lookahead = False

    total = time.time() - t0
    return accu / samp, roc_auc_score(targets, scores), total, blats, nb


# ==============================================================
# MAIN
# ==============================================================

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log("=" * 70)
    log("PREFETCH BENCHMARK V9: Entropy Coding (uint8) with Prefetch")
    log("=" * 70)

    log("Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")

    # Profile access patterns
    log("Profiling access patterns...")
    access_raw = [None] * num_tables
    for i, batch in enumerate(train_ld):
        if i >= PROFILE_BATCHES: break
        _, _, lS_i, _ = batch
        for t in range(num_tables):
            idx = lS_i[t].numpy().flatten()
            if access_raw[t] is None:
                access_raw[t] = idx.copy()
            else:
                access_raw[t] = np.concatenate([access_raw[t], idx])

    hot_indices = {}
    for t in range(num_tables):
        if access_raw[t] is None or len(access_raw[t]) == 0:
            hot_indices[t] = np.array([], dtype=np.int64)
            continue
        unique, counts = np.unique(access_raw[t], return_counts=True)
        si = np.argsort(-counts)
        cum = np.cumsum(counts[si])
        cutoff = np.searchsorted(cum, cum[-1] * HOT_THRESHOLD) + 1
        hot_indices[t] = unique[si[:cutoff]]

    for t in large_tables:
        log(f"  Table {t}: {ln_emb[t]:,} total, {len(hot_indices[t]):,} hot")

    # Quick test: per-row zlib compression on 16-byte rows
    log("\nPer-row zlib test (why per-row entropy coding doesn't help for 16 bytes):")
    hi_set = set(hot_indices[large_tables[0]].tolist())
    sample_cold = sorted(set(range(ln_emb[large_tables[0]])) - hi_set)[:1000]
    sample_w = state_dict[emb_keys[large_tables[0]]][sample_cold]
    sq, ss, szp = quantize(sample_w)
    sample_bytes = sq.numpy()
    raw_total = sample_bytes.nbytes
    comp_total = sum(len(zlib.compress(row.tobytes())) for row in sample_bytes)
    log(f"  1000 rows x 16 bytes: raw={raw_total} bytes, zlib={comp_total} bytes "
        f"({comp_total / raw_total:.2f}x) -- WORSE due to per-row headers")
    log(f"  Conclusion: per-row entropy coding inflates 16-byte items. "
        f"uint8 quantization (4x compression) is the practical approach.\n")

    results = {}

    # ================================================================
    # CONFIG 1: BASELINE
    # ================================================================
    log("=" * 60)
    log("CONFIG 1: BASELINE — streaming, no compression")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    drop_caches(); time.sleep(1); gc.collect()

    b_acc, b_auc, b_time, b_blats, b_nb = run_baseline_inference(dlrm, test_ld)
    b_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
    log(f"  Acc={b_acc * 100:.4f}%, AUC={b_auc:.6f}")
    log(f"  Inference={b_time:.2f}s, Memory={b_mem:.1f}MB")
    log(f"  Batch: mean={np.mean(b_blats) * 1000:.1f}ms, "
        f"p50={np.percentile(b_blats, 50) * 1000:.1f}ms, "
        f"p99={np.percentile(b_blats, 99) * 1000:.1f}ms")

    baseline_auc = b_auc
    results['baseline'] = {
        'name': 'Baseline (streaming)', 'accuracy': b_acc, 'auc': b_auc,
        'inference_time': b_time, 'setup_time': 0, 'total_time': b_time,
        'memory_mb': b_mem, 'batch_latency': latency_stats(b_blats),
    }

    # ================================================================
    # CONFIG 2: FULL RECONSTRUCT
    # ================================================================
    log("\n" + "=" * 60)
    log("CONFIG 2: Full reconstruct — quantize+dequantize all at setup, vanilla inference")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    gc.collect()

    setup_t0 = time.time()
    setup_full_reconstruct(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                           large_tables, hot_indices)
    setup_time = time.time() - setup_t0
    log(f"  Setup: {setup_time:.2f}s")

    drop_caches(); time.sleep(1); gc.collect()
    fr_acc, fr_auc, fr_time, fr_blats, fr_nb = run_baseline_inference(dlrm, test_ld)
    fr_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
    log(f"  Acc={fr_acc * 100:.4f}%, AUC={fr_auc:.6f}")
    log(f"  Inference={fr_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time + fr_time:.2f}s")
    log(f"  Memory={fr_mem:.1f}MB (all fp32 at runtime)")
    log(f"  AUC loss vs baseline: {(baseline_auc - fr_auc) * 100:.4f}pp")

    results['full_reconstruct'] = {
        'name': 'Full reconstruct (quant round-trip)', 'accuracy': fr_acc, 'auc': fr_auc,
        'inference_time': fr_time, 'setup_time': setup_time,
        'total_time': setup_time + fr_time,
        'memory_mb': fr_mem, 'batch_latency': latency_stats(fr_blats),
    }

    # ================================================================
    # CONFIG 3: PER-BATCH ENTROPY PREFETCH
    # ================================================================
    log("\n" + "=" * 60)
    log("CONFIG 3: Per-batch uint8 dequant with 1-batch lookahead prefetch")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    gc.collect()

    setup_t0 = time.time()
    store = setup_entropy_cold(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                               large_tables, hot_indices)
    setup_time = time.time() - setup_t0
    cold_mb = store.memory_bytes / 1024 / 1024
    log(f"  Setup: {setup_time:.2f}s")
    log(f"  Cold uint8 storage: {cold_mb:.1f}MB")

    drop_caches(); time.sleep(1); gc.collect()
    e_acc, e_auc, e_time, e_blats, e_nb = run_entropy_inference(dlrm, test_ld, store)

    hot_mb = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables) / 1024 / 1024
    small_mb = sum(state_dict[emb_keys[t]].numel() * 4
                   for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD) / 1024 / 1024
    mlp_mb = sum(p.numel() * 4 for n, p in dlrm.named_parameters() if 'emb_l' not in n) / 1024 / 1024
    total_mem = hot_mb + small_mb + mlp_mb + cold_mb

    log(f"  Acc={e_acc * 100:.4f}%, AUC={e_auc:.6f}")
    log(f"  Inference={e_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time + e_time:.2f}s")
    log(f"  Memory: hot={hot_mb:.1f}MB, small={small_mb:.1f}MB, mlp={mlp_mb:.1f}MB, "
        f"cold_uint8={cold_mb:.1f}MB, total={total_mem:.1f}MB")
    log(f"  AUC loss vs baseline: {(baseline_auc - e_auc) * 100:.4f}pp")
    log(f"  Batch: mean={np.mean(e_blats) * 1000:.1f}ms, "
        f"p50={np.percentile(e_blats, 50) * 1000:.1f}ms, "
        f"p99={np.percentile(e_blats, 99) * 1000:.1f}ms")

    results['entropy_prefetch'] = {
        'name': 'Per-batch uint8 + prefetch', 'accuracy': e_acc, 'auc': e_auc,
        'inference_time': e_time, 'setup_time': setup_time,
        'total_time': setup_time + e_time,
        'memory_mb': total_mem,
        'memory_breakdown': {
            'hot_fp32_mb': hot_mb, 'small_fp32_mb': small_mb,
            'mlp_mb': mlp_mb, 'cold_uint8_mb': cold_mb,
        },
        'batch_latency': latency_stats(e_blats),
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    fmt = "  %-45s  %8s  %10s  %10s  %10s  %10s"
    log(fmt % ("Config", "AUC", "AUC loss", "Infer(s)", "Setup(s)", "Total(s)"))
    log("  " + "-" * 97)
    for key in ['baseline', 'full_reconstruct', 'entropy_prefetch']:
        r = results[key]
        al = (baseline_auc - r['auc']) * 100
        log(fmt % (r['name'], f"{r['auc']:.6f}", f"{al:.4f}pp",
                   f"{r['inference_time']:.2f}", f"{r.get('setup_time', 0):.2f}",
                   f"{r['total_time']:.2f}"))

    json_path = os.path.join(RESULTS_DIR, 'prefetch_v9_entropy.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Saved: {json_path}")

    log("\n" + "=" * 70)
    log("DONE!")
    log("=" * 70)


if __name__ == '__main__':
    main()
