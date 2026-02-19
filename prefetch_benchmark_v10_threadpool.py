#!/usr/bin/env python3
"""
Prefetch Benchmark V10 Thread Pool: Multi-threaded decompression + block size sweep.

During inference, embedding lookup is mostly memory-bound, freeing CPU cycles
for decompression threads. Test whether a thread pool can hide decompression latency.

Grid:
  Part 1 - Thread sweep (block_size=4096):
    3 codecs (zlib, lz4, zstd) × 4 thread counts (1, 2, 4, 8)
  Part 2 - Block size sweep (threads=4):
    3 codecs × 3 block sizes (1080p=388800, 1440p=691200, 4K=1555200)
"""

import os, sys, time, json, threading, gc, zlib, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import lz4.block
import zstandard as zstd

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

# Block sizes
BLOCK_ORIGINAL = 4096
BLOCK_1080P = 388800   # 1920*1080*3 / 16
BLOCK_1440P = 691200   # 2560*1440*3 / 16
BLOCK_4K = 1555200     # 3840*2160*3 / 16

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
# COLD STORES
# ==============================================================

class EntropyColdStore:
    """V9 uint8 cold store — no block compression."""
    def __init__(self, large_tables):
        self.cold_uint8 = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}
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


class ThreadPoolBlockStore:
    """
    Block-compressed cold store with thread pool for parallel decompression.

    During inference, embedding lookup is memory-bound, leaving CPU cycles
    free for decompression threads. Each table can decompress independently.
    """
    def __init__(self, large_tables, codec='zlib', block_size=BLOCK_ORIGINAL,
                 num_threads=1):
        self.blocks = {}           # {t: [(compressed_bytes, num_rows, raw_size), ...]}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.large_tables = large_tables
        self.block_size = block_size
        self.codec = codec
        self.num_threads = num_threads
        self.block_cache = {}      # {(t, block_id): np.array}
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.compress_time = 0.0
        self._cache_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        # Set up codec functions
        # Note: zstd decompressor is NOT thread-safe, so create per-call instance
        if codec == 'zlib':
            self._compress = lambda data: zlib.compress(data, 1)
            self._decompress = lambda data, size: zlib.decompress(data)
        elif codec == 'lz4':
            self._compress = lambda data: lz4.block.compress(data, store_size=False)
            self._decompress = lambda data, size: lz4.block.decompress(data, uncompressed_size=size)
        elif codec == 'zstd':
            cctx = zstd.ZstdCompressor(level=1)
            self._compress = cctx.compress
            self._decompress = lambda data, size: zstd.ZstdDecompressor().decompress(data)
        else:
            raise ValueError(f"Unknown codec: {codec}")

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (s, zp)

        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

        n_rows = q_np.shape[0]
        table_blocks = []
        raw_total = 0
        comp_total = 0
        ct0 = time.time()
        for start in range(0, n_rows, self.block_size):
            end = min(start + self.block_size, n_rows)
            block_data = q_np[start:end]
            raw = block_data.tobytes()
            compressed = self._compress(raw)
            table_blocks.append((compressed, end - start, len(raw)))
            raw_total += len(raw)
            comp_total += len(compressed)
        self.compress_time += time.time() - ct0
        self.blocks[t] = table_blocks
        self.raw_bytes += raw_total
        self.compressed_bytes += comp_total

    def _get_block(self, t, block_id):
        """Get a block, decompressing if not cached. Thread-safe."""
        key = (t, block_id)
        # Fast path: check cache without lock
        cached = self.block_cache.get(key)
        if cached is not None:
            return cached

        # Slow path: decompress
        t0 = time.time()
        compressed, num_rows, raw_size = self.blocks[t][block_id]
        raw = self._decompress(compressed, raw_size)
        arr = np.frombuffer(raw, dtype=np.uint8).copy().reshape(num_rows, EMB_DIM)
        dt = time.time() - t0

        with self._cache_lock:
            self.block_cache[key] = arr
        with self._stats_lock:
            self.decompress_count += 1
            self.decompress_time += dt
        return arr

    def _dequant_one_table(self, t, lS_i):
        """Dequantize cold rows for one table. Called from thread pool."""
        if t not in self.blocks:
            return None
        indices = lS_i[t].numpy().flatten()
        unique = np.unique(indices)
        seq = self.cold_seq_lookup[t][unique]
        mask = seq >= 0
        if not mask.any():
            return None
        cold_orig = unique[mask]
        cold_seq = seq[mask]

        block_ids = cold_seq // self.block_size
        unique_blocks = np.unique(block_ids)

        q_rows = np.empty((len(cold_seq), EMB_DIM), dtype=np.uint8)
        for bid in unique_blocks:
            block_data = self._get_block(t, bid)
            block_mask = block_ids == bid
            local_offsets = cold_seq[block_mask] - bid * self.block_size
            q_rows[block_mask] = block_data[local_offsets]

        s, zp = self.quant_params[t]
        fp_rows = (q_rows.astype(np.float32) - zp) * s
        return (t, torch.from_numpy(cold_orig.astype(np.int64)),
                torch.from_numpy(fp_rows))

    def dequantize_for_batch(self, lS_i):
        """Parallel dequantization: each table processed by a thread pool worker."""
        result = {}
        if self.num_threads <= 1:
            # Single-threaded path
            for t in self.large_tables:
                r = self._dequant_one_table(t, lS_i)
                if r is not None:
                    _, idx, val = r
                    result[r[0]] = (idx, val)
        else:
            # Thread pool: one task per table
            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                futures = {pool.submit(self._dequant_one_table, t, lS_i): t
                           for t in self.large_tables}
                for future in futures:
                    r = future.result()
                    if r is not None:
                        result[r[0]] = (r[1], r[2])
        return result

    @staticmethod
    def inject(dlrm, result):
        with torch.no_grad():
            for t, (idx_tensor, val_tensor) in result.items():
                dlrm.emb_l[t].weight.data[idx_tensor] = val_tensor

    @property
    def memory_bytes(self):
        return self.compressed_bytes

    @property
    def compression_ratio(self):
        if self.compressed_bytes == 0:
            return 0.0
        return self.raw_bytes / self.compressed_bytes

    @property
    def num_blocks_total(self):
        return sum(len(bl) for bl in self.blocks.values())

    @property
    def cache_size_bytes(self):
        return sum(a.nbytes for a in self.block_cache.values())


# ==============================================================
# PREFETCHER (with thread pool support)
# ==============================================================

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
# SETUP
# ==============================================================

def setup_cold_store(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                     large_tables, hot_indices, store):
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
        log(f"    Table {t}: {len(hot_idx):,} hot, {len(cold_idx):,} cold")
    return store


# ==============================================================
# INFERENCE
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


def run_prefetch_inference(dlrm, test_ld, store):
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
    log("PREFETCH BENCHMARK V10: Thread Pool + Block Size Sweep")
    log("=" * 70)

    log("Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")

    # Profile
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
        log(f"  Table {t}: {ln_emb[t]:,} total, {len(hot_indices[t]):,} hot@80%")

    def compute_memory(store):
        cold_mb = store.memory_bytes / 1024 / 1024
        hot_mb = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables) / 1024 / 1024
        small_mb = sum(state_dict[emb_keys[t]].numel() * 4
                       for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD) / 1024 / 1024
        mlp_mb = sum(p.numel() * 4 for n, p in dlrm.named_parameters() if 'emb_l' not in n) / 1024 / 1024
        return {'hot_fp32_mb': hot_mb, 'small_fp32_mb': small_mb,
                'mlp_mb': mlp_mb, 'cold_store_mb': cold_mb,
                'total_mb': hot_mb + small_mb + mlp_mb + cold_mb}

    results = {}

    # ================================================================
    # CONFIG 0: BASELINE
    # ================================================================
    log("\n" + "=" * 60)
    log("BASELINE — streaming, no compression")
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
        'name': 'Baseline', 'accuracy': b_acc, 'auc': b_auc, 'auc_loss_pp': 0.0,
        'inference_time': b_time, 'setup_time': 0, 'total_time': b_time,
        'memory_mb': b_mem, 'batch_latency': latency_stats(b_blats),
    }

    # ================================================================
    # CONFIG 1: V9 reference (uint8, no compression)
    # ================================================================
    log("\n" + "=" * 60)
    log("V9 REFERENCE — uint8, 80% hot, no block compression")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    gc.collect()

    store = EntropyColdStore(large_tables)
    setup_t0 = time.time()
    setup_cold_store(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                     large_tables, hot_indices, store)
    setup_time = time.time() - setup_t0
    mem = compute_memory(store)
    log(f"  Setup: {setup_time:.2f}s, Cold: {mem['cold_store_mb']:.1f}MB")

    drop_caches(); time.sleep(1); gc.collect()
    acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
    auc_loss = (baseline_auc - auc) * 100
    log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}, AUC loss={auc_loss:.4f}pp")
    log(f"  Inference={inf_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time + inf_time:.2f}s")
    log(f"  Batch: mean={np.mean(blats) * 1000:.1f}ms, p50={np.percentile(blats, 50) * 1000:.1f}ms, "
        f"p99={np.percentile(blats, 99) * 1000:.1f}ms")

    results['v9_uint8'] = {
        'name': 'V9 uint8 (no compression)', 'accuracy': acc, 'auc': auc,
        'auc_loss_pp': auc_loss,
        'inference_time': inf_time, 'setup_time': setup_time,
        'total_time': setup_time + inf_time,
        'memory_mb': mem['total_mb'], 'memory_breakdown': mem,
        'batch_latency': latency_stats(blats),
    }

    # ================================================================
    # PART 1: THREAD SWEEP (block_size=4096)
    # ================================================================
    log("\n" + "=" * 70)
    log("PART 1: THREAD SWEEP — block_size=4096, varying thread count")
    log("=" * 70)

    codecs = ['zlib', 'lz4', 'zstd']
    thread_counts = [1, 2, 4, 8]

    for codec in codecs:
        for nthreads in thread_counts:
            key = f"thread_{codec}_t{nthreads}"
            name = f"{codec} block=4096 threads={nthreads}"
            log(f"\n{'=' * 60}")
            log(f"{name}")
            log(f"{'=' * 60}")

            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()

            store = ThreadPoolBlockStore(large_tables, codec=codec,
                                         block_size=BLOCK_ORIGINAL,
                                         num_threads=nthreads)
            setup_t0 = time.time()
            setup_cold_store(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                             large_tables, hot_indices, store)
            setup_time = time.time() - setup_t0
            mem = compute_memory(store)

            log(f"  Setup: {setup_time:.2f}s (compress: {store.compress_time:.2f}s)")
            log(f"  Blocks: {store.num_blocks_total}, "
                f"Compression: {store.raw_bytes / 1024 / 1024:.1f}MB -> "
                f"{store.compressed_bytes / 1024 / 1024:.1f}MB ({store.compression_ratio:.2f}x)")

            drop_caches(); time.sleep(1); gc.collect()
            acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
            auc_loss = (baseline_auc - auc) * 100

            cache_mb = store.cache_size_bytes / 1024 / 1024
            log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}, AUC loss={auc_loss:.4f}pp")
            log(f"  Inference={inf_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time + inf_time:.2f}s")
            log(f"  Memory: cold={mem['cold_store_mb']:.1f}MB, total={mem['total_mb']:.1f}MB, "
                f"cache={cache_mb:.1f}MB")
            log(f"  Decompress: count={store.decompress_count}, time={store.decompress_time * 1000:.1f}ms")
            log(f"  Batch: mean={np.mean(blats) * 1000:.1f}ms, "
                f"p50={np.percentile(blats, 50) * 1000:.1f}ms, "
                f"p99={np.percentile(blats, 99) * 1000:.1f}ms")

            results[key] = {
                'name': name, 'accuracy': acc, 'auc': auc, 'auc_loss_pp': auc_loss,
                'inference_time': inf_time, 'setup_time': setup_time,
                'total_time': setup_time + inf_time,
                'memory_mb': mem['total_mb'], 'memory_breakdown': mem,
                'batch_latency': latency_stats(blats),
                'codec': codec, 'block_size': BLOCK_ORIGINAL,
                'num_threads': nthreads,
                'compression_ratio': store.compression_ratio,
                'compressed_mb': store.compressed_bytes / 1024 / 1024,
                'decompress_count': store.decompress_count,
                'decompress_time_ms': store.decompress_time * 1000,
                'cache_mb': cache_mb,
                'num_blocks': store.num_blocks_total,
            }

    # ================================================================
    # PART 2: BLOCK SIZE SWEEP (threads=4)
    # ================================================================
    log("\n" + "=" * 70)
    log("PART 2: BLOCK SIZE SWEEP — threads=4, varying block size")
    log("=" * 70)

    block_sizes = [
        ('1080p', BLOCK_1080P),
        ('1440p', BLOCK_1440P),
        ('4K', BLOCK_4K),
    ]

    for codec in codecs:
        for bname, bsize in block_sizes:
            key = f"block_{codec}_{bname}"
            name = f"{codec} block={bname} threads=4"
            log(f"\n{'=' * 60}")
            log(f"{name}")
            log(f"{'=' * 60}")

            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()

            store = ThreadPoolBlockStore(large_tables, codec=codec,
                                         block_size=bsize, num_threads=4)
            setup_t0 = time.time()
            setup_cold_store(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                             large_tables, hot_indices, store)
            setup_time = time.time() - setup_t0
            mem = compute_memory(store)

            log(f"  Setup: {setup_time:.2f}s (compress: {store.compress_time:.2f}s)")
            log(f"  Blocks: {store.num_blocks_total}, "
                f"Compression: {store.raw_bytes / 1024 / 1024:.1f}MB -> "
                f"{store.compressed_bytes / 1024 / 1024:.1f}MB ({store.compression_ratio:.2f}x)")

            drop_caches(); time.sleep(1); gc.collect()
            acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
            auc_loss = (baseline_auc - auc) * 100

            cache_mb = store.cache_size_bytes / 1024 / 1024
            log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}, AUC loss={auc_loss:.4f}pp")
            log(f"  Inference={inf_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time + inf_time:.2f}s")
            log(f"  Memory: cold={mem['cold_store_mb']:.1f}MB, total={mem['total_mb']:.1f}MB, "
                f"cache={cache_mb:.1f}MB")
            log(f"  Decompress: count={store.decompress_count}, time={store.decompress_time * 1000:.1f}ms")
            log(f"  Batch: mean={np.mean(blats) * 1000:.1f}ms, "
                f"p50={np.percentile(blats, 50) * 1000:.1f}ms, "
                f"p99={np.percentile(blats, 99) * 1000:.1f}ms")

            results[key] = {
                'name': name, 'accuracy': acc, 'auc': auc, 'auc_loss_pp': auc_loss,
                'inference_time': inf_time, 'setup_time': setup_time,
                'total_time': setup_time + inf_time,
                'memory_mb': mem['total_mb'], 'memory_breakdown': mem,
                'batch_latency': latency_stats(blats),
                'codec': codec, 'block_size': bsize, 'block_name': bname,
                'num_threads': 4,
                'compression_ratio': store.compression_ratio,
                'compressed_mb': store.compressed_bytes / 1024 / 1024,
                'decompress_count': store.decompress_count,
                'decompress_time_ms': store.decompress_time * 1000,
                'cache_mb': cache_mb,
                'num_blocks': store.num_blocks_total,
            }

    # ================================================================
    # SUMMARY
    # ================================================================
    log("\n" + "=" * 70)
    log("SUMMARY — THREAD POOL + BLOCK SIZE SWEEP")
    log("=" * 70)

    # Part 1 summary: Thread sweep
    log("\n--- PART 1: Thread Count Effect (block=4096) ---")
    fmt = "  %-40s  %8s  %10s  %10s  %10s  %10s  %8s  %12s"
    log(fmt % ("Config", "AUC", "Infer(s)", "Setup(s)", "Total(s)", "Mem(MB)", "Ratio", "Decomp(ms)"))
    log("  " + "-" * 130)
    log(fmt % ("Baseline", f"{baseline_auc:.6f}", f"{results['baseline']['inference_time']:.2f}",
               "0.00", f"{results['baseline']['total_time']:.2f}",
               f"{results['baseline']['memory_mb']:.1f}", "-", "-"))
    log(fmt % ("V9 uint8", f"{results['v9_uint8']['auc']:.6f}",
               f"{results['v9_uint8']['inference_time']:.2f}",
               f"{results['v9_uint8']['setup_time']:.2f}",
               f"{results['v9_uint8']['total_time']:.2f}",
               f"{results['v9_uint8']['memory_mb']:.1f}", "-", "-"))
    for codec in codecs:
        for nthreads in thread_counts:
            key = f"thread_{codec}_t{nthreads}"
            r = results[key]
            log(fmt % (r['name'], f"{r['auc']:.6f}", f"{r['inference_time']:.2f}",
                       f"{r['setup_time']:.2f}", f"{r['total_time']:.2f}",
                       f"{r['memory_mb']:.1f}", f"{r['compression_ratio']:.2f}x",
                       f"{r['decompress_time_ms']:.1f}"))

    # Part 2 summary: Block size sweep
    log("\n--- PART 2: Block Size Effect (threads=4) ---")
    log(fmt % ("Config", "AUC", "Infer(s)", "Setup(s)", "Total(s)", "Mem(MB)", "Ratio", "Decomp(ms)"))
    log("  " + "-" * 130)
    for codec in codecs:
        for bname, _ in block_sizes:
            key = f"block_{codec}_{bname}"
            r = results[key]
            log(fmt % (r['name'], f"{r['auc']:.6f}", f"{r['inference_time']:.2f}",
                       f"{r['setup_time']:.2f}", f"{r['total_time']:.2f}",
                       f"{r['memory_mb']:.1f}", f"{r['compression_ratio']:.2f}x",
                       f"{r['decompress_time_ms']:.1f}"))

    json_path = os.path.join(RESULTS_DIR, 'prefetch_v10_threadpool.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Saved: {json_path}")

    log("\n" + "=" * 70)
    log("DONE!")
    log("=" * 70)


if __name__ == '__main__':
    main()
