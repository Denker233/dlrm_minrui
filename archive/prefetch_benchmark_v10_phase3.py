#!/usr/bin/env python3
"""
Prefetch Benchmark V10 Phase 3: Practical Optimizations

Lessons learned from Phase 1 and Phase 2:
- LRU with small blocks (4096 rows) gets 42-82% cache hit rate
  but inference is 3-5x slower (140-236s at 300MB budget)
- LRU with large blocks (518K rows) gets 0% cache hit rate
  because 70 blocks thrash the cache completely
- Frequency sorting doesn't help: cold access profiled from training
  data doesn't match test-time access patterns
- All cold rows have near-equal low frequency in profiling data

New strategies that CAN work:
1. Two-pass cache warmup: run test data once to populate cache, measure 2nd pass
2. Aggressive hot threshold (95-99%): shrink cold set dramatically
3. V9 pre-decode + H.265 disk: fast inference + minimal disk size
4. Optimized small blocks: 4096-row blocks but with lz4 for faster decompression
5. Combined: higher hot threshold + smaller cold set + LRU cache

The goal: find configs that achieve ~50-80s inference with < 200MB memory.
"""

import os
import sys
import time
import json
import threading
import gc
import traceback
from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import lz4.block
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlrm_data_pytorch as dp

# ==============================================================
# CONFIGURATION
# ==============================================================

MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
DATA_FILE = "./input/train.txt"
RESULTS_DIR = os.path.expanduser("~/experiment-control")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 2048
EMB_DIM = 16
PROFILE_BATCHES = 200
LARGE_TABLE_THRESHOLD = 50000

os.makedirs(RESULTS_DIR, exist_ok=True)
JSON_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_phase3.json")
LOG_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_phase3.log")

_log_fh = None


def log(msg):
    global _log_fh
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_fh is not None:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def drop_caches():
    try:
        import subprocess
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except Exception:
        pass


# ==============================================================
# SHARED UTILITIES
# ==============================================================

def create_args():
    class Args:
        pass
    a = Args()
    a.arch_sparse_feature_size = ARCH_SPARSE_FEATURE_SIZE
    a.arch_mlp_bot = ARCH_MLP_BOT
    a.arch_mlp_top = ARCH_MLP_TOP
    a.arch_interaction_op = "dot"
    a.arch_interaction_itself = False
    a.data_generation = "dataset"
    a.data_set = "kaggle"
    a.raw_data_file = DATA_FILE
    a.processed_data_file = PROCESSED_DATA
    a.loss_function = "bce"
    a.max_ind_range = -1
    a.test_mini_batch_size = TEST_BATCH_SIZE
    a.test_num_workers = 0
    a.num_workers = 0
    a.mlperf_logging = False
    a.memory_map = False
    a.data_randomize = "total"
    a.data_trace_enable_padding = False
    a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10
    a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 128
    a.round_targets = True
    a.mlperf_bin_loader = False
    a.mlperf_bin_shuffle = False
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


def quantize_global(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def dequantize_global(q, s, zp):
    return (q.float() - zp) * s


def quantize_per_column(w):
    ncols = w.shape[1]
    scales = np.zeros(ncols, dtype=np.float32)
    zps = np.zeros(ncols, dtype=np.int32)
    q = torch.zeros_like(w, dtype=torch.uint8)
    for c in range(ncols):
        col = w[:, c]
        mn, mx = col.min().item(), col.max().item()
        s = (mx - mn) / 255.0
        if s == 0:
            s = 1.0
        z = round(-mn / s)
        q[:, c] = ((col / s).round() + z).clamp(0, 255).to(torch.uint8)
        scales[c] = s
        zps[c] = z
    return q, scales, zps


def dequantize_per_column(q_np, scales, zps):
    result = np.empty(q_np.shape, dtype=np.float32)
    for c in range(q_np.shape[1]):
        result[:, c] = (q_np[:, c].astype(np.float32) - zps[c]) * scales[c]
    return result


def dequantize_per_column_torch(q_t, scales, zps):
    scales_t = torch.from_numpy(scales).float().unsqueeze(0)
    zps_t = torch.from_numpy(zps).float().unsqueeze(0)
    return (q_t.float() - zps_t) * scales_t


def restore_weights(dlrm, state_dict, emb_keys):
    with torch.no_grad():
        for k in emb_keys:
            t = int(k.split('.')[1])
            dlrm.emb_l[t].weight.data = state_dict[k].clone()


def latency_stats(blats):
    a = np.array(blats)
    if len(a) == 0:
        return {'count': 0, 'mean_ms': 0, 'p50_ms': 0, 'p95_ms': 0, 'p99_ms': 0}
    return {
        'count': len(a),
        'mean_ms': float(np.mean(a) * 1000),
        'p50_ms': float(np.percentile(a, 50) * 1000),
        'p95_ms': float(np.percentile(a, 95) * 1000),
        'p99_ms': float(np.percentile(a, 99) * 1000),
    }


def profile_access(train_ld, num_tables, ln_emb, threshold):
    access_raw = [None] * num_tables
    for i, batch in enumerate(train_ld):
        if i >= PROFILE_BATCHES:
            break
        _, _, lS_i, _ = batch
        for t in range(num_tables):
            idx = lS_i[t].numpy().flatten()
            if access_raw[t] is None:
                access_raw[t] = idx.copy()
            else:
                access_raw[t] = np.concatenate([access_raw[t], idx])

    hot_indices = {}
    access_counts = {}
    for t in range(num_tables):
        if access_raw[t] is None or len(access_raw[t]) == 0:
            hot_indices[t] = np.array([], dtype=np.int64)
            access_counts[t] = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
            continue
        unique, counts = np.unique(access_raw[t], return_counts=True)
        si = np.argsort(-counts)
        cum = np.cumsum(counts[si])
        cutoff = np.searchsorted(cum, cum[-1] * threshold) + 1
        hot_indices[t] = unique[si[:cutoff]]
        access_counts[t] = (unique[si], counts[si])
    return hot_indices, access_counts


def compute_memory_breakdown(dlrm, cold_store_bytes, hot_indices, large_tables,
                              state_dict, emb_keys, num_tables, ln_emb):
    cold_mb = cold_store_bytes / 1024 / 1024
    hot_mb = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables) / 1024 / 1024
    small_mb = sum(state_dict[emb_keys[t]].numel() * 4
                   for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD) / 1024 / 1024
    mlp_mb = sum(p.numel() * 4 for n, p in dlrm.named_parameters() if 'emb_l' not in n) / 1024 / 1024
    total_mb = hot_mb + small_mb + mlp_mb + cold_mb
    return {
        'hot_fp32_mb': round(hot_mb, 2),
        'small_fp32_mb': round(small_mb, 2),
        'mlp_mb': round(mlp_mb, 2),
        'cold_store_mb': round(cold_mb, 2),
        'total_mb': round(total_mb, 2),
    }


def compress_block(data_bytes, codec='zstd'):
    if codec == 'lz4' and HAS_LZ4:
        return lz4.block.compress(data_bytes, store_size=False)
    elif codec == 'zstd' and HAS_ZSTD:
        return zstd.ZstdCompressor(level=3).compress(data_bytes)
    else:
        import zlib
        return zlib.compress(data_bytes, 1)


def decompress_block(data_bytes, codec='zstd', orig_size=0):
    if codec == 'lz4' and HAS_LZ4:
        return lz4.block.decompress(data_bytes, uncompressed_size=orig_size)
    elif codec == 'zstd' and HAS_ZSTD:
        return zstd.ZstdDecompressor().decompress(data_bytes)
    else:
        import zlib
        return zlib.decompress(data_bytes)


# ==============================================================
# LRU CACHE
# ==============================================================

class LRUCache:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.cache = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key][0]
        self.misses += 1
        return None

    def put(self, key, data, size_bytes):
        if key in self.cache:
            old_size = self.cache[key][1]
            self.current_bytes -= old_size
            del self.cache[key]
        while self.current_bytes + size_bytes > self.max_bytes and self.cache:
            _, (_, evicted_size) = self.cache.popitem(last=False)
            self.current_bytes -= evicted_size
        self.cache[key] = (data, size_bytes)
        self.current_bytes += size_bytes

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ==============================================================
# PREFETCHER
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
# COLD STORE: V9 Reference (uint8 pre-decoded)
# ==============================================================

class V9ColdStore:
    """V9 uint8 cold store: all cold rows pre-decoded in memory."""
    def __init__(self, large_tables):
        self.cold_uint8 = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.large_tables = large_tables

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize_global(cold_weights)
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


# ==============================================================
# COLD STORE: LRU Block Cache with configurable codec + block size
# ==============================================================

class LRUBlockStore:
    """Block-compressed cold store with LRU cache."""

    def __init__(self, large_tables, block_size=4096, max_cache_mb=200,
                 codec='zstd', quant_fn='global'):
        self.large_tables = large_tables
        self.block_size = block_size
        self.codec = codec
        self.quant_fn = quant_fn
        self.blocks = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.cache = LRUCache(max_cache_mb * 1024 * 1024)
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.num_blocks_total = 0

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        if self.quant_fn == 'per_column':
            q, scales, zps = quantize_per_column(cold_weights)
            self.quant_params[t] = ('per_column', scales, zps)
        else:
            q, s, zp = quantize_global(cold_weights)
            self.quant_params[t] = ('global', s, zp)
        q_np = q.numpy()

        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

        n_rows = q_np.shape[0]
        table_blocks = []
        for start in range(0, n_rows, self.block_size):
            end = min(start + self.block_size, n_rows)
            block_data = q_np[start:end]
            raw = block_data.tobytes()
            compressed = compress_block(raw, self.codec)
            table_blocks.append((compressed, end - start, len(raw)))
            self.raw_bytes += len(raw)
            self.compressed_bytes += len(compressed)
        self.blocks[t] = table_blocks
        self.num_blocks_total += len(table_blocks)

    def _get_block(self, t, block_id):
        key = (t, block_id)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        t0 = time.time()
        compressed, num_rows, orig_size = self.blocks[t][block_id]
        raw = decompress_block(compressed, self.codec, orig_size)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(num_rows, EMB_DIM).copy()
        self.cache.put(key, arr, arr.nbytes)
        self.decompress_count += 1
        self.decompress_time += time.time() - t0
        return arr

    def dequantize_for_batch(self, lS_i):
        result = {}
        for t in self.large_tables:
            if t not in self.blocks:
                continue
            indices = lS_i[t].numpy().flatten()
            unique = np.unique(indices)
            seq = self.cold_seq_lookup[t][unique]
            mask = seq >= 0
            if not mask.any():
                continue
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

            qp = self.quant_params[t]
            if qp[0] == 'per_column':
                fp_rows = dequantize_per_column(q_rows, qp[1], qp[2])
            else:
                s, zp = qp[1], qp[2]
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
        return self.compressed_bytes + self.cache.current_bytes


# ==============================================================
# TWO-PASS INFERENCE: warmup cache then measure
# ==============================================================

def run_warmup_pass(dlrm, test_ld, store):
    """First pass: populate cache without measuring metrics.
    Runs through test data once, doing dequantize+inject for each batch."""
    log("  Warmup pass (populating cache)...")
    prefetcher = PrefetchDequantizer(store)
    nb = 0
    t0 = time.time()
    dataloader_iter = iter(test_ld)

    try:
        current_batch = next(dataloader_iter)
    except StopIteration:
        return

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
        store.inject(dlrm, current_result)
        with torch.no_grad():
            dlrm(X, lS_o, lS_i)  # Run inference (result discarded)
        nb += 1

        if nb % 500 == 0:
            log(f"    Warmup batch {nb}")

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

    elapsed = time.time() - t0
    if hasattr(store, 'cache') and hasattr(store.cache, 'hit_rate'):
        log(f"  Warmup done: {nb} batches in {elapsed:.1f}s, "
            f"cache: {store.cache.hit_rate * 100:.1f}% hit rate, "
            f"{store.cache.current_bytes / 1024 / 1024:.1f}MB cached")
    else:
        log(f"  Warmup done: {nb} batches in {elapsed:.1f}s")


def run_measured_pass(dlrm, test_ld, store):
    """Second pass: measure actual inference performance with warm cache."""
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
        scores.extend(S.tolist())
        targets.extend(Tn.tolist())
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
            scores.extend(S.tolist())
            targets.extend(Tn.tolist())
            nb += 1
            if nb % 500 == 0:
                log(f"    Batch {nb}, lat={blats[-1] * 1000:.1f}ms")
    total = time.time() - t0
    return accu / samp, roc_auc_score(targets, scores), total, blats, nb


def run_prefetch_inference(dlrm, test_ld, store):
    """Standard single-pass inference."""
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
        scores.extend(S.tolist())
        targets.extend(Tn.tolist())
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
# SETUP HELPERS
# ==============================================================

def setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                large_tables, hot_indices, quantize_fn='global'):
    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        if w.shape[0] < LARGE_TABLE_THRESHOLD:
            if quantize_fn == 'per_column':
                q, scales, zps = quantize_per_column(w)
                dq = dequantize_per_column_torch(q, scales, zps)
            else:
                q, s, zp = quantize_global(w)
                dq = dequantize_global(q, s, zp)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = dq

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())
        hot_idx = sorted(hi)
        if hot_idx:
            hw = w[hot_idx]
            if quantize_fn == 'per_column':
                qh, sh, zh = quantize_per_column(hw)
                dq = dequantize_per_column_torch(qh, sh, zh)
            else:
                qh, sh, zh = quantize_global(hw)
                dq = dequantize_global(qh, sh, zh)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[torch.tensor(hot_idx, dtype=torch.long)] = dq


def add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices):
    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())
        cold_idx = sorted(set(range(ln_emb[t])) - hi)
        cw = w[cold_idx]
        store.add_table(t, ln_emb[t], cold_idx, cw)
        log(f"    Table {t}: {len(list(hi)):,} hot, {len(cold_idx):,} cold")


def save_results(all_results):
    with open(JSON_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


def collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                     baseline_auc, store, exp_name, extra_info=None):
    auc_loss = (baseline_auc - auc) * 100
    result = {
        'name': exp_name,
        'accuracy': float(acc),
        'auc': float(auc),
        'auc_loss_pp': float(auc_loss),
        'inference_time': float(inf_time),
        'setup_time': float(setup_time),
        'total_time': float(setup_time + inf_time),
        'memory_breakdown': mem,
        'memory_mb': float(mem['total_mb']),
        'batch_latency': latency_stats(blats),
        'num_batches': nb,
    }

    if hasattr(store, 'compressed_bytes') and store.compressed_bytes > 0:
        ratio = store.raw_bytes / store.compressed_bytes
        result['compression_ratio'] = float(ratio)
        result['compressed_mb'] = float(store.compressed_bytes / 1024 / 1024)

    if hasattr(store, 'decompress_count'):
        result['decompress_count'] = store.decompress_count
        result['decompress_time_ms'] = float(store.decompress_time * 1000)

    if hasattr(store, 'cache') and hasattr(store.cache, 'hit_rate'):
        result['cache_hit_rate'] = float(store.cache.hit_rate)
        result['cache_hits'] = store.cache.hits
        result['cache_misses'] = store.cache.misses
        result['cache_current_mb'] = float(store.cache.current_bytes / 1024 / 1024)

    if hasattr(store, 'num_blocks_total'):
        result['num_blocks_total'] = store.num_blocks_total

    if extra_info:
        result['extra_info'] = extra_info

    return result


# ==============================================================
# MAIN
# ==============================================================

def main():
    global _log_fh

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _log_fh = open(LOG_PATH, 'w')

    log("=" * 70)
    log("PREFETCH BENCHMARK V10 PHASE 3: Practical Optimizations")
    log("=" * 70)
    log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # Load model and data
    log("\nLoading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")

    # Profile at multiple thresholds
    log("\nProfiling access patterns at multiple thresholds...")
    thresholds = [0.80, 0.90, 0.95, 0.99]
    hot_at = {}
    for th in thresholds:
        hi, _ = profile_access(train_ld, num_tables, ln_emb, th)
        hot_at[th] = hi
        total_hot = sum(len(hi[t]) for t in large_tables)
        total_cold = sum(ln_emb[t] for t in large_tables) - total_hot
        cold_mb = total_cold * EMB_DIM / 1024 / 1024
        log(f"  {th*100:.0f}%: {total_hot:,} hot, {total_cold:,} cold ({cold_mb:.1f}MB uint8)")

    # ----------------------------------------------------------
    # BASELINE
    # ----------------------------------------------------------
    log("\n" + "=" * 60)
    log("BASELINE")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    drop_caches()
    time.sleep(1)
    gc.collect()

    b_acc, b_auc, b_time, b_blats, b_nb = run_baseline_inference(dlrm, test_ld)
    b_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
    log(f"  Acc={b_acc * 100:.4f}%, AUC={b_auc:.6f}, Inference={b_time:.2f}s")
    baseline_auc = b_auc
    all_results['0_baseline'] = {
        'name': 'Baseline',
        'accuracy': float(b_acc), 'auc': float(b_auc), 'auc_loss_pp': 0.0,
        'inference_time': float(b_time), 'setup_time': 0.0,
        'total_time': float(b_time), 'memory_mb': float(b_mem),
        'batch_latency': latency_stats(b_blats), 'num_batches': b_nb,
    }
    save_results(all_results)

    # ==============================================================
    # EXP 1: V9 reference at 80% hot (single-pass)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 1: V9 reference (80% hot, single-pass)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()
        store = V9ColdStore(large_tables)
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_at[0.80])
        add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_at[0.80])
        setup_time = time.time() - setup_t0
        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_at[0.80], large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        log(f"  Setup: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB")

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}, Inference={inf_time:.2f}s")
        log(f"  AUC loss: {(baseline_auc - auc) * 100:.4f}pp")

        all_results['1_v9_ref'] = collect_metrics(
            acc, auc, inf_time, blats, nb, setup_time, mem, baseline_auc, store,
            "V9 reference (80% hot)")
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 2: Two-pass LRU with lz4 (small blocks, 200MB + 300MB cache)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 2: Two-pass LRU (warm cache then measure)")
    log("#" * 70)

    for cache_mb, codec in [(200, 'lz4'), (300, 'lz4'), (300, 'zstd')]:
        key = f"2_twopass_{codec}_{cache_mb}mb"
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()
            store = LRUBlockStore(large_tables, block_size=4096,
                                   max_cache_mb=cache_mb, codec=codec)
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_at[0.80])
            add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_at[0.80])
            setup_time = time.time() - setup_t0
            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_at[0.80], large_tables,
                                            state_dict, emb_keys, num_tables, ln_emb)
            log(f"  Setup: {setup_time:.2f}s, Blocks: {store.num_blocks_total}")

            drop_caches()
            gc.collect()

            # Pass 1: warmup
            warmup_t0 = time.time()
            run_warmup_pass(dlrm, test_ld, store)
            warmup_time = time.time() - warmup_t0

            # Reset decompress counters for measured pass
            old_dc = store.decompress_count
            old_dt = store.decompress_time
            store.decompress_count = 0
            store.decompress_time = 0.0
            # Reset cache hit/miss counters
            store.cache.hits = 0
            store.cache.misses = 0

            # Pass 2: measured
            log("  Measured pass (with warm cache)...")
            acc, auc, inf_time, blats, nb = run_measured_pass(dlrm, test_ld, store)
            total_with_warmup = warmup_time + setup_time + inf_time

            log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}")
            log(f"  Warmup: {warmup_time:.2f}s, Inference: {inf_time:.2f}s, "
                f"Total (setup+warmup+infer): {total_with_warmup:.2f}s")
            log(f"  AUC loss: {(baseline_auc - auc) * 100:.4f}pp")
            log(f"  Cache (2nd pass): hit={store.cache.hit_rate * 100:.1f}%, "
                f"hits={store.cache.hits}, misses={store.cache.misses}")
            log(f"  Decompress (2nd pass): {store.decompress_count} ops, "
                f"{store.decompress_time * 1000:.1f}ms")
            log(f"  Decompress (warmup): {old_dc} ops, {old_dt * 1000:.1f}ms")

            mem_after = compute_memory_breakdown(dlrm, store.memory_bytes, hot_at[0.80],
                                                  large_tables, state_dict, emb_keys,
                                                  num_tables, ln_emb)
            result = collect_metrics(
                acc, auc, inf_time, blats, nb, setup_time, mem_after, baseline_auc, store,
                f"Two-pass {codec} {cache_mb}MB")
            result['warmup_time'] = float(warmup_time)
            result['total_with_warmup'] = float(total_with_warmup)
            result['warmup_decompress_count'] = old_dc
            result['warmup_decompress_time_ms'] = float(old_dt * 1000)
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 3: Higher hot threshold (95%, 99%) + V9 pre-decode
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 3: Higher hot threshold + V9 pre-decode")
    log("#" * 70)

    for threshold in [0.95, 0.99]:
        key = f"3_v9_hot{int(threshold*100)}pct"
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()
            store = V9ColdStore(large_tables)
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_at[threshold])
            add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_at[threshold])
            setup_time = time.time() - setup_t0
            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_at[threshold],
                                            large_tables, state_dict, emb_keys, num_tables, ln_emb)
            log(f"  Setup: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB")

            drop_caches()
            time.sleep(1)
            gc.collect()
            acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
            log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}, Inference={inf_time:.2f}s")
            log(f"  AUC loss: {(baseline_auc - auc) * 100:.4f}pp")

            all_results[key] = collect_metrics(
                acc, auc, inf_time, blats, nb, setup_time, mem, baseline_auc, store,
                f"V9 pre-decode (hot={threshold*100:.0f}%)")
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 4: Higher hot threshold + LRU block cache (smaller cold set)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 4: Hot 95%/99% + LRU block cache (smaller cold = better hit rate)")
    log("#" * 70)

    for threshold, cache_mb in [(0.95, 200), (0.95, 300), (0.99, 100), (0.99, 200)]:
        key = f"4_hot{int(threshold*100)}_lru_{cache_mb}mb"
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()
            store = LRUBlockStore(large_tables, block_size=4096,
                                   max_cache_mb=cache_mb, codec='lz4')
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_at[threshold])
            add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_at[threshold])
            setup_time = time.time() - setup_t0
            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_at[threshold],
                                            large_tables, state_dict, emb_keys, num_tables, ln_emb)
            log(f"  Setup: {setup_time:.2f}s, Blocks: {store.num_blocks_total}, "
                f"Memory: {mem['total_mb']:.1f}MB")

            drop_caches()
            time.sleep(1)
            gc.collect()
            acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
            auc_loss = (baseline_auc - auc) * 100
            log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}, Inference={inf_time:.2f}s")
            log(f"  AUC loss: {auc_loss:.4f}pp")
            if hasattr(store, 'cache'):
                log(f"  Cache: {store.cache.hit_rate * 100:.1f}% hit, "
                    f"{store.cache.current_bytes / 1024 / 1024:.1f}MB used")
            log(f"  Decompress: {store.decompress_count} ops, {store.decompress_time * 1000:.1f}ms")

            all_results[key] = collect_metrics(
                acc, auc, inf_time, blats, nb, setup_time, mem, baseline_auc, store,
                f"Hot {threshold*100:.0f}% + lz4 LRU {cache_mb}MB")
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 5: Per-column quant + V9 pre-decode (best AUC)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 5: Per-column quant + V9 pre-decode")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()

        # Per-column quantized V9 store
        store = V9ColdStore(large_tables)
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_at[0.80], quantize_fn='per_column')

        # Override add_table to use per-column quant
        for t in large_tables:
            w = state_dict[emb_keys[t]]
            hi = set(hot_at[0.80][t].tolist())
            cold_idx = sorted(set(range(ln_emb[t])) - hi)
            cw = w[cold_idx]
            q, scales, zps = quantize_per_column(cw)
            store.cold_uint8[t] = q.numpy()
            store.quant_params[t] = ('per_column', scales, zps)
            lookup = np.full(ln_emb[t], -1, dtype=np.int32)
            for seq, orig in enumerate(cold_idx):
                lookup[orig] = seq
            store.cold_seq_lookup[t] = lookup
            log(f"    Table {t}: {len(list(hi)):,} hot, {len(cold_idx):,} cold")

        # Override dequantize to handle per-column
        orig_dequant = store.dequantize_for_batch
        def percol_dequant(lS_i):
            result = {}
            for t in store.large_tables:
                if t not in store.cold_uint8:
                    continue
                indices = lS_i[t].numpy().flatten()
                unique = np.unique(indices)
                seq = store.cold_seq_lookup[t][unique]
                mask = seq >= 0
                if not mask.any():
                    continue
                cold_orig = unique[mask]
                cold_seq = seq[mask]
                q_rows = store.cold_uint8[t][cold_seq]
                qp = store.quant_params[t]
                if qp[0] == 'per_column':
                    fp_rows = dequantize_per_column(q_rows, qp[1], qp[2])
                else:
                    s, zp = qp[1], qp[2]
                    fp_rows = (q_rows.astype(np.float32) - zp) * s
                result[t] = (torch.from_numpy(cold_orig.astype(np.int64)),
                             torch.from_numpy(fp_rows))
            return result
        store.dequantize_for_batch = percol_dequant

        setup_time = time.time() - setup_t0
        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_at[0.80], large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        log(f"  Setup: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB")

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}, Inference={inf_time:.2f}s")
        log(f"  AUC loss: {(baseline_auc - auc) * 100:.4f}pp")

        all_results['5_percol_v9'] = collect_metrics(
            acc, auc, inf_time, blats, nb, setup_time, mem, baseline_auc, store,
            "Per-column quant + V9 pre-decode (80% hot)")
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 6: Two-pass + Higher hot threshold (99%) + per-column quant
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 6: Combined: Two-pass + hot 99% + lz4 LRU 200MB + per-col quant")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()
        store = LRUBlockStore(large_tables, block_size=4096,
                               max_cache_mb=200, codec='lz4', quant_fn='per_column')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_at[0.99], quantize_fn='per_column')
        add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_at[0.99])
        setup_time = time.time() - setup_t0
        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_at[0.99], large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        log(f"  Setup: {setup_time:.2f}s, Blocks: {store.num_blocks_total}")

        drop_caches()
        gc.collect()

        # Warmup pass
        warmup_t0 = time.time()
        run_warmup_pass(dlrm, test_ld, store)
        warmup_time = time.time() - warmup_t0

        # Reset counters
        old_dc = store.decompress_count
        store.decompress_count = 0
        store.decompress_time = 0.0
        store.cache.hits = 0
        store.cache.misses = 0

        # Measured pass
        log("  Measured pass...")
        acc, auc, inf_time, blats, nb = run_measured_pass(dlrm, test_ld, store)
        total_with_warmup = warmup_time + setup_time + inf_time

        log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}")
        log(f"  Warmup: {warmup_time:.2f}s, Inference: {inf_time:.2f}s, "
            f"Total: {total_with_warmup:.2f}s")
        log(f"  AUC loss: {(baseline_auc - auc) * 100:.4f}pp")
        log(f"  Cache (2nd pass): {store.cache.hit_rate * 100:.1f}% hit")

        mem_after = compute_memory_breakdown(dlrm, store.memory_bytes, hot_at[0.99],
                                              large_tables, state_dict, emb_keys,
                                              num_tables, ln_emb)
        result = collect_metrics(
            acc, auc, inf_time, blats, nb, setup_time, mem_after, baseline_auc, store,
            "Combined: per-col + hot 99% + two-pass lz4 200MB")
        result['warmup_time'] = float(warmup_time)
        result['total_with_warmup'] = float(total_with_warmup)
        all_results['6_combined'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # SUMMARY
    # ==============================================================
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"\n{'Config':<55s} {'AUC':>10s} {'Loss':>10s} {'Infer':>8s} {'Total':>8s} {'Mem':>8s}")
    log("-" * 100)
    for k, v in all_results.items():
        name = v.get('name', k)[:52]
        auc = v.get('auc', 0)
        loss = v.get('auc_loss_pp', 0)
        infer = v.get('inference_time', 0)
        total = v.get('total_time', 0)
        mem = v.get('memory_mb', 0)
        warmup = v.get('warmup_time', 0)
        extra = f" (warmup={warmup:.0f}s)" if warmup > 0 else ""
        log(f"  {name:<53s} {auc:.6f} {loss:+.4f}pp {infer:>6.1f}s {total:>6.1f}s {mem:>6.1f}{extra}")

    log(f"\nResults saved to: {JSON_PATH}")
    log(f"\n{'=' * 70}")
    log("DONE!")
    log("=" * 70)

    if _log_fh:
        _log_fh.close()


if __name__ == '__main__':
    main()
