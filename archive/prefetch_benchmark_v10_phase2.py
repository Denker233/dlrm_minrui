#!/usr/bin/env python3
"""
Prefetch Benchmark V10 Phase 2: Frequency-Sorted Large Blocks

Key insight from Phase 1: LRU cache with 4096-row blocks gives 55MB memory
but 3-5x slower inference (too many cache misses). Meanwhile, 4K-video blocks
(1.5M rows) give only ~27 blocks total, all cached early = near-baseline speed.

The gap: How to get fast inference (~50-60s) with low memory (~50-150MB)?

Solution: Sort cold rows by access frequency BEFORE blocking. This clusters
the most-accessed cold rows in the first blocks. An LRU cache retains hot
blocks, giving high cache hit rates even with small memory budgets.

Experiments:
  1. Large-block LRU (unsorted baseline) - 4K blocks + memory budgets
  2. Frequency-sorted large blocks - THE KEY EXPERIMENT
  3. Pinned + LRU (pinned hot blocks, LRU for rest)
  4. Partial pre-decode (hot blocks decoded at startup)
  5. Per-column quant + freq-sorted (quality + speed)
  6. Combined optimal

Reference results:
  Baseline: AUC=0.802698, Inference=43.9s, Memory=2062.5MB
  V9 uint8: AUC=0.802697, Inference=44.6s, Setup=9.4s, Memory=520.8MB
  LRU 300MB (4K-row blocks): Inference=140.4s, Memory=55.1MB
  H.265 4K predecode: Inference=59.2s, Memory=520.8+37.2MB, Disk=37.2MB
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
HOT_THRESHOLD = 0.80
PROFILE_BATCHES = 200
LARGE_TABLE_THRESHOLD = 50000

# 4K video resolution block size: 3840*2160/16 = 518400 rows
# We want ~20-30 blocks total across all 8 tables
LARGE_BLOCK_SIZE = 518400  # rows per block (~7.9MB per block in uint8)

os.makedirs(RESULTS_DIR, exist_ok=True)

JSON_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_phase2.json")
LOG_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_phase2.log")

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
# SHARED UTILITIES (from v10_optimize.py)
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


def zstd_compress(data, level=3):
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data)
    else:
        import zlib
        return zlib.compress(data, min(level, 9))


def zstd_decompress(data):
    if HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    else:
        import zlib
        return zlib.decompress(data)


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
# HELPER: Sort cold rows by access frequency
# ==============================================================

def sort_cold_by_frequency(cold_indices, cold_weights, access_counts_for_table):
    """Sort cold rows by descending access frequency.

    Args:
        cold_indices: list of original indices (sorted by index)
        cold_weights: torch.Tensor (N, 16) of cold row weights
        access_counts_for_table: tuple (unique_indices, counts) sorted by freq desc

    Returns:
        sorted_cold_indices: np.array of original indices sorted by freq desc
        sorted_cold_weights: torch.Tensor sorted by freq desc
        cold_freq_counts: np.array of access counts for each cold row (sorted)
    """
    all_unique, all_counts = access_counts_for_table
    # Build freq dict for cold rows
    freq_dict = {}
    for u, c in zip(all_unique, all_counts):
        freq_dict[int(u)] = int(c)

    # Get freq count for each cold row
    cold_freq = np.array([freq_dict.get(ci, 0) for ci in cold_indices], dtype=np.int64)

    # Sort by descending frequency (stable sort to preserve order among ties)
    order = np.argsort(-cold_freq, kind='stable')

    sorted_indices = np.array([cold_indices[i] for i in order], dtype=np.int64)
    sorted_weights = cold_weights[order]
    sorted_freq = cold_freq[order]

    return sorted_indices, sorted_weights, sorted_freq


# ==============================================================
# HELPER: Profile block-level access
# ==============================================================

def profile_block_access(cold_seq_lookup, block_size, access_counts_for_table, num_blocks):
    """Compute per-block access counts for a table.

    Returns: np.array of shape (num_blocks,) with total access count per block.
    """
    all_unique, all_counts = access_counts_for_table
    block_counts = np.zeros(num_blocks, dtype=np.int64)

    for u, c in zip(all_unique, all_counts):
        seq = cold_seq_lookup[int(u)]
        if seq >= 0:
            bid = seq // block_size
            if bid < num_blocks:
                block_counts[bid] += c

    return block_counts


# ==============================================================
# EXP 1: Large-Block LRU Store (unsorted)
# ==============================================================

class LargeBlockLRUStore:
    """LRU block cache with large blocks. Cold rows in original order (not freq-sorted)."""

    def __init__(self, large_tables, block_size=LARGE_BLOCK_SIZE, max_cache_mb=150):
        self.large_tables = large_tables
        self.block_size = block_size
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
        q, s, zp = quantize_global(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (s, zp)

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
            compressed = zstd_compress(raw, level=3)
            table_blocks.append((compressed, end - start))
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
        compressed, num_rows = self.blocks[t][block_id]
        raw = zstd_decompress(compressed)
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
        return self.compressed_bytes + self.cache.current_bytes


# ==============================================================
# EXP 2: Frequency-Sorted Large-Block LRU Store
# ==============================================================

class FreqSortedLargeBlockStore:
    """Large blocks with cold rows sorted by access frequency (descending).
    Hot cold rows cluster in early blocks -> LRU cache retains them."""

    def __init__(self, large_tables, block_size=LARGE_BLOCK_SIZE, max_cache_mb=150):
        self.large_tables = large_tables
        self.block_size = block_size
        self.blocks = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.cache = LRUCache(max_cache_mb * 1024 * 1024)
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.num_blocks_total = 0
        self.block_access_stats = {}  # {t: np.array of per-block access counts}

    def add_table(self, t, num_emb, cold_indices, cold_weights,
                  cold_access_counts=None, access_counts_for_table=None):
        """Add table with frequency-sorted cold rows.

        cold_indices/cold_weights are already sorted by frequency (desc)
        if the caller pre-sorted them."""
        q, s, zp = quantize_global(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (s, zp)

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
            compressed = zstd_compress(raw, level=3)
            table_blocks.append((compressed, end - start))
            self.raw_bytes += len(raw)
            self.compressed_bytes += len(compressed)
        self.blocks[t] = table_blocks
        self.num_blocks_total += len(table_blocks)

        # Compute per-block access stats
        if access_counts_for_table is not None:
            num_blks = len(table_blocks)
            self.block_access_stats[t] = profile_block_access(
                lookup, self.block_size, access_counts_for_table, num_blks)

    def _get_block(self, t, block_id):
        key = (t, block_id)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        t0 = time.time()
        compressed, num_rows = self.blocks[t][block_id]
        raw = zstd_decompress(compressed)
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
        return self.compressed_bytes + self.cache.current_bytes


# ==============================================================
# EXP 3: Pinned Hot Blocks + LRU
# ==============================================================

class PinnedBlockStore:
    """Pin the most-accessed blocks permanently (never evicted).
    Remaining blocks use LRU cache."""

    def __init__(self, large_tables, block_size=LARGE_BLOCK_SIZE,
                 pin_budget_mb=100, lru_budget_mb=50):
        self.large_tables = large_tables
        self.block_size = block_size
        self.blocks = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}

        self.pin_budget = pin_budget_mb * 1024 * 1024
        self.pinned = {}         # {(t, block_id): np.array}
        self.pinned_bytes = 0
        self.pinned_hits = 0

        self.cache = LRUCache(lru_budget_mb * 1024 * 1024)
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.num_blocks_total = 0

    def add_table(self, t, num_emb, cold_indices, cold_weights,
                  access_counts_for_table=None):
        q, s, zp = quantize_global(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (s, zp)

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
            compressed = zstd_compress(raw, level=3)
            table_blocks.append((compressed, end - start))
            self.raw_bytes += len(raw)
            self.compressed_bytes += len(compressed)
        self.blocks[t] = table_blocks
        self.num_blocks_total += len(table_blocks)

    def pin_hot_blocks(self, access_counts, large_tables):
        """After all tables added, pin the most-accessed blocks."""
        # Collect all (table, block_id, access_count, block_size_bytes)
        candidates = []
        for t in large_tables:
            if t not in self.blocks:
                continue
            num_blks = len(self.blocks[t])
            blk_counts = profile_block_access(
                self.cold_seq_lookup[t], self.block_size,
                access_counts[t], num_blks)
            for bid in range(num_blks):
                _, num_rows = self.blocks[t][bid]
                block_bytes = num_rows * EMB_DIM
                candidates.append((blk_counts[bid], t, bid, block_bytes))

        # Sort by access count descending, pin greedily
        candidates.sort(key=lambda x: -x[0])
        for count, t, bid, block_bytes in candidates:
            if self.pinned_bytes + block_bytes > self.pin_budget:
                continue
            # Decompress and pin
            compressed, num_rows = self.blocks[t][bid]
            raw = zstd_decompress(compressed)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(num_rows, EMB_DIM).copy()
            self.pinned[(t, bid)] = arr
            self.pinned_bytes += block_bytes

        log(f"  Pinned {len(self.pinned)} blocks ({self.pinned_bytes / 1024 / 1024:.1f}MB)")

    def _get_block(self, t, block_id):
        key = (t, block_id)
        # Check pinned first
        if key in self.pinned:
            self.pinned_hits += 1
            return self.pinned[key]
        # Then LRU
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        t0 = time.time()
        compressed, num_rows = self.blocks[t][block_id]
        raw = zstd_decompress(compressed)
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
        return self.compressed_bytes + self.pinned_bytes + self.cache.current_bytes


# ==============================================================
# EXP 4: Partial Pre-Decode (hot blocks decoded at startup)
# ==============================================================

class PartialPreDecodeStore:
    """Decode hot blocks at startup (like V9 for those blocks).
    Keep cold blocks compressed. On miss: decompress transiently."""

    def __init__(self, large_tables, block_size=LARGE_BLOCK_SIZE,
                 predecode_budget_mb=150):
        self.large_tables = large_tables
        self.block_size = block_size
        self.blocks = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}

        self.predecode_budget = predecode_budget_mb * 1024 * 1024
        self.predecoded = {}     # {(t, block_id): np.array}
        self.predecoded_bytes = 0

        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.num_blocks_total = 0
        self.predecoded_hits = 0
        self.cold_hits = 0

    def add_table(self, t, num_emb, cold_indices, cold_weights,
                  access_counts_for_table=None):
        """cold_indices/cold_weights should be freq-sorted (hot first)."""
        q, s, zp = quantize_global(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (s, zp)

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
            compressed = zstd_compress(raw, level=3)
            table_blocks.append((compressed, end - start))
            self.raw_bytes += len(raw)
            self.compressed_bytes += len(compressed)
        self.blocks[t] = table_blocks
        self.num_blocks_total += len(table_blocks)

    def predecode_hot_blocks(self):
        """After all tables added, pre-decode the first N blocks (hottest due to freq sort)."""
        # With freq sorting, block 0 of each table has the most-accessed rows
        # Greedily decode blocks in round-robin across tables
        remaining = self.predecode_budget
        # Collect all blocks with their table and block_id
        all_blocks = []
        for t in self.large_tables:
            if t not in self.blocks:
                continue
            for bid in range(len(self.blocks[t])):
                _, num_rows = self.blocks[t][bid]
                block_bytes = num_rows * EMB_DIM
                all_blocks.append((t, bid, block_bytes))

        # Decode block 0 of each table first, then block 1, etc.
        max_blks = max(len(self.blocks[t]) for t in self.large_tables if t in self.blocks)
        for bid in range(max_blks):
            for t in self.large_tables:
                if t not in self.blocks:
                    continue
                if bid >= len(self.blocks[t]):
                    continue
                _, num_rows = self.blocks[t][bid]
                block_bytes = num_rows * EMB_DIM
                if self.predecoded_bytes + block_bytes > self.predecode_budget:
                    continue
                compressed, nr = self.blocks[t][bid]
                raw = zstd_decompress(compressed)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(nr, EMB_DIM).copy()
                self.predecoded[(t, bid)] = arr
                self.predecoded_bytes += block_bytes

        log(f"  Pre-decoded {len(self.predecoded)} blocks "
            f"({self.predecoded_bytes / 1024 / 1024:.1f}MB)")

    def _get_block(self, t, block_id):
        key = (t, block_id)
        if key in self.predecoded:
            self.predecoded_hits += 1
            return self.predecoded[key]
        # Cold miss: decompress transiently
        self.cold_hits += 1
        t0 = time.time()
        compressed, num_rows = self.blocks[t][block_id]
        raw = zstd_decompress(compressed)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(num_rows, EMB_DIM).copy()
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
        return self.compressed_bytes + self.predecoded_bytes


# ==============================================================
# EXP 5: Per-Column Quant + Freq-Sorted Large Blocks
# ==============================================================

class PerColFreqSortedStore:
    """Combines per-column quantization (better AUC) with freq-sorted large blocks."""

    def __init__(self, large_tables, block_size=LARGE_BLOCK_SIZE, max_cache_mb=150):
        self.large_tables = large_tables
        self.block_size = block_size
        self.blocks = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}  # {t: (scales_16, zps_16)}
        self.cache = LRUCache(max_cache_mb * 1024 * 1024)
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.num_blocks_total = 0

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        """cold_indices/cold_weights should be freq-sorted."""
        q, scales, zps = quantize_per_column(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (scales, zps)

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
            compressed = zstd_compress(raw, level=3)
            table_blocks.append((compressed, end - start))
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
        compressed, num_rows = self.blocks[t][block_id]
        raw = zstd_decompress(compressed)
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

            scales, zps = self.quant_params[t]
            fp_rows = dequantize_per_column(q_rows, scales, zps)
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
# EXP 6: Combined Optimal
# ==============================================================

class CombinedOptimalStore:
    """Per-column quant + freq-sorted + partial pre-decode + LRU for rest.
    Best quality + fast inference + bounded memory."""

    def __init__(self, large_tables, block_size=LARGE_BLOCK_SIZE,
                 predecode_budget_mb=100, lru_budget_mb=50):
        self.large_tables = large_tables
        self.block_size = block_size
        self.blocks = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}

        self.predecode_budget = predecode_budget_mb * 1024 * 1024
        self.predecoded = {}
        self.predecoded_bytes = 0
        self.predecoded_hits = 0

        self.cache = LRUCache(lru_budget_mb * 1024 * 1024)
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.num_blocks_total = 0

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        """cold_indices/cold_weights should be freq-sorted."""
        q, scales, zps = quantize_per_column(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (scales, zps)

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
            compressed = zstd_compress(raw, level=3)
            table_blocks.append((compressed, end - start))
            self.raw_bytes += len(raw)
            self.compressed_bytes += len(compressed)
        self.blocks[t] = table_blocks
        self.num_blocks_total += len(table_blocks)

    def predecode_hot_blocks(self):
        """Pre-decode the first N blocks across tables (hottest due to freq sort)."""
        max_blks = max(len(self.blocks[t]) for t in self.large_tables if t in self.blocks)
        for bid in range(max_blks):
            for t in self.large_tables:
                if t not in self.blocks or bid >= len(self.blocks[t]):
                    continue
                _, num_rows = self.blocks[t][bid]
                block_bytes = num_rows * EMB_DIM
                if self.predecoded_bytes + block_bytes > self.predecode_budget:
                    continue
                compressed, nr = self.blocks[t][bid]
                raw = zstd_decompress(compressed)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(nr, EMB_DIM).copy()
                self.predecoded[(t, bid)] = arr
                self.predecoded_bytes += block_bytes

        log(f"  Pre-decoded {len(self.predecoded)} blocks "
            f"({self.predecoded_bytes / 1024 / 1024:.1f}MB)")

    def _get_block(self, t, block_id):
        key = (t, block_id)
        if key in self.predecoded:
            self.predecoded_hits += 1
            return self.predecoded[key]
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        t0 = time.time()
        compressed, num_rows = self.blocks[t][block_id]
        raw = zstd_decompress(compressed)
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

            scales, zps = self.quant_params[t]
            fp_rows = dequantize_per_column(q_rows, scales, zps)
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
        return self.compressed_bytes + self.predecoded_bytes + self.cache.current_bytes


# ==============================================================
# SETUP + EXPERIMENT RUNNER
# ==============================================================

def setup_model_for_experiment(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                                large_tables, hot_indices, quantize_fn='global'):
    """Set up small tables and hot rows in model."""
    # Small tables: quantize round-trip
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

    # Hot rows: quantize round-trip
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


def save_results(all_results):
    with open(JSON_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


def run_experiment(exp_name, exp_key, dlrm, test_ld, state_dict, emb_keys,
                   ln_emb, num_tables, large_tables, hot_indices, baseline_auc,
                   store, extra_setup_fn=None, quantize_fn='global',
                   extra_info=None):
    """Run experiment: restore -> setup -> inference -> collect metrics."""
    log(f"\n{'=' * 60}")
    log(f"EXPERIMENT {exp_key}: {exp_name}")
    log(f"{'=' * 60}")

    restore_weights(dlrm, state_dict, emb_keys)
    gc.collect()

    setup_t0 = time.time()
    setup_model_for_experiment(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                                large_tables, hot_indices, quantize_fn)

    # Add cold tables
    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())
        cold_idx = sorted(set(range(ln_emb[t])) - hi)
        cw = w[cold_idx]
        store.add_table(t, ln_emb[t], cold_idx, cw)
        log(f"    Table {t}: {len(list(hi)):,} hot, {len(cold_idx):,} cold")

    # Extra setup (pinning, pre-decode, etc.)
    if extra_setup_fn:
        extra_setup_fn()

    setup_time = time.time() - setup_t0

    mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                    state_dict, emb_keys, num_tables, ln_emb)
    log(f"  Setup: {setup_time:.2f}s")
    log(f"  Cold storage: {mem['cold_store_mb']:.1f}MB, Total: {mem['total_mb']:.1f}MB")
    if hasattr(store, 'num_blocks_total'):
        log(f"  Total blocks: {store.num_blocks_total}")

    drop_caches()
    time.sleep(1)
    gc.collect()

    acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
    auc_loss = (baseline_auc - auc) * 100

    log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}")
    log(f"  Inference={inf_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time + inf_time:.2f}s")
    log(f"  Memory: {mem}")
    log(f"  AUC loss vs baseline: {auc_loss:.4f}pp")
    log(f"  Batch: mean={np.mean(blats) * 1000:.1f}ms, "
        f"p50={np.percentile(blats, 50) * 1000:.1f}ms, "
        f"p95={np.percentile(blats, 95) * 1000:.1f}ms, "
        f"p99={np.percentile(blats, 99) * 1000:.1f}ms")

    result = {
        'name': exp_name,
        'experiment_key': exp_key,
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

    # Store-specific metrics
    if hasattr(store, 'compressed_bytes') and store.compressed_bytes > 0:
        ratio = store.raw_bytes / store.compressed_bytes
        log(f"  Compression: ratio={ratio:.2f}x, "
            f"raw={store.raw_bytes / 1024 / 1024:.1f}MB, "
            f"compressed={store.compressed_bytes / 1024 / 1024:.1f}MB")
        result['compression_ratio'] = float(ratio)

    if hasattr(store, 'decompress_count'):
        log(f"  Decompress: count={store.decompress_count}, time={store.decompress_time * 1000:.1f}ms")
        result['decompress_count'] = store.decompress_count
        result['decompress_time_ms'] = float(store.decompress_time * 1000)

    if hasattr(store, 'cache') and hasattr(store.cache, 'hit_rate'):
        log(f"  Cache: hit_rate={store.cache.hit_rate * 100:.1f}%, "
            f"hits={store.cache.hits}, misses={store.cache.misses}, "
            f"current_bytes={store.cache.current_bytes / 1024 / 1024:.1f}MB")
        result['cache_hit_rate'] = float(store.cache.hit_rate)
        result['cache_hits'] = store.cache.hits
        result['cache_misses'] = store.cache.misses

    if hasattr(store, 'pinned_hits'):
        log(f"  Pinned: hits={store.pinned_hits}, bytes={store.pinned_bytes / 1024 / 1024:.1f}MB")
        result['pinned_hits'] = store.pinned_hits
        result['pinned_bytes_mb'] = float(store.pinned_bytes / 1024 / 1024)

    if hasattr(store, 'predecoded_hits'):
        log(f"  Pre-decoded: hits={store.predecoded_hits}, bytes={store.predecoded_bytes / 1024 / 1024:.1f}MB")
        if hasattr(store, 'cold_hits'):
            log(f"  Cold block hits: {store.cold_hits}")
        result['predecoded_hits'] = store.predecoded_hits
        result['predecoded_bytes_mb'] = float(store.predecoded_bytes / 1024 / 1024)

    if hasattr(store, 'num_blocks_total'):
        result['num_blocks_total'] = store.num_blocks_total

    if hasattr(store, 'block_access_stats') and store.block_access_stats:
        # Log top blocks per table
        for t, ba in store.block_access_stats.items():
            total_acc = ba.sum()
            if total_acc > 0:
                top3 = np.argsort(-ba)[:3]
                top3_pct = ba[top3] / total_acc * 100
                log(f"  Table {t} block access: top blocks={list(zip(top3.tolist(), top3_pct.tolist()))}")

    if extra_info:
        result['extra_info'] = extra_info

    return result


def run_experiment_freq_sorted(exp_name, exp_key, dlrm, test_ld, state_dict, emb_keys,
                                ln_emb, num_tables, large_tables, hot_indices, baseline_auc,
                                store, access_counts, extra_setup_fn=None,
                                quantize_fn='global', extra_info=None):
    """Like run_experiment but with frequency-sorted cold rows."""
    log(f"\n{'=' * 60}")
    log(f"EXPERIMENT {exp_key}: {exp_name}")
    log(f"{'=' * 60}")

    restore_weights(dlrm, state_dict, emb_keys)
    gc.collect()

    setup_t0 = time.time()
    setup_model_for_experiment(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                                large_tables, hot_indices, quantize_fn)

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())
        cold_idx = sorted(set(range(ln_emb[t])) - hi)
        cw = w[cold_idx]

        # Sort by frequency
        sorted_ci, sorted_cw, sorted_freq = sort_cold_by_frequency(
            cold_idx, cw, access_counts[t])

        # Check if store has extended add_table signature
        if hasattr(store.add_table, '__code__') and \
           'access_counts_for_table' in store.add_table.__code__.co_varnames:
            store.add_table(t, ln_emb[t], sorted_ci, sorted_cw,
                          access_counts_for_table=access_counts[t])
        else:
            store.add_table(t, ln_emb[t], sorted_ci, sorted_cw)

        log(f"    Table {t}: {len(list(hi)):,} hot, {len(cold_idx):,} cold "
            f"(top-cold freq={sorted_freq[0] if len(sorted_freq) > 0 else 0})")

    if extra_setup_fn:
        extra_setup_fn()

    setup_time = time.time() - setup_t0

    mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                    state_dict, emb_keys, num_tables, ln_emb)
    log(f"  Setup: {setup_time:.2f}s")
    log(f"  Cold storage: {mem['cold_store_mb']:.1f}MB, Total: {mem['total_mb']:.1f}MB")
    if hasattr(store, 'num_blocks_total'):
        log(f"  Total blocks: {store.num_blocks_total}")

    drop_caches()
    time.sleep(1)
    gc.collect()

    acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
    auc_loss = (baseline_auc - auc) * 100

    log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}")
    log(f"  Inference={inf_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time + inf_time:.2f}s")
    log(f"  Memory: {mem}")
    log(f"  AUC loss vs baseline: {auc_loss:.4f}pp")
    log(f"  Batch: mean={np.mean(blats) * 1000:.1f}ms, "
        f"p50={np.percentile(blats, 50) * 1000:.1f}ms, "
        f"p95={np.percentile(blats, 95) * 1000:.1f}ms, "
        f"p99={np.percentile(blats, 99) * 1000:.1f}ms")

    result = {
        'name': exp_name,
        'experiment_key': exp_key,
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
        log(f"  Compression: ratio={ratio:.2f}x")
        result['compression_ratio'] = float(ratio)

    if hasattr(store, 'decompress_count'):
        log(f"  Decompress: count={store.decompress_count}, time={store.decompress_time * 1000:.1f}ms")
        result['decompress_count'] = store.decompress_count
        result['decompress_time_ms'] = float(store.decompress_time * 1000)

    if hasattr(store, 'cache') and hasattr(store.cache, 'hit_rate'):
        log(f"  Cache: hit_rate={store.cache.hit_rate * 100:.1f}%, "
            f"hits={store.cache.hits}, misses={store.cache.misses}")
        result['cache_hit_rate'] = float(store.cache.hit_rate)
        result['cache_hits'] = store.cache.hits
        result['cache_misses'] = store.cache.misses

    if hasattr(store, 'pinned_hits'):
        log(f"  Pinned: hits={store.pinned_hits}")
        result['pinned_hits'] = store.pinned_hits

    if hasattr(store, 'predecoded_hits'):
        log(f"  Pre-decoded: hits={store.predecoded_hits}, cold_hits={getattr(store, 'cold_hits', 0)}")
        result['predecoded_hits'] = store.predecoded_hits
        result['predecoded_bytes_mb'] = float(store.predecoded_bytes / 1024 / 1024)

    if hasattr(store, 'num_blocks_total'):
        result['num_blocks_total'] = store.num_blocks_total

    if hasattr(store, 'block_access_stats') and store.block_access_stats:
        block_stats = {}
        for t, ba in store.block_access_stats.items():
            total_acc = ba.sum()
            if total_acc > 0:
                cumulative = np.cumsum(np.sort(ba)[::-1]) / total_acc * 100
                log(f"  Table {t}: block0 covers {ba[0]/total_acc*100:.1f}% of accesses, "
                    f"top-2: {cumulative[1] if len(cumulative) > 1 else cumulative[0]:.1f}%")
                block_stats[str(t)] = {
                    'block0_pct': float(ba[0] / total_acc * 100),
                    'top2_pct': float(cumulative[min(1, len(cumulative)-1)]),
                    'top3_pct': float(cumulative[min(2, len(cumulative)-1)]),
                }
        result['block_access_stats'] = block_stats

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
    log("PREFETCH BENCHMARK V10 PHASE 2: Frequency-Sorted Large Blocks")
    log("=" * 70)
    log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Block size: {LARGE_BLOCK_SIZE} rows ({LARGE_BLOCK_SIZE * EMB_DIM / 1024 / 1024:.1f}MB per block)")
    log(f"JSON output: {JSON_PATH}")

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

    # Profile access patterns
    log("\nProfiling access patterns...")
    hot_indices, access_counts = profile_access(train_ld, num_tables, ln_emb, HOT_THRESHOLD)
    for t in large_tables:
        log(f"  Table {t}: {ln_emb[t]:,} total, {len(hot_indices[t]):,} hot@{HOT_THRESHOLD*100:.0f}%")

    # Count total cold rows and estimate blocks
    total_cold = 0
    for t in large_tables:
        hi = set(hot_indices[t].tolist())
        total_cold += ln_emb[t] - len(hi)
    est_blocks = (total_cold + LARGE_BLOCK_SIZE - 1) // LARGE_BLOCK_SIZE
    log(f"\nTotal cold rows: {total_cold:,}")
    log(f"Estimated blocks at {LARGE_BLOCK_SIZE} rows/block: ~{est_blocks}")
    log(f"Estimated block size: {LARGE_BLOCK_SIZE * EMB_DIM / 1024 / 1024:.1f}MB")

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
    log(f"  Acc={b_acc * 100:.4f}%, AUC={b_auc:.6f}")
    log(f"  Inference={b_time:.2f}s, Memory={b_mem:.1f}MB")
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
    # EXP 1: Large-Block LRU (unsorted, 3 cache sizes)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 1: Large-Block LRU (unsorted)")
    log("#" * 70)

    for cache_mb in [100, 150, 200]:
        key = f"1_largeblock_lru_{cache_mb}mb"
        try:
            store = LargeBlockLRUStore(large_tables, LARGE_BLOCK_SIZE, cache_mb)
            result = run_experiment(
                f"Large-block LRU {cache_mb}MB (unsorted)", key,
                dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
                large_tables, hot_indices, baseline_auc, store,
                extra_info={'cache_mb': cache_mb, 'block_size': LARGE_BLOCK_SIZE, 'sorted': False})
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 2: Frequency-Sorted Large Blocks (THE KEY EXPERIMENT)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 2: Frequency-Sorted Large Blocks")
    log("#" * 70)

    for cache_mb in [50, 100, 150, 200]:
        key = f"2_freqsorted_{cache_mb}mb"
        try:
            store = FreqSortedLargeBlockStore(large_tables, LARGE_BLOCK_SIZE, cache_mb)
            result = run_experiment_freq_sorted(
                f"Freq-sorted large blocks {cache_mb}MB", key,
                dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
                large_tables, hot_indices, baseline_auc, store, access_counts,
                extra_info={'cache_mb': cache_mb, 'block_size': LARGE_BLOCK_SIZE, 'sorted': True})
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 3: Pinned + LRU (freq-sorted)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 3: Pinned Hot Blocks + LRU (freq-sorted)")
    log("#" * 70)

    for pin_mb, lru_mb in [(100, 50), (75, 75)]:
        key = f"3_pinned_{pin_mb}mb_lru_{lru_mb}mb"
        try:
            store = PinnedBlockStore(large_tables, LARGE_BLOCK_SIZE, pin_mb, lru_mb)
            result = run_experiment_freq_sorted(
                f"Pinned {pin_mb}MB + LRU {lru_mb}MB (freq-sorted)", key,
                dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
                large_tables, hot_indices, baseline_auc, store, access_counts,
                extra_setup_fn=lambda: store.pin_hot_blocks(access_counts, large_tables),
                extra_info={'pin_mb': pin_mb, 'lru_mb': lru_mb, 'sorted': True})
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 4: Partial Pre-Decode (freq-sorted)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 4: Partial Pre-Decode (freq-sorted)")
    log("#" * 70)

    for predecode_mb in [100, 150, 200]:
        key = f"4_predecode_{predecode_mb}mb"
        try:
            store = PartialPreDecodeStore(large_tables, LARGE_BLOCK_SIZE, predecode_mb)
            result = run_experiment_freq_sorted(
                f"Partial pre-decode {predecode_mb}MB (freq-sorted)", key,
                dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
                large_tables, hot_indices, baseline_auc, store, access_counts,
                extra_setup_fn=lambda: store.predecode_hot_blocks(),
                extra_info={'predecode_mb': predecode_mb, 'sorted': True})
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 5: Per-Column Quant + Freq-Sorted
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 5: Per-Column Quant + Freq-Sorted Large Blocks")
    log("#" * 70)

    for cache_mb in [100, 150]:
        key = f"5_percol_freqsorted_{cache_mb}mb"
        try:
            store = PerColFreqSortedStore(large_tables, LARGE_BLOCK_SIZE, cache_mb)
            result = run_experiment_freq_sorted(
                f"Per-col quant + freq-sorted {cache_mb}MB", key,
                dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
                large_tables, hot_indices, baseline_auc, store, access_counts,
                quantize_fn='per_column',
                extra_info={'cache_mb': cache_mb, 'sorted': True, 'quant': 'per_column'})
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 6: Combined Optimal
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 6: Combined Optimal (per-col + freq-sorted + pre-decode + LRU)")
    log("#" * 70)

    for predecode_mb, lru_mb in [(100, 50), (150, 50), (100, 100)]:
        key = f"6_combined_pd{predecode_mb}_lru{lru_mb}"
        try:
            store = CombinedOptimalStore(large_tables, LARGE_BLOCK_SIZE,
                                          predecode_mb, lru_mb)
            result = run_experiment_freq_sorted(
                f"Combined: per-col + predecode {predecode_mb}MB + LRU {lru_mb}MB", key,
                dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
                large_tables, hot_indices, baseline_auc, store, access_counts,
                extra_setup_fn=lambda: store.predecode_hot_blocks(),
                quantize_fn='per_column',
                extra_info={'predecode_mb': predecode_mb, 'lru_mb': lru_mb,
                           'sorted': True, 'quant': 'per_column'})
            all_results[key] = result
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
    log(f"\n{'Config':<55s} {'AUC':>10s} {'Loss':>10s} {'Infer(s)':>10s} {'Total(s)':>10s} {'Mem(MB)':>10s}")
    log("-" * 105)
    for k, v in all_results.items():
        name = v.get('name', k)[:52]
        auc = v.get('auc', 0)
        loss = v.get('auc_loss_pp', 0)
        infer = v.get('inference_time', 0)
        total = v.get('total_time', 0)
        mem = v.get('memory_mb', 0)
        log(f"  {name:<53s} {auc:.6f} {loss:+.4f}pp {infer:>8.1f}s {total:>8.1f}s {mem:>8.1f}")

    log(f"\nResults saved to: {JSON_PATH}")
    log(f"Log saved to: {LOG_PATH}")
    log("\n" + "=" * 70)
    log("DONE!")
    log("=" * 70)

    if _log_fh:
        _log_fh.close()


if __name__ == '__main__':
    main()
