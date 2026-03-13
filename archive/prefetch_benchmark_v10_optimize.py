#!/usr/bin/env python3
"""
Prefetch Benchmark V10 Optimize: Extended optimization exploration.

Runs for several hours exploring 9 experiments across 4 phases:

Phase 1 - Reduce Runtime Memory (the key gap vs CAFE+):
  Exp 1: Partial pre-decode with LRU cache (100/200/300 MB budgets)
  Exp 2: On-demand per-table decode (minimal memory, slow)
  Exp 3: Hybrid memory tiers (top-2000 decoded + LRU for rest)

Phase 2 - Faster Setup:
  Exp 4: Parallel table encoding (multiprocessing Pool)
  Exp 5: Skip encode, load pre-compressed from disk

Phase 3 - Better Compression + Quality:
  Exp 6: Per-column quantization (finer scale/zp per embedding column)
  Exp 7: Mixed precision cold storage (fp16 top-20% + uint8 bottom-80%)
  Exp 8: Hot threshold sweep (70%-99% in 5% steps)

Phase 4 - Combined Best Configuration:
  Exp 9: Best of everything combined

Reference results:
  Baseline: AUC=0.802698, Inference=42-52s, Memory=2062.5MB
  V9 uint8: AUC=0.802692, Inference=43-47s, Setup=9s, Memory=520.8MB
  CAFE+:    AUC=0.8010, 16x comp=0.7882 (1.45pp loss)
"""

import os
import sys
import time
import json
import threading
import gc
import zlib
import subprocess
import struct
import pickle
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

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
COMPRESSED_CACHE_DIR = os.path.join(RESULTS_DIR, "compressed_cache")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 2048
EMB_DIM = 16
HOT_THRESHOLD_DEFAULT = 0.80
PROFILE_BATCHES = 200
LARGE_TABLE_THRESHOLD = 50000
ZSTD_BLOCK_SIZE = 4096  # rows per block

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(COMPRESSED_CACHE_DIR, exist_ok=True)

# Output files
JSON_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_optimize.json")
LOG_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_optimize.log")

# Global log file handle
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
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except Exception:
        pass


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


# ==============================================================
# QUANTIZATION UTILITIES
# ==============================================================

def quantize_global(w):
    """Global uint8 quantization (single scale/zp for entire tensor)."""
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def dequantize_global(q, s, zp):
    """Dequantize from uint8 with global scale/zp."""
    return (q.float() - zp) * s


def quantize_per_column(w):
    """Per-column uint8 quantization: separate scale/zp for each of the 16 embedding columns.
    Returns q (uint8), scales (float array, 16), zps (int array, 16)."""
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
    """Dequantize uint8 numpy array using per-column scales/zps.
    q_np: (N, 16) uint8 numpy, scales: (16,) float, zps: (16,) int.
    Returns: (N, 16) float32 numpy."""
    result = np.empty(q_np.shape, dtype=np.float32)
    for c in range(q_np.shape[1]):
        result[:, c] = (q_np[:, c].astype(np.float32) - zps[c]) * scales[c]
    return result


def dequantize_per_column_torch(q_t, scales, zps):
    """Dequantize uint8 torch tensor using per-column scales/zps.
    q_t: (N, 16) uint8 torch, scales: (16,), zps: (16,).
    Returns: (N, 16) float32 torch."""
    scales_t = torch.from_numpy(scales).float().unsqueeze(0)  # (1, 16)
    zps_t = torch.from_numpy(zps).float().unsqueeze(0)  # (1, 16)
    return (q_t.float() - zps_t) * scales_t


# ==============================================================
# GENERAL UTILITIES
# ==============================================================

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
    """Profile access patterns from training data and compute hot indices at given threshold."""
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
        access_counts[t] = (unique[si], counts[si])  # sorted by frequency descending
    return hot_indices, access_counts


def profile_access_simple(train_ld, num_tables, ln_emb, threshold):
    """Simplified version that only returns hot_indices (no access_counts)."""
    hi, _ = profile_access(train_ld, num_tables, ln_emb, threshold)
    return hi


def compute_memory_breakdown(dlrm, cold_store_bytes, hot_indices, large_tables,
                              state_dict, emb_keys, num_tables, ln_emb):
    """Compute memory breakdown in MB."""
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


def save_results_incremental(all_results):
    """Save results to JSON incrementally (after each experiment)."""
    with open(JSON_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


# ==============================================================
# ZSTD COMPRESSION HELPERS
# ==============================================================

def zstd_compress(data, level=3):
    """Compress bytes with zstd."""
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data)
    else:
        return zlib.compress(data, min(level, 9))


def zstd_decompress(data):
    """Decompress zstd bytes."""
    if HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    else:
        return zlib.decompress(data)


# ==============================================================
# COLD STORE: V9 Reference (uint8 + vectorized dequant)
# ==============================================================

class EntropyColdStore:
    """V9 uint8 cold store with vectorized dequantization."""
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
# EXP 1: LRU BLOCK CACHE with zstd compression
# ==============================================================

class LRUCache:
    """Simple LRU cache with memory budget."""
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.cache = OrderedDict()  # key -> (data, size_bytes)
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
        # Evict until there is room
        while self.current_bytes + size_bytes > self.max_bytes and self.cache:
            _, (_, evicted_size) = self.cache.popitem(last=False)
            self.current_bytes -= evicted_size
        self.cache[key] = (data, size_bytes)
        self.current_bytes += size_bytes

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class LRUBlockColdStore:
    """zstd-compressed blocks with LRU cache budget.
    Only keeps up to max_cache_mb of decoded blocks in memory at a time."""

    def __init__(self, large_tables, block_size=ZSTD_BLOCK_SIZE, max_cache_mb=200):
        self.blocks = {}           # {t: [(compressed_bytes, num_rows), ...]}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.large_tables = large_tables
        self.block_size = block_size
        self.max_cache_mb = max_cache_mb
        self.cache = LRUCache(max_cache_mb * 1024 * 1024)
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0

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
        raw_total = 0
        comp_total = 0
        for start in range(0, n_rows, self.block_size):
            end = min(start + self.block_size, n_rows)
            block_data = q_np[start:end]
            raw = block_data.tobytes()
            compressed = zstd_compress(raw, level=3)
            table_blocks.append((compressed, end - start))
            raw_total += len(raw)
            comp_total += len(compressed)
        self.blocks[t] = table_blocks
        self.raw_bytes += raw_total
        self.compressed_bytes += comp_total

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
        EntropyColdStore.inject(dlrm, result)

    @property
    def memory_bytes(self):
        """Return compressed bytes + current cache usage."""
        return self.compressed_bytes + self.cache.current_bytes


# ==============================================================
# EXP 2: On-Demand Per-Table Decode (minimal memory)
# ==============================================================

class OnDemandColdStore:
    """Keep tables fully compressed. Decompress entire table per batch to temp buffer,
    extract needed rows, discard. Uses minimal persistent memory."""

    def __init__(self, large_tables):
        self.compressed_tables = {}  # {t: compressed_bytes}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.table_shapes = {}       # {t: (num_cold_rows, EMB_DIM)}
        self.large_tables = large_tables
        self.compressed_bytes = 0
        self.raw_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize_global(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (s, zp)
        self.table_shapes[t] = q_np.shape

        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

        raw = q_np.tobytes()
        compressed = zstd_compress(raw, level=3)
        self.compressed_tables[t] = compressed
        self.raw_bytes += len(raw)
        self.compressed_bytes += len(compressed)

    def dequantize_for_batch(self, lS_i):
        result = {}
        for t in self.large_tables:
            if t not in self.compressed_tables:
                continue
            indices = lS_i[t].numpy().flatten()
            unique = np.unique(indices)
            seq = self.cold_seq_lookup[t][unique]
            mask = seq >= 0
            if not mask.any():
                continue
            cold_orig = unique[mask]
            cold_seq = seq[mask]

            # Decompress full table to temp buffer
            t0 = time.time()
            raw = zstd_decompress(self.compressed_tables[t])
            shape = self.table_shapes[t]
            full_table = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            self.decompress_count += 1
            self.decompress_time += time.time() - t0

            # Extract only needed rows
            q_rows = full_table[cold_seq]
            s, zp = self.quant_params[t]
            fp_rows = (q_rows.astype(np.float32) - zp) * s
            result[t] = (torch.from_numpy(cold_orig.astype(np.int64)),
                         torch.from_numpy(fp_rows))
            # full_table goes out of scope and is garbage collected
        return result

    @staticmethod
    def inject(dlrm, result):
        EntropyColdStore.inject(dlrm, result)

    @property
    def memory_bytes(self):
        """Only compressed tables + lookup tables."""
        lookup_bytes = sum(a.nbytes for a in self.cold_seq_lookup.values())
        return self.compressed_bytes + lookup_bytes


# ==============================================================
# EXP 3: Hybrid Memory Tiers
# ==============================================================

class HybridTierColdStore:
    """Hybrid: top-K most frequently accessed cold rows decoded in memory (fp32),
    rest zstd compressed in blocks with LRU cache."""

    def __init__(self, large_tables, top_k_per_table=2000,
                 block_size=ZSTD_BLOCK_SIZE, max_cache_mb=100):
        self.large_tables = large_tables
        self.top_k = top_k_per_table
        self.block_size = block_size

        # Tier 1: top-K cold rows stored decoded as fp32
        self.tier1_weights = {}     # {t: torch.Tensor (K, 16) float32}
        self.tier1_indices = {}     # {t: np.array of original indices in tier1}
        self.tier1_lookup = {}      # {t: np.array (num_emb,) int32, -1 if not in tier1}

        # Tier 2: remaining cold rows compressed in blocks
        self.blocks = {}
        self.tier2_seq_lookup = {}
        self.quant_params = {}
        self.cache = LRUCache(max_cache_mb * 1024 * 1024)

        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.tier1_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.tier1_hits = 0
        self.tier2_hits = 0

    def add_table(self, t, num_emb, cold_indices, cold_weights, cold_access_counts=None):
        """cold_access_counts: array of access counts aligned with cold_indices (most frequent first)."""
        n_cold = len(cold_indices)
        # Determine tier1 rows (top-K by frequency)
        if cold_access_counts is not None and len(cold_access_counts) > 0:
            # cold_access_counts[i] = count for cold_indices[i]
            top_k = min(self.top_k, n_cold)
            freq_order = np.argsort(-cold_access_counts)
            tier1_local_idx = freq_order[:top_k]
            tier2_local_idx = freq_order[top_k:]
        else:
            # No frequency info, just take first top_k
            top_k = min(self.top_k, n_cold)
            tier1_local_idx = np.arange(top_k)
            tier2_local_idx = np.arange(top_k, n_cold)

        # Tier 1: store as fp32
        tier1_orig_indices = np.array([cold_indices[i] for i in tier1_local_idx], dtype=np.int64)
        tier1_w = cold_weights[tier1_local_idx]  # torch tensor
        self.tier1_weights[t] = tier1_w.clone()
        self.tier1_indices[t] = tier1_orig_indices
        tier1_lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(tier1_orig_indices):
            tier1_lookup[orig] = seq
        self.tier1_lookup[t] = tier1_lookup
        self.tier1_bytes += tier1_w.numel() * 4

        # Tier 2: compress remaining cold rows
        tier2_orig_indices = np.array([cold_indices[i] for i in tier2_local_idx], dtype=np.int64)
        tier2_w = cold_weights[tier2_local_idx]
        q, s, zp = quantize_global(tier2_w)
        q_np = q.numpy()
        self.quant_params[t] = (s, zp)

        tier2_lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(tier2_orig_indices):
            tier2_lookup[orig] = seq
        self.tier2_seq_lookup[t] = tier2_lookup

        n_tier2 = q_np.shape[0]
        table_blocks = []
        raw_total = 0
        comp_total = 0
        for start in range(0, n_tier2, self.block_size):
            end = min(start + self.block_size, n_tier2)
            block_data = q_np[start:end]
            raw = block_data.tobytes()
            compressed = zstd_compress(raw, level=3)
            table_blocks.append((compressed, end - start))
            raw_total += len(raw)
            comp_total += len(compressed)
        self.blocks[t] = table_blocks
        self.raw_bytes += raw_total
        self.compressed_bytes += comp_total

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

            all_idx = []
            all_vals = []

            # Check tier 1 (fp32 preloaded)
            tier1_seq = self.tier1_lookup[t][unique]
            tier1_mask = tier1_seq >= 0
            if tier1_mask.any():
                t1_orig = unique[tier1_mask]
                t1_seq = tier1_seq[tier1_mask]
                t1_vals = self.tier1_weights[t][t1_seq]
                all_idx.append(torch.from_numpy(t1_orig.astype(np.int64)))
                all_vals.append(t1_vals)
                self.tier1_hits += len(t1_orig)

            # Check tier 2 (compressed)
            tier2_seq = self.tier2_seq_lookup[t][unique]
            tier2_mask = tier2_seq >= 0
            if tier2_mask.any():
                t2_orig = unique[tier2_mask]
                t2_seq = tier2_seq[tier2_mask]
                block_ids = t2_seq // self.block_size
                unique_blocks = np.unique(block_ids)

                q_rows = np.empty((len(t2_seq), EMB_DIM), dtype=np.uint8)
                for bid in unique_blocks:
                    block_data = self._get_block(t, bid)
                    block_mask = block_ids == bid
                    local_offsets = t2_seq[block_mask] - bid * self.block_size
                    q_rows[block_mask] = block_data[local_offsets]

                s, zp = self.quant_params[t]
                fp_rows = (q_rows.astype(np.float32) - zp) * s
                all_idx.append(torch.from_numpy(t2_orig.astype(np.int64)))
                all_vals.append(torch.from_numpy(fp_rows))
                self.tier2_hits += len(t2_orig)

            if all_idx:
                result[t] = (torch.cat(all_idx), torch.cat(all_vals))
        return result

    @staticmethod
    def inject(dlrm, result):
        EntropyColdStore.inject(dlrm, result)

    @property
    def memory_bytes(self):
        return self.tier1_bytes + self.compressed_bytes + self.cache.current_bytes


# ==============================================================
# EXP 6: Per-Column Quantization Cold Store
# ==============================================================

class PerColumnColdStore:
    """Per-column uint8 quantization: each of the 16 embedding dimensions
    gets its own scale/zero-point for finer quantization."""

    def __init__(self, large_tables):
        self.cold_uint8 = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}  # {t: (scales_16, zps_16)}
        self.large_tables = large_tables

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, scales, zps = quantize_per_column(cold_weights)
        self.cold_uint8[t] = q.numpy()
        self.quant_params[t] = (scales, zps)
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
            scales, zps = self.quant_params[t]
            fp_rows = dequantize_per_column(q_rows, scales, zps)
            result[t] = (torch.from_numpy(cold_orig.astype(np.int64)),
                         torch.from_numpy(fp_rows))
        return result

    @staticmethod
    def inject(dlrm, result):
        EntropyColdStore.inject(dlrm, result)

    @property
    def memory_bytes(self):
        return sum(a.nbytes for a in self.cold_uint8.values())


# ==============================================================
# EXP 7: Mixed Precision Cold Store (fp16 + uint8)
# ==============================================================

class MixedPrecisionColdStore:
    """Top 20% of cold rows by access frequency: fp16.
    Bottom 80%: uint8. Better quality for frequently-accessed cold rows."""

    def __init__(self, large_tables, fp16_fraction=0.20):
        self.large_tables = large_tables
        self.fp16_fraction = fp16_fraction

        # fp16 tier
        self.fp16_data = {}        # {t: np.array (N_fp16, 16) float16}
        self.fp16_lookup = {}      # {t: np.array (num_emb,) int32}

        # uint8 tier
        self.uint8_data = {}       # {t: np.array (N_u8, 16) uint8}
        self.uint8_lookup = {}     # {t: np.array (num_emb,) int32}
        self.quant_params = {}     # {t: (scale, zp)} for uint8 tier

        self.fp16_bytes = 0
        self.uint8_bytes = 0

    def add_table(self, t, num_emb, cold_indices, cold_weights, cold_access_counts=None):
        n_cold = len(cold_indices)
        n_fp16 = max(1, int(n_cold * self.fp16_fraction))

        if cold_access_counts is not None and len(cold_access_counts) > 0:
            freq_order = np.argsort(-cold_access_counts)
        else:
            freq_order = np.arange(n_cold)

        fp16_local = freq_order[:n_fp16]
        uint8_local = freq_order[n_fp16:]

        # fp16 tier
        fp16_orig = np.array([cold_indices[i] for i in fp16_local], dtype=np.int64)
        fp16_w = cold_weights[fp16_local].numpy().astype(np.float16)
        self.fp16_data[t] = fp16_w
        fp16_lk = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(fp16_orig):
            fp16_lk[orig] = seq
        self.fp16_lookup[t] = fp16_lk
        self.fp16_bytes += fp16_w.nbytes

        # uint8 tier
        if len(uint8_local) > 0:
            uint8_orig = np.array([cold_indices[i] for i in uint8_local], dtype=np.int64)
            uint8_w = cold_weights[uint8_local]
            q, s, zp = quantize_global(uint8_w)
            self.uint8_data[t] = q.numpy()
            self.quant_params[t] = (s, zp)
            uint8_lk = np.full(num_emb, -1, dtype=np.int32)
            for seq, orig in enumerate(uint8_orig):
                uint8_lk[orig] = seq
            self.uint8_lookup[t] = uint8_lk
            self.uint8_bytes += q.numpy().nbytes
        else:
            self.uint8_data[t] = np.empty((0, EMB_DIM), dtype=np.uint8)
            self.uint8_lookup[t] = np.full(num_emb, -1, dtype=np.int32)
            self.quant_params[t] = (1.0, 0)

    def dequantize_for_batch(self, lS_i):
        result = {}
        for t in self.large_tables:
            if t not in self.fp16_data:
                continue
            indices = lS_i[t].numpy().flatten()
            unique = np.unique(indices)

            all_idx = []
            all_vals = []

            # fp16 tier
            fp16_seq = self.fp16_lookup[t][unique]
            fp16_mask = fp16_seq >= 0
            if fp16_mask.any():
                f_orig = unique[fp16_mask]
                f_seq = fp16_seq[fp16_mask]
                f_rows = self.fp16_data[t][f_seq].astype(np.float32)
                all_idx.append(torch.from_numpy(f_orig.astype(np.int64)))
                all_vals.append(torch.from_numpy(f_rows))

            # uint8 tier
            uint8_seq = self.uint8_lookup[t][unique]
            uint8_mask = uint8_seq >= 0
            if uint8_mask.any():
                u_orig = unique[uint8_mask]
                u_seq = uint8_seq[uint8_mask]
                q_rows = self.uint8_data[t][u_seq]
                s, zp = self.quant_params[t]
                fp_rows = (q_rows.astype(np.float32) - zp) * s
                all_idx.append(torch.from_numpy(u_orig.astype(np.int64)))
                all_vals.append(torch.from_numpy(fp_rows))

            if all_idx:
                result[t] = (torch.cat(all_idx), torch.cat(all_vals))
        return result

    @staticmethod
    def inject(dlrm, result):
        EntropyColdStore.inject(dlrm, result)

    @property
    def memory_bytes(self):
        return self.fp16_bytes + self.uint8_bytes


# ==============================================================
# EXP 9: Combined Best Store
# ==============================================================

class CombinedBestColdStore:
    """Combined best configuration: per-column quantization + lazy materialization
    + zstd compression + configurable hot threshold.

    Lazy: skip re-injection for already-materialized cold rows.
    Per-column: finer quantization for better AUC.
    zstd blocks: compressed storage for lower memory."""

    def __init__(self, large_tables, block_size=ZSTD_BLOCK_SIZE, max_cache_mb=200):
        self.large_tables = large_tables
        self.block_size = block_size

        self.blocks = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}  # {t: (scales_16, zps_16)}
        self.materialized = {}

        self.cache = LRUCache(max_cache_mb * 1024 * 1024)
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.total_injections = 0
        self.total_skipped = 0
        self.injection_counts = []

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, scales, zps = quantize_per_column(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (scales, zps)

        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup
        self.materialized[t] = np.zeros(num_emb, dtype=bool)

        n_rows = q_np.shape[0]
        table_blocks = []
        raw_total = 0
        comp_total = 0
        for start in range(0, n_rows, self.block_size):
            end = min(start + self.block_size, n_rows)
            block_data = q_np[start:end]
            raw = block_data.tobytes()
            compressed = zstd_compress(raw, level=3)
            table_blocks.append((compressed, end - start))
            raw_total += len(raw)
            comp_total += len(compressed)
        self.blocks[t] = table_blocks
        self.raw_bytes += raw_total
        self.compressed_bytes += comp_total

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
        batch_injections = 0
        batch_skipped = 0
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

            # Lazy: skip already materialized
            not_mat = ~self.materialized[t][cold_orig]
            batch_skipped += int(np.sum(~not_mat))
            if not not_mat.any():
                continue
            new_cold = cold_orig[not_mat]
            new_seq = self.cold_seq_lookup[t][new_cold]

            block_ids = new_seq // self.block_size
            unique_blocks = np.unique(block_ids)

            q_rows = np.empty((len(new_seq), EMB_DIM), dtype=np.uint8)
            for bid in unique_blocks:
                block_data = self._get_block(t, bid)
                block_mask = block_ids == bid
                local_offsets = new_seq[block_mask] - bid * self.block_size
                q_rows[block_mask] = block_data[local_offsets]

            scales, zps = self.quant_params[t]
            fp_rows = dequantize_per_column(q_rows, scales, zps)
            result[t] = (torch.from_numpy(new_cold.astype(np.int64)),
                         torch.from_numpy(fp_rows))
            self.materialized[t][new_cold] = True
            batch_injections += len(new_cold)

        self.total_injections += batch_injections
        self.total_skipped += batch_skipped
        self.injection_counts.append(batch_injections)
        return result

    @staticmethod
    def inject(dlrm, result):
        EntropyColdStore.inject(dlrm, result)

    @property
    def memory_bytes(self):
        return self.compressed_bytes + self.cache.current_bytes


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
# SETUP FUNCTIONS
# ==============================================================

def setup_small_tables_quantized(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                                  quantize_fn='global'):
    """Quantize small tables (round-trip) and set in model."""
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


def setup_hot_rows(dlrm, state_dict, emb_keys, large_tables, hot_indices,
                    quantize_fn='global'):
    """Set hot rows in model (quantize round-trip)."""
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


def setup_cold_store_generic(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                              large_tables, hot_indices, store,
                              access_counts=None, quantize_fn='global'):
    """Generic setup for any cold store."""
    setup_small_tables_quantized(dlrm, state_dict, emb_keys, ln_emb, num_tables, quantize_fn)
    setup_hot_rows(dlrm, state_dict, emb_keys, large_tables, hot_indices, quantize_fn)

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())
        hot_idx = sorted(hi)
        cold_idx = sorted(set(range(ln_emb[t])) - hi)
        cw = w[cold_idx]

        # Compute cold access counts if available
        cold_acc = None
        if access_counts is not None and t in access_counts:
            all_unique, all_counts = access_counts[t]
            if len(all_unique) > 0:
                cold_set = set(cold_idx)
                cold_acc_dict = {}
                for u, c in zip(all_unique, all_counts):
                    if u in cold_set:
                        cold_acc_dict[u] = c
                cold_acc = np.array([cold_acc_dict.get(ci, 0) for ci in cold_idx], dtype=np.int64)

        # Call add_table with or without access counts
        if cold_acc is not None and hasattr(store.add_table, '__code__') and \
           'cold_access_counts' in store.add_table.__code__.co_varnames:
            store.add_table(t, ln_emb[t], cold_idx, cw, cold_access_counts=cold_acc)
        else:
            store.add_table(t, ln_emb[t], cold_idx, cw)

        log(f"    Table {t}: {len(hot_idx):,} hot, {len(cold_idx):,} cold")
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
            scores.extend(S.tolist())
            targets.extend(Tn.tolist())
            nb += 1
            if nb % 500 == 0:
                log(f"    Batch {nb}, lat={blats[-1] * 1000:.1f}ms")
    total = time.time() - t0
    return accu / samp, roc_auc_score(targets, scores), total, blats, nb


def run_prefetch_inference(dlrm, test_ld, store):
    """Generic prefetch inference loop."""
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
# PARALLEL ENCODING HELPER (Exp 4)
# ==============================================================

def _encode_table_worker(args):
    """Worker function for parallel table encoding.
    Runs in a separate process to encode a single table."""
    t, cold_indices, cold_weights_np, num_emb = args
    # Quantize
    w_t = torch.from_numpy(cold_weights_np)
    mn, mx = w_t.min().item(), w_t.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((w_t / s).round() + zp).clamp(0, 255).to(torch.uint8)
    q_np = q.numpy()

    lookup = np.full(num_emb, -1, dtype=np.int32)
    for seq, orig in enumerate(cold_indices):
        lookup[orig] = seq

    return (t, q_np, s, zp, lookup)


# ==============================================================
# EXPERIMENT RUNNER
# ==============================================================

def run_experiment(exp_name, exp_num, dlrm, test_ld, state_dict, emb_keys,
                   ln_emb, num_tables, large_tables, hot_indices, baseline_auc,
                   store, access_counts=None, quantize_fn='global',
                   extra_info=None):
    """Run a single experiment: setup + inference + collect metrics."""
    log(f"\n{'=' * 60}")
    log(f"EXPERIMENT {exp_num}: {exp_name}")
    log(f"{'=' * 60}")

    restore_weights(dlrm, state_dict, emb_keys)
    gc.collect()

    setup_t0 = time.time()
    setup_cold_store_generic(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                              large_tables, hot_indices, store,
                              access_counts=access_counts,
                              quantize_fn=quantize_fn)
    setup_time = time.time() - setup_t0

    mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                    state_dict, emb_keys, num_tables, ln_emb)
    log(f"  Setup: {setup_time:.2f}s")
    log(f"  Cold storage: {mem['cold_store_mb']:.1f}MB, Total: {mem['total_mb']:.1f}MB")

    drop_caches()
    time.sleep(1)
    gc.collect()

    acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
    auc_loss = (baseline_auc - auc) * 100

    log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}")
    log(f"  Inference={inf_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time + inf_time:.2f}s")
    log(f"  Memory: hot={mem['hot_fp32_mb']:.1f}MB, small={mem['small_fp32_mb']:.1f}MB, "
        f"mlp={mem['mlp_mb']:.1f}MB, cold={mem['cold_store_mb']:.1f}MB, total={mem['total_mb']:.1f}MB")
    log(f"  AUC loss vs baseline: {auc_loss:.4f}pp")
    log(f"  Batch: mean={np.mean(blats) * 1000:.1f}ms, "
        f"p50={np.percentile(blats, 50) * 1000:.1f}ms, "
        f"p95={np.percentile(blats, 95) * 1000:.1f}ms, "
        f"p99={np.percentile(blats, 99) * 1000:.1f}ms")

    result = {
        'name': exp_name,
        'experiment_number': exp_num,
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

    # Extra metrics for specific stores
    if hasattr(store, 'total_injections'):
        log(f"  Lazy: {store.total_injections:,} injections, {store.total_skipped:,} skipped")
        result['lazy_total_injections'] = store.total_injections
        result['lazy_total_skipped'] = store.total_skipped
        result['lazy_injection_counts_first50'] = store.injection_counts[:50]
        result['lazy_injection_counts_last20'] = store.injection_counts[-20:] if len(store.injection_counts) > 20 else store.injection_counts

    if hasattr(store, 'compression_ratio') and callable(getattr(store, 'compression_ratio', None)):
        pass
    elif hasattr(store, 'compressed_bytes') and hasattr(store, 'raw_bytes') and store.compressed_bytes > 0:
        ratio = store.raw_bytes / store.compressed_bytes
        log(f"  Compression: ratio={ratio:.2f}x, "
            f"raw={store.raw_bytes / 1024 / 1024:.1f}MB, "
            f"compressed={store.compressed_bytes / 1024 / 1024:.1f}MB")
        result['compression_ratio'] = float(ratio)
        result['raw_bytes'] = store.raw_bytes
        result['compressed_bytes'] = store.compressed_bytes

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
        result['cache_current_mb'] = float(store.cache.current_bytes / 1024 / 1024)

    if hasattr(store, 'tier1_hits'):
        log(f"  Tiers: tier1_hits={store.tier1_hits}, tier2_hits={store.tier2_hits}")
        result['tier1_hits'] = store.tier1_hits
        result['tier2_hits'] = store.tier2_hits
        result['tier1_bytes'] = store.tier1_bytes

    if hasattr(store, 'fp16_bytes'):
        log(f"  Mixed precision: fp16={store.fp16_bytes / 1024 / 1024:.1f}MB, "
            f"uint8={store.uint8_bytes / 1024 / 1024:.1f}MB")
        result['fp16_mb'] = float(store.fp16_bytes / 1024 / 1024)
        result['uint8_mb'] = float(store.uint8_bytes / 1024 / 1024)

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
    log("PREFETCH BENCHMARK V10 OPTIMIZE: Extended Optimization Exploration")
    log("=" * 70)
    log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"JSON output: {JSON_PATH}")
    log(f"Log output: {LOG_PATH}")
    log(f"zstd available: {HAS_ZSTD}")
    log(f"lz4 available: {HAS_LZ4}")

    all_results = {}
    start_time = time.time()

    # ----------------------------------------------------------
    # LOAD MODEL AND DATA
    # ----------------------------------------------------------
    log("\nLoading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")
    log(f"Table sizes: {[int(ln_emb[t]) for t in range(num_tables)]}")

    # ----------------------------------------------------------
    # PROFILE ACCESS PATTERNS (once, reuse across experiments)
    # ----------------------------------------------------------
    log("\nProfiling access patterns (default threshold 80%)...")
    hot_80, access_counts_80 = profile_access(train_ld, num_tables, ln_emb, 0.80)
    for t in large_tables:
        log(f"  Table {t}: {ln_emb[t]:,} total, {len(hot_80[t]):,} hot@80%")

    # ----------------------------------------------------------
    # BASELINE
    # ----------------------------------------------------------
    log("\n" + "=" * 60)
    log("BASELINE: streaming, no compression")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    drop_caches()
    time.sleep(1)
    gc.collect()

    b_acc, b_auc, b_time, b_blats, b_nb = run_baseline_inference(dlrm, test_ld)
    b_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
    log(f"  Acc={b_acc * 100:.4f}%, AUC={b_auc:.6f}")
    log(f"  Inference={b_time:.2f}s, Memory={b_mem:.1f}MB")
    log(f"  Batch: mean={np.mean(b_blats) * 1000:.1f}ms, "
        f"p50={np.percentile(b_blats, 50) * 1000:.1f}ms, "
        f"p99={np.percentile(b_blats, 99) * 1000:.1f}ms")

    baseline_auc = b_auc
    all_results['0_baseline'] = {
        'name': 'Baseline (streaming, no compression)',
        'experiment_number': 0,
        'accuracy': float(b_acc),
        'auc': float(b_auc),
        'auc_loss_pp': 0.0,
        'inference_time': float(b_time),
        'setup_time': 0.0,
        'total_time': float(b_time),
        'memory_mb': float(b_mem),
        'batch_latency': latency_stats(b_blats),
        'num_batches': b_nb,
    }
    save_results_incremental(all_results)

    # ----------------------------------------------------------
    # V9 REFERENCE
    # ----------------------------------------------------------
    log("\n" + "=" * 60)
    log("V9 REFERENCE: uint8, 80% hot, per-batch prefetch")
    log("=" * 60)
    store_v9 = EntropyColdStore(large_tables)
    result_v9 = run_experiment(
        "V9 reference (uint8, 80% hot, prefetch)", "0_v9ref",
        dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
        large_tables, hot_80, baseline_auc, store_v9)
    all_results['0_v9_reference'] = result_v9
    save_results_incremental(all_results)

    # ==============================================================
    # PHASE 1: REDUCE RUNTIME MEMORY
    # ==============================================================
    log("\n" + "#" * 70)
    log("PHASE 1: REDUCE RUNTIME MEMORY")
    log("#" * 70)

    # ----------------------------------------------------------
    # EXP 1: Partial Pre-Decode with LRU Cache
    # ----------------------------------------------------------
    for cache_mb in [100, 200, 300]:
        exp_key = f"1_lru_cache_{cache_mb}mb"
        try:
            log(f"\n--- Exp 1: LRU Block Cache (budget={cache_mb}MB) ---")
            store = LRUBlockColdStore(large_tables, block_size=ZSTD_BLOCK_SIZE,
                                       max_cache_mb=cache_mb)
            result = run_experiment(
                f"LRU block cache (zstd, 4K blocks, {cache_mb}MB budget)",
                exp_key, dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
                large_tables, hot_80, baseline_auc, store,
                extra_info={'cache_budget_mb': cache_mb, 'block_size': ZSTD_BLOCK_SIZE})
            all_results[exp_key] = result
            save_results_incremental(all_results)
        except Exception as e:
            log(f"  ERROR in Exp 1 (cache={cache_mb}MB): {e}")
            log(traceback.format_exc())
            all_results[exp_key] = {'name': exp_key, 'error': str(e)}
            save_results_incremental(all_results)

    # ----------------------------------------------------------
    # EXP 2: On-Demand Per-Table Decode
    # ----------------------------------------------------------
    exp_key = "2_on_demand"
    try:
        log(f"\n--- Exp 2: On-Demand Per-Table Decode (minimal memory) ---")
        store = OnDemandColdStore(large_tables)
        result = run_experiment(
            "On-demand per-table decode (zstd, minimal memory)",
            exp_key, dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
            large_tables, hot_80, baseline_auc, store)
        all_results[exp_key] = result
        save_results_incremental(all_results)
    except Exception as e:
        log(f"  ERROR in Exp 2: {e}")
        log(traceback.format_exc())
        all_results[exp_key] = {'name': exp_key, 'error': str(e)}
        save_results_incremental(all_results)

    # ----------------------------------------------------------
    # EXP 3: Hybrid Memory Tiers
    # ----------------------------------------------------------
    for top_k in [2000]:
        for tier_cache_mb in [100]:
            exp_key = f"3_hybrid_tier_top{top_k}_cache{tier_cache_mb}mb"
            try:
                log(f"\n--- Exp 3: Hybrid Tiers (top-{top_k} fp32, rest zstd, cache={tier_cache_mb}MB) ---")
                store = HybridTierColdStore(large_tables, top_k_per_table=top_k,
                                             block_size=ZSTD_BLOCK_SIZE,
                                             max_cache_mb=tier_cache_mb)
                result = run_experiment(
                    f"Hybrid tiers (top-{top_k} fp32 + zstd blocks, cache={tier_cache_mb}MB)",
                    exp_key, dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_80, baseline_auc, store,
                    access_counts=access_counts_80,
                    extra_info={'top_k': top_k, 'tier_cache_mb': tier_cache_mb})
                all_results[exp_key] = result
                save_results_incremental(all_results)
            except Exception as e:
                log(f"  ERROR in Exp 3: {e}")
                log(traceback.format_exc())
                all_results[exp_key] = {'name': exp_key, 'error': str(e)}
                save_results_incremental(all_results)

    # ==============================================================
    # PHASE 2: FASTER SETUP
    # ==============================================================
    log("\n" + "#" * 70)
    log("PHASE 2: FASTER SETUP")
    log("#" * 70)

    # ----------------------------------------------------------
    # EXP 4: Parallel Table Encoding
    # ----------------------------------------------------------
    exp_key = "4_parallel_encode"
    try:
        log(f"\n--- Exp 4: Parallel Table Encoding (multiprocessing) ---")
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()

        setup_t0 = time.time()

        # Prepare small tables (sequential, fast)
        setup_small_tables_quantized(dlrm, state_dict, emb_keys, ln_emb, num_tables)

        # Prepare hot rows (sequential)
        setup_hot_rows(dlrm, state_dict, emb_keys, large_tables, hot_80)

        # Prepare cold table encoding tasks
        encode_tasks = []
        for t in large_tables:
            w = state_dict[emb_keys[t]]
            hi = set(hot_80[t].tolist())
            cold_idx = sorted(set(range(ln_emb[t])) - hi)
            cw = w[cold_idx].numpy()
            encode_tasks.append((t, cold_idx, cw, int(ln_emb[t])))

        # Parallel encoding using threads (ProcessPoolExecutor has pickle overhead
        # for large arrays, ThreadPoolExecutor avoids that)
        parallel_t0 = time.time()
        store = EntropyColdStore(large_tables)
        with ThreadPoolExecutor(max_workers=min(4, len(large_tables))) as pool:
            futures = []
            for task in encode_tasks:
                t_id, cold_idx, cw_np, n_emb = task
                def encode_one(t_id=t_id, cold_idx=cold_idx, cw_np=cw_np, n_emb=n_emb):
                    w_t = torch.from_numpy(cw_np)
                    q, s, zp = quantize_global(w_t)
                    q_np = q.numpy()
                    lookup = np.full(n_emb, -1, dtype=np.int32)
                    for seq, orig in enumerate(cold_idx):
                        lookup[orig] = seq
                    return (t_id, q_np, s, zp, lookup)
                futures.append(pool.submit(encode_one))

            for fut in futures:
                t_id, q_np, s, zp, lookup = fut.result()
                store.cold_uint8[t_id] = q_np
                store.quant_params[t_id] = (s, zp)
                store.cold_seq_lookup[t_id] = lookup
                log(f"    Table {t_id}: encoded {q_np.shape[0]:,} cold rows")

        parallel_time = time.time() - parallel_t0
        setup_time = time.time() - setup_t0
        log(f"  Parallel encode time: {parallel_time:.2f}s")
        log(f"  Total setup time: {setup_time:.2f}s")

        # Now run inference
        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_80, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        drop_caches()
        time.sleep(1)
        gc.collect()

        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        auc_loss = (baseline_auc - auc) * 100

        log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}")
        log(f"  Inference={inf_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time + inf_time:.2f}s")
        log(f"  Memory: {mem['total_mb']:.1f}MB")
        log(f"  AUC loss: {auc_loss:.4f}pp")
        log(f"  Batch: mean={np.mean(blats) * 1000:.1f}ms, "
            f"p50={np.percentile(blats, 50) * 1000:.1f}ms, "
            f"p99={np.percentile(blats, 99) * 1000:.1f}ms")

        all_results[exp_key] = {
            'name': 'Parallel table encoding (ThreadPool)',
            'experiment_number': exp_key,
            'accuracy': float(acc), 'auc': float(auc),
            'auc_loss_pp': float(auc_loss),
            'inference_time': float(inf_time),
            'setup_time': float(setup_time),
            'parallel_encode_time': float(parallel_time),
            'total_time': float(setup_time + inf_time),
            'memory_breakdown': mem, 'memory_mb': float(mem['total_mb']),
            'batch_latency': latency_stats(blats),
            'num_batches': nb,
        }
        save_results_incremental(all_results)
    except Exception as e:
        log(f"  ERROR in Exp 4: {e}")
        log(traceback.format_exc())
        all_results[exp_key] = {'name': exp_key, 'error': str(e)}
        save_results_incremental(all_results)

    # ----------------------------------------------------------
    # EXP 5: Skip Encode, Load Pre-Compressed from Disk
    # ----------------------------------------------------------
    exp_key = "5_precompressed_disk"
    try:
        log(f"\n--- Exp 5: Pre-Compressed Disk Cache ---")

        cache_file = os.path.join(COMPRESSED_CACHE_DIR, "cold_uint8_80pct.pkl")

        # Step 1: Save compressed tables to disk (one-time cost)
        if not os.path.exists(cache_file):
            log("  Creating pre-compressed cache file (one-time)...")
            save_t0 = time.time()
            cache_data = {}
            for t in large_tables:
                w = state_dict[emb_keys[t]]
                hi = set(hot_80[t].tolist())
                cold_idx = sorted(set(range(ln_emb[t])) - hi)
                cw = w[cold_idx]
                q, s, zp = quantize_global(cw)
                q_np = q.numpy()
                lookup = np.full(int(ln_emb[t]), -1, dtype=np.int32)
                for seq, orig in enumerate(cold_idx):
                    lookup[orig] = seq
                cache_data[t] = {
                    'q_np': q_np, 'scale': s, 'zp': zp,
                    'lookup': lookup, 'cold_idx': cold_idx,
                }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            save_time = time.time() - save_t0
            file_size_mb = os.path.getsize(cache_file) / 1024 / 1024
            log(f"  Saved cache: {cache_file} ({file_size_mb:.1f}MB) in {save_time:.2f}s")
        else:
            log(f"  Cache file already exists: {cache_file}")

        # Step 2: Load from disk (simulating subsequent runs)
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()

        load_t0 = time.time()
        setup_small_tables_quantized(dlrm, state_dict, emb_keys, ln_emb, num_tables)
        setup_hot_rows(dlrm, state_dict, emb_keys, large_tables, hot_80)

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        store = EntropyColdStore(large_tables)
        for t in large_tables:
            cd = cache_data[t]
            store.cold_uint8[t] = cd['q_np']
            store.quant_params[t] = (cd['scale'], cd['zp'])
            store.cold_seq_lookup[t] = cd['lookup']
            log(f"    Table {t}: loaded {cd['q_np'].shape[0]:,} cold rows from disk")

        load_time = time.time() - load_t0
        log(f"  Load-from-disk setup time: {load_time:.2f}s")

        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_80, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        drop_caches()
        time.sleep(1)
        gc.collect()

        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        auc_loss = (baseline_auc - auc) * 100

        log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}")
        log(f"  Inference={inf_time:.2f}s, Load setup={load_time:.2f}s, Total={load_time + inf_time:.2f}s")
        log(f"  Memory: {mem['total_mb']:.1f}MB")
        log(f"  AUC loss: {auc_loss:.4f}pp")

        cache_size_mb = os.path.getsize(cache_file) / 1024 / 1024

        all_results[exp_key] = {
            'name': 'Pre-compressed disk cache (load from pkl)',
            'experiment_number': exp_key,
            'accuracy': float(acc), 'auc': float(auc),
            'auc_loss_pp': float(auc_loss),
            'inference_time': float(inf_time),
            'setup_time': float(load_time),
            'total_time': float(load_time + inf_time),
            'memory_breakdown': mem, 'memory_mb': float(mem['total_mb']),
            'batch_latency': latency_stats(blats),
            'num_batches': nb,
            'cache_file_size_mb': float(cache_size_mb),
        }
        save_results_incremental(all_results)
        del cache_data
    except Exception as e:
        log(f"  ERROR in Exp 5: {e}")
        log(traceback.format_exc())
        all_results[exp_key] = {'name': exp_key, 'error': str(e)}
        save_results_incremental(all_results)

    # ==============================================================
    # PHASE 3: BETTER COMPRESSION + QUALITY
    # ==============================================================
    log("\n" + "#" * 70)
    log("PHASE 3: BETTER COMPRESSION + QUALITY")
    log("#" * 70)

    # ----------------------------------------------------------
    # EXP 6: Per-Column Quantization
    # ----------------------------------------------------------
    exp_key = "6_per_column_quant"
    try:
        log(f"\n--- Exp 6: Per-Column Quantization ---")
        store = PerColumnColdStore(large_tables)
        result = run_experiment(
            "Per-column uint8 quantization (16 scale/zp pairs)",
            exp_key, dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
            large_tables, hot_80, baseline_auc, store,
            quantize_fn='per_column')
        all_results[exp_key] = result
        save_results_incremental(all_results)
    except Exception as e:
        log(f"  ERROR in Exp 6: {e}")
        log(traceback.format_exc())
        all_results[exp_key] = {'name': exp_key, 'error': str(e)}
        save_results_incremental(all_results)

    # ----------------------------------------------------------
    # EXP 7: Mixed Precision Cold Storage
    # ----------------------------------------------------------
    exp_key = "7_mixed_precision"
    try:
        log(f"\n--- Exp 7: Mixed Precision Cold Storage (fp16 top-20% + uint8 bottom-80%) ---")
        store = MixedPrecisionColdStore(large_tables, fp16_fraction=0.20)
        result = run_experiment(
            "Mixed precision (fp16 top-20% + uint8 bottom-80%)",
            exp_key, dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
            large_tables, hot_80, baseline_auc, store,
            access_counts=access_counts_80)
        all_results[exp_key] = result
        save_results_incremental(all_results)
    except Exception as e:
        log(f"  ERROR in Exp 7: {e}")
        log(traceback.format_exc())
        all_results[exp_key] = {'name': exp_key, 'error': str(e)}
        save_results_incremental(all_results)

    # ----------------------------------------------------------
    # EXP 8: Hot Threshold Sweep
    # ----------------------------------------------------------
    log(f"\n--- Exp 8: Hot Threshold Sweep (70%-99%, step 5%) ---")
    sweep_results = []
    for pct in range(70, 100, 5):
        threshold = pct / 100.0
        exp_key = f"8_hot_sweep_{pct}pct"
        try:
            log(f"\n  --- Hot threshold = {pct}% ---")
            hot_sweep = profile_access_simple(train_ld, num_tables, ln_emb, threshold)
            for t in large_tables:
                log(f"    Table {t}: {len(hot_sweep[t]):,} hot@{pct}%")

            store = EntropyColdStore(large_tables)
            result = run_experiment(
                f"Hot threshold sweep ({pct}%)",
                exp_key, dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
                large_tables, hot_sweep, baseline_auc, store,
                extra_info={'hot_threshold': threshold, 'hot_pct': pct})

            # Add hot/cold counts per table
            for t_idx in large_tables:
                result[f'table_{t_idx}_hot_count'] = len(hot_sweep[t_idx])
                result[f'table_{t_idx}_cold_count'] = int(ln_emb[t_idx]) - len(hot_sweep[t_idx])

            all_results[exp_key] = result
            sweep_results.append(result)
            save_results_incremental(all_results)
        except Exception as e:
            log(f"  ERROR in Exp 8 (threshold={pct}%): {e}")
            log(traceback.format_exc())
            all_results[exp_key] = {'name': exp_key, 'error': str(e)}
            save_results_incremental(all_results)

    # Print sweep summary
    if sweep_results:
        log("\n  Hot Threshold Sweep Summary:")
        log(f"  {'Threshold':>10s}  {'AUC':>10s}  {'AUC loss':>10s}  {'Infer(s)':>10s}  "
            f"{'Setup(s)':>10s}  {'Memory(MB)':>10s}")
        log("  " + "-" * 65)
        for r in sweep_results:
            pct = r.get('extra_info', {}).get('hot_pct', '?')
            log(f"  {str(pct) + '%':>10s}  {r['auc']:.6f}  {r['auc_loss_pp']:.4f}pp  "
                f"{r['inference_time']:10.2f}  {r['setup_time']:10.2f}  {r['memory_mb']:10.1f}")

    # ==============================================================
    # PHASE 4: COMBINED BEST CONFIGURATION
    # ==============================================================
    log("\n" + "#" * 70)
    log("PHASE 4: COMBINED BEST CONFIGURATION")
    log("#" * 70)

    # ----------------------------------------------------------
    # EXP 9: Best of Everything
    # ----------------------------------------------------------
    # Determine optimal hot threshold from sweep
    best_sweep = None
    best_score = -1.0
    for r in sweep_results:
        if 'error' in r:
            continue
        # Score: prioritize low AUC loss, then low memory, then low inference time
        auc_loss = r.get('auc_loss_pp', 999)
        mem = r.get('memory_mb', 9999)
        infer = r.get('inference_time', 9999)
        # Only consider configs with <0.01pp AUC loss and <1.5x baseline inference
        if auc_loss <= 0.01 and infer <= b_time * 1.5:
            # Minimize memory
            score = -mem  # higher is better (less memory)
            if score > best_score:
                best_score = score
                best_sweep = r

    if best_sweep is not None:
        optimal_threshold = best_sweep.get('extra_info', {}).get('hot_threshold', 0.80)
        optimal_pct = best_sweep.get('extra_info', {}).get('hot_pct', 80)
    else:
        # Fallback: use 90% if no sweep met the criteria
        log("  No sweep config met both AUC and speed criteria. Checking relaxed criteria...")
        # Relaxed: just pick lowest memory with <0.05pp loss
        for r in sweep_results:
            if 'error' in r:
                continue
            auc_loss = r.get('auc_loss_pp', 999)
            mem = r.get('memory_mb', 9999)
            if auc_loss <= 0.05:
                score = -mem
                if score > best_score:
                    best_score = score
                    best_sweep = r
        if best_sweep is not None:
            optimal_threshold = best_sweep.get('extra_info', {}).get('hot_threshold', 0.90)
            optimal_pct = best_sweep.get('extra_info', {}).get('hot_pct', 90)
        else:
            optimal_threshold = 0.90
            optimal_pct = 90
            log("  Using default 90% threshold for combined experiment.")

    log(f"\n  Optimal hot threshold from sweep: {optimal_pct}%")

    exp_key = "9_combined_best"
    try:
        log(f"\n--- Exp 9: Combined Best (per-column quant + lazy + zstd + hot {optimal_pct}%) ---")

        # Profile at optimal threshold
        hot_optimal = profile_access_simple(train_ld, num_tables, ln_emb, optimal_threshold)
        for t in large_tables:
            log(f"  Table {t}: {len(hot_optimal[t]):,} hot@{optimal_pct}%")

        store = CombinedBestColdStore(large_tables, block_size=ZSTD_BLOCK_SIZE,
                                       max_cache_mb=200)
        result = run_experiment(
            f"Combined best (per-col quant + lazy + zstd + hot {optimal_pct}%)",
            exp_key, dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
            large_tables, hot_optimal, baseline_auc, store,
            quantize_fn='per_column',
            extra_info={
                'hot_threshold': optimal_threshold,
                'hot_pct': optimal_pct,
                'block_size': ZSTD_BLOCK_SIZE,
                'cache_budget_mb': 200,
                'quantization': 'per_column',
                'lazy_materialization': True,
                'compression': 'zstd',
            })
        all_results[exp_key] = result
        save_results_incremental(all_results)
    except Exception as e:
        log(f"  ERROR in Exp 9: {e}")
        log(traceback.format_exc())
        all_results[exp_key] = {'name': exp_key, 'error': str(e)}
        save_results_incremental(all_results)

    # Also try a more aggressive combined: minimal cache for absolute lowest memory
    exp_key = "9b_combined_minimal_mem"
    try:
        log(f"\n--- Exp 9b: Combined Minimal Memory (per-col quant + lazy + zstd + hot {optimal_pct}% + 50MB cache) ---")

        store = CombinedBestColdStore(large_tables, block_size=ZSTD_BLOCK_SIZE,
                                       max_cache_mb=50)
        result = run_experiment(
            f"Combined minimal memory (per-col quant + lazy + zstd + hot {optimal_pct}% + 50MB cache)",
            exp_key, dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
            large_tables, hot_optimal, baseline_auc, store,
            quantize_fn='per_column',
            extra_info={
                'hot_threshold': optimal_threshold,
                'hot_pct': optimal_pct,
                'block_size': ZSTD_BLOCK_SIZE,
                'cache_budget_mb': 50,
                'quantization': 'per_column',
                'lazy_materialization': True,
                'compression': 'zstd',
            })
        all_results[exp_key] = result
        save_results_incremental(all_results)
    except Exception as e:
        log(f"  ERROR in Exp 9b: {e}")
        log(traceback.format_exc())
        all_results[exp_key] = {'name': exp_key, 'error': str(e)}
        save_results_incremental(all_results)

    # Try combined with on-demand decode for absolute minimum memory
    exp_key = "9c_combined_on_demand"
    try:
        log(f"\n--- Exp 9c: Combined On-Demand (per-col quant + on-demand decode + hot {optimal_pct}%) ---")

        # Create an on-demand store with per-column quantization
        # We adapt OnDemandColdStore but with per-column quant
        class OnDemandPerColStore:
            """On-demand decode with per-column quantization."""
            def __init__(self, large_tables_list):
                self.compressed_tables = {}
                self.cold_seq_lookup = {}
                self.quant_params = {}
                self.table_shapes = {}
                self.large_tables = large_tables_list
                self.compressed_bytes = 0
                self.raw_bytes = 0
                self.decompress_count = 0
                self.decompress_time = 0.0

            def add_table(self, t, num_emb, cold_indices, cold_weights):
                q, scales, zps = quantize_per_column(cold_weights)
                q_np = q.numpy()
                self.quant_params[t] = (scales, zps)
                self.table_shapes[t] = q_np.shape
                lookup = np.full(num_emb, -1, dtype=np.int32)
                for seq, orig in enumerate(cold_indices):
                    lookup[orig] = seq
                self.cold_seq_lookup[t] = lookup
                raw = q_np.tobytes()
                compressed = zstd_compress(raw, level=3)
                self.compressed_tables[t] = compressed
                self.raw_bytes += len(raw)
                self.compressed_bytes += len(compressed)

            def dequantize_for_batch(self, lS_i):
                result = {}
                for t in self.large_tables:
                    if t not in self.compressed_tables:
                        continue
                    indices = lS_i[t].numpy().flatten()
                    unique = np.unique(indices)
                    seq = self.cold_seq_lookup[t][unique]
                    mask = seq >= 0
                    if not mask.any():
                        continue
                    cold_orig = unique[mask]
                    cold_seq = seq[mask]
                    t0 = time.time()
                    raw = zstd_decompress(self.compressed_tables[t])
                    shape = self.table_shapes[t]
                    full_table = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
                    self.decompress_count += 1
                    self.decompress_time += time.time() - t0
                    q_rows = full_table[cold_seq]
                    scales, zps = self.quant_params[t]
                    fp_rows = dequantize_per_column(q_rows, scales, zps)
                    result[t] = (torch.from_numpy(cold_orig.astype(np.int64)),
                                 torch.from_numpy(fp_rows))
                return result

            @staticmethod
            def inject(dlrm_model, result):
                EntropyColdStore.inject(dlrm_model, result)

            @property
            def memory_bytes(self):
                lookup_bytes = sum(a.nbytes for a in self.cold_seq_lookup.values())
                return self.compressed_bytes + lookup_bytes

        store = OnDemandPerColStore(large_tables)
        result = run_experiment(
            f"Combined on-demand (per-col quant + zstd + on-demand + hot {optimal_pct}%)",
            exp_key, dlrm, test_ld, state_dict, emb_keys, ln_emb, num_tables,
            large_tables, hot_optimal, baseline_auc, store,
            quantize_fn='per_column',
            extra_info={
                'hot_threshold': optimal_threshold,
                'hot_pct': optimal_pct,
                'quantization': 'per_column',
                'decode_strategy': 'on_demand',
                'compression': 'zstd',
            })
        all_results[exp_key] = result
        save_results_incremental(all_results)
    except Exception as e:
        log(f"  ERROR in Exp 9c: {e}")
        log(traceback.format_exc())
        all_results[exp_key] = {'name': exp_key, 'error': str(e)}
        save_results_incremental(all_results)

    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================
    elapsed = time.time() - start_time
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"Total elapsed time: {elapsed / 60:.1f} minutes ({elapsed / 3600:.2f} hours)")

    # Sort results by experiment key
    sorted_keys = sorted(all_results.keys())

    fmt = "  %-65s  %8s  %10s  %10s  %10s  %10s"
    log(fmt % ("Experiment", "AUC", "AUC loss", "Infer(s)", "Setup(s)", "Mem(MB)"))
    log("  " + "-" * 115)

    for key in sorted_keys:
        r = all_results[key]
        if 'error' in r:
            log(f"  {key:<65s}  {'ERROR':>8s}  {r.get('error', 'unknown')[:50]}")
            continue
        name = r.get('name', key)
        if len(name) > 65:
            name = name[:62] + "..."
        auc_val = r.get('auc', 0)
        auc_loss = r.get('auc_loss_pp', 0)
        infer = r.get('inference_time', 0)
        setup = r.get('setup_time', 0)
        mem = r.get('memory_mb', 0)
        log(fmt % (name, f"{auc_val:.6f}", f"{auc_loss:.4f}pp",
                   f"{infer:.2f}", f"{setup:.2f}", f"{mem:.1f}"))

    # Compute comparison vs CAFE+
    log("\n  Comparison vs CAFE+ (AUC=0.8010, 16x: AUC=0.7882):")
    log("  " + "-" * 80)
    for key in sorted_keys:
        r = all_results[key]
        if 'error' in r or 'auc' not in r:
            continue
        auc_val = r.get('auc', 0)
        cafe_gap = (auc_val - 0.8010) * 100
        cafe16_gap = (auc_val - 0.7882) * 100
        mem = r.get('memory_mb', 0)
        name = r.get('name', key)
        if len(name) > 55:
            name = name[:52] + "..."
        log(f"  {name:<55s}  AUC={auc_val:.6f}  vs CAFE+: +{cafe_gap:.2f}pp  "
            f"vs CAFE+16x: +{cafe16_gap:.2f}pp  Mem={mem:.1f}MB")

    # Find Pareto-optimal configs (best tradeoff of AUC loss vs memory)
    log("\n  Pareto-optimal configurations (AUC loss vs memory):")
    log("  " + "-" * 80)
    valid_results = [(k, r) for k, r in all_results.items()
                     if 'error' not in r and 'auc' in r]
    valid_results.sort(key=lambda x: x[1].get('memory_mb', 9999))

    pareto = []
    best_auc_so_far = -1
    for key, r in valid_results:
        auc_val = r.get('auc', 0)
        if auc_val > best_auc_so_far:
            pareto.append((key, r))
            best_auc_so_far = auc_val

    for key, r in pareto:
        name = r.get('name', key)
        if len(name) > 55:
            name = name[:52] + "..."
        log(f"  {name:<55s}  AUC={r['auc']:.6f}  loss={r.get('auc_loss_pp', 0):.4f}pp  "
            f"Mem={r.get('memory_mb', 0):.1f}MB  Infer={r.get('inference_time', 0):.1f}s")

    # Save final results
    save_results_incremental(all_results)
    log(f"\nResults saved to: {JSON_PATH}")
    log(f"Log saved to: {LOG_PATH}")

    log("\n" + "=" * 70)
    log("DONE!")
    log("=" * 70)

    if _log_fh is not None:
        _log_fh.close()
        _log_fh = None


if __name__ == '__main__':
    main()
