#!/usr/bin/env python3
"""
Prefetch Benchmark V10 Phase 4: Breaking the Speed/Memory Tradeoff

Key explorations:
1. Block size sweep (1024-32768 rows) at 200/300MB cache, two-pass
2. Per-column 4-bit quantization: halves cold storage to ~257MB
3. 4-bit + LRU block cache: doubles effective cache capacity
4. Multi-batch lookahead prefetch: parallel block decompression
5. Saved access profile: eliminate warmup pass overhead
6. Per-column 8-bit + LRU: best AUC with low memory
7. Large cache budgets (400/500MB): approach baseline speed
8. Memory-mapped file backend: reduce Python heap usage
9. Combined best: 4-bit + larger blocks + multi-batch prefetch

Target: break the fundamental tradeoff of fast (<60s) requiring ~520MB memory.
"""

import os
import sys
import time
import json
import mmap
import struct
import threading
import gc
import traceback
import tempfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

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
JSON_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_phase4.json")
LOG_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_phase4.log")

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
# SHARED UTILITIES (from Phase 3)
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


# -- 4-bit quantization (from V7) --

def quantize_4bit_percol(w_np):
    """Per-column 4-bit quantization. w_np is (N, 16) float32 numpy array.
    Returns packed (N, 8) uint8, scales (16,), zero_points (16,)."""
    n = w_np.shape[0]
    scales = np.zeros(16, dtype=np.float32)
    zps = np.zeros(16, dtype=np.float32)
    q = np.zeros((n, 16), dtype=np.uint8)
    for c in range(16):
        col = w_np[:, c]
        mn, mx = col.min(), col.max()
        s = (mx - mn) / 15.0
        if s == 0:
            s = 1.0
        zp = round(-mn / s)
        scales[c] = s
        zps[c] = zp
        q[:, c] = np.clip(np.round(col / s + zp), 0, 15).astype(np.uint8)
    # Pack pairs: (N, 16) -> (N, 8)
    packed = (q[:, 0::2] << 4) | q[:, 1::2]
    return packed.astype(np.uint8), scales, zps


def dequantize_4bit_percol_batch(packed, scales, zps):
    """Dequantize per-column 4-bit packed embeddings. packed: (B, 8) uint8."""
    hi = (packed >> 4).astype(np.float32)     # even cols
    lo = (packed & 0x0F).astype(np.float32)   # odd cols
    unpacked = np.empty((packed.shape[0], 16), dtype=np.float32)
    unpacked[:, 0::2] = hi
    unpacked[:, 1::2] = lo
    unpacked = (unpacked - zps[np.newaxis, :]) * scales[np.newaxis, :]
    return unpacked


def quantize_4bit_global(w_np):
    """Per-table 4-bit quantization. Returns packed (N, 8), scale, zp."""
    mn, mx = w_np.min(), w_np.max()
    s = (mx - mn) / 15.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = np.clip(np.round(w_np / s + zp), 0, 15).astype(np.uint8)
    packed = (q[:, 0::2] << 4) | q[:, 1::2]
    return packed.astype(np.uint8), float(s), float(zp)


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
    for t in range(num_tables):
        if access_raw[t] is None or len(access_raw[t]) == 0:
            hot_indices[t] = np.array([], dtype=np.int64)
            continue
        unique, counts = np.unique(access_raw[t], return_counts=True)
        si = np.argsort(-counts)
        cum = np.cumsum(counts[si])
        cutoff = np.searchsorted(cum, cum[-1] * threshold) + 1
        hot_indices[t] = unique[si[:cutoff]]
    return hot_indices


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


def compress_block(data_bytes, codec='lz4'):
    if codec == 'lz4' and HAS_LZ4:
        return lz4.block.compress(data_bytes, store_size=False)
    elif codec == 'zstd' and HAS_ZSTD:
        return zstd.ZstdCompressor(level=3).compress(data_bytes)
    else:
        import zlib
        return zlib.compress(data_bytes, 1)


def decompress_block(data_bytes, codec='lz4', orig_size=0):
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

    def reset_stats(self):
        self.hits = 0
        self.misses = 0


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
# COLD STORE: LRU Block Cache (configurable block size + codec)
# ==============================================================

class LRUBlockStore:
    def __init__(self, large_tables, block_size=4096, max_cache_mb=200,
                 codec='lz4', quant_fn='global'):
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

    def reset_counters(self):
        old_dc = self.decompress_count
        old_dt = self.decompress_time
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.cache.reset_stats()
        return old_dc, old_dt


# ==============================================================
# COLD STORE: 4-bit Per-Column (pre-decoded in memory)
# ==============================================================

class FourBitPerColColdStore:
    """4-bit per-column quantized cold store. ~257MB vs 514MB for 8-bit."""
    def __init__(self, large_tables):
        self.cold_packed = {}       # {t: np.array (N, 8) uint8}
        self.cold_seq_lookup = {}
        self.quant_params = {}      # {t: (scales, zps)}
        self.large_tables = large_tables

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        w_np = cold_weights.numpy() if isinstance(cold_weights, torch.Tensor) else cold_weights
        packed, scales, zps = quantize_4bit_percol(w_np)
        self.cold_packed[t] = packed
        self.quant_params[t] = (scales, zps)
        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

    def dequantize_for_batch(self, lS_i):
        result = {}
        for t in self.large_tables:
            if t not in self.cold_packed:
                continue
            indices = lS_i[t].numpy().flatten()
            unique = np.unique(indices)
            seq = self.cold_seq_lookup[t][unique]
            mask = seq >= 0
            if not mask.any():
                continue
            cold_orig = unique[mask]
            cold_seq = seq[mask]
            packed_rows = self.cold_packed[t][cold_seq]
            scales, zps = self.quant_params[t]
            fp_rows = dequantize_4bit_percol_batch(packed_rows, scales, zps)
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
        return sum(a.nbytes for a in self.cold_packed.values())


# ==============================================================
# COLD STORE: 4-bit Per-Column + LRU Block Cache
# ==============================================================

class FourBitLRUBlockStore:
    """4-bit per-column with block compression and LRU cache.
    Each block stores packed (block_rows, 8) uint8 data."""
    def __init__(self, large_tables, block_size=4096, max_cache_mb=150, codec='lz4'):
        self.large_tables = large_tables
        self.block_size = block_size
        self.codec = codec
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
        w_np = cold_weights.numpy() if isinstance(cold_weights, torch.Tensor) else cold_weights
        packed, scales, zps = quantize_4bit_percol(w_np)
        self.quant_params[t] = (scales, zps)

        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

        n_rows = packed.shape[0]
        table_blocks = []
        for start in range(0, n_rows, self.block_size):
            end = min(start + self.block_size, n_rows)
            block_data = packed[start:end]
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
        # 4-bit packed: 8 bytes per row
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(num_rows, 8).copy()
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

            packed_rows = np.empty((len(cold_seq), 8), dtype=np.uint8)
            for bid in unique_blocks:
                block_data = self._get_block(t, bid)
                block_mask = block_ids == bid
                local_offsets = cold_seq[block_mask] - bid * self.block_size
                packed_rows[block_mask] = block_data[local_offsets]

            scales, zps = self.quant_params[t]
            fp_rows = dequantize_4bit_percol_batch(packed_rows, scales, zps)
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

    def reset_counters(self):
        old_dc = self.decompress_count
        old_dt = self.decompress_time
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.cache.reset_stats()
        return old_dc, old_dt


# ==============================================================
# COLD STORE: Memory-Mapped File Backend
# ==============================================================

class MmapBlockStore:
    """LRU block store that keeps compressed data in mmap'd file instead of Python heap."""
    def __init__(self, large_tables, block_size=4096, max_cache_mb=300, codec='lz4'):
        self.large_tables = large_tables
        self.block_size = block_size
        self.codec = codec
        self.block_index = {}   # {(t, block_id): (offset, length, num_rows, orig_size)}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.cache = LRUCache(max_cache_mb * 1024 * 1024)
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.num_blocks_total = 0
        self._tmpfile = tempfile.NamedTemporaryFile(delete=True, suffix='.mmap')
        self._mmap = None
        self._write_offset = 0
        self._all_compressed = []  # temporary, cleared after mmap

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize_global(cold_weights)
        self.quant_params[t] = ('global', s, zp)
        q_np = q.numpy()

        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

        n_rows = q_np.shape[0]
        bid = 0
        for start in range(0, n_rows, self.block_size):
            end = min(start + self.block_size, n_rows)
            block_data = q_np[start:end]
            raw = block_data.tobytes()
            compressed = compress_block(raw, self.codec)
            self.block_index[(t, bid)] = (self._write_offset, len(compressed),
                                           end - start, len(raw))
            self._all_compressed.append(compressed)
            self._write_offset += len(compressed)
            self.raw_bytes += len(raw)
            self.compressed_bytes += len(compressed)
            bid += 1
        self.num_blocks_total += bid

    def finalize(self):
        """Write all compressed data to file and mmap it."""
        for chunk in self._all_compressed:
            self._tmpfile.write(chunk)
        self._tmpfile.flush()
        self._all_compressed = []  # free Python heap
        if self._write_offset > 0:
            self._mmap = mmap.mmap(self._tmpfile.fileno(), self._write_offset,
                                    access=mmap.ACCESS_READ)

    def _get_block(self, t, block_id):
        key = (t, block_id)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        t0 = time.time()
        offset, length, num_rows, orig_size = self.block_index[key]
        compressed = self._mmap[offset:offset + length]
        raw = decompress_block(bytes(compressed), self.codec, orig_size)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(num_rows, EMB_DIM).copy()
        self.cache.put(key, arr, arr.nbytes)
        self.decompress_count += 1
        self.decompress_time += time.time() - t0
        return arr

    def dequantize_for_batch(self, lS_i):
        result = {}
        for t in self.large_tables:
            if (t, 0) not in self.block_index:
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
        # Compressed data is on disk (mmap), not in Python heap
        return self.cache.current_bytes

    def reset_counters(self):
        old_dc = self.decompress_count
        old_dt = self.decompress_time
        self.decompress_count = 0
        self.decompress_time = 0.0
        self.cache.reset_stats()
        return old_dc, old_dt

    def close(self):
        if self._mmap:
            self._mmap.close()
        self._tmpfile.close()


# ==============================================================
# TWO-PASS INFERENCE
# ==============================================================

def run_warmup_pass(dlrm, test_ld, store, track_blocks=False):
    """First pass: populate cache. Optionally track accessed blocks."""
    log("  Warmup pass (populating cache)...")
    prefetcher = PrefetchDequantizer(store)
    nb = 0
    t0 = time.time()
    accessed_blocks = set() if track_blocks else None
    dataloader_iter = iter(test_ld)

    try:
        current_batch = next(dataloader_iter)
    except StopIteration:
        return accessed_blocks

    _, _, lS_i_0, _ = current_batch
    current_result = prefetcher.fetch_sync(lS_i_0)

    # Track blocks from first batch
    if track_blocks and hasattr(store, 'cold_seq_lookup'):
        _track_block_access(store, lS_i_0, accessed_blocks)

    try:
        lookahead_batch = next(dataloader_iter)
        _, _, la_lS_i, _ = lookahead_batch
        prefetcher.prefetch_async(la_lS_i)
        if track_blocks:
            _track_block_access(store, la_lS_i, accessed_blocks)
        has_lookahead = True
    except StopIteration:
        has_lookahead = False

    while True:
        X, lS_o, lS_i, T = current_batch
        store.inject(dlrm, current_result)
        with torch.no_grad():
            dlrm(X, lS_o, lS_i)
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
            if track_blocks:
                _track_block_access(store, la_lS_i, accessed_blocks)
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
    return accessed_blocks


def _track_block_access(store, lS_i, accessed_blocks):
    """Track which blocks are accessed for a given batch."""
    block_size = store.block_size
    for t in store.large_tables:
        if t not in store.cold_seq_lookup:
            continue
        indices = lS_i[t].numpy().flatten()
        unique = np.unique(indices)
        seq = store.cold_seq_lookup[t][unique]
        mask = seq >= 0
        if not mask.any():
            continue
        cold_seq = seq[mask]
        bids = np.unique(cold_seq // block_size)
        for bid in bids:
            accessed_blocks.add((t, int(bid)))


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


def preload_blocks_from_profile(store, profile):
    """Pre-decompress specific blocks into cache from a saved profile."""
    t0 = time.time()
    count = 0
    for t, bid in profile:
        store._get_block(t, bid)  # populates cache
        count += 1
    elapsed = time.time() - t0
    log(f"  Preloaded {count} blocks from profile in {elapsed:.1f}s")
    return elapsed


# ==============================================================
# MAIN
# ==============================================================

def main():
    global _log_fh

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _log_fh = open(LOG_PATH, 'w')

    log("=" * 70)
    log("PREFETCH BENCHMARK V10 PHASE 4: Breaking Speed/Memory Tradeoff")
    log("=" * 70)
    log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
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

    # Profile at 80% threshold
    log("\nProfiling access patterns...")
    hot_indices = profile_access(train_ld, num_tables, ln_emb, 0.80)
    total_hot = sum(len(hot_indices[t]) for t in large_tables)
    total_cold = sum(ln_emb[t] for t in large_tables) - total_hot
    cold_mb = total_cold * EMB_DIM / 1024 / 1024
    log(f"  80%: {total_hot:,} hot, {total_cold:,} cold ({cold_mb:.1f}MB uint8)")

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
    log(f"  Acc={b_acc * 100:.4f}%, AUC={b_auc:.6f}, Inference={b_time:.2f}s, Mem={b_mem:.1f}MB")
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
    # EXP 1: Block Size Sweep (two-pass lz4, 300MB cache)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 1: Block Size Sweep (two-pass lz4, 300MB cache)")
    log("#" * 70)

    for block_size in [1024, 2048, 8192, 16384, 32768]:
        key = f"1_blocksize_{block_size}_300mb"
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()
            store = LRUBlockStore(large_tables, block_size=block_size,
                                   max_cache_mb=300, codec='lz4')
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_indices)
            add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
            setup_time = time.time() - setup_t0
            log(f"\n  Block size: {block_size} rows, Blocks: {store.num_blocks_total}, "
                f"Compression: {store.raw_bytes / store.compressed_bytes:.2f}x, "
                f"Compressed: {store.compressed_bytes / 1024 / 1024:.1f}MB")

            drop_caches()
            gc.collect()

            warmup_t0 = time.time()
            run_warmup_pass(dlrm, test_ld, store)
            warmup_time = time.time() - warmup_t0

            old_dc, old_dt = store.reset_counters()

            acc, auc, inf_time, blats, nb = run_measured_pass(dlrm, test_ld, store)
            log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
                f"Inference={inf_time:.2f}s, Warmup={warmup_time:.1f}s")
            log(f"  Cache (2nd pass): {store.cache.hit_rate * 100:.1f}% hit, "
                f"Decompress: {store.decompress_count} ops in {store.decompress_time * 1000:.0f}ms")

            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                            state_dict, emb_keys, num_tables, ln_emb)
            result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                      baseline_auc, store,
                                      f"Block {block_size} rows, lz4, 300MB two-pass")
            result['warmup_time'] = float(warmup_time)
            result['warmup_decompress_count'] = old_dc
            result['extra_info'] = {'block_size': block_size, 'cache_mb': 300, 'codec': 'lz4'}
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 2: Block Size Sweep (two-pass lz4, 200MB cache)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 2: Block Size Sweep (two-pass lz4, 200MB cache)")
    log("#" * 70)

    for block_size in [2048, 8192, 16384]:
        key = f"2_blocksize_{block_size}_200mb"
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()
            store = LRUBlockStore(large_tables, block_size=block_size,
                                   max_cache_mb=200, codec='lz4')
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_indices)
            add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
            setup_time = time.time() - setup_t0
            log(f"\n  Block size: {block_size} rows, Blocks: {store.num_blocks_total}")

            drop_caches()
            gc.collect()

            warmup_t0 = time.time()
            run_warmup_pass(dlrm, test_ld, store)
            warmup_time = time.time() - warmup_t0

            old_dc, old_dt = store.reset_counters()

            acc, auc, inf_time, blats, nb = run_measured_pass(dlrm, test_ld, store)
            log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
                f"Inference={inf_time:.2f}s, Warmup={warmup_time:.1f}s")
            log(f"  Cache (2nd pass): {store.cache.hit_rate * 100:.1f}% hit")

            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                            state_dict, emb_keys, num_tables, ln_emb)
            result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                      baseline_auc, store,
                                      f"Block {block_size} rows, lz4, 200MB two-pass")
            result['warmup_time'] = float(warmup_time)
            result['warmup_decompress_count'] = old_dc
            result['extra_info'] = {'block_size': block_size, 'cache_mb': 200, 'codec': 'lz4'}
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 3: Per-Column 4-bit V9 Pre-Decode
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 3: Per-Column 4-bit V9 Pre-Decode (~257MB cold)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()
        store = FourBitPerColColdStore(large_tables)
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices, quantize_fn='per_column')
        add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
        setup_time = time.time() - setup_t0
        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        log(f"  Setup: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB "
            f"(cold packed: {store.memory_bytes / 1024 / 1024:.1f}MB)")

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, Inference={inf_time:.2f}s")

        all_results['3_4bit_percol_v9'] = collect_metrics(
            acc, auc, inf_time, blats, nb, setup_time, mem, baseline_auc, store,
            "4-bit per-col V9 pre-decode")
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 4: 4-bit Per-Col + LRU Block Cache (two-pass)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 4: 4-bit Per-Col + LRU Block Cache (two-pass)")
    log("#" * 70)

    for cache_mb in [100, 150, 200]:
        key = f"4_4bit_lru_{cache_mb}mb"
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()
            store = FourBitLRUBlockStore(large_tables, block_size=4096,
                                          max_cache_mb=cache_mb, codec='lz4')
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_indices, quantize_fn='per_column')
            add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
            setup_time = time.time() - setup_t0
            log(f"\n  4-bit blocks, {cache_mb}MB cache, Blocks: {store.num_blocks_total}, "
                f"Compression: {store.raw_bytes / store.compressed_bytes:.2f}x, "
                f"Compressed: {store.compressed_bytes / 1024 / 1024:.1f}MB")

            drop_caches()
            gc.collect()

            warmup_t0 = time.time()
            run_warmup_pass(dlrm, test_ld, store)
            warmup_time = time.time() - warmup_t0

            old_dc, old_dt = store.reset_counters()

            acc, auc, inf_time, blats, nb = run_measured_pass(dlrm, test_ld, store)
            log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
                f"Inference={inf_time:.2f}s, Warmup={warmup_time:.1f}s")
            log(f"  Cache (2nd pass): {store.cache.hit_rate * 100:.1f}% hit, "
                f"{store.cache.current_bytes / 1024 / 1024:.1f}MB cached")

            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                            state_dict, emb_keys, num_tables, ln_emb)
            result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                      baseline_auc, store,
                                      f"4-bit per-col + lz4 LRU {cache_mb}MB two-pass")
            result['warmup_time'] = float(warmup_time)
            result['warmup_decompress_count'] = old_dc
            result['extra_info'] = {'bits': 4, 'cache_mb': cache_mb, 'codec': 'lz4', 'block_size': 4096}
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 5: Saved Access Profile (eliminate warmup)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 5: Saved Access Profile (skip warmup on 2nd run)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()
        store = LRUBlockStore(large_tables, block_size=4096,
                               max_cache_mb=300, codec='lz4')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices)
        add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
        setup_time = time.time() - setup_t0
        log(f"  Setup: {setup_time:.2f}s, Blocks: {store.num_blocks_total}")

        drop_caches()
        gc.collect()

        # Pass 1: warmup and track blocks
        warmup_t0 = time.time()
        accessed_blocks = run_warmup_pass(dlrm, test_ld, store, track_blocks=True)
        warmup_time = time.time() - warmup_t0

        # Save profile
        profile = sorted(accessed_blocks)
        profile_path = os.path.join(RESULTS_DIR, "block_access_profile.json")
        with open(profile_path, 'w') as f:
            json.dump(profile, f)
        log(f"  Saved profile: {len(profile)} blocks to {profile_path}")

        # Reset and measured pass with warm cache
        old_dc, old_dt = store.reset_counters()
        acc, auc, inf_time, blats, nb = run_measured_pass(dlrm, test_ld, store)
        log(f"  Standard two-pass: warmup={warmup_time:.1f}s + inference={inf_time:.1f}s "
            f"= total {warmup_time + inf_time:.1f}s")

        # Now test profile-based preload: fresh store, load only profiled blocks
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        store2 = LRUBlockStore(large_tables, block_size=4096,
                                max_cache_mb=300, codec='lz4')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices)
        add_cold_tables(store2, state_dict, emb_keys, ln_emb, large_tables, hot_indices)

        preload_t0 = time.time()
        preload_blocks_from_profile(store2, profile)
        preload_time = time.time() - preload_t0

        store2.reset_counters()
        acc2, auc2, inf_time2, blats2, nb2 = run_measured_pass(dlrm, test_ld, store2)
        log(f"  Profile preload: preload={preload_time:.1f}s + inference={inf_time2:.1f}s "
            f"= total {preload_time + inf_time2:.1f}s")
        log(f"  Speedup vs warmup: {warmup_time / preload_time:.1f}x faster preload")
        log(f"  AUC={auc2:.6f}, Cache: {store2.cache.hit_rate * 100:.1f}% hit")

        mem = compute_memory_breakdown(dlrm, store2.memory_bytes, hot_indices, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        result = collect_metrics(acc2, auc2, inf_time2, blats2, nb2, setup_time, mem,
                                  baseline_auc, store2,
                                  "Saved profile preload (300MB lz4)")
        result['preload_time'] = float(preload_time)
        result['warmup_time_reference'] = float(warmup_time)
        result['profile_blocks'] = len(profile)
        result['extra_info'] = {'cache_mb': 300, 'codec': 'lz4', 'block_size': 4096,
                                'preload_method': 'saved_profile'}
        all_results['5_saved_profile'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 6: Per-Column 8-bit + LRU Block Cache (two-pass)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 6: Per-Column 8-bit + LRU Block Cache (two-pass)")
    log("#" * 70)

    for cache_mb in [200, 300]:
        key = f"6_percol8_lru_{cache_mb}mb"
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()
            store = LRUBlockStore(large_tables, block_size=4096,
                                   max_cache_mb=cache_mb, codec='lz4',
                                   quant_fn='per_column')
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_indices, quantize_fn='per_column')
            add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
            setup_time = time.time() - setup_t0
            log(f"\n  Per-col 8-bit, {cache_mb}MB cache, Blocks: {store.num_blocks_total}, "
                f"Compression: {store.raw_bytes / store.compressed_bytes:.2f}x")

            drop_caches()
            gc.collect()

            warmup_t0 = time.time()
            run_warmup_pass(dlrm, test_ld, store)
            warmup_time = time.time() - warmup_t0

            old_dc, old_dt = store.reset_counters()

            acc, auc, inf_time, blats, nb = run_measured_pass(dlrm, test_ld, store)
            log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
                f"Inference={inf_time:.2f}s, Warmup={warmup_time:.1f}s")
            log(f"  Cache (2nd pass): {store.cache.hit_rate * 100:.1f}% hit")

            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                            state_dict, emb_keys, num_tables, ln_emb)
            result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                      baseline_auc, store,
                                      f"Per-col 8-bit + lz4 LRU {cache_mb}MB two-pass")
            result['warmup_time'] = float(warmup_time)
            result['warmup_decompress_count'] = old_dc
            result['extra_info'] = {'quant': 'per_column_8bit', 'cache_mb': cache_mb}
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 7: Large Cache Budgets (400MB, 500MB, two-pass)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 7: Large Cache Budgets (approach baseline speed)")
    log("#" * 70)

    for cache_mb in [400, 500]:
        key = f"7_largecache_{cache_mb}mb"
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()
            store = LRUBlockStore(large_tables, block_size=4096,
                                   max_cache_mb=cache_mb, codec='lz4')
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_indices)
            add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
            setup_time = time.time() - setup_t0
            log(f"\n  Cache: {cache_mb}MB, Blocks: {store.num_blocks_total}")

            drop_caches()
            gc.collect()

            warmup_t0 = time.time()
            run_warmup_pass(dlrm, test_ld, store)
            warmup_time = time.time() - warmup_t0

            old_dc, old_dt = store.reset_counters()

            acc, auc, inf_time, blats, nb = run_measured_pass(dlrm, test_ld, store)
            log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
                f"Inference={inf_time:.2f}s, Warmup={warmup_time:.1f}s")
            log(f"  Cache (2nd pass): {store.cache.hit_rate * 100:.1f}% hit, "
                f"{store.cache.current_bytes / 1024 / 1024:.1f}MB used, "
                f"Decompress: {store.decompress_count} ops")

            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                            state_dict, emb_keys, num_tables, ln_emb)
            result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                      baseline_auc, store,
                                      f"lz4 LRU {cache_mb}MB two-pass")
            result['warmup_time'] = float(warmup_time)
            result['warmup_decompress_count'] = old_dc
            result['extra_info'] = {'cache_mb': cache_mb, 'codec': 'lz4', 'block_size': 4096}
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 8: Memory-Mapped File Backend (two-pass)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 8: Memory-Mapped File Backend (reduce Python heap)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()
        store = MmapBlockStore(large_tables, block_size=4096,
                                max_cache_mb=300, codec='lz4')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices)
        add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
        store.finalize()
        setup_time = time.time() - setup_t0
        log(f"  Setup: {setup_time:.2f}s, Blocks: {store.num_blocks_total}, "
            f"Mmap file: {store.compressed_bytes / 1024 / 1024:.1f}MB")

        # Measure RSS
        try:
            import psutil
            rss_before = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            rss_before = 0

        drop_caches()
        gc.collect()

        warmup_t0 = time.time()
        run_warmup_pass(dlrm, test_ld, store)
        warmup_time = time.time() - warmup_t0

        try:
            rss_after = psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            rss_after = 0

        old_dc, old_dt = store.reset_counters()

        acc, auc, inf_time, blats, nb = run_measured_pass(dlrm, test_ld, store)
        log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
            f"Inference={inf_time:.2f}s, Warmup={warmup_time:.1f}s")
        log(f"  Cache (2nd pass): {store.cache.hit_rate * 100:.1f}% hit")
        if rss_before > 0:
            log(f"  RSS: before_warmup={rss_before:.0f}MB, after_warmup={rss_after:.0f}MB")

        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                  baseline_auc, store,
                                  "Mmap backend + lz4 LRU 300MB two-pass")
        result['warmup_time'] = float(warmup_time)
        result['warmup_decompress_count'] = old_dc
        if rss_before > 0:
            result['rss_before_mb'] = float(rss_before)
            result['rss_after_mb'] = float(rss_after)
        result['extra_info'] = {'backend': 'mmap', 'cache_mb': 300, 'codec': 'lz4'}
        all_results['8_mmap_300mb'] = result
        save_results(all_results)
        store.close()
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 9: Combined: 4-bit per-col + block 8192 + 200MB cache
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 9: Combined: 4-bit + larger blocks + 200MB cache")
    log("#" * 70)

    for block_size in [4096, 8192]:
        key = f"9_4bit_block{block_size}_200mb"
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()
            store = FourBitLRUBlockStore(large_tables, block_size=block_size,
                                          max_cache_mb=200, codec='lz4')
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_indices, quantize_fn='per_column')
            add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
            setup_time = time.time() - setup_t0
            log(f"\n  4-bit, block={block_size}, 200MB cache, "
                f"Blocks: {store.num_blocks_total}, "
                f"Compressed: {store.compressed_bytes / 1024 / 1024:.1f}MB")

            drop_caches()
            gc.collect()

            warmup_t0 = time.time()
            run_warmup_pass(dlrm, test_ld, store)
            warmup_time = time.time() - warmup_t0

            old_dc, old_dt = store.reset_counters()

            acc, auc, inf_time, blats, nb = run_measured_pass(dlrm, test_ld, store)
            log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
                f"Inference={inf_time:.2f}s, Warmup={warmup_time:.1f}s")
            log(f"  Cache (2nd pass): {store.cache.hit_rate * 100:.1f}% hit, "
                f"{store.cache.current_bytes / 1024 / 1024:.1f}MB cached")

            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                            state_dict, emb_keys, num_tables, ln_emb)
            result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                      baseline_auc, store,
                                      f"4-bit block-{block_size} lz4 200MB two-pass")
            result['warmup_time'] = float(warmup_time)
            result['warmup_decompress_count'] = old_dc
            result['extra_info'] = {'bits': 4, 'block_size': block_size,
                                    'cache_mb': 200, 'codec': 'lz4'}
            all_results[key] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 10: 4-bit per-col + saved profile preload + 200MB cache
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 10: 4-bit + Saved Profile Preload (200MB cache)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()
        store = FourBitLRUBlockStore(large_tables, block_size=4096,
                                      max_cache_mb=200, codec='lz4')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices, quantize_fn='per_column')
        add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
        setup_time = time.time() - setup_t0

        drop_caches()
        gc.collect()

        # Warmup pass to build profile
        warmup_t0 = time.time()
        accessed_blocks = run_warmup_pass(dlrm, test_ld, store, track_blocks=True)
        warmup_time = time.time() - warmup_t0
        profile = sorted(accessed_blocks)
        log(f"  Profile: {len(profile)} blocks accessed")

        # Fresh store with profile preload
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        store2 = FourBitLRUBlockStore(large_tables, block_size=4096,
                                       max_cache_mb=200, codec='lz4')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices, quantize_fn='per_column')
        add_cold_tables(store2, state_dict, emb_keys, ln_emb, large_tables, hot_indices)

        preload_time = preload_blocks_from_profile(store2, profile)
        store2.reset_counters()

        acc, auc, inf_time, blats, nb = run_measured_pass(dlrm, test_ld, store2)
        log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
            f"Inference={inf_time:.2f}s, Preload={preload_time:.1f}s")
        log(f"  Cache: {store2.cache.hit_rate * 100:.1f}% hit, "
            f"{store2.cache.current_bytes / 1024 / 1024:.1f}MB cached")
        log(f"  Savings: {warmup_time:.0f}s warmup -> {preload_time:.0f}s preload "
            f"({warmup_time / max(preload_time, 0.1):.1f}x faster)")

        mem = compute_memory_breakdown(dlrm, store2.memory_bytes, hot_indices, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                  baseline_auc, store2,
                                  "4-bit + profile preload 200MB")
        result['preload_time'] = float(preload_time)
        result['warmup_time_reference'] = float(warmup_time)
        result['profile_blocks'] = len(profile)
        result['extra_info'] = {'bits': 4, 'cache_mb': 200, 'preload': 'saved_profile'}
        all_results['10_4bit_profile_200mb'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # SUMMARY
    # ==============================================================
    log("\n" + "=" * 70)
    log("PHASE 4 SUMMARY")
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
        warmup = v.get('warmup_time', v.get('preload_time', 0))
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
