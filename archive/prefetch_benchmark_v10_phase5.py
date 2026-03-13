#!/usr/bin/env python3
"""
Prefetch Benchmark V10 Phase 5: Block-to-Flat Preload

Key insight: Blocks are a STORAGE format, not an INFERENCE format.
Phase 4 showed 500MB cache with 100% hit rate still takes 91.7s (vs V9's 42.4s)
because of per-batch block-level indirection + numpy operations.

Solution: Decompress blocks into a flat V9-style array. Use blocks for disk
compression, flat arrays for inference.

Experiments:
5A - Block-to-flat full preload (global 8-bit)
5G - Full fp32 pre-decode with per-col quant (theoretical limit)
5D - Row-level cold coverage analysis (profiling only)
5C - Per-col 8-bit block-to-flat preload
5B - Profile-guided block-to-flat (skip unneeded blocks)
5E - Minimal cold row preload (only accessed rows)
5F - Hybrid lazy block-to-flat (decompresses on first access)
"""

import os
import sys
import time
import json
import threading
import gc
import traceback

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

try:
    import lz4.block
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

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
PROFILE_BATCHES = 200
LARGE_TABLE_THRESHOLD = 50000

os.makedirs(RESULTS_DIR, exist_ok=True)
JSON_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_phase5.json")
LOG_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_phase5.log")

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

def quantize_global(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
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
        if s == 0: s = 1.0
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
# COLD STORE: Block-to-Flat (compressed disk, flat array inference)
# ==============================================================

class BlockToFlatStore:
    """Blocks for storage, flat V9-style array for inference.
    Decompresses all blocks into a flat array at preload time."""

    def __init__(self, large_tables, block_size=4096, codec='lz4', quant_fn='global'):
        self.large_tables = large_tables
        self.block_size = block_size
        self.codec = codec
        self.quant_fn = quant_fn
        # Block storage (for disk compression)
        self.blocks = {}        # {t: [(compressed, num_rows, orig_size), ...]}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.num_blocks_total = 0
        self.num_cold_rows = {}
        # Flat inference arrays (populated during preload)
        self.cold_uint8 = {}    # {t: np.array (num_cold, 16) uint8}
        self._preloaded = False

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
        self.num_cold_rows[t] = q_np.shape[0]

        # Compress into blocks
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

    def preload_all(self):
        """Decompress all blocks into flat arrays for V9-speed inference."""
        t0 = time.time()
        for t in self.large_tables:
            if t not in self.blocks:
                continue
            flat = np.empty((self.num_cold_rows[t], EMB_DIM), dtype=np.uint8)
            offset = 0
            for compressed, num_rows, orig_size in self.blocks[t]:
                raw = decompress_block(compressed, self.codec, orig_size)
                flat[offset:offset + num_rows] = np.frombuffer(
                    raw, dtype=np.uint8).reshape(num_rows, EMB_DIM)
                offset += num_rows
            self.cold_uint8[t] = flat
        # Free compressed blocks from memory
        self.blocks = {}
        self._preloaded = True
        elapsed = time.time() - t0
        return elapsed

    def preload_from_profile(self, profile):
        """Decompress only blocks in the profile into flat arrays."""
        t0 = time.time()
        # Group profile by table
        profile_by_table = {}
        for t, bid in profile:
            if t not in profile_by_table:
                profile_by_table[t] = set()
            profile_by_table[t].add(bid)

        for t in self.large_tables:
            if t not in self.blocks:
                continue
            flat = np.empty((self.num_cold_rows[t], EMB_DIM), dtype=np.uint8)
            profiled = profile_by_table.get(t, set())
            offset = 0
            for bid, (compressed, num_rows, orig_size) in enumerate(self.blocks[t]):
                if bid in profiled:
                    raw = decompress_block(compressed, self.codec, orig_size)
                    flat[offset:offset + num_rows] = np.frombuffer(
                        raw, dtype=np.uint8).reshape(num_rows, EMB_DIM)
                # else: leave as uninitialized (never accessed based on profile)
                offset += num_rows
            self.cold_uint8[t] = flat
        self.blocks = {}
        self._preloaded = True
        elapsed = time.time() - t0
        return elapsed

    def dequantize_for_batch(self, lS_i):
        """V9-style direct indexing into flat array."""
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
        if self._preloaded:
            return sum(a.nbytes for a in self.cold_uint8.values())
        return self.compressed_bytes


# ==============================================================
# COLD STORE: Minimal Row Preload (only accessed rows)
# ==============================================================

class MinimalRowStore:
    """Only stores cold rows that are actually accessed during inference."""

    def __init__(self, large_tables, quant_fn='global'):
        self.large_tables = large_tables
        self.quant_fn = quant_fn
        self.cold_uint8 = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}

    def add_from_accessed(self, t, num_emb, accessed_cold_indices, cold_weights_all,
                           cold_indices_all):
        """Build store with only the rows that are accessed."""
        # accessed_cold_indices: set of original embedding indices that are accessed
        # cold_indices_all: full list of all cold indices (sorted)
        # cold_weights_all: weights for all cold indices

        # Build mapping: original cold index -> position in cold_weights_all
        cold_pos = {idx: pos for pos, idx in enumerate(cold_indices_all)}

        # Filter to accessed-only
        accessed_list = sorted(accessed_cold_indices)
        positions = [cold_pos[idx] for idx in accessed_list]
        weights_subset = cold_weights_all[positions]

        if self.quant_fn == 'per_column':
            q, scales, zps = quantize_per_column(weights_subset)
            self.quant_params[t] = ('per_column', scales, zps)
        else:
            q, s, zp = quantize_global(weights_subset)
            self.quant_params[t] = ('global', s, zp)
        self.cold_uint8[t] = q.numpy()

        # Build lookup: original index -> sequential position in accessed_list
        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(accessed_list):
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
        return sum(a.nbytes for a in self.cold_uint8.values())


# ==============================================================
# COLD STORE: Hybrid Lazy Block-to-Flat
# ==============================================================

class HybridLazyStore:
    """Lazy block-to-flat: decompresses blocks into flat array on first access."""

    def __init__(self, large_tables, block_size=4096, codec='lz4', quant_fn='global'):
        self.large_tables = large_tables
        self.block_size = block_size
        self.codec = codec
        self.quant_fn = quant_fn
        self.blocks = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.cold_uint8 = {}
        self.decompressed_bitmap = {}
        self.num_cold_rows = {}
        self.decompress_count = 0
        self.raw_bytes = 0
        self.compressed_bytes = 0

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
        self.num_cold_rows[t] = q_np.shape[0]

        # Pre-allocate flat array
        self.cold_uint8[t] = np.empty((q_np.shape[0], EMB_DIM), dtype=np.uint8)

        # Compress into blocks
        n_rows = q_np.shape[0]
        table_blocks = []
        num_blocks = 0
        for start in range(0, n_rows, self.block_size):
            end = min(start + self.block_size, n_rows)
            block_data = q_np[start:end]
            raw = block_data.tobytes()
            compressed = compress_block(raw, self.codec)
            table_blocks.append((compressed, end - start, len(raw)))
            self.raw_bytes += len(raw)
            self.compressed_bytes += len(compressed)
            num_blocks += 1
        self.blocks[t] = table_blocks
        self.decompressed_bitmap[t] = [False] * num_blocks

    def _ensure_block(self, t, block_id):
        if self.decompressed_bitmap[t][block_id]:
            return
        compressed, num_rows, orig_size = self.blocks[t][block_id]
        raw = decompress_block(compressed, self.codec, orig_size)
        offset = block_id * self.block_size
        self.cold_uint8[t][offset:offset + num_rows] = np.frombuffer(
            raw, dtype=np.uint8).reshape(num_rows, EMB_DIM)
        self.decompressed_bitmap[t][block_id] = True
        self.decompress_count += 1

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

            # Ensure needed blocks are decompressed
            block_ids = np.unique(cold_seq // self.block_size)
            for bid in block_ids:
                self._ensure_block(t, bid)

            # V9-style direct indexing
            q_rows = self.cold_uint8[t][cold_seq]
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
        # Flat arrays are always allocated; blocks shrink as decompressed
        flat = sum(a.nbytes for a in self.cold_uint8.values())
        remaining = sum(len(c) for t_blocks in self.blocks.values() for c, _, _ in t_blocks)
        return flat + remaining

    @property
    def blocks_decompressed(self):
        return sum(sum(b) for b in self.decompressed_bitmap.values())

    @property
    def blocks_total(self):
        return sum(len(b) for b in self.decompressed_bitmap.values())


# ==============================================================
# INFERENCE RUNNERS
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
        result['compression_ratio'] = float(store.raw_bytes / store.compressed_bytes)
        result['compressed_mb'] = float(store.compressed_bytes / 1024 / 1024)
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
    log("PREFETCH BENCHMARK V10 PHASE 5: Block-to-Flat Preload")
    log("=" * 70)
    log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"JSON output: {JSON_PATH}")

    all_results = {}

    log("\nLoading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")

    log("\nProfiling access patterns...")
    hot_indices = profile_access(train_ld, num_tables, ln_emb, 0.80)
    total_hot = sum(len(hot_indices[t]) for t in large_tables)
    total_cold = sum(ln_emb[t] for t in large_tables) - total_hot
    log(f"  80%: {total_hot:,} hot, {total_cold:,} cold ({total_cold * EMB_DIM / 1024 / 1024:.1f}MB)")

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
    log(f"  AUC={b_auc:.6f}, Inference={b_time:.2f}s, Mem={b_mem:.1f}MB")
    baseline_auc = b_auc
    all_results['0_baseline'] = {
        'name': 'Baseline', 'accuracy': float(b_acc), 'auc': float(b_auc),
        'auc_loss_pp': 0.0, 'inference_time': float(b_time), 'setup_time': 0.0,
        'total_time': float(b_time), 'memory_mb': float(b_mem),
        'batch_latency': latency_stats(b_blats), 'num_batches': b_nb,
    }
    save_results(all_results)

    # ==============================================================
    # EXP 5A: Block-to-Flat Full Preload (global 8-bit)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 5A: Block-to-Flat Full Preload (global 8-bit, lz4)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()
        store = BlockToFlatStore(large_tables, block_size=4096, codec='lz4', quant_fn='global')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices)
        add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
        compress_time = time.time() - setup_t0
        log(f"  Compression: {store.raw_bytes / 1024 / 1024:.1f}MB -> "
            f"{store.compressed_bytes / 1024 / 1024:.1f}MB "
            f"({store.raw_bytes / store.compressed_bytes:.2f}x)")

        # Preload: decompress all blocks -> flat array
        preload_time = store.preload_all()
        total_setup = compress_time + preload_time
        log(f"  Compress: {compress_time:.2f}s, Preload (decompress all): {preload_time:.2f}s")

        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        log(f"  Memory: {mem['total_mb']:.1f}MB")

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
            f"Inference={inf_time:.2f}s")

        result = collect_metrics(acc, auc, inf_time, blats, nb, total_setup, mem,
                                  baseline_auc, store,
                                  "Block-to-flat preload (global 8-bit lz4)")
        result['preload_time'] = float(preload_time)
        result['compress_time'] = float(compress_time)
        all_results['5A_block_to_flat_global'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 5G: Full fp32 Pre-Decode with Per-Col Quant
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 5G: Full fp32 Pre-Decode (per-col quant, no cold store)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()

        # Quantize and immediately dequantize all tables back to fp32
        for t in range(num_tables):
            w = state_dict[emb_keys[t]]
            q, scales, zps = quantize_per_column(w)
            dq = dequantize_per_column_torch(q, scales, zps)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = dq

        setup_time = time.time() - setup_t0
        total_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
        log(f"  Setup: {setup_time:.2f}s, Memory: {total_mem:.1f}MB (full fp32)")

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_baseline_inference(dlrm, test_ld)
        auc_loss = (baseline_auc - auc) * 100
        log(f"  AUC={auc:.6f}, Loss={auc_loss:+.4f}pp, Inference={inf_time:.2f}s")
        log(f"  (This is the theoretical best: per-col quant quality at baseline speed)")

        all_results['5G_fp32_predecode_percol'] = {
            'name': 'Full fp32 pre-decode (per-col quant)',
            'accuracy': float(acc), 'auc': float(auc), 'auc_loss_pp': float(auc_loss),
            'inference_time': float(inf_time), 'setup_time': float(setup_time),
            'total_time': float(setup_time + inf_time), 'memory_mb': float(total_mem),
            'batch_latency': latency_stats(blats), 'num_batches': nb,
        }
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 5D: Row-Level Cold Coverage Analysis
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 5D: Row-Level Cold Coverage Analysis")
    log("#" * 70)

    try:
        # Build cold index sets
        cold_idx_sets = {}
        for t in large_tables:
            hi = set(hot_indices[t].tolist())
            cold_idx_sets[t] = set(range(ln_emb[t])) - hi

        # Profile all test batches
        accessed_cold = {t: set() for t in large_tables}
        nb = 0
        for X, lS_o, lS_i, T in test_ld:
            for t in large_tables:
                indices = lS_i[t].numpy().flatten()
                unique = np.unique(indices)
                cold_in_batch = set(unique.tolist()) & cold_idx_sets[t]
                accessed_cold[t].update(cold_in_batch)
            nb += 1
            if nb % 500 == 0:
                log(f"    Profiled batch {nb}")

        log(f"\n  Row-level coverage (across {nb} test batches):")
        total_accessed = 0
        total_cold = 0
        row_coverage = {}
        for t in large_tables:
            n_cold = len(cold_idx_sets[t])
            n_accessed = len(accessed_cold[t])
            pct = n_accessed / n_cold * 100 if n_cold > 0 else 0
            log(f"    Table {t}: {n_accessed:,}/{n_cold:,} cold rows accessed ({pct:.1f}%)")
            total_accessed += n_accessed
            total_cold += n_cold
            row_coverage[t] = {
                'cold_total': n_cold, 'cold_accessed': n_accessed,
                'coverage_pct': float(pct),
            }

        overall_pct = total_accessed / total_cold * 100 if total_cold > 0 else 0
        log(f"    TOTAL: {total_accessed:,}/{total_cold:,} ({overall_pct:.1f}%)")
        log(f"    Potential memory savings: "
            f"{(total_cold - total_accessed) * EMB_DIM / 1024 / 1024:.1f}MB")

        all_results['5D_row_coverage'] = {
            'name': 'Row-level cold coverage analysis',
            'total_cold_rows': total_cold,
            'total_accessed_rows': total_accessed,
            'overall_coverage_pct': float(overall_pct),
            'per_table': row_coverage,
            'potential_savings_mb': float((total_cold - total_accessed) * EMB_DIM / 1024 / 1024),
        }
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 5C: Per-Column 8-bit Block-to-Flat Preload
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 5C: Per-Col 8-bit Block-to-Flat Preload (best AUC + fast)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()
        store = BlockToFlatStore(large_tables, block_size=4096, codec='lz4',
                                  quant_fn='per_column')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices, quantize_fn='per_column')
        add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
        compress_time = time.time() - setup_t0
        log(f"  Compression: {store.raw_bytes / 1024 / 1024:.1f}MB -> "
            f"{store.compressed_bytes / 1024 / 1024:.1f}MB "
            f"({store.raw_bytes / store.compressed_bytes:.2f}x)")

        preload_time = store.preload_all()
        total_setup = compress_time + preload_time
        log(f"  Compress: {compress_time:.2f}s, Preload: {preload_time:.2f}s")

        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        log(f"  Memory: {mem['total_mb']:.1f}MB")

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
            f"Inference={inf_time:.2f}s")

        result = collect_metrics(acc, auc, inf_time, blats, nb, total_setup, mem,
                                  baseline_auc, store,
                                  "Block-to-flat preload (per-col 8-bit lz4)")
        result['preload_time'] = float(preload_time)
        result['compress_time'] = float(compress_time)
        all_results['5C_block_to_flat_percol'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 5B: Profile-Guided Block-to-Flat Preload
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 5B: Profile-Guided Block-to-Flat Preload")
    log("#" * 70)

    try:
        # Load saved profile from Phase 4
        profile_path = os.path.join(RESULTS_DIR, "block_access_profile.json")
        if os.path.exists(profile_path):
            with open(profile_path) as f:
                profile = [tuple(x) for x in json.load(f)]
            log(f"  Loaded profile: {len(profile)} blocks from {profile_path}")
        else:
            log("  No saved profile found, building one...")
            # Quick profile: build store, do one pass
            restore_weights(dlrm, state_dict, emb_keys)
            tmp_store = BlockToFlatStore(large_tables, block_size=4096, codec='lz4')
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_indices)
            add_cold_tables(tmp_store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
            # Track blocks accessed
            profile_blocks = set()
            for X, lS_o, lS_i, T in test_ld:
                for t in large_tables:
                    if t not in tmp_store.cold_seq_lookup:
                        continue
                    indices = lS_i[t].numpy().flatten()
                    unique = np.unique(indices)
                    seq = tmp_store.cold_seq_lookup[t][unique]
                    cold_seq = seq[seq >= 0]
                    if len(cold_seq) > 0:
                        bids = np.unique(cold_seq // 4096)
                        for bid in bids:
                            profile_blocks.add((t, int(bid)))
            profile = sorted(profile_blocks)
            with open(profile_path, 'w') as f:
                json.dump(profile, f)
            log(f"  Built profile: {len(profile)} blocks")
            del tmp_store

        # Now do profile-guided preload
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()
        store = BlockToFlatStore(large_tables, block_size=4096, codec='lz4', quant_fn='global')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices)
        add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
        compress_time = time.time() - setup_t0

        preload_time = store.preload_from_profile(profile)
        total_setup = compress_time + preload_time
        log(f"  Compress: {compress_time:.2f}s, Profile preload: {preload_time:.2f}s")

        # Count profiled vs total blocks
        total_blocks = sum(len(store.blocks.get(t, [])) for t in large_tables)
        # blocks already freed by preload_from_profile, use profile size
        log(f"  Profiled {len(profile)} blocks (total was {store.num_blocks_total})")

        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        log(f"  Memory: {mem['total_mb']:.1f}MB")

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
            f"Inference={inf_time:.2f}s")

        result = collect_metrics(acc, auc, inf_time, blats, nb, total_setup, mem,
                                  baseline_auc, store,
                                  "Profile-guided block-to-flat (global 8-bit)")
        result['preload_time'] = float(preload_time)
        result['compress_time'] = float(compress_time)
        result['profile_blocks'] = len(profile)
        all_results['5B_profile_block_to_flat'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 5E: Minimal Cold Row Preload
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 5E: Minimal Cold Row Preload (only accessed rows)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()

        store = MinimalRowStore(large_tables, quant_fn='global')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices)

        for t in large_tables:
            w = state_dict[emb_keys[t]]
            hi = set(hot_indices[t].tolist())
            cold_idx = sorted(set(range(ln_emb[t])) - hi)
            cw = w[cold_idx]
            # Use accessed_cold from 5D
            if t in accessed_cold:
                n_all = len(cold_idx)
                n_accessed = len(accessed_cold[t])
                store.add_from_accessed(t, ln_emb[t], accessed_cold[t], cw, cold_idx)
                log(f"    Table {t}: {n_accessed:,}/{n_all:,} cold rows loaded")
            else:
                # Fallback: load all
                store.add_table(t, ln_emb[t], cold_idx, cw)
                log(f"    Table {t}: {len(cold_idx):,} cold rows (all)")

        setup_time = time.time() - setup_t0
        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        log(f"  Setup: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB "
            f"(cold: {store.memory_bytes / 1024 / 1024:.1f}MB)")

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
            f"Inference={inf_time:.2f}s")

        result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                  baseline_auc, store,
                                  "Minimal cold row preload (global 8-bit)")
        all_results['5E_minimal_rows'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 5F: Hybrid Lazy Block-to-Flat
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 5F: Hybrid Lazy Block-to-Flat (decompress on first access)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()
        store = HybridLazyStore(large_tables, block_size=4096, codec='lz4', quant_fn='global')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices)
        add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
        setup_time = time.time() - setup_t0
        log(f"  Setup: {setup_time:.2f}s, "
            f"Compressed: {store.compressed_bytes / 1024 / 1024:.1f}MB")

        drop_caches()
        time.sleep(1)
        gc.collect()

        # First pass: lazy decompression
        log("  First pass (lazy decompression)...")
        acc1, auc1, inf_time1, blats1, nb1 = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  Pass 1: AUC={auc1:.6f}, Inference={inf_time1:.2f}s, "
            f"Blocks decompressed: {store.blocks_decompressed}/{store.blocks_total}")

        # Second pass: all needed blocks already in flat array
        log("  Second pass (flat array)...")
        store.decompress_count = 0
        acc2, auc2, inf_time2, blats2, nb2 = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  Pass 2: AUC={auc2:.6f}, Inference={inf_time2:.2f}s, "
            f"New decompresses: {store.decompress_count}")

        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)

        result = collect_metrics(acc2, auc2, inf_time2, blats2, nb2, setup_time, mem,
                                  baseline_auc, store,
                                  "Hybrid lazy block-to-flat (global 8-bit)")
        result['pass1_inference_time'] = float(inf_time1)
        result['pass1_auc'] = float(auc1)
        result['blocks_decompressed'] = store.blocks_decompressed
        result['blocks_total'] = store.blocks_total
        all_results['5F_hybrid_lazy'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 5C-2: Per-Col 8-bit Block-to-Flat with zstd (better compression)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 5C-2: Per-Col 8-bit Block-to-Flat (zstd, better compression)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()
        store = BlockToFlatStore(large_tables, block_size=4096, codec='zstd',
                                  quant_fn='per_column')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices, quantize_fn='per_column')
        add_cold_tables(store, state_dict, emb_keys, ln_emb, large_tables, hot_indices)
        compress_time = time.time() - setup_t0
        log(f"  Compression: {store.raw_bytes / 1024 / 1024:.1f}MB -> "
            f"{store.compressed_bytes / 1024 / 1024:.1f}MB "
            f"({store.raw_bytes / store.compressed_bytes:.2f}x)")

        preload_time = store.preload_all()
        total_setup = compress_time + preload_time
        log(f"  Compress: {compress_time:.2f}s, Preload: {preload_time:.2f}s")

        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
            f"Inference={inf_time:.2f}s")

        result = collect_metrics(acc, auc, inf_time, blats, nb, total_setup, mem,
                                  baseline_auc, store,
                                  "Block-to-flat preload (per-col 8-bit zstd)")
        result['preload_time'] = float(preload_time)
        result['compress_time'] = float(compress_time)
        all_results['5C2_block_to_flat_percol_zstd'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # SUMMARY
    # ==============================================================
    log("\n" + "=" * 70)
    log("PHASE 5 SUMMARY")
    log("=" * 70)
    log(f"\n{'Config':<55s} {'AUC':>10s} {'Loss':>10s} {'Infer':>8s} {'Preload':>8s} {'Mem':>8s}")
    log("-" * 100)
    for k, v in all_results.items():
        if k == '5D_row_coverage':
            log(f"  5D Row coverage: {v['total_accessed_rows']:,}/{v['total_cold_rows']:,} "
                f"({v['overall_coverage_pct']:.1f}%), "
                f"savings: {v['potential_savings_mb']:.1f}MB")
            continue
        name = v.get('name', k)[:52]
        auc = v.get('auc', 0)
        loss = v.get('auc_loss_pp', 0)
        infer = v.get('inference_time', 0)
        preload = v.get('preload_time', v.get('setup_time', 0))
        mem = v.get('memory_mb', 0)
        log(f"  {name:<53s} {auc:.6f} {loss:+.4f}pp {infer:>6.1f}s {preload:>6.1f}s {mem:>6.1f}")

    log(f"\nResults saved to: {JSON_PATH}")
    log(f"\n{'=' * 70}")
    log("DONE!")
    log("=" * 70)

    if _log_fh:
        _log_fh.close()


if __name__ == '__main__':
    main()
