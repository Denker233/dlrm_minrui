#!/usr/bin/env python3
"""
True On-Demand H.265 Codec Benchmark for DLRM Embedding Tables.

Key changes from codec_prefetch_benchmark.py:
1. Proper video-resolution frames (1080p, 4K) instead of width=16
2. Per-frame compressed files for true on-demand decode (option 2)
3. NO pre-decode — each cache miss does disk read + H.265 decode + dequantize

Reuses Phases 1-2 (profiling, hot/cold split) and reordering from previous runs.
Only re-encodes with new frame geometry and runs Markov predictor experiments.
"""

import os, sys, time, json, threading, gc, subprocess, tempfile, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import psutil

# C++ extension for fast hot/cold embedding lookup
try:
    import compressed_emb as _C
    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import av

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
DATA_FILE = os.path.expanduser("~/input/train.txt")
PROCESSED_DATA = os.path.expanduser("~/input/kaggleAdDisplayChallenge_processed.npz")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 2048
EMB_DIM = 16
HOT_COVERAGE = 0.80
LARGE_TABLE_THRESHOLD = 50000
H265_CRF = 0  # lossless

# Resolution configs: (name, width, height, pixels_per_frame)
RESOLUTIONS = {
    '480p':  (640, 480),
    '1080p': (1920, 1080),
    '4K':    (3840, 2160),
}

RESULTS_DIR = "results"
PROFILING_DIR = os.path.join(RESULTS_DIR, "profiling")
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
ONDEMAND_DIR = os.path.join(RESULTS_DIR, "ondemand")
LOG_FILE = "logs/codec_ondemand_benchmark.log"

for d in [PROFILING_DIR, HOTCOLD_DIR, REORDER_DIR, ONDEMAND_DIR, "logs"]:
    os.makedirs(d, exist_ok=True)

log_fh = open(LOG_FILE, 'a')

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()

def get_rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def drop_caches():
    try:
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except Exception:
        pass


# ============================================================
# MODEL + DATA LOADING (same as before)
# ============================================================
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
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net
    args = create_args()
    log("Loading dataset...")
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
    log("Loading model checkpoint...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    log(f"Model loaded. {len(ln_emb)} tables, emb_dim={m_spa}")
    return dlrm, test_ld, train_ld, ln_emb

def quantize_table(w):
    """Global quantization: single scale/zero-point for entire table."""
    mn = w.min().item()
    mx = w.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


# ============================================================
# 4x4 TILING: Map each embedding (dim=16) to a 4x4 spatial tile
# ============================================================
TILE_H = 4
TILE_W = 4

def rows_to_tiled_frame(emb_rows, width, height):
    """
    Convert embedding rows (N, 16) into a tiled 2D frame (height, width).
    Each embedding becomes a 4x4 tile arranged in a grid.
    """
    tiles_per_row = width // TILE_W   # e.g., 1920/4 = 480
    tiles_per_col = height // TILE_H  # e.g., 1080/4 = 270
    rows_per_frame = tiles_per_row * tiles_per_col  # 129600

    # Reshape: (rows_per_frame, 16) -> (tiles_per_col, tiles_per_row, 4, 4)
    tiles = emb_rows[:rows_per_frame].reshape(tiles_per_col, tiles_per_row, TILE_H, TILE_W)
    # Rearrange to image: (tiles_per_col, 4, tiles_per_row, 4) -> (height, width)
    frame = tiles.transpose(0, 2, 1, 3).reshape(height, width)
    return frame

def tiled_frame_to_rows(frame, width, height):
    """
    Convert a tiled 2D frame (height, width) back to embedding rows (N, 16).
    Inverse of rows_to_tiled_frame.
    """
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col

    # Reshape: (height, width) -> (tiles_per_col, 4, tiles_per_row, 4)
    grid = frame.reshape(tiles_per_col, TILE_H, tiles_per_row, TILE_W)
    # Rearrange: (tiles_per_col, tiles_per_row, 4, 4) -> (rows_per_frame, 16)
    rows = grid.transpose(0, 2, 1, 3).reshape(rows_per_frame, EMB_DIM)
    return rows


# ============================================================
# NEW H.265 ENCODING: Proper video resolution frames with 4x4 tiling
# ============================================================
def encode_h265_perframe(q_np, width, height, crf=0, output_dir=None, table_id=0):
    """
    Encode uint8 cold embeddings as individual H.265 compressed frame files.

    Each embedding (dim=16) is reshaped into a 4x4 spatial tile.
    Tiles are arranged in a grid to fill video-resolution frames (e.g., 1920x1080).
    Each frame is encoded as a separate ALL-INTRA .h265 file.

    Returns: (num_frames, frame_dir, total_compressed_bytes, encode_time, rows_per_frame)
    """
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    num_rows = q_np.shape[0]
    num_frames = max(1, (num_rows + rows_per_frame - 1) // rows_per_frame)

    # Pad to fill complete frames
    padded_rows = num_frames * rows_per_frame
    padded = np.zeros((padded_rows, EMB_DIM), dtype=np.uint8)
    padded[:num_rows] = q_np

    frame_dir = os.path.join(output_dir, f'table_{table_id}')
    os.makedirs(frame_dir, exist_ok=True)

    t0 = time.time()
    total_compressed = 0

    for i in range(num_frames):
        # Get this frame's embedding rows
        frame_rows = padded[i * rows_per_frame:(i + 1) * rows_per_frame]
        # Tile into 2D frame
        frame_2d = rows_to_tiled_frame(frame_rows, width, height)
        frame_path = os.path.join(frame_dir, f'frame_{i:05d}.h265')

        # Encode single frame via ffmpeg
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', f'{width}x{height}',
            '-r', '1',
            '-i', 'pipe:0',
            '-c:v', 'libx265',
            '-preset', 'ultrafast',
            '-pix_fmt', 'gray',
            '-x265-params',
            f'keyint=1:min-keyint=1:{"lossless=1" if crf == 0 else f"crf={crf}"}:log-level=error',
            '-f', 'matroska',
            frame_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.stdin.write(frame_2d.tobytes())
        proc.stdin.close()
        proc.wait()

        total_compressed += os.path.getsize(frame_path)

    encode_time = time.time() - t0
    raw_bytes = num_rows * EMB_DIM
    ratio = raw_bytes / total_compressed if total_compressed > 0 else 0

    log(f"  H.265 encode ({width}x{height}, 4x4 tiles): {num_frames} frames, "
        f"{raw_bytes/1024/1024:.1f}MB -> {total_compressed/1024/1024:.1f}MB "
        f"({ratio:.2f}x uint8 ratio), {encode_time:.1f}s")

    return num_frames, frame_dir, total_compressed, encode_time, rows_per_frame


# ============================================================
# TRUE ON-DEMAND FRAME DECODER: Per-frame file, no pre-decode
# ============================================================
class OnDemandFrameDecoder:
    """
    Decode a single H.265 frame file on demand using PyAV.
    No pre-decoding — reads from disk and decodes each time.
    Thread-safe: each call is independent (opens its own file).
    """
    def __init__(self, frame_dir, rows_per_frame, emb_dim, width, height):
        self.frame_dir = frame_dir
        self.rows_per_frame = rows_per_frame
        self.emb_dim = emb_dim
        self.width = width
        self.height = height
        # Count frames
        self.num_frames = len([f for f in os.listdir(frame_dir)
                               if f.startswith('frame_') and f.endswith('.h265')])

    def decode_frame(self, frame_id):
        """
        Read and decode a single frame file from disk.
        Untiles 4x4 tiles back to embedding rows.
        Returns uint8 ndarray (rows_per_frame, emb_dim).
        Thread-safe: each call opens its own file/container.
        """
        frame_path = os.path.join(self.frame_dir, f'frame_{frame_id:05d}.h265')
        container = av.open(frame_path)
        frame = next(container.decode(video=0))
        arr = frame.to_ndarray(format='gray')  # (height, width)
        container.close()
        # Untile: 2D frame back to embedding rows
        rows = tiled_frame_to_rows(arr, self.width, self.height)
        return rows

    def close(self):
        pass


class InMemoryFrameDecoder:
    """
    Decode H.265 frames from in-memory compressed bytes using PyAV.
    Same interface as OnDemandFrameDecoder but reads from memory, not disk.
    """
    def __init__(self, frame_data_list, rows_per_frame, emb_dim, width, height):
        self.frame_data = frame_data_list  # list of bytes, one per frame
        self.rows_per_frame = rows_per_frame
        self.emb_dim = emb_dim
        self.width = width
        self.height = height
        self.num_frames = len(frame_data_list)

    def decode_frame(self, frame_id):
        """Decode a single frame from in-memory bytes. Returns uint8 (rows_per_frame, emb_dim)."""
        data = self.frame_data[frame_id]
        container = av.open(io.BytesIO(data))
        frame = next(container.decode(video=0))
        arr = frame.to_ndarray(format='gray')
        container.close()
        rows = tiled_frame_to_rows(arr, self.width, self.height)
        return rows

    def close(self):
        pass


# ============================================================
# MARKOV PREDICTOR (same as before)
# ============================================================
class MarkovPredictor:
    def __init__(self, num_frames, lookahead_depth=3):
        self.transition = defaultdict(Counter)
        self.last_frames = set()
        self.lookahead_depth = lookahead_depth
        self.ema_scores = np.zeros(num_frames)
        self.ema_alpha = 0.2
        self.warmup_batches = 0

    def observe(self, current_frames):
        current_set = set(current_frames)
        for prev in self.last_frames:
            for curr in current_set:
                self.transition[prev][curr] += 1
        self.last_frames = current_set
        self.warmup_batches += 1
        self.ema_scores *= (1 - self.ema_alpha)
        for fid in current_frames:
            if fid < len(self.ema_scores):
                self.ema_scores[fid] += self.ema_alpha

    def predict(self, current_frames, budget_frames=50):
        self.observe(current_frames)
        predictions = {}
        step1 = Counter()
        for fid in current_frames:
            for nxt, cnt in self.transition[fid].most_common(budget_frames):
                step1[nxt] += cnt
        if step1:
            mx = max(step1.values())
            for fid, s in step1.most_common(budget_frames):
                predictions[fid] = 1.0 * (s / mx)
        if self.lookahead_depth >= 2:
            step1_top = [f for f, _ in step1.most_common(20)]
            step2 = Counter()
            for fid in step1_top:
                for nxt, cnt in self.transition[fid].most_common(budget_frames):
                    step2[nxt] += cnt
            if step2:
                mx2 = max(step2.values())
                for fid, s in step2.most_common(budget_frames):
                    if fid not in predictions:
                        predictions[fid] = 0.5 * (s / mx2)
        if self.lookahead_depth >= 3:
            for fid in np.argsort(self.ema_scores)[-budget_frames:]:
                fid = int(fid)
                if fid not in predictions and self.ema_scores[fid] > 0.01:
                    predictions[fid] = 0.25 * self.ema_scores[fid]
        return predictions


# ============================================================
# GLOBAL LRU FRAME CACHE (shared across all tables)
# ============================================================
class GlobalFrameCache:
    """
    Shared LRU frame cache across all embedding tables.
    Key: (table_id, frame_id) -> decoded frame data (uint8 or fp32).
    Total capacity: N frames regardless of which table they belong to.
    """
    def __init__(self, capacity, store_uint8=True):
        self.capacity = capacity
        self.store_uint8 = store_uint8
        self.cache = OrderedDict()  # (table_id, frame_id) -> frame_data
        self.lock = threading.Lock()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}

    def get(self, table_id, frame_id):
        key = (table_id, frame_id)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.stats['hits'] += 1
                return self.cache[key]
            self.stats['misses'] += 1
            return None

    def put(self, table_id, frame_id, data):
        key = (table_id, frame_id)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = data
                return
            while len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)  # evict LRU
                self.stats['evictions'] += 1
            self.cache[key] = data

    def clear(self):
        with self.lock:
            self.cache.clear()

    def __len__(self):
        return len(self.cache)

    def report(self):
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        log(f"  GlobalCache: {len(self.cache)}/{self.capacity} frames, "
            f"{hit_rate:.1%} hit rate, {self.stats['evictions']} evictions")


# ============================================================
# TRUE ON-DEMAND PREFETCH FRAME CACHE
# ============================================================
class OnDemandPrefetchCache:
    """
    Bounded frame cache with true on-demand H.265 decode.
    Supports two modes:
    1. Per-table local cache (legacy, global_cache=None)
    2. Global shared cache (global_cache=GlobalFrameCache instance)

    When using global cache, frames are stored as uint8 to save 4x memory.
    Dequantization happens during gather (only for needed rows).
    """
    def __init__(self, frame_dir, rows_per_frame, emb_dim, num_cold_rows,
                 width, height, quant_scale, quant_zp,
                 cache_capacity=100, predictor=None, num_prefetch_workers=2,
                 global_cache=None, table_id=-1):
        self.frame_dir = frame_dir
        self.rows_per_frame = rows_per_frame
        self.emb_dim = emb_dim
        self.num_cold_rows = num_cold_rows
        self.num_frames = (num_cold_rows + rows_per_frame - 1) // rows_per_frame
        self.quant_scale = quant_scale
        self.quant_zp = quant_zp
        self.width = width
        self.height = height
        self.table_id = table_id

        # On-demand decoder (no pre-decode)
        self._decoder = OnDemandFrameDecoder(
            frame_dir, rows_per_frame, emb_dim, width, height)

        # Global vs local cache
        self.global_cache = global_cache
        self.use_global = global_cache is not None

        if not self.use_global:
            # Legacy per-table LRU cache: frame_id -> fp32 tensor
            self.cache = {}
            self.cache_priority = {}
            self.cache_capacity = cache_capacity
        else:
            # Global cache handles storage and eviction
            self.cache = None
            self.cache_priority = None
            self.cache_capacity = 0  # managed globally

        self.lock = threading.Lock()

        frame_bytes_fp32 = rows_per_frame * emb_dim * 4
        frame_bytes_uint8 = rows_per_frame * emb_dim
        if self.use_global:
            log(f"  Table {table_id}: using global cache "
                f"({self.num_frames} frames, {frame_bytes_uint8/1024:.1f}KB/frame uint8)")
        else:
            total_cache_bytes = cache_capacity * frame_bytes_fp32
            log(f"Cache: {cache_capacity} frames x {frame_bytes_fp32/1024:.1f}KB = "
                f"{total_cache_bytes/1024/1024:.1f}MB budget "
                f"({rows_per_frame} rows/frame, {self.num_frames} total frames)")

        # Prefetch workers
        self.executor = ThreadPoolExecutor(max_workers=num_prefetch_workers)
        self.pending = set()
        self.prefetch_lock = threading.Lock()
        self.predictor = predictor

        # Stats
        self.stats = {
            'cache_hits': 0, 'cache_misses': 0,
            'prefetch_hits': 0, 'demand_decomps': 0,
            'prefetch_submissions': 0, 'prefetch_wastes': 0,
            'total_demand_ms': 0.0, 'total_prefetch_ms': 0.0,
            'batch_demand_decomps': [], 'batch_hit_rates': [],
        }

    def _decode_raw(self, frame_id):
        """Decode H.265 frame, return uint8 ndarray (actual_rows, emb_dim)."""
        q_uint8 = self._decoder.decode_frame(frame_id)
        row_start = frame_id * self.rows_per_frame
        row_end = min(row_start + self.rows_per_frame, self.num_cold_rows)
        actual_rows = row_end - row_start
        return q_uint8[:actual_rows]

    def _decode_and_dequant(self, frame_id):
        """True on-demand: read H.265 file from disk, decode, dequantize to fp32."""
        q_slice = self._decode_raw(frame_id)
        fp32 = (q_slice.astype(np.float32) - self.quant_zp) * self.quant_scale
        return torch.from_numpy(fp32)

    def _dequant_rows(self, uint8_data, row_offsets):
        """Dequantize specific rows from uint8 frame data to fp32 tensor.
        Uses C++ extension when available for ~2x speedup."""
        if HAS_CPP_EXT and isinstance(uint8_data, np.ndarray):
            uint8_t = torch.from_numpy(uint8_data)
            offsets_t = torch.from_numpy(row_offsets).long() if isinstance(row_offsets, np.ndarray) else row_offsets.long()
            return _C.gather_dequant_uint8(uint8_t, offsets_t, self.quant_scale, self.quant_zp)
        selected = uint8_data[row_offsets]
        fp32 = (selected.astype(np.float32) - self.quant_zp) * self.quant_scale
        return torch.from_numpy(fp32)

    def _prefetch_worker(self, frame_id, priority):
        t0 = time.time()
        if self.use_global:
            frame_data = self._decode_raw(frame_id)
            self.global_cache.put(self.table_id, frame_id, frame_data)
        else:
            frame_data = self._decode_and_dequant(frame_id)
            with self.lock:
                if frame_id not in self.cache:
                    self.cache[frame_id] = frame_data
                    self.cache_priority[frame_id] = priority
                    self._evict_if_needed()
        elapsed = (time.time() - t0) * 1000
        self.stats['total_prefetch_ms'] += elapsed
        with self.prefetch_lock:
            self.pending.discard(frame_id)

    def _evict_if_needed(self):
        if self.use_global:
            return  # global cache handles eviction
        while len(self.cache) > self.cache_capacity:
            evict_id = min(self.cache_priority, key=self.cache_priority.get)
            del self.cache[evict_id]
            del self.cache_priority[evict_id]
            self.stats['prefetch_wastes'] += 1

    def launch_prefetch(self, current_batch_frames):
        if self.predictor is None:
            return
        cap = self.global_cache.capacity if self.use_global else self.cache_capacity
        predictions = self.predictor.predict(
            current_batch_frames, budget_frames=cap)
        if self.use_global:
            cached_set = set()
            for key in list(self.global_cache.cache.keys()):
                if key[0] == self.table_id:
                    cached_set.add(key[1])
            cur_size = len(self.global_cache)
        else:
            with self.lock:
                cached_set = set(self.cache.keys())
                cur_size = len(self.cache)
        with self.prefetch_lock:
            in_flight = self.pending.copy()
        candidates = {fid: pri for fid, pri in predictions.items()
                      if fid not in cached_set and fid not in in_flight}
        sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])
        budget = max(0, int(cap * 0.7) - cur_size - len(in_flight))
        submitted = 0
        for fid, pri in sorted_cands:
            if submitted >= budget:
                break
            with self.prefetch_lock:
                self.pending.add(fid)
            self.executor.submit(self._prefetch_worker, fid, pri)
            submitted += 1
        self.stats['prefetch_submissions'] += submitted

    def lookup(self, cold_indices_reordered):
        if len(cold_indices_reordered) == 0:
            return torch.zeros(0, self.emb_dim), set()

        sorted_order = torch.argsort(cold_indices_reordered)
        sorted_indices = cold_indices_reordered[sorted_order]

        frame_ids = (sorted_indices // self.rows_per_frame).long()
        unique_frames = torch.unique(frame_ids).tolist()

        batch_hits = 0
        batch_misses = 0

        # Ensure all needed frames are in cache (global or local)
        for fid in unique_frames:
            if self.use_global:
                cached = self.global_cache.get(self.table_id, fid)
                if cached is not None:
                    self.stats['cache_hits'] += 1
                    batch_hits += 1
                    continue
            else:
                with self.lock:
                    if fid in self.cache:
                        self.cache_priority[fid] = 2.0
                        self.stats['cache_hits'] += 1
                        batch_hits += 1
                        continue

            # Wait for in-flight prefetch
            for _ in range(10):
                with self.prefetch_lock:
                    if fid not in self.pending:
                        break
                time.sleep(0.0005)

            if self.use_global:
                cached = self.global_cache.get(self.table_id, fid)
                if cached is not None:
                    self.stats['prefetch_hits'] += 1
                    batch_hits += 1
                    continue
            else:
                with self.lock:
                    if fid in self.cache:
                        self.stats['prefetch_hits'] += 1
                        self.cache_priority[fid] = 2.0
                        batch_hits += 1
                        continue

            # DEMAND DECODE
            t0 = time.time()
            if self.use_global:
                frame_data = self._decode_raw(fid)  # uint8
                self.global_cache.put(self.table_id, fid, frame_data)
            else:
                frame_data = self._decode_and_dequant(fid)  # fp32
                with self.lock:
                    self.cache[fid] = frame_data
                    self.cache_priority[fid] = 2.0
                    self._evict_if_needed()
            elapsed = (time.time() - t0) * 1000
            self.stats['demand_decomps'] += 1
            self.stats['total_demand_ms'] += elapsed
            self.stats['cache_misses'] += 1
            batch_misses += 1

        # Gather embeddings from cached frames
        if self.use_global:
            # uint8 path: gather from uint8 frames, dequantize only needed rows
            results = torch.zeros(len(sorted_indices), self.emb_dim)
            for fid in unique_frames:
                mask = (frame_ids == fid)
                offsets_in_frame = (sorted_indices[mask] % self.rows_per_frame).numpy()
                frame_data = self.global_cache.get(self.table_id, fid)
                if frame_data is None:
                    # Fallback: decode again (shouldn't happen normally)
                    frame_data = self._decode_raw(fid)
                    self.stats['demand_decomps'] += 1
                actual_rows = frame_data.shape[0]
                safe_offsets = np.clip(offsets_in_frame, 0, actual_rows - 1)
                results[mask] = self._dequant_rows(frame_data, safe_offsets)
        else:
            # fp32 path: gather from pre-dequantized frames
            frame_data_copies = {}
            with self.lock:
                for fid in unique_frames:
                    if fid in self.cache:
                        frame_data_copies[fid] = self.cache[fid]
            for fid in unique_frames:
                if fid not in frame_data_copies:
                    frame_data_copies[fid] = self._decode_and_dequant(fid)
                    self.stats['demand_decomps'] += 1

            results = torch.zeros(len(sorted_indices), self.emb_dim)
            for fid in unique_frames:
                mask = (frame_ids == fid)
                offsets_in_frame = (sorted_indices[mask] % self.rows_per_frame).long()
                frame_data = frame_data_copies[fid]
                actual_rows = frame_data.shape[0]
                safe_offsets = torch.clamp(offsets_in_frame, max=actual_rows - 1)
                results[mask] = frame_data[safe_offsets]

        final = torch.zeros_like(results)
        final[sorted_order] = results

        total_batch = batch_hits + batch_misses
        self.stats['batch_demand_decomps'].append(batch_misses)
        self.stats['batch_hit_rates'].append(
            batch_hits / total_batch if total_batch > 0 else 1.0)

        return final, set(unique_frames)

    def clear_cache(self):
        if self.use_global:
            # Only clear this table's entries from global cache
            keys_to_remove = [k for k in self.global_cache.cache
                              if k[0] == self.table_id]
            for k in keys_to_remove:
                del self.global_cache.cache[k]
        else:
            with self.lock:
                self.cache.clear()
                self.cache_priority.clear()

    def reset_stats(self):
        self.stats = {
            'cache_hits': 0, 'cache_misses': 0,
            'prefetch_hits': 0, 'demand_decomps': 0,
            'prefetch_submissions': 0, 'prefetch_wastes': 0,
            'total_demand_ms': 0.0, 'total_prefetch_ms': 0.0,
            'batch_demand_decomps': [], 'batch_hit_rates': [],
        }

    def report_stats(self):
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total if total else 0
        avg_demand = (np.mean(self.stats['batch_demand_decomps'])
                      if self.stats['batch_demand_decomps'] else 0)
        cache_size = len(self.global_cache) if self.use_global else len(self.cache)
        cache_cap = self.global_cache.capacity if self.use_global else self.cache_capacity
        log(f"  Table {self.table_id} Cache: {hit_rate:.1%} hit | "
            f"demand={self.stats['demand_decomps']} ({avg_demand:.1f}/batch) | "
            f"prefetch_hits={self.stats['prefetch_hits']} | "
            f"demand_time={self.stats['total_demand_ms']:.0f}ms")
        return {
            'hit_rate': hit_rate,
            'demand_decomps': self.stats['demand_decomps'],
            'avg_demand_per_batch': avg_demand,
            'prefetch_hits': self.stats['prefetch_hits'],
            'demand_time_ms': self.stats['total_demand_ms'],
        }

    def close(self):
        self.executor.shutdown(wait=False)
        self._decoder.close()


# ============================================================
# COMPRESSED EMBEDDING BAG: Hot fp32 + Cold H.265 in memory
# ============================================================
class CompressedEmbeddingBag(nn.Module):
    """
    Drop-in replacement for nn.EmbeddingBag that stores:
    - Hot rows: uncompressed fp32 OR quantized uint8 in a compact tensor
    - Cold rows: H.265 compressed in memory, decoded on demand

    Uses merged int32 mapping to save ~418MB vs separate is_hot/orig_to_hot/o2c tensors.
    Mapping encoding: >=0 = hot index, <0 = cold index (-(val+1)), INT32_MIN = invalid
    """
    def __init__(self, hot_weight, is_hot, orig_to_hot, orig_to_cold_reordered,
                 cold_cache, num_embeddings, embedding_dim,
                 quantize_hot=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.cold_cache = cold_cache
        self.last_frames_used = set()
        self.mode = 'sum'
        self.quantize_hot = quantize_hot

        # Build merged int32 mapping: saves ~418MB for 8 large tables
        # vs storing separate is_hot (bool), orig_to_hot (int64), o2c (int64)
        INT32_MIN = -2147483648  # torch.iinfo(torch.int32).min
        mapping = torch.full((num_embeddings,), INT32_MIN, dtype=torch.int32)
        hot_mask = is_hot.bool()
        # Hot indices: mapping[i] = hot_compact_idx (>= 0)
        mapping[hot_mask] = orig_to_hot[hot_mask].to(torch.int32)
        # Cold indices: mapping[i] = -(cold_reordered_idx + 1) (< 0, != INT32_MIN)
        cold_mask = orig_to_cold_reordered >= 0
        mapping[cold_mask] = (-(orig_to_cold_reordered[cold_mask] + 1)).to(torch.int32)
        self.mapping = mapping

        # Keep legacy tensors only if no C++ merged support
        self._has_merged = HAS_CPP_EXT and hasattr(_C, 'compressed_emb_bag_forward_merged')
        if not self._has_merged:
            self.is_hot = is_hot
            self.orig_to_hot = orig_to_hot
            self.o2c = orig_to_cold_reordered
        else:
            # Free the large legacy tensors
            self.is_hot = None
            self.orig_to_hot = None
            self.o2c = None

        if quantize_hot:
            mn = hot_weight.min().item()
            mx = hot_weight.max().item()
            s = (mx - mn) / 255.0
            if s == 0:
                s = 1.0
            zp = round(-mn / s)
            self.hot_weight_q8 = ((hot_weight / s).round() + zp).clamp(0, 255).to(torch.uint8)
            self.hot_scale = s
            self.hot_zp = zp
            self.hot_weight = None
        else:
            self.hot_weight = hot_weight
            self.hot_weight_q8 = None
            self.hot_scale = 0.0
            self.hot_zp = 0

        self._empty_psw = torch.empty(0)

    def forward(self, indices, offsets, per_sample_weights=None):
        if HAS_CPP_EXT:
            return self._forward_cpp(indices, offsets, per_sample_weights)
        return self._forward_python(indices, offsets, per_sample_weights)

    def _forward_cpp(self, indices, offsets, per_sample_weights=None):
        psw = per_sample_weights if per_sample_weights is not None else self._empty_psw

        if self._has_merged:
            if self.quantize_hot:
                output, cold_mask, cold_count = _C.compressed_emb_bag_forward_q8_merged(
                    indices, offsets, self.hot_weight_q8, self.mapping,
                    psw, self.hot_scale, self.hot_zp)
            else:
                output, cold_mask, cold_count = _C.compressed_emb_bag_forward_merged(
                    indices, offsets, self.hot_weight, self.mapping, psw)
        else:
            if self.quantize_hot:
                output, cold_mask, cold_count = _C.compressed_emb_bag_forward_q8(
                    indices, offsets, self.hot_weight_q8, self.is_hot,
                    self.orig_to_hot, psw, self.hot_scale, self.hot_zp)
            else:
                output, cold_mask, cold_count = _C.compressed_emb_bag_forward(
                    indices, offsets, self.hot_weight, self.is_hot,
                    self.orig_to_hot, psw)

        if cold_count.item() > 0:
            cold_positions = torch.where(cold_mask)[0]
            cold_orig = indices[cold_positions]
            # Extract cold reordered indices from merged mapping
            cold_map_vals = self.mapping[cold_orig]
            cold_reordered = -(cold_map_vals.long() + 1)
            valid = cold_reordered >= 0
            if valid.any():
                cold_result, frames_used = self.cold_cache.lookup(cold_reordered[valid])
                valid_positions = cold_positions[valid]
                _C.cold_fixup(output, indices, offsets, cold_mask,
                              cold_result, valid_positions, psw)
                self.last_frames_used = frames_used
            else:
                self.last_frames_used = set()
        else:
            self.last_frames_used = set()

        return output

    def _forward_python(self, indices, offsets, per_sample_weights=None):
        map_vals = self.mapping[indices]
        hot_mask = map_vals >= 0
        all_embeds = torch.zeros(len(indices), self.embedding_dim)

        # Hot lookups from compact tensor
        if hot_mask.any():
            hot_compact_idx = map_vals[hot_mask].long()
            if self.quantize_hot:
                q = self.hot_weight_q8[hot_compact_idx]
                all_embeds[hot_mask] = (q.float() - self.hot_zp) * self.hot_scale
            else:
                all_embeds[hot_mask] = self.hot_weight[hot_compact_idx]

        # Cold lookups via on-demand H.265 decode
        INT32_MIN = -2147483648
        cold_mask = (map_vals < 0) & (map_vals != INT32_MIN)
        if cold_mask.any():
            cold_map = map_vals[cold_mask]
            cold_reordered = -(cold_map.long() + 1)
            valid = cold_reordered >= 0
            if valid.any():
                cold_result, frames_used = self.cold_cache.lookup(cold_reordered[valid])
                cold_positions = torch.where(cold_mask)[0]
                all_embeds[cold_positions[valid]] = cold_result
                self.last_frames_used = frames_used
            else:
                self.last_frames_used = set()
        else:
            self.last_frames_used = set()

        if per_sample_weights is not None:
            all_embeds = all_embeds * per_sample_weights.unsqueeze(1)

        # Sum pooling per bag using offsets (vectorized)
        num_bags = len(offsets)
        output = torch.zeros(num_bags, self.embedding_dim)
        bag_ids = torch.bucketize(torch.arange(len(indices)), offsets, right=True) - 1
        bag_ids = bag_ids.clamp(min=0)
        output.scatter_add_(0, bag_ids.unsqueeze(1).expand_as(all_embeds), all_embeds)

        return output


# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("TRUE ON-DEMAND H.265 CODEC BENCHMARK")
    log("Per-frame files, proper video resolution, no pre-decode")
    log("=" * 70)
    log(f"RSS at start: {get_rss_mb():.0f}MB")

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")
    log(f"C++ extension: {'ENABLED' if HAS_CPP_EXT else 'DISABLED (falling back to Python)'}")

    total_emb_mb = sum(state_dict[k].numel() * 4 for k in emb_keys) / 1024 / 1024
    log(f"Total embedding memory: {total_emb_mb:.1f}MB")

    # ---- Load existing Phase 1-2 results ----
    log("\nLoading existing profiling and hot/cold data...")
    is_hot = {}
    hot_indices = {}
    cold_indices = {}
    orig_to_cold = {}
    for t in large_tables:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]
        cold_indices[t] = torch.where(~is_hot[t])[0]
        orig_to_cold[t] = torch.load(os.path.join(HOTCOLD_DIR, f'orig_to_cold_{t}.pt'),
                                      map_location='cpu', weights_only=True)

    # Load reordering
    orig_to_cold_reordered = {}
    cold_quant_scale = {}
    cold_quant_zp = {}
    cold_num_rows = {}

    for t in large_tables:
        fp = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        orig_to_cold_reordered[t] = torch.load(fp, map_location='cpu', weights_only=True)
        n_cold_path = os.path.join(REORDER_DIR, f'num_cold_{t}.txt')
        with open(n_cold_path) as f:
            cold_num_rows[t] = int(f.read().strip())

    log("Loaded profiling, hot/cold, and reorder data.")

    # ---- Phase 3b: Re-encode with proper video resolution ----
    for res_name, (width, height) in RESOLUTIONS.items():
        res_dir = os.path.join(ONDEMAND_DIR, res_name)
        done_marker = os.path.join(res_dir, '.done')

        if os.path.exists(done_marker):
            log(f"\nEncoding for {res_name} SKIPPED — already done")
            # Load quant params from saved metadata
            for t in large_tables:
                if t not in cold_quant_scale:
                    meta_path = os.path.join(res_dir, f'table_{t}', 'meta.json')
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            meta = json.load(f)
                        cold_quant_scale[t] = meta['quant_scale']
                        cold_quant_zp[t] = meta['quant_zp']
            continue

        log(f"\n{'='*70}")
        log(f"ENCODING: {res_name} ({width}x{height})")
        log(f"{'='*70}")

        os.makedirs(res_dir, exist_ok=True)
        pixels_per_frame = width * height
        rows_per_frame = pixels_per_frame // EMB_DIM

        total_compressed = 0
        total_raw = 0

        for t in large_tables:
            n_cold = cold_num_rows[t]
            if n_cold == 0:
                continue

            log(f"\n  Table {t}: {n_cold:,} cold embeddings")

            # Load reordered cold weights and quantize (global)
            cold_order = np.load(os.path.join(REORDER_DIR, f'cold_order_{t}.npy'))
            cold_w = state_dict[emb_keys[t]][cold_indices[t]]
            reordered_w = cold_w[cold_order]
            q, s, zp = quantize_table(reordered_w)
            cold_quant_scale[t] = s
            cold_quant_zp[t] = zp

            num_frames, frame_dir, compressed_bytes, enc_time, rpf = \
                encode_h265_perframe(q.numpy(), width, height, crf=H265_CRF,
                                     output_dir=res_dir, table_id=t)

            # Save metadata
            meta = {
                'num_frames': num_frames,
                'rows_per_frame': rpf,
                'width': width,
                'height': height,
                'n_cold': n_cold,
                'compressed_bytes': compressed_bytes,
                'raw_bytes': n_cold * EMB_DIM,
                'quant_scale': s,
                'quant_zp': zp,
            }
            with open(os.path.join(frame_dir, 'meta.json'), 'w') as f:
                json.dump(meta, f)

            total_compressed += compressed_bytes
            total_raw += n_cold * EMB_DIM

            del cold_w, reordered_w, q
            gc.collect()

        ratio = total_raw / total_compressed if total_compressed > 0 else 0
        log(f"\n  {res_name} total: {total_raw/1024/1024:.1f}MB uint8 -> "
            f"{total_compressed/1024/1024:.1f}MB compressed ({ratio:.2f}x uint8 ratio)")

        with open(done_marker, 'w') as f:
            f.write(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ---- Phase 4: Inference benchmarks ----
    log(f"\n{'='*70}")
    log("INFERENCE BENCHMARKS — True On-Demand Decode")
    log(f"{'='*70}")

    def restore_weights():
        with torch.no_grad():
            for k in emb_keys:
                t_idx = int(k.split('.')[1])
                dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

    # --- Baseline ---
    log("\n--- Baseline: Full fp32 ---")
    restore_weights()
    gc.collect()
    drop_caches()
    time.sleep(0.5)
    scores, targets, blats = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            scores.extend(Z.detach().cpu().numpy().flatten().tolist())
            targets.extend(T.detach().cpu().numpy().flatten().tolist())
            if batch_idx % 500 == 0:
                log(f"    Batch {batch_idx}")
    baseline_time = time.time() - t0
    baseline_auc = roc_auc_score(targets, scores)
    baseline_rss = get_rss_mb()
    log(f"  AUC={baseline_auc:.6f}, Time={baseline_time:.2f}s, RSS={baseline_rss:.0f}MB")
    log(f"  Batch latency: mean={np.mean(blats)*1000:.2f}ms, "
        f"p50={np.percentile(blats,50)*1000:.2f}ms, p99={np.percentile(blats,99)*1000:.2f}ms")

    all_results = {
        'A_baseline': {
            'auc': baseline_auc, 'total_time': baseline_time,
            'num_batches': len(blats), 'rss_mb': baseline_rss,
            'mean_lat_ms': np.mean(blats)*1000,
            'p50_lat_ms': np.percentile(blats,50)*1000,
            'p99_lat_ms': np.percentile(blats,99)*1000,
            'emb_memory_mb': total_emb_mb,
        }
    }

    # --- Run codec experiments ---
    # Save original EmbeddingBag modules so we can restore between experiments
    original_emb_modules = {}
    for t in large_tables:
        original_emb_modules[t] = dlrm.emb_l[t]

    o2c_map = orig_to_cold_reordered

    def run_ondemand_inference(res_name, cache_capacity, predictor_type='markov',
                               lookahead_depth=3, tag="",
                               use_global_cache=False, disk_decode=False,
                               quantize_hot=False, use_hash_table=False,
                               use_bitmap=False, warmup_batches=0):
        width, height = RESOLUTIONS[res_name]
        pixels_per_frame = width * height
        rows_per_frame = pixels_per_frame // EMB_DIM
        res_dir = os.path.join(ONDEMAND_DIR, res_name)

        mode_str = "GLOBAL" if use_global_cache else "per-table"
        decode_str = "disk" if disk_decode else "in-memory"
        log(f"\n--- {tag}: {res_name}, cache={cache_capacity} ({mode_str}), "
            f"decode={decode_str}, predictor={predictor_type} ---")

        # Restore original EmbeddingBag modules for small tables
        restore_weights()

        # Create global cache if requested
        global_cache = None
        if use_global_cache:
            global_cache = GlobalFrameCache(capacity=cache_capacity, store_uint8=True)
            log(f"  Global cache: {cache_capacity} frames total across all tables "
                f"(uint8, ~{cache_capacity * rows_per_frame * EMB_DIM / 1024 / 1024:.1f}MB)")

        # Build CompressedEmbeddingBag for each large table
        caches = {}
        total_compressed_bytes = 0
        total_hot_mb = 0
        total_mapping_mb = 0

        for t_idx in large_tables:
            n_cold = cold_num_rows.get(t_idx, 0)
            if n_cold == 0:
                continue
            frame_dir = os.path.join(res_dir, f'table_{t_idx}')
            if not os.path.exists(frame_dir):
                log(f"  WARNING: No frame dir for table {t_idx}")
                continue

            frame_files = sorted([f for f in os.listdir(frame_dir)
                                  if f.startswith('frame_') and f.endswith('.h265')])

            if disk_decode:
                # Disk-based: don't load compressed bytes into memory
                # Just count sizes for reporting
                comp_bytes = sum(os.path.getsize(os.path.join(frame_dir, ff))
                                 for ff in frame_files)
                total_compressed_bytes += comp_bytes
                # Use disk-based decoder (already the default in OnDemandPrefetchCache)
                decoder = None  # will use the default OnDemandFrameDecoder
            else:
                # In-memory: load all compressed frame files into memory
                frame_data_list = []
                for ff in frame_files:
                    with open(os.path.join(frame_dir, ff), 'rb') as f:
                        frame_data_list.append(f.read())
                comp_bytes = sum(len(d) for d in frame_data_list)
                total_compressed_bytes += comp_bytes
                decoder = InMemoryFrameDecoder(
                    frame_data_list, rows_per_frame, EMB_DIM, width, height)

            num_frames_t = len(frame_files)
            pred = None
            if predictor_type == 'markov':
                pred = MarkovPredictor(num_frames_t, lookahead_depth=lookahead_depth)

            cache = OnDemandPrefetchCache(
                frame_dir=frame_dir,
                rows_per_frame=rows_per_frame,
                emb_dim=EMB_DIM,
                num_cold_rows=n_cold,
                width=width, height=height,
                quant_scale=cold_quant_scale[t_idx],
                quant_zp=cold_quant_zp[t_idx],
                cache_capacity=cache_capacity,
                predictor=pred,
                num_prefetch_workers=2,
                global_cache=global_cache,
                table_id=t_idx,
            )
            # Replace decoder if using in-memory mode
            if decoder is not None:
                cache._decoder = decoder
            caches[t_idx] = cache

            # Build compact hot embedding
            w = state_dict[emb_keys[t_idx]]
            h_idx = hot_indices[t_idx]
            hot_weight = w[h_idx].clone()
            if quantize_hot:
                total_hot_mb += hot_weight.numel() / 1024 / 1024  # uint8 = 1 byte
            else:
                total_hot_mb += hot_weight.numel() * 4 / 1024 / 1024  # fp32 = 4 bytes

            # Build orig-to-hot mapping
            orig_to_hot = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
            orig_to_hot[h_idx] = torch.arange(len(h_idx))

            # Replace nn.EmbeddingBag with CompressedEmbeddingBag
            comp_emb = CompressedEmbeddingBag(
                hot_weight=hot_weight,
                is_hot=is_hot[t_idx],
                orig_to_hot=orig_to_hot,
                orig_to_cold_reordered=o2c_map[t_idx],
                cold_cache=cache,
                num_embeddings=ln_emb[t_idx],
                embedding_dim=EMB_DIM,
                quantize_hot=quantize_hot,
            )
            # Track mapping memory: int32 = 4 bytes/row
            total_mapping_mb += ln_emb[t_idx] * 4 / 1024 / 1024
            dlrm.emb_l[t_idx] = comp_emb

        # Free full fp32 weights to show true memory savings:
        # 1. Free state_dict entries for large tables
        freed_state = {}
        for t_idx in large_tables:
            k = emb_keys[t_idx]
            if k in state_dict:
                freed_state[k] = state_dict.pop(k)
        # 2. Free the saved original EmbeddingBag weight data (shrink to 1-row placeholder)
        for t_idx in large_tables:
            if t_idx in original_emb_modules:
                original_emb_modules[t_idx].weight = nn.Parameter(
                    torch.zeros(1, EMB_DIM), requires_grad=False)
        gc.collect()

        total_compressed_mb = total_compressed_bytes / 1024 / 1024
        compressed_in_mem_mb = 0.0 if disk_decode else total_compressed_mb
        rss_after_setup = get_rss_mb()
        log(f"  Memory: hot={total_hot_mb:.1f}MB + compressed_cold="
            f"{'0 (on disk)' if disk_decode else f'{total_compressed_mb:.1f}MB'} "
            f"= {total_hot_mb + compressed_in_mem_mb:.1f}MB "
            f"(vs {total_emb_mb:.1f}MB full fp32)")
        log(f"  RSS after setup: {rss_after_setup:.0f}MB")

        # Warmup Markov
        if predictor_type == 'markov':
            log(f"    Warming up Markov predictor (500 batches)...")
            for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
                if batch_idx >= 500:
                    break
                for t_idx in caches:
                    indices = lS_i[t_idx]
                    cold_mask = ~is_hot[t_idx][indices]
                    if cold_mask.any():
                        cold_mapped = o2c_map[t_idx][indices[cold_mask]]
                        valid = cold_mapped >= 0
                        if valid.any():
                            fids = (cold_mapped[valid] // rows_per_frame).unique().tolist()
                            caches[t_idx].predictor.observe(fids)
            for t_idx in caches:
                caches[t_idx].reset_stats()
                caches[t_idx].clear_cache()

        # Fast apply_emb: register all tables in C++ then use fast_forward (single call)
        # This eliminates the Python for-loop entirely — zero Python overhead per batch.
        compressed_table_ids = set(caches.keys())
        _orig_apply_emb = dlrm.apply_emb

        if HAS_CPP_EXT and compressed_table_ids:
            num_tabs = len(dlrm.emb_l)
            # Register tables in C++ for fast_forward
            table_kinds = []
            weights = []
            mappings = []
            scales = []
            zero_points = []
            _cold_caches = {}  # table_idx -> cold_cache (for cold fixup in Python)
            _mappings = {}     # table_idx -> mapping tensor (for cold fixup)
            _empty_psw = {}    # table_idx -> empty per_sample_weights
            _empty_set = set()

            for k in range(num_tabs):
                E = dlrm.emb_l[k]
                if k in compressed_table_ids and isinstance(E, CompressedEmbeddingBag):
                    if E.quantize_hot:
                        table_kinds.append(2)  # COMPRESSED_Q8
                        weights.append(E.hot_weight_q8)
                        mappings.append(E.mapping)
                        scales.append(float(E.hot_scale))
                        zero_points.append(int(E.hot_zp))
                    else:
                        table_kinds.append(1)  # COMPRESSED_FP32
                        weights.append(E.hot_weight)
                        mappings.append(E.mapping)
                        scales.append(0.0)
                        zero_points.append(0)
                    _cold_caches[k] = E.cold_cache
                    _mappings[k] = E.mapping
                    _empty_psw[k] = E._empty_psw
                else:
                    table_kinds.append(0)  # STANDARD
                    weights.append(E.weight)
                    mappings.append(torch.empty(0, dtype=torch.int32))  # placeholder
                    scales.append(0.0)
                    zero_points.append(0)

            # Use hash table or bitmap-rank to replace mapping tensors
            use_hash = use_hash_table and not use_bitmap
            _C.register_tables(table_kinds, weights, mappings, scales, zero_points,
                               use_hash_table=use_hash, use_bitmap=use_bitmap)
            if use_hash or use_bitmap:
                # C++ side releases mapping tensors.
                # For cold fixup, we need orig_to_cold_reordered.
                # Load as numpy memory-mapped arrays (only accessed pages enter RAM).
                _cold_reordered_mmap = {}
                for k in _cold_caches:
                    E = dlrm.emb_l[k]
                    mmap_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{k}.npy')
                    if not os.path.exists(mmap_path):
                        pt_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{k}.pt')
                        o2c = torch.load(pt_path, map_location='cpu', weights_only=True)
                        np.save(mmap_path, o2c.numpy())
                        del o2c
                    _cold_reordered_mmap[k] = np.load(mmap_path, mmap_mode='r')
                    if hasattr(E, 'mapping'):
                        del E.mapping
                del mappings
                _mappings.clear()
                gc.collect()
                mode_name = "Bitmap-rank" if use_bitmap else "Hash table"
                log(f"  {mode_name} mode: mapping tensors released, cold lookup via mmap")

            def _fast_apply_emb(lS_o, lS_i, emb_l, v_W_l):
                # Ensure lS_i and lS_o are 2D contiguous tensors
                if isinstance(lS_i, (list, tuple)):
                    lS_i_2d = torch.stack(lS_i)
                elif lS_i.dim() == 2:
                    lS_i_2d = lS_i
                else:
                    lS_i_2d = lS_i.view(num_tabs, -1)
                if isinstance(lS_o, (list, tuple)):
                    lS_o_2d = torch.stack(lS_o)
                elif lS_o.dim() == 2:
                    lS_o_2d = lS_o
                else:
                    lS_o_2d = lS_o.view(num_tabs, -1)

                # Single C++ call processes all 26 tables
                results = _C.fast_forward(lS_i_2d, lS_o_2d)
                # Unpack: [outputs..., cold_masks..., cold_counts...]
                outputs = results[:num_tabs]
                cold_masks = results[num_tabs:2*num_tabs]
                cold_counts = results[2*num_tabs:]

                # Handle cold fixup for compressed tables (rare: ~0.1% of batches)
                for k in _cold_caches:
                    cc = cold_counts[k].item()
                    if cc > 0:
                        cold_mask = cold_masks[k]
                        idx = lS_i_2d[k]
                        off = lS_o_2d[k]
                        cold_positions = torch.where(cold_mask)[0]
                        cold_orig_indices = idx[cold_positions].numpy()
                        if (use_hash or use_bitmap) and k in _cold_reordered_mmap:
                            # Hash mode: look up cold indices from mmap'd file
                            cold_reordered_np = _cold_reordered_mmap[k][cold_orig_indices]
                            cold_reordered = torch.from_numpy(cold_reordered_np.copy()).long()
                        else:
                            # Array mode: extract from merged mapping
                            cold_map_vals = _mappings[k][idx[cold_positions]]
                            cold_reordered = -(cold_map_vals.long() + 1)
                        valid = cold_reordered >= 0
                        if valid.any():
                            cold_result, frames_used = _cold_caches[k].lookup(cold_reordered[valid])
                            _C.cold_fixup(outputs[k], idx, off, cold_mask,
                                          cold_result, cold_positions[valid], _empty_psw[k])
                            emb_l[k].last_frames_used = frames_used
                        else:
                            emb_l[k].last_frames_used = _empty_set
                    else:
                        emb_l[k].last_frames_used = _empty_set

                return list(outputs)
            dlrm.apply_emb = _fast_apply_emb
            log(f"  Fast apply_emb enabled (C++ fast_forward, zero Python loop overhead)")

        # Run inference — CompressedEmbeddingBag handles lookups inside dlrm.forward()
        drop_caches()
        time.sleep(0.5)
        gc.collect()

        # Warmup: run first N batches to populate LRU cache before timing
        if warmup_batches > 0:
            log(f"  Warming up cache ({warmup_batches} batches)...")
            with torch.no_grad():
                for wb_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
                    if wb_idx >= warmup_batches:
                        break
                    Z = dlrm(X, lS_o, lS_i)
                    for t_idx in caches:
                        if isinstance(dlrm.emb_l[t_idx], CompressedEmbeddingBag):
                            frames = dlrm.emb_l[t_idx].last_frames_used
                            caches[t_idx].launch_prefetch(frames)
            # Reset cache stats after warmup
            for c in caches.values():
                c.reset_stats()
            log(f"  Cache warmed ({len(global_cache) if global_cache else sum(len(c.cache) for c in caches.values())} frames loaded)")

        scores, targets, blats, rss_trace = [], [], [], []
        t0 = time.time()

        with torch.no_grad():
            for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
                bt0 = time.time()

                Z = dlrm(X, lS_o, lS_i)
                blats.append(time.time() - bt0)

                # Launch prefetch based on frames accessed during forward
                for t_idx in caches:
                    if isinstance(dlrm.emb_l[t_idx], CompressedEmbeddingBag):
                        frames = dlrm.emb_l[t_idx].last_frames_used
                        caches[t_idx].launch_prefetch(frames)

                scores.extend(Z.detach().cpu().numpy().flatten().tolist())
                targets.extend(T.detach().cpu().numpy().flatten().tolist())

                if batch_idx % 500 == 0:
                    rss = get_rss_mb()
                    rss_trace.append((batch_idx, rss))
                    log(f"    Batch {batch_idx}: lat={blats[-1]*1000:.1f}ms, RSS={rss:.0f}MB")

        total_time = time.time() - t0
        auc = roc_auc_score(targets, scores)
        rss = get_rss_mb()

        total_hits = sum(c.stats['cache_hits'] for c in caches.values())
        total_misses = sum(c.stats['cache_misses'] for c in caches.values())
        total_demand = sum(c.stats['demand_decomps'] for c in caches.values())
        total_prefetch_hits = sum(c.stats['prefetch_hits'] for c in caches.values())
        total_demand_ms = sum(c.stats['total_demand_ms'] for c in caches.values())
        all_batch_demands = []
        for c in caches.values():
            all_batch_demands.extend(c.stats['batch_demand_decomps'])
        avg_demand = np.mean(all_batch_demands) if all_batch_demands else 0
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0

        # Compressed size = bytes in memory
        compressed_mb = total_compressed_mb

        orig_cold_mb = sum(cold_num_rows.get(t, 0) * EMB_DIM * 4
                           for t in caches) / 1024 / 1024
        compression_ratio = orig_cold_mb / compressed_mb if compressed_mb > 0 else 0

        # LRU cache actual memory: count decoded frames actually held
        if use_global_cache and global_cache is not None:
            lru_frames = len(global_cache)
            # uint8 frames in global cache
            lru_mb = lru_frames * rows_per_frame * EMB_DIM / 1024 / 1024
        else:
            lru_frames = sum(len(c.cache) for c in caches.values())
            lru_mb = sum(
                len(c.cache) * c.rows_per_frame * c.emb_dim * 4 / 1024 / 1024
                for c in caches.values()
            )
        # When hash/bitmap tables are used, mapping memory is replaced
        effective_mapping_mb = total_mapping_mb
        if HAS_CPP_EXT and compressed_table_ids and (use_hash or use_bitmap):
            if use_bitmap:
                # Bitmap: 12 bytes per 64 rows (8 byte bitmap word + 4 byte rank)
                bm_mb = 0
                for t_idx in caches:
                    n_rows = ln_emb[t_idx]
                    n_words = (n_rows + 63) // 64
                    bm_mb += (n_words * 8 + (n_words + 1) * 4) / 1024 / 1024
                effective_mapping_mb = bm_mb
            else:
                # Hash table: ~60% load factor, 8 bytes per slot
                hash_mb = 0
                for t_idx in caches:
                    E = dlrm.emb_l[t_idx]
                    hw = getattr(E, 'hot_weight_q8', None)
                    if hw is None:
                        hw = getattr(E, 'hot_weight', None)
                    if hw is None:
                        continue
                    n_hot = hw.size(0)
                    capacity = 1
                    while capacity < n_hot * 5 // 3:
                        capacity *= 2
                    hash_mb += capacity * 8 / 1024 / 1024
                effective_mapping_mb = hash_mb

        total_mem_mb = total_hot_mb + compressed_in_mem_mb + lru_mb + effective_mapping_mb

        log(f"  AUC={auc:.6f}, Time={total_time:.2f}s, RSS={rss:.0f}MB")
        log(f"  Hit rate={hit_rate:.1%}, Demand={total_demand} ({avg_demand:.1f}/batch), "
            f"Prefetch hits={total_prefetch_hits}, Demand time={total_demand_ms:.0f}ms")
        log(f"  Compression: {orig_cold_mb:.1f}MB fp32 -> {compressed_mb:.1f}MB "
            f"({compression_ratio:.1f}x)")
        log(f"  LRU cache: {lru_frames} frames = {lru_mb:.1f}MB")
        if HAS_CPP_EXT and compressed_table_ids and use_bitmap:
            log(f"  Bitmap-rank: {effective_mapping_mb:.1f}MB (vs {total_mapping_mb:.1f}MB mapping)")
        elif HAS_CPP_EXT and compressed_table_ids and use_hash:
            log(f"  Hot hash table: {effective_mapping_mb:.1f}MB (vs {total_mapping_mb:.1f}MB mapping)")
        else:
            log(f"  Mapping: {total_mapping_mb:.1f}MB (merged int32)")
        log(f"  Total memory: hot={total_hot_mb:.1f} + cold={compressed_in_mem_mb:.1f} "
            f"+ lru={lru_mb:.1f} + map={effective_mapping_mb:.1f} = {total_mem_mb:.1f}MB "
            f"(vs {total_emb_mb:.1f}MB baseline, {total_emb_mb/total_mem_mb:.1f}x)")
        log(f"  Batch latency: mean={np.mean(blats)*1000:.2f}ms")

        for t_idx in caches:
            caches[t_idx].report_stats()
        if use_global_cache and global_cache is not None:
            global_cache.report()

        result = {
            'auc': auc, 'total_time': total_time,
            'num_batches': len(blats),
            'mean_lat_ms': np.mean(blats)*1000,
            'p50_lat_ms': np.percentile(blats,50)*1000,
            'p99_lat_ms': np.percentile(blats,99)*1000,
            'rss_mb': rss,
            'rss_after_setup_mb': rss_after_setup,
            'hot_mb': total_hot_mb,
            'mapping_mb': effective_mapping_mb,
            'compressed_cold_mb': compressed_in_mem_mb,
            'compressed_on_disk_mb': total_compressed_mb,
            'disk_decode': disk_decode,
            'lru_mb': lru_mb,
            'total_mem_mb': total_mem_mb,
            'hit_rate': hit_rate,
            'demand_decomps': total_demand,
            'avg_demand_per_batch': avg_demand,
            'prefetch_hits': total_prefetch_hits,
            'demand_time_ms': total_demand_ms,
            'compressed_mb': compressed_mb,
            'orig_cold_mb': orig_cold_mb,
            'compression_ratio': compression_ratio,
            'cache_capacity': cache_capacity,
            'predictor': predictor_type,
            'lookahead_depth': lookahead_depth,
            'resolution': res_name,
            'rows_per_frame': rows_per_frame,
            'rss_trace': rss_trace,
        }

        # Cleanup: close caches, restore original embeddings and state_dict
        for c in caches.values():
            c.close()
        del caches

        # Restore original apply_emb
        if HAS_CPP_EXT and compressed_table_ids:
            dlrm.apply_emb = _orig_apply_emb

        # Restore state_dict entries first
        for k, v in freed_state.items():
            state_dict[k] = v

        # Restore original EmbeddingBag modules with full weights from state_dict
        for t_idx in large_tables:
            dlrm.emb_l[t_idx] = original_emb_modules[t_idx]
            k = emb_keys[t_idx]
            with torch.no_grad():
                dlrm.emb_l[t_idx].weight = nn.Parameter(
                    state_dict[k].clone(), requires_grad=False)

        gc.collect()
        return result

    # ---- Run experiments ----
    # === Bitmap-rank mode (best memory + speed balance) ===
    log(f"\n{'='*70}")
    log("EXPERIMENTS: Bitmap-rank mode (~2MB mapping)")
    log(f"{'='*70}")

    # 4K bitmap (speed + low memory): ~88MB
    key = '4K_g8_q8hot_bitmap'
    all_results[key] = run_ondemand_inference(
        res_name='4K', cache_capacity=8,
        predictor_type='none', lookahead_depth=1, tag=key,
        use_global_cache=True, disk_decode=True, quantize_hot=True,
        use_bitmap=True, warmup_batches=1)

    # 1080p bitmap: ~68MB
    key = '1080p_g32_q8hot_bitmap'
    all_results[key] = run_ondemand_inference(
        res_name='1080p', cache_capacity=32,
        predictor_type='none', lookahead_depth=1, tag=key,
        use_global_cache=True, disk_decode=True, quantize_hot=True,
        use_bitmap=True, warmup_batches=1)

    # 480p bitmap g64: ~43MB (extreme compression)
    key = '480p_g64_q8hot_bitmap'
    all_results[key] = run_ondemand_inference(
        res_name='480p', cache_capacity=64,
        predictor_type='none', lookahead_depth=1, tag=key,
        use_global_cache=True, disk_decode=True, quantize_hot=True,
        use_bitmap=True, warmup_batches=1)

    # === Hash mode for comparison ===
    log(f"\n{'='*70}")
    log("EXPERIMENTS: Hash table comparison")
    log(f"{'='*70}")

    key = '4K_g8_q8hot_hash'
    all_results[key] = run_ondemand_inference(
        res_name='4K', cache_capacity=8,
        predictor_type='none', lookahead_depth=1, tag=key,
        use_global_cache=True, disk_decode=True, quantize_hot=True,
        use_hash_table=True, warmup_batches=1)

    # === Non-hash speed mode for comparison ===
    key = '1080p_g32_q8hot'
    all_results[key] = run_ondemand_inference(
        res_name='1080p', cache_capacity=32,
        predictor_type='none', lookahead_depth=1, tag=key,
        use_global_cache=True, disk_decode=True, quantize_hot=True,
        warmup_batches=1)

    # ---- Save results ----
    log(f"\n{'='*70}")
    log("GENERATING REPORTS")
    log(f"{'='*70}")

    results_json = os.path.join(RESULTS_DIR, 'ondemand_results.json')
    serializable = {}
    for k, v in all_results.items():
        sv = {}
        for kk, vv in v.items():
            if isinstance(vv, (int, float, str, bool, type(None))):
                sv[kk] = vv
            elif isinstance(vv, list) and all(isinstance(x, (int, float, tuple, list)) for x in vv):
                sv[kk] = vv
        serializable[k] = sv
    with open(results_json, 'w') as f:
        json.dump(serializable, f, indent=2)
    log(f"  Saved: {results_json}")

    # Summary table
    summary = []
    summary.append("# On-Demand Codec Results\n")
    summary.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Baseline AUC: {baseline_auc:.6f}, Time: {baseline_time:.2f}s\n")

    header = f"{'Config':<30} {'AUC':>8} {'Time(s)':>7} {'Hit%':>6} {'Hot':>6} {'Cold':>6} {'LRU':>6} {'Total':>7} {'Reduc':>6} {'RSS':>7} {'BLat':>7}"
    summary.append(header)
    summary.append("-" * len(header))

    for key in sorted(all_results.keys()):
        r = all_results[key]
        auc = r.get('auc', 0)
        t = r.get('total_time', 0)
        hit = r.get('hit_rate', 0)
        hot_mb = r.get('hot_mb', r.get('emb_memory_mb', 0))
        cold_mb = r.get('compressed_cold_mb', 0)
        lru = r.get('lru_mb', 0)
        tmem = r.get('total_mem_mb', hot_mb + cold_mb + lru)
        reduc = total_emb_mb / tmem if tmem > 0 else 0
        rss = r.get('rss_mb', 0)
        blat = r.get('mean_lat_ms', 0)
        line = f"{key:<30} {auc:>8.6f} {t:>7.1f} {hit:>5.1%} {hot_mb:>5.0f}M {cold_mb:>5.0f}M {lru:>5.0f}M {tmem:>6.0f}M {reduc:>5.1f}x {rss:>6.0f}M {blat:>6.1f}ms"
        summary.append(line)

    summary_text = "\n".join(summary)
    summary_path = os.path.join(RESULTS_DIR, 'ondemand_results.md')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    log(f"\n{summary_text}")
    log(f"\n  Saved: {summary_path}")

    log(f"\n{'='*70}")
    log("ALL ON-DEMAND EXPERIMENTS COMPLETE")
    log(f"{'='*70}")
    log(f"Final RSS: {get_rss_mb():.0f}MB")


if __name__ == '__main__':
    main()
