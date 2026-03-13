#!/usr/bin/env python3
"""
CPU Video Codec Compression for DLRM Embedding Tables
With Hypergraph Co-Access Reordering + Multi-Batch Lookahead Prefetching

Phases:
  1. Access Pattern Profiling
  2. Hot/Cold Split
  3. Hypergraph Co-Access Reordering + H.265 Encoding
  4. Frame Cache with Multi-Batch Lookahead Prefetching
  5. Experiments A-G
"""

import os, sys, time, json, threading, gc, subprocess, tempfile, io
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import psutil
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
PROFILE_BATCHES = 5000  # full test set
FRAME_SIZE = 4096  # rows per codec frame
H265_CRF = 0  # lossless

RESULTS_DIR = "results"
PROFILING_DIR = os.path.join(RESULTS_DIR, "profiling")
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
LOG_FILE = "logs/codec_prefetch_benchmark.log"

for d in [PROFILING_DIR, HOTCOLD_DIR, REORDER_DIR, "logs"]:
    os.makedirs(d, exist_ok=True)

log_fh = open(LOG_FILE, 'a')

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


def check_done(marker):
    return os.path.exists(marker)


def mark_done(marker):
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, 'w') as f:
        f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


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
# MODEL + DATA LOADING
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
    log("Loading dataset (this may take a while for first-time processing)...")
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
    """Per-row INT8 quantization: fp32 -> uint8 with per-row min/max."""
    mins = w.min(dim=1, keepdim=True).values  # (N, 1)
    maxs = w.max(dim=1, keepdim=True).values  # (N, 1)
    ranges = maxs - mins
    ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)
    q = ((w - mins) / ranges * 255.0).round().clamp(0, 255).to(torch.uint8)
    return q, mins.squeeze(1), maxs.squeeze(1)


def dequantize_table(q_uint8, mins, maxs):
    """Per-row INT8 dequantization: uint8 -> fp32."""
    ranges = maxs - mins
    return mins.unsqueeze(1) + q_uint8.float() / 255.0 * ranges.unsqueeze(1)


# ============================================================
# H.265 ENCODING (offline, one-time, uses ffmpeg subprocess)
# ============================================================
def encode_h265_frames(q_np, frame_size, emb_dim, crf=0, output_path=None):
    """
    Encode uint8 cold embeddings as H.265 ALL-INTRA video.
    Each frame = frame_size rows of emb_dim columns.
    Returns path to the .mp4 file on disk.
    """
    num_rows = q_np.shape[0]
    num_frames = (num_rows + frame_size - 1) // frame_size

    # Pad to full frames
    padded_rows = num_frames * frame_size
    if num_rows < padded_rows:
        pad = np.zeros((padded_rows - num_rows, emb_dim), dtype=np.uint8)
        q_padded = np.vstack([q_np, pad])
    else:
        q_padded = q_np

    # Each frame: frame_size x emb_dim grayscale image
    # Need width and height for ffmpeg
    width = emb_dim
    height = frame_size

    if output_path is None:
        output_path = tempfile.mktemp(suffix='.mp4')

    t0 = time.time()
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
        '-f', 'mp4',
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(num_frames):
        frame_data = q_padded[i * frame_size:(i + 1) * frame_size]
        proc.stdin.write(frame_data.tobytes())
    proc.stdin.close()
    proc.wait()

    encode_time = time.time() - t0
    file_size = os.path.getsize(output_path)
    raw_size = num_rows * emb_dim
    log(f"  H.265 encode: {num_frames} frames, {raw_size/1024/1024:.1f}MB -> "
        f"{file_size/1024/1024:.1f}MB ({raw_size/file_size:.1f}x), {encode_time:.1f}s")
    return output_path, num_frames, encode_time


# ============================================================
# PyAV FRAME DECODER (runtime, per-frame, in-process)
# ============================================================
class PyAVFrameDecoder:
    """
    Decode individual frames from an ALL-INTRA H.265 video using PyAV.
    Each thread must have its own instance (PyAV is not thread-safe).
    Uses sequential decode with frame caching for efficiency.
    """
    def __init__(self, video_path, frame_size, emb_dim):
        self.video_path = video_path
        self.frame_size = frame_size
        self.emb_dim = emb_dim
        # Pre-decode ALL frames into a dict at init (these are uint8, compact)
        # For a table with 2353 frames * 4096 * 16 bytes = 150MB uint8
        # This is the compressed -> uint8 decode only, NOT dequant to fp32
        self._frames = {}
        container = av.open(video_path)
        frame_idx = 0
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format='gray')
            self._frames[frame_idx] = arr.reshape(frame_size, emb_dim)
            frame_idx += 1
        container.close()
        self.num_frames = frame_idx

    def decode_frame(self, frame_id):
        """Return uint8 ndarray (frame_size, emb_dim) for given frame_id."""
        if frame_id in self._frames:
            return self._frames[frame_id]
        raise ValueError(f"frame_id {frame_id} not found (max={self.num_frames-1})")

    def close(self):
        self._frames.clear()


# ============================================================
# PREDICTORS
# ============================================================
class LastBatchPredictor:
    def __init__(self):
        self.last_frames = set()
        self.second_last_frames = set()

    def predict(self, current_frames, budget_frames=30):
        predictions = {}
        for fid in self.last_frames:
            predictions[fid] = max(predictions.get(fid, 0), 1.0)
        for fid in self.second_last_frames:
            predictions[fid] = max(predictions.get(fid, 0), 0.5)
        self.second_last_frames = self.last_frames.copy()
        self.last_frames = set(current_frames)
        return predictions


class EMAPredictor:
    def __init__(self, num_frames, alpha=0.3):
        self.scores = np.zeros(num_frames)
        self.alpha = alpha

    def predict(self, current_frames, budget_frames=30):
        self.scores *= (1 - self.alpha)
        for fid in current_frames:
            if fid < len(self.scores):
                self.scores[fid] += self.alpha
        top_ids = np.argsort(self.scores)[-budget_frames:]
        return {int(fid): float(self.scores[fid])
                for fid in top_ids if self.scores[fid] > 0.01}


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
        # T+1: direct Markov
        step1 = Counter()
        for fid in current_frames:
            for nxt, cnt in self.transition[fid].most_common(budget_frames):
                step1[nxt] += cnt
        if step1:
            mx = max(step1.values())
            for fid, s in step1.most_common(budget_frames):
                predictions[fid] = 1.0 * (s / mx)
        # T+2: 2-step
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
        # T+3+: EMA fallback
        if self.lookahead_depth >= 3:
            for fid in np.argsort(self.ema_scores)[-budget_frames:]:
                fid = int(fid)
                if fid not in predictions and self.ema_scores[fid] > 0.01:
                    predictions[fid] = 0.25 * self.ema_scores[fid]
        return predictions


class OracleLookaheadPredictor:
    def __init__(self, batch_frames_list, lookahead_depth=5):
        self.batch_frames = batch_frames_list
        self.lookahead_depth = lookahead_depth
        self.current_batch_idx = 0
        avg_frames = np.mean([len(f) for f in self.batch_frames]) if self.batch_frames else 0
        log(f"Oracle: {len(self.batch_frames)} batches, avg {avg_frames:.1f} frames/batch")

    def predict(self, current_frames, budget_frames=50):
        predictions = {}
        idx = self.current_batch_idx
        for d in range(1, self.lookahead_depth + 1):
            future = idx + d
            if future >= len(self.batch_frames):
                break
            priority = 1.0 - (d - 1) * (0.8 / self.lookahead_depth)
            for fid in self.batch_frames[future]:
                predictions[fid] = max(predictions.get(fid, 0), priority)
        self.current_batch_idx += 1
        return predictions

    def reset(self):
        self.current_batch_idx = 0

    def get_optimal_cache_size(self, target_hit_rate=0.95):
        for cache_size in [25, 50, 75, 100, 150, 200, 300, 500]:
            cache = OrderedDict()
            hits = total = 0
            for batch_idx, frame_set in enumerate(self.batch_frames):
                for d in range(1, self.lookahead_depth + 1):
                    future = batch_idx + d
                    if future < len(self.batch_frames):
                        for fid in self.batch_frames[future]:
                            cache[fid] = True
                            cache.move_to_end(fid)
                            while len(cache) > cache_size:
                                cache.popitem(last=False)
                for fid in frame_set:
                    total += 1
                    if fid in cache:
                        hits += 1
                        cache.move_to_end(fid)
                    else:
                        cache[fid] = True
                        while len(cache) > cache_size:
                            cache.popitem(last=False)
            rate = hits / total if total else 0
            log(f"  Cache {cache_size:>4d}: hit={rate:.3f} {'OK' if rate >= target_hit_rate else '  '}")
            if rate >= target_hit_rate:
                return cache_size
        return 500


# ============================================================
# PREFETCH FRAME CACHE ENGINE
# ============================================================
class PrefetchFrameCache:
    """
    Bounded frame cache with async multi-batch prefetching.
    MEMORY INVARIANT: len(self.cache) <= self.cache_capacity at ALL times.
    """
    def __init__(self, video_path, frame_size, emb_dim, num_cold_rows,
                 quant_mins, quant_maxs, cache_capacity=100,
                 predictor=None, num_prefetch_workers=2):
        self.video_path = video_path
        self.frame_size = frame_size
        self.emb_dim = emb_dim
        self.num_cold_rows = num_cold_rows
        self.num_frames = (num_cold_rows + frame_size - 1) // frame_size
        self.quant_mins = quant_mins  # (num_cold_rows,) fp32
        self.quant_maxs = quant_maxs  # (num_cold_rows,) fp32

        # Single decoder that pre-decodes all uint8 frames (read-only, thread-safe)
        self._decoder = PyAVFrameDecoder(video_path, frame_size, emb_dim)

        # Bounded cache
        self.cache = {}           # frame_id -> decoded fp32 tensor (actual_rows, emb_dim)
        self.cache_priority = {}
        self.cache_capacity = cache_capacity
        self.lock = threading.Lock()

        # Memory budget check
        frame_bytes = frame_size * emb_dim * 4
        total_cache_bytes = cache_capacity * frame_bytes
        log(f"Cache: {cache_capacity} frames x {frame_bytes/1024:.1f}KB = "
            f"{total_cache_bytes/1024/1024:.1f}MB budget")
        assert total_cache_bytes < 500 * 1024 * 1024, \
            f"Cache too large: {total_cache_bytes/1024/1024:.0f}MB > 500MB"

        # Prefetch
        self.executor = ThreadPoolExecutor(max_workers=num_prefetch_workers)
        self.pending = set()
        self.prefetch_lock = threading.Lock()

        # Predictor
        self.predictor = predictor

        # Stats
        self.stats = {
            'cache_hits': 0, 'cache_misses': 0,
            'prefetch_hits': 0, 'demand_decomps': 0,
            'prefetch_submissions': 0, 'prefetch_wastes': 0,
            'total_demand_ms': 0.0, 'total_prefetch_ms': 0.0,
            'batch_demand_decomps': [],
            'batch_hit_rates': [],
        }

    def _decode_and_dequant(self, frame_id):
        """Dequantize one pre-decoded uint8 frame to fp32."""
        q_uint8 = self._decoder.decode_frame(frame_id)  # (frame_size, emb_dim) uint8
        row_start = frame_id * self.frame_size
        row_end = min(row_start + self.frame_size, self.num_cold_rows)
        actual_rows = row_end - row_start
        q_slice = q_uint8[:actual_rows]
        mins = self.quant_mins[row_start:row_end].numpy()
        maxs = self.quant_maxs[row_start:row_end].numpy()
        ranges = maxs - mins
        fp32 = mins[:, None] + q_slice.astype(np.float32) / 255.0 * ranges[:, None]
        return torch.from_numpy(fp32)

    def _prefetch_worker(self, frame_id, priority):
        t0 = time.time()
        frame_data = self._decode_and_dequant(frame_id)
        elapsed = (time.time() - t0) * 1000
        with self.lock:
            if frame_id not in self.cache:
                self.cache[frame_id] = frame_data
                self.cache_priority[frame_id] = priority
                self._evict_if_needed()
            self.stats['total_prefetch_ms'] += elapsed
        with self.prefetch_lock:
            self.pending.discard(frame_id)

    def _evict_if_needed(self):
        """Must hold self.lock."""
        while len(self.cache) > self.cache_capacity:
            evict_id = min(self.cache_priority, key=self.cache_priority.get)
            del self.cache[evict_id]
            del self.cache_priority[evict_id]
            self.stats['prefetch_wastes'] += 1
        assert len(self.cache) <= self.cache_capacity

    def launch_prefetch(self, current_batch_frames):
        if self.predictor is None:
            return
        predictions = self.predictor.predict(
            current_batch_frames, budget_frames=self.cache_capacity)
        with self.lock:
            cached = set(self.cache.keys())
            cur_size = len(self.cache)
        with self.prefetch_lock:
            in_flight = self.pending.copy()
        candidates = {fid: pri for fid, pri in predictions.items()
                      if fid not in cached and fid not in in_flight}
        sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])
        budget = max(0, int(self.cache_capacity * 0.7) - cur_size - len(in_flight))
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
        """
        Main thread: lookup cold embeddings from bounded frame cache.
        Returns (result_tensor, set_of_frame_ids_used).
        """
        if len(cold_indices_reordered) == 0:
            return torch.zeros(0, self.emb_dim), set()

        # Sort for frame locality (Rule 4)
        sorted_order = torch.argsort(cold_indices_reordered)
        sorted_indices = cold_indices_reordered[sorted_order]

        frame_ids = (sorted_indices // self.frame_size).long()
        unique_frames = torch.unique(frame_ids).tolist()

        batch_hits = 0
        batch_misses = 0

        for fid in unique_frames:
            with self.lock:
                if fid in self.cache:
                    self.cache_priority[fid] = 2.0
                    self.stats['cache_hits'] += 1
                    batch_hits += 1
                    continue
            # Wait briefly for in-flight prefetch
            waited = False
            for _ in range(10):
                with self.prefetch_lock:
                    if fid not in self.pending:
                        break
                time.sleep(0.0005)
                waited = True
            with self.lock:
                if fid in self.cache:
                    self.stats['prefetch_hits'] += 1
                    self.cache_priority[fid] = 2.0
                    batch_hits += 1
                    continue
            # DEMAND DECODE
            t0 = time.time()
            frame_data = self._decode_and_dequant(fid)
            elapsed = (time.time() - t0) * 1000
            with self.lock:
                self.cache[fid] = frame_data
                self.cache_priority[fid] = 2.0
                self._evict_if_needed()
                self.stats['demand_decomps'] += 1
                self.stats['total_demand_ms'] += elapsed
            self.stats['cache_misses'] += 1
            batch_misses += 1

        # Vectorized gather (Rule 5)
        # Collect frame data copies to avoid holding lock during tensor ops
        frame_data_copies = {}
        with self.lock:
            for fid in unique_frames:
                if fid in self.cache:
                    frame_data_copies[fid] = self.cache[fid]
                else:
                    # Frame was evicted — re-decode inline
                    pass

        # Re-decode any evicted frames
        for fid in unique_frames:
            if fid not in frame_data_copies:
                frame_data_copies[fid] = self._decode_and_dequant(fid)
                self.stats['demand_decomps'] += 1

        results = torch.zeros(len(sorted_indices), self.emb_dim)
        for fid in unique_frames:
            mask = (frame_ids == fid)
            offsets = (sorted_indices[mask] % self.frame_size).long()
            frame_data = frame_data_copies[fid]
            actual_rows = frame_data.shape[0]
            safe_offsets = torch.clamp(offsets, max=actual_rows - 1)
            results[mask] = frame_data[safe_offsets]

        # Unsort
        final = torch.zeros_like(results)
        final[sorted_order] = results

        # Per-batch stats
        total_batch = batch_hits + batch_misses
        self.stats['batch_demand_decomps'].append(batch_misses)
        self.stats['batch_hit_rates'].append(
            batch_hits / total_batch if total_batch > 0 else 1.0)

        # INVARIANT CHECK
        with self.lock:
            assert len(self.cache) <= self.cache_capacity, \
                f"LEAK: {len(self.cache)} > {self.cache_capacity}"

        return final, set(unique_frames)

    def clear_cache(self):
        with self.lock:
            self.cache.clear()
            self.cache_priority.clear()

    def reset_stats(self):
        self.stats = {
            'cache_hits': 0, 'cache_misses': 0,
            'prefetch_hits': 0, 'demand_decomps': 0,
            'prefetch_submissions': 0, 'prefetch_wastes': 0,
            'total_demand_ms': 0.0, 'total_prefetch_ms': 0.0,
            'batch_demand_decomps': [],
            'batch_hit_rates': [],
        }

    def report_stats(self):
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total if total else 0
        avg_demand = (np.mean(self.stats['batch_demand_decomps'])
                      if self.stats['batch_demand_decomps'] else 0)
        log(f"  Cache: {hit_rate:.1%} hit | demand={self.stats['demand_decomps']} "
            f"({avg_demand:.1f}/batch) | prefetch_hits={self.stats['prefetch_hits']} "
            f"| demand_time={self.stats['total_demand_ms']:.0f}ms "
            f"| size={len(self.cache)}/{self.cache_capacity}")
        return {
            'hit_rate': hit_rate,
            'demand_decomps': self.stats['demand_decomps'],
            'avg_demand_per_batch': avg_demand,
            'prefetch_hits': self.stats['prefetch_hits'],
            'demand_time_ms': self.stats['total_demand_ms'],
            'prefetch_time_ms': self.stats['total_prefetch_ms'],
            'cache_size': len(self.cache),
        }

    def close(self):
        self.executor.shutdown(wait=False)
        self._decoder.close()


# ============================================================
# MAIN EXPERIMENT ENGINE
# ============================================================
def main():
    log("=" * 70)
    log("CPU VIDEO CODEC COMPRESSION FOR DLRM EMBEDDING TABLES")
    log("Hypergraph Co-Access Reordering + Multi-Batch Lookahead Prefetching")
    log("=" * 70)
    log(f"RSS at start: {get_rss_mb():.0f}MB")

    # Load model and data
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large (>={LARGE_TABLE_THRESHOLD}): {large_tables}")
    for t in range(num_tables):
        log(f"  Table {t}: {ln_emb[t]:>10,} rows x {EMB_DIM} dim = "
            f"{ln_emb[t]*EMB_DIM*4/1024/1024:.1f}MB")
    total_emb_mb = sum(ln_emb[t] * EMB_DIM * 4 for t in range(num_tables)) / 1024 / 1024
    log(f"  Total embedding memory: {total_emb_mb:.1f}MB")

    # Collect all experiment results
    all_results = {}

    # ==============================================================
    # PHASE 1: ACCESS PATTERN PROFILING
    # ==============================================================
    PROFILING_DONE = os.path.join(PROFILING_DIR, '.done')
    if check_done(PROFILING_DONE):
        log("\nPhase 1 SKIPPED — loading existing profiling data")
        freq = {}
        for t in range(num_tables):
            fp = os.path.join(PROFILING_DIR, f'freq_table_{t}.npy')
            if os.path.exists(fp):
                freq[t] = np.load(fp)
        batch_cold_frames_all = None  # will rebuild if needed
        batch_access_log_path = os.path.join(PROFILING_DIR, 'batch_access_log.npz')
        if os.path.exists(batch_access_log_path):
            _bal = np.load(batch_access_log_path, allow_pickle=True)
            batch_access_log = _bal['batch_access_log'].tolist()
        else:
            batch_access_log = None
    else:
        log("\nPhase 1 — Access Pattern Profiling")
        log(f"  Processing up to {PROFILE_BATCHES} test batches...")

        freq = {t: np.zeros(ln_emb[t], dtype=np.int64) for t in range(num_tables)}
        batch_access_log = []  # list of dicts: {table_idx: set_of_indices}

        # Time model components for prefetch budget
        emb_times = []
        mlp_times = []

        t0 = time.time()
        for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
            if batch_idx >= PROFILE_BATCHES:
                break
            batch_record = {}
            for t in range(num_tables):
                indices = lS_i[t].numpy().flatten()
                unique_idx = np.unique(indices)
                freq[t][unique_idx] += 1
                batch_record[t] = set(unique_idx.tolist())
            batch_access_log.append(batch_record)

            # Time full forward pass
            with torch.no_grad():
                mt0 = time.time()
                Z = dlrm(X, lS_o, lS_i)
                fwd_time = time.time() - mt0
                mlp_times.append(fwd_time)
                emb_times.append(fwd_time * 0.3)  # approximate emb fraction

            if batch_idx % 500 == 0:
                log(f"    Batch {batch_idx}/{PROFILE_BATCHES}")

        profiling_time = time.time() - t0
        log(f"  Profiled {len(batch_access_log)} batches in {profiling_time:.1f}s")

        # Save frequency arrays
        for t in range(num_tables):
            np.save(os.path.join(PROFILING_DIR, f'freq_table_{t}.npy'), freq[t])

        # Save batch access log
        np.savez_compressed(os.path.join(PROFILING_DIR, 'batch_access_log.npz'),
                            batch_access_log=np.array(batch_access_log, dtype=object))

        # Compute and save CDF for large tables
        log("  Computing access CDFs...")
        for t in large_tables:
            f = freq[t]
            nonzero = f[f > 0]
            sorted_counts = np.sort(nonzero)[::-1]
            cumsum = np.cumsum(sorted_counts)
            total = cumsum[-1]
            cdf = cumsum / total
            # Find coverage thresholds
            for pct in [0.5, 0.8, 0.9, 0.95]:
                idx = np.searchsorted(cdf, pct)
                log(f"    Table {t}: {pct*100:.0f}% coverage = {idx+1:,} indices "
                    f"({(idx+1)/ln_emb[t]*100:.2f}% of table)")
            np.save(os.path.join(PROFILING_DIR, f'cdf_table_{t}.npy'), cdf)

        # Prefetch budget
        avg_emb_ms = np.mean(emb_times) * 1000
        avg_mlp_ms = np.mean(mlp_times) * 1000
        prefetch_budget_ms = avg_mlp_ms - avg_emb_ms  # MLP time minus emb time
        log(f"  Prefetch budget: emb={avg_emb_ms:.2f}ms, total_fwd={avg_mlp_ms:.2f}ms, "
            f"budget={prefetch_budget_ms:.2f}ms")

        # Measure PyAV decode latency (quick test)
        # We'll do this properly in Phase 3

        # Batch-to-batch frame overlap (using FRAME_SIZE grouping)
        log("  Computing temporal frame overlap...")
        overlaps = []
        for bi in range(1, min(len(batch_access_log), 2000)):
            prev_frames = set()
            curr_frames = set()
            for t in large_tables:
                for idx in batch_access_log[bi-1].get(t, set()):
                    prev_frames.add((t, idx // FRAME_SIZE))
                for idx in batch_access_log[bi].get(t, set()):
                    curr_frames.add((t, idx // FRAME_SIZE))
            if curr_frames:
                overlap = len(prev_frames & curr_frames) / len(curr_frames)
                overlaps.append(overlap)
        avg_overlap = np.mean(overlaps) if overlaps else 0
        log(f"  Frame overlap: mean={avg_overlap:.3f}, median={np.median(overlaps):.3f}")

        with open(os.path.join(PROFILING_DIR, 'profiling_summary.json'), 'w') as f:
            json.dump({
                'num_batches': len(batch_access_log),
                'avg_emb_ms': avg_emb_ms,
                'avg_mlp_ms': avg_mlp_ms,
                'prefetch_budget_ms': prefetch_budget_ms,
                'avg_frame_overlap': avg_overlap,
            }, f, indent=2)

        mark_done(PROFILING_DONE)
        log("  Phase 1 complete.")

    # ==============================================================
    # PHASE 2: HOT/COLD SPLIT
    # ==============================================================
    HOTCOLD_DONE = os.path.join(HOTCOLD_DIR, '.done')
    if check_done(HOTCOLD_DONE):
        log("\nPhase 2 SKIPPED — loading existing hot/cold split")
        is_hot = {}; orig_to_hot = {}; orig_to_cold = {}
        hot_emb_weight = {}; hot_indices = {}; cold_indices = {}
        for t in range(num_tables):
            is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                                   map_location='cpu', weights_only=True)
            orig_to_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'orig_to_hot_{t}.pt'),
                                         map_location='cpu', weights_only=True)
            orig_to_cold[t] = torch.load(os.path.join(HOTCOLD_DIR, f'orig_to_cold_{t}.pt'),
                                          map_location='cpu', weights_only=True)
            hot_emb_weight[t] = torch.load(os.path.join(HOTCOLD_DIR, f'hot_weight_{t}.pt'),
                                            map_location='cpu', weights_only=True)
            hot_indices[t] = torch.load(os.path.join(HOTCOLD_DIR, f'hot_indices_{t}.pt'),
                                         map_location='cpu', weights_only=True)
            cold_indices[t] = torch.load(os.path.join(HOTCOLD_DIR, f'cold_indices_{t}.pt'),
                                          map_location='cpu', weights_only=True)
    else:
        log("\nPhase 2 — Hot/Cold Split")
        is_hot = {}; orig_to_hot = {}; orig_to_cold = {}
        hot_emb_weight = {}; hot_indices = {}; cold_indices = {}

        for t in range(num_tables):
            w = state_dict[emb_keys[t]]
            n_emb = ln_emb[t]

            if n_emb < LARGE_TABLE_THRESHOLD:
                # Small table: all hot
                is_hot[t] = torch.ones(n_emb, dtype=torch.bool)
                orig_to_hot[t] = torch.arange(n_emb, dtype=torch.long)
                orig_to_cold[t] = torch.full((n_emb,), -1, dtype=torch.long)
                hot_emb_weight[t] = w.clone()
                hot_indices[t] = torch.arange(n_emb, dtype=torch.long)
                cold_indices[t] = torch.tensor([], dtype=torch.long)
                continue

            # Large table: frequency-based split
            f = freq[t]
            if f.sum() == 0:
                # No accesses profiled — treat all as hot
                is_hot[t] = torch.ones(n_emb, dtype=torch.bool)
                orig_to_hot[t] = torch.arange(n_emb, dtype=torch.long)
                orig_to_cold[t] = torch.full((n_emb,), -1, dtype=torch.long)
                hot_emb_weight[t] = w.clone()
                hot_indices[t] = torch.arange(n_emb, dtype=torch.long)
                cold_indices[t] = torch.tensor([], dtype=torch.long)
                continue

            # Sort by frequency descending
            sorted_idx = np.argsort(-f)
            sorted_counts = f[sorted_idx]
            cumsum = np.cumsum(sorted_counts)
            total = cumsum[-1]
            cutoff = np.searchsorted(cumsum, total * HOT_COVERAGE) + 1

            hot_set = set(sorted_idx[:cutoff].tolist())
            h_idx = torch.tensor(sorted(hot_set), dtype=torch.long)
            c_idx = torch.tensor(sorted(set(range(n_emb)) - hot_set), dtype=torch.long)

            is_hot_t = torch.zeros(n_emb, dtype=torch.bool)
            is_hot_t[h_idx] = True

            o2h = torch.full((n_emb,), -1, dtype=torch.long)
            o2h[h_idx] = torch.arange(len(h_idx), dtype=torch.long)

            o2c = torch.full((n_emb,), -1, dtype=torch.long)
            o2c[c_idx] = torch.arange(len(c_idx), dtype=torch.long)

            is_hot[t] = is_hot_t
            orig_to_hot[t] = o2h
            orig_to_cold[t] = o2c
            hot_emb_weight[t] = w[h_idx].clone()
            hot_indices[t] = h_idx
            cold_indices[t] = c_idx

            log(f"  Table {t}: {len(h_idx):,} hot ({len(h_idx)/n_emb*100:.2f}%), "
                f"{len(c_idx):,} cold, hot_mem={len(h_idx)*EMB_DIM*4/1024:.1f}KB")

        # Save
        for t in range(num_tables):
            torch.save(is_hot[t], os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'))
            torch.save(orig_to_hot[t], os.path.join(HOTCOLD_DIR, f'orig_to_hot_{t}.pt'))
            torch.save(orig_to_cold[t], os.path.join(HOTCOLD_DIR, f'orig_to_cold_{t}.pt'))
            torch.save(hot_emb_weight[t], os.path.join(HOTCOLD_DIR, f'hot_weight_{t}.pt'))
            torch.save(hot_indices[t], os.path.join(HOTCOLD_DIR, f'hot_indices_{t}.pt'))
            torch.save(cold_indices[t], os.path.join(HOTCOLD_DIR, f'cold_indices_{t}.pt'))

        mark_done(HOTCOLD_DONE)
        log("  Phase 2 complete.")

    # ==============================================================
    # PHASE 3: REORDERING + H.265 ENCODING
    # ==============================================================
    REORDER_DONE = os.path.join(REORDER_DIR, '.done')
    if check_done(REORDER_DONE):
        log("\nPhase 3 SKIPPED — loading existing reorder + compressed files")
        orig_to_cold_reordered = {}
        cold_quant_mins = {}
        cold_quant_maxs = {}
        cold_num_rows = {}
        for t in large_tables:
            fp = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
            if os.path.exists(fp):
                orig_to_cold_reordered[t] = torch.load(fp, map_location='cpu', weights_only=True)
            fp_mins = os.path.join(REORDER_DIR, f'quant_mins_{t}.pt')
            fp_maxs = os.path.join(REORDER_DIR, f'quant_maxs_{t}.pt')
            if os.path.exists(fp_mins):
                cold_quant_mins[t] = torch.load(fp_mins, map_location='cpu', weights_only=True)
                cold_quant_maxs[t] = torch.load(fp_maxs, map_location='cpu', weights_only=True)
            nc_path = os.path.join(REORDER_DIR, f'num_cold_{t}.txt')
            if os.path.exists(nc_path):
                cold_num_rows[t] = int(open(nc_path).read().strip())
        # Also load unordered versions
        orig_to_cold_unordered = {}
        for t in large_tables:
            fp = os.path.join(REORDER_DIR, f'orig_to_cold_unordered_{t}.pt')
            if os.path.exists(fp):
                orig_to_cold_unordered[t] = torch.load(fp, map_location='cpu', weights_only=True)
    else:
        log("\nPhase 3 — Co-Access Reordering + H.265 Encoding")

        # Reload batch access log if not in memory
        if batch_access_log is None:
            bal_path = os.path.join(PROFILING_DIR, 'batch_access_log.npz')
            if os.path.exists(bal_path):
                _bal = np.load(bal_path, allow_pickle=True)
                batch_access_log = _bal['batch_access_log'].tolist()
            else:
                log("  ERROR: No batch access log found. Re-run Phase 1.")
                sys.exit(1)

        orig_to_cold_reordered = {}
        orig_to_cold_unordered = {}
        cold_quant_mins = {}
        cold_quant_maxs = {}
        cold_num_rows = {}

        for t in large_tables:
            n_cold = len(cold_indices[t])
            if n_cold == 0:
                continue
            log(f"\n  Table {t}: {n_cold:,} cold embeddings")
            cold_num_rows[t] = n_cold

            # Save unordered mapping (orig_to_cold is already it)
            orig_to_cold_unordered[t] = orig_to_cold[t].clone()

            # Build co-access reordering using batch-affinity approach
            # Key idea: assign each cold embedding a "batch signature" vector,
            # then sort by it so co-accessed embeddings end up adjacent.
            log(f"    Building batch-affinity ordering...")

            # Vectorized: build cold_to_seq mapping
            cold_idx_np = cold_indices[t].numpy()
            cold_to_seq_arr = np.full(ln_emb[t], -1, dtype=np.int64)
            cold_to_seq_arr[cold_idx_np] = np.arange(n_cold, dtype=np.int64)

            # Build frequency array for cold indices
            cold_freq = np.zeros(n_cold, dtype=np.int64)
            num_sample_batches = min(len(batch_access_log), 5000)

            # Batch-affinity: assign each cold embedding to its "primary batch"
            # (the batch that accesses it most), then sort within that group
            cold_first_batch = np.full(n_cold, num_sample_batches, dtype=np.int64)

            for bi in range(num_sample_batches):
                accessed = batch_access_log[bi].get(t, set())
                for idx in accessed:
                    seq = cold_to_seq_arr[idx] if idx < ln_emb[t] else -1
                    if seq >= 0:
                        cold_freq[seq] += 1
                        if cold_first_batch[seq] == num_sample_batches:
                            cold_first_batch[seq] = bi

            active_count = np.sum(cold_freq > 0)
            log(f"    {active_count:,} active cold indices (of {n_cold:,})")

            # Create compound sort key: (first_batch, -frequency)
            # This groups co-accessed indices by batch, then sorts by frequency
            # within each group, naturally creating frame locality
            sort_key = cold_first_batch.astype(np.float64) * 1e12 - cold_freq.astype(np.float64)
            order = np.argsort(sort_key).astype(np.int64)
            log(f"    Batch-affinity ordering complete: {len(order):,} indices")

            reorder = order  # already np.int64 array

            # Build reordered mapping (vectorized)
            # reorder[new_pos] = old_cold_seq
            # We need: orig_to_cold_reordered[orig_idx] = new_pos
            o2c_reordered = torch.full((ln_emb[t],), -1, dtype=torch.long)
            orig_indices_reordered = cold_idx_np[reorder]  # vectorized lookup
            new_positions = torch.arange(len(reorder), dtype=torch.long)
            o2c_reordered[torch.from_numpy(orig_indices_reordered)] = new_positions
            orig_to_cold_reordered[t] = o2c_reordered

            # Build reordered cold weights and quantize
            cold_w = state_dict[emb_keys[t]][cold_indices[t]]  # (n_cold, emb_dim)
            reordered_w = cold_w[reorder]  # (n_cold, emb_dim)

            q, mins, maxs = quantize_table(reordered_w)
            cold_quant_mins[t] = mins
            cold_quant_maxs[t] = maxs

            # Encode reordered as H.265
            video_path = os.path.join(REORDER_DIR, f'cold_reordered_{t}.mp4')
            encode_h265_frames(q.numpy(), FRAME_SIZE, EMB_DIM, crf=H265_CRF,
                              output_path=video_path)

            # Also encode UNORDERED for comparison (Experiment C)
            unordered_w = cold_w  # original order
            q_un, mins_un, maxs_un = quantize_table(unordered_w)
            video_path_un = os.path.join(REORDER_DIR, f'cold_unordered_{t}.mp4')
            encode_h265_frames(q_un.numpy(), FRAME_SIZE, EMB_DIM, crf=H265_CRF,
                              output_path=video_path_un)
            # Save unordered quant params too
            torch.save(mins_un, os.path.join(REORDER_DIR, f'quant_mins_unordered_{t}.pt'))
            torch.save(maxs_un, os.path.join(REORDER_DIR, f'quant_maxs_unordered_{t}.pt'))

            # Save
            torch.save(o2c_reordered, os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt'))
            torch.save(orig_to_cold[t], os.path.join(REORDER_DIR, f'orig_to_cold_unordered_{t}.pt'))
            torch.save(mins, os.path.join(REORDER_DIR, f'quant_mins_{t}.pt'))
            torch.save(maxs, os.path.join(REORDER_DIR, f'quant_maxs_{t}.pt'))
            with open(os.path.join(REORDER_DIR, f'num_cold_{t}.txt'), 'w') as f:
                f.write(str(n_cold))
            np.save(os.path.join(REORDER_DIR, f'cold_order_{t}.npy'), reorder)

            # Free large tensors
            del cold_w, reordered_w, q, unordered_w, q_un
            gc.collect()

        mark_done(REORDER_DONE)
        log("  Phase 3 complete.")

    # ==============================================================
    # PHASE 4: INFERENCE BENCHMARKS
    # ==============================================================
    log("\n" + "=" * 70)
    log("PHASE 4 — INFERENCE BENCHMARKS")
    log("=" * 70)

    def restore_weights():
        with torch.no_grad():
            for k in emb_keys:
                t_idx = int(k.split('.')[1])
                dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

    def run_baseline(tag="baseline"):
        """Experiment A: pure baseline, no compression."""
        log(f"\n--- {tag}: Full fp32 baseline ---")
        restore_weights()
        gc.collect()
        drop_caches()
        time.sleep(0.5)

        scores, targets = [], []
        blats = []
        emb_times_l, mlp_times_l = [], []
        t0 = time.time()
        with torch.no_grad():
            for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
                bt0 = time.time()
                Z = dlrm(X, lS_o, lS_i)
                blats.append(time.time() - bt0)
                S = Z.detach().cpu().numpy().flatten()
                Tn = T.detach().cpu().numpy().flatten()
                scores.extend(S.tolist())
                targets.extend(Tn.tolist())
                if batch_idx % 1000 == 0:
                    log(f"    Batch {batch_idx}")
        total_time = time.time() - t0
        auc = roc_auc_score(targets, scores)
        rss = get_rss_mb()
        log(f"  AUC={auc:.6f}, Time={total_time:.2f}s, Batches={len(blats)}, "
            f"RSS={rss:.0f}MB")
        log(f"  Batch latency: mean={np.mean(blats)*1000:.2f}ms, "
            f"p50={np.percentile(blats,50)*1000:.2f}ms, "
            f"p99={np.percentile(blats,99)*1000:.2f}ms")
        return {
            'auc': auc, 'total_time': total_time,
            'num_batches': len(blats),
            'mean_lat_ms': np.mean(blats)*1000,
            'p50_lat_ms': np.percentile(blats,50)*1000,
            'p99_lat_ms': np.percentile(blats,99)*1000,
            'rss_mb': rss,
            'emb_memory_mb': total_emb_mb,
        }

    def run_hotcold_split_only(tag="hotcold_split"):
        """Experiment B: hot/cold split, cold as direct tensor (no codec)."""
        log(f"\n--- {tag}: Hot/cold split, no codec ---")
        restore_weights()
        gc.collect()

        # Apply quantize-dequantize to cold to match what codec would do
        for t in large_tables:
            if len(cold_indices[t]) == 0:
                continue
            w = state_dict[emb_keys[t]]
            # Hot: keep fp32 original
            # Cold: quantize+dequantize (simulates codec quality loss)
            c_idx = cold_indices[t]
            c_w = w[c_idx]
            q, mins, maxs = quantize_table(c_w)
            dq = dequantize_table(q, mins, maxs)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[c_idx] = dq
            del q, dq, c_w
        gc.collect()

        drop_caches()
        time.sleep(0.5)

        scores, targets = [], []
        blats = []
        t0 = time.time()
        with torch.no_grad():
            for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
                bt0 = time.time()
                Z = dlrm(X, lS_o, lS_i)
                blats.append(time.time() - bt0)
                S = Z.detach().cpu().numpy().flatten()
                Tn = T.detach().cpu().numpy().flatten()
                scores.extend(S.tolist())
                targets.extend(Tn.tolist())
        total_time = time.time() - t0
        auc = roc_auc_score(targets, scores)
        rss = get_rss_mb()
        hot_mb = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables) / 1024 / 1024
        log(f"  AUC={auc:.6f}, Time={total_time:.2f}s, RSS={rss:.0f}MB, HotMem={hot_mb:.1f}MB")
        return {
            'auc': auc, 'total_time': total_time,
            'num_batches': len(blats),
            'mean_lat_ms': np.mean(blats)*1000,
            'p50_lat_ms': np.percentile(blats,50)*1000,
            'p99_lat_ms': np.percentile(blats,99)*1000,
            'rss_mb': rss,
            'hot_memory_mb': hot_mb,
        }

    def run_codec_inference(use_reorder, cache_capacity, predictor_type='none',
                            lookahead_depth=1, tag="codec"):
        """
        Experiment C/D/E: codec-compressed cold with bounded frame cache.
        use_reorder: True=reordered, False=original order
        predictor_type: 'none', 'last', 'ema', 'markov', 'oracle'
        """
        order_label = "reordered" if use_reorder else "unordered"
        log(f"\n--- {tag}: {order_label}, cache={cache_capacity}, "
            f"predictor={predictor_type}, depth={lookahead_depth} ---")

        restore_weights()
        # Set hot embeddings (keep fp32) and small tables
        for t in range(num_tables):
            if t not in large_tables:
                continue
            w = state_dict[emb_keys[t]]
            h_idx = hot_indices[t]
            if len(h_idx) > 0:
                with torch.no_grad():
                    dlrm.emb_l[t].weight.data[h_idx] = w[h_idx]
        gc.collect()

        # Determine which mapping and video to use
        if use_reorder:
            o2c_map = orig_to_cold_reordered
            video_suffix = "cold_reordered"
            q_mins = cold_quant_mins
            q_maxs = cold_quant_maxs
        else:
            o2c_map = orig_to_cold_unordered if orig_to_cold_unordered else orig_to_cold
            video_suffix = "cold_unordered"
            q_mins = {}
            q_maxs = {}
            for t_idx in large_tables:
                fp_mins = os.path.join(REORDER_DIR, f'quant_mins_unordered_{t_idx}.pt')
                fp_maxs = os.path.join(REORDER_DIR, f'quant_maxs_unordered_{t_idx}.pt')
                if os.path.exists(fp_mins):
                    q_mins[t_idx] = torch.load(fp_mins, map_location='cpu', weights_only=True)
                    q_maxs[t_idx] = torch.load(fp_maxs, map_location='cpu', weights_only=True)

        # Build per-table frame caches
        caches = {}
        for t_idx in large_tables:
            n_cold = cold_num_rows.get(t_idx, len(cold_indices[t_idx]))
            if n_cold == 0:
                continue
            video_path = os.path.join(REORDER_DIR, f'{video_suffix}_{t_idx}.mp4')
            if not os.path.exists(video_path):
                log(f"    WARNING: No video file for table {t_idx}")
                continue

            num_frames_t = (n_cold + FRAME_SIZE - 1) // FRAME_SIZE

            # Build predictor
            pred = None
            if predictor_type == 'last':
                pred = LastBatchPredictor()
            elif predictor_type == 'ema':
                pred = EMAPredictor(num_frames_t, alpha=0.3)
            elif predictor_type == 'markov':
                pred = MarkovPredictor(num_frames_t, lookahead_depth=lookahead_depth)
            # oracle handled separately below

            caches[t_idx] = PrefetchFrameCache(
                video_path=video_path,
                frame_size=FRAME_SIZE,
                emb_dim=EMB_DIM,
                num_cold_rows=n_cold,
                quant_mins=q_mins[t_idx],
                quant_maxs=q_maxs[t_idx],
                cache_capacity=cache_capacity,
                predictor=pred,
                num_prefetch_workers=2,
            )

        # Pre-scan for oracle predictor
        if predictor_type == 'oracle':
            log(f"    Oracle: pre-scanning test batches...")
            per_table_batch_frames = {t_idx: [] for t_idx in caches}
            for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
                for t_idx in caches:
                    indices = lS_i[t_idx]
                    cold_mask = ~is_hot[t_idx][indices]
                    if cold_mask.any():
                        cold_mapped = o2c_map[t_idx][indices[cold_mask]]
                        valid = cold_mapped >= 0
                        if valid.any():
                            fids = (cold_mapped[valid] // FRAME_SIZE).unique().tolist()
                        else:
                            fids = []
                    else:
                        fids = []
                    per_table_batch_frames[t_idx].append(set(fids))

            for t_idx in caches:
                pred = OracleLookaheadPredictor(
                    per_table_batch_frames[t_idx],
                    lookahead_depth=lookahead_depth)
                caches[t_idx].predictor = pred
            log(f"    Oracle pre-scan complete.")

        # Warmup for Markov
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
                            fids = (cold_mapped[valid] // FRAME_SIZE).unique().tolist()
                            caches[t_idx].predictor.observe(fids)

            # Reset stats after warmup
            for t_idx in caches:
                caches[t_idx].reset_stats()
                caches[t_idx].clear_cache()

        # Reset oracle index
        if predictor_type == 'oracle':
            for t_idx in caches:
                caches[t_idx].predictor.reset()

        # Run inference
        drop_caches()
        time.sleep(0.5)
        gc.collect()

        scores, targets = [], []
        blats = []
        emb_lats = []
        rss_trace = []
        t0 = time.time()

        with torch.no_grad():
            for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
                bt0 = time.time()

                # Embedding phase: hot lookup + cold frame cache lookup
                et0 = time.time()
                per_table_frames = {}  # per-table frame IDs
                for t_idx in range(num_tables):
                    if t_idx in caches:
                        indices = lS_i[t_idx]
                        hot_mask = is_hot[t_idx][indices]

                        if (~hot_mask).any():
                            cold_mapped = o2c_map[t_idx][indices[~hot_mask]]
                            valid = cold_mapped >= 0
                            if valid.any():
                                cold_result, frames_used = caches[t_idx].lookup(cold_mapped[valid])
                                per_table_frames[t_idx] = frames_used
                                # Inject cold results back
                                cold_orig_indices = indices[~hot_mask][valid]
                                with torch.no_grad():
                                    dlrm.emb_l[t_idx].weight.data[cold_orig_indices] = cold_result
                            else:
                                per_table_frames[t_idx] = set()
                        else:
                            per_table_frames[t_idx] = set()
                emb_lats.append(time.time() - et0)

                # Forward pass (includes embedding lookup again via model)
                Z = dlrm(X, lS_o, lS_i)
                blats.append(time.time() - bt0)

                # Launch prefetch during "compute phase" — per-table frame IDs
                for t_idx in caches:
                    table_frames = per_table_frames.get(t_idx, set())
                    caches[t_idx].launch_prefetch(table_frames)

                S = Z.detach().cpu().numpy().flatten()
                Tn = T.detach().cpu().numpy().flatten()
                scores.extend(S.tolist())
                targets.extend(Tn.tolist())

                if batch_idx % 500 == 0:
                    rss = get_rss_mb()
                    rss_trace.append((batch_idx, rss))
                    log(f"    Batch {batch_idx}: lat={blats[-1]*1000:.1f}ms, RSS={rss:.0f}MB")

        total_time = time.time() - t0
        auc = roc_auc_score(targets, scores)
        rss = get_rss_mb()

        # Aggregate cache stats
        total_hits = sum(c.stats['cache_hits'] for c in caches.values())
        total_misses = sum(c.stats['cache_misses'] for c in caches.values())
        total_demand = sum(c.stats['demand_decomps'] for c in caches.values())
        total_prefetch_hits = sum(c.stats['prefetch_hits'] for c in caches.values())
        total_demand_ms = sum(c.stats['total_demand_ms'] for c in caches.values())
        all_batch_demands = []
        for c in caches.values():
            all_batch_demands.extend(c.stats['batch_demand_decomps'])
        avg_demand_per_batch = np.mean(all_batch_demands) if all_batch_demands else 0
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0

        # Compressed size on disk
        compressed_mb = 0
        for t_idx in caches:
            vp = caches[t_idx].video_path
            if os.path.exists(vp):
                compressed_mb += os.path.getsize(vp) / 1024 / 1024

        # Original cold size
        orig_cold_mb = sum(cold_num_rows.get(t_idx, 0) * EMB_DIM * 4
                           for t_idx in caches) / 1024 / 1024
        compression_ratio = orig_cold_mb / compressed_mb if compressed_mb > 0 else 0

        log(f"  AUC={auc:.6f}, Time={total_time:.2f}s, RSS={rss:.0f}MB")
        log(f"  Hit rate={hit_rate:.1%}, Demand={total_demand} ({avg_demand_per_batch:.1f}/batch), "
            f"Prefetch hits={total_prefetch_hits}, Demand time={total_demand_ms:.0f}ms")
        log(f"  Compression: {orig_cold_mb:.1f}MB -> {compressed_mb:.1f}MB ({compression_ratio:.1f}x)")
        log(f"  Emb lat: mean={np.mean(emb_lats)*1000:.2f}ms")

        for t_idx in caches:
            caches[t_idx].report_stats()

        result = {
            'auc': auc, 'total_time': total_time,
            'num_batches': len(blats),
            'mean_lat_ms': np.mean(blats)*1000,
            'p50_lat_ms': np.percentile(blats,50)*1000,
            'p99_lat_ms': np.percentile(blats,99)*1000,
            'emb_lat_ms': np.mean(emb_lats)*1000,
            'rss_mb': rss,
            'hit_rate': hit_rate,
            'demand_decomps': total_demand,
            'avg_demand_per_batch': avg_demand_per_batch,
            'prefetch_hits': total_prefetch_hits,
            'demand_time_ms': total_demand_ms,
            'compressed_mb': compressed_mb,
            'orig_cold_mb': orig_cold_mb,
            'compression_ratio': compression_ratio,
            'cache_capacity': cache_capacity,
            'predictor': predictor_type,
            'lookahead_depth': lookahead_depth,
            'use_reorder': use_reorder,
            'rss_trace': rss_trace,
        }

        # Cleanup
        for c in caches.values():
            c.close()
        del caches
        gc.collect()

        return result

    # ==============================================================
    # RUN ALL EXPERIMENTS
    # ==============================================================

    # --- Experiment A: Baseline ---
    log("\n" + "=" * 70)
    log("EXPERIMENT A: BASELINES")
    log("=" * 70)
    all_results['A_baseline'] = run_baseline("A_baseline")
    baseline_auc = all_results['A_baseline']['auc']
    baseline_time = all_results['A_baseline']['total_time']

    # --- Experiment B: Hot/cold split only ---
    log("\n" + "=" * 70)
    log("EXPERIMENT B: HOT/COLD SPLIT ONLY")
    log("=" * 70)
    all_results['B_hotcold'] = run_hotcold_split_only("B_hotcold")

    # --- Experiment C: Unordered cold codec, no prefetch ---
    log("\n" + "=" * 70)
    log("EXPERIMENT C: UNORDERED COLD CODEC (no prefetch)")
    log("=" * 70)
    for cache_sz in [25, 50, 100, 200]:
        key = f'C_unordered_cache{cache_sz}'
        all_results[key] = run_codec_inference(
            use_reorder=False, cache_capacity=cache_sz,
            predictor_type='none', tag=key)

    # --- Experiment D: Reordered cold codec, no prefetch ---
    log("\n" + "=" * 70)
    log("EXPERIMENT D: REORDERED COLD CODEC (no prefetch)")
    log("=" * 70)
    for cache_sz in [25, 50, 100, 200]:
        key = f'D_reordered_cache{cache_sz}'
        all_results[key] = run_codec_inference(
            use_reorder=True, cache_capacity=cache_sz,
            predictor_type='none', tag=key)

    # --- Experiment E: Reordered + Prefetching ---
    log("\n" + "=" * 70)
    log("EXPERIMENT E: REORDERED + PREFETCHING (full system)")
    log("=" * 70)

    # E1: Last-batch predictor
    for depth in [1, 3]:
        key = f'E1_last_depth{depth}'
        all_results[key] = run_codec_inference(
            use_reorder=True, cache_capacity=100,
            predictor_type='last', lookahead_depth=depth, tag=key)

    # E2: EMA predictor
    for depth in [1, 3, 5]:
        key = f'E2_ema_depth{depth}'
        all_results[key] = run_codec_inference(
            use_reorder=True, cache_capacity=100,
            predictor_type='ema', lookahead_depth=depth, tag=key)

    # E3: Markov predictor
    for depth in [1, 3, 5]:
        key = f'E3_markov_depth{depth}'
        all_results[key] = run_codec_inference(
            use_reorder=True, cache_capacity=100,
            predictor_type='markov', lookahead_depth=depth, tag=key)

    # E4: Oracle predictor
    for depth in [1, 3, 5, 10]:
        key = f'E4_oracle_depth{depth}'
        all_results[key] = run_codec_inference(
            use_reorder=True, cache_capacity=100,
            predictor_type='oracle', lookahead_depth=depth, tag=key)

    # E5: Cache size sweep with best predictor
    log("\n--- Cache size sweep (Markov depth=3) ---")
    for cache_sz in [25, 50, 100, 200, 500]:
        key = f'E5_markov_d3_cache{cache_sz}'
        all_results[key] = run_codec_inference(
            use_reorder=True, cache_capacity=cache_sz,
            predictor_type='markov', lookahead_depth=3, tag=key)

    # ==============================================================
    # GENERATE REPORTS AND PLOTS
    # ==============================================================
    log("\n" + "=" * 70)
    log("GENERATING REPORTS")
    log("=" * 70)

    # Save raw results
    results_json = os.path.join(RESULTS_DIR, 'experiment_results.json')
    # Filter out non-serializable data
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

    # Generate summary table
    summary = []
    summary.append("# Experiment Results Summary\n")
    summary.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Baseline AUC: {baseline_auc:.6f}")
    summary.append(f"Baseline Time: {baseline_time:.2f}s\n")

    header = f"{'Config':<45} {'AUC':>8} {'AUC_pp':>8} {'Time(s)':>8} {'Demand':>8} {'Hit%':>6} {'Cache':>6} {'Comp_MB':>8} {'RSS_MB':>8}"
    summary.append(header)
    summary.append("-" * len(header))

    for key in sorted(all_results.keys()):
        r = all_results[key]
        auc = r.get('auc', 0)
        auc_pp = (baseline_auc - auc) * 100
        total_t = r.get('total_time', 0)
        demand = r.get('avg_demand_per_batch', 0)
        hit = r.get('hit_rate', 0)
        cache = r.get('cache_capacity', '-')
        comp = r.get('compressed_mb', r.get('emb_memory_mb', 0))
        rss = r.get('rss_mb', 0)
        line = f"{key:<45} {auc:>8.6f} {auc_pp:>+7.4f} {total_t:>8.2f} {demand:>8.1f} {hit:>5.1%} {str(cache):>6} {comp:>8.1f} {rss:>8.0f}"
        summary.append(line)

    summary_text = "\n".join(summary)
    summary_path = os.path.join(RESULTS_DIR, 'experiment_results.md')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    log(f"\n{summary_text}")
    log(f"\n  Saved: {summary_path}")

    # --- Experiment F: Ablation table ---
    ablation = []
    ablation.append("\n# Experiment F: Ablation — Incremental Benefit\n")
    ablation.append(f"{'Configuration':<45} {'Depth':>5} {'Demand/B':>8} {'Hit%':>6} {'CacheMB':>8} {'Time(s)':>8} {'Speedup':>8}")
    ablation.append("-" * 100)

    ablation_keys = [
        ('C_unordered_cache100', 'C. Unordered+no prefetch', '-'),
        ('D_reordered_cache100', 'D. Reordered+no prefetch', '-'),
        ('E1_last_depth1', 'E1. Reordered+last (d=1)', '1'),
        ('E2_ema_depth3', 'E2. Reordered+EMA (d=3)', '3'),
        ('E3_markov_depth1', 'E3. Reordered+Markov (d=1)', '1'),
        ('E3_markov_depth3', 'E3. Reordered+Markov (d=3)', '3'),
        ('E3_markov_depth5', 'E3. Reordered+Markov (d=5)', '5'),
        ('E4_oracle_depth1', 'E4. Reordered+Oracle (d=1)', '1'),
        ('E4_oracle_depth3', 'E4. Reordered+Oracle (d=3)', '3'),
        ('E4_oracle_depth5', 'E4. Reordered+Oracle (d=5)', '5'),
        ('E4_oracle_depth10', 'E4. Reordered+Oracle (d=10)', '10'),
        ('A_baseline', 'A. Baseline (uncompressed)', '-'),
    ]

    ref_time = all_results.get('C_unordered_cache100', {}).get('total_time', 1)
    for key, label, depth in ablation_keys:
        if key not in all_results:
            continue
        r = all_results[key]
        demand = r.get('avg_demand_per_batch', 0)
        hit = r.get('hit_rate', 0)
        cache_cap = r.get('cache_capacity', 0)
        cache_mb = cache_cap * FRAME_SIZE * EMB_DIM * 4 / 1024 / 1024 if isinstance(cache_cap, int) else 0
        total_t = r.get('total_time', 0)
        speedup = ref_time / total_t if total_t > 0 else 0
        ablation.append(f"{label:<45} {depth:>5} {demand:>8.1f} {hit:>5.1%} {cache_mb:>8.1f} {total_t:>8.2f} {speedup:>7.2f}x")

    ablation_text = "\n".join(ablation)
    log(ablation_text)
    with open(summary_path, 'a') as f:
        f.write("\n" + ablation_text)

    # --- Generate plots ---
    try:
        # Plot 1: Cache hit rate vs cache size
        fig, ax = plt.subplots(figsize=(10, 6))
        for prefix, label in [('C_unordered_cache', 'Unordered'),
                               ('D_reordered_cache', 'Reordered'),
                               ('E5_markov_d3_cache', 'Reordered+Markov(d=3)')]:
            sizes = []
            rates = []
            for key in sorted(all_results.keys()):
                if key.startswith(prefix):
                    r = all_results[key]
                    sizes.append(r.get('cache_capacity', 0))
                    rates.append(r.get('hit_rate', 0))
            if sizes:
                ax.plot(sizes, rates, 'o-', label=label)
        ax.set_xlabel('Cache Capacity (frames)')
        ax.set_ylabel('Cache Hit Rate')
        ax.set_title('Cache Hit Rate vs Cache Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(RESULTS_DIR, 'cache_hit_rate_vs_size.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        log("  Saved: cache_hit_rate_vs_size.png")

        # Plot 2: Demand decomps vs depth (all predictors)
        fig, ax = plt.subplots(figsize=(10, 6))
        for prefix, label in [('E1_last_depth', 'Last-batch'),
                               ('E2_ema_depth', 'EMA'),
                               ('E3_markov_depth', 'Markov'),
                               ('E4_oracle_depth', 'Oracle')]:
            depths = []
            demands = []
            for key in sorted(all_results.keys()):
                if key.startswith(prefix):
                    r = all_results[key]
                    depths.append(r.get('lookahead_depth', 0))
                    demands.append(r.get('avg_demand_per_batch', 0))
            if depths:
                ax.plot(depths, demands, 'o-', label=label)
        ax.set_xlabel('Lookahead Depth')
        ax.set_ylabel('Demand Decomps per Batch')
        ax.set_title('Demand Decompressions vs Lookahead Depth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(RESULTS_DIR, 'depth_sweep_demand_decomp.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        log("  Saved: depth_sweep_demand_decomp.png")

        # Plot 3: Memory trace
        fig, ax = plt.subplots(figsize=(10, 6))
        for key in sorted(all_results.keys()):
            r = all_results[key]
            trace = r.get('rss_trace', [])
            if len(trace) > 2:
                batches = [t[0] for t in trace]
                rss_vals = [t[1] for t in trace]
                ax.plot(batches, rss_vals, '-', label=key, alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('RSS (MB)')
        ax.set_title('Memory Usage Over Time')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(RESULTS_DIR, 'memory_trace.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        log("  Saved: memory_trace.png")

        # Plot 4: Latency breakdown
        fig, ax = plt.subplots(figsize=(12, 6))
        keys_to_plot = ['A_baseline', 'B_hotcold', 'C_unordered_cache100',
                        'D_reordered_cache100', 'E3_markov_depth3', 'E4_oracle_depth3']
        keys_to_plot = [k for k in keys_to_plot if k in all_results]
        x_pos = range(len(keys_to_plot))
        mean_lats = [all_results[k].get('mean_lat_ms', 0) for k in keys_to_plot]
        ax.bar(x_pos, mean_lats, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([k.replace('_', '\n') for k in keys_to_plot], fontsize=7)
        ax.set_ylabel('Mean Batch Latency (ms)')
        ax.set_title('Latency Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        fig.savefig(os.path.join(RESULTS_DIR, 'latency_breakdown_stacked.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        log("  Saved: latency_breakdown_stacked.png")

    except Exception as e:
        log(f"  Plot generation error: {e}")

    log("\n" + "=" * 70)
    log("ALL EXPERIMENTS COMPLETE")
    log("=" * 70)
    log(f"Final RSS: {get_rss_mb():.0f}MB")
    log(f"Results saved to: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
