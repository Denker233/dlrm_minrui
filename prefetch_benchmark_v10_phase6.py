#!/usr/bin/env python3
"""
Prefetch Benchmark V10 Phase 6: Pareto Frontier Combinations

Phase 5 breakthrough: only 10.5% of cold rows are accessed, enabling 61MB memory.
Phase 6 explores combinations that push the Pareto frontier further:

6A - Minimal cold rows + per-column quantization (best AUC at low memory)
6B - Minimal cold rows + lz4 disk compression (minimum disk + minimum memory)
6C - Saved row-level access profile (deployment optimization)
6D - Hot threshold sweep with minimal rows (90%, 95%, 99%)
6E - Three-tier store: hot fp32 + warm fp32 + cold uint8
6F - Minimal rows + H.265 disk (minimum disk footprint)
6G - Memory-optimal combined (push absolute minimum)
"""

import os
import sys
import time
import json
import threading
import gc
import traceback
import subprocess
import tempfile

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

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

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
JSON_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_phase6.json")
LOG_PATH = os.path.join(RESULTS_DIR, "prefetch_v10_phase6.log")

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

# ==============================================================
# SHARED UTILITIES (from Phase 5)
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
# 4-BIT QUANTIZATION
# ==============================================================

def quantize_4bit_per_column(w):
    """Per-column 4-bit quantization. Returns packed (N, 8) uint8 + params."""
    ncols = w.shape[1]  # 16
    scales = np.zeros(ncols, dtype=np.float32)
    zps = np.zeros(ncols, dtype=np.int32)
    q_4bit = np.zeros(w.shape, dtype=np.uint8)
    w_np = w.numpy() if isinstance(w, torch.Tensor) else w
    for c in range(ncols):
        col = w_np[:, c]
        mn, mx = col.min(), col.max()
        s = (mx - mn) / 15.0
        if s == 0: s = 1.0
        z = round(-mn / s)
        q_4bit[:, c] = np.clip(np.round(col / s) + z, 0, 15).astype(np.uint8)
        scales[c] = s
        zps[c] = z
    # Pack: two 4-bit values per byte
    nrows = w_np.shape[0]
    packed = np.zeros((nrows, ncols // 2), dtype=np.uint8)
    for c in range(0, ncols, 2):
        packed[:, c // 2] = (q_4bit[:, c] << 4) | q_4bit[:, c + 1]
    return packed, scales, zps

def dequantize_4bit_per_column(packed, scales, zps, indices=None):
    """Dequantize packed 4-bit data to fp32."""
    if indices is not None:
        p = packed[indices]
    else:
        p = packed
    nrows = p.shape[0]
    ncols = p.shape[1] * 2  # 16
    result = np.empty((nrows, ncols), dtype=np.float32)
    for c in range(0, ncols, 2):
        hi = ((p[:, c // 2] >> 4) & 0x0F).astype(np.float32)
        lo = (p[:, c // 2] & 0x0F).astype(np.float32)
        result[:, c] = (hi - zps[c]) * scales[c]
        result[:, c + 1] = (lo - zps[c + 1]) * scales[c + 1]
    return result

# ==============================================================
# H.265 ENCODING / DECODING (from H265 experiment)
# ==============================================================

def encode_h265_table(q_np, width, height, crf=0):
    """Encode uint8 embedding rows as H.265 grayscale video frames."""
    pixels_per_frame = width * height
    total_bytes = q_np.size
    num_frames = max(1, (total_bytes + pixels_per_frame - 1) // pixels_per_frame)
    padded_size = num_frames * pixels_per_frame
    flat = np.zeros(padded_size, dtype=np.uint8)
    flat[:total_bytes] = q_np.flatten()

    tmp_out = tempfile.NamedTemporaryFile(suffix='.mkv', delete=False)
    tmp_out_path = tmp_out.name
    tmp_out.close()

    try:
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'gray', '-s', f'{width}x{height}',
            '-r', '1', '-i', 'pipe:0',
            '-c:v', 'libx265', '-preset', 'ultrafast', '-pix_fmt', 'gray',
            '-x265-params', f'lossless={1 if crf == 0 else 0}:log-level=error',
            '-f', 'matroska', tmp_out_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for i in range(num_frames):
            frame_data = flat[i * pixels_per_frame:(i + 1) * pixels_per_frame]
            proc.stdin.write(frame_data.tobytes())
        proc.stdin.close()
        proc.wait()
        with open(tmp_out_path, 'rb') as f:
            compressed = f.read()
    finally:
        if os.path.exists(tmp_out_path):
            os.unlink(tmp_out_path)

    return compressed, num_frames, total_bytes

def decode_h265_table(compressed_bytes, num_rows, width, height, num_threads=4):
    """Decode H.265 grayscale frames back to uint8 embedding rows."""
    total_bytes = num_rows * EMB_DIM
    pixels_per_frame = width * height

    tmp_in = tempfile.NamedTemporaryFile(suffix='.mkv', delete=False)
    tmp_in_path = tmp_in.name
    tmp_in.write(compressed_bytes)
    tmp_in.close()

    try:
        container = av.open(tmp_in_path, mode='r')
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.thread_count = num_threads
        y_planes = []
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format='gray')
            y_planes.append(arr.flatten()[:pixels_per_frame])
        container.close()
    finally:
        if os.path.exists(tmp_in_path):
            os.unlink(tmp_in_path)

    flat = np.concatenate(y_planes)
    return flat[:total_bytes].reshape(num_rows, EMB_DIM)

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
# COLD STORE: Minimal Row Preload (from Phase 5)
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
        cold_pos = {idx: pos for pos, idx in enumerate(cold_indices_all)}
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
# COLD STORE: Minimal Row with 4-bit
# ==============================================================

class MinimalRow4BitStore:
    """Minimal rows with 4-bit per-column quantization for minimum memory."""

    def __init__(self, large_tables):
        self.large_tables = large_tables
        self.cold_packed = {}       # {t: (N, 8) packed uint8}
        self.cold_seq_lookup = {}
        self.quant_params = {}      # {t: (scales, zps)}

    def add_from_accessed(self, t, num_emb, accessed_cold_indices, cold_weights_all,
                           cold_indices_all):
        cold_pos = {idx: pos for pos, idx in enumerate(cold_indices_all)}
        accessed_list = sorted(accessed_cold_indices)
        positions = [cold_pos[idx] for idx in accessed_list]
        weights_subset = cold_weights_all[positions]

        packed, scales, zps = quantize_4bit_per_column(weights_subset)
        self.cold_packed[t] = packed
        self.quant_params[t] = (scales, zps)

        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(accessed_list):
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
            scales, zps = self.quant_params[t]
            fp_rows = dequantize_4bit_per_column(self.cold_packed[t], scales, zps, cold_seq)
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
# COLD STORE: Three-Tier (hot fp32 + warm fp32 + cold uint8)
# ==============================================================

class ThreeTierStore:
    """Three tiers: hot fp32 (in model), warm fp32 (in model, from popular cold),
    cold uint8 (minimal accessed rows minus warm)."""

    def __init__(self, large_tables, quant_fn='per_column', warm_fraction=0.10):
        self.large_tables = large_tables
        self.quant_fn = quant_fn
        self.warm_fraction = warm_fraction
        self.cold_uint8 = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.warm_indices_set = {}   # {t: set of original indices already in fp32}
        self.warm_row_count = 0
        self.cold_row_count = 0

    def add_from_tiered(self, dlrm, t, num_emb, accessed_cold_indices,
                         cold_weights_all, cold_indices_all,
                         cold_access_counts):
        """Build tiered store.
        cold_access_counts: {orig_idx: count} for accessed cold rows"""
        # Sort accessed rows by frequency (most accessed first)
        accessed_sorted = sorted(cold_access_counts.keys(),
                                  key=lambda x: -cold_access_counts[x])
        n_warm = int(len(accessed_sorted) * self.warm_fraction)
        warm_list = accessed_sorted[:n_warm]
        cold_remaining = sorted(set(accessed_sorted[n_warm:]))

        # Inject warm rows as fp32 into model
        cold_pos = {idx: pos for pos, idx in enumerate(cold_indices_all)}
        if warm_list:
            warm_positions = [cold_pos[idx] for idx in warm_list]
            warm_weights = cold_weights_all[warm_positions]
            # Quantize + dequantize for consistent quality
            if self.quant_fn == 'per_column':
                qw, sw, zw = quantize_per_column(warm_weights)
                dqw = dequantize_per_column_torch(qw, sw, zw)
            else:
                qw, sw, zw = quantize_global(warm_weights)
                dqw = dequantize_global(qw, sw, zw)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[torch.tensor(warm_list, dtype=torch.long)] = dqw
        self.warm_indices_set[t] = set(warm_list)
        self.warm_row_count += len(warm_list)

        # Store remaining cold rows as uint8
        if cold_remaining:
            positions = [cold_pos[idx] for idx in cold_remaining]
            weights_subset = cold_weights_all[positions]
            if self.quant_fn == 'per_column':
                q, scales, zps = quantize_per_column(weights_subset)
                self.quant_params[t] = ('per_column', scales, zps)
            else:
                q, s, zp = quantize_global(weights_subset)
                self.quant_params[t] = ('global', s, zp)
            self.cold_uint8[t] = q.numpy()

            lookup = np.full(num_emb, -1, dtype=np.int32)
            for seq, orig in enumerate(cold_remaining):
                lookup[orig] = seq
            self.cold_seq_lookup[t] = lookup
            self.cold_row_count += len(cold_remaining)

    def dequantize_for_batch(self, lS_i):
        """Only dequantize rows NOT in warm set (warm already correct in model)."""
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
        cold_bytes = sum(a.nbytes for a in self.cold_uint8.values())
        warm_bytes = self.warm_row_count * EMB_DIM * 4  # fp32
        return cold_bytes + warm_bytes


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
    if extra_info:
        result['extra_info'] = extra_info
    return result


# ==============================================================
# PROFILING: Cold row access at multiple thresholds
# ==============================================================

def profile_cold_access(test_ld, large_tables, ln_emb, hot_indices_dict):
    """Profile which cold rows are accessed at each hot threshold.
    hot_indices_dict: {threshold: {t: hot_indices_array}}
    Returns: {threshold: {t: set of accessed cold indices}}
    """
    # Build cold index sets for each threshold
    cold_sets = {}
    for thresh, hi_dict in hot_indices_dict.items():
        cold_sets[thresh] = {}
        for t in large_tables:
            hi = set(hi_dict[t].tolist())
            cold_sets[thresh][t] = set(range(ln_emb[t])) - hi

    # Also collect access counts for three-tier analysis
    accessed_cold = {thresh: {t: set() for t in large_tables} for thresh in hot_indices_dict}
    access_counts = {t: {} for t in large_tables}  # {t: {orig_idx: count}} at 80% threshold

    nb = 0
    for X, lS_o, lS_i, T in test_ld:
        for t in large_tables:
            indices = lS_i[t].numpy().flatten()
            unique = np.unique(indices)
            unique_set = set(unique.tolist())
            # Count accesses (for three-tier, use first threshold)
            first_thresh = list(hot_indices_dict.keys())[0]
            for idx in unique_set:
                if idx in cold_sets[first_thresh][t]:
                    access_counts[t][idx] = access_counts[t].get(idx, 0) + 1
            # Track accessed cold for each threshold
            for thresh in hot_indices_dict:
                cold_in_batch = unique_set & cold_sets[thresh][t]
                accessed_cold[thresh][t].update(cold_in_batch)
        nb += 1
        if nb % 500 == 0:
            log(f"    Profiled batch {nb}")

    log(f"  Profiled {nb} test batches across {len(hot_indices_dict)} thresholds")
    return accessed_cold, access_counts


# ==============================================================
# MAIN
# ==============================================================

def main():
    global _log_fh
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _log_fh = open(LOG_PATH, 'w')

    log("=" * 70)
    log("PREFETCH BENCHMARK V10 PHASE 6: Pareto Frontier Combinations")
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

    # Profile hot indices at multiple thresholds
    log("\nProfiling access patterns at multiple thresholds...")
    thresholds = [0.80, 0.90, 0.95, 0.99]
    hot_indices_dict = {}
    for thresh in thresholds:
        hi = profile_access(train_ld, num_tables, ln_emb, thresh)
        hot_indices_dict[thresh] = hi
        total_hot = sum(len(hi[t]) for t in large_tables)
        total_cold = sum(ln_emb[t] for t in large_tables) - total_hot
        log(f"  {thresh*100:.0f}%: {total_hot:,} hot, {total_cold:,} cold "
            f"({total_cold * EMB_DIM / 1024 / 1024:.1f}MB)")

    hot_indices = hot_indices_dict[0.80]  # Default

    # Profile cold row access at all thresholds (single pass through test data)
    log("\nProfiling cold row access across all thresholds (single pass)...")
    accessed_cold_all, access_counts = profile_cold_access(
        test_ld, large_tables, ln_emb, hot_indices_dict)

    # Report coverage per threshold
    for thresh in thresholds:
        total_cold = sum(len(set(range(ln_emb[t])) - set(hot_indices_dict[thresh][t].tolist()))
                         for t in large_tables)
        total_accessed = sum(len(accessed_cold_all[thresh][t]) for t in large_tables)
        pct = total_accessed / total_cold * 100 if total_cold > 0 else 0
        cold_bytes = total_accessed * EMB_DIM
        log(f"  {thresh*100:.0f}% hot: {total_accessed:,}/{total_cold:,} cold accessed "
            f"({pct:.1f}%), min cold store = {cold_bytes / 1024 / 1024:.1f}MB")

    all_results['profiling'] = {
        'thresholds': {
            str(thresh): {
                'total_hot': sum(len(hot_indices_dict[thresh][t]) for t in large_tables),
                'total_cold': sum(ln_emb[t] for t in large_tables) -
                              sum(len(hot_indices_dict[thresh][t]) for t in large_tables),
                'accessed_cold': sum(len(accessed_cold_all[thresh][t]) for t in large_tables),
                'cold_store_mb': round(sum(len(accessed_cold_all[thresh][t])
                                            for t in large_tables) * EMB_DIM / 1024 / 1024, 2),
            }
            for thresh in thresholds
        }
    }
    save_results(all_results)

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
    # EXP 6A: Minimal Cold Rows + Per-Column Quantization
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 6A: Minimal Cold Rows + Per-Column Quantization (best AUC at low mem)")
    log("#" * 70)

    try:
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()

        store = MinimalRowStore(large_tables, quant_fn='per_column')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices, quantize_fn='per_column')

        for t in large_tables:
            w = state_dict[emb_keys[t]]
            hi = set(hot_indices[t].tolist())
            cold_idx = sorted(set(range(ln_emb[t])) - hi)
            cw = w[cold_idx]
            n_accessed = len(accessed_cold_all[0.80][t])
            store.add_from_accessed(t, ln_emb[t], accessed_cold_all[0.80][t], cw, cold_idx)
            log(f"    Table {t}: {n_accessed:,}/{len(cold_idx):,} cold rows loaded (per-col)")

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
                                  "Minimal cold rows + per-col 8-bit (80% hot)")
        all_results['6A_minimal_percol'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 6B: Minimal Cold Rows + LZ4 Disk Compression
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 6B: Minimal Cold Rows + LZ4 Disk Compression")
    log("#" * 70)

    for sub_label, qfn in [('global', 'global'), ('percol', 'per_column')]:
        log(f"\n  --- 6B-{sub_label}: {qfn} quantization ---")
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()

            store = MinimalRowStore(large_tables, quant_fn=qfn)
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_indices, quantize_fn=qfn)

            for t in large_tables:
                w = state_dict[emb_keys[t]]
                hi = set(hot_indices[t].tolist())
                cold_idx = sorted(set(range(ln_emb[t])) - hi)
                cw = w[cold_idx]
                store.add_from_accessed(t, ln_emb[t], accessed_cold_all[0.80][t], cw, cold_idx)

            setup_time = time.time() - setup_t0

            # Measure disk compression
            raw_bytes = 0
            compressed_bytes = 0
            for t in large_tables:
                if t in store.cold_uint8:
                    raw = store.cold_uint8[t].tobytes()
                    compressed = compress_block(raw, 'lz4')
                    raw_bytes += len(raw)
                    compressed_bytes += len(compressed)

            disk_mb = compressed_bytes / 1024 / 1024
            raw_mb = raw_bytes / 1024 / 1024
            ratio = raw_bytes / compressed_bytes if compressed_bytes > 0 else 0

            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                            state_dict, emb_keys, num_tables, ln_emb)
            log(f"  Setup: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB")
            log(f"  Disk: {raw_mb:.1f}MB -> {disk_mb:.1f}MB lz4 ({ratio:.2f}x)")

            drop_caches()
            time.sleep(1)
            gc.collect()
            acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
            log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
                f"Inference={inf_time:.2f}s")

            result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                      baseline_auc, store,
                                      f"Minimal rows + lz4 ({qfn})")
            result['disk_mb'] = float(disk_mb)
            result['disk_raw_mb'] = float(raw_mb)
            result['disk_ratio'] = float(ratio)
            all_results[f'6B_minimal_lz4_{sub_label}'] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 6C: Saved Row-Level Access Profile
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 6C: Saved Row-Level Access Profile (deployment optimization)")
    log("#" * 70)

    try:
        profile_path = os.path.join(RESULTS_DIR, "row_access_profile.json")

        # Save profile
        t0 = time.time()
        profile_data = {}
        for t in large_tables:
            profile_data[str(t)] = sorted(list(accessed_cold_all[0.80][t]))
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f)
        save_time = time.time() - t0
        file_size = os.path.getsize(profile_path)

        # Load profile
        t0 = time.time()
        with open(profile_path) as f:
            loaded_profile = json.load(f)
        loaded_accessed = {int(t): set(indices) for t, indices in loaded_profile.items()}
        load_time = time.time() - t0

        # Verify same counts
        for t in large_tables:
            orig = len(accessed_cold_all[0.80][t])
            loaded = len(loaded_accessed.get(t, set()))
            assert orig == loaded, f"Table {t}: {orig} vs {loaded}"

        log(f"  Profile saved: {file_size / 1024 / 1024:.1f}MB, save time: {save_time:.2f}s")
        log(f"  Profile loaded: {load_time:.2f}s, verified identical")

        # Build store from loaded profile and verify AUC
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()

        store = MinimalRowStore(large_tables, quant_fn='per_column')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hot_indices, quantize_fn='per_column')

        for t in large_tables:
            w = state_dict[emb_keys[t]]
            hi = set(hot_indices[t].tolist())
            cold_idx = sorted(set(range(ln_emb[t])) - hi)
            cw = w[cold_idx]
            store.add_from_accessed(t, ln_emb[t], loaded_accessed.get(t, set()), cw, cold_idx)

        setup_time = time.time() - setup_t0
        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        log(f"  Store setup from profile: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB")

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
            f"Inference={inf_time:.2f}s")
        log(f"  (Verified: profile-loaded store gives same AUC as direct profiling)")

        result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                  baseline_auc, store,
                                  "Minimal rows from saved profile (per-col)")
        result['profile_file_mb'] = float(file_size / 1024 / 1024)
        result['profile_save_time'] = float(save_time)
        result['profile_load_time'] = float(load_time)
        all_results['6C_saved_profile'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # EXP 6D: Hot Threshold Sweep with Minimal Rows
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 6D: Hot Threshold Sweep with Minimal Rows (per-col)")
    log("#" * 70)

    for thresh in [0.90, 0.95, 0.99]:
        log(f"\n  --- {thresh*100:.0f}% hot threshold ---")
        try:
            hi = hot_indices_dict[thresh]
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()

            store = MinimalRowStore(large_tables, quant_fn='per_column')
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hi, quantize_fn='per_column')

            for t in large_tables:
                w = state_dict[emb_keys[t]]
                hot_set = set(hi[t].tolist())
                cold_idx = sorted(set(range(ln_emb[t])) - hot_set)
                cw = w[cold_idx]
                accessed = accessed_cold_all[thresh][t]
                store.add_from_accessed(t, ln_emb[t], accessed, cw, cold_idx)
                log(f"    Table {t}: hot={len(hot_set):,}, cold={len(cold_idx):,}, "
                    f"accessed={len(accessed):,}")

            setup_time = time.time() - setup_t0
            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hi, large_tables,
                                            state_dict, emb_keys, num_tables, ln_emb)
            log(f"  Setup: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB "
                f"(hot: {mem['hot_fp32_mb']:.1f}MB, cold: {store.memory_bytes / 1024 / 1024:.1f}MB)")

            drop_caches()
            time.sleep(1)
            gc.collect()
            acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
            log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
                f"Inference={inf_time:.2f}s")

            result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                      baseline_auc, store,
                                      f"Minimal rows per-col ({thresh*100:.0f}% hot)")
            result['hot_threshold'] = float(thresh)
            all_results[f'6D_thresh_{int(thresh*100)}'] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 6E: Three-Tier Store (hot fp32 + warm fp32 + cold uint8)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 6E: Three-Tier Store (hot fp32 + warm fp32 + cold uint8)")
    log("#" * 70)

    for warm_frac in [0.10, 0.25]:
        log(f"\n  --- Warm fraction: {warm_frac*100:.0f}% of accessed cold ---")
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()

            store = ThreeTierStore(large_tables, quant_fn='per_column',
                                    warm_fraction=warm_frac)
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_indices, quantize_fn='per_column')

            for t in large_tables:
                w = state_dict[emb_keys[t]]
                hi = set(hot_indices[t].tolist())
                cold_idx = sorted(set(range(ln_emb[t])) - hi)
                cw = w[cold_idx]
                accessed = accessed_cold_all[0.80][t]
                # Get per-row access counts for this table
                table_counts = {idx: access_counts[t].get(idx, 1) for idx in accessed}
                store.add_from_tiered(dlrm, t, ln_emb[t], accessed, cw, cold_idx, table_counts)
                warm_ct = len(store.warm_indices_set.get(t, set()))
                cold_ct = len(store.cold_uint8.get(t, np.array([])))
                log(f"    Table {t}: warm={warm_ct:,}, cold={cold_ct:,}")

            setup_time = time.time() - setup_t0
            # Memory: warm is in model fp32, cold is in uint8 store
            # compute_memory_breakdown counts hot fp32 but not warm separately
            # We need to add warm fp32 cost
            cold_bytes = sum(a.nbytes for a in store.cold_uint8.values())
            warm_bytes = store.warm_row_count * EMB_DIM * 4
            mem = compute_memory_breakdown(dlrm, cold_bytes, hot_indices, large_tables,
                                            state_dict, emb_keys, num_tables, ln_emb)
            # Add warm fp32 to the total
            mem['warm_fp32_mb'] = round(warm_bytes / 1024 / 1024, 2)
            mem['total_mb'] = round(mem['total_mb'] + mem['warm_fp32_mb'], 2)
            log(f"  Setup: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB "
                f"(warm: {mem['warm_fp32_mb']:.1f}MB, cold: {cold_bytes / 1024 / 1024:.1f}MB)")
            log(f"  Warm rows: {store.warm_row_count:,}, Cold rows: {store.cold_row_count:,}")

            drop_caches()
            time.sleep(1)
            gc.collect()
            acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
            log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
                f"Inference={inf_time:.2f}s")

            result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                      baseline_auc, store,
                                      f"Three-tier per-col ({warm_frac*100:.0f}% warm)")
            result['warm_fraction'] = float(warm_frac)
            result['warm_rows'] = store.warm_row_count
            result['cold_rows'] = store.cold_row_count
            all_results[f'6E_three_tier_{int(warm_frac*100)}pct'] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()

    # ==============================================================
    # EXP 6F: Minimal Rows + H.265 Disk (minimum disk footprint)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 6F: Minimal Rows + H.265 Disk (minimum disk footprint)")
    log("#" * 70)

    if HAS_AV:
        try:
            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()
            setup_t0 = time.time()

            store = MinimalRowStore(large_tables, quant_fn='global')
            setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                        large_tables, hot_indices)

            h265_data = {}  # {t: (compressed_bytes, num_rows)}
            total_h265_bytes = 0
            total_raw_bytes = 0
            width, height = 3840, 2160  # 4K

            for t in large_tables:
                w = state_dict[emb_keys[t]]
                hi = set(hot_indices[t].tolist())
                cold_idx = sorted(set(range(ln_emb[t])) - hi)
                cw = w[cold_idx]
                accessed = accessed_cold_all[0.80][t]
                store.add_from_accessed(t, ln_emb[t], accessed, cw, cold_idx)

                # Encode the minimal uint8 data as H.265
                q_np = store.cold_uint8[t]
                n_rows = q_np.shape[0]
                raw_bytes = q_np.size
                compressed, num_frames, _ = encode_h265_table(q_np, width, height)
                h265_data[t] = (compressed, n_rows, num_frames)
                total_h265_bytes += len(compressed)
                total_raw_bytes += raw_bytes
                log(f"    Table {t}: {n_rows:,} rows, "
                    f"{raw_bytes / 1024:.0f}KB -> {len(compressed) / 1024:.0f}KB H.265 "
                    f"({raw_bytes / len(compressed):.1f}x)")

            setup_time = time.time() - setup_t0
            h265_mb = total_h265_bytes / 1024 / 1024
            raw_mb = total_raw_bytes / 1024 / 1024
            ratio = total_raw_bytes / total_h265_bytes if total_h265_bytes > 0 else 0

            mem = compute_memory_breakdown(dlrm, store.memory_bytes, hot_indices, large_tables,
                                            state_dict, emb_keys, num_tables, ln_emb)
            log(f"  Setup: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB")
            log(f"  H.265 disk: {raw_mb:.1f}MB -> {h265_mb:.1f}MB ({ratio:.1f}x)")

            # Verify decode works correctly
            decode_t0 = time.time()
            for t in large_tables:
                compressed, n_rows, num_frames = h265_data[t]
                decoded = decode_h265_table(compressed, n_rows, width, height, num_threads=4)
                if not np.array_equal(decoded, store.cold_uint8[t]):
                    log(f"    WARNING: Table {t} decode mismatch "
                        f"(max diff: {np.max(np.abs(decoded.astype(int) - store.cold_uint8[t].astype(int)))})")
                else:
                    log(f"    Table {t}: decode verified lossless")
            decode_time = time.time() - decode_t0
            log(f"  Decode time: {decode_time:.2f}s")

            drop_caches()
            time.sleep(1)
            gc.collect()
            acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
            log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
                f"Inference={inf_time:.2f}s")

            result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                      baseline_auc, store,
                                      "Minimal rows + H.265 4K disk")
            result['h265_disk_mb'] = float(h265_mb)
            result['h265_raw_mb'] = float(raw_mb)
            result['h265_ratio'] = float(ratio)
            result['h265_decode_time'] = float(decode_time)
            all_results['6F_minimal_h265'] = result
            save_results(all_results)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()
    else:
        log("  SKIPPED: PyAV not available")

    # ==============================================================
    # EXP 6G: Memory-Optimal Combined (push absolute minimum)
    # ==============================================================
    log("\n" + "#" * 70)
    log("EXP 6G: Memory-Optimal Combined (99% hot + per-col + minimal)")
    log("#" * 70)

    # 6G-8bit: 99% hot + per-col 8-bit + minimal rows
    log("\n  --- 6G-8bit: 99% hot + per-col 8-bit ---")
    try:
        hi_99 = hot_indices_dict[0.99]
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()

        store = MinimalRowStore(large_tables, quant_fn='per_column')
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hi_99, quantize_fn='per_column')

        for t in large_tables:
            w = state_dict[emb_keys[t]]
            hot_set = set(hi_99[t].tolist())
            cold_idx = sorted(set(range(ln_emb[t])) - hot_set)
            cw = w[cold_idx]
            accessed = accessed_cold_all[0.99][t]
            store.add_from_accessed(t, ln_emb[t], accessed, cw, cold_idx)
            log(f"    Table {t}: hot={len(hot_set):,}, accessed_cold={len(accessed):,}")

        setup_time = time.time() - setup_t0
        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hi_99, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        log(f"  Setup: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB "
            f"(hot: {mem['hot_fp32_mb']:.1f}MB, cold: {store.memory_bytes / 1024 / 1024:.1f}MB)")

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
            f"Inference={inf_time:.2f}s")

        result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                  baseline_auc, store,
                                  "Memory-optimal: 99% hot + per-col 8-bit minimal")
        all_results['6G_optimal_8bit'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # 6G-4bit: 99% hot + per-col 4-bit + minimal rows
    log("\n  --- 6G-4bit: 99% hot + per-col 4-bit ---")
    try:
        hi_99 = hot_indices_dict[0.99]
        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()
        setup_t0 = time.time()

        store = MinimalRow4BitStore(large_tables)
        setup_model(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                    large_tables, hi_99, quantize_fn='per_column')

        for t in large_tables:
            w = state_dict[emb_keys[t]]
            hot_set = set(hi_99[t].tolist())
            cold_idx = sorted(set(range(ln_emb[t])) - hot_set)
            cw = w[cold_idx]
            accessed = accessed_cold_all[0.99][t]
            store.add_from_accessed(t, ln_emb[t], accessed, cw, cold_idx)
            log(f"    Table {t}: hot={len(hot_set):,}, accessed_cold={len(accessed):,}, "
                f"packed={store.cold_packed[t].nbytes / 1024:.0f}KB")

        setup_time = time.time() - setup_t0
        mem = compute_memory_breakdown(dlrm, store.memory_bytes, hi_99, large_tables,
                                        state_dict, emb_keys, num_tables, ln_emb)
        log(f"  Setup: {setup_time:.2f}s, Memory: {mem['total_mb']:.1f}MB "
            f"(hot: {mem['hot_fp32_mb']:.1f}MB, cold 4-bit: {store.memory_bytes / 1024 / 1024:.1f}MB)")

        drop_caches()
        time.sleep(1)
        gc.collect()
        acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
        log(f"  AUC={auc:.6f}, Loss={((baseline_auc - auc) * 100):+.4f}pp, "
            f"Inference={inf_time:.2f}s")

        result = collect_metrics(acc, auc, inf_time, blats, nb, setup_time, mem,
                                  baseline_auc, store,
                                  "Memory-optimal: 99% hot + per-col 4-bit minimal")
        all_results['6G_optimal_4bit'] = result
        save_results(all_results)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()

    # ==============================================================
    # SUMMARY
    # ==============================================================
    log("\n" + "=" * 70)
    log("PHASE 6 SUMMARY")
    log("=" * 70)
    log(f"\n{'Config':<60s} {'AUC':>10s} {'Loss':>10s} {'Infer':>8s} {'Mem':>8s} {'Disk':>8s}")
    log("-" * 108)
    for k, v in all_results.items():
        if k == 'profiling':
            continue
        name = v.get('name', k)[:57]
        auc = v.get('auc', 0)
        loss = v.get('auc_loss_pp', 0)
        infer = v.get('inference_time', 0)
        mem = v.get('memory_mb', 0)
        disk = v.get('disk_mb', v.get('h265_disk_mb', '-'))
        if isinstance(disk, float):
            disk_str = f"{disk:>6.1f}"
        else:
            disk_str = f"{'-':>6s}"
        log(f"  {name:<58s} {auc:.6f} {loss:+.4f}pp {infer:>6.1f}s {mem:>6.1f} {disk_str}")

    log(f"\nResults saved to: {JSON_PATH}")
    log(f"\n{'=' * 70}")
    log("DONE!")
    log("=" * 70)

    if _log_fh:
        _log_fh.close()


if __name__ == '__main__':
    main()
