#!/usr/bin/env python3
"""
Prefetch Benchmark V10 H.265: Software HEVC codec for embedding compression.

Encode embedding tables as H.265 video frames at different resolutions,
decode with configurable thread count, measure all metrics.

Grid:
  Resolutions: 1080p (1920x1080), 1440p (2560x1440), 4K (3840x2160)
  Thread counts: 1, 2, 4, 8
  + Baseline and V9 uint8 reference

Embedding data → uint8 quantize → reshape to grayscale frames → H.265 encode
H.265 decode → reshape back to rows → dequantize to fp32 → inject into model
"""

import os, sys, time, json, threading, gc, subprocess, io, tempfile
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlrm_data_pytorch as dp

try:
    import av
except ImportError:
    print("ERROR: PyAV not installed. Run: pip install av")
    sys.exit(1)

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

# Resolution configs: (name, width, height)
RESOLUTIONS = [
    ('1080p', 1920, 1080),
    ('1440p', 2560, 1440),
    ('4K', 3840, 2160),
]

THREAD_COUNTS = [1, 2, 4, 8]

# H.265 CRF for near-lossless quality on uint8 data
H265_CRF = 0  # lossless mode for x265

os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


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
# H.265 ENCODING / DECODING
# ==============================================================

def encode_h265_table(q_np, width, height, crf=0):
    """
    Encode uint8 embedding rows as H.265 grayscale video frames via ffmpeg.

    q_np: (N, 16) uint8 array
    Returns: (compressed_bytes, num_frames, raw_total_bytes, encode_time)
    """
    pixels_per_frame = width * height
    total_bytes = q_np.size  # N * 16
    num_frames = max(1, (total_bytes + pixels_per_frame - 1) // pixels_per_frame)
    padded_size = num_frames * pixels_per_frame
    flat = np.zeros(padded_size, dtype=np.uint8)
    flat[:total_bytes] = q_np.flatten()

    t0 = time.time()
    tmp_out = tempfile.NamedTemporaryFile(suffix='.mkv', delete=False)
    tmp_out_path = tmp_out.name
    tmp_out.close()

    try:
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', f'{width}x{height}',
            '-r', '1',
            '-i', 'pipe:0',
            '-c:v', 'libx265',
            '-preset', 'ultrafast',
            '-pix_fmt', 'gray',
            '-x265-params', f'lossless={1 if crf == 0 else 0}:log-level=error',
            '-f', 'matroska',
            tmp_out_path,
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

    encode_time = time.time() - t0
    return compressed, num_frames, total_bytes, encode_time


def decode_h265_table(compressed_bytes, num_rows, width, height, num_threads=1):
    """
    Decode H.265 grayscale frames back to uint8 embedding rows.
    Uses PyAV for thread-controllable decoding.

    Returns: (q_np, decode_time)
    """
    total_bytes = num_rows * EMB_DIM
    pixels_per_frame = width * height

    t0 = time.time()
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
    q_np = flat[:total_bytes].reshape(num_rows, EMB_DIM)

    decode_time = time.time() - t0
    return q_np, decode_time


# ==============================================================
# H.265 COLD STORE
# ==============================================================

class H265ColdStore:
    """
    H.265 compressed cold store. Each table encoded as a short H.265 video.
    Frames are decoded and cached on first access per block.
    """
    def __init__(self, large_tables, width=1920, height=1080, num_threads=1,
                 crf=0):
        self.large_tables = large_tables
        self.width = width
        self.height = height
        self.num_threads = num_threads
        self.crf = crf
        self.pixels_per_frame = width * height

        # Per-table data
        self.compressed_data = {}    # {t: compressed_bytes}
        self.cold_seq_lookup = {}    # {t: np.array (num_emb,) int32}
        self.quant_params = {}       # {t: (scale, zp)}
        self.num_cold_rows = {}      # {t: int}
        self.num_frames = {}         # {t: int}

        # Cache: decoded uint8 arrays (full table)
        self.decoded_cache = {}      # {t: np.array (num_cold, 16)}

        # Metrics
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.encode_time = 0.0
        self.decode_time = 0.0
        self.decode_count = 0

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (s, zp)
        self.num_cold_rows[t] = len(cold_indices)

        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

        # Encode as H.265
        compressed, num_frames, total_bytes, enc_time = encode_h265_table(
            q_np, self.width, self.height, self.crf)

        self.compressed_data[t] = compressed
        self.num_frames[t] = num_frames
        self.raw_bytes += total_bytes
        self.compressed_bytes += len(compressed)
        self.encode_time += enc_time

    def _ensure_decoded(self, t):
        """Decode table if not already cached."""
        if t in self.decoded_cache:
            return
        compressed = self.compressed_data[t]
        num_rows = self.num_cold_rows[t]
        q_np, dec_time = decode_h265_table(
            compressed, num_rows, self.width, self.height, self.num_threads)
        self.decoded_cache[t] = q_np
        self.decode_time += dec_time
        self.decode_count += 1

    def dequantize_for_batch(self, lS_i):
        result = {}
        for t in self.large_tables:
            if t not in self.compressed_data:
                continue
            indices = lS_i[t].numpy().flatten()
            unique = np.unique(indices)
            seq = self.cold_seq_lookup[t][unique]
            mask = seq >= 0
            if not mask.any():
                continue
            cold_orig = unique[mask]
            cold_seq = seq[mask]

            # Decode full table on first access
            self._ensure_decoded(t)

            q_rows = self.decoded_cache[t][cold_seq]
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
        return self.compressed_bytes

    @property
    def compression_ratio(self):
        if self.compressed_bytes == 0:
            return 0.0
        return self.raw_bytes / self.compressed_bytes

    @property
    def cache_size_bytes(self):
        return sum(a.nbytes for a in self.decoded_cache.values())


class H265PredecodeColdStore:
    """
    H.265 cold store that pre-decodes all tables at setup time.
    The compressed data is kept for memory accounting but the decoded
    cache is what's used at inference.
    """
    def __init__(self, large_tables, width=1920, height=1080, num_threads=1,
                 crf=0):
        self.large_tables = large_tables
        self.width = width
        self.height = height
        self.num_threads = num_threads
        self.crf = crf

        self.cold_uint8 = {}         # {t: np.array (num_cold, 16) decoded}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.num_cold_rows = {}

        # Keep compressed for size accounting
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.encode_time = 0.0
        self.decode_time = 0.0

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (s, zp)
        self.num_cold_rows[t] = len(cold_indices)

        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

        # Encode
        compressed, num_frames, total_bytes, enc_time = encode_h265_table(
            q_np, self.width, self.height, self.crf)
        self.raw_bytes += total_bytes
        self.compressed_bytes += len(compressed)
        self.encode_time += enc_time

        # Immediately decode back
        decoded, dec_time = decode_h265_table(
            compressed, len(cold_indices), self.width, self.height, self.num_threads)
        self.cold_uint8[t] = decoded
        self.decode_time += dec_time

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
        # Runtime memory = decoded uint8 arrays (same as V9)
        return sum(a.nbytes for a in self.cold_uint8.values())

    @property
    def disk_bytes(self):
        return self.compressed_bytes

    @property
    def compression_ratio(self):
        if self.compressed_bytes == 0:
            return 0.0
        return self.raw_bytes / self.compressed_bytes


# ==============================================================
# V9 REFERENCE STORE
# ==============================================================

class EntropyColdStore:
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


# ==============================================================
# PREFETCHER + INFERENCE
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
    log("PREFETCH BENCHMARK V10 H.265: HEVC Codec for Embedding Compression")
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
    # BASELINE
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
    log(f"  Batch: mean={np.mean(b_blats) * 1000:.1f}ms, p99={np.percentile(b_blats, 99) * 1000:.1f}ms")
    baseline_auc = b_auc
    baseline_acc = b_acc
    results['baseline'] = {
        'name': 'Baseline', 'accuracy': b_acc, 'auc': b_auc, 'auc_loss_pp': 0.0,
        'inference_time': b_time, 'setup_time': 0, 'total_time': b_time,
        'memory_mb': b_mem, 'batch_latency': latency_stats(b_blats),
    }

    # ================================================================
    # V9 REFERENCE
    # ================================================================
    log("\n" + "=" * 60)
    log("V9 REFERENCE — uint8, 80% hot")
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
    log(f"  Batch: mean={np.mean(blats) * 1000:.1f}ms, p99={np.percentile(blats, 99) * 1000:.1f}ms")

    results['v9_uint8'] = {
        'name': 'V9 uint8 (no H.265)', 'accuracy': acc, 'auc': auc,
        'auc_loss_pp': auc_loss,
        'inference_time': inf_time, 'setup_time': setup_time,
        'total_time': setup_time + inf_time,
        'memory_mb': mem['total_mb'], 'memory_breakdown': mem,
        'batch_latency': latency_stats(blats),
    }

    # ================================================================
    # H.265 CONFIGS: resolution × thread count (pre-decode at setup)
    # ================================================================
    log("\n" + "=" * 70)
    log("H.265 EXPERIMENTS: Pre-decode at setup, resolution × threads")
    log("=" * 70)

    for res_name, width, height in RESOLUTIONS:
        for nthreads in THREAD_COUNTS:
            key = f"h265_{res_name}_t{nthreads}"
            name = f"H.265 {res_name} threads={nthreads}"
            log(f"\n{'=' * 60}")
            log(f"{name}")
            log(f"{'=' * 60}")

            restore_weights(dlrm, state_dict, emb_keys)
            gc.collect()

            store = H265PredecodeColdStore(large_tables, width=width,
                                            height=height,
                                            num_threads=nthreads, crf=0)
            setup_t0 = time.time()
            setup_cold_store(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                             large_tables, hot_indices, store)
            setup_time = time.time() - setup_t0
            mem = compute_memory(store)

            log(f"  Setup: {setup_time:.2f}s (encode: {store.encode_time:.2f}s, "
                f"decode: {store.decode_time:.2f}s)")
            log(f"  Compression: {store.raw_bytes / 1024 / 1024:.1f}MB -> "
                f"{store.compressed_bytes / 1024 / 1024:.1f}MB "
                f"(ratio={store.compression_ratio:.2f}x)")
            log(f"  Disk size: {store.disk_bytes / 1024 / 1024:.1f}MB, "
                f"Runtime memory: {mem['cold_store_mb']:.1f}MB (decoded uint8)")

            drop_caches(); time.sleep(1); gc.collect()
            acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
            auc_loss = (baseline_auc - auc) * 100
            acc_loss = (baseline_acc - acc) * 100

            log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}")
            log(f"  AUC loss={auc_loss:.4f}pp, Acc loss={acc_loss:.4f}pp")
            log(f"  Inference={inf_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time + inf_time:.2f}s")
            log(f"  Batch: mean={np.mean(blats) * 1000:.1f}ms, "
                f"p50={np.percentile(blats, 50) * 1000:.1f}ms, "
                f"p99={np.percentile(blats, 99) * 1000:.1f}ms")

            results[key] = {
                'name': name, 'accuracy': acc, 'auc': auc,
                'auc_loss_pp': auc_loss, 'acc_loss_pp': acc_loss,
                'inference_time': inf_time, 'setup_time': setup_time,
                'total_time': setup_time + inf_time,
                'memory_mb': mem['total_mb'], 'memory_breakdown': mem,
                'disk_mb': store.disk_bytes / 1024 / 1024,
                'batch_latency': latency_stats(blats),
                'compression_ratio': store.compression_ratio,
                'encode_time_s': store.encode_time,
                'decode_time_s': store.decode_time,
                'resolution': res_name, 'width': width, 'height': height,
                'num_threads': nthreads,
            }

    # ================================================================
    # SUMMARY
    # ================================================================
    log("\n" + "=" * 70)
    log("SUMMARY — H.265 CODEC COMPARISON")
    log("=" * 70)

    fmt = "  %-35s  %8s  %10s  %10s  %10s  %10s  %10s  %8s  %10s  %10s"
    log(fmt % ("Config", "AUC", "AUC loss", "Acc%", "Infer(s)", "Setup(s)",
               "Total(s)", "Ratio", "Disk(MB)", "Mem(MB)"))
    log("  " + "-" * 145)
    for key in ['baseline', 'v9_uint8'] + [
        f"h265_{r}_{t}" for r, _, _ in RESOLUTIONS
        for t in [f"t{n}" for n in THREAD_COUNTS]
    ]:
        if key not in results:
            continue
        r = results[key]
        ratio = r.get('compression_ratio', '-')
        ratio_s = f"{ratio:.2f}x" if isinstance(ratio, float) else '-'
        disk = r.get('disk_mb', r.get('memory_mb', 0))
        log(fmt % (r['name'],
                   f"{r['auc']:.6f}", f"{r['auc_loss_pp']:.4f}pp",
                   f"{r['accuracy'] * 100:.4f}",
                   f"{r['inference_time']:.2f}", f"{r.get('setup_time', 0):.2f}",
                   f"{r['total_time']:.2f}",
                   ratio_s, f"{disk:.1f}", f"{r['memory_mb']:.1f}"))

    # CAFE+ comparison
    log("\n  --- CAFE+ Reference (from paper) ---")
    log("  CAFE+ Baseline:    AUC=0.8010, 1.0x compression")
    log("  CAFE+ 16x:         AUC=0.7882, 1.45pp loss")
    log("  CAFE+ 64x:         AUC=0.7769, 2.58pp loss")
    log("  CAFE+ 1000x:       AUC=0.7736, 2.74pp loss")

    json_path = os.path.join(RESULTS_DIR, 'prefetch_v10_h265.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Saved: {json_path}")

    log("\n" + "=" * 70)
    log("DONE!")
    log("=" * 70)


if __name__ == '__main__':
    main()
