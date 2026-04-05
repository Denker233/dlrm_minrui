#!/usr/bin/env python3
"""
Experiment: Warmup-based profiling sweep.

Splits test_ld into warmup (profiling) and eval (measurement) portions.
Sweeps warmup fractions: 5%, 10%, 20%, 50%, 100%.
Also tests train_ld profiling with subsampling and hybrid approaches.

Results saved to results/methodology_experiments/
"""
import os, sys, time, json, gc, io, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# ============================================================
# Config
# ============================================================
ARCH_SPARSE_FEATURE_SIZE = 16
EMB_DIM = 16
TILE_H, TILE_W = 4, 4
HOT_FRACTION = 0.043
LARGE_TABLE_THRESHOLD = 50000
MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
DATA_FILE = "./input/train.txt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 2048
RESULTS_DIR = "results/methodology_experiments"
CRF = 18
WIDTH, HEIGHT = 1920, 1080
ROWS_PER_FRAME = (WIDTH // TILE_W) * (HEIGHT // TILE_H)  # 129600
NUM_EVAL_BATCHES = 50  # batches for AUC/latency evaluation

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Import C++ extension
# ============================================================
try:
    import compressed_emb as _C
    HAS_CPP = True
except ImportError:
    try:
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if torch_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
            os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        import compressed_emb as _C
        HAS_CPP = True
    except ImportError:
        HAS_CPP = False
        _C = None

import subprocess
try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024


# ============================================================
# Model & Data Loading (from demo)
# ============================================================
def load_model_and_data():
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net

    class Args:
        pass
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

    log("Loading Kaggle dataset...")
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(a)
    ln_emb = np.array(train_data.counts)
    m_spa = a.arch_sparse_feature_size
    ln_bot = np.fromstring(a.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[-1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + a.arch_mlp_top, dtype=int, sep="-")

    dlrm = DLRM_Net(m_spa, ln_emb, ln_bot, ln_top,
                     arch_interaction_op="dot", arch_interaction_itself=False,
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")
    log("Loading model checkpoint...")
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ckpt["state_dict"])
    dlrm.eval()

    state_dict = ckpt["state_dict"]
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))

    log(f"Model loaded: {len(ln_emb)} tables, emb_dim={m_spa}")
    return dlrm, train_ld, test_ld, ln_emb, state_dict, emb_keys


# ============================================================
# Profiling
# ============================================================
def profile_frequency(batches, ln_emb, large_tables, max_batches=0):
    """Count access frequency per row. Returns freq dict and first_batch dict."""
    freq = {}
    first_batch = {}
    n_batches = 0
    for X, lS_o, lS_i, T in batches:
        for t in large_tables:
            idx = lS_i[t].flatten() if isinstance(lS_i, (list, tuple)) else lS_i[t].flatten()
            if t not in freq:
                freq[t] = torch.zeros(ln_emb[t], dtype=torch.long)
                first_batch[t] = torch.full((ln_emb[t],), 999999999, dtype=torch.long)
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
            # Track first batch appearance
            unseen_mask = first_batch[t][idx.long()] == 999999999
            if unseen_mask.any():
                first_batch[t].scatter_reduce_(
                    0, idx.long()[unseen_mask],
                    torch.full((unseen_mask.sum(),), n_batches, dtype=torch.long),
                    reduce='amin', include_self=True
                )
        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break
    return freq, first_batch, n_batches


def hot_cold_split(freq, ln_emb, large_tables, state_dict, emb_keys,
                   reorder_method='frequency', first_batch_data=None):
    """Compute hot/cold split and cold row ordering."""
    is_hot = {}
    hot_indices = {}
    cold_indices = {}
    orig_to_cold_reordered = {}
    cold_weights_q = {}
    cold_quant_params = {}

    for t in large_tables:
        n = ln_emb[t]
        n_hot = max(1, int(n * HOT_FRACTION))
        sorted_idx = freq[t].argsort(descending=True)
        hot_idx = sorted_idx[:n_hot]

        is_hot_t = torch.zeros(n, dtype=torch.bool)
        is_hot_t[hot_idx] = True
        is_hot[t] = is_hot_t
        hot_indices[t] = hot_idx

        # Cold row ordering
        cold_idx = sorted_idx[n_hot:]  # by descending frequency
        if reorder_method == 'batch_affinity' and first_batch_data is not None:
            cold_freq = freq[t][cold_idx]
            cold_fb = first_batch_data[t][cold_idx]
            sort_key = cold_fb.float() * 1e12 - cold_freq.float()
            ba_order = sort_key.argsort()
            cold_idx = cold_idx[ba_order]

        cold_indices[t] = cold_idx

        # Build orig→cold_reordered mapping
        o2c = torch.full((n,), -1, dtype=torch.long)
        o2c[cold_idx] = torch.arange(len(cold_idx))
        orig_to_cold_reordered[t] = o2c

        # Quantize cold embeddings
        w = state_dict[emb_keys[t]]
        cold_w = w[cold_idx]
        mn = cold_w.min().item()
        mx = cold_w.max().item()
        s = (mx - mn) / 255.0
        if s == 0:
            s = 1.0
        zp = round(-mn / s)
        q = ((cold_w / s).round() + zp).clamp(0, 255).to(torch.uint8)
        cold_weights_q[t] = q
        cold_quant_params[t] = (s, zp)

    return (large_tables, is_hot, hot_indices, cold_indices,
            orig_to_cold_reordered, cold_weights_q, cold_quant_params)


# ============================================================
# H.265 Encoding
# ============================================================
def rows_to_tiled_frame(emb_rows, width, height):
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rpf = tiles_per_row * tiles_per_col
    n = emb_rows.shape[0]
    if n < rpf:
        padded = torch.zeros(rpf, EMB_DIM, dtype=torch.uint8)
        padded[:n] = emb_rows
        emb_rows = padded
    elif n > rpf:
        emb_rows = emb_rows[:rpf]
    tiles = emb_rows.reshape(tiles_per_col, tiles_per_row, TILE_H, TILE_W)
    frame = tiles.permute(0, 2, 1, 3).reshape(height, width)
    return frame.numpy()


def encode_h265_frames(cold_q, width, height, crf, output_dir, table_id):
    rpf = (width // TILE_W) * (height // TILE_H)
    n_cold = cold_q.shape[0]
    num_frames = (n_cold + rpf - 1) // rpf
    frame_dir = os.path.join(output_dir, f"table_{table_id}")
    os.makedirs(frame_dir, exist_ok=True)
    total_bytes = 0

    for f_idx in range(num_frames):
        start = f_idx * rpf
        end = min(start + rpf, n_cold)
        chunk = cold_q[start:end]
        frame_2d = rows_to_tiled_frame(chunk, width, height)
        out_path = os.path.join(frame_dir, f"frame_{f_idx:04d}.h265")
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{width}x{height}', '-i', 'pipe:0',
               '-c:v', 'libx265', '-preset', 'ultrafast',
               '-x265-params', f'qp={crf}:keyint=1:min-keyint=1' if crf == 0
               else f'crf={crf}:keyint=1:min-keyint=1',
               '-pix_fmt', 'gray', out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.stdin.write(frame_2d.tobytes())
        proc.stdin.close()
        proc.wait()
        total_bytes += os.path.getsize(out_path)

    return num_frames, frame_dir, total_bytes, rpf


def load_frames_to_ram(frame_dir, num_frames):
    frame_bytes = []
    frame_files = sorted([f for f in os.listdir(frame_dir)
                          if f.startswith('frame_') and
                          (f.endswith('.h265') or f.endswith('.h264') or f.endswith('.mkv'))])
    for ff in frame_files:
        with open(os.path.join(frame_dir, ff), 'rb') as fh:
            frame_bytes.append(fh.read())
    return frame_bytes


# ============================================================
# Decode helpers
# ============================================================
def decode_frame_from_bytes(data):
    if HAS_CPP and hasattr(_C, 'decode_h265_frame_from_bytes'):
        compressed_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        return _C.decode_h265_frame_from_bytes(compressed_t)
    container = av.open(io.BytesIO(data))
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='gray')
        container.close()
        return torch.from_numpy(img)
    container.close()
    return None


# ============================================================
# Compressed Embedding Bag (simplified from demo)
# ============================================================
class SimpleCompressedEmbeddingBag(nn.Module):
    def __init__(self, hot_weight, mapping, compressed_frames, rows_per_frame,
                 width, height, quant_scale, quant_zp, n_cold, cache_size=20):
        super().__init__()
        self.embedding_dim = EMB_DIM
        self.hot_weight = hot_weight
        self.mapping = mapping
        self.compressed_frames = compressed_frames
        self.rows_per_frame = rows_per_frame
        self.width = width
        self.height = height
        self.quant_scale = quant_scale
        self.quant_zp = quant_zp
        self.n_cold = n_cold
        self.tiles_per_row = width // TILE_W
        self.mode = 'sum'
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self._has_merged = HAS_CPP and hasattr(_C, 'compressed_emb_bag_forward_merged')
        self._empty_psw = torch.empty(0)

    def _get_frame(self, frame_id):
        if frame_id in self.cache:
            self.cache_hits += 1
            return self.cache[frame_id]
        self.cache_misses += 1
        frame_data = decode_frame_from_bytes(self.compressed_frames[frame_id])
        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            self.cache.pop(oldest, None)
        self.cache[frame_id] = frame_data
        self.cache_order.append(frame_id)
        return frame_data

    def _gather_cold(self, cold_reordered_indices):
        if len(cold_reordered_indices) == 0:
            return torch.zeros(0, EMB_DIM)
        frame_ids = (cold_reordered_indices // self.rows_per_frame).long()
        offsets_in_frame = (cold_reordered_indices % self.rows_per_frame).long()
        result = torch.zeros(len(cold_reordered_indices), EMB_DIM)
        unique_frames = frame_ids.unique()
        for fid in unique_frames:
            fid_val = fid.item()
            if fid_val < 0 or fid_val >= len(self.compressed_frames):
                continue
            mask = frame_ids == fid
            offsets = offsets_in_frame[mask]
            frame = self._get_frame(fid_val)
            if frame is None:
                continue
            for i, off in enumerate(offsets):
                off_val = off.item()
                ty = off_val // self.tiles_per_row
                tx = off_val % self.tiles_per_row
                y0, x0 = ty * TILE_H, tx * TILE_W
                tile = frame[y0:y0+TILE_H, x0:x0+TILE_W]
                if isinstance(tile, torch.Tensor):
                    row = tile.reshape(-1).float()
                else:
                    row = torch.from_numpy(tile.reshape(-1)).float()
                result[mask.nonzero(as_tuple=True)[0][i]] = (row - self.quant_zp) * self.quant_scale

        return result

    def forward(self, indices, offsets, per_sample_weights=None):
        psw = per_sample_weights if per_sample_weights is not None else self._empty_psw
        if self._has_merged:
            output, cold_mask, cold_count = _C.compressed_emb_bag_forward_merged(
                indices, offsets, self.hot_weight, self.mapping, psw)
            n_cold = cold_count.item()
            if n_cold > 0:
                cold_indices_reordered = -(self.mapping[indices[cold_mask]] + 1).long()
                cold_embs = self._gather_cold(cold_indices_reordered)
                bag_indices = torch.searchsorted(offsets[1:], cold_mask.nonzero(as_tuple=True)[0].to(offsets.dtype))
                output.scatter_add_(0, bag_indices.unsqueeze(1).expand(-1, EMB_DIM), cold_embs)
            return output
        else:
            # Python fallback
            B = offsets.shape[0]
            output = torch.zeros(B, EMB_DIM)
            for b in range(B):
                start = offsets[b].item()
                end = offsets[b+1].item() if b+1 < B else len(indices)
                for j in range(start, end):
                    idx = indices[j].item()
                    m = self.mapping[idx].item()
                    if m >= 0:
                        output[b] += self.hot_weight[m]
                    elif m != -2147483648:
                        cold_idx = -(m + 1)
                        cold_emb = self._gather_cold(torch.tensor([cold_idx]))
                        output[b] += cold_emb[0]
            return output

    def reset_cache_stats(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache = {}
        self.cache_order = []


# ============================================================
# Inference runner
# ============================================================
def run_inference(dlrm, batches, num_batches=0):
    """Run inference and return AUC + latency stats."""
    all_scores = []
    all_targets = []
    latencies = []

    n = num_batches if num_batches > 0 else len(batches)
    n = min(n, len(batches))

    with torch.no_grad():
        for i in range(n):
            X, lS_o, lS_i, T = batches[i]
            t0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            t1 = time.time()
            latencies.append((t1 - t0) * 1000)
            S = Z.detach().sigmoid().numpy().flatten()
            all_scores.append(S)
            all_targets.append(T.numpy().flatten())

    scores = np.concatenate(all_scores)
    targets = np.concatenate(all_targets)
    auc = roc_auc_score(targets, scores)

    return {
        'auc': float(auc),
        'mean_lat_ms': float(np.mean(latencies)),
        'p50_lat_ms': float(np.median(latencies)),
        'p99_lat_ms': float(np.percentile(latencies, 99)),
        'n_batches': n,
    }


# ============================================================
# Build compressed model
# ============================================================
def build_compressed_model(dlrm, ln_emb, state_dict, emb_keys,
                           large_tables, is_hot, hot_indices, cold_indices,
                           orig_to_cold_reordered, cold_quant_params,
                           compressed_frames_per_table, rpf_per_table,
                           cache_size=20):
    """Replace embedding tables with compressed versions. Returns memory stats."""
    # Restore original EmbeddingBag modules first
    with torch.no_grad():
        for k in emb_keys:
            t_idx = int(k.split('.')[1])
            if not hasattr(dlrm.emb_l[t_idx], 'weight'):
                # Re-create EmbeddingBag if it was replaced
                n_rows, dim = state_dict[k].shape
                dlrm.emb_l[t_idx] = nn.EmbeddingBag(n_rows, dim, mode='sum', sparse=False)
            dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

    total_hot_mb = 0
    total_compressed_bytes = 0
    total_mapping_mb = 0

    for t in large_tables:
        if t not in compressed_frames_per_table:
            continue
        w = state_dict[emb_keys[t]]
        h_idx = hot_indices[t]
        hot_weight = w[h_idx].clone()
        total_hot_mb += hot_weight.numel() * 4 / 1024 / 1024

        INT32_MIN = -2147483648
        mapping = torch.full((ln_emb[t],), INT32_MIN, dtype=torch.int32)
        orig_to_hot = torch.full((ln_emb[t],), -1, dtype=torch.long)
        orig_to_hot[h_idx] = torch.arange(len(h_idx))
        mapping[is_hot[t]] = orig_to_hot[is_hot[t]].to(torch.int32)
        o2c = orig_to_cold_reordered[t]
        cold_mask = o2c >= 0
        mapping[cold_mask] = (-(o2c[cold_mask] + 1)).to(torch.int32)
        total_mapping_mb += ln_emb[t] * 4 / 1024 / 1024

        s, zp = cold_quant_params[t]
        comp_bytes = sum(len(b) for b in compressed_frames_per_table[t])
        total_compressed_bytes += comp_bytes

        comp_emb = SimpleCompressedEmbeddingBag(
            hot_weight=hot_weight, mapping=mapping,
            compressed_frames=compressed_frames_per_table[t],
            rows_per_frame=rpf_per_table[t],
            width=WIDTH, height=HEIGHT,
            quant_scale=s, quant_zp=zp,
            n_cold=len(cold_indices[t]),
            cache_size=cache_size,
        )
        dlrm.emb_l[t] = comp_emb

    compressed_mb = total_compressed_bytes / 1024 / 1024
    cache_mb = cache_size * ROWS_PER_FRAME * EMB_DIM / 1024 / 1024

    return {
        'hot_mb': total_hot_mb,
        'compressed_mb': compressed_mb,
        'cache_mb': cache_mb,
        'mapping_mb': total_mapping_mb,
        'total_mb': total_hot_mb + compressed_mb + cache_mb + total_mapping_mb,
    }


def get_cache_stats(dlrm, large_tables):
    """Collect cache hit/miss stats from all compressed tables."""
    total_hits = 0
    total_misses = 0
    per_table = {}
    for t in large_tables:
        if isinstance(dlrm.emb_l[t], SimpleCompressedEmbeddingBag):
            e = dlrm.emb_l[t]
            total_hits += e.cache_hits
            total_misses += e.cache_misses
            total = e.cache_hits + e.cache_misses
            per_table[t] = {
                'hits': e.cache_hits, 'misses': e.cache_misses,
                'hit_rate': e.cache_hits / total if total > 0 else 0,
            }
    total = total_hits + total_misses
    return {
        'total_hits': total_hits,
        'total_misses': total_misses,
        'hit_rate': total_hits / total if total > 0 else 0,
        'per_table': per_table,
    }


def reset_cache_stats(dlrm, large_tables):
    for t in large_tables:
        if isinstance(dlrm.emb_l[t], SimpleCompressedEmbeddingBag):
            dlrm.emb_l[t].reset_cache_stats()


# ============================================================
# Run one config
# ============================================================
def run_config(name, dlrm, ln_emb, state_dict, emb_keys,
               profile_batches, eval_batches, baseline_auc,
               reorder_method='frequency', max_profile_batches=0):
    """Profile, encode, build, evaluate one configuration."""
    log(f"\n{'='*60}")
    log(f"Config: {name}")
    log(f"{'='*60}")

    num_tables = len(ln_emb)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]

    # Profile
    t0 = time.time()
    freq, first_batch, n_profiled = profile_frequency(
        profile_batches, ln_emb, large_tables, max_batches=max_profile_batches)
    profile_time = time.time() - t0
    log(f"  Profiled {n_profiled} batches in {profile_time:.1f}s")

    # Hot/cold split
    (_, is_hot, hot_indices, cold_indices,
     orig_to_cold_reordered, cold_weights_q, cold_quant_params) = \
        hot_cold_split(freq, ln_emb, large_tables, state_dict, emb_keys,
                       reorder_method=reorder_method, first_batch_data=first_batch)

    # Encode
    t0 = time.time()
    output_dir = os.path.join(RESULTS_DIR, f"compressed_{name}")
    compressed_frames_per_table = {}
    rpf_per_table = {}
    total_compressed_bytes = 0
    total_raw_bytes = 0

    for t in large_tables:
        n_cold = len(cold_indices[t])
        if n_cold == 0:
            continue
        num_frames, frame_dir, comp_bytes, rpf = encode_h265_frames(
            cold_weights_q[t], WIDTH, HEIGHT, CRF, output_dir, t)
        frame_bytes = load_frames_to_ram(frame_dir, num_frames)
        compressed_frames_per_table[t] = frame_bytes
        rpf_per_table[t] = rpf
        total_compressed_bytes += comp_bytes
        total_raw_bytes += n_cold * EMB_DIM

    encode_time = time.time() - t0

    uint8_ratio = total_raw_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0
    fp32_cold_bytes = sum(len(cold_indices[t]) * EMB_DIM * 4 for t in large_tables)
    fp32_ratio = fp32_cold_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0

    log(f"  Encoded in {encode_time:.1f}s: {total_raw_bytes/1024/1024:.1f}MB uint8 → "
        f"{total_compressed_bytes/1024/1024:.2f}MB H.265 ({uint8_ratio:.1f}x)")

    # Build compressed model
    mem = build_compressed_model(
        dlrm, ln_emb, state_dict, emb_keys,
        large_tables, is_hot, hot_indices, cold_indices,
        orig_to_cold_reordered, cold_quant_params,
        compressed_frames_per_table, rpf_per_table,
        cache_size=20)

    log(f"  Memory: hot={mem['hot_mb']:.1f}MB, cold={mem['compressed_mb']:.2f}MB, "
        f"cache={mem['cache_mb']:.1f}MB, mapping={mem['mapping_mb']:.1f}MB, "
        f"total={mem['total_mb']:.1f}MB")

    # Evaluate
    result = run_inference(dlrm, eval_batches, num_batches=NUM_EVAL_BATCHES)
    cache_stats = get_cache_stats(dlrm, large_tables)

    auc_delta = (result['auc'] - baseline_auc) * 100

    log(f"  AUC={result['auc']:.6f} (delta={auc_delta:+.4f}%)")
    log(f"  Latency={result['mean_lat_ms']:.2f}ms, Cache hit={cache_stats['hit_rate']:.1%}")

    return {
        'name': name,
        'reorder_method': reorder_method,
        'profiled_batches': n_profiled,
        'eval_batches': min(NUM_EVAL_BATCHES, len(eval_batches)),
        'profile_time_s': profile_time,
        'encode_time_s': encode_time,
        'uint8_compression_ratio': uint8_ratio,
        'fp32_compression_ratio': fp32_ratio,
        'compressed_cold_mb': total_compressed_bytes / 1024 / 1024,
        'memory': mem,
        'auc': result['auc'],
        'auc_delta_pct': auc_delta,
        'mean_latency_ms': result['mean_lat_ms'],
        'p50_latency_ms': result['p50_lat_ms'],
        'p99_latency_ms': result['p99_lat_ms'],
        'cache_hit_rate': cache_stats['hit_rate'],
        'cache_stats': {str(k): v for k, v in cache_stats['per_table'].items()},
    }


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 70)
    log("EXPERIMENT: Warmup Profiling Sweep + Train Profiling + Batch-Affinity")
    log("=" * 70)

    dlrm, train_ld, test_ld, ln_emb, state_dict, emb_keys = load_model_and_data()

    # Cache all test batches
    log("\nCaching test batches...")
    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    log(f"  {len(test_batches)} test batches cached")

    # Baseline AUC (on first 50 batches for speed)
    log("\nRunning baseline...")
    baseline = run_inference(dlrm, test_batches, num_batches=NUM_EVAL_BATCHES)
    baseline_auc = baseline['auc']
    log(f"  Baseline AUC: {baseline_auc:.6f}")

    results = {'baseline': baseline, 'configs': {}}

    # ==========================================
    # Part 1: Warmup fraction sweep (test_ld)
    # ==========================================
    log("\n" + "=" * 70)
    log("PART 1: Warmup Fraction Sweep (test_ld)")
    log("=" * 70)

    warmup_fractions = [0.05, 0.10, 0.20, 0.50]
    for wf in warmup_fractions:
        n_warmup = max(1, int(len(test_batches) * wf))
        warmup_batches = test_batches[:n_warmup]
        eval_batches = test_batches[n_warmup:]

        name = f"warmup_{int(wf*100)}pct"
        r = run_config(name, dlrm, ln_emb, state_dict, emb_keys,
                       warmup_batches, eval_batches, baseline_auc,
                       reorder_method='frequency')
        r['warmup_fraction'] = wf
        r['warmup_batches'] = n_warmup
        r['disjoint_eval'] = True
        results['configs'][name] = r
        gc.collect()

    # 100% test (original circular method — for comparison)
    name = "test_100pct"
    r = run_config(name, dlrm, ln_emb, state_dict, emb_keys,
                   test_batches, test_batches, baseline_auc,
                   reorder_method='frequency')
    r['warmup_fraction'] = 1.0
    r['disjoint_eval'] = False
    results['configs'][name] = r
    gc.collect()

    # ==========================================
    # Part 2: Train_ld profiling with subsampling
    # ==========================================
    log("\n" + "=" * 70)
    log("PART 2: Train_ld Profiling (subsample sweep)")
    log("=" * 70)

    train_batch_counts = [1599, 5000, 20000]
    for nbc in train_batch_counts:
        name = f"train_{nbc}batches"
        r = run_config(name, dlrm, ln_emb, state_dict, emb_keys,
                       train_ld, test_batches, baseline_auc,
                       reorder_method='frequency', max_profile_batches=nbc)
        r['disjoint_eval'] = True
        results['configs'][name] = r
        gc.collect()

    # ==========================================
    # Part 3: Hybrid — train for split, warmup for ordering
    # ==========================================
    log("\n" + "=" * 70)
    log("PART 3: Hybrid (train split + warmup ordering)")
    log("=" * 70)

    num_tables = len(ln_emb)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]

    # Train profiling for hot/cold split
    log("  Profiling train_ld (5000 batches) for hot/cold split...")
    train_freq, _, train_n = profile_frequency(train_ld, ln_emb, large_tables, max_batches=5000)

    # Warmup profiling for ordering
    n_warmup = int(len(test_batches) * 0.10)
    warmup_batches = test_batches[:n_warmup]
    eval_batches = test_batches[n_warmup:]
    log(f"  Profiling test warmup ({n_warmup} batches) for cold ordering...")
    warmup_freq, warmup_fb, warmup_n = profile_frequency(warmup_batches, ln_emb, large_tables)

    # Use train freq for hot/cold split, warmup freq for cold ordering
    is_hot = {}
    hot_indices = {}
    cold_indices = {}
    orig_to_cold_reordered = {}
    cold_weights_q = {}
    cold_quant_params = {}

    for t in large_tables:
        n = ln_emb[t]
        n_hot = max(1, int(n * HOT_FRACTION))

        # Hot/cold split from TRAIN frequency
        sorted_idx = train_freq[t].argsort(descending=True)
        hot_idx = sorted_idx[:n_hot]
        cold_idx_set = sorted_idx[n_hot:]

        is_hot_t = torch.zeros(n, dtype=torch.bool)
        is_hot_t[hot_idx] = True
        is_hot[t] = is_hot_t
        hot_indices[t] = hot_idx

        # Cold row ordering from WARMUP frequency
        cold_warmup_freq = warmup_freq[t][cold_idx_set]
        cold_order = cold_warmup_freq.argsort(descending=True)
        cold_idx = cold_idx_set[cold_order]
        cold_indices[t] = cold_idx

        # Build mapping
        o2c = torch.full((n,), -1, dtype=torch.long)
        o2c[cold_idx] = torch.arange(len(cold_idx))
        orig_to_cold_reordered[t] = o2c

        # Quantize cold
        w = state_dict[emb_keys[t]]
        cold_w = w[cold_idx]
        mn = cold_w.min().item()
        mx = cold_w.max().item()
        s = (mx - mn) / 255.0
        if s == 0:
            s = 1.0
        zp = round(-mn / s)
        q = ((cold_w / s).round() + zp).clamp(0, 255).to(torch.uint8)
        cold_weights_q[t] = q
        cold_quant_params[t] = (s, zp)

    # Encode
    output_dir = os.path.join(RESULTS_DIR, "compressed_hybrid")
    compressed_frames_per_table = {}
    rpf_per_table = {}
    total_compressed_bytes = 0
    total_raw_bytes = 0
    for t in large_tables:
        n_cold = len(cold_indices[t])
        if n_cold == 0:
            continue
        num_frames, frame_dir, comp_bytes, rpf = encode_h265_frames(
            cold_weights_q[t], WIDTH, HEIGHT, CRF, output_dir, t)
        frame_bytes = load_frames_to_ram(frame_dir, num_frames)
        compressed_frames_per_table[t] = frame_bytes
        rpf_per_table[t] = rpf
        total_compressed_bytes += comp_bytes
        total_raw_bytes += n_cold * EMB_DIM

    # Build model
    mem = build_compressed_model(
        dlrm, ln_emb, state_dict, emb_keys,
        large_tables, is_hot, hot_indices, cold_indices,
        orig_to_cold_reordered, cold_quant_params,
        compressed_frames_per_table, rpf_per_table, cache_size=20)

    # Evaluate
    hybrid_result = run_inference(dlrm, eval_batches, num_batches=NUM_EVAL_BATCHES)
    cache_stats = get_cache_stats(dlrm, large_tables)
    auc_delta = (hybrid_result['auc'] - baseline_auc) * 100

    uint8_ratio = total_raw_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0
    fp32_cold = sum(len(cold_indices[t]) * EMB_DIM * 4 for t in large_tables)
    fp32_ratio = fp32_cold / total_compressed_bytes if total_compressed_bytes > 0 else 0

    log(f"  Hybrid: AUC={hybrid_result['auc']:.6f} (delta={auc_delta:+.4f}%), "
        f"cache={cache_stats['hit_rate']:.1%}, "
        f"compression={uint8_ratio:.1f}x uint8")

    results['configs']['hybrid_train5k_warmup10pct'] = {
        'name': 'hybrid_train5k_warmup10pct',
        'description': 'Train (5K batches) for hot/cold split, test warmup (10%) for cold ordering',
        'reorder_method': 'frequency',
        'train_profiled_batches': train_n,
        'warmup_profiled_batches': warmup_n,
        'eval_batches': min(NUM_EVAL_BATCHES, len(eval_batches)),
        'disjoint_eval': True,
        'uint8_compression_ratio': uint8_ratio,
        'fp32_compression_ratio': fp32_ratio,
        'compressed_cold_mb': total_compressed_bytes / 1024 / 1024,
        'memory': mem,
        'auc': hybrid_result['auc'],
        'auc_delta_pct': auc_delta,
        'mean_latency_ms': hybrid_result['mean_lat_ms'],
        'cache_hit_rate': cache_stats['hit_rate'],
        'cache_stats': {str(k): v for k, v in cache_stats['per_table'].items()},
    }

    # ==========================================
    # Part 4: Batch-affinity reordering
    # ==========================================
    log("\n" + "=" * 70)
    log("PART 4: Batch-Affinity Reordering")
    log("=" * 70)

    # Warmup 10% + batch_affinity
    n_warmup = int(len(test_batches) * 0.10)
    warmup_batches = test_batches[:n_warmup]
    eval_batches = test_batches[n_warmup:]

    name = "warmup_10pct_batch_affinity"
    r = run_config(name, dlrm, ln_emb, state_dict, emb_keys,
                   warmup_batches, eval_batches, baseline_auc,
                   reorder_method='batch_affinity')
    r['warmup_fraction'] = 0.10
    r['disjoint_eval'] = True
    results['configs'][name] = r
    gc.collect()

    # Test 100% + batch_affinity
    name = "test_100pct_batch_affinity"
    r = run_config(name, dlrm, ln_emb, state_dict, emb_keys,
                   test_batches, test_batches, baseline_auc,
                   reorder_method='batch_affinity')
    r['warmup_fraction'] = 1.0
    r['disjoint_eval'] = False
    results['configs'][name] = r
    gc.collect()

    # ==========================================
    # Save results
    # ==========================================
    results_path = os.path.join(RESULTS_DIR, "warmup_sweep.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {results_path}")

    # Print summary table
    log("\n" + "=" * 70)
    log("SUMMARY TABLE")
    log("=" * 70)
    log(f"{'Config':<35} {'Compr':>6} {'Cache':>6} {'AUC Δ':>8} {'Lat':>8} {'Mem':>6} {'Disjoint':>8}")
    log("-" * 80)
    for name, cfg in results['configs'].items():
        log(f"{name:<35} {cfg['uint8_compression_ratio']:>5.1f}x "
            f"{cfg['cache_hit_rate']:>5.1%} "
            f"{cfg['auc_delta_pct']:>+7.4f}% "
            f"{cfg['mean_latency_ms']:>7.1f}ms "
            f"{cfg['memory']['total_mb']:>5.0f}MB "
            f"{'YES' if cfg.get('disjoint_eval', False) else 'NO':>8}")


if __name__ == '__main__':
    main()
