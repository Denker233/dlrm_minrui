#!/usr/bin/env python3
"""
End-to-End Demo: H.265 Video Codec Compression for DLRM Embedding Tables

Walks through the full pipeline:
  1. Load trained DLRM model
  2. Profile embedding access patterns → hot/cold split
  3. Quantize cold embeddings to uint8
  4. Tile into 4×4 blocks, encode as H.265 video frames
  5. Run inference with compressed model (on-demand decode)
  6. Compare AUC and latency vs uncompressed baseline

Usage:
    python3 demo_e2e_pipeline.py [--crf 0] [--resolution 1080p] [--num-batches 0]

    --crf: H.265 quality (0=lossless, 18=lossy sweet spot)
    --resolution: 1080p or 4K
    --num-batches: limit test batches (0 = all)
"""

import os, sys, time, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import roc_auc_score
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import compressed_emb as _C
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input/train.txt")
PROCESSED_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input/kaggleAdDisplayChallenge_processed.npz")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 2048
EMB_DIM = 16
TILE_H, TILE_W = 4, 4

HOT_COVERAGE = 0.80           # top 80% accesses → hot
LARGE_TABLE_THRESHOLD = 50000  # only compress tables with >50K rows
PROFILE_BATCHES = 200          # batches to profile for access frequency

RESOLUTIONS = {
    '1080p': (1920, 1080),
    '4K':    (3840, 2160),
}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


# ============================================================
# Step 0: Load model and data
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
    log("Loading Kaggle dataset...")
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    ln_emb = np.array(train_data.counts)
    m_spa = args.arch_sparse_feature_size
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[-1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + args.arch_mlp_top, dtype=int, sep="-")

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
    total_emb_mb = sum(state_dict[k].numel() * 4 for k in emb_keys) / 1024 / 1024
    log(f"Total embedding memory: {total_emb_mb:.1f} MB (fp32)")

    return dlrm, train_ld, test_ld, ln_emb, state_dict, emb_keys


# ============================================================
# Step 1: Profile access patterns → hot/cold split
# ============================================================
def profile_and_split(train_ld, ln_emb, state_dict, emb_keys):
    num_tables = len(ln_emb)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"\nStep 1: Profiling access patterns ({PROFILE_BATCHES} batches)")
    log(f"  {len(large_tables)} large tables (>= {LARGE_TABLE_THRESHOLD} rows): {large_tables}")

    # Count access frequency per row per table
    access_counts = {t: Counter() for t in large_tables}
    n_batches = 0
    for X, lS_o, lS_i, T in train_ld:
        for t in large_tables:
            indices = lS_i[t].numpy()
            for idx in indices:
                access_counts[t][idx] += 1
        n_batches += 1
        if n_batches >= PROFILE_BATCHES:
            break
    log(f"  Profiled {n_batches} batches")

    # Hot/cold split: top rows covering HOT_COVERAGE of total accesses → hot
    is_hot = {}
    hot_indices = {}
    cold_indices = {}
    orig_to_cold_reordered = {}
    cold_weights_q = {}
    cold_quant_params = {}

    for t in large_tables:
        counts = access_counts[t]
        total_accesses = sum(counts.values())
        sorted_rows = sorted(counts.items(), key=lambda x: -x[1])

        hot_set = set()
        cumulative = 0
        for row_id, cnt in sorted_rows:
            if cumulative >= total_accesses * HOT_COVERAGE:
                break
            hot_set.add(row_id)
            cumulative += cnt

        # Build masks
        n = ln_emb[t]
        is_hot_t = torch.zeros(n, dtype=torch.bool)
        for r in hot_set:
            is_hot_t[r] = True
        is_hot[t] = is_hot_t
        hot_indices[t] = torch.where(is_hot_t)[0]
        cold_indices[t] = torch.where(~is_hot_t)[0]

        n_hot = len(hot_indices[t])
        n_cold = len(cold_indices[t])

        # Sort cold rows by access frequency (most accessed first)
        cold_set = cold_indices[t].tolist()
        cold_freq = [(r, counts.get(r, 0)) for r in cold_set]
        cold_freq.sort(key=lambda x: -x[1])
        cold_order = [r for r, _ in cold_freq]

        # Build orig→cold_reordered mapping
        o2c = torch.full((n,), -1, dtype=torch.long)
        for new_idx, orig_idx in enumerate(cold_order):
            o2c[orig_idx] = new_idx
        orig_to_cold_reordered[t] = o2c

        # Quantize cold embeddings (global min-max to uint8)
        w = state_dict[emb_keys[t]]
        cold_w = w[torch.tensor(cold_order)]
        mn = cold_w.min().item()
        mx = cold_w.max().item()
        s = (mx - mn) / 255.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        q = ((cold_w / s).round() + zp).clamp(0, 255).to(torch.uint8)

        cold_weights_q[t] = q
        cold_quant_params[t] = (s, zp)

        hot_mb = n_hot * EMB_DIM * 4 / 1024 / 1024
        cold_mb = n_cold * EMB_DIM * 4 / 1024 / 1024
        cold_q_mb = n_cold * EMB_DIM / 1024 / 1024
        log(f"  Table {t}: {n:,} rows → {n_hot:,} hot ({hot_mb:.1f}MB fp32) + "
            f"{n_cold:,} cold ({cold_mb:.1f}MB fp32 → {cold_q_mb:.1f}MB uint8)")

    return large_tables, is_hot, hot_indices, cold_indices, orig_to_cold_reordered, \
           cold_weights_q, cold_quant_params


# ============================================================
# Step 2: Tile + H.265 encode
# ============================================================
def rows_to_tiled_frame_py(emb_rows, width, height):
    """Python: reshape (N,16) → 4×4 tiles → (H,W) frame."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    n = min(len(emb_rows), rows_per_frame)
    padded = np.zeros((rows_per_frame, EMB_DIM), dtype=np.uint8)
    padded[:n] = emb_rows[:n]
    tiles = padded.reshape(tiles_per_col, tiles_per_row, TILE_H, TILE_W)
    frame = tiles.transpose(0, 2, 1, 3).reshape(height, width)
    return frame


def tiled_frame_to_rows_py(frame, width, height):
    """Python: (H,W) frame → untile → (N,16) rows."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    grid = frame.reshape(tiles_per_col, TILE_H, tiles_per_row, TILE_W)
    rows = grid.transpose(0, 2, 1, 3).reshape(rows_per_frame, EMB_DIM)
    return rows


def encode_h265_frames(cold_q, width, height, crf, output_dir, table_id):
    """Encode uint8 cold embeddings as H.265 video frames."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    num_rows = cold_q.shape[0]
    num_frames = max(1, (num_rows + rows_per_frame - 1) // rows_per_frame)

    frame_dir = os.path.join(output_dir, f'table_{table_id}')
    os.makedirs(frame_dir, exist_ok=True)

    t0 = time.time()
    total_compressed = 0
    q_np = cold_q.numpy() if isinstance(cold_q, torch.Tensor) else cold_q

    if HAS_CPP:
        # C++ path: fused multiframe tiling + batch encode
        q_t = torch.from_numpy(q_np) if isinstance(q_np, np.ndarray) else q_np
        padded_rows = num_frames * rows_per_frame
        if q_t.shape[0] < padded_rows:
            padded = torch.zeros(padded_rows, EMB_DIM, dtype=torch.uint8)
            padded[:q_t.shape[0]] = q_t
            q_t = padded
        tiled_frames = _C.fused_quantize_tile_multiframe(q_t, width, height)

        if hasattr(_C, 'batch_encode_h265_frames') and crf == 0:
            total_compressed = _C.batch_encode_h265_frames(tiled_frames, frame_dir, True)
        elif hasattr(_C, 'batch_encode_frames_codec') and crf == 0:
            total_compressed = _C.batch_encode_frames_codec(
                tiled_frames, frame_dir, 'h265', True)
        else:
            # Per-frame encode
            for i in range(num_frames):
                frame_2d = tiled_frames[i].numpy()
                frame_path = os.path.join(frame_dir, f'frame_{i:05d}.h265')
                _encode_single_frame(frame_2d, frame_path, width, height, crf)
                total_compressed += os.path.getsize(frame_path)
    else:
        # Python-only path
        padded_rows = num_frames * rows_per_frame
        padded = np.zeros((padded_rows, EMB_DIM), dtype=np.uint8)
        padded[:num_rows] = q_np
        for i in range(num_frames):
            chunk = padded[i * rows_per_frame:(i + 1) * rows_per_frame]
            frame_2d = rows_to_tiled_frame_py(chunk, width, height)
            frame_path = os.path.join(frame_dir, f'frame_{i:05d}.h265')
            _encode_single_frame(frame_2d, frame_path, width, height, crf)
            total_compressed += os.path.getsize(frame_path)

    encode_time = time.time() - t0
    raw_bytes = num_rows * EMB_DIM
    ratio = raw_bytes / total_compressed if total_compressed > 0 else 0
    log(f"    Encoded: {num_frames} frames, "
        f"{raw_bytes/1024/1024:.2f}MB uint8 → {total_compressed/1024/1024:.2f}MB H.265 "
        f"({ratio:.1f}x), {encode_time:.1f}s")

    return num_frames, frame_dir, total_compressed, rows_per_frame


def _encode_single_frame(frame_2d, frame_path, width, height, crf):
    """Encode a single frame using PyAV or ffmpeg subprocess."""
    if HAS_CPP and hasattr(_C, 'encode_h265_frame'):
        frame_t = torch.from_numpy(np.ascontiguousarray(frame_2d))
        if crf == 0:
            _C.encode_h265_frame_to_file(frame_t, frame_path, True)
        else:
            compressed = _C.encode_frame_with_params(
                frame_t, frame_path, "h265", False, crf, "")
        return

    if HAS_PYAV:
        container = av.open(frame_path, mode='w')
        stream = container.add_stream('libx265', rate=1)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'gray'
        if crf == 0:
            stream.options = {'x265-params': 'lossless=1:log-level=error'}
        else:
            stream.options = {'x265-params': f'crf={crf}:log-level=error'}
        frame = av.VideoFrame.from_ndarray(frame_2d, format='gray')
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return

    import subprocess
    lossless = "lossless=1" if crf == 0 else f"crf={crf}"
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
        '-s', f'{width}x{height}', '-r', '1', '-i', 'pipe:0',
        '-c:v', 'libx265', '-preset', 'ultrafast', '-pix_fmt', 'gray',
        '-x265-params', f'keyint=1:min-keyint=1:{lossless}:log-level=error',
        '-f', 'matroska', frame_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.stdin.write(frame_2d.tobytes())
    proc.stdin.close()
    proc.wait()


# ============================================================
# Step 3: Decode frame (for inference)
# ============================================================
def decode_frame(frame_path, width, height):
    """Decode a single H.265 frame file → (H, W) uint8 array."""
    if HAS_CPP and hasattr(_C, 'decode_h265_frame_from_file'):
        return _C.decode_h265_frame_from_file(frame_path)

    container = av.open(frame_path)
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='gray')
        container.close()
        return torch.from_numpy(img) if HAS_CPP else img
    container.close()
    return None


# ============================================================
# Step 4: Inference with compressed embeddings
# ============================================================
class SimpleCompressedEmbeddingBag(nn.Module):
    """
    Simplified compressed embedding lookup for demo.
    Hot rows: fp32 compact tensor.
    Cold rows: on-demand H.265 decode with LRU frame cache.
    """
    def __init__(self, hot_weight, mapping, cold_frames_dir, rows_per_frame,
                 width, height, quant_scale, quant_zp, n_cold, cache_size=20):
        super().__init__()
        self.embedding_dim = EMB_DIM
        self.hot_weight = hot_weight
        self.mapping = mapping  # int32: >=0 = hot idx, <0 = -(cold_idx+1)
        self.cold_frames_dir = cold_frames_dir
        self.rows_per_frame = rows_per_frame
        self.width = width
        self.height = height
        self.quant_scale = quant_scale
        self.quant_zp = quant_zp
        self.n_cold = n_cold
        self.tiles_per_row = width // TILE_W
        self.mode = 'sum'

        # LRU frame cache
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

        # Detect frame extension
        self._frame_ext = '.h265'
        for ext in ['.h265', '.h264', '.mkv']:
            test_path = os.path.join(cold_frames_dir, f'frame_00000{ext}')
            if os.path.exists(test_path):
                self._frame_ext = ext
                break

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0

        self._has_merged = HAS_CPP and hasattr(_C, 'compressed_emb_bag_forward_merged')
        self._empty_psw = torch.empty(0)

    def _get_frame(self, frame_id):
        """Get decoded frame from cache or decode from disk."""
        if frame_id in self.cache:
            self.cache_hits += 1
            return self.cache[frame_id]

        self.cache_misses += 1
        frame_path = os.path.join(
            self.cold_frames_dir, f'frame_{frame_id:05d}{self._frame_ext}')
        frame_data = decode_frame(frame_path, self.width, self.height)

        # LRU eviction
        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            self.cache.pop(oldest, None)
        self.cache[frame_id] = frame_data
        self.cache_order.append(frame_id)
        return frame_data

    def _gather_cold(self, cold_reordered_indices):
        """Gather cold embeddings by decoding needed frames."""
        if len(cold_reordered_indices) == 0:
            return torch.zeros(0, EMB_DIM)

        frame_ids = (cold_reordered_indices // self.rows_per_frame).long()
        row_offsets = (cold_reordered_indices % self.rows_per_frame).long()
        unique_frames = torch.unique(frame_ids).tolist()

        # Decode needed frames
        for fid in unique_frames:
            self._get_frame(fid)

        # Gather rows from cached frames
        result = torch.zeros(len(cold_reordered_indices), EMB_DIM)
        for fid in unique_frames:
            mask = frame_ids == fid
            offsets = row_offsets[mask]
            frame_data = self.cache[fid]

            if HAS_CPP and isinstance(frame_data, torch.Tensor) and frame_data.dim() == 2:
                # C++ gather + dequant from tiled frame
                rows = _C.gather_dequant_from_tiled_frame(
                    frame_data, offsets, self.tiles_per_row,
                    self.quant_scale, self.quant_zp)
            else:
                # Python: untile, index, dequantize
                if isinstance(frame_data, torch.Tensor):
                    frame_np = frame_data.numpy()
                else:
                    frame_np = frame_data
                all_rows = tiled_frame_to_rows_py(frame_np, self.width, self.height)
                selected = all_rows[offsets.numpy()]
                rows = torch.from_numpy(
                    (selected.astype(np.float32) - self.quant_zp) * self.quant_scale)

            result[mask] = rows
        return result

    def forward(self, indices, offsets, per_sample_weights=None):
        psw = per_sample_weights if per_sample_weights is not None else self._empty_psw

        # C++ hot forward with merged mapping
        if self._has_merged:
            output, cold_mask, cold_count = _C.compressed_emb_bag_forward_merged(
                indices, offsets, self.hot_weight, self.mapping, psw)

            if cold_count.item() > 0:
                cold_positions = torch.where(cold_mask)[0]
                cold_orig = indices[cold_positions]
                cold_map_vals = self.mapping[cold_orig]
                cold_reordered = -(cold_map_vals.long() + 1)
                valid = cold_reordered >= 0
                if valid.any():
                    cold_result = self._gather_cold(cold_reordered[valid])
                    _C.cold_fixup(output, indices, offsets, cold_mask,
                                  cold_result, cold_positions[valid], psw)
            return output

        # Python fallback
        map_vals = self.mapping[indices]
        hot_mask = map_vals >= 0
        all_embeds = torch.zeros(len(indices), EMB_DIM)

        if hot_mask.any():
            hot_idx = map_vals[hot_mask].long()
            all_embeds[hot_mask] = self.hot_weight[hot_idx]

        INT32_MIN = -2147483648
        cold_mask = (map_vals < 0) & (map_vals != INT32_MIN)
        if cold_mask.any():
            cold_map = map_vals[cold_mask]
            cold_reordered = -(cold_map.long() + 1)
            valid = cold_reordered >= 0
            if valid.any():
                cold_result = self._gather_cold(cold_reordered[valid])
                cold_positions = torch.where(cold_mask)[0]
                all_embeds[cold_positions[valid]] = cold_result

        if per_sample_weights is not None:
            all_embeds = all_embeds * per_sample_weights.unsqueeze(1)

        num_bags = len(offsets)
        output = torch.zeros(num_bags, EMB_DIM)
        bag_ids = torch.bucketize(torch.arange(len(indices)), offsets, right=True) - 1
        bag_ids = bag_ids.clamp(min=0)
        output.scatter_add_(0, bag_ids.unsqueeze(1).expand_as(all_embeds), all_embeds)
        return output


# ============================================================
# Inference runner
# ============================================================
def run_inference(dlrm, test_batches, label, num_batches=0):
    """Run inference and measure AUC + latency."""
    n = len(test_batches) if num_batches == 0 else min(num_batches, len(test_batches))
    max_samples = n * TEST_BATCH_SIZE + TEST_BATCH_SIZE
    all_scores = np.empty(max_samples, dtype=np.float32)
    all_targets = np.empty(max_samples, dtype=np.float32)
    sample_idx = 0
    latencies = []

    dlrm.time_look_up = 0
    dlrm.time_interact = 0
    dlrm.time_mlp = 0

    t0 = time.time()
    with torch.no_grad():
        for batch_idx in range(n):
            X, lS_o, lS_i, T = test_batches[batch_idx]
            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            latencies.append(time.time() - bt0)

            z_np = Z.detach().cpu().numpy().ravel()
            t_np = T.detach().cpu().numpy().ravel()
            bs = z_np.shape[0]
            all_scores[sample_idx:sample_idx + bs] = z_np
            all_targets[sample_idx:sample_idx + bs] = t_np
            sample_idx += bs

            if batch_idx % 500 == 0 and batch_idx > 0:
                log(f"    [{label}] Batch {batch_idx}/{n}")

    total_time = time.time() - t0
    auc = roc_auc_score(all_targets[:sample_idx], all_scores[:sample_idx])
    mem = rss_mb()

    log(f"  [{label}] AUC = {auc:.6f}")
    log(f"  [{label}] Total = {total_time:.2f}s ({n} batches)")
    log(f"  [{label}] Latency: mean={np.mean(latencies)*1000:.2f}ms, "
        f"p50={np.percentile(latencies,50)*1000:.2f}ms, "
        f"p99={np.percentile(latencies,99)*1000:.2f}ms")
    log(f"  [{label}] Breakdown: emb={dlrm.time_look_up/n*1000:.2f}ms, "
        f"interact={dlrm.time_interact/n*1000:.2f}ms, "
        f"mlp={dlrm.time_mlp/n*1000:.2f}ms")
    log(f"  [{label}] RSS = {mem:.0f}MB")

    return {
        'auc': auc, 'total_time': total_time, 'num_batches': n,
        'mean_lat_ms': np.mean(latencies) * 1000,
        'p50_lat_ms': np.percentile(latencies, 50) * 1000,
        'p99_lat_ms': np.percentile(latencies, 99) * 1000,
        'rss_mb': mem,
        'emb_ms': dlrm.time_look_up / n * 1000,
        'interact_ms': dlrm.time_interact / n * 1000,
        'mlp_ms': dlrm.time_mlp / n * 1000,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="E2E H.265 DLRM Compression Demo")
    parser.add_argument('--crf', type=int, default=0, help='H.265 CRF (0=lossless, 18=lossy)')
    parser.add_argument('--resolution', type=str, default='1080p', choices=['1080p', '4K'])
    parser.add_argument('--num-batches', type=int, default=0, help='Limit test batches (0=all)')
    parser.add_argument('--cache-size', type=int, default=20, help='LRU cache frames')
    args = parser.parse_args()

    width, height = RESOLUTIONS[args.resolution]
    rows_per_frame = (width // TILE_W) * (height // TILE_H)
    output_dir = f"results/demo_crf{args.crf}_{args.resolution}"
    os.makedirs(output_dir, exist_ok=True)

    log("=" * 70)
    log("END-TO-END H.265 CODEC COMPRESSION DEMO")
    log(f"CRF={args.crf}, Resolution={args.resolution} ({width}x{height})")
    log(f"Rows/frame={rows_per_frame:,}, Cache={args.cache_size} frames")
    log(f"C++ extension: {'YES' if HAS_CPP else 'NO (Python fallback)'}")
    log("=" * 70)

    # Step 0: Load
    dlrm, train_ld, test_ld, ln_emb, state_dict, emb_keys = load_model_and_data()

    # Pre-cache test batches (eliminate DataLoader overhead from timing)
    log("\nPre-caching test batches...")
    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    log(f"  {len(test_batches)} batches cached")

    # Step 0.5: Baseline inference
    log("\n" + "=" * 70)
    log("BASELINE: Full fp32 (uncompressed)")
    log("=" * 70)
    baseline = run_inference(dlrm, test_batches, "Baseline", args.num_batches)

    # Step 1: Profile + split
    log("\n" + "=" * 70)
    log("STEP 1: Profile Access Patterns + Hot/Cold Split")
    log("=" * 70)
    (large_tables, is_hot, hot_indices, cold_indices,
     orig_to_cold_reordered, cold_weights_q, cold_quant_params) = \
        profile_and_split(train_ld, ln_emb, state_dict, emb_keys)

    # Step 2: Encode
    log("\n" + "=" * 70)
    log("STEP 2: Tile + H.265 Encode Cold Embeddings")
    log("=" * 70)
    frame_dirs = {}
    rpf_per_table = {}
    total_raw_bytes = 0
    total_compressed_bytes = 0

    for t in large_tables:
        n_cold = len(cold_indices[t])
        if n_cold == 0:
            continue
        log(f"  Table {t}: {n_cold:,} cold rows")
        num_frames, frame_dir, comp_bytes, rpf = encode_h265_frames(
            cold_weights_q[t], width, height, args.crf, output_dir, t)
        frame_dirs[t] = frame_dir
        rpf_per_table[t] = rpf
        total_raw_bytes += n_cold * EMB_DIM
        total_compressed_bytes += comp_bytes

    if total_compressed_bytes > 0:
        overall_ratio = total_raw_bytes / total_compressed_bytes
        log(f"\n  Total: {total_raw_bytes/1024/1024:.1f}MB uint8 → "
            f"{total_compressed_bytes/1024/1024:.2f}MB H.265 ({overall_ratio:.1f}x)")

        # Also compute fp32-to-compressed ratio
        total_fp32_cold = sum(len(cold_indices[t]) * EMB_DIM * 4
                              for t in large_tables) / 1024 / 1024
        fp32_ratio = total_fp32_cold / (total_compressed_bytes / 1024 / 1024)
        log(f"  Cold fp32→H.265: {total_fp32_cold:.1f}MB → "
            f"{total_compressed_bytes/1024/1024:.2f}MB ({fp32_ratio:.0f}x)")

    # Step 3: Replace embeddings with compressed version
    log("\n" + "=" * 70)
    log("STEP 3: Build Compressed Model + Run Inference")
    log("=" * 70)

    # Restore weights first
    with torch.no_grad():
        for k in emb_keys:
            t_idx = int(k.split('.')[1])
            dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

    total_hot_mb = 0
    total_mapping_mb = 0
    for t in large_tables:
        if t not in frame_dirs:
            continue
        w = state_dict[emb_keys[t]]
        h_idx = hot_indices[t]
        hot_weight = w[h_idx].clone()
        total_hot_mb += hot_weight.numel() * 4 / 1024 / 1024

        # Build merged int32 mapping
        INT32_MIN = -2147483648
        mapping = torch.full((ln_emb[t],), INT32_MIN, dtype=torch.int32)
        # Hot: mapping[i] = compact hot index
        orig_to_hot = torch.full((ln_emb[t],), -1, dtype=torch.long)
        orig_to_hot[h_idx] = torch.arange(len(h_idx))
        mapping[is_hot[t]] = orig_to_hot[is_hot[t]].to(torch.int32)
        # Cold: mapping[i] = -(cold_reordered_idx + 1)
        o2c = orig_to_cold_reordered[t]
        cold_mask = o2c >= 0
        mapping[cold_mask] = (-(o2c[cold_mask] + 1)).to(torch.int32)

        total_mapping_mb += ln_emb[t] * 4 / 1024 / 1024

        s, zp = cold_quant_params[t]
        comp_emb = SimpleCompressedEmbeddingBag(
            hot_weight=hot_weight,
            mapping=mapping,
            cold_frames_dir=frame_dirs[t],
            rows_per_frame=rpf_per_table[t],
            width=width, height=height,
            quant_scale=s, quant_zp=zp,
            n_cold=len(cold_indices[t]),
            cache_size=args.cache_size,
        )
        dlrm.emb_l[t] = comp_emb

    compressed_mb = total_compressed_bytes / 1024 / 1024
    cache_mb = args.cache_size * rows_per_frame * EMB_DIM / 1024 / 1024
    log(f"\n  Memory breakdown:")
    log(f"    Hot embeddings (fp32):  {total_hot_mb:.1f} MB")
    log(f"    Cold compressed (H.265): {compressed_mb:.2f} MB")
    log(f"    LRU cache ({args.cache_size} frames): {cache_mb:.1f} MB")
    log(f"    Mapping (int32):        {total_mapping_mb:.1f} MB")
    log(f"    Total:                  {total_hot_mb + compressed_mb + cache_mb + total_mapping_mb:.1f} MB")

    gc.collect()

    # Run compressed inference
    compressed = run_inference(dlrm, test_batches, "Compressed", args.num_batches)

    # Print cache stats
    for t in large_tables:
        if isinstance(dlrm.emb_l[t], SimpleCompressedEmbeddingBag):
            e = dlrm.emb_l[t]
            total = e.cache_hits + e.cache_misses
            hit_rate = e.cache_hits / total if total > 0 else 0
            log(f"  Table {t} cache: {e.cache_hits} hits, {e.cache_misses} misses "
                f"({hit_rate:.1%} hit rate)")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    auc_delta = (compressed['auc'] - baseline['auc']) * 100
    overhead = (compressed['mean_lat_ms'] / baseline['mean_lat_ms'] - 1) * 100

    total_orig_mb = sum(state_dict[k].numel() * 4 for k in emb_keys) / 1024 / 1024
    total_comp_mb = total_hot_mb + compressed_mb + total_mapping_mb
    storage_ratio = total_orig_mb / total_comp_mb if total_comp_mb > 0 else 0

    log(f"  Baseline AUC:     {baseline['auc']:.6f}")
    log(f"  Compressed AUC:   {compressed['auc']:.6f}")
    log(f"  AUC delta:        {auc_delta:+.4f}%")
    log(f"")
    log(f"  Baseline latency: {baseline['mean_lat_ms']:.2f} ms/batch")
    log(f"  Compressed:       {compressed['mean_lat_ms']:.2f} ms/batch")
    log(f"  Overhead:         {overhead:+.1f}%")
    log(f"")
    log(f"  Original size:    {total_orig_mb:.1f} MB (all fp32)")
    log(f"  Compressed size:  {total_comp_mb:.1f} MB (hot fp32 + H.265 + mapping)")
    log(f"  Storage ratio:    {storage_ratio:.1f}x")

    # Save results
    results = {
        'config': {
            'crf': args.crf, 'resolution': args.resolution,
            'width': width, 'height': height, 'cache_size': args.cache_size,
            'has_cpp': HAS_CPP,
        },
        'baseline': baseline,
        'compressed': compressed,
        'summary': {
            'auc_delta_pct': auc_delta,
            'overhead_pct': overhead,
            'storage_ratio': storage_ratio,
            'original_mb': total_orig_mb,
            'compressed_mb': total_comp_mb,
        }
    }
    results_path = os.path.join(output_dir, 'demo_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Results saved to {results_path}")


if __name__ == '__main__':
    main()
