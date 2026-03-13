#!/usr/bin/env python3
"""
Benchmark: Python vs C++ Overhead During DLRM Inference

Compares the per-batch overhead of Python-only vs C++-optimized embedding
lookup paths during actual inference with compressed embeddings.

Measures:
  1. Baseline (fp32, no compression)
  2. Compressed + Python-only path (reshape/transpose tiling, numpy dequant)
  3. Compressed + C++ path (fused gather+dequant, merged mapping)
  4. Compressed + Full C++ fast_forward with LRU frame cache

Usage:
    python3 benchmark_python_vs_cpp_inference.py [--num-batches 500] [--resolution 1080p]
    python3 benchmark_python_vs_cpp_inference.py --cache-size 20  # LRU cache for fast_forward (default)
    python3 benchmark_python_vs_cpp_inference.py --cache-size 0   # pre-decode all frames (no LRU)
"""

import os, sys, time, json, gc, argparse, copy, io, tempfile
import numpy as np
import torch
import torch.nn as nn
from collections import Counter, OrderedDict
from sklearn.metrics import roc_auc_score
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import compressed_emb as _C
    HAS_CPP = True
except ImportError:
    _C = None
    HAS_CPP = False
    print("WARNING: C++ extension not available. Only Python path will run.")

try:
    import av
except ImportError:
    pass

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

HOT_COVERAGE = 0.80
LARGE_TABLE_THRESHOLD = 50000
PROFILE_BATCHES = 0  # 0 = profile ALL training batches (recommended for accuracy)

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
# Tiling: Python implementations
# ============================================================
def rows_to_tiled_frame_py(emb_rows, width, height):
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
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    grid = frame.reshape(tiles_per_col, TILE_H, tiles_per_row, TILE_W)
    rows = grid.transpose(0, 2, 1, 3).reshape(rows_per_frame, EMB_DIM)
    return rows


# ============================================================
# Model loading (reuse from existing code)
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
    return dlrm, train_ld, test_ld, ln_emb, state_dict, emb_keys


def profile_and_split(train_ld, ln_emb, state_dict, emb_keys):
    """Profile access patterns and build hot/cold split."""
    num_tables = len(ln_emb)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    profile_limit = "ALL" if PROFILE_BATCHES == 0 else str(PROFILE_BATCHES)
    log(f"Profiling {profile_limit} batches for {len(large_tables)} large tables...")

    access_counts = {t: Counter() for t in large_tables}
    n_batches = 0
    for X, lS_o, lS_i, T in train_ld:
        for t in large_tables:
            for idx in lS_i[t].numpy():
                access_counts[t][idx] += 1
        n_batches += 1
        if PROFILE_BATCHES > 0 and n_batches >= PROFILE_BATCHES:
            break

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

        n = ln_emb[t]
        is_hot_t = torch.zeros(n, dtype=torch.bool)
        for r in hot_set:
            is_hot_t[r] = True
        is_hot[t] = is_hot_t
        hot_indices[t] = torch.where(is_hot_t)[0]
        cold_indices[t] = torch.where(~is_hot_t)[0]

        # Frequency-sorted cold order
        cold_set = cold_indices[t].tolist()
        cold_freq = [(r, counts.get(r, 0)) for r in cold_set]
        cold_freq.sort(key=lambda x: -x[1])
        cold_order = [r for r, _ in cold_freq]

        o2c = torch.full((n,), -1, dtype=torch.long)
        for new_idx, orig_idx in enumerate(cold_order):
            o2c[orig_idx] = new_idx
        orig_to_cold_reordered[t] = o2c

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

        n_hot = len(hot_indices[t])
        n_cold = len(cold_indices[t])
        log(f"  Table {t}: {n_hot:,} hot + {n_cold:,} cold")

    return (large_tables, is_hot, hot_indices, cold_indices,
            orig_to_cold_reordered, cold_weights_q, cold_quant_params)


HOTCOLD_DIR = os.path.join("results", "hotcold")
REORDER_DIR = os.path.join("results", "reorder")


def load_saved_profile(ln_emb, state_dict, emb_keys):
    """Load saved profiling data from results/hotcold/ and results/reorder/.
    Returns the same tuple as profile_and_split, or None if files don't exist."""
    num_tables = len(ln_emb)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]

    # Check if saved data exists for all large tables
    for t in large_tables:
        if not os.path.exists(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt')):
            return None
        if not (os.path.exists(os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')) or
                os.path.exists(os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.npy'))):
            return None

    log(f"Loading saved profiling data for {len(large_tables)} large tables...")
    is_hot = {}
    hot_indices = {}
    cold_indices = {}
    orig_to_cold_reordered = {}
    cold_weights_q = {}
    cold_quant_params = {}

    for t in large_tables:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.load(os.path.join(HOTCOLD_DIR, f'hot_indices_{t}.pt'),
                                    map_location='cpu', weights_only=True)
        cold_indices[t] = torch.load(os.path.join(HOTCOLD_DIR, f'cold_indices_{t}.pt'),
                                     map_location='cpu', weights_only=True)

        # Load frequency-reordered cold mapping
        npy_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.npy')
        pt_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        if os.path.exists(npy_path):
            orig_to_cold_reordered[t] = torch.from_numpy(np.load(npy_path).copy()).long()
        else:
            orig_to_cold_reordered[t] = torch.load(pt_path, map_location='cpu',
                                                    weights_only=True).long()

        # Load quantization params (global min/max across all cold rows)
        mins_path = os.path.join(REORDER_DIR, f'quant_mins_{t}.pt')
        maxs_path = os.path.join(REORDER_DIR, f'quant_maxs_{t}.pt')
        if os.path.exists(mins_path) and os.path.exists(maxs_path):
            mins_t = torch.load(mins_path, map_location='cpu', weights_only=True)
            maxs_t = torch.load(maxs_path, map_location='cpu', weights_only=True)
            mn = mins_t.min().item()
            mx = maxs_t.max().item()
        else:
            # Fallback: recompute from cold weights
            cold_order_path = os.path.join(REORDER_DIR, f'cold_order_{t}.npy')
            if os.path.exists(cold_order_path):
                cold_order = np.load(cold_order_path)
                cold_w = state_dict[emb_keys[t]][torch.from_numpy(cold_order)]
            else:
                cold_w = state_dict[emb_keys[t]][cold_indices[t]]
            mn = cold_w.min().item()
            mx = cold_w.max().item()

        s = (mx - mn) / 255.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        cold_quant_params[t] = (s, zp)

        # cold_weights_q not needed when loading pre-compressed frames
        cold_weights_q[t] = torch.empty(0)

        n_hot = len(hot_indices[t])
        n_cold = len(cold_indices[t])
        log(f"  Table {t}: {n_hot:,} hot + {n_cold:,} cold (loaded from disk)")

    return (large_tables, is_hot, hot_indices, cold_indices,
            orig_to_cold_reordered, cold_weights_q, cold_quant_params)


# ============================================================
# Compressed EmbeddingBag: PYTHON-ONLY path
# ============================================================
class PythonCompressedEmbeddingBag(nn.Module):
    """Pure Python implementation — NO C++ calls.
    Decodes H.265 compressed bytes on cache miss using PyAV."""
    def __init__(self, hot_weight, is_hot, orig_to_hot, orig_to_cold_reordered,
                 compressed_bytes_list, rows_per_frame, width, height, quant_scale, quant_zp,
                 n_cold, num_embeddings):
        super().__init__()
        self.embedding_dim = EMB_DIM
        self.num_embeddings = num_embeddings
        self.hot_weight = hot_weight
        self.mode = 'sum'

        # Build mapping
        INT32_MIN = -2147483648
        mapping = torch.full((num_embeddings,), INT32_MIN, dtype=torch.int32)
        hot_mask = is_hot.bool()
        mapping[hot_mask] = orig_to_hot[hot_mask].to(torch.int32)
        cold_mask = orig_to_cold_reordered >= 0
        mapping[cold_mask] = (-(orig_to_cold_reordered[cold_mask] + 1)).to(torch.int32)
        self.mapping = mapping

        self.compressed_bytes_list = compressed_bytes_list  # list of bytes objects (H.265)
        self.rows_per_frame = rows_per_frame
        self.width = width
        self.height = height
        self.quant_scale = quant_scale
        self.quant_zp = quant_zp
        self.n_cold = n_cold

        self.emb_time = 0.0
        self.cold_time = 0.0

        # Fine-grained operation timers (seconds)
        self.t_mapping = 0.0       # mapping lookup + hot/cold split
        self.t_hot_gather = 0.0    # hot embedding gather
        self.t_decode = 0.0        # H.265 in-memory decode time
        self.t_untile = 0.0        # frame untiling (reshape+transpose)
        self.t_gather = 0.0        # row indexing from untiled frame
        self.t_dequant = 0.0       # uint8 → fp32 dequantization
        self.t_scatter = 0.0       # cold result scatter back
        self.t_pooling = 0.0       # sum pooling (scatter_add)
        self.n_cold_rows = 0       # total cold rows looked up
        self.n_cold_batches = 0    # batches that had cold lookups

    def forward(self, indices, offsets, per_sample_weights=None):
        t0 = time.time()

        # Mapping lookup + hot/cold split
        map_vals = self.mapping[indices]
        hot_mask = map_vals >= 0
        all_embeds = torch.zeros(len(indices), EMB_DIM)
        t1 = time.time()
        self.t_mapping += t1 - t0

        # Hot gather
        if hot_mask.any():
            hot_idx = map_vals[hot_mask].long()
            all_embeds[hot_mask] = self.hot_weight[hot_idx]
        t2 = time.time()
        self.t_hot_gather += t2 - t1

        # Cold lookup: decode H.265 from memory, untile, index, dequantize
        INT32_MIN = -2147483648
        cold_mask = (map_vals < 0) & (map_vals != INT32_MIN)
        if cold_mask.any():
            cold_map = map_vals[cold_mask]
            cold_reordered = -(cold_map.long() + 1)
            valid = cold_reordered >= 0
            if valid.any():
                self.n_cold_batches += 1
                self.n_cold_rows += valid.sum().item()
                valid_idx = cold_reordered[valid]
                frame_ids = (valid_idx // self.rows_per_frame).long()
                row_offsets = (valid_idx % self.rows_per_frame).long()
                unique_frames = torch.unique(frame_ids).tolist()

                result = torch.zeros(valid.sum(), EMB_DIM)
                for fid in unique_frames:
                    mask = frame_ids == fid
                    offsets_in_frame = row_offsets[mask].numpy()

                    # DECODE H.265 from in-memory bytes (PyAV)
                    td0 = time.time()
                    data = self.compressed_bytes_list[fid]
                    container = av.open(io.BytesIO(data))
                    frame_av = next(container.decode(video=0))
                    frame_np = frame_av.to_ndarray(format='gray')
                    container.close()
                    td1 = time.time()
                    self.t_decode += td1 - td0

                    # PYTHON untile: reshape + transpose
                    tu0 = time.time()
                    all_rows = tiled_frame_to_rows_py(frame_np, self.width, self.height)
                    tu1 = time.time()
                    self.t_untile += tu1 - tu0

                    # Row gather (indexing)
                    selected = all_rows[offsets_in_frame]
                    tg1 = time.time()
                    self.t_gather += tg1 - tu1

                    # PYTHON dequantize: numpy arithmetic
                    fp32 = (selected.astype(np.float32) - self.quant_zp) * self.quant_scale
                    result[mask] = torch.from_numpy(fp32)
                    td1 = time.time()
                    self.t_dequant += td1 - tg1

                # Scatter cold results back
                ts0 = time.time()
                cold_positions = torch.where(cold_mask)[0]
                all_embeds[cold_positions[valid]] = result
                self.t_scatter += time.time() - ts0

        t_cold = time.time()
        self.cold_time += t_cold - t2

        if per_sample_weights is not None:
            all_embeds = all_embeds * per_sample_weights.unsqueeze(1)

        # Sum pooling (Python scatter_add)
        tp0 = time.time()
        num_bags = len(offsets)
        output = torch.zeros(num_bags, EMB_DIM)
        bag_ids = torch.bucketize(torch.arange(len(indices)), offsets, right=True) - 1
        bag_ids = bag_ids.clamp(min=0)
        output.scatter_add_(0, bag_ids.unsqueeze(1).expand_as(all_embeds), all_embeds)
        self.t_pooling += time.time() - tp0

        self.emb_time += time.time() - t0
        return output


# ============================================================
# Compressed EmbeddingBag: C++ path
# ============================================================
class CppCompressedEmbeddingBag(nn.Module):
    """C++ optimized path — merged mapping, fused gather+dequant.
    Decodes H.265 compressed bytes on cache miss using C++ decode_h265_frame_from_bytes."""
    def __init__(self, hot_weight, is_hot, orig_to_hot, orig_to_cold_reordered,
                 compressed_bytes_list, rows_per_frame, width, height, quant_scale, quant_zp,
                 n_cold, num_embeddings):
        super().__init__()
        self.embedding_dim = EMB_DIM
        self.num_embeddings = num_embeddings
        self.hot_weight = hot_weight
        self.mode = 'sum'

        INT32_MIN = -2147483648
        mapping = torch.full((num_embeddings,), INT32_MIN, dtype=torch.int32)
        hot_mask = is_hot.bool()
        mapping[hot_mask] = orig_to_hot[hot_mask].to(torch.int32)
        cold_mask = orig_to_cold_reordered >= 0
        mapping[cold_mask] = (-(orig_to_cold_reordered[cold_mask] + 1)).to(torch.int32)
        self.mapping = mapping

        self.compressed_bytes_list = compressed_bytes_list  # list of bytes objects (H.265)
        self.rows_per_frame = rows_per_frame
        self.tiles_per_row = width // TILE_W
        self.width = width
        self.height = height
        self.quant_scale = quant_scale
        self.quant_zp = quant_zp
        self.n_cold = n_cold

        self._has_merged = hasattr(_C, 'compressed_emb_bag_forward_merged')
        self._empty_psw = torch.empty(0)

        self.emb_time = 0.0
        self.cold_time = 0.0

        # Fine-grained operation timers (seconds)
        self.t_mapping = 0.0           # C++ merged mapping + hot forward + pooling
        self.t_hot_gather = 0.0        # (included in t_mapping for C++)
        self.t_decode = 0.0            # H.265 in-memory decode time
        self.t_gather_dequant = 0.0    # C++ fused gather+dequant from tiled frame
        self.t_cold_fixup = 0.0        # C++ cold_fixup (scatter into output)
        self.t_cold_index = 0.0        # cold index extraction from mapping
        self.t_pooling = 0.0           # (included in t_mapping for C++)
        # Note: C++ has no separate untile, gather, dequant — they are fused
        self.t_untile = 0.0            # always 0 for C++ (fused)
        self.t_gather = 0.0            # always 0 for C++ (fused)
        self.t_dequant = 0.0           # always 0 for C++ (fused)
        self.t_scatter = 0.0           # alias for cold_fixup
        self.n_cold_rows = 0
        self.n_cold_batches = 0

    def forward(self, indices, offsets, per_sample_weights=None):
        t0 = time.time()
        psw = per_sample_weights if per_sample_weights is not None else self._empty_psw

        # C++ hot forward with merged mapping (includes mapping + hot gather + pooling)
        output, cold_mask, cold_count = _C.compressed_emb_bag_forward_merged(
            indices, offsets, self.hot_weight, self.mapping, psw)
        t_hot = time.time()
        self.t_mapping += t_hot - t0

        if cold_count.item() > 0:
            # Cold index extraction
            tc0 = time.time()
            cold_positions = torch.where(cold_mask)[0]
            cold_orig = indices[cold_positions]
            cold_map_vals = self.mapping[cold_orig]
            cold_reordered = -(cold_map_vals.long() + 1)
            valid = cold_reordered >= 0
            self.t_cold_index += time.time() - tc0

            if valid.any():
                self.n_cold_batches += 1
                self.n_cold_rows += valid.sum().item()
                valid_idx = cold_reordered[valid]
                frame_ids = (valid_idx // self.rows_per_frame).long()
                row_offsets = (valid_idx % self.rows_per_frame).long()
                unique_frames = torch.unique(frame_ids).tolist()

                # Decode H.265 from in-memory bytes + C++ fused gather+dequant
                tgd0 = time.time()
                result = torch.zeros(valid.sum(), EMB_DIM)
                for fid in unique_frames:
                    mask = frame_ids == fid
                    offsets_in_frame = row_offsets[mask]
                    # Decode from in-memory bytes
                    data = self.compressed_bytes_list[fid]
                    compressed_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                    tiled_frame = _C.decode_h265_frame_from_bytes(compressed_t)
                    td1 = time.time()
                    self.t_decode += td1 - tgd0
                    # C++ fused gather+dequant from tiled frame
                    rows = _C.gather_dequant_from_tiled_frame(
                        tiled_frame, offsets_in_frame,
                        self.tiles_per_row, self.quant_scale, self.quant_zp)
                    result[mask] = rows
                self.t_gather_dequant += time.time() - tgd0

                # C++ cold_fixup: scatter cold results into output
                tf0 = time.time()
                _C.cold_fixup(output, indices, offsets, cold_mask,
                              result, cold_positions[valid], psw)
                self.t_cold_fixup += time.time() - tf0
                self.t_scatter += time.time() - tf0

        t_cold = time.time()
        self.cold_time += t_cold - t_hot
        self.emb_time += time.time() - t0
        return output


# ============================================================
# Inference runner with detailed timing
# ============================================================
def run_inference(dlrm, test_batches, label, num_batches, large_tables):
    n = len(test_batches) if num_batches == 0 else min(num_batches, len(test_batches))
    max_samples = n * TEST_BATCH_SIZE + TEST_BATCH_SIZE
    all_scores = np.empty(max_samples, dtype=np.float32)
    all_targets = np.empty(max_samples, dtype=np.float32)
    sample_idx = 0
    latencies = []

    dlrm.time_look_up = 0; dlrm.time_interact = 0; dlrm.time_mlp = 0

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

    total_time = time.time() - t0
    auc = roc_auc_score(all_targets[:sample_idx], all_scores[:sample_idx])
    mem = rss_mb()

    # Collect per-table timing (aggregate across all large tables)
    cold_time_total = 0
    emb_time_total = 0
    op_timers = {
        't_mapping': 0.0, 't_hot_gather': 0.0,
        't_decode': 0.0,
        't_untile': 0.0, 't_gather': 0.0, 't_dequant': 0.0,
        't_scatter': 0.0, 't_pooling': 0.0,
        't_gather_dequant': 0.0, 't_cold_fixup': 0.0, 't_cold_index': 0.0,
    }
    total_cold_rows = 0
    total_cold_batches = 0

    total_cache_hits = 0
    total_cache_misses = 0

    for t in large_tables:
        E = dlrm.emb_l[t]
        if hasattr(E, 'cold_time'):
            cold_time_total += E.cold_time
            emb_time_total += E.emb_time
        for key in op_timers:
            if hasattr(E, key):
                op_timers[key] += getattr(E, key)
        if hasattr(E, 'n_cold_rows'):
            total_cold_rows += E.n_cold_rows
            total_cold_batches += E.n_cold_batches
        if hasattr(E, 'cache_hits'):
            total_cache_hits += E.cache_hits
            total_cache_misses += E.cache_misses

    total_cache_accesses = total_cache_hits + total_cache_misses

    result = {
        'auc': auc,
        'total_time': total_time,
        'mean_lat_ms': np.mean(latencies) * 1000,
        'p50_lat_ms': np.percentile(latencies, 50) * 1000,
        'p99_lat_ms': np.percentile(latencies, 99) * 1000,
        'rss_mb': mem,
        'emb_lookup_ms': dlrm.time_look_up / n * 1000,
        'interact_ms': dlrm.time_interact / n * 1000,
        'mlp_ms': dlrm.time_mlp / n * 1000,
        'cold_total_ms': cold_time_total * 1000,
        'cold_per_batch_ms': cold_time_total / n * 1000,
        'emb_total_ms': emb_time_total * 1000,
        'emb_per_batch_ms': emb_time_total / n * 1000,
        'total_cold_rows': total_cold_rows,
        'total_cold_batches': total_cold_batches,
        'cache_hits': total_cache_hits,
        'cache_misses': total_cache_misses,
        'cache_hit_rate': total_cache_hits / total_cache_accesses if total_cache_accesses > 0 else 0.0,
    }
    # Per-operation timing (ms per batch)
    for key, val in op_timers.items():
        result[f'op_{key}_ms'] = val / n * 1000

    log(f"\n  [{label}] Results:")
    log(f"    AUC = {auc:.6f}")
    log(f"    Latency: mean={result['mean_lat_ms']:.2f}ms, "
        f"p50={result['p50_lat_ms']:.2f}ms, p99={result['p99_lat_ms']:.2f}ms")
    log(f"    Forward breakdown: emb={result['emb_lookup_ms']:.2f}ms, "
        f"interact={result['interact_ms']:.2f}ms, mlp={result['mlp_ms']:.2f}ms")
    if emb_time_total > 0:
        log(f"    Cold lookups: {total_cold_rows:,} rows across "
            f"{total_cold_batches} batches ({total_cold_rows/n:.1f} cold rows/batch)")
        if total_cache_accesses > 0:
            log(f"    Frame cache: {total_cache_hits}/{total_cache_accesses} hits "
                f"({result['cache_hit_rate']*100:.1f}%), "
                f"{total_cache_misses} decodes")
        log(f"    ---- Per-operation timing (ms/batch, summed across all large tables) ----")
        log(f"    Mapping + hot/cold split: {result['op_t_mapping_ms']:.4f} ms")
        log(f"    Hot gather:               {result['op_t_hot_gather_ms']:.4f} ms")
        if result['op_t_decode_ms'] > 0:
            log(f"    H.265 decode (memory):    {result['op_t_decode_ms']:.4f} ms")
        log(f"    Frame untiling:           {result['op_t_untile_ms']:.4f} ms")
        log(f"    Row gather (indexing):    {result['op_t_gather_ms']:.4f} ms")
        log(f"    Dequantization:           {result['op_t_dequant_ms']:.4f} ms")
        if result['op_t_gather_dequant_ms'] > 0:
            log(f"    Fused gather+dequant:     {result['op_t_gather_dequant_ms']:.4f} ms")
        if result['op_t_cold_fixup_ms'] > 0:
            log(f"    Cold fixup (scatter):     {result['op_t_cold_fixup_ms']:.4f} ms")
        if result['op_t_cold_index_ms'] > 0:
            log(f"    Cold index extraction:    {result['op_t_cold_index_ms']:.4f} ms")
        log(f"    Result scatter:           {result['op_t_scatter_ms']:.4f} ms")
        log(f"    Sum pooling:              {result['op_t_pooling_ms']:.4f} ms")
    log(f"    RSS = {mem:.0f}MB")

    return result


def _find_existing_compressed(compressed_dir, large_tables, rows_per_frame):
    """Check if compressed frames already exist on disk. Returns dict or None."""
    if not compressed_dir or not os.path.isdir(compressed_dir):
        return None
    result = {}
    for t in large_tables:
        table_dir = os.path.join(compressed_dir, f'table_{t}')
        if not os.path.isdir(table_dir):
            continue
        frame_files = sorted([f for f in os.listdir(table_dir)
                              if f.startswith('frame_') and
                              (f.endswith('.h265') or f.endswith('.h264') or f.endswith('.mkv'))])
        if not frame_files:
            continue
        frame_bytes = []
        for ff in frame_files:
            with open(os.path.join(table_dir, ff), 'rb') as fh:
                frame_bytes.append(fh.read())
        result[t] = (frame_bytes, rows_per_frame)
    return result if result else None


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Python vs C++ Inference Overhead")
    parser.add_argument('--num-batches', type=int, default=500,
                        help='Number of test batches (0=all)')
    parser.add_argument('--resolution', type=str, default='1080p', choices=['1080p', '4K'])
    parser.add_argument('--crf', type=int, default=0, help='H.265 CRF (0=lossless)')
    parser.add_argument('--compressed-dir', type=str, default=None,
                        help='Load pre-compressed frames from this dir (skip encoding)')
    parser.add_argument('--cache-size', type=int, default=20,
                        help='LRU frame cache size per table (0=no cache, default 20)')
    args = parser.parse_args()

    width, height = RESOLUTIONS[args.resolution]
    rows_per_frame = (width // TILE_W) * (height // TILE_H)

    log("=" * 70)
    log("PYTHON vs C++ INFERENCE OVERHEAD BENCHMARK")
    log(f"Resolution={args.resolution} ({width}x{height}), "
        f"Rows/frame={rows_per_frame:,}, Batches={args.num_batches}")
    log(f"C++ extension: {'YES' if HAS_CPP else 'NO'}, "
        f"LRU cache: {args.cache_size} frames/table")
    log("=" * 70)

    # Load
    dlrm, train_ld, test_ld, ln_emb, state_dict, emb_keys = load_model_and_data()
    log("Pre-caching test batches...")
    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    log(f"  {len(test_batches)} batches cached")

    # Profile — use saved data when loading pre-compressed frames to avoid mapping mismatch
    saved = load_saved_profile(ln_emb, state_dict, emb_keys) if args.compressed_dir else None
    if saved:
        (large_tables, is_hot, hot_indices, cold_indices,
         orig_to_cold_reordered, cold_weights_q, cold_quant_params) = saved
    else:
        (large_tables, is_hot, hot_indices, cold_indices,
         orig_to_cold_reordered, cold_weights_q, cold_quant_params) = \
            profile_and_split(train_ld, ln_emb, state_dict, emb_keys)

    # Load or encode compressed frames into RAM.
    # During inference, cache misses decode from these in-memory bytes (no disk I/O).
    cold_compressed_bytes_per_table = {}  # table -> list of bytes objects
    cold_num_frames_per_table = {}

    existing = _find_existing_compressed(args.compressed_dir, large_tables, rows_per_frame) \
        if args.compressed_dir else None

    if existing:
        log(f"\nLoading pre-compressed frames from {args.compressed_dir}...")
        for t, (frame_bytes, rpf) in existing.items():
            cold_compressed_bytes_per_table[t] = frame_bytes
            cold_num_frames_per_table[t] = len(frame_bytes)
            comp_bytes = sum(len(b) for b in frame_bytes)
            n_cold = len(cold_indices[t])
            log(f"  Table {t}: loaded {len(frame_bytes)} frames "
                f"({comp_bytes/1024:.1f}KB) for {n_cold:,} cold rows")
    else:
        log("\nEncoding cold frames to H.265 and storing compressed bytes in RAM...")
        for t in large_tables:
            q = cold_weights_q[t].numpy()
            n_cold = len(q)
            num_frames = max(1, (n_cold + rows_per_frame - 1) // rows_per_frame)
            padded = np.zeros((num_frames * rows_per_frame, EMB_DIM), dtype=np.uint8)
            padded[:n_cold] = q

            # Tile frames
            if HAS_CPP:
                q_t = torch.from_numpy(padded)
                tiled = _C.fused_quantize_tile_multiframe(q_t, width, height)
            else:
                tiled = []
                for i in range(num_frames):
                    chunk = padded[i * rows_per_frame:(i + 1) * rows_per_frame]
                    tiled.append(torch.from_numpy(rows_to_tiled_frame_py(chunk, width, height)))

            # Encode each tiled frame to H.265 and store compressed bytes in memory
            compressed_list = []
            total_compressed = 0
            for i in range(num_frames):
                frame_t = tiled[i] if isinstance(tiled, list) else tiled[i]
                if not isinstance(frame_t, torch.Tensor):
                    frame_t = torch.from_numpy(frame_t)
                frame_t = frame_t.to(torch.uint8).contiguous()
                if HAS_CPP and hasattr(_C, 'encode_h265_frame'):
                    compressed_t = _C.encode_h265_frame(frame_t, "", True, 0)
                    compressed_bytes = bytes(compressed_t.numpy().tobytes())
                else:
                    with tempfile.NamedTemporaryFile(suffix='.h265', delete=True) as tmp:
                        tmp_path = tmp.name
                        container = av.open(tmp_path, mode='w', format='matroska')
                        stream = container.add_stream('libx265', rate=1)
                        stream.width = width; stream.height = height
                        stream.pix_fmt = 'gray'
                        stream.options = {'preset': 'ultrafast',
                                          'x265-params': 'lossless=1:log-level=error'}
                        avframe = av.VideoFrame.from_ndarray(frame_t.numpy(), format='gray')
                        for pkt in stream.encode(avframe):
                            container.mux(pkt)
                        for pkt in stream.encode():
                            container.mux(pkt)
                        container.close()
                        with open(tmp_path, 'rb') as f:
                            compressed_bytes = f.read()
                compressed_list.append(compressed_bytes)
                total_compressed += len(compressed_bytes)

            cold_compressed_bytes_per_table[t] = compressed_list
            cold_num_frames_per_table[t] = num_frames
            raw_bytes = n_cold * EMB_DIM
            ratio = raw_bytes / total_compressed if total_compressed > 0 else 0
            log(f"  Table {t}: {num_frames} frames ({n_cold:,} cold rows), "
                f"{raw_bytes/1024:.0f}KB → {total_compressed/1024:.0f}KB H.265 ({ratio:.1f}x)")

    # Helper to build compressed emb for each table
    def build_hot_data(t):
        w = state_dict[emb_keys[t]]
        h_idx = hot_indices[t]
        hot_weight = w[h_idx].clone()
        orig_to_hot = torch.full((ln_emb[t],), -1, dtype=torch.long)
        orig_to_hot[h_idx] = torch.arange(len(h_idx))
        return hot_weight, orig_to_hot

    # Save original EmbeddingBag modules so we can restore between experiments
    original_emb_modules = {t: dlrm.emb_l[t] for t in range(len(dlrm.emb_l))}

    def restore_weights():
        with torch.no_grad():
            for t_idx, orig_mod in original_emb_modules.items():
                dlrm.emb_l[t_idx] = orig_mod
            for k in emb_keys:
                t_idx = int(k.split('.')[1])
                dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

    all_results = {}

    # ---- Experiment 1: Baseline ----
    log("\n" + "=" * 70)
    log("EXPERIMENT 1: Baseline (fp32, no compression)")
    log("=" * 70)
    restore_weights()
    gc.collect()
    all_results['baseline'] = run_inference(
        dlrm, test_batches, "Baseline", args.num_batches, large_tables)

    # ---- Experiment 2: Python-only compressed ----
    log("\n" + "=" * 70)
    log("EXPERIMENT 2: Compressed (PYTHON-ONLY path)")
    log("  - H.265 in-memory decode via PyAV")
    log("  - Python reshape+transpose untiling")
    log("  - Python numpy dequantization")
    log("  - Python scatter_add pooling")
    log("=" * 70)
    restore_weights()
    for t in large_tables:
        hot_weight, orig_to_hot = build_hot_data(t)
        s, zp = cold_quant_params[t]
        dlrm.emb_l[t] = PythonCompressedEmbeddingBag(
            hot_weight=hot_weight, is_hot=is_hot[t],
            orig_to_hot=orig_to_hot,
            orig_to_cold_reordered=orig_to_cold_reordered[t],
            compressed_bytes_list=cold_compressed_bytes_per_table[t],
            rows_per_frame=rows_per_frame,
            width=width, height=height,
            quant_scale=s, quant_zp=zp,
            n_cold=len(cold_indices[t]),
            num_embeddings=ln_emb[t],
        )
    gc.collect()
    all_results['python'] = run_inference(
        dlrm, test_batches, "Python", args.num_batches, large_tables)

    # ---- Experiment 3: C++ compressed ----
    if HAS_CPP:
        log("\n" + "=" * 70)
        log("EXPERIMENT 3: Compressed (C++ path)")
        log("  - H.265 in-memory decode via C++ on cache miss")
        log("  - C++ merged mapping forward")
        log("  - C++ fused gather+dequant from tiled frame")
        log("  - C++ cold_fixup (in-place)")
        log("=" * 70)
        restore_weights()
        for t in large_tables:
            hot_weight, orig_to_hot = build_hot_data(t)
            s, zp = cold_quant_params[t]
            dlrm.emb_l[t] = CppCompressedEmbeddingBag(
                hot_weight=hot_weight, is_hot=is_hot[t],
                orig_to_hot=orig_to_hot,
                orig_to_cold_reordered=orig_to_cold_reordered[t],
                compressed_bytes_list=cold_compressed_bytes_per_table[t],
                rows_per_frame=rows_per_frame,
                width=width, height=height,
                quant_scale=s, quant_zp=zp,
                n_cold=len(cold_indices[t]),
                num_embeddings=ln_emb[t],
            )
        gc.collect()
        all_results['cpp'] = run_inference(
            dlrm, test_batches, "C++", args.num_batches, large_tables)

    # ---- Experiment 4: Full C++ (fast_forward + LRU cache) ----
    if HAS_CPP and hasattr(_C, 'fast_forward') and hasattr(_C, 'register_tables'):
        log("\n" + "=" * 70)
        cache_label = f"LRU cache={args.cache_size}" if args.cache_size > 0 else "all frames pre-decoded"
        log(f"EXPERIMENT 4: Compressed (Full C++ fast_forward, {cache_label})")
        log("  - All tables registered in C++")
        log("  - Single C++ fast_forward call per batch")
        if args.cache_size > 0:
            log(f"  - LRU frame cache ({args.cache_size} frames), on-demand H.265 decode on miss")
            log("  - Re-registers dirty tables in C++ when cache changes")
        else:
            log("  - All cold frames pre-decoded from H.265 bytes and registered")
        log("=" * 70)
        restore_weights()

        # Register tables (hot weights + mappings)
        num_tabs = len(dlrm.emb_l)
        table_kinds = []
        weights_list = []
        mappings_list = []
        scales_list = []
        zp_list = []
        compressed_tables = set()

        for k in range(num_tabs):
            if k in large_tables and k in cold_compressed_bytes_per_table:
                hot_weight, orig_to_hot = build_hot_data(k)
                INT32_MIN = -2147483648
                mapping = torch.full((ln_emb[k],), INT32_MIN, dtype=torch.int32)
                mapping[is_hot[k]] = orig_to_hot[is_hot[k]].to(torch.int32)
                o2c = orig_to_cold_reordered[k]
                cold_mask = o2c >= 0
                mapping[cold_mask] = (-(o2c[cold_mask] + 1)).to(torch.int32)

                table_kinds.append(1)
                weights_list.append(hot_weight)
                mappings_list.append(mapping)
                s, zp = cold_quant_params[k]
                scales_list.append(float(s))
                zp_list.append(int(zp))
                compressed_tables.add(k)

                dlrm.emb_l[k] = nn.EmbeddingBag(ln_emb[k], EMB_DIM, mode='sum',
                                                  _weight=torch.zeros(ln_emb[k], EMB_DIM))
                dlrm.emb_l[k].mode = 'sum'
            else:
                table_kinds.append(0)
                weights_list.append(dlrm.emb_l[k].weight)
                mappings_list.append(torch.empty(0, dtype=torch.int32))
                scales_list.append(0.0)
                zp_list.append(0)

        _C.register_tables(table_kinds, weights_list, mappings_list, scales_list, zp_list)

        # Override apply_emb with fast_forward (no cold fixup needed —
        # all needed frames are registered before each forward call)
        _orig_apply_emb = dlrm.apply_emb

        def _fast_apply_emb(lS_o, lS_i, emb_l, v_W_l):
            start_time = time.time()
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
            results = _C.fast_forward(lS_i_2d, lS_o_2d)
            dlrm.time_look_up += time.time() - start_time
            return results[-1]

        dlrm.apply_emb = _fast_apply_emb

        if args.cache_size > 0:
            # ---- LRU cache mode: decode on-demand, register per batch ----
            frame_cache = OrderedDict()  # (table_id, frame_id) -> decoded rows (uint8)
            cache_hits = 0
            cache_misses = 0
            total_frames_decoded = 0

            n = len(test_batches) if args.num_batches == 0 else min(args.num_batches, len(test_batches))
            max_samples = n * TEST_BATCH_SIZE + TEST_BATCH_SIZE
            all_scores = np.empty(max_samples, dtype=np.float32)
            all_targets = np.empty(max_samples, dtype=np.float32)
            sample_idx = 0
            latencies = []
            decode_times = []
            dlrm.time_look_up = 0; dlrm.time_interact = 0; dlrm.time_mlp = 0

            gc.collect()
            t0 = time.time()
            with torch.no_grad():
                for batch_idx in range(n):
                    X, lS_o, lS_i, T = test_batches[batch_idx]

                    # Scan for needed frames + manage LRU cache
                    t_dec = time.time()
                    dirty_tables = set()
                    for t_idx in compressed_tables:
                        if isinstance(lS_i, (list, tuple)):
                            indices = lS_i[t_idx]
                        elif lS_i.dim() == 2:
                            indices = lS_i[t_idx]
                        else:
                            indices = lS_i
                        cold_mask_t = ~is_hot[t_idx][indices]
                        if not cold_mask_t.any():
                            continue
                        cold_orig = indices[cold_mask_t]
                        cold_mapped = orig_to_cold_reordered[t_idx][cold_orig]
                        valid = cold_mapped >= 0
                        if not valid.any():
                            continue
                        fids = (cold_mapped[valid] // rows_per_frame).unique().tolist()
                        for fid in fids:
                            key = (t_idx, fid)
                            if key in frame_cache:
                                frame_cache.move_to_end(key)
                                cache_hits += 1
                            else:
                                cache_misses += 1
                                total_frames_decoded += 1
                                # Decode from in-memory bytes
                                data = cold_compressed_bytes_per_table[t_idx][fid]
                                compressed_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                                tiled_frame = _C.decode_h265_frame_from_bytes(compressed_t)
                                rows = _C.untile_frame_to_rows(tiled_frame, rows_per_frame)
                                if rows.shape[0] < rows_per_frame:
                                    rows = torch.cat([rows, torch.zeros(
                                        rows_per_frame - rows.shape[0], EMB_DIM, dtype=torch.uint8)])
                                frame_cache[key] = rows
                                dirty_tables.add(t_idx)

                    # Evict if over capacity
                    while len(frame_cache) > args.cache_size:
                        evicted_key, _ = frame_cache.popitem(last=False)
                        dirty_tables.add(evicted_key[0])

                    # Re-register dirty tables in C++
                    for t_idx in dirty_tables:
                        table_frames = [(fid, data) for (tid, fid), data
                                        in frame_cache.items() if tid == t_idx]
                        if table_frames:
                            table_frames.sort(key=lambda x: x[0])
                            fids_t = torch.tensor([f[0] for f in table_frames], dtype=torch.long)
                            all_data = torch.cat([f[1] for f in table_frames], dim=0)
                            s, zp = cold_quant_params[t_idx]
                            _C.register_cold_frames_for_table(
                                t_idx, fids_t, all_data,
                                float(s), int(zp),
                                rows_per_frame, torch.empty(0, dtype=torch.long))

                    decode_times.append((time.time() - t_dec) * 1000)

                    # Forward
                    bt0 = time.time()
                    Z = dlrm(X, lS_o, lS_i)
                    latencies.append(time.time() - bt0)

                    z_np = Z.detach().cpu().numpy().ravel()
                    t_np = T.detach().cpu().numpy().ravel()
                    bs = z_np.shape[0]
                    all_scores[sample_idx:sample_idx + bs] = z_np
                    all_targets[sample_idx:sample_idx + bs] = t_np
                    sample_idx += bs

            total_time = time.time() - t0
            auc = roc_auc_score(all_targets[:sample_idx], all_scores[:sample_idx])
            total_cache_accesses = cache_hits + cache_misses

            all_results['full_cpp'] = {
                'auc': auc,
                'total_time': total_time,
                'mean_lat_ms': np.mean(latencies) * 1000,
                'p50_lat_ms': np.percentile(latencies, 50) * 1000,
                'p99_lat_ms': np.percentile(latencies, 99) * 1000,
                'rss_mb': rss_mb(),
                'emb_lookup_ms': dlrm.time_look_up / n * 1000,
                'interact_ms': dlrm.time_interact / n * 1000,
                'mlp_ms': dlrm.time_mlp / n * 1000,
                'cold_total_ms': 0, 'cold_per_batch_ms': 0.0,
                'emb_total_ms': 0, 'emb_per_batch_ms': 0.0,
                'total_cold_rows': 0, 'total_cold_batches': 0,
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'cache_hit_rate': cache_hits / total_cache_accesses if total_cache_accesses > 0 else 0.0,
                'total_frames_decoded': total_frames_decoded,
                'decode_ms_per_batch': np.mean(decode_times),
                'op_t_mapping_ms': 0.0, 'op_t_hot_gather_ms': 0.0,
                'op_t_decode_ms': 0.0, 'op_t_untile_ms': 0.0,
                'op_t_gather_ms': 0.0, 'op_t_dequant_ms': 0.0,
                'op_t_scatter_ms': 0.0, 'op_t_pooling_ms': 0.0,
                'op_t_gather_dequant_ms': 0.0,
                'op_t_cold_fixup_ms': 0.0, 'op_t_cold_index_ms': 0.0,
            }

            r = all_results['full_cpp']
            log(f"\n  [Full C++ + LRU] Results:")
            log(f"    AUC = {auc:.6f}")
            log(f"    Latency: mean={r['mean_lat_ms']:.2f}ms, "
                f"p50={r['p50_lat_ms']:.2f}ms, p99={r['p99_lat_ms']:.2f}ms")
            log(f"    Forward breakdown: emb={r['emb_lookup_ms']:.2f}ms, "
                f"interact={r['interact_ms']:.2f}ms, mlp={r['mlp_ms']:.2f}ms")
            log(f"    Frame cache: {cache_hits}/{total_cache_accesses} hits "
                f"({r['cache_hit_rate']*100:.1f}%), "
                f"{total_frames_decoded} total decodes")
            log(f"    Decode overhead: {r['decode_ms_per_batch']:.4f} ms/batch avg")
            log(f"    RSS = {r['rss_mb']:.0f}MB")

        else:
            # ---- Pre-decode all mode (cache_size=0): original behavior ----
            _cold_data = {}
            _mapping_tensors = {}
            for k in compressed_tables:
                s, zp = cold_quant_params[k]
                decoded_frames = []
                for data in cold_compressed_bytes_per_table[k]:
                    compressed_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                    decoded_frames.append(_C.decode_h265_frame_from_bytes(compressed_t))
                _cold_data[k] = {
                    'frames': decoded_frames,
                    'scale': s, 'zp': zp,
                    'tiles_per_row': width // TILE_W,
                }
                _mapping_tensors[k] = mappings_list[k]

            # Register all cold frames
            for t in _cold_data:
                frames = _cold_data[t]['frames']
                if not frames:
                    continue
                sorted_fids = list(range(len(frames)))
                frame_data_list = []
                for f in frames:
                    if f.dim() == 2:
                        rows = _C.untile_frame_to_rows(f, rows_per_frame)
                        frame_data_list.append(rows)
                    else:
                        frame_data_list.append(f)
                all_data = torch.cat(frame_data_list, dim=0)
                frame_ids_t = torch.tensor(sorted_fids, dtype=torch.long)
                _C.register_cold_frames_for_table(
                    t, frame_ids_t, all_data,
                    float(_cold_data[t]['scale']),
                    float(_cold_data[t]['zp']),
                    rows_per_frame,
                    torch.empty(0, dtype=torch.long))

            gc.collect()
            all_results['full_cpp'] = run_inference(
                dlrm, test_batches, "Full C++", args.num_batches, large_tables)

        dlrm.apply_emb = _orig_apply_emb

    # ============================================================
    # Summary table
    # ============================================================
    log("\n" + "=" * 70)
    log("SUMMARY: Python vs C++ Inference Overhead")
    log("=" * 70)

    header = f"{'Experiment':<25} {'AUC':>10} {'Mean(ms)':>10} {'P50(ms)':>10} {'P99(ms)':>10} {'Emb(ms)':>10} {'Cold(ms)':>10} {'Cache%':>8} {'RSS(MB)':>10}"
    log(header)
    log("-" * len(header))

    baseline_lat = all_results['baseline']['mean_lat_ms']

    for name, label in [('baseline', 'Baseline (fp32)'),
                         ('python', 'Python compressed'),
                         ('cpp', 'C++ compressed'),
                         ('full_cpp', 'Full C++ fast_fwd')]:
        if name not in all_results:
            continue
        r = all_results[name]
        overhead = ""
        if name != 'baseline':
            pct = (r['mean_lat_ms'] / baseline_lat - 1) * 100
            overhead = f" ({pct:+.0f}%)"
        hit_rate = r.get('cache_hit_rate', 0)
        cache_str = f"{hit_rate*100:.1f}%" if r.get('cache_hits', 0) + r.get('cache_misses', 0) > 0 else "—"
        log(f"{label:<25} {r['auc']:>10.6f} {r['mean_lat_ms']:>9.2f}{overhead:>0} "
            f"{r['p50_lat_ms']:>10.2f} {r['p99_lat_ms']:>10.2f} "
            f"{r['emb_lookup_ms']:>10.2f} {r['cold_per_batch_ms']:>10.3f} "
            f"{cache_str:>8} {r['rss_mb']:>10.0f}")

    # Per-operation comparison: Python vs C++
    if 'python' in all_results and 'cpp' in all_results:
        py = all_results['python']
        cpp = all_results['cpp']

        log(f"\n  Overall speedup: C++ is "
            f"{py['mean_lat_ms']/cpp['mean_lat_ms']:.2f}x faster per batch")
        log(f"    Python total emb: {py['emb_per_batch_ms']:.3f} ms/batch")
        log(f"    C++    total emb: {cpp['emb_per_batch_ms']:.3f} ms/batch")
        emb_speedup = py['emb_per_batch_ms'] / max(0.001, cpp['emb_per_batch_ms'])
        log(f"    Embedding speedup: {emb_speedup:.1f}x")

        log(f"\n  ---- Per-Operation Comparison (ms/batch) ----")
        ops = [
            ('Mapping + hot/cold split', 'op_t_mapping_ms'),
            ('Hot gather',               'op_t_hot_gather_ms'),
            ('H.265 decode (memory)',    'op_t_decode_ms'),
            ('Frame untiling',           'op_t_untile_ms'),
            ('Row gather (indexing)',     'op_t_gather_ms'),
            ('Dequantization',           'op_t_dequant_ms'),
            ('Fused gather+dequant',     'op_t_gather_dequant_ms'),
            ('Cold fixup / scatter',     'op_t_scatter_ms'),
            ('Sum pooling',              'op_t_pooling_ms'),
        ]
        log(f"  {'Operation':<28} {'Python':>10} {'C++':>10} {'Speedup':>10}")
        log(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}")
        for op_label, key in ops:
            py_val = py.get(key, 0)
            cpp_val = cpp.get(key, 0)
            # Skip rows where both are zero
            if py_val < 0.0001 and cpp_val < 0.0001:
                continue
            if cpp_val > 0.0001:
                speedup = f"{py_val/cpp_val:.1f}x"
            elif py_val > 0:
                speedup = "inf"
            else:
                speedup = "-"
            # For C++ fused ops, annotate
            note = ""
            if key == 'op_t_gather_dequant_ms' and cpp_val > 0 and py_val == 0:
                note = " (C++ fused)"
            if key in ('op_t_untile_ms', 'op_t_gather_ms', 'op_t_dequant_ms') and cpp_val == 0 and py_val > 0:
                note = " (fused in C++)"
            log(f"  {op_label + note:<28} {py_val:>10.4f} {cpp_val:>10.4f} {speedup:>10}")

        # Show what the C++ fused operation replaces
        py_cold_ops = py.get('op_t_untile_ms', 0) + py.get('op_t_gather_ms', 0) + py.get('op_t_dequant_ms', 0)
        cpp_fused = cpp.get('op_t_gather_dequant_ms', 0)
        if py_cold_ops > 0 and cpp_fused > 0:
            log(f"\n  Key insight: Python untile+gather+dequant = {py_cold_ops:.4f} ms/batch")
            log(f"               C++ fused gather_dequant     = {cpp_fused:.4f} ms/batch")
            log(f"               Speedup: {py_cold_ops/cpp_fused:.1f}x")

        # Mapping comparison
        py_map = py.get('op_t_mapping_ms', 0)
        cpp_map = cpp.get('op_t_mapping_ms', 0)
        if py_map > 0 and cpp_map > 0:
            log(f"\n  Mapping+hot+pool: Python={py_map:.4f} ms, "
                f"C++ merged={cpp_map:.4f} ms ({py_map/cpp_map:.1f}x)")

    if 'full_cpp' in all_results:
        full_lat = all_results['full_cpp']['mean_lat_ms']
        log(f"\n  Full C++ (fast_forward) vs Baseline: "
            f"{(full_lat/baseline_lat - 1)*100:+.1f}% overhead")

    # AUC comparison
    log(f"\n  AUC preservation:")
    for name, label in [('python', 'Python'), ('cpp', 'C++'), ('full_cpp', 'Full C++')]:
        if name in all_results:
            delta = (all_results[name]['auc'] - all_results['baseline']['auc']) * 100
            log(f"    {label}: {delta:+.4f}% vs baseline")

    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = f"results/python_vs_cpp_inference_{args.resolution}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\n  Results saved to {results_path}")


if __name__ == '__main__':
    main()
