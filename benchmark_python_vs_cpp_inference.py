#!/usr/bin/env python3
"""
Benchmark: Python vs C++ Overhead During DLRM Inference

Compares the per-batch overhead of Python-only vs C++-optimized embedding
lookup paths during actual inference with compressed embeddings.

Measures:
  1. Baseline (fp32, no compression)
  2. Compressed + Python-only path (reshape/transpose tiling, numpy dequant)
  3. Compressed + C++ path (fused gather+dequant, merged mapping, fast_forward)
  4. Compressed + Full C++ path (all cold frames pre-registered in C++)

Usage:
    python3 benchmark_python_vs_cpp_inference.py [--num-batches 500] [--resolution 1080p]
"""

import os, sys, time, json, gc, argparse, copy
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
PROFILE_BATCHES = 200

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
    log(f"Profiling {PROFILE_BATCHES} batches for {len(large_tables)} large tables...")

    access_counts = {t: Counter() for t in large_tables}
    n_batches = 0
    for X, lS_o, lS_i, T in train_ld:
        for t in large_tables:
            for idx in lS_i[t].numpy():
                access_counts[t][idx] += 1
        n_batches += 1
        if n_batches >= PROFILE_BATCHES:
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


# ============================================================
# Compressed EmbeddingBag: PYTHON-ONLY path
# ============================================================
class PythonCompressedEmbeddingBag(nn.Module):
    """Pure Python implementation — NO C++ calls."""
    def __init__(self, hot_weight, is_hot, orig_to_hot, orig_to_cold_reordered,
                 cold_frames, rows_per_frame, width, height, quant_scale, quant_zp,
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

        self.cold_frames = cold_frames  # list of (H, W) numpy arrays
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

        # Cold lookup: untile full frame, index, dequantize (Python only)
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

                    # PYTHON untile: reshape + transpose
                    tu0 = time.time()
                    frame_np = self.cold_frames[fid]
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
    """C++ optimized path — merged mapping, fused gather+dequant."""
    def __init__(self, hot_weight, is_hot, orig_to_hot, orig_to_cold_reordered,
                 cold_frames, rows_per_frame, width, height, quant_scale, quant_zp,
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

        # Store cold frames as torch tensors (tiled layout for C++ gather)
        self.cold_frames = []
        for f in cold_frames:
            self.cold_frames.append(torch.from_numpy(f) if isinstance(f, np.ndarray) else f)
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

                # C++ fused gather + dequant from tiled frames (no separate untile)
                tgd0 = time.time()
                result = torch.zeros(valid.sum(), EMB_DIM)
                for fid in unique_frames:
                    mask = frame_ids == fid
                    offsets_in_frame = row_offsets[mask]
                    rows = _C.gather_dequant_from_tiled_frame(
                        self.cold_frames[fid], offsets_in_frame,
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
        't_untile': 0.0, 't_gather': 0.0, 't_dequant': 0.0,
        't_scatter': 0.0, 't_pooling': 0.0,
        't_gather_dequant': 0.0, 't_cold_fixup': 0.0, 't_cold_index': 0.0,
    }
    total_cold_rows = 0
    total_cold_batches = 0

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
        log(f"    ---- Per-operation timing (ms/batch, summed across all large tables) ----")
        log(f"    Mapping + hot/cold split: {result['op_t_mapping_ms']:.4f} ms")
        log(f"    Hot gather:               {result['op_t_hot_gather_ms']:.4f} ms")
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


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Python vs C++ Inference Overhead")
    parser.add_argument('--num-batches', type=int, default=500,
                        help='Number of test batches (0=all)')
    parser.add_argument('--resolution', type=str, default='1080p', choices=['1080p', '4K'])
    parser.add_argument('--crf', type=int, default=0, help='H.265 CRF (0=lossless)')
    args = parser.parse_args()

    width, height = RESOLUTIONS[args.resolution]
    rows_per_frame = (width // TILE_W) * (height // TILE_H)

    log("=" * 70)
    log("PYTHON vs C++ INFERENCE OVERHEAD BENCHMARK")
    log(f"Resolution={args.resolution} ({width}x{height}), "
        f"Rows/frame={rows_per_frame:,}, Batches={args.num_batches}")
    log(f"C++ extension: {'YES' if HAS_CPP else 'NO'}")
    log("=" * 70)

    # Load
    dlrm, train_ld, test_ld, ln_emb, state_dict, emb_keys = load_model_and_data()
    log("Pre-caching test batches...")
    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    log(f"  {len(test_batches)} batches cached")

    # Profile
    (large_tables, is_hot, hot_indices, cold_indices,
     orig_to_cold_reordered, cold_weights_q, cold_quant_params) = \
        profile_and_split(train_ld, ln_emb, state_dict, emb_keys)

    # Pre-tile cold frames into memory (skip H.265 for fair comparison —
    # we want to isolate the embedding lookup overhead, not codec speed)
    log("\nPre-tiling cold frames into memory (bypass H.265 for overhead isolation)...")
    cold_frames_per_table = {}
    for t in large_tables:
        q = cold_weights_q[t].numpy()
        n_cold = len(q)
        num_frames = max(1, (n_cold + rows_per_frame - 1) // rows_per_frame)
        padded = np.zeros((num_frames * rows_per_frame, EMB_DIM), dtype=np.uint8)
        padded[:n_cold] = q

        frames = []
        if HAS_CPP:
            q_t = torch.from_numpy(padded)
            tiled = _C.fused_quantize_tile_multiframe(q_t, width, height)
            for i in range(num_frames):
                frames.append(tiled[i].numpy())
        else:
            for i in range(num_frames):
                chunk = padded[i * rows_per_frame:(i + 1) * rows_per_frame]
                frames.append(rows_to_tiled_frame_py(chunk, width, height))

        cold_frames_per_table[t] = frames
        log(f"  Table {t}: {num_frames} frames ({n_cold:,} cold rows)")

    # Helper to build compressed emb for each table
    def build_hot_data(t):
        w = state_dict[emb_keys[t]]
        h_idx = hot_indices[t]
        hot_weight = w[h_idx].clone()
        orig_to_hot = torch.full((ln_emb[t],), -1, dtype=torch.long)
        orig_to_hot[h_idx] = torch.arange(len(h_idx))
        return hot_weight, orig_to_hot

    def restore_weights():
        with torch.no_grad():
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
            cold_frames=cold_frames_per_table[t],
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
                cold_frames=cold_frames_per_table[t],
                rows_per_frame=rows_per_frame,
                width=width, height=height,
                quant_scale=s, quant_zp=zp,
                n_cold=len(cold_indices[t]),
                num_embeddings=ln_emb[t],
            )
        gc.collect()
        all_results['cpp'] = run_inference(
            dlrm, test_batches, "C++", args.num_batches, large_tables)

    # ---- Experiment 4: Full C++ (fast_forward) ----
    if HAS_CPP and hasattr(_C, 'fast_forward') and hasattr(_C, 'register_tables'):
        log("\n" + "=" * 70)
        log("EXPERIMENT 4: Compressed (Full C++ fast_forward)")
        log("  - All tables registered in C++")
        log("  - Single C++ call per batch (zero Python loop)")
        log("  - Cold frames pre-registered for O(1) lookup")
        log("=" * 70)
        restore_weights()

        # Build compressed embedding bags first
        num_tabs = len(dlrm.emb_l)
        table_kinds = []
        weights = []
        mappings = []
        scales = []
        zero_points = []
        _cold_data = {}  # for cold fixup
        _mapping_tensors = {}

        for k in range(num_tabs):
            if k in large_tables and k in cold_frames_per_table:
                hot_weight, orig_to_hot = build_hot_data(k)
                INT32_MIN = -2147483648
                mapping = torch.full((ln_emb[k],), INT32_MIN, dtype=torch.int32)
                mapping[is_hot[k]] = orig_to_hot[is_hot[k]].to(torch.int32)
                o2c = orig_to_cold_reordered[k]
                cold_mask = o2c >= 0
                mapping[cold_mask] = (-(o2c[cold_mask] + 1)).to(torch.int32)

                table_kinds.append(1)  # COMPRESSED_FP32
                weights.append(hot_weight)
                mappings.append(mapping)
                scales.append(0.0)
                zero_points.append(0)
                _mapping_tensors[k] = mapping

                # Store cold cache info for fixup
                s, zp = cold_quant_params[k]
                _cold_data[k] = {
                    'frames': [torch.from_numpy(f) if isinstance(f, np.ndarray) else f
                               for f in cold_frames_per_table[k]],
                    'scale': s, 'zp': zp,
                    'tiles_per_row': width // TILE_W,
                }

                # Set a simple passthrough on the model's emb_l
                dlrm.emb_l[k] = nn.EmbeddingBag(ln_emb[k], EMB_DIM, mode='sum',
                                                  _weight=torch.zeros(ln_emb[k], EMB_DIM))
                dlrm.emb_l[k].mode = 'sum'
            else:
                table_kinds.append(0)  # STANDARD
                weights.append(dlrm.emb_l[k].weight)
                mappings.append(torch.empty(0, dtype=torch.int32))
                scales.append(0.0)
                zero_points.append(0)

        _C.register_tables(table_kinds, weights, mappings, scales, zero_points)

        # Register cold frames
        for t in _cold_data:
            frames = _cold_data[t]['frames']
            if not frames:
                continue
            sorted_fids = list(range(len(frames)))
            frame_data_list = []
            for f in frames:
                if f.dim() == 2:
                    # Tiled frame → untile to rows for registration
                    rows = _C.untile_frame_to_rows(f, rows_per_frame)
                    frame_data_list.append(rows)
                else:
                    frame_data_list.append(f)
            all_data = torch.cat(frame_data_list, dim=0)
            frame_ids = torch.tensor(sorted_fids, dtype=torch.long)
            _C.register_cold_frames_for_table(
                t, frame_ids, all_data,
                float(_cold_data[t]['scale']),
                float(_cold_data[t]['zp']),
                rows_per_frame,
                torch.empty(0, dtype=torch.long))

        _empty_psw = torch.empty(0)

        # Override apply_emb with fast_forward
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
            outputs = results[:num_tabs]
            cold_masks = results[num_tabs:2*num_tabs]
            cold_counts = results[2*num_tabs:]

            # Cold fixup (rare)
            for k in _cold_data:
                cc = cold_counts[k].item()
                if cc > 0:
                    cold_mask = cold_masks[k]
                    idx = lS_i_2d[k]
                    off = lS_o_2d[k]
                    cold_positions = torch.where(cold_mask)[0]
                    cold_map_vals = _mapping_tensors[k][idx[cold_positions]]
                    cold_reordered = -(cold_map_vals.long() + 1)
                    valid = cold_reordered >= 0
                    if valid.any():
                        valid_idx = cold_reordered[valid]
                        frame_ids_v = (valid_idx // rows_per_frame).long()
                        row_offsets_v = (valid_idx % rows_per_frame).long()
                        unique_fids = torch.unique(frame_ids_v).tolist()
                        cd = _cold_data[k]
                        cold_result = torch.zeros(valid.sum(), EMB_DIM)
                        for fid in unique_fids:
                            mask = frame_ids_v == fid
                            offs = row_offsets_v[mask]
                            rows = _C.gather_dequant_from_tiled_frame(
                                cd['frames'][fid], offs, cd['tiles_per_row'],
                                cd['scale'], cd['zp'])
                            cold_result[mask] = rows
                        _C.cold_fixup(outputs[k], idx, off, cold_mask,
                                      cold_result, cold_positions[valid], _empty_psw)

            dlrm.time_look_up += time.time() - start_time
            return list(outputs)

        dlrm.apply_emb = _fast_apply_emb
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

    header = f"{'Experiment':<25} {'AUC':>10} {'Mean(ms)':>10} {'P50(ms)':>10} {'P99(ms)':>10} {'Emb(ms)':>10} {'Cold(ms)':>10} {'RSS(MB)':>10}"
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
        log(f"{label:<25} {r['auc']:>10.6f} {r['mean_lat_ms']:>9.2f}{overhead:>0} "
            f"{r['p50_lat_ms']:>10.2f} {r['p99_lat_ms']:>10.2f} "
            f"{r['emb_lookup_ms']:>10.2f} {r['cold_per_batch_ms']:>10.3f} "
            f"{r['rss_mb']:>10.0f}")

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
