#!/usr/bin/env python3
"""
Experiment: Hot embedding compression.

Tests quantizing the hot embeddings from fp32 to uint8.
The C++ extension already supports q8 hot weights via
compressed_emb_bag_forward_q8_merged().

Configs:
1. fp32 hot (baseline, 88.5 MB hot)
2. uint8 hot per-tensor (C++ q8_merged, ~22 MB hot)
3. H.265 on hot embeddings (experimental)

Results saved to results/methodology_experiments/hot_compression.json
"""
import os, sys, time, json, gc, io, subprocess
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# ============================================================
# Config — same as warmup sweep
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
ROWS_PER_FRAME = (WIDTH // TILE_W) * (HEIGHT // TILE_H)
NUM_EVAL_BATCHES = 50

os.makedirs(RESULTS_DIR, exist_ok=True)

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

try:
    import av
except ImportError:
    pass


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# Reuse functions from warmup sweep
# ============================================================
from experiment_warmup_sweep import (
    load_model_and_data, profile_frequency, hot_cold_split,
    encode_h265_frames, load_frames_to_ram, decode_frame_from_bytes,
    run_inference, get_cache_stats,
)


# ============================================================
# Compressed Embedding Bag with q8 hot support
# ============================================================
class CompressedEmbeddingBagQ8(nn.Module):
    """Extended version that supports uint8 hot weights."""
    def __init__(self, hot_weight, mapping, compressed_frames, rows_per_frame,
                 width, height, quant_scale, quant_zp, n_cold, cache_size=20,
                 quantize_hot=False):
        super().__init__()
        self.embedding_dim = EMB_DIM
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
        self.quantize_hot = quantize_hot

        if quantize_hot:
            hot_min = hot_weight.min().item()
            hot_max = hot_weight.max().item()
            hot_scale = (hot_max - hot_min) / 255.0
            if hot_scale == 0:
                hot_scale = 1.0
            hot_zp = round(-hot_min / hot_scale)
            self.hot_weight = ((hot_weight / hot_scale).round() + hot_zp).clamp(0, 255).to(torch.uint8)
            self.hot_scale = hot_scale
            self.hot_zp = hot_zp
            self.hot_weight_fp32 = None  # don't keep fp32 copy
        else:
            self.hot_weight = hot_weight
            self.hot_scale = 0.0
            self.hot_zp = 0
            self.hot_weight_fp32 = hot_weight

        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self._empty_psw = torch.empty(0)
        self._has_q8_merged = (quantize_hot and HAS_CPP and
                                hasattr(_C, 'compressed_emb_bag_forward_q8_merged'))
        self._has_merged = (not quantize_hot and HAS_CPP and
                           hasattr(_C, 'compressed_emb_bag_forward_merged'))

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

        if self._has_q8_merged:
            output, cold_mask, cold_count = _C.compressed_emb_bag_forward_q8_merged(
                indices, offsets, self.hot_weight, self.mapping, psw,
                self.hot_scale, self.hot_zp)
        elif self._has_merged:
            output, cold_mask, cold_count = _C.compressed_emb_bag_forward_merged(
                indices, offsets, self.hot_weight, self.mapping, psw)
        else:
            # Python fallback
            B = offsets.shape[0]
            output = torch.zeros(B, EMB_DIM)
            cold_mask = torch.zeros(len(indices), dtype=torch.bool)
            cold_count = torch.tensor(0)
            for b in range(B):
                start = offsets[b].item()
                end = offsets[b+1].item() if b+1 < B else len(indices)
                for j in range(start, end):
                    idx = indices[j].item()
                    m = self.mapping[idx].item()
                    if m >= 0:
                        if self.quantize_hot:
                            row = self.hot_weight[m].float()
                            output[b] += (row - self.hot_zp) * self.hot_scale
                        else:
                            output[b] += self.hot_weight[m]
                    elif m != -2147483648:
                        cold_mask[j] = True
            cold_count = torch.tensor(cold_mask.sum().item())

        n_cold = cold_count.item()
        if n_cold > 0:
            cold_indices_reordered = -(self.mapping[indices[cold_mask]] + 1).long()
            cold_embs = self._gather_cold(cold_indices_reordered)
            bag_indices = torch.searchsorted(offsets[1:], cold_mask.nonzero(as_tuple=True)[0].to(offsets.dtype))
            output.scatter_add_(0, bag_indices.unsqueeze(1).expand(-1, EMB_DIM), cold_embs)

        return output

    def reset_cache_stats(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache = {}
        self.cache_order = []


# ============================================================
# Build compressed model with q8 hot option
# ============================================================
def build_compressed_model_q8(dlrm, ln_emb, state_dict, emb_keys,
                               large_tables, is_hot, hot_indices, cold_indices,
                               orig_to_cold_reordered, cold_quant_params,
                               compressed_frames_per_table, rpf_per_table,
                               cache_size=20, quantize_hot=False):
    with torch.no_grad():
        for k in emb_keys:
            t_idx = int(k.split('.')[1])
            if not hasattr(dlrm.emb_l[t_idx], 'weight'):
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

        if quantize_hot:
            total_hot_mb += len(h_idx) * EMB_DIM / 1024 / 1024  # uint8 size
        else:
            total_hot_mb += hot_weight.numel() * 4 / 1024 / 1024  # fp32 size

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

        comp_emb = CompressedEmbeddingBagQ8(
            hot_weight=hot_weight, mapping=mapping,
            compressed_frames=compressed_frames_per_table[t],
            rows_per_frame=rpf_per_table[t],
            width=WIDTH, height=HEIGHT,
            quant_scale=s, quant_zp=zp,
            n_cold=len(cold_indices[t]),
            cache_size=cache_size,
            quantize_hot=quantize_hot,
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


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 70)
    log("EXPERIMENT: Hot Embedding Compression")
    log("=" * 70)

    dlrm, train_ld, test_ld, ln_emb, state_dict, emb_keys = load_model_and_data()

    log("\nCaching test batches...")
    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    log(f"  {len(test_batches)} test batches cached")

    # Baseline
    log("\nRunning baseline...")
    baseline = run_inference(dlrm, test_batches, num_batches=NUM_EVAL_BATCHES)
    baseline_auc = baseline['auc']
    log(f"  Baseline AUC: {baseline_auc:.6f}")

    num_tables = len(ln_emb)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]

    # Profile (use test_ld for consistency with existing results)
    log("\nProfiling test_ld...")
    freq, first_batch, n_profiled = profile_frequency(test_batches, ln_emb, large_tables)

    # Hot/cold split
    (_, is_hot, hot_indices, cold_indices,
     orig_to_cold_reordered, cold_weights_q, cold_quant_params) = \
        hot_cold_split(freq, ln_emb, large_tables, state_dict, emb_keys)

    # Encode cold once (shared across experiments)
    log("\nEncoding cold embeddings...")
    output_dir = os.path.join(RESULTS_DIR, "compressed_hot_exp")
    compressed_frames_per_table = {}
    rpf_per_table = {}
    total_compressed_bytes = 0

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

    results = {'baseline': baseline, 'configs': {}}

    # ===== Config 1: fp32 hot (current default) =====
    log("\n" + "=" * 60)
    log("Config 1: fp32 hot (baseline)")
    log("=" * 60)
    mem = build_compressed_model_q8(
        dlrm, ln_emb, state_dict, emb_keys,
        large_tables, is_hot, hot_indices, cold_indices,
        orig_to_cold_reordered, cold_quant_params,
        compressed_frames_per_table, rpf_per_table,
        cache_size=20, quantize_hot=False)
    log(f"  Memory: {mem}")
    r1 = run_inference(dlrm, test_batches, num_batches=NUM_EVAL_BATCHES)
    cs1 = get_cache_stats(dlrm, large_tables)
    log(f"  AUC={r1['auc']:.6f} (Δ={(r1['auc']-baseline_auc)*100:+.4f}%)")
    log(f"  Latency={r1['mean_lat_ms']:.2f}ms, Cache hit={cs1['hit_rate']:.1%}")
    results['configs']['fp32_hot'] = {
        'quantize_hot': False,
        'memory': mem,
        'auc': r1['auc'], 'auc_delta_pct': (r1['auc'] - baseline_auc) * 100,
        'mean_latency_ms': r1['mean_lat_ms'],
        'cache_hit_rate': cs1['hit_rate'],
    }
    gc.collect()

    # ===== Config 2: uint8 hot (C++ q8_merged) =====
    log("\n" + "=" * 60)
    log("Config 2: uint8 hot per-tensor (C++ q8_merged)")
    log("=" * 60)
    mem = build_compressed_model_q8(
        dlrm, ln_emb, state_dict, emb_keys,
        large_tables, is_hot, hot_indices, cold_indices,
        orig_to_cold_reordered, cold_quant_params,
        compressed_frames_per_table, rpf_per_table,
        cache_size=20, quantize_hot=True)
    log(f"  Memory: {mem}")
    log(f"  Using C++ q8_merged: {HAS_CPP and hasattr(_C, 'compressed_emb_bag_forward_q8_merged')}")
    r2 = run_inference(dlrm, test_batches, num_batches=NUM_EVAL_BATCHES)
    cs2 = get_cache_stats(dlrm, large_tables)
    log(f"  AUC={r2['auc']:.6f} (Δ={(r2['auc']-baseline_auc)*100:+.4f}%)")
    log(f"  Latency={r2['mean_lat_ms']:.2f}ms, Cache hit={cs2['hit_rate']:.1%}")
    results['configs']['uint8_hot'] = {
        'quantize_hot': True,
        'memory': mem,
        'auc': r2['auc'], 'auc_delta_pct': (r2['auc'] - baseline_auc) * 100,
        'mean_latency_ms': r2['mean_lat_ms'],
        'cache_hit_rate': cs2['hit_rate'],
    }
    gc.collect()

    # ===== Config 3: H.265 on hot embeddings =====
    log("\n" + "=" * 60)
    log("Config 3: H.265 on hot embeddings (experimental)")
    log("=" * 60)

    # Encode hot rows with H.265 and decode immediately to measure error
    hot_h265_total_bytes = 0
    hot_h265_mse = {}
    for t in large_tables:
        w = state_dict[emb_keys[t]]
        h_idx = hot_indices[t]
        hot_w = w[h_idx]
        n_hot = len(h_idx)

        # Quantize hot to uint8
        mn = hot_w.min().item()
        mx = hot_w.max().item()
        s = (mx - mn) / 255.0
        if s == 0:
            s = 1.0
        zp = round(-mn / s)
        hot_q = ((hot_w / s).round() + zp).clamp(0, 255).to(torch.uint8)

        # Encode
        hot_dir = os.path.join(RESULTS_DIR, "compressed_hot_h265")
        num_frames, frame_dir, comp_bytes, rpf = encode_h265_frames(
            hot_q, WIDTH, HEIGHT, CRF, hot_dir, f"hot_{t}")
        hot_h265_total_bytes += comp_bytes

        # Decode and compute MSE
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_')])
        decoded_rows = []
        for ff in frame_files:
            with open(os.path.join(frame_dir, ff), 'rb') as fh:
                frame_data = fh.read()
            frame = decode_frame_from_bytes(frame_data)
            if frame is None:
                continue
            # Untile
            if isinstance(frame, torch.Tensor):
                frame_np = frame.numpy()
            else:
                frame_np = frame
            tiles_per_row = WIDTH // TILE_W
            tiles_per_col = HEIGHT // TILE_H
            grid = frame_np.reshape(tiles_per_col, TILE_H, tiles_per_row, TILE_W)
            rows = grid.transpose(0, 2, 1, 3).reshape(-1, EMB_DIM)
            decoded_rows.append(rows)

        decoded_all = np.concatenate(decoded_rows)[:n_hot]
        # Dequantize decoded
        decoded_fp32 = (decoded_all.astype(np.float32) - zp) * s
        # Original
        original_fp32 = hot_w.numpy()
        mse = float(np.mean((decoded_fp32 - original_fp32) ** 2))
        hot_h265_mse[t] = mse
        log(f"  Table {t}: {n_hot:,} hot rows, {num_frames} frames, "
            f"{comp_bytes/1024:.1f}KB H.265, MSE={mse:.6f}")

    hot_fp32_bytes = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables)
    hot_uint8_bytes = sum(len(hot_indices[t]) * EMB_DIM for t in large_tables)
    log(f"\n  Hot H.265 total: {hot_h265_total_bytes/1024:.1f}KB "
        f"(fp32: {hot_fp32_bytes/1024/1024:.1f}MB, uint8: {hot_uint8_bytes/1024/1024:.1f}MB)")
    log(f"  Hot compression ratio: {hot_fp32_bytes/hot_h265_total_bytes:.1f}x (fp32), "
        f"{hot_uint8_bytes/hot_h265_total_bytes:.1f}x (uint8)")

    results['hot_h265'] = {
        'total_bytes': hot_h265_total_bytes,
        'fp32_compression_ratio': hot_fp32_bytes / hot_h265_total_bytes if hot_h265_total_bytes > 0 else 0,
        'uint8_compression_ratio': hot_uint8_bytes / hot_h265_total_bytes if hot_h265_total_bytes > 0 else 0,
        'per_table_mse': {str(k): v for k, v in hot_h265_mse.items()},
    }

    # Save
    results_path = os.path.join(RESULTS_DIR, "hot_compression.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {results_path}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY: Hot Embedding Compression")
    log("=" * 70)
    log(f"{'Config':<25} {'Hot MB':>7} {'Total MB':>9} {'AUC Δ':>9} {'Latency':>9}")
    log("-" * 60)
    for name, cfg in results['configs'].items():
        log(f"{name:<25} {cfg['memory']['hot_mb']:>6.1f}  {cfg['memory']['total_mb']:>8.1f}  "
            f"{cfg['auc_delta_pct']:>+8.4f}% {cfg['mean_latency_ms']:>8.2f}ms")
    log(f"\nHot H.265: {hot_h265_total_bytes/1024:.0f}KB ({results['hot_h265']['fp32_compression_ratio']:.0f}x from fp32)")


if __name__ == '__main__':
    main()
