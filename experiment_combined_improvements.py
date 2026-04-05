#!/usr/bin/env python3
"""
Experiment: Combined improvements.

Compares configurations that combine the best findings from:
1. Warmup-based profiling (disjoint eval)
2. Train_ld profiling for hot/cold split
3. Batch-affinity reordering
4. Hot embedding compression (uint8)

Runs with LARGER eval set (all remaining batches) for more reliable AUC.

Results saved to results/methodology_experiments/combined_improvements.json
"""
import os, sys, time, json, gc
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
ROWS_PER_FRAME = (WIDTH // TILE_W) * (HEIGHT // TILE_H)

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

import subprocess
try:
    import av
except ImportError:
    pass


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


from experiment_warmup_sweep import (
    load_model_and_data, profile_frequency, hot_cold_split,
    encode_h265_frames, load_frames_to_ram, decode_frame_from_bytes,
    run_inference, get_cache_stats, SimpleCompressedEmbeddingBag,
    reset_cache_stats,
)


# ============================================================
# Compressed Embedding Bag with q8 hot support
# ============================================================
class CompressedEmbeddingBagQ8(nn.Module):
    """Extended version supporting uint8 hot weights."""
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
        else:
            self.hot_weight = hot_weight
            self.hot_scale = 0.0
            self.hot_zp = 0

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


def build_model(dlrm, ln_emb, state_dict, emb_keys, large_tables,
                is_hot, hot_indices, cold_indices,
                orig_to_cold_reordered, cold_quant_params,
                compressed_frames_per_table, rpf_per_table,
                cache_size=20, quantize_hot=False):
    """Build compressed model, optionally with uint8 hot."""
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
            total_hot_mb += len(h_idx) * EMB_DIM / 1024 / 1024
        else:
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

        if quantize_hot:
            comp_emb = CompressedEmbeddingBagQ8(
                hot_weight=hot_weight, mapping=mapping,
                compressed_frames=compressed_frames_per_table[t],
                rows_per_frame=rpf_per_table[t],
                width=WIDTH, height=HEIGHT,
                quant_scale=s, quant_zp=zp,
                n_cold=len(cold_indices[t]),
                cache_size=cache_size,
                quantize_hot=True)
        else:
            comp_emb = SimpleCompressedEmbeddingBag(
                hot_weight=hot_weight, mapping=mapping,
                compressed_frames=compressed_frames_per_table[t],
                rows_per_frame=rpf_per_table[t],
                width=WIDTH, height=HEIGHT,
                quant_scale=s, quant_zp=zp,
                n_cold=len(cold_indices[t]),
                cache_size=cache_size)
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


def get_all_cache_stats(dlrm, large_tables):
    """Get cache stats from both SimpleCompressedEmbeddingBag and CompressedEmbeddingBagQ8."""
    total_hits = 0
    total_misses = 0
    per_table = {}
    for t in large_tables:
        e = dlrm.emb_l[t]
        if hasattr(e, 'cache_hits'):
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


def reset_all_cache_stats(dlrm, large_tables):
    for t in large_tables:
        if hasattr(dlrm.emb_l[t], 'reset_cache_stats'):
            dlrm.emb_l[t].reset_cache_stats()


def run_config_combined(name, dlrm, ln_emb, state_dict, emb_keys,
                        large_tables, profile_source, eval_batches,
                        baseline_auc, train_ld=None, test_batches=None,
                        warmup_fraction=0.10, profile_batches_limit=0,
                        reorder_method='frequency', quantize_hot=False,
                        use_hybrid=False):
    """Run a single combined config. Returns result dict."""
    log(f"\n{'='*60}")
    log(f"Config: {name}")
    log(f"{'='*60}")

    desc_parts = []

    if use_hybrid:
        # Hybrid: train for split, warmup for ordering
        log(f"  Profiling train_ld ({profile_batches_limit} batches) for hot/cold split...")
        train_freq, _, _ = profile_frequency(train_ld, ln_emb, large_tables,
                                              max_batches=profile_batches_limit)
        n_warmup = max(1, int(len(test_batches) * warmup_fraction))
        warmup_batches = test_batches[:n_warmup]
        log(f"  Profiling warmup ({n_warmup} batches) for cold ordering...")
        warmup_freq, warmup_fb, _ = profile_frequency(warmup_batches, ln_emb, large_tables)

        is_hot = {}
        hot_indices = {}
        cold_indices = {}
        orig_to_cold_reordered = {}
        cold_weights_q = {}
        cold_quant_params = {}

        for t in large_tables:
            n = ln_emb[t]
            n_hot = max(1, int(n * HOT_FRACTION))
            sorted_idx = train_freq[t].argsort(descending=True)
            hot_idx = sorted_idx[:n_hot]
            cold_idx_set = sorted_idx[n_hot:]

            is_hot_t = torch.zeros(n, dtype=torch.bool)
            is_hot_t[hot_idx] = True
            is_hot[t] = is_hot_t
            hot_indices[t] = hot_idx

            if reorder_method == 'batch_affinity' and warmup_fb is not None:
                cold_warmup_freq = warmup_freq[t][cold_idx_set]
                cold_fb = warmup_fb[t][cold_idx_set]
                sort_key = cold_fb.float() * 1e12 - cold_warmup_freq.float()
                cold_order = sort_key.argsort()
            else:
                cold_warmup_freq = warmup_freq[t][cold_idx_set]
                cold_order = cold_warmup_freq.argsort(descending=True)
            cold_idx = cold_idx_set[cold_order]
            cold_indices[t] = cold_idx

            o2c = torch.full((n,), -1, dtype=torch.long)
            o2c[cold_idx] = torch.arange(len(cold_idx))
            orig_to_cold_reordered[t] = o2c

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

        desc_parts.append(f"hybrid: train {profile_batches_limit} batches + warmup {warmup_fraction*100:.0f}%")
    elif profile_source == 'warmup':
        n_warmup = max(1, int(len(test_batches) * warmup_fraction))
        profile_data = test_batches[:n_warmup]
        log(f"  Profiling warmup ({n_warmup} batches)...")
        freq, fb, _ = profile_frequency(profile_data, ln_emb, large_tables)
        (_, is_hot, hot_indices, cold_indices,
         orig_to_cold_reordered, cold_weights_q, cold_quant_params) = \
            hot_cold_split(freq, ln_emb, large_tables, state_dict, emb_keys,
                           reorder_method=reorder_method, first_batch_data=fb)
        desc_parts.append(f"warmup {warmup_fraction*100:.0f}%")
    elif profile_source == 'test_circular':
        log(f"  Profiling all test batches (circular)...")
        freq, fb, _ = profile_frequency(test_batches, ln_emb, large_tables)
        (_, is_hot, hot_indices, cold_indices,
         orig_to_cold_reordered, cold_weights_q, cold_quant_params) = \
            hot_cold_split(freq, ln_emb, large_tables, state_dict, emb_keys,
                           reorder_method=reorder_method, first_batch_data=fb)
        desc_parts.append("test 100% (circular)")
    elif profile_source == 'train':
        log(f"  Profiling train_ld ({profile_batches_limit} batches)...")
        freq, fb, _ = profile_frequency(train_ld, ln_emb, large_tables,
                                          max_batches=profile_batches_limit)
        (_, is_hot, hot_indices, cold_indices,
         orig_to_cold_reordered, cold_weights_q, cold_quant_params) = \
            hot_cold_split(freq, ln_emb, large_tables, state_dict, emb_keys,
                           reorder_method=reorder_method, first_batch_data=fb)
        desc_parts.append(f"train {profile_batches_limit} batches")

    desc_parts.append(f"reorder={reorder_method}")
    desc_parts.append(f"hot={'uint8' if quantize_hot else 'fp32'}")

    # Encode cold
    t0 = time.time()
    output_dir = os.path.join(RESULTS_DIR, f"compressed_combined_{name}")
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

    # Build
    mem = build_model(
        dlrm, ln_emb, state_dict, emb_keys, large_tables,
        is_hot, hot_indices, cold_indices,
        orig_to_cold_reordered, cold_quant_params,
        compressed_frames_per_table, rpf_per_table,
        cache_size=20, quantize_hot=quantize_hot)

    log(f"  Memory: hot={mem['hot_mb']:.1f}MB, cold={mem['compressed_mb']:.2f}MB, "
        f"cache={mem['cache_mb']:.1f}MB, mapping={mem['mapping_mb']:.1f}MB, "
        f"total={mem['total_mb']:.1f}MB")

    # Evaluate — use ALL eval batches for better AUC estimate
    result = run_inference(dlrm, eval_batches, num_batches=0)
    cache_stats = get_all_cache_stats(dlrm, large_tables)
    auc_delta = (result['auc'] - baseline_auc) * 100

    log(f"  AUC={result['auc']:.6f} (delta={auc_delta:+.4f}%)")
    log(f"  Latency={result['mean_lat_ms']:.2f}ms, Cache hit={cache_stats['hit_rate']:.1%}")
    log(f"  Eval batches: {result['n_batches']}")

    return {
        'name': name,
        'description': ', '.join(desc_parts),
        'uint8_compression_ratio': uint8_ratio,
        'fp32_compression_ratio': fp32_ratio,
        'compressed_cold_mb': total_compressed_bytes / 1024 / 1024,
        'memory': mem,
        'auc': result['auc'],
        'auc_delta_pct': auc_delta,
        'mean_latency_ms': result['mean_lat_ms'],
        'p50_latency_ms': result['p50_lat_ms'],
        'p99_latency_ms': result['p99_lat_ms'],
        'n_eval_batches': result['n_batches'],
        'cache_hit_rate': cache_stats['hit_rate'],
        'cache_stats': {str(k): v for k, v in cache_stats['per_table'].items()},
        'quantize_hot': quantize_hot,
        'encode_time_s': encode_time,
    }


def main():
    log("=" * 70)
    log("EXPERIMENT: Combined Improvements")
    log("=" * 70)

    dlrm, train_ld, test_ld, ln_emb, state_dict, emb_keys = load_model_and_data()

    # Cache test batches
    log("\nCaching test batches...")
    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    log(f"  {len(test_batches)} test batches cached")

    num_tables = len(ln_emb)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]

    # Split: use 10% warmup, rest for eval (for all warmup configs)
    warmup_frac = 0.10
    n_warmup = max(1, int(len(test_batches) * warmup_frac))
    eval_batches_disjoint = test_batches[n_warmup:]

    # Baseline on eval portion (disjoint)
    log("\nRunning baseline on eval portion (disjoint)...")
    baseline_disjoint = run_inference(dlrm, eval_batches_disjoint, num_batches=0)
    log(f"  Baseline AUC (disjoint, {baseline_disjoint['n_batches']} batches): {baseline_disjoint['auc']:.6f}")

    # Also compute baseline on all test batches (for circular comparison)
    log("Running baseline on all test batches...")
    baseline_all = run_inference(dlrm, test_batches, num_batches=0)
    log(f"  Baseline AUC (all, {baseline_all['n_batches']} batches): {baseline_all['auc']:.6f}")

    results = {
        'baseline_disjoint': baseline_disjoint,
        'baseline_all': baseline_all,
        'configs': {},
    }

    configs = [
        # Config 1: Original circular (test 100% profile + eval, fp32 hot, frequency)
        {
            'name': 'original_circular',
            'profile_source': 'test_circular',
            'eval_batches': test_batches,
            'baseline_auc': baseline_all['auc'],
            'reorder_method': 'frequency',
            'quantize_hot': False,
        },
        # Config 2: Warmup 10% disjoint (fp32 hot, frequency)
        {
            'name': 'warmup_10pct_freq',
            'profile_source': 'warmup',
            'eval_batches': eval_batches_disjoint,
            'baseline_auc': baseline_disjoint['auc'],
            'warmup_fraction': warmup_frac,
            'reorder_method': 'frequency',
            'quantize_hot': False,
        },
        # Config 3: Warmup 10% disjoint + batch_affinity
        {
            'name': 'warmup_10pct_ba',
            'profile_source': 'warmup',
            'eval_batches': eval_batches_disjoint,
            'baseline_auc': baseline_disjoint['auc'],
            'warmup_fraction': warmup_frac,
            'reorder_method': 'batch_affinity',
            'quantize_hot': False,
        },
        # Config 4: Warmup 10% disjoint + uint8 hot
        {
            'name': 'warmup_10pct_q8hot',
            'profile_source': 'warmup',
            'eval_batches': eval_batches_disjoint,
            'baseline_auc': baseline_disjoint['auc'],
            'warmup_fraction': warmup_frac,
            'reorder_method': 'frequency',
            'quantize_hot': True,
        },
        # Config 5: Hybrid (train 5K + warmup 10%), fp32 hot
        {
            'name': 'hybrid_5k_freq',
            'use_hybrid': True,
            'eval_batches': eval_batches_disjoint,
            'baseline_auc': baseline_disjoint['auc'],
            'profile_batches_limit': 5000,
            'warmup_fraction': warmup_frac,
            'reorder_method': 'frequency',
            'quantize_hot': False,
        },
        # Config 6: Hybrid + batch_affinity + uint8 hot (best combined)
        {
            'name': 'hybrid_5k_ba_q8hot',
            'use_hybrid': True,
            'eval_batches': eval_batches_disjoint,
            'baseline_auc': baseline_disjoint['auc'],
            'profile_batches_limit': 5000,
            'warmup_fraction': warmup_frac,
            'reorder_method': 'batch_affinity',
            'quantize_hot': True,
        },
        # Config 7: Train-only 5K batches (no warmup, fp32 hot)
        {
            'name': 'train_5k_freq',
            'profile_source': 'train',
            'eval_batches': eval_batches_disjoint,
            'baseline_auc': baseline_disjoint['auc'],
            'profile_batches_limit': 5000,
            'reorder_method': 'frequency',
            'quantize_hot': False,
        },
    ]

    for cfg in configs:
        name = cfg.pop('name')
        eval_b = cfg.pop('eval_batches')
        base_auc = cfg.pop('baseline_auc')

        r = run_config_combined(
            name, dlrm, ln_emb, state_dict, emb_keys,
            large_tables,
            profile_source=cfg.get('profile_source', 'warmup'),
            eval_batches=eval_b,
            baseline_auc=base_auc,
            train_ld=train_ld,
            test_batches=test_batches,
            warmup_fraction=cfg.get('warmup_fraction', 0.10),
            profile_batches_limit=cfg.get('profile_batches_limit', 0),
            reorder_method=cfg.get('reorder_method', 'frequency'),
            quantize_hot=cfg.get('quantize_hot', False),
            use_hybrid=cfg.get('use_hybrid', False),
        )
        results['configs'][name] = r
        gc.collect()

    # Save results
    results_path = os.path.join(RESULTS_DIR, "combined_improvements.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {results_path}")

    # Summary table
    log("\n" + "=" * 80)
    log("COMBINED IMPROVEMENTS SUMMARY")
    log("=" * 80)
    log(f"{'Config':<30} {'Compr':>6} {'Hot':>6} {'Total':>7} {'AUC Δ':>9} {'Lat':>8} {'Cache':>6} {'#Eval':>5}")
    log("-" * 85)
    for name, cfg in results['configs'].items():
        hot_label = 'q8' if cfg.get('quantize_hot') else 'fp32'
        log(f"{name:<30} {cfg['uint8_compression_ratio']:>5.1f}x "
            f"{cfg['memory']['hot_mb']:>5.1f}M "
            f"{cfg['memory']['total_mb']:>6.1f}M "
            f"{cfg['auc_delta_pct']:>+8.4f}% "
            f"{cfg['mean_latency_ms']:>7.1f}ms "
            f"{cfg['cache_hit_rate']:>5.1%} "
            f"{cfg['n_eval_batches']:>5d}")


if __name__ == '__main__':
    main()
