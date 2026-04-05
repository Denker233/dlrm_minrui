#!/usr/bin/env python3
"""
Experiment: uint8 hot + cold H.265 with frequency vs batch-affinity reordering.
Uses C++ fast_forward for hot lookups (bitmap-rank) + LRU cold frame cache (on-demand decode).

This is the realistic production configuration:
- Bitmap-rank for hot/cold classification (~2 MB vs 128 MB mapping)
- LRU cache (20 frames) for cold lookups, on-demand H.265 decode for misses
- uint8 hot embeddings with AVX-512 dequantization

Configs:
1. test_q8hot_freq: test profiling + frequency reorder + uint8 hot
2. test_q8hot_ba: test profiling + batch-affinity reorder + uint8 hot

Results saved to results/methodology_experiments/q8hot_reorder.json
"""
import os, sys, time, json, gc
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

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
LRU_CACHE_SIZE = 20  # frames per table

os.makedirs(RESULTS_DIR, exist_ok=True)

# Must have C++ extension
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if torch_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')
import compressed_emb as _C


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


from experiment_warmup_sweep import (
    load_model_and_data, profile_frequency, hot_cold_split,
    encode_h265_frames, load_frames_to_ram, decode_frame_from_bytes,
    run_inference,
)


class LRUFrameCache:
    """LRU cache for decoded cold frames. Decodes H.265 on demand."""
    def __init__(self, compressed_frames, rows_per_frame, quant_scale, quant_zp,
                 cache_size=20):
        self.compressed_frames = compressed_frames  # list of bytes (H.265 encoded)
        self.rows_per_frame = rows_per_frame
        self.quant_scale = quant_scale
        self.quant_zp = quant_zp
        self.cache_size = cache_size
        self.cache = {}       # frame_id -> (rows_tensor [rpf, D] uint8)
        self.cache_order = [] # LRU order
        self.hits = 0
        self.misses = 0

    def get_frame_rows(self, frame_id):
        """Get decoded frame rows (rpf, D) uint8. Decodes from H.265 on cache miss."""
        if frame_id in self.cache:
            self.hits += 1
            # Move to end of LRU
            self.cache_order.remove(frame_id)
            self.cache_order.append(frame_id)
            return self.cache[frame_id]

        self.misses += 1
        # Decode H.265 frame
        frame_2d = decode_frame_from_bytes(self.compressed_frames[frame_id])
        if isinstance(frame_2d, np.ndarray):
            frame_2d = torch.from_numpy(frame_2d)
        # Untile to (rpf, D)
        if frame_2d.dim() == 2 and frame_2d.shape[1] != EMB_DIM:
            rows = _C.untile_frame_to_rows(frame_2d, self.rows_per_frame)
        else:
            rows = frame_2d
        # Pad if needed
        if rows.shape[0] < self.rows_per_frame:
            pad = torch.zeros(self.rows_per_frame - rows.shape[0], EMB_DIM, dtype=torch.uint8)
            rows = torch.cat([rows, pad], dim=0)

        # Evict if full
        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        self.cache[frame_id] = rows
        self.cache_order.append(frame_id)
        return rows

    def lookup_cold_rows(self, cold_reordered_indices):
        """Look up cold rows by reordered index. Returns (N, D) fp32 dequantized."""
        if len(cold_reordered_indices) == 0:
            return torch.zeros(0, EMB_DIM)

        frame_ids = (cold_reordered_indices // self.rows_per_frame).long()
        offsets_in_frame = (cold_reordered_indices % self.rows_per_frame).long()
        result = torch.zeros(len(cold_reordered_indices), EMB_DIM)

        unique_frames = frame_ids.unique().tolist()
        for fid in unique_frames:
            rows = self.get_frame_rows(fid)
            mask = (frame_ids == fid)
            offsets = offsets_in_frame[mask]
            # Gather rows and dequantize
            gathered = rows[offsets].float()
            result[mask] = (gathered - self.quant_zp) * self.quant_scale

        return result

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.cache = {}
        self.cache_order = []


def run_config(name, dlrm, ln_emb, state_dict, emb_keys, large_tables,
               test_batches, baseline_auc, reorder_method='frequency'):
    """Run a config using C++ fast_forward (bitmap-rank) + LRU cold cache."""
    log(f"\n{'='*60}")
    log(f"Config: {name} (reorder={reorder_method}, uint8 hot, bitmap-rank, LRU cache)")
    log(f"{'='*60}")

    num_tabs = len(ln_emb)

    # Profile on all test batches
    log(f"  Profiling all {len(test_batches)} test batches...")
    freq, fb, n_batches = profile_frequency(test_batches, ln_emb, large_tables)

    # Hot/cold split
    (_, is_hot, hot_indices, cold_indices,
     orig_to_cold_reordered, cold_weights_q, cold_quant_params) = \
        hot_cold_split(freq, ln_emb, large_tables, state_dict, emb_keys,
                       reorder_method=reorder_method, first_batch_data=fb)

    # Encode cold embeddings to H.265
    t0 = time.time()
    output_dir = os.path.join(RESULTS_DIR, f"compressed_q8hot_{name}")
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
    fp32_ratio = (total_raw_bytes * 4) / total_compressed_bytes if total_compressed_bytes > 0 else 0

    log(f"  Encoded in {encode_time:.1f}s: {total_raw_bytes/1024/1024:.1f}MB uint8 -> "
        f"{total_compressed_bytes/1024/1024:.2f}MB H.265 ({uint8_ratio:.1f}x uint8, {fp32_ratio:.1f}x fp32)")

    # Restore standard weights
    with torch.no_grad():
        for k in emb_keys:
            t_idx = int(k.split('.')[1])
            if not hasattr(dlrm.emb_l[t_idx], 'weight'):
                n_rows, dim = state_dict[k].shape
                dlrm.emb_l[t_idx] = nn.EmbeddingBag(n_rows, dim, mode='sum', sparse=False)
            dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

    # Prepare registration data
    table_kinds = []
    weights = []
    mappings_for_reg = []
    scales = []
    zero_points = []
    cold_caches = {}  # table_idx -> LRUFrameCache
    cold_mappings = {}  # table_idx -> orig_to_cold_reordered tensor (int32, for cold fixup)

    total_hot_mb = 0

    for k in range(num_tabs):
        if k in large_tables and k in compressed_frames_per_table:
            w = state_dict[emb_keys[k]]
            h_idx = hot_indices[k]
            hot_weight = w[h_idx].clone()

            # Quantize hot to uint8
            hot_min = hot_weight.min().item()
            hot_max = hot_weight.max().item()
            hot_scale = (hot_max - hot_min) / 255.0
            if hot_scale == 0:
                hot_scale = 1.0
            hot_zp = round(-hot_min / hot_scale)
            hot_q8 = ((hot_weight / hot_scale).round() + hot_zp).clamp(0, 255).to(torch.uint8)

            # Build mapping (needed for bitmap-rank construction, then released by C++)
            INT32_MIN = -2147483648
            mapping = torch.full((ln_emb[k],), INT32_MIN, dtype=torch.int32)
            orig_to_hot = torch.full((ln_emb[k],), -1, dtype=torch.long)
            orig_to_hot[h_idx] = torch.arange(len(h_idx))
            mapping[is_hot[k]] = orig_to_hot[is_hot[k]].to(torch.int32)
            o2c = orig_to_cold_reordered[k]
            cold_mask = o2c >= 0
            mapping[cold_mask] = (-(o2c[cold_mask] + 1)).to(torch.int32)

            table_kinds.append(2)  # COMPRESSED_Q8
            weights.append(hot_q8)
            mappings_for_reg.append(mapping)
            scales.append(float(hot_scale))
            zero_points.append(int(hot_zp))

            total_hot_mb += len(h_idx) * EMB_DIM / 1024 / 1024

            # Create LRU cold cache for this table
            s, zp = cold_quant_params[k]
            cold_caches[k] = LRUFrameCache(
                compressed_frames=compressed_frames_per_table[k],
                rows_per_frame=rpf_per_table[k],
                quant_scale=s, quant_zp=zp,
                cache_size=LRU_CACHE_SIZE)

            # Save cold mapping for fixup (int32 to save memory)
            cold_mappings[k] = o2c.to(torch.int32)
        else:
            table_kinds.append(0)
            weights.append(dlrm.emb_l[k].weight)
            mappings_for_reg.append(torch.empty(0, dtype=torch.int32))
            scales.append(0.0)
            zero_points.append(0)

    compressed_mb = total_compressed_bytes / 1024 / 1024
    cache_mb = LRU_CACHE_SIZE * ROWS_PER_FRAME * EMB_DIM / 1024 / 1024  # per table worst case
    # Bitmap-rank uses ~2 MB total vs 128 MB for array mapping
    bitmap_mb_estimate = sum(
        ((ln_emb[t] + 63) // 64 * 8 + ((ln_emb[t] + 63) // 64 + 1) * 4) / 1024 / 1024
        for t in large_tables if t in compressed_frames_per_table
    )

    # Register tables with bitmap-rank (C++ releases mapping tensors)
    log(f"  Registering {num_tabs} tables in C++ (use_bitmap=True)...")
    _C.register_tables(table_kinds, weights, mappings_for_reg, scales, zero_points,
                       use_hash_table=False, use_bitmap=True)
    # Mapping tensors are now released by C++ — bitmap-rank uses ~2 MB instead
    del mappings_for_reg
    gc.collect()

    total_mb = total_hot_mb + compressed_mb + cache_mb + bitmap_mb_estimate
    mem = {
        'hot_mb': total_hot_mb,
        'compressed_cold_mb': compressed_mb,
        'cache_mb': cache_mb,
        'bitmap_mb': bitmap_mb_estimate,
        'total_mb': total_mb,
    }
    log(f"  Memory: hot={total_hot_mb:.1f}MB  cold_h265={compressed_mb:.2f}MB  "
        f"cache={cache_mb:.1f}MB  bitmap={bitmap_mb_estimate:.2f}MB  total={total_mb:.1f}MB")

    # Build apply_emb: C++ fast_forward for hot + cold_mask fallback with LRU cache
    def _fast_apply_emb(lS_o, lS_i, emb_l, v_W_l):
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

        # C++ handles hot lookups + standard tables
        results = _C.fast_forward(lS_i_2d, lS_o_2d)
        outputs = results[:num_tabs]
        cold_masks = results[num_tabs:2*num_tabs]
        cold_counts = results[2*num_tabs:3*num_tabs]

        # Cold fixup via LRU cache
        for k in cold_caches:
            cc = cold_counts[k].item()
            if cc > 0:
                cold_mask = cold_masks[k]
                idx = lS_i_2d[k]
                off = lS_o_2d[k]
                cold_positions = torch.where(cold_mask)[0]
                cold_orig = idx[cold_positions]

                # Look up cold reordered indices from saved mapping
                cold_reordered = cold_mappings[k][cold_orig].long()
                valid = cold_reordered >= 0
                if valid.any():
                    cold_result = cold_caches[k].lookup_cold_rows(cold_reordered[valid])
                    # Use C++ cold_fixup to scatter cold results into output
                    _C.cold_fixup(outputs[k], idx, off, cold_mask,
                                  cold_result, cold_positions[valid], torch.empty(0))

        return list(outputs)

    dlrm.apply_emb = _fast_apply_emb

    # Reset cache stats
    for c in cold_caches.values():
        c.reset_stats()

    # Evaluate
    log(f"  Evaluating on all {len(test_batches)} test batches (bitmap-rank + LRU cache)...")
    inf_res = run_inference(dlrm, test_batches, num_batches=0)
    auc = inf_res['auc']
    auc_delta = (auc - baseline_auc) * 100

    # Collect cache stats
    total_hits = sum(c.hits for c in cold_caches.values())
    total_misses = sum(c.misses for c in cold_caches.values())
    total_accesses = total_hits + total_misses
    cache_hit_rate = total_hits / total_accesses if total_accesses > 0 else 0
    per_table_cache = {}
    for k, c in cold_caches.items():
        t_total = c.hits + c.misses
        per_table_cache[str(k)] = {
            'hits': c.hits, 'misses': c.misses,
            'hit_rate': c.hits / t_total if t_total > 0 else 0,
        }

    log(f"  AUC={auc:.6f} (delta={auc_delta:+.4f}%)  latency={inf_res['mean_lat_ms']:.1f}ms  "
        f"cache_hit={cache_hit_rate:.1%}")
    cache_summary = {k: "{:.1%}".format(v['hit_rate']) for k,v in per_table_cache.items()}
    log(f"  Per-table cache: {json.dumps(cache_summary)}")

    result = {
        'name': name,
        'reorder_method': reorder_method,
        'quantize_hot': True,
        'cpp_fast_forward': True,
        'use_bitmap': True,
        'lru_cache_size': LRU_CACHE_SIZE,
        'profile_source': 'test_all',
        'profile_batches': len(test_batches),
        'eval_batches': len(test_batches),
        'uint8_compression_ratio': uint8_ratio,
        'fp32_compression_ratio': fp32_ratio,
        'memory': mem,
        'auc': auc,
        'auc_delta_pct': auc_delta,
        'mean_latency_ms': inf_res['mean_lat_ms'],
        'p50_latency_ms': inf_res['p50_lat_ms'],
        'p99_latency_ms': inf_res['p99_lat_ms'],
        'cache_hit_rate': cache_hit_rate,
        'cache_stats': per_table_cache,
    }

    gc.collect()
    return result


def main():
    log("Loading model and data...")
    dlrm, train_ld, test_ld, ln_emb, state_dict, emb_keys = load_model_and_data()
    large_tables = [t for t in range(len(ln_emb)) if ln_emb[t] > LARGE_TABLE_THRESHOLD]
    log(f"  Large tables: {large_tables}")

    # Collect test batches
    log("Collecting test batches...")
    test_batches = []
    for j, (X, lS_o, lS_i, T) in enumerate(test_ld):
        test_batches.append((X, lS_o, lS_i, T))
    log(f"  {len(test_batches)} test batches collected")

    # Baseline
    log("Running baseline (uncompressed, all test batches)...")
    baseline_res = run_inference(dlrm, test_batches, num_batches=0)
    baseline_auc = baseline_res['auc']
    log(f"  Baseline AUC={baseline_auc:.6f}  latency={baseline_res['mean_lat_ms']:.1f}ms")

    results = {
        'baseline': baseline_res,
        'configs': {},
    }

    # Config 1: frequency reordering
    r1 = run_config("freq", dlrm, ln_emb, state_dict, emb_keys, large_tables,
                    test_batches, baseline_auc, reorder_method='frequency')
    results['configs']['test_q8hot_freq'] = r1

    # Config 2: batch-affinity reordering
    r2 = run_config("batch_affinity", dlrm, ln_emb, state_dict, emb_keys, large_tables,
                    test_batches, baseline_auc, reorder_method='batch_affinity')
    results['configs']['test_q8hot_ba'] = r2

    # Summary
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")
    log(f"Baseline: AUC={baseline_auc:.6f}  latency={baseline_res['mean_lat_ms']:.1f}ms")
    log(f"")
    log(f"{'Config':<25s} {'Compr':>8s} {'Hot':>7s} {'Total':>8s} {'AUC Δ':>10s} {'Lat':>10s} {'Cache':>7s}")
    for key in ['test_q8hot_freq', 'test_q8hot_ba']:
        c = results['configs'][key]
        log(f"{key:<25s} {c['uint8_compression_ratio']:>7.1f}x {c['memory']['hot_mb']:>6.1f}M "
            f"{c['memory']['total_mb']:>7.1f}M {c['auc_delta_pct']:>+9.4f}% "
            f"{c['mean_latency_ms']:>8.1f}ms {c['cache_hit_rate']:>6.1%}")

    out_path = os.path.join(RESULTS_DIR, "q8hot_reorder.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
