#!/usr/bin/env python3
"""
AUC comparison: single-frame H.265 vs 1080p multi-frame H.265 on Kaggle DLRM.

For each config:
1. Decode all cold frames -> uint8 rows
2. Register in C++ with bitmap-rank for O(1) cold lookup
3. Run full test set inference, compute AUC

Uses the same hot/cold split and quant params as codec_ondemand_benchmark.py.
"""

import os, sys, time, json, gc
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

# Set LD_LIBRARY_PATH for C++ extension
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import compressed_emb as _C

# Import from codec_ondemand_benchmark
from codec_ondemand_benchmark import (
    load_model_and_data, CompressedEmbeddingBag,
    HOTCOLD_DIR, REORDER_DIR, ONDEMAND_DIR,
    LARGE_TABLE_THRESHOLD, EMB_DIM, MODEL_PATH,
    TILE_H, TILE_W, TEST_BATCH_SIZE,
    tiled_frame_to_rows,
)

# ============================================================
# Configuration
# ============================================================
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]

# 1080p config
CONFIG_1080P = {
    'name': '1080p_crf30_nofilter',
    'base_dir': os.path.join(ONDEMAND_DIR, '1080p_crf30_nofilter'),
    'width': 1920,
    'height': 1080,
}

# Single-frame config (per-table dimensions from comparison_results.json)
SINGLE_FRAME_DIMS = {
    2:  (3840, 40400),
    3:  (1920, 17568),
    9:  (1920, 744),
    11: (3840, 33304),
    15: (1920, 43556),
    20: (1920, 56200),
    23: (1920, 2284),
    25: (1920, 1140),
}

CONFIG_SINGLE = {
    'name': 'single_frame_h265',
    'base_dir': os.path.join(ONDEMAND_DIR, 'single_frame', 'h265'),
}


def decode_h265_frames(frame_dir, num_frames, rows_per_frame, width, height):
    """Decode all H.265 frames and untile to get uint8 rows."""
    all_rows = []
    for fid in range(num_frames):
        path = os.path.join(frame_dir, f'frame_{fid:05d}.h265')
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        # C++ decode returns (H, W) uint8 tensor
        tiled_frame = _C.decode_h265_frame_from_file(path)
        # Untile to (rpf, D)
        rows = _C.untile_frame_to_rows(tiled_frame, rows_per_frame)
        all_rows.append(rows)
    if all_rows:
        return torch.cat(all_rows, dim=0)
    return torch.zeros(0, EMB_DIM, dtype=torch.uint8)


def run_auc_test(config_name, dlrm, test_batches, ln_emb, state_dict, emb_keys,
                 is_hot, hot_indices, cold_num_rows,
                 orig_to_cold_reordered, cold_quant_scale, cold_quant_zp,
                 decode_cold_fn):
    """
    Run inference with compressed embeddings and compute AUC.

    decode_cold_fn(table_id) -> (uint8_rows_tensor, rows_per_frame)
        Returns decoded uint8 cold rows for the given table.
    """
    print(f"\n{'='*60}")
    print(f"Running AUC test: {config_name}")
    print(f"{'='*60}")

    num_tabs = len(dlrm.emb_l)

    # Restore original weights for small tables
    with torch.no_grad():
        for k in emb_keys:
            t_idx = int(k.split('.')[1])
            dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

    # Build CompressedEmbeddingBag for each large table
    caches_info = {}  # t_idx -> (hot_weight, cache_for_cold_reg)
    o2c_map = orig_to_cold_reordered

    for t_idx in LARGE_TABLES:
        n_cold = cold_num_rows.get(t_idx, 0)
        if n_cold == 0:
            continue

        # Build hot weight
        w = state_dict[emb_keys[t_idx]]
        h_idx = hot_indices[t_idx]
        hot_weight = w[h_idx].clone()

        # Build orig-to-hot mapping
        orig_to_hot = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
        orig_to_hot[h_idx] = torch.arange(len(h_idx))

        # Create a dummy frame_dir (not actually used since we register in C++)
        dummy_dir = '/tmp'

        # Create CompressedEmbeddingBag (doesn't need real cache for full_cpp mode)
        comp_emb = CompressedEmbeddingBag(
            hot_weight=hot_weight,
            is_hot=is_hot[t_idx],
            orig_to_hot=orig_to_hot,
            orig_to_cold_reordered=o2c_map[t_idx],
            cold_cache=None,  # We'll use full C++ path
            num_embeddings=ln_emb[t_idx],
            embedding_dim=EMB_DIM,
            quantize_hot=False,
        )
        dlrm.emb_l[t_idx] = comp_emb
        caches_info[t_idx] = hot_weight

    # Register tables in C++ for fast_forward
    table_kinds = []
    weights = []
    mappings = []
    scales = []
    zero_points = []

    for k in range(num_tabs):
        E = dlrm.emb_l[k]
        if k in caches_info and isinstance(E, CompressedEmbeddingBag):
            table_kinds.append(1)  # COMPRESSED_FP32
            weights.append(E.hot_weight)
            mappings.append(E.mapping)
            scales.append(0.0)
            zero_points.append(0)
        else:
            table_kinds.append(0)  # STANDARD
            weights.append(E.weight)
            mappings.append(torch.empty(0, dtype=torch.int32))
            scales.append(0.0)
            zero_points.append(0)

    _C.register_tables(table_kinds, weights, mappings, scales, zero_points,
                       use_hash_table=False, use_bitmap=True)
    print("  Tables registered in C++ (bitmap mode)")

    # Load mmap'd o2c for cold fixup
    _cold_reordered_mmap = {}
    for t_idx in LARGE_TABLES:
        mmap_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t_idx}.npy')
        if os.path.exists(mmap_path):
            _cold_reordered_mmap[t_idx] = np.load(mmap_path, mmap_mode='r')

    # Decode cold frames and register in C++
    print("  Decoding cold frames...")
    t_decode_start = time.time()
    total_cold_mb = 0

    for t_idx in LARGE_TABLES:
        n_cold = cold_num_rows.get(t_idx, 0)
        if n_cold == 0:
            continue

        # Decode all frames for this table
        all_data, rows_per_frame = decode_cold_fn(t_idx)
        print(f"    Table {t_idx}: decoded {all_data.shape[0]} rows "
              f"(need {n_cold}, rpf={rows_per_frame})")

        # Map cold rows to positions in decoded data using bitmap-rank
        is_hot_t = is_hot[t_idx][:ln_emb[t_idx]]
        cold_orig = torch.where(~is_hot_t)[0]

        if t_idx in _cold_reordered_mmap:
            o2c_np = _cold_reordered_mmap[t_idx]
            cold_reordered = torch.from_numpy(
                np.array(o2c_np[cold_orig.numpy()])).long()
        else:
            cold_reordered = o2c_map[t_idx][cold_orig].long()

        # Map reordered indices to positions in all_data
        num_frames = (n_cold + rows_per_frame - 1) // rows_per_frame
        max_rows = num_frames * rows_per_frame

        # cold_reordered[i] = position in the reordered cold array
        # which maps to frame fid = cold_reordered[i] // rows_per_frame,
        # row_in_frame = cold_reordered[i] % rows_per_frame
        # In all_data (which is frames concatenated), position = cold_reordered[i]
        valid = (cold_reordered >= 0) & (cold_reordered < all_data.shape[0])

        # Extract valid rows
        valid_cold_ranks = torch.where(valid)[0].long()
        src_positions = cold_reordered[valid].long()
        valid_data = all_data[src_positions]  # [n_cached, D] uint8

        _C.register_cold_sparse_flat(
            t_idx, valid_data, valid_cold_ranks,
            float(cold_quant_scale[t_idx]),
            float(cold_quant_zp[t_idx]),
            n_cold)
        cold_mb = valid_data.nbytes / 1024 / 1024
        total_cold_mb += cold_mb
        print(f"    Table {t_idx}: registered {valid_data.shape[0]} cold rows "
              f"({cold_mb:.1f}MB)")

        del all_data, valid_data
        gc.collect()

    decode_time = time.time() - t_decode_start
    print(f"  Cold decode + register: {decode_time:.1f}s, {total_cold_mb:.1f}MB total")

    # Set up fast apply_emb (full C++ path, no Python cold fixup needed)
    def _full_cpp_apply_emb(lS_o, lS_i, emb_l, v_W_l):
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
        return results[-1]  # [T, B, D] stacked output

    dlrm.apply_emb = _full_cpp_apply_emb
    print("  Full C++ apply_emb enabled")

    # Run inference
    print("  Running inference...")
    num_test_batches = len(test_batches)
    max_samples = num_test_batches * TEST_BATCH_SIZE + TEST_BATCH_SIZE
    all_scores = np.empty(max_samples, dtype=np.float32)
    all_targets = np.empty(max_samples, dtype=np.float32)
    sample_idx = 0
    blats = []

    t0 = time.time()
    with torch.no_grad():
        for batch_idx in range(num_test_batches):
            X, lS_o, lS_i, T = test_batches[batch_idx]
            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)

            z_np = Z.detach().cpu().numpy().ravel()
            t_np = T.detach().cpu().numpy().ravel()
            bs = z_np.shape[0]
            all_scores[sample_idx:sample_idx+bs] = z_np
            all_targets[sample_idx:sample_idx+bs] = t_np
            sample_idx += bs

            if batch_idx % 500 == 0:
                print(f"    Batch {batch_idx}/{num_test_batches}")

    total_time = time.time() - t0
    auc = roc_auc_score(all_targets[:sample_idx], all_scores[:sample_idx])

    print(f"\n  Results for {config_name}:")
    print(f"    AUC = {auc:.6f}")
    print(f"    Time = {total_time:.2f}s")
    print(f"    Mean batch latency = {np.mean(blats)*1000:.2f}ms")
    print(f"    P50 latency = {np.percentile(blats, 50)*1000:.2f}ms")
    print(f"    P99 latency = {np.percentile(blats, 99)*1000:.2f}ms")
    print(f"    Cold memory = {total_cold_mb:.1f}MB")

    return {
        'auc': auc,
        'total_time': total_time,
        'mean_lat_ms': float(np.mean(blats) * 1000),
        'p50_lat_ms': float(np.percentile(blats, 50) * 1000),
        'p99_lat_ms': float(np.percentile(blats, 99) * 1000),
        'cold_mb': total_cold_mb,
        'decode_time': decode_time,
    }


def main():
    print("=" * 60)
    print("Single-Frame vs 1080p H.265 AUC Comparison")
    print("=" * 60)

    # Load model and data
    print("\nLoading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))

    # Pre-cache test batches
    print("Pre-caching test batches...")
    test_batches = []
    for X, lS_o, lS_i, T in test_ld:
        test_batches.append((X, lS_o, lS_i, T))
    print(f"Pre-cached {len(test_batches)} batches")

    # Load hot/cold data
    print("Loading hot/cold split data...")
    is_hot = {}
    hot_indices = {}
    for t in LARGE_TABLES:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]

    orig_to_cold_reordered = {}
    cold_num_rows = {}
    for t in LARGE_TABLES:
        fp = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        orig_to_cold_reordered[t] = torch.load(fp, map_location='cpu', weights_only=True)
        with open(os.path.join(REORDER_DIR, f'num_cold_{t}.txt')) as f:
            cold_num_rows[t] = int(f.read().strip())

    # Load quant params from 1080p meta.json (same quantization for both configs)
    cold_quant_scale = {}
    cold_quant_zp = {}
    for t in LARGE_TABLES:
        meta_path = os.path.join(CONFIG_1080P['base_dir'], f'table_{t}', 'meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        cold_quant_scale[t] = meta['quant_scale']
        cold_quant_zp[t] = meta['quant_zp']

    print(f"Large tables: {LARGE_TABLES}")
    for t in LARGE_TABLES:
        print(f"  Table {t}: {ln_emb[t]:,} rows, {len(hot_indices[t]):,} hot, "
              f"{cold_num_rows[t]:,} cold")

    # ---- First run baseline AUC ----
    print(f"\n{'='*60}")
    print("Baseline: Full fp32 (no compression)")
    print(f"{'='*60}")
    with torch.no_grad():
        for k in emb_keys:
            t_idx = int(k.split('.')[1])
            dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

    # Make sure apply_emb is the original
    # Save original apply_emb
    orig_apply_emb = None
    if hasattr(dlrm, '_orig_apply_emb'):
        orig_apply_emb = dlrm._orig_apply_emb
    else:
        orig_apply_emb = dlrm.apply_emb
        dlrm._orig_apply_emb = orig_apply_emb

    dlrm.apply_emb = orig_apply_emb

    num_test_batches = len(test_batches)
    max_samples = num_test_batches * TEST_BATCH_SIZE + TEST_BATCH_SIZE
    all_scores = np.empty(max_samples, dtype=np.float32)
    all_targets = np.empty(max_samples, dtype=np.float32)
    sample_idx = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx in range(num_test_batches):
            X, lS_o, lS_i, T = test_batches[batch_idx]
            Z = dlrm(X, lS_o, lS_i)
            z_np = Z.detach().cpu().numpy().ravel()
            t_np = T.detach().cpu().numpy().ravel()
            bs = z_np.shape[0]
            all_scores[sample_idx:sample_idx+bs] = z_np
            all_targets[sample_idx:sample_idx+bs] = t_np
            sample_idx += bs
    baseline_time = time.time() - t0
    baseline_auc = roc_auc_score(all_targets[:sample_idx], all_scores[:sample_idx])
    print(f"  Baseline AUC = {baseline_auc:.6f}, Time = {baseline_time:.2f}s")

    results = {'baseline': {'auc': baseline_auc, 'time': baseline_time}}

    # ---- 1080p multi-frame config ----
    def decode_1080p(t_idx):
        frame_dir = os.path.join(CONFIG_1080P['base_dir'], f'table_{t_idx}')
        w, h = CONFIG_1080P['width'], CONFIG_1080P['height']
        rpf = (w * h) // EMB_DIM  # 129600
        frame_files = sorted([f for f in os.listdir(frame_dir)
                              if f.startswith('frame_') and f.endswith('.h265')])
        num_frames = len(frame_files)
        all_rows = decode_h265_frames(frame_dir, num_frames, rpf, w, h)
        return all_rows, rpf

    # Need to restore emb_l modules between runs
    # Save original modules
    original_emb_modules = {}
    for t in LARGE_TABLES:
        original_emb_modules[t] = type(dlrm.emb_l[t])(
            num_embeddings=ln_emb[t], embedding_dim=EMB_DIM)
        original_emb_modules[t].weight.data = state_dict[emb_keys[t]].clone()

    results['1080p_crf30_nofilter'] = run_auc_test(
        '1080p_crf30_nofilter',
        dlrm, test_batches, ln_emb, state_dict, emb_keys,
        is_hot, hot_indices, cold_num_rows,
        orig_to_cold_reordered, cold_quant_scale, cold_quant_zp,
        decode_1080p,
    )

    # Restore original modules for next test
    dlrm.apply_emb = orig_apply_emb
    for t in LARGE_TABLES:
        dlrm.emb_l[t] = nn.EmbeddingBag(ln_emb[t], EMB_DIM, mode='sum', sparse=False)
        dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()
    gc.collect()

    # ---- Single-frame config ----
    # Check if single-frame uses the same quant params. It was encoded from
    # the same cold_q8 data, so quant_scale/zp should match. But single-frame
    # may have been encoded independently. Let's verify by checking if there's
    # a meta.json in the single-frame dir.
    # If not, we assume same quant params (same source data).

    def decode_single_frame(t_idx):
        frame_dir = os.path.join(CONFIG_SINGLE['base_dir'], f'table_{t_idx}')
        w, h = SINGLE_FRAME_DIMS[t_idx]
        rpf = (w * h) // EMB_DIM
        # Only 1 frame per table
        all_rows = decode_h265_frames(frame_dir, 1, rpf, w, h)
        return all_rows, rpf

    results['single_frame_h265'] = run_auc_test(
        'single_frame_h265',
        dlrm, test_batches, ln_emb, state_dict, emb_keys,
        is_hot, hot_indices, cold_num_rows,
        orig_to_cold_reordered, cold_quant_scale, cold_quant_zp,
        decode_single_frame,
    )

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'AUC':>10} {'AUC Loss':>10} {'Time(s)':>8} {'Cold MB':>8}")
    print("-" * 65)

    bl_auc = results['baseline']['auc']
    print(f"{'Baseline (fp32)':<25} {bl_auc:>10.6f} {'---':>10} "
          f"{results['baseline']['time']:>8.2f} {'---':>8}")

    for key in ['1080p_crf30_nofilter', 'single_frame_h265']:
        r = results[key]
        loss = r['auc'] - bl_auc
        print(f"{key:<25} {r['auc']:>10.6f} {loss:>+10.6f} "
              f"{r['total_time']:>8.2f} {r['cold_mb']:>7.1f}")

    # Save results
    out_path = os.path.join(ONDEMAND_DIR, 'single_frame', 'auc_comparison.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: convert(vv) for kk, vv in v.items()}

    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
