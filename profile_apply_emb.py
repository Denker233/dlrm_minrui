#!/usr/bin/env python3
"""
Profile the apply_emb function to find where the ~3.5ms overhead comes from.
Runs 100 batches, measures time in:
1. Compressed table C++ calls (8 tables)
2. Standard F.embedding_bag calls (18 tables)
3. Python overhead (loop, attribute access, isinstance, list.append)
"""

import os, sys, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the C++ extension
try:
    import compressed_emb as _C
    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False
    print("ERROR: C++ extension not available")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from codec_ondemand_benchmark import (
    load_model_and_data, CompressedEmbeddingBag, OnDemandPrefetchCache,
    GlobalFrameCache, log, EMB_DIM, LARGE_TABLE_THRESHOLD,
    RESOLUTIONS, REORDER_DIR, HOTCOLD_DIR, ONDEMAND_DIR, RESULTS_DIR
)

MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"

def main():
    log("Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]

    # Load hot/cold data
    is_hot = {}
    hot_indices = {}
    orig_to_cold_reordered = {}
    cold_num_rows = {}
    cold_quant_scale = {}
    cold_quant_zp = {}

    for t in large_tables:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]
        fp = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        orig_to_cold_reordered[t] = torch.load(fp, map_location='cpu', weights_only=True)
        n_cold_path = os.path.join(REORDER_DIR, f'num_cold_{t}.txt')
        with open(n_cold_path) as f:
            cold_num_rows[t] = int(f.read().strip())
        import json as _json
        meta_path = os.path.join(ONDEMAND_DIR, '4K', f'table_{t}', 'meta.json')
        with open(meta_path) as f:
            meta = _json.load(f)
        cold_quant_scale[t] = meta['quant_scale']
        cold_quant_zp[t] = meta['quant_zp']

    # Setup 4K, cache=8, fp32hot
    res_name = '4K'
    width, height = RESOLUTIONS[res_name]
    rows_per_frame = (width * height) // EMB_DIM

    global_cache = GlobalFrameCache(capacity=8, store_uint8=True)
    caches = {}
    for t_idx in large_tables:
        n_cold = cold_num_rows.get(t_idx, 0)
        if n_cold == 0:
            continue
        frame_dir = os.path.join(ONDEMAND_DIR, res_name, f'table_{t_idx}')
        if not os.path.exists(frame_dir):
            continue
        cache = OnDemandPrefetchCache(
            frame_dir=frame_dir, rows_per_frame=rows_per_frame,
            emb_dim=EMB_DIM, num_cold_rows=n_cold,
            width=width, height=height,
            quant_scale=cold_quant_scale[t_idx],
            quant_zp=cold_quant_zp[t_idx],
            cache_capacity=8, predictor=None,
            num_prefetch_workers=2,
            global_cache=global_cache, table_id=t_idx)
        caches[t_idx] = cache

        w = state_dict[emb_keys[t_idx]]
        h_idx = hot_indices[t_idx]
        hot_weight = w[h_idx].clone()
        orig_to_hot = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
        orig_to_hot[h_idx] = torch.arange(len(h_idx))

        comp_emb = CompressedEmbeddingBag(
            hot_weight=hot_weight, is_hot=is_hot[t_idx],
            orig_to_hot=orig_to_hot,
            orig_to_cold_reordered=orig_to_cold_reordered[t_idx],
            cold_cache=cache, num_embeddings=ln_emb[t_idx],
            embedding_dim=EMB_DIM, quantize_hot=False)
        dlrm.emb_l[t_idx] = comp_emb

    compressed_table_ids = set(caches.keys())
    log(f"Compressed tables: {sorted(compressed_table_ids)}")
    log(f"C++ extension: {HAS_CPP_EXT}")

    # Profile 100 batches with detailed timing
    N_PROFILE = 200
    warmup = 20

    times_compressed = []
    times_standard = []
    times_cold_check = []
    times_total = []
    times_python = []

    gc.collect()
    log(f"\nProfiling {N_PROFILE} batches (warmup={warmup})...")

    with torch.no_grad():
        for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
            if batch_idx >= N_PROFILE:
                break

            t_total_start = time.perf_counter()
            t_compressed = 0.0
            t_standard = 0.0
            t_cold_check = 0.0
            ly = []

            for k in range(len(dlrm.emb_l)):
                E = dlrm.emb_l[k]
                idx = lS_i[k]
                off = lS_o[k]

                if k in compressed_table_ids and isinstance(E, CompressedEmbeddingBag):
                    t0 = time.perf_counter()
                    output, cold_mask, cold_count = _C.compressed_emb_bag_forward_merged(
                        idx, off, E.hot_weight, E.mapping, E._empty_psw)
                    t1 = time.perf_counter()
                    t_compressed += (t1 - t0)

                    t0 = time.perf_counter()
                    cc = cold_count.item()
                    t1 = time.perf_counter()
                    t_cold_check += (t1 - t0)

                    if cc > 0:
                        cold_positions = torch.where(cold_mask)[0]
                        cold_orig = idx[cold_positions]
                        cold_map_vals = E.mapping[cold_orig]
                        cold_reordered = -(cold_map_vals.long() + 1)
                        valid = cold_reordered >= 0
                        if valid.any():
                            cold_result, frames_used = E.cold_cache.lookup(cold_reordered[valid])
                            _C.cold_fixup(output, idx, off, cold_mask,
                                          cold_result, cold_positions[valid], E._empty_psw)
                            E.last_frames_used = frames_used
                        else:
                            E.last_frames_used = set()
                    else:
                        E.last_frames_used = set()
                    ly.append(output)
                else:
                    t0 = time.perf_counter()
                    psw = None  # no per_sample_weights in baseline
                    ly.append(F.embedding_bag(idx, E.weight, off,
                                              per_sample_weights=psw, mode='sum'))
                    t1 = time.perf_counter()
                    t_standard += (t1 - t0)

            t_total = time.perf_counter() - t_total_start
            t_python_overhead = t_total - t_compressed - t_standard - t_cold_check

            if batch_idx >= warmup:
                times_compressed.append(t_compressed * 1000)
                times_standard.append(t_standard * 1000)
                times_cold_check.append(t_cold_check * 1000)
                times_total.append(t_total * 1000)
                times_python.append(t_python_overhead * 1000)

    log(f"\nResults ({N_PROFILE - warmup} batches, excluding warmup):")
    log(f"  Total apply_emb:  mean={np.mean(times_total):.3f}ms  p50={np.percentile(times_total,50):.3f}ms  p99={np.percentile(times_total,99):.3f}ms")
    log(f"  Compressed (8 tabs):  mean={np.mean(times_compressed):.3f}ms  p50={np.percentile(times_compressed,50):.3f}ms")
    log(f"  Standard (18 tabs):   mean={np.mean(times_standard):.3f}ms  p50={np.percentile(times_standard,50):.3f}ms")
    log(f"  Cold checks (.item):  mean={np.mean(times_cold_check):.3f}ms  p50={np.percentile(times_cold_check,50):.3f}ms")
    log(f"  Python overhead:      mean={np.mean(times_python):.3f}ms  p50={np.percentile(times_python,50):.3f}ms")
    log(f"  ")
    log(f"  Per compressed table: mean={np.mean(times_compressed)/8:.3f}ms")
    log(f"  Per standard table:   mean={np.mean(times_standard)/18:.3f}ms")

    # Also profile baseline apply_emb (standard EmbeddingBag.__call__)
    log(f"\nProfiling baseline apply_emb (nn.Module.__call__)...")
    # Restore original EmbeddingBag for large tables
    for t_idx in large_tables:
        dlrm.emb_l[t_idx] = nn.EmbeddingBag(ln_emb[t_idx], EMB_DIM, mode='sum')
        dlrm.emb_l[t_idx].weight = nn.Parameter(
            state_dict[emb_keys[t_idx]].clone(), requires_grad=False)

    times_baseline = []
    times_baseline_emb_call = []
    with torch.no_grad():
        for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
            if batch_idx >= N_PROFILE:
                break
            t0 = time.perf_counter()
            t_emb = 0.0
            ly = []
            for k in range(len(dlrm.emb_l)):
                E = dlrm.emb_l[k]
                idx = lS_i[k]
                off = lS_o[k]
                t1 = time.perf_counter()
                V = E(idx, off)
                t2 = time.perf_counter()
                t_emb += (t2 - t1)
                ly.append(V)
            t_total = time.perf_counter() - t0
            if batch_idx >= warmup:
                times_baseline.append(t_total * 1000)
                times_baseline_emb_call.append(t_emb * 1000)

    log(f"\nBaseline apply_emb ({N_PROFILE - warmup} batches):")
    log(f"  Total:     mean={np.mean(times_baseline):.3f}ms  p50={np.percentile(times_baseline,50):.3f}ms")
    log(f"  EmbBag calls: mean={np.mean(times_baseline_emb_call):.3f}ms  p50={np.percentile(times_baseline_emb_call,50):.3f}ms")
    log(f"  Per table:    mean={np.mean(times_baseline_emb_call)/26:.3f}ms")

    # Cleanup
    for c in caches.values():
        c.close()

if __name__ == '__main__':
    main()
