#!/usr/bin/env python3
"""Profile per-batch latency breakdown for CompressedEmbeddingBag vs baseline."""
import os, sys, time, json, gc
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import compressed_emb as _C
    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False

# Configuration
MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
PROCESSED_DATA = os.path.expanduser("~/input/kaggleAdDisplayChallenge_processed.npz")
EMB_DIM = 16
LARGE_TABLE_THRESHOLD = 50000
HOTCOLD_DIR = "results/hotcold"
REORDER_DIR = "results/reorder"
ONDEMAND_DIR = "results/ondemand"

from codec_ondemand_benchmark import (
    create_args, load_model_and_data, quantize_table,
    GlobalFrameCache, OnDemandPrefetchCache, CompressedEmbeddingBag,
    InMemoryFrameDecoder, RESOLUTIONS, HOT_COVERAGE
)

def main():
    print("Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]

    # ========== Baseline profiling ==========
    print("\n=== BASELINE PROFILING ===")
    with torch.no_grad():
        for k in emb_keys:
            t_idx = int(k.split('.')[1])
            dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

    emb_times_baseline = []
    mlp_times = []
    total_times_baseline = []

    with torch.no_grad():
        for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
            if batch_idx < 100:  # skip warmup
                continue
            if batch_idx >= 600:
                break

            bt0 = time.time()

            # Embedding lookups
            et0 = time.time()
            ly = dlrm.apply_emb(lS_o, lS_i, dlrm.emb_l, dlrm.v_W_l)
            emb_time = time.time() - et0

            # MLP + interaction
            mt0 = time.time()
            x = dlrm.apply_mlp(X, dlrm.bot_l)
            z = dlrm.interact_features(x, ly)
            p = dlrm.apply_mlp(z, dlrm.top_l)
            mlp_time = time.time() - mt0

            total_time = time.time() - bt0
            emb_times_baseline.append(emb_time * 1000)
            mlp_times.append(mlp_time * 1000)
            total_times_baseline.append(total_time * 1000)

    print(f"Baseline embedding: {np.mean(emb_times_baseline):.2f}ms")
    print(f"Baseline MLP+interact: {np.mean(mlp_times):.2f}ms")
    print(f"Baseline total: {np.mean(total_times_baseline):.2f}ms")

    # ========== Codec profiling ==========
    print("\n=== CODEC PROFILING (global=8, disk, fp32 hot) ===")
    res_name = '4K'
    width, height = RESOLUTIONS[res_name]
    rows_per_frame = (width * height) // EMB_DIM
    res_dir = os.path.join(ONDEMAND_DIR, res_name)

    # Load hot/cold data
    is_hot = {}
    hot_indices = {}
    cold_num_rows = {}
    cold_quant_scale = {}
    cold_quant_zp = {}
    orig_to_cold_reordered = {}

    for t in large_tables:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]
        orig_to_cold_reordered[t] = torch.load(
            os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt'),
            map_location='cpu', weights_only=True)
        with open(os.path.join(REORDER_DIR, f'num_cold_{t}.txt')) as f:
            cold_num_rows[t] = int(f.read().strip())
        meta_path = os.path.join(res_dir, f'table_{t}', 'meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        cold_quant_scale[t] = meta['quant_scale']
        cold_quant_zp[t] = meta['quant_zp']

    # Set up codec
    global_cache = GlobalFrameCache(capacity=8, store_uint8=True)
    caches = {}

    for t_idx in large_tables:
        n_cold = cold_num_rows.get(t_idx, 0)
        if n_cold == 0:
            continue
        frame_dir = os.path.join(res_dir, f'table_{t_idx}')
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
            embedding_dim=EMB_DIM)
        dlrm.emb_l[t_idx] = comp_emb

    # Warmup cache (1 batch to load all 8 frames)
    with torch.no_grad():
        for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
            Z = dlrm(X, lS_o, lS_i)
            if batch_idx >= 5:
                break

    emb_times_codec = []
    forward_breakdown = {'cpp_call': [], 'cold_check': [], 'total_emb': []}

    with torch.no_grad():
        for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_ld):
            if batch_idx < 100:
                continue
            if batch_idx >= 600:
                break

            bt0 = time.time()

            # Embedding lookups (same as dlrm.apply_emb)
            et0 = time.time()
            ly = dlrm.apply_emb(lS_o, lS_i, dlrm.emb_l, dlrm.v_W_l)
            emb_time = time.time() - et0
            emb_times_codec.append(emb_time * 1000)

    print(f"Codec embedding: {np.mean(emb_times_codec):.2f}ms")
    print(f"  Overhead vs baseline: {np.mean(emb_times_codec) - np.mean(emb_times_baseline):.2f}ms")
    print(f"\n  Detail: codec calls apply_emb which iterates 26 tables.")
    print(f"  8 large tables use CompressedEmbeddingBag.forward()")
    print(f"  18 small tables use standard nn.EmbeddingBag.forward()")

    # Profile just the C++ call overhead for 1 table
    t_idx = large_tables[0]
    comp_emb = dlrm.emb_l[t_idx]
    test_indices = lS_i[t_idx] if isinstance(lS_i, list) else lS_i[t_idx]
    test_offsets = lS_o[t_idx] if isinstance(lS_o, list) else lS_o[t_idx]
    psw = torch.empty(0)

    cpp_times = []
    for _ in range(1000):
        t0 = time.time()
        if comp_emb.quantize_hot:
            out, cm, cc = _C.compressed_emb_bag_forward_q8(
                test_indices, test_offsets, comp_emb.hot_weight_q8,
                comp_emb.is_hot, comp_emb.orig_to_hot, psw,
                comp_emb.hot_scale, comp_emb.hot_zp)
        else:
            out, cm, cc = _C.compressed_emb_bag_forward(
                test_indices, test_offsets, comp_emb.hot_weight,
                comp_emb.is_hot, comp_emb.orig_to_hot, psw)
        cpp_times.append((time.time() - t0) * 1000)

    print(f"\n  C++ forward call alone: {np.mean(cpp_times):.3f}ms")

    # Profile Python wrapper overhead
    wrapper_times = []
    for _ in range(1000):
        t0 = time.time()
        out = comp_emb.forward(test_indices, test_offsets)
        wrapper_times.append((time.time() - t0) * 1000)

    print(f"  Python wrapper call: {np.mean(wrapper_times):.3f}ms")
    print(f"  Python overhead: {np.mean(wrapper_times) - np.mean(cpp_times):.3f}ms per table")
    print(f"  Estimated 8-table Python overhead: {(np.mean(wrapper_times) - np.mean(cpp_times)) * 8:.3f}ms")

    # Compare with standard EmbeddingBag
    small_t = [t for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD][0]
    std_emb = dlrm.emb_l[small_t]
    small_indices = lS_i[small_t] if isinstance(lS_i, list) else lS_i[small_t]
    small_offsets = lS_o[small_t] if isinstance(lS_o, list) else lS_o[small_t]
    std_times = []
    for _ in range(1000):
        t0 = time.time()
        out = std_emb(small_indices, small_offsets)
        std_times.append((time.time() - t0) * 1000)
    print(f"\n  Standard nn.EmbeddingBag call: {np.mean(std_times):.3f}ms")


if __name__ == '__main__':
    main()
