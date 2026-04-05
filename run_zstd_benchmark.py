#!/usr/bin/env python3
"""Quick standalone Zstd benchmark: encode + fullcpp + LRU inference runs.
Reuses the codec_ondemand_benchmark infrastructure but only runs Zstd configs.
"""
import os, sys, time, json, gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C
from codec_ondemand_benchmark import (
    log, load_model_and_data, quantize_table, encode_zstd_perframe,
    OnDemandFrameDecoder, OnDemandPrefetchCache, GlobalFrameCache,
    CompressedEmbeddingBag, get_rss_mb, drop_caches,
    EMB_DIM, RESOLUTIONS, RES_ZSTD, REORDER_DIR, HOTCOLD_DIR, ONDEMAND_DIR,
)
from sklearn.metrics import roc_auc_score
import torch.nn as nn

def main():
    log("=" * 70)
    log("ZSTD BENCHMARK — encode + full_cpp + LRU inference")
    log("=" * 70)

    # Load model and data
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = dlrm.state_dict()
    emb_keys = {i: k for i, k in enumerate(
        [k for k in state_dict if k.startswith('emb_l.') and k.endswith('.weight')])}
    # Fix: proper key mapping
    emb_keys = {}
    for k in state_dict:
        if k.startswith('emb_l.') and k.endswith('.weight'):
            t_idx = int(k.split('.')[1])
            emb_keys[t_idx] = k

    LARGE_TABLE_THRESHOLD = 50000
    large_tables = [i for i in range(len(ln_emb)) if ln_emb[i] > LARGE_TABLE_THRESHOLD]
    num_tabs = len(ln_emb)
    total_emb_mb = sum(ln_emb[i] * EMB_DIM * 4 for i in range(len(ln_emb))) / 1024 / 1024

    # Load hot/cold data
    is_hot, hot_indices, cold_indices = {}, {}, {}
    orig_to_cold_reordered, cold_quant_scale, cold_quant_zp, cold_num_rows = {}, {}, {}, {}

    for t in large_tables:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]
        fp = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        orig_to_cold_reordered[t] = torch.load(fp, map_location='cpu', weights_only=True)
        n_cold_path = os.path.join(REORDER_DIR, f'num_cold_{t}.txt')
        with open(n_cold_path) as f:
            cold_num_rows[t] = int(f.read().strip())

    o2c_map = orig_to_cold_reordered

    # Phase 1: Encode Zstd frames (if not already done)
    for res_name in ['1080p_zstd3', '1080p_zstd9']:
        width, height = RESOLUTIONS[res_name]
        pixels_per_frame = width * height
        rows_per_frame = pixels_per_frame // EMB_DIM
        res_dir = os.path.join(ONDEMAND_DIR, res_name)
        done_marker = os.path.join(res_dir, '.done')

        if os.path.exists(done_marker):
            log(f"\nEncoding for {res_name} SKIPPED — already done")
            for t in large_tables:
                meta_path = os.path.join(res_dir, f'table_{t}', 'meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    cold_quant_scale[t] = meta['quant_scale']
                    cold_quant_zp[t] = meta['quant_zp']
            continue

        log(f"\n{'='*70}")
        log(f"ENCODING: {res_name}")
        log(f"{'='*70}")
        os.makedirs(res_dir, exist_ok=True)

        zstd_level = RES_ZSTD[res_name]
        total_compressed, total_raw = 0, 0

        for t in large_tables:
            n_cold = cold_num_rows[t]
            if n_cold == 0:
                continue
            log(f"\n  Table {t}: {n_cold:,} cold embeddings")
            cold_order = np.load(os.path.join(REORDER_DIR, f'cold_order_{t}.npy'))
            reordered_w = state_dict[emb_keys[t]][cold_order]
            q, s, zp = quantize_table(reordered_w)
            cold_quant_scale[t] = s
            cold_quant_zp[t] = zp

            num_frames, frame_dir, compressed_bytes, enc_time, rpf = \
                encode_zstd_perframe(q.numpy(), rows_per_frame,
                                     output_dir=res_dir, table_id=t, level=zstd_level)
            meta = {
                'num_frames': num_frames, 'rows_per_frame': rpf,
                'width': width, 'height': height, 'n_cold': n_cold,
                'compressed_bytes': compressed_bytes, 'raw_bytes': n_cold * EMB_DIM,
                'quant_scale': s, 'quant_zp': zp,
            }
            with open(os.path.join(frame_dir, 'meta.json'), 'w') as f:
                json.dump(meta, f)
            total_compressed += compressed_bytes
            total_raw += n_cold * EMB_DIM
            del reordered_w, q; gc.collect()

        ratio = total_raw / total_compressed if total_compressed > 0 else 0
        log(f"\n  {res_name} total: {total_raw/1024/1024:.1f}MB uint8 -> "
            f"{total_compressed/1024/1024:.1f}MB ({ratio:.2f}x uint8, {ratio*4:.2f}x fp32)")
        with open(done_marker, 'w') as f:
            f.write(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Phase 2: Pre-cache test batches
    log("\nPre-caching test batches...")
    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    log(f"Pre-cached {len(test_batches)} batches")

    # Baseline
    log("\n--- Baseline ---")
    drop_caches(); time.sleep(0.3); gc.collect()
    scores, targets, blats = [], [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            t0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - t0)
            scores.extend(Z.detach().cpu().numpy().ravel().tolist())
            targets.extend(T.detach().cpu().numpy().ravel().tolist())
    baseline_auc = roc_auc_score(targets, scores)
    baseline_lat = np.mean(blats) * 1000
    log(f"  AUC={baseline_auc:.6f}, mean_lat={baseline_lat:.2f}ms")

    original_emb_modules = {t: dlrm.emb_l[t] for t in large_tables}

    def restore_weights():
        with torch.no_grad():
            for t_idx in emb_keys:
                dlrm.emb_l[t_idx].weight.data = state_dict[emb_keys[t_idx]].clone()

    # Phase 3: Run Zstd inference configs
    all_results = {'baseline': {'auc': baseline_auc, 'mean_lat_ms': baseline_lat}}

    for res_name, full_cpp, cache_cap, tag in [
        ('1080p_zstd3', True, 32, 'zstd3_bitmap_fullcpp'),
        ('1080p_zstd9', True, 32, 'zstd9_bitmap_fullcpp'),
        ('1080p_zstd3', False, 20, 'zstd3_bitmap_lru20'),
    ]:
        width, height = RESOLUTIONS[res_name]
        rows_per_frame = (width * height) // EMB_DIM
        res_dir = os.path.join(ONDEMAND_DIR, res_name)

        log(f"\n--- {tag} ---")
        restore_weights()
        gc.collect()

        global_cache = GlobalFrameCache(capacity=9999 if full_cpp else cache_cap, store_uint8=True)
        caches = {}
        total_compressed_bytes = 0
        total_hot_mb = 0
        total_mapping_mb = 0

        for t_idx in large_tables:
            n_cold = cold_num_rows.get(t_idx, 0)
            if n_cold == 0:
                continue
            frame_dir = os.path.join(res_dir, f'table_{t_idx}')
            if not os.path.exists(frame_dir):
                continue

            frame_files = sorted([f for f in os.listdir(frame_dir)
                                  if f.startswith('frame_') and f.endswith('.zst')])
            comp_bytes = sum(os.path.getsize(os.path.join(frame_dir, ff)) for ff in frame_files)
            total_compressed_bytes += comp_bytes

            cache = OnDemandPrefetchCache(
                frame_dir=frame_dir, rows_per_frame=rows_per_frame,
                emb_dim=EMB_DIM, num_cold_rows=n_cold,
                width=width, height=height,
                quant_scale=cold_quant_scale[t_idx],
                quant_zp=cold_quant_zp[t_idx],
                cache_capacity=cache_cap, predictor=None,
                num_prefetch_workers=2, global_cache=global_cache,
                table_id=t_idx,
            )
            caches[t_idx] = cache

            w = state_dict[emb_keys[t_idx]]
            h_idx = hot_indices[t_idx]
            hot_weight = w[h_idx].clone()
            total_hot_mb += hot_weight.numel() / 1024 / 1024  # uint8

            orig_to_hot = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
            orig_to_hot[h_idx] = torch.arange(len(h_idx))

            comp_emb = CompressedEmbeddingBag(
                hot_weight=hot_weight, is_hot=is_hot[t_idx],
                orig_to_hot=orig_to_hot,
                orig_to_cold_reordered=o2c_map[t_idx],
                cold_cache=cache, num_embeddings=ln_emb[t_idx],
                embedding_dim=EMB_DIM, quantize_hot=True,
            )
            total_mapping_mb += ln_emb[t_idx] * 4 / 1024 / 1024
            dlrm.emb_l[t_idx] = comp_emb

        # Free weights
        freed_state = {}
        for t_idx in large_tables:
            k = emb_keys[t_idx]
            if k in state_dict:
                freed_state[k] = state_dict.pop(k)
        for t_idx in large_tables:
            if t_idx in original_emb_modules:
                original_emb_modules[t_idx].weight = nn.Parameter(
                    torch.zeros(1, EMB_DIM), requires_grad=False)
        gc.collect()

        compressed_table_ids = set(caches.keys())

        # Register C++ tables
        if compressed_table_ids:
            _orig_apply_emb = dlrm.apply_emb
            table_kinds, weights, mappings, scales, zero_points = [], [], [], [], []
            for i in range(num_tabs):
                E = dlrm.emb_l[i]
                if isinstance(E, CompressedEmbeddingBag):
                    table_kinds.append(2)  # q8_merged
                    weights.append(E.hot_weight_q8)
                    mappings.append(E.mapping)
                    scales.append(float(E.hot_scale))
                    zero_points.append(int(E.hot_zp))
                else:
                    table_kinds.append(0)
                    weights.append(E.weight)
                    mappings.append(torch.empty(0, dtype=torch.int32))
                    scales.append(0.0)
                    zero_points.append(0)
            _C.register_tables(table_kinds, weights, mappings, scales, zero_points,
                              False, True)  # use_bitmap=True

        # Full_cpp: scan + decode all needed frames
        _full_cpp_active = False
        _cold_frame_mb = 0.0
        if full_cpp and compressed_table_ids and global_cache is not None:
            log(f"  Scanning all batches for cold frame coverage...")
            t_scan0 = time.time()
            needed_frames = {t_idx: set() for t_idx in caches}
            for X_s, lS_o_s, lS_i_s, T_s in test_batches:
                for t_idx in caches:
                    indices = lS_i_s[t_idx]
                    cold_mask = ~is_hot[t_idx][indices]
                    if cold_mask.any():
                        cold_orig = indices[cold_mask]
                        cold_mapped = o2c_map[t_idx][cold_orig]
                        valid = cold_mapped >= 0
                        if valid.any():
                            fids = (cold_mapped[valid] // rows_per_frame).unique().tolist()
                            needed_frames[t_idx].update(fids)
            total_needed = sum(len(v) for v in needed_frames.values())
            log(f"  Found {total_needed} frames ({time.time()-t_scan0:.1f}s)")

            # Batch Zstd decode
            t_decode0 = time.time()
            for t_idx in caches:
                fids = sorted(needed_frames[t_idx])
                if not fids:
                    continue
                decoder = caches[t_idx]._decoder
                paths = [os.path.join(decoder.frame_dir,
                         f'frame_{fid:05d}{decoder._frame_ext}') for fid in fids]
                original_size = decoder.rows_per_frame * decoder.emb_dim
                decoded = _C.batch_zstd_decompress_files(paths, original_size, 0)
                for i, fid in enumerate(fids):
                    rows = decoded[i].view(decoder.rows_per_frame, decoder.emb_dim)
                    global_cache.put(t_idx, fid, rows)
            decode_ms = (time.time() - t_decode0) * 1000
            log(f"  Zstd batch decode: {total_needed} frames in {decode_ms:.1f}ms")

            # Register in C++
            total_cold_frame_mb = 0
            for t_idx in caches:
                table_frames = {}
                for (tid, fid), data in global_cache.cache.items():
                    if tid == t_idx:
                        table_frames[fid] = data
                if not table_frames:
                    continue
                sorted_fids = sorted(table_frames.keys())
                padded_frames = []
                for fid in sorted_fids:
                    frame = table_frames[fid]
                    rows = torch.from_numpy(frame) if isinstance(frame, np.ndarray) else frame
                    if rows.dim() == 1:
                        rows = rows.view(-1, EMB_DIM)
                    if rows.shape[0] < rows_per_frame:
                        pad = torch.zeros(rows_per_frame - rows.shape[0], EMB_DIM, dtype=torch.uint8)
                        rows = torch.cat([rows, pad], dim=0)
                    padded_frames.append(rows)
                all_data = torch.cat(padded_frames, dim=0)

                # Bitmap mode
                n_rows = ln_emb[t_idx]
                is_hot_t = is_hot[t_idx][:n_rows]
                cold_orig = torch.where(~is_hot_t)[0]
                n_cold = cold_orig.size(0)
                cold_reordered = o2c_map[t_idx][cold_orig].long()

                max_fid = max(sorted_fids) + 1
                fid_to_offset = torch.full((max_fid,), -1, dtype=torch.long)
                for ci, fid in enumerate(sorted_fids):
                    fid_to_offset[fid] = ci * rows_per_frame
                fids_t = cold_reordered // rows_per_frame
                rows_in_frame = cold_reordered % rows_per_frame
                valid = (cold_reordered >= 0) & (fids_t < max_fid)
                offsets = torch.where(valid,
                    fid_to_offset[fids_t.clamp(0, max_fid - 1)], torch.tensor(-1))
                valid = valid & (offsets >= 0)
                src_pos = offsets + rows_in_frame
                valid_cold_ranks = torch.where(valid)[0].long()
                valid_data = all_data[src_pos[valid]]

                _C.register_cold_sparse_flat(
                    t_idx, valid_data, valid_cold_ranks,
                    float(cold_quant_scale[t_idx]),
                    float(cold_quant_zp[t_idx]), n_cold)
                total_cold_frame_mb += valid_data.nbytes / 1024 / 1024

            _full_cpp_active = True
            _cold_frame_mb = total_cold_frame_mb
            log(f"  Cold frames registered: {total_cold_frame_mb:.1f}MB uint8")
            global_cache.cache.clear(); gc.collect()

            def _full_cpp_apply_emb(lS_o, lS_i, emb_l, v_W_l):
                st = time.time()
                lS_i_2d = torch.stack(lS_i) if isinstance(lS_i, (list, tuple)) else (
                    lS_i if lS_i.dim() == 2 else lS_i.view(num_tabs, -1))
                lS_o_2d = torch.stack(lS_o) if isinstance(lS_o, (list, tuple)) else (
                    lS_o if lS_o.dim() == 2 else lS_o.view(num_tabs, -1))
                results = _C.fast_forward(lS_i_2d, lS_o_2d)
                dlrm.time_look_up += time.time() - st
                return results[-1]
            dlrm.apply_emb = _full_cpp_apply_emb

        # Run inference
        drop_caches(); time.sleep(0.3); gc.collect()
        torch.set_num_threads(40)

        scores_arr = np.empty(len(test_batches) * 2048 + 2048, dtype=np.float32)
        targets_arr = np.empty(len(test_batches) * 2048 + 2048, dtype=np.float32)
        si = 0
        blats = []
        dlrm.time_look_up = 0; dlrm.time_interact = 0; dlrm.time_mlp = 0

        t0 = time.time()
        with torch.no_grad():
            for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
                bt0 = time.time()
                Z = dlrm(X, lS_o, lS_i)
                blats.append(time.time() - bt0)
                z_np = Z.detach().cpu().numpy().ravel()
                t_np = T.detach().cpu().numpy().ravel()
                bs = z_np.shape[0]
                scores_arr[si:si+bs] = z_np
                targets_arr[si:si+bs] = t_np
                si += bs
        total_time = time.time() - t0

        auc = roc_auc_score(targets_arr[:si], scores_arr[:si])
        n_b = len(blats)
        rss = get_rss_mb()

        # Bitmap mapping MB
        bm_mb = 0
        for t_idx in caches:
            n_rows = ln_emb[t_idx]
            n_words = (n_rows + 63) // 64
            bm_mb += (n_words * 8 + (n_words + 1) * 4) / 1024 / 1024
        effective_mapping_mb = bm_mb
        if _full_cpp_active:
            for t_idx in caches:
                n_cold = int((~is_hot[t_idx][:ln_emb[t_idx]]).sum().item())
                n_words = (n_cold + 63) // 64
                effective_mapping_mb += (n_words * 8 + (n_words + 1) * 4) / 1024 / 1024

        total_mem_mb = total_hot_mb + 0.0 + _cold_frame_mb + effective_mapping_mb

        log(f"  AUC={auc:.6f}, Time={total_time:.2f}s")
        log(f"  Batch latency: mean={np.mean(blats)*1000:.2f}ms, "
            f"p50={np.percentile(blats,50)*1000:.2f}ms, "
            f"p99={np.percentile(blats,99)*1000:.2f}ms")
        log(f"  Forward: emb={dlrm.time_look_up/n_b*1000:.2f}ms, "
            f"interact={dlrm.time_interact/n_b*1000:.2f}ms, "
            f"mlp={dlrm.time_mlp/n_b*1000:.2f}ms")
        log(f"  Memory: hot={total_hot_mb:.1f} + cold_frames={_cold_frame_mb:.1f} "
            f"+ map={effective_mapping_mb:.1f} = {total_mem_mb:.1f}MB "
            f"(vs {total_emb_mb:.1f}MB baseline, {total_emb_mb/total_mem_mb:.1f}x)")
        log(f"  RSS={rss:.0f}MB")

        all_results[tag] = {
            'auc': auc, 'total_time': total_time,
            'mean_lat_ms': np.mean(blats)*1000,
            'p50_lat_ms': np.percentile(blats,50)*1000,
            'p99_lat_ms': np.percentile(blats,99)*1000,
            'hot_mb': total_hot_mb,
            'cold_frame_mb': _cold_frame_mb,
            'mapping_mb': effective_mapping_mb,
            'total_mem_mb': total_mem_mb,
            'compressed_on_disk_mb': total_compressed_bytes / 1024 / 1024,
            'rss_mb': rss,
            'decode_setup_ms': decode_ms if full_cpp else 0,
        }

        # Cleanup
        for c in caches.values():
            c.close()
        del caches
        if compressed_table_ids:
            dlrm.apply_emb = _orig_apply_emb
        for k, v in freed_state.items():
            state_dict[k] = v
        for t_idx in large_tables:
            dlrm.emb_l[t_idx] = original_emb_modules[t_idx]
            with torch.no_grad():
                dlrm.emb_l[t_idx].weight = nn.Parameter(
                    state_dict[emb_keys[t_idx]].clone(), requires_grad=False)
        gc.collect()

    # Save
    json_path = os.path.join('results/codec_exploration', 'zstd_benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {json_path}")

    # Summary
    log(f"\n{'='*70}")
    log("COMPARISON SUMMARY")
    log(f"{'='*70}")
    log(f"{'Config':<30} {'AUC':>10} {'Lat(ms)':>8} {'p99(ms)':>8} {'Mem(MB)':>8}")
    log("-" * 70)
    for k, v in all_results.items():
        log(f"{k:<30} {v.get('auc',0):>10.6f} {v.get('mean_lat_ms',0):>8.2f} "
            f"{v.get('p99_lat_ms',0):>8.2f} {v.get('total_mem_mb',0):>8.1f}")

if __name__ == '__main__':
    main()
