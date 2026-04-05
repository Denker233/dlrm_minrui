#!/usr/bin/env python3
"""Pipelined Zstd decode benchmark.

Compares three modes:
1. full_cpp: decode ALL frames before inference (current approach)
2. pipelined: decode frames incrementally with lookahead, limit memory budget
3. lru_noprefetch: LRU cache without any prefetch (worst case)

The pipeline works:
- Pre-scan all batches to find per-batch frame needs
- Before batch 0: decode its frames synchronously
- While batch N inference runs: background thread decodes batch N+1's new frames
- LRU eviction keeps memory bounded

Since Zstd decode (0.83ms/frame) overlaps with inference (4.5ms/batch),
the pipeline should have near-zero interference.
"""
import os, sys, time, json, gc, threading
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, Future

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


def prescan_batch_frames(test_batches, caches, is_hot, o2c_map, rows_per_frame):
    """Pre-scan all batches to find per-batch frame needs for each table."""
    n_batches = len(test_batches)
    # batch_frames[batch_idx] = {table_idx: set of frame_ids}
    batch_frames = [{} for _ in range(n_batches)]
    all_frames = {t: set() for t in caches}

    for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
        for t_idx in caches:
            indices = lS_i[t_idx]
            cold_mask = ~is_hot[t_idx][indices]
            if cold_mask.any():
                cold_orig = indices[cold_mask]
                cold_mapped = o2c_map[t_idx][cold_orig]
                valid = cold_mapped >= 0
                if valid.any():
                    fids = (cold_mapped[valid] // rows_per_frame).unique().tolist()
                    batch_frames[bi][t_idx] = set(fids)
                    all_frames[t_idx].update(fids)
            if t_idx not in batch_frames[bi]:
                batch_frames[bi][t_idx] = set()

    total = sum(len(v) for v in all_frames.values())
    log(f"  Pre-scan: {total} unique frames across all batches")
    return batch_frames, all_frames


def decode_frames_for_tables(caches, frame_ids_per_table, rows_per_frame):
    """Decode specific frames for each table. Returns {table_idx: {fid: tensor}}."""
    results = {}
    for t_idx, fids in frame_ids_per_table.items():
        if not fids:
            continue
        decoder = caches[t_idx]._decoder
        fids_sorted = sorted(fids)
        paths = [os.path.join(decoder.frame_dir,
                 f'frame_{fid:05d}{decoder._frame_ext}') for fid in fids_sorted]
        original_size = decoder.rows_per_frame * decoder.emb_dim
        decoded = _C.batch_zstd_decompress_files(paths, original_size, 0)
        results[t_idx] = {}
        for i, fid in enumerate(fids_sorted):
            results[t_idx][fid] = decoded[i].view(rows_per_frame, EMB_DIM)
    return results


def register_frames_in_cpp(caches, frame_cache, is_hot, o2c_map, ln_emb,
                           rows_per_frame, cold_quant_scale, cold_quant_zp):
    """Register all cached frames in C++ for fast_forward."""
    total_mb = 0
    for t_idx in caches:
        table_frames = {}
        for (tid, fid), data in frame_cache.items():
            if tid == t_idx:
                table_frames[fid] = data
        if not table_frames:
            continue

        sorted_fids = sorted(table_frames.keys())
        padded = []
        for fid in sorted_fids:
            frame = table_frames[fid]
            rows = torch.from_numpy(frame) if isinstance(frame, np.ndarray) else frame
            if rows.dim() == 1:
                rows = rows.view(-1, EMB_DIM)
            if rows.shape[0] < rows_per_frame:
                pad = torch.zeros(rows_per_frame - rows.shape[0], EMB_DIM, dtype=torch.uint8)
                rows = torch.cat([rows, pad], dim=0)
            padded.append(rows)
        all_data = torch.cat(padded, dim=0)

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
        total_mb += valid_data.nbytes / 1024 / 1024
    return total_mb


def main():
    log("=" * 70)
    log("PIPELINED ZSTD DECODE BENCHMARK")
    log("=" * 70)

    # Load model and data
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = dlrm.state_dict()
    emb_keys = {}
    for k in state_dict:
        if k.startswith('emb_l.') and k.endswith('.weight'):
            emb_keys[int(k.split('.')[1])] = k

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
        orig_to_cold_reordered[t] = torch.load(
            os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt'),
            map_location='cpu', weights_only=True)
        with open(os.path.join(REORDER_DIR, f'num_cold_{t}.txt')) as f:
            cold_num_rows[t] = int(f.read().strip())

    o2c_map = orig_to_cold_reordered

    # Ensure Zstd frames are encoded
    res_name = '1080p_zstd3'
    width, height = RESOLUTIONS[res_name]
    rows_per_frame = (width * height) // EMB_DIM
    res_dir = os.path.join(ONDEMAND_DIR, res_name)
    done_marker = os.path.join(res_dir, '.done')

    if os.path.exists(done_marker):
        log(f"\nEncoding for {res_name} SKIPPED — already done")
        for t in large_tables:
            meta_path = os.path.join(res_dir, f'table_{t}', 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as fp:
                    meta = json.load(fp)
                cold_quant_scale[t] = meta['quant_scale']
                cold_quant_zp[t] = meta['quant_zp']
    else:
        log(f"ERROR: Run run_zstd_benchmark.py first to encode Zstd frames")
        return

    # Pre-cache test batches
    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    log(f"Pre-cached {len(test_batches)} test batches")

    original_emb_modules = {t: dlrm.emb_l[t] for t in large_tables}
    freed_state = {}

    def restore_weights():
        for k, v in freed_state.items():
            state_dict[k] = v
        for t_idx in large_tables:
            dlrm.emb_l[t_idx] = original_emb_modules[t_idx]
            with torch.no_grad():
                dlrm.emb_l[t_idx].weight = nn.Parameter(
                    state_dict[emb_keys[t_idx]].clone(), requires_grad=False)

    def setup_compressed_tables():
        """Set up CompressedEmbeddingBag for each large table. Returns caches dict."""
        caches = {}
        total_hot_mb = 0

        for t_idx in large_tables:
            n_cold = cold_num_rows.get(t_idx, 0)
            if n_cold == 0:
                continue
            frame_dir = os.path.join(res_dir, f'table_{t_idx}')
            if not os.path.exists(frame_dir):
                continue

            cache = OnDemandPrefetchCache(
                frame_dir=frame_dir, rows_per_frame=rows_per_frame,
                emb_dim=EMB_DIM, num_cold_rows=n_cold,
                width=width, height=height,
                quant_scale=cold_quant_scale[t_idx],
                quant_zp=cold_quant_zp[t_idx],
                cache_capacity=32, predictor=None,
                num_prefetch_workers=2,
                global_cache=GlobalFrameCache(capacity=9999, store_uint8=True),
                table_id=t_idx,
            )
            caches[t_idx] = cache

            w = state_dict[emb_keys[t_idx]]
            h_idx = hot_indices[t_idx]
            hot_weight = w[h_idx].clone()
            total_hot_mb += hot_weight.numel() / 1024 / 1024

            orig_to_hot = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
            orig_to_hot[h_idx] = torch.arange(len(h_idx))

            comp_emb = CompressedEmbeddingBag(
                hot_weight=hot_weight, is_hot=is_hot[t_idx],
                orig_to_hot=orig_to_hot,
                orig_to_cold_reordered=o2c_map[t_idx],
                cold_cache=cache, num_embeddings=ln_emb[t_idx],
                embedding_dim=EMB_DIM, quantize_hot=True,
            )
            dlrm.emb_l[t_idx] = comp_emb

        # Free state dict weights for large tables
        freed_state.clear()
        for t_idx in large_tables:
            k = emb_keys[t_idx]
            if k in state_dict:
                freed_state[k] = state_dict.pop(k)
        for t_idx in large_tables:
            if t_idx in original_emb_modules:
                original_emb_modules[t_idx].weight = nn.Parameter(
                    torch.zeros(1, EMB_DIM), requires_grad=False)
        gc.collect()

        # Register in C++
        table_kinds, weights, mappings, scales, zero_points = [], [], [], [], []
        for i in range(num_tabs):
            E = dlrm.emb_l[i]
            if isinstance(E, CompressedEmbeddingBag):
                table_kinds.append(2)
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
                          False, True)
        return caches, total_hot_mb

    def enable_full_cpp_apply():
        """Enable full C++ apply_emb (zero Python cold overhead)."""
        orig_apply = dlrm.apply_emb
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
        return orig_apply

    def run_inference(tag):
        """Run inference and return results dict."""
        drop_caches(); time.sleep(0.2); gc.collect()
        torch.set_num_threads(40)

        n_test = len(test_batches)
        scores_arr = np.empty(n_test * 2048 + 2048, dtype=np.float32)
        targets_arr = np.empty(n_test * 2048 + 2048, dtype=np.float32)
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
                if bi % 500 == 0:
                    log(f"    Batch {bi}: lat={blats[-1]*1000:.1f}ms")
        total_time = time.time() - t0
        auc = roc_auc_score(targets_arr[:si], scores_arr[:si])

        log(f"  [{tag}] AUC={auc:.6f}, Time={total_time:.2f}s")
        log(f"  Batch latency: mean={np.mean(blats)*1000:.2f}ms, "
            f"p50={np.percentile(blats,50)*1000:.2f}ms, "
            f"p99={np.percentile(blats,99)*1000:.2f}ms")
        return {
            'auc': auc, 'total_time': total_time,
            'mean_lat_ms': np.mean(blats) * 1000,
            'p50_lat_ms': np.percentile(blats, 50) * 1000,
            'p99_lat_ms': np.percentile(blats, 99) * 1000,
            'blats': blats,
        }

    all_results = {}

    # ====================================================================
    # MODE 1: full_cpp (baseline — decode ALL frames before inference)
    # ====================================================================
    log(f"\n{'='*70}")
    log("MODE 1: full_cpp (decode all frames before inference)")
    log(f"{'='*70}")

    caches, total_hot_mb = setup_compressed_tables()

    # Pre-scan + decode all
    batch_frames, all_frames = prescan_batch_frames(
        test_batches, caches, is_hot, o2c_map, rows_per_frame)

    t_decode0 = time.time()
    frame_cache = {}  # (table_idx, frame_id) -> tensor
    decoded = decode_frames_for_tables(caches, all_frames, rows_per_frame)
    for t_idx, fid_data in decoded.items():
        for fid, tensor in fid_data.items():
            frame_cache[(t_idx, fid)] = tensor
    decode_all_ms = (time.time() - t_decode0) * 1000
    total_frames = sum(len(v) for v in all_frames.values())
    log(f"  Decoded ALL {total_frames} frames in {decode_all_ms:.1f}ms")

    # Register all in C++
    cold_frame_mb = register_frames_in_cpp(
        caches, frame_cache, is_hot, o2c_map, ln_emb,
        rows_per_frame, cold_quant_scale, cold_quant_zp)
    log(f"  Registered {cold_frame_mb:.1f}MB cold frames in C++")

    orig_apply = enable_full_cpp_apply()

    res = run_inference("full_cpp")
    res['cold_frame_mb'] = cold_frame_mb
    res['decode_setup_ms'] = decode_all_ms
    res['mode'] = 'full_cpp'
    res['max_frames_in_memory'] = total_frames
    all_results['full_cpp'] = res

    # Cleanup
    dlrm.apply_emb = orig_apply
    restore_weights()
    del frame_cache; gc.collect()

    # ====================================================================
    # MODE 2: Pipelined decode (decode incrementally with lookahead)
    # ====================================================================
    for budget in [5, 10, 20]:
        log(f"\n{'='*70}")
        log(f"MODE 2: Pipelined decode (frame budget={budget})")
        log(f"{'='*70}")

        caches, total_hot_mb = setup_compressed_tables()

        batch_frames, all_frames = prescan_batch_frames(
            test_batches, caches, is_hot, o2c_map, rows_per_frame)

        # Compute per-batch NEW frames (frames not seen in previous batches)
        seen_frames = {t: set() for t in caches}
        new_per_batch = []
        for bi in range(len(test_batches)):
            new = {}
            for t_idx in caches:
                batch_needs = batch_frames[bi].get(t_idx, set())
                new_fids = batch_needs - seen_frames[t_idx]
                if new_fids:
                    new[t_idx] = new_fids
                seen_frames[t_idx].update(batch_needs)
            new_per_batch.append(new)

        total_new_after_first = sum(
            sum(len(v) for v in new_per_batch[bi].values())
            for bi in range(1, len(test_batches)))
        log(f"  Batch 0 needs {sum(len(v) for v in new_per_batch[0].values())} new frames")
        log(f"  Batches 1-{len(test_batches)-1} need {total_new_after_first} new frames total")

        # Pipeline: frame cache with LRU eviction
        frame_cache = {}  # (t_idx, fid) -> tensor
        frame_lru = {}    # (t_idx, fid) -> last_used_batch
        decode_times = []  # per-batch decode time
        pipeline_futures = []
        executor = ThreadPoolExecutor(max_workers=2)

        def evict_to_budget():
            """Evict oldest frames if over budget."""
            while len(frame_cache) > budget:
                oldest_key = min(frame_lru, key=frame_lru.get)
                del frame_cache[oldest_key]
                del frame_lru[oldest_key]

        def decode_and_cache_batch(bi):
            """Decode new frames for batch bi, add to cache."""
            t0 = time.time()
            new = new_per_batch[bi]
            if not new:
                return 0.0
            decoded = decode_frames_for_tables(caches, new, rows_per_frame)
            n_decoded = 0
            for t_idx, fid_data in decoded.items():
                for fid, tensor in fid_data.items():
                    frame_cache[(t_idx, fid)] = tensor
                    frame_lru[(t_idx, fid)] = bi
                    n_decoded += 1
            evict_to_budget()
            return (time.time() - t0) * 1000

        # Decode batch 0 synchronously
        t_pipe0 = time.time()
        ms = decode_and_cache_batch(0)
        decode_times.append(ms)

        # Touch all frames used by batch 0 for LRU
        for t_idx, fids in batch_frames[0].items():
            for fid in fids:
                frame_lru[(t_idx, fid)] = 0

        # Register initial frames in C++
        cold_frame_mb = register_frames_in_cpp(
            caches, frame_cache, is_hot, o2c_map, ln_emb,
            rows_per_frame, cold_quant_scale, cold_quant_zp)
        orig_apply = enable_full_cpp_apply()

        pipeline_setup_ms = (time.time() - t_pipe0) * 1000
        log(f"  Initial decode: {ms:.1f}ms, {len(frame_cache)} frames cached")

        # Pipelined inference
        drop_caches(); time.sleep(0.2); gc.collect()
        torch.set_num_threads(40)

        n_test = len(test_batches)
        scores_arr = np.empty(n_test * 2048 + 2048, dtype=np.float32)
        targets_arr = np.empty(n_test * 2048 + 2048, dtype=np.float32)
        si = 0
        blats = []
        prefetch_overlaps = 0  # batches where decode finished before inference
        re_registrations = 0
        total_pipeline_decode_ms = 0
        dlrm.time_look_up = 0

        # Pre-submit decode for batch 1
        pending_future = None
        if len(test_batches) > 1 and new_per_batch[1]:
            pending_future = executor.submit(decode_and_cache_batch, 1)

        t0 = time.time()
        with torch.no_grad():
            for bi in range(n_test):
                # Wait for any pending prefetch to complete
                if pending_future is not None:
                    ms = pending_future.result()
                    total_pipeline_decode_ms += ms
                    decode_times.append(ms)
                    pending_future = None

                    # If new frames were decoded, we need to re-register in C++
                    if ms > 0:
                        re_reg_t0 = time.time()
                        cold_frame_mb = register_frames_in_cpp(
                            caches, frame_cache, is_hot, o2c_map, ln_emb,
                            rows_per_frame, cold_quant_scale, cold_quant_zp)
                        re_registrations += 1

                # Update LRU for this batch's frames
                for t_idx, fids in batch_frames[bi].items():
                    for fid in fids:
                        if (t_idx, fid) in frame_lru:
                            frame_lru[(t_idx, fid)] = bi

                # Run inference
                X, lS_o, lS_i, T = test_batches[bi]
                bt0 = time.time()
                Z = dlrm(X, lS_o, lS_i)
                blats.append(time.time() - bt0)

                # Submit decode for batch bi+2 (bi+1 was already submitted)
                next_bi = bi + 2
                if next_bi < n_test and new_per_batch[next_bi]:
                    pending_future = executor.submit(decode_and_cache_batch, next_bi)

                z_np = Z.detach().cpu().numpy().ravel()
                t_np = T.detach().cpu().numpy().ravel()
                bs = z_np.shape[0]
                scores_arr[si:si+bs] = z_np
                targets_arr[si:si+bs] = t_np
                si += bs

                if bi % 500 == 0:
                    log(f"    Batch {bi}: lat={blats[-1]*1000:.1f}ms, "
                        f"cached={len(frame_cache)} frames")

        total_time = time.time() - t0
        auc = roc_auc_score(targets_arr[:si], scores_arr[:si])
        executor.shutdown(wait=False)

        log(f"  [pipelined_budget{budget}] AUC={auc:.6f}, Time={total_time:.2f}s")
        log(f"  Batch latency: mean={np.mean(blats)*1000:.2f}ms, "
            f"p50={np.percentile(blats,50)*1000:.2f}ms, "
            f"p99={np.percentile(blats,99)*1000:.2f}ms")
        log(f"  Pipeline: {re_registrations} re-registrations, "
            f"{total_pipeline_decode_ms:.1f}ms total decode")
        log(f"  Max frames in memory: {budget}")

        all_results[f'pipelined_budget{budget}'] = {
            'auc': auc, 'total_time': total_time,
            'mean_lat_ms': np.mean(blats) * 1000,
            'p50_lat_ms': np.percentile(blats, 50) * 1000,
            'p99_lat_ms': np.percentile(blats, 99) * 1000,
            'cold_frame_mb': cold_frame_mb,
            'mode': f'pipelined_budget{budget}',
            'max_frames_in_memory': budget,
            'total_pipeline_decode_ms': total_pipeline_decode_ms,
            're_registrations': re_registrations,
        }

        dlrm.apply_emb = orig_apply
        restore_weights()
        del frame_cache, frame_lru; gc.collect()

    # ====================================================================
    # Summary
    # ====================================================================
    log(f"\n{'='*70}")
    log("PIPELINED DECODE COMPARISON")
    log(f"{'='*70}")
    log(f"{'Config':<25} {'AUC':>10} {'Mean(ms)':>9} {'P50(ms)':>8} {'P99(ms)':>8} "
        f"{'ColdMB':>7} {'MaxFrm':>7}")
    log("-" * 80)
    for k, v in all_results.items():
        log(f"{k:<25} {v['auc']:>10.6f} {v['mean_lat_ms']:>9.2f} "
            f"{v['p50_lat_ms']:>8.2f} {v['p99_lat_ms']:>8.2f} "
            f"{v.get('cold_frame_mb',0):>7.1f} {v.get('max_frames_in_memory',0):>7}")

    # Interference analysis
    if 'full_cpp' in all_results:
        base_lat = all_results['full_cpp']['mean_lat_ms']
        for k, v in all_results.items():
            if k.startswith('pipelined'):
                overhead = (v['mean_lat_ms'] - base_lat) / base_lat * 100
                log(f"  {k} overhead vs full_cpp: {overhead:+.1f}%")

    # Save
    json_path = 'results/codec_exploration/pipelined_benchmark.json'
    save_data = {}
    for k, v in all_results.items():
        sv = {kk: vv for kk, vv in v.items() if kk != 'blats'}
        save_data[k] = sv
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
