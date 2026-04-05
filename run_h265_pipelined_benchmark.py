#!/usr/bin/env python3
"""
Pipelined H.265 decode benchmark.

Tests whether H.265 decode can be fully hidden behind inference latency.

Pipeline approach:
  1. Pre-scan all batches → per-batch frame requirements
  2. Batch 0: decode all needed frames synchronously
  3. While batch N runs inference: background thread decodes batch N+1's NEW frames
  4. LRU eviction keeps memory bounded

Configs tested:
  - CRF=18 ultrafast + filter (existing baseline)
  - CRF=30 medium + no-deblock:no-sao (faster decode, similar compression)

Uses decode_hevc_file_fast (context pool) for accelerated single-frame decode.
"""
import os, sys, time, json, gc
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C
from codec_ondemand_benchmark import (
    log, load_model_and_data, EMB_DIM, TILE_W, TILE_H,
    REORDER_DIR, HOTCOLD_DIR, ONDEMAND_DIR, LARGE_TABLE_THRESHOLD,
    OnDemandPrefetchCache, GlobalFrameCache, CompressedEmbeddingBag,
)
from sklearn.metrics import roc_auc_score

WIDTH, HEIGHT = 1920, 1080
ROWS_PER_FRAME = (WIDTH * HEIGHT) // EMB_DIM

# H.265 configs to test: (tag, res_dir_name)
H265_CONFIGS = [
    ('crf18_filter',    '1080p_crf18'),
    ('crf30_nofilter',  '1080p_crf30_nofilter'),
]


def prescan_batch_frames(test_batches, large_tables, is_hot, o2c_map):
    """Pre-scan all batches to find per-batch frame needs."""
    n_batches = len(test_batches)
    batch_frames = [{} for _ in range(n_batches)]
    all_frames = {t: set() for t in large_tables}

    for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
        for t_idx in large_tables:
            indices = lS_i[t_idx]
            cold_mask = ~is_hot[t_idx][indices]
            if cold_mask.any():
                cold_orig = indices[cold_mask]
                cold_mapped = o2c_map[t_idx][cold_orig]
                valid = cold_mapped >= 0
                if valid.any():
                    fids = (cold_mapped[valid] // ROWS_PER_FRAME).unique().tolist()
                    batch_frames[bi][t_idx] = set(fids)
                    all_frames[t_idx].update(fids)
            if t_idx not in batch_frames[bi]:
                batch_frames[bi][t_idx] = set()

    total = sum(len(v) for v in all_frames.values())
    log(f"  Pre-scan: {total} unique frames across {n_batches} batches")
    return batch_frames, all_frames


def decode_h265_frames(res_dir, frame_ids_per_table, use_pool=True):
    """Decode H.265 frames for specified tables. Returns {t_idx: {fid: uint8_tensor}}."""
    results = {}
    mode = 3 if use_pool else 0  # 3 = pool+fast_probe, 0 = original

    for t_idx, fids in frame_ids_per_table.items():
        if not fids:
            continue
        table_dir = os.path.join(res_dir, f'table_{t_idx}')

        # Find extension
        ext = '.h265'
        for e in ['.h265', '.mkv']:
            if os.path.exists(os.path.join(table_dir, f'frame_00000{e}')):
                ext = e
                break

        fids_sorted = sorted(fids)
        paths = [os.path.join(table_dir, f'frame_{fid:05d}{ext}') for fid in fids_sorted]

        # Batch decode with pool
        decoded = _C.batch_decode_fast(paths, mode, len(paths), True)  # skip_loop=True

        results[t_idx] = {}
        for i, fid in enumerate(fids_sorted):
            # H.265 returns (H, W) tiled frame — untile to (rows_per_frame, D)
            frame_2d = decoded[i]
            if frame_2d.dim() == 2:
                rows = _C.untile_frame_to_rows(frame_2d, ROWS_PER_FRAME)
            else:
                rows = frame_2d.view(ROWS_PER_FRAME, EMB_DIM)
            results[t_idx][fid] = rows

    return results


def register_frames_cpp(frame_cache, large_tables, is_hot, o2c_map, ln_emb,
                        cold_quant_scale, cold_quant_zp):
    """Register cached frames in C++ for fast_forward."""
    total_mb = 0
    for t_idx in large_tables:
        table_frames = {}
        for (tid, fid), data in frame_cache.items():
            if tid == t_idx:
                table_frames[fid] = data
        if not table_frames:
            continue

        sorted_fids = sorted(table_frames.keys())
        padded = []
        for fid in sorted_fids:
            rows = table_frames[fid]
            if rows.dim() == 1:
                rows = rows.view(-1, EMB_DIM)
            if rows.shape[0] < ROWS_PER_FRAME:
                pad = torch.zeros(ROWS_PER_FRAME - rows.shape[0], EMB_DIM, dtype=torch.uint8)
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
            fid_to_offset[fid] = ci * ROWS_PER_FRAME
        fids_t = cold_reordered // ROWS_PER_FRAME
        rows_in_frame = cold_reordered % ROWS_PER_FRAME
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
    log("PIPELINED H.265 DECODE BENCHMARK")
    log("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = dlrm.state_dict()
    emb_keys = {}
    for k in state_dict:
        if k.startswith('emb_l.') and k.endswith('.weight'):
            emb_keys[int(k.split('.')[1])] = k

    num_tabs = len(ln_emb)
    large_tables = [i for i in range(num_tabs) if ln_emb[i] > LARGE_TABLE_THRESHOLD]

    # Load hot/cold data
    is_hot, hot_indices = {}, {}
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

    def setup_compressed_tables(res_dir):
        caches = {}
        for t_idx in large_tables:
            n_cold = cold_num_rows.get(t_idx, 0)
            if n_cold == 0:
                continue
            frame_dir = os.path.join(res_dir, f'table_{t_idx}')
            if not os.path.exists(frame_dir):
                continue

            # Load quant params
            meta_path = os.path.join(frame_dir, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                cold_quant_scale[t_idx] = meta['quant_scale']
                cold_quant_zp[t_idx] = meta['quant_zp']

            cache = OnDemandPrefetchCache(
                frame_dir=frame_dir, rows_per_frame=ROWS_PER_FRAME,
                emb_dim=EMB_DIM, num_cold_rows=n_cold,
                width=WIDTH, height=HEIGHT,
                quant_scale=cold_quant_scale[t_idx],
                quant_zp=cold_quant_zp[t_idx],
                cache_capacity=32, predictor=None,
                num_prefetch_workers=2,
                global_cache=GlobalFrameCache(capacity=9999, store_uint8=True),
                table_id=t_idx)
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
                embedding_dim=EMB_DIM, quantize_hot=True)
            dlrm.emb_l[t_idx] = comp_emb

        # Free state dict
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

        # Register tables in C++
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
        return caches

    def enable_full_cpp():
        orig_apply = dlrm.apply_emb
        def _fast(lS_o, lS_i, emb_l, v_W_l):
            lS_i_2d = torch.stack(lS_i) if isinstance(lS_i, (list, tuple)) else (
                lS_i if lS_i.dim() == 2 else lS_i.view(num_tabs, -1))
            lS_o_2d = torch.stack(lS_o) if isinstance(lS_o, (list, tuple)) else (
                lS_o if lS_o.dim() == 2 else lS_o.view(num_tabs, -1))
            return _C.fast_forward(lS_i_2d, lS_o_2d)[-1]
        dlrm.apply_emb = _fast
        return orig_apply

    # Pre-scan batch frames (same for all H.265 configs)
    batch_frames, all_frames = prescan_batch_frames(
        test_batches, large_tables, is_hot, orig_to_cold_reordered)

    # Compute per-batch NEW frames
    seen_frames = {t: set() for t in large_tables}
    new_per_batch = []
    for bi in range(len(test_batches)):
        new = {}
        for t_idx in large_tables:
            batch_needs = batch_frames[bi].get(t_idx, set())
            new_fids = batch_needs - seen_frames[t_idx]
            if new_fids:
                new[t_idx] = new_fids
            seen_frames[t_idx].update(batch_needs)
        new_per_batch.append(new)

    total_new_b0 = sum(len(v) for v in new_per_batch[0].values())
    total_new_rest = sum(sum(len(v) for v in new_per_batch[bi].values())
                        for bi in range(1, len(test_batches)))
    log(f"  Batch 0: {total_new_b0} new frames")
    log(f"  Batches 1-{len(test_batches)-1}: {total_new_rest} new frames")

    # --- Baseline ---
    log(f"\n{'='*70}")
    log("BASELINE: fp32 (no compression)")
    log(f"{'='*70}")
    torch.set_num_threads(32)

    scores, targets, blats = [], [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            t0 = time.perf_counter()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.perf_counter() - t0)
            scores.append(Z.detach().numpy().ravel())
            targets.append(T.numpy().ravel())
    baseline_auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    baseline_lat = np.mean(blats) * 1000
    log(f"  AUC={baseline_auc:.6f}, mean_lat={baseline_lat:.2f}ms")

    all_results = {'baseline': {'auc': float(baseline_auc), 'mean_lat_ms': float(baseline_lat)}}

    # --- Test each H.265 config ---
    for tag, res_dir_name in H265_CONFIGS:
        res_dir = os.path.join(ONDEMAND_DIR, res_dir_name)
        if not os.path.exists(os.path.join(res_dir, '.done')):
            log(f"\n  SKIP {tag}: not encoded yet ({res_dir})")
            continue

        for mode in ['full_predecode', 'pipelined']:
            mode_tag = f'{tag}_{mode}'
            log(f"\n{'='*70}")
            log(f"{mode_tag}")
            log(f"{'='*70}")

            restore_weights()
            caches = setup_compressed_tables(res_dir)

            if mode == 'full_predecode':
                # Decode ALL frames before inference
                t_dec0 = time.perf_counter()
                frame_cache = {}
                decoded = decode_h265_frames(res_dir, all_frames, use_pool=True)
                for t_idx, fid_data in decoded.items():
                    for fid, tensor in fid_data.items():
                        frame_cache[(t_idx, fid)] = tensor
                decode_ms = (time.perf_counter() - t_dec0) * 1000
                total_f = sum(len(v) for v in all_frames.values())
                log(f"  Pre-decoded ALL {total_f} frames in {decode_ms:.1f}ms")

                cold_mb = register_frames_cpp(
                    frame_cache, large_tables, is_hot, orig_to_cold_reordered,
                    ln_emb, cold_quant_scale, cold_quant_zp)
                orig_apply = enable_full_cpp()

                torch.set_num_threads(32)
                gc.collect()

                scores, targets, blats = [], [], []
                with torch.no_grad():
                    for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
                        t0 = time.perf_counter()
                        Z = dlrm(X, lS_o, lS_i)
                        blats.append(time.perf_counter() - t0)
                        scores.append(Z.detach().numpy().ravel())
                        targets.append(T.numpy().ravel())

                auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
                mean_lat = np.mean(blats) * 1000
                p99_lat = np.percentile(blats, 99) * 1000

                log(f"  AUC={auc:.6f}, mean_lat={mean_lat:.2f}ms, p99={p99_lat:.2f}ms")
                log(f"  AUC loss: {(baseline_auc - auc)*100:+.4f}%")
                log(f"  Latency vs baseline: {mean_lat - baseline_lat:+.2f}ms")

                all_results[mode_tag] = {
                    'auc': float(auc), 'mean_lat_ms': float(mean_lat),
                    'p99_lat_ms': float(p99_lat),
                    'decode_setup_ms': float(decode_ms),
                    'auc_loss_pct': float((baseline_auc - auc) * 100),
                }

                dlrm.apply_emb = orig_apply
                del frame_cache; gc.collect()

            elif mode == 'pipelined':
                # Pipelined: decode next batch's frames during current inference
                executor = ThreadPoolExecutor(max_workers=2)
                frame_cache = {}
                frame_lru = {}

                def decode_and_cache(bi):
                    """Decode new frames for batch bi."""
                    new = new_per_batch[bi]
                    if not new:
                        return 0.0
                    t0 = time.perf_counter()
                    decoded = decode_h265_frames(res_dir, new, use_pool=True)
                    for t_idx, fid_data in decoded.items():
                        for fid, tensor in fid_data.items():
                            frame_cache[(t_idx, fid)] = tensor
                            frame_lru[(t_idx, fid)] = bi
                    return (time.perf_counter() - t0) * 1000

                # Decode batch 0 synchronously
                t_pipe0 = time.perf_counter()
                decode_and_cache(0)
                for t_idx, fids in batch_frames[0].items():
                    for fid in fids:
                        frame_lru[(t_idx, fid)] = 0

                cold_mb = register_frames_cpp(
                    frame_cache, large_tables, is_hot, orig_to_cold_reordered,
                    ln_emb, cold_quant_scale, cold_quant_zp)
                orig_apply = enable_full_cpp()
                setup_ms = (time.perf_counter() - t_pipe0) * 1000
                log(f"  Initial decode: {setup_ms:.1f}ms, {len(frame_cache)} frames")

                torch.set_num_threads(32)
                gc.collect()

                scores, targets, blats = [], [], []
                decode_overlaps = 0
                total_decode_ms = 0
                re_registrations = 0

                # Submit decode for batch 1
                pending = None
                if len(test_batches) > 1 and new_per_batch[1]:
                    pending = executor.submit(decode_and_cache, 1)

                with torch.no_grad():
                    for bi in range(len(test_batches)):
                        # Wait for pending decode
                        if pending is not None:
                            ms = pending.result()
                            total_decode_ms += ms
                            pending = None

                            if ms > 0:
                                # Re-register frames with new data
                                register_frames_cpp(
                                    frame_cache, large_tables, is_hot,
                                    orig_to_cold_reordered, ln_emb,
                                    cold_quant_scale, cold_quant_zp)
                                re_registrations += 1

                        # Update LRU
                        for t_idx, fids in batch_frames[bi].items():
                            for fid in fids:
                                if (t_idx, fid) in frame_lru:
                                    frame_lru[(t_idx, fid)] = bi

                        # Submit decode for batch bi+2
                        next_bi = bi + 2
                        if next_bi < len(test_batches) and new_per_batch[next_bi]:
                            pending = executor.submit(decode_and_cache, next_bi)

                        # Run inference
                        X, lS_o, lS_i, T = test_batches[bi]
                        t0 = time.perf_counter()
                        Z = dlrm(X, lS_o, lS_i)
                        blats.append(time.perf_counter() - t0)
                        scores.append(Z.detach().numpy().ravel())
                        targets.append(T.numpy().ravel())

                # Wait for any remaining
                if pending is not None:
                    pending.result()

                executor.shutdown(wait=False)

                auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
                mean_lat = np.mean(blats) * 1000
                p99_lat = np.percentile(blats, 99) * 1000

                log(f"  AUC={auc:.6f}, mean_lat={mean_lat:.2f}ms, p99={p99_lat:.2f}ms")
                log(f"  AUC loss: {(baseline_auc - auc)*100:+.4f}%")
                log(f"  Latency vs baseline: {mean_lat - baseline_lat:+.2f}ms")
                log(f"  Pipeline decode: {total_decode_ms:.1f}ms total, "
                    f"{re_registrations} re-registrations")

                all_results[mode_tag] = {
                    'auc': float(auc), 'mean_lat_ms': float(mean_lat),
                    'p99_lat_ms': float(p99_lat),
                    'total_pipeline_decode_ms': float(total_decode_ms),
                    're_registrations': re_registrations,
                    'auc_loss_pct': float((baseline_auc - auc) * 100),
                }

                dlrm.apply_emb = orig_apply
                del frame_cache; gc.collect()

    # Summary
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")
    log(f"{'Config':<35s} {'AUC':>9s} {'Loss':>8s} {'MeanLat':>8s} {'P99':>8s} {'vs base':>8s}")
    log("-" * 82)
    log(f"{'baseline':<35s} {baseline_auc:>9.6f} {'':>8s} {baseline_lat:>7.2f}ms {'':>8s} {'':>8s}")
    for tag, r in all_results.items():
        if tag == 'baseline':
            continue
        loss = r.get('auc_loss_pct', 0)
        lat = r['mean_lat_ms']
        p99 = r.get('p99_lat_ms', 0)
        diff = lat - baseline_lat
        log(f"{tag:<35s} {r['auc']:>9.6f} {loss:>+7.4f}% {lat:>7.2f}ms {p99:>7.2f}ms {diff:>+7.2f}ms")

    out_file = 'results/h265_pipelined_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
