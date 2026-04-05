#!/usr/bin/env python3
"""
Pipelined H.265 decode v2 — honest timing, no re-registration.

Uses CompressedEmbeddingBag + GlobalFrameCache (LRU) directly.
No C++ registration — cold lookups go through cache.lookup().

Pipeline:
  1. Pre-scan all batches → per-batch frame needs
  2. Batch 0: pre-decode needed frames into LRU cache
  3. While batch N inference runs: background thread decodes batch N+1's new frames
  4. Total wall-clock time per batch includes ALL overhead (wait, decode, etc.)

Tests CRF=18 with filter and CRF=30 medium no-filter.
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
    log, load_model_and_data, EMB_DIM,
    REORDER_DIR, HOTCOLD_DIR, ONDEMAND_DIR, LARGE_TABLE_THRESHOLD,
    OnDemandPrefetchCache, GlobalFrameCache, CompressedEmbeddingBag,
)
from sklearn.metrics import roc_auc_score

WIDTH, HEIGHT = 1920, 1080
ROWS_PER_FRAME = (WIDTH * HEIGHT) // EMB_DIM

H265_CONFIGS = [
    ('crf18_filter', '1080p_crf18'),
    ('crf30_nofilter', '1080p_crf30_nofilter'),
]


def prescan_batch_frames(test_batches, large_tables, is_hot, o2c_map):
    """Pre-scan all batches to find per-batch frame needs per table."""
    batch_frames = [{} for _ in range(len(test_batches))]
    all_frames = {t: set() for t in large_tables}

    for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
        for t in large_tables:
            indices = lS_i[t]
            cold_mask = ~is_hot[t][indices]
            if cold_mask.any():
                cold_orig = indices[cold_mask]
                cold_mapped = o2c_map[t][cold_orig]
                valid = cold_mapped >= 0
                if valid.any():
                    fids = (cold_mapped[valid] // ROWS_PER_FRAME).unique().tolist()
                    batch_frames[bi][t] = set(fids)
                    all_frames[t].update(fids)
            if t not in batch_frames[bi]:
                batch_frames[bi][t] = set()

    return batch_frames, all_frames


def predecode_frames(global_cache, caches, frame_ids_per_table):
    """Decode frames and put into the GlobalFrameCache."""
    for t, fids in frame_ids_per_table.items():
        if not fids:
            continue
        cache = caches[t]
        frame_dir = cache.frame_dir
        ext = cache._decoder._frame_ext

        fids_sorted = sorted(fids)
        paths = [os.path.join(frame_dir, f'frame_{fid:05d}{ext}')
                 for fid in fids_sorted]

        # Use pool-accelerated batch decode
        decoded = _C.batch_decode_fast(paths, 2, len(paths), True)  # mode=2 (pool)

        for i, fid in enumerate(fids_sorted):
            global_cache.put(t, fid, decoded[i])


def main():
    log("=" * 70)
    log("PIPELINED H.265 DECODE v2 — Honest Timing")
    log("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = {k: v.clone() for k, v in dlrm.state_dict().items()}
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tabs = len(ln_emb)
    large_tables = [i for i in range(num_tabs) if ln_emb[i] > LARGE_TABLE_THRESHOLD]

    # Load hot/cold data
    is_hot, hot_indices, o2c_map = {}, {}, {}
    cold_num_rows, cold_quant_scale, cold_quant_zp = {}, {}, {}

    for t in large_tables:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]
        o2c_map[t] = torch.load(
            os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt'),
            map_location='cpu', weights_only=True)
        with open(os.path.join(REORDER_DIR, f'num_cold_{t}.txt')) as f:
            cold_num_rows[t] = int(f.read().strip())

    original_embs = {t: dlrm.emb_l[t] for t in large_tables}

    # Pre-cache test batches
    test_batches = list(test_ld)
    log(f"Pre-cached {len(test_batches)} test batches")

    # Pre-scan frames
    batch_frames, all_frames = prescan_batch_frames(
        test_batches, large_tables, is_hot, o2c_map)

    # Per-batch new frames
    seen = {t: set() for t in large_tables}
    new_per_batch = []
    for bi in range(len(test_batches)):
        new = {}
        for t in large_tables:
            batch_needs = batch_frames[bi].get(t, set())
            new_fids = batch_needs - seen[t]
            if new_fids:
                new[t] = new_fids
            seen[t].update(batch_needs)
        new_per_batch.append(new)

    total_new_b0 = sum(len(v) for v in new_per_batch[0].values())
    total_new_rest = sum(sum(len(v) for v in nb.values()) for nb in new_per_batch[1:])
    total_frames = sum(len(v) for v in all_frames.values())
    log(f"  Total unique frames: {total_frames}")
    log(f"  Batch 0 needs: {total_new_b0} new frames")
    log(f"  Batches 1-{len(test_batches)-1}: {total_new_rest} new frames")

    def setup_compressed(res_dir, global_cache):
        """Setup CompressedEmbeddingBag for each large table with shared global cache."""
        caches = {}
        for t in large_tables:
            n_cold = cold_num_rows.get(t, 0)
            if n_cold == 0:
                continue
            frame_dir = os.path.join(res_dir, f'table_{t}')
            meta_path = os.path.join(frame_dir, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                cold_quant_scale[t] = meta['quant_scale']
                cold_quant_zp[t] = meta['quant_zp']

            cache = OnDemandPrefetchCache(
                frame_dir=frame_dir, rows_per_frame=ROWS_PER_FRAME,
                emb_dim=EMB_DIM, num_cold_rows=n_cold,
                width=WIDTH, height=HEIGHT,
                quant_scale=cold_quant_scale[t], quant_zp=cold_quant_zp[t],
                cache_capacity=9999, predictor=None,
                num_prefetch_workers=1, global_cache=global_cache, table_id=t)
            caches[t] = cache

            w = state_dict[emb_keys[t]]
            h_idx = hot_indices[t]
            hot_weight = w[h_idx].clone()
            orig_to_hot = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
            orig_to_hot[h_idx] = torch.arange(len(h_idx))

            comp_emb = CompressedEmbeddingBag(
                hot_weight=hot_weight, is_hot=is_hot[t],
                orig_to_hot=orig_to_hot,
                orig_to_cold_reordered=o2c_map[t],
                cold_cache=cache, num_embeddings=int(ln_emb[t]),
                embedding_dim=EMB_DIM, quantize_hot=True)
            dlrm.emb_l[t] = comp_emb
        return caches

    def restore():
        for t in large_tables:
            dlrm.emb_l[t] = original_embs[t]

    def run_inference(tag):
        """Run inference, return (auc, blats). Timer wraps FULL batch processing."""
        gc.collect()
        torch.set_num_threads(32)
        scores, targets, blats = [], [], []
        with torch.no_grad():
            for X, lS_o, lS_i, T in test_batches:
                t0 = time.perf_counter()
                Z = dlrm(X, lS_o, lS_i)
                blats.append(time.perf_counter() - t0)
                scores.append(Z.detach().numpy().ravel())
                targets.append(T.numpy().ravel())
        auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
        mean_lat = np.mean(blats) * 1000
        p99 = np.percentile(blats, 99) * 1000
        total = sum(blats)
        log(f"  [{tag}] AUC={auc:.6f}, mean={mean_lat:.2f}ms, "
            f"p99={p99:.2f}ms, total={total:.2f}s")
        return auc, blats

    # --- Baseline ---
    log(f"\n{'='*70}")
    log("BASELINE (fp32)")
    log(f"{'='*70}")
    baseline_auc, baseline_blats = run_inference("baseline")
    baseline_lat = np.mean(baseline_blats) * 1000
    baseline_total = sum(baseline_blats)

    all_results = {'baseline': {
        'auc': float(baseline_auc),
        'mean_lat_ms': float(baseline_lat),
        'total_s': float(baseline_total),
    }}

    # --- Test each config ---
    for tag, res_name in H265_CONFIGS:
        res_dir = os.path.join(ONDEMAND_DIR, res_name)
        if not os.path.exists(os.path.join(res_dir, '.done')):
            log(f"\n  SKIP {tag}: not encoded")
            continue

        # === Mode A: Full pre-decode (all frames decoded before inference) ===
        mode_tag = f'{tag}_predecode'
        log(f"\n{'='*70}")
        log(f"{mode_tag}: decode ALL frames, then run inference")
        log(f"{'='*70}")

        restore()
        global_cache = GlobalFrameCache(capacity=9999, store_uint8=True)
        caches = setup_compressed(res_dir, global_cache)

        # Pre-decode all frames
        t_dec = time.perf_counter()
        predecode_frames(global_cache, caches, all_frames)
        decode_ms = (time.perf_counter() - t_dec) * 1000
        log(f"  Pre-decoded {total_frames} frames in {decode_ms:.1f}ms")

        auc, blats = run_inference(mode_tag)
        mean_lat = np.mean(blats) * 1000
        total_s = sum(blats)

        all_results[mode_tag] = {
            'auc': float(auc),
            'auc_loss_pct': float((baseline_auc - auc) * 100),
            'mean_lat_ms': float(mean_lat),
            'p99_lat_ms': float(np.percentile(blats, 99) * 1000),
            'total_s': float(total_s),
            'decode_setup_ms': float(decode_ms),
        }

        # === Mode B: Pipelined (background decode during inference) ===
        mode_tag = f'{tag}_pipelined'
        log(f"\n{'='*70}")
        log(f"{mode_tag}: decode-ahead with LRU cache")
        log(f"{'='*70}")

        restore()
        global_cache = GlobalFrameCache(capacity=9999, store_uint8=True)
        caches = setup_compressed(res_dir, global_cache)
        executor = ThreadPoolExecutor(max_workers=2)

        # Pre-decode batch 0's frames synchronously
        t_dec = time.perf_counter()
        predecode_frames(global_cache, caches, new_per_batch[0])
        warmup_ms = (time.perf_counter() - t_dec) * 1000
        log(f"  Warmup: decoded {total_new_b0} frames in {warmup_ms:.1f}ms")

        # Pipelined inference — HONEST timing: timer wraps the ENTIRE loop body
        gc.collect()
        torch.set_num_threads(32)
        scores, targets, blats = [], [], []
        decode_futures = []

        # Submit decode for batch 1's new frames
        pending = None
        if len(test_batches) > 1 and new_per_batch[1]:
            pending = executor.submit(
                predecode_frames, global_cache, caches, new_per_batch[1])

        with torch.no_grad():
            for bi in range(len(test_batches)):
                t0 = time.perf_counter()  # ← timer starts HERE

                # Wait for any pending background decode
                if pending is not None:
                    pending.result()
                    pending = None

                # Submit decode for batch bi+2
                next_bi = bi + 2
                if next_bi < len(test_batches) and new_per_batch[next_bi]:
                    pending = executor.submit(
                        predecode_frames, global_cache, caches, new_per_batch[next_bi])

                # Run inference (cache.lookup will find frames in global_cache)
                X, lS_o, lS_i, T = test_batches[bi]
                Z = dlrm(X, lS_o, lS_i)

                blats.append(time.perf_counter() - t0)  # ← timer ends HERE

                scores.append(Z.detach().numpy().ravel())
                targets.append(T.numpy().ravel())

        if pending is not None:
            pending.result()
        executor.shutdown(wait=False)

        auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
        mean_lat = np.mean(blats) * 1000
        p99 = np.percentile(blats, 99) * 1000
        total_s = sum(blats)
        max_lat = max(blats) * 1000

        log(f"  [{mode_tag}] AUC={auc:.6f}, mean={mean_lat:.2f}ms, "
            f"p99={p99:.2f}ms, max={max_lat:.2f}ms, total={total_s:.2f}s")

        all_results[mode_tag] = {
            'auc': float(auc),
            'auc_loss_pct': float((baseline_auc - auc) * 100),
            'mean_lat_ms': float(mean_lat),
            'p99_lat_ms': float(p99),
            'max_lat_ms': float(max_lat),
            'total_s': float(total_s),
            'warmup_ms': float(warmup_ms),
        }

    # --- Summary ---
    log(f"\n{'='*70}")
    log("SUMMARY (honest end-to-end timing)")
    log(f"{'='*70}")
    log(f"{'Config':<30s} {'AUC':>9s} {'Loss':>8s} {'Mean':>7s} {'P99':>7s} "
        f"{'Total':>7s} {'vs base':>8s}")
    log("-" * 85)

    for tag, r in all_results.items():
        loss = r.get('auc_loss_pct', 0)
        lat = r['mean_lat_ms']
        p99 = r.get('p99_lat_ms', 0)
        total = r['total_s']
        diff = lat - baseline_lat
        loss_str = f"{loss:+.4f}%" if loss else ""
        p99_str = f"{p99:.2f}ms" if p99 else ""
        log(f"{tag:<30s} {r['auc']:>9.6f} {loss_str:>8s} {lat:>6.2f}ms "
            f"{p99_str:>7s} {total:>6.2f}s {diff:>+7.2f}ms")

    out_file = 'results/h265_pipeline_v2_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
