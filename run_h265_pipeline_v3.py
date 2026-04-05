#!/usr/bin/env python3
"""
Pipelined H.265 decode v3 — fully C++ pipeline with dynamic frame cache.

Uses pipeline_forward() which:
  1. Waits for background decode of current batch's frames (started during prev batch)
  2. Launches background decode of NEXT batch's missing frames
  3. Runs fast_forward on current batch (cold_frame_accum uses dynamic per-frame pointers)

No Python in the hot path. No re-registration. O(1) frame add/remove.
Honest wall-clock timing wraps the entire pipeline_forward call.
"""
import os, sys, time, json, gc
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C
from codec_ondemand_benchmark import (
    log, load_model_and_data, EMB_DIM,
    REORDER_DIR, HOTCOLD_DIR, ONDEMAND_DIR, LARGE_TABLE_THRESHOLD,
    CompressedEmbeddingBag,
)
from sklearn.metrics import roc_auc_score

WIDTH, HEIGHT = 1920, 1080
ROWS_PER_FRAME = (WIDTH * HEIGHT) // EMB_DIM

H265_CONFIGS = [
    ('crf18_filter', '1080p_crf18'),
    ('crf30_nofilter', '1080p_crf30_nofilter'),
]


def main():
    log("=" * 70)
    log("PIPELINED H.265 v3 — Full C++ Pipeline")
    log("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = dlrm.state_dict()
    emb_keys = {}
    for k in state_dict:
        if k.startswith('emb_l.') and k.endswith('.weight'):
            emb_keys[int(k.split('.')[1])] = k

    num_tabs = len(ln_emb)
    large_tables = [i for i in range(num_tabs) if ln_emb[i] > LARGE_TABLE_THRESHOLD]

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

    test_batches = list(test_ld)
    log(f"Pre-cached {len(test_batches)} test batches")

    original_embs = {t: dlrm.emb_l[t] for t in large_tables}

    def restore():
        for t in large_tables:
            dlrm.emb_l[t] = original_embs[t]

    # --- Baseline ---
    log(f"\n{'='*70}")
    log("BASELINE (fp32)")
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
    baseline_total = sum(blats)
    log(f"  AUC={baseline_auc:.6f}, mean={baseline_lat:.2f}ms, total={baseline_total:.2f}s")

    all_results = {'baseline': {
        'auc': float(baseline_auc), 'mean_lat_ms': float(baseline_lat),
        'total_s': float(baseline_total),
    }}

    # --- Test each config ---
    for tag, res_name in H265_CONFIGS:
        res_dir = os.path.join(ONDEMAND_DIR, res_name)
        if not os.path.exists(os.path.join(res_dir, '.done')):
            log(f"\n  SKIP {tag}: not encoded"); continue

        log(f"\n{'='*70}")
        log(f"{tag} — C++ pipeline_forward")
        log(f"{'='*70}")

        restore()

        # Load quant params
        for t in large_tables:
            meta_path = os.path.join(res_dir, f'table_{t}', 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                cold_quant_scale[t] = meta['quant_scale']
                cold_quant_zp[t] = meta['quant_zp']

        # Setup CompressedEmbeddingBag for hot lookups + mapping
        for t in large_tables:
            w = state_dict[emb_keys[t]]
            h_idx = hot_indices[t]
            hot_weight = w[h_idx].clone()
            orig_to_hot = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
            orig_to_hot[h_idx] = torch.arange(len(h_idx))

            comp_emb = CompressedEmbeddingBag(
                hot_weight=hot_weight, is_hot=is_hot[t],
                orig_to_hot=orig_to_hot,
                orig_to_cold_reordered=o2c_map[t],
                cold_cache=None,  # not needed — pipeline handles cold
                num_embeddings=int(ln_emb[t]),
                embedding_dim=EMB_DIM, quantize_hot=True)
            dlrm.emb_l[t] = comp_emb

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

        # Initialize pipeline
        frame_dirs = [os.path.join(res_dir, f'table_{t}') for t in large_tables]
        rpfs = [ROWS_PER_FRAME] * len(large_tables)
        c_scales = [float(cold_quant_scale[t]) for t in large_tables]
        c_zps = [float(cold_quant_zp[t]) for t in large_tables]
        n_colds = [int(cold_num_rows[t]) for t in large_tables]
        is_hot_list = [is_hot[t] for t in large_tables]
        o2c_list = [o2c_map[t].int() for t in large_tables]

        _C.pipeline_init(large_tables, frame_dirs, rpfs, c_scales, c_zps,
                        n_colds, is_hot_list, o2c_list, 9999)

        # Prepare lS_i/lS_o tensors for pipeline
        def make_2d(lS_i, lS_o):
            lS_i_2d = torch.stack(lS_i) if isinstance(lS_i, (list, tuple)) else (
                lS_i if lS_i.dim() == 2 else lS_i.view(num_tabs, -1))
            lS_o_2d = torch.stack(lS_o) if isinstance(lS_o, (list, tuple)) else (
                lS_o if lS_o.dim() == 2 else lS_o.view(num_tabs, -1))
            return lS_i_2d, lS_o_2d

        # No explicit warmup — pipeline_forward handles batch 0's missing frames
        # automatically (synchronous decode on first call, then prefetch for subsequent)

        # Override apply_emb to use pipeline_forward
        def _pipeline_apply_emb(lS_o, lS_i, emb_l, v_W_l):
            nonlocal _next_lS_i_2d
            lS_i_2d, lS_o_2d = make_2d(lS_i, lS_o)
            next_t = _next_lS_i_2d if _next_lS_i_2d is not None else torch.empty(0)
            results = _C.pipeline_forward(lS_i_2d, lS_o_2d, next_t)
            return results[-1]

        orig_apply = dlrm.apply_emb
        dlrm.apply_emb = _pipeline_apply_emb

        # Run inference
        gc.collect()
        torch.set_num_threads(32)
        scores, targets, blats = [], [], []
        _next_lS_i_2d = None

        n = len(test_batches)
        with torch.no_grad():
            for bi in range(n):
                X, lS_o, lS_i, T = test_batches[bi]

                # Set next batch's indices for prefetch
                if bi + 1 < n:
                    next_lS_i = test_batches[bi + 1][2]
                    _next_lS_i_2d, _ = make_2d(next_lS_i, test_batches[bi + 1][1])
                else:
                    _next_lS_i_2d = None

                t0 = time.perf_counter()
                Z = dlrm(X, lS_o, lS_i)
                blats.append(time.perf_counter() - t0)

                scores.append(Z.detach().numpy().ravel())
                targets.append(T.numpy().ravel())

        auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
        mean_lat = np.mean(blats) * 1000
        p99 = np.percentile(blats, 99) * 1000
        max_lat = max(blats) * 1000
        total_s = sum(blats)

        log(f"  AUC={auc:.6f}, mean={mean_lat:.2f}ms, p99={p99:.2f}ms, "
            f"max={max_lat:.2f}ms, total={total_s:.2f}s")
        log(f"  AUC loss: {(baseline_auc - auc)*100:+.4f}%")
        log(f"  Latency vs baseline: {mean_lat - baseline_lat:+.2f}ms")

        all_results[tag] = {
            'auc': float(auc),
            'auc_loss_pct': float((baseline_auc - auc) * 100),
            'mean_lat_ms': float(mean_lat),
            'p99_lat_ms': float(p99),
            'max_lat_ms': float(max_lat),
            'total_s': float(total_s),
            'batch0_lat_ms': float(blats[0] * 1000),
        }

        log(f"  Batch 0 latency: {blats[0]*1000:.2f}ms (includes decode of missing frames)")

        dlrm.apply_emb = orig_apply
        restore()

    # Summary
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")
    log(f"{'Config':<25s} {'AUC':>9s} {'Loss':>8s} {'Mean':>7s} {'P99':>7s} "
        f"{'Max':>7s} {'Total':>7s} {'vs base':>8s}")
    log("-" * 85)
    for tag, r in all_results.items():
        loss = r.get('auc_loss_pct', 0)
        lat = r['mean_lat_ms']
        p99 = r.get('p99_lat_ms', 0)
        mx = r.get('max_lat_ms', 0)
        total = r['total_s']
        diff = lat - baseline_lat
        loss_s = f"{loss:+.4f}%" if loss else ""
        log(f"{tag:<25s} {r['auc']:>9.6f} {loss_s:>8s} {lat:>6.2f}ms {p99:>6.2f}ms "
            f"{mx:>6.2f}ms {total:>6.2f}s {diff:>+7.2f}ms")

    out_file = 'results/h265_pipeline_v3_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
