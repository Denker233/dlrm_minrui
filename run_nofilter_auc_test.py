#!/usr/bin/env python3
"""
Controlled AUC test: isolate the effect of removing deblock/SAO filters.

Runs full inference (all 1599 test batches) for each config and compares AUC,
compression ratio, and decode speed against the existing CRF=18 ultrafast baseline.

Configs tested:
  - CRF=18 ultrafast no-filter  (isolates filter removal only)
  - CRF=18 medium no-filter     (isolates preset change + filter removal)
  - CRF=30 medium no-filter     (already have this from prior run)
"""
import os, sys, time, json, gc
import numpy as np

os.environ.setdefault('CRITEO_DAYS', '4')

import torch
import torch.nn as nn
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')

torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if torch_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import compressed_emb as _C

from codec_ondemand_benchmark import (
    EMB_DIM, HOTCOLD_DIR, REORDER_DIR, ONDEMAND_DIR, LARGE_TABLE_THRESHOLD,
    encode_h265_perframe, quantize_table, log,
    load_model_and_data,
    OnDemandPrefetchCache, GlobalFrameCache, CompressedEmbeddingBag,
)
from sklearn.metrics import roc_auc_score

# Configs: (res_name, crf, preset, extra_x265)
CONFIGS = [
    ('1080p_crf18_uf_nofilter',  18, 'ultrafast', 'no-deblock=1:no-sao=1'),
    ('1080p_crf18_med_nofilter', 18, 'medium',    'no-deblock=1:no-sao=1'),
]

WIDTH, HEIGHT = 1920, 1080
ROWS_PER_FRAME = (WIDTH * HEIGHT) // EMB_DIM


def run_inference(dlrm, test_ld, label):
    """Run full inference on all test batches."""
    all_targets, all_scores, batch_lats = [], [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_ld:
            t0 = time.perf_counter()
            Z = dlrm(X, lS_o, lS_i)
            batch_lats.append(time.perf_counter() - t0)
            all_scores.append(Z.detach().numpy().flatten())
            all_targets.append(T.numpy().flatten())
    auc = roc_auc_score(np.concatenate(all_targets), np.concatenate(all_scores))
    mean_lat = np.mean(batch_lats) * 1000
    print(f"  {label}: AUC={auc:.6f}, mean_lat={mean_lat:.2f}ms, {len(batch_lats)} batches")
    return auc, mean_lat, len(batch_lats)


def setup_compressed_model(dlrm, state_dict, emb_keys, ln_emb, large_tables,
                           is_hot, hot_indices, orig_to_cold_reordered,
                           cold_num_rows, cold_quant_scale, cold_quant_zp,
                           res_dir):
    """Replace large tables with CompressedEmbeddingBag, return total compressed bytes."""
    global_cache = GlobalFrameCache(capacity=9999, store_uint8=True)
    total_compressed_bytes = 0

    for t in large_tables:
        n_cold = cold_num_rows[t]
        if n_cold == 0:
            continue
        frame_dir = os.path.join(res_dir, f'table_{t}')
        frame_files = sorted([f for f in os.listdir(frame_dir)
                              if f.startswith('frame_') and
                              (f.endswith('.h265') or f.endswith('.mkv'))])
        total_compressed_bytes += sum(
            os.path.getsize(os.path.join(frame_dir, ff)) for ff in frame_files)

        cache = OnDemandPrefetchCache(
            frame_dir=frame_dir, rows_per_frame=ROWS_PER_FRAME,
            emb_dim=EMB_DIM, num_cold_rows=n_cold,
            width=WIDTH, height=HEIGHT,
            quant_scale=cold_quant_scale[t], quant_zp=cold_quant_zp[t],
            cache_capacity=9999, predictor=None,
            num_prefetch_workers=1, global_cache=global_cache, table_id=t)

        w = state_dict[emb_keys[t]]
        h_idx = hot_indices[t]
        hot_weight = w[h_idx].clone()

        orig_to_hot = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
        orig_to_hot[h_idx] = torch.arange(len(h_idx))

        comp_emb = CompressedEmbeddingBag(
            hot_weight=hot_weight,
            is_hot=is_hot[t],
            orig_to_hot=orig_to_hot,
            orig_to_cold_reordered=orig_to_cold_reordered[t],
            cold_cache=cache,
            num_embeddings=int(ln_emb[t]),
            embedding_dim=EMB_DIM,
            quantize_hot=True,
        )
        dlrm.emb_l[t] = comp_emb

    return total_compressed_bytes


def restore_original_embs(dlrm, original_embs):
    """Restore original EmbeddingBag modules."""
    for t, emb in original_embs.items():
        dlrm.emb_l[t] = emb


def main():
    print("=" * 70)
    print("Controlled No-Filter AUC Test")
    print("=" * 70)

    # --- Load model and data ---
    print("\n[1] Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = {k: v.clone() for k, v in dlrm.state_dict().items()}
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    large_tables = [t for t in range(len(emb_keys)) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]

    # Save original embeddings for restoration between configs
    original_embs = {t: dlrm.emb_l[t] for t in large_tables}

    # --- Load precomputed hot/cold data ---
    print("\n[2] Loading precomputed hot/cold/reorder data...")
    is_hot = {}
    hot_indices = {}
    orig_to_cold_reordered = {}
    cold_num_rows = {}

    for t in large_tables:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]
        fp = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        orig_to_cold_reordered[t] = torch.load(fp, map_location='cpu', weights_only=True)
        with open(os.path.join(REORDER_DIR, f'num_cold_{t}.txt')) as f:
            cold_num_rows[t] = int(f.read().strip())

    # --- Baseline ---
    print("\n[3] Baseline inference (fp32, all test batches)...")
    torch.set_num_threads(32)
    baseline_auc, baseline_lat, n_batches = run_inference(dlrm, test_ld, "Baseline")

    # --- Run each config ---
    all_results = {
        'baseline': {'auc': float(baseline_auc), 'mean_lat_ms': float(baseline_lat)},
    }

    for config_idx, (res_name, crf, preset, extra_x265) in enumerate(CONFIGS):
        print(f"\n{'='*70}")
        print(f"CONFIG {config_idx+1}/{len(CONFIGS)}: {res_name}")
        print(f"  CRF={crf}, preset={preset}, x265={extra_x265}")
        print(f"{'='*70}")

        # Restore original embeddings
        restore_original_embs(dlrm, original_embs)

        # Encode
        res_dir = os.path.join(ONDEMAND_DIR, res_name)
        done_marker = os.path.join(res_dir, '.done')
        cold_quant_scale = {}
        cold_quant_zp = {}

        if os.path.exists(done_marker):
            print(f"  Encoding SKIPPED — already done")
            for t in large_tables:
                meta_path = os.path.join(res_dir, f'table_{t}', 'meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    cold_quant_scale[t] = meta['quant_scale']
                    cold_quant_zp[t] = meta['quant_zp']
        else:
            print(f"  Encoding...")
            os.makedirs(res_dir, exist_ok=True)
            total_comp = 0
            total_raw = 0

            for t in large_tables:
                n_cold = cold_num_rows[t]
                if n_cold == 0:
                    continue
                cold_order = np.load(os.path.join(REORDER_DIR, f'cold_order_{t}.npy'))
                reordered_w = state_dict[emb_keys[t]][cold_order]
                q, s, zp = quantize_table(reordered_w)
                cold_quant_scale[t] = s
                cold_quant_zp[t] = zp

                nf, fd, cb, et, rpf = encode_h265_perframe(
                    q.numpy(), WIDTH, HEIGHT, crf=crf,
                    output_dir=res_dir, table_id=t,
                    preset=preset, extra_x265_params=extra_x265)

                meta = {'num_frames': nf, 'rows_per_frame': rpf,
                        'width': WIDTH, 'height': HEIGHT,
                        'n_cold': n_cold, 'compressed_bytes': cb,
                        'raw_bytes': n_cold * EMB_DIM, 'quant_scale': s, 'quant_zp': zp}
                with open(os.path.join(fd, 'meta.json'), 'w') as f:
                    json.dump(meta, f)

                total_comp += cb
                total_raw += n_cold * EMB_DIM
                del reordered_w, q; gc.collect()

            with open(done_marker, 'w') as f:
                f.write(f'CRF={crf} preset={preset} extra={extra_x265}\n')
            print(f"  Encoded: {total_raw/1024/1024:.1f}MB -> {total_comp/1024:.1f}KB "
                  f"({total_raw/total_comp:.0f}x)")

        # Setup compressed model
        print(f"  Setting up compressed model...")
        total_compressed_bytes = setup_compressed_model(
            dlrm, state_dict, emb_keys, ln_emb, large_tables,
            is_hot, hot_indices, orig_to_cold_reordered,
            cold_num_rows, cold_quant_scale, cold_quant_zp, res_dir)

        # Warmup
        with torch.no_grad():
            for j, (X, lS_o, lS_i, T) in enumerate(test_ld):
                if j >= 1: break
                dlrm(X, lS_o, lS_i)

        # Full inference
        print(f"  Running inference ({n_batches} batches)...")
        comp_auc, comp_lat, _ = run_inference(dlrm, test_ld, res_name)

        # Ratios
        total_cold_raw = sum(cold_num_rows[t] * EMB_DIM for t in large_tables
                             if cold_num_rows[t] > 0)
        total_fp32 = sum(state_dict[emb_keys[t]].numel() * 4 for t in large_tables)
        cold_ratio = total_cold_raw / total_compressed_bytes if total_compressed_bytes > 0 else 0
        fp32_ratio = total_fp32 / total_compressed_bytes if total_compressed_bytes > 0 else 0
        auc_loss = (baseline_auc - comp_auc) * 100

        all_results[res_name] = {
            'crf': crf, 'preset': preset, 'extra_x265': extra_x265,
            'auc': float(comp_auc),
            'auc_loss_pct': float(auc_loss),
            'mean_lat_ms': float(comp_lat),
            'cold_compressed_bytes': int(total_compressed_bytes),
            'cold_ratio': float(cold_ratio),
            'fp32_ratio': float(fp32_ratio),
        }
        print(f"  AUC loss: {auc_loss:+.4f}%, storage: {fp32_ratio:.0f}x vs fp32")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("CONTROLLED COMPARISON")
    print(f"{'='*70}")
    print(f"{'Config':<35s} {'CRF':>4s} {'Preset':<10s} {'Filter':>8s} "
          f"{'AUC':>9s} {'Loss':>8s} {'Ratio':>7s}")
    print("-" * 90)
    print(f"{'Baseline (fp32)':<35s} {'':>4s} {'':>10s} {'':>8s} "
          f"{baseline_auc:>9.6f} {'':>8s} {'1x':>7s}")

    # Load existing CRF=18 results
    existing = "results/ondemand_results.json"
    if os.path.exists(existing):
        with open(existing) as f:
            old = json.load(f)
        if '1080p_crf18_bitmap_fullcpp' in old:
            r = old['1080p_crf18_bitmap_fullcpp']
            loss = (baseline_auc - r['auc']) * 100
            print(f"{'CRF=18 ultrafast (existing)':<35s} {'18':>4s} {'ultrafast':<10s} {'yes':>8s} "
                  f"{r['auc']:>9.6f} {loss:>+7.4f}% {'1360x':>7s}")

    for res_name, crf, preset, extra in CONFIGS:
        if res_name in all_results:
            r = all_results[res_name]
            print(f"{res_name:<35s} {crf:>4d} {preset:<10s} {'no':>8s} "
                  f"{r['auc']:>9.6f} {r['auc_loss_pct']:>+7.4f}% {r['fp32_ratio']:>6.0f}x")

    # Also show the CRF=30 no-filter result if available
    nofilter30 = 'results/nofilter_auc_results.json'
    if os.path.exists(nofilter30):
        with open(nofilter30) as f:
            r30 = json.load(f)
        print(f"{'CRF=30 medium no-filter':<35s} {'30':>4s} {'medium':<10s} {'no':>8s} "
              f"{r30['compressed_auc']:>9.6f} {r30['auc_loss_pct']:>+7.4f}% "
              f"{r30['fp32_ratio']:>6.0f}x")

    print("-" * 90)

    # Save all results
    out_file = 'results/nofilter_controlled_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
