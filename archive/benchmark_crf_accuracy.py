#!/usr/bin/env python3
"""
CRF Accuracy Sweep: Measure AUC impact of lossy H.265 encoding at different CRF values.

For each CRF value (0=lossless, 10, 18, 23, 28):
1. Encode cold embeddings as H.265 frames
2. Decode them back
3. Measure quantization error (MSE, max error)
4. Run full inference with the lossy-decoded embeddings
5. Report AUC, log-loss, accuracy

Also measures Zstd at same granularity for comparison (lossless only).
"""

import os, sys, time, json, gc
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '.')

RESULTS_DIR = "results"
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "codec_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
TILE_H, TILE_W = 4, 4
FRAME_W, FRAME_H = 1920, 1080
ROWS_PER_FRAME = (FRAME_W // TILE_W) * (FRAME_H // TILE_H)
EMB_DIM = 16
TEST_BATCH_SIZE = 2048
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tile_to_frame(rows_uint8):
    n_rows = rows_uint8.shape[0]
    if n_rows < ROWS_PER_FRAME:
        padded = np.zeros((ROWS_PER_FRAME, 16), dtype=np.uint8)
        padded[:n_rows] = rows_uint8
        rows_uint8 = padded
    tiles_w = FRAME_W // TILE_W
    tiles_h = FRAME_H // TILE_H
    frame = rows_uint8.reshape(tiles_h, tiles_w, TILE_H, TILE_W)
    frame = frame.transpose(0, 2, 1, 3).reshape(FRAME_H, FRAME_W)
    return frame


def untile_frame(frame, n_rows):
    tiles_w = FRAME_W // TILE_W
    tiles_h = FRAME_H // TILE_H
    rows = frame.reshape(tiles_h, TILE_H, tiles_w, TILE_W)
    rows = rows.transpose(0, 2, 1, 3).reshape(-1, 16)
    return rows[:n_rows]


def main():
    import compressed_emb as _C
    from benchmark_full_comparison import load_model_and_data, compute_metrics

    # ---- Load model ----
    dlrm, test_ld, ln_emb, state_dict = load_model_and_data()

    # ---- Load hot/cold split ----
    log("Loading hot/cold split...")
    cold_indices = {}
    hot_indices = {}
    cold_quant = {}
    cold_scale = {}
    cold_zp = {}
    emb_keys = {}

    for t_idx in LARGE_TABLES:
        cold_idx_path = os.path.join(HOTCOLD_DIR, f"cold_indices_{t_idx}.pt")
        hot_idx_path = os.path.join(HOTCOLD_DIR, f"hot_indices_{t_idx}.pt")
        if not os.path.exists(cold_idx_path) or not os.path.exists(hot_idx_path):
            continue

        cold_idx = torch.load(cold_idx_path, weights_only=False)
        hot_idx = torch.load(hot_idx_path, weights_only=False)
        if cold_idx.numel() == 0:
            continue

        key = f'emb_l.{t_idx}.weight'
        weight = state_dict[key]
        cold_weight = weight[cold_idx]

        mn, mx = cold_weight.min().item(), cold_weight.max().item()
        s = (mx - mn) / 255.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        quant = ((cold_weight / s).round() + zp).clamp(0, 255).to(torch.uint8).numpy()

        # Apply reordering
        order_path = os.path.join(REORDER_DIR, f"cold_order_{t_idx}.npy")
        if os.path.exists(order_path):
            order = np.load(order_path)
            quant = quant[order]
            cold_idx = cold_idx[torch.from_numpy(order.astype(np.int64))]

        cold_indices[t_idx] = cold_idx
        hot_indices[t_idx] = hot_idx
        cold_quant[t_idx] = quant
        cold_scale[t_idx] = s
        cold_zp[t_idx] = zp
        emb_keys[t_idx] = key
        log(f"  Table {t_idx}: {len(cold_idx):,} cold rows, scale={s:.6f}, zp={zp}")

    # ---- Baseline AUC (fp32, unmodified) ----
    log(f"\n{'='*60}")
    log("BASELINE (fp32)")
    log(f"{'='*60}")

    max_samples = 2000 * TEST_BATCH_SIZE
    all_scores = np.empty(max_samples, dtype=np.float32)
    all_targets = np.empty(max_samples, dtype=np.float32)

    def run_inference_auc(dlrm):
        sample_idx = 0
        with torch.no_grad():
            for inputBatch in test_ld:
                X, lS_o, lS_i, T = inputBatch[0], inputBatch[1], inputBatch[2], inputBatch[3]
                Z = dlrm(X, lS_o, lS_i)
                z_np = Z.detach().cpu().numpy().ravel()
                t_np = T.detach().cpu().numpy().ravel()
                bs = z_np.shape[0]
                all_scores[sample_idx:sample_idx+bs] = z_np
                all_targets[sample_idx:sample_idx+bs] = t_np
                sample_idx += bs
        scores, targets = all_scores[:sample_idx], all_targets[:sample_idx]
        return compute_metrics(scores, targets)

    baseline_auc, baseline_ll, baseline_acc = run_inference_auc(dlrm)
    log(f"  Baseline: AUC={baseline_auc:.6f}, LogLoss={baseline_ll:.6f}, Acc={baseline_acc:.4f}")

    # ---- Lossless quantization AUC (uint8 quantize + dequantize, no codec) ----
    log(f"\n{'='*60}")
    log("LOSSLESS QUANTIZATION (uint8 round-trip, no codec)")
    log(f"{'='*60}")

    for t_idx in cold_quant:
        q = cold_quant[t_idx]
        s, zp = cold_scale[t_idx], cold_zp[t_idx]
        dequant = (q.astype(np.float32) - zp) * s
        dequant_t = torch.from_numpy(dequant)
        # Write back to model
        c_idx = cold_indices[t_idx]
        dlrm.emb_l[t_idx].weight.data[c_idx] = dequant_t

    quant_auc, quant_ll, quant_acc = run_inference_auc(dlrm)
    log(f"  Quantized: AUC={quant_auc:.6f} (delta={quant_auc - baseline_auc:+.6f}), "
        f"LogLoss={quant_ll:.6f}, Acc={quant_acc:.4f}")

    # Restore
    for t_idx, k in emb_keys.items():
        dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

    # ---- CRF sweep ----
    crf_values = [0, 10, 18, 23, 28]
    results = {
        'baseline': {'auc': baseline_auc, 'log_loss': baseline_ll, 'accuracy': baseline_acc},
        'quantized_uint8': {'auc': quant_auc, 'log_loss': quant_ll, 'accuracy': quant_acc},
    }

    for crf in crf_values:
        log(f"\n{'='*60}")
        log(f"CRF = {crf} ({'lossless' if crf == 0 else 'lossy'})")
        log(f"{'='*60}")

        total_mse = 0
        total_max_err = 0
        total_rows = 0
        total_raw_bytes = 0
        total_compressed_bytes = 0
        encode_times = []
        decode_times = []

        for t_idx in cold_quant:
            q = cold_quant[t_idx]
            s, zp = cold_scale[t_idx], cold_zp[t_idx]
            n_rows = q.shape[0]
            n_frames = (n_rows + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME

            reconstructed = np.empty_like(q)

            for i in range(n_frames):
                start = i * ROWS_PER_FRAME
                end = min(start + ROWS_PER_FRAME, n_rows)
                chunk = q[start:end]
                frame = tile_to_frame(chunk)
                frame_t = torch.from_numpy(frame)

                total_raw_bytes += len(chunk) * 16

                # Encode
                tmp_path = f"/tmp/_crf_sweep_{t_idx}_{i}.h265"
                t0 = time.perf_counter()
                _C.encode_h265_frame(frame_t, tmp_path, crf == 0, crf)
                t1 = time.perf_counter()
                encode_times.append((t1 - t0) * 1000)
                total_compressed_bytes += os.path.getsize(tmp_path)

                # Decode
                t0 = time.perf_counter()
                decoded_frame = _C.decode_h265_frame_from_file(tmp_path)
                t1 = time.perf_counter()
                decode_times.append((t1 - t0) * 1000)

                decoded = untile_frame(decoded_frame.numpy(), end - start)
                reconstructed[start:end] = decoded
                os.remove(tmp_path)

            # Compute error in uint8 domain
            diff = reconstructed.astype(np.float32) - q.astype(np.float32)
            mse = np.mean(diff ** 2)
            max_err = np.max(np.abs(diff))
            total_mse += mse * n_rows
            total_max_err = max(total_max_err, max_err)
            total_rows += n_rows

            # Dequantize and write to model
            dequant = (reconstructed.astype(np.float32) - zp) * s
            dequant_t = torch.from_numpy(dequant)
            c_idx = cold_indices[t_idx]
            dlrm.emb_l[t_idx].weight.data[c_idx] = dequant_t

        avg_mse = total_mse / total_rows if total_rows > 0 else 0
        ratio = total_raw_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0

        log(f"  Compression: {total_compressed_bytes/1e6:.1f}MB, ratio={ratio:.1f}x vs uint8")
        log(f"  Error (uint8): MSE={avg_mse:.4f}, MaxErr={total_max_err:.0f}/255")
        log(f"  Error (fp32): MSE~{avg_mse * cold_scale[2]**2:.10f}")
        log(f"  Encode: avg={np.mean(encode_times):.1f}ms/frame")
        log(f"  Decode: avg={np.mean(decode_times):.3f}ms/frame")

        # Measure AUC
        crf_auc, crf_ll, crf_acc = run_inference_auc(dlrm)
        auc_delta = crf_auc - baseline_auc

        log(f"  AUC={crf_auc:.6f} (delta={auc_delta:+.6f}), "
            f"LogLoss={crf_ll:.6f}, Acc={crf_acc:.4f}")

        results[f'crf_{crf}'] = {
            'crf': crf,
            'auc': crf_auc,
            'log_loss': crf_ll,
            'accuracy': crf_acc,
            'auc_delta': auc_delta,
            'mse_uint8': float(avg_mse),
            'max_err_uint8': float(total_max_err),
            'compression_ratio': ratio,
            'compressed_mb': total_compressed_bytes / 1e6,
            'avg_encode_ms': np.mean(encode_times),
            'avg_decode_ms': np.mean(decode_times),
        }

        # Restore weights for next iteration
        for t_idx, k in emb_keys.items():
            dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

    # ---- Summary ----
    log(f"\n{'='*60}")
    log("CRF ACCURACY SWEEP SUMMARY")
    log(f"{'='*60}\n")

    log(f"  {'Config':<20s} | {'AUC':>10s} | {'Delta':>10s} | {'Ratio':>6s} | "
        f"{'Size':>7s} | {'Decode':>8s} | {'MaxErr':>7s}")
    log(f"  {'-'*80}")

    log(f"  {'Baseline (fp32)':<20s} | {baseline_auc:>10.6f} | {'---':>10s} | {'1.0x':>6s} | "
        f"{'2061MB':>7s} | {'N/A':>8s} | {'0':>7s}")
    log(f"  {'Quantized uint8':<20s} | {quant_auc:>10.6f} | {quant_auc-baseline_auc:>+10.6f} | "
        f"{'4.0x':>6s} | {'516MB':>7s} | {'N/A':>8s} | {'0':>7s}")

    for crf in crf_values:
        r = results[f'crf_{crf}']
        name = f"CRF={crf}" if crf > 0 else "CRF=0 (lossless)"
        log(f"  {name:<20s} | {r['auc']:>10.6f} | {r['auc_delta']:>+10.6f} | "
            f"{r['compression_ratio']:>5.1f}x | {r['compressed_mb']:>5.1f}MB | "
            f"{r['avg_decode_ms']:>6.2f}ms | {r['max_err_uint8']:>5.0f}/255")

    # Save
    json_path = os.path.join(OUTPUT_DIR, 'crf_accuracy_sweep.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
