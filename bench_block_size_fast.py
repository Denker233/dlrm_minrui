#!/usr/bin/env python3
"""
Fast block size sweep: skip AC extraction (DC-only for AUC test).
For each block size: compute DC → reconstruct as DC-only → measure AUC.
"""
import os, sys, time, json, gc
import numpy as np
from scipy.fft import dctn, idctn
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
from codec_ondemand_benchmark import (
    load_model_and_data, EMB_DIM, MODEL_PATH,
    HOTCOLD_DIR, REORDER_DIR, quantize_table, LARGE_TABLE_THRESHOLD,
)

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]


def dc_only_reconstruct_4x4(uint8_rows, step):
    """4×4: 1 row per block. DC = mean of 16 values. Reconstruct = fill with mean."""
    n = uint8_rows.shape[0]
    blocks = uint8_rows.reshape(n, 4, 4).astype(np.float32)
    dct = dctn(blocks, axes=(-2, -1), type=2, norm='ortho')
    dc_quant = np.round(dct[:, 0, 0] / step).astype(np.int16)

    # DC-only reconstruct: each pixel = DC * step * (1/block_size)
    # For 4×4 ortho DCT: C(0) = 1/sqrt(4) = 0.5, so DC weight = 0.5 * 0.5 = 0.25
    dc_weight = 1.0 / 4  # 1/sqrt(4) * 1/sqrt(4) = 1/4
    reconstructed = (dc_quant * step * dc_weight).astype(np.float32)
    # Each row: all 16 dims get the same value
    recon = np.broadcast_to(reconstructed[:, np.newaxis], (n, EMB_DIM)).copy()
    recon = np.clip(np.round(recon), 0, 255).astype(np.uint8)

    n_blocks = n
    unique, counts = np.unique(dc_quant, return_counts=True)
    n_dc_only_est = n  # all DC-only in this mode
    default_dc = unique[counts.argmax()]
    n_default = counts.max()

    return recon, dc_quant, n_blocks, n_dc_only_est, default_dc, n_default


def dc_only_reconstruct_8x8(uint8_rows, step):
    """8×8: 4 rows per block."""
    RPB = 4; TILE = 4; BLOCK = 8
    n = uint8_rows.shape[0]
    n_pad = ((n + RPB - 1) // RPB) * RPB
    if n_pad > n:
        p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8); p[:n] = uint8_rows; uint8_rows = p
    n_blocks = n_pad // RPB
    rows = uint8_rows.reshape(n_blocks, RPB, EMB_DIM).astype(np.float32)
    blocks = np.zeros((n_blocks, BLOCK, BLOCK), dtype=np.float32)
    for r in range(RPB):
        tiles = rows[:, r, :].reshape(n_blocks, TILE, TILE)
        blocks[:, (r // 2) * TILE:(r // 2) * TILE + TILE,
               (r % 2) * TILE:(r % 2) * TILE + TILE] = tiles

    dct = dctn(blocks, axes=(-2, -1), type=2, norm='ortho')
    dc_quant = np.round(dct[:, 0, 0] / step).astype(np.int16)

    # DC weight for 8×8: 1/sqrt(8) * 1/sqrt(8) = 1/8 = 0.125
    dc_weight = 1.0 / BLOCK
    dc_vals = (dc_quant * step * dc_weight).astype(np.float32)
    # Each of 4 rows gets the same value across all dims
    recon = np.zeros((n_pad, EMB_DIM), dtype=np.uint8)
    for i in range(n_blocks):
        v = np.clip(np.round(dc_vals[i]), 0, 255).astype(np.uint8)
        recon[i * RPB:(i + 1) * RPB, :] = v

    unique, counts = np.unique(dc_quant, return_counts=True)
    default_dc = unique[counts.argmax()]
    n_default = counts.max()

    return recon[:n], dc_quant, n_blocks, n_blocks, default_dc, n_default


def dc_only_reconstruct_16x16(uint8_rows, step):
    """16×16: 16 rows per block."""
    RPB = 16; TILE = 4; BLOCK = 16
    n = uint8_rows.shape[0]
    n_pad = ((n + RPB - 1) // RPB) * RPB
    if n_pad > n:
        p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8); p[:n] = uint8_rows; uint8_rows = p
    n_blocks = n_pad // RPB
    rows = uint8_rows.reshape(n_blocks, RPB, EMB_DIM).astype(np.float32)
    blocks = np.zeros((n_blocks, BLOCK, BLOCK), dtype=np.float32)
    for r in range(RPB):
        tiles = rows[:, r, :].reshape(n_blocks, TILE, TILE)
        r_grid = r // 4
        c_grid = r % 4
        blocks[:, r_grid * TILE:(r_grid + 1) * TILE,
               c_grid * TILE:(c_grid + 1) * TILE] = tiles

    dct = dctn(blocks, axes=(-2, -1), type=2, norm='ortho')
    dc_quant = np.round(dct[:, 0, 0] / step).astype(np.int16)

    dc_weight = 1.0 / BLOCK  # 1/16
    dc_vals = (dc_quant * step * dc_weight).astype(np.float32)
    recon = np.zeros((n_pad, EMB_DIM), dtype=np.uint8)
    for i in range(n_blocks):
        v = np.clip(np.round(dc_vals[i]), 0, 255).astype(np.uint8)
        recon[i * RPB:(i + 1) * RPB, :] = v

    unique, counts = np.unique(dc_quant, return_counts=True)
    default_dc = unique[counts.argmax()]
    n_default = counts.max()

    return recon[:n], dc_quant, n_blocks, n_blocks, default_dc, n_default


def main():
    print("=" * 70)
    print("BLOCK SIZE SWEEP (DC-only, fast)")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    nt = len(ln_emb)
    torch.set_num_threads(32)

    # Baseline
    scores, targets = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_base = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"Baseline: {auc_base:.6f}")

    step = 32
    configs = [
        ("4×4 (1 row/block)", 4, 1, dc_only_reconstruct_4x4),
        ("8×8 (4 rows/block)", 8, 4, dc_only_reconstruct_8x8),
        ("16×16 (16 rows/block)", 16, 16, dc_only_reconstruct_16x16),
    ]

    results = []
    for label, block_sz, rpb, recon_fn in configs:
        print(f"\n{'='*60}")
        print(f"{label}, step={step}")
        print(f"{'='*60}")

        total_blocks = 0
        total_dc_bytes = 0
        total_default = 0

        cold_decoded = {}
        for t in TABLES:
            cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
            w = sd[ek[t]][cold_order]
            q, s, zp = quantize_table(w)
            q_np = q.numpy()
            n_cold = len(cold_order)

            t0 = time.perf_counter()
            recon, dc_quant, n_blocks, _, default_dc, n_default = recon_fn(q_np, step)
            t_enc = time.perf_counter() - t0

            # Error
            err = np.abs(recon.astype(np.int16) - q_np[:n_cold].astype(np.int16))

            total_blocks += n_blocks
            total_dc_bytes += n_blocks  # 1 byte per block
            total_default += n_default

            cold_decoded[t] = (recon, s, zp, cold_order)
            print(f"  t{t}: {n_blocks:>8} blocks, default={n_default/n_blocks*100:.0f}%, "
                  f"max_err={err.max():.0f}, mean_err={err.mean():.3f}, enc={t_enc:.1f}s")

        default_pct = total_default / total_blocks * 100
        sparse_kb = (total_blocks - total_default) * 5 / 1024  # exception list
        dc_kb = total_dc_bytes / 1024
        print(f"  Total: {total_blocks} blocks, DC={dc_kb:.0f}KB, "
              f"default={default_pct:.1f}%, sparse≈{sparse_kb:.0f}KB")

        # Build full embedding table with decoded cold rows
        with torch.no_grad():
            for k in ek:
                t_idx = int(k.split('.')[1])
                dlrm.emb_l[t_idx] = nn.EmbeddingBag(int(ln_emb[t_idx]), EMB_DIM,
                                                      mode='sum', sparse=True)
                w = sd[k].clone()
                if t_idx in cold_decoded:
                    recon, s, zp, cold_order = cold_decoded[t_idx]
                    decoded_fp32 = (torch.from_numpy(recon).float() - zp) * s
                    n = min(len(cold_order), decoded_fp32.shape[0])
                    w[cold_order[:n]] = decoded_fp32[:n]
                dlrm.emb_l[t_idx].weight.data = w

        # Warmup
        with torch.no_grad():
            for X, o, i, T in test_batches[:10]: dlrm(X, o, i)

        # AUC + speed
        scores, targets, batch_times = [], [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                t0 = time.perf_counter()
                Z = dlrm(X, o, i)
                batch_times.append(time.perf_counter() - t0)
                scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())

        auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
        batch_ms = np.median(batch_times) * 1000

        hot_mb = 22.1; small_mb = 2.9; bitmap_mb = 6.0
        total_mb = hot_mb + small_mb + bitmap_mb + dc_kb / 1024
        ratio = 2060.7 / total_mb

        print(f"\n  AUC:      {auc:.6f} (loss: {(auc_base - auc) * 100:+.4f}%)")
        print(f"  Batch:    {batch_ms:.2f}ms")
        print(f"  Cold DC:  {dc_kb:.0f}KB (sparse≈{sparse_kb:.0f}KB)")
        print(f"  Total:    {total_mb:.1f}MB ({ratio:.0f}x)")

        results.append({
            'label': label, 'block_size': block_sz, 'rows_per_block': rpb,
            'auc': float(auc), 'auc_loss_pct': float((auc_base - auc) * 100),
            'batch_ms': float(batch_ms),
            'n_blocks': total_blocks, 'dc_kb': float(dc_kb),
            'default_pct': float(default_pct), 'sparse_kb': float(sparse_kb),
            'total_mb': float(total_mb), 'ratio': float(ratio),
        })
        del cold_decoded; gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Config':<25} {'AUC':>10} {'Loss':>10} {'Batch':>8} {'Blocks':>10} "
          f"{'DC KB':>7} {'Sparse':>7} {'Total':>8} {'Ratio':>6}")
    print("-" * 95)
    for r in results:
        print(f"{r['label']:<25} {r['auc']:>10.6f} {r['auc_loss_pct']:>+9.4f}% "
              f"{r['batch_ms']:>7.2f} {r['n_blocks']:>10,} "
              f"{r['dc_kb']:>6.0f} {r['sparse_kb']:>6.0f} {r['total_mb']:>7.1f}MB {r['ratio']:>5.0f}x")


if __name__ == '__main__':
    main()
