#!/usr/bin/env python3
"""
Block size sweep: 4×4 (1 row/block), 8×8 (4 rows/block), 16×16 (16 rows/block).
For each: measure AUC, latency, memory.

For 4×4: DC = mean of 16 values in one row. No tiling needed.
For 8×8: current approach (4 rows tiled into 8×8).
For 16×16: 16 rows tiled into 16×16.
"""
import os, sys, time, json, gc
import numpy as np
from scipy.fft import dctn
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
import compressed_emb as _C
from codec_ondemand_benchmark import (
    load_model_and_data, CompressedEmbeddingBag, EMB_DIM, MODEL_PATH,
    HOTCOLD_DIR, REORDER_DIR, quantize_table, LARGE_TABLE_THRESHOLD,
)

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]


def compute_dc_4x4(uint8_rows, step):
    """4×4 block = 1 row per block. DC = mean of 16 values."""
    n = uint8_rows.shape[0]
    # Each row (16 values) → reshape to 4×4 → DCT → DC coefficient
    blocks = uint8_rows.reshape(n, 4, 4).astype(np.float32)
    dct = dctn(blocks, axes=(-2, -1), type=2, norm='ortho')
    dc_quant = np.round(dct[:, 0, 0] / step).astype(np.int16)

    # AC coefficients
    ac_data, ac_pos, ac_off = [], [], [0]
    quantized = np.round(dct / step).astype(np.int16)
    for bi in range(n):
        q = quantized[bi]
        for u in range(4):
            for v in range(4):
                if (u, v) == (0, 0): continue
                if q[u, v] != 0:
                    ac_pos.append(u * 4 + v)
                    ac_data.append(q[u, v])
        ac_off.append(len(ac_data))

    return dc_quant, np.array(ac_data, dtype=np.int16), np.array(ac_pos, dtype=np.uint8), \
           np.array(ac_off, dtype=np.int64), n


def compute_dc_8x8(uint8_rows, step):
    """8×8 block = 4 rows per block. Current approach."""
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

    quantized = np.round(dct / step).astype(np.int16)
    ac_data, ac_pos, ac_off = [], [], [0]
    for bi in range(n_blocks):
        q = quantized[bi]
        for u in range(8):
            for v in range(8):
                if (u, v) == (0, 0): continue
                if q[u, v] != 0:
                    ac_pos.append(u * 8 + v)
                    ac_data.append(q[u, v])
        ac_off.append(len(ac_data))

    return dc_quant, np.array(ac_data, dtype=np.int16), np.array(ac_pos, dtype=np.uint8), \
           np.array(ac_off, dtype=np.int64), n_blocks


def compute_dc_16x16(uint8_rows, step):
    """16×16 block = 16 rows per block."""
    RPB = 16; TILE = 4; BLOCK = 16
    n = uint8_rows.shape[0]
    n_pad = ((n + RPB - 1) // RPB) * RPB
    if n_pad > n:
        p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8); p[:n] = uint8_rows; uint8_rows = p
    n_blocks = n_pad // RPB
    rows = uint8_rows.reshape(n_blocks, RPB, EMB_DIM).astype(np.float32)
    # 16 rows × 16 dims → 4×4 grid of 4×4 tiles in 16×16 block
    blocks = np.zeros((n_blocks, BLOCK, BLOCK), dtype=np.float32)
    for r in range(RPB):
        tiles = rows[:, r, :].reshape(n_blocks, TILE, TILE)
        r_grid = r // 4  # which tile row (0-3)
        c_grid = r % 4   # which tile col (0-3)
        blocks[:, r_grid * TILE:(r_grid + 1) * TILE,
               c_grid * TILE:(c_grid + 1) * TILE] = tiles
    dct = dctn(blocks, axes=(-2, -1), type=2, norm='ortho')
    dc_quant = np.round(dct[:, 0, 0] / step).astype(np.int16)

    quantized = np.round(dct / step).astype(np.int16)
    ac_data, ac_pos, ac_off = [], [], [0]
    for bi in range(n_blocks):
        q = quantized[bi]
        for u in range(16):
            for v in range(16):
                if (u, v) == (0, 0): continue
                if q[u, v] != 0:
                    ac_pos.append(u * 16 + v)
                    ac_data.append(q[u, v])
        ac_off.append(len(ac_data))

    return dc_quant, np.array(ac_data, dtype=np.int16), np.array(ac_pos, dtype=np.uint8), \
           np.array(ac_off, dtype=np.int64), n_blocks


def reconstruct_rows(dc_quant, ac_data, ac_pos, ac_off, n_blocks, step, block_size, rows_per_block, n_rows, D=16):
    """Reconstruct uint8 rows from DC+AC coefficients via IDCT."""
    from scipy.fft import idctn
    TILE = 4
    all_rows = np.zeros((n_blocks * rows_per_block, D), dtype=np.float32)

    for bi in range(n_blocks):
        coeffs = np.zeros((block_size, block_size), dtype=np.float32)
        coeffs[0, 0] = dc_quant[bi] * step
        start, end = ac_off[bi], ac_off[bi + 1]
        for j in range(start, end):
            pos = ac_pos[j]
            u, v = pos // block_size, pos % block_size
            coeffs[u, v] = ac_data[j] * step

        block = idctn(coeffs.reshape(1, block_size, block_size),
                      axes=(-2, -1), type=2, norm='ortho')[0]
        block = np.clip(np.round(block), 0, 255)

        for r in range(rows_per_block):
            if block_size == 4:
                tile = block.ravel()
            elif block_size == 8:
                rs = (r // 2) * TILE
                cs = (r % 2) * TILE
                tile = block[rs:rs + TILE, cs:cs + TILE].ravel()
            elif block_size == 16:
                rs = (r // 4) * TILE
                cs = (r % 4) * TILE
                tile = block[rs:rs + TILE, cs:cs + TILE].ravel()
            row_idx = bi * rows_per_block + r
            if row_idx < n_rows:
                all_rows[row_idx] = tile

    return all_rows[:n_rows].astype(np.uint8)


def main():
    print("=" * 70)
    print("BLOCK SIZE SWEEP: 4×4 vs 8×8 vs 16×16")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    nt = len(ln_emb)
    torch.set_num_threads(32)
    is_hot = {t: torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True) for t in TABLES}
    hi = {t: torch.where(is_hot[t])[0] for t in TABLES}

    # Baseline
    scores, targets = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_base = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"Baseline: {auc_base:.6f}")

    step = 32
    block_configs = [
        ("4×4 (1 row/block)", 4, 1, compute_dc_4x4),
        ("8×8 (4 rows/block)", 8, 4, compute_dc_8x8),
        ("16×16 (16 rows/block)", 16, 16, compute_dc_16x16),
    ]

    results = []
    total_fp32 = sum(int(ln_emb[t]) * EMB_DIM * 4 for t in TABLES)

    for label, block_size, rpb, compute_fn in block_configs:
        print(f"\n{'='*60}")
        print(f"{label}, step={step}")
        print(f"{'='*60}")

        # Compute DC for each table, measure compression
        dc_data = {}
        total_dc_bytes = 0
        total_ac_bytes = 0
        total_blocks = 0
        dc_only_count = 0
        recon_errors = {}

        for t in TABLES:
            cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
            w = sd[ek[t]][cold_order]
            q, s, zp = quantize_table(w)
            q_np = q.numpy()
            n_cold = len(cold_order)

            dc, ac_d, ac_p, ac_o, nb = compute_fn(q_np, step)
            dc_data[t] = (dc, ac_d, ac_p, ac_o, nb, s, zp, n_cold, q_np)

            # Stats
            dc_bytes = nb  # uint8 (1 byte per block)
            ac_bytes = len(ac_d) * 2 + len(ac_p)  # int16 + uint8 per AC
            total_dc_bytes += dc_bytes
            total_ac_bytes += ac_bytes
            total_blocks += nb
            n_dc_only = sum(1 for i in range(nb) if ac_o[i + 1] == ac_o[i])
            dc_only_count += n_dc_only

            # Reconstruction error
            recon = reconstruct_rows(dc, ac_d, ac_p, ac_o, nb, step, block_size, rpb, n_cold)
            err = np.abs(recon.astype(np.int16) - q_np[:n_cold].astype(np.int16))
            recon_errors[t] = (err.max(), err.mean())

            print(f"  t{t}: {nb:>8} blocks, DC-only={n_dc_only / nb * 100:.0f}%, "
                  f"AC={len(ac_d)}, max_err={err.max():.0f}, mean_err={err.mean():.3f}")

        dc_only_pct = dc_only_count / total_blocks * 100
        total_cold_kb = (total_dc_bytes + total_ac_bytes) / 1024
        print(f"  Total: {total_blocks} blocks, DC-only={dc_only_pct:.1f}%, "
              f"DC={total_dc_bytes / 1024:.0f}KB, AC={total_ac_bytes / 1024:.0f}KB, "
              f"cold={total_cold_kb:.0f}KB")

        # AUC via full decode → build embedding table
        cold_decoded = {}
        for t in TABLES:
            dc, ac_d, ac_p, ac_o, nb, s, zp, n_cold, q_orig = dc_data[t]
            recon = reconstruct_rows(dc, ac_d, ac_p, ac_o, nb, step, block_size, rpb, n_cold)
            cold_decoded[t] = (recon, s, zp)

        with torch.no_grad():
            for k in ek:
                t_idx = int(k.split('.')[1])
                dlrm.emb_l[t_idx] = nn.EmbeddingBag(int(ln_emb[t_idx]), EMB_DIM,
                                                      mode='sum', sparse=True)
                w = sd[k].clone()
                if t_idx in cold_decoded:
                    recon, s, zp = cold_decoded[t_idx]
                    cold_order = np.load(f'{REORDER_DIR}/cold_order_{t_idx}.npy')
                    decoded_fp32 = (torch.from_numpy(recon).float() - zp) * s
                    n = min(len(cold_order), decoded_fp32.shape[0])
                    w[cold_order[:n]] = decoded_fp32[:n]
                dlrm.emb_l[t_idx].weight.data = w

        # Warmup
        with torch.no_grad():
            for X, o, i, T in test_batches[:10]: dlrm(X, o, i)

        scores, targets, batch_times = [], [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                t0 = time.perf_counter()
                Z = dlrm(X, o, i)
                batch_times.append(time.perf_counter() - t0)
                scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())

        auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
        batch_ms = np.median(batch_times) * 1000

        # Memory: hot (22.1MB) + small (2.9MB) + bitmap (6MB) + cold DCT
        hot_mb = 22.1
        total_mb = hot_mb + 2.9 + 6.0 + total_cold_kb / 1024
        ratio = 2060.7 / total_mb

        print(f"\n  AUC:      {auc:.6f} (loss: {(auc_base - auc) * 100:+.4f}%)")
        print(f"  Batch:    {batch_ms:.2f}ms")
        print(f"  Cold mem: {total_cold_kb:.0f}KB")
        print(f"  Total:    {total_mb:.1f}MB ({ratio:.0f}x)")

        results.append({
            'label': label, 'block_size': block_size, 'rows_per_block': rpb,
            'auc': float(auc), 'auc_loss': float(auc_base - auc),
            'batch_ms': float(batch_ms),
            'total_blocks': total_blocks, 'dc_only_pct': dc_only_pct,
            'dc_kb': total_dc_bytes / 1024, 'ac_kb': total_ac_bytes / 1024,
            'cold_kb': total_cold_kb, 'total_mb': total_mb, 'ratio': ratio,
        })
        del cold_decoded; gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Config':<25} {'AUC':>10} {'Loss':>10} {'Batch':>8} {'DC-only':>8} "
          f"{'Cold KB':>8} {'Total':>8} {'Ratio':>6}")
    print("-" * 85)
    for r in results:
        print(f"{r['label']:<25} {r['auc']:>10.6f} {r['auc_loss']*100:>+9.4f}% "
              f"{r['batch_ms']:>7.2f} {r['dc_only_pct']:>7.1f}% "
              f"{r['cold_kb']:>7.0f} {r['total_mb']:>7.1f}MB {r['ratio']:>5.0f}x")

    os.makedirs('results/dct_domain', exist_ok=True)
    with open('results/dct_domain/block_size_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
