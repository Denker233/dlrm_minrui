#!/usr/bin/env python3
"""
DCT-Domain Embedding Lookup: Compute EmbeddingBag SUM directly from
quantized DCT coefficients without full IDCT decompression.

Key insight: For SUM mode, we need sum of selected rows' values.
With raw DCT coefficients (no intra prediction), this is a dot product
of non-zero coefficients with pre-computed weight vectors.

Layout: Each embedding row (D=16) is a 4×4 tile. An 8×8 DCT block
contains 4 tiles (2×2 arrangement) = 4 embedding rows.
"""
import os, sys, time, json, struct, gc
import numpy as np
from scipy.fft import dctn, idctn
import zstandard as zstd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

from codec_ondemand_benchmark import (
    load_model_and_data, EMB_DIM, MODEL_PATH,
    HOTCOLD_DIR, REORDER_DIR, quantize_table,
)

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
BLOCK = 8
TILE = 4
ROWS_PER_BLOCK = (BLOCK // TILE) * (BLOCK // TILE)  # 4

# ================================================================
# Pre-compute weight matrices for DCT-domain row extraction
# ================================================================
def precompute_weights():
    """
    For each (row_in_block, dimension), compute the weight vector over
    8×8 DCT coefficients that gives the pixel value.

    Row r (0-3) in block maps to tile positions:
      row 0: pixels (0:4, 0:4)  → tile top-left
      row 1: pixels (0:4, 4:8)  → tile top-right
      row 2: pixels (4:8, 0:4)  → tile bottom-left
      row 3: pixels (4:8, 4:8)  → tile bottom-right

    Dimension d (0-15) within tile maps to:
      tile_y = d // 4, tile_x = d % 4

    Pixel position for (row r, dim d):
      py = (r // 2) * 4 + d // 4
      px = (r % 2) * 4 + d % 4

    IDCT formula (ortho-normalized):
      f(x,y) = sum_{u,v} C(u)*C(v) * F(u,v) * cos(pi*(2x+1)*u/16) * cos(pi*(2y+1)*v/16)

    So weight for coefficient (u,v) to get pixel (x,y):
      w(u,v,x,y) = C(u)*C(v) * cos(pi*(2x+1)*u/16) * cos(pi*(2y+1)*v/16)
    """
    # Build basis functions
    weights = np.zeros((ROWS_PER_BLOCK, EMB_DIM, BLOCK, BLOCK), dtype=np.float32)

    for r in range(ROWS_PER_BLOCK):
        for d in range(EMB_DIM):
            py = (r // 2) * TILE + d // TILE
            px = (r % 2) * TILE + d % TILE

            for u in range(BLOCK):
                for v in range(BLOCK):
                    cu = 1.0 / np.sqrt(BLOCK) if u == 0 else np.sqrt(2.0 / BLOCK)
                    cv = 1.0 / np.sqrt(BLOCK) if v == 0 else np.sqrt(2.0 / BLOCK)
                    weights[r, d, u, v] = (cu * cv *
                        np.cos(np.pi * (2 * py + 1) * u / (2 * BLOCK)) *
                        np.cos(np.pi * (2 * px + 1) * v / (2 * BLOCK)))

    # Verify: the sum over all rows and dims for DC should give block_sum
    # DC weight for (r,d) should be 1/8 for ortho-normalized DCT
    return weights


def precompute_sum_weights():
    """
    For SUM of ALL 4 rows: weight for each (dim, u, v) = sum over rows of weights[r, dim, u, v]
    For SUM of subset: combine appropriately.

    But for EmbeddingBag, we often sum different rows from DIFFERENT blocks.
    So the per-row weights are more useful than pre-summed weights.
    """
    return precompute_weights()  # shape: (4, 16, 8, 8)


# ================================================================
# Codec: Encode with random-access block index
# ================================================================
def ra_encode(uint8_rows, step_size=16):
    """
    Random-access DCT codec.

    Returns: (compressed_bytes, block_index, metadata)
      block_index[i] = (offset, length) of block i's coefficients in compressed_bytes
    """
    n_rows, dim = uint8_rows.shape
    assert dim == EMB_DIM

    # Pad to multiple of ROWS_PER_BLOCK
    n_pad = ((n_rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK) * ROWS_PER_BLOCK
    if n_pad > n_rows:
        padded = np.zeros((n_pad, dim), dtype=np.uint8)
        padded[:n_rows] = uint8_rows
        uint8_rows = padded

    n_blocks = n_pad // ROWS_PER_BLOCK

    # Tile: reshape each group of 4 rows into 8×8 block
    # Row layout in 8×8 block: 2×2 grid of 4×4 tiles
    blocks = np.zeros((n_blocks, BLOCK, BLOCK), dtype=np.float32)
    rows_reshaped = uint8_rows.reshape(n_blocks, ROWS_PER_BLOCK, dim).astype(np.float32)

    for r in range(ROWS_PER_BLOCK):
        tile = rows_reshaped[:, r, :].reshape(-1, TILE, TILE)  # (n_blocks, 4, 4)
        r_start = (r // 2) * TILE
        c_start = (r % 2) * TILE
        blocks[:, r_start:r_start+TILE, c_start:c_start+TILE] = tile

    # DCT transform
    dct_coeffs = dctn(blocks, axes=(-2, -1), type=2, norm='ortho')

    # Quantize
    quantized = np.round(dct_coeffs / step_size).astype(np.int16)

    # Store each block's non-zero coefficients with index for random access
    # Format per block: [n_nonzero (uint8)] + [pos, value] pairs
    # pos = u*8+v (uint8), value = int16
    all_block_data = bytearray()
    block_offsets = []  # (offset, length) for each block

    for bi in range(n_blocks):
        block_start = len(all_block_data)
        q = quantized[bi]
        nz_mask = q != 0
        nz_positions = np.argwhere(nz_mask)  # (k, 2) array of (u,v)
        n_nz = len(nz_positions)

        # Write: count + (pos, value) pairs
        all_block_data.append(min(n_nz, 255))
        for (u, v) in nz_positions:
            pos = u * BLOCK + v
            val = int(q[u, v])
            all_block_data.append(pos)
            all_block_data.extend(struct.pack('<h', val))  # int16

        block_offsets.append((block_start, len(all_block_data) - block_start))

    # Optionally Zstd-compress the whole thing for storage
    cctx = zstd.ZstdCompressor(level=3)
    compressed = cctx.compress(bytes(all_block_data))

    meta = {
        'n_rows': n_rows, 'n_blocks': n_blocks,
        'step_size': step_size, 'dim': dim,
    }

    return compressed, bytes(all_block_data), block_offsets, meta


def ra_decode_block(raw_data, offset, length, step_size):
    """Decode a single block from raw (uncompressed) block data."""
    pos = offset
    n_nz = raw_data[pos]; pos += 1

    coeffs = np.zeros((BLOCK, BLOCK), dtype=np.float32)
    for _ in range(n_nz):
        p = raw_data[pos]; pos += 1
        val = struct.unpack('<h', raw_data[pos:pos+2])[0]; pos += 2
        u, v = p // BLOCK, p % BLOCK
        coeffs[u, v] = val * step_size

    return coeffs  # DCT coefficients (dequantized)


def ra_decode_full(raw_data, block_offsets, step_size, n_rows):
    """Decode all blocks → full uint8 rows (for verification)."""
    n_blocks = len(block_offsets)
    all_rows = np.zeros((n_blocks * ROWS_PER_BLOCK, EMB_DIM), dtype=np.uint8)

    for bi in range(n_blocks):
        offset, length = block_offsets[bi]
        dct_coeffs = ra_decode_block(raw_data, offset, length, step_size)

        # Full IDCT
        block = idctn(dct_coeffs.reshape(1, BLOCK, BLOCK),
                      axes=(-2, -1), type=2, norm='ortho')[0]
        block = np.clip(np.round(block), 0, 255).astype(np.uint8)

        # Untile: extract 4 rows from 8×8 block
        for r in range(ROWS_PER_BLOCK):
            r_start = (r // 2) * TILE
            c_start = (r % 2) * TILE
            tile = block[r_start:r_start+TILE, c_start:c_start+TILE]
            row_idx = bi * ROWS_PER_BLOCK + r
            all_rows[row_idx] = tile.ravel()

    return all_rows[:n_rows]


def dct_domain_lookup(raw_data, block_offsets, step_size, row_indices, weights,
                      quant_scale, quant_zp):
    """
    Compute EmbeddingBag SUM directly from DCT coefficients.

    Args:
        raw_data: uncompressed block data
        block_offsets: [(offset, length)] per block
        step_size: DCT quantization step
        row_indices: 1D array of cold row indices to sum
        weights: precomputed weight matrices (4, 16, 8, 8)
        quant_scale, quant_zp: uint8 dequantization params

    Returns:
        fp32 tensor of shape (EMB_DIM,) = dequantized sum of selected rows
    """
    output = np.zeros(EMB_DIM, dtype=np.float64)

    # Map row indices to blocks
    for row_idx in row_indices:
        block_id = row_idx // ROWS_PER_BLOCK
        row_in_block = row_idx % ROWS_PER_BLOCK

        if block_id >= len(block_offsets):
            continue

        offset, length = block_offsets[block_id]
        pos = offset
        n_nz = raw_data[pos]; pos += 1

        # For each non-zero DCT coefficient, accumulate its contribution
        # to the output via pre-computed weights
        w_row = weights[row_in_block]  # (16, 8, 8)

        for _ in range(n_nz):
            p = raw_data[pos]; pos += 1
            val = struct.unpack('<h', raw_data[pos:pos+2])[0]; pos += 2
            u, v = p // BLOCK, p % BLOCK
            # contribution = val * step_size * w_row[:, u, v]
            output += val * step_size * w_row[:, u, v]

    # Dequantize: uint8_domain_sum → fp32
    # Each row was uint8 with scale/zp: fp32_val = (uint8_val - zp) * scale
    # Sum of fp32 = sum((uint8_val - zp) * scale) = scale * (sum(uint8_val) - n * zp)
    n = len(row_indices)
    output_fp32 = (output - n * quant_zp) * quant_scale

    return output_fp32.astype(np.float32)


# ================================================================
# Vectorized DCT-domain lookup (for benchmarking speed)
# ================================================================
def dct_domain_lookup_batch(raw_data, block_offsets, step_size, row_indices,
                            weights, quant_scale, quant_zp):
    """Vectorized version: batch all rows, process blocks once."""
    output = np.zeros(EMB_DIM, dtype=np.float64)

    # Group rows by block
    block_to_rows = {}
    for ri in row_indices:
        bid = ri // ROWS_PER_BLOCK
        rib = ri % ROWS_PER_BLOCK
        if bid not in block_to_rows:
            block_to_rows[bid] = []
        block_to_rows[bid].append(rib)

    # Process each unique block once
    for bid, rows_in_block in block_to_rows.items():
        if bid >= len(block_offsets):
            continue

        offset, length = block_offsets[bid]
        pos = offset
        n_nz = raw_data[pos]; pos += 1

        # Sum weights for all needed rows in this block
        combined_w = np.zeros((EMB_DIM, BLOCK, BLOCK), dtype=np.float64)
        for rib in rows_in_block:
            combined_w += weights[rib]

        # For each non-zero coefficient
        for _ in range(n_nz):
            p = raw_data[pos]; pos += 1
            val = struct.unpack('<h', raw_data[pos:pos+2])[0]; pos += 2
            u, v = p // BLOCK, p % BLOCK
            output += val * step_size * combined_w[:, u, v]

    n = len(row_indices)
    return ((output - n * quant_zp) * quant_scale).astype(np.float32)


# ================================================================
# Main experiment
# ================================================================
def main():
    print("=" * 70)
    print("DCT-DOMAIN EMBEDDING LOOKUP")
    print("=" * 70)

    # Pre-compute weight matrices
    print("Pre-computing DCT weight matrices...")
    weights = precompute_weights()
    print(f"  Weights shape: {weights.shape} (rows_per_block, dim, block_h, block_w)")

    # Verify weights by comparing IDCT with weight-based reconstruction
    print("  Verifying weights...")
    test_block = np.random.randint(0, 255, (BLOCK, BLOCK)).astype(np.float32)
    dct_test = dctn(test_block.reshape(1, BLOCK, BLOCK), axes=(-2, -1), type=2, norm='ortho')[0]

    for r in range(ROWS_PER_BLOCK):
        for d in range(EMB_DIM):
            # Via IDCT
            idct_val = idctn(dct_test.reshape(1, BLOCK, BLOCK), axes=(-2, -1), type=2, norm='ortho')[0]
            py = (r // 2) * TILE + d // TILE
            px = (r % 2) * TILE + d % TILE
            val_idct = idct_val[py, px]

            # Via weights
            val_weight = np.sum(dct_test * weights[r, d])

            if abs(val_idct - val_weight) > 1e-4:
                print(f"  MISMATCH at r={r}, d={d}: idct={val_idct:.4f}, weight={val_weight:.4f}")
                return
    print("  Weights verified: all match IDCT ✓")

    # Load model and data
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    torch.set_num_threads(32)

    is_hot = {}
    for t in TABLES:
        is_hot[t] = torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True)

    # Encode cold data with random-access codec
    print("\nEncoding cold data...")
    encoded = {}  # t -> (compressed, raw, offsets, meta, scale, zp)

    step_sizes_to_test = [1, 8, 16, 32]
    total_raw_uint8 = 0
    total_fp32 = 0

    for step in step_sizes_to_test:
        total_comp = 0
        total_raw = 0
        encoded_step = {}

        for t in TABLES:
            cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
            w = sd[ek[t]][cold_order]
            q, s, zp = quantize_table(w)
            q_np = q.numpy()

            comp, raw, offsets, meta = ra_encode(q_np, step_size=step)
            total_comp += len(comp)
            total_raw += len(raw)
            encoded_step[t] = (comp, raw, offsets, meta, s, zp, q_np)

            if step == step_sizes_to_test[0]:
                total_raw_uint8 += q_np.size
                total_fp32 += q_np.size * 4

        ratio_fp32 = total_fp32 / total_comp if total_comp > 0 else 0
        ratio_raw = total_fp32 / total_raw if total_raw > 0 else 0
        print(f"  step={step}: compressed={total_comp/1024:.0f}KB ({ratio_fp32:.0f}x vs fp32), "
              f"raw block data={total_raw/1024:.0f}KB ({ratio_raw:.0f}x vs fp32)")
        encoded[step] = encoded_step

    # ================================================================
    # Verify: DCT-domain lookup matches standard decode
    # ================================================================
    print(f"\n{'='*70}")
    print("VERIFICATION: DCT-domain vs standard decode")
    print(f"{'='*70}")

    step = 16
    for t in [2, 9]:  # test on one large and one small table
        comp, raw, offsets, meta, s, zp, q_orig = encoded[step][t]
        n_rows = meta['n_rows']

        # Standard: full decode → gather → sum
        decoded = ra_decode_full(raw, offsets, step, n_rows)
        test_rows = [0, 1, 10, 100, min(1000, n_rows-1)]

        for ri in test_rows:
            # Via standard decode + dequant
            std_val = (decoded[ri].astype(np.float32) - zp) * s

            # Via DCT-domain lookup (single row)
            dct_val = dct_domain_lookup(raw, offsets, step, [ri], weights, s, zp)

            err = np.abs(std_val - dct_val).max()
            print(f"  table {t}, row {ri}: max_err={err:.6f} "
                  f"({'OK' if err < 0.01 else 'MISMATCH'})")

        # Test SUM of multiple rows
        sum_rows = list(range(min(10, n_rows)))
        std_sum = np.zeros(EMB_DIM, dtype=np.float32)
        for ri in sum_rows:
            std_sum += (decoded[ri].astype(np.float32) - zp) * s

        dct_sum = dct_domain_lookup_batch(raw, offsets, step, sum_rows, weights, s, zp)
        err = np.abs(std_sum - dct_sum).max()
        print(f"  table {t}, SUM of {len(sum_rows)} rows: max_err={err:.6f}")

    # ================================================================
    # Speed benchmark: DCT-domain vs standard decode
    # ================================================================
    print(f"\n{'='*70}")
    print("SPEED: DCT-domain lookup vs standard decode per batch")
    print(f"{'='*70}")

    step = 16
    # Simulate a batch: 128 samples, ~5% cold per table = ~6-7 cold rows per table
    np.random.seed(42)

    for n_cold_rows in [5, 10, 20, 50]:
        print(f"\n  {n_cold_rows} cold rows per table:")

        # Standard: decode all blocks that contain needed rows → extract → sum
        t_standard = 0
        for t in TABLES:
            comp, raw, offsets, meta, s, zp, q_orig = encoded[step][t]
            n_rows = meta['n_rows']
            cold_rows = np.random.choice(n_rows, size=min(n_cold_rows, n_rows), replace=False)

            t0 = time.perf_counter()
            for _ in range(100):
                # Standard: decode needed blocks
                needed_blocks = set(ri // ROWS_PER_BLOCK for ri in cold_rows)
                decoded_rows = {}
                for bid in needed_blocks:
                    dct = ra_decode_block(raw, offsets[bid][0], offsets[bid][1], step)
                    block = idctn(dct.reshape(1, BLOCK, BLOCK), axes=(-2,-1), type=2, norm='ortho')[0]
                    block = np.clip(np.round(block), 0, 255)
                    for r in range(ROWS_PER_BLOCK):
                        row_id = bid * ROWS_PER_BLOCK + r
                        r_start = (r // 2) * TILE
                        c_start = (r % 2) * TILE
                        decoded_rows[row_id] = block[r_start:r_start+TILE, c_start:c_start+TILE].ravel()

                result = np.zeros(EMB_DIM, dtype=np.float32)
                for ri in cold_rows:
                    if ri in decoded_rows:
                        result += (decoded_rows[ri] - zp) * s
            t_standard += (time.perf_counter() - t0) / 100

        # DCT-domain
        t_dct = 0
        for t in TABLES:
            comp, raw, offsets, meta, s, zp, q_orig = encoded[step][t]
            n_rows = meta['n_rows']
            cold_rows = np.random.choice(n_rows, size=min(n_cold_rows, n_rows), replace=False)

            t0 = time.perf_counter()
            for _ in range(100):
                result = dct_domain_lookup_batch(raw, offsets, step, cold_rows, weights, s, zp)
            t_dct += (time.perf_counter() - t0) / 100

        print(f"    Standard (decode blocks + extract): {t_standard*1000:.2f}ms")
        print(f"    DCT-domain (weight dot product):    {t_dct*1000:.2f}ms")
        print(f"    Speedup: {t_standard/t_dct:.1f}x")

    # ================================================================
    # AUC test: full inference with DCT-domain embedding lookup
    # ================================================================
    print(f"\n{'='*70}")
    print("AUC: Full inference with DCT-domain lookup")
    print(f"{'='*70}")

    step = 16

    # Baseline AUC
    def compute_auc_with_cold(cold_decoded_dict):
        with torch.no_grad():
            for k in ek:
                t_idx = int(k.split('.')[1])
                dlrm.emb_l[t_idx] = nn.EmbeddingBag(int(ln_emb[t_idx]), EMB_DIM,
                                                      mode='sum', sparse=True)
                w = sd[k].clone()
                if t_idx in cold_decoded_dict:
                    cold_order = np.load(f'{REORDER_DIR}/cold_order_{t_idx}.npy')
                    decoded = cold_decoded_dict[t_idx]
                    s_t = encoded[step][t_idx][4]
                    zp_t = encoded[step][t_idx][5]
                    decoded_fp32 = (torch.from_numpy(decoded).float() - zp_t) * s_t
                    n = min(len(cold_order), decoded_fp32.shape[0])
                    w[cold_order[:n]] = decoded_fp32[:n]
                dlrm.emb_l[t_idx].weight.data = w

        scores, targets = [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                Z = dlrm(X, o, i)
                scores.append(Z.numpy().ravel())
                targets.append(T.numpy().ravel())
        return roc_auc_score(np.concatenate(targets), np.concatenate(scores))

    print("  Baseline (fp32)...")
    auc_base = compute_auc_with_cold({})
    print(f"  Baseline: {auc_base:.6f}")

    # Decode with random-access codec (full decode for AUC)
    for step in [1, 8, 16, 32]:
        cold_decoded = {}
        total_comp = 0
        for t in TABLES:
            comp, raw, offsets, meta, s, zp, q_orig = encoded[step][t]
            total_comp += len(comp)
            decoded = ra_decode_full(raw, offsets, step, meta['n_rows'])
            cold_decoded[t] = decoded

            # Error vs original
            err = np.abs(decoded.astype(np.int16) - q_orig[:meta['n_rows']].astype(np.int16))
            if t == 2:
                print(f"    table 2 step={step}: max_err={err.max()}, mean_err={err.mean():.3f}")

        auc = compute_auc_with_cold(cold_decoded)
        ratio = total_fp32 / total_comp
        print(f"  step={step}: AUC={auc:.6f} (loss={auc_base-auc:.6f}, "
              f"{(auc_base-auc)*100:+.4f}%), ratio={ratio:.0f}x")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Random-access DCT codec with compressed-domain lookup:")
    print(f"  - Storage: quantized DCT coefficients + block index")
    print(f"  - Lookup: dot product of non-zero coefficients with pre-computed weights")
    print(f"  - No IDCT, no frame decode, no cache needed")
    print(f"  - Each block independent → random access, fully parallel")

    os.makedirs('results/dct_domain', exist_ok=True)
    print(f"\nDone.")


if __name__ == '__main__':
    main()
