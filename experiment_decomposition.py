#!/usr/bin/env python3
"""
Decomposition experiment: isolate each compression component's contribution.

Pipeline: fp32 → uint8 → [frequency sort] → [tiling] → [DCT+quant] → [intra pred] → [CABAC]

Each step measured independently by comparing with/without.
All experiments on the SAME data (8 large Kaggle tables, cold rows only).
"""
import os, sys, time, json, subprocess, struct, gc, tempfile, shutil
import numpy as np
import zstandard as zstd
from scipy.fft import dctn, idctn
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

SF_DIMS = {
    2: (3840, 40400), 3: (1920, 17568), 9: (1920, 744), 11: (3840, 33304),
    15: (1920, 43556), 20: (1920, 56200), 23: (1920, 2284), 25: (1920, 1140),
}

import compressed_emb as _C


def zstd_compress(data_bytes, level=3):
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data_bytes)


def encode_h265(raw_frame_bytes, width, height, outpath, crf=30, extra=''):
    """Encode raw gray frame with H.265."""
    x265 = f'keyint=1:min-keyint=1:crf={crf}:log-level=error'
    if extra:
        x265 += ':' + extra
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
           '-s', f'{width}x{height}', '-r', '1', '-i', 'pipe:0',
           '-c:v', 'libx265', '-preset', 'medium', '-pix_fmt', 'gray',
           '-x265-params', x265, '-f', 'matroska', outpath]
    subprocess.run(cmd, input=raw_frame_bytes, capture_output=True, timeout=120)
    return os.path.getsize(outpath) if os.path.exists(outpath) else 0


def dct_quantize_zstd(uint8_flat, step_size=16, zstd_level=3):
    """DCT + uniform quantize + Zstd (no intra prediction, no CABAC)."""
    flat = uint8_flat.astype(np.float32)
    bs = BLOCK * BLOCK
    n = len(flat)
    n_pad = ((n + bs - 1) // bs) * bs
    if n_pad > n:
        flat = np.concatenate([flat, np.zeros(n_pad - n, dtype=np.float32)])
    blocks = flat.reshape(-1, BLOCK, BLOCK)
    dct = dctn(blocks, axes=(-2,-1), type=2, norm='ortho')
    q = np.round(dct / step_size).astype(np.int16)
    return zstd_compress(q.tobytes(), zstd_level)


def main():
    print("=" * 70)
    print("DECOMPOSITION: Isolating each compression component")
    print("=" * 70)

    # Load model and cold data
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    torch.set_num_threads(32)

    is_hot = {}
    for t in TABLES:
        is_hot[t] = torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True)

    # Get fp32 cold weights (SORTED and UNSORTED)
    cold_fp32_sorted = {}      # frequency sorted
    cold_fp32_unsorted = {}    # natural order
    cold_uint8_sorted = {}
    cold_uint8_unsorted = {}
    cold_scale, cold_zp = {}, {}

    for t in TABLES:
        cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
        cold_indices = torch.where(~is_hot[t])[0].numpy()

        # Sorted (by access frequency)
        w_sorted = sd[ek[t]][cold_order].numpy()
        cold_fp32_sorted[t] = w_sorted

        # Unsorted (natural index order)
        w_unsorted = sd[ek[t]][cold_indices].numpy()
        cold_fp32_unsorted[t] = w_unsorted

        # Quantize (same scale for fair comparison — use sorted data for scale)
        q_s, s, zp = quantize_table(torch.from_numpy(w_sorted))
        cold_uint8_sorted[t] = q_s.numpy()
        cold_scale[t] = s
        cold_zp[t] = zp

        # Quantize unsorted with SAME scale/zp
        q_u = np.clip(np.round(w_unsorted / s + zp), 0, 255).astype(np.uint8)
        cold_uint8_unsorted[t] = q_u

    total_fp32 = sum(cold_fp32_sorted[t].size * 4 for t in TABLES)
    total_uint8 = sum(cold_uint8_sorted[t].size for t in TABLES)
    n_cold_total = sum(cold_uint8_sorted[t].shape[0] for t in TABLES)
    print(f"Total cold: {n_cold_total:,} rows, {total_fp32/1024/1024:.0f}MB fp32, "
          f"{total_uint8/1024/1024:.0f}MB uint8")

    tmpdir = tempfile.mkdtemp(prefix='decomp_')
    results = {}

    # ================================================================
    # Level 0: Raw fp32 (baseline)
    # ================================================================
    print(f"\n{'='*70}")
    print("Level 0: Raw fp32 (no compression)")
    print(f"{'='*70}")
    print(f"  Size: {total_fp32/1024/1024:.0f}MB = {total_fp32:,} bytes")
    results['L0_fp32'] = {'bytes': total_fp32, 'ratio_vs_fp32': 1.0}

    # ================================================================
    # Level 1: uint8 quantization only
    # ================================================================
    print(f"\n{'='*70}")
    print("Level 1: uint8 quantization (fp32 → uint8)")
    print(f"{'='*70}")
    print(f"  Size: {total_uint8/1024/1024:.0f}MB = {total_uint8:,} bytes")
    print(f"  Ratio vs fp32: {total_fp32/total_uint8:.1f}x")
    results['L1_uint8'] = {'bytes': total_uint8, 'ratio_vs_fp32': total_fp32/total_uint8}

    # ================================================================
    # Level 2: uint8 + Zstd (entropy coding, no DCT)
    # Measure sorted vs unsorted separately
    # ================================================================
    print(f"\n{'='*70}")
    print("Level 2: uint8 + Zstd-3 (entropy coding only)")
    print(f"{'='*70}")

    for label, data in [("unsorted", cold_uint8_unsorted), ("sorted", cold_uint8_sorted)]:
        total_comp = 0
        for t in TABLES:
            comp = zstd_compress(data[t].tobytes())
            total_comp += len(comp)
        ratio = total_fp32 / total_comp
        print(f"  {label}: {total_comp/1024:.0f}KB ({ratio:.0f}x vs fp32, "
              f"{total_uint8/total_comp:.1f}x vs uint8)")
        results[f'L2_zstd_{label}'] = {'bytes': total_comp, 'ratio_vs_fp32': ratio,
                                         'ratio_vs_uint8': total_uint8/total_comp}

    # ================================================================
    # Level 3: uint8 + DCT + Zstd (DCT decorrelation, no lossy quantization)
    # step_size=1 means lossless DCT (just transform, don't drop coefficients)
    # ================================================================
    print(f"\n{'='*70}")
    print("Level 3: uint8 + lossless DCT + Zstd (DCT decorrelation only)")
    print(f"{'='*70}")

    for label, data in [("unsorted", cold_uint8_unsorted), ("sorted", cold_uint8_sorted)]:
        total_comp = 0
        for t in TABLES:
            comp = dct_quantize_zstd(data[t].ravel(), step_size=1)
            total_comp += len(comp)
        ratio = total_fp32 / total_comp
        prev = results[f'L2_zstd_{label}']['bytes']
        print(f"  {label}: {total_comp/1024:.0f}KB ({ratio:.0f}x vs fp32, "
              f"{prev/total_comp:.2f}x improvement from DCT)")
        results[f'L3_dct_lossless_{label}'] = {
            'bytes': total_comp, 'ratio_vs_fp32': ratio,
            'improvement_from_dct': prev/total_comp}

    # ================================================================
    # Level 4: uint8 + DCT + lossy quantization + Zstd
    # (like our custom codec — isolates DCT quantization benefit)
    # ================================================================
    print(f"\n{'='*70}")
    print("Level 4: uint8 + DCT + lossy quant + Zstd (DCT quantization benefit)")
    print(f"{'='*70}")

    for step in [4, 8, 16, 32, 64, 128]:
        total_comp = 0
        for t in TABLES:
            comp = dct_quantize_zstd(cold_uint8_sorted[t].ravel(), step_size=step)
            total_comp += len(comp)
        ratio = total_fp32 / total_comp
        l3 = results['L3_dct_lossless_sorted']['bytes']
        print(f"  step={step:>3}: {total_comp/1024:>7.0f}KB ({ratio:>6.0f}x vs fp32, "
              f"{l3/total_comp:>5.1f}x from lossy quant)")
        results[f'L4_dct_lossy_step{step}'] = {
            'bytes': total_comp, 'ratio_vs_fp32': ratio,
            'improvement_from_lossy_quant': l3/total_comp}

    # ================================================================
    # Level 5: H.265 lossless (CRF=0) — adds CABAC + intra prediction
    # over lossless DCT. Compare with Level 3.
    # ================================================================
    print(f"\n{'='*70}")
    print("Level 5: H.265 CRF=0 (lossless) — isolates CABAC + intra vs Zstd")
    print(f"{'='*70}")

    for label, data in [("unsorted", cold_uint8_unsorted), ("sorted", cold_uint8_sorted)]:
        total_comp = 0
        for t in TABLES:
            w, h = SF_DIMS[t]
            rpf = (w * h) // EMB_DIM
            n = data[t].shape[0]
            q_t = torch.from_numpy(data[t])
            if n < rpf:
                q_t = torch.cat([q_t, torch.zeros(rpf - n, EMB_DIM, dtype=torch.uint8)])
            tiled = _C.fused_quantize_tile_multiframe(q_t[:rpf], w, h)
            raw = tiled[0].numpy().tobytes()
            outpath = os.path.join(tmpdir, f'L5_{label}_t{t}.h265')
            comp = encode_h265(raw, w, h, outpath, crf=0, extra='no-deblock=1:no-sao=1')
            total_comp += comp

        ratio = total_fp32 / total_comp
        l3 = results[f'L3_dct_lossless_{label}']['bytes']
        print(f"  {label}: {total_comp/1024:.0f}KB ({ratio:.0f}x vs fp32, "
              f"{l3/total_comp:.2f}x better than DCT+Zstd lossless)")
        results[f'L5_h265_crf0_{label}'] = {
            'bytes': total_comp, 'ratio_vs_fp32': ratio,
            'improvement_over_dct_zstd': l3/total_comp}

    # ================================================================
    # Level 5b: H.265 CRF=0 WITHOUT intra prediction
    # Use --no-strong-intra-smoothing + constrained-intra + ctu=16
    # to minimize intra prediction contribution
    # ================================================================
    print(f"\n{'='*70}")
    print("Level 5b: H.265 CRF=0 with minimal intra prediction")
    print(f"{'='*70}")

    total_comp_nointra = 0
    for t in TABLES:
        w, h = SF_DIMS[t]
        rpf = (w * h) // EMB_DIM
        q_t = torch.from_numpy(cold_uint8_sorted[t])
        if q_t.shape[0] < rpf:
            q_t = torch.cat([q_t, torch.zeros(rpf - q_t.shape[0], EMB_DIM, dtype=torch.uint8)])
        tiled = _C.fused_quantize_tile_multiframe(q_t[:rpf], w, h)
        raw = tiled[0].numpy().tobytes()
        outpath = os.path.join(tmpdir, f'L5b_t{t}.h265')
        comp = encode_h265(raw, w, h, outpath, crf=0,
                           extra='no-deblock=1:no-sao=1:ctu=16:no-rect=1:no-amp=1')
        total_comp_nointra += comp

    ratio = total_fp32 / total_comp_nointra
    l5_full = results['L5_h265_crf0_sorted']['bytes']
    print(f"  sorted, minimal intra: {total_comp_nointra/1024:.0f}KB ({ratio:.0f}x, "
          f"full intra {l5_full/1024:.0f}KB, diff={total_comp_nointra/l5_full:.2f}x)")
    results['L5b_h265_crf0_nointra'] = {
        'bytes': total_comp_nointra, 'ratio_vs_fp32': ratio,
        'intra_contribution': total_comp_nointra / l5_full}

    # ================================================================
    # Level 6: H.265 CRF=30 (lossy) — full pipeline
    # ================================================================
    print(f"\n{'='*70}")
    print("Level 6: H.265 CRF=30 (lossy) — full pipeline")
    print(f"{'='*70}")

    for label, data, extra_params in [
        ("full (CABAC+intra+deblock)", cold_uint8_sorted, 'no-deblock=1:no-sao=1'),
        ("no intra (ctu=16)", cold_uint8_sorted, 'no-deblock=1:no-sao=1:ctu=16:no-rect=1:no-amp=1'),
    ]:
        total_comp = 0
        for t in TABLES:
            w, h = SF_DIMS[t]
            rpf = (w * h) // EMB_DIM
            q_t = torch.from_numpy(data[t])
            if q_t.shape[0] < rpf:
                q_t = torch.cat([q_t, torch.zeros(rpf - q_t.shape[0], EMB_DIM, dtype=torch.uint8)])
            tiled = _C.fused_quantize_tile_multiframe(q_t[:rpf], w, h)
            raw = tiled[0].numpy().tobytes()
            outpath = os.path.join(tmpdir, f'L6_{label[:10]}_t{t}.h265')
            comp = encode_h265(raw, w, h, outpath, crf=30, extra=extra_params)
            total_comp += comp

        ratio = total_fp32 / total_comp
        print(f"  {label}: {total_comp/1024:.0f}KB ({ratio:.0f}x vs fp32)")
        results[f'L6_h265_crf30_{label[:10]}'] = {'bytes': total_comp, 'ratio_vs_fp32': ratio}

    # ================================================================
    # Level 6b: Isolate CABAC contribution at CRF=30
    # Compare: DCT+quant+Zstd (level 4) vs H.265 CRF=30 (level 6)
    # The difference = CABAC + intra over Zstd
    # ================================================================

    # ================================================================
    # Summary decomposition table
    # ================================================================
    print(f"\n{'='*70}")
    print("DECOMPOSITION SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Level':<45} {'Size':>10} {'Ratio':>8} {'Step contrib':>14}")
    print("-" * 80)

    levels = [
        ('L0: fp32 (raw)', results['L0_fp32']),
        ('L1: + uint8 quantization', results['L1_uint8']),
        ('L2: + Zstd entropy (sorted)', results['L2_zstd_sorted']),
        ('L3: + lossless DCT (sorted)', results['L3_dct_lossless_sorted']),
        ('L4: + lossy DCT quant (step=16)', results.get('L4_dct_lossy_step16', {})),
        ('L4: + lossy DCT quant (step=64)', results.get('L4_dct_lossy_step64', {})),
        ('L5: H.265 CRF=0 lossless (sorted)', results['L5_h265_crf0_sorted']),
        ('L5b: H.265 CRF=0 no-intra (sorted)', results['L5b_h265_crf0_nointra']),
        ('L6: H.265 CRF=30 full', results.get('L6_h265_crf30_full (CABA', {})),
        ('L6: H.265 CRF=30 no-intra', results.get('L6_h265_crf30_no intra ', {})),
    ]

    prev_bytes = total_fp32
    for label, r in levels:
        if not r:
            continue
        b = r.get('bytes', 0)
        ratio = r.get('ratio_vs_fp32', 0)
        step = prev_bytes / b if b > 0 else 0
        sz = f"{b/1024:.0f}KB" if b < 1024*1024 else f"{b/1024/1024:.0f}MB"
        print(f"  {label:<43} {sz:>10} {ratio:>7.0f}x {step:>10.1f}x")
        prev_bytes = b

    # Key isolated contributions
    print(f"\n--- Isolated Component Contributions ---")

    # uint8 quant
    r_uint8 = total_fp32 / total_uint8
    print(f"  uint8 quantization:       {r_uint8:.1f}x (fp32/uint8)")

    # Zstd entropy (on uint8)
    r_zstd = total_uint8 / results['L2_zstd_sorted']['bytes']
    print(f"  Zstd entropy coding:      {r_zstd:.1f}x (uint8/Zstd)")

    # Frequency sorting (Zstd unsorted vs sorted)
    r_sort = results['L2_zstd_unsorted']['bytes'] / results['L2_zstd_sorted']['bytes']
    print(f"  Frequency sorting (Zstd): {r_sort:.2f}x ({(r_sort-1)*100:.1f}% improvement)")

    # DCT decorrelation (lossless: L2 vs L3)
    r_dct = results['L2_zstd_sorted']['bytes'] / results['L3_dct_lossless_sorted']['bytes']
    print(f"  DCT decorrelation:        {r_dct:.2f}x (Zstd/DCT+Zstd lossless)")

    # CABAC+intra over Zstd (lossless: L3 vs L5)
    r_cabac_intra = results['L3_dct_lossless_sorted']['bytes'] / results['L5_h265_crf0_sorted']['bytes']
    print(f"  CABAC+intra (lossless):   {r_cabac_intra:.2f}x (DCT+Zstd/H.265 CRF=0)")

    # Intra prediction alone (L5b vs L5 at CRF=0)
    r_intra = results['L5b_h265_crf0_nointra']['bytes'] / results['L5_h265_crf0_sorted']['bytes']
    print(f"  Intra prediction (CRF=0): {r_intra:.2f}x ({(r_intra-1)*100:.1f}% from intra)")

    # CABAC alone (approximately: L5/L5b gives intra, L3/L5 gives CABAC+intra)
    r_cabac_only = r_cabac_intra / r_intra
    print(f"  CABAC alone (approx):     {r_cabac_only:.2f}x")

    # Lossy DCT quantization (L5 CRF=0 vs L6 CRF=30)
    l6_key = [k for k in results if k.startswith('L6_h265_crf30_full')]
    if l6_key:
        r_lossy = results['L5_h265_crf0_sorted']['bytes'] / results[l6_key[0]]['bytes']
        print(f"  Lossy DCT quant (CRF=30): {r_lossy:.1f}x (H.265 CRF=0 / CRF=30)")

    # Intra prediction at CRF=30
    l6_full = [k for k in results if 'crf30_full' in k]
    l6_nointra = [k for k in results if 'crf30_no intra' in k]
    if l6_full and l6_nointra:
        r_intra30 = results[l6_nointra[0]]['bytes'] / results[l6_full[0]]['bytes']
        print(f"  Intra prediction (CRF=30):{r_intra30:.2f}x ({(r_intra30-1)*100:.1f}% from intra)")

    # Save
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.makedirs('results/decomposition', exist_ok=True)
    with open('results/decomposition/results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to results/decomposition/results.json")


if __name__ == '__main__':
    main()
