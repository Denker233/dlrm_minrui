#!/usr/bin/env python3
"""
Two experiments:
1. CRF sweep (30-51) on single-frame sorted → Pareto curve vs CAFE+
2. Actual model loading/transfer time measurement
"""
import os, sys, time, json, subprocess, gc, tempfile, shutil
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import compressed_emb as _C
from codec_ondemand_benchmark import (
    load_model_and_data, EMB_DIM, MODEL_PATH,
    HOTCOLD_DIR, REORDER_DIR, ONDEMAND_DIR, LARGE_TABLE_THRESHOLD,
    quantize_table,
)

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
TILE_W, TILE_H = 4, 4

# Single-frame dimensions (from comparison_results.json)
SF_DIMS = {
    2: (3840, 40400), 3: (1920, 17568), 9: (1920, 744), 11: (3840, 33304),
    15: (1920, 43556), 20: (1920, 56200), 23: (1920, 2284), 25: (1920, 1140),
}


def encode_single_frame_h265(q_np, width, height, outpath, crf, preset='medium'):
    """Encode uint8 data as single H.265 frame."""
    rpf = (width * height) // EMB_DIM
    n = q_np.shape[0]
    if n < rpf:
        padded = np.zeros((rpf, EMB_DIM), dtype=np.uint8)
        padded[:n] = q_np
        q_np = padded

    # Tile using C++ (reshape to match expected format)
    q_t = torch.from_numpy(q_np[:rpf])
    tiled = _C.fused_quantize_tile_multiframe(q_t, width, height)
    raw = tiled[0].numpy().tobytes()

    x265 = f'keyint=1:min-keyint=1:crf={crf}:log-level=error:no-deblock=1:no-sao=1'
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
           '-s', f'{width}x{height}', '-r', '1', '-i', 'pipe:0',
           '-c:v', 'libx265', '-preset', preset, '-pix_fmt', 'gray',
           '-x265-params', x265, '-f', 'matroska', outpath]
    subprocess.run(cmd, input=raw, capture_output=True, timeout=120)
    return os.path.getsize(outpath) if os.path.exists(outpath) else 0


def decode_and_get_cold_rows(frame_path, width, height, n_rows):
    """Decode H.265 frame and untile to get cold rows."""
    tiled = _C.decode_h265_frame_from_file(frame_path, 1, True, False, False)
    rpf = (width * height) // EMB_DIM
    rows = _C.untile_frame_to_rows(tiled, rpf)
    return rows[:n_rows]


def run_auc(dlrm, test_batches, ln_emb, state_dict, emb_keys,
            is_hot, hot_indices, cold_rows_per_table, cold_scale, cold_zp):
    """Run AUC test with decoded cold rows."""
    nt = len(ln_emb)
    with torch.no_grad():
        for k in emb_keys:
            t = int(k.split('.')[1])
            dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
            w = state_dict[k].clone()
            if t in cold_rows_per_table:
                # Replace cold rows with decoded+dequantized values
                cold_indices = torch.where(~is_hot[t])[0]
                cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
                decoded = cold_rows_per_table[t]  # (n_cold, D) uint8
                # Dequantize
                s, zp = cold_scale[t], cold_zp[t]
                decoded_fp32 = (decoded.float() - zp) * s
                # Map back to original positions
                w[cold_order[:decoded.shape[0]]] = decoded_fp32[:len(cold_order)]
            dlrm.emb_l[t].weight.data = w

    scores, targets = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            Z = dlrm(X, o, i)
            scores.append(Z.numpy().ravel())
            targets.append(T.numpy().ravel())
    return roc_auc_score(np.concatenate(targets), np.concatenate(scores))


def main():
    print("=" * 70)
    print("EXPERIMENT: CRF Pareto Curve + Transfer Time")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    torch.set_num_threads(32)

    test_batches = list(test_ld)
    print(f"{len(test_batches)} test batches")

    # Load hot/cold
    is_hot, hot_indices = {}, {}
    for t in TABLES:
        is_hot[t] = torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]

    # Load cold data (sorted)
    cold_data, cold_scale, cold_zp, cold_n = {}, {}, {}, {}
    for t in TABLES:
        cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
        w = state_dict[ek[t]][cold_order]
        q, s, zp = quantize_table(w)
        cold_data[t] = q.numpy()
        cold_scale[t] = s
        cold_zp[t] = zp
        cold_n[t] = len(cold_order)

    total_fp32 = sum(cold_n[t] * EMB_DIM * 4 for t in TABLES)
    total_uint8 = sum(cold_n[t] * EMB_DIM for t in TABLES)
    print(f"Total cold: {total_fp32/1024/1024:.0f}MB fp32, {total_uint8/1024/1024:.0f}MB uint8")

    # Baseline AUC
    print("\n--- Baseline AUC ---")
    auc_base = run_auc(dlrm, test_batches, ln_emb, state_dict, ek,
                       is_hot, hot_indices, {}, cold_scale, cold_zp)
    print(f"  Baseline fp32: AUC={auc_base:.6f}")

    # ================================================================
    # Part 1: CRF Sweep
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 1: CRF SWEEP (single-frame, sorted)")
    print(f"{'='*70}")

    crfs = [0, 10, 18, 25, 30, 35, 40, 45, 51]
    results = {'baseline_auc': float(auc_base), 'total_fp32': total_fp32,
               'total_uint8': total_uint8, 'crf_sweep': {}}

    tmpdir = tempfile.mkdtemp(prefix='crf_sweep_')
    print(f"Temp dir: {tmpdir}")

    for crf in crfs:
        print(f"\n--- CRF={crf} ---")
        total_comp = 0
        cold_decoded = {}

        for t in TABLES:
            w, h = SF_DIMS[t]
            outpath = os.path.join(tmpdir, f't{t}_crf{crf}.h265')
            comp_bytes = encode_single_frame_h265(cold_data[t], w, h, outpath, crf)
            total_comp += comp_bytes

            # Decode for AUC
            if comp_bytes > 0:
                decoded = decode_and_get_cold_rows(outpath, w, h, cold_n[t])
                cold_decoded[t] = decoded
                # Compute error vs original
                orig = torch.from_numpy(cold_data[t][:cold_n[t]])
                err = (decoded.float() - orig.float()).abs()
                max_err = err.max().item()
                mean_err = err.mean().item()
                psnr = 10 * np.log10(255**2 / (err**2).mean().item()) if mean_err > 0 else float('inf')
                print(f"  t{t}: {comp_bytes/1024:.1f}KB, PSNR={psnr:.1f}, maxerr={max_err:.0f}, meanerr={mean_err:.3f}")
            os.remove(outpath) if os.path.exists(outpath) else None

        ratio_fp32 = total_fp32 / total_comp if total_comp > 0 else 0
        ratio_uint8 = total_uint8 / total_comp if total_comp > 0 else 0
        print(f"  Total: {total_comp/1024:.1f}KB, {ratio_fp32:.0f}x vs fp32, {ratio_uint8:.0f}x vs uint8")

        # AUC
        auc = run_auc(dlrm, test_batches, ln_emb, state_dict, ek,
                      is_hot, hot_indices, cold_decoded, cold_scale, cold_zp)
        auc_loss = (auc - auc_base) * 100
        print(f"  AUC={auc:.6f} (loss={auc_loss:+.4f}%)")

        results['crf_sweep'][str(crf)] = {
            'compressed_bytes': total_comp,
            'ratio_fp32': float(ratio_fp32),
            'ratio_uint8': float(ratio_uint8),
            'auc': float(auc),
            'auc_loss_pct': float(auc_loss),
        }
        del cold_decoded; gc.collect()

    shutil.rmtree(tmpdir, ignore_errors=True)

    # Print Pareto table
    print(f"\n{'='*70}")
    print("CRF PARETO TABLE")
    print(f"{'='*70}")
    print(f"{'CRF':>4} {'Size':>10} {'Ratio(fp32)':>12} {'AUC':>10} {'AUC Loss':>10}")
    print("-" * 50)
    for crf in crfs:
        r = results['crf_sweep'][str(crf)]
        print(f"{crf:>4} {r['compressed_bytes']/1024:>9.1f}KB {r['ratio_fp32']:>11.0f}x "
              f"{r['auc']:>10.6f} {r['auc_loss_pct']:>+9.4f}%")

    # ================================================================
    # Part 2: Actual Transfer/Loading Time
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: ACTUAL MODEL LOADING TIME")
    print(f"{'='*70}")

    # Prepare compressed files (CRF=30, single-frame)
    comp_dir = 'results/ondemand/single_frame/h265'
    comp_files = []
    for t in TABLES:
        f = os.path.join(comp_dir, f'table_{t}', 'frame_00000.h265')
        if os.path.exists(f):
            comp_files.append(f)

    comp_total = sum(os.path.getsize(f) for f in comp_files)
    print(f"Compressed files: {len(comp_files)}, total {comp_total/1024:.1f}KB")

    # Also prepare fp32 tensors for uncompressed comparison
    fp32_sizes = {t: state_dict[ek[t]].numel() * 4 for t in TABLES}
    fp32_total = sum(fp32_sizes.values())

    # Test 1: Write compressed to disk + read back
    print("\n--- Test 1: Write/Read Compressed ---")
    test_dir = '/tmp/transfer_test'
    os.makedirs(test_dir, exist_ok=True)

    # Write compressed
    t0 = time.perf_counter()
    for f in comp_files:
        shutil.copy2(f, test_dir)
    t_write_comp = time.perf_counter() - t0

    # Read compressed back
    t0 = time.perf_counter()
    for f in comp_files:
        fname = os.path.basename(os.path.dirname(f)) + '_' + os.path.basename(f)
        with open(os.path.join(test_dir, os.path.basename(f)), 'rb') as fh:
            _ = fh.read()
    t_read_comp = time.perf_counter() - t0

    # Decode compressed (H.265)
    t0 = time.perf_counter()
    for t in TABLES:
        f = os.path.join(comp_dir, f'table_{t}', 'frame_00000.h265')
        if os.path.exists(f):
            w, h = SF_DIMS[t]
            decoded = _C.decode_h265_frame_from_file(f, 1, True, False, False)
    t_decode_h265 = time.perf_counter() - t0

    # Decode with pool (batch)
    t0 = time.perf_counter()
    _C.batch_decode_fast(comp_files, 2, len(comp_files), True, False, False)
    t_decode_h265_pool = time.perf_counter() - t0

    print(f"  Write compressed: {t_write_comp*1000:.1f}ms ({comp_total/1024:.0f}KB)")
    print(f"  Read compressed:  {t_read_comp*1000:.1f}ms")
    print(f"  Decode H.265 (serial): {t_decode_h265*1000:.1f}ms")
    print(f"  Decode H.265 (pool, parallel): {t_decode_h265_pool*1000:.1f}ms")

    # Test 2: Write/Read uncompressed fp32
    print("\n--- Test 2: Write/Read Uncompressed fp32 ---")
    # Save each table's fp32 weights
    t0 = time.perf_counter()
    for t in TABLES:
        torch.save(state_dict[ek[t]], os.path.join(test_dir, f'table_{t}_fp32.pt'))
    t_write_fp32 = time.perf_counter() - t0

    # Read back
    t0 = time.perf_counter()
    for t in TABLES:
        _ = torch.load(os.path.join(test_dir, f'table_{t}_fp32.pt'), weights_only=True)
    t_read_fp32 = time.perf_counter() - t0

    # Raw binary write/read (no torch overhead)
    t0 = time.perf_counter()
    for t in TABLES:
        w = state_dict[ek[t]]
        with open(os.path.join(test_dir, f'table_{t}_raw.bin'), 'wb') as fh:
            fh.write(w.numpy().tobytes())
    t_write_raw = time.perf_counter() - t0

    t0 = time.perf_counter()
    for t in TABLES:
        with open(os.path.join(test_dir, f'table_{t}_raw.bin'), 'rb') as fh:
            data = fh.read()
    t_read_raw = time.perf_counter() - t0

    # Drop page cache and re-read (cold read)
    os.system('sync')
    # Can't drop caches without root, so measure cached reads as lower bound

    print(f"  Write fp32 (torch.save): {t_write_fp32*1000:.0f}ms ({fp32_total/1024/1024:.0f}MB)")
    print(f"  Read fp32 (torch.load):  {t_read_fp32*1000:.0f}ms")
    print(f"  Write fp32 (raw binary): {t_write_raw*1000:.0f}ms")
    print(f"  Read fp32 (raw binary):  {t_read_raw*1000:.0f}ms")

    # Test 3: Full pipeline — read compressed + decode vs read uncompressed
    print("\n--- Test 3: End-to-End Loading Pipeline ---")

    # Compressed: read + decode (parallel)
    times_comp = []
    for _ in range(5):
        t0 = time.perf_counter()
        for f in comp_files:
            with open(f, 'rb') as fh:
                _ = fh.read()
        _C.batch_decode_fast(comp_files, 2, len(comp_files), True, False, False)
        times_comp.append(time.perf_counter() - t0)

    # Uncompressed: just read
    times_uncomp = []
    for _ in range(5):
        t0 = time.perf_counter()
        for t in TABLES:
            with open(os.path.join(test_dir, f'table_{t}_raw.bin'), 'rb') as fh:
                data = fh.read()
        times_uncomp.append(time.perf_counter() - t0)

    t_comp_e2e = np.median(times_comp)
    t_uncomp_e2e = np.median(times_uncomp)

    print(f"  Compressed (read + decode): {t_comp_e2e*1000:.1f}ms (median of 5)")
    print(f"  Uncompressed (read raw):    {t_uncomp_e2e*1000:.1f}ms (median of 5)")
    print(f"  Speedup: {t_uncomp_e2e/t_comp_e2e:.1f}x")

    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\nCRF Pareto (vs CAFE+ ~10,000x @ ~0.75% AUC loss):")
    print(f"{'CRF':>4} {'Ratio':>8} {'AUC Loss':>10} {'Better than CAFE+?':>20}")
    for crf in crfs:
        r = results['crf_sweep'][str(crf)]
        better = "YES (AUC)" if abs(r['auc_loss_pct']) < 0.75 else "NO"
        if r['ratio_fp32'] > 10000:
            better += " + ratio"
        print(f"{crf:>4} {r['ratio_fp32']:>7.0f}x {r['auc_loss_pct']:>+9.4f}% {better:>20}")

    print(f"\nTransfer time:")
    print(f"  Compressed (312KB, read+decode): {t_comp_e2e*1000:.1f}ms")
    print(f"  Uncompressed ({fp32_total/1024/1024:.0f}MB, read): {t_uncomp_e2e*1000:.1f}ms")
    print(f"  Speedup: {t_uncomp_e2e/t_comp_e2e:.1f}x")

    results['transfer'] = {
        'compressed_read_decode_ms': float(t_comp_e2e * 1000),
        'uncompressed_read_ms': float(t_uncomp_e2e * 1000),
        'speedup': float(t_uncomp_e2e / t_comp_e2e),
        'compressed_size_bytes': comp_total,
        'uncompressed_size_bytes': fp32_total,
        'decode_h265_serial_ms': float(t_decode_h265 * 1000),
        'decode_h265_pool_ms': float(t_decode_h265_pool * 1000),
    }

    os.makedirs('results/crf_pareto', exist_ok=True)
    with open('results/crf_pareto/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/crf_pareto/results.json")


if __name__ == '__main__':
    main()
