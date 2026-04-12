#!/usr/bin/env python3
"""
AV1 CRF sweep — AUC vs compression Pareto vs H.265.

For each codec (H.265, AV1) and each CRF:
  1. Encode COLD rows as compressed frames
  2. Decode them back, reconstruct table = HOT fp32 + dequantized cold
  3. Measure AUC, save compression ratio

This is the fair "hot=fp32, cold=compressed" scenario, matching paper headline.
Both codecs use the SAME pipeline so comparison is apples-to-apples.
"""
import os, sys, time, json, subprocess, tempfile, shutil
import concurrent.futures
import numpy as np
import torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
import compressed_emb as _C
from sklearn.metrics import roc_auc_score
from codec_ondemand_benchmark import (
    load_model_and_data, quantize_table, REORDER_DIR
)

# ============================================================
# Config
# ============================================================
TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
HOT_FRACTION = 0.043
EMB_DIM = 16
WIDTH, HEIGHT = 1920, 1080
TILE_W, TILE_H = 4, 4
RPF = (WIDTH // TILE_W) * (HEIGHT // TILE_H)  # 129600

# CRF sweep
H265_CRFS = [0, 10, 18, 23, 28, 35, 45]
AV1_CRFS  = [0, 10, 20, 30, 40, 50, 63]
PARALLEL_ENCODE_WORKERS = 16

# ============================================================
# Encode helpers
# ============================================================

def encode_one_frame(args):
    """args = (frame_bytes, w, h, codec, crf, out_path)"""
    frame_bytes, w, h, codec, crf, out_path = args
    if codec == 'h265':
        if crf == 0:
            x265_params = 'keyint=1:min-keyint=1:lossless=1:log-level=error:no-deblock=1:no-sao=1'
        else:
            x265_params = f'keyint=1:min-keyint=1:crf={crf}:log-level=error:no-deblock=1:no-sao=1'
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{w}x{h}', '-r', '1', '-i', 'pipe:0',
               '-c:v', 'libx265', '-preset', 'medium', '-pix_fmt', 'gray',
               '-x265-params', x265_params,
               '-f', 'matroska', out_path]
    elif codec == 'av1':
        if crf == 0:
            cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                   '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{w}x{h}', '-r', '1', '-i', 'pipe:0',
                   '-c:v', 'libaom-av1', '-pix_fmt', 'gray',
                   '-cpu-used', '8', '-g', '1',
                   '-aom-params', 'lossless=1',
                   '-f', 'matroska', out_path]
        else:
            cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                   '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{w}x{h}', '-r', '1', '-i', 'pipe:0',
                   '-c:v', 'libaom-av1', '-pix_fmt', 'gray',
                   '-cpu-used', '8', '-crf', str(crf), '-g', '1',
                   '-f', 'matroska', out_path]
    else:
        raise ValueError(f"Unknown codec: {codec}")

    proc = subprocess.run(cmd, input=frame_bytes, capture_output=True, timeout=300)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return out_path, 0, proc.stderr.decode()[-300:]
    return out_path, os.path.getsize(out_path), None


def encode_table(q_uint8_rows, codec, crf, table_id, frame_dir):
    """Encode all cold rows as frames in parallel."""
    n_rows = q_uint8_rows.shape[0]
    n_frames = (n_rows + RPF - 1) // RPF

    # Pad
    padded_rows = n_frames * RPF
    if n_rows < padded_rows:
        padded = torch.zeros(padded_rows, EMB_DIM, dtype=torch.uint8)
        padded[:n_rows] = q_uint8_rows
        q_uint8_rows = padded

    tiled_frames = _C.fused_quantize_tile_multiframe(q_uint8_rows, WIDTH, HEIGHT)

    ext = 'h265' if codec == 'h265' else 'av1.mkv'
    tasks = []
    frame_paths = []
    for fi in range(n_frames):
        out_path = os.path.join(frame_dir, f't{table_id}_f{fi:04d}.{ext}')
        frame_bytes = tiled_frames[fi].numpy().tobytes()
        tasks.append((frame_bytes, WIDTH, HEIGHT, codec, crf, out_path))
        frame_paths.append(out_path)

    total_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_ENCODE_WORKERS) as ex:
        for path, sz, err in ex.map(encode_one_frame, tasks):
            if err is not None:
                raise RuntimeError(f"{codec.upper()} encode failed for {path}: {err}")
            total_bytes += sz

    return frame_paths, total_bytes, n_frames


def decode_table(frame_paths, n_rows):
    """Decode frames via C++ pool path, untile, return (n_rows, EMB_DIM) uint8 tensor."""
    decoded = _C.batch_decode_fast(frame_paths, 0, len(frame_paths), True, False, False)
    rows_list = []
    for tiled in decoded:
        rows = _C.untile_frame_to_rows(tiled, RPF)
        rows_list.append(rows)
    all_rows = torch.cat(rows_list, dim=0)[:n_rows]
    return all_rows


def run_auc(dlrm, test_ld):
    all_scores, all_labels = [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_ld:
            Z = dlrm(X, lS_o, lS_i)
            all_scores.append(Z.detach().cpu().numpy().flatten())
            all_labels.append(T.numpy().flatten())
    return roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores))


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("CODEC CRF SWEEP — AUC vs compression Pareto (H.265 vs AV1)")
    print("=" * 70)
    print(f"H.265 CRFs: {H265_CRFS}")
    print(f"AV1 CRFs:   {AV1_CRFS}")
    print(f"Tables: {TABLES} (hot fraction = {HOT_FRACTION})")
    print()

    # Load model + data
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = dlrm.state_dict()
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))

    # Baseline AUC
    print("Computing baseline AUC...")
    t0 = time.time()
    baseline_auc = run_auc(dlrm, test_ld)
    print(f"  Baseline AUC: {baseline_auc:.6f} ({time.time()-t0:.1f}s)")

    # Pre-compute hot/cold split + quantization params for each table
    print("\nLoading hot/cold splits + quantizing cold rows...")
    table_data = {}
    total_cold_rows = 0
    for t in TABLES:
        w_orig = state_dict[emb_keys[t]].clone()
        n = w_orig.shape[0]

        cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
        is_hot = torch.load(f'results/hotcold/is_hot_{t}.pt', weights_only=False)
        hot_idx = torch.where(is_hot)[0].numpy()

        # Quantize ONLY cold rows (own min/max for cold)
        cold_w = w_orig[cold_order]
        q_cold, s, zp = quantize_table(cold_w)
        # q_cold is (n_cold, EMB_DIM) uint8 in cold_order

        table_data[t] = {
            'w_orig': w_orig,
            'n': n,
            'cold_order': cold_order,
            'hot_idx': hot_idx,
            's': s,
            'zp': zp,
            'q_cold': q_cold,
            'n_cold': len(cold_order),
        }
        total_cold_rows += len(cold_order)
        print(f"  table {t}: n={n}, hot={len(hot_idx)}, cold={len(cold_order)}, s={s:.4g}, zp={zp}")

    total_cold_fp32_mb = total_cold_rows * EMB_DIM * 4 / 1024 / 1024
    total_cold_uint8_mb = total_cold_rows * EMB_DIM / 1024 / 1024
    print(f"\nTotal cold storage: fp32={total_cold_fp32_mb:.1f} MB, uint8={total_cold_uint8_mb:.1f} MB")

    # ============================================================
    # SWEEP
    # ============================================================
    results = {
        'baseline_auc': baseline_auc,
        'total_cold_fp32_mb': total_cold_fp32_mb,
        'total_cold_uint8_mb': total_cold_uint8_mb,
        'codecs': {},
    }
    tmpdir = tempfile.mkdtemp(prefix='codec_sweep_')
    print(f"\nTemp dir: {tmpdir}")

    for codec, crfs in [('h265', H265_CRFS), ('av1', AV1_CRFS)]:
        print(f"\n{'#'*70}\n# CODEC: {codec.upper()}\n{'#'*70}")
        results['codecs'][codec] = {}

        for crf in crfs:
            crf_dir = os.path.join(tmpdir, f'{codec}_crf{crf}')
            os.makedirs(crf_dir, exist_ok=True)

            print(f"\n  --- {codec.upper()} CRF={crf} ---")
            crf_t0 = time.time()

            # Encode + decode each table, reconstruct
            total_compressed = 0
            reconstructed = {}
            for t in TABLES:
                td = table_data[t]
                t_enc0 = time.time()
                frame_paths, sz, nf = encode_table(
                    td['q_cold'], codec, crf, t, crf_dir)
                t_enc = time.time() - t_enc0
                total_compressed += sz

                # Decode back
                recon_uint8 = decode_table(frame_paths, td['n_cold'])

                # Dequantize cold rows
                recon_cold_fp32 = (recon_uint8.float() - td['zp']) * td['s']

                # Reconstruct full table: hot=fp32 (original), cold=dequantized
                recon_full = td['w_orig'].clone()  # start with original (preserves hot)
                recon_full[td['cold_order']] = recon_cold_fp32
                reconstructed[t] = recon_full

                print(f"    table {t}: {nf}f, {sz/1024:>8.1f}KB enc, "
                      f"enc={t_enc:.1f}s")

            # Replace weights in model
            for t in TABLES:
                dlrm.emb_l[t].weight.data = reconstructed[t]

            # Measure AUC
            auc = run_auc(dlrm, test_ld)
            delta = auc - baseline_auc

            # Restore
            for t in TABLES:
                dlrm.emb_l[t].weight.data = table_data[t]['w_orig']

            compressed_mb = total_compressed / 1024 / 1024
            ratio_uint8 = total_cold_uint8_mb / compressed_mb if compressed_mb > 0 else 0
            ratio_fp32 = total_cold_fp32_mb / compressed_mb if compressed_mb > 0 else 0
            crf_time = time.time() - crf_t0

            print(f"  ==> {codec.upper()} CRF={crf}: {compressed_mb:.3f} MB, "
                  f"{ratio_uint8:.0f}x uint8 ({ratio_fp32:.0f}x fp32)")
            print(f"      AUC={auc:.6f} (delta {delta:+.6f}, {delta*100:+.4f}%)  [{crf_time:.0f}s]")

            results['codecs'][codec][str(crf)] = {
                'crf': crf,
                'compressed_mb': compressed_mb,
                'compressed_bytes': total_compressed,
                'ratio_uint8': ratio_uint8,
                'ratio_fp32': ratio_fp32,
                'auc': auc,
                'delta': delta,
                'time_sec': crf_time,
            }

            # Save incrementally
            out_path = 'results/codec_crf_sweep.json'
            os.makedirs('results', exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Clean up encoded frames to save disk
            shutil.rmtree(crf_dir)

    # Print final table
    print(f"\n{'='*78}")
    print(f"SUMMARY: hot=fp32 (top {HOT_FRACTION*100:.1f}%), cold=compressed")
    print(f"{'='*78}")
    print(f"{'codec':>6} {'CRF':>5} {'MB':>10} {'uint8 ratio':>12} {'fp32 ratio':>12} {'AUC':>10} {'delta %':>10}")
    print(f"{'fp32':>6} {'-':>5} {total_cold_fp32_mb:>10.2f} {'1x':>12} {'1x':>12} {baseline_auc:>10.6f} {'0.0000':>10}")
    for codec, crfs in [('h265', H265_CRFS), ('av1', AV1_CRFS)]:
        for crf in crfs:
            r = results['codecs'][codec].get(str(crf))
            if r is None: continue
            print(f"{codec:>6} {crf:>5} {r['compressed_mb']:>10.3f} {r['ratio_uint8']:>11.0f}x "
                  f"{r['ratio_fp32']:>11.0f}x {r['auc']:>10.6f} {r['delta']*100:>+10.4f}")

    print(f"\nResults saved to {out_path}")
    shutil.rmtree(tmpdir)
    print("Done.")


if __name__ == '__main__':
    main()
