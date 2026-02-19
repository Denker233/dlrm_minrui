#!/usr/bin/env python3
"""
Algorithm Isolation with AUC — adds inference to all configs.
Encodes, decodes, replaces weights, runs inference for each config.
"""

import os
import sys
import time
import json
import tempfile
import subprocess
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlrm_data_pytorch as dp
from dlrm_s_pytorch import DLRM_Net, tile_embeddings, untile_embeddings

MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
DATA_FILE = "./input/train.txt"
RESULTS_DIR = os.path.expanduser("~/experiment-control")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 16384

MULTI_FRAME_W = 512
MULTI_FRAME_H = 1024
MIN_WIDTH = 64
MIN_HEIGHT = 64
MAX_DIM = 16384
TILING_THRESHOLD = 50000
TILE_SIZE = 4


def drop_caches():
    try:
        subprocess.run(['sync'], check=True)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], check=True)
        print("  [CACHE] Dropped caches")
    except Exception as e:
        print(f"  [CACHE] Warning: {e}")


def create_args():
    class Args: pass
    a = Args()
    a.arch_sparse_feature_size = ARCH_SPARSE_FEATURE_SIZE
    a.arch_mlp_bot = ARCH_MLP_BOT
    a.arch_mlp_top = ARCH_MLP_TOP
    a.arch_interaction_op = "dot"
    a.arch_interaction_itself = False
    a.data_generation = "dataset"
    a.data_set = "kaggle"
    a.raw_data_file = DATA_FILE
    a.processed_data_file = PROCESSED_DATA
    a.loss_function = "bce"
    a.max_ind_range = -1
    a.test_mini_batch_size = TEST_BATCH_SIZE
    a.test_num_workers = 0
    a.num_workers = 0
    a.mlperf_logging = False
    a.memory_map = False
    a.data_randomize = "total"
    a.data_trace_enable_padding = False
    a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10
    a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 128
    a.round_targets = True
    a.mlperf_bin_loader = False
    a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False
    return a


def load_model_and_data():
    print("Loading model and data...")
    args = create_args()
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    ln_emb = np.array(train_data.counts)
    m_spa = args.arch_sparse_feature_size
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + args.arch_mlp_top, dtype=int, sep="-")
    dlrm = DLRM_Net(m_spa, ln_emb, ln_bot, ln_top,
                     arch_interaction_op="dot", arch_interaction_itself=False,
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2,
                     loss_function="bce")
    ld_model = torch.load(MODEL_PATH, map_location='cpu')
    dlrm.load_state_dict(ld_model["state_dict"])
    dlrm.eval()
    return dlrm, test_ld


def run_inference(dlrm, test_ld):
    all_scores, all_targets = [], []
    test_accu, test_samp = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(test_ld):
            X, lS_o, lS_i, T = batch
            Z = dlrm(X, lS_o, lS_i)
            S = Z.detach().cpu().numpy().flatten()
            T_np = T.detach().cpu().numpy().flatten()
            test_accu += np.sum((np.round(S, 0) == T_np).astype(np.uint8))
            test_samp += T_np.shape[0]
            all_scores.extend(S.tolist())
            all_targets.extend(T_np.tolist())
    elapsed = time.time() - t0
    return test_accu / test_samp, roc_auc_score(all_targets, all_scores), elapsed


def quantize_table(weights):
    w_min, w_max = weights.min().item(), weights.max().item()
    scale = (w_max - w_min) / 255.0
    if scale == 0: scale = 1.0
    zp = round(-w_min / scale)
    q = ((weights / scale).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, scale, zp


def dequantize(q_uint8, scale, zp):
    return (q_uint8.float() - zp) * scale


def prepare_single_frame(pixels_np, num_emb, emb_dim):
    width, height = emb_dim, num_emb
    tiling_meta = {'tiled': False}
    if num_emb > TILING_THRESHOLD:
        data_flat = pixels_np.reshape(-1)
        image, grid_size, tiles_per_emb = tile_embeddings(data_flat, emb_dim, num_emb, TILE_SIZE)
        width, height = image.shape[1], image.shape[0]
        raw_data = image.tobytes()
        tiling_meta = {'tiled': True, 'grid_size': grid_size,
                       'tiles_per_emb': tiles_per_emb, 'tile_size': TILE_SIZE}
    else:
        raw_data = pixels_np.tobytes()

    total_pixels = width * height
    if height > MAX_DIM or width < MIN_WIDTH or height < MIN_HEIGHT:
        min_required = MIN_WIDTH * MIN_HEIGHT
        if total_pixels < min_required:
            width, height = MIN_WIDTH, MIN_HEIGHT
        elif height > MAX_DIM:
            width = (total_pixels + MAX_DIM - 1) // MAX_DIM
            height = MAX_DIM
            if width < MIN_WIDTH:
                width = MIN_WIDTH
                height = (total_pixels + width - 1) // width
        elif width < MIN_WIDTH:
            width = MIN_WIDTH
            height = (total_pixels + width - 1) // width
            if height < MIN_HEIGHT: height = MIN_HEIGHT
        elif height < MIN_HEIGHT:
            height = MIN_HEIGHT
            width = (total_pixels + height - 1) // height
            if width < MIN_WIDTH:
                width, height = MIN_WIDTH, MIN_HEIGHT
        padded_pixels = width * height
        if padded_pixels > len(raw_data):
            padded = bytearray(padded_pixels)
            padded[:len(raw_data)] = raw_data
            raw_data = bytes(padded)
    return raw_data, width, height, tiling_meta


def decode_single_frame(compressed, width, height, num_emb, emb_dim, tiling_meta, container='mp4'):
    with tempfile.TemporaryDirectory() as tmpdir:
        ext = 'mkv' if container == 'mkv' else 'mp4'
        vf = os.path.join(tmpdir, f'v.{ext}')
        rf = os.path.join(tmpdir, 'dec.raw')
        with open(vf, 'wb') as f: f.write(compressed)
        subprocess.run(['ffmpeg', '-y', '-i', vf, '-pix_fmt', 'gray', '-f', 'rawvideo', rf],
                       capture_output=True, check=True)
        pixels = np.fromfile(rf, dtype=np.uint8)

    if tiling_meta.get('tiled'):
        gs = tiling_meta['grid_size']
        ts = tiling_meta['tile_size']
        img_size = gs * ts
        pixels = untile_embeddings(pixels[:img_size*img_size].reshape(img_size, img_size),
                                   emb_dim, num_emb, gs, tiling_meta['tiles_per_emb'], ts)
    else:
        pixels = pixels[:num_emb * emb_dim]
    return torch.from_numpy(pixels.copy()).reshape(num_emb, emb_dim)


# ============================================================
# SINGLE-FRAME: encode + decode + inference per config
# ============================================================
def run_single_frame_config(name, codec_args, container, dlrm, test_ld, state_dict, emb_keys):
    num_tables = len(emb_keys)
    total_compressed = 0
    total_ct = 0.0

    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        num_emb, emb_dim = w.shape
        quantized, scale, zp = quantize_table(w)
        pixels = quantized.numpy()
        raw_data, width, height, tiling_meta = prepare_single_frame(pixels, num_emb, emb_dim)

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_file = os.path.join(tmpdir, 'in.raw')
            ext = 'mkv' if container == 'mkv' else 'mp4'
            vid_file = os.path.join(tmpdir, f'out.{ext}')
            with open(raw_file, 'wb') as f: f.write(raw_data)

            cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
                   '-s', f'{width}x{height}', '-r', '1', '-i', raw_file
                   ] + codec_args + ['-frames:v', '1', vid_file]
            t0 = time.time()
            subprocess.run(cmd, capture_output=True, check=True)
            total_ct += time.time() - t0
            with open(vid_file, 'rb') as f: compressed = f.read()
            total_compressed += len(compressed)

        # Decode and replace weights
        decoded_uint8 = decode_single_frame(compressed, width, height, num_emb, emb_dim,
                                            tiling_meta, container)
        decoded_fp32 = dequantize(decoded_uint8, scale, zp)
        with torch.no_grad():
            dlrm.emb_l[t].weight.data = decoded_fp32

    # Run inference
    drop_caches()
    acc, auc, infer_t = run_inference(dlrm, test_ld)

    # Restore original weights
    for t in range(num_tables):
        with torch.no_grad():
            dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

    return total_compressed, total_ct, acc, auc, infer_t


# ============================================================
# MULTI-FRAME: encode + decode + unpack + inference per config
# ============================================================
def run_multiframe_config(name, codec_args, container, dlrm, test_ld, state_dict, emb_keys):
    num_tables = len(emb_keys)
    frame_size = MULTI_FRAME_W * MULTI_FRAME_H

    # Quantize all tables and build frame data + metadata
    table_meta = []
    all_frames = bytearray()
    total_frames = 0

    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        quantized, scale, zp = quantize_table(w)
        pixels = quantized.numpy().reshape(-1)
        num_pixels = len(pixels)
        n_frames = (num_pixels + frame_size - 1) // frame_size
        padded_size = n_frames * frame_size
        if padded_size > num_pixels:
            pixels = np.concatenate([pixels, np.zeros(padded_size - num_pixels, dtype=np.uint8)])
        all_frames.extend(pixels.tobytes())
        table_meta.append({
            'table': t, 'shape': list(w.shape),
            'n_frames': n_frames, 'num_pixels': w.numel(),
            'scale': scale, 'zp': zp,
        })
        total_frames += n_frames

    # Encode
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_file = os.path.join(tmpdir, 'all.raw')
        ext = 'mkv' if container == 'mkv' else 'mp4'
        vid_file = os.path.join(tmpdir, f'out.{ext}')
        dec_file = os.path.join(tmpdir, 'dec.raw')

        with open(raw_file, 'wb') as f: f.write(bytes(all_frames))

        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{MULTI_FRAME_W}x{MULTI_FRAME_H}', '-r', '30',
               '-i', raw_file] + codec_args + [vid_file]
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ct = time.time() - t0
        if result.returncode != 0:
            raise RuntimeError(f"Encode failed: {result.stderr[:500]}")
        with open(vid_file, 'rb') as f: compressed = f.read()
        comp_size = len(compressed)

        # Decode
        t0 = time.time()
        subprocess.run(['ffmpeg', '-y', '-i', vid_file, '-pix_fmt', 'gray',
                        '-f', 'rawvideo', dec_file], capture_output=True, check=True)
        dt = time.time() - t0
        decoded_all = np.fromfile(dec_file, dtype=np.uint8)

    # Unpack per-table and replace weights
    offset = 0
    for tm in table_meta:
        t = tm['table']
        num_emb, emb_dim = tm['shape']
        n_pixels = tm['num_pixels']
        n_frame_pixels = tm['n_frames'] * frame_size
        table_pixels = decoded_all[offset:offset + n_frame_pixels][:n_pixels]
        offset += n_frame_pixels

        decoded_uint8 = torch.from_numpy(table_pixels.copy()).reshape(num_emb, emb_dim)
        decoded_fp32 = dequantize(decoded_uint8, tm['scale'], tm['zp'])
        with torch.no_grad():
            dlrm.emb_l[t].weight.data = decoded_fp32

    # Run inference
    drop_caches()
    acc, auc, infer_t = run_inference(dlrm, test_ld)

    # Restore original weights
    for t in range(num_tables):
        with torch.no_grad():
            dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

    return comp_size, ct, dt, acc, auc, infer_t, total_frames


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 80)
    print("ALGORITHM ISOLATION WITH AUC (RETRAINED MODEL)")
    print("=" * 80)

    dlrm, test_ld = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]
    num_tables = len(emb_keys)
    total_fp32 = sum(state_dict[k].numel() * 4 for k in emb_keys)
    total_uint8 = sum(state_dict[k].numel() for k in emb_keys)

    print(f"  Tables: {num_tables}, FP32: {total_fp32/1024/1024:.2f} MB, UINT8: {total_uint8/1024/1024:.2f} MB")

    # Baseline
    print("\n  Running baseline inference...")
    drop_caches()
    bl_acc, bl_auc, bl_infer = run_inference(dlrm, test_ld)
    print(f"  BASELINE: acc={bl_acc*100:.4f}%, AUC={bl_auc:.6f}, infer={bl_infer:.2f}s")

    # ================================================================
    # Single-frame configs
    # ================================================================
    sf_configs = [
        ("Full H.265 (CRF 23)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1']),
        ("Entropy only (FFV1)", 'mkv',
         ['-c:v', 'ffv1']),
        ("Lossless H.265 (CRF 0)", 'mp4',
         ['-c:v', 'libx265', '-x265-params', 'lossless=1:log-level=error',
          '-preset', 'ultrafast']),
        ("H.264 (CRF 23)", 'mp4',
         ['-c:v', 'libx264', '-crf', '23', '-preset', 'ultrafast']),
    ]

    sf_results = []
    for name, container, codec_args in sf_configs:
        print(f"\n{'='*60}")
        print(f"SINGLE-FRAME: {name}")
        print(f"{'='*60}")
        comp_size, ct, acc, auc, infer_t = run_single_frame_config(
            name, codec_args, container, dlrm, test_ld, state_dict, emb_keys)
        auc_loss = bl_auc - auc
        r = {
            'name': name, 'compressed_bytes': comp_size,
            'compressed_mb': comp_size / 1024 / 1024,
            'ratio_vs_fp32': total_fp32 / comp_size,
            'ratio_vs_uint8': total_uint8 / comp_size,
            'compress_time': ct,
            'accuracy': acc, 'auc': auc,
            'auc_loss': auc_loss, 'auc_loss_pp': auc_loss * 100,
            'inference_time': infer_t,
        }
        sf_results.append(r)
        print(f"  {name}: {comp_size/1024/1024:.2f} MB, {total_fp32/comp_size:.1f}x vs FP32, "
              f"AUC={auc:.6f} (loss {auc_loss*100:.4f}pp), ct={ct:.2f}s, infer={infer_t:.2f}s")

    # ================================================================
    # Multi-frame configs
    # ================================================================
    mf_configs = [
        ("I-frames only (keyint=1, CRF 23)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1:keyint=1']),
        ("With inter-frame (keyint=250, CRF 23)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1:keyint=250']),
        ("With inter-frame (keyint=999, CRF 23)", 'mp4',
         ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
          '-x265-params', 'log-level=error:allow-non-conformance=1:keyint=999']),
        ("I-frames only lossless (keyint=1, CRF 0)", 'mp4',
         ['-c:v', 'libx265', '-preset', 'ultrafast',
          '-x265-params', 'lossless=1:log-level=error:keyint=1']),
        ("With inter lossless (keyint=250, CRF 0)", 'mp4',
         ['-c:v', 'libx265', '-preset', 'ultrafast',
          '-x265-params', 'lossless=1:log-level=error:keyint=250']),
    ]

    mf_results = []
    for name, container, codec_args in mf_configs:
        print(f"\n{'='*60}")
        print(f"MULTI-FRAME: {name}")
        print(f"{'='*60}")
        try:
            comp_size, ct, dt, acc, auc, infer_t, nf = run_multiframe_config(
                name, codec_args, container, dlrm, test_ld, state_dict, emb_keys)
            auc_loss = bl_auc - auc
            r = {
                'name': name, 'compressed_bytes': comp_size,
                'compressed_mb': comp_size / 1024 / 1024,
                'ratio_vs_fp32': total_fp32 / comp_size,
                'ratio_vs_uint8': total_uint8 / comp_size,
                'compress_time': ct, 'decompress_time': dt,
                'accuracy': acc, 'auc': auc,
                'auc_loss': auc_loss, 'auc_loss_pp': auc_loss * 100,
                'inference_time': infer_t, 'num_frames': nf,
            }
            mf_results.append(r)
            print(f"  {name}: {comp_size/1024/1024:.2f} MB, {total_fp32/comp_size:.1f}x vs FP32, "
                  f"AUC={auc:.6f} (loss {auc_loss*100:.4f}pp), ct={ct:.2f}s, dt={dt:.2f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            mf_results.append({'name': name, 'failed': True, 'error': str(e)})

    # ================================================================
    # Write results
    # ================================================================
    out_path = os.path.join(RESULTS_DIR, "algorithm_isolation_retrained.md")
    with open(out_path, 'w') as f:
        f.write("# Algorithm Isolation with AUC: Retrained Model\n\n")
        f.write("## Baseline\n\n")
        f.write(f"- **FP32 size:** {total_fp32:,} bytes ({total_fp32/1024/1024:.2f} MB)\n")
        f.write(f"- **UINT8 size:** {total_uint8:,} bytes ({total_uint8/1024/1024:.2f} MB)\n")
        f.write(f"- **Baseline accuracy:** {bl_acc*100:.4f}%\n")
        f.write(f"- **Baseline AUC:** {bl_auc:.6f}\n")
        f.write(f"- **Baseline inference:** {bl_infer:.2f}s\n\n")

        f.write("## Single-Frame Results (each table = 1 frame)\n\n")
        f.write("| Config | Size (MB) | Ratio vs FP32 | AUC | AUC Loss (pp) | Compress (s) | Inference (s) |\n")
        f.write("|--------|----------|--------------|------|--------------|-------------|---------------|\n")
        for r in sf_results:
            f.write(f"| {r['name']} | {r['compressed_mb']:.2f} | "
                    f"{r['ratio_vs_fp32']:.2f}x | {r['auc']:.6f} | "
                    f"{r['auc_loss_pp']:.4f} | {r['compress_time']:.2f} | "
                    f"{r['inference_time']:.2f} |\n")

        f.write(f"\n## Multi-Frame Results (all tables in one video, 1051 frames of {MULTI_FRAME_W}x{MULTI_FRAME_H})\n\n")
        f.write("| Config | Size (MB) | Ratio vs FP32 | AUC | AUC Loss (pp) | Compress (s) | Decompress (s) | Inference (s) |\n")
        f.write("|--------|----------|--------------|------|--------------|-------------|---------------|---------------|\n")
        for r in mf_results:
            if not r.get('failed'):
                f.write(f"| {r['name']} | {r['compressed_mb']:.2f} | "
                        f"{r['ratio_vs_fp32']:.2f}x | {r['auc']:.6f} | "
                        f"{r['auc_loss_pp']:.4f} | {r['compress_time']:.2f} | "
                        f"{r.get('decompress_time', 0):.2f} | {r['inference_time']:.2f} |\n")

        # Algorithm contribution with AUC
        ffv1 = next((r for r in sf_results if 'FFV1' in r['name']), None)
        h265_l = next((r for r in sf_results if 'CRF 0' in r['name']), None)
        h265_23 = next((r for r in sf_results if 'CRF 23' in r['name'] and 'H.265' in r['name']), None)
        i_only_23 = next((r for r in mf_results if 'keyint=1' in r['name'] and 'CRF 23' in r['name'] and not r.get('failed')), None)
        inter_250_23 = next((r for r in mf_results if 'keyint=250' in r['name'] and 'CRF 23' in r['name'] and not r.get('failed')), None)
        i_only_l = next((r for r in mf_results if 'keyint=1' in r['name'] and 'CRF 0' in r['name'] and not r.get('failed')), None)
        inter_250_l = next((r for r in mf_results if 'keyint=250' in r['name'] and 'CRF 0' in r['name'] and not r.get('failed')), None)

        f.write("\n## Algorithm Contribution (with AUC)\n\n")
        if ffv1 and h265_l and h265_23:
            total_savings = total_uint8 - h265_23['compressed_bytes']
            ent_savings = total_uint8 - ffv1['compressed_bytes']
            intra_savings = ffv1['compressed_bytes'] - h265_l['compressed_bytes']
            lossy_savings = h265_l['compressed_bytes'] - h265_23['compressed_bytes']

            f.write("### Single-Frame Breakdown (uint8 → H.265 CRF 23)\n\n")
            f.write("| Component | Bytes Saved | % of Savings | AUC After | AUC Loss (pp) |\n")
            f.write("|-----------|-------------|-------------|-----------|---------------|\n")
            f.write(f"| Entropy (FFV1) | {ent_savings:,} | {ent_savings/total_savings*100:.1f}% | "
                    f"{ffv1['auc']:.6f} | {ffv1['auc_loss_pp']:.4f} |\n")
            f.write(f"| Intra prediction (H.265 lossless vs FFV1) | {intra_savings:,} | "
                    f"{intra_savings/total_savings*100:.1f}% | {h265_l['auc']:.6f} | "
                    f"{h265_l['auc_loss_pp']:.4f} |\n")
            f.write(f"| Lossy quantization (CRF 23 vs lossless) | {lossy_savings:,} | "
                    f"{lossy_savings/total_savings*100:.1f}% | {h265_23['auc']:.6f} | "
                    f"{h265_23['auc_loss_pp']:.4f} |\n")
            f.write(f"| **Total** | **{total_savings:,}** | **100%** | | |\n")

        if i_only_23 and inter_250_23:
            f.write("\n### Inter-Frame Prediction (CRF 23)\n\n")
            savings = i_only_23['compressed_bytes'] - inter_250_23['compressed_bytes']
            pct = savings / i_only_23['compressed_bytes'] * 100
            f.write(f"| Config | Size (MB) | Ratio vs FP32 | AUC | AUC Loss (pp) |\n")
            f.write(f"|--------|----------|--------------|------|---------------|\n")
            f.write(f"| I-only (keyint=1) | {i_only_23['compressed_mb']:.2f} | "
                    f"{i_only_23['ratio_vs_fp32']:.2f}x | {i_only_23['auc']:.6f} | "
                    f"{i_only_23['auc_loss_pp']:.4f} |\n")
            f.write(f"| With inter (keyint=250) | {inter_250_23['compressed_mb']:.2f} | "
                    f"{inter_250_23['ratio_vs_fp32']:.2f}x | {inter_250_23['auc']:.6f} | "
                    f"{inter_250_23['auc_loss_pp']:.4f} |\n")
            f.write(f"\nInter-frame saves {savings:,} bytes ({pct:.1f}% of I-only size).\n")
            f.write(f"AUC impact: {inter_250_23['auc_loss_pp'] - i_only_23['auc_loss_pp']:.4f}pp additional loss.\n")

        if i_only_l and inter_250_l:
            f.write("\n### Inter-Frame Prediction (Lossless)\n\n")
            savings = i_only_l['compressed_bytes'] - inter_250_l['compressed_bytes']
            pct = savings / i_only_l['compressed_bytes'] * 100
            f.write(f"| Config | Size (MB) | Ratio vs FP32 | AUC | AUC Loss (pp) |\n")
            f.write(f"|--------|----------|--------------|------|---------------|\n")
            f.write(f"| I-only lossless | {i_only_l['compressed_mb']:.2f} | "
                    f"{i_only_l['ratio_vs_fp32']:.2f}x | {i_only_l['auc']:.6f} | "
                    f"{i_only_l['auc_loss_pp']:.4f} |\n")
            f.write(f"| With inter lossless | {inter_250_l['compressed_mb']:.2f} | "
                    f"{inter_250_l['ratio_vs_fp32']:.2f}x | {inter_250_l['auc']:.6f} | "
                    f"{inter_250_l['auc_loss_pp']:.4f} |\n")
            f.write(f"\nInter-frame at lossless: {savings:,} bytes ({pct:.1f}%).\n")

    print(f"\nResults written to: {out_path}")

    json_path = os.path.join(RESULTS_DIR, "algorithm_isolation_retrained.json")
    with open(json_path, 'w') as f:
        json.dump({
            'baseline': {'accuracy': bl_acc, 'auc': bl_auc, 'inference_time': bl_infer,
                         'total_fp32': total_fp32, 'total_uint8': total_uint8},
            'single_frame': sf_results,
            'multi_frame': mf_results,
        }, f, indent=2, default=str)
    print(f"JSON saved to: {json_path}")

    print("\n" + "=" * 80)
    print("ALGORITHM ISOLATION WITH AUC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
