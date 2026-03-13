#!/usr/bin/env python3
"""
Retrained Model: Uniform Codec Compression Experiment
Uses the fully-trained 1-epoch model for Phase 2A of the retrain task.
Tests 16 CRF levels: 0, 15, 18, 20, 23, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48, 51
"""

import os
import sys
import time
import tempfile
import subprocess
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlrm_data_pytorch as dp
from dlrm_s_pytorch import DLRM_Net, quantize_asymmetric, dequantize_asymmetric
from dlrm_s_pytorch import tile_embeddings, untile_embeddings

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
DATA_FILE = "./input/train.txt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
RESULTS_DIR = os.path.expanduser("~/experiment-control")
STATUS_FILE = os.path.join(RESULTS_DIR, "status.md")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 16384

CRF_LEVELS = [0, 15, 18, 20, 23, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48, 51]


def update_status(msg):
    """Append status message"""
    try:
        with open(STATUS_FILE, 'a') as f:
            f.write(f"\n{msg}")
    except:
        pass
    print(msg, flush=True)


def create_args():
    class Args:
        pass
    args = Args()
    args.arch_sparse_feature_size = ARCH_SPARSE_FEATURE_SIZE
    args.arch_mlp_bot = ARCH_MLP_BOT
    args.arch_mlp_top = ARCH_MLP_TOP
    args.arch_interaction_op = "dot"
    args.arch_interaction_itself = False
    args.data_generation = "dataset"
    args.data_set = "kaggle"
    args.raw_data_file = DATA_FILE
    args.processed_data_file = PROCESSED_DATA
    args.loss_function = "bce"
    args.max_ind_range = -1
    args.test_mini_batch_size = TEST_BATCH_SIZE
    args.test_num_workers = 0
    args.num_workers = 0
    args.mlperf_logging = False
    args.memory_map = False
    args.data_randomize = "total"
    args.data_trace_enable_padding = False
    args.data_sub_sample_rate = 0.0
    args.num_indices_per_lookup = 10
    args.num_indices_per_lookup_fixed = False
    args.mini_batch_size = 128
    args.round_targets = True
    args.mlperf_bin_loader = False
    args.mlperf_bin_shuffle = False
    args.dataset_multiprocessing = False
    return args


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
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    dlrm = DLRM_Net(
        m_spa, ln_emb, ln_bot, ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        loss_function=args.loss_function,
    )

    ld_model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    dlrm.load_state_dict(ld_model["state_dict"])
    dlrm.eval()

    return dlrm, test_ld, ln_emb, ln_bot, ln_top, m_spa, args


def run_inference(dlrm, test_ld):
    """Run inference and return accuracy, AUC, and inference time"""
    all_scores = []
    all_targets = []
    test_accu = 0
    test_samp = 0

    t_start = time.time()
    with torch.no_grad():
        for i, testBatch in enumerate(test_ld):
            X_test = testBatch[0]
            lS_o_test = testBatch[1]
            lS_i_test = testBatch[2]
            T_test = testBatch[3]

            Z_test = dlrm(X_test, lS_o_test, lS_i_test)

            S_test = Z_test.detach().cpu().numpy().flatten()
            T_test_np = T_test.detach().cpu().numpy().flatten()

            mbs_test = T_test_np.shape[0]
            A_test = np.sum((np.round(S_test, 0) == T_test_np).astype(np.uint8))

            test_accu += A_test
            test_samp += mbs_test

            all_scores.extend(S_test.tolist())
            all_targets.extend(T_test_np.tolist())

            if i % 50 == 0:
                print(f"  Batch {i}/{len(test_ld)}", end='\r')

    inference_time = time.time() - t_start
    accuracy = test_accu / test_samp
    auc = roc_auc_score(all_targets, all_scores)
    print(f"  Inference complete: {test_samp} samples in {inference_time:.2f}s")
    return accuracy, auc, inference_time


def compress_table_with_codec(weights, codec_cmd_args, quality, tmpdir_base=None):
    num_emb, emb_dim = weights.shape
    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / 255.0
    zero_point = -(w_min / scale).round() if scale > 0 else 0.0
    quantized = ((weights / scale).round() + zero_point).clamp(0, 255).to(torch.uint8)
    quant_metadata = {'scale': float(scale), 'zero_point': float(zero_point)}
    pixels = quantized.numpy()

    MAX_DIM = 16384
    MIN_WIDTH = 64
    MIN_HEIGHT = 64
    TILE_SIZE = 4
    TILING_THRESHOLD = 50000

    width = emb_dim
    height = num_emb
    tiling_meta = {'tiled': False}

    if num_emb > TILING_THRESHOLD:
        data_flat = pixels.reshape(-1)
        image, grid_size, tiles_per_emb = tile_embeddings(data_flat, emb_dim, num_emb, TILE_SIZE)
        width = image.shape[1]
        height = image.shape[0]
        raw_data = image.tobytes()
        tiling_meta = {'tiled': True, 'grid_size': grid_size, 'tiles_per_emb': tiles_per_emb, 'tile_size': TILE_SIZE}
    else:
        raw_data = pixels.tobytes()

    total_pixels = width * height
    original_width, original_height = width, height

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
            if height < MIN_HEIGHT:
                height = MIN_HEIGHT
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

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_file = os.path.join(tmpdir, 'pixels.raw')
        video_file = os.path.join(tmpdir, 'compressed.mp4')

        with open(raw_file, 'wb') as f:
            f.write(raw_data)

        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
            '-s', f'{width}x{height}', '-r', '1', '-i', raw_file,
        ] + codec_cmd_args + ['-frames:v', '1', video_file]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        compress_time = time.time() - t0

        if result.returncode != 0:
            raise RuntimeError(f"Encoding failed: {result.stderr[:500]}")

        with open(video_file, 'rb') as f:
            compressed_data = f.read()

    metadata = {
        'shape': (num_emb, emb_dim),
        'quant_params': quant_metadata,
        'frame_width': width,
        'frame_height': height,
        'tiling': tiling_meta,
        'original_pixels': num_emb * emb_dim,
    }

    return compressed_data, metadata, compress_time


def decompress_table(compressed_data, metadata):
    num_emb, emb_dim = metadata['shape']

    with tempfile.TemporaryDirectory() as tmpdir:
        video_file = os.path.join(tmpdir, 'compressed.mp4')
        raw_file = os.path.join(tmpdir, 'decoded.raw')

        with open(video_file, 'wb') as f:
            f.write(compressed_data)

        cmd = ['ffmpeg', '-y', '-i', video_file, '-pix_fmt', 'gray', '-f', 'rawvideo', raw_file]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        decompress_time = time.time() - t0

        if result.returncode != 0:
            raise RuntimeError(f"Decoding failed: {result.stderr[:500]}")

        pixels_uint8 = np.fromfile(raw_file, dtype=np.uint8)

    if metadata.get('tiling', {}).get('tiled', False):
        tiling = metadata['tiling']
        grid_size = tiling['grid_size']
        tile_size = tiling['tile_size']
        image_size = grid_size * tile_size
        pixels_uint8 = untile_embeddings(
            pixels_uint8[:image_size * image_size].reshape(image_size, image_size),
            emb_dim, num_emb, grid_size, tiling['tiles_per_emb'], tile_size
        )
    else:
        expected_size = num_emb * emb_dim
        pixels_uint8 = pixels_uint8[:expected_size]

    pixels_uint8 = torch.from_numpy(pixels_uint8.copy()).reshape(num_emb, emb_dim)
    qp = metadata['quant_params']
    weights = (pixels_uint8.float() - qp['zero_point']) * qp['scale']

    return weights, decompress_time


def get_codec_args(crf):
    if crf == 0:
        return ['-c:v', 'libx265', '-x265-params', 'lossless=1:log-level=error', '-preset', 'ultrafast']
    else:
        return ['-c:v', 'libx265', '-crf', str(crf), '-preset', 'ultrafast',
                '-x265-params', 'log-level=error:allow-non-conformance=1']


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 80)
    print("RETRAINED MODEL: UNIFORM CODEC COMPRESSION SWEEP")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"CRF levels: {CRF_LEVELS}")

    # Baseline
    print("\n--- BASELINE ---")
    dlrm_orig, test_ld, ln_emb, ln_bot, ln_top, m_spa, args = load_model_and_data()

    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]
    total_emb_size = sum(state_dict[k].numel() * 4 for k in emb_keys)

    print(f"  Embedding tables: {len(emb_keys)}")
    print(f"  Total embedding size: {total_emb_size / 1024 / 1024:.2f} MB")

    baseline_acc, baseline_auc, baseline_infer_time = run_inference(dlrm_orig, test_ld)
    print(f"  Baseline Accuracy: {baseline_acc * 100:.4f}%")
    print(f"  Baseline AUC: {baseline_auc:.6f}")
    print(f"  Baseline Inference Time: {baseline_infer_time:.2f}s")

    baseline = {
        'accuracy': baseline_acc,
        'auc': baseline_auc,
        'inference_time': baseline_infer_time,
        'total_emb_size': total_emb_size,
        'num_tables': len(emb_keys),
    }

    # CRF sweep
    results = []
    for crf in CRF_LEVELS:
        print(f"\n{'='*60}")
        print(f"CRF = {crf}")
        print(f"{'='*60}")

        codec_args = get_codec_args(crf)

        total_compressed_size = 0
        total_compress_time = 0.0
        total_decompress_time = 0.0

        # Create fresh model
        dlrm_test = DLRM_Net(
            m_spa, ln_emb, ln_bot, ln_top,
            arch_interaction_op="dot", arch_interaction_itself=False,
            sigmoid_bot=-1, sigmoid_top=ln_top.size - 2,
            loss_function="bce",
        )
        dlrm_test.load_state_dict(state_dict)
        dlrm_test.eval()

        for idx, key in enumerate(emb_keys):
            weights = state_dict[key]
            table_size = weights.numel() * 4

            try:
                compressed_data, metadata, ct = compress_table_with_codec(weights, codec_args, crf)
                total_compress_time += ct
                total_compressed_size += len(compressed_data)

                decompressed_weights, dt = decompress_table(compressed_data, metadata)
                total_decompress_time += dt

                with torch.no_grad():
                    dlrm_test.emb_l[idx].weight.data = decompressed_weights

                if idx % 5 == 0:
                    print(f"  Table {idx}/{len(emb_keys)}: {weights.shape} → {len(compressed_data)} bytes ({table_size / len(compressed_data):.1f}x)")
            except Exception as e:
                print(f"  Table {idx} FAILED: {e}")
                total_compressed_size += table_size

        # Run inference with timing
        accuracy, auc, infer_time = run_inference(dlrm_test, test_ld)

        compression_ratio = total_emb_size / total_compressed_size if total_compressed_size > 0 else 0

        result = {
            'crf': crf,
            'compression_ratio': compression_ratio,
            'accuracy': accuracy,
            'auc': auc,
            'compressed_size': total_compressed_size,
            'compress_time': total_compress_time,
            'decompress_time': total_decompress_time,
            'inference_time': infer_time,
            'accuracy_loss': (baseline_acc - accuracy) * 100,
            'auc_loss': baseline_auc - auc,
        }
        results.append(result)

        print(f"\n  CRF {crf}: ratio={compression_ratio:.2f}x, acc={accuracy*100:.4f}%, auc={auc:.6f}, "
              f"size={total_compressed_size/1024/1024:.2f}MB, infer={infer_time:.2f}s")

        # Update status periodically
        if crf in [0, 23, 51]:
            update_status(f"- Uniform CRF {crf}: ratio={compression_ratio:.2f}x, AUC={auc:.6f}")

    # Write results
    write_results(baseline, results)

    print("\n" + "=" * 80)
    print("UNIFORM COMPRESSION SWEEP COMPLETE")
    print("=" * 80)


def write_results(baseline, results):
    path = os.path.join(RESULTS_DIR, "retrained_uniform_results.md")
    with open(path, 'w') as f:
        f.write("# Retrained Model: Uniform Codec Compression Results\n\n")
        f.write("## Baseline (Retrained 1 Full Epoch)\n\n")
        f.write(f"- **Model:** `{MODEL_PATH}`\n")
        f.write(f"- **Embedding tables:** {baseline['num_tables']}\n")
        f.write(f"- **Total embedding size:** {baseline['total_emb_size']} bytes ({baseline['total_emb_size']/1024/1024:.2f} MB)\n")
        f.write(f"- **Baseline accuracy:** {baseline['accuracy']*100:.4f}%\n")
        f.write(f"- **Baseline AUC:** {baseline['auc']:.6f}\n")
        f.write(f"- **Baseline inference time:** {baseline['inference_time']:.2f}s\n\n")

        f.write("## CRF Sweep Results\n\n")
        f.write("| CRF | Comp Ratio | Accuracy (%) | AUC | Acc Loss (pp) | AUC Loss | Size (MB) | Compress (s) | Decompress (s) | Inference (s) |\n")
        f.write("|-----|-----------|-------------|------|--------------|----------|----------|-------------|---------------|---------------|\n")

        for r in results:
            f.write(f"| {r['crf']} | {r['compression_ratio']:.2f}x | {r['accuracy']*100:.4f} | {r['auc']:.6f} | {r['accuracy_loss']:.4f} | {r['auc_loss']:.6f} | {r['compressed_size']/1024/1024:.2f} | {r['compress_time']:.2f} | {r['decompress_time']:.2f} | {r['inference_time']:.2f} |\n")

        # Validation
        f.write("\n## Validation\n\n")
        all_valid = True
        for r in results:
            issues = []
            if r['compression_ratio'] <= 1:
                issues.append(f"ratio={r['compression_ratio']:.2f}")
            if not (0.5 <= r['accuracy'] <= 1.0):
                issues.append(f"acc={r['accuracy']:.4f}")
            if not (0.5 <= r['auc'] <= 1.0):
                issues.append(f"auc={r['auc']:.6f}")
            if r['compressed_size'] <= 0:
                issues.append("size<=0")
            if r['inference_time'] <= 0:
                issues.append("infer<=0")
            if issues:
                f.write(f"- FAIL CRF {r['crf']}: {', '.join(issues)}\n")
                all_valid = False
        if all_valid:
            f.write("- ALL CHECKS PASSED\n")

    print(f"\nResults written to: {path}")


if __name__ == "__main__":
    main()
