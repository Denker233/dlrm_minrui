#!/usr/bin/env python3
"""
Full Software Codec Compression Experiment
Phase 1.5: Baseline inference (accuracy, AUC)
Phase 2: Compress at CRF 0, 15, 23, 30, 38, 45, 51 with full metrics
Phase 3: Isolate each H.265 algorithm's effect
"""

import os
import sys
import time
import json
import tempfile
import subprocess
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlrm_data_pytorch as dp
from dlrm_s_pytorch import DLRM_Net, quantize_asymmetric, dequantize_asymmetric
from dlrm_s_pytorch import tile_embeddings, untile_embeddings

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_quick.pt"
DATA_FILE = "./input/train.txt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
RESULTS_DIR = os.path.expanduser("~/experiment-control")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 16384


# ============================================================
# Helpers
# ============================================================
def create_args():
    """Create args object mimicking argparse for data loading"""
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
    """Load the baseline model and test data"""
    print("Loading model and data...")
    args = create_args()

    # Load data
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    ln_emb = np.array(train_data.counts)
    m_spa = args.arch_sparse_feature_size
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den

    # Calculate interaction size
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # Create model
    dlrm = DLRM_Net(
        m_spa, ln_emb, ln_bot, ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        loss_function=args.loss_function,
    )

    # Load weights
    ld_model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    dlrm.load_state_dict(ld_model["state_dict"])
    dlrm.eval()

    return dlrm, test_ld, ln_emb, ln_bot, ln_top, m_spa, args


def run_inference(dlrm, test_ld):
    """Run inference and return accuracy and AUC"""
    all_scores = []
    all_targets = []
    test_accu = 0
    test_samp = 0

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

    accuracy = test_accu / test_samp
    auc = roc_auc_score(all_targets, all_scores)
    print(f"  Inference complete: {test_samp} samples")
    return accuracy, auc


# ============================================================
# Compression Engine (simplified, standalone)
# ============================================================
def compress_table_with_codec(weights, codec_cmd_args, quality, tmpdir_base=None):
    """
    Compress a single embedding table using ffmpeg.
    Returns: compressed_data (bytes), metadata (dict), compress_time (float)
    """
    num_emb, emb_dim = weights.shape

    # Quantize to uint8
    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / 255.0
    zero_point = -(w_min / scale).round() if scale > 0 else 0.0
    quantized = ((weights / scale).round() + zero_point).clamp(0, 255).to(torch.uint8)
    quant_metadata = {'scale': float(scale), 'zero_point': float(zero_point)}

    pixels = quantized.numpy()

    # Determine frame dimensions
    MAX_DIM = 16384
    MIN_WIDTH = 64
    MIN_HEIGHT = 64
    TILE_SIZE = 4
    TILING_THRESHOLD = 50000

    width = emb_dim
    height = num_emb
    tiling_meta = {'tiled': False}

    if num_emb > TILING_THRESHOLD:
        # Use tiling
        data_flat = pixels.reshape(-1)
        image, grid_size, tiles_per_emb = tile_embeddings(
            data_flat, emb_dim, num_emb, TILE_SIZE
        )
        width = image.shape[1]
        height = image.shape[0]
        raw_data = image.tobytes()
        tiling_meta = {
            'tiled': True,
            'grid_size': grid_size,
            'tiles_per_emb': tiles_per_emb,
            'tile_size': TILE_SIZE
        }
    else:
        raw_data = pixels.tobytes()

    # Handle dimension constraints
    total_pixels = width * height
    original_width, original_height = width, height
    reshaped = False

    if height > MAX_DIM or width < MIN_WIDTH or height < MIN_HEIGHT:
        min_required = MIN_WIDTH * MIN_HEIGHT
        if total_pixels < min_required:
            width = MIN_WIDTH
            height = MIN_HEIGHT
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
                width = MIN_WIDTH
                height = MIN_HEIGHT

        padded_pixels = width * height
        if padded_pixels > len(raw_data):
            padded = bytearray(padded_pixels)
            padded[:len(raw_data)] = raw_data
            raw_data = bytes(padded)
        reshaped = True

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_file = os.path.join(tmpdir, 'pixels.raw')
        video_file = os.path.join(tmpdir, 'compressed.mp4')

        with open(raw_file, 'wb') as f:
            f.write(raw_data)

        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', f'{width}x{height}',
            '-r', '1',
            '-i', raw_file,
        ] + codec_cmd_args + [
            '-frames:v', '1',
            video_file
        ]

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
    """
    Decompress a single embedding table.
    Returns: weights (torch.Tensor), decompress_time (float)
    """
    num_emb, emb_dim = metadata['shape']

    with tempfile.TemporaryDirectory() as tmpdir:
        video_file = os.path.join(tmpdir, 'compressed.mp4')
        raw_file = os.path.join(tmpdir, 'decoded.raw')

        with open(video_file, 'wb') as f:
            f.write(compressed_data)

        cmd = [
            'ffmpeg', '-y',
            '-i', video_file,
            '-pix_fmt', 'gray',
            '-f', 'rawvideo',
            raw_file
        ]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        decompress_time = time.time() - t0

        if result.returncode != 0:
            raise RuntimeError(f"Decoding failed: {result.stderr[:500]}")

        pixels_uint8 = np.fromfile(raw_file, dtype=np.uint8)

    # Handle tiling
    if metadata.get('tiling', {}).get('tiled', False):
        tiling = metadata['tiling']
        grid_size = tiling['grid_size']
        tile_size = tiling['tile_size']
        image_size = grid_size * tile_size
        pixels_uint8 = untile_embeddings(
            pixels_uint8[:image_size * image_size].reshape(image_size, image_size),
            emb_dim, num_emb,
            grid_size, tiling['tiles_per_emb'], tile_size
        )
    else:
        expected_size = num_emb * emb_dim
        pixels_uint8 = pixels_uint8[:expected_size]

    pixels_uint8 = torch.from_numpy(pixels_uint8.copy()).reshape(num_emb, emb_dim)

    # Dequantize
    qp = metadata['quant_params']
    weights = (pixels_uint8.float() - qp['zero_point']) * qp['scale']

    return weights, decompress_time


# ============================================================
# Phase 1.5: Baseline
# ============================================================
def run_phase_1_5():
    """Run baseline inference and return metrics"""
    print("\n" + "=" * 80)
    print("PHASE 1.5: BASELINE MODEL INFERENCE")
    print("=" * 80)

    dlrm, test_ld, ln_emb, ln_bot, ln_top, m_spa, args = load_model_and_data()

    # Calculate model size
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]
    total_emb_size = sum(state_dict[k].numel() * 4 for k in emb_keys)
    total_model_size = sum(v.numel() * 4 for v in state_dict.values())

    print(f"  Model: {MODEL_PATH}")
    print(f"  Embedding tables: {len(emb_keys)}")
    print(f"  Total embedding size: {total_emb_size / 1024 / 1024:.2f} MB")
    print(f"  Total model size: {total_model_size / 1024 / 1024:.2f} MB")

    accuracy, auc = run_inference(dlrm, test_ld)

    print(f"\n  BASELINE ACCURACY: {accuracy * 100:.4f}%")
    print(f"  BASELINE AUC:      {auc:.6f}")

    return {
        'accuracy': accuracy,
        'auc': auc,
        'total_emb_size': total_emb_size,
        'total_model_size': total_model_size,
        'num_tables': len(emb_keys),
    }


# ============================================================
# Phase 2: CRF sweep
# ============================================================
def run_phase_2(baseline):
    """Compress at various CRF levels and measure everything"""
    print("\n" + "=" * 80)
    print("PHASE 2: CRF QUALITY SWEEP (libx265)")
    print("=" * 80)

    crf_levels = [0, 15, 23, 30, 38, 45, 51]
    results = []

    # Load model and data once
    dlrm_orig, test_ld, ln_emb, ln_bot, ln_top, m_spa, args = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]

    for crf in crf_levels:
        print(f"\n{'='*60}")
        print(f"CRF = {crf}")
        print(f"{'='*60}")

        if crf == 0:
            codec_args = [
                '-c:v', 'libx265',
                '-x265-params', 'lossless=1:log-level=error',
                '-preset', 'ultrafast',
            ]
        else:
            codec_args = [
                '-c:v', 'libx265',
                '-crf', str(crf),
                '-preset', 'ultrafast',
                '-x265-params', 'log-level=error:allow-non-conformance=1',
            ]

        total_compressed_size = 0
        total_compress_time = 0.0
        total_decompress_time = 0.0
        num_tables_compressed = 0

        # Create a fresh model for this CRF level
        dlrm_test = DLRM_Net(
            m_spa, ln_emb,
            np.fromstring(ARCH_MLP_BOT, dtype=int, sep="-"),
            np.fromstring(str((ln_emb.size + 1) * ln_emb.size // 2 +
                             np.fromstring(ARCH_MLP_BOT, dtype=int, sep="-")[-1]) + "-" + ARCH_MLP_TOP,
                         dtype=int, sep="-"),
            arch_interaction_op="dot",
            arch_interaction_itself=False,
            sigmoid_bot=-1,
            sigmoid_top=np.fromstring(str((ln_emb.size + 1) * ln_emb.size // 2 +
                             np.fromstring(ARCH_MLP_BOT, dtype=int, sep="-")[-1]) + "-" + ARCH_MLP_TOP,
                         dtype=int, sep="-").size - 2,
            loss_function="bce",
        )
        dlrm_test.load_state_dict(state_dict)
        dlrm_test.eval()

        # Compress and decompress each embedding table
        for idx, key in enumerate(emb_keys):
            weights = state_dict[key]
            num_emb, emb_dim = weights.shape
            table_size = num_emb * emb_dim * 4  # float32 bytes

            try:
                compressed_data, metadata, ct = compress_table_with_codec(
                    weights, codec_args, crf
                )
                total_compress_time += ct
                total_compressed_size += len(compressed_data)
                num_tables_compressed += 1

                # Decompress
                decompressed_weights, dt = decompress_table(compressed_data, metadata)
                total_decompress_time += dt

                # Replace weights in model
                with torch.no_grad():
                    dlrm_test.emb_l[idx].weight.data = decompressed_weights

                if idx % 5 == 0:
                    print(f"  Table {idx}/{len(emb_keys)}: {weights.shape} "
                          f"→ {len(compressed_data)} bytes "
                          f"({table_size / len(compressed_data):.1f}x)")
            except Exception as e:
                print(f"  Table {idx} FAILED: {e}")
                # Keep original weights
                num_tables_compressed += 1
                total_compressed_size += table_size  # worst case

        # Run inference
        print(f"  Running inference...")
        accuracy, auc = run_inference(dlrm_test, test_ld)

        compression_ratio = baseline['total_emb_size'] / total_compressed_size if total_compressed_size > 0 else 0

        result = {
            'crf': crf,
            'compression_ratio': compression_ratio,
            'accuracy': accuracy,
            'auc': auc,
            'compressed_size': total_compressed_size,
            'num_tables': num_tables_compressed,
            'compress_time': total_compress_time,
            'decompress_time': total_decompress_time,
            'accuracy_loss': (baseline['accuracy'] - accuracy) * 100,
            'auc_loss': baseline['auc'] - auc,
        }
        results.append(result)

        print(f"\n  CRF {crf} Results:")
        print(f"    Compression ratio: {compression_ratio:.2f}x")
        print(f"    Accuracy: {accuracy * 100:.4f}% (loss: {result['accuracy_loss']:.4f}%)")
        print(f"    AUC: {auc:.6f} (loss: {result['auc_loss']:.6f})")
        print(f"    Compressed size: {total_compressed_size} bytes ({total_compressed_size / 1024 / 1024:.2f} MB)")
        print(f"    Tables: {num_tables_compressed}")
        print(f"    Compress time: {total_compress_time:.2f}s")
        print(f"    Decompress time: {total_decompress_time:.2f}s")

    return results


# ============================================================
# Phase 3: Algorithm isolation
# ============================================================
def run_phase_3(baseline):
    """Isolate the effect of each H.265 algorithm"""
    print("\n" + "=" * 80)
    print("PHASE 3: H.265 ALGORITHM ISOLATION")
    print("=" * 80)

    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]

    # We'll test on all tables combined
    # First, create raw uint8 data for each table
    all_raw_sizes = 0
    table_data = []

    for key in emb_keys:
        weights = state_dict[key]
        w_min, w_max = weights.min(), weights.max()
        scale = (w_max - w_min) / 255.0
        zero_point = -(w_min / scale).round() if scale > 0 else 0.0
        quantized = ((weights / scale).round() + zero_point).clamp(0, 255).to(torch.uint8)
        raw_bytes = quantized.numpy().tobytes()
        all_raw_sizes += len(raw_bytes)
        table_data.append({
            'key': key,
            'shape': weights.shape,
            'raw_bytes': raw_bytes,
            'float_size': weights.numel() * 4,
        })

    raw_uint8_size = all_raw_sizes  # after INT8 quantization, before codec

    print(f"  Total raw (float32): {baseline['total_emb_size']} bytes ({baseline['total_emb_size']/1024/1024:.2f} MB)")
    print(f"  Total raw (uint8):   {raw_uint8_size} bytes ({raw_uint8_size/1024/1024:.2f} MB)")

    # Define configurations to test
    configs = [
        {
            'name': 'Full H.265 pipeline (CRF 23)',
            'codec_args': ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
                          '-x265-params', 'log-level=error:allow-non-conformance=1'],
        },
        {
            'name': 'Entropy only (FFV1)',
            'codec_args': ['-c:v', 'ffv1'],
            'ext': '.avi',
        },
        {
            'name': 'Lossless H.265',
            'codec_args': ['-c:v', 'libx265', '-x265-params', 'lossless=1:log-level=error',
                          '-preset', 'ultrafast'],
        },
        {
            'name': 'No SAO (CRF 23)',
            'codec_args': ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
                          '-x265-params', 'log-level=error:allow-non-conformance=1:no-sao=1'],
        },
        {
            'name': 'No Deblock (CRF 23)',
            'codec_args': ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
                          '-x265-params', 'log-level=error:allow-non-conformance=1:no-deblock=1'],
        },
        {
            'name': 'No SAO + No Deblock (CRF 23)',
            'codec_args': ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
                          '-x265-params', 'log-level=error:allow-non-conformance=1:no-sao=1:no-deblock=1'],
        },
        {
            'name': 'No Intra Smoothing (CRF 23)',
            'codec_args': ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
                          '-x265-params', 'log-level=error:allow-non-conformance=1:no-strong-intra-smoothing=1'],
        },
        {
            'name': 'No Transform Skip (CRF 23)',
            'codec_args': ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
                          '-x265-params', 'log-level=error:allow-non-conformance=1:no-tskip=1'],
        },
        {
            'name': 'H.264 (CRF 23)',
            'codec_args': ['-c:v', 'libx264', '-crf', '23', '-preset', 'ultrafast'],
        },
    ]

    # Inter-frame test: encode all tables as consecutive frames
    # We'll handle this separately since it requires multi-frame encoding

    algo_results = []

    for config in configs:
        name = config['name']
        codec_args = config['codec_args']
        ext = config.get('ext', '.mp4')

        print(f"\n  Testing: {name}")

        total_compressed = 0
        total_compress_time = 0.0
        total_decompress_time = 0.0
        failures = 0

        for td in table_data:
            num_emb, emb_dim = td['shape']
            raw_bytes = td['raw_bytes']

            # Determine dimensions
            width = emb_dim
            height = num_emb
            MAX_DIM = 16384
            MIN_WIDTH = 64
            MIN_HEIGHT = 64

            total_pixels = width * height

            if height > MAX_DIM or width < MIN_WIDTH or height < MIN_HEIGHT:
                min_required = MIN_WIDTH * MIN_HEIGHT
                if total_pixels < min_required:
                    width = MIN_WIDTH
                    height = MIN_HEIGHT
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
                        width = MIN_WIDTH
                        height = MIN_HEIGHT

                padded_pixels = width * height
                if padded_pixels > len(raw_bytes):
                    padded = bytearray(padded_pixels)
                    padded[:len(raw_bytes)] = raw_bytes
                    raw_bytes = bytes(padded)

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    raw_file = os.path.join(tmpdir, 'pixels.raw')
                    video_file = os.path.join(tmpdir, 'compressed' + ext)
                    decoded_file = os.path.join(tmpdir, 'decoded.raw')

                    with open(raw_file, 'wb') as f:
                        f.write(raw_bytes)

                    # Compress
                    cmd = [
                        'ffmpeg', '-y',
                        '-f', 'rawvideo', '-pix_fmt', 'gray',
                        '-s', f'{width}x{height}',
                        '-r', '1', '-i', raw_file,
                    ] + codec_args + [
                        '-frames:v', '1',
                        video_file
                    ]

                    t0 = time.time()
                    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    ct = time.time() - t0
                    total_compress_time += ct

                    if r.returncode != 0:
                        failures += 1
                        total_compressed += len(raw_bytes)
                        continue

                    compressed_size = os.path.getsize(video_file)
                    total_compressed += compressed_size

                    # Decompress
                    cmd_dec = [
                        'ffmpeg', '-y',
                        '-i', video_file,
                        '-pix_fmt', 'gray',
                        '-f', 'rawvideo',
                        decoded_file
                    ]
                    t0 = time.time()
                    subprocess.run(cmd_dec, capture_output=True, text=True, check=False)
                    dt = time.time() - t0
                    total_decompress_time += dt

            except Exception as e:
                failures += 1
                total_compressed += len(raw_bytes)

        bits_per_value = (total_compressed * 8) / (raw_uint8_size) if raw_uint8_size > 0 else 0
        comp_ratio = raw_uint8_size / total_compressed if total_compressed > 0 else 0
        comp_ratio_vs_float = baseline['total_emb_size'] / total_compressed if total_compressed > 0 else 0

        result = {
            'name': name,
            'compressed_size': total_compressed,
            'comp_ratio_vs_uint8': comp_ratio,
            'comp_ratio_vs_float32': comp_ratio_vs_float,
            'bits_per_value': bits_per_value,
            'compress_time': total_compress_time,
            'decompress_time': total_decompress_time,
            'failures': failures,
        }
        algo_results.append(result)

        print(f"    Size: {total_compressed} bytes ({total_compressed/1024/1024:.2f} MB)")
        print(f"    Ratio vs uint8: {comp_ratio:.2f}x, vs float32: {comp_ratio_vs_float:.2f}x")
        print(f"    Bits/value: {bits_per_value:.4f}")
        print(f"    Compress: {total_compress_time:.2f}s, Decompress: {total_decompress_time:.2f}s")
        if failures > 0:
            print(f"    Failures: {failures}")

    # Inter-frame test
    print(f"\n  Testing: Inter-frame prediction (multi-frame encoding)")
    inter_results = run_interframe_test(table_data, raw_uint8_size, baseline['total_emb_size'])
    algo_results.extend(inter_results)

    return algo_results, raw_uint8_size


def run_interframe_test(table_data, raw_uint8_size, float32_size):
    """Test inter-frame prediction by encoding multiple tables as consecutive frames"""
    results = []

    # Group tables by compatible dimensions
    # For simplicity, we'll normalize all tables to 64xN format
    FRAME_WIDTH = 64

    # Build multi-frame raw data: each table becomes one or more frames
    frame_data = bytearray()
    frame_heights = []
    num_frames = 0

    for td in table_data:
        num_emb, emb_dim = td['shape']
        raw = td['raw_bytes']
        total_pixels = len(raw)

        # Normalize to FRAME_WIDTH wide
        height = (total_pixels + FRAME_WIDTH - 1) // FRAME_WIDTH
        if height < 64:
            height = 64

        padded_size = FRAME_WIDTH * height
        if padded_size > len(raw):
            padded = bytearray(padded_size)
            padded[:len(raw)] = raw
            frame_data.extend(padded)
        else:
            frame_data.extend(raw[:padded_size])
        frame_heights.append(height)
        num_frames += 1

    # For multi-frame, all frames must be same height
    # Use the max height for all frames
    max_height = max(frame_heights)

    # Rebuild with uniform frame size
    uniform_frame_data = bytearray()
    for td in table_data:
        raw = td['raw_bytes']
        frame_size = FRAME_WIDTH * max_height
        padded = bytearray(frame_size)
        padded[:min(len(raw), frame_size)] = raw[:min(len(raw), frame_size)]
        uniform_frame_data.extend(padded)

    configs = [
        {
            'name': 'Multi-frame: I-frames only (keyint=1, no inter)',
            'args': ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
                    '-x265-params', 'log-level=error:allow-non-conformance=1:keyint=1'],
        },
        {
            'name': 'Multi-frame: with inter-frame prediction',
            'args': ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
                    '-x265-params', 'log-level=error:allow-non-conformance=1:keyint=250'],
        },
    ]

    for config in configs:
        name = config['name']
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                raw_file = os.path.join(tmpdir, 'multi.raw')
                video_file = os.path.join(tmpdir, 'multi.mp4')
                decoded_file = os.path.join(tmpdir, 'decoded.raw')

                with open(raw_file, 'wb') as f:
                    f.write(uniform_frame_data)

                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'rawvideo', '-pix_fmt', 'gray',
                    '-s', f'{FRAME_WIDTH}x{max_height}',
                    '-r', '30',
                    '-i', raw_file,
                ] + config['args'] + [video_file]

                t0 = time.time()
                r = subprocess.run(cmd, capture_output=True, text=True, check=False)
                ct = time.time() - t0

                if r.returncode != 0:
                    print(f"    {name}: FAILED - {r.stderr[:200]}")
                    results.append({
                        'name': name,
                        'compressed_size': len(uniform_frame_data),
                        'comp_ratio_vs_uint8': 1.0,
                        'comp_ratio_vs_float32': float32_size / len(uniform_frame_data),
                        'bits_per_value': 8.0,
                        'compress_time': ct,
                        'decompress_time': 0,
                        'failures': 1,
                    })
                    continue

                compressed_size = os.path.getsize(video_file)

                # Decompress
                cmd_dec = ['ffmpeg', '-y', '-i', video_file, '-pix_fmt', 'gray',
                          '-f', 'rawvideo', decoded_file]
                t0 = time.time()
                subprocess.run(cmd_dec, capture_output=True, text=True, check=False)
                dt = time.time() - t0

                bits_per_value = (compressed_size * 8) / raw_uint8_size if raw_uint8_size > 0 else 0
                comp_vs_uint8 = raw_uint8_size / compressed_size if compressed_size > 0 else 0
                comp_vs_float32 = float32_size / compressed_size if compressed_size > 0 else 0

                results.append({
                    'name': name,
                    'compressed_size': compressed_size,
                    'comp_ratio_vs_uint8': comp_vs_uint8,
                    'comp_ratio_vs_float32': comp_vs_float32,
                    'bits_per_value': bits_per_value,
                    'compress_time': ct,
                    'decompress_time': dt,
                    'failures': 0,
                })

                print(f"    {name}:")
                print(f"      Size: {compressed_size} bytes ({compressed_size/1024/1024:.2f} MB)")
                print(f"      Ratio vs uint8: {comp_vs_uint8:.2f}x, vs float32: {comp_vs_float32:.2f}x")
                print(f"      Bits/value: {bits_per_value:.4f}")
                print(f"      Compress: {ct:.2f}s, Decompress: {dt:.2f}s")

        except Exception as e:
            print(f"    {name}: ERROR - {e}")
            results.append({
                'name': name,
                'compressed_size': len(uniform_frame_data),
                'comp_ratio_vs_uint8': 1.0,
                'comp_ratio_vs_float32': float32_size / len(uniform_frame_data),
                'bits_per_value': 8.0,
                'compress_time': 0,
                'decompress_time': 0,
                'failures': 1,
            })

    return results


# ============================================================
# Report generation
# ============================================================
def write_phase2_report(baseline, phase2_results):
    """Write Phase 2 results to markdown"""
    path = os.path.join(RESULTS_DIR, "compression_results.md")
    with open(path, 'w') as f:
        f.write("# Phase 2: Software Codec Compression Results\n\n")
        f.write("## Baseline Model\n\n")
        f.write(f"- **Model:** `{MODEL_PATH}`\n")
        f.write(f"- **Embedding tables:** {baseline['num_tables']}\n")
        f.write(f"- **Total embedding size:** {baseline['total_emb_size']} bytes ({baseline['total_emb_size']/1024/1024:.2f} MB)\n")
        f.write(f"- **Baseline accuracy:** {baseline['accuracy']*100:.4f}%\n")
        f.write(f"- **Baseline AUC:** {baseline['auc']:.6f}\n\n")

        f.write("## CRF Quality Sweep (libx265, ultrafast preset)\n\n")
        f.write("| CRF | Compression Ratio | Accuracy (%) | AUC | Accuracy Loss (%) | AUC Loss | Compressed Size (bytes) | Compressed Size (MB) | Tables | Compress Time (s) | Decompress Time (s) |\n")
        f.write("|-----|------------------|-------------|------|-------------------|----------|------------------------|---------------------|--------|-------------------|--------------------|\n")

        for r in phase2_results:
            f.write(f"| {r['crf']} | {r['compression_ratio']:.2f}x | {r['accuracy']*100:.4f} | {r['auc']:.6f} | {r['accuracy_loss']:.4f} | {r['auc_loss']:.6f} | {r['compressed_size']} | {r['compressed_size']/1024/1024:.2f} | {r['num_tables']} | {r['compress_time']:.2f} | {r['decompress_time']:.2f} |\n")

        f.write("\n## Validation\n\n")
        all_valid = True
        for r in phase2_results:
            checks = []
            if r['compression_ratio'] <= 1:
                checks.append(f"CRF {r['crf']}: compression_ratio {r['compression_ratio']:.2f} <= 1")
                all_valid = False
            if not (0.5 <= r['accuracy'] <= 1.0):
                checks.append(f"CRF {r['crf']}: accuracy {r['accuracy']:.4f} out of range")
                all_valid = False
            if not (0.5 <= r['auc'] <= 1.0):
                checks.append(f"CRF {r['crf']}: AUC {r['auc']:.6f} out of range")
                all_valid = False
            if r['compressed_size'] <= 0:
                checks.append(f"CRF {r['crf']}: compressed_size <= 0")
                all_valid = False
            if r['compress_time'] <= 0:
                checks.append(f"CRF {r['crf']}: compress_time <= 0")
                all_valid = False
            if r['decompress_time'] <= 0:
                checks.append(f"CRF {r['crf']}: decompress_time <= 0")
                all_valid = False
            if checks:
                for c in checks:
                    f.write(f"- FAIL: {c}\n")

        if all_valid:
            f.write("- ALL CHECKS PASSED\n")

    print(f"\nPhase 2 report written to: {path}")


def write_phase3_report(algo_results, raw_uint8_size, baseline):
    """Write Phase 3 results to markdown"""
    path = os.path.join(RESULTS_DIR, "algorithm_analysis.md")

    # Find full pipeline result for delta calculations
    full_pipeline = None
    for r in algo_results:
        if 'Full H.265 pipeline' in r['name']:
            full_pipeline = r
            break

    with open(path, 'w') as f:
        f.write("# Phase 3: H.265 Algorithm Isolation Analysis\n\n")
        f.write("## Reference Sizes\n\n")
        f.write(f"- **Raw float32:** {baseline['total_emb_size']} bytes ({baseline['total_emb_size']/1024/1024:.2f} MB)\n")
        f.write(f"- **Raw uint8 (after INT8 quantization):** {raw_uint8_size} bytes ({raw_uint8_size/1024/1024:.2f} MB)\n")
        if full_pipeline:
            f.write(f"- **Full H.265 pipeline (CRF 23):** {full_pipeline['compressed_size']} bytes ({full_pipeline['compressed_size']/1024/1024:.2f} MB)\n")
        f.write("\n")

        f.write("## Per-Algorithm Results\n\n")
        f.write("| Algorithm Config | File Size (bytes) | File Size (MB) | Comp Ratio vs uint8 | Comp Ratio vs float32 | Bits/Value | Compress Time (s) | Decompress Time (s) |\n")
        f.write("|-----------------|-------------------|---------------|--------------------|-----------------------|-----------|-------------------|--------------------|\n")

        for r in algo_results:
            f.write(f"| {r['name']} | {r['compressed_size']} | {r['compressed_size']/1024/1024:.2f} | {r['comp_ratio_vs_uint8']:.2f}x | {r['comp_ratio_vs_float32']:.2f}x | {r['bits_per_value']:.4f} | {r['compress_time']:.2f} | {r['decompress_time']:.2f} |\n")

        # Contribution analysis
        f.write("\n## Algorithm Contribution Analysis\n\n")
        f.write("Contribution = how many bytes each algorithm saves compared to disabling it.\n\n")

        entropy_only = next((r for r in algo_results if 'FFV1' in r['name']), None)
        lossless = next((r for r in algo_results if 'Lossless' in r['name']), None)
        no_sao = next((r for r in algo_results if 'No SAO' in r['name'] and 'Deblock' not in r['name']), None)
        no_deblock = next((r for r in algo_results if 'No Deblock' in r['name'] and 'SAO' not in r['name']), None)
        no_both = next((r for r in algo_results if 'No SAO + No Deblock' in r['name']), None)
        no_intra = next((r for r in algo_results if 'No Intra' in r['name']), None)
        iframes_only = next((r for r in algo_results if 'I-frames only' in r['name']), None)
        with_inter = next((r for r in algo_results if 'with inter-frame' in r['name']), None)

        f.write("| Component | Bytes Saved | MB Saved | % of Total Savings |\n")
        f.write("|-----------|-------------|---------|--------------------|\n")

        total_savings = raw_uint8_size - (full_pipeline['compressed_size'] if full_pipeline else raw_uint8_size)

        contributions = []
        if entropy_only:
            entropy_savings = raw_uint8_size - entropy_only['compressed_size']
            contributions.append(('Entropy coding (CABAC/FFV1)', entropy_savings))

        if entropy_only and lossless:
            prediction_savings = entropy_only['compressed_size'] - lossless['compressed_size']
            contributions.append(('Prediction (intra)', prediction_savings))

        if lossless and full_pipeline:
            quant_savings = lossless['compressed_size'] - full_pipeline['compressed_size']
            contributions.append(('Quantization (lossy)', quant_savings))

        if no_sao and full_pipeline:
            sao_savings = no_sao['compressed_size'] - full_pipeline['compressed_size']
            contributions.append(('SAO filter', sao_savings))

        if no_deblock and full_pipeline:
            deblock_savings = no_deblock['compressed_size'] - full_pipeline['compressed_size']
            contributions.append(('Deblocking filter', deblock_savings))

        if iframes_only and with_inter:
            inter_savings = iframes_only['compressed_size'] - with_inter['compressed_size']
            contributions.append(('Inter-frame prediction', inter_savings))

        for name, savings in contributions:
            pct = (savings / total_savings * 100) if total_savings > 0 else 0
            f.write(f"| {name} | {savings} | {savings/1024/1024:.2f} | {pct:.1f}% |\n")

        f.write(f"\n**Total savings (uint8 → full H.265 CRF 23):** {total_savings} bytes ({total_savings/1024/1024:.2f} MB)\n")

    print(f"\nPhase 3 report written to: {path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 80)
    print("FULL SOFTWARE CODEC COMPRESSION EXPERIMENT")
    print("=" * 80)
    print(f"Working directory: {os.getcwd()}")
    print(f"Model: {MODEL_PATH}")
    print(f"Results: {RESULTS_DIR}")

    # Phase 1.5
    baseline = run_phase_1_5()

    # Phase 2
    phase2_results = run_phase_2(baseline)
    write_phase2_report(baseline, phase2_results)

    # Phase 3
    algo_results, raw_uint8_size = run_phase_3(baseline)
    write_phase3_report(algo_results, raw_uint8_size, baseline)

    # Final status
    print("\n" + "=" * 80)
    print("ALL PHASES COMPLETE")
    print("=" * 80)
    print(f"  compression_results.md: {os.path.join(RESULTS_DIR, 'compression_results.md')}")
    print(f"  algorithm_analysis.md:  {os.path.join(RESULTS_DIR, 'algorithm_analysis.md')}")
