#!/usr/bin/env python3
"""
Codec Pipeline Isolation Experiments
Measures each algorithm's individual contribution to compression and AUC loss.

Experiment A: Raw INT8 (no codec) — isolates quantization cost
Experiment B: INT8 + generic entropy coding (gzip, zstd, bzip2) — no spatial prediction
Experiments C/D/E: use existing Phase 2A and Phase 8 data
"""

import os
import sys
import time
import json
import gzip
import bz2
import subprocess
import struct
import tempfile
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlrm_data_pytorch as dp
from dlrm_s_pytorch import DLRM_Net

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
DATA_FILE = "./input/train.txt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
RESULTS_DIR = os.path.expanduser("~/experiment-control")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 16384


def drop_caches():
    try:
        subprocess.run(['sync'], check=True)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], check=True)
        print("  [CACHE] Dropped caches")
    except Exception as e:
        print(f"  [CACHE] Warning: {e}")


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
    ln_top = np.fromstring(str(num_int) + "-" + args.arch_mlp_top, dtype=int, sep="-")
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
    return dlrm, test_ld, ln_emb


def run_inference(dlrm, test_ld):
    all_scores = []
    all_targets = []
    test_accu = 0
    test_samp = 0
    t0 = time.time()
    with torch.no_grad():
        for i, testBatch in enumerate(test_ld):
            X_test, lS_o_test, lS_i_test, T_test = testBatch
            Z_test = dlrm(X_test, lS_o_test, lS_i_test)
            S_test = Z_test.detach().cpu().numpy().flatten()
            T_test_np = T_test.detach().cpu().numpy().flatten()
            test_accu += np.sum((np.round(S_test, 0) == T_test_np).astype(np.uint8))
            test_samp += T_test_np.shape[0]
            all_scores.extend(S_test.tolist())
            all_targets.extend(T_test_np.tolist())
            if i % 50 == 0:
                print(f"    Batch {i}/{len(test_ld)}", end='\r')
    inference_time = time.time() - t0
    accuracy = test_accu / test_samp
    auc = roc_auc_score(all_targets, all_scores)
    return accuracy, auc, inference_time


def quantize_table_to_uint8(weights):
    """Quantize FP32 weights to UINT8 using asymmetric min/max scaling.
    Returns (uint8_data, scale, zero_point) — same method as the codec pipeline."""
    w_min = weights.min().item()
    w_max = weights.max().item()
    scale = (w_max - w_min) / 255.0
    if scale == 0:
        scale = 1.0
    zero_point = round(-w_min / scale)
    quantized = ((weights / scale).round() + zero_point).clamp(0, 255).to(torch.uint8)
    return quantized, scale, zero_point


def dequantize_uint8(quantized, scale, zero_point):
    """Dequantize UINT8 back to FP32."""
    return (quantized.float() - zero_point) * scale


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 80)
    print("CODEC PIPELINE ISOLATION EXPERIMENTS")
    print("Measuring each algorithm's individual contribution")
    print("=" * 80)

    dlrm, test_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]
    num_tables = len(emb_keys)

    # Compute original FP32 sizes
    table_fp32_sizes = []
    total_fp32 = 0
    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        sz = w.numel() * 4  # 4 bytes per float32
        table_fp32_sizes.append(sz)
        total_fp32 += sz

    print(f"  Model: {MODEL_PATH}")
    print(f"  Tables: {num_tables}")
    print(f"  Total FP32 embedding size: {total_fp32:,} bytes ({total_fp32/1024/1024:.2f} MB)")

    # ================================================================
    # BASELINE: FP32 inference
    # ================================================================
    print("\n" + "=" * 60)
    print("BASELINE: FP32 Inference")
    print("=" * 60)
    drop_caches()
    baseline_acc, baseline_auc, baseline_infer = run_inference(dlrm, test_ld)
    print(f"  BASELINE: acc={baseline_acc*100:.4f}%, AUC={baseline_auc:.6f}, infer={baseline_infer:.2f}s")

    # ================================================================
    # EXPERIMENT A: Raw INT8 (quantize only, no compression)
    # ================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Raw INT8 Quantization (no codec)")
    print("  FP32 -> UINT8 asymmetric min/max, raw bytes")
    print("=" * 60)

    table_a_results = []
    total_int8_size = 0

    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        num_emb, emb_dim = w.shape
        fp32_size = w.numel() * 4
        int8_size = w.numel()  # 1 byte per uint8

        quantized, scale, zp = quantize_table_to_uint8(w)
        raw_bytes = quantized.numpy().tobytes()

        total_int8_size += int8_size

        table_a_results.append({
            'table': t,
            'shape': list(w.shape),
            'fp32_bytes': fp32_size,
            'int8_bytes': int8_size,
            'raw_bytes_len': len(raw_bytes),
            'ratio': fp32_size / int8_size,
            'scale': scale,
            'zero_point': zp,
        })

        # Replace weights with dequantized version
        dec_w = dequantize_uint8(quantized, scale, zp)
        with torch.no_grad():
            dlrm.emb_l[t].weight.data = dec_w

        if t % 5 == 0 or t == num_tables - 1:
            print(f"  Table {t:2d}: {w.shape} -> {int8_size:,} bytes (4.00x), "
                  f"scale={scale:.8f}, zp={zp}")

    # Run inference with INT8-dequantized weights
    drop_caches()
    int8_acc, int8_auc, int8_infer = run_inference(dlrm, test_ld)
    int8_auc_loss = baseline_auc - int8_auc
    int8_acc_loss = (baseline_acc - int8_acc) * 100

    print(f"\n  EXPERIMENT A RESULT:")
    print(f"    Total INT8 size: {total_int8_size:,} bytes ({total_int8_size/1024/1024:.2f} MB)")
    print(f"    Compression ratio: {total_fp32/total_int8_size:.2f}x")
    print(f"    Accuracy: {int8_acc*100:.4f}% (loss {int8_acc_loss:.4f}%)")
    print(f"    AUC: {int8_auc:.6f} (loss {int8_auc_loss:.6f} = {int8_auc_loss*100:.4f}pp)")
    print(f"    Inference: {int8_infer:.2f}s")

    # Restore original weights for next experiment's data collection
    for t in range(num_tables):
        with torch.no_grad():
            dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

    # ================================================================
    # EXPERIMENT B: INT8 + Generic Entropy Coding (no spatial prediction)
    # ================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT B: INT8 + Generic Entropy Coding")
    print("  gzip -9, zstd -19, bzip2 -9 on raw UINT8 bytes")
    print("  No spatial prediction — pure entropy coding only")
    print("=" * 60)

    # Check if zstd is available
    has_zstd = True
    try:
        subprocess.run(['zstd', '--version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        has_zstd = False
        print("  WARNING: zstd not found, will skip zstd compression")

    table_b_results = []
    total_gzip = 0
    total_zstd = 0
    total_bzip2 = 0

    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        quantized, scale, zp = quantize_table_to_uint8(w)
        raw_bytes = quantized.numpy().tobytes()
        raw_size = len(raw_bytes)

        # gzip -9
        t0 = time.time()
        gzip_data = gzip.compress(raw_bytes, compresslevel=9)
        gzip_time = time.time() - t0
        gzip_size = len(gzip_data)
        total_gzip += gzip_size

        # zstd -19
        if has_zstd:
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
                f.write(raw_bytes)
                raw_path = f.name
            zstd_path = raw_path + '.zst'
            t0 = time.time()
            subprocess.run(['zstd', '-19', '-f', '-q', raw_path, '-o', zstd_path],
                           check=True, capture_output=True)
            zstd_time = time.time() - t0
            zstd_size = os.path.getsize(zstd_path)
            total_zstd += zstd_size
            os.unlink(raw_path)
            os.unlink(zstd_path)
        else:
            zstd_size = 0
            zstd_time = 0

        # bzip2 -9
        t0 = time.time()
        bzip2_data = bz2.compress(raw_bytes, compresslevel=9)
        bzip2_time = time.time() - t0
        bzip2_size = len(bzip2_data)
        total_bzip2 += bzip2_size

        table_b_results.append({
            'table': t,
            'shape': list(w.shape),
            'raw_int8_bytes': raw_size,
            'fp32_bytes': w.numel() * 4,
            'gzip_bytes': gzip_size,
            'gzip_ratio_vs_fp32': (w.numel() * 4) / gzip_size if gzip_size > 0 else 0,
            'gzip_ratio_vs_int8': raw_size / gzip_size if gzip_size > 0 else 0,
            'gzip_time': gzip_time,
            'zstd_bytes': zstd_size,
            'zstd_ratio_vs_fp32': (w.numel() * 4) / zstd_size if zstd_size > 0 else 0,
            'zstd_ratio_vs_int8': raw_size / zstd_size if zstd_size > 0 else 0,
            'zstd_time': zstd_time,
            'bzip2_bytes': bzip2_size,
            'bzip2_ratio_vs_fp32': (w.numel() * 4) / bzip2_size if bzip2_size > 0 else 0,
            'bzip2_ratio_vs_int8': raw_size / bzip2_size if bzip2_size > 0 else 0,
            'bzip2_time': bzip2_time,
        })

        if t % 5 == 0 or t == num_tables - 1:
            print(f"  Table {t:2d}: INT8={raw_size:>12,}  "
                  f"gzip={gzip_size:>10,} ({(w.numel()*4)/gzip_size:>6.1f}x)  "
                  f"zstd={zstd_size:>10,} ({(w.numel()*4)/zstd_size if zstd_size else 0:>6.1f}x)  "
                  f"bzip2={bzip2_size:>10,} ({(w.numel()*4)/bzip2_size:>6.1f}x)")

    print(f"\n  EXPERIMENT B TOTALS:")
    print(f"    gzip -9:   {total_gzip:>12,} bytes ({total_gzip/1024/1024:>8.2f} MB) = "
          f"{total_fp32/total_gzip:.2f}x vs FP32, {total_int8_size/total_gzip:.2f}x vs INT8")
    if has_zstd:
        print(f"    zstd -19:  {total_zstd:>12,} bytes ({total_zstd/1024/1024:>8.2f} MB) = "
              f"{total_fp32/total_zstd:.2f}x vs FP32, {total_int8_size/total_zstd:.2f}x vs INT8")
    print(f"    bzip2 -9:  {total_bzip2:>12,} bytes ({total_bzip2/1024/1024:>8.2f} MB) = "
          f"{total_fp32/total_bzip2:.2f}x vs FP32, {total_int8_size/total_bzip2:.2f}x vs INT8")
    print(f"    (AUC: same as Experiment A = {int8_auc:.6f}, all lossless)")

    # ================================================================
    # EXPERIMENT B2: Entropy coding on FP32 directly (for reference)
    # ================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT B2: Entropy Coding on Raw FP32 (reference)")
    print("  gzip -9, zstd -19, bzip2 -9 on original FP32 bytes")
    print("=" * 60)

    total_gzip_fp32 = 0
    total_zstd_fp32 = 0
    total_bzip2_fp32 = 0

    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        # Convert to raw bytes in native format
        raw_fp32 = w.numpy().tobytes()
        raw_size = len(raw_fp32)

        gzip_data = gzip.compress(raw_fp32, compresslevel=9)
        total_gzip_fp32 += len(gzip_data)

        if has_zstd:
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
                f.write(raw_fp32)
                raw_path = f.name
            zstd_path = raw_path + '.zst'
            subprocess.run(['zstd', '-19', '-f', '-q', raw_path, '-o', zstd_path],
                           check=True, capture_output=True)
            total_zstd_fp32 += os.path.getsize(zstd_path)
            os.unlink(raw_path)
            os.unlink(zstd_path)

        bzip2_data = bz2.compress(raw_fp32, compresslevel=9)
        total_bzip2_fp32 += len(bzip2_data)

        if t % 5 == 0 or t == num_tables - 1:
            print(f"  Table {t:2d}: FP32={raw_size:>12,}  "
                  f"gzip={len(gzip_data):>10,} ({raw_size/len(gzip_data):>5.1f}x)  "
                  f"bzip2={len(bzip2_data):>10,} ({raw_size/len(bzip2_data):>5.1f}x)")

    print(f"\n  EXPERIMENT B2 TOTALS (FP32 entropy coding):")
    print(f"    gzip -9:   {total_gzip_fp32:>12,} bytes ({total_gzip_fp32/1024/1024:>8.2f} MB) = "
          f"{total_fp32/total_gzip_fp32:.2f}x")
    if has_zstd:
        print(f"    zstd -19:  {total_zstd_fp32:>12,} bytes ({total_zstd_fp32/1024/1024:>8.2f} MB) = "
              f"{total_fp32/total_zstd_fp32:.2f}x")
    print(f"    bzip2 -9:  {total_bzip2_fp32:>12,} bytes ({total_bzip2_fp32/1024/1024:>8.2f} MB) = "
          f"{total_fp32/total_bzip2_fp32:.2f}x")

    # ================================================================
    # Load existing Phase 2A and Phase 8 data for C/D/E
    # ================================================================
    phase2a_path = os.path.join(RESULTS_DIR, "retrained_uniform_results.json")
    phase8_path = os.path.join(RESULTS_DIR, "retrained_16x16_frame_results.json")

    phase2a_data = None
    phase8_data = None

    if os.path.exists(phase2a_path):
        with open(phase2a_path) as f:
            phase2a_data = json.load(f)
        print(f"\n  Loaded Phase 2A data: {len(phase2a_data['results'])} CRF levels")

    if os.path.exists(phase8_path):
        with open(phase8_path) as f:
            phase8_data = json.load(f)
        print(f"  Loaded Phase 8 data: {len(phase8_data['results'])} CRF levels")

    # ================================================================
    # WRITE COMPREHENSIVE RESULTS
    # ================================================================
    print("\n" + "=" * 60)
    print("WRITING RESULTS")
    print("=" * 60)

    # Helper to find Phase 2A result by CRF
    def get_phase2a(crf):
        if phase2a_data:
            for r in phase2a_data['results']:
                if r['crf'] == crf:
                    return r
        return None

    # Helper to find Phase 8 result by CRF
    def get_phase8(crf):
        if phase8_data:
            for r in phase8_data['results']:
                if r['crf'] == crf:
                    return r
        return None

    out_path = os.path.join(RESULTS_DIR, "codec_pipeline_isolation_results.md")
    with open(out_path, 'w') as f:
        f.write("# Codec Pipeline Isolation: Per-Algorithm Contribution\n\n")

        f.write("## Overview\n\n")
        f.write("This experiment measures the individual contribution of each algorithm\n")
        f.write("in the video codec embedding compression pipeline.\n\n")

        f.write("## Baseline\n\n")
        f.write(f"- **Model:** `{MODEL_PATH}`\n")
        f.write(f"- **Embedding tables:** {num_tables}\n")
        f.write(f"- **Total FP32 size:** {total_fp32:,} bytes ({total_fp32/1024/1024:.2f} MB)\n")
        f.write(f"- **Baseline accuracy:** {baseline_acc*100:.4f}%\n")
        f.write(f"- **Baseline AUC:** {baseline_auc:.6f}\n")
        f.write(f"- **Baseline inference time:** {baseline_infer:.2f}s\n\n")

        # ============================================================
        # MASTER SUMMARY TABLE
        # ============================================================
        f.write("## Master Pipeline Summary\n\n")
        f.write("| Stage | Method | Total Size (MB) | Ratio vs FP32 | Ratio vs Previous | AUC | AUC Loss (pp) |\n")
        f.write("|-------|--------|----------------|--------------|-------------------|-----|---------------|\n")

        # Row 1: Original FP32
        f.write(f"| Original | FP32 | {total_fp32/1024/1024:.2f} | 1.00x | — | "
                f"{baseline_auc:.6f} | 0.0000 |\n")

        # Row 2: INT8 Quantization
        f.write(f"| +INT8 Quantization | Raw UINT8 | {total_int8_size/1024/1024:.2f} | "
                f"{total_fp32/total_int8_size:.2f}x | "
                f"{total_fp32/total_int8_size:.2f}x | "
                f"{int8_auc:.6f} | {int8_auc_loss*100:.4f} |\n")

        # Row 3: gzip on FP32 (reference)
        f.write(f"| Entropy only (FP32) | gzip -9 on FP32 | {total_gzip_fp32/1024/1024:.2f} | "
                f"{total_fp32/total_gzip_fp32:.2f}x | "
                f"{total_fp32/total_gzip_fp32:.2f}x | "
                f"{baseline_auc:.6f} | 0.0000 |\n")

        # Row 4: gzip on INT8
        gzip_vs_prev = total_int8_size / total_gzip
        f.write(f"| +Entropy (INT8) | gzip -9 | {total_gzip/1024/1024:.2f} | "
                f"{total_fp32/total_gzip:.2f}x | "
                f"{gzip_vs_prev:.2f}x | "
                f"{int8_auc:.6f} | {int8_auc_loss*100:.4f} |\n")

        # Row 5: zstd on INT8
        if has_zstd and total_zstd > 0:
            zstd_vs_prev = total_int8_size / total_zstd
            f.write(f"| +Entropy (INT8) | zstd -19 | {total_zstd/1024/1024:.2f} | "
                    f"{total_fp32/total_zstd:.2f}x | "
                    f"{zstd_vs_prev:.2f}x | "
                    f"{int8_auc:.6f} | {int8_auc_loss*100:.4f} |\n")

        # Row 6: bzip2 on INT8
        bzip2_vs_prev = total_int8_size / total_bzip2
        f.write(f"| +Entropy (INT8) | bzip2 -9 | {total_bzip2/1024/1024:.2f} | "
                f"{total_fp32/total_bzip2:.2f}x | "
                f"{bzip2_vs_prev:.2f}x | "
                f"{int8_auc:.6f} | {int8_auc_loss*100:.4f} |\n")

        # Rows from Phase 2A: H.265 single-frame
        crf_key_levels = [0, 18, 20, 23, 25, 28, 30, 35, 38, 43, 51]
        best_entropy = min(total_gzip, total_zstd if total_zstd > 0 else float('inf'), total_bzip2)

        for crf in crf_key_levels:
            r = get_phase2a(crf)
            if r:
                comp_bytes = r['compressed_size']
                comp_mb = comp_bytes / 1024 / 1024
                cumul_ratio = total_fp32 / comp_bytes if comp_bytes > 0 else 0
                auc_loss_pp = r.get('auc_loss', 0) * 100
                vs_prev_label = ""
                if crf == 0:
                    # vs best entropy coding
                    vs_prev = best_entropy / comp_bytes if comp_bytes > 0 else 0
                    stage = "+Intra Pred + CABAC"
                    method = "H.265 CRF 0 (lossless)"
                else:
                    stage = "+Lossy Transform Quant"
                    method = f"H.265 CRF {crf}"
                    # vs CRF 0
                    r0 = get_phase2a(0)
                    if r0:
                        vs_prev = r0['compressed_size'] / comp_bytes if comp_bytes > 0 else 0
                    else:
                        vs_prev = 0

                f.write(f"| {stage} | {method} | {comp_mb:.2f} | "
                        f"{cumul_ratio:.2f}x | "
                        f"{vs_prev:.2f}x | "
                        f"{r['auc']:.6f} | {auc_loss_pp:.4f} |\n")

        # Rows from Phase 8: 16x16 multi-frame
        for crf in [0, 23]:
            r = get_phase8(crf)
            if r:
                comp_bytes = r['compressed_size']
                comp_mb = comp_bytes / 1024 / 1024
                cumul_ratio = total_fp32 / comp_bytes if comp_bytes > 0 else 0
                auc_loss_pp = r.get('auc_loss_pp', r.get('auc_loss', 0) * 100)

                # Compare to single-frame at same CRF
                r_sf = get_phase2a(crf)
                if r_sf:
                    # multi-frame is WORSE, so ratio < 1
                    vs_single = comp_bytes / r_sf['compressed_size']
                    note = f"{vs_single:.1f}x larger than single-frame"
                else:
                    note = ""

                f.write(f"| +Inter-frame (16x16) | Multi-frame CRF {crf} | {comp_mb:.2f} | "
                        f"{cumul_ratio:.2f}x | {note} | "
                        f"{r['auc']:.6f} | {auc_loss_pp:.4f} |\n")

        # ============================================================
        # DETAILED EXPERIMENT A RESULTS
        # ============================================================
        f.write("\n## Experiment A: Raw INT8 Quantization (Per-Table)\n\n")
        f.write("FP32 → UINT8 asymmetric min/max quantization, no compression.\n\n")
        f.write(f"- **Total INT8 size:** {total_int8_size:,} bytes ({total_int8_size/1024/1024:.2f} MB)\n")
        f.write(f"- **Compression ratio:** {total_fp32/total_int8_size:.2f}x\n")
        f.write(f"- **Accuracy:** {int8_acc*100:.4f}% (loss {int8_acc_loss:.4f}%)\n")
        f.write(f"- **AUC:** {int8_auc:.6f} (loss {int8_auc_loss*100:.4f}pp)\n")
        f.write(f"- **Inference time:** {int8_infer:.2f}s\n\n")

        f.write("| Table | Shape | FP32 (bytes) | INT8 (bytes) | Ratio | Scale | Zero Point |\n")
        f.write("|-------|-------|-------------|-------------|-------|-------|------------|\n")
        for r in table_a_results:
            f.write(f"| {r['table']} | {r['shape']} | {r['fp32_bytes']:,} | "
                    f"{r['int8_bytes']:,} | {r['ratio']:.2f}x | "
                    f"{r['scale']:.8f} | {r['zero_point']} |\n")

        # ============================================================
        # DETAILED EXPERIMENT B RESULTS
        # ============================================================
        f.write("\n## Experiment B: INT8 + Generic Entropy Coding (Per-Table)\n\n")
        f.write("Lossless compression of UINT8 bytes — no spatial prediction.\n")
        f.write("AUC is identical to Experiment A (all lossless).\n\n")

        f.write("### Totals\n\n")
        f.write("| Method | Total (bytes) | Total (MB) | Ratio vs FP32 | Ratio vs INT8 |\n")
        f.write("|--------|-------------|-----------|--------------|---------------|\n")
        f.write(f"| Raw INT8 | {total_int8_size:,} | {total_int8_size/1024/1024:.2f} | "
                f"{total_fp32/total_int8_size:.2f}x | 1.00x |\n")
        f.write(f"| gzip -9 | {total_gzip:,} | {total_gzip/1024/1024:.2f} | "
                f"{total_fp32/total_gzip:.2f}x | {total_int8_size/total_gzip:.2f}x |\n")
        if has_zstd and total_zstd > 0:
            f.write(f"| zstd -19 | {total_zstd:,} | {total_zstd/1024/1024:.2f} | "
                    f"{total_fp32/total_zstd:.2f}x | {total_int8_size/total_zstd:.2f}x |\n")
        f.write(f"| bzip2 -9 | {total_bzip2:,} | {total_bzip2/1024/1024:.2f} | "
                f"{total_fp32/total_bzip2:.2f}x | {total_int8_size/total_bzip2:.2f}x |\n")

        f.write("\n### Per-Table Detail\n\n")
        f.write("| Table | Shape | INT8 (bytes) | gzip (bytes) | gzip Ratio | "
                "zstd (bytes) | zstd Ratio | bzip2 (bytes) | bzip2 Ratio |\n")
        f.write("|-------|-------|-------------|-------------|-----------|"
                "-------------|-----------|--------------|-------------|\n")
        for r in table_b_results:
            f.write(f"| {r['table']} | {r['shape']} | {r['raw_int8_bytes']:,} | "
                    f"{r['gzip_bytes']:,} | {r['gzip_ratio_vs_fp32']:.1f}x | "
                    f"{r['zstd_bytes']:,} | {r['zstd_ratio_vs_fp32']:.1f}x | "
                    f"{r['bzip2_bytes']:,} | {r['bzip2_ratio_vs_fp32']:.1f}x |\n")

        # ============================================================
        # EXPERIMENT B2: FP32 entropy coding
        # ============================================================
        f.write("\n## Experiment B2: Entropy Coding on Raw FP32 (Reference)\n\n")
        f.write("How much can generic compressors reduce FP32 without quantization?\n\n")
        f.write("| Method | Total (bytes) | Total (MB) | Ratio vs FP32 |\n")
        f.write("|--------|-------------|-----------|---------------|\n")
        f.write(f"| Raw FP32 | {total_fp32:,} | {total_fp32/1024/1024:.2f} | 1.00x |\n")
        f.write(f"| gzip -9 | {total_gzip_fp32:,} | {total_gzip_fp32/1024/1024:.2f} | "
                f"{total_fp32/total_gzip_fp32:.2f}x |\n")
        if has_zstd and total_zstd_fp32 > 0:
            f.write(f"| zstd -19 | {total_zstd_fp32:,} | {total_zstd_fp32/1024/1024:.2f} | "
                    f"{total_fp32/total_zstd_fp32:.2f}x |\n")
        f.write(f"| bzip2 -9 | {total_bzip2_fp32:,} | {total_bzip2_fp32/1024/1024:.2f} | "
                f"{total_fp32/total_bzip2_fp32:.2f}x |\n")

        # ============================================================
        # ALGORITHM CONTRIBUTION ANALYSIS
        # ============================================================
        f.write("\n## Algorithm Contribution Analysis\n\n")
        f.write("Incremental contribution of each algorithm stage:\n\n")

        r0 = get_phase2a(0)
        r23 = get_phase2a(23)
        r35 = get_phase2a(35)

        f.write("### Compression Ratio Buildup\n\n")
        f.write("| Step | What It Does | Cumulative Ratio | Incremental Factor | AUC Loss (pp) | Incremental AUC Cost |\n")
        f.write("|------|-------------|-----------------|-------------------|--------------|---------------------|\n")

        f.write(f"| 1. INT8 Quantization | FP32→UINT8 (4:1 fixed) | "
                f"{total_fp32/total_int8_size:.1f}x | {total_fp32/total_int8_size:.1f}x | "
                f"{int8_auc_loss*100:.4f} | {int8_auc_loss*100:.4f}pp |\n")

        best_ent_name = "gzip -9"
        best_ent_size = total_gzip
        if has_zstd and total_zstd > 0 and total_zstd < total_gzip:
            best_ent_name = "zstd -19"
            best_ent_size = total_zstd
        if total_bzip2 < best_ent_size:
            best_ent_name = "bzip2 -9"
            best_ent_size = total_bzip2

        ent_incr = total_int8_size / best_ent_size
        f.write(f"| 2. Entropy coding only | {best_ent_name} on UINT8 | "
                f"{total_fp32/best_ent_size:.1f}x | {ent_incr:.1f}x | "
                f"{int8_auc_loss*100:.4f} | 0.0000pp (lossless) |\n")

        if r0:
            h265_crf0_bytes = r0['compressed_size']
            intra_incr = best_ent_size / h265_crf0_bytes
            h265_crf0_auc_loss = r0.get('auc_loss', 0)
            # CRF 0 is nominally lossless for the codec, but there's a tiny
            # difference vs raw INT8 due to the uint8 roundtrip
            f.write(f"| 3. Intra prediction + CABAC | H.265 CRF 0 spatial pred | "
                    f"{total_fp32/h265_crf0_bytes:.1f}x | {intra_incr:.1f}x | "
                    f"{h265_crf0_auc_loss*100:.4f} | ~0pp (lossless) |\n")

            if r23:
                r23_bytes = r23['compressed_size']
                lossy23_incr = h265_crf0_bytes / r23_bytes
                r23_auc_loss = r23.get('auc_loss', 0) * 100
                r23_auc_incr = r23_auc_loss - h265_crf0_auc_loss * 100
                f.write(f"| 4a. Lossy quant (CRF 23) | Quantize DCT coefficients | "
                        f"{total_fp32/r23_bytes:.1f}x | {lossy23_incr:.1f}x | "
                        f"{r23_auc_loss:.4f} | {r23_auc_incr:.4f}pp |\n")

            if r35:
                r35_bytes = r35['compressed_size']
                lossy35_incr = h265_crf0_bytes / r35_bytes
                r35_auc_loss = r35.get('auc_loss', 0) * 100
                r35_auc_incr = r35_auc_loss - h265_crf0_auc_loss * 100
                f.write(f"| 4b. Lossy quant (CRF 35) | More aggressive quantization | "
                        f"{total_fp32/r35_bytes:.1f}x | {lossy35_incr:.1f}x | "
                        f"{r35_auc_loss:.4f} | {r35_auc_incr:.4f}pp |\n")

        # Inter-frame comparison
        r8_crf0 = get_phase8(0)
        if r8_crf0 and r0:
            f.write(f"| 5. Inter-frame (16x16) | Multi-frame CRF 0 | "
                    f"{total_fp32/r8_crf0['compressed_size']:.1f}x | "
                    f"{r0['compressed_size']/r8_crf0['compressed_size']:.2f}x (WORSE) | "
                    f"{r8_crf0.get('auc_loss_pp', r8_crf0.get('auc_loss',0)*100):.4f} | "
                    f"N/A (different pipeline) |\n")

        # ============================================================
        # KEY FINDINGS
        # ============================================================
        f.write("\n## Key Findings\n\n")

        f.write("### 1. INT8 Quantization\n")
        f.write(f"- Fixed 4.00x compression (FP32→UINT8)\n")
        f.write(f"- AUC cost: {int8_auc_loss*100:.4f}pp — extremely cheap\n")
        f.write(f"- This is a prerequisite for the codec pipeline (converts to pixel values)\n\n")

        f.write("### 2. Generic Entropy Coding (no spatial prediction)\n")
        f.write(f"- Best generic compressor: {best_ent_name} = {total_fp32/best_ent_size:.1f}x total\n")
        f.write(f"- Incremental factor over raw INT8: {ent_incr:.1f}x\n")
        f.write(f"- No additional AUC cost (lossless)\n")
        f.write(f"- Entropy in the embedding weights is limited — generic compressors can only do so much\n\n")

        if r0:
            f.write("### 3. Intra Prediction + CABAC (H.265 CRF 0)\n")
            f.write(f"- Total: {total_fp32/h265_crf0_bytes:.1f}x (vs {total_fp32/best_ent_size:.1f}x best generic)\n")
            f.write(f"- Incremental over best entropy coding: {intra_incr:.1f}x\n")
            f.write(f"- H.265's spatial prediction (intra-frame) + CABAC dramatically outperforms generic entropy coders\n")
            f.write(f"- The spatial structure of embedding tables (nearby rows have similar values) is key\n\n")

        if r23:
            f.write("### 4. Lossy Transform Quantization (CRF > 0)\n")
            f.write(f"- CRF 23: {total_fp32/r23_bytes:.0f}x total, {lossy23_incr:.1f}x over lossless, "
                    f"costs {r23_auc_incr:.3f}pp AUC\n")
            if r35:
                f.write(f"- CRF 35: {total_fp32/r35_bytes:.0f}x total, {lossy35_incr:.1f}x over lossless, "
                        f"costs {r35_auc_incr:.3f}pp AUC\n")
            f.write(f"- Lossy quantization of DCT coefficients provides huge additional compression\n")
            f.write(f"- AUC degrades gracefully with increasing CRF\n\n")

        if r8_crf0 and r0:
            f.write("### 5. Inter-Frame Prediction (16x16 multi-frame)\n")
            f.write(f"- CRF 0 multi-frame: {total_fp32/r8_crf0['compressed_size']:.1f}x "
                    f"vs single-frame CRF 0: {total_fp32/r0['compressed_size']:.1f}x\n")
            f.write(f"- Inter-frame is {r8_crf0['compressed_size']/r0['compressed_size']:.1f}x WORSE "
                    f"than single-frame\n")
            f.write(f"- Per-frame overhead of encoding ~2M tiny 16x16 frames dominates\n")
            f.write(f"- Inter-frame prediction between consecutive embedding groups does not help\n")
            f.write(f"- The dominant compression comes from intra-frame prediction within large single frames\n\n")

        f.write("### Summary: Where Does the Compression Come From?\n\n")
        f.write("For the best single-frame pipeline (H.265 CRF 23 = 2054x):\n\n")

        if r0 and r23:
            total_ratio = total_fp32 / r23_bytes
            int8_contrib = 4.0
            ent_contrib = ent_incr
            intra_contrib = intra_incr
            lossy_contrib = lossy23_incr

            f.write(f"| Component | Factor | Cumulative | % of log(ratio) |\n")
            f.write(f"|-----------|--------|-----------|------------------|\n")

            import math
            total_log = math.log(total_ratio)
            log_int8 = math.log(int8_contrib)
            log_ent = math.log(ent_contrib)
            log_intra = math.log(intra_contrib)
            log_lossy = math.log(lossy_contrib)

            cum = int8_contrib
            f.write(f"| INT8 Quantization | {int8_contrib:.1f}x | {cum:.0f}x | "
                    f"{log_int8/total_log*100:.1f}% |\n")
            cum *= ent_contrib
            f.write(f"| Entropy Coding ({best_ent_name}) | {ent_contrib:.1f}x | {cum:.0f}x | "
                    f"{log_ent/total_log*100:.1f}% |\n")
            # Intra adds on top of the best entropy
            intra_over_total = best_ent_size / h265_crf0_bytes
            cum_after_intra = total_fp32 / h265_crf0_bytes
            f.write(f"| Intra Prediction + CABAC | {intra_over_total:.1f}x | {cum_after_intra:.0f}x | "
                    f"{math.log(intra_over_total)/total_log*100:.1f}% |\n")
            cum_final = total_fp32 / r23_bytes
            lossy_over = h265_crf0_bytes / r23_bytes
            f.write(f"| Lossy Transform Quant (CRF 23) | {lossy_over:.1f}x | {cum_final:.0f}x | "
                    f"{math.log(lossy_over)/total_log*100:.1f}% |\n")

    print(f"\nResults written to: {out_path}")

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "codec_pipeline_isolation_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            'baseline': {
                'accuracy': baseline_acc,
                'auc': baseline_auc,
                'inference_time': baseline_infer,
                'total_fp32_size': total_fp32,
                'num_tables': num_tables,
            },
            'experiment_a': {
                'description': 'Raw INT8 quantization (no codec)',
                'total_int8_size': total_int8_size,
                'compression_ratio': total_fp32 / total_int8_size,
                'accuracy': int8_acc,
                'auc': int8_auc,
                'auc_loss': int8_auc_loss,
                'auc_loss_pp': int8_auc_loss * 100,
                'inference_time': int8_infer,
                'per_table': table_a_results,
            },
            'experiment_b': {
                'description': 'INT8 + generic entropy coding',
                'gzip': {
                    'total_bytes': total_gzip,
                    'ratio_vs_fp32': total_fp32 / total_gzip,
                    'ratio_vs_int8': total_int8_size / total_gzip,
                },
                'zstd': {
                    'total_bytes': total_zstd,
                    'ratio_vs_fp32': total_fp32 / total_zstd if total_zstd > 0 else 0,
                    'ratio_vs_int8': total_int8_size / total_zstd if total_zstd > 0 else 0,
                },
                'bzip2': {
                    'total_bytes': total_bzip2,
                    'ratio_vs_fp32': total_fp32 / total_bzip2,
                    'ratio_vs_int8': total_int8_size / total_bzip2,
                },
                'auc': int8_auc,
                'auc_loss_pp': int8_auc_loss * 100,
                'per_table': table_b_results,
            },
            'experiment_b2_fp32_entropy': {
                'description': 'Entropy coding on raw FP32 (reference)',
                'gzip_fp32_bytes': total_gzip_fp32,
                'gzip_fp32_ratio': total_fp32 / total_gzip_fp32,
                'zstd_fp32_bytes': total_zstd_fp32,
                'zstd_fp32_ratio': total_fp32 / total_zstd_fp32 if total_zstd_fp32 > 0 else 0,
                'bzip2_fp32_bytes': total_bzip2_fp32,
                'bzip2_fp32_ratio': total_fp32 / total_bzip2_fp32,
            },
        }, f, indent=2)
    print(f"JSON saved to: {json_path}")

    print("\n" + "=" * 80)
    print("CODEC PIPELINE ISOLATION EXPERIMENTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
