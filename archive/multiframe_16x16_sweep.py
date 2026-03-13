#!/usr/bin/env python3
"""
Multi-frame 16x16 encoding experiment.
Instead of packing all embeddings into one frame, group every 16 consecutive
embedding rows into a 16x16 frame. This enables H.265 inter-frame prediction
between consecutive groups of 16 embeddings.

CRF sweep: 0, 15, 18, 20, 23, 25, 28, 30, 35, 38, 43, 51
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
from dlrm_s_pytorch import DLRM_Net

MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
DATA_FILE = "./input/train.txt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
RESULTS_DIR = os.path.expanduser("~/experiment-control")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 16384

CRF_LEVELS = [0, 15, 18, 20, 23, 25, 28, 30, 35, 38, 43, 51]

FRAME_WIDTH = 16   # = emb_dim
FRAME_HEIGHT = 16  # = 16 embeddings per frame
ROWS_PER_FRAME = 16


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
    return dlrm, test_ld, ln_emb, ln_bot, ln_top, m_spa, args


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
    inference_time = time.time() - t0
    accuracy = test_accu / test_samp
    auc = roc_auc_score(all_targets, all_scores)
    return accuracy, auc, inference_time


def get_codec_args(crf):
    if crf == 0:
        return ['-c:v', 'libx265', '-x265-params', 'lossless=1:log-level=error',
                '-preset', 'ultrafast']
    else:
        return ['-c:v', 'libx265', '-crf', str(crf), '-preset', 'ultrafast',
                '-x265-params', 'log-level=error:allow-non-conformance=1']


def compress_table_multiframe(weights, codec_cmd_args):
    """Compress an embedding table as a multi-frame 16x16 video.

    Each frame is 16x16 = 16 consecutive embeddings (each row is one embedding of dim 16).
    Inter-frame prediction exploits similarity between consecutive groups of embeddings.
    """
    num_emb, emb_dim = weights.shape
    assert emb_dim == 16, f"Expected emb_dim=16, got {emb_dim}"

    # Quantize to uint8
    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / 255.0
    zero_point = -(w_min / scale).round() if scale > 0 else 0.0
    quantized = ((weights / scale).round() + zero_point).clamp(0, 255).to(torch.uint8)
    pixels = quantized.numpy()

    # Pad to multiple of ROWS_PER_FRAME
    num_frames = (num_emb + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
    padded_rows = num_frames * ROWS_PER_FRAME
    if padded_rows > num_emb:
        pad = np.zeros((padded_rows - num_emb, emb_dim), dtype=np.uint8)
        pixels = np.vstack([pixels, pad])

    # Raw data: num_frames frames, each 16x16
    raw_data = pixels.tobytes()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_file = os.path.join(tmpdir, 'pixels.raw')
        video_file = os.path.join(tmpdir, 'compressed.mp4')
        with open(raw_file, 'wb') as f:
            f.write(raw_data)

        cmd = ['ffmpeg', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}',
               '-r', '30',
               '-i', raw_file
               ] + codec_cmd_args + [video_file]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        compress_time = time.time() - t0

        if result.returncode != 0:
            raise RuntimeError(f"Encoding failed: {result.stderr[:500]}")

        with open(video_file, 'rb') as f:
            compressed_data = f.read()

    metadata = {
        'shape': (num_emb, emb_dim),
        'quant_params': {'scale': float(scale), 'zero_point': float(zero_point)},
        'num_frames': num_frames,
        'padded_rows': padded_rows,
    }
    return compressed_data, metadata, compress_time


def decompress_table_multiframe(compressed_data, metadata):
    """Decompress a multi-frame 16x16 video back to embedding weights."""
    num_emb, emb_dim = metadata['shape']

    with tempfile.TemporaryDirectory() as tmpdir:
        video_file = os.path.join(tmpdir, 'compressed.mp4')
        raw_file = os.path.join(tmpdir, 'decoded.raw')
        with open(video_file, 'wb') as f:
            f.write(compressed_data)

        cmd = ['ffmpeg', '-y', '-i', video_file,
               '-pix_fmt', 'gray', '-f', 'rawvideo', raw_file]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        decompress_time = time.time() - t0

        if result.returncode != 0:
            raise RuntimeError(f"Decoding failed: {result.stderr[:500]}")

        pixels_uint8 = np.fromfile(raw_file, dtype=np.uint8)

    # Trim padding and reshape
    pixels_uint8 = pixels_uint8[:num_emb * emb_dim]
    pixels_uint8 = torch.from_numpy(pixels_uint8.copy()).reshape(num_emb, emb_dim)

    qp = metadata['quant_params']
    weights = (pixels_uint8.float() - qp['zero_point']) * qp['scale']
    return weights, decompress_time


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 80)
    print("MULTI-FRAME 16x16 ENCODING EXPERIMENT")
    print(f"Each frame = {ROWS_PER_FRAME} embeddings x {FRAME_WIDTH} dim = {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print("Inter-frame prediction enabled between consecutive embedding groups")
    print("=" * 80)

    dlrm, test_ld, ln_emb, ln_bot, ln_top, m_spa, args = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]
    total_emb_size = sum(state_dict[k].numel() * 4 for k in emb_keys)
    num_tables = len(emb_keys)

    print(f"  Model: {MODEL_PATH}")
    print(f"  Tables: {num_tables}")
    print(f"  Total emb size: {total_emb_size / 1024 / 1024:.2f} MB")

    # Print per-table frame counts
    for t, key in enumerate(emb_keys):
        n = state_dict[key].shape[0]
        nf = (n + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
        if t < 5 or n > 100000:
            print(f"  Table {t}: {n:,} rows -> {nf:,} frames")

    # Baseline
    print("\n  Running baseline inference...")
    drop_caches()
    baseline_acc, baseline_auc, baseline_infer = run_inference(dlrm, test_ld)
    print(f"  BASELINE: acc={baseline_acc * 100:.4f}%, AUC={baseline_auc:.6f}, "
          f"infer={baseline_infer:.2f}s")

    results = []

    for crf in CRF_LEVELS:
        drop_caches()
        print(f"\n{'=' * 60}")
        print(f"CRF = {crf}")
        print(f"{'=' * 60}")

        codec_args = get_codec_args(crf)
        tot_comp = 0
        tot_ct = 0.0
        tot_dt = 0.0

        for t in range(num_tables):
            weights = state_dict[emb_keys[t]]
            table_size = weights.numel() * 4
            try:
                cdata, meta, ct = compress_table_multiframe(weights, codec_args)
                dec_w, dt = decompress_table_multiframe(cdata, meta)
                tot_comp += len(cdata)
                tot_ct += ct
                tot_dt += dt
                with torch.no_grad():
                    dlrm.emb_l[t].weight.data = dec_w
                if t % 5 == 0:
                    print(f"  Table {t}/{num_tables}: {weights.shape} -> "
                          f"{len(cdata)} bytes ({table_size / len(cdata):.1f}x), "
                          f"{meta['num_frames']} frames")
            except Exception as e:
                print(f"  Table {t} FAILED: {e}")
                tot_comp += table_size

        # Run inference
        acc, auc, infer_t = run_inference(dlrm, test_ld)

        # Restore original weights
        for t in range(num_tables):
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

        comp_ratio = total_emb_size / tot_comp if tot_comp > 0 else 0

        result = {
            'crf': crf,
            'compression_ratio': comp_ratio,
            'accuracy': acc,
            'accuracy_pct': acc * 100,
            'auc': auc,
            'accuracy_loss_pct': (baseline_acc - acc) * 100,
            'auc_loss': baseline_auc - auc,
            'auc_loss_pp': (baseline_auc - auc) * 100,
            'compressed_size': tot_comp,
            'compressed_mb': tot_comp / 1024 / 1024,
            'compress_time': tot_ct,
            'decompress_time': tot_dt,
            'inference_time': infer_t,
        }
        results.append(result)

        print(f"  CRF {crf}: ratio={comp_ratio:.2f}x, "
              f"acc={acc * 100:.4f}% (loss {result['accuracy_loss_pct']:.4f}%), "
              f"AUC={auc:.6f} (loss {result['auc_loss_pp']:.4f}pp), "
              f"size={tot_comp / 1024 / 1024:.2f}MB, "
              f"ct={tot_ct:.2f}s, dt={tot_dt:.2f}s, infer={infer_t:.2f}s")

    # Write results
    out_path = os.path.join(RESULTS_DIR, "retrained_16x16_frame_results.md")
    with open(out_path, 'w') as f:
        f.write("# Multi-Frame 16x16 Encoding: Uniform CRF Sweep\n\n")
        f.write("## Method\n\n")
        f.write(f"- Each frame = {ROWS_PER_FRAME} consecutive embeddings x {FRAME_WIDTH} dim = "
                f"{FRAME_WIDTH}x{FRAME_HEIGHT} pixels\n")
        f.write("- Inter-frame prediction enabled (H.265 uses P/B frames between consecutive groups)\n")
        f.write("- No tiling, no hot/cold splitting\n")
        f.write("- ffmpeg `-r 30` framerate, libx265 ultrafast preset\n\n")

        f.write("## Baseline Model\n\n")
        f.write(f"- **Model:** `{MODEL_PATH}`\n")
        f.write(f"- **Embedding tables:** {num_tables}\n")
        f.write(f"- **Total embedding size:** {total_emb_size:,} bytes "
                f"({total_emb_size / 1024 / 1024:.2f} MB)\n")
        f.write(f"- **Baseline accuracy:** {baseline_acc * 100:.4f}%\n")
        f.write(f"- **Baseline AUC:** {baseline_auc:.6f}\n")
        f.write(f"- **Baseline inference time:** {baseline_infer:.2f}s\n\n")

        f.write("## CRF Sweep Results\n\n")
        f.write("| CRF | Comp Ratio | Accuracy (%) | AUC | AUC Loss (pp) | "
                "Compressed (MB) | Compress (s) | Decompress (s) | Inference (s) |\n")
        f.write("|-----|-----------|-------------|------|---------------|"
                "----------------|-------------|---------------|---------------|\n")
        for r in results:
            f.write(f"| {r['crf']} | {r['compression_ratio']:.2f}x "
                    f"| {r['accuracy_pct']:.4f} | {r['auc']:.6f} "
                    f"| {r['auc_loss_pp']:.4f} "
                    f"| {r['compressed_mb']:.2f} | {r['compress_time']:.2f} "
                    f"| {r['decompress_time']:.2f} | {r['inference_time']:.2f} |\n")

        # Comparison with Phase 2A single-frame results
        f.write("\n## Comparison with Phase 2A (Single-Frame per Table)\n\n")
        f.write("| CRF | 16x16 Multi-Frame Ratio | Single-Frame Ratio | "
                "16x16 AUC Loss (pp) | Single-Frame AUC Loss (pp) | Better? |\n")
        f.write("|-----|------------------------|-------------------|-"
                "--------------------|---------------------------|--------|\n")

        # Load Phase 2A results for comparison
        phase2a_path = os.path.join(RESULTS_DIR, "retrained_uniform_results.json")
        phase2a = {}
        if os.path.exists(phase2a_path):
            with open(phase2a_path) as pf:
                p2a_data = json.load(pf)
                for r2a in p2a_data.get('results', []):
                    phase2a[r2a['crf']] = r2a

        for r in results:
            crf = r['crf']
            if crf in phase2a:
                p = phase2a[crf]
                p_ratio = p['compression_ratio']
                p_auc_loss = p.get('auc_loss', 0) * 100
                better = "16x16" if r['compression_ratio'] > p_ratio else "Single"
                f.write(f"| {crf} | {r['compression_ratio']:.2f}x | {p_ratio:.2f}x "
                        f"| {r['auc_loss_pp']:.4f} | {p_auc_loss:.4f} | {better} |\n")
            else:
                f.write(f"| {crf} | {r['compression_ratio']:.2f}x | N/A "
                        f"| {r['auc_loss_pp']:.4f} | N/A | — |\n")

        f.write("\n## Validation\n\n")
        all_valid = True
        for r in results:
            issues = []
            if r['compression_ratio'] <= 1:
                issues.append(f"compression_ratio={r['compression_ratio']:.2f}")
            if not (0.5 <= r['accuracy'] <= 1.0):
                issues.append(f"accuracy={r['accuracy']:.4f}")
            if not (0.5 <= r['auc'] <= 1.0):
                issues.append(f"AUC={r['auc']:.6f}")
            if r['compressed_size'] <= 0:
                issues.append("compressed_size<=0")
            if r['compress_time'] <= 0:
                issues.append("compress_time<=0")
            if r['decompress_time'] <= 0:
                issues.append("decompress_time<=0")
            if issues:
                f.write(f"- FAIL CRF {r['crf']}: {', '.join(issues)}\n")
                all_valid = False
        if all_valid:
            f.write("- ALL CHECKS PASSED\n")

    print(f"\nResults written to: {out_path}")

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "retrained_16x16_frame_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            'method': f'{ROWS_PER_FRAME} embeddings per {FRAME_WIDTH}x{FRAME_HEIGHT} frame, multi-frame',
            'baseline': {
                'accuracy': baseline_acc,
                'auc': baseline_auc,
                'inference_time': baseline_infer,
                'total_emb_size': total_emb_size,
                'num_tables': num_tables,
            },
            'results': results,
        }, f, indent=2)
    print(f"JSON saved to: {json_path}")

    print("\n" + "=" * 80)
    print("MULTI-FRAME 16x16 EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
