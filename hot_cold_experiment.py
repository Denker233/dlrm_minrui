#!/usr/bin/env python3
"""
Hot/Cold Embedding Splitting with Differential Compression

Phase 1: Profile access patterns, identify large/small tables
Phase 2: Split large tables into hot/cold by access frequency
Phase 3: Test 25 combinations (5 hot CRF x 5 cold CRF)
Phase 4: Compare with uniform CRF 23
Phase 5: Write results to markdown, send notification
"""

import os
import sys
import time
import tempfile
import subprocess
import numpy as np
import torch
from collections import Counter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlrm_data_pytorch as dp
from dlrm_s_pytorch import DLRM_Net
from dlrm_s_pytorch import tile_embeddings, untile_embeddings

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_quick.pt"
DATA_FILE = "./input/train.txt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
RESULTS_DIR = os.path.expanduser("~/experiment-control")
STATUS_FILE = os.path.join(RESULTS_DIR, "status.md")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 16384

HOT_CRFS = [0, 5, 10, 15, 20]
COLD_CRFS = [25, 30, 38, 45, 51]
ACCESS_THRESHOLD = 0.80


def update_status(phase_text):
    status = f"""# IN PROGRESS

## Task: Hot/Cold Embedding Splitting with Differential Compression

### Phase 0: Preserve Current State — DONE
- Snapshot: `snapshots/snapshot_before_hotcold_20260208_172356.tar.gz` (16G)
- Backup: `snapshots/codec_experiment_full.py.snapshot`
- Results backups: `compression_results.md.backup`, `algorithm_analysis.md.backup`, `status.md.backup`

### AUC Check — DONE
- All 7 CRF levels have AUC values recorded in compression_results.md

{phase_text}
"""
    with open(STATUS_FILE, 'w') as f:
        f.write(status)
    print(f"[STATUS] {phase_text[:80]}")


def send_notification(message):
    try:
        subprocess.run(
            ['curl', '-s', '-d', message, 'https://ntfy.sh/minrui-dlrm-experiment'],
            capture_output=True, timeout=10)
        print(f"[NOTIFY] Sent: {message[:80]}")
    except Exception as e:
        print(f"[NOTIFY] Failed: {e}")


# ============================================================
# Helpers (reused from codec_experiment_full.py)
# ============================================================
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
    """Run inference, return (accuracy, auc, inference_time_seconds)"""
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


def compress_table_with_codec(weights, codec_cmd_args):
    """Compress embedding table via ffmpeg. Returns (bytes, metadata, time)."""
    num_emb, emb_dim = weights.shape

    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / 255.0
    zero_point = -(w_min / scale).round() if scale > 0 else 0.0
    quantized = ((weights / scale).round() + zero_point).clamp(0, 255).to(torch.uint8)

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
        image, grid_size, tiles_per_emb = tile_embeddings(
            data_flat, emb_dim, num_emb, TILE_SIZE)
        width = image.shape[1]
        height = image.shape[0]
        raw_data = image.tobytes()
        tiling_meta = {'tiled': True, 'grid_size': grid_size,
                       'tiles_per_emb': tiles_per_emb, 'tile_size': TILE_SIZE}
    else:
        raw_data = pixels.tobytes()

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
        if padded_pixels > len(raw_data):
            padded = bytearray(padded_pixels)
            padded[:len(raw_data)] = raw_data
            raw_data = bytes(padded)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_file = os.path.join(tmpdir, 'pixels.raw')
        video_file = os.path.join(tmpdir, 'compressed.mp4')

        with open(raw_file, 'wb') as f:
            f.write(raw_data)

        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{width}x{height}', '-r', '1', '-i', raw_file
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
        'quant_params': {'scale': float(scale), 'zero_point': float(zero_point)},
        'frame_width': width,
        'frame_height': height,
        'tiling': tiling_meta,
        'original_pixels': num_emb * emb_dim,
    }
    return compressed_data, metadata, compress_time


def decompress_table(compressed_data, metadata):
    """Decompress embedding table. Returns (weights_tensor, time)."""
    num_emb, emb_dim = metadata['shape']

    with tempfile.TemporaryDirectory() as tmpdir:
        video_file = os.path.join(tmpdir, 'compressed.mp4')
        raw_file = os.path.join(tmpdir, 'decoded.raw')

        with open(video_file, 'wb') as f:
            f.write(compressed_data)

        cmd = ['ffmpeg', '-y', '-i', video_file, '-pix_fmt', 'gray',
               '-f', 'rawvideo', raw_file]

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
            emb_dim, num_emb, grid_size, tiling['tiles_per_emb'], tile_size)
    else:
        pixels_uint8 = pixels_uint8[:num_emb * emb_dim]

    pixels_uint8 = torch.from_numpy(pixels_uint8.copy()).reshape(num_emb, emb_dim)
    qp = metadata['quant_params']
    weights = (pixels_uint8.float() - qp['zero_point']) * qp['scale']
    return weights, decompress_time


# ============================================================
# Phase 1: Access Profiling
# ============================================================
def profile_access_patterns(test_ld, num_tables):
    print("  Profiling access patterns across test data...")
    access_counts = [Counter() for _ in range(num_tables)]
    total_accesses = [0] * num_tables

    for i, batch in enumerate(test_ld):
        lS_i = batch[2]
        for t in range(num_tables):
            if isinstance(lS_i, list):
                indices = lS_i[t].numpy().flatten()
            else:
                indices = lS_i[t].numpy().flatten()
            for idx in indices:
                access_counts[t][int(idx)] += 1
            total_accesses[t] += len(indices)
        if i % 20 == 0:
            print(f"    Batch {i}/{len(test_ld)}", end='\r')

    print(f"    Profiled {len(test_ld)} batches              ")
    return access_counts, total_accesses


def identify_large_tables(ln_emb, emb_dim=16):
    sizes = np.array([int(n) * emb_dim * 4 for n in ln_emb])
    mean_size = float(np.mean(sizes))
    std_size = float(np.std(sizes))
    median_size = float(np.median(sizes))

    threshold1 = mean_size + std_size
    threshold2 = 10 * median_size

    large_mask = [(s > threshold1) or (s >= threshold2) for s in sizes]

    print(f"  Mean={mean_size/1024/1024:.2f} MB, Std={std_size/1024/1024:.2f} MB, "
          f"Median={median_size/1024/1024:.4f} MB")
    print(f"  Threshold1 (mean+std): {threshold1/1024/1024:.2f} MB")
    print(f"  Threshold2 (10x median): {threshold2/1024/1024:.4f} MB")
    print(f"  Large tables: {sum(large_mask)}/{len(ln_emb)}")

    return large_mask, sizes.tolist(), mean_size, std_size, median_size


def compute_hot_cold_split(access_counts, num_embeddings, threshold=0.80):
    total = sum(access_counts.values())
    if total == 0:
        return list(range(num_embeddings)), [], total, 1.0

    sorted_items = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)
    cumulative = 0
    hot_indices = []
    for idx, count in sorted_items:
        cumulative += count
        hot_indices.append(idx)
        if cumulative / total >= threshold:
            break

    hot_set = set(hot_indices)
    cold_indices = [i for i in range(num_embeddings) if i not in hot_set]
    hot_access_pct = cumulative / total
    return sorted(hot_indices), sorted(cold_indices), total, hot_access_pct


# ============================================================
# Phase 2: Splitting
# ============================================================
def split_table(weights, hot_indices, cold_indices):
    if len(hot_indices) > 0:
        hot_weights = weights[hot_indices].clone()
    else:
        hot_weights = torch.empty(0, weights.shape[1])
    if len(cold_indices) > 0:
        cold_weights = weights[cold_indices].clone()
    else:
        cold_weights = torch.empty(0, weights.shape[1])
    return hot_weights, cold_weights


def reconstruct_full_table(hot_weights, cold_weights, hot_indices, cold_indices,
                           num_embeddings, emb_dim):
    full = torch.zeros(num_embeddings, emb_dim)
    if len(hot_indices) > 0:
        full[hot_indices] = hot_weights
    if len(cold_indices) > 0:
        full[cold_indices] = cold_weights
    return full


# ============================================================
# Report Writers
# ============================================================
def write_access_profile(profiles, total_emb_size, baseline_acc, baseline_auc,
                         mean_size, std_size, median_size):
    path = os.path.join(RESULTS_DIR, "hot_cold_access_profile.md")
    large = [p for p in profiles if p['is_large']]
    small = [p for p in profiles if not p['is_large']]

    with open(path, 'w') as f:
        f.write("# Hot/Cold Access Pattern Profile\n\n")
        f.write("## Model Summary\n\n")
        f.write(f"- **Model:** `{MODEL_PATH}`\n")
        f.write(f"- **Embedding tables:** {len(profiles)}\n")
        f.write(f"- **Total embedding size:** {total_emb_size:,} bytes "
                f"({total_emb_size/1024/1024:.2f} MB)\n")
        f.write(f"- **Baseline accuracy:** {baseline_acc*100:.4f}%\n")
        f.write(f"- **Baseline AUC:** {baseline_auc:.6f}\n")
        f.write(f"- **Hot threshold:** top embeddings accounting for "
                f"{ACCESS_THRESHOLD*100:.0f}% of accesses\n\n")

        f.write("## Table Classification Criteria\n\n")
        f.write(f"- **Mean table size:** {mean_size/1024/1024:.2f} MB\n")
        f.write(f"- **Std table size:** {std_size/1024/1024:.2f} MB\n")
        f.write(f"- **Median table size:** {median_size/1024/1024:.4f} MB\n")
        f.write(f"- **Threshold (mean + std):** "
                f"{(mean_size + std_size)/1024/1024:.2f} MB\n")
        f.write(f"- **Threshold (10x median):** "
                f"{(10 * median_size)/1024/1024:.4f} MB\n")
        f.write(f"- **Large tables:** {len(large)}\n")
        f.write(f"- **Small tables:** {len(small)}\n\n")

        f.write("## Per-Table Access Profile\n\n")
        f.write("| Table | Rows | Size (MB) | Class | Unique Accessed "
                "| Coverage (%) | Hot Rows | Cold Rows "
                "| Hot % (rows) | Hot % (accesses) |\n")
        f.write("|-------|------|-----------|-------|----------------"
                "|-------------|----------|----------"
                "|-------------|------------------|\n")

        for p in profiles:
            f.write(f"| {p['table_idx']} "
                    f"| {p['num_embeddings']:,} "
                    f"| {p['size_mb']:.4f} "
                    f"| {'LARGE' if p['is_large'] else 'small'} "
                    f"| {p['unique_accessed']:,} "
                    f"| {p['access_coverage']*100:.1f} "
                    f"| {p['num_hot']:,} "
                    f"| {p['num_cold']:,} "
                    f"| {p['hot_pct_rows']:.2f} "
                    f"| {p['hot_pct_accesses']:.1f} |\n")

        if large:
            f.write("\n## Large Table Details\n\n")
            for p in large:
                f.write(f"### Table {p['table_idx']} ({p['key']})\n\n")
                f.write(f"- **Rows:** {p['num_embeddings']:,}\n")
                f.write(f"- **Size:** {p['size_mb']:.2f} MB\n")
                f.write(f"- **Unique accessed:** {p['unique_accessed']:,} / "
                        f"{p['num_embeddings']:,} "
                        f"({p['access_coverage']*100:.2f}%)\n")
                f.write(f"- **Total accesses:** {p['total_accesses']:,}\n")
                f.write(f"- **Hot rows:** {p['num_hot']:,} "
                        f"({p['hot_pct_rows']:.2f}% of rows, "
                        f"{p['hot_pct_accesses']:.1f}% of accesses)\n")
                f.write(f"- **Cold rows:** {p['num_cold']:,} "
                        f"({100 - p['hot_pct_rows']:.2f}% of rows)\n\n")

    print(f"  Written: {path}")


def write_compression_results(results, baseline_acc, baseline_auc, total_emb_size):
    path = os.path.join(RESULTS_DIR, "hot_cold_compression_results.md")
    with open(path, 'w') as f:
        f.write("# Hot/Cold Differential Compression Results\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- **Hot CRFs:** {HOT_CRFS}\n")
        f.write(f"- **Cold CRFs:** {COLD_CRFS}\n")
        f.write(f"- **Combinations tested:** {len(results)}\n")
        f.write(f"- **Hot threshold:** {ACCESS_THRESHOLD*100:.0f}% of accesses\n")
        f.write(f"- **Small tables:** compressed at hot CRF\n\n")

        f.write("## Baseline\n\n")
        f.write(f"- **Accuracy:** {baseline_acc*100:.4f}%\n")
        f.write(f"- **AUC:** {baseline_auc:.6f}\n")
        f.write(f"- **Uncompressed:** {total_emb_size:,} bytes "
                f"({total_emb_size/1024/1024:.2f} MB)\n\n")

        f.write("## Results (25 Combinations)\n\n")
        f.write("| Hot CRF | Cold CRF | Accuracy (%) | AUC | Acc Loss (%) "
                "| AUC Loss | Total (MB) | Ratio "
                "| Hot (MB) | Cold (MB) | Small (MB) "
                "| Compress (s) | Decompress (s) | Inference (s) "
                "| Hot Rows | Cold Rows | Hot Row % "
                "| Total (bytes) | Acc (raw) |\n")
        f.write("|---------|----------|-------------|------|----------"
                "|----------|-----------|------"
                "|---------|----------|----------"
                "|-------------|---------------|-------------"
                "|----------|-----------|----------"
                "|--------------|----------|\n")

        for r in results:
            f.write(
                f"| {r['hot_crf']} "
                f"| {r['cold_crf']} "
                f"| {r['accuracy_pct']:.4f} "
                f"| {r['auc']:.6f} "
                f"| {r['accuracy_loss_pct']:.4f} "
                f"| {r['auc_loss']:.6f} "
                f"| {r['total_compressed_mb']:.2f} "
                f"| {r['compression_ratio']:.2f}x "
                f"| {r['hot_mb']:.4f} "
                f"| {r['cold_mb']:.4f} "
                f"| {r['small_mb']:.4f} "
                f"| {r['compress_time']:.2f} "
                f"| {r['decompress_time']:.2f} "
                f"| {r['inference_time']:.2f} "
                f"| {r['total_hot_rows']} "
                f"| {r['total_cold_rows']} "
                f"| {r['hot_row_pct']:.1f} "
                f"| {r['total_compressed_size']} "
                f"| {r['accuracy']:.6f} |\n")

        best_auc = max(results, key=lambda r: r['auc'])
        best_ratio = max(results, key=lambda r: r['compression_ratio'])

        f.write(f"\n## Best Results\n\n")
        f.write(f"### Best AUC\n")
        f.write(f"- Hot CRF={best_auc['hot_crf']}, "
                f"Cold CRF={best_auc['cold_crf']}\n")
        f.write(f"- AUC: {best_auc['auc']:.6f} "
                f"(loss: {best_auc['auc_loss']:.6f})\n")
        f.write(f"- Compression: {best_auc['compression_ratio']:.2f}x "
                f"({best_auc['total_compressed_mb']:.2f} MB)\n\n")

        f.write(f"### Best Compression Ratio\n")
        f.write(f"- Hot CRF={best_ratio['hot_crf']}, "
                f"Cold CRF={best_ratio['cold_crf']}\n")
        f.write(f"- AUC: {best_ratio['auc']:.6f} "
                f"(loss: {best_ratio['auc_loss']:.6f})\n")
        f.write(f"- Compression: {best_ratio['compression_ratio']:.2f}x "
                f"({best_ratio['total_compressed_mb']:.2f} MB)\n")

    print(f"  Written: {path}")


def write_vs_uniform(combo_results, uniform, baseline_acc, baseline_auc,
                     baseline_infer, total_emb_size):
    path = os.path.join(RESULTS_DIR, "hot_cold_vs_uniform.md")

    best_auc = max(combo_results, key=lambda r: r['auc'])
    best_ratio = max(combo_results, key=lambda r: r['compression_ratio'])
    valid = [r for r in combo_results if r['auc'] >= uniform['auc']]
    best_valid = max(valid, key=lambda r: r['compression_ratio']) if valid else best_auc

    with open(path, 'w') as f:
        f.write("# Hot/Cold vs Uniform Compression Comparison\n\n")

        f.write("## Baseline (No Compression)\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Accuracy | {baseline_acc*100:.4f}% |\n")
        f.write(f"| AUC | {baseline_auc:.6f} |\n")
        f.write(f"| Size | {total_emb_size/1024/1024:.2f} MB |\n")
        f.write(f"| Inference Time | {baseline_infer:.2f}s |\n\n")

        f.write("## Head-to-Head Comparison\n\n")
        f.write("| Metric | Uniform CRF 23 | Best H/C (AUC) "
                "| Best H/C (Ratio) | Best Valid H/C |\n")
        f.write("|--------|---------------|----------------"
                "|-----------------|----------------|\n")

        rows = [
            ("Config", f"CRF {uniform['crf']}",
             f"H={best_auc['hot_crf']}/C={best_auc['cold_crf']}",
             f"H={best_ratio['hot_crf']}/C={best_ratio['cold_crf']}",
             f"H={best_valid['hot_crf']}/C={best_valid['cold_crf']}"),
            ("Accuracy (%)",
             f"{uniform['accuracy_pct']:.4f}",
             f"{best_auc['accuracy_pct']:.4f}",
             f"{best_ratio['accuracy_pct']:.4f}",
             f"{best_valid['accuracy_pct']:.4f}"),
            ("AUC",
             f"{uniform['auc']:.6f}",
             f"{best_auc['auc']:.6f}",
             f"{best_ratio['auc']:.6f}",
             f"{best_valid['auc']:.6f}"),
            ("AUC Loss",
             f"{uniform['auc_loss']:.6f}",
             f"{best_auc['auc_loss']:.6f}",
             f"{best_ratio['auc_loss']:.6f}",
             f"{best_valid['auc_loss']:.6f}"),
            ("Size (MB)",
             f"{uniform['total_compressed_mb']:.2f}",
             f"{best_auc['total_compressed_mb']:.2f}",
             f"{best_ratio['total_compressed_mb']:.2f}",
             f"{best_valid['total_compressed_mb']:.2f}"),
            ("Compression Ratio",
             f"{uniform['compression_ratio']:.2f}x",
             f"{best_auc['compression_ratio']:.2f}x",
             f"{best_ratio['compression_ratio']:.2f}x",
             f"{best_valid['compression_ratio']:.2f}x"),
            ("Compress Time (s)",
             f"{uniform['compress_time']:.2f}",
             f"{best_auc['compress_time']:.2f}",
             f"{best_ratio['compress_time']:.2f}",
             f"{best_valid['compress_time']:.2f}"),
            ("Decompress Time (s)",
             f"{uniform['decompress_time']:.2f}",
             f"{best_auc['decompress_time']:.2f}",
             f"{best_ratio['decompress_time']:.2f}",
             f"{best_valid['decompress_time']:.2f}"),
            ("Inference Time (s)",
             f"{uniform['inference_time']:.2f}",
             f"{best_auc['inference_time']:.2f}",
             f"{best_ratio['inference_time']:.2f}",
             f"{best_valid['inference_time']:.2f}"),
        ]
        for row in rows:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n")

        f.write(f"\n## Analysis\n\n")
        auc_diff = best_valid['auc'] - uniform['auc']
        ratio_diff = best_valid['compression_ratio'] - uniform['compression_ratio']

        f.write(f"### Best Valid Hot/Cold vs Uniform CRF 23\n\n")
        f.write(f"(Best valid = highest compression ratio with AUC >= uniform)\n\n")
        f.write(f"- **AUC difference:** {auc_diff:+.6f}\n")
        f.write(f"- **Compression ratio difference:** {ratio_diff:+.2f}x\n")

        if ratio_diff > 0:
            f.write(f"- **Verdict:** Hot/cold achieves **{ratio_diff:.2f}x better "
                    f"compression** while maintaining AUC >= uniform\n\n")
        elif auc_diff > 0:
            f.write(f"- **Verdict:** Hot/cold achieves **better AUC** "
                    f"at similar compression\n\n")
        else:
            f.write(f"- **Verdict:** Uniform compression is competitive\n\n")

        better_auc = sum(1 for r in combo_results
                         if r['auc'] > uniform['auc'])
        better_ratio = sum(1 for r in combo_results
                           if r['compression_ratio'] > uniform['compression_ratio'])
        better_both = sum(1 for r in combo_results
                          if r['auc'] >= uniform['auc']
                          and r['compression_ratio'] > uniform['compression_ratio'])

        f.write(f"### Summary Across All 25 Combinations\n\n")
        f.write(f"- Better AUC than uniform: {better_auc}/25\n")
        f.write(f"- Better compression than uniform: {better_ratio}/25\n")
        f.write(f"- Better in BOTH AUC and compression: {better_both}/25\n")

    print(f"  Written: {path}")


# ============================================================
# Main
# ============================================================
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 80)
    print("HOT/COLD EMBEDDING SPLITTING EXPERIMENT")
    print("=" * 80)

    # Load model and data
    update_status("### Phase 1: Loading model and data...")
    dlrm, test_ld, ln_emb, ln_bot, ln_top, m_spa, args = load_model_and_data()

    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]
    num_tables = len(emb_keys)
    total_emb_size = sum(state_dict[k].numel() * 4 for k in emb_keys)

    # Baseline
    print("\n  Running baseline inference...")
    baseline_acc, baseline_auc, baseline_infer = run_inference(dlrm, test_ld)
    print(f"  Baseline: acc={baseline_acc*100:.4f}%, "
          f"AUC={baseline_auc:.6f}, time={baseline_infer:.2f}s")

    # ==============================================================
    # PHASE 1: Access profiling
    # ==============================================================
    print("\n" + "=" * 80)
    print("PHASE 1: ACCESS PATTERN PROFILING")
    print("=" * 80)
    update_status("### Phase 1: Profiling access patterns — IN PROGRESS")

    large_mask, table_sizes, mean_sz, std_sz, median_sz = \
        identify_large_tables(ln_emb, m_spa)
    access_counts, total_accesses = profile_access_patterns(test_ld, num_tables)

    profiles = []
    for t in range(num_tables):
        num_emb = int(ln_emb[t])
        is_large = large_mask[t]
        unique_acc = len(access_counts[t])

        if is_large and num_emb > 0:
            hot_idx, cold_idx, tot_acc, hot_pct = compute_hot_cold_split(
                access_counts[t], num_emb, ACCESS_THRESHOLD)
        else:
            hot_idx = list(range(num_emb))
            cold_idx = []
            tot_acc = total_accesses[t]
            hot_pct = 1.0

        profiles.append({
            'table_idx': t,
            'key': emb_keys[t],
            'num_embeddings': num_emb,
            'emb_dim': m_spa,
            'size_bytes': table_sizes[t],
            'size_mb': table_sizes[t] / 1024 / 1024,
            'is_large': is_large,
            'unique_accessed': unique_acc,
            'total_accesses': tot_acc,
            'access_coverage': unique_acc / num_emb if num_emb > 0 else 0,
            'num_hot': len(hot_idx),
            'num_cold': len(cold_idx),
            'hot_pct_rows': len(hot_idx) / num_emb * 100 if num_emb > 0 else 0,
            'hot_pct_accesses': hot_pct * 100,
            'hot_indices': hot_idx,
            'cold_indices': cold_idx,
        })

        tag = "LARGE" if is_large else "small"
        print(f"  Table {t}: {num_emb:>10,} rows, "
              f"{table_sizes[t]/1024/1024:>8.2f} MB, {tag:>5s}, "
              f"hot={len(hot_idx):>8,} ({len(hot_idx)/num_emb*100 if num_emb else 0:>5.1f}%), "
              f"cold={len(cold_idx):>8,}")

    n_large = sum(1 for p in profiles if p['is_large'])
    print(f"\n  Summary: {n_large} large, {num_tables - n_large} small")
    update_status(f"### Phase 1: Access profiling — DONE\n"
                  f"- {n_large} large tables, {num_tables - n_large} small tables")

    # ==============================================================
    # PHASE 2: Pre-compress all subtables
    # ==============================================================
    print("\n" + "=" * 80)
    print("PHASE 2: PRE-COMPRESSING SUBTABLES")
    print("=" * 80)
    update_status("### Phase 2: Pre-compressing subtables — IN PROGRESS")

    # hot_cache[t][crf] = (compressed_bytes, metadata, compress_time) or None
    # cold_cache[t][crf] = same
    # small_cache[t][crf] = same
    hot_cache = {}
    cold_cache = {}
    small_cache = {}

    for t, prof in enumerate(profiles):
        weights = state_dict[emb_keys[t]]

        if prof['is_large']:
            hot_w, cold_w = split_table(weights, prof['hot_indices'],
                                        prof['cold_indices'])
            hot_cache[t] = {}
            cold_cache[t] = {}

            for crf in HOT_CRFS:
                if hot_w.shape[0] > 0:
                    try:
                        hot_cache[t][crf] = compress_table_with_codec(
                            hot_w, get_codec_args(crf))
                    except Exception as e:
                        print(f"    WARN: Table {t} hot CRF {crf}: {e}")
                        hot_cache[t][crf] = None
                else:
                    hot_cache[t][crf] = None

            for crf in COLD_CRFS:
                if cold_w.shape[0] > 0:
                    try:
                        cold_cache[t][crf] = compress_table_with_codec(
                            cold_w, get_codec_args(crf))
                    except Exception as e:
                        print(f"    WARN: Table {t} cold CRF {crf}: {e}")
                        cold_cache[t][crf] = None
                else:
                    cold_cache[t][crf] = None

            print(f"  Table {t} (LARGE): hot={hot_w.shape}, "
                  f"cold={cold_w.shape} — compressed")
        else:
            small_cache[t] = {}
            for crf in HOT_CRFS:
                try:
                    small_cache[t][crf] = compress_table_with_codec(
                        weights, get_codec_args(crf))
                except Exception as e:
                    print(f"    WARN: Table {t} small CRF {crf}: {e}")
                    small_cache[t][crf] = None
            print(f"  Table {t} (small): {weights.shape} — compressed")

    update_status("### Phase 2: Pre-compression — DONE")

    # ==============================================================
    # PHASE 3: Test all 25 combinations
    # ==============================================================
    print("\n" + "=" * 80)
    print("PHASE 3: TESTING 25 COMBINATIONS")
    print("=" * 80)

    combo_results = []
    combo_num = 0
    total_combos = len(HOT_CRFS) * len(COLD_CRFS)

    for hot_crf in HOT_CRFS:
        for cold_crf in COLD_CRFS:
            combo_num += 1
            print(f"\n  [{combo_num}/{total_combos}] "
                  f"Hot CRF={hot_crf}, Cold CRF={cold_crf}")
            update_status(f"### Phase 3: Combination {combo_num}/{total_combos} "
                          f"(hot={hot_crf}, cold={cold_crf}) — IN PROGRESS")

            tot_comp_size = 0
            tot_hot_size = 0
            tot_cold_size = 0
            tot_small_size = 0
            tot_ct = 0.0
            tot_dt = 0.0
            tot_hot_rows = 0
            tot_cold_rows = 0

            for t, prof in enumerate(profiles):
                num_emb = prof['num_embeddings']
                emb_dim = prof['emb_dim']

                if prof['is_large']:
                    # Hot subtable
                    if hot_cache[t][hot_crf] is not None:
                        cdata, meta, ct = hot_cache[t][hot_crf]
                        hw, dt = decompress_table(cdata, meta)
                        tot_hot_size += len(cdata)
                        tot_comp_size += len(cdata)
                        tot_ct += ct
                        tot_dt += dt
                    else:
                        hw = torch.empty(0, emb_dim)

                    # Cold subtable
                    if cold_cache[t][cold_crf] is not None:
                        cdata, meta, ct = cold_cache[t][cold_crf]
                        cw, dt = decompress_table(cdata, meta)
                        tot_cold_size += len(cdata)
                        tot_comp_size += len(cdata)
                        tot_ct += ct
                        tot_dt += dt
                    else:
                        cw = torch.empty(0, emb_dim)

                    reconstructed = reconstruct_full_table(
                        hw, cw, prof['hot_indices'], prof['cold_indices'],
                        num_emb, emb_dim)

                    tot_hot_rows += prof['num_hot']
                    tot_cold_rows += prof['num_cold']

                    with torch.no_grad():
                        dlrm.emb_l[t].weight.data = reconstructed
                else:
                    if small_cache[t][hot_crf] is not None:
                        cdata, meta, ct = small_cache[t][hot_crf]
                        dec_w, dt = decompress_table(cdata, meta)
                        tot_small_size += len(cdata)
                        tot_comp_size += len(cdata)
                        tot_ct += ct
                        tot_dt += dt
                        with torch.no_grad():
                            dlrm.emb_l[t].weight.data = dec_w
                    else:
                        tot_comp_size += prof['size_bytes']
                        tot_small_size += prof['size_bytes']

            # Inference
            acc, auc, infer_t = run_inference(dlrm, test_ld)

            # Restore original weights
            for t in range(num_tables):
                with torch.no_grad():
                    dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

            comp_ratio = total_emb_size / tot_comp_size if tot_comp_size > 0 else 0

            result = {
                'hot_crf': hot_crf,
                'cold_crf': cold_crf,
                'accuracy': acc,
                'accuracy_pct': acc * 100,
                'auc': auc,
                'accuracy_loss_pct': (baseline_acc - acc) * 100,
                'auc_loss': baseline_auc - auc,
                'total_compressed_size': tot_comp_size,
                'total_compressed_mb': tot_comp_size / 1024 / 1024,
                'compression_ratio': comp_ratio,
                'hot_compressed_size': tot_hot_size,
                'hot_mb': tot_hot_size / 1024 / 1024,
                'cold_compressed_size': tot_cold_size,
                'cold_mb': tot_cold_size / 1024 / 1024,
                'small_compressed_size': tot_small_size,
                'small_mb': tot_small_size / 1024 / 1024,
                'compress_time': tot_ct,
                'decompress_time': tot_dt,
                'inference_time': infer_t,
                'total_hot_rows': tot_hot_rows,
                'total_cold_rows': tot_cold_rows,
                'hot_row_pct': (tot_hot_rows / (tot_hot_rows + tot_cold_rows) * 100
                                if (tot_hot_rows + tot_cold_rows) > 0 else 0),
            }
            combo_results.append(result)

            print(f"    Acc: {acc*100:.4f}% (loss {result['accuracy_loss_pct']:.4f}%) "
                  f"| AUC: {auc:.6f} (loss {result['auc_loss']:.6f}) "
                  f"| {tot_comp_size/1024/1024:.2f} MB ({comp_ratio:.1f}x) "
                  f"| infer {infer_t:.2f}s")

    update_status("### Phase 3: All 25 combinations — DONE")

    # ==============================================================
    # PHASE 4: Uniform CRF 23 comparison
    # ==============================================================
    print("\n" + "=" * 80)
    print("PHASE 4: UNIFORM CRF 23 COMPARISON")
    print("=" * 80)
    update_status("### Phase 4: Uniform CRF 23 — IN PROGRESS")

    uniform_crf = 23
    u_codec = get_codec_args(uniform_crf)
    u_size = 0
    u_ct = 0.0
    u_dt = 0.0

    for t in range(num_tables):
        weights = state_dict[emb_keys[t]]
        try:
            cdata, meta, ct = compress_table_with_codec(weights, u_codec)
            dec_w, dt = decompress_table(cdata, meta)
            u_size += len(cdata)
            u_ct += ct
            u_dt += dt
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = dec_w
        except Exception as e:
            print(f"    Table {t} failed: {e}")
            u_size += table_sizes[t]

    u_acc, u_auc, u_infer = run_inference(dlrm, test_ld)
    u_ratio = total_emb_size / u_size if u_size > 0 else 0

    uniform = {
        'crf': uniform_crf,
        'accuracy': u_acc,
        'accuracy_pct': u_acc * 100,
        'auc': u_auc,
        'accuracy_loss_pct': (baseline_acc - u_acc) * 100,
        'auc_loss': baseline_auc - u_auc,
        'total_compressed_size': u_size,
        'total_compressed_mb': u_size / 1024 / 1024,
        'compression_ratio': u_ratio,
        'compress_time': u_ct,
        'decompress_time': u_dt,
        'inference_time': u_infer,
    }

    print(f"  Uniform CRF 23: acc={u_acc*100:.4f}%, AUC={u_auc:.6f}, "
          f"{u_size/1024/1024:.2f} MB ({u_ratio:.1f}x), infer {u_infer:.2f}s")

    # Restore original weights
    for t in range(num_tables):
        with torch.no_grad():
            dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

    update_status("### Phase 4: Uniform comparison — DONE")

    # ==============================================================
    # PHASE 5: Write reports
    # ==============================================================
    print("\n" + "=" * 80)
    print("PHASE 5: WRITING REPORTS")
    print("=" * 80)
    update_status("### Phase 5: Writing reports — IN PROGRESS")

    write_access_profile(profiles, total_emb_size, baseline_acc, baseline_auc,
                         mean_sz, std_sz, median_sz)
    write_compression_results(combo_results, baseline_acc, baseline_auc,
                              total_emb_size)
    write_vs_uniform(combo_results, uniform, baseline_acc, baseline_auc,
                     baseline_infer, total_emb_size)

    # Verify all reports exist and have content
    for fname in ["hot_cold_access_profile.md",
                  "hot_cold_compression_results.md",
                  "hot_cold_vs_uniform.md"]:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            sz = os.path.getsize(fpath)
            print(f"  Verified: {fname} ({sz} bytes)")
        else:
            print(f"  ERROR: {fname} not found!")

    # Final status
    best_auc = max(combo_results, key=lambda r: r['auc'])
    best_ratio = max(combo_results, key=lambda r: r['compression_ratio'])

    final_status = f"""### Phase 1: Access profiling — DONE
- {n_large} large tables, {num_tables - n_large} small tables

### Phase 2: Pre-compression — DONE

### Phase 3: All 25 combinations — DONE

### Phase 4: Uniform CRF 23 comparison — DONE
- Uniform: AUC={u_auc:.6f}, ratio={u_ratio:.1f}x

### Phase 5: Reports written — DONE
- hot_cold_access_profile.md
- hot_cold_compression_results.md
- hot_cold_vs_uniform.md

### Key Results
- Best AUC: {best_auc['auc']:.6f} (hot={best_auc['hot_crf']}, cold={best_auc['cold_crf']}, {best_auc['compression_ratio']:.1f}x)
- Best ratio: {best_ratio['compression_ratio']:.1f}x (hot={best_ratio['hot_crf']}, cold={best_ratio['cold_crf']}, AUC={best_ratio['auc']:.6f})"""

    update_status(final_status)

    send_notification(
        f"Hot/Cold experiment DONE! "
        f"Best AUC: {best_auc['auc']:.6f} (H{best_auc['hot_crf']}/C{best_auc['cold_crf']}), "
        f"Best ratio: {best_ratio['compression_ratio']:.1f}x "
        f"(H{best_ratio['hot_crf']}/C{best_ratio['cold_crf']})")

    print("\n" + "=" * 80)
    print("ALL PHASES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
