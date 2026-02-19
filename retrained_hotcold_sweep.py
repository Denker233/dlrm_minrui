#!/usr/bin/env python3
"""
Phase 2B: Hot/Cold Compression with Retrained Model
25 combinations: Hot CRF {0,5,10,15,20} x Cold CRF {25,30,38,45,51}
"""

import os
import sys
import time
import json
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
MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
DATA_FILE = "./input/train.txt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
RESULTS_DIR = os.path.expanduser("~/experiment-control")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 16384

HOT_CRFS = [0, 5, 10, 15, 20]
COLD_CRFS = [25, 30, 38, 45, 51]
ACCESS_THRESHOLD = 0.80


def drop_caches():
    """Drop OS page/buffer caches between experiments for consistent timing."""
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


def compress_table_with_codec(weights, codec_cmd_args):
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
        'frame_width': width, 'frame_height': height,
        'tiling': tiling_meta, 'original_pixels': num_emb * emb_dim,
    }
    return compressed_data, metadata, compress_time


def decompress_table(compressed_data, metadata):
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


def profile_access_patterns(test_ld, num_tables):
    print("  Profiling access patterns...")
    access_counts = [Counter() for _ in range(num_tables)]
    for i, batch in enumerate(test_ld):
        lS_i = batch[2]
        for t in range(num_tables):
            indices = lS_i[t].numpy().flatten() if isinstance(lS_i, list) else lS_i[t].numpy().flatten()
            for idx in indices:
                access_counts[t][int(idx)] += 1
        if i % 20 == 0:
            print(f"    Batch {i}/{len(test_ld)}", end='\r')
    print(f"    Profiled {len(test_ld)} batches")
    return access_counts


def identify_large_tables(ln_emb, emb_dim=16):
    sizes = np.array([int(n) * emb_dim * 4 for n in ln_emb])
    mean_size = float(np.mean(sizes))
    std_size = float(np.std(sizes))
    median_size = float(np.median(sizes))
    threshold1 = mean_size + std_size
    threshold2 = 10 * median_size
    large_mask = [(s > threshold1) or (s >= threshold2) for s in sizes]
    print(f"  Large tables: {sum(large_mask)}/{len(ln_emb)}")
    return large_mask, sizes.tolist()


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
    return sorted(hot_indices), sorted(cold_indices), total, cumulative / total


def split_table(weights, hot_indices, cold_indices):
    hot_w = weights[hot_indices].clone() if len(hot_indices) > 0 else torch.empty(0, weights.shape[1])
    cold_w = weights[cold_indices].clone() if len(cold_indices) > 0 else torch.empty(0, weights.shape[1])
    return hot_w, cold_w


def reconstruct_full_table(hot_w, cold_w, hot_idx, cold_idx, num_emb, emb_dim):
    full = torch.zeros(num_emb, emb_dim)
    if len(hot_idx) > 0:
        full[hot_idx] = hot_w
    if len(cold_idx) > 0:
        full[cold_idx] = cold_w
    return full


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 80)
    print("PHASE 2B: HOT/COLD COMPRESSION (RETRAINED MODEL)")
    print("=" * 80)

    dlrm, test_ld, ln_emb, ln_bot, ln_top, m_spa, args = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]
    num_tables = len(emb_keys)
    total_emb_size = sum(state_dict[k].numel() * 4 for k in emb_keys)
    table_sizes = [state_dict[k].numel() * 4 for k in emb_keys]

    print(f"  Model: {MODEL_PATH}")
    print(f"  Tables: {num_tables}, Total: {total_emb_size/1024/1024:.2f} MB")

    # Baseline
    print("\n  Running baseline inference...")
    baseline_acc, baseline_auc, baseline_infer = run_inference(dlrm, test_ld)
    print(f"  BASELINE: acc={baseline_acc*100:.4f}%, AUC={baseline_auc:.6f}, "
          f"infer={baseline_infer:.2f}s")

    # Phase 1: Profile access patterns
    print("\n" + "=" * 60)
    print("STEP 1: ACCESS PROFILING")
    print("=" * 60)
    large_mask, sizes_list = identify_large_tables(ln_emb, m_spa)
    access_counts = profile_access_patterns(test_ld, num_tables)

    profiles = []
    for t in range(num_tables):
        num_emb = int(ln_emb[t])
        is_large = large_mask[t]
        if is_large and num_emb > 0:
            hot_idx, cold_idx, tot_acc, hot_pct = compute_hot_cold_split(
                access_counts[t], num_emb, ACCESS_THRESHOLD)
        else:
            hot_idx = list(range(num_emb))
            cold_idx = []
            hot_pct = 1.0
        profiles.append({
            'table_idx': t, 'num_emb': num_emb, 'is_large': is_large,
            'hot_indices': hot_idx, 'cold_indices': cold_idx,
            'num_hot': len(hot_idx), 'num_cold': len(cold_idx),
        })
        tag = "LARGE" if is_large else "small"
        print(f"  Table {t}: {num_emb:>10,} rows, {tag}, "
              f"hot={len(hot_idx):>8,}, cold={len(cold_idx):>8,}")

    # Phase 2: Pre-compress all subtables
    print("\n" + "=" * 60)
    print("STEP 2: PRE-COMPRESSING")
    print("=" * 60)

    hot_cache = {}
    cold_cache = {}
    small_cache = {}

    for t, prof in enumerate(profiles):
        weights = state_dict[emb_keys[t]]
        if prof['is_large']:
            hot_w, cold_w = split_table(weights, prof['hot_indices'], prof['cold_indices'])
            hot_cache[t] = {}
            cold_cache[t] = {}
            for crf in HOT_CRFS:
                if hot_w.shape[0] > 0:
                    try:
                        hot_cache[t][crf] = compress_table_with_codec(hot_w, get_codec_args(crf))
                    except Exception as e:
                        print(f"    WARN: Table {t} hot CRF {crf}: {e}")
                        hot_cache[t][crf] = None
                else:
                    hot_cache[t][crf] = None
            for crf in COLD_CRFS:
                if cold_w.shape[0] > 0:
                    try:
                        cold_cache[t][crf] = compress_table_with_codec(cold_w, get_codec_args(crf))
                    except Exception as e:
                        print(f"    WARN: Table {t} cold CRF {crf}: {e}")
                        cold_cache[t][crf] = None
                else:
                    cold_cache[t][crf] = None
            print(f"  Table {t} (LARGE): hot={hot_w.shape}, cold={cold_w.shape}")
        else:
            small_cache[t] = {}
            for crf in HOT_CRFS:
                try:
                    small_cache[t][crf] = compress_table_with_codec(weights, get_codec_args(crf))
                except Exception as e:
                    print(f"    WARN: Table {t} small CRF {crf}: {e}")
                    small_cache[t][crf] = None
            print(f"  Table {t} (small): {weights.shape}")

    # Phase 3: Test all 25 combinations
    print("\n" + "=" * 60)
    print("STEP 3: TESTING 25 COMBINATIONS")
    print("=" * 60)

    combo_results = []
    combo_num = 0
    total_combos = len(HOT_CRFS) * len(COLD_CRFS)

    for hot_crf in HOT_CRFS:
        for cold_crf in COLD_CRFS:
            combo_num += 1
            drop_caches()
            print(f"\n  [{combo_num}/{total_combos}] Hot={hot_crf}, Cold={cold_crf}")

            tot_comp = 0
            tot_hot_size = 0
            tot_cold_size = 0
            tot_small_size = 0
            tot_ct = 0.0
            tot_dt = 0.0
            tot_hot_rows = 0
            tot_cold_rows = 0

            for t, prof in enumerate(profiles):
                num_emb = prof['num_emb']
                emb_dim = m_spa

                if prof['is_large']:
                    if hot_cache[t][hot_crf] is not None:
                        cdata, meta, ct = hot_cache[t][hot_crf]
                        hw, dt = decompress_table(cdata, meta)
                        tot_hot_size += len(cdata)
                        tot_comp += len(cdata)
                        tot_ct += ct
                        tot_dt += dt
                    else:
                        hw = torch.empty(0, emb_dim)

                    if cold_cache[t][cold_crf] is not None:
                        cdata, meta, ct = cold_cache[t][cold_crf]
                        cw, dt = decompress_table(cdata, meta)
                        tot_cold_size += len(cdata)
                        tot_comp += len(cdata)
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
                        tot_comp += len(cdata)
                        tot_ct += ct
                        tot_dt += dt
                        with torch.no_grad():
                            dlrm.emb_l[t].weight.data = dec_w
                    else:
                        tot_comp += table_sizes[t]
                        tot_small_size += table_sizes[t]

            acc, auc, infer_t = run_inference(dlrm, test_ld)

            # Restore original weights
            for t in range(num_tables):
                with torch.no_grad():
                    dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

            comp_ratio = total_emb_size / tot_comp if tot_comp > 0 else 0

            result = {
                'hot_crf': hot_crf, 'cold_crf': cold_crf,
                'accuracy': acc, 'accuracy_pct': acc * 100, 'auc': auc,
                'accuracy_loss_pct': (baseline_acc - acc) * 100,
                'auc_loss': baseline_auc - auc,
                'total_compressed_size': tot_comp,
                'total_compressed_mb': tot_comp / 1024 / 1024,
                'compression_ratio': comp_ratio,
                'hot_mb': tot_hot_size / 1024 / 1024,
                'cold_mb': tot_cold_size / 1024 / 1024,
                'small_mb': tot_small_size / 1024 / 1024,
                'compress_time': tot_ct, 'decompress_time': tot_dt,
                'inference_time': infer_t,
                'total_hot_rows': tot_hot_rows, 'total_cold_rows': tot_cold_rows,
                'hot_row_pct': (tot_hot_rows / (tot_hot_rows + tot_cold_rows) * 100
                                if (tot_hot_rows + tot_cold_rows) > 0 else 0),
            }
            combo_results.append(result)

            print(f"    acc={acc*100:.4f}%, AUC={auc:.6f}, "
                  f"{tot_comp/1024/1024:.2f}MB ({comp_ratio:.1f}x), infer={infer_t:.2f}s")

    # Write results
    out_path = os.path.join(RESULTS_DIR, "retrained_hotcold_results.md")
    with open(out_path, 'w') as f:
        f.write("# Retrained Model: Hot/Cold Differential Compression Results\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- **Model:** `{MODEL_PATH}`\n")
        f.write(f"- **Hot CRFs:** {HOT_CRFS}\n")
        f.write(f"- **Cold CRFs:** {COLD_CRFS}\n")
        f.write(f"- **Hot threshold:** {ACCESS_THRESHOLD*100:.0f}% of accesses\n")
        f.write(f"- **Small tables:** compressed at hot CRF\n\n")

        f.write("## Baseline\n\n")
        f.write(f"- **Accuracy:** {baseline_acc*100:.4f}%\n")
        f.write(f"- **AUC:** {baseline_auc:.6f}\n")
        f.write(f"- **Uncompressed:** {total_emb_size:,} bytes ({total_emb_size/1024/1024:.2f} MB)\n")
        f.write(f"- **Inference time:** {baseline_infer:.2f}s\n\n")

        f.write("## Results (25 Combinations)\n\n")
        f.write("| Hot CRF | Cold CRF | Accuracy (%) | AUC | Acc Loss (%) | AUC Loss "
                "| Total (MB) | Ratio | Hot (MB) | Cold (MB) | Small (MB) "
                "| Compress (s) | Decompress (s) | Inference (s) |\n")
        f.write("|---------|----------|-------------|------|-------------|----------"
                "|-----------|-------|---------|----------|----------"
                "|-------------|---------------|---------------|\n")

        for r in combo_results:
            f.write(f"| {r['hot_crf']} | {r['cold_crf']} "
                    f"| {r['accuracy_pct']:.4f} | {r['auc']:.6f} "
                    f"| {r['accuracy_loss_pct']:.4f} | {r['auc_loss']:.6f} "
                    f"| {r['total_compressed_mb']:.2f} | {r['compression_ratio']:.2f}x "
                    f"| {r['hot_mb']:.4f} | {r['cold_mb']:.4f} | {r['small_mb']:.4f} "
                    f"| {r['compress_time']:.2f} | {r['decompress_time']:.2f} "
                    f"| {r['inference_time']:.2f} |\n")

        best_auc = max(combo_results, key=lambda r: r['auc'])
        best_ratio = max(combo_results, key=lambda r: r['compression_ratio'])

        f.write(f"\n## Best Results\n\n")
        f.write(f"### Best AUC\n")
        f.write(f"- Hot CRF={best_auc['hot_crf']}, Cold CRF={best_auc['cold_crf']}\n")
        f.write(f"- AUC: {best_auc['auc']:.6f} (loss: {best_auc['auc_loss']:.6f})\n")
        f.write(f"- Compression: {best_auc['compression_ratio']:.2f}x ({best_auc['total_compressed_mb']:.2f} MB)\n\n")

        f.write(f"### Best Compression Ratio\n")
        f.write(f"- Hot CRF={best_ratio['hot_crf']}, Cold CRF={best_ratio['cold_crf']}\n")
        f.write(f"- AUC: {best_ratio['auc']:.6f} (loss: {best_ratio['auc_loss']:.6f})\n")
        f.write(f"- Compression: {best_ratio['compression_ratio']:.2f}x ({best_ratio['total_compressed_mb']:.2f} MB)\n")

    print(f"\nResults written to: {out_path}")

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "retrained_hotcold_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            'baseline': {
                'accuracy': baseline_acc, 'auc': baseline_auc,
                'inference_time': baseline_infer,
                'total_emb_size': total_emb_size,
            },
            'results': combo_results,
        }, f, indent=2)
    print(f"JSON saved to: {json_path}")

    print("\n" + "=" * 80)
    print("PHASE 2B COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
