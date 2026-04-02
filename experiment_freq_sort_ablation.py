#!/usr/bin/env python3
"""
Ablation: Impact of frequency sorting on H.265 compression ratio and AUC.

Compares two configs, both using single-frame-per-table H.265 (CRF=30, medium, no-deblock, no-sao):
1. With frequency sorting (current approach): cold rows ordered by cold_order_{t}.npy
2. Without frequency sorting: cold rows in natural index order (all non-hot rows in ascending index)

Measures:
- Per-table compressed size, PSNR, max_error, mean_error
- Total compression ratios
- AUC for both configs + fp32 baseline
"""

import os, sys, time, json, gc, subprocess, tempfile, math
import numpy as np

os.chdir('/home/cc/expr/dlrm_minrui')
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')

import torch
import torch.nn as nn
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import compressed_emb as _C
from sklearn.metrics import roc_auc_score

# Import from existing code
from codec_ondemand_benchmark import (
    load_model_and_data, CompressedEmbeddingBag,
    HOTCOLD_DIR, REORDER_DIR, ONDEMAND_DIR,
    EMB_DIM, MODEL_PATH, TILE_H, TILE_W, TEST_BATCH_SIZE,
)

# ============================================================
# Configuration
# ============================================================
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
H265_CRF = 30
H265_PRESET = 'medium'
H265_EXTRA = 'no-deblock=1:no-sao=1'

OUTPUT_DIR = 'results/freq_sort_ablation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Single-frame dimensions from existing results
SINGLE_FRAME_DIMS = {
    2:  (3840, 40400),
    3:  (1920, 17568),
    9:  (1920, 744),
    11: (3840, 33304),
    15: (1920, 43556),
    20: (1920, 56200),
    23: (1920, 2284),
    25: (1920, 1140),
}

# ============================================================
# Helpers
# ============================================================
def quantize_table(w):
    """Global quantization: single scale/zero-point for entire table."""
    mn = w.min().item()
    mx = w.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def rows_to_tiled_frame(emb_rows, width, height):
    """Convert embedding rows (N, 16) into a tiled 2D frame (height, width)."""
    if isinstance(emb_rows, np.ndarray):
        emb_rows = torch.from_numpy(emb_rows)
    rows_per_frame = (width // TILE_W) * (height // TILE_H)
    if emb_rows.shape[0] < rows_per_frame:
        padded = torch.zeros(rows_per_frame, EMB_DIM, dtype=torch.uint8)
        padded[:emb_rows.shape[0]] = emb_rows
        emb_rows = padded
    frame = _C.tile_rows_to_frame(emb_rows, width, height)
    return frame.numpy()


def untile_frame(frame_np, width, height):
    """Convert tiled 2D frame back to embedding rows."""
    frame_t = torch.from_numpy(frame_np) if isinstance(frame_np, np.ndarray) else frame_np
    rows_per_frame = (width // TILE_W) * (height // TILE_H)
    rows = _C.untile_frame_to_rows(frame_t, rows_per_frame)
    return rows.numpy()


def encode_h265_single_frame(q_np, width, height, output_path):
    """Encode uint8 rows as a single H.265 frame."""
    frame_2d = rows_to_tiled_frame(q_np, width, height)

    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(frame_2d.tobytes())

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'gray',
        '-s', f'{width}x{height}',
        '-r', '1',
        '-i', tmp_path,
        '-c:v', 'libx265',
        '-preset', H265_PRESET,
        '-pix_fmt', 'gray',
        '-x265-params',
        f'keyint=1:min-keyint=1:crf={H265_CRF}:log-level=error'
        + (f':{H265_EXTRA}' if H265_EXTRA else ''),
        '-f', 'matroska',
        output_path,
    ]

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, stderr = proc.communicate()
    encode_time = time.time() - t0

    os.unlink(tmp_path)

    if proc.returncode != 0:
        print(f"  WARNING: ffmpeg returned {proc.returncode}")
        print(f"  stderr: {stderr.decode()[:500]}")
        return 0, encode_time

    compressed_size = os.path.getsize(output_path)
    return compressed_size, encode_time


def decode_h265_single_frame(h265_path, width, height):
    """Decode an H.265 file back to a 2D frame."""
    cmd = [
        'ffmpeg', '-y',
        '-i', h265_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'gray',
        'pipe:1',
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_data, _ = proc.communicate()
    decode_time = time.time() - t0

    expected = width * height
    if len(raw_data) != expected:
        print(f"  WARNING: decoded {len(raw_data)} bytes, expected {expected}")
        if len(raw_data) < expected:
            raw_data = raw_data + b'\x00' * (expected - len(raw_data))
        else:
            raw_data = raw_data[:expected]

    frame = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width)
    return frame, decode_time


def compute_metrics(original_uint8, decoded_uint8, n_rows):
    """Compute PSNR, max error, mean error between original and decoded uint8 data."""
    orig = original_uint8[:n_rows].flatten().astype(np.float64)
    dec = decoded_uint8[:n_rows].flatten().astype(np.float64)

    diff = orig - dec
    mse = np.mean(diff ** 2)
    max_err = np.max(np.abs(diff))
    mean_err = np.mean(np.abs(diff))

    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * math.log10(255.0**2 / mse)

    return psnr, max_err, mean_err


# ============================================================
# Phase 1: Encode/decode sorted and unsorted, measure compression + quality
# ============================================================
def run_compression_experiment():
    print("=" * 80)
    print("PHASE 1: Compression & Quality Comparison (Sorted vs Unsorted)")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = ld['state_dict'] if 'state_dict' in ld else ld
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda k: int(k.split('.')[1]))

    # Load is_hot masks
    is_hot = {}
    for t in LARGE_TABLES:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)

    results = {'sorted': {}, 'unsorted': {}}

    for t in LARGE_TABLES:
        print(f"\n--- Table {t} ---")
        w = state_dict[emb_keys[t]]
        n_total = w.shape[0]
        is_hot_t = is_hot[t][:n_total]

        # Cold indices in natural (unsorted) order
        cold_indices_natural = torch.where(~is_hot_t)[0]
        n_cold = len(cold_indices_natural)
        raw_bytes = n_cold * EMB_DIM

        # Load frequency-sorted cold order
        cold_order = np.load(os.path.join(REORDER_DIR, f'cold_order_{t}.npy'))

        # Get frame dimensions
        sf_width, sf_height = SINGLE_FRAME_DIMS[t]
        rpf = (sf_width // TILE_W) * (sf_height // TILE_H)

        print(f"  n_cold={n_cold:,}, frame={sf_width}x{sf_height}, rpf={rpf:,}")

        # ---- SORTED (frequency-sorted) ----
        print(f"  [Sorted] Quantizing and encoding...")
        w_sorted = w[cold_order]
        q_sorted, s_sorted, zp_sorted = quantize_table(w_sorted)
        q_sorted_np = q_sorted.numpy()

        sorted_dir = os.path.join(OUTPUT_DIR, 'sorted', f'table_{t}')
        os.makedirs(sorted_dir, exist_ok=True)
        sorted_path = os.path.join(sorted_dir, 'frame_00000.h265')

        sorted_size, sorted_enc_time = encode_h265_single_frame(
            q_sorted_np, sf_width, sf_height, sorted_path)

        # Decode and measure quality
        sorted_frame, sorted_dec_time = decode_h265_single_frame(
            sorted_path, sf_width, sf_height)
        sorted_decoded_rows = untile_frame(sorted_frame, sf_width, sf_height)
        sorted_psnr, sorted_maxerr, sorted_meanerr = compute_metrics(
            q_sorted_np, sorted_decoded_rows, n_cold)
        sorted_ratio = raw_bytes / sorted_size if sorted_size > 0 else 0

        results['sorted'][str(t)] = {
            'n_cold': n_cold,
            'raw_bytes': raw_bytes,
            'compressed_bytes': sorted_size,
            'ratio': sorted_ratio,
            'psnr': sorted_psnr if not math.isinf(sorted_psnr) else 'inf',
            'max_err': float(sorted_maxerr),
            'mean_err': float(sorted_meanerr),
            'encode_time': sorted_enc_time,
            'decode_time': sorted_dec_time,
            'quant_scale': s_sorted,
            'quant_zp': zp_sorted,
        }
        print(f"  [Sorted]   Size={sorted_size:>10,} bytes, Ratio={sorted_ratio:>8.1f}x, "
              f"PSNR={sorted_psnr:.1f}, MaxErr={sorted_maxerr:.0f}, MeanErr={sorted_meanerr:.4f}")

        # ---- UNSORTED (natural index order) ----
        print(f"  [Unsorted] Quantizing and encoding...")
        w_unsorted = w[cold_indices_natural]
        q_unsorted, s_unsorted, zp_unsorted = quantize_table(w_unsorted)
        q_unsorted_np = q_unsorted.numpy()

        unsorted_dir = os.path.join(OUTPUT_DIR, 'unsorted', f'table_{t}')
        os.makedirs(unsorted_dir, exist_ok=True)
        unsorted_path = os.path.join(unsorted_dir, 'frame_00000.h265')

        unsorted_size, unsorted_enc_time = encode_h265_single_frame(
            q_unsorted_np, sf_width, sf_height, unsorted_path)

        # Decode and measure quality
        unsorted_frame, unsorted_dec_time = decode_h265_single_frame(
            unsorted_path, sf_width, sf_height)
        unsorted_decoded_rows = untile_frame(unsorted_frame, sf_width, sf_height)
        unsorted_psnr, unsorted_maxerr, unsorted_meanerr = compute_metrics(
            q_unsorted_np, unsorted_decoded_rows, n_cold)
        unsorted_ratio = raw_bytes / unsorted_size if unsorted_size > 0 else 0

        results['unsorted'][str(t)] = {
            'n_cold': n_cold,
            'raw_bytes': raw_bytes,
            'compressed_bytes': unsorted_size,
            'ratio': unsorted_ratio,
            'psnr': unsorted_psnr if not math.isinf(unsorted_psnr) else 'inf',
            'max_err': float(unsorted_maxerr),
            'mean_err': float(unsorted_meanerr),
            'encode_time': unsorted_enc_time,
            'decode_time': unsorted_dec_time,
            'quant_scale': s_unsorted,
            'quant_zp': zp_unsorted,
        }
        print(f"  [Unsorted] Size={unsorted_size:>10,} bytes, Ratio={unsorted_ratio:>8.1f}x, "
              f"PSNR={unsorted_psnr:.1f}, MaxErr={unsorted_maxerr:.0f}, MeanErr={unsorted_meanerr:.4f}")

        # Comparison
        if unsorted_size > 0 and sorted_size > 0:
            size_change = (sorted_size - unsorted_size) / unsorted_size * 100
            print(f"  >> Sorting impact: size {size_change:+.1f}% "
                  f"({'sorting helps' if size_change < 0 else 'sorting hurts'})")

        del w_sorted, w_unsorted, q_sorted, q_unsorted
        gc.collect()

    # Save compression results
    comp_path = os.path.join(OUTPUT_DIR, 'compression_results.json')
    with open(comp_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nCompression results saved to {comp_path}")

    return results


# ============================================================
# Phase 2: AUC comparison
# ============================================================
def run_auc_experiment(comp_results):
    print("\n" + "=" * 80)
    print("PHASE 2: AUC Comparison (Sorted vs Unsorted vs Baseline)")
    print("=" * 80)

    # Load model and data
    print("\nLoading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))

    # Pre-cache test batches
    print("Pre-caching test batches...")
    test_batches = []
    for X, lS_o, lS_i, T in test_ld:
        test_batches.append((X, lS_o, lS_i, T))
    print(f"Pre-cached {len(test_batches)} batches")

    # Load hot/cold data
    print("Loading hot/cold split data...")
    is_hot = {}
    hot_indices = {}
    for t in LARGE_TABLES:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]

    cold_num_rows = {}
    for t in LARGE_TABLES:
        with open(os.path.join(REORDER_DIR, f'num_cold_{t}.txt')) as f:
            cold_num_rows[t] = int(f.read().strip())

    # Load orig_to_cold_reordered for sorted config
    orig_to_cold_reordered = {}
    for t in LARGE_TABLES:
        fp = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        orig_to_cold_reordered[t] = torch.load(fp, map_location='cpu', weights_only=True)

    num_tabs = len(dlrm.emb_l)

    # Save original apply_emb
    orig_apply_emb = dlrm.apply_emb

    # ---- Helper: run inference and compute AUC ----
    def run_inference(tag):
        print(f"\n  Running inference for {tag}...")
        num_test_batches = len(test_batches)
        max_samples = num_test_batches * TEST_BATCH_SIZE + TEST_BATCH_SIZE
        all_scores = np.empty(max_samples, dtype=np.float32)
        all_targets = np.empty(max_samples, dtype=np.float32)
        sample_idx = 0

        t0 = time.time()
        with torch.no_grad():
            for batch_idx in range(num_test_batches):
                X, lS_o, lS_i, T = test_batches[batch_idx]
                Z = dlrm(X, lS_o, lS_i)
                z_np = Z.detach().cpu().numpy().ravel()
                t_np = T.detach().cpu().numpy().ravel()
                bs = z_np.shape[0]
                all_scores[sample_idx:sample_idx+bs] = z_np
                all_targets[sample_idx:sample_idx+bs] = t_np
                sample_idx += bs
                if batch_idx % 500 == 0:
                    print(f"    Batch {batch_idx}/{num_test_batches}")

        total_time = time.time() - t0
        auc = roc_auc_score(all_targets[:sample_idx], all_scores[:sample_idx])
        print(f"  {tag}: AUC = {auc:.6f}, Time = {total_time:.2f}s")
        return auc, total_time

    # ---- Baseline (fp32) ----
    print(f"\n{'='*60}")
    print("Baseline: Full fp32 (no compression)")
    print(f"{'='*60}")
    with torch.no_grad():
        for k in emb_keys:
            t_idx = int(k.split('.')[1])
            dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()
    dlrm.apply_emb = orig_apply_emb
    baseline_auc, baseline_time = run_inference("Baseline fp32")

    auc_results = {
        'baseline': {'auc': float(baseline_auc), 'time': float(baseline_time)}
    }

    # ---- Helper: setup compressed inference ----
    def setup_compressed_and_run(tag, decode_fn, o2c_map_fn, quant_params_fn):
        """
        Setup compressed embedding tables and run inference.

        decode_fn(t_idx) -> uint8_rows tensor (n_cold_or_more, D)
        o2c_map_fn(t_idx) -> orig_to_cold mapping tensor (n_total,) int32
        quant_params_fn(t_idx) -> (scale, zp)
        """
        print(f"\n{'='*60}")
        print(f"Config: {tag}")
        print(f"{'='*60}")

        # Restore original weights
        with torch.no_grad():
            for k in emb_keys:
                t_idx = int(k.split('.')[1])
                dlrm.emb_l[t_idx] = nn.EmbeddingBag(
                    ln_emb[t_idx], EMB_DIM, mode='sum', sparse=False)
                dlrm.emb_l[t_idx].weight.data = state_dict[k].clone()

        # Build CompressedEmbeddingBag for each large table
        for t_idx in LARGE_TABLES:
            n_cold = cold_num_rows.get(t_idx, 0)
            if n_cold == 0:
                continue

            w = state_dict[emb_keys[t_idx]]
            h_idx = hot_indices[t_idx]
            hot_weight = w[h_idx].clone()

            orig_to_hot = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
            orig_to_hot[h_idx] = torch.arange(len(h_idx))

            o2c = o2c_map_fn(t_idx)

            comp_emb = CompressedEmbeddingBag(
                hot_weight=hot_weight,
                is_hot=is_hot[t_idx],
                orig_to_hot=orig_to_hot,
                orig_to_cold_reordered=o2c,
                cold_cache=None,
                num_embeddings=ln_emb[t_idx],
                embedding_dim=EMB_DIM,
                quantize_hot=False,
            )
            dlrm.emb_l[t_idx] = comp_emb

        # Register tables in C++
        table_kinds = []
        weights = []
        mappings = []
        scales = []
        zero_points = []

        for k_idx in range(num_tabs):
            E = dlrm.emb_l[k_idx]
            if k_idx in cold_num_rows and isinstance(E, CompressedEmbeddingBag):
                table_kinds.append(1)  # COMPRESSED_FP32
                weights.append(E.hot_weight)
                mappings.append(E.mapping)
                scales.append(0.0)
                zero_points.append(0)
            else:
                table_kinds.append(0)  # STANDARD
                weights.append(E.weight)
                mappings.append(torch.empty(0, dtype=torch.int32))
                scales.append(0.0)
                zero_points.append(0)

        _C.register_tables(table_kinds, weights, mappings, scales, zero_points,
                           use_hash_table=False, use_bitmap=True)
        print("  Tables registered in C++ (bitmap mode)")

        # Decode and register cold frames
        print("  Decoding cold frames...")
        total_cold_mb = 0

        for t_idx in LARGE_TABLES:
            n_cold = cold_num_rows.get(t_idx, 0)
            if n_cold == 0:
                continue

            # Decode
            decoded_uint8 = decode_fn(t_idx)  # (n_rows, D) uint8 tensor

            # For the cold registration, we need to map cold-rank indices to
            # positions in the decoded data. Since we decode ALL cold rows and
            # they are stored sequentially, valid_cold_ranks = arange(n_cold)
            # and src_positions = arange(n_cold).
            valid_data = decoded_uint8[:n_cold]
            if isinstance(valid_data, np.ndarray):
                valid_data = torch.from_numpy(valid_data)
            valid_cold_ranks = torch.arange(n_cold, dtype=torch.long)

            scale, zp = quant_params_fn(t_idx)
            _C.register_cold_sparse_flat(
                t_idx, valid_data, valid_cold_ranks,
                float(scale), float(zp), n_cold)

            cold_mb = valid_data.nbytes / 1024 / 1024
            total_cold_mb += cold_mb
            print(f"    Table {t_idx}: registered {valid_data.shape[0]:,} cold rows ({cold_mb:.1f}MB)")

            del decoded_uint8, valid_data
            gc.collect()

        print(f"  Total cold memory: {total_cold_mb:.1f}MB")

        # Setup fast_forward
        def _full_cpp_apply_emb(lS_o, lS_i, emb_l, v_W_l):
            if isinstance(lS_i, (list, tuple)):
                lS_i_2d = torch.stack(lS_i)
            elif lS_i.dim() == 2:
                lS_i_2d = lS_i
            else:
                lS_i_2d = lS_i.view(num_tabs, -1)
            if isinstance(lS_o, (list, tuple)):
                lS_o_2d = torch.stack(lS_o)
            elif lS_o.dim() == 2:
                lS_o_2d = lS_o
            else:
                lS_o_2d = lS_o.view(num_tabs, -1)
            results = _C.fast_forward(lS_i_2d, lS_o_2d)
            return results[-1]

        dlrm.apply_emb = _full_cpp_apply_emb

        auc, total_time = run_inference(tag)
        return {'auc': float(auc), 'time': float(total_time), 'cold_mb': float(total_cold_mb)}

    # ---- Sorted config ----
    # For sorted: decode from the sorted frames, use existing orig_to_cold_reordered mapping
    def decode_sorted(t_idx):
        path = os.path.join(OUTPUT_DIR, 'sorted', f'table_{t_idx}', 'frame_00000.h265')
        w, h = SINGLE_FRAME_DIMS[t_idx]
        frame, _ = decode_h265_single_frame(path, w, h)
        rows = untile_frame(frame, w, h)
        return rows  # numpy array (rpf, D)

    def o2c_sorted(t_idx):
        return orig_to_cold_reordered[t_idx]

    def quant_sorted(t_idx):
        r = comp_results['sorted'][str(t_idx)]
        return r['quant_scale'], r['quant_zp']

    auc_results['sorted'] = setup_compressed_and_run(
        'Sorted (freq-sorted)', decode_sorted, o2c_sorted, quant_sorted)

    # Reset
    dlrm.apply_emb = orig_apply_emb
    gc.collect()

    # ---- Unsorted config ----
    # For unsorted: decode from the unsorted frames, build new orig_to_cold mapping
    # In the unsorted case: cold_indices_natural = torch.where(~is_hot_t)[0]
    # The cold rows are stored in this natural order.
    # orig_to_cold_unsorted[cold_indices_natural[i]] = i
    unsorted_o2c = {}
    for t_idx in LARGE_TABLES:
        is_hot_t = is_hot[t_idx][:ln_emb[t_idx]]
        cold_indices_natural = torch.where(~is_hot_t)[0]
        n_cold = len(cold_indices_natural)
        o2c = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
        o2c[cold_indices_natural] = torch.arange(n_cold, dtype=torch.long)
        unsorted_o2c[t_idx] = o2c

    def decode_unsorted(t_idx):
        path = os.path.join(OUTPUT_DIR, 'unsorted', f'table_{t_idx}', 'frame_00000.h265')
        w, h = SINGLE_FRAME_DIMS[t_idx]
        frame, _ = decode_h265_single_frame(path, w, h)
        rows = untile_frame(frame, w, h)
        return rows  # numpy array (rpf, D)

    def o2c_unsorted(t_idx):
        return unsorted_o2c[t_idx]

    def quant_unsorted(t_idx):
        r = comp_results['unsorted'][str(t_idx)]
        return r['quant_scale'], r['quant_zp']

    auc_results['unsorted'] = setup_compressed_and_run(
        'Unsorted (natural order)', decode_unsorted, o2c_unsorted, quant_unsorted)

    return auc_results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 80)
    print("EXPERIMENT: Frequency Sorting Ablation for H.265 Compression")
    print(f"Config: CRF={H265_CRF}, preset={H265_PRESET}, {H265_EXTRA}")
    print(f"Single frame per table, 4x4 tiling, EMB_DIM={EMB_DIM}")
    print("=" * 80)

    # Phase 1: Compression & quality
    comp_results = run_compression_experiment()

    # Phase 2: AUC
    auc_results = run_auc_experiment(comp_results)

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    # Compression summary
    print(f"\n{'Table':>6} | {'n_cold':>12} | {'Sorted Size':>12} | {'Sorted Ratio':>12} | "
          f"{'Unsorted Size':>14} | {'Unsorted Ratio':>14} | {'Size Change':>12} | "
          f"{'Sorted PSNR':>11} | {'Unsorted PSNR':>13}")
    print("-" * 130)

    total_sorted_size = 0
    total_unsorted_size = 0
    total_raw = 0

    for t in LARGE_TABLES:
        ts = str(t)
        s = comp_results['sorted'][ts]
        u = comp_results['unsorted'][ts]
        total_sorted_size += s['compressed_bytes']
        total_unsorted_size += u['compressed_bytes']
        total_raw += s['raw_bytes']

        size_change = (s['compressed_bytes'] - u['compressed_bytes']) / u['compressed_bytes'] * 100 \
            if u['compressed_bytes'] > 0 else 0

        s_psnr = f"{s['psnr']:.1f}" if s['psnr'] != 'inf' else 'inf'
        u_psnr = f"{u['psnr']:.1f}" if u['psnr'] != 'inf' else 'inf'

        print(f"{t:>6} | {s['n_cold']:>12,} | {s['compressed_bytes']:>12,} | "
              f"{s['ratio']:>11.1f}x | {u['compressed_bytes']:>14,} | "
              f"{u['ratio']:>13.1f}x | {size_change:>+11.1f}% | "
              f"{s_psnr:>11} | {u_psnr:>13}")

    print("-" * 130)
    sorted_total_ratio = total_raw / total_sorted_size if total_sorted_size > 0 else 0
    unsorted_total_ratio = total_raw / total_unsorted_size if total_unsorted_size > 0 else 0
    total_size_change = (total_sorted_size - total_unsorted_size) / total_unsorted_size * 100 \
        if total_unsorted_size > 0 else 0
    print(f"{'TOTAL':>6} | {'':>12} | {total_sorted_size:>12,} | "
          f"{sorted_total_ratio:>11.1f}x | {total_unsorted_size:>14,} | "
          f"{unsorted_total_ratio:>13.1f}x | {total_size_change:>+11.1f}% |")

    total_raw_fp32 = total_raw * 4
    print(f"\nRaw uint8 total: {total_raw:,} bytes ({total_raw/1024/1024:.1f} MB)")
    print(f"Raw fp32 total:  {total_raw_fp32:,} bytes ({total_raw_fp32/1024/1024:.1f} MB)")
    print(f"Sorted fp32-ratio:   {total_raw_fp32/total_sorted_size:.1f}x")
    print(f"Unsorted fp32-ratio: {total_raw_fp32/total_unsorted_size:.1f}x")

    # AUC summary
    print(f"\n{'Config':<30} {'AUC':>10} {'AUC Loss':>12} {'Time(s)':>8}")
    print("-" * 65)
    bl_auc = auc_results['baseline']['auc']
    print(f"{'Baseline (fp32)':<30} {bl_auc:>10.6f} {'---':>12} "
          f"{auc_results['baseline']['time']:>8.2f}")
    for key in ['sorted', 'unsorted']:
        r = auc_results[key]
        loss = r['auc'] - bl_auc
        loss_pct = loss / bl_auc * 100
        print(f"{key.capitalize() + ' H.265 CRF=30':<30} {r['auc']:>10.6f} "
              f"{loss:>+12.6f} {r['time']:>8.2f}")

    print(f"\nAUC difference (sorted - unsorted): "
          f"{auc_results['sorted']['auc'] - auc_results['unsorted']['auc']:+.6f}")

    # Save all results
    all_results = {
        'compression': comp_results,
        'auc': auc_results,
        'config': {
            'h265_crf': H265_CRF,
            'h265_preset': H265_PRESET,
            'h265_extra': H265_EXTRA,
            'emb_dim': EMB_DIM,
            'tile_w': TILE_W,
            'tile_h': TILE_H,
            'tables': LARGE_TABLES,
            'frame_dims': {str(k): list(v) for k, v in SINGLE_FRAME_DIMS.items()},
        },
        'summary': {
            'sorted_total_compressed': total_sorted_size,
            'unsorted_total_compressed': total_unsorted_size,
            'total_raw_uint8': total_raw,
            'total_raw_fp32': total_raw_fp32,
            'sorted_ratio_uint8': sorted_total_ratio,
            'unsorted_ratio_uint8': unsorted_total_ratio,
            'sorted_ratio_fp32': total_raw_fp32 / total_sorted_size if total_sorted_size > 0 else 0,
            'unsorted_ratio_fp32': total_raw_fp32 / total_unsorted_size if total_unsorted_size > 0 else 0,
            'size_change_pct': total_size_change,
            'baseline_auc': float(bl_auc),
            'sorted_auc': float(auc_results['sorted']['auc']),
            'unsorted_auc': float(auc_results['unsorted']['auc']),
            'sorted_auc_loss': float(auc_results['sorted']['auc'] - bl_auc),
            'unsorted_auc_loss': float(auc_results['unsorted']['auc'] - bl_auc),
            'sorting_auc_diff': float(auc_results['sorted']['auc'] - auc_results['unsorted']['auc']),
        }
    }

    results_path = os.path.join(OUTPUT_DIR, 'full_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {results_path}")


if __name__ == '__main__':
    main()
