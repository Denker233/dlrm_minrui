#!/usr/bin/env python3
"""
Two experiments:
1. CRF sweep with 4x4 tiling + frequency sorting on Kaggle → AUC at each CRF
2. Deep entropy analysis on Terabyte → why some tables compress 1400x and others 12x
"""

import os, sys, time, json, gc, tempfile, subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import compressed_emb as _C
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

KAGGLE_MODEL = "./models/dlrm_kaggle_correct.pt"
TERABYTE_MODEL = "./models/dlrm_terabyte_4day.pt"
LARGE_TABLE_THRESHOLD = 50000
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

RESULTS_DIR = "results/crf_and_entropy"
os.makedirs(RESULTS_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)


# ============================================================
# SHARED UTILITIES
# ============================================================
def quantize_table_uint8(w):
    """Global min/max uint8 quantization."""
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = torch.clamp(torch.round(w / s + zp), 0, 255).to(torch.uint8)
    return q, s, zp

def dequantize_uint8(q, s, zp):
    return (q.float() - zp) * s

def encode_frame_h265(frame_np, crf):
    """Encode a (H,W) uint8 frame via ffmpeg, return (compressed_size, decoded_frame)."""
    with tempfile.NamedTemporaryFile(suffix='.h265', delete=False) as f:
        tmp = f.name
    try:
        h, w = frame_np.shape
        lossless = (crf == 0)
        params = 'lossless=1:log-level=error' if lossless else f'crf={crf}:log-level=error'
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
            '-s', f'{w}x{h}', '-r', '1', '-i', 'pipe:0',
            '-c:v', 'libx265', '-preset', 'ultrafast', '-pix_fmt', 'gray',
            '-x265-params', f'keyint=1:min-keyint=1:{params}',
            '-f', 'hevc', tmp
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.stdin.write(frame_np.tobytes())
        proc.stdin.close()
        proc.wait()
        comp_size = os.path.getsize(tmp)

        # Decode
        if HAS_CPP and hasattr(_C, 'decode_h265_frame_from_file'):
            decoded = _C.decode_h265_frame_from_file(tmp).numpy()
        else:
            import av
            container = av.open(tmp)
            for frame in container.decode(video=0):
                decoded = frame.to_ndarray(format='gray')
                break
            container.close()
        return comp_size, decoded
    finally:
        os.unlink(tmp)


def pack_rows_tiled_4x4(rows_uint8, D, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Pack rows into a (H, W) frame using 4x4 tiling via C++."""
    rpf = (width // 4) * (height // 4)
    N = rows_uint8.shape[0]
    if N < rpf:
        pad = torch.zeros(rpf - N, D, dtype=torch.uint8)
        rows_uint8 = torch.cat([rows_uint8, pad])
    return _C.tile_rows_to_frame(rows_uint8[:rpf], width, height)

def unpack_frame_tiled_4x4(frame, n_rows, width=FRAME_WIDTH):
    """Unpack a tiled frame back to rows."""
    rpf = (width // 4) * (FRAME_HEIGHT // 4)
    rows = _C.untile_frame_to_rows(frame if isinstance(frame, torch.Tensor) else torch.from_numpy(frame), rpf)
    return rows[:n_rows]

def pack_rows_flat(rows_uint8, D, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Pack rows into a (H, W) frame using flat layout."""
    N = rows_uint8.shape[0]
    rpr = width // D
    padded_N = ((N + rpr - 1) // rpr) * rpr
    if padded_N > N:
        data = torch.cat([rows_uint8, torch.zeros(padded_N - N, D, dtype=torch.uint8)])
    else:
        data = rows_uint8
    H_used = padded_N // rpr
    frame = np.zeros((height, width), dtype=np.uint8)
    frame[:H_used] = data.reshape(H_used, rpr * D).numpy()
    return frame

def unpack_frame_flat(frame_np, n_rows, D, width=FRAME_WIDTH):
    """Unpack flat frame to rows."""
    rpr = width // D
    H_used = (n_rows + rpr - 1) // rpr
    data = frame_np[:H_used].reshape(-1, D)
    return data[:n_rows]


# ============================================================
# PART 1: CRF SWEEP ON KAGGLE (4x4 tiled + freq sort)
# ============================================================
def load_kaggle():
    """Load Kaggle model and test data."""
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net

    class Args: pass
    a = Args()
    a.arch_sparse_feature_size = 16
    a.arch_mlp_bot = "13-512-256-64-16"; a.arch_mlp_top = "512-256-1"
    a.raw_data_file = os.path.expanduser("~/input/train.txt")
    a.processed_data_file = os.path.expanduser("~/input/kaggleAdDisplayChallenge_processed.npz")
    a.data_set = "kaggle"; a.max_ind_range = -1
    a.arch_interaction_op = "dot"; a.arch_interaction_itself = False
    a.data_generation = "dataset"; a.loss_function = "bce"
    a.test_mini_batch_size = 2048; a.test_num_workers = 0; a.num_workers = 0
    a.mlperf_logging = False; a.memory_map = False; a.data_randomize = "total"
    a.data_trace_enable_padding = False; a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10; a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 2048; a.round_targets = True
    a.mlperf_bin_loader = False; a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False

    log("  Loading kaggle dataset...")
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(a)
    ln_emb = np.array(train_data.counts)
    D = a.arch_sparse_feature_size
    ln_bot = np.fromstring(a.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + a.arch_mlp_top, dtype=int, sep="-")

    dlrm = DLRM_Net(D, ln_emb, ln_bot, ln_top,
                     arch_interaction_op="dot", arch_interaction_itself=False,
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")

    ckpt = torch.load(KAGGLE_MODEL, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']
    dlrm.load_state_dict(state_dict)
    dlrm.eval()

    return dlrm, test_ld, ln_emb, state_dict, D


def run_auc(dlrm, test_ld):
    from sklearn.metrics import roc_auc_score
    all_s, all_t = [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_ld:
            Z = dlrm(X, lS_o, lS_i)
            all_s.append(Z.cpu().numpy().flatten())
            all_t.append(T.cpu().numpy().flatten())
    scores = np.concatenate(all_s)
    targets = np.concatenate(all_t)
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
    return roc_auc_score(targets, probs)


def kaggle_crf_sweep():
    """CRF sweep: for each CRF, encode all large tables with 4x4 tiling + freq sort,
    decode, dequantize, replace weights, measure AUC."""

    log("=" * 70)
    log("PART 1: CRF SWEEP — Kaggle, 4×4 tiled, frequency-sorted")
    log("=" * 70)

    dlrm, test_ld, ln_emb, state_dict, D = load_kaggle()
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                       key=lambda x: int(x.split('.')[1]))

    large_tables = [i for i, n in enumerate(ln_emb) if n >= LARGE_TABLE_THRESHOLD]
    log(f"  {len(ln_emb)} tables, D={D}, large={large_tables}")

    # Profile access frequencies
    log("Profiling access frequencies...")
    freq = {}
    for X, lS_o, lS_i, T in test_ld:
        for t in large_tables:
            if isinstance(lS_i, (list, tuple)):
                idx = lS_i[t].flatten()
            elif lS_i.dim() == 2:
                idx = lS_i[t].flatten()
            else:
                idx = lS_i.flatten()
            if t not in freq:
                freq[t] = torch.zeros(ln_emb[t], dtype=torch.long)
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))

    # Pre-compute: quantize + freq-sort for each large table
    table_data = {}
    for t in large_tables:
        w = state_dict[emb_keys[t]]
        q, s, zp = quantize_table_uint8(w)
        sorted_idx = freq[t].argsort(descending=True)
        q_sorted = q[sorted_idx]
        inv_perm = torch.empty_like(sorted_idx)
        inv_perm[sorted_idx] = torch.arange(len(sorted_idx))
        table_data[t] = {
            'w': w, 'q_sorted': q_sorted, 's': s, 'zp': zp,
            'sorted_idx': sorted_idx, 'inv_perm': inv_perm,
            'n_rows': w.shape[0]
        }
        log(f"  Table {t}: {w.shape[0]:,} rows, scale={s:.6f}, zp={zp}")

    rpf = (FRAME_WIDTH // 4) * (FRAME_HEIGHT // 4)  # 480 * 270 = 129600
    log(f"  rows_per_frame (4×4 tiled) = {rpf}")

    # Baseline AUC
    baseline_auc = run_auc(dlrm, test_ld)
    log(f"  Baseline AUC: {baseline_auc:.6f}")

    # CRF sweep
    crfs = [0, 10, 18, 23, 28, 33]
    results = {'baseline_auc': baseline_auc, 'crfs': {}}

    for crf in crfs:
        log(f"\n  --- CRF={crf} ---")
        total_raw = 0
        total_compressed = 0
        total_fp32 = 0

        # For each large table: encode all frames, decode, dequantize, get weights
        reconstructed_weights = {}

        for t in large_tables:
            td = table_data[t]
            q_sorted = td['q_sorted']
            n = td['n_rows']
            s, zp = td['s'], td['zp']

            n_frames = (n + rpf - 1) // rpf
            all_recon = []
            t_comp = 0

            for fi in range(n_frames):
                start = fi * rpf
                end = min(start + rpf, n)
                chunk = q_sorted[start:end]

                # Tile using C++
                if chunk.shape[0] < rpf:
                    chunk_padded = torch.cat([chunk, torch.zeros(rpf - chunk.shape[0], D, dtype=torch.uint8)])
                else:
                    chunk_padded = chunk
                tiled_frame = _C.tile_rows_to_frame(chunk_padded, FRAME_WIDTH, FRAME_HEIGHT)

                # Encode + decode
                comp_size, decoded_np = encode_frame_h265(tiled_frame.numpy(), crf)
                t_comp += comp_size

                # Untile decoded frame
                decoded_t = torch.from_numpy(decoded_np)
                rows_back = _C.untile_frame_to_rows(decoded_t, rpf)
                actual = min(rpf, end - start)
                all_recon.append(rows_back[:actual])

            recon_sorted = torch.cat(all_recon, dim=0)  # uint8, freq-sorted order
            # Inverse permutation back to original order
            recon_orig = recon_sorted[td['inv_perm']]
            # Dequantize
            recon_fp32 = dequantize_uint8(recon_orig, s, zp)
            reconstructed_weights[t] = recon_fp32

            raw = n * D
            fp32 = n * D * 4
            total_raw += raw
            total_compressed += t_comp
            total_fp32 += fp32
            log(f"    Table {t}: {n_frames} frames, {raw/1024:.0f}KB → {t_comp/1024:.0f}KB "
                f"(uint8={raw/t_comp:.1f}x, fp32={fp32/t_comp:.1f}x)")

        # Replace weights and measure AUC
        orig_weights = {}
        for t in large_tables:
            orig_weights[t] = dlrm.emb_l[t].weight.data.clone()
            dlrm.emb_l[t].weight.data = reconstructed_weights[t]

        auc = run_auc(dlrm, test_ld)
        delta = auc - baseline_auc

        # Restore
        for t in large_tables:
            dlrm.emb_l[t].weight.data = orig_weights[t]

        ratio_uint8 = total_raw / total_compressed
        ratio_fp32 = total_fp32 / total_compressed
        log(f"  CRF={crf}: AUC={auc:.6f} (delta={delta:+.6f}), "
            f"ratio_uint8={ratio_uint8:.1f}x, ratio_fp32={ratio_fp32:.1f}x")

        results['crfs'][crf] = {
            'auc': auc, 'auc_delta': delta,
            'ratio_uint8': ratio_uint8, 'ratio_fp32': ratio_fp32,
            'total_compressed_bytes': total_compressed,
            'total_uint8_bytes': total_raw,
            'total_fp32_bytes': total_fp32,
        }

    # Summary table
    log(f"\n  === CRF SWEEP SUMMARY (Kaggle, 4×4 tiled + freq sort) ===")
    log(f"  Baseline AUC: {baseline_auc:.6f}")
    log(f"  {'CRF':>4s} | {'AUC':>10s} | {'Delta':>10s} | {'vs uint8':>10s} | {'vs fp32':>10s}")
    log(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for crf in crfs:
        r = results['crfs'][crf]
        log(f"  {crf:4d} | {r['auc']:10.6f} | {r['auc_delta']:+10.6f} | "
            f"{r['ratio_uint8']:>9.1f}x | {r['ratio_fp32']:>9.1f}x")

    return results


# ============================================================
# PART 2: DEEP ENTROPY ANALYSIS FOR TERABYTE
# ============================================================
def load_terabyte_weights():
    """Load just the Terabyte model weights (no data loader needed for entropy analysis)."""
    ckpt = torch.load(TERABYTE_MODEL, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                       key=lambda x: int(x.split('.')[1]))
    return state_dict, emb_keys


def compute_entropy(data_uint8):
    """Compute Shannon entropy in bits per byte."""
    flat = data_uint8.flatten().numpy().astype(np.int32)
    counts = np.bincount(flat, minlength=256).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -(probs * np.log2(probs)).sum()


def terabyte_entropy_analysis():
    """Deep analysis of why Terabyte tables have such different compression ratios."""

    log("\n" + "=" * 70)
    log("PART 2: DEEP ENTROPY ANALYSIS — Terabyte")
    log("=" * 70)

    state_dict, emb_keys = load_terabyte_weights()
    D = state_dict[emb_keys[0]].shape[1]
    log(f"  D={D}, {len(emb_keys)} tables")

    # Known compression ratios from intrinsic_compressibility experiment
    known_ratios = {0: 1409, 9: 176, 10: 19, 11: 12, 19: 33, 20: 25, 21: 645, 22: 12}

    # Identify large tables
    large_tables = []
    for i, k in enumerate(emb_keys):
        n = state_dict[k].shape[0]
        if n >= LARGE_TABLE_THRESHOLD:
            large_tables.append(i)

    log(f"  Large tables: {large_tables}")

    results = {}

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        n_rows, dim = w.shape
        log(f"\n  === Table {t}: {n_rows:,} rows × {dim} dims ===")

        # 1. Basic fp32 statistics
        fp32_mean = w.mean().item()
        fp32_std = w.std().item()
        fp32_min = w.min().item()
        fp32_max = w.max().item()
        fp32_abs_mean = w.abs().mean().item()
        log(f"    fp32: mean={fp32_mean:.6f}, std={fp32_std:.6f}, "
            f"range=[{fp32_min:.4f}, {fp32_max:.4f}], |mean|={fp32_abs_mean:.6f}")

        # 2. Near-zero analysis
        row_norms = torch.norm(w, dim=1)
        norm_mean = row_norms.mean().item()
        norm_std = row_norms.std().item()
        near_zero_1 = (row_norms < 0.01).float().mean().item()
        near_zero_01 = (row_norms < 0.001).float().mean().item()
        near_zero_1e4 = (row_norms < 0.0001).float().mean().item()
        log(f"    Row L2 norm: mean={norm_mean:.6f}, std={norm_std:.6f}")
        log(f"    Near-zero rows: <0.01={near_zero_1*100:.1f}%, <0.001={near_zero_01*100:.1f}%, "
            f"<1e-4={near_zero_1e4*100:.1f}%")

        # 3. Quantize to uint8 and analyze
        q, s, zp = quantize_table_uint8(w)
        q_flat = q.flatten().float()
        uint8_mean = q_flat.mean().item()
        uint8_std = q_flat.std().item()
        uint8_entropy = compute_entropy(q)
        log(f"    uint8: mean={uint8_mean:.1f}, std={uint8_std:.1f}, "
            f"entropy={uint8_entropy:.2f} bits/byte, scale={s:.8f}, zp={zp}")

        # 4. Per-dimension analysis
        dim_means = w.mean(dim=0)
        dim_stds = w.std(dim=0)
        dim_ranges = w.max(dim=0).values - w.min(dim=0).values
        log(f"    Per-dim: mean_of_means={dim_means.mean().item():.6f}, "
            f"mean_of_stds={dim_stds.mean().item():.6f}, "
            f"mean_range={dim_ranges.mean().item():.6f}")

        # 5. Value concentration: what % of values fall in a narrow band?
        # After uint8: how many unique values are actually used?
        unique_uint8 = len(torch.unique(q))
        # What fraction falls within ±2 of the zero point?
        near_zp = ((q_flat - zp).abs() <= 2).float().mean().item()
        near_zp_10 = ((q_flat - zp).abs() <= 10).float().mean().item()
        log(f"    uint8 unique values: {unique_uint8}/256, "
            f"within ±2 of zp={near_zp*100:.1f}%, ±10={near_zp_10*100:.1f}%")

        # 6. Row-to-row similarity (sample 10000 adjacent pairs)
        n_sample = min(10000, n_rows - 1)
        sample_idx = torch.randint(0, n_rows - 1, (n_sample,))
        row_diffs = (q[sample_idx + 1].float() - q[sample_idx].float())
        adj_mse = (row_diffs ** 2).mean().item()
        adj_mae = row_diffs.abs().mean().item()
        log(f"    Adjacent row diff (uint8): MSE={adj_mse:.2f}, MAE={adj_mae:.2f}")

        # 7. Histogram: value distribution in uint8 space
        hist = torch.bincount(q.flatten().int(), minlength=256).float()
        hist_norm = hist / hist.sum()
        # Top 5 values
        top5_vals = torch.argsort(hist, descending=True)[:5]
        top5_pct = hist[top5_vals] / hist.sum() * 100
        top5_str = ", ".join([f"{v.item()}({p.item():.1f}%)" for v, p in zip(top5_vals, top5_pct)])
        log(f"    Top-5 uint8 values: {top5_str}")

        # 8. How spread is the distribution?  Gini coefficient of uint8 histogram
        sorted_hist = torch.sort(hist)[0]
        n_vals = 256
        cumsum = torch.cumsum(sorted_hist, 0)
        gini = (n_vals + 1 - 2 * (cumsum / cumsum[-1]).sum().item()) / n_vals
        log(f"    uint8 Gini coefficient: {gini:.4f} (1=concentrated, 0=uniform)")

        # 9. Effective dimensionality via PCA (sample 50K rows)
        n_pca = min(50000, n_rows)
        pca_sample = w[torch.randint(0, n_rows, (n_pca,))]
        centered = pca_sample - pca_sample.mean(dim=0)
        _, S, _ = torch.svd_lowrank(centered, q=min(dim, 32))
        var_explained = (S ** 2) / (S ** 2).sum()
        cumvar = torch.cumsum(var_explained, 0)
        dims_90 = (cumvar < 0.90).sum().item() + 1
        dims_99 = (cumvar < 0.99).sum().item() + 1
        top1_var = var_explained[0].item()
        log(f"    PCA: top-1 var={top1_var*100:.1f}%, dims for 90%={dims_90}, 99%={dims_99}")

        # 10. Compression test: flat vs 8x8 tiled at CRF=18
        rpf_flat = (FRAME_WIDTH // D) * FRAME_HEIGHT  # 30 * 1080 = 32400
        n_test = min(rpf_flat, n_rows)  # test on one frame
        test_q = q[:n_test]

        # Flat encode
        frame_flat = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        rpr = FRAME_WIDTH // D
        H_used = (n_test + rpr - 1) // rpr
        frame_flat[:H_used] = test_q[:H_used * rpr].reshape(H_used, rpr * D).numpy()
        flat_sz_18, _ = encode_frame_h265(frame_flat, 18)
        flat_sz_0, _ = encode_frame_h265(frame_flat, 0)

        # 8x8 tiled encode (reshape each D=64 row to 8x8)
        frame_8x8 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        tiles_per_row = FRAME_WIDTH // 8  # 240
        for r in range(n_test):
            ty = r // tiles_per_row
            tx = r % tiles_per_row
            tile = test_q[r].numpy().reshape(8, 8)
            for ly in range(8):
                frame_8x8[ty * 8 + ly, tx * 8: tx * 8 + 8] = tile[ly]
        tiled_sz_18, _ = encode_frame_h265(frame_8x8, 18)
        tiled_sz_0, _ = encode_frame_h265(frame_8x8, 0)

        raw_bytes = n_test * D
        log(f"    1-frame compression (n={n_test}):")
        log(f"      Flat  CRF=0: {raw_bytes/flat_sz_0:.1f}x,  CRF=18: {raw_bytes/flat_sz_18:.1f}x")
        log(f"      8×8   CRF=0: {raw_bytes/tiled_sz_0:.1f}x,  CRF=18: {raw_bytes/tiled_sz_18:.1f}x")
        log(f"      Tiling benefit at CRF=18: {flat_sz_18/tiled_sz_18:.2f}x smaller")

        crf18_ratio = known_ratios.get(t, 0)
        results[t] = {
            'n_rows': n_rows, 'dim': dim,
            'fp32_mean': fp32_mean, 'fp32_std': fp32_std,
            'fp32_abs_mean': fp32_abs_mean,
            'fp32_min': fp32_min, 'fp32_max': fp32_max,
            'norm_mean': norm_mean, 'norm_std': norm_std,
            'near_zero_001': near_zero_01,
            'uint8_entropy': uint8_entropy,
            'uint8_std': uint8_std,
            'uint8_unique': unique_uint8,
            'near_zp_pct': near_zp,
            'near_zp_10_pct': near_zp_10,
            'adj_row_mse': adj_mse,
            'gini': gini,
            'pca_top1_var': top1_var,
            'pca_dims_90': dims_90,
            'pca_dims_99': dims_99,
            'flat_crf18_ratio': raw_bytes / flat_sz_18,
            'tiled_8x8_crf18_ratio': raw_bytes / tiled_sz_18,
            'flat_crf0_ratio': raw_bytes / flat_sz_0,
            'tiled_8x8_crf0_ratio': raw_bytes / tiled_sz_0,
            'known_crf18_ratio': crf18_ratio,
        }

        del w, q
        gc.collect()

    # Summary comparison: good compressors vs bad compressors
    log(f"\n  === SUMMARY: What predicts compressibility? ===")
    log(f"  {'Table':>5s} | {'Rows':>10s} | {'CRF18':>6s} | {'Entropy':>7s} | "
        f"{'uint8 std':>9s} | {'near_zp%':>8s} | {'Gini':>6s} | "
        f"{'PCA top1':>8s} | {'norm':>8s} | {'flat':>6s} | {'8×8':>6s}")
    log(f"  {'-'*5}-+-{'-'*10}-+-{'-'*6}-+-{'-'*7}-+-"
        f"{'-'*9}-+-{'-'*8}-+-{'-'*6}-+-"
        f"{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")

    for t in sorted(results.keys(), key=lambda t: -results[t].get('known_crf18_ratio', 0)):
        r = results[t]
        log(f"  {t:5d} | {r['n_rows']:>10,} | {r['known_crf18_ratio']:>5d}x | "
            f"{r['uint8_entropy']:>6.2f}b | {r['uint8_std']:>8.1f} | "
            f"{r['near_zp_pct']*100:>7.1f}% | {r['gini']:>6.4f} | "
            f"{r['pca_top1_var']*100:>7.1f}% | {r['norm_mean']:>8.5f} | "
            f"{r['flat_crf18_ratio']:>5.1f}x | {r['tiled_8x8_crf18_ratio']:>5.1f}x")

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    all_results = {}

    # Part 1: CRF sweep on Kaggle
    log("Starting CRF sweep on Kaggle...")
    crf_results = kaggle_crf_sweep()
    all_results['kaggle_crf_sweep'] = crf_results

    # Part 2: Terabyte entropy analysis
    log("\nStarting Terabyte entropy analysis...")
    entropy_results = terabyte_entropy_analysis()
    all_results['terabyte_entropy'] = entropy_results

    # Save
    # Convert numpy/torch types for JSON
    def jsonify(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [jsonify(x) for x in obj]
        return obj

    with open(os.path.join(RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(jsonify(all_results), f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR}/all_results.json")
    log("DONE.")
