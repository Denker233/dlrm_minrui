#!/usr/bin/env python3
"""
Intrinsic Compressibility Experiments

Experiments to characterize WHY embedding tables are compressible by lossy video codecs
and whether row reordering matters.

Experiment 1: Lossy CRF sweep WITH vs WITHOUT reordering
Experiment 2: Spatial smoothness (Total Variation) across orderings
Experiment 3: Intrinsic compressibility (entropy, PCA effective rank, value distribution)
Experiment 4: Random data baseline comparison

Runs on both Kaggle (D=16) and Terabyte (D=64) models.
"""

import os, sys, time, json, gc, tempfile
import numpy as np
import torch
import torch.nn as nn

try:
    import compressed_emb as _C
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("WARNING: C++ extension not available, using PyAV fallback")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIG
# ============================================================
KAGGLE_MODEL = "./models/dlrm_kaggle_correct.pt"
TERABYTE_MODEL = "./models/dlrm_terabyte_4day.pt"
LARGE_TABLE_THRESHOLD = 50000
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CRF_VALUES = [0, 10, 18, 23, 28]
RESULTS_DIR = "results/intrinsic_compressibility"
os.makedirs(RESULTS_DIR, exist_ok=True)

log_fh = open(os.path.join(RESULTS_DIR, "experiment.log"), "w")
def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n"); log_fh.flush()

# ============================================================
# UTILITIES
# ============================================================
def quantize_table_uint8(w):
    """Quantize fp32 weights to uint8 with table-wise min/max."""
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = torch.clamp(torch.round(w / s + zp), 0, 255).to(torch.uint8)
    return q, s, zp

def dequantize_uint8(q, s, zp):
    """Dequantize uint8 back to fp32."""
    return (q.float() - zp) * s

def pack_rows_flat(rows_uint8, D, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Pack uint8 rows into a (height, width) frame using flat layout."""
    N = rows_uint8.shape[0]
    rows_per_row = width // D
    padded_N = ((N + rows_per_row - 1) // rows_per_row) * rows_per_row
    if padded_N > N:
        pad = torch.zeros(padded_N - N, D, dtype=torch.uint8)
        data = torch.cat([rows_uint8, pad], dim=0)
    else:
        data = rows_uint8
    H_used = padded_N // rows_per_row
    frame = torch.zeros(height, width, dtype=torch.uint8)
    frame[:H_used] = data.reshape(H_used, rows_per_row * D)
    return frame

def unpack_frame_flat(frame, n_rows, D, width=FRAME_WIDTH):
    """Unpack frame back to rows."""
    rows_per_row = width // D
    H_used = (n_rows + rows_per_row - 1) // rows_per_row
    data = frame[:H_used].reshape(-1, D)
    return data[:n_rows]

def encode_frame_h265(frame, crf, lossless=False):
    """Encode frame to H.265, return compressed bytes size and decoded frame."""
    with tempfile.NamedTemporaryFile(suffix='.h265', delete=False) as f:
        tmp_path = f.name
    try:
        if HAS_CPP:
            _C.encode_h265_frame(frame, tmp_path, lossless, crf)
        else:
            encode_frame_pyav(frame.numpy(), tmp_path, crf, lossless)
        comp_size = os.path.getsize(tmp_path)
        # Decode back
        if HAS_CPP:
            decoded = _C.decode_h265_frame_from_file(tmp_path)
        else:
            decoded = decode_frame_pyav(tmp_path)
        return comp_size, decoded
    finally:
        os.unlink(tmp_path)

def encode_frame_pyav(frame_np, out_path, crf=0, lossless=False):
    import av
    h, w = frame_np.shape
    container = av.open(out_path, mode='w', format='hevc')
    stream = container.add_stream('libx265', rate=1)
    stream.width = w; stream.height = h
    stream.pix_fmt = 'gray'
    if lossless:
        stream.options = {'x265-params': 'lossless=1'}
    else:
        stream.options = {'crf': str(crf), 'x265-params': f'qp={crf}'}
    video_frame = av.VideoFrame.from_ndarray(frame_np, format='gray')
    for packet in stream.encode(video_frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

def decode_frame_pyav(path):
    import av
    container = av.open(path)
    for frame in container.decode(video=0):
        return torch.from_numpy(frame.to_ndarray(format='gray'))

def rows_per_frame(D):
    return (FRAME_WIDTH // D) * FRAME_HEIGHT

# ============================================================
# LOAD MODELS
# ============================================================
def load_model_weights(model_path, label):
    log(f"Loading {label} model from {model_path}...")
    ld = torch.load(model_path, map_location='cpu', weights_only=False)
    sd = ld['state_dict']
    tables = {}
    for k, v in sd.items():
        if 'emb_l' in k and 'weight' in k:
            idx = int(k.split('.')[1])
            tables[idx] = v
    D = next(iter(tables.values())).shape[1]
    large = {i: w for i, w in tables.items() if w.shape[0] > LARGE_TABLE_THRESHOLD}
    log(f"  {len(tables)} tables, D={D}, {len(large)} large (>{LARGE_TABLE_THRESHOLD} rows)")
    for i in sorted(large.keys()):
        log(f"    Table {i}: {large[i].shape[0]:,} rows")
    return tables, large, D

# ============================================================
# EXPERIMENT 1: LOSSY CRF x REORDERING SWEEP
# ============================================================
def reorder_rows(uint8_rows, order, freq=None):
    """Reorder rows according to specified order."""
    N = uint8_rows.shape[0]
    if order == 'natural':
        return uint8_rows, torch.arange(N)
    elif order == 'random':
        perm = torch.randperm(N)
        return uint8_rows[perm], perm
    elif order == 'frequency':
        assert freq is not None
        perm = freq.argsort(descending=True)
        return uint8_rows[perm], perm
    elif order == 'reverse_frequency':
        assert freq is not None
        perm = freq.argsort(descending=False)
        return uint8_rows[perm], perm
    else:
        raise ValueError(f"Unknown order: {order}")

def experiment1_crf_reorder(large_tables, D, label, access_freq=None):
    """Experiment 1: CRF sweep x reordering."""
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 1: CRF x Reordering ({label}, D={D})")
    log(f"{'='*70}")

    orderings = ['natural', 'random', 'frequency']
    results = {}

    for t_idx in sorted(large_tables.keys()):
        w = large_tables[t_idx]
        N = w.shape[0]
        q, s, zp = quantize_table_uint8(w)

        # Build frequency counts (use uniform if no access data)
        if access_freq is not None and t_idx in access_freq:
            freq = access_freq[t_idx]
        else:
            # Use value magnitude as proxy for frequency
            freq = w.abs().sum(dim=1)

        rpf = rows_per_frame(D)
        raw_bytes = N * D  # uint8 raw size
        raw_fp32_bytes = N * D * 4

        log(f"\n  Table {t_idx}: {N:,} rows, {raw_bytes/1024/1024:.1f}MB uint8, {raw_fp32_bytes/1024/1024:.1f}MB fp32")

        table_results = {}
        for order in orderings:
            q_ordered, perm = reorder_rows(q, order, freq)
            order_results = {}

            for crf in CRF_VALUES:
                lossless = (crf == 0)
                total_comp = 0
                total_mse = 0.0
                total_max_err = 0
                n_frames = (N + rpf - 1) // rpf

                for fid in range(n_frames):
                    start = fid * rpf
                    end = min(start + rpf, N)
                    frame_rows = q_ordered[start:end]
                    frame = pack_rows_flat(frame_rows, D)
                    comp_size, decoded_frame = encode_frame_h265(frame, crf, lossless)
                    total_comp += comp_size

                    # Compute reconstruction error
                    recon_rows = unpack_frame_flat(decoded_frame, end - start, D)
                    mse = ((frame_rows.float() - recon_rows.float()) ** 2).mean().item()
                    max_err = (frame_rows.float() - recon_rows.float()).abs().max().item()
                    total_mse += mse * (end - start)
                    total_max_err = max(total_max_err, int(max_err))

                avg_mse = total_mse / N
                ratio_uint8 = raw_bytes / max(1, total_comp)
                ratio_fp32 = raw_fp32_bytes / max(1, total_comp)

                order_results[crf] = {
                    'compressed_bytes': total_comp,
                    'ratio_vs_uint8': round(ratio_uint8, 2),
                    'ratio_vs_fp32': round(ratio_fp32, 2),
                    'mse_uint8': round(avg_mse, 4),
                    'max_error_uint8': total_max_err,
                    'psnr_uint8': round(10 * np.log10(255**2 / max(avg_mse, 1e-10)), 2),
                }

                log(f"    {order:20s} CRF={crf:2d}: {ratio_uint8:6.1f}x (uint8), "
                    f"{ratio_fp32:7.1f}x (fp32), MSE={avg_mse:.2f}, MaxErr={total_max_err}")

            table_results[order] = order_results
        results[t_idx] = table_results

    return results

# ============================================================
# EXPERIMENT 2: SPATIAL SMOOTHNESS
# ============================================================
def total_variation(frame):
    """Total variation of a 2D frame. Lower = smoother."""
    f = frame.float()
    dx = torch.abs(f[:, 1:] - f[:, :-1]).mean().item()
    dy = torch.abs(f[1:, :] - f[:-1, :]).mean().item()
    return dx + dy

def row_to_row_l2(rows):
    """Mean L2 distance between consecutive rows."""
    if rows.shape[0] < 2:
        return 0.0
    diffs = (rows[1:].float() - rows[:-1].float()).norm(dim=1)
    return diffs.mean().item()

def experiment2_smoothness(large_tables, D, label, access_freq=None):
    """Experiment 2: Spatial smoothness metrics."""
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 2: Spatial Smoothness ({label}, D={D})")
    log(f"{'='*70}")

    orderings = ['natural', 'random', 'frequency', 'reverse_frequency']
    results = {}

    for t_idx in sorted(large_tables.keys()):
        w = large_tables[t_idx]
        N = w.shape[0]
        q, s, zp = quantize_table_uint8(w)

        if access_freq is not None and t_idx in access_freq:
            freq = access_freq[t_idx]
        else:
            freq = w.abs().sum(dim=1)

        rpf = rows_per_frame(D)
        table_results = {}

        for order in orderings:
            q_ordered, _ = reorder_rows(q, order, freq)

            # Compute TV for first frame only (representative)
            n_first = min(rpf, N)
            frame = pack_rows_flat(q_ordered[:n_first], D)
            tv = total_variation(frame)
            r2r = row_to_row_l2(q_ordered[:min(10000, N)])

            # Also compute column-wise variation (across rows, per dimension)
            sample = q_ordered[:min(10000, N)].float()
            col_var = sample.var(dim=0).mean().item()  # mean variance across dimensions
            col_std = np.sqrt(col_var)

            table_results[order] = {
                'total_variation': round(tv, 4),
                'row_to_row_l2': round(r2r, 4),
                'column_variance': round(col_var, 4),
                'column_std': round(col_std, 4),
            }

            log(f"  Table {t_idx} [{order:20s}]: TV={tv:.4f}, row-L2={r2r:.2f}, col_std={col_std:.2f}")

        results[t_idx] = table_results

    return results

# ============================================================
# EXPERIMENT 3: INTRINSIC COMPRESSIBILITY
# ============================================================
def byte_entropy(data_uint8):
    """Shannon entropy in bits per byte for uint8 data."""
    flat = data_uint8.flatten().numpy().astype(np.int32)
    counts = np.bincount(flat, minlength=256).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -(probs * np.log2(probs)).sum()

def experiment3_intrinsic(large_tables, D, label):
    """Experiment 3: Entropy, PCA rank, value distribution."""
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 3: Intrinsic Compressibility ({label}, D={D})")
    log(f"{'='*70}")

    results = {}

    for t_idx in sorted(large_tables.keys()):
        w = large_tables[t_idx]
        N = w.shape[0]
        q, s, zp = quantize_table_uint8(w)

        # --- 3a: Entropy ---
        ent = byte_entropy(q)

        # --- 3b: Value distribution ---
        flat = q.flatten().float()
        val_mean = flat.mean().item()
        val_std = flat.std().item()
        val_median = flat.median().item()
        # Fraction of values within ±16 of mean
        near_mean = ((flat - val_mean).abs() <= 16).float().mean().item()
        # Fraction of unique uint8 values actually used
        unique_vals = len(torch.unique(q.flatten()))

        # --- 3c: PCA / SVD effective rank ---
        # Sample rows for SVD (full table may be too large)
        sample_n = min(50000, N)
        sample_idx = torch.randperm(N)[:sample_n]
        sample_fp32 = w[sample_idx].float()
        # Center
        sample_centered = sample_fp32 - sample_fp32.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(sample_centered, full_matrices=False)
            var_explained = (S ** 2) / (S ** 2).sum()
            cumvar = torch.cumsum(var_explained, 0)
            rank_90 = int((cumvar < 0.90).sum().item()) + 1
            rank_95 = int((cumvar < 0.95).sum().item()) + 1
            rank_99 = int((cumvar < 0.99).sum().item()) + 1

            # Effective rank (exponential of entropy of normalized singular values)
            sv_norm = S / S.sum()
            sv_norm = sv_norm[sv_norm > 0]
            eff_rank = torch.exp(-(sv_norm * torch.log(sv_norm)).sum()).item()

            # Fraction of variance in top-1 component
            top1_var = var_explained[0].item()
        except Exception as e:
            log(f"    SVD failed for table {t_idx}: {e}")
            rank_90 = rank_95 = rank_99 = D
            eff_rank = D
            top1_var = 1.0 / D
            var_explained = torch.ones(D) / D
            cumvar = torch.linspace(0, 1, D)

        # --- 3d: Per-dimension stats ---
        dim_std = w.std(dim=0)  # std of each dimension across all rows
        dim_range = w.max(dim=0).values - w.min(dim=0).values
        avg_dim_std = dim_std.mean().item()
        avg_dim_range = dim_range.mean().item()

        # --- 3e: Row norms ---
        row_norms = w.norm(dim=1)
        avg_norm = row_norms.mean().item()
        std_norm = row_norms.std().item()
        # Fraction of near-zero rows (norm < 0.01)
        near_zero_rows = (row_norms < 0.01).float().mean().item()

        table_result = {
            'n_rows': N,
            'D': D,
            'entropy_bits_per_byte': round(ent, 4),
            'uint8_mean': round(val_mean, 2),
            'uint8_std': round(val_std, 2),
            'uint8_median': round(val_median, 2),
            'uint8_near_mean_frac': round(near_mean, 4),
            'unique_uint8_values': unique_vals,
            'svd_rank_90': rank_90,
            'svd_rank_95': rank_95,
            'svd_rank_99': rank_99,
            'effective_rank': round(eff_rank, 2),
            'top1_variance_frac': round(top1_var, 4),
            'cumulative_variance': [round(x, 4) for x in cumvar.tolist()],
            'avg_dim_std': round(avg_dim_std, 6),
            'avg_dim_range': round(avg_dim_range, 6),
            'avg_row_norm': round(avg_norm, 6),
            'std_row_norm': round(std_norm, 6),
            'near_zero_row_frac': round(near_zero_rows, 4),
            'quant_scale': round(s, 8),
            'quant_zp': zp,
        }
        results[t_idx] = table_result

        log(f"\n  Table {t_idx}: {N:,} rows, D={D}")
        log(f"    Entropy: {ent:.2f} bits/byte (max 8.0)")
        log(f"    uint8 distribution: mean={val_mean:.1f}, std={val_std:.1f}, "
            f"median={val_median:.0f}, unique={unique_vals}/256")
        log(f"    {near_mean*100:.1f}% of values within ±16 of mean")
        log(f"    SVD effective rank: {eff_rank:.1f} / {D}")
        log(f"    Dims for 90% var: {rank_90}, 95%: {rank_95}, 99%: {rank_99} (of {D})")
        log(f"    Top-1 component: {top1_var*100:.1f}% of variance")
        log(f"    Row norms: mean={avg_norm:.4f}, std={std_norm:.4f}")
        log(f"    Near-zero rows (norm<0.01): {near_zero_rows*100:.1f}%")

    return results

# ============================================================
# EXPERIMENT 4: RANDOM DATA BASELINE
# ============================================================
def experiment4_random(large_tables, D, label):
    """Experiment 4: Compare compression of real vs random data."""
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 4: Random Data Baseline ({label}, D={D})")
    log(f"{'='*70}")

    results = {}
    rpf = rows_per_frame(D)

    for t_idx in sorted(large_tables.keys()):
        w = large_tables[t_idx]
        N = w.shape[0]
        q_real, s, zp = quantize_table_uint8(w)

        # Use first frame for comparison (representative)
        n_frame = min(rpf, N)
        frame_real = pack_rows_flat(q_real[:n_frame], D)

        # Random variants
        # 1. Fully uniform random
        frame_uniform = torch.randint(0, 256, frame_real.shape, dtype=torch.uint8)

        # 2. Random with same mean/std as real data
        real_mean = q_real[:n_frame].float().mean().item()
        real_std = q_real[:n_frame].float().std().item()
        gaussian = torch.normal(real_mean, real_std, frame_real.shape)
        frame_gaussian = torch.clamp(gaussian, 0, 255).to(torch.uint8)

        # 3. Row-shuffled (same rows, random order)
        perm = torch.randperm(n_frame)
        q_shuffled = q_real[:n_frame][perm]
        frame_shuffled = pack_rows_flat(q_shuffled, D)

        # 4. Column-shuffled (same values per column, but rows mixed across columns)
        q_col_shuffle = q_real[:n_frame].clone()
        for d in range(D):
            q_col_shuffle[:, d] = q_col_shuffle[torch.randperm(n_frame), d]
        frame_col_shuffled = pack_rows_flat(q_col_shuffle, D)

        table_results = {}

        for crf in CRF_VALUES:
            lossless = (crf == 0)
            sizes = {}
            for name, frame in [('real', frame_real),
                                 ('uniform_random', frame_uniform),
                                 ('gaussian_random', frame_gaussian),
                                 ('row_shuffled', frame_shuffled),
                                 ('col_shuffled', frame_col_shuffled)]:
                comp_size, _ = encode_frame_h265(frame, crf, lossless)
                sizes[name] = comp_size

            raw_bytes = frame_real.numel()
            table_results[crf] = {
                name: {
                    'bytes': sz,
                    'ratio': round(raw_bytes / max(1, sz), 2),
                }
                for name, sz in sizes.items()
            }

            log(f"  Table {t_idx} CRF={crf:2d}: "
                f"real={sizes['real']/1024:.0f}KB ({raw_bytes/max(1,sizes['real']):.1f}x), "
                f"uniform={sizes['uniform_random']/1024:.0f}KB ({raw_bytes/max(1,sizes['uniform_random']):.1f}x), "
                f"gaussian={sizes['gaussian_random']/1024:.0f}KB ({raw_bytes/max(1,sizes['gaussian_random']):.1f}x), "
                f"row_shuf={sizes['row_shuffled']/1024:.0f}KB ({raw_bytes/max(1,sizes['row_shuffled']):.1f}x), "
                f"col_shuf={sizes['col_shuffled']/1024:.0f}KB ({raw_bytes/max(1,sizes['col_shuffled']):.1f}x)")

        results[t_idx] = table_results

    return results

# ============================================================
# AUC IMPACT OF LOSSY COMPRESSION (requires model + data)
# ============================================================
def experiment1b_auc_impact(model_path, dataset_name, D, large_table_indices):
    """Measure AUC loss from lossy compression per ordering."""
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 1b: AUC Impact ({dataset_name}, D={D})")
    log(f"{'='*70}")

    # Only run for Kaggle (Terabyte data loading is too slow for quick experiment)
    if dataset_name == 'terabyte':
        os.environ['CRITEO_DAYS'] = '4'

    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net

    # Build args
    class Args: pass
    a = Args()
    a.arch_sparse_feature_size = D
    if D == 16:
        a.arch_mlp_bot = "13-512-256-64-16"; a.arch_mlp_top = "512-256-1"
        a.raw_data_file = os.path.expanduser("~/input/train.txt")
        a.processed_data_file = os.path.expanduser("~/input/kaggleAdDisplayChallenge_processed.npz")
        a.data_set = "kaggle"
    else:
        a.arch_mlp_bot = "13-512-256-64"; a.arch_mlp_top = "512-512-256-1"
        a.raw_data_file = os.path.expanduser("~/input/terabyte/day")
        a.processed_data_file = os.path.expanduser("~/input/terabyte/terabyte_processed.npz")
        a.data_set = "terabyte"
    a.arch_interaction_op = "dot"; a.arch_interaction_itself = False
    a.data_generation = "dataset"; a.loss_function = "bce"
    a.max_ind_range = 10000000 if D == 64 else -1
    a.test_mini_batch_size = 2048; a.test_num_workers = 0; a.num_workers = 0
    a.mlperf_logging = False; a.memory_map = False; a.data_randomize = "total"
    a.data_trace_enable_padding = False; a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10; a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 2048; a.round_targets = True
    a.mlperf_bin_loader = False; a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False

    log("  Loading dataset...")
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(a)
    ln_emb = np.array(train_data.counts)
    ln_bot = np.fromstring(a.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    if D == 16:
        num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
        ln_top = np.fromstring(str(num_int) + "-" + a.arch_mlp_top, dtype=int, sep="-")
    else:
        num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
        ln_top = np.fromstring(str(num_int) + "-" + a.arch_mlp_top, dtype=int, sep="-")

    dlrm = DLRM_Net(D, ln_emb, ln_bot, ln_top,
                     arch_interaction_op="dot", arch_interaction_itself=False,
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")
    ld = torch.load(model_path, map_location='cpu', weights_only=False)
    sd = ld['state_dict']
    dlrm.load_state_dict(sd)
    dlrm.eval()

    # Collect access frequencies
    log("  Profiling access frequencies...")
    freq = {}
    for t in large_table_indices:
        freq[t] = torch.zeros(ln_emb[t], dtype=torch.long)
    for X, lS_o, lS_i, T in test_ld:
        for t in large_table_indices:
            if isinstance(lS_i, (list, tuple)):
                indices = lS_i[t].flatten()
            elif lS_i.dim() == 2:
                indices = lS_i[t].flatten()
            else:
                indices = lS_i.flatten()
            valid = indices[indices < ln_emb[t]]
            freq[t].scatter_add_(0, valid.long(), torch.ones_like(valid, dtype=torch.long))

    # Run baseline
    log("  Running baseline inference...")
    from sklearn.metrics import roc_auc_score
    def run_auc(model, loader):
        all_s, all_t = [], []
        with torch.no_grad():
            for X, lS_o, lS_i, T in loader:
                Z = model(X, lS_o, lS_i)
                all_s.append(Z.cpu().numpy().flatten())
                all_t.append(T.cpu().numpy().flatten())
        scores = np.concatenate(all_s)
        targets = np.concatenate(all_t)
        probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
        return roc_auc_score(targets, probs)

    baseline_auc = run_auc(dlrm, test_ld)
    log(f"  Baseline AUC: {baseline_auc:.6f}")

    # Test lossy compression impact
    orderings = ['natural', 'random', 'frequency']
    crfs = [0, 10, 18, 23, 28]
    results = {'baseline_auc': baseline_auc}

    for crf in crfs:
        for order in orderings:
            # Quantize, reorder, encode, decode, dequant, replace weights
            for t in large_table_indices:
                w_orig = sd[f'emb_l.{t}.weight']
                q, s, zp = quantize_table_uint8(w_orig)

                q_ordered, perm = reorder_rows(q, order, freq.get(t))
                inv_perm = torch.argsort(perm)

                rpf = rows_per_frame(D)
                N = q.shape[0]
                recon_rows = []
                for fid in range((N + rpf - 1) // rpf):
                    start = fid * rpf
                    end = min(start + rpf, N)
                    frame = pack_rows_flat(q_ordered[start:end], D)
                    _, decoded = encode_frame_h265(frame, crf, lossless=(crf == 0))
                    recon = unpack_frame_flat(decoded, end - start, D)
                    recon_rows.append(recon)
                recon_all = torch.cat(recon_rows, dim=0)
                # Undo reorder
                recon_orig_order = recon_all[inv_perm]
                # Dequantize
                w_recon = dequantize_uint8(recon_orig_order, s, zp)
                dlrm.emb_l[t].weight.data = w_recon

            auc = run_auc(dlrm, test_ld)
            delta = auc - baseline_auc

            key = f"crf{crf}_{order}"
            results[key] = {'auc': auc, 'delta': delta}
            log(f"  CRF={crf:2d} {order:20s}: AUC={auc:.6f}, delta={delta:+.6f}")

            # Restore original weights
            for t in large_table_indices:
                dlrm.emb_l[t].weight.data = sd[f'emb_l.{t}.weight'].clone()

    return results, freq

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("INTRINSIC COMPRESSIBILITY EXPERIMENTS")
    log("=" * 70)
    log(f"Frame: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    log(f"CRF values: {CRF_VALUES}")

    all_results = {}

    # ---- KAGGLE (D=16) ----
    if os.path.exists(KAGGLE_MODEL):
        log("\n" + "#" * 70)
        log("# KAGGLE DATASET (D=16)")
        log("#" * 70)

        tables_k, large_k, D_k = load_model_weights(KAGGLE_MODEL, "Kaggle")

        # Experiments 2, 3, 4 don't need data loader
        r3_k = experiment3_intrinsic(large_k, D_k, "Kaggle")
        all_results['kaggle_exp3'] = r3_k

        r2_k = experiment2_smoothness(large_k, D_k, "Kaggle")
        all_results['kaggle_exp2'] = r2_k

        r4_k = experiment4_random(large_k, D_k, "Kaggle")
        all_results['kaggle_exp4'] = r4_k

        # Experiment 1 (compression only, no AUC yet)
        r1_k = experiment1_crf_reorder(large_k, D_k, "Kaggle")
        all_results['kaggle_exp1'] = r1_k

        # Experiment 1b (AUC impact — needs data loader)
        large_k_indices = sorted(large_k.keys())
        r1b_k, freq_k = experiment1b_auc_impact(KAGGLE_MODEL, 'kaggle', D_k, large_k_indices)
        all_results['kaggle_exp1b_auc'] = r1b_k

        # Re-run experiment 2 with real access freq
        r2_k_freq = experiment2_smoothness(large_k, D_k, "Kaggle (with freq)", freq_k)
        all_results['kaggle_exp2_freq'] = r2_k_freq

        del tables_k, large_k
        gc.collect()

    # ---- TERABYTE (D=64) ----
    if os.path.exists(TERABYTE_MODEL):
        log("\n" + "#" * 70)
        log("# TERABYTE DATASET (D=64)")
        log("#" * 70)

        tables_t, large_t, D_t = load_model_weights(TERABYTE_MODEL, "Terabyte")

        r3_t = experiment3_intrinsic(large_t, D_t, "Terabyte")
        all_results['terabyte_exp3'] = r3_t

        r2_t = experiment2_smoothness(large_t, D_t, "Terabyte")
        all_results['terabyte_exp2'] = r2_t

        r4_t = experiment4_random(large_t, D_t, "Terabyte")
        all_results['terabyte_exp4'] = r4_t

        r1_t = experiment1_crf_reorder(large_t, D_t, "Terabyte")
        all_results['terabyte_exp1'] = r1_t

        # Experiment 1b for Terabyte (with AUC)
        large_t_indices = sorted(large_t.keys())
        r1b_t, freq_t = experiment1b_auc_impact(TERABYTE_MODEL, 'terabyte', D_t, large_t_indices)
        all_results['terabyte_exp1b_auc'] = r1b_t

        del tables_t, large_t
        gc.collect()

    # ---- SAVE ALL RESULTS ----
    results_path = os.path.join(RESULTS_DIR, "all_experiments.json")

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    log(f"\nAll results saved to {results_path}")

    # ---- GENERATE SUMMARY ----
    generate_summary(all_results)

    log("\n" + "=" * 70)
    log("ALL EXPERIMENTS COMPLETE")
    log("=" * 70)
    log_fh.close()

# ============================================================
# SUMMARY GENERATION
# ============================================================
def generate_summary(results):
    """Generate markdown summary of all experiments."""
    summary_path = os.path.join(RESULTS_DIR, "intrinsic_compressibility.md")
    with open(summary_path, 'w') as f:
        f.write("# Intrinsic Compressibility of Embedding Tables\n\n")
        f.write("## Key Question\n")
        f.write("Why does lossy H.265 compression (CRF=18) achieve 225x compression with <0.01% AUC loss?\n")
        f.write("Is it due to row reordering (frequency sorting), or is it intrinsic to the embedding values?\n\n")

        # Experiment 3: Intrinsic properties
        for dataset in ['kaggle', 'terabyte']:
            key = f'{dataset}_exp3'
            if key not in results:
                continue
            exp3 = results[key]
            D_label = "D=16" if dataset == "kaggle" else "D=64"
            f.write(f"## Experiment 3: Intrinsic Properties ({dataset.title()}, {D_label})\n\n")
            f.write("| Table | Rows | Entropy (bits/byte) | Effective Rank | Rank for 99% var | Top-1 Var% | uint8 std | Near-zero rows |\n")
            f.write("|-------|------|--------------------|--------------|--------------------|-----------|-----------|----------------|\n")
            for t_idx in sorted(exp3.keys(), key=lambda x: int(x)):
                t = exp3[t_idx]
                f.write(f"| {t_idx} | {t['n_rows']:,} | {t['entropy_bits_per_byte']:.2f} | "
                        f"{t['effective_rank']:.1f}/{t['D']} | {t['svd_rank_99']}/{t['D']} | "
                        f"{t['top1_variance_frac']*100:.1f}% | {t['uint8_std']:.1f} | "
                        f"{t['near_zero_row_frac']*100:.1f}% |\n")
            f.write("\n")

        # Experiment 2: Smoothness
        for dataset in ['kaggle', 'terabyte']:
            key = f'{dataset}_exp2'
            if key not in results:
                continue
            exp2 = results[key]
            D_label = "D=16" if dataset == "kaggle" else "D=64"
            f.write(f"## Experiment 2: Spatial Smoothness ({dataset.title()}, {D_label})\n\n")
            f.write("| Table | Natural TV | Random TV | Freq-sorted TV | Rev-freq TV | Natural row-L2 | Random row-L2 | Freq row-L2 |\n")
            f.write("|-------|-----------|-----------|---------------|-------------|---------------|--------------|-------------|\n")
            for t_idx in sorted(exp2.keys(), key=lambda x: int(x)):
                t = exp2[t_idx]
                f.write(f"| {t_idx} "
                        f"| {t['natural']['total_variation']:.2f} "
                        f"| {t['random']['total_variation']:.2f} "
                        f"| {t['frequency']['total_variation']:.2f} "
                        f"| {t['reverse_frequency']['total_variation']:.2f} "
                        f"| {t['natural']['row_to_row_l2']:.2f} "
                        f"| {t['random']['row_to_row_l2']:.2f} "
                        f"| {t['frequency']['row_to_row_l2']:.2f} |\n")
            f.write("\n")

        # Experiment 1: CRF x Reordering
        for dataset in ['kaggle', 'terabyte']:
            key = f'{dataset}_exp1'
            if key not in results:
                continue
            exp1 = results[key]
            D_label = "D=16" if dataset == "kaggle" else "D=64"
            f.write(f"## Experiment 1: CRF x Reordering ({dataset.title()}, {D_label})\n\n")

            # Pick one representative table
            first_table = sorted(exp1.keys(), key=lambda x: int(x))[0]
            f.write(f"### Representative table: {first_table}\n\n")
            f.write("| CRF | Natural ratio | Random ratio | Freq ratio | Natural MSE | Random MSE | Freq MSE |\n")
            f.write("|-----|--------------|-------------|-----------|------------|-----------|----------|\n")
            for crf in CRF_VALUES:
                t = exp1[first_table]
                f.write(f"| {crf} "
                        f"| {t['natural'][crf]['ratio_vs_uint8']:.1f}x "
                        f"| {t['random'][crf]['ratio_vs_uint8']:.1f}x "
                        f"| {t['frequency'][crf]['ratio_vs_uint8']:.1f}x "
                        f"| {t['natural'][crf]['mse_uint8']:.2f} "
                        f"| {t['random'][crf]['mse_uint8']:.2f} "
                        f"| {t['frequency'][crf]['mse_uint8']:.2f} |\n")
            f.write("\n")

        # Experiment 4: Random baseline
        for dataset in ['kaggle', 'terabyte']:
            key = f'{dataset}_exp4'
            if key not in results:
                continue
            exp4 = results[key]
            D_label = "D=16" if dataset == "kaggle" else "D=64"
            f.write(f"## Experiment 4: Real vs Random Data ({dataset.title()}, {D_label})\n\n")
            first_table = sorted(exp4.keys(), key=lambda x: int(x))[0]
            f.write(f"### Table {first_table}, CRF=18\n\n")
            if 18 in exp4[first_table]:
                crf_data = exp4[first_table][18]
                f.write("| Data Type | Compressed Size | Ratio |\n")
                f.write("|-----------|----------------|-------|\n")
                for name in ['real', 'uniform_random', 'gaussian_random', 'row_shuffled', 'col_shuffled']:
                    if name in crf_data:
                        d = crf_data[name]
                        f.write(f"| {name} | {d['bytes']/1024:.0f} KB | {d['ratio']:.1f}x |\n")
            f.write("\n")

        # Experiment 1b: AUC impact
        for dataset in ['kaggle', 'terabyte']:
            key = f'{dataset}_exp1b_auc'
            if key not in results:
                continue
            exp1b = results[key]
            D_label = "D=16" if dataset == "kaggle" else "D=64"
            f.write(f"## Experiment 1b: AUC Impact of Lossy Compression ({dataset.title()}, {D_label})\n\n")
            f.write(f"Baseline AUC: {exp1b['baseline_auc']:.6f}\n\n")
            f.write("| CRF | Natural AUC | Natural delta | Random AUC | Random delta | Freq AUC | Freq delta |\n")
            f.write("|-----|-----------|--------------|----------|-------------|---------|------------|\n")
            for crf in [0, 10, 18, 23, 28]:
                row = f"| {crf} "
                for order in ['natural', 'random', 'frequency']:
                    k = f"crf{crf}_{order}"
                    if k in exp1b:
                        row += f"| {exp1b[k]['auc']:.6f} | {exp1b[k]['delta']:+.6f} "
                    else:
                        row += "| — | — "
                row += "|\n"
                f.write(row)
            f.write("\n")

        # Conclusions
        f.write("## Conclusions\n\n")
        f.write("(To be filled after reviewing results)\n")

    log(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
