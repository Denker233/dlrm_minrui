#!/usr/bin/env python3
"""
Cold Multi-Frame Granularity Experiment
Tests different frame sizes for cold embeddings with multi-frame encoding.
Hot embeddings: single-frame at CRF 0 or CRF 5.
Small tables: single-frame at hot CRF level.
84 combinations total.
"""

import os, sys, time, json, tempfile, subprocess
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

MIN_WIDTH = 64
MIN_HEIGHT = 64
MAX_DIM = 16384
TILING_THRESHOLD = 50000
TILE_SIZE = 4

HOT_THRESHOLD = 0.80
PROFILE_BATCHES = 200
LARGE_TABLE_THRESHOLD = 50000

HOT_CRFS = [0, 5]

COLD_FRAME_SIZES = [
    ("16x16", 16, 16),
    ("64x64", 64, 64),
    ("256x256", 256, 256),
    ("512x512", 512, 512),
    ("1920x1080", 1920, 1080),
    ("3840x2160", 3840, 2160),
    ("single-frame", 0, 0),
]

COLD_CRFS = [0, 23, 35]
KEYINT_MODES = [("I-only", 1), ("inter", 250)]


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
    a.arch_mlp_bot = ARCH_MLP_BOT; a.arch_mlp_top = ARCH_MLP_TOP
    a.arch_interaction_op = "dot"; a.arch_interaction_itself = False
    a.data_generation = "dataset"; a.data_set = "kaggle"
    a.raw_data_file = DATA_FILE; a.processed_data_file = PROCESSED_DATA
    a.loss_function = "bce"; a.max_ind_range = -1
    a.test_mini_batch_size = TEST_BATCH_SIZE; a.test_num_workers = 0; a.num_workers = 0
    a.mlperf_logging = False; a.memory_map = False; a.data_randomize = "total"
    a.data_trace_enable_padding = False; a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10; a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 128; a.round_targets = True
    a.mlperf_bin_loader = False; a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False
    return a


def load_model_and_data():
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
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")
    ld = torch.load(MODEL_PATH, map_location='cpu')
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    return dlrm, test_ld, train_ld


def run_inference(dlrm, test_ld):
    scores, targets = [], []
    accu, samp = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for batch in test_ld:
            X, lS_o, lS_i, T = batch
            Z = dlrm(X, lS_o, lS_i)
            S = Z.detach().cpu().numpy().flatten()
            Tn = T.detach().cpu().numpy().flatten()
            accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
            samp += Tn.shape[0]
            scores.extend(S.tolist()); targets.extend(Tn.tolist())
    return accu / samp, roc_auc_score(targets, scores), time.time() - t0


def quantize(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def dequantize(q, s, zp):
    return (q.float() - zp) * s


def get_codec_args(crf, keyint=None):
    ki = f":keyint={keyint}" if keyint else ""
    if crf == 0:
        return ['-c:v', 'libx265', '-preset', 'ultrafast',
                '-x265-params', f'lossless=1:log-level=error{ki}']
    else:
        return ['-c:v', 'libx265', '-crf', str(crf), '-preset', 'ultrafast',
                '-x265-params', f'log-level=error:allow-non-conformance=1{ki}']


def prepare_single_frame(pixels_np, num_emb, emb_dim):
    """Prepare single-frame buffer with tiling/padding (same as Phase 2A)."""
    w, h = emb_dim, num_emb
    tiling_meta = {'tiled': False}
    if num_emb > TILING_THRESHOLD:
        img, gs, tpe = tile_embeddings(pixels_np.reshape(-1), emb_dim, num_emb, TILE_SIZE)
        w, h = img.shape[1], img.shape[0]
        raw = img.tobytes()
        tiling_meta = {'tiled': True, 'grid_size': gs, 'tiles_per_emb': tpe, 'tile_size': TILE_SIZE}
    else:
        raw = pixels_np.tobytes()
    tp = w * h
    if h > MAX_DIM or w < MIN_WIDTH or h < MIN_HEIGHT:
        mr = MIN_WIDTH * MIN_HEIGHT
        if tp < mr: w, h = MIN_WIDTH, MIN_HEIGHT
        elif h > MAX_DIM:
            w = (tp + MAX_DIM - 1) // MAX_DIM; h = MAX_DIM
            if w < MIN_WIDTH: w = MIN_WIDTH; h = (tp + w - 1) // w
        elif w < MIN_WIDTH:
            w = MIN_WIDTH; h = (tp + w - 1) // w
            if h < MIN_HEIGHT: h = MIN_HEIGHT
        elif h < MIN_HEIGHT:
            h = MIN_HEIGHT; w = (tp + h - 1) // h
            if w < MIN_WIDTH: w, h = MIN_WIDTH, MIN_HEIGHT
        pp = w * h
        if pp > len(raw):
            p = bytearray(pp); p[:len(raw)] = raw; raw = bytes(p)
    return raw, w, h, tiling_meta


def encode_single_frame(raw, w, h, codec_args):
    with tempfile.TemporaryDirectory() as d:
        rf = os.path.join(d, 'i.raw'); vf = os.path.join(d, 'o.mp4')
        with open(rf, 'wb') as f: f.write(raw)
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{w}x{h}', '-r', '1', '-i', rf] + codec_args + ['-frames:v', '1', vf]
        t0 = time.time()
        subprocess.run(cmd, capture_output=True, check=True)
        ct = time.time() - t0
        with open(vf, 'rb') as f: data = f.read()
    return data, ct


def decode_single_frame(comp, w, h, num_emb, emb_dim, tiling_meta):
    with tempfile.TemporaryDirectory() as d:
        vf = os.path.join(d, 'i.mp4'); rf = os.path.join(d, 'o.raw')
        with open(vf, 'wb') as f: f.write(comp)
        subprocess.run(['ffmpeg', '-y', '-i', vf, '-pix_fmt', 'gray', '-f', 'rawvideo', rf],
                       capture_output=True, check=True)
        px = np.fromfile(rf, dtype=np.uint8)
    if tiling_meta.get('tiled'):
        gs, ts = tiling_meta['grid_size'], tiling_meta['tile_size']
        isz = gs * ts
        px = untile_embeddings(px[:isz*isz].reshape(isz, isz),
                               emb_dim, num_emb, gs, tiling_meta['tiles_per_emb'], ts)
    else:
        px = px[:num_emb * emb_dim]
    return torch.from_numpy(px.copy()).reshape(num_emb, emb_dim)


def encode_multiframe(pixels_flat, frame_w, frame_h, codec_args):
    """Encode flat uint8 array as multi-frame video."""
    frame_size = frame_w * frame_h
    n_frames = (len(pixels_flat) + frame_size - 1) // frame_size
    padded = np.zeros(n_frames * frame_size, dtype=np.uint8)
    padded[:len(pixels_flat)] = pixels_flat
    with tempfile.TemporaryDirectory() as d:
        rf = os.path.join(d, 'i.raw'); vf = os.path.join(d, 'o.mp4')
        with open(rf, 'wb') as f: f.write(padded.tobytes())
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{frame_w}x{frame_h}', '-r', '30',
               '-i', rf] + codec_args + [vf]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ct = time.time() - t0
        if r.returncode != 0:
            raise RuntimeError(f"Encode fail: {r.stderr[:300]}")
        with open(vf, 'rb') as f: data = f.read()
    return data, ct, n_frames


def decode_multiframe(comp, num_pixels):
    """Decode multi-frame video back to flat uint8 array."""
    with tempfile.TemporaryDirectory() as d:
        vf = os.path.join(d, 'i.mp4'); rf = os.path.join(d, 'o.raw')
        with open(vf, 'wb') as f: f.write(comp)
        t0 = time.time()
        subprocess.run(['ffmpeg', '-y', '-i', vf, '-pix_fmt', 'gray', '-f', 'rawvideo', rf],
                       capture_output=True, check=True)
        dt = time.time() - t0
        px = np.fromfile(rf, dtype=np.uint8)
    return px[:num_pixels], dt


def profile_access(train_ld, num_tables, n_batches=PROFILE_BATCHES):
    """Profile which embedding rows are accessed."""
    access_counts = [None] * num_tables
    for i, batch in enumerate(train_ld):
        if i >= n_batches: break
        _, _, lS_i, _ = batch
        for t in range(num_tables):
            indices = lS_i[t].numpy().flatten()
            if access_counts[t] is None:
                access_counts[t] = np.zeros(0, dtype=np.int64)
            access_counts[t] = np.concatenate([access_counts[t], indices])
        if i % 40 == 0: print(f"    Batch {i}/{n_batches}", end='\r')
    print(f"    Profiled {n_batches} batches")

    hot_indices = {}
    for t in range(num_tables):
        if access_counts[t] is None or len(access_counts[t]) == 0:
            hot_indices[t] = np.array([], dtype=np.int64)
            continue
        unique, counts = np.unique(access_counts[t], return_counts=True)
        sorted_idx = np.argsort(-counts)
        cum = np.cumsum(counts[sorted_idx])
        total = cum[-1]
        cutoff = np.searchsorted(cum, total * HOT_THRESHOLD) + 1
        hot_indices[t] = unique[sorted_idx[:cutoff]]
    return hot_indices


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 80)
    print("COLD MULTI-FRAME GRANULARITY EXPERIMENT")
    print("84 combinations: 2 hot CRFs × 7 granularities × 3 cold CRFs × 2 modes")
    print("=" * 80)

    print("\nLoading model and data...")
    dlrm, test_ld, train_ld = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]
    num_tables = len(emb_keys)
    total_fp32 = sum(state_dict[k].numel() * 4 for k in emb_keys)

    print(f"  Tables: {num_tables}, Total FP32: {total_fp32/1024/1024:.2f} MB")

    # Baseline
    print("\n  Running baseline inference...")
    drop_caches()
    bl_acc, bl_auc, bl_infer = run_inference(dlrm, test_ld)
    print(f"  BASELINE: acc={bl_acc*100:.4f}%, AUC={bl_auc:.6f}, infer={bl_infer:.2f}s")

    # ================================================================
    # ACCESS PROFILING
    # ================================================================
    print("\n" + "=" * 60)
    print("ACCESS PROFILING")
    print("=" * 60)
    hot_indices = profile_access(train_ld, num_tables)

    large_tables = []
    small_tables = []
    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        if w.shape[0] >= LARGE_TABLE_THRESHOLD:
            large_tables.append(t)
        else:
            small_tables.append(t)
    print(f"  Large tables: {len(large_tables)}, Small tables: {len(small_tables)}")

    # Split weights into hot/cold for large tables
    table_splits = {}
    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = hot_indices[t]
        all_idx = np.arange(w.shape[0])
        cold_mask = np.ones(w.shape[0], dtype=bool)
        cold_mask[hi] = False
        cold_idx = all_idx[cold_mask]
        table_splits[t] = {
            'hot_idx': hi, 'cold_idx': cold_idx,
            'hot_w': w[hi], 'cold_w': w[cold_idx],
        }
        print(f"  Table {t}: {w.shape[0]} rows -> hot={len(hi)}, cold={len(cold_idx)}")

    # ================================================================
    # PRE-COMPRESS HOT + SMALL at CRF 0 and CRF 5
    # ================================================================
    print("\n" + "=" * 60)
    print("PRE-COMPRESSING HOT + SMALL EMBEDDINGS")
    print("=" * 60)

    # Cache: hot_cache[crf] = {table: decoded_fp32_weights}
    hot_cache = {}
    hot_sizes = {}  # hot_sizes[crf] = total bytes
    small_cache = {}
    small_sizes = {}

    for hcrf in HOT_CRFS:
        print(f"\n  Hot CRF {hcrf}:")
        hot_cache[hcrf] = {}
        hot_sizes[hcrf] = 0

        codec_args = get_codec_args(hcrf)
        for t in large_tables:
            hw = table_splits[t]['hot_w']
            q, s, zp = quantize(hw)
            raw, w, h, tm = prepare_single_frame(q.numpy(), hw.shape[0], hw.shape[1])
            comp, ct = encode_single_frame(raw, w, h, codec_args)
            dec_uint8 = decode_single_frame(comp, w, h, hw.shape[0], hw.shape[1], tm)
            hot_cache[hcrf][t] = dequantize(dec_uint8, s, zp)
            hot_sizes[hcrf] += len(comp)

        print(f"    Hot total: {hot_sizes[hcrf]:,} bytes ({hot_sizes[hcrf]/1024/1024:.2f} MB)")

        small_cache[hcrf] = {}
        small_sizes[hcrf] = 0
        for t in small_tables:
            sw = state_dict[emb_keys[t]]
            q, s, zp = quantize(sw)
            raw, w, h, tm = prepare_single_frame(q.numpy(), sw.shape[0], sw.shape[1])
            comp, ct = encode_single_frame(raw, w, h, codec_args)
            dec_uint8 = decode_single_frame(comp, w, h, sw.shape[0], sw.shape[1], tm)
            small_cache[hcrf][t] = dequantize(dec_uint8, s, zp)
            small_sizes[hcrf] += len(comp)

        print(f"    Small total: {small_sizes[hcrf]:,} bytes ({small_sizes[hcrf]/1024/1024:.2f} MB)")

    # ================================================================
    # RUN 84 COMBINATIONS
    # ================================================================
    print("\n" + "=" * 60)
    print("RUNNING 84 COMBINATIONS")
    print("=" * 60)

    all_results = []
    combo_num = 0
    total_combos = len(HOT_CRFS) * len(COLD_FRAME_SIZES) * len(COLD_CRFS) * len(KEYINT_MODES)

    for fs_name, fw, fh in COLD_FRAME_SIZES:
        for ccrf in COLD_CRFS:
            for mode_name, keyint in KEYINT_MODES:
                # Compress cold embeddings for this config
                cold_compressed = {}
                cold_decoded = {}
                total_cold_bytes = 0
                total_cold_ct = 0
                total_cold_dt = 0
                total_cold_frames = 0

                for t in large_tables:
                    cw = table_splits[t]['cold_w']
                    q, s, zp = quantize(cw)
                    num_emb, emb_dim = cw.shape

                    if fs_name == "single-frame":
                        # Use existing single-frame approach
                        codec_args = get_codec_args(ccrf)
                        raw, w, h, tm = prepare_single_frame(q.numpy(), num_emb, emb_dim)
                        comp, ct = encode_single_frame(raw, w, h, codec_args)
                        dec_uint8 = decode_single_frame(comp, w, h, num_emb, emb_dim, tm)
                        n_frames = 1
                        dt = 0  # included in decode_single_frame but not timed separately
                    else:
                        # Multi-frame encoding
                        codec_args = get_codec_args(ccrf, keyint)
                        pixels_flat = q.numpy().reshape(-1)
                        comp, ct, n_frames = encode_multiframe(pixels_flat, fw, fh, codec_args)
                        dec_flat, dt = decode_multiframe(comp, num_emb * emb_dim)
                        dec_uint8 = torch.from_numpy(dec_flat.copy()).reshape(num_emb, emb_dim)
                        total_cold_dt += dt

                    cold_compressed[t] = comp
                    cold_decoded[t] = dequantize(dec_uint8, s, zp)
                    total_cold_bytes += len(comp)
                    total_cold_ct += ct
                    total_cold_frames += n_frames

                # Now run inference with each hot CRF
                for hcrf in HOT_CRFS:
                    combo_num += 1
                    label = f"[{combo_num}/{total_combos}] H{hcrf}/Cold_{fs_name}_CRF{ccrf}_{mode_name}"
                    print(f"\n  {label}")

                    # Set weights
                    with torch.no_grad():
                        for t in large_tables:
                            hi = table_splits[t]['hot_idx']
                            ci = table_splits[t]['cold_idx']
                            dlrm.emb_l[t].weight.data[hi] = hot_cache[hcrf][t]
                            dlrm.emb_l[t].weight.data[ci] = cold_decoded[t]
                        for t in small_tables:
                            dlrm.emb_l[t].weight.data = small_cache[hcrf][t]

                    drop_caches()
                    acc, auc, infer_t = run_inference(dlrm, test_ld)
                    auc_loss = bl_auc - auc

                    total_size = hot_sizes[hcrf] + total_cold_bytes + small_sizes[hcrf]
                    ratio = total_fp32 / total_size if total_size > 0 else 0

                    result = {
                        'hot_crf': hcrf,
                        'cold_frame_size': fs_name,
                        'cold_crf': ccrf,
                        'mode': mode_name,
                        'keyint': keyint,
                        'cold_frames': total_cold_frames,
                        'hot_mb': hot_sizes[hcrf] / 1024 / 1024,
                        'cold_mb': total_cold_bytes / 1024 / 1024,
                        'small_mb': small_sizes[hcrf] / 1024 / 1024,
                        'total_mb': total_size / 1024 / 1024,
                        'total_bytes': total_size,
                        'ratio': ratio,
                        'accuracy': acc,
                        'accuracy_pct': acc * 100,
                        'auc': auc,
                        'auc_loss': auc_loss,
                        'auc_loss_pp': auc_loss * 100,
                        'cold_compress_time': total_cold_ct,
                        'cold_decompress_time': total_cold_dt,
                        'inference_time': infer_t,
                    }
                    all_results.append(result)

                    print(f"    {total_size/1024/1024:.2f} MB ({ratio:.1f}x), "
                          f"AUC={auc:.6f} (loss {auc_loss*100:.4f}pp), "
                          f"cold_frames={total_cold_frames}, infer={infer_t:.2f}s")

                    # Restore weights
                    with torch.no_grad():
                        for t in range(num_tables):
                            dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

    # ================================================================
    # WRITE RESULTS
    # ================================================================
    print("\n" + "=" * 60)
    print("WRITING RESULTS")
    print("=" * 60)

    out_path = os.path.join(RESULTS_DIR, "cold_multiframe_granularity_results.md")
    with open(out_path, 'w') as f:
        f.write("# Cold Multi-Frame Granularity Experiment\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- **Model:** `{MODEL_PATH}`\n")
        f.write(f"- **Total FP32:** {total_fp32:,} bytes ({total_fp32/1024/1024:.2f} MB)\n")
        f.write(f"- **Baseline AUC:** {bl_auc:.6f}, Accuracy: {bl_acc*100:.4f}%\n")
        f.write(f"- **Hot threshold:** {HOT_THRESHOLD*100:.0f}% of accesses\n")
        f.write(f"- **Large tables:** {len(large_tables)}/26, Small tables: {len(small_tables)}/26\n")
        f.write(f"- **Hot CRF levels:** {HOT_CRFS}\n")
        f.write(f"- **Cold frame sizes:** {[n for n,_,_ in COLD_FRAME_SIZES]}\n")
        f.write(f"- **Cold CRF levels:** {COLD_CRFS}\n")
        f.write(f"- **Modes:** I-only (keyint=1), inter (keyint=250)\n\n")

        f.write("## Pre-computed Hot/Small Sizes\n\n")
        f.write("| Hot CRF | Hot (MB) | Small (MB) | Hot+Small (MB) |\n")
        f.write("|---------|---------|-----------|----------------|\n")
        for hcrf in HOT_CRFS:
            hs = hot_sizes[hcrf] / 1024 / 1024
            ss = small_sizes[hcrf] / 1024 / 1024
            f.write(f"| {hcrf} | {hs:.2f} | {ss:.2f} | {hs+ss:.2f} |\n")

        f.write("\n## Full Results (84 Combinations)\n\n")
        f.write("| # | Hot CRF | Cold Frame | Cold CRF | Mode | Cold Frames | "
                "Hot (MB) | Cold (MB) | Small (MB) | Total (MB) | Ratio | "
                "Acc (%) | AUC | AUC Loss (pp) | Cold CT (s) | Cold DT (s) | Infer (s) |\n")
        f.write("|---|---------|-----------|----------|------|------------|"
                "---------|----------|-----------|-----------|-------|"
                "--------|------|--------------|------------|------------|----------|\n")
        for i, r in enumerate(all_results):
            f.write(f"| {i+1} | {r['hot_crf']} | {r['cold_frame_size']} | {r['cold_crf']} | "
                    f"{r['mode']} | {r['cold_frames']:,} | "
                    f"{r['hot_mb']:.2f} | {r['cold_mb']:.2f} | {r['small_mb']:.2f} | "
                    f"{r['total_mb']:.2f} | {r['ratio']:.1f}x | "
                    f"{r['accuracy_pct']:.4f} | {r['auc']:.6f} | {r['auc_loss_pp']:.4f} | "
                    f"{r['cold_compress_time']:.1f} | {r['cold_decompress_time']:.1f} | "
                    f"{r['inference_time']:.1f} |\n")

        # Summary by frame size (best cold CRF per granularity)
        f.write("\n## Summary: Best AUC per Frame Granularity (Hot CRF 0)\n\n")
        f.write("| Frame Size | Emb/Frame | Best Cold CRF | Mode | Ratio | AUC Loss (pp) | Total (MB) |\n")
        f.write("|-----------|----------|--------------|------|-------|--------------|------------|\n")
        for fs_name, fw, fh in COLD_FRAME_SIZES:
            matching = [r for r in all_results
                        if r['hot_crf'] == 0 and r['cold_frame_size'] == fs_name]
            if matching:
                best = min(matching, key=lambda r: r['auc_loss_pp'])
                epf = (fw * fh) // 16 if fw > 0 else "all"
                f.write(f"| {fs_name} | {epf} | {best['cold_crf']} | {best['mode']} | "
                        f"{best['ratio']:.1f}x | {best['auc_loss_pp']:.4f} | "
                        f"{best['total_mb']:.2f} |\n")

        # Inter-frame vs I-only comparison
        f.write("\n## Inter-Frame vs I-Only Comparison (Hot CRF 0, Cold CRF 23)\n\n")
        f.write("| Frame Size | I-Only Ratio | I-Only AUC Loss | Inter Ratio | Inter AUC Loss | "
                "Inter Better? |\n")
        f.write("|-----------|-------------|----------------|------------|----------------|"
                "---------------|\n")
        for fs_name, fw, fh in COLD_FRAME_SIZES:
            io = next((r for r in all_results if r['hot_crf'] == 0 and r['cold_frame_size'] == fs_name
                        and r['cold_crf'] == 23 and r['mode'] == 'I-only'), None)
            inter = next((r for r in all_results if r['hot_crf'] == 0 and r['cold_frame_size'] == fs_name
                           and r['cold_crf'] == 23 and r['mode'] == 'inter'), None)
            if io and inter:
                better = "Yes" if inter['ratio'] > io['ratio'] and inter['auc_loss_pp'] <= io['auc_loss_pp'] * 1.1 else "No"
                f.write(f"| {fs_name} | {io['ratio']:.1f}x | {io['auc_loss_pp']:.4f} | "
                        f"{inter['ratio']:.1f}x | {inter['auc_loss_pp']:.4f} | {better} |\n")

    print(f"\nResults written to: {out_path}")

    json_path = os.path.join(RESULTS_DIR, "cold_multiframe_granularity_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            'baseline': {'accuracy': bl_acc, 'auc': bl_auc, 'inference_time': bl_infer,
                         'total_fp32': total_fp32},
            'hot_sizes': {str(k): v for k, v in hot_sizes.items()},
            'small_sizes': {str(k): v for k, v in small_sizes.items()},
            'results': all_results,
        }, f, indent=2)
    print(f"JSON saved to: {json_path}")

    print("\n" + "=" * 80)
    print("COLD MULTI-FRAME GRANULARITY EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
