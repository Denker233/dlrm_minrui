#!/usr/bin/env python3
"""
DEMO: Video Codec Embedding Table Compression
Shows real-time compression of DLRM embedding tables using H.265
at different frame sizes and quality levels.

Usage:
    python3 demo_codec_compression.py
    python3 demo_codec_compression.py --crf 0         # lossless only
    python3 demo_codec_compression.py --crf 23        # single CRF
    python3 demo_codec_compression.py --skip-baseline  # skip baseline inference
"""

import os, sys, time, argparse, tempfile, subprocess, shutil
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlrm_data_pytorch as dp
from dlrm_s_pytorch import DLRM_Net, tile_embeddings, untile_embeddings

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
DATA_FILE = "./input/train.txt"

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 16384

MIN_WIDTH = 64
MIN_HEIGHT = 64
MAX_DIM = 16384
TILING_THRESHOLD = 50000
TILE_SIZE = 4

# ── Pretty printing ────────────────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"
CLEAR_LINE = "\r\033[K"

def header(text, width=72):
    print()
    print(f"{BOLD}{CYAN}{'=' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * width}{RESET}")

def subheader(text):
    print(f"\n{BOLD}  >> {text}{RESET}")

def status(text):
    print(f"{DIM}     {text}{RESET}")

def progress(text):
    print(f"{CLEAR_LINE}     {text}", end='', flush=True)

def result_line(label, value, color=GREEN):
    print(f"     {label}: {color}{value}{RESET}")

def fmt_size(mb):
    if mb >= 1000:
        return f"{mb:,.0f} MB"
    elif mb >= 1:
        return f"{mb:.2f} MB"
    else:
        return f"{mb*1024:.0f} KB"

def fmt_ratio(ratio):
    if ratio < 1:
        return f"{ratio:.2f}x (expansion!)"
    elif ratio >= 1000:
        return f"{ratio:,.0f}x"
    else:
        return f"{ratio:.1f}x"

def fmt_auc_loss(loss_pp):
    if abs(loss_pp) < 0.001:
        return f"{loss_pp:+.4f}pp (negligible)"
    elif abs(loss_pp) < 0.1:
        return f"{loss_pp:+.4f}pp"
    else:
        return f"{loss_pp:+.2f}pp"


# ── Model / Data ───────────────────────────────────────────────────────────
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
    return dlrm, test_ld


def run_inference(dlrm, test_ld, label="Inference"):
    scores, targets = [], []
    accu, samp = 0, 0
    total_batches = len(test_ld)
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(test_ld):
            X, lS_o, lS_i, T = batch
            Z = dlrm(X, lS_o, lS_i)
            S = Z.detach().cpu().numpy().flatten()
            Tn = T.detach().cpu().numpy().flatten()
            accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
            samp += Tn.shape[0]
            scores.extend(S.tolist()); targets.extend(Tn.tolist())
            if (i + 1) % 200 == 0 or i == total_batches - 1:
                elapsed = time.time() - t0
                progress(f"{label}... batch {i+1}/{total_batches} "
                         f"({elapsed:.1f}s elapsed)")
    elapsed = time.time() - t0
    print(f"{CLEAR_LINE}", end='')
    return accu / samp, roc_auc_score(targets, scores), elapsed


# ── Quantization ───────────────────────────────────────────────────────────
def quantize(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def dequantize(q, s, zp):
    return (q.float() - zp) * s


# ── H.265 Encoding / Decoding ─────────────────────────────────────────────
def get_codec_args(crf, keyint=None):
    ki = f":keyint={keyint}" if keyint else ""
    if crf == 0:
        return ['-c:v', 'libx265', '-preset', 'ultrafast',
                '-x265-params', f'lossless=1:log-level=error{ki}']
    else:
        return ['-c:v', 'libx265', '-crf', str(crf), '-preset', 'ultrafast',
                '-x265-params', f'log-level=error:allow-non-conformance=1{ki}']


def prepare_single_frame(pixels_np, num_emb, emb_dim):
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
        subprocess.run(cmd, capture_output=True, check=True)
        with open(vf, 'rb') as f: data = f.read()
    return data


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
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"Encode failed: {r.stderr[:300]}")
        with open(vf, 'rb') as f: data = f.read()
    return data, n_frames


def decode_multiframe(comp, num_pixels):
    with tempfile.TemporaryDirectory() as d:
        vf = os.path.join(d, 'i.mp4'); rf = os.path.join(d, 'o.raw')
        with open(vf, 'wb') as f: f.write(comp)
        subprocess.run(['ffmpeg', '-y', '-i', vf, '-pix_fmt', 'gray', '-f', 'rawvideo', rf],
                       capture_output=True, check=True)
        px = np.fromfile(rf, dtype=np.uint8)
    return px[:num_pixels]


# ── Compress all tables with a given config ────────────────────────────────
def compress_all_tables(emb_keys, state_dict, frame_name, frame_w, frame_h,
                        crf, keyint):
    """Compress all 26 embedding tables. Returns total compressed bytes,
    encode time, decode time, list of decoded fp32 weights, and frame count."""
    codec_args = get_codec_args(crf, keyint if frame_name != "single-frame" else None)
    num_tables = len(emb_keys)
    total_bytes = 0
    total_encode = 0
    total_decode = 0
    total_frames = 0
    decoded_weights = []

    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        num_emb, emb_dim = w.shape
        q, s, zp = quantize(w)

        progress(f"Encoding table {t+1}/{num_tables} "
                 f"({num_emb:,} rows, {frame_name})...")

        t0 = time.time()
        if frame_name == "single-frame":
            raw, fw, fh, tm = prepare_single_frame(q.numpy(), num_emb, emb_dim)
            comp = encode_single_frame(raw, fw, fh, codec_args)
            n_frames = 1
        else:
            pixels_flat = q.numpy().reshape(-1)
            comp, n_frames = encode_multiframe(pixels_flat, frame_w, frame_h, codec_args)
        encode_time = time.time() - t0

        t0 = time.time()
        if frame_name == "single-frame":
            dec_uint8 = decode_single_frame(comp, fw, fh, num_emb, emb_dim, tm)
        else:
            dec_flat = decode_multiframe(comp, num_emb * emb_dim)
            dec_uint8 = torch.from_numpy(dec_flat.copy()).reshape(num_emb, emb_dim)
        decode_time = time.time() - t0

        decoded_weights.append(dequantize(dec_uint8, s, zp))
        total_bytes += len(comp)
        total_encode += encode_time
        total_decode += decode_time
        total_frames += n_frames

    print(f"{CLEAR_LINE}", end='')
    return total_bytes, total_encode, total_decode, decoded_weights, total_frames


# ── Main demo ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Demo: Video Codec Embedding Compression")
    parser.add_argument('--crf', type=int, nargs='+', default=[0, 23],
                        help='CRF levels to test (default: 0 23)')
    parser.add_argument('--frames', type=str, nargs='+',
                        default=['single-frame', '3840x2160', '1920x1080', '512x512'],
                        help='Frame sizes to test')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline inference (faster demo)')
    parser.add_argument('--skip-inference', action='store_true',
                        help='Skip all inference (show compression only)')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    frame_map = {
        '16x16': (16, 16), '64x64': (64, 64), '128x128': (128, 128),
        '256x256': (256, 256), '512x512': (512, 512),
        '1920x1080': (1920, 1080), '3840x2160': (3840, 2160),
        'single-frame': (0, 0),
    }
    frame_configs = []
    for f in args.frames:
        if f not in frame_map:
            print(f"Unknown frame size: {f}")
            print(f"Available: {', '.join(frame_map.keys())}")
            sys.exit(1)
        fw, fh = frame_map[f]
        frame_configs.append((f, fw, fh))

    total_configs = len(frame_configs) * len(args.crf)

    # ── Title ──
    header("VIDEO CODEC EMBEDDING TABLE COMPRESSION")
    print(f"     Model: DLRM (Kaggle/Criteo)")
    print(f"     Codec: H.265/HEVC")
    print(f"     Frame sizes: {', '.join(args.frames)}")
    print(f"     CRF levels: {args.crf}")
    print(f"     Configurations: {total_configs}")

    # ── Load ──
    header("LOADING MODEL & DATA")
    t0 = time.time()
    status("Loading DLRM model and Criteo test data...")
    dlrm, test_ld = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]
    num_tables = len(emb_keys)
    total_fp32 = sum(state_dict[k].numel() * 4 for k in emb_keys)
    total_uint8 = sum(state_dict[k].numel() for k in emb_keys)
    load_time = time.time() - t0

    result_line("Tables", f"{num_tables}")
    result_line("Total rows", f"{sum(state_dict[k].shape[0] for k in emb_keys):,}")
    result_line("Embedding dim", f"{ARCH_SPARSE_FEATURE_SIZE}")
    result_line("FP32 size", f"{fmt_size(total_fp32 / 1024 / 1024)}")
    result_line("UINT8 size", f"{fmt_size(total_uint8 / 1024 / 1024)}")
    result_line("Load time", f"{load_time:.1f}s")

    # ── Baseline ──
    bl_acc, bl_auc, bl_time = None, None, None
    if not args.skip_baseline and not args.skip_inference:
        header("BASELINE INFERENCE (FP32)")
        status("Running inference with original FP32 weights...")
        bl_acc, bl_auc, bl_time = run_inference(dlrm, test_ld, "Baseline")
        result_line("Accuracy", f"{bl_acc * 100:.4f}%")
        result_line("AUC", f"{bl_auc:.6f}")
        result_line("Inference", f"{bl_time:.1f}s")
        result_line("Memory", f"{fmt_size(total_fp32 / 1024 / 1024)}")
    elif args.skip_baseline and not args.skip_inference:
        # Use known baseline from prior experiments
        bl_auc = 0.802698
        bl_acc = 0.788722
        status(f"Using cached baseline: AUC={bl_auc:.6f}")

    # ── Run each config ──
    results = []
    config_num = 0

    for crf in args.crf:
        for frame_name, fw, fh in frame_configs:
            config_num += 1
            mode = "I-only"
            keyint = 1
            label = f"H.265 {frame_name} CRF {crf}"

            header(f"[{config_num}/{total_configs}] {label}")

            # Step 1: Compress
            subheader("Step 1: INT8 Quantization + H.265 Encoding")

            t_start = time.time()
            comp_bytes, enc_time, dec_time, decoded_weights, n_frames = \
                compress_all_tables(emb_keys, state_dict, frame_name, fw, fh,
                                    crf, keyint)
            comp_mb = comp_bytes / 1024 / 1024
            ratio = total_fp32 / comp_bytes if comp_bytes > 0 else 0

            result_line("Compressed size", f"{BOLD}{fmt_size(comp_mb)}{RESET}",
                        GREEN if ratio > 1 else RED)
            result_line("Compression ratio", f"{BOLD}{fmt_ratio(ratio)}{RESET}",
                        GREEN if ratio > 1 else RED)
            result_line("Total frames", f"{n_frames:,}")
            result_line("Encode time", f"{enc_time:.1f}s")
            result_line("Decode time", f"{dec_time:.1f}s")

            # Step 2: Replace weights and run inference
            acc, auc, infer_time = None, None, None
            auc_loss_pp = None

            if not args.skip_inference:
                subheader("Step 2: Replace Weights & Run Inference")

                with torch.no_grad():
                    for t in range(num_tables):
                        dlrm.emb_l[t].weight.data = decoded_weights[t]

                acc, auc, infer_time = run_inference(dlrm, test_ld, label)

                if bl_auc is not None:
                    auc_loss_pp = (bl_auc - auc) * 100

                result_line("AUC", f"{BOLD}{auc:.6f}{RESET}")
                if auc_loss_pp is not None:
                    color = GREEN if abs(auc_loss_pp) < 0.1 else (YELLOW if abs(auc_loss_pp) < 0.5 else RED)
                    result_line("AUC loss", f"{BOLD}{fmt_auc_loss(auc_loss_pp)}{RESET}", color)
                result_line("Accuracy", f"{acc * 100:.4f}%")
                result_line("Inference", f"{infer_time:.1f}s")

                # Restore original weights
                with torch.no_grad():
                    for t in range(num_tables):
                        dlrm.emb_l[t].weight.data = state_dict[emb_keys[t]].clone()

            total_time = time.time() - t_start
            result_line("Total time", f"{total_time:.1f}s")

            results.append({
                'label': label,
                'frame': frame_name,
                'crf': crf,
                'comp_mb': comp_mb,
                'ratio': ratio,
                'frames': n_frames,
                'encode_s': enc_time,
                'decode_s': dec_time,
                'auc': auc,
                'auc_loss_pp': auc_loss_pp,
                'accuracy': acc,
                'infer_s': infer_time,
                'total_s': total_time,
            })

    # ── Summary table ──
    header("SUMMARY")

    print(f"\n     {BOLD}{'Config':<35} {'Size':>10} {'Ratio':>10}", end='')
    if not args.skip_inference:
        print(f" {'AUC':>10} {'AUC Loss':>12} {'Infer':>8} {'Total':>8}", end='')
    print(f" {'Enc':>7} {'Dec':>7} {'Frames':>10}{RESET}")
    print(f"     {'─' * 35} {'─' * 10} {'─' * 10}", end='')
    if not args.skip_inference:
        print(f" {'─' * 10} {'─' * 12} {'─' * 8} {'─' * 8}", end='')
    print(f" {'─' * 7} {'─' * 7} {'─' * 10}")

    # Baseline row
    if bl_auc is not None and not args.skip_inference:
        fp32_mb = total_fp32 / 1024 / 1024
        bl_total = bl_time if bl_time is not None else 0
        print(f"     {DIM}{'Baseline (FP32)':<35}{RESET} "
              f"{fp32_mb:>9.1f}M {'1.0x':>10}"
              f" {bl_auc:>10.6f} {'0.0000pp':>12} {bl_total:>7.1f}s {bl_total:>7.1f}s"
              f" {'─':>7} {'─':>7} {'─':>10}")

    for r in results:
        comp_str = f"{r['comp_mb']:>8.2f}M" if r['comp_mb'] >= 1 else f"{r['comp_mb']*1024:>7.0f}KB"
        ratio_str = f"{r['ratio']:>9.1f}x" if r['ratio'] < 10000 else f"{r['ratio']:>9,.0f}x"

        # Color the ratio
        if r['ratio'] < 1:
            ratio_str = f"{RED}{ratio_str}{RESET}"
        elif r['ratio'] >= 100:
            ratio_str = f"{GREEN}{ratio_str}{RESET}"

        line = f"     {r['label']:<35} {comp_str} {ratio_str}"

        if not args.skip_inference and r['auc'] is not None:
            if r['auc_loss_pp'] is not None:
                if abs(r['auc_loss_pp']) < 0.01:
                    loss_color = GREEN
                elif abs(r['auc_loss_pp']) < 0.2:
                    loss_color = YELLOW
                else:
                    loss_color = RED
                loss_str = f"{r['auc_loss_pp']:+.4f}pp"
                line += f" {r['auc']:>10.6f} {loss_color}{loss_str:>12}{RESET} {r['infer_s']:>7.1f}s {r['total_s']:>7.1f}s"
            else:
                line += f" {r['auc']:>10.6f} {'N/A':>12} {r['infer_s']:>7.1f}s {r['total_s']:>7.1f}s"

        line += f" {r['encode_s']:>6.1f}s {r['decode_s']:>6.1f}s {r['frames']:>10,}"
        print(line)

    # ── Key takeaways ──
    if results and not args.skip_inference:
        print()
        lossless = [r for r in results if r['crf'] == 0 and r['auc'] is not None]
        lossy = [r for r in results if r['crf'] > 0 and r['auc'] is not None]

        if lossless:
            best_ll = max(lossless, key=lambda r: r['ratio'])
            print(f"     {BOLD}Best lossless:{RESET} {best_ll['label']} "
                  f"= {fmt_ratio(best_ll['ratio'])}, "
                  f"AUC loss {fmt_auc_loss(best_ll['auc_loss_pp'])}")

        if lossy:
            best_quality = min(lossy, key=lambda r: abs(r['auc_loss_pp']))
            best_ratio = max(lossy, key=lambda r: r['ratio'])
            print(f"     {BOLD}Best quality (lossy):{RESET} {best_quality['label']} "
                  f"= {fmt_ratio(best_quality['ratio'])}, "
                  f"AUC loss {fmt_auc_loss(best_quality['auc_loss_pp'])}")
            if best_ratio != best_quality:
                print(f"     {BOLD}Best ratio (lossy):{RESET} {best_ratio['label']} "
                      f"= {fmt_ratio(best_ratio['ratio'])}, "
                      f"AUC loss {fmt_auc_loss(best_ratio['auc_loss_pp'])}")

    print(f"\n{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"{BOLD}{CYAN}  DEMO COMPLETE{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}\n")


if __name__ == "__main__":
    main()
