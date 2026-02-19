#!/usr/bin/env python3
"""
Prefetch Benchmark V7: Improved quantization + optimized injection.

Key improvements over V6:
  1. Per-column quantization for 4-bit (each of 16 dims gets own scale/zp)
  2. 5-bit quantization (32 levels, middle ground between 4 and 6)
  3. Dense all-cold storage at 6-bit (with full reconstruction at startup)
  4. Mixed 8b warm + 6b ice
  5. Vectorized batch injection (numpy-based cold index lookup, no Python loop)
  6. All-cold per-column 4-bit (lowest memory config with good quality)
"""

import os, sys, time, json, tempfile, subprocess, io
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import av

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlrm_data_pytorch as dp

MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
DATA_FILE = "./input/train.txt"
RESULTS_DIR = os.path.expanduser("~/experiment-control")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 2048
EMB_DIM = 16

HOT_THRESHOLD = 0.80
PROFILE_BATCHES = 200
LARGE_TABLE_THRESHOLD = 50000

os.makedirs(RESULTS_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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
    from dlrm_s_pytorch import DLRM_Net
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
    return dlrm, test_ld, train_ld, ln_emb

# ======================== Quantization ========================

def quantize_8bit(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp

def quantize_4bit_percol(w_np):
    """Per-column 4-bit quantization. w_np is (N, 16) float32 numpy array.
    Returns packed (N, 8) uint8, scales (16,), zero_points (16,)."""
    n = w_np.shape[0]
    scales = np.zeros(16, dtype=np.float32)
    zps = np.zeros(16, dtype=np.float32)
    q = np.zeros((n, 16), dtype=np.uint8)
    for c in range(16):
        col = w_np[:, c]
        mn, mx = col.min(), col.max()
        s = (mx - mn) / 15.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        scales[c] = s
        zps[c] = zp
        q[:, c] = np.clip(np.round(col / s + zp), 0, 15).astype(np.uint8)
    # Pack pairs: (N, 16) -> (N, 8)
    packed = (q[:, 0::2] << 4) | q[:, 1::2]
    return packed, scales, zps

def dequantize_4bit_percol_batch(packed, scales, zps, indices):
    """Dequantize per-column 4-bit packed embeddings for a batch of indices."""
    batch = packed[indices]  # (B, 8) uint8
    hi = (batch >> 4).astype(np.float32)   # (B, 8) - even cols
    lo = (batch & 0x0F).astype(np.float32) # (B, 8) - odd cols
    unpacked = np.empty((batch.shape[0], 16), dtype=np.float32)
    unpacked[:, 0::2] = hi
    unpacked[:, 1::2] = lo
    # Per-column dequant
    unpacked = (unpacked - zps[np.newaxis, :]) * scales[np.newaxis, :]
    return unpacked

def quantize_4bit_pertable(w_np):
    """Per-table 4-bit quantization. Returns packed (N, 8), scale, zp."""
    mn, mx = w_np.min(), w_np.max()
    s = (mx - mn) / 15.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = np.clip(np.round(w_np / s + zp), 0, 15).astype(np.uint8)
    packed = (q[:, 0::2] << 4) | q[:, 1::2]
    return packed, float(s), float(zp)

def dequantize_4bit_pertable_batch(packed, s, zp, indices):
    batch = packed[indices]
    hi = (batch >> 4).astype(np.float32)
    lo = (batch & 0x0F).astype(np.float32)
    unpacked = np.empty((batch.shape[0], 16), dtype=np.float32)
    unpacked[:, 0::2] = hi
    unpacked[:, 1::2] = lo
    return (unpacked - zp) * s

def quantize_5bit_percol(w_np):
    """Per-column 5-bit quantization (32 levels). Pack 8 values into 5 bytes.
    For simplicity, store as uint8 with values 0-31 (not bit-packed)."""
    n = w_np.shape[0]
    scales = np.zeros(16, dtype=np.float32)
    zps = np.zeros(16, dtype=np.float32)
    q = np.zeros((n, 16), dtype=np.uint8)
    for c in range(16):
        col = w_np[:, c]
        mn, mx = col.min(), col.max()
        s = (mx - mn) / 31.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        scales[c] = s
        zps[c] = zp
        q[:, c] = np.clip(np.round(col / s + zp), 0, 31).astype(np.uint8)
    # Store unpacked uint8 with values 0-31 (1 byte per value, not bit-packed)
    # Memory = N * 16 bytes (same as uint8), but only 5 bits of info per value
    # For actual memory savings, we'd need bit-packing. Let's do it:
    # Pack: 16 values * 5 bits = 80 bits = 10 bytes per embedding
    packed = np.zeros((n, 10), dtype=np.uint8)
    # Pack 8 values into 5 bytes: [v0:5][v1:5][v2:5][v3:5][v4:5][v5:5][v6:5][v7:5] = 40 bits = 5 bytes
    # First 8 values -> bytes 0-4, next 8 values -> bytes 5-9
    for group in range(2):
        base = group * 8
        out_base = group * 5
        v = q[:, base:base+8].astype(np.uint16)  # (N, 8) with values 0-31
        # Byte 0: v0[4:0] << 3 | v1[4:2]
        packed[:, out_base+0] = ((v[:, 0] << 3) | (v[:, 1] >> 2)).astype(np.uint8)
        # Byte 1: v1[1:0] << 6 | v2[4:0] << 1 | v3[4]
        packed[:, out_base+1] = ((v[:, 1] << 6) | (v[:, 2] << 1) | (v[:, 3] >> 4)).astype(np.uint8)
        # Byte 2: v3[3:0] << 4 | v4[4:1]
        packed[:, out_base+2] = ((v[:, 3] << 4) | (v[:, 4] >> 1)).astype(np.uint8)
        # Byte 3: v4[0] << 7 | v5[4:0] << 2 | v6[4:3]
        packed[:, out_base+3] = ((v[:, 4] << 7) | (v[:, 5] << 2) | (v[:, 6] >> 3)).astype(np.uint8)
        # Byte 4: v6[2:0] << 5 | v7[4:0]
        packed[:, out_base+4] = ((v[:, 6] << 5) | v[:, 7]).astype(np.uint8)
    return packed, scales, zps

def dequantize_5bit_percol_batch(packed, scales, zps, indices):
    """Dequantize per-column 5-bit packed embeddings."""
    batch = packed[indices]  # (B, 10) uint8
    b = batch.astype(np.uint16)
    q = np.zeros((batch.shape[0], 16), dtype=np.float32)
    for group in range(2):
        base = group * 8
        out_base = group * 5
        # Unpack 5 bytes -> 8 values
        q[:, base+0] = ((b[:, out_base+0] >> 3) & 0x1F)
        q[:, base+1] = (((b[:, out_base+0] & 0x07) << 2) | (b[:, out_base+1] >> 6)) & 0x1F
        q[:, base+2] = ((b[:, out_base+1] >> 1) & 0x1F)
        q[:, base+3] = (((b[:, out_base+1] & 0x01) << 4) | (b[:, out_base+2] >> 4)) & 0x1F
        q[:, base+4] = (((b[:, out_base+2] & 0x0F) << 1) | (b[:, out_base+3] >> 7)) & 0x1F
        q[:, base+5] = ((b[:, out_base+3] >> 2) & 0x1F)
        q[:, base+6] = (((b[:, out_base+3] & 0x03) << 3) | (b[:, out_base+4] >> 5)) & 0x1F
        q[:, base+7] = (b[:, out_base+4] & 0x1F)
    return (q - zps[np.newaxis, :]) * scales[np.newaxis, :]

def quantize_6bit_percol(w_np):
    """Per-column 6-bit quantization (64 levels). Pack 4 values into 3 bytes.
    16 values = 4 groups of 4 = 12 bytes per embedding."""
    n = w_np.shape[0]
    scales = np.zeros(16, dtype=np.float32)
    zps = np.zeros(16, dtype=np.float32)
    q = np.zeros((n, 16), dtype=np.uint8)
    for c in range(16):
        col = w_np[:, c]
        mn, mx = col.min(), col.max()
        s = (mx - mn) / 63.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        scales[c] = s
        zps[c] = zp
        q[:, c] = np.clip(np.round(col / s + zp), 0, 63).astype(np.uint8)
    # Pack: 4 values * 6 bits = 24 bits = 3 bytes. 16 values = 4 groups = 12 bytes
    packed = np.zeros((n, 12), dtype=np.uint8)
    for group in range(4):
        base = group * 4
        out_base = group * 3
        v = q[:, base:base+4]  # (N, 4) with values 0-63
        # Byte 0: v0[5:0] << 2 | v1[5:4]
        packed[:, out_base+0] = (v[:, 0] << 2) | (v[:, 1] >> 4)
        # Byte 1: v1[3:0] << 4 | v2[5:2]
        packed[:, out_base+1] = ((v[:, 1] & 0x0F) << 4) | (v[:, 2] >> 2)
        # Byte 2: v2[1:0] << 6 | v3[5:0]
        packed[:, out_base+2] = ((v[:, 2] & 0x03) << 6) | v[:, 3]
    return packed, scales, zps

def dequantize_6bit_percol_batch(packed, scales, zps, indices):
    """Dequantize per-column 6-bit packed embeddings."""
    batch = packed[indices]  # (B, 12) uint8
    q = np.zeros((batch.shape[0], 16), dtype=np.float32)
    for group in range(4):
        base = group * 4
        out_base = group * 3
        b0, b1, b2 = batch[:, out_base], batch[:, out_base+1], batch[:, out_base+2]
        q[:, base+0] = (b0 >> 2) & 0x3F
        q[:, base+1] = ((b0 & 0x03) << 4) | (b1 >> 4)
        q[:, base+2] = ((b1 & 0x0F) << 2) | (b2 >> 6)
        q[:, base+3] = b2 & 0x3F
    return (q - zps[np.newaxis, :]) * scales[np.newaxis, :]

def dequantize_8bit_batch(data, s, zp, indices):
    batch = data[indices]
    return (batch.astype(np.float32) - zp) * s

# ======================== Codec helpers ========================

def get_codec_args(crf, keyint=None):
    ki = f":keyint={keyint}" if keyint else ""
    if crf == 0:
        return ['-c:v', 'libx265', '-preset', 'ultrafast',
                '-x265-params', f'lossless=1:log-level=error{ki}']
    else:
        return ['-c:v', 'libx265', '-crf', str(crf), '-preset', 'ultrafast',
                '-x265-params', f'log-level=error:allow-non-conformance=1{ki}']

MIN_WIDTH = 64; MIN_HEIGHT = 64; MAX_DIM = 16384
TILING_THRESHOLD = 50000; TILE_SIZE = 4

def prepare_single_frame(pixels_np, num_emb, emb_dim):
    from dlrm_s_pytorch import tile_embeddings
    w, h = emb_dim, num_emb
    tm = {'tiled': False}
    if num_emb > TILING_THRESHOLD:
        img, gs, tpe = tile_embeddings(pixels_np.reshape(-1), emb_dim, num_emb, TILE_SIZE)
        w, h = img.shape[1], img.shape[0]
        raw = img.tobytes()
        tm = {'tiled': True, 'grid_size': gs, 'tiles_per_emb': tpe, 'tile_size': TILE_SIZE}
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
    return raw, w, h, tm

def encode_single_frame(raw, w, h, codec_args):
    with tempfile.TemporaryDirectory() as d:
        rf = os.path.join(d, 'i.raw'); vf = os.path.join(d, 'o.mp4')
        with open(rf, 'wb') as f: f.write(raw)
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{w}x{h}', '-r', '1', '-i', rf] + codec_args + ['-frames:v', '1', vf]
        subprocess.run(cmd, capture_output=True, check=True)
        with open(vf, 'rb') as f: data = f.read()
    return data

def decode_single_frame_legacy(comp, w, h, num_emb, emb_dim, tm):
    from dlrm_s_pytorch import untile_embeddings
    with tempfile.TemporaryDirectory() as d:
        vf = os.path.join(d, 'i.mp4'); rf = os.path.join(d, 'o.raw')
        with open(vf, 'wb') as f: f.write(comp)
        subprocess.run(['ffmpeg', '-y', '-i', vf, '-pix_fmt', 'gray', '-f', 'rawvideo', rf],
                       capture_output=True, check=True)
        px = np.fromfile(rf, dtype=np.uint8)
    if tm.get('tiled'):
        gs, ts = tm['grid_size'], tm['tile_size']
        isz = gs * ts
        px = untile_embeddings(px[:isz*isz].reshape(isz, isz),
                               emb_dim, num_emb, gs, tm['tiles_per_emb'], ts)
    else:
        px = px[:num_emb * emb_dim]
    return torch.from_numpy(px.copy()).reshape(num_emb, emb_dim)

def encode_multiframe(pixels_flat, frame_w, frame_h, crf, keyint=1):
    frame_size = frame_w * frame_h
    n_frames = (len(pixels_flat) + frame_size - 1) // frame_size
    padded = np.zeros(n_frames * frame_size, dtype=np.uint8)
    padded[:len(pixels_flat)] = pixels_flat
    codec_args = get_codec_args(crf, keyint)
    with tempfile.TemporaryDirectory() as d:
        rf = os.path.join(d, 'i.raw'); vf = os.path.join(d, 'o.mp4')
        with open(rf, 'wb') as f: f.write(padded.tobytes())
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{frame_w}x{frame_h}', '-r', '30',
               '-i', rf] + codec_args + [vf]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0: raise RuntimeError(f"Encode fail: {r.stderr[:500]}")
        with open(vf, 'rb') as f: data = f.read()
    return data, n_frames

def decode_all_frames_pyav(comp_bytes, frame_w, frame_h):
    frame_size = frame_w * frame_h
    frames = []
    container = av.open(io.BytesIO(comp_bytes))
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format='gray').flatten()[:frame_size])
    container.close()
    return frames

def restore_weights(dlrm, state_dict, emb_keys):
    with torch.no_grad():
        for k in emb_keys:
            t = int(k.split('.')[1])
            dlrm.emb_l[t].weight.data = state_dict[k].clone()

def run_inference(dlrm, all_batches):
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    t0 = time.time()
    with torch.no_grad():
        for X, lS_o, lS_i, T in all_batches:
            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            S = Z.detach().cpu().numpy().flatten()
            Tn = T.detach().cpu().numpy().flatten()
            accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
            samp += Tn.shape[0]
            scores.extend(S.tolist()); targets.extend(Tn.tolist())
    return accu/samp, roc_auc_score(targets, scores), time.time()-t0, blats

def latency_stats(blats):
    a = np.array(blats)
    return {'count': len(a), 'mean': float(np.mean(a)),
            'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
            'p99': float(np.percentile(a, 99))}

def drop_caches():
    try:
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except: pass

def setup_small_and_hot(dlrm, state_dict, emb_keys, ln_emb, num_tables, large_tables, hot_indices):
    total_comp = 0
    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        if w.shape[0] < LARGE_TABLE_THRESHOLD:
            q, s, zp = quantize_8bit(w)
            raw, fw2, fh2, tm = prepare_single_frame(q.numpy(), w.shape[0], EMB_DIM)
            comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
            total_comp += len(comp)
            dec = decode_single_frame_legacy(comp, fw2, fh2, w.shape[0], EMB_DIM, tm)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = (dec.float() - zp) * s

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())
        hot_idx = sorted(hi)
        if hot_idx:
            hw = w[hot_idx]; qh, sh, zh = quantize_8bit(hw)
            raw, fw2, fh2, tm = prepare_single_frame(qh.numpy(), hw.shape[0], EMB_DIM)
            comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
            total_comp += len(comp)
            dec = decode_single_frame_legacy(comp, fw2, fh2, hw.shape[0], EMB_DIM, tm)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[hot_idx] = (dec.float() - zh) * sh
    return total_comp

def decode_cold_uint8_via_codec(w_tensor, cold_idx, fw, fh, epf):
    """Encode cold embeddings through CRF 0, decode, return as fp32 numpy."""
    cw = w_tensor[cold_idx]
    n_cold = len(cold_idx)
    if n_cold == 0:
        return np.zeros((0, EMB_DIM), dtype=np.float32), 0, 0, 0
    q8, s8, zp8 = quantize_8bit(cw)
    pf = q8.numpy().reshape(-1)
    comp_data, n_frames = encode_multiframe(pf, fw, fh, 0, keyint=1)
    comp_bytes = len(comp_data)
    all_frames = decode_all_frames_pyav(comp_data, fw, fh)

    decoded_uint8 = np.zeros((n_cold, EMB_DIM), dtype=np.uint8)
    for fi, fd in enumerate(all_frames):
        start = fi * epf
        end = min(start + epf, n_cold)
        count = end - start
        decoded_uint8[start:end] = fd[:count * EMB_DIM].reshape(count, EMB_DIM)

    fp32 = (decoded_uint8.astype(np.float32) - zp8) * s8
    return fp32, comp_bytes, s8, zp8


def build_cold_lookup(cold_idx, table_size):
    """Build a numpy array mapping original index -> sequential cold index.
    Returns array of size table_size where lookup[orig_idx] = seq_idx or -1."""
    lookup = np.full(table_size, -1, dtype=np.int32)
    for seq, orig in enumerate(cold_idx):
        lookup[orig] = seq
    return lookup


def vectorized_inject(dlrm, t, indices_np, cold_lookup, cold_data, dequant_fn):
    """Vectorized cold injection using numpy lookup array.
    cold_lookup: array[table_size] -> seq_idx or -1
    cold_data: quantized cold storage
    dequant_fn: callable(cold_data, seq_indices) -> fp32 (B, 16)
    """
    # Map original indices to sequential cold indices
    # Clamp to valid range for lookup
    valid_mask = (indices_np >= 0) & (indices_np < len(cold_lookup))
    seq = np.full_like(indices_np, -1, dtype=np.int32)
    seq[valid_mask] = cold_lookup[indices_np[valid_mask]]

    # Find unique cold indices that need injection
    cold_mask = seq >= 0
    if not cold_mask.any():
        return

    cold_orig = indices_np[cold_mask]
    cold_seq = seq[cold_mask]

    # Unique
    unique_seq, first_idx = np.unique(cold_seq, return_index=True)
    unique_orig = cold_orig[first_idx]

    # Dequantize
    fp32 = dequant_fn(cold_data, unique_seq)

    # Write back
    dlrm.emb_l[t].weight.data[torch.from_numpy(unique_orig.astype(np.int64))] = \
        torch.from_numpy(fp32)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log("=" * 70)
    log("PREFETCH BENCHMARK V7: Improved Quantization + Vectorized Injection")
    log("=" * 70)

    log("Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]

    log("Pre-loading all test batches...")
    all_batches = list(test_ld)
    nb = len(all_batches)
    log(f"  {nb} batches")

    log("Profiling access patterns...")
    access_raw = [None] * num_tables
    for i, batch in enumerate(train_ld):
        if i >= PROFILE_BATCHES: break
        _, _, lS_i, _ = batch
        for t in range(num_tables):
            idx = lS_i[t].numpy().flatten()
            access_raw[t] = idx.copy() if access_raw[t] is None else np.concatenate([access_raw[t], idx])

    hot_indices = {}
    access_counts = {}
    for t in range(num_tables):
        if access_raw[t] is None or len(access_raw[t]) == 0:
            hot_indices[t] = np.array([], dtype=np.int64)
            access_counts[t] = {}
            continue
        unique, counts = np.unique(access_raw[t], return_counts=True)
        access_counts[t] = {int(u): int(c) for u, c in zip(unique, counts)}
        si = np.argsort(-counts)
        cum = np.cumsum(counts[si])
        cutoff = np.searchsorted(cum, cum[-1] * HOT_THRESHOLD) + 1
        hot_indices[t] = unique[si[:cutoff]]

    freq_sorted_cold = {}
    for t in large_tables:
        hi_set = set(hot_indices[t].tolist())
        all_cold = sorted(set(range(ln_emb[t])) - hi_set)
        counts = access_counts[t]
        freq_sorted_cold[t] = sorted(all_cold, key=lambda x: (-counts.get(x, 0), x))
        log(f"  Table {t}: {ln_emb[t]:,} total, {len(hot_indices[t]):,} hot, {len(all_cold):,} cold")

    fw, fh = 3840, 2160
    epf = (fw * fh) // EMB_DIM

    # Pre-compute memory constants
    hot_mem = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables)
    small_mem = sum(state_dict[emb_keys[t]].numel() * 4
                    for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD)
    mlp_mem = sum(p.numel() * 4 for n, p in dlrm.named_parameters() if 'emb_l' not in n)

    results = {}

    # ============================================================
    # BASELINE
    # ============================================================
    log("\n" + "=" * 60)
    log("BASELINE")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    drop_caches(); time.sleep(1)
    a_acc, a_auc, a_time, a_blats = run_inference(dlrm, all_batches)
    a_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
    log(f"  AUC={a_auc:.6f}, Time={a_time:.1f}s, Mem={a_mem:.1f}MB")
    baseline_auc = a_auc
    results['baseline'] = {
        'name': 'Baseline (fp32)', 'auc': a_auc, 'accuracy': a_acc,
        'inference_time': a_time, 'memory_mb': a_mem, 'total_time': a_time,
    }

    # ============================================================
    # EXP 1: Pure quantization quality sweep (no codec, per-column)
    # ============================================================
    log("\n" + "=" * 60)
    log("EXP 1: Per-column quantization quality (no codec)")
    log("=" * 60)

    for bits in [8, 6, 5, 4]:
        restore_weights(dlrm, state_dict, emb_keys)
        with torch.no_grad():
            for t in large_tables:
                w = state_dict[emb_keys[t]]
                cold_idx = freq_sorted_cold[t]
                cw = w[cold_idx].numpy()

                if bits == 8:
                    # Per-table 8-bit (reference from V6)
                    q, s, zp = quantize_8bit(w[cold_idx])
                    deq = (q.float() - zp) * s
                elif bits == 6:
                    packed, scales, zps = quantize_6bit_percol(cw)
                    all_idx = np.arange(len(cold_idx))
                    deq = torch.from_numpy(dequantize_6bit_percol_batch(packed, scales, zps, all_idx))
                elif bits == 5:
                    packed, scales, zps = quantize_5bit_percol(cw)
                    all_idx = np.arange(len(cold_idx))
                    deq = torch.from_numpy(dequantize_5bit_percol_batch(packed, scales, zps, all_idx))
                elif bits == 4:
                    packed, scales, zps = quantize_4bit_percol(cw)
                    all_idx = np.arange(len(cold_idx))
                    deq = torch.from_numpy(dequantize_4bit_percol_batch(packed, scales, zps, all_idx))

                idx_t = torch.tensor(cold_idx, dtype=torch.long)
                dlrm.emb_l[t].weight.data[idx_t] = deq

        acc, auc, infer_time, blats = run_inference(dlrm, all_batches)
        loss = (baseline_auc - auc) * 100

        # Memory calculation
        bytes_per_emb = {8: 16, 6: 12, 5: 10, 4: 8}[bits]
        cold_mem = sum(len(freq_sorted_cold[t]) * bytes_per_emb for t in large_tables) / 1024 / 1024
        # Overhead for per-column scales/zps: 16 * 4 bytes * 2 * num_large_tables
        overhead = len(large_tables) * 16 * 4 * 2 / 1024 / 1024

        label = f"{bits}-bit percol" if bits <= 6 else f"{bits}-bit"
        log(f"  {label}: AUC={auc:.6f}, loss={loss:.4f}pp, cold_mem={cold_mem:.1f}MB")
        results[f'pure_{bits}bit_percol'] = {
            'name': f'Pure {bits}-bit per-column', 'bits': bits,
            'auc': auc, 'accuracy': acc, 'auc_loss_pp': loss,
            'cold_memory_mb': cold_mem,
        }

    # Also test per-table 4-bit for comparison
    restore_weights(dlrm, state_dict, emb_keys)
    with torch.no_grad():
        for t in large_tables:
            w = state_dict[emb_keys[t]]
            cold_idx = freq_sorted_cold[t]
            cw = w[cold_idx].numpy()
            packed, s, zp = quantize_4bit_pertable(cw)
            all_idx = np.arange(len(cold_idx))
            deq = torch.from_numpy(dequantize_4bit_pertable_batch(packed, s, zp, all_idx))
            idx_t = torch.tensor(cold_idx, dtype=torch.long)
            dlrm.emb_l[t].weight.data[idx_t] = deq
    acc, auc, infer_time, blats = run_inference(dlrm, all_batches)
    loss = (baseline_auc - auc) * 100
    log(f"  4-bit pertable: AUC={auc:.6f}, loss={loss:.4f}pp (V6 reference)")
    results['pure_4bit_pertable'] = {
        'name': 'Pure 4-bit per-table (V6 ref)', 'bits': 4,
        'auc': auc, 'accuracy': acc, 'auc_loss_pp': loss,
        'cold_memory_mb': sum(len(freq_sorted_cold[t]) * 8 for t in large_tables) / 1024 / 1024,
    }

    # ============================================================
    # EXP 2: Dense cold storage with per-column 4-bit (via CRF 0 codec)
    # Vectorized batch injection
    # ============================================================
    log("\n" + "=" * 60)
    log("EXP 2: All-cold per-column 4-bit (CRF 0 + vectorized inject)")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    setup_t0 = time.time()
    total_comp = setup_small_and_hot(dlrm, state_dict, emb_keys, ln_emb, num_tables, large_tables, hot_indices)

    cold_4bit_pc = {}     # {table: packed (n_cold, 8)}
    cold_4bit_pc_q = {}   # {table: (scales, zps)}
    cold_lookups = {}      # {table: lookup array}

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        cold_idx = freq_sorted_cold[t]
        fp32, comp_bytes, s8, zp8 = decode_cold_uint8_via_codec(w, cold_idx, fw, fh, epf)
        total_comp += comp_bytes
        if fp32.shape[0] == 0: continue

        packed, scales, zps = quantize_4bit_percol(fp32)
        cold_4bit_pc[t] = packed
        cold_4bit_pc_q[t] = (scales, zps)
        cold_lookups[t] = build_cold_lookup(cold_idx, ln_emb[t])
        log(f"    Table {t}: {len(cold_idx):,} cold, packed={packed.nbytes/1024/1024:.1f}MB")

    setup_time = time.time() - setup_t0

    fourbit_mem = sum(v.nbytes for v in cold_4bit_pc.values())
    # Scale/zp overhead
    qparam_mem = sum(s.nbytes + z.nbytes for s, z in cold_4bit_pc_q.values())
    total_mem = (fourbit_mem + qparam_mem + hot_mem + small_mem + mlp_mem) / 1024 / 1024

    log(f"  Setup: {setup_time:.1f}s")
    log(f"  Memory: 4bit_cold={fourbit_mem/1024/1024:.1f}MB, total={total_mem:.1f}MB")

    # Run inference with vectorized injection
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    drop_caches(); time.sleep(1)
    t0 = time.time()
    with torch.no_grad():
        for bi in range(nb):
            X, lS_o, lS_i, T = all_batches[bi]
            bt0 = time.time()

            for t in large_tables:
                if t not in cold_4bit_pc: continue
                indices_np = lS_i[t].numpy().flatten()
                scales, zps = cold_4bit_pc_q[t]

                def dequant_fn(data, seq_idx, _s=scales, _z=zps):
                    return dequantize_4bit_percol_batch(data, _s, _z, seq_idx)

                vectorized_inject(dlrm, t, indices_np, cold_lookups[t],
                                  cold_4bit_pc[t], dequant_fn)

            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            S = Z.detach().cpu().numpy().flatten()
            Tn = T.detach().cpu().numpy().flatten()
            accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
            samp += Tn.shape[0]
            scores.extend(S.tolist()); targets.extend(Tn.tolist())

            if bi % 400 == 0:
                log(f"    Batch {bi}/{nb}, batch={blats[-1]*1000:.1f}ms")

    infer_time = time.time() - t0
    acc = accu / samp
    auc = roc_auc_score(targets, scores)
    loss = (baseline_auc - auc) * 100
    log(f"  AUC={auc:.6f}, loss={loss:.4f}pp")
    log(f"  Inference={infer_time:.1f}s, Setup={setup_time:.1f}s, Total={setup_time+infer_time:.1f}s")
    log(f"  Batch: avg={np.mean(blats)*1000:.1f}ms")

    results['all_4bit_percol'] = {
        'name': 'All cold 4-bit per-col (CRF 0)', 'auc': auc, 'accuracy': acc,
        'auc_loss_pp': loss, 'inference_time': infer_time, 'setup_time': setup_time,
        'total_time': setup_time + infer_time, 'memory_mb': total_mem,
        'compressed_mb': total_comp / 1024 / 1024,
        'batch_latencies': latency_stats(blats),
    }

    # ============================================================
    # EXP 3: Dense cold 6-bit per-column (CRF 0 + vectorized inject)
    # ============================================================
    log("\n" + "=" * 60)
    log("EXP 3: All-cold per-column 6-bit (CRF 0 + vectorized inject)")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    setup_t0 = time.time()
    total_comp = setup_small_and_hot(dlrm, state_dict, emb_keys, ln_emb, num_tables, large_tables, hot_indices)

    cold_6bit = {}
    cold_6bit_q = {}
    cold_lookups_6 = {}

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        cold_idx = freq_sorted_cold[t]
        fp32, comp_bytes, s8, zp8 = decode_cold_uint8_via_codec(w, cold_idx, fw, fh, epf)
        total_comp += comp_bytes
        if fp32.shape[0] == 0: continue

        packed, scales, zps = quantize_6bit_percol(fp32)
        cold_6bit[t] = packed
        cold_6bit_q[t] = (scales, zps)
        cold_lookups_6[t] = build_cold_lookup(cold_idx, ln_emb[t])
        log(f"    Table {t}: {len(cold_idx):,} cold, packed={packed.nbytes/1024/1024:.1f}MB")

    setup_time = time.time() - setup_t0

    sixbit_mem = sum(v.nbytes for v in cold_6bit.values())
    qparam_mem = sum(s.nbytes + z.nbytes for s, z in cold_6bit_q.values())
    total_mem = (sixbit_mem + qparam_mem + hot_mem + small_mem + mlp_mem) / 1024 / 1024

    log(f"  Setup: {setup_time:.1f}s")
    log(f"  Memory: 6bit_cold={sixbit_mem/1024/1024:.1f}MB, total={total_mem:.1f}MB")

    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    drop_caches(); time.sleep(1)
    t0 = time.time()
    with torch.no_grad():
        for bi in range(nb):
            X, lS_o, lS_i, T = all_batches[bi]
            bt0 = time.time()

            for t in large_tables:
                if t not in cold_6bit: continue
                indices_np = lS_i[t].numpy().flatten()
                scales, zps = cold_6bit_q[t]

                def dequant_fn(data, seq_idx, _s=scales, _z=zps):
                    return dequantize_6bit_percol_batch(data, _s, _z, seq_idx)

                vectorized_inject(dlrm, t, indices_np, cold_lookups_6[t],
                                  cold_6bit[t], dequant_fn)

            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            S = Z.detach().cpu().numpy().flatten()
            Tn = T.detach().cpu().numpy().flatten()
            accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
            samp += Tn.shape[0]
            scores.extend(S.tolist()); targets.extend(Tn.tolist())

            if bi % 400 == 0:
                log(f"    Batch {bi}/{nb}, batch={blats[-1]*1000:.1f}ms")

    infer_time = time.time() - t0
    acc = accu / samp
    auc = roc_auc_score(targets, scores)
    loss = (baseline_auc - auc) * 100
    log(f"  AUC={auc:.6f}, loss={loss:.4f}pp")
    log(f"  Inference={infer_time:.1f}s, Setup={setup_time:.1f}s, Total={setup_time+infer_time:.1f}s")
    log(f"  Batch: avg={np.mean(blats)*1000:.1f}ms")

    results['all_6bit_percol'] = {
        'name': 'All cold 6-bit per-col (CRF 0)', 'auc': auc, 'accuracy': acc,
        'auc_loss_pp': loss, 'inference_time': infer_time, 'setup_time': setup_time,
        'total_time': setup_time + infer_time, 'memory_mb': total_mem,
        'compressed_mb': total_comp / 1024 / 1024,
        'batch_latencies': latency_stats(blats),
    }

    # ============================================================
    # EXP 4: Dense cold 5-bit per-column (CRF 0 + vectorized inject)
    # ============================================================
    log("\n" + "=" * 60)
    log("EXP 4: All-cold per-column 5-bit (CRF 0 + vectorized inject)")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    setup_t0 = time.time()
    total_comp = setup_small_and_hot(dlrm, state_dict, emb_keys, ln_emb, num_tables, large_tables, hot_indices)

    cold_5bit = {}
    cold_5bit_q = {}
    cold_lookups_5 = {}

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        cold_idx = freq_sorted_cold[t]
        fp32, comp_bytes, s8, zp8 = decode_cold_uint8_via_codec(w, cold_idx, fw, fh, epf)
        total_comp += comp_bytes
        if fp32.shape[0] == 0: continue

        packed, scales, zps = quantize_5bit_percol(fp32)
        cold_5bit[t] = packed
        cold_5bit_q[t] = (scales, zps)
        cold_lookups_5[t] = build_cold_lookup(cold_idx, ln_emb[t])
        log(f"    Table {t}: {len(cold_idx):,} cold, packed={packed.nbytes/1024/1024:.1f}MB")

    setup_time = time.time() - setup_t0

    fivebit_mem = sum(v.nbytes for v in cold_5bit.values())
    qparam_mem = sum(s.nbytes + z.nbytes for s, z in cold_5bit_q.values())
    total_mem = (fivebit_mem + qparam_mem + hot_mem + small_mem + mlp_mem) / 1024 / 1024

    log(f"  Setup: {setup_time:.1f}s")
    log(f"  Memory: 5bit_cold={fivebit_mem/1024/1024:.1f}MB, total={total_mem:.1f}MB")

    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    drop_caches(); time.sleep(1)
    t0 = time.time()
    with torch.no_grad():
        for bi in range(nb):
            X, lS_o, lS_i, T = all_batches[bi]
            bt0 = time.time()

            for t in large_tables:
                if t not in cold_5bit: continue
                indices_np = lS_i[t].numpy().flatten()
                scales, zps = cold_5bit_q[t]

                def dequant_fn(data, seq_idx, _s=scales, _z=zps):
                    return dequantize_5bit_percol_batch(data, _s, _z, seq_idx)

                vectorized_inject(dlrm, t, indices_np, cold_lookups_5[t],
                                  cold_5bit[t], dequant_fn)

            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            S = Z.detach().cpu().numpy().flatten()
            Tn = T.detach().cpu().numpy().flatten()
            accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
            samp += Tn.shape[0]
            scores.extend(S.tolist()); targets.extend(Tn.tolist())

            if bi % 400 == 0:
                log(f"    Batch {bi}/{nb}, batch={blats[-1]*1000:.1f}ms")

    infer_time = time.time() - t0
    acc = accu / samp
    auc = roc_auc_score(targets, scores)
    loss = (baseline_auc - auc) * 100
    log(f"  AUC={auc:.6f}, loss={loss:.4f}pp")
    log(f"  Inference={infer_time:.1f}s, Setup={setup_time:.1f}s, Total={setup_time+infer_time:.1f}s")
    log(f"  Batch: avg={np.mean(blats)*1000:.1f}ms")

    results['all_5bit_percol'] = {
        'name': 'All cold 5-bit per-col (CRF 0)', 'auc': auc, 'accuracy': acc,
        'auc_loss_pp': loss, 'inference_time': infer_time, 'setup_time': setup_time,
        'total_time': setup_time + infer_time, 'memory_mb': total_mem,
        'compressed_mb': total_comp / 1024 / 1024,
        'batch_latencies': latency_stats(blats),
    }

    # ============================================================
    # EXP 5: Mixed 8b warm + per-col 4b ice (best memory/quality tradeoff)
    # ============================================================
    log("\n" + "=" * 60)
    log("EXP 5: Mixed 8-bit warm + per-col 4-bit ice (CRF 0)")
    log("=" * 60)

    for warm_k in [500000, 1000000, 2000000]:
        log(f"\n  --- warm_k={warm_k:,} ---")
        restore_weights(dlrm, state_dict, emb_keys)
        setup_t0 = time.time()
        total_comp = setup_small_and_hot(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                                          large_tables, hot_indices)

        warm_uint8 = {}; warm_quant = {}; warm_lookups = {}
        ice_4bit_pc = {}; ice_4bit_pc_q = {}; ice_lookups = {}

        for t in large_tables:
            w = state_dict[emb_keys[t]]
            cold_sorted = freq_sorted_cold[t]
            wk = min(warm_k, len(cold_sorted))
            warm_idx = cold_sorted[:wk]
            ice_idx = cold_sorted[wk:]

            if warm_idx:
                fp32_w, comp_bytes, s8, zp8 = decode_cold_uint8_via_codec(w, warm_idx, fw, fh, epf)
                total_comp += comp_bytes
                # Store as uint8
                warm_data = np.clip(np.round(fp32_w / s8 + zp8), 0, 255).astype(np.uint8)
                warm_uint8[t] = warm_data
                warm_quant[t] = (s8, zp8)
                warm_lookups[t] = build_cold_lookup(warm_idx, ln_emb[t])

            if ice_idx:
                fp32_i, comp_bytes, s8_i, zp8_i = decode_cold_uint8_via_codec(w, ice_idx, fw, fh, epf)
                total_comp += comp_bytes
                packed, scales, zps = quantize_4bit_percol(fp32_i)
                ice_4bit_pc[t] = packed
                ice_4bit_pc_q[t] = (scales, zps)
                ice_lookups[t] = build_cold_lookup(ice_idx, ln_emb[t])

        setup_time = time.time() - setup_t0

        warm_mem = sum(v.nbytes for v in warm_uint8.values())
        ice_mem = sum(v.nbytes for v in ice_4bit_pc.values())
        total_mem = (warm_mem + ice_mem + hot_mem + small_mem + mlp_mem) / 1024 / 1024

        log(f"  Setup: {setup_time:.1f}s")
        log(f"  Memory: warm_8bit={warm_mem/1024/1024:.1f}MB, ice_4bit={ice_mem/1024/1024:.1f}MB, total={total_mem:.1f}MB")

        scores, targets = [], []
        accu, samp = 0, 0
        blats = []
        drop_caches(); time.sleep(1)
        t0 = time.time()
        with torch.no_grad():
            for bi in range(nb):
                X, lS_o, lS_i, T = all_batches[bi]
                bt0 = time.time()

                for t in large_tables:
                    indices_np = lS_i[t].numpy().flatten()

                    # Warm injection
                    if t in warm_uint8:
                        s8, zp8 = warm_quant[t]
                        def dequant_warm(data, seq_idx, _s=s8, _z=zp8):
                            return dequantize_8bit_batch(data, _s, _z, seq_idx)
                        vectorized_inject(dlrm, t, indices_np, warm_lookups[t],
                                          warm_uint8[t], dequant_warm)

                    # Ice injection
                    if t in ice_4bit_pc:
                        scales, zps = ice_4bit_pc_q[t]
                        def dequant_ice(data, seq_idx, _s=scales, _z=zps):
                            return dequantize_4bit_percol_batch(data, _s, _z, seq_idx)
                        vectorized_inject(dlrm, t, indices_np, ice_lookups[t],
                                          ice_4bit_pc[t], dequant_ice)

                Z = dlrm(X, lS_o, lS_i)
                blats.append(time.time() - bt0)
                S = Z.detach().cpu().numpy().flatten()
                Tn = T.detach().cpu().numpy().flatten()
                accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
                samp += Tn.shape[0]
                scores.extend(S.tolist()); targets.extend(Tn.tolist())

                if bi % 400 == 0:
                    log(f"    Batch {bi}/{nb}, batch={blats[-1]*1000:.1f}ms")

        infer_time = time.time() - t0
        acc = accu / samp
        auc = roc_auc_score(targets, scores)
        loss = (baseline_auc - auc) * 100
        log(f"  AUC={auc:.6f}, loss={loss:.4f}pp")
        log(f"  Inference={infer_time:.1f}s, Total={setup_time+infer_time:.1f}s")
        log(f"  Batch: avg={np.mean(blats)*1000:.1f}ms")

        results[f'mixed_8b_4bpc_warm{warm_k}'] = {
            'name': f'Mixed 8b warm({warm_k:,}) + 4b-percol ice',
            'warm_k': warm_k, 'auc': auc, 'accuracy': acc,
            'auc_loss_pp': loss, 'inference_time': infer_time,
            'setup_time': setup_time, 'total_time': setup_time + infer_time,
            'memory_mb': total_mem,
            'warm_8bit_mb': warm_mem / 1024 / 1024,
            'ice_4bit_mb': ice_mem / 1024 / 1024,
            'batch_latencies': latency_stats(blats),
        }

    # ============================================================
    # EXP 6: Dense uint8 full reconstruction (V5 reference, no per-batch injection)
    # ============================================================
    log("\n" + "=" * 60)
    log("EXP 6: Dense uint8 full reconstruction (CRF 0, no injection)")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    setup_t0 = time.time()
    total_comp = 0

    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        if w.shape[0] < LARGE_TABLE_THRESHOLD:
            q, s, zp = quantize_8bit(w)
            raw, fw2, fh2, tm = prepare_single_frame(q.numpy(), w.shape[0], EMB_DIM)
            comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
            total_comp += len(comp)
            dec = decode_single_frame_legacy(comp, fw2, fh2, w.shape[0], EMB_DIM, tm)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = (dec.float() - zp) * s
        else:
            # Full reconstruction: encode all, decode all, write back
            q, s, zp = quantize_8bit(w)
            pf = q.numpy().reshape(-1)
            comp_data, n_f = encode_multiframe(pf, fw, fh, 0, keyint=1)
            total_comp += len(comp_data)
            frames = decode_all_frames_pyav(comp_data, fw, fh)
            n_emb = w.shape[0]
            decoded = np.zeros((n_emb, EMB_DIM), dtype=np.uint8)
            for fi, fd in enumerate(frames):
                start = fi * epf
                end = min(start + epf, n_emb)
                count = end - start
                decoded[start:end] = fd[:count * EMB_DIM].reshape(count, EMB_DIM)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = torch.from_numpy(
                    (decoded.astype(np.float32) - zp) * s)
            log(f"    Table {t}: {n_emb:,} embs, full reconstruction")

    setup_time = time.time() - setup_t0
    total_mem_recon = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024

    log(f"  Setup: {setup_time:.1f}s")
    log(f"  Memory: {total_mem_recon:.1f}MB (full fp32)")

    drop_caches(); time.sleep(1)
    acc, auc, infer_time, blats = run_inference(dlrm, all_batches)
    loss = (baseline_auc - auc) * 100
    log(f"  AUC={auc:.6f}, loss={loss:.4f}pp")
    log(f"  Inference={infer_time:.1f}s, Setup={setup_time:.1f}s, Total={setup_time+infer_time:.1f}s")

    results['full_recon_uint8'] = {
        'name': 'Full reconstruction (uint8 CRF 0)', 'auc': auc, 'accuracy': acc,
        'auc_loss_pp': loss, 'inference_time': infer_time, 'setup_time': setup_time,
        'total_time': setup_time + infer_time, 'memory_mb': total_mem_recon,
        'compressed_mb': total_comp / 1024 / 1024,
        'batch_latencies': latency_stats(blats),
    }

    # ============================================================
    # FINAL REPORT
    # ============================================================
    log("\n" + "=" * 70)
    log("FINAL REPORT")
    log("=" * 70)

    report = []
    report.append("# Prefetch V7: Improved Quantization Results\n\n")
    report.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Baseline AUC:** {baseline_auc:.6f}\n")
    report.append(f"**Baseline Memory:** {results['baseline']['memory_mb']:.1f}MB\n\n")

    report.append("## Per-Column Quantization Quality (no codec)\n\n")
    report.append("| Bits | Type | AUC | AUC Loss (pp) | Cold Memory (MB) |\n")
    report.append("|------|------|-----|---------------|------------------|\n")
    for bits in [8, 6, 5, 4]:
        r = results.get(f'pure_{bits}bit_percol')
        if r:
            report.append(f"| {bits} | per-col | {r['auc']:.6f} | {r['auc_loss_pp']:.4f} | {r['cold_memory_mb']:.1f} |\n")
    r = results.get('pure_4bit_pertable')
    if r:
        report.append(f"| 4 | per-table | {r['auc']:.6f} | {r['auc_loss_pp']:.4f} | {r['cold_memory_mb']:.1f} |\n")

    report.append("\n## Storage Layout Comparison (with codec + injection)\n\n")
    report.append("| Config | AUC | AUC Loss (pp) | Inference (s) | Setup (s) | Total (s) | Memory (MB) |\n")
    report.append("|--------|-----|---------------|--------------|----------|----------|------------|\n")

    layout_keys = ['baseline', 'full_recon_uint8', 'all_6bit_percol', 'all_5bit_percol', 'all_4bit_percol']
    layout_keys += [k for k in sorted(results.keys()) if k.startswith('mixed_8b_4bpc_')]
    for key in layout_keys:
        r = results.get(key)
        if not r or 'auc' not in r: continue
        report.append(f"| {r['name']} | {r['auc']:.6f} | {r.get('auc_loss_pp',0):.4f} | "
                      f"{r.get('inference_time',0):.1f} | {r.get('setup_time',0):.1f} | "
                      f"{r.get('total_time',0):.1f} | {r.get('memory_mb',0):.1f} |\n")

    md_path = os.path.join(RESULTS_DIR, 'prefetch_v7_improved_quant.md')
    with open(md_path, 'w') as f: f.writelines(report)
    log(f"  Saved: {md_path}")

    json_path = os.path.join(RESULTS_DIR, 'prefetch_v7_improved_quant.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  Saved: {json_path}")

    log("=" * 70)
    log("V7 COMPLETE!")
    log("=" * 70)


if __name__ == '__main__':
    main()
