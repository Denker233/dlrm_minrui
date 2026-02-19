#!/usr/bin/env python3
"""
Prefetch Benchmark V6: Sub-byte quantization for lower memory.

Key idea: use 4-bit quantization for ice-cold embeddings to halve their storage.
- Hot: fp32 in weight.data (CRF 0 encode/decode)
- Warm-cold: uint8 from CRF 0 decode (top K cold by frequency)
- Ice-cold: 4-bit packed in memory (no decode needed, just unpack + dequantize)
- Small tables: fp32 (CRF 0)

Experiments:
  1. Pure 4-bit quantization quality test (no codec)
  2. Mixed 8b/4b: warm uint8 + ice 4-bit
  3. Full all-cold 4-bit (lowest memory)
  4. Extended warm_k sweep for three-tier (larger warm tiers)
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

def quantize_8bit(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp

def quantize_4bit(w):
    """Per-table 4-bit quantization. Returns packed uint8 (2 values per byte)."""
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 15.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 15).to(torch.uint8)
    # Pack: (N, 16) -> (N, 8), two 4-bit values per byte
    q_np = q.numpy()
    packed = (q_np[:, 0::2] << 4) | q_np[:, 1::2]  # (N, 8) uint8
    return packed, s, zp

def dequantize_4bit_batch(packed, s, zp, indices):
    """Dequantize a batch of 4-bit packed embeddings."""
    batch = packed[indices]  # (B, 8) uint8
    # Unpack
    hi = (batch >> 4).astype(np.float32)  # (B, 8)
    lo = (batch & 0x0F).astype(np.float32)  # (B, 8)
    # Interleave: [hi0, lo0, hi1, lo1, ...]
    unpacked = np.empty((batch.shape[0], 16), dtype=np.float32)
    unpacked[:, 0::2] = hi
    unpacked[:, 1::2] = lo
    return (unpacked - zp) * s

def dequantize_8bit_batch(data, s, zp, indices):
    """Dequantize a batch of uint8 embeddings."""
    batch = data[indices]  # (B, 16) uint8
    return (batch.astype(np.float32) - zp) * s

def get_codec_args(crf, keyint=None):
    ki = f":keyint={keyint}" if keyint else ""
    if crf == 0:
        return ['-c:v', 'libx265', '-preset', 'ultrafast',
                '-x265-params', f'lossless=1:log-level=error{ki}']
    else:
        return ['-c:v', 'libx265', '-crf', str(crf), '-preset', 'ultrafast',
                '-x265-params', f'log-level=error:allow-non-conformance=1{ki}']

# Single-frame encode/decode
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
    """Common setup: CRF 0 for small tables + hot embeddings."""
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


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log("=" * 70)
    log("PREFETCH BENCHMARK V6: Sub-byte Quantization")
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

    # Build frequency-sorted cold order
    freq_sorted_cold = {}
    for t in large_tables:
        hi_set = set(hot_indices[t].tolist())
        all_cold = sorted(set(range(ln_emb[t])) - hi_set)
        counts = access_counts[t]
        freq_sorted_cold[t] = sorted(all_cold, key=lambda x: (-counts.get(x, 0), x))
        log(f"  Table {t}: {ln_emb[t]:,} total, {len(hot_indices[t]):,} hot, {len(all_cold):,} cold")

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
        'name': 'Baseline', 'auc': a_auc, 'accuracy': a_acc,
        'inference_time': a_time, 'memory_mb': a_mem, 'total_time': a_time,
    }

    # ============================================================
    # EXP 1: Pure quantization quality test (no codec at all)
    # ============================================================
    log("\n" + "=" * 60)
    log("EXP 1: Pure quantization quality (no codec)")
    log("=" * 60)

    for bits in [8, 6, 4]:
        restore_weights(dlrm, state_dict, emb_keys)
        with torch.no_grad():
            for t in large_tables:
                w = state_dict[emb_keys[t]]
                hi = set(hot_indices[t].tolist())
                cold_idx = freq_sorted_cold[t]
                cw = w[cold_idx]

                if bits == 8:
                    q, s, zp = quantize_8bit(cw)
                    deq = (q.float() - zp) * s
                elif bits == 6:
                    mn, mx = cw.min().item(), cw.max().item()
                    s = (mx - mn) / 63.0
                    if s == 0: s = 1.0
                    zp = round(-mn / s)
                    q = ((cw / s).round() + zp).clamp(0, 63).to(torch.uint8)
                    deq = (q.float() - zp) * s
                elif bits == 4:
                    mn, mx = cw.min().item(), cw.max().item()
                    s = (mx - mn) / 15.0
                    if s == 0: s = 1.0
                    zp = round(-mn / s)
                    q = ((cw / s).round() + zp).clamp(0, 15).to(torch.uint8)
                    deq = (q.float() - zp) * s

                idx_t = torch.tensor(cold_idx, dtype=torch.long)
                dlrm.emb_l[t].weight.data[idx_t] = deq

        acc, auc, infer_time, blats = run_inference(dlrm, all_batches)
        loss = (baseline_auc - auc) * 100
        # Theoretical memory for cold
        if bits == 8:
            cold_mem = sum(len(freq_sorted_cold[t]) * EMB_DIM for t in large_tables) / 1024 / 1024
        elif bits == 6:
            cold_mem = sum(len(freq_sorted_cold[t]) * EMB_DIM * 6 / 8 for t in large_tables) / 1024 / 1024
        elif bits == 4:
            cold_mem = sum(len(freq_sorted_cold[t]) * EMB_DIM / 2 for t in large_tables) / 1024 / 1024
        log(f"  {bits}-bit: AUC={auc:.6f}, loss={loss:.4f}pp, cold_mem={cold_mem:.0f}MB")
        results[f'pure_{bits}bit'] = {
            'name': f'Pure {bits}-bit quantization', 'bits': bits,
            'auc': auc, 'accuracy': acc, 'auc_loss_pp': loss,
            'cold_memory_mb': cold_mem,
        }

    # ============================================================
    # EXP 2: All-cold 4-bit with codec (encode 8-bit -> CRF 0 -> decode -> requantize to 4-bit)
    # ============================================================
    log("\n" + "=" * 60)
    log("EXP 2: All-cold 4-bit packed storage (via CRF 0 codec)")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    setup_t0 = time.time()
    total_comp = setup_small_and_hot(dlrm, state_dict, emb_keys, ln_emb, num_tables, large_tables, hot_indices)

    fw, fh = 3840, 2160
    epf = (fw * fh) // EMB_DIM

    cold_4bit = {}    # {table: packed np.ndarray (n_cold, 8)}
    cold_4bit_q = {}  # {table: (scale, zp)}
    cold_idx_maps = {}

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        cold_idx = freq_sorted_cold[t]
        cw = w[cold_idx]
        if cw.shape[0] == 0: continue

        # Encode through codec (CRF 0 for lossless uint8 round-trip)
        q8, s8, zp8 = quantize_8bit(cw)
        pf = q8.numpy().reshape(-1)
        comp_data, n_frames = encode_multiframe(pf, fw, fh, 0, keyint=1)
        total_comp += len(comp_data)
        all_frames = decode_all_frames_pyav(comp_data, fw, fh)

        # Reconstruct uint8 from decoded frames
        n_cold = len(cold_idx)
        decoded_uint8 = np.zeros((n_cold, EMB_DIM), dtype=np.uint8)
        for fi, fd in enumerate(all_frames):
            start = fi * epf
            end = min(start + epf, n_cold)
            for j in range(start, end):
                off = (j - start) * EMB_DIM
                decoded_uint8[j] = fd[off:off + EMB_DIM]

        # Dequantize from uint8 then requantize to 4-bit
        fp32 = (decoded_uint8.astype(np.float32) - zp8) * s8  # (n_cold, 16) fp32
        mn4, mx4 = fp32.min(), fp32.max()
        s4 = (mx4 - mn4) / 15.0
        if s4 == 0: s4 = 1.0
        zp4 = round(-mn4 / s4)
        q4 = np.clip(np.round(fp32 / s4 + zp4), 0, 15).astype(np.uint8)
        # Pack: (N, 16) -> (N, 8)
        packed = (q4[:, 0::2] << 4) | q4[:, 1::2]
        cold_4bit[t] = packed
        cold_4bit_q[t] = (float(s4), float(zp4))
        cold_idx_maps[t] = {orig: seq for seq, orig in enumerate(cold_idx)}
        log(f"    Table {t}: {n_cold:,} cold, 4bit_packed={packed.nbytes/1024/1024:.1f}MB")

    setup_time = time.time() - setup_t0

    fourbit_mem = sum(v.nbytes for v in cold_4bit.values())
    hot_mem = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables)
    small_mem = sum(state_dict[emb_keys[t]].numel() * 4
                    for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD)
    mlp_mem = sum(p.numel() * 4 for n, p in dlrm.named_parameters() if 'emb_l' not in n)
    total_mem = (fourbit_mem + hot_mem + small_mem + mlp_mem) / 1024 / 1024

    log(f"  Setup: {setup_time:.1f}s")
    log(f"  Memory: 4bit_cold={fourbit_mem/1024/1024:.1f}MB, total={total_mem:.1f}MB")

    # Run inference
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
                if t not in cold_4bit: continue
                indices = lS_i[t].numpy().flatten()
                cmap = cold_idx_maps[t]
                unique_cold = []
                seq_list = []
                seen = set()
                for i in indices:
                    ii = int(i)
                    if ii in cmap and ii not in seen:
                        unique_cold.append(ii)
                        seq_list.append(cmap[ii])
                        seen.add(ii)
                if not unique_cold: continue

                seq_arr = np.array(seq_list)
                s4, zp4 = cold_4bit_q[t]
                fp_batch = dequantize_4bit_batch(cold_4bit[t], s4, zp4, seq_arr)
                dlrm.emb_l[t].weight.data[torch.tensor(unique_cold, dtype=torch.long)] = \
                    torch.from_numpy(fp_batch)

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

    results['all_4bit_crf0'] = {
        'name': 'All cold 4-bit (via CRF 0)', 'auc': auc, 'accuracy': acc,
        'auc_loss_pp': loss, 'inference_time': infer_time, 'setup_time': setup_time,
        'total_time': setup_time + infer_time, 'memory_mb': total_mem,
        'compressed_mb': total_comp / 1024 / 1024,
        'batch_latencies': latency_stats(blats),
    }

    # ============================================================
    # EXP 3: Mixed 8b warm + 4b ice (via CRF 0 codec)
    # ============================================================
    log("\n" + "=" * 60)
    log("EXP 3: Mixed precision — 8-bit warm + 4-bit ice (CRF 0)")
    log("=" * 60)

    # Use warm_k = 500000 (from V5: 60.8% hit rate at embedding level)
    for warm_k in [500000, 1000000, 2000000]:
        log(f"\n  --- warm_k={warm_k:,} ---")
        restore_weights(dlrm, state_dict, emb_keys)
        setup_t0 = time.time()
        total_comp = setup_small_and_hot(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                                          large_tables, hot_indices)

        warm_uint8 = {}
        warm_quant = {}
        warm_maps = {}
        ice_4bit = {}
        ice_4bit_q = {}
        ice_maps = {}

        for t in large_tables:
            w = state_dict[emb_keys[t]]
            cold_sorted = freq_sorted_cold[t]
            warm_idx = cold_sorted[:min(warm_k, len(cold_sorted))]
            ice_idx = cold_sorted[min(warm_k, len(cold_sorted)):]

            # Warm: encode → CRF 0 → decode → store uint8
            if warm_idx:
                ww = w[warm_idx]
                q8, s8, zp8 = quantize_8bit(ww)
                pf = q8.numpy().reshape(-1)
                comp_data, n_f = encode_multiframe(pf, fw, fh, 0, keyint=1)
                total_comp += len(comp_data)
                frames = decode_all_frames_pyav(comp_data, fw, fh)
                dense = np.zeros((len(warm_idx), EMB_DIM), dtype=np.uint8)
                for fi, fd in enumerate(frames):
                    start = fi * epf
                    end = min(start + epf, len(warm_idx))
                    for j in range(start, end):
                        off = (j - start) * EMB_DIM
                        dense[j] = fd[off:off + EMB_DIM]
                warm_uint8[t] = dense
                warm_quant[t] = (s8, zp8)
                warm_maps[t] = {orig: seq for seq, orig in enumerate(warm_idx)}

            # Ice: encode → CRF 0 → decode → requantize to 4-bit
            if ice_idx:
                iw = w[ice_idx]
                q8, s8, zp8 = quantize_8bit(iw)
                pf = q8.numpy().reshape(-1)
                comp_data, n_f = encode_multiframe(pf, fw, fh, 0, keyint=1)
                total_comp += len(comp_data)
                frames = decode_all_frames_pyav(comp_data, fw, fh)
                n_ice = len(ice_idx)
                decoded = np.zeros((n_ice, EMB_DIM), dtype=np.uint8)
                for fi, fd in enumerate(frames):
                    start = fi * epf
                    end = min(start + epf, n_ice)
                    for j in range(start, end):
                        off = (j - start) * EMB_DIM
                        decoded[j] = fd[off:off + EMB_DIM]
                fp32 = (decoded.astype(np.float32) - zp8) * s8
                mn4, mx4 = fp32.min(), fp32.max()
                s4 = (mx4 - mn4) / 15.0
                if s4 == 0: s4 = 1.0
                zp4 = round(-mn4 / s4)
                q4 = np.clip(np.round(fp32 / s4 + zp4), 0, 15).astype(np.uint8)
                packed = (q4[:, 0::2] << 4) | q4[:, 1::2]
                ice_4bit[t] = packed
                ice_4bit_q[t] = (float(s4), float(zp4))
                ice_maps[t] = {orig: seq for seq, orig in enumerate(ice_idx)}

        setup_time = time.time() - setup_t0

        warm_mem = sum(v.nbytes for v in warm_uint8.values())
        ice_mem = sum(v.nbytes for v in ice_4bit.values())
        total_mem = (warm_mem + ice_mem + hot_mem + small_mem + mlp_mem) / 1024 / 1024

        log(f"  Setup: {setup_time:.1f}s")
        log(f"  Memory: warm_8bit={warm_mem/1024/1024:.1f}MB, ice_4bit={ice_mem/1024/1024:.1f}MB, "
            f"total={total_mem:.1f}MB")

        # Run inference
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
                    indices = lS_i[t].numpy().flatten()
                    hi_set = set(hot_indices[t].tolist())
                    seen = set()

                    w_orig, w_seq = [], []
                    i_orig, i_seq = [], []

                    for i in indices:
                        ii = int(i)
                        if ii in hi_set or ii in seen: continue
                        seen.add(ii)
                        wm = warm_maps.get(t, {})
                        im = ice_maps.get(t, {})
                        if ii in wm:
                            w_orig.append(ii); w_seq.append(wm[ii])
                        elif ii in im:
                            i_orig.append(ii); i_seq.append(im[ii])

                    if w_orig and t in warm_uint8:
                        seq_arr = np.array(w_seq)
                        sw, zw = warm_quant[t]
                        fp = dequantize_8bit_batch(warm_uint8[t], sw, zw, seq_arr)
                        dlrm.emb_l[t].weight.data[torch.tensor(w_orig, dtype=torch.long)] = \
                            torch.from_numpy(fp)

                    if i_orig and t in ice_4bit:
                        seq_arr = np.array(i_seq)
                        s4, zp4 = ice_4bit_q[t]
                        fp = dequantize_4bit_batch(ice_4bit[t], s4, zp4, seq_arr)
                        dlrm.emb_l[t].weight.data[torch.tensor(i_orig, dtype=torch.long)] = \
                            torch.from_numpy(fp)

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

        results[f'mixed_warm{warm_k}'] = {
            'name': f'Mixed 8b warm({warm_k:,}) + 4b ice (CRF 0)',
            'warm_k': warm_k, 'auc': auc, 'accuracy': acc,
            'auc_loss_pp': loss, 'inference_time': infer_time,
            'setup_time': setup_time, 'total_time': setup_time + infer_time,
            'memory_mb': total_mem,
            'warm_8bit_mb': warm_mem / 1024 / 1024,
            'ice_4bit_mb': ice_mem / 1024 / 1024,
            'batch_latencies': latency_stats(blats),
        }

    # ============================================================
    # FINAL REPORT
    # ============================================================
    log("\n" + "=" * 70)
    log("FINAL REPORT")
    log("=" * 70)

    report = []
    report.append("# Prefetch V6: Sub-byte Quantization Results\n\n")
    report.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Baseline AUC:** {baseline_auc:.6f}\n")
    report.append(f"**Baseline Memory:** {results['baseline']['memory_mb']:.1f}MB\n\n")

    report.append("## Pure Quantization Quality (no codec)\n\n")
    report.append("| Bits | AUC | AUC Loss (pp) | Cold Memory (MB) |\n")
    report.append("|------|-----|---------------|------------------|\n")
    for bits in [8, 6, 4]:
        r = results[f'pure_{bits}bit']
        report.append(f"| {bits} | {r['auc']:.6f} | {r['auc_loss_pp']:.4f} | {r['cold_memory_mb']:.0f} |\n")

    report.append("\n## Storage Layout Comparison\n\n")
    report.append("| Config | AUC | AUC Loss (pp) | Inference (s) | Setup (s) | Total (s) | Memory (MB) |\n")
    report.append("|--------|-----|---------------|--------------|----------|----------|------------|\n")

    all_keys = ['baseline', 'all_4bit_crf0']
    all_keys += [k for k in sorted(results.keys()) if k.startswith('mixed_')]
    for key in all_keys:
        r = results[key]
        if 'auc' not in r: continue
        report.append(f"| {r['name']} | {r['auc']:.6f} | {r.get('auc_loss_pp',0):.4f} | "
                      f"{r.get('inference_time',0):.1f} | {r.get('setup_time',0):.1f} | "
                      f"{r.get('total_time',0):.1f} | {r.get('memory_mb',0):.1f} |\n")

    md_path = os.path.join(RESULTS_DIR, 'prefetch_v6_subbyte.md')
    with open(md_path, 'w') as f: f.writelines(report)
    log(f"  Saved: {md_path}")

    json_path = os.path.join(RESULTS_DIR, 'prefetch_v6_subbyte.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  Saved: {json_path}")

    log("=" * 70)
    log("V6 COMPLETE!")
    log("=" * 70)


if __name__ == '__main__':
    main()
