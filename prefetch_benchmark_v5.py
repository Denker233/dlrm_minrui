#!/usr/bin/env python3
"""
Prefetch Benchmark V5: Optimize memory + AUC together.

Experiments:
  A) CRF sweep — find AUC/compression Pareto frontier
  B) Dense uint8 cold storage — no frame cache overhead, cleaner lookup
  C) Three-tier embedding — hot(fp32) + warm(uint8) + ice(compressed on-demand)
  D) Combined optimal — best CRF + best storage layout

All experiments use pre-loaded batches for fair timing.
"""

import os, sys, time, json, tempfile, subprocess, threading, io, gc
from collections import OrderedDict
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


def quantize(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def dequantize(q_tensor, s, zp):
    return (q_tensor.float() - zp) * s


def get_codec_args(crf, keyint=None):
    ki = f":keyint={keyint}" if keyint else ""
    if crf == 0:
        return ['-c:v', 'libx265', '-preset', 'ultrafast',
                '-x265-params', f'lossless=1:log-level=error{ki}']
    else:
        return ['-c:v', 'libx265', '-crf', str(crf), '-preset', 'ultrafast',
                '-x265-params', f'log-level=error:allow-non-conformance=1{ki}']


# ======== Single-frame encode/decode (for hot + small) ========
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


# ======== Multi-frame encode/decode ========
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


def decode_all_frames_pyav(comp_bytes, frame_w, frame_h, n_frames):
    """Decode all frames from compressed bytes using PyAV."""
    frame_size = frame_w * frame_h
    frames = []
    container = av.open(io.BytesIO(comp_bytes))
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format='gray').flatten()[:frame_size])
    container.close()
    return frames


def decode_specific_frames_pyav(comp_bytes, frame_w, frame_h, frame_indices):
    """Decode specific frames using PyAV (sequential scan, stop early)."""
    frame_size = frame_w * frame_h
    results = {}
    target_set = set(frame_indices)
    container = av.open(io.BytesIO(comp_bytes))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    tb = float(stream.time_base)
    for frame in container.decode(video=0):
        fn = round(frame.pts * tb * fps) if frame.pts is not None else len(results)
        if fn in target_set:
            results[fn] = frame.to_ndarray(format='gray').flatten()[:frame_size]
            if len(results) == len(target_set): break
    container.close()
    return results


# ======== Helpers ========
def restore_weights(dlrm, state_dict, emb_keys):
    with torch.no_grad():
        for k in emb_keys:
            t = int(k.split('.')[1])
            dlrm.emb_l[t].weight.data = state_dict[k].clone()


def run_inference(dlrm, all_batches):
    """Vanilla inference on pre-loaded batches."""
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


# ================================================================
# EXPERIMENT A: CRF Sweep
# ================================================================
def experiment_a_crf_sweep(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                            large_tables, hot_indices, cold_order, all_batches):
    """Test multiple CRF values for cold embeddings.
    For each CRF: encode cold -> multi-frame H.265 -> decode all -> reconstruct -> measure AUC.
    """
    log("\n" + "=" * 70)
    log("EXPERIMENT A: CRF Sweep (4K, full reconstruction)")
    log("=" * 70)

    fw, fh = 3840, 2160
    epf = (fw * fh) // EMB_DIM
    crf_values = [0, 5, 10, 15, 18, 20, 23]
    results_a = {}

    for crf in crf_values:
        log(f"\n  --- CRF {crf} ---")
        restore_weights(dlrm, state_dict, emb_keys)
        setup_t0 = time.time()
        total_comp = 0

        # Small tables: always CRF 0
        for t in range(num_tables):
            w = state_dict[emb_keys[t]]
            if w.shape[0] < LARGE_TABLE_THRESHOLD:
                q, s, zp = quantize(w)
                raw, fw2, fh2, tm = prepare_single_frame(q.numpy(), w.shape[0], EMB_DIM)
                comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
                total_comp += len(comp)
                dec = decode_single_frame_legacy(comp, fw2, fh2, w.shape[0], EMB_DIM, tm)
                with torch.no_grad():
                    dlrm.emb_l[t].weight.data = dequantize(dec, s, zp)

        # Large tables: hot CRF 0, cold at test CRF
        for t in large_tables:
            w = state_dict[emb_keys[t]]
            hi = set(hot_indices[t].tolist())

            # Hot (CRF 0 always)
            hot_idx = sorted(hi)
            if hot_idx:
                hw = w[hot_idx]; qh, sh, zh = quantize(hw)
                raw, fw2, fh2, tm = prepare_single_frame(qh.numpy(), hw.shape[0], EMB_DIM)
                comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
                total_comp += len(comp)
                dec = decode_single_frame_legacy(comp, fw2, fh2, hw.shape[0], EMB_DIM, tm)
                with torch.no_grad():
                    dlrm.emb_l[t].weight.data[hot_idx] = dequantize(dec, sh, zh)

            # Cold at test CRF
            cold_idx = cold_order[t]
            cw = w[cold_idx]
            if cw.shape[0] > 0:
                qc, sc, zc = quantize(cw)
                pf = qc.numpy().reshape(-1)
                comp_data, n_frames = encode_multiframe(pf, fw, fh, crf, keyint=1)
                total_comp += len(comp_data)

                # Decode all frames and reconstruct
                all_frames = decode_all_frames_pyav(comp_data, fw, fh, n_frames)
                all_q = []
                valid_idx = []
                for fi, fd in enumerate(all_frames):
                    start = fi * epf
                    end = min(start + epf, len(cold_idx))
                    for j in range(start, end):
                        off = (j - start) * EMB_DIM
                        all_q.append(fd[off:off + EMB_DIM])
                        valid_idx.append(cold_idx[j])
                q_arr = np.stack(all_q)
                fp_arr = (q_arr.astype(np.float32) - zc) * sc
                with torch.no_grad():
                    dlrm.emb_l[t].weight.data[torch.tensor(valid_idx, dtype=torch.long)] = \
                        torch.from_numpy(fp_arr)

        setup_time = time.time() - setup_t0

        # Run inference
        acc, auc, infer_time, blats = run_inference(dlrm, all_batches)
        log(f"    AUC={auc:.6f}, loss={((0.802698-auc)*100):.4f}pp, "
            f"compressed={total_comp/1024/1024:.2f}MB, "
            f"setup={setup_time:.1f}s, infer={infer_time:.1f}s")

        results_a[crf] = {
            'crf': crf, 'auc': auc, 'accuracy': acc,
            'auc_loss_pp': (0.802698 - auc) * 100,
            'compressed_mb': total_comp / 1024 / 1024,
            'setup_time': setup_time, 'inference_time': infer_time,
            'total_time': setup_time + infer_time,
            'batch_latencies': latency_stats(blats),
        }

    return results_a


# ================================================================
# EXPERIMENT B: Dense uint8 Cold Storage
# ================================================================
def experiment_b_dense_uint8(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                              large_tables, hot_indices, cold_order, all_batches,
                              cold_crf=0):
    """
    Store cold embeddings as dense uint8 tensors (no frame cache).
    Per-batch: index uint8 tensor, dequantize, vectorized inject.
    """
    log("\n" + "=" * 70)
    log(f"EXPERIMENT B: Dense uint8 cold storage (CRF {cold_crf})")
    log("=" * 70)

    fw, fh = 3840, 2160
    epf = (fw * fh) // EMB_DIM
    restore_weights(dlrm, state_dict, emb_keys)
    setup_t0 = time.time()
    total_comp = 0

    # Small tables (CRF 0)
    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        if w.shape[0] < LARGE_TABLE_THRESHOLD:
            q, s, zp = quantize(w)
            raw, fw2, fh2, tm = prepare_single_frame(q.numpy(), w.shape[0], EMB_DIM)
            comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
            total_comp += len(comp)
            dec = decode_single_frame_legacy(comp, fw2, fh2, w.shape[0], EMB_DIM, tm)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = dequantize(dec, s, zp)

    # Large tables: hot fp32, cold as dense uint8
    cold_uint8 = {}      # {table: np.ndarray (n_cold, 16) uint8}
    cold_quant = {}      # {table: (scale, zp)}
    cold_idx_maps = {}   # {table: {orig_idx: seq_idx}}

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())

        # Hot (CRF 0)
        hot_idx = sorted(hi)
        if hot_idx:
            hw = w[hot_idx]; qh, sh, zh = quantize(hw)
            raw, fw2, fh2, tm = prepare_single_frame(qh.numpy(), hw.shape[0], EMB_DIM)
            comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
            total_comp += len(comp)
            dec = decode_single_frame_legacy(comp, fw2, fh2, hw.shape[0], EMB_DIM, tm)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[hot_idx] = dequantize(dec, sh, zh)

        # Cold: encode -> decode -> extract into dense uint8
        cold_idx = cold_order[t]
        cw = w[cold_idx]
        if cw.shape[0] > 0:
            qc, sc, zc = quantize(cw)
            pf = qc.numpy().reshape(-1)
            comp_data, n_frames = encode_multiframe(pf, fw, fh, cold_crf, keyint=1)
            total_comp += len(comp_data)

            all_frames = decode_all_frames_pyav(comp_data, fw, fh, n_frames)
            # Extract embeddings into dense array
            n_cold = len(cold_idx)
            dense = np.zeros((n_cold, EMB_DIM), dtype=np.uint8)
            for fi, fd in enumerate(all_frames):
                start = fi * epf
                end = min(start + epf, n_cold)
                for j in range(start, end):
                    off = (j - start) * EMB_DIM
                    dense[j] = fd[off:off + EMB_DIM]

            cold_uint8[t] = dense
            cold_quant[t] = (sc, zc)
            cold_idx_maps[t] = {orig: seq for seq, orig in enumerate(cold_idx)}
            log(f"    Table {t}: {n_cold:,} cold, dense uint8={dense.nbytes/1024/1024:.1f}MB, "
                f"compressed={len(comp_data)/1024:.1f}KB")

    setup_time = time.time() - setup_t0
    uint8_mem = sum(v.nbytes for v in cold_uint8.values())
    hot_mem = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables)
    small_mem = sum(state_dict[emb_keys[t]].numel() * 4
                    for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD)
    mlp_mem = sum(p.numel() * 4 for n, p in dlrm.named_parameters() if 'emb_l' not in n)
    total_mem = (uint8_mem + hot_mem + small_mem + mlp_mem) / 1024 / 1024

    log(f"  Setup: {setup_time:.1f}s")
    log(f"  Memory: uint8_cold={uint8_mem/1024/1024:.1f}MB, hot={hot_mem/1024/1024:.1f}MB, "
        f"small={small_mem/1024/1024:.1f}MB, mlp={mlp_mem/1024/1024:.1f}MB, total={total_mem:.1f}MB")
    log(f"  Compressed on disk: {total_comp/1024/1024:.2f}MB")

    # Run inference with per-batch uint8 lookup + dequantize + inject
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    nb = len(all_batches)

    drop_caches(); time.sleep(1)
    t0 = time.time()
    with torch.no_grad():
        for bi in range(nb):
            X, lS_o, lS_i, T = all_batches[bi]
            bt0 = time.time()

            # Inject cold embeddings from uint8 storage
            for t in large_tables:
                if t not in cold_uint8: continue
                indices = lS_i[t].numpy().flatten()
                cmap = cold_idx_maps[t]
                # Find unique cold indices
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

                # Batch uint8 lookup + dequantize
                seq_arr = np.array(seq_list)
                q_batch = cold_uint8[t][seq_arr]  # (N, 16) uint8
                sc, zc = cold_quant[t]
                fp_batch = (q_batch.astype(np.float32) - zc) * sc
                idx_tensor = torch.tensor(unique_cold, dtype=torch.long)
                dlrm.emb_l[t].weight.data[idx_tensor] = torch.from_numpy(fp_batch)

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

    log(f"  Acc={acc*100:.4f}%, AUC={auc:.6f}")
    log(f"  Inference={infer_time:.1f}s, Setup={setup_time:.1f}s, Total={setup_time+infer_time:.1f}s")
    log(f"  Batch: avg={np.mean(blats)*1000:.1f}ms, p50={np.percentile(blats,50)*1000:.1f}ms")

    return {
        'name': f'Dense uint8 (CRF {cold_crf})', 'auc': auc, 'accuracy': acc,
        'auc_loss_pp': (0.802698 - auc) * 100,
        'inference_time': infer_time, 'setup_time': setup_time,
        'total_time': setup_time + infer_time,
        'memory_mb': total_mem, 'compressed_mb': total_comp / 1024 / 1024,
        'uint8_cold_mb': uint8_mem / 1024 / 1024,
        'batch_latencies': latency_stats(blats),
    }


# ================================================================
# EXPERIMENT C: Three-tier embedding
# ================================================================
def experiment_c_three_tier(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                             large_tables, hot_indices, freq_sorted_cold,
                             access_counts, all_batches, cold_crf=0):
    """
    Three tiers:
      1. Hot: fp32 in weight.data (top 80% of accesses)
      2. Warm-cold: uint8 tensor in memory (most-accessed cold, variable K)
      3. Ice-cold: H.265 compressed bytes, decoded on demand (rest)

    First simulate to find best K, then run full inference.
    """
    log("\n" + "=" * 70)
    log(f"EXPERIMENT C: Three-tier embedding (CRF {cold_crf})")
    log("=" * 70)

    fw, fh = 3840, 2160
    epf = (fw * fh) // EMB_DIM
    nb = len(all_batches)

    # Determine how many unique cold embeddings are accessed per batch
    # and what fraction the top-K captures
    log("  Profiling per-batch cold access coverage...")
    warm_sizes = [10000, 50000, 100000, 200000, 500000]  # embeddings per table

    for warm_k in warm_sizes:
        # Build warm set per table
        warm_sets = {}
        for t in large_tables:
            warm_sets[t] = set(freq_sorted_cold[t][:min(warm_k, len(freq_sorted_cold[t]))])

        total_cold_accesses = 0
        warm_hits = 0
        ice_misses_per_batch = []

        for bi in range(nb):
            _, _, lS_i, _ = all_batches[bi]
            batch_ice = 0
            for t in large_tables:
                hi_set = set(hot_indices[t].tolist())
                indices = lS_i[t].numpy().flatten()
                for i in indices:
                    ii = int(i)
                    if ii not in hi_set and ii < ln_emb[t]:
                        total_cold_accesses += 1
                        if ii in warm_sets[t]:
                            warm_hits += 1
                        else:
                            batch_ice += 1
            ice_misses_per_batch.append(batch_ice)

        hit_rate = warm_hits / total_cold_accesses if total_cold_accesses > 0 else 0
        warm_mem = warm_k * EMB_DIM * len(large_tables) / 1024 / 1024
        avg_ice = np.mean(ice_misses_per_batch) if ice_misses_per_batch else 0
        log(f"    warm_k={warm_k:,}: hit={hit_rate:.4f}, warm_mem~{warm_mem:.0f}MB, "
            f"avg_ice_misses/batch={avg_ice:.1f}")

    # Find smallest warm_k with >99% hit rate
    best_k = None
    for warm_k in warm_sizes:
        warm_sets = {}
        for t in large_tables:
            warm_sets[t] = set(freq_sorted_cold[t][:min(warm_k, len(freq_sorted_cold[t]))])
        total = 0; hits = 0
        for bi in range(min(200, nb)):  # sample
            _, _, lS_i, _ = all_batches[bi]
            for t in large_tables:
                hi_set = set(hot_indices[t].tolist())
                for i in lS_i[t].numpy().flatten():
                    ii = int(i)
                    if ii not in hi_set and ii < ln_emb[t]:
                        total += 1
                        if ii in warm_sets[t]: hits += 1
        hr = hits / total if total > 0 else 0
        if hr >= 0.99:
            best_k = warm_k
            break

    if best_k is None:
        best_k = warm_sizes[-1]
        log(f"  No warm_k achieves 99%. Using largest: {best_k}")
    else:
        log(f"  Best warm_k for 99% hit: {best_k:,}")

    # Run full inference with three-tier
    log(f"\n  Running three-tier inference with warm_k={best_k:,}...")
    restore_weights(dlrm, state_dict, emb_keys)
    setup_t0 = time.time()
    total_comp = 0

    # Small tables
    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        if w.shape[0] < LARGE_TABLE_THRESHOLD:
            q, s, zp = quantize(w)
            raw, fw2, fh2, tm = prepare_single_frame(q.numpy(), w.shape[0], EMB_DIM)
            comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
            total_comp += len(comp)
            dec = decode_single_frame_legacy(comp, fw2, fh2, w.shape[0], EMB_DIM, tm)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = dequantize(dec, s, zp)

    warm_uint8 = {}    # {table: np.ndarray (warm_k, 16)}
    warm_quant = {}    # {table: (scale, zp)}
    warm_orig_to_seq = {}  # {table: {orig: seq}}
    ice_comp = {}      # {table: compressed_bytes}
    ice_quant = {}     # {table: (scale, zp, n_cold)}
    ice_idx_maps = {}  # {table: {orig: seq_in_ice}}
    ice_n_frames = {}

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())

        # Hot (CRF 0)
        hot_idx = sorted(hi)
        if hot_idx:
            hw = w[hot_idx]; qh, sh, zh = quantize(hw)
            raw, fw2, fh2, tm = prepare_single_frame(qh.numpy(), hw.shape[0], EMB_DIM)
            comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
            total_comp += len(comp)
            dec = decode_single_frame_legacy(comp, fw2, fh2, hw.shape[0], EMB_DIM, tm)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[hot_idx] = dequantize(dec, sh, zh)

        # Split cold into warm + ice
        cold_sorted = freq_sorted_cold[t]
        warm_idx = cold_sorted[:min(best_k, len(cold_sorted))]
        ice_idx = cold_sorted[min(best_k, len(cold_sorted)):]

        # Warm: quantize, encode with CRF, decode, store as uint8
        if warm_idx:
            ww = w[warm_idx]; qw, sw, zw = quantize(ww)
            # Encode/decode through codec for paper consistency
            pf = qw.numpy().reshape(-1)
            comp_data, n_f = encode_multiframe(pf, fw, fh, cold_crf, keyint=1)
            total_comp += len(comp_data)
            all_frames = decode_all_frames_pyav(comp_data, fw, fh, n_f)
            dense = np.zeros((len(warm_idx), EMB_DIM), dtype=np.uint8)
            for fi, fd in enumerate(all_frames):
                start = fi * epf
                end = min(start + epf, len(warm_idx))
                for j in range(start, end):
                    off = (j - start) * EMB_DIM
                    dense[j] = fd[off:off + EMB_DIM]
            warm_uint8[t] = dense
            warm_quant[t] = (sw, zw)
            warm_orig_to_seq[t] = {orig: seq for seq, orig in enumerate(warm_idx)}

        # Ice: encode, keep compressed
        if ice_idx:
            iw = w[ice_idx]; qi, si, zi = quantize(iw)
            pf = qi.numpy().reshape(-1)
            comp_data, n_f = encode_multiframe(pf, fw, fh, cold_crf, keyint=1)
            total_comp += len(comp_data)
            ice_comp[t] = comp_data
            ice_quant[t] = (si, zi, len(ice_idx))
            ice_idx_maps[t] = {orig: seq for seq, orig in enumerate(ice_idx)}
            ice_n_frames[t] = n_f
            log(f"    Table {t}: warm={len(warm_idx):,} ({dense.nbytes/1024/1024:.1f}MB uint8), "
                f"ice={len(ice_idx):,} ({len(comp_data)/1024:.1f}KB compressed, {n_f} frames)")

    setup_time = time.time() - setup_t0

    warm_mem = sum(v.nbytes for v in warm_uint8.values())
    ice_mem = sum(len(v) for v in ice_comp.values())
    hot_mem = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables)
    small_mem = sum(state_dict[emb_keys[t]].numel() * 4
                    for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD)
    mlp_mem = sum(p.numel() * 4 for n, p in dlrm.named_parameters() if 'emb_l' not in n)
    total_mem = (warm_mem + ice_mem + hot_mem + small_mem + mlp_mem) / 1024 / 1024

    log(f"  Setup: {setup_time:.1f}s")
    log(f"  Memory: warm_uint8={warm_mem/1024/1024:.1f}MB, ice_comp={ice_mem/1024/1024:.1f}MB, "
        f"hot={hot_mem/1024/1024:.1f}MB, total={total_mem:.1f}MB")

    # Run inference
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    ice_decode_total = 0
    ice_miss_total = 0

    drop_caches(); time.sleep(1)
    t0 = time.time()
    with torch.no_grad():
        for bi in range(nb):
            X, lS_o, lS_i, T = all_batches[bi]
            bt0 = time.time()

            for t in large_tables:
                indices = lS_i[t].numpy().flatten()
                hi_set = set(hot_indices[t].tolist())

                # Warm lookup
                warm_map = warm_orig_to_seq.get(t, {})
                warm_orig = []
                warm_seq = []
                ice_orig = []
                ice_seq = []
                seen = set()

                for i in indices:
                    ii = int(i)
                    if ii in hi_set or ii in seen: continue
                    seen.add(ii)
                    if ii in warm_map:
                        warm_orig.append(ii)
                        warm_seq.append(warm_map[ii])
                    elif t in ice_idx_maps and ii in ice_idx_maps[t]:
                        ice_orig.append(ii)
                        ice_seq.append(ice_idx_maps[t][ii])

                # Inject warm
                if warm_orig:
                    seq_arr = np.array(warm_seq)
                    q_batch = warm_uint8[t][seq_arr]
                    sw, zw = warm_quant[t]
                    fp_batch = (q_batch.astype(np.float32) - zw) * sw
                    dlrm.emb_l[t].weight.data[torch.tensor(warm_orig, dtype=torch.long)] = \
                        torch.from_numpy(fp_batch)

                # Inject ice (on-demand decode)
                if ice_orig and t in ice_comp:
                    ice_t0 = time.time()
                    si, zi, n_ice = ice_quant[t]
                    # Find frames containing these embeddings
                    frames_needed = set()
                    for seq in ice_seq:
                        frames_needed.add(seq // epf)
                    # Decode those frames
                    decoded = decode_specific_frames_pyav(
                        ice_comp[t], fw, fh, list(frames_needed))
                    # Extract embeddings
                    fp_vals = []
                    for orig, seq in zip(ice_orig, ice_seq):
                        fn = seq // epf
                        off = (seq % epf) * EMB_DIM
                        fd = decoded.get(fn)
                        if fd is not None and off + EMB_DIM <= len(fd):
                            q = fd[off:off + EMB_DIM]
                            fp_vals.append((q.astype(np.float32) - zi) * si)
                        else:
                            fp_vals.append(np.zeros(EMB_DIM, dtype=np.float32))
                    fp_arr = np.stack(fp_vals)
                    dlrm.emb_l[t].weight.data[torch.tensor(ice_orig, dtype=torch.long)] = \
                        torch.from_numpy(fp_arr)
                    ice_decode_total += time.time() - ice_t0
                    ice_miss_total += len(ice_orig)

            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            S = Z.detach().cpu().numpy().flatten()
            Tn = T.detach().cpu().numpy().flatten()
            accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
            samp += Tn.shape[0]
            scores.extend(S.tolist()); targets.extend(Tn.tolist())

            if bi % 200 == 0:
                log(f"    Batch {bi}/{nb}, batch={blats[-1]*1000:.1f}ms, "
                    f"ice_decode_so_far={ice_decode_total:.1f}s")

    infer_time = time.time() - t0
    acc = accu / samp
    auc = roc_auc_score(targets, scores)

    log(f"  Acc={acc*100:.4f}%, AUC={auc:.6f}")
    log(f"  Inference={infer_time:.1f}s, Setup={setup_time:.1f}s, Total={setup_time+infer_time:.1f}s")
    log(f"  Ice decode: {ice_decode_total:.1f}s total, {ice_miss_total:,} ice misses")
    log(f"  Batch: avg={np.mean(blats)*1000:.1f}ms")

    return {
        'name': f'Three-tier (warm_k={best_k:,}, CRF {cold_crf})',
        'warm_k': best_k, 'auc': auc, 'accuracy': acc,
        'auc_loss_pp': (0.802698 - auc) * 100,
        'inference_time': infer_time, 'setup_time': setup_time,
        'total_time': setup_time + infer_time,
        'memory_mb': total_mem,
        'warm_uint8_mb': warm_mem / 1024 / 1024,
        'ice_compressed_mb': ice_mem / 1024 / 1024,
        'compressed_total_mb': total_comp / 1024 / 1024,
        'ice_decode_time': ice_decode_total,
        'ice_miss_count': ice_miss_total,
        'batch_latencies': latency_stats(blats),
    }


# ================================================================
# MAIN
# ================================================================
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log("=" * 70)
    log("PREFETCH BENCHMARK V5: Memory + AUC Optimization")
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
    log(f"  {len(all_batches)} batches")

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

    # Build cold orders
    cold_order_orig = {}
    freq_sorted_cold = {}
    for t in large_tables:
        hi_set = set(hot_indices[t].tolist())
        all_cold = sorted(set(range(ln_emb[t])) - hi_set)
        cold_order_orig[t] = all_cold
        counts = access_counts[t]
        freq_sorted_cold[t] = sorted(all_cold, key=lambda x: (-counts.get(x, 0), x))

    for t in large_tables:
        log(f"  Table {t}: {ln_emb[t]:,} total, {len(hot_indices[t]):,} hot, "
            f"{len(cold_order_orig[t]):,} cold")

    results = {}

    # Baseline
    log("\n" + "=" * 70)
    log("BASELINE")
    log("=" * 70)
    restore_weights(dlrm, state_dict, emb_keys)
    drop_caches(); time.sleep(1)
    a_acc, a_auc, a_time, a_blats = run_inference(dlrm, all_batches)
    a_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
    log(f"  AUC={a_auc:.6f}, Time={a_time:.1f}s, Mem={a_mem:.1f}MB")
    results['baseline'] = {
        'name': 'Baseline', 'auc': a_auc, 'accuracy': a_acc,
        'auc_loss_pp': 0, 'inference_time': a_time, 'setup_time': 0,
        'total_time': a_time, 'memory_mb': a_mem,
        'batch_latencies': latency_stats(a_blats),
    }
    baseline_auc = a_auc

    # Experiment A
    results['exp_a'] = experiment_a_crf_sweep(
        dlrm, state_dict, emb_keys, ln_emb, num_tables,
        large_tables, hot_indices, cold_order_orig, all_batches)

    # Experiment B: Dense uint8 with CRF 0
    results['exp_b_crf0'] = experiment_b_dense_uint8(
        dlrm, state_dict, emb_keys, ln_emb, num_tables,
        large_tables, hot_indices, cold_order_orig, all_batches, cold_crf=0)

    # Experiment B: Dense uint8 with CRF 23
    results['exp_b_crf23'] = experiment_b_dense_uint8(
        dlrm, state_dict, emb_keys, ln_emb, num_tables,
        large_tables, hot_indices, cold_order_orig, all_batches, cold_crf=23)

    # Experiment C: Three-tier with CRF 0
    results['exp_c'] = experiment_c_three_tier(
        dlrm, state_dict, emb_keys, ln_emb, num_tables,
        large_tables, hot_indices, freq_sorted_cold, access_counts,
        all_batches, cold_crf=0)

    # ========================================
    # FINAL REPORT
    # ========================================
    log("\n" + "=" * 70)
    log("FINAL REPORT")
    log("=" * 70)

    report = []
    report.append("# Prefetch V5: Memory + AUC Optimization\n\n")
    report.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Baseline AUC:** {baseline_auc:.6f}\n")
    report.append(f"**Baseline Inference:** {results['baseline']['inference_time']:.1f}s\n")
    report.append(f"**Baseline Memory:** {results['baseline']['memory_mb']:.1f}MB\n\n")

    # CRF sweep table
    report.append("## Experiment A: CRF Sweep (4K, full reconstruction)\n\n")
    report.append("| CRF | AUC | AUC Loss (pp) | Compressed (MB) | Setup (s) | Inference (s) | Total (s) |\n")
    report.append("|-----|-----|---------------|----------------|----------|--------------|----------|\n")
    for crf in sorted(results['exp_a'].keys()):
        r = results['exp_a'][crf]
        report.append(f"| {crf} | {r['auc']:.6f} | {r['auc_loss_pp']:.4f} | "
                      f"{r['compressed_mb']:.2f} | {r['setup_time']:.1f} | "
                      f"{r['inference_time']:.1f} | {r['total_time']:.1f} |\n")

    # Main results table
    report.append("\n## Experiment B & C: Storage Layouts\n\n")
    report.append("| Config | AUC | AUC Loss (pp) | Inference (s) | Setup (s) | Total (s) | "
                  "Memory (MB) | Compressed (MB) |\n")
    report.append("|--------|-----|---------------|--------------|----------|----------|"
                  "------------|----------------|\n")

    for key in ['baseline', 'exp_b_crf0', 'exp_b_crf23', 'exp_c']:
        r = results[key]
        if isinstance(r, dict) and 'auc' in r:
            report.append(f"| {r['name']} | {r['auc']:.6f} | {r.get('auc_loss_pp',0):.4f} | "
                          f"{r['inference_time']:.1f} | {r.get('setup_time',0):.1f} | "
                          f"{r['total_time']:.1f} | {r.get('memory_mb',0):.1f} | "
                          f"{r.get('compressed_mb', r.get('compressed_total_mb', 0)):.2f} |\n")

    report.append("\n## Key Findings\n\n")
    report.append("See JSON for full data.\n")

    md_path = os.path.join(RESULTS_DIR, 'prefetch_v5_optimization.md')
    with open(md_path, 'w') as f: f.writelines(report)
    log(f"  Saved: {md_path}")

    json_path = os.path.join(RESULTS_DIR, 'prefetch_v5_optimization.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  Saved: {json_path}")

    log("=" * 70)
    log("V5 COMPLETE!")
    log("=" * 70)


if __name__ == '__main__':
    main()
