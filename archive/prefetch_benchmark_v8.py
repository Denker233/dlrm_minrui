#!/usr/bin/env python3
"""
Prefetch Benchmark V8: Fair Comparison (No Pre-loading)

Previous versions (V4-V7) pre-loaded all test batches into memory before timing,
which artificially reduced baseline inference time from ~50-60s to ~10s. This version
streams batches from the dataloader for ALL configs, giving a fair comparison.

4 configs, back-to-back:
  1. Baseline - no compression, iterate test_ld directly
  2. 1080p prefetch (1920x1080, 20 frames/table LRU cache, CRF 23)
  3. 2K prefetch   (3072x1728, 20 frames/table LRU cache, CRF 23)
  4. 4K prefetch   (3840x2160, 20 frames/table LRU cache, CRF 23)

Prefetch uses a 1-batch lookahead buffer (Option B): read batch i+1 while
processing batch i. This is realistic for production deployment.
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

COLD_CRF = 23
CACHE_SIZE = 20  # frames per table

os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def drop_caches():
    try:
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except Exception:
        pass


# ==============================================================
# MODEL / DATA LOADING
# ==============================================================

def create_args():
    class Args:
        pass
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


# ==============================================================
# QUANTIZATION
# ==============================================================

def quantize(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def dequantize(q, s, zp):
    return (q.float() - zp) * s


# ==============================================================
# CODEC HELPERS
# ==============================================================

MIN_WIDTH = 64; MIN_HEIGHT = 64; MAX_DIM = 16384
TILING_THRESHOLD = 50000; TILE_SIZE = 4


def get_codec_args(crf, keyint=None):
    ki = f":keyint={keyint}" if keyint else ""
    if crf == 0:
        return ['-c:v', 'libx265', '-preset', 'ultrafast',
                '-x265-params', f'lossless=1:log-level=error{ki}']
    else:
        return ['-c:v', 'libx265', '-crf', str(crf), '-preset', 'ultrafast',
                '-x265-params', f'log-level=error:allow-non-conformance=1{ki}']


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
        if tp < mr:
            w, h = MIN_WIDTH, MIN_HEIGHT
        elif h > MAX_DIM:
            w = (tp + MAX_DIM - 1) // MAX_DIM; h = MAX_DIM
            if w < MIN_WIDTH:
                w = MIN_WIDTH; h = (tp + w - 1) // w
        elif w < MIN_WIDTH:
            w = MIN_WIDTH; h = (tp + w - 1) // w
            if h < MIN_HEIGHT:
                h = MIN_HEIGHT
        elif h < MIN_HEIGHT:
            h = MIN_HEIGHT; w = (tp + h - 1) // h
            if w < MIN_WIDTH:
                w, h = MIN_WIDTH, MIN_HEIGHT
        pp = w * h
        if pp > len(raw):
            p = bytearray(pp); p[:len(raw)] = raw; raw = bytes(p)
    return raw, w, h, tm


def encode_single_frame(raw, w, h, codec_args):
    with tempfile.TemporaryDirectory() as d:
        rf = os.path.join(d, 'i.raw'); vf = os.path.join(d, 'o.mp4')
        with open(rf, 'wb') as f:
            f.write(raw)
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{w}x{h}', '-r', '1', '-i', rf] + codec_args + ['-frames:v', '1', vf]
        subprocess.run(cmd, capture_output=True, check=True)
        with open(vf, 'rb') as f:
            data = f.read()
    return data


def decode_single_frame_legacy(comp, w, h, num_emb, emb_dim, tm):
    from dlrm_s_pytorch import untile_embeddings
    with tempfile.TemporaryDirectory() as d:
        vf = os.path.join(d, 'i.mp4'); rf = os.path.join(d, 'o.raw')
        with open(vf, 'wb') as f:
            f.write(comp)
        subprocess.run(['ffmpeg', '-y', '-i', vf, '-pix_fmt', 'gray', '-f', 'rawvideo', rf],
                       capture_output=True, check=True)
        px = np.fromfile(rf, dtype=np.uint8)
    if tm.get('tiled'):
        gs, ts = tm['grid_size'], tm['tile_size']
        isz = gs * ts
        px = untile_embeddings(px[:isz * isz].reshape(isz, isz),
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
        with open(rf, 'wb') as f:
            f.write(padded.tobytes())
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
               '-s', f'{frame_w}x{frame_h}', '-r', '30',
               '-i', rf] + codec_args + [vf]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"Encode fail: {r.stderr[:500]}")
        with open(vf, 'rb') as f:
            data = f.read()
    return data, n_frames


# ==============================================================
# PyAV DECODER
# ==============================================================

class PyAVDecoder:
    def __init__(self, comp_bytes, frame_w, frame_h, n_frames):
        self.comp = comp_bytes
        self.frame_w = frame_w; self.frame_h = frame_h
        self.frame_size = frame_w * frame_h
        self.n_frames = n_frames

    def decode_frames(self, frame_indices):
        if not frame_indices:
            return {}
        results = {}
        container = av.open(io.BytesIO(self.comp))
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        tb = float(stream.time_base)
        target_set = set(frame_indices)
        for frame in container.decode(video=0):
            fn = round(frame.pts * tb * fps) if frame.pts is not None else len(results)
            if fn in target_set:
                results[fn] = frame.to_ndarray(format='gray').flatten()[:self.frame_size]
                if len(results) == len(target_set):
                    break
        container.close()
        for i in frame_indices:
            if i not in results:
                results[i] = np.zeros(self.frame_size, dtype=np.uint8)
        return results

    def decode_all(self):
        frames = []
        container = av.open(io.BytesIO(self.comp))
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='gray').flatten()[:self.frame_size])
        container.close()
        return frames


# ==============================================================
# LRU CACHE
# ==============================================================

class LRUCache:
    def __init__(self, max_per_table):
        self.max_per_table = max_per_table
        self.caches = {}
        self.hits = 0; self.misses = 0; self.evictions = 0

    def get(self, table, frame):
        if table not in self.caches:
            self.caches[table] = OrderedDict()
        c = self.caches[table]
        if frame in c:
            c.move_to_end(frame); self.hits += 1; return c[frame]
        self.misses += 1; return None

    def put(self, table, frame, data):
        if table not in self.caches:
            self.caches[table] = OrderedDict()
        c = self.caches[table]
        if frame in c:
            c.move_to_end(frame); c[frame] = data
        else:
            if len(c) >= self.max_per_table:
                c.popitem(last=False); self.evictions += 1
            c[frame] = data

    @property
    def hit_rate(self):
        t = self.hits + self.misses
        return self.hits / t if t > 0 else 0.0

    @property
    def mem_bytes(self):
        return sum(d.nbytes for c in self.caches.values() for d in c.values())


# ==============================================================
# PREFETCH ENGINE
# ==============================================================

class PrefetchEngine:
    def __init__(self, decoders, quant_meta, cold_idx_maps,
                 frame_w, frame_h, cache_size, emb_dim=16):
        self.decoders = decoders
        self.quant_meta = quant_meta
        self.cold_idx_maps = cold_idx_maps
        self.frame_w = frame_w; self.frame_h = frame_h
        self.emb_dim = emb_dim
        self.embs_per_frame = (frame_w * frame_h) // emb_dim
        self.cache = LRUCache(cache_size)
        self.lock = threading.Lock()
        self._thread = None
        self._thread_result = [0.0, 0]
        self.total_decode_time = 0.0

    def get_needed_frames(self, table, indices):
        idx_map = self.cold_idx_maps.get(table, {})
        frames = set()
        for i in indices:
            ii = int(i)
            if ii in idx_map:
                frames.add(idx_map[ii] // self.embs_per_frame)
        return frames

    def _do_fetch(self, needed_by_table):
        t0 = time.time()
        n = 0
        for table, frames in needed_by_table.items():
            misses = []
            with self.lock:
                for f in frames:
                    if self.cache.get(table, f) is None:
                        misses.append(f)
            if misses:
                decoded = self.decoders[table].decode_frames(misses)
                with self.lock:
                    for fi, px in decoded.items():
                        self.cache.put(table, fi, px)
                n += len(misses)
        elapsed = time.time() - t0
        self._thread_result = [elapsed, n]
        self.total_decode_time += elapsed

    def prefetch_async(self, needed_by_table):
        if self._thread:
            self._thread.join()
        self._thread_result = [0.0, 0]
        self._thread = threading.Thread(target=self._do_fetch, args=(needed_by_table,))
        self._thread.start()

    def wait(self):
        if self._thread:
            self._thread.join()
            self._thread = None
        return self._thread_result

    def get_cold_embeddings_batch(self, table, orig_indices_unique):
        idx_map = self.cold_idx_maps[table]
        s, zp, _ = self.quant_meta[table]
        epf = self.embs_per_frame
        indices_out = []
        values_list = []
        with self.lock:
            for orig in orig_indices_unique:
                if orig not in idx_map:
                    continue
                cold_seq = idx_map[orig]
                fn = cold_seq // epf
                offset = (cold_seq % epf) * self.emb_dim
                frame_data = self.cache.get(table, fn)
                if frame_data is not None and offset + self.emb_dim <= len(frame_data):
                    q = frame_data[offset:offset + self.emb_dim]
                    indices_out.append(orig)
                    values_list.append(q)
        if not indices_out:
            return None, None
        q_arr = np.stack(values_list)
        fp_arr = (q_arr.astype(np.float32) - zp) * s
        return indices_out, torch.from_numpy(fp_arr)


# ==============================================================
# SETUP HELPERS
# ==============================================================

def setup_hot_cold_split(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                         large_tables, hot_indices, cold_order,
                         frame_w, frame_h, cold_crf=23):
    """Compress hot (CRF0) + cold (multi-frame, given CRF) for large tables.
    Small tables get CRF 0 single-frame.
    Returns: decoders, cold_idx_maps, cold_quant_meta, comp_tables, comp_bytes.
    """
    comp_bytes_total = 0

    # Small tables
    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        if w.shape[0] < LARGE_TABLE_THRESHOLD:
            q, s, zp = quantize(w)
            raw, fw2, fh2, tm = prepare_single_frame(q.numpy(), w.shape[0], EMB_DIM)
            comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
            comp_bytes_total += len(comp)
            dec = decode_single_frame_legacy(comp, fw2, fh2, w.shape[0], EMB_DIM, tm)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = dequantize(dec, s, zp)

    decoders = {}
    cold_idx_maps = {}
    cold_quant_meta = {}
    comp_tables = {}

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())

        # Hot embeddings (CRF 0)
        hot_idx = sorted(hi)
        if hot_idx:
            hw = w[hot_idx]; qh, sh, zh = quantize(hw)
            raw, fw2, fh2, tm = prepare_single_frame(qh.numpy(), hw.shape[0], EMB_DIM)
            comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
            comp_bytes_total += len(comp)
            dec = decode_single_frame_legacy(comp, fw2, fh2, hw.shape[0], EMB_DIM, tm)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[hot_idx] = dequantize(dec, sh, zh)

        # Cold embeddings (multi-frame)
        cold_idx = cold_order[t]
        cw = w[cold_idx]
        if cw.shape[0] > 0:
            qc, sc, zc = quantize(cw)
            pf = qc.numpy().reshape(-1)
            comp_data, n_frames = encode_multiframe(pf, frame_w, frame_h, cold_crf, keyint=1)
            comp_tables[t] = comp_data
            cold_quant_meta[t] = (sc, zc, cw.shape[0])
            cold_idx_maps[t] = {orig: seq for seq, orig in enumerate(cold_idx)}
            decoders[t] = PyAVDecoder(comp_data, frame_w, frame_h, n_frames)
            comp_bytes_total += len(comp_data)
            log(f"    Table {t}: {cw.shape[0]:,} cold -> {n_frames} frames, "
                f"{len(comp_data) / 1024:.1f} KB (CRF {cold_crf})")

    return decoders, cold_idx_maps, cold_quant_meta, comp_tables, comp_bytes_total


def compute_memory(dlrm, hot_indices, large_tables, num_tables, ln_emb,
                   emb_keys, state_dict, comp_tables, cache_bytes=0):
    hot_mb = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables) / 1024 / 1024
    small_mb = sum(state_dict[emb_keys[t]].numel() * 4
                   for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD) / 1024 / 1024
    mlp_mb = sum(p.numel() * 4 for n, p in dlrm.named_parameters()
                 if 'emb_l' not in n) / 1024 / 1024
    comp_cold_mb = sum(len(v) for v in comp_tables.values()) / 1024 / 1024
    cache_mb = cache_bytes / 1024 / 1024
    return {
        'hot_fp32_mb': hot_mb, 'small_fp32_mb': small_mb, 'mlp_mb': mlp_mb,
        'compressed_cold_mb': comp_cold_mb, 'frame_cache_mb': cache_mb,
        'total_mb': hot_mb + small_mb + mlp_mb + comp_cold_mb + cache_mb,
    }


def latency_stats(blats):
    a = np.array(blats)
    return {
        'count': len(a), 'mean_ms': float(np.mean(a) * 1000),
        'p50_ms': float(np.percentile(a, 50) * 1000),
        'p95_ms': float(np.percentile(a, 95) * 1000),
        'p99_ms': float(np.percentile(a, 99) * 1000),
    }


def restore_weights(dlrm, state_dict, emb_keys):
    with torch.no_grad():
        for k in emb_keys:
            t = int(k.split('.')[1])
            dlrm.emb_l[t].weight.data = state_dict[k].clone()


# ==============================================================
# INFERENCE: BASELINE (streaming from dataloader)
# ==============================================================

def run_baseline_inference(dlrm, test_ld):
    """Vanilla inference streaming from dataloader — no pre-loading."""
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    nb = 0
    t0 = time.time()
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_ld:
            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            S = Z.detach().cpu().numpy().flatten()
            Tn = T.detach().cpu().numpy().flatten()
            accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
            samp += Tn.shape[0]
            scores.extend(S.tolist()); targets.extend(Tn.tolist())
            nb += 1
            if nb % 500 == 0:
                log(f"    Batch {nb}, lat={blats[-1] * 1000:.1f}ms")
    total = time.time() - t0
    acc = accu / samp
    auc = roc_auc_score(targets, scores)
    return acc, auc, total, blats, nb


# ==============================================================
# INFERENCE: PREFETCH (streaming with 1-batch lookahead)
# ==============================================================

def run_prefetch_inference(dlrm, test_ld, engine, large_tables, cold_idx_maps):
    """Inference with prefetch, streaming from dataloader.

    Uses a 1-batch lookahead: while processing batch i, we have already
    read batch i+1 from the dataloader and started prefetching its frames.
    """
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    dec_times = []
    miss_counts = []
    nb = 0

    dataloader_iter = iter(test_ld)

    # Read first batch
    try:
        current_batch = next(dataloader_iter)
    except StopIteration:
        return 0, 0, 0, [], 0, [], []

    # Synchronously fetch frames for the first batch (no prefetch possible)
    X, lS_o, lS_i, T = current_batch
    needed_0 = {}
    for t in large_tables:
        if t in engine.decoders:
            frames = engine.get_needed_frames(t, lS_i[t].numpy().flatten())
            if frames:
                needed_0[t] = frames
    engine._do_fetch(needed_0)

    # Read lookahead batch and start async prefetch
    try:
        lookahead_batch = next(dataloader_iter)
        _, _, la_lS_i, _ = lookahead_batch
        needed_la = {}
        for t in large_tables:
            if t in engine.decoders:
                frames = engine.get_needed_frames(t, la_lS_i[t].numpy().flatten())
                if frames:
                    needed_la[t] = frames
        engine.prefetch_async(needed_la)
        has_lookahead = True
    except StopIteration:
        has_lookahead = False

    t0 = time.time()

    while True:
        X, lS_o, lS_i, T = current_batch
        bt0 = time.time()

        # Inject cold embeddings for current batch
        with torch.no_grad():
            for t in large_tables:
                if t not in engine.decoders:
                    continue
                indices = lS_i[t].numpy().flatten()
                cold_map = cold_idx_maps[t]
                unique_cold = list(set(int(i) for i in indices if int(i) in cold_map))
                if not unique_cold:
                    continue
                idx_list, val_tensor = engine.get_cold_embeddings_batch(t, unique_cold)
                if idx_list is not None:
                    idx_tensor = torch.tensor(idx_list, dtype=torch.long)
                    dlrm.emb_l[t].weight.data[idx_tensor] = val_tensor

        # Forward pass
        with torch.no_grad():
            Z = dlrm(X, lS_o, lS_i)
        blats.append(time.time() - bt0)

        S = Z.detach().cpu().numpy().flatten()
        Tn = T.detach().cpu().numpy().flatten()
        accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
        samp += Tn.shape[0]
        scores.extend(S.tolist()); targets.extend(Tn.tolist())
        nb += 1

        if nb % 200 == 0:
            c = engine.cache
            log(f"    Batch {nb}, hit={c.hit_rate:.4f}, "
                f"cached={sum(len(x) for x in c.caches.values())} frames, "
                f"cache={c.mem_bytes / 1024 / 1024:.1f}MB, "
                f"batch={blats[-1] * 1000:.1f}ms")

        # Advance: current <- lookahead, read next lookahead
        if not has_lookahead:
            break

        # Wait for lookahead prefetch to complete
        dt, nd = engine.wait()
        dec_times.append(dt)
        miss_counts.append(nd)

        current_batch = lookahead_batch

        # Ensure frames for new current batch are in cache (handle misses from prefetch)
        X_next, lS_o_next, lS_i_next, T_next = current_batch
        needed_now = {}
        for t in large_tables:
            if t in engine.decoders:
                frames = engine.get_needed_frames(t, lS_i_next[t].numpy().flatten())
                if frames:
                    needed_now[t] = frames
        still_missing = {}
        for t, frames in needed_now.items():
            misses = []
            with engine.lock:
                for f in frames:
                    if engine.cache.get(t, f) is None:
                        misses.append(f)
            if misses:
                still_missing[t] = set(misses)
        if still_missing:
            engine._do_fetch(still_missing)

        # Read next lookahead batch
        try:
            lookahead_batch = next(dataloader_iter)
            _, _, la_lS_i, _ = lookahead_batch
            needed_la = {}
            for t in large_tables:
                if t in engine.decoders:
                    frames = engine.get_needed_frames(t, la_lS_i[t].numpy().flatten())
                    if frames:
                        needed_la[t] = frames
            engine.prefetch_async(needed_la)
            has_lookahead = True
        except StopIteration:
            has_lookahead = False

    engine.wait()  # final cleanup
    total = time.time() - t0
    acc = accu / samp
    auc = roc_auc_score(targets, scores)
    return acc, auc, total, blats, nb, dec_times, miss_counts


# ==============================================================
# MAIN
# ==============================================================

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log("=" * 70)
    log("PREFETCH BENCHMARK V8: Fair Comparison (No Pre-loading)")
    log("=" * 70)

    log("Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")

    # Profile access patterns from training data
    log("Profiling access patterns...")
    access_raw = [None] * num_tables
    for i, batch in enumerate(train_ld):
        if i >= PROFILE_BATCHES:
            break
        _, _, lS_i, _ = batch
        for t in range(num_tables):
            idx = lS_i[t].numpy().flatten()
            if access_raw[t] is None:
                access_raw[t] = idx.copy()
            else:
                access_raw[t] = np.concatenate([access_raw[t], idx])

    hot_indices = {}
    for t in range(num_tables):
        if access_raw[t] is None or len(access_raw[t]) == 0:
            hot_indices[t] = np.array([], dtype=np.int64)
            continue
        unique, counts = np.unique(access_raw[t], return_counts=True)
        si = np.argsort(-counts)
        cum = np.cumsum(counts[si])
        cutoff = np.searchsorted(cum, cum[-1] * HOT_THRESHOLD) + 1
        hot_indices[t] = unique[si[:cutoff]]

    for t in large_tables:
        log(f"  Table {t}: {ln_emb[t]:,} total, {len(hot_indices[t]):,} hot")

    # Build cold order (sorted index order)
    cold_order = {}
    for t in large_tables:
        hi = set(hot_indices[t].tolist())
        cold_order[t] = sorted(set(range(ln_emb[t])) - hi)

    results = {}

    # ================================================================
    # CONFIG 1: BASELINE (no compression, streaming from dataloader)
    # ================================================================
    log("\n" + "=" * 60)
    log("CONFIG 1: BASELINE — streaming from dataloader, no compression")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    drop_caches(); time.sleep(1)
    gc.collect()

    b_acc, b_auc, b_time, b_blats, b_nb = run_baseline_inference(dlrm, test_ld)
    b_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
    log(f"  Acc={b_acc * 100:.4f}%, AUC={b_auc:.6f}")
    log(f"  Inference={b_time:.2f}s, Batches={b_nb}")
    log(f"  Memory={b_mem:.1f}MB (all fp32)")
    log(f"  Batch: mean={np.mean(b_blats) * 1000:.1f}ms, "
        f"p50={np.percentile(b_blats, 50) * 1000:.1f}ms, "
        f"p95={np.percentile(b_blats, 95) * 1000:.1f}ms, "
        f"p99={np.percentile(b_blats, 99) * 1000:.1f}ms")
    log(f"  Throughput: {b_nb / b_time:.1f} batches/s, "
        f"{b_nb * TEST_BATCH_SIZE / b_time:.0f} samples/s")

    baseline_auc = b_auc
    baseline_time = b_time

    results['baseline'] = {
        'name': 'Baseline (streaming, no compression)',
        'accuracy': b_acc, 'auc': b_auc,
        'auc_loss_pp': 0.0,
        'inference_time': b_time, 'setup_time': 0, 'total_time': b_time,
        'num_batches': b_nb,
        'memory': {
            'hot_fp32_mb': 0, 'small_fp32_mb': 0, 'mlp_mb': 0,
            'compressed_cold_mb': 0, 'frame_cache_mb': 0,
            'total_mb': b_mem,
        },
        'batch_latency': latency_stats(b_blats),
        'cache_hit_rate': None, 'cache_evictions': None, 'total_decode_time': None,
        'throughput_batches_per_sec': b_nb / b_time,
        'throughput_samples_per_sec': b_nb * TEST_BATCH_SIZE / b_time,
    }

    # ================================================================
    # PREFETCH CONFIGS: 1080p, 2K, 4K
    # ================================================================
    prefetch_configs = [
        ('1080p', 1920, 1080),
        ('2K',    3072, 1728),
        ('4K',    3840, 2160),
    ]

    for cfg_name, fw, fh in prefetch_configs:
        log("\n" + "=" * 60)
        log(f"CONFIG: {cfg_name} prefetch ({fw}x{fh}, cache={CACHE_SIZE}, CRF {COLD_CRF})")
        log("=" * 60)

        restore_weights(dlrm, state_dict, emb_keys)
        gc.collect()

        # Setup: encode + decode hot/small tables
        setup_t0 = time.time()
        decoders, cold_idx_maps, cold_quant_meta, comp_tables, comp_bytes = \
            setup_hot_cold_split(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                                large_tables, hot_indices, cold_order,
                                fw, fh, cold_crf=COLD_CRF)
        setup_time = time.time() - setup_t0
        log(f"  Setup: {setup_time:.2f}s")

        engine = PrefetchEngine(decoders, cold_quant_meta, cold_idx_maps,
                                fw, fh, CACHE_SIZE, EMB_DIM)
        drop_caches(); time.sleep(1)
        gc.collect()

        p_acc, p_auc, p_time, p_blats, p_nb, p_dec, p_miss = \
            run_prefetch_inference(dlrm, test_ld, engine, large_tables, cold_idx_maps)

        mem = compute_memory(dlrm, hot_indices, large_tables, num_tables, ln_emb,
                             emb_keys, state_dict, comp_tables, engine.cache.mem_bytes)
        auc_loss = (baseline_auc - p_auc) * 100

        log(f"  Acc={p_acc * 100:.4f}%, AUC={p_auc:.6f}, AUC loss={auc_loss:.4f}pp")
        log(f"  Inference={p_time:.2f}s, Setup={setup_time:.2f}s, "
            f"Total={setup_time + p_time:.2f}s")
        log(f"  Cache hit={engine.cache.hit_rate:.4f}, "
            f"evictions={engine.cache.evictions:,}, "
            f"decode_time={engine.total_decode_time:.2f}s")
        log(f"  Memory: hot={mem['hot_fp32_mb']:.1f}MB, small={mem['small_fp32_mb']:.1f}MB, "
            f"cold_comp={mem['compressed_cold_mb']:.1f}MB, "
            f"cache={mem['frame_cache_mb']:.1f}MB, mlp={mem['mlp_mb']:.1f}MB, "
            f"total={mem['total_mb']:.1f}MB")
        log(f"  Batch: mean={np.mean(p_blats) * 1000:.1f}ms, "
            f"p50={np.percentile(p_blats, 50) * 1000:.1f}ms, "
            f"p95={np.percentile(p_blats, 95) * 1000:.1f}ms, "
            f"p99={np.percentile(p_blats, 99) * 1000:.1f}ms")
        log(f"  Throughput: {p_nb / p_time:.1f} batches/s, "
            f"{p_nb * TEST_BATCH_SIZE / p_time:.0f} samples/s")

        key = f"prefetch_{cfg_name.lower()}"
        results[key] = {
            'name': f'{cfg_name} prefetch ({fw}x{fh})',
            'resolution': f'{fw}x{fh}',
            'accuracy': p_acc, 'auc': p_auc,
            'auc_loss_pp': auc_loss,
            'inference_time': p_time, 'setup_time': setup_time,
            'total_time': setup_time + p_time,
            'num_batches': p_nb,
            'memory': mem,
            'batch_latency': latency_stats(p_blats),
            'cache_hit_rate': engine.cache.hit_rate,
            'cache_evictions': engine.cache.evictions,
            'total_decode_time': engine.total_decode_time,
            'throughput_batches_per_sec': p_nb / p_time,
            'throughput_samples_per_sec': p_nb * TEST_BATCH_SIZE / p_time,
        }

    # ================================================================
    # SUMMARY
    # ================================================================
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    report = []
    report.append("# Prefetch V8: Fair Benchmark (No Pre-loading)\n\n")
    report.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Batch Size:** {TEST_BATCH_SIZE}\n")
    report.append(f"**Cold CRF:** {COLD_CRF}\n")
    report.append(f"**Cache Size:** {CACHE_SIZE} frames/table\n")
    report.append(f"**Baseline AUC:** {baseline_auc:.6f}\n")
    report.append(f"**Baseline Inference:** {baseline_time:.2f}s (streaming from dataloader)\n\n")

    report.append("## Results\n\n")
    report.append("| Config | AUC | AUC Loss (pp) | Inference (s) | Setup (s) | Total (s) | "
                  "Slowdown | Memory (MB) | Cache Hit | Evictions | Decode (s) |\n")
    report.append("|--------|-----|---------------|--------------|----------|----------|"
                  "----------|------------|-----------|-----------|------------|\n")

    for key in ['baseline', 'prefetch_1080p', 'prefetch_2k', 'prefetch_4k']:
        r = results[key]
        al = r['auc_loss_pp']
        sd = r['total_time'] / baseline_time if baseline_time > 0 else 0
        ch = f"{r['cache_hit_rate']:.4f}" if r['cache_hit_rate'] is not None else "N/A"
        ev = f"{r['cache_evictions']:,}" if r['cache_evictions'] is not None else "N/A"
        dt = f"{r['total_decode_time']:.2f}" if r['total_decode_time'] is not None else "N/A"
        mem_total = r['memory']['total_mb'] if isinstance(r['memory'], dict) else r['memory']
        report.append(f"| {r['name']} | {r['auc']:.6f} | {al:.4f} | "
                      f"{r['inference_time']:.2f} | {r.get('setup_time', 0):.2f} | "
                      f"{r['total_time']:.2f} | {sd:.2f}x | "
                      f"{mem_total:.1f} | {ch} | {ev} | {dt} |\n")

    report.append("\n## Batch Latency\n\n")
    report.append("| Config | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | "
                  "Batches/s | Samples/s |\n")
    report.append("|--------|-----------|----------|----------|----------|"
                  "-----------|------------|\n")
    for key in ['baseline', 'prefetch_1080p', 'prefetch_2k', 'prefetch_4k']:
        r = results[key]
        bl = r['batch_latency']
        report.append(f"| {r['name']} | {bl['mean_ms']:.2f} | {bl['p50_ms']:.2f} | "
                      f"{bl['p95_ms']:.2f} | {bl['p99_ms']:.2f} | "
                      f"{r['throughput_batches_per_sec']:.1f} | "
                      f"{r['throughput_samples_per_sec']:.0f} |\n")

    report.append("\n## Memory Breakdown (prefetch configs)\n\n")
    report.append("| Config | Hot FP32 | Small FP32 | Cold Compressed | Frame Cache | "
                  "MLP | Total |\n")
    report.append("|--------|----------|------------|-----------------|-------------|"
                  "-----|-------|\n")
    for key in ['prefetch_1080p', 'prefetch_2k', 'prefetch_4k']:
        r = results[key]
        m = r['memory']
        report.append(f"| {r['name']} | {m['hot_fp32_mb']:.1f} | {m['small_fp32_mb']:.1f} | "
                      f"{m['compressed_cold_mb']:.1f} | {m['frame_cache_mb']:.1f} | "
                      f"{m['mlp_mb']:.1f} | {m['total_mb']:.1f} |\n")

    report.append("\n## Notes\n\n")
    report.append("- All configs iterate `test_ld` directly (no pre-loading into memory)\n")
    report.append("- Prefetch uses 1-batch lookahead: read batch i+1 while processing batch i\n")
    report.append("- Hot embeddings: top 80% of training accesses, CRF 0 (lossless)\n")
    report.append(f"- Cold embeddings: multi-frame H.265, CRF {COLD_CRF}\n")
    report.append(f"- LRU cache: {CACHE_SIZE} frames per table\n")
    report.append("- Baseline inference should be ~50-60s (real dataloader overhead)\n")

    md_path = os.path.join(RESULTS_DIR, 'prefetch_v8_fair.md')
    with open(md_path, 'w') as f:
        f.writelines(report)
    log(f"  Report: {md_path}")

    json_path = os.path.join(RESULTS_DIR, 'prefetch_v8_fair.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  JSON:   {json_path}")

    # Print summary table to console
    log("\n  %-40s  %10s  %10s  %10s  %10s" % ("Config", "AUC", "Loss(pp)", "Infer(s)", "Total(s)"))
    log("  " + "-" * 84)
    for key in ['baseline', 'prefetch_1080p', 'prefetch_2k', 'prefetch_4k']:
        r = results[key]
        log("  %-40s  %10.6f  %10.4f  %10.2f  %10.2f" % (
            r['name'], r['auc'], r['auc_loss_pp'], r['inference_time'], r['total_time']))

    log("\n" + "=" * 70)
    log("ALL DONE!")
    log("=" * 70)


if __name__ == '__main__':
    main()
