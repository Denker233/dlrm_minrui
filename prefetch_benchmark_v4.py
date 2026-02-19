#!/usr/bin/env python3
"""
Prefetch Benchmark V4: 6 Improvement Methods

Tests these improvements over V3 4K/1080p prefetch:
  Method 1: Full reconstruction (decode all frames at startup, no per-batch injection)
  Method 2: (skipped — equivalent to Method 1 for D with 100% cache)
  Method 3: Resolution sweep (find sweet spot between 1080p and 4K)
  Method 4: Two-tier cold split (permanent warm-cold + LRU for rest)
  Method 5: CRF 0 for cold (lossless, reduces AUC loss)
  Method 6: Batch reordering (sort by frame-access similarity for 1080p)

Key fix from V3: baseline pre-loads all batches for fair timing comparison.
"""

import os, sys, time, json, tempfile, subprocess, threading, io, gc, psutil
from collections import OrderedDict, Counter
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
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def drop_caches():
    try:
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except:
        pass


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


# ==============================================================
# SINGLE-FRAME TILING
# ==============================================================
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


# ==============================================================
# MULTI-FRAME ENCODE / DECODE
# ==============================================================
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


class PyAVDecoder:
    def __init__(self, comp_bytes, frame_w, frame_h, n_frames):
        self.comp = comp_bytes
        self.frame_w = frame_w; self.frame_h = frame_h
        self.frame_size = frame_w * frame_h
        self.n_frames = n_frames

    def decode_frames(self, frame_indices):
        if not frame_indices: return {}
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
                if len(results) == len(target_set): break
        container.close()
        for i in frame_indices:
            if i not in results: results[i] = np.zeros(self.frame_size, dtype=np.uint8)
        return results

    def decode_all(self):
        frames = []
        container = av.open(io.BytesIO(self.comp))
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='gray').flatten()[:self.frame_size])
        container.close()
        return frames


class LRUCache:
    def __init__(self, max_per_table):
        self.max_per_table = max_per_table
        self.caches = {}
        self.hits = 0; self.misses = 0; self.evictions = 0

    def get(self, table, frame):
        if table not in self.caches: self.caches[table] = OrderedDict()
        c = self.caches[table]
        if frame in c:
            c.move_to_end(frame); self.hits += 1; return c[frame]
        self.misses += 1; return None

    def put(self, table, frame, data):
        if table not in self.caches: self.caches[table] = OrderedDict()
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
# PREFETCH ENGINE (from V3)
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
        self._thread_result = [time.time() - t0, n]

    def prefetch_async(self, needed_by_table):
        if self._thread: self._thread.join()
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
                if orig not in idx_map: continue
                cold_seq = idx_map[orig]
                fn = cold_seq // epf
                offset = (cold_seq % epf) * self.emb_dim
                frame_data = self.cache.get(table, fn)
                if frame_data is not None and offset + self.emb_dim <= len(frame_data):
                    q = frame_data[offset:offset + self.emb_dim]
                    indices_out.append(orig)
                    values_list.append(q)
        if not indices_out: return None, None
        q_arr = np.stack(values_list)
        fp_arr = (q_arr.astype(np.float32) - zp) * s
        return indices_out, torch.from_numpy(fp_arr)


# ==============================================================
# INFERENCE FUNCTIONS
# ==============================================================

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


def run_prefetch_inference(dlrm, all_batches, engine, large_tables, cold_idx_maps):
    """Inference with prefetch and vectorized injection."""
    nb = len(all_batches)
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    dec_times = []
    miss_counts = []

    # Pre-fetch batch 0
    _, _, lS_i_0, _ = all_batches[0]
    needed_0 = {}
    for t in large_tables:
        if t in engine.decoders:
            frames = engine.get_needed_frames(t, lS_i_0[t].numpy().flatten())
            if frames: needed_0[t] = frames
    engine._do_fetch(needed_0)

    # Start prefetch for batch 1
    if nb > 1:
        _, _, lS_i_1, _ = all_batches[1]
        needed_1 = {}
        for t in large_tables:
            if t in engine.decoders:
                frames = engine.get_needed_frames(t, lS_i_1[t].numpy().flatten())
                if frames: needed_1[t] = frames
        engine.prefetch_async(needed_1)

    t0 = time.time()

    for bi in range(nb):
        X, lS_o, lS_i, T = all_batches[bi]
        bt0 = time.time()

        if bi >= 1:
            dt, nd = engine.wait()
            dec_times.append(dt)
            miss_counts.append(nd)
            # Check for remaining misses
            needed_now = {}
            for t in large_tables:
                if t in engine.decoders:
                    frames = engine.get_needed_frames(t, lS_i[t].numpy().flatten())
                    if frames: needed_now[t] = frames
            still_missing = {}
            for t, frames in needed_now.items():
                misses = []
                with engine.lock:
                    for f in frames:
                        if engine.cache.get(t, f) is None:
                            misses.append(f)
                if misses: still_missing[t] = set(misses)
            if still_missing:
                engine._do_fetch(still_missing)
        else:
            dec_times.append(0)
            miss_counts.append(0)

        # Prefetch next batch
        if bi + 1 < nb:
            _, _, next_lS_i, _ = all_batches[bi + 1]
            needed_next = {}
            for t in large_tables:
                if t in engine.decoders:
                    frames = engine.get_needed_frames(t, next_lS_i[t].numpy().flatten())
                    if frames: needed_next[t] = frames
            if bi + 1 >= 2:
                engine.prefetch_async(needed_next)

        # Vectorized cold injection
        with torch.no_grad():
            for t in large_tables:
                if t not in engine.decoders: continue
                indices = lS_i[t].numpy().flatten()
                cold_map = cold_idx_maps[t]
                unique_cold = list(set(int(i) for i in indices if int(i) in cold_map))
                if not unique_cold: continue
                idx_list, val_tensor = engine.get_cold_embeddings_batch(t, unique_cold)
                if idx_list is not None:
                    idx_tensor = torch.tensor(idx_list, dtype=torch.long)
                    dlrm.emb_l[t].weight.data[idx_tensor] = val_tensor

        Z = dlrm(X, lS_o, lS_i)
        blats.append(time.time() - bt0)

        S = Z.detach().cpu().numpy().flatten()
        Tn = T.detach().cpu().numpy().flatten()
        accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
        samp += Tn.shape[0]
        scores.extend(S.tolist()); targets.extend(Tn.tolist())

        if bi % 200 == 0:
            c = engine.cache
            log(f"    Batch {bi}/{nb}, hit={c.hit_rate:.4f}, "
                f"cached={sum(len(x) for x in c.caches.values())} frames, "
                f"cache={c.mem_bytes/1024/1024:.1f}MB, "
                f"batch={blats[-1]*1000:.1f}ms")

    engine.wait()
    total = time.time() - t0
    return accu/samp, roc_auc_score(targets, scores), total, blats, dec_times, miss_counts


# ==============================================================
# SHARED SETUP HELPERS
# ==============================================================

def setup_hot_cold_split(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                         large_tables, hot_indices, cold_order,
                         frame_w, frame_h, cold_crf=23):
    """Compress hot (CRF0) + cold (multi-frame, given CRF) for large tables.
    Small tables get CRF 0 single-frame.
    Returns: decoders, cold_idx_maps, cold_quant_meta, comp_tables, comp_bytes.
    """
    comp_bytes_total = 0
    epf = (frame_w * frame_h) // EMB_DIM

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
        ne = w.shape[0]
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
                f"{len(comp_data)/1024:.1f} KB (CRF {cold_crf})")

    return decoders, cold_idx_maps, cold_quant_meta, comp_tables, comp_bytes_total


def compute_memory(dlrm, hot_indices, large_tables, num_tables, ln_emb,
                   emb_keys, state_dict, comp_tables, cache_bytes=0):
    hot_mb = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables) / 1024 / 1024
    small_mb = sum(state_dict[emb_keys[t]].numel() * 4
                   for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD) / 1024 / 1024
    mlp_mb = sum(p.numel() * 4 for n, p in dlrm.named_parameters() if 'emb_l' not in n) / 1024 / 1024
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
        'count': len(a), 'mean': float(np.mean(a)),
        'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
        'p99': float(np.percentile(a, 99)),
    }


def restore_weights(dlrm, state_dict, emb_keys):
    with torch.no_grad():
        for k in emb_keys:
            t = int(k.split('.')[1])
            dlrm.emb_l[t].weight.data = state_dict[k].clone()


# ==============================================================
# MAIN
# ==============================================================

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log("=" * 70)
    log("PREFETCH BENCHMARK V4: 6 Improvement Methods")
    log("=" * 70)

    log("Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")

    # Pre-load ALL test batches for fair timing
    log("Pre-loading all test batches...")
    all_batches = list(test_ld)
    nb = len(all_batches)
    log(f"  {nb} batches pre-loaded")

    # Profile access patterns
    log("Profiling access patterns...")
    access_raw = [None] * num_tables
    for i, batch in enumerate(train_ld):
        if i >= PROFILE_BATCHES: break
        _, _, lS_i, _ = batch
        for t in range(num_tables):
            idx = lS_i[t].numpy().flatten()
            if access_raw[t] is None:
                access_raw[t] = idx.copy()
            else:
                access_raw[t] = np.concatenate([access_raw[t], idx])

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

    for t in large_tables:
        log(f"  Table {t}: {ln_emb[t]:,} total, {len(hot_indices[t]):,} hot")

    # Build cold order (original = sorted index order)
    cold_order_orig = {}
    for t in large_tables:
        hi = set(hot_indices[t].tolist())
        cold_order_orig[t] = sorted(set(range(ln_emb[t])) - hi)

    # Pre-build idx_maps once (used by multiple methods)
    cold_idx_maps_prebuilt = {}
    for t in large_tables:
        cold_idx_maps_prebuilt[t] = {orig: seq for seq, orig in enumerate(cold_order_orig[t])}

    results = {}

    # ================================================================
    # BASELINE (pre-loaded batches)
    # ================================================================
    log("\n" + "=" * 60)
    log("BASELINE: Pre-loaded batches, original weights")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    drop_caches(); time.sleep(1)
    a_acc, a_auc, a_time, a_blats = run_inference(dlrm, all_batches)
    a_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
    log(f"  Acc={a_acc*100:.4f}%, AUC={a_auc:.6f}, Time={a_time:.2f}s, Mem={a_mem:.1f}MB")
    log(f"  Batch: avg={np.mean(a_blats)*1000:.1f}ms, p50={np.percentile(a_blats,50)*1000:.1f}ms")

    results['baseline'] = {
        'name': 'Baseline (pre-loaded)', 'accuracy': a_acc, 'auc': a_auc,
        'inference_time': a_time, 'setup_time': 0, 'total_time': a_time,
        'memory_mb': a_mem, 'batch_latencies': latency_stats(a_blats),
    }
    baseline_auc = a_auc
    baseline_time = a_time

    # ================================================================
    # V3-STYLE 4K PREFETCH (reference, pre-loaded batches)
    # ================================================================
    log("\n" + "=" * 60)
    log("REFERENCE: V3-style 4K prefetch (vectorized inject, pre-loaded)")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    fw, fh = 3840, 2160
    setup_t0 = time.time()
    decoders, cold_idx_maps, cold_quant_meta, comp_tables, _ = setup_hot_cold_split(
        dlrm, state_dict, emb_keys, ln_emb, num_tables,
        large_tables, hot_indices, cold_order_orig, fw, fh, cold_crf=23)
    setup_time = time.time() - setup_t0
    log(f"  Setup: {setup_time:.2f}s")

    engine = PrefetchEngine(decoders, cold_quant_meta, cold_idx_maps, fw, fh, 20, EMB_DIM)
    drop_caches(); time.sleep(1)
    ref_acc, ref_auc, ref_time, ref_blats, ref_dec, ref_miss = run_prefetch_inference(
        dlrm, all_batches, engine, large_tables, cold_idx_maps)
    mem = compute_memory(dlrm, hot_indices, large_tables, num_tables, ln_emb,
                         emb_keys, state_dict, comp_tables, engine.cache.mem_bytes)
    log(f"  Acc={ref_acc*100:.4f}%, AUC={ref_auc:.6f}")
    log(f"  Inference={ref_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time+ref_time:.2f}s")
    log(f"  Cache hit={engine.cache.hit_rate:.4f}, Mem={mem['total_mb']:.1f}MB")
    log(f"  Batch: avg={np.mean(ref_blats)*1000:.1f}ms")

    results['ref_4k'] = {
        'name': 'V3 4K prefetch (reference)', 'accuracy': ref_acc, 'auc': ref_auc,
        'inference_time': ref_time, 'setup_time': setup_time,
        'total_time': setup_time + ref_time, 'memory_mb': mem['total_mb'],
        'cache_hit_rate': engine.cache.hit_rate,
        'cache_evictions': engine.cache.evictions,
        'batch_latencies': latency_stats(ref_blats),
        'total_decode_time': float(np.sum(ref_dec)),
    }

    # ================================================================
    # METHOD 1: Full reconstruction (no per-batch injection)
    # ================================================================
    log("\n" + "=" * 60)
    log("METHOD 1: Full reconstruction from 4K frames")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    fw, fh = 3840, 2160
    epf = (fw * fh) // EMB_DIM
    setup_t0 = time.time()

    # Encode and immediately decode ALL cold frames, reconstruct full tables
    comp_tables_m1 = {}
    comp_bytes_m1 = 0

    # Small tables (same as before)
    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        if w.shape[0] < LARGE_TABLE_THRESHOLD:
            q, s, zp = quantize(w)
            raw, fw2, fh2, tm = prepare_single_frame(q.numpy(), w.shape[0], EMB_DIM)
            comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
            comp_bytes_m1 += len(comp)
            dec = decode_single_frame_legacy(comp, fw2, fh2, w.shape[0], EMB_DIM, tm)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = dequantize(dec, s, zp)

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        ne = w.shape[0]
        hi = set(hot_indices[t].tolist())

        # Hot (CRF 0) - same as before
        hot_idx = sorted(hi)
        if hot_idx:
            hw = w[hot_idx]; qh, sh, zh = quantize(hw)
            raw, fw2, fh2, tm = prepare_single_frame(qh.numpy(), hw.shape[0], EMB_DIM)
            comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
            comp_bytes_m1 += len(comp)
            dec = decode_single_frame_legacy(comp, fw2, fh2, hw.shape[0], EMB_DIM, tm)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[hot_idx] = dequantize(dec, sh, zh)

        # Cold: encode to multi-frame, then decode ALL and reconstruct
        cold_idx = cold_order_orig[t]
        cw = w[cold_idx]
        if cw.shape[0] > 0:
            qc, sc, zc = quantize(cw)
            pf = qc.numpy().reshape(-1)
            comp_data, n_frames = encode_multiframe(pf, fw, fh, 23, keyint=1)
            comp_tables_m1[t] = comp_data
            comp_bytes_m1 += len(comp_data)
            log(f"    Table {t}: {cw.shape[0]:,} cold -> {n_frames} frames, {len(comp_data)/1024:.1f} KB")

            # Decode all frames and write into weight.data
            decoder = PyAVDecoder(comp_data, fw, fh, n_frames)
            all_frames = decoder.decode_all()
            # Vectorized reconstruction: build full array then scatter write
            all_q = []
            valid_idx = []
            for fi, frame_data in enumerate(all_frames):
                start_emb = fi * epf
                end_emb = min(start_emb + epf, len(cold_idx))
                for j in range(start_emb, end_emb):
                    offset = (j - start_emb) * EMB_DIM
                    q = frame_data[offset:offset + EMB_DIM]
                    all_q.append(q)
                    valid_idx.append(cold_idx[j])
            # Single vectorized dequantize + write
            q_arr = np.stack(all_q)  # (N, 16)
            fp_arr = (q_arr.astype(np.float32) - zc) * sc
            with torch.no_grad():
                idx_tensor = torch.tensor(valid_idx, dtype=torch.long)
                dlrm.emb_l[t].weight.data[idx_tensor] = torch.from_numpy(fp_arr)

    setup_time = time.time() - setup_t0
    log(f"  Full reconstruction: {setup_time:.2f}s")

    # Now run vanilla inference — no injection needed
    m1_acc, m1_auc, m1_time, m1_blats = run_inference(dlrm, all_batches)
    m1_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
    log(f"  Acc={m1_acc*100:.4f}%, AUC={m1_auc:.6f}")
    log(f"  Inference={m1_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time+m1_time:.2f}s")
    log(f"  Memory (runtime): {m1_mem:.1f}MB, Compressed on disk: {comp_bytes_m1/1024/1024:.2f}MB")
    log(f"  Batch: avg={np.mean(m1_blats)*1000:.1f}ms")

    results['method1'] = {
        'name': 'Method 1: Full reconstruction (4K)', 'accuracy': m1_acc, 'auc': m1_auc,
        'inference_time': m1_time, 'setup_time': setup_time,
        'total_time': setup_time + m1_time, 'memory_mb': m1_mem,
        'compressed_mb': comp_bytes_m1 / 1024 / 1024,
        'batch_latencies': latency_stats(m1_blats),
    }

    # ================================================================
    # METHOD 3: Resolution sweep (simulation only for cache hit)
    # ================================================================
    log("\n" + "=" * 60)
    log("METHOD 3: Resolution sweep — cache hit simulation")
    log("=" * 60)

    resolutions = [
        ('1080p', 1920, 1080),
        ('1440p', 2560, 1440),
        ('2K', 3072, 1728),
        ('3K', 3840, 2160),  # same as 4K reference
    ]

    cache_sizes = [20, 30, 50]

    # Simulate LRU cache hit rates for each resolution + cache_size
    log("  Simulating cache hit rates...")
    for res_name, fw, fh in resolutions:
        epf = (fw * fh) // EMB_DIM
        for cs in cache_sizes:
            sim_cache = {}
            total_hits = 0
            total_accesses = 0
            total_evictions = 0

            for bi in range(nb):
                _, _, lS_i, _ = all_batches[bi]
                for t in large_tables:
                    idx_map = cold_idx_maps_prebuilt[t]
                    indices = lS_i[t].numpy().flatten()
                    frames_needed = set()
                    for i in indices:
                        ii = int(i)
                        if ii in idx_map:
                            frames_needed.add(idx_map[ii] // epf)

                    if not frames_needed: continue
                    if t not in sim_cache:
                        sim_cache[t] = OrderedDict()
                    c = sim_cache[t]

                    for f in frames_needed:
                        total_accesses += 1
                        if f in c:
                            c.move_to_end(f)
                            total_hits += 1
                        else:
                            if len(c) >= cs:
                                c.popitem(last=False)
                                total_evictions += 1
                            c[f] = True

            hit_rate = total_hits / total_accesses if total_accesses > 0 else 0
            max_cache_frames = sum(min(cs, (len(cold_order_orig[t]) + epf - 1) // epf) for t in large_tables)
            cache_mem_mb = max_cache_frames * fw * fh / 1024 / 1024
            n_frames_max = max((len(cold_order_orig[t]) + epf - 1) // epf for t in large_tables)

            log(f"    {res_name} cache={cs}: hit={hit_rate:.4f}, evictions={total_evictions:,}, "
                f"cache~{cache_mem_mb:.0f}MB, max_frames/table={n_frames_max}")

    # Find the best non-4K resolution that gets >99% hit rate
    # Run full inference for that config
    best_res = None
    for res_name, fw, fh in resolutions:
        if res_name == '3K': continue  # skip — same as reference
        epf = (fw * fh) // EMB_DIM
        n_frames_max = max((len(cold_order_orig[t]) + epf - 1) // epf for t in large_tables)
        # We need cache_size >= n_frames_max for 100% hit
        if n_frames_max <= 50:  # reasonable cache size
            best_res = (res_name, fw, fh, n_frames_max)
            break

    if best_res:
        res_name, fw, fh, needed_cache = best_res
        log(f"\n  Running full inference for {res_name} with cache={needed_cache}...")
        restore_weights(dlrm, state_dict, emb_keys)
        setup_t0 = time.time()
        decoders, cold_idx_maps, cold_quant_meta, comp_tables, _ = setup_hot_cold_split(
            dlrm, state_dict, emb_keys, ln_emb, num_tables,
            large_tables, hot_indices, cold_order_orig, fw, fh, cold_crf=23)
        setup_time = time.time() - setup_t0
        log(f"  Setup: {setup_time:.2f}s")

        engine = PrefetchEngine(decoders, cold_quant_meta, cold_idx_maps, fw, fh, needed_cache, EMB_DIM)
        drop_caches(); time.sleep(1)
        m3_acc, m3_auc, m3_time, m3_blats, m3_dec, m3_miss = run_prefetch_inference(
            dlrm, all_batches, engine, large_tables, cold_idx_maps)
        mem = compute_memory(dlrm, hot_indices, large_tables, num_tables, ln_emb,
                             emb_keys, state_dict, comp_tables, engine.cache.mem_bytes)
        log(f"  Acc={m3_acc*100:.4f}%, AUC={m3_auc:.6f}")
        log(f"  Inference={m3_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time+m3_time:.2f}s")
        log(f"  Cache hit={engine.cache.hit_rate:.4f}, Mem={mem['total_mb']:.1f}MB")
        log(f"  Batch: avg={np.mean(m3_blats)*1000:.1f}ms")

        results['method3'] = {
            'name': f'Method 3: {res_name} cache={needed_cache}',
            'accuracy': m3_acc, 'auc': m3_auc,
            'inference_time': m3_time, 'setup_time': setup_time,
            'total_time': setup_time + m3_time, 'memory_mb': mem['total_mb'],
            'cache_hit_rate': engine.cache.hit_rate,
            'cache_evictions': engine.cache.evictions,
            'batch_latencies': latency_stats(m3_blats),
            'total_decode_time': float(np.sum(m3_dec)),
        }
    else:
        log("  No intermediate resolution achieves >99% hit with cache<=50. Skipping full run.")
        results['method3'] = {'name': 'Method 3: SKIPPED', 'status': 'no suitable resolution'}

    # ================================================================
    # METHOD 4: Two-tier cold split for 1080p
    # ================================================================
    log("\n" + "=" * 60)
    log("METHOD 4: Two-tier cold split (1080p)")
    log("=" * 60)

    # Concept: keep top-K cold embeddings (by frequency) permanently decompressed in weight.data
    # The rest go into frame cache as usual.
    # The "warm cold" embeddings don't need frame cache — they're always in weight.data.
    # This reduces the number of frames touched per batch.

    # First: figure out optimal warm tier size using simulation
    fw, fh = 1920, 1080
    epf = (fw * fh) // EMB_DIM

    # Sort cold by frequency for each table
    freq_sorted_cold = {}
    for t in large_tables:
        hi_set = set(hot_indices[t].tolist())
        all_cold = sorted(set(range(ln_emb[t])) - hi_set)
        counts = access_counts[t]
        freq_sorted_cold[t] = sorted(all_cold, key=lambda x: (-counts.get(x, 0), x))

    warm_tiers = [1, 2, 5, 10]  # number of 1080p frames worth of "warm cold" embeddings
    best_warm = None
    best_warm_hitrate = 0

    for warm_frames in warm_tiers:
        warm_count = warm_frames * epf  # number of warm cold embeddings per table

        # Pre-build ice-cold maps (avoid rebuilding in inner loop)
        ice_cold_maps = {}
        for t in large_tables:
            cold_sorted = freq_sorted_cold[t]
            ice_start = min(warm_count, len(cold_sorted))
            ice_cold_maps[t] = {cold_sorted[i]: i - ice_start for i in range(ice_start, len(cold_sorted))}

        sim_cache = {}
        total_hits = 0
        total_accesses = 0

        for bi in range(nb):
            _, _, lS_i, _ = all_batches[bi]
            for t in large_tables:
                ice_cold_set = ice_cold_maps[t]

                indices = lS_i[t].numpy().flatten()
                frames_needed = set()
                for i in indices:
                    ii = int(i)
                    if ii in ice_cold_set:
                        frames_needed.add(ice_cold_set[ii] // epf)

                if not frames_needed: continue

                if t not in sim_cache:
                    sim_cache[t] = OrderedDict()
                c = sim_cache[t]

                for f in frames_needed:
                    total_accesses += 1
                    if f in c:
                        c.move_to_end(f)
                        total_hits += 1
                    else:
                        if len(c) >= 20:  # same cache size
                            c.popitem(last=False)
                        c[f] = True

        hit_rate = total_hits / total_accesses if total_accesses > 0 else 1.0
        warm_mem_mb = warm_count * EMB_DIM * 4 * len(large_tables) / 1024 / 1024
        log(f"  warm={warm_frames} frames ({warm_count} embs/table): "
            f"ice-cold hit={hit_rate:.4f}, warm_mem~{warm_mem_mb:.0f}MB, "
            f"accesses={total_accesses:,}")
        if hit_rate > best_warm_hitrate:
            best_warm_hitrate = hit_rate
            best_warm = warm_frames

    # Run full inference for best warm tier
    if best_warm and best_warm_hitrate > 0.9:
        warm_count = best_warm * epf
        log(f"\n  Running full inference: warm={best_warm} frames ({warm_count} embs/table)...")
        restore_weights(dlrm, state_dict, emb_keys)
        fw, fh = 1920, 1080
        setup_t0 = time.time()

        comp_bytes_m4 = 0
        # Small tables
        for t in range(num_tables):
            w = state_dict[emb_keys[t]]
            if w.shape[0] < LARGE_TABLE_THRESHOLD:
                q, s, zp = quantize(w)
                raw, fw2, fh2, tm = prepare_single_frame(q.numpy(), w.shape[0], EMB_DIM)
                comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
                comp_bytes_m4 += len(comp)
                dec = decode_single_frame_legacy(comp, fw2, fh2, w.shape[0], EMB_DIM, tm)
                with torch.no_grad():
                    dlrm.emb_l[t].weight.data = dequantize(dec, s, zp)

        decoders_m4 = {}
        cold_idx_maps_m4 = {}
        cold_quant_meta_m4 = {}
        comp_tables_m4 = {}

        for t in large_tables:
            w = state_dict[emb_keys[t]]
            ne = w.shape[0]
            hi = set(hot_indices[t].tolist())

            # Hot (CRF 0)
            hot_idx = sorted(hi)
            if hot_idx:
                hw = w[hot_idx]; qh, sh, zh = quantize(hw)
                raw, fw2, fh2, tm = prepare_single_frame(qh.numpy(), hw.shape[0], EMB_DIM)
                comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
                comp_bytes_m4 += len(comp)
                dec = decode_single_frame_legacy(comp, fw2, fh2, hw.shape[0], EMB_DIM, tm)
                with torch.no_grad():
                    dlrm.emb_l[t].weight.data[hot_idx] = dequantize(dec, sh, zh)

            # Warm cold: top warm_count, permanently decompressed
            cold_sorted = freq_sorted_cold[t]
            warm_idx = cold_sorted[:min(warm_count, len(cold_sorted))]
            if warm_idx:
                ww = w[warm_idx]; qw, sw, zw = quantize(ww)
                # Decompress warm cold with CRF 23 (same quality as other cold)
                raw, fw2, fh2, tm = prepare_single_frame(qw.numpy(), ww.shape[0], EMB_DIM)
                comp = encode_single_frame(raw, fw2, fh2, get_codec_args(23))
                comp_bytes_m4 += len(comp)
                dec = decode_single_frame_legacy(comp, fw2, fh2, ww.shape[0], EMB_DIM, tm)
                with torch.no_grad():
                    idx_t = torch.tensor(warm_idx, dtype=torch.long)
                    dlrm.emb_l[t].weight.data[idx_t] = dequantize(dec, sw, zw)

            # Ice cold: rest, into frame cache
            ice_cold_idx = cold_sorted[min(warm_count, len(cold_sorted)):]
            if len(ice_cold_idx) > 0:
                iw = w[ice_cold_idx]
                qi, si, zi = quantize(iw)
                pf = qi.numpy().reshape(-1)
                comp_data, n_frames = encode_multiframe(pf, fw, fh, 23, keyint=1)
                comp_tables_m4[t] = comp_data
                cold_quant_meta_m4[t] = (si, zi, iw.shape[0])
                cold_idx_maps_m4[t] = {orig: seq for seq, orig in enumerate(ice_cold_idx)}
                decoders_m4[t] = PyAVDecoder(comp_data, fw, fh, n_frames)
                comp_bytes_m4 += len(comp_data)
                log(f"    Table {t}: warm={len(warm_idx):,}, ice={len(ice_cold_idx):,} -> "
                    f"{n_frames} frames, {len(comp_data)/1024:.1f} KB")

        setup_time = time.time() - setup_t0
        log(f"  Setup: {setup_time:.2f}s")

        engine = PrefetchEngine(decoders_m4, cold_quant_meta_m4, cold_idx_maps_m4, fw, fh, 20, EMB_DIM)
        drop_caches(); time.sleep(1)
        m4_acc, m4_auc, m4_time, m4_blats, m4_dec, m4_miss = run_prefetch_inference(
            dlrm, all_batches, engine, large_tables, cold_idx_maps_m4)
        warm_mem = sum(min(warm_count, len(freq_sorted_cold[t])) * EMB_DIM * 4 for t in large_tables)
        total_mem_m4 = compute_memory(dlrm, hot_indices, large_tables, num_tables, ln_emb,
                                       emb_keys, state_dict, comp_tables_m4, engine.cache.mem_bytes)
        # Add warm cold memory (it's in weight.data but we track it separately)
        log(f"  Acc={m4_acc*100:.4f}%, AUC={m4_auc:.6f}")
        log(f"  Inference={m4_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time+m4_time:.2f}s")
        log(f"  Cache hit={engine.cache.hit_rate:.4f}, evictions={engine.cache.evictions:,}")
        log(f"  Mem={total_mem_m4['total_mb']:.1f}MB (warm_cold={warm_mem/1024/1024:.1f}MB)")
        log(f"  Batch: avg={np.mean(m4_blats)*1000:.1f}ms")
        log(f"  Decode: total={np.sum(m4_dec):.2f}s, avg misses={np.mean(m4_miss):.1f}")

        results['method4'] = {
            'name': f'Method 4: Two-tier 1080p (warm={best_warm} frames)',
            'accuracy': m4_acc, 'auc': m4_auc,
            'inference_time': m4_time, 'setup_time': setup_time,
            'total_time': setup_time + m4_time,
            'memory_mb': total_mem_m4['total_mb'],
            'warm_cold_mb': warm_mem / 1024 / 1024,
            'cache_hit_rate': engine.cache.hit_rate,
            'cache_evictions': engine.cache.evictions,
            'batch_latencies': latency_stats(m4_blats),
            'total_decode_time': float(np.sum(m4_dec)),
            'avg_misses_per_batch': float(np.mean(m4_miss)),
        }
    else:
        log(f"  Best warm tier hit rate: {best_warm_hitrate:.4f} — not worth it. Skipping.")
        results['method4'] = {'name': 'Method 4: SKIPPED', 'status': 'insufficient hit rate'}

    # ================================================================
    # METHOD 5: CRF 0 for cold (lossless)
    # ================================================================
    log("\n" + "=" * 60)
    log("METHOD 5: CRF 0 (lossless) for cold embeddings — 4K")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    fw, fh = 3840, 2160
    setup_t0 = time.time()
    decoders, cold_idx_maps, cold_quant_meta, comp_tables, comp_total = setup_hot_cold_split(
        dlrm, state_dict, emb_keys, ln_emb, num_tables,
        large_tables, hot_indices, cold_order_orig, fw, fh, cold_crf=0)  # CRF 0!
    setup_time = time.time() - setup_t0
    log(f"  Setup: {setup_time:.2f}s")
    log(f"  Compressed size: {comp_total/1024/1024:.2f}MB (vs CRF23 reference)")

    engine = PrefetchEngine(decoders, cold_quant_meta, cold_idx_maps, fw, fh, 20, EMB_DIM)
    drop_caches(); time.sleep(1)
    m5_acc, m5_auc, m5_time, m5_blats, m5_dec, m5_miss = run_prefetch_inference(
        dlrm, all_batches, engine, large_tables, cold_idx_maps)
    mem = compute_memory(dlrm, hot_indices, large_tables, num_tables, ln_emb,
                         emb_keys, state_dict, comp_tables, engine.cache.mem_bytes)
    log(f"  Acc={m5_acc*100:.4f}%, AUC={m5_auc:.6f}")
    log(f"  Inference={m5_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time+m5_time:.2f}s")
    log(f"  Cache hit={engine.cache.hit_rate:.4f}, Mem={mem['total_mb']:.1f}MB")
    log(f"  Batch: avg={np.mean(m5_blats)*1000:.1f}ms")

    results['method5'] = {
        'name': 'Method 5: CRF 0 cold (4K)', 'accuracy': m5_acc, 'auc': m5_auc,
        'inference_time': m5_time, 'setup_time': setup_time,
        'total_time': setup_time + m5_time, 'memory_mb': mem['total_mb'],
        'compressed_total_mb': comp_total / 1024 / 1024,
        'cache_hit_rate': engine.cache.hit_rate,
        'batch_latencies': latency_stats(m5_blats),
        'total_decode_time': float(np.sum(m5_dec)),
    }

    # ================================================================
    # METHOD 6: Batch reordering for 1080p
    # ================================================================
    log("\n" + "=" * 60)
    log("METHOD 6: Batch reordering for 1080p")
    log("=" * 60)

    # Sort batches by frame access pattern similarity
    # Strategy: for each batch, compute the set of frames accessed for the largest table (table 2)
    # Sort batches so consecutive batches share the most frames
    fw, fh = 1920, 1080
    epf = (fw * fh) // EMB_DIM
    target_table = 2  # largest table

    cold_idx_map_t2 = cold_idx_maps_prebuilt[target_table]

    # Compute frame fingerprint for each batch (set of frames accessed for table 2)
    log("  Computing batch frame fingerprints...")
    batch_frames = []
    for bi in range(nb):
        _, _, lS_i, _ = all_batches[bi]
        indices = lS_i[target_table].numpy().flatten()
        frames = set()
        for i in indices:
            ii = int(i)
            if ii in cold_idx_map_t2:
                frames.add(cold_idx_map_t2[ii] // epf)
        batch_frames.append(frames)

    # Greedy nearest-neighbor ordering: start with batch 0, always pick the batch
    # with the most frame overlap as next
    log("  Computing greedy ordering...")
    used = set()
    order = [0]
    used.add(0)
    for _ in range(nb - 1):
        cur = order[-1]
        cur_frames = batch_frames[cur]
        best_bi = -1
        best_overlap = -1
        for bi in range(nb):
            if bi in used: continue
            overlap = len(cur_frames & batch_frames[bi])
            if overlap > best_overlap:
                best_overlap = overlap
                best_bi = bi
        order.append(best_bi)
        used.add(best_bi)

    reordered_batches = [all_batches[i] for i in order]

    # Simulate cache hit rate with reordered batches (reuse pre-built cold_idx_maps_orig)
    sim_cache = {}
    total_hits = 0
    total_accesses = 0
    total_evictions = 0
    for bi in range(nb):
        _, _, lS_i, _ = reordered_batches[bi]
        for t in large_tables:
            idx_map = cold_idx_maps_prebuilt[t]
            indices = lS_i[t].numpy().flatten()
            frames_needed = set()
            for i in indices:
                ii = int(i)
                if ii in idx_map:
                    frames_needed.add(idx_map[ii] // epf)

            if t not in sim_cache:
                sim_cache[t] = OrderedDict()
            c = sim_cache[t]
            for f in frames_needed:
                total_accesses += 1
                if f in c:
                    c.move_to_end(f); total_hits += 1
                else:
                    if len(c) >= 20:
                        c.popitem(last=False); total_evictions += 1
                    c[f] = True

    reorder_hit = total_hits / total_accesses if total_accesses > 0 else 0
    log(f"  Simulated reordered hit rate: {reorder_hit:.4f} (vs ~0.46 original)")
    log(f"  Evictions: {total_evictions:,}")

    if reorder_hit > 0.5:  # meaningful improvement
        log(f"\n  Running full inference with reordered batches...")
        restore_weights(dlrm, state_dict, emb_keys)
        setup_t0 = time.time()
        decoders, cold_idx_maps, cold_quant_meta, comp_tables, _ = setup_hot_cold_split(
            dlrm, state_dict, emb_keys, ln_emb, num_tables,
            large_tables, hot_indices, cold_order_orig, fw, fh, cold_crf=23)
        setup_time = time.time() - setup_t0
        log(f"  Setup: {setup_time:.2f}s")

        engine = PrefetchEngine(decoders, cold_quant_meta, cold_idx_maps, fw, fh, 20, EMB_DIM)
        drop_caches(); time.sleep(1)
        m6_acc, m6_auc, m6_time, m6_blats, m6_dec, m6_miss = run_prefetch_inference(
            dlrm, reordered_batches, engine, large_tables, cold_idx_maps)
        mem = compute_memory(dlrm, hot_indices, large_tables, num_tables, ln_emb,
                             emb_keys, state_dict, comp_tables, engine.cache.mem_bytes)
        log(f"  Acc={m6_acc*100:.4f}%, AUC={m6_auc:.6f}")
        log(f"  Inference={m6_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time+m6_time:.2f}s")
        log(f"  Cache hit={engine.cache.hit_rate:.4f}, evictions={engine.cache.evictions:,}")
        log(f"  Mem={mem['total_mb']:.1f}MB")
        log(f"  Batch: avg={np.mean(m6_blats)*1000:.1f}ms")
        log(f"  Decode: total={np.sum(m6_dec):.2f}s")

        results['method6'] = {
            'name': 'Method 6: Batch reorder (1080p)',
            'accuracy': m6_acc, 'auc': m6_auc,
            'inference_time': m6_time, 'setup_time': setup_time,
            'total_time': setup_time + m6_time, 'memory_mb': mem['total_mb'],
            'cache_hit_rate': engine.cache.hit_rate,
            'cache_evictions': engine.cache.evictions,
            'batch_latencies': latency_stats(m6_blats),
            'total_decode_time': float(np.sum(m6_dec)),
            'avg_misses_per_batch': float(np.mean(m6_miss)),
        }
    else:
        log(f"  Reorder hit rate {reorder_hit:.4f} — no meaningful improvement. Skipping full run.")
        results['method6'] = {'name': 'Method 6: SKIPPED', 'status': 'insufficient improvement'}

    # ================================================================
    # SUMMARY REPORT
    # ================================================================
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)

    report = []
    report.append("# Prefetch V4: Improvement Methods\n\n")
    report.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Batch Size:** {TEST_BATCH_SIZE}\n")
    report.append(f"**Baseline AUC:** {baseline_auc:.6f}\n")
    report.append(f"**Baseline Inference:** {baseline_time:.2f}s (pre-loaded batches)\n\n")

    report.append("## Results\n\n")
    report.append("| Method | AUC | AUC Loss (pp) | Inference (s) | Setup (s) | Total (s) | "
                  "Slowdown | Memory (MB) | Cache Hit | Notes |\n")
    report.append("|--------|-----|---------------|--------------|----------|----------|"
                  "----------|------------|-----------|-------|\n")

    for key in ['baseline', 'ref_4k', 'method1', 'method3', 'method4', 'method5', 'method6']:
        r = results.get(key, {})
        if 'auc' not in r:
            report.append(f"| {r.get('name', key)} | — | — | — | — | — | — | — | — | {r.get('status', 'skipped')} |\n")
            continue
        al = (baseline_auc - r['auc']) * 100
        sd = r['total_time'] / baseline_time
        ch = f"{r.get('cache_hit_rate', 0):.4f}" if r.get('cache_hit_rate') is not None else "N/A"
        notes = ""
        if 'compressed_mb' in r:
            notes = f"disk={r['compressed_mb']:.1f}MB"
        if 'compressed_total_mb' in r:
            notes = f"comp={r['compressed_total_mb']:.1f}MB"
        report.append(f"| {r['name']} | {r['auc']:.6f} | {al:.4f} | "
                      f"{r['inference_time']:.2f} | {r.get('setup_time', 0):.2f} | "
                      f"{r['total_time']:.2f} | {sd:.2f}x | "
                      f"{r.get('memory_mb', 0):.1f} | {ch} | {notes} |\n")

    report.append("\n## Method Details\n\n")

    report.append("### Method 1: Full Reconstruction\n")
    report.append("Decode all 4K frames at startup, reconstruct full embedding tables. "
                  "No per-batch injection. Runtime memory = baseline. Benefit = storage compression.\n\n")

    report.append("### Method 3: Resolution Sweep\n")
    report.append("Simulate cache hit rates at 1080p/1440p/2K/3K with various cache sizes. "
                  "Run full inference for best non-4K config that achieves >99% hit.\n\n")

    report.append("### Method 4: Two-Tier Cold\n")
    report.append("Keep most-accessed cold embeddings permanently decompressed (warm tier). "
                  "Only cache ice-cold frames. Reduces frame-cache working set for 1080p.\n\n")

    report.append("### Method 5: CRF 0 Cold\n")
    report.append("Lossless H.265 for cold embeddings. Eliminates codec quality loss. "
                  "Larger compressed size but better AUC.\n\n")

    report.append("### Method 6: Batch Reordering\n")
    report.append("Greedy nearest-neighbor sort by frame-access overlap. "
                  "Improves temporal locality for 1080p LRU cache.\n\n")

    md_path = os.path.join(RESULTS_DIR, 'prefetch_v4_methods.md')
    with open(md_path, 'w') as f: f.writelines(report)
    log(f"  Saved: {md_path}")

    json_path = os.path.join(RESULTS_DIR, 'prefetch_v4_methods.json')
    # Remove raw latency lists for JSON
    json_results = {}
    for k, v in results.items():
        json_results[k] = {kk: vv for kk, vv in v.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    log(f"  Saved: {json_path}")

    log("=" * 70)
    log("ALL DONE!")
    log("=" * 70)


if __name__ == '__main__':
    main()
