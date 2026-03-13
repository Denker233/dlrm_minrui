#!/usr/bin/env python3
"""
Prefetch Benchmark V3: Frequency-sorted cold embeddings + improved overlap.

Key change from V2: cold embeddings are sorted by descending access frequency
before packing into frames. Most-accessed cold embeddings go in frame 0, 1, etc.
This concentrates cache hits in the first few frames.

Also improves overlap: decode for batch N+1 runs fully in parallel with
batch N's weight injection + forward pass.
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
# SINGLE-FRAME TILING (from codebase)
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
# MULTI-FRAME ENCODE
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


# ==============================================================
# LRU CACHE (real memory management)
# ==============================================================
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
# FREQUENCY-BASED COLD REORDERING
# ==============================================================

def compute_cold_frequency_order(access_counts_per_table, hot_indices, ln_emb, large_tables):
    """Sort cold indices by descending access frequency.
    Returns {table: sorted_cold_indices} where index 0 is most accessed.
    """
    freq_sorted_cold = {}
    for t in large_tables:
        hi_set = set(hot_indices[t].tolist())
        all_cold = sorted(set(range(ln_emb[t])) - hi_set)

        if t in access_counts_per_table and access_counts_per_table[t] is not None:
            # Count accesses to each cold index
            counts = access_counts_per_table[t]
            cold_counts = {}
            for idx in all_cold:
                cold_counts[idx] = counts.get(idx, 0)
            # Sort by descending count, then by index for stability
            sorted_cold = sorted(all_cold, key=lambda x: (-cold_counts[x], x))
        else:
            sorted_cold = all_cold

        freq_sorted_cold[t] = sorted_cold
    return freq_sorted_cold


# ==============================================================
# PREFETCH ENGINE V3
# ==============================================================

class PrefetchEngineV3:
    """Prefetch with real LRU, PyAV decoder, and improved overlap."""

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
        self._thread_result = [0.0, 0]  # [decode_time, n_decoded]

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
        """Start background prefetch."""
        if self._thread: self._thread.join()
        self._thread_result = [0.0, 0]
        self._thread = threading.Thread(target=self._do_fetch, args=(needed_by_table,))
        self._thread.start()

    def wait(self):
        if self._thread:
            self._thread.join()
            self._thread = None
        return self._thread_result

    def fetch_sync(self, needed_by_table):
        """Synchronous fetch for remaining misses."""
        self._do_fetch(needed_by_table)
        return self._thread_result

    def get_cold_embeddings_batch(self, table, orig_indices_unique):
        """Vectorized: get all cold embeddings for a set of original indices.
        Returns (indices_list, values_tensor) for a single scatter write.
        """
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

        # Vectorized dequantize
        q_arr = np.stack(values_list)  # (N, 16)
        fp_arr = (q_arr.astype(np.float32) - zp) * s
        return indices_out, torch.from_numpy(fp_arr)


# ==============================================================
# INFERENCE
# ==============================================================

def run_baseline(dlrm, test_ld):
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
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
    return accu/samp, roc_auc_score(targets, scores), time.time()-t0, blats


def run_prefetch_inference(dlrm, test_ld, engine, large_tables, cold_idx_maps, cold_quant_meta):
    """Inference with prefetch, vectorized injection, and full overlap."""
    all_batches = list(test_ld)
    nb = len(all_batches)
    embs_per_frame = engine.embs_per_frame

    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    dec_times = []
    miss_counts = []

    # Pre-identify needed frames for batch 0
    _, _, lS_i_0, _ = all_batches[0]
    needed_0 = {}
    for t in large_tables:
        if t in engine.decoders:
            frames = engine.get_needed_frames(t, lS_i_0[t].numpy().flatten())
            if frames: needed_0[t] = frames

    # Synchronously fetch batch 0's frames (cold start)
    engine.fetch_sync(needed_0)

    # Pre-identify batch 1's frames and start prefetch
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

        # If bi >= 1, wait for prefetch of THIS batch (started in previous iteration)
        if bi >= 1:
            dt, nd = engine.wait()
            dec_times.append(dt)
            miss_counts.append(nd)

            # Check for any remaining misses (frames evicted between prefetch start and now)
            needed_now = {}
            for t in large_tables:
                if t in engine.decoders:
                    frames = engine.get_needed_frames(t, lS_i[t].numpy().flatten())
                    if frames: needed_now[t] = frames
            # Only fetch truly missing ones
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
                engine.fetch_sync(still_missing)
        else:
            dec_times.append(0)
            miss_counts.append(0)

        # Start prefetch for batch bi+2 (bi+1 was already started)
        # Actually: start prefetch for the NEXT batch (bi+1)
        if bi + 1 < nb:
            _, _, next_lS_i, _ = all_batches[bi + 1]
            needed_next = {}
            for t in large_tables:
                if t in engine.decoders:
                    frames = engine.get_needed_frames(t, next_lS_i[t].numpy().flatten())
                    if frames: needed_next[t] = frames
            if bi + 1 >= 2:
                # prefetch for bi+1 hasn't been started yet if bi >= 1
                # Actually we need to rethink: we start prefetch for bi+1 during bi's processing
                engine.prefetch_async(needed_next)

        # Vectorized cold embedding injection
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
    acc = accu / samp
    auc = roc_auc_score(targets, scores)
    return acc, auc, total, blats, dec_times, miss_counts


# ==============================================================
# MAIN
# ==============================================================

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log("=" * 70)
    log("PREFETCH BENCHMARK V3: Frequency-Sorted Cold + Vectorized Inject")
    log("=" * 70)

    log("Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")

    # Profile access with FULL COUNT tracking
    log("Profiling access patterns (full counts)...")
    access_counts = {}  # {table: {idx: count}}
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
    for t in range(num_tables):
        if access_raw[t] is None or len(access_raw[t]) == 0:
            hot_indices[t] = np.array([], dtype=np.int64)
            access_counts[t] = {}
            continue
        unique, counts = np.unique(access_raw[t], return_counts=True)
        # Build full count dict
        access_counts[t] = {int(u): int(c) for u, c in zip(unique, counts)}
        si = np.argsort(-counts)
        cum = np.cumsum(counts[si])
        cutoff = np.searchsorted(cum, cum[-1] * HOT_THRESHOLD) + 1
        hot_indices[t] = unique[si[:cutoff]]

    for t in large_tables:
        log(f"  Table {t}: {ln_emb[t]:,} total, {len(hot_indices[t]):,} hot")

    # Compute frequency-sorted cold order
    log("Computing frequency-sorted cold order...")
    freq_sorted_cold = compute_cold_frequency_order(access_counts, hot_indices, ln_emb, large_tables)

    # Analyze: how many frames do the top-K cold embeddings span?
    EMBS_1080P = (1920 * 1080) // EMB_DIM
    for t in large_tables:
        sc = freq_sorted_cold[t]
        n_cold = len(sc)
        # How many unique frames needed for top 80%, 90%, 95% of cold accesses?
        total_cold_accesses = sum(access_counts[t].get(idx, 0) for idx in sc)
        if total_cold_accesses > 0:
            cum = 0
            for rank, idx in enumerate(sc):
                cum += access_counts[t].get(idx, 0)
                if cum >= total_cold_accesses * 0.80 and rank > 0:
                    frames_80 = (rank // EMBS_1080P) + 1
                    break
            else:
                frames_80 = (n_cold // EMBS_1080P) + 1

            cum = 0
            for rank, idx in enumerate(sc):
                cum += access_counts[t].get(idx, 0)
                if cum >= total_cold_accesses * 0.95:
                    frames_95 = (rank // EMBS_1080P) + 1
                    break
            else:
                frames_95 = (n_cold // EMBS_1080P) + 1

            log(f"  Table {t}: {n_cold:,} cold, 80% accesses in {frames_80} frames, "
                f"95% in {frames_95} frames (of {(n_cold+EMBS_1080P-1)//EMBS_1080P} total)")

    # ========================================
    # Config A: Baseline
    # ========================================
    log("\n--- Config A: Baseline ---")
    with torch.no_grad():
        for k in emb_keys:
            t = int(k.split('.')[1])
            dlrm.emb_l[t].weight.data = state_dict[k].clone()
    drop_caches(); time.sleep(1)
    a_acc, a_auc, a_time, a_blats = run_baseline(dlrm, test_ld)
    a_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
    log(f"  Acc={a_acc*100:.4f}%, AUC={a_auc:.6f}, Time={a_time:.2f}s, Mem={a_mem:.1f}MB")

    results = {
        'A': {'config': 'A', 'name': 'Baseline', 'accuracy': a_acc, 'auc': a_auc,
              'inference_time': a_time, 'total_time': a_time, 'decompress_time': 0,
              'memory_mb': a_mem, 'batch_latencies': a_blats, 'cache_hit_rate': None,
              'avg_decomp_per_batch': 0}
    }

    # ========================================
    # Configs: Prefetch with freq-sorted vs original order
    # ========================================
    configs_to_run = [
        ('C_orig', 'V2 1080p original order', 1920, 1080, 20, False),
        ('C_freq', 'V3 1080p freq-sorted', 1920, 1080, 20, True),
        ('D_orig', 'V2 4K original order', 3840, 2160, 20, False),
        ('D_freq', 'V3 4K freq-sorted', 3840, 2160, 20, True),
    ]

    for label, name, fw, fh, cache_size, use_freq_sort in configs_to_run:
        log(f"\n--- Config {label}: {name} ---")

        # Restore weights
        with torch.no_grad():
            for k in emb_keys:
                t = int(k.split('.')[1])
                dlrm.emb_l[t].weight.data = state_dict[k].clone()

        setup_t0 = time.time()

        # Decompress small tables (CRF 0)
        small_comp = 0
        for t in range(num_tables):
            w = state_dict[emb_keys[t]]
            if w.shape[0] < LARGE_TABLE_THRESHOLD:
                q, s, zp = quantize(w)
                raw, fw2, fh2, tm = prepare_single_frame(q.numpy(), w.shape[0], EMB_DIM)
                comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
                small_comp += len(comp)
                dec = decode_single_frame_legacy(comp, fw2, fh2, w.shape[0], EMB_DIM, tm)
                with torch.no_grad():
                    dlrm.emb_l[t].weight.data = dequantize(dec, s, zp)

        # For large tables: hot at CRF 0, cold at multi-frame CRF 23
        decoders = {}
        cold_idx_maps = {}
        cold_quant_meta = {}
        comp_tables = {}
        hot_comp = 0

        for t in large_tables:
            w = state_dict[emb_keys[t]]
            ne = w.shape[0]
            hi = set(hot_indices[t].tolist())

            # Hot
            hot_idx = sorted(hi)
            if hot_idx:
                hw = w[hot_idx]; qh, sh, zh = quantize(hw)
                raw, fw2, fh2, tm = prepare_single_frame(qh.numpy(), hw.shape[0], EMB_DIM)
                comp = encode_single_frame(raw, fw2, fh2, get_codec_args(0))
                hot_comp += len(comp)
                dec = decode_single_frame_legacy(comp, fw2, fh2, hw.shape[0], EMB_DIM, tm)
                with torch.no_grad():
                    dlrm.emb_l[t].weight.data[hot_idx] = dequantize(dec, sh, zh)

            # Cold: choose order
            if use_freq_sort:
                cold_idx = freq_sorted_cold[t]
            else:
                cold_idx = sorted(set(range(ne)) - hi)

            cw = w[cold_idx]
            if cw.shape[0] > 0:
                qc, sc, zc = quantize(cw)
                pf = qc.numpy().reshape(-1)
                comp_data, n_frames = encode_multiframe(pf, fw, fh, 23, keyint=1)
                comp_tables[t] = comp_data
                cold_quant_meta[t] = (sc, zc, cw.shape[0])
                # Map: original_index -> sequential position in freq-sorted order
                cold_idx_maps[t] = {orig: seq for seq, orig in enumerate(cold_idx)}
                decoders[t] = PyAVDecoder(comp_data, fw, fh, n_frames)
                log(f"    Table {t}: {cw.shape[0]:,} cold -> {n_frames} frames, "
                    f"{len(comp_data)/1024:.1f} KB")

        setup_time = time.time() - setup_t0
        log(f"  Setup: {setup_time:.2f}s")

        # Measure per-frame decode
        test_t = max(decoders.keys(), key=lambda t: decoders[t].n_frames)
        lats = []
        for _ in range(10):
            fi = np.random.randint(0, decoders[test_t].n_frames)
            t0 = time.time()
            decoders[test_t].decode_frames([fi])
            lats.append(time.time() - t0)
        pf_ms = np.median(lats) * 1000
        log(f"  Per-frame decode: {pf_ms:.1f}ms")

        # Run inference
        engine = PrefetchEngineV3(decoders, cold_quant_meta, cold_idx_maps,
                                   fw, fh, cache_size, EMB_DIM)
        drop_caches(); time.sleep(1)

        acc, auc, infer_time, blats, dec_times, miss_counts = run_prefetch_inference(
            dlrm, test_ld, engine, large_tables, cold_idx_maps, cold_quant_meta)

        # Memory
        comp_cold_mb = sum(len(v) for v in comp_tables.values()) / 1024 / 1024
        cache_mb = engine.cache.mem_bytes / 1024 / 1024
        hot_mb = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables) / 1024 / 1024
        small_mb = sum(state_dict[emb_keys[t]].numel() * 4
                       for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD) / 1024 / 1024
        mlp_mb = sum(p.numel() * 4 for n, p in dlrm.named_parameters() if 'emb_l' not in n) / 1024 / 1024
        total_mem = hot_mb + small_mb + mlp_mb + comp_cold_mb + cache_mb

        blats_arr = np.array(blats)
        dec_arr = np.array(dec_times)
        miss_arr = np.array(miss_counts)

        log(f"  Acc={acc*100:.4f}%, AUC={auc:.6f}, Infer={infer_time:.2f}s")
        log(f"  Cache: hit={engine.cache.hit_rate:.4f}, evictions={engine.cache.evictions:,}")
        log(f"  Memory: {total_mem:.1f}MB (cache={cache_mb:.1f}MB)")
        log(f"  Batch: avg={np.mean(blats_arr)*1000:.1f}ms p50={np.percentile(blats_arr,50)*1000:.1f}ms "
            f"p95={np.percentile(blats_arr,95)*1000:.1f}ms")
        log(f"  Decode: total={np.sum(dec_arr):.2f}s, avg/batch={np.mean(dec_arr)*1000:.1f}ms, "
            f"avg misses={np.mean(miss_arr):.1f}")

        results[label] = {
            'config': label, 'name': name, 'accuracy': acc, 'auc': auc,
            'inference_time': infer_time, 'total_time': setup_time + infer_time,
            'decompress_time': setup_time, 'memory_mb': total_mem,
            'hot_fp32_mb': hot_mb, 'small_fp32_mb': small_mb,
            'compressed_cold_mb': comp_cold_mb, 'frame_cache_mb': cache_mb,
            'mlp_mb': mlp_mb, 'batch_latencies': blats,
            'cache_hit_rate': engine.cache.hit_rate,
            'cache_evictions': engine.cache.evictions,
            'avg_decomp_per_batch': float(np.mean(miss_arr)),
            'avg_batch_lat': float(np.mean(blats_arr)),
            'p50_batch_lat': float(np.percentile(blats_arr, 50)),
            'p95_batch_lat': float(np.percentile(blats_arr, 95)),
            'p99_batch_lat': float(np.percentile(blats_arr, 99)),
            'total_decode_time': float(np.sum(dec_arr)),
            'per_frame_decode_ms': pf_ms,
        }

    # ========================================
    # REPORT
    # ========================================
    log("\n" + "=" * 70)
    log("REPORT")
    log("=" * 70)

    baseline = results['A']
    report = []
    report.append("# Prefetch V3: Frequency-Sorted Cold Embeddings\n\n")
    report.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Batch Size:** {TEST_BATCH_SIZE}\n")
    report.append(f"**Decoder:** PyAV in-process\n")
    report.append(f"**Baseline AUC:** {baseline['auc']:.6f}\n\n")

    report.append("## Comparison: Original Order vs Frequency-Sorted\n\n")
    report.append("| Config | Description | AUC | AUC Loss (pp) | Inference (s) | Slowdown | "
                  "Memory (MB) | Cache Hit | Evictions | Avg Misses/Batch | Total Decode (s) |\n")
    report.append("|--------|-------------|-----|---------------|--------------|----------|"
                  "------------|-----------|-----------|-----------------|------------------|\n")

    for c in ['A', 'C_orig', 'C_freq', 'D_orig', 'D_freq']:
        r = results[c]
        al = (baseline['auc'] - r['auc']) * 100
        sd = r['inference_time'] / baseline['inference_time']
        ch = f"{r['cache_hit_rate']:.4f}" if r['cache_hit_rate'] is not None else "N/A"
        ev = r.get('cache_evictions', 0)
        am = r.get('avg_decomp_per_batch', 0)
        td = r.get('total_decode_time', 0)
        report.append(f"| {c} | {r['name']} | {r['auc']:.6f} | {al:.4f} | "
                      f"{r['inference_time']:.2f} | {sd:.2f}x | {r['memory_mb']:.1f} | "
                      f"{ch} | {ev:,} | {am:.1f} | {td:.2f} |\n")

    report.append("\n## Batch Latency\n\n")
    report.append("| Config | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) |\n")
    report.append("|--------|---------|---------|---------|--------|\n")
    for c in ['A', 'C_orig', 'C_freq', 'D_orig', 'D_freq']:
        r = results[c]
        bl = np.array(r['batch_latencies'])
        report.append(f"| {c} | {np.mean(bl)*1000:.1f} | {np.percentile(bl,50)*1000:.1f} | "
                      f"{np.percentile(bl,95)*1000:.1f} | {np.percentile(bl,99)*1000:.1f} |\n")

    report.append("\n## Key Finding: Frequency Sort Impact on 1080p\n\n")
    if 'C_orig' in results and 'C_freq' in results:
        co = results['C_orig']; cf = results['C_freq']
        report.append(f"| Metric | Original | Freq-Sorted | Improvement |\n")
        report.append(f"|--------|---------|------------|-------------|\n")
        report.append(f"| Cache Hit Rate | {co['cache_hit_rate']:.4f} | {cf['cache_hit_rate']:.4f} | "
                      f"{(cf['cache_hit_rate']-co['cache_hit_rate'])*100:.1f}pp |\n")
        report.append(f"| Evictions | {co['cache_evictions']:,} | {cf['cache_evictions']:,} | "
                      f"{co['cache_evictions']-cf['cache_evictions']:,} fewer |\n")
        report.append(f"| Avg Misses/Batch | {co['avg_decomp_per_batch']:.1f} | "
                      f"{cf['avg_decomp_per_batch']:.1f} | "
                      f"{co['avg_decomp_per_batch']-cf['avg_decomp_per_batch']:.1f} fewer |\n")
        report.append(f"| Inference Time | {co['inference_time']:.1f}s | {cf['inference_time']:.1f}s | "
                      f"{co['inference_time']-cf['inference_time']:.1f}s faster |\n")
        report.append(f"| Slowdown | {co['inference_time']/baseline['inference_time']:.2f}x | "
                      f"{cf['inference_time']/baseline['inference_time']:.2f}x | |\n")

    md_path = os.path.join(RESULTS_DIR, "prefetch_v3_freq_sorted.md")
    with open(md_path, 'w') as f:
        f.write(''.join(report))
    log(f"  Saved: {md_path}")

    jr = {}
    for c, r in results.items():
        rd = dict(r)
        bl = rd['batch_latencies']
        rd['batch_latencies'] = {
            'count': len(bl), 'mean': float(np.mean(bl)),
            'p50': float(np.percentile(bl, 50)), 'p95': float(np.percentile(bl, 95)),
            'p99': float(np.percentile(bl, 99)),
        }
        jr[c] = rd
    json_path = os.path.join(RESULTS_DIR, "prefetch_v3_freq_sorted.json")
    with open(json_path, 'w') as f:
        json.dump(jr, f, indent=2, default=str)
    log(f"  Saved: {json_path}")

    log("=" * 70)
    log("ALL DONE!")
    log("=" * 70)


if __name__ == "__main__":
    main()
