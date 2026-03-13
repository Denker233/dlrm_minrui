#!/usr/bin/env python3
"""
Benchmark hot embedding reordering in the codec pipeline.

Tests whether reordering hot embedding rows (by access frequency) improves
lookup latency in the compressed embedding setup.

Hot rows are stored as compact fp32 or uint8 tensors. By sorting them so that
the most frequently accessed rows are at the top (low indices), we can
improve CPU cache utilization during the hot embedding gather operations.

Conditions:
  1. Original hot order (baseline codec)
  2. Frequency-sorted hot (most accessed first)
  3. Batch-affinity hot (group by first batch, then frequency)
"""

import os, sys, time, gc, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import psutil
import subprocess

# C++ extension
try:
    import compressed_emb as _C
    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import av

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
DATA_FILE = os.path.expanduser("~/input/train.txt")
PROCESSED_DATA = os.path.expanduser("~/input/kaggleAdDisplayChallenge_processed.npz")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 2048
EMB_DIM = 16
HOT_COVERAGE = 0.80
LARGE_TABLE_THRESHOLD = 50000
H265_CRF = 0

RESOLUTIONS = {
    '1080p': (1920, 1080),
}

RESULTS_DIR = "results"
PROFILING_DIR = os.path.join(RESULTS_DIR, "profiling")
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
ONDEMAND_DIR = os.path.join(RESULTS_DIR, "ondemand")
LOG_FILE = "logs/hot_reorder_benchmark.log"

os.makedirs("logs", exist_ok=True)
log_fh = open(LOG_FILE, 'w')

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()

def get_rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def drop_caches():
    try:
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except Exception:
        pass


# ============================================================
# MODEL + DATA LOADING
# ============================================================
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
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net
    args = create_args()
    log("Loading dataset...")
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
    log("Loading model checkpoint...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    log(f"Model loaded. {len(ln_emb)} tables, emb_dim={m_spa}")
    return dlrm, test_ld, train_ld, ln_emb


def quantize_table(w):
    mn = w.min().item()
    mx = w.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def main():
    log("=" * 70)
    log("HOT EMBEDDING REORDERING BENCHMARK (Codec Pipeline)")
    log("Original vs Frequency-sorted vs Batch-affinity hot rows")
    log("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")
    log(f"C++ extension: {'ENABLED' if HAS_CPP_EXT else 'DISABLED'}")

    total_emb_mb = sum(state_dict[k].numel() * 4 for k in emb_keys) / 1024 / 1024
    log(f"Total embedding memory: {total_emb_mb:.1f}MB")

    # Load hot/cold data
    is_hot = {}
    hot_indices = {}
    cold_indices = {}
    for t in large_tables:
        is_hot[t] = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{t}.pt'),
                               map_location='cpu', weights_only=True)
        hot_indices[t] = torch.where(is_hot[t])[0]
        cold_indices[t] = torch.where(~is_hot[t])[0]

    # Load batch access log
    bal_path = os.path.join(PROFILING_DIR, 'batch_access_log.npz')
    _bal = np.load(bal_path, allow_pickle=True)
    batch_access_log = _bal['batch_access_log'].tolist()
    log(f"Batch access log: {len(batch_access_log)} batches")

    # Load cold reorder data (unchanged across experiments — only hot ordering changes)
    orig_to_cold_reordered = {}
    cold_quant_scale = {}
    cold_quant_zp = {}
    cold_num_rows = {}

    for t in large_tables:
        fp = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t}.pt')
        orig_to_cold_reordered[t] = torch.load(fp, map_location='cpu', weights_only=True)
        n_cold_path = os.path.join(REORDER_DIR, f'num_cold_{t}.txt')
        with open(n_cold_path) as f:
            cold_num_rows[t] = int(f.read().strip())

    res_name = '1080p'
    res_dir = os.path.join(ONDEMAND_DIR, res_name)
    for t in large_tables:
        meta_path = os.path.join(res_dir, f'table_{t}', 'meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        cold_quant_scale[t] = meta['quant_scale']
        cold_quant_zp[t] = meta['quant_zp']

    # Pre-cache test batches
    log("Pre-caching test batches...")
    test_batches = []
    for X, lS_o, lS_i, T in test_ld:
        test_batches.append((X, lS_o, lS_i, T))
    log(f"  {len(test_batches)} test batches")

    # Import classes from codec_ondemand_benchmark
    import codec_ondemand_benchmark as cob
    CompressedEmbeddingBag = cob.CompressedEmbeddingBag
    OnDemandPrefetchCache = cob.OnDemandPrefetchCache
    GlobalFrameCache = cob.GlobalFrameCache

    # Build hot access frequency per table
    log("\nProfiling hot access frequencies...")
    hot_freq = {}
    hot_first_batch = {}
    for t in large_tables:
        h_idx = hot_indices[t].numpy()
        hot_to_compact = np.full(ln_emb[t], -1, dtype=np.int64)
        hot_to_compact[h_idx] = np.arange(len(h_idx), dtype=np.int64)

        freq = np.zeros(len(h_idx), dtype=np.int64)
        first_batch = np.full(len(h_idx), len(batch_access_log), dtype=np.int64)
        num_sample = min(len(batch_access_log), 5000)

        for bi in range(num_sample):
            accessed = batch_access_log[bi].get(t, set())
            for idx in accessed:
                if idx < ln_emb[t]:
                    compact = hot_to_compact[idx]
                    if compact >= 0:
                        freq[compact] += 1
                        if first_batch[compact] == len(batch_access_log):
                            first_batch[compact] = bi

        hot_freq[t] = freq
        hot_first_batch[t] = first_batch
        active = np.sum(freq > 0)
        log(f"  Table {t}: {len(h_idx):,} hot rows, {active:,} active, "
            f"avg freq={np.mean(freq[freq>0]):.1f}")

    # Store original modules for restoration
    original_emb_modules = {}
    for t in large_tables:
        original_emb_modules[t] = dlrm.emb_l[t]

    def restore_weights():
        for t in range(num_tables):
            with torch.no_grad():
                dlrm.emb_l[t].weight = nn.Parameter(
                    state_dict[emb_keys[t]].clone(), requires_grad=False)
        for t in large_tables:
            original_emb_modules[t] = dlrm.emb_l[t]
        gc.collect()

    NUM_RUNS = 3
    all_results = {}

    def run_codec_experiment(hot_reorder_mode, tag):
        """
        Run full codec inference with a specific hot reordering.
        hot_reorder_mode: 'original', 'frequency', 'batch_affinity'
        """
        width, height = RESOLUTIONS[res_name]
        rows_per_frame = (width * height) // EMB_DIM

        log(f"\n  Setting up {tag}...")
        restore_weights()

        # Build hot weight tensors with specified ordering
        o2c_map = {}
        caches = {}
        total_hot_mb = 0
        total_mapping_mb = 0
        total_compressed_bytes = 0

        cap = 9999  # unlimited for full_cpp
        global_cache = GlobalFrameCache(capacity=cap, store_uint8=True)

        for t_idx in large_tables:
            n_cold = cold_num_rows.get(t_idx, 0)
            if n_cold == 0:
                continue

            frame_dir = os.path.join(res_dir, f'table_{t_idx}')
            frame_files = sorted([f for f in os.listdir(frame_dir)
                                  if f.startswith('frame_') and f.endswith('.h265')])
            comp_bytes = sum(os.path.getsize(os.path.join(frame_dir, ff))
                             for ff in frame_files)
            total_compressed_bytes += comp_bytes

            cache_obj = OnDemandPrefetchCache(
                frame_dir=frame_dir,
                rows_per_frame=rows_per_frame,
                emb_dim=EMB_DIM,
                num_cold_rows=n_cold,
                width=width, height=height,
                quant_scale=cold_quant_scale[t_idx],
                quant_zp=cold_quant_zp[t_idx],
                cache_capacity=32,
                predictor=None,
                num_prefetch_workers=2,
                global_cache=global_cache,
                table_id=t_idx,
            )
            caches[t_idx] = cache_obj

            # Build hot weight with specified ordering
            w = state_dict[emb_keys[t_idx]]
            h_idx = hot_indices[t_idx]
            n_hot = len(h_idx)

            if hot_reorder_mode == 'frequency':
                # Sort hot rows by descending frequency
                freq_order = np.argsort(-hot_freq[t_idx])
                hot_weight = w[h_idx[freq_order]].clone()
                # Rebuild orig_to_hot mapping
                orig_to_hot = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
                orig_to_hot[h_idx[freq_order]] = torch.arange(n_hot)
            elif hot_reorder_mode == 'batch_affinity':
                # Sort by (first_batch, -freq)
                sort_key = (hot_first_batch[t_idx].astype(np.float64) * 1e12
                           - hot_freq[t_idx].astype(np.float64))
                ba_order = np.argsort(sort_key)
                hot_weight = w[h_idx[ba_order]].clone()
                orig_to_hot = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
                orig_to_hot[h_idx[ba_order]] = torch.arange(n_hot)
            else:
                # Original order
                hot_weight = w[h_idx].clone()
                orig_to_hot = torch.full((ln_emb[t_idx],), -1, dtype=torch.long)
                orig_to_hot[h_idx] = torch.arange(n_hot)

            total_hot_mb += hot_weight.numel() / 1024 / 1024  # quantized uint8

            o2c_map[t_idx] = orig_to_cold_reordered[t_idx]

            comp_emb = CompressedEmbeddingBag(
                hot_weight=hot_weight,
                is_hot=is_hot[t_idx],
                orig_to_hot=orig_to_hot,
                orig_to_cold_reordered=o2c_map[t_idx],
                cold_cache=cache_obj,
                num_embeddings=ln_emb[t_idx],
                embedding_dim=EMB_DIM,
                quantize_hot=True,
            )
            total_mapping_mb += ln_emb[t_idx] * 4 / 1024 / 1024
            dlrm.emb_l[t_idx] = comp_emb

        # Free state_dict entries
        freed_state = {}
        for t_idx in large_tables:
            k = emb_keys[t_idx]
            if k in state_dict:
                freed_state[k] = state_dict.pop(k)
        for t_idx in large_tables:
            original_emb_modules[t_idx].weight = nn.Parameter(
                torch.zeros(1, EMB_DIM), requires_grad=False)
        gc.collect()

        # Register in C++ with bitmap
        compressed_table_ids = set(caches.keys())
        num_tabs = len(dlrm.emb_l)

        if HAS_CPP_EXT and compressed_table_ids:
            table_kinds = []
            weights = []
            mappings = []
            scales = []
            zero_points = []

            for k in range(num_tabs):
                E = dlrm.emb_l[k]
                if k in compressed_table_ids and isinstance(E, CompressedEmbeddingBag):
                    table_kinds.append(2)  # COMPRESSED_Q8
                    weights.append(E.hot_weight_q8)
                    mappings.append(E.mapping)
                    scales.append(float(E.hot_scale))
                    zero_points.append(int(E.hot_zp))
                else:
                    table_kinds.append(0)
                    weights.append(E.weight)
                    mappings.append(torch.empty(0, dtype=torch.int32))
                    scales.append(0.0)
                    zero_points.append(0)

            # NOTE: bitmap mode computes hot indices via popcount rank, which
            # assumes hot_weight is sorted by original index. When hot reordering
            # is active, we must use the mapping tensor (which stores the correct
            # reordered orig_to_hot values).
            use_bm = (hot_reorder_mode == 'original')
            _C.register_tables(table_kinds, weights, mappings, scales, zero_points,
                               use_hash_table=False, use_bitmap=use_bm)

            # Warmup
            with torch.no_grad():
                X, lS_o, lS_i, T = test_batches[0]
                Z = dlrm(X, lS_o, lS_i)

            # Full C++ mode: register cold frames
            needed_frames = {t_idx: set() for t_idx in caches}
            for X_s, lS_o_s, lS_i_s, T_s in test_batches:
                for t_idx in caches:
                    indices = lS_i_s[t_idx]
                    cold_mask = ~is_hot[t_idx][indices]
                    if cold_mask.any():
                        cold_orig = indices[cold_mask]
                        cold_mapped = o2c_map[t_idx][cold_orig]
                        valid = cold_mapped >= 0
                        if valid.any():
                            fids = (cold_mapped[valid] // rows_per_frame).unique().tolist()
                            needed_frames[t_idx].update(fids)

            for t_idx in caches:
                for fid in sorted(needed_frames[t_idx]):
                    cached = global_cache.get(t_idx, fid)
                    if cached is None:
                        frame_data = caches[t_idx]._decode_raw(fid)
                        global_cache.put(t_idx, fid, frame_data)

            for t_idx in caches:
                table_frames = {}
                for (tid, fid), data in global_cache.cache.items():
                    if tid == t_idx:
                        table_frames[fid] = data
                if not table_frames:
                    continue
                sorted_fids = sorted(table_frames.keys())
                padded_frames = []
                for fid in sorted_fids:
                    frame = table_frames[fid]
                    if frame.shape[0] < rows_per_frame:
                        pad = np.zeros((rows_per_frame - frame.shape[0], EMB_DIM), dtype=np.uint8)
                        frame = np.concatenate([frame, pad], axis=0)
                    padded_frames.append(frame)
                all_data = np.concatenate(padded_frames, axis=0)
                frame_data_tensor = torch.from_numpy(all_data.copy())
                frame_ids_tensor = torch.tensor(sorted_fids, dtype=torch.long)

                if use_bm:
                    # Bitmap mode: need cold mapping
                    mmap_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t_idx}.npy')
                    if not os.path.exists(mmap_path):
                        pt_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{t_idx}.pt')
                        o2c = torch.load(pt_path, map_location='cpu', weights_only=True)
                        np.save(mmap_path, o2c.numpy())
                    cold_mapping_tensor = torch.from_numpy(
                        np.load(mmap_path).copy()).int()
                else:
                    cold_mapping_tensor = torch.empty(0, dtype=torch.long)
                _C.register_cold_frames_for_table(
                    t_idx, frame_ids_tensor, frame_data_tensor,
                    float(cold_quant_scale[t_idx]),
                    float(cold_quant_zp[t_idx]),
                    rows_per_frame,
                    cold_mapping_tensor)

            def _full_cpp_apply_emb(lS_o, lS_i, emb_l, v_W_l):
                start_time = time.time()
                if isinstance(lS_i, (list, tuple)):
                    lS_i_2d = torch.stack(lS_i)
                elif lS_i.dim() == 2:
                    lS_i_2d = lS_i
                else:
                    lS_i_2d = lS_i.view(num_tabs, -1)
                if isinstance(lS_o, (list, tuple)):
                    lS_o_2d = torch.stack(lS_o)
                elif lS_o.dim() == 2:
                    lS_o_2d = lS_o
                else:
                    lS_o_2d = lS_o.view(num_tabs, -1)
                results = _C.fast_forward(lS_i_2d, lS_o_2d)
                dlrm.time_look_up += time.time() - start_time
                return results[-1]
            dlrm.apply_emb = _full_cpp_apply_emb

        # Run inference
        drop_caches()
        gc.collect()

        times_list = []
        all_lats = []

        for run_idx in range(NUM_RUNS):
            max_samples = len(test_batches) * TEST_BATCH_SIZE + TEST_BATCH_SIZE
            scores = np.empty(max_samples, dtype=np.float32)
            targets_arr = np.empty(max_samples, dtype=np.float32)
            sample_idx = 0
            blats = []
            t0 = time.time()

            with torch.no_grad():
                for batch_idx in range(len(test_batches)):
                    X, lS_o, lS_i, T = test_batches[batch_idx]
                    bt0 = time.time()
                    Z = dlrm(X, lS_o, lS_i)
                    blats.append(time.time() - bt0)
                    z_np = Z.detach().cpu().numpy().ravel()
                    t_np = T.detach().cpu().numpy().ravel()
                    bs = z_np.shape[0]
                    scores[sample_idx:sample_idx+bs] = z_np
                    targets_arr[sample_idx:sample_idx+bs] = t_np
                    sample_idx += bs

            total_time = time.time() - t0
            auc = roc_auc_score(targets_arr[:sample_idx], scores[:sample_idx])
            times_list.append(total_time)
            all_lats.extend(blats)
            log(f"    [{tag}] Run {run_idx+1}: AUC={auc:.6f}, Time={total_time:.2f}s, "
                f"Mean lat={np.mean(blats)*1000:.2f}ms")

        result = {
            'auc': auc,
            'avg_time': float(np.mean(times_list)),
            'std_time': float(np.std(times_list)),
            'mean_lat_ms': float(np.mean(all_lats) * 1000),
            'p50_lat_ms': float(np.percentile(all_lats, 50) * 1000),
            'p99_lat_ms': float(np.percentile(all_lats, 99) * 1000),
            'hot_mb': total_hot_mb,
            'method': tag,
        }

        # Restore state_dict
        for k, v in freed_state.items():
            state_dict[k] = v
        restore_weights()

        return result

    # ---- Run experiments ----
    torch.set_num_threads(40)
    log(f"Thread count: 40")

    log("\n" + "=" * 70)
    log("EXPERIMENT 1: ORIGINAL HOT ORDER (codec baseline)")
    log("=" * 70)
    all_results['original_hot'] = run_codec_experiment('original', 'original_hot')

    log("\n" + "=" * 70)
    log("EXPERIMENT 2: FREQUENCY-SORTED HOT")
    log("=" * 70)
    all_results['freq_hot'] = run_codec_experiment('frequency', 'freq_hot')

    log("\n" + "=" * 70)
    log("EXPERIMENT 3: BATCH-AFFINITY HOT")
    log("=" * 70)
    all_results['batch_affinity_hot'] = run_codec_experiment('batch_affinity', 'batch_affinity_hot')

    # ---- Results ----
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY — HOT REORDERING")
    log("=" * 70)

    header = f"{'Method':<25} {'AUC':>8} {'AvgTime':>8} {'±Std':>6} {'MeanLat':>8} {'P50Lat':>8} {'P99Lat':>8}"
    log(header)
    log("-" * len(header))

    baseline_time = all_results['original_hot']['avg_time']
    for key in ['original_hot', 'freq_hot', 'batch_affinity_hot']:
        r = all_results[key]
        speedup = baseline_time / r['avg_time'] if r['avg_time'] > 0 else 0
        log(f"{key:<25} {r['auc']:>8.6f} {r['avg_time']:>7.2f}s {r['std_time']:>5.2f}s "
            f"{r['mean_lat_ms']:>7.2f}ms {r['p50_lat_ms']:>7.2f}ms {r['p99_lat_ms']:>7.2f}ms "
            f"({speedup:.3f}x)")

    # Save
    results_json = os.path.join(RESULTS_DIR, 'hot_reorder_results.json')
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved: {results_json}")

    summary_lines = ["# Hot Embedding Reordering Results (Codec Pipeline)\n"]
    summary_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Config: 1080p, full_cpp, quantize_hot, 40 threads (bitmap only for original)")
    summary_lines.append(f"Runs per experiment: {NUM_RUNS}\n")
    summary_lines.append(header)
    summary_lines.append("-" * len(header))
    for key in ['original_hot', 'freq_hot', 'batch_affinity_hot']:
        r = all_results[key]
        speedup = baseline_time / r['avg_time'] if r['avg_time'] > 0 else 0
        summary_lines.append(
            f"{key:<25} {r['auc']:>8.6f} {r['avg_time']:>7.2f}s {r['std_time']:>5.2f}s "
            f"{r['mean_lat_ms']:>7.2f}ms {r['p50_lat_ms']:>7.2f}ms {r['p99_lat_ms']:>7.2f}ms "
            f"({speedup:.3f}x)")

    summary_lines.append("\n\n## Delta vs Original Hot Order\n")
    for key in ['freq_hot', 'batch_affinity_hot']:
        r = all_results[key]
        b = all_results['original_hot']
        summary_lines.append(f"### {key}")
        summary_lines.append(f"  AUC delta:    {r['auc'] - b['auc']:+.6f}")
        summary_lines.append(f"  Time delta:   {r['avg_time'] - b['avg_time']:+.2f}s "
                           f"({(r['avg_time']/b['avg_time'] - 1)*100:+.1f}%)")
        summary_lines.append(f"  Lat delta:    {r['mean_lat_ms'] - b['mean_lat_ms']:+.2f}ms")
        summary_lines.append("")

    summary_path = os.path.join(RESULTS_DIR, 'hot_reorder_results.md')
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    log(f"Saved: {summary_path}")

    log(f"\n{'='*70}")
    log("HOT REORDER BENCHMARK COMPLETE")
    log(f"{'='*70}")


if __name__ == '__main__':
    main()
