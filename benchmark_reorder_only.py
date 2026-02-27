#!/usr/bin/env python3
"""
Benchmark reordering-only impact on DLRM inference (no codec compression).

Tests whether physically rearranging embedding rows improves CPU cache locality
during EmbeddingBag lookups. Three conditions:
  1. Baseline: original row order
  2. Batch-affinity: rows sorted by (first_batch, -frequency)
  3. Rec-AD: rows sorted by (community_id, -frequency) via Louvain

The key insight: if frequently co-accessed rows are physically adjacent in memory,
EmbeddingBag lookups benefit from hardware prefetching and fewer cache misses.
"""

import os, sys, time, gc, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
LARGE_TABLE_THRESHOLD = 50000

RESULTS_DIR = "results"
PROFILING_DIR = os.path.join(RESULTS_DIR, "profiling")
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR_ORIG = os.path.join(RESULTS_DIR, "reorder")
REORDER_DIR_RECAD = os.path.join(RESULTS_DIR, "reorder_recad")
LOG_FILE = "logs/reorder_only_benchmark.log"

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


def build_reorder_mapping(cold_order, cold_indices, hot_indices, ln_emb_t):
    """
    Build a full-table reorder permutation from a cold reorder.

    Hot rows keep their relative order but are placed first.
    Cold rows are reordered according to cold_order.

    Returns:
        full_order: np.int64 array of length ln_emb_t — new row order
        index_remap: np.int64 array of length ln_emb_t — maps old_idx -> new_idx
    """
    n_hot = len(hot_indices)
    n_cold = len(cold_indices)
    cold_idx_np = cold_indices.numpy()

    # Full order: hot rows first (original order), then cold rows (reordered)
    full_order = np.empty(ln_emb_t, dtype=np.int64)
    # Hot rows: sorted by original index for consistency
    hot_sorted = np.sort(hot_indices.numpy())
    full_order[:n_hot] = hot_sorted
    # Cold rows: apply cold_order permutation
    full_order[n_hot:n_hot + n_cold] = cold_idx_np[cold_order]
    # Any remaining rows (if ln_emb_t > n_hot + n_cold, shouldn't happen)
    used = set(full_order[:n_hot + n_cold].tolist())
    remaining = [i for i in range(ln_emb_t) if i not in used]
    if remaining:
        full_order[n_hot + n_cold:] = remaining

    # Build reverse mapping: old_idx -> new_idx
    index_remap = np.empty(ln_emb_t, dtype=np.int64)
    index_remap[full_order[:n_hot + n_cold]] = np.arange(n_hot + n_cold, dtype=np.int64)
    for i, r in enumerate(remaining):
        index_remap[r] = n_hot + n_cold + i

    return full_order, index_remap


def build_frequency_reorder(batch_access_log, ln_emb_t, table_id, num_batches=5000):
    """
    Build a frequency-based reorder for ALL rows (not just cold).
    Most frequently accessed rows go first.
    """
    freq = np.zeros(ln_emb_t, dtype=np.int64)
    num_sample = min(len(batch_access_log), num_batches)
    for bi in range(num_sample):
        accessed = batch_access_log[bi].get(table_id, set())
        for idx in accessed:
            if idx < ln_emb_t:
                freq[idx] += 1

    # Sort by descending frequency (most accessed first)
    full_order = np.argsort(-freq).astype(np.int64)

    # Build reverse mapping
    index_remap = np.empty(ln_emb_t, dtype=np.int64)
    index_remap[full_order] = np.arange(ln_emb_t, dtype=np.int64)

    return full_order, index_remap


def run_inference(dlrm, test_batches, label, num_runs=3):
    """Run inference and collect AUC + timing. Average over num_runs."""
    times = []
    all_lats = []

    for run_idx in range(num_runs):
        gc.collect()
        max_samples = len(test_batches) * TEST_BATCH_SIZE + TEST_BATCH_SIZE
        scores = np.empty(max_samples, dtype=np.float32)
        targets = np.empty(max_samples, dtype=np.float32)
        sample_idx = 0
        blats = []

        t0 = time.time()
        with torch.no_grad():
            for X, lS_o, lS_i, T in test_batches:
                bt0 = time.time()
                Z = dlrm(X, lS_o, lS_i)
                blats.append(time.time() - bt0)

                z_np = Z.detach().cpu().numpy().ravel()
                t_np = T.detach().cpu().numpy().ravel()
                bs = z_np.shape[0]
                scores[sample_idx:sample_idx+bs] = z_np
                targets[sample_idx:sample_idx+bs] = t_np
                sample_idx += bs
        total_time = time.time() - t0
        auc = roc_auc_score(targets[:sample_idx], scores[:sample_idx])
        times.append(total_time)
        all_lats.extend(blats)

        log(f"  [{label}] Run {run_idx+1}/{num_runs}: AUC={auc:.6f}, "
            f"Time={total_time:.2f}s, Mean lat={np.mean(blats)*1000:.2f}ms")

    avg_time = np.mean(times)
    std_time = np.std(times)
    mean_lat = np.mean(all_lats) * 1000
    p50_lat = np.percentile(all_lats, 50) * 1000
    p99_lat = np.percentile(all_lats, 99) * 1000

    log(f"  [{label}] Average: Time={avg_time:.2f}s (±{std_time:.2f}s), "
        f"Mean lat={mean_lat:.2f}ms, P50={p50_lat:.2f}ms, P99={p99_lat:.2f}ms")

    return {
        'auc': auc,
        'avg_time': float(avg_time),
        'std_time': float(std_time),
        'mean_lat_ms': float(mean_lat),
        'p50_lat_ms': float(p50_lat),
        'p99_lat_ms': float(p99_lat),
        'rss_mb': get_rss_mb(),
        'num_runs': num_runs,
    }


def main():
    log("=" * 70)
    log("REORDERING-ONLY BENCHMARK (No Codec Compression)")
    log("Baseline vs Batch-Affinity vs Rec-AD vs Frequency-Sort")
    log("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]

    log(f"Tables: {num_tables}, Large: {large_tables}")
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

    # Pre-cache test batches
    log("Pre-caching test batches...")
    test_batches = []
    for X, lS_o, lS_i, T in test_ld:
        test_batches.append((X, lS_o, lS_i, T))
    log(f"  {len(test_batches)} test batches")

    NUM_RUNS = 3
    all_results = {}

    # ---- Experiment 1: Baseline (original order) ----
    log("\n" + "=" * 70)
    log("EXPERIMENT 1: BASELINE (original row order)")
    log("=" * 70)

    # Restore original weights
    for t in range(num_tables):
        with torch.no_grad():
            dlrm.emb_l[t].weight = nn.Parameter(
                state_dict[emb_keys[t]].clone(), requires_grad=False)
    gc.collect()

    all_results['baseline'] = run_inference(dlrm, test_batches, 'Baseline', NUM_RUNS)

    # ---- Experiment 2: Frequency-sorted (all rows, most accessed first) ----
    log("\n" + "=" * 70)
    log("EXPERIMENT 2: FREQUENCY-SORTED (most accessed rows first)")
    log("=" * 70)

    freq_remaps = {}
    for t in large_tables:
        full_order, index_remap = build_frequency_reorder(
            batch_access_log, ln_emb[t], t)
        freq_remaps[t] = index_remap

        # Physically reorder embedding weights
        w = state_dict[emb_keys[t]]
        reordered_w = w[full_order]
        with torch.no_grad():
            dlrm.emb_l[t].weight = nn.Parameter(reordered_w, requires_grad=False)
        log(f"  Table {t}: reordered {ln_emb[t]:,} rows by frequency")

    # Remap test batch indices
    test_batches_freq = []
    for X, lS_o, lS_i, T in test_batches:
        new_lS_i = []
        for t in range(num_tables):
            if t in freq_remaps:
                new_lS_i.append(torch.from_numpy(
                    freq_remaps[t][lS_i[t].numpy()]).long())
            else:
                new_lS_i.append(lS_i[t])
        test_batches_freq.append((X, lS_o, new_lS_i, T))

    all_results['frequency_sort'] = run_inference(
        dlrm, test_batches_freq, 'Frequency-sort', NUM_RUNS)

    # ---- Experiment 3: Batch-affinity reorder (hot first, cold reordered) ----
    log("\n" + "=" * 70)
    log("EXPERIMENT 3: BATCH-AFFINITY REORDER")
    log("=" * 70)

    # Restore original weights first
    for t in range(num_tables):
        with torch.no_grad():
            dlrm.emb_l[t].weight = nn.Parameter(
                state_dict[emb_keys[t]].clone(), requires_grad=False)

    ba_remaps = {}
    for t in large_tables:
        cold_order = np.load(os.path.join(REORDER_DIR_ORIG, f'cold_order_{t}.npy'))
        full_order, index_remap = build_reorder_mapping(
            cold_order, cold_indices[t], hot_indices[t], ln_emb[t])
        ba_remaps[t] = index_remap

        w = state_dict[emb_keys[t]]
        reordered_w = w[full_order]
        with torch.no_grad():
            dlrm.emb_l[t].weight = nn.Parameter(reordered_w, requires_grad=False)
        log(f"  Table {t}: reordered (hot first + cold batch-affinity)")

    test_batches_ba = []
    for X, lS_o, lS_i, T in test_batches:
        new_lS_i = []
        for t in range(num_tables):
            if t in ba_remaps:
                new_lS_i.append(torch.from_numpy(
                    ba_remaps[t][lS_i[t].numpy()]).long())
            else:
                new_lS_i.append(lS_i[t])
        test_batches_ba.append((X, lS_o, new_lS_i, T))

    all_results['batch_affinity'] = run_inference(
        dlrm, test_batches_ba, 'Batch-affinity', NUM_RUNS)

    # ---- Experiment 4: Rec-AD reorder (hot first, cold community-sorted) ----
    log("\n" + "=" * 70)
    log("EXPERIMENT 4: REC-AD COMMUNITY DETECTION REORDER")
    log("=" * 70)

    # Restore original weights first
    for t in range(num_tables):
        with torch.no_grad():
            dlrm.emb_l[t].weight = nn.Parameter(
                state_dict[emb_keys[t]].clone(), requires_grad=False)

    recad_remaps = {}
    for t in large_tables:
        cold_order_path = os.path.join(REORDER_DIR_RECAD, f'cold_order_{t}.npy')
        if not os.path.exists(cold_order_path):
            log(f"  Table {t}: Rec-AD reorder not found, using batch-affinity")
            cold_order = np.load(os.path.join(REORDER_DIR_ORIG, f'cold_order_{t}.npy'))
        else:
            cold_order = np.load(cold_order_path)
        full_order, index_remap = build_reorder_mapping(
            cold_order, cold_indices[t], hot_indices[t], ln_emb[t])
        recad_remaps[t] = index_remap

        w = state_dict[emb_keys[t]]
        reordered_w = w[full_order]
        with torch.no_grad():
            dlrm.emb_l[t].weight = nn.Parameter(reordered_w, requires_grad=False)
        log(f"  Table {t}: reordered (hot first + cold Rec-AD)")

    test_batches_recad = []
    for X, lS_o, lS_i, T in test_batches:
        new_lS_i = []
        for t in range(num_tables):
            if t in recad_remaps:
                new_lS_i.append(torch.from_numpy(
                    recad_remaps[t][lS_i[t].numpy()]).long())
            else:
                new_lS_i.append(lS_i[t])
        test_batches_recad.append((X, lS_o, new_lS_i, T))

    all_results['recad_louvain'] = run_inference(
        dlrm, test_batches_recad, 'Rec-AD', NUM_RUNS)

    # ---- Results ----
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)

    header = f"{'Method':<25} {'AUC':>8} {'AvgTime':>8} {'±Std':>6} {'MeanLat':>8} {'P50Lat':>8} {'P99Lat':>8}"
    log(header)
    log("-" * len(header))

    baseline_time = all_results['baseline']['avg_time']
    for key in ['baseline', 'frequency_sort', 'batch_affinity', 'recad_louvain']:
        r = all_results[key]
        speedup = baseline_time / r['avg_time'] if r['avg_time'] > 0 else 0
        log(f"{key:<25} {r['auc']:>8.6f} {r['avg_time']:>7.2f}s {r['std_time']:>5.2f}s "
            f"{r['mean_lat_ms']:>7.2f}ms {r['p50_lat_ms']:>7.2f}ms {r['p99_lat_ms']:>7.2f}ms "
            f"({speedup:.3f}x)")

    # Save
    results_json = os.path.join(RESULTS_DIR, 'reorder_only_results.json')
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved: {results_json}")

    summary_lines = ["# Reordering-Only Benchmark Results\n"]
    summary_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Runs per experiment: {NUM_RUNS}\n")
    summary_lines.append(header)
    summary_lines.append("-" * len(header))
    for key in ['baseline', 'frequency_sort', 'batch_affinity', 'recad_louvain']:
        r = all_results[key]
        speedup = baseline_time / r['avg_time'] if r['avg_time'] > 0 else 0
        summary_lines.append(
            f"{key:<25} {r['auc']:>8.6f} {r['avg_time']:>7.2f}s {r['std_time']:>5.2f}s "
            f"{r['mean_lat_ms']:>7.2f}ms {r['p50_lat_ms']:>7.2f}ms {r['p99_lat_ms']:>7.2f}ms "
            f"({speedup:.3f}x)")

    summary_lines.append("\n\n## Delta vs Baseline\n")
    for key in ['frequency_sort', 'batch_affinity', 'recad_louvain']:
        r = all_results[key]
        b = all_results['baseline']
        summary_lines.append(f"### {key}")
        summary_lines.append(f"  AUC delta:    {r['auc'] - b['auc']:+.6f}")
        summary_lines.append(f"  Time delta:   {r['avg_time'] - b['avg_time']:+.2f}s "
                           f"({(r['avg_time']/b['avg_time'] - 1)*100:+.1f}%)")
        summary_lines.append(f"  Lat delta:    {r['mean_lat_ms'] - b['mean_lat_ms']:+.2f}ms")
        summary_lines.append("")

    summary_path = os.path.join(RESULTS_DIR, 'reorder_only_results.md')
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    log(f"Saved: {summary_path}")

    log(f"\n{'='*70}")
    log("REORDER-ONLY BENCHMARK COMPLETE")
    log(f"{'='*70}")


if __name__ == '__main__':
    main()
