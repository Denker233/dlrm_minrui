#!/usr/bin/env python3
"""
SSD cold embedding benchmark at batch_size=2048.
Compare: fp32 DRAM, SSD cold, DC value-sort 4-bit.
Report batch sizes in MB, embedding rows per batch, cold row counts.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
EMB_DIM = 16
MODEL_PATH = 'models/dlrm_kaggle_correct.pt'
HOTCOLD_DIR = 'results/hotcold'
SSD_FILE = '/tmp/cold_embeddings_bs2048.bin'
OUT_DIR = 'results/ssd_benchmark'
os.makedirs(OUT_DIR, exist_ok=True)
BATCH_SIZE = 2048

def load_model_bs2048():
    """Load DLRM with batch_size=2048 test loader."""
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net

    class Args:
        pass
    a = Args()
    a.arch_sparse_feature_size = 16
    a.arch_mlp_bot = "13-512-256-64-16"
    a.arch_mlp_top = "512-256-1"
    a.arch_interaction_op = "dot"
    a.arch_interaction_itself = False
    a.data_generation = "dataset"
    a.data_set = "kaggle"
    a.raw_data_file = "./input/train.txt"
    a.processed_data_file = "./input/kaggleAdDisplayChallenge_processed.npz"
    a.loss_function = "bce"
    a.max_ind_range = -1
    a.test_mini_batch_size = BATCH_SIZE
    a.test_num_workers = 0
    a.num_workers = 0
    a.mlperf_logging = False
    a.memory_map = False
    a.data_randomize = "total"
    a.data_trace_enable_padding = False
    a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10
    a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = BATCH_SIZE
    a.round_targets = True
    a.mlperf_bin_loader = False
    a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False

    print("  Loading dataset (bs=2048)...")
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(a)
    ln_emb = np.array(train_data.counts)

    ln_bot = np.fromstring(a.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + a.arch_mlp_top, dtype=int, sep="-")

    dlrm = DLRM_Net(16, ln_emb, ln_bot, ln_top,
                     arch_interaction_op="dot", arch_interaction_itself=False,
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")
    print("  Loading checkpoint...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    return dlrm, test_ld, train_ld, ln_emb


def pca_sort_order(cold_w):
    sample_size = min(50000, len(cold_w))
    pca = PCA(n_components=1, random_state=42)
    if sample_size < len(cold_w):
        idx = np.random.RandomState(42).choice(len(cold_w), sample_size, replace=False)
        pca.fit(cold_w[idx])
    else:
        pca.fit(cold_w)
    return np.argsort(pca.transform(cold_w).ravel())


def apply_dc_pca_4bit(weight, freq, hf=0.01, block_size=16):
    """Apply DC block-mean with PCA sort, 4-bit, return modified weights and ratio."""
    w = weight.numpy().copy()
    n_rows, dim = w.shape
    n_hot = max(1, int(n_rows * hf))
    n_cold = n_rows - n_hot

    freq_np = freq.numpy()
    sorted_by_freq = np.argsort(-freq_np)
    cold_idx = sorted_by_freq[n_hot:]
    cold_w = w[cold_idx]

    order = pca_sort_order(cold_w)
    cold_w_sorted = cold_w[order]

    n_blocks = (n_cold + block_size - 1) // block_size
    padded = np.zeros((n_blocks * block_size, dim))
    padded[:n_cold] = cold_w_sorted
    blocks = padded.reshape(n_blocks, block_size, dim)
    block_means = blocks.mean(axis=1)

    bm_min, bm_max = block_means.min(), block_means.max()
    bm_scale = (bm_max - bm_min) / 15.0 if bm_max > bm_min else 1.0
    bm_q = np.clip(np.round((block_means - bm_min) / bm_scale), 0, 15)
    bm_deq = bm_q * bm_scale + bm_min

    reconstructed = np.repeat(bm_deq, block_size, axis=0)[:n_cold]
    unsort = np.argsort(order)

    result = weight.clone()
    result[cold_idx] = torch.from_numpy(reconstructed[unsort]).float()

    orig_bytes = n_rows * dim * 4
    hot_bytes = n_hot * dim * 1
    cold_bytes = n_blocks * dim * 0.5
    map_bytes = n_rows * 0.5
    ratio = orig_bytes / (hot_bytes + cold_bytes + map_bytes)

    return result, ratio


def main():
    print("=" * 70)
    print(f"SSD COLD EMBEDDING BENCHMARK — batch_size={BATCH_SIZE}")
    print("=" * 70)

    # Load
    print("\n[1] Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_bs2048()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    nt = len(ln_emb)
    torch.set_num_threads(32)
    print(f"  {len(test_batches)} test batches, batch_size={BATCH_SIZE}")

    # Load hot/cold masks
    is_hot = {t: torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True) for t in TABLES}

    # ================================================================
    # [2] Batch size analysis
    # ================================================================
    print(f"\n[2] Batch size analysis...")

    # Profile one batch
    X, lS_o, lS_i, T = test_batches[0]
    actual_bs = T.shape[0]
    print(f"  Actual batch size: {actual_bs}")
    print(f"  Dense features (X): {X.shape} = {X.numel() * 4 / 1024:.1f} KB")

    total_emb_lookups = 0
    total_cold_lookups = 0
    total_emb_bytes_fp32 = 0
    total_cold_bytes_fp32 = 0

    # Profile access patterns across all batches
    print(f"\n  Profiling embedding access across {len(test_batches)} batches...")
    freq_counts = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    all_emb_lookups = []
    all_cold_lookups = []
    all_emb_bytes = []
    all_cold_bytes = []

    for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
        batch_emb = 0
        batch_cold = 0
        for ti in range(nt):
            idx = lS_i[ti] if isinstance(lS_i, list) else lS_i[ti]
            n_idx = len(idx)
            batch_emb += n_idx

            if ti in TABLES:
                freq_counts[ti].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
                cold_mask = ~is_hot[ti][idx]
                batch_cold += cold_mask.sum().item()

        all_emb_lookups.append(batch_emb)
        all_cold_lookups.append(batch_cold)
        all_emb_bytes.append(batch_emb * EMB_DIM * 4)  # fp32
        all_cold_bytes.append(batch_cold * EMB_DIM * 4)

    all_emb_lookups = np.array(all_emb_lookups)
    all_cold_lookups = np.array(all_cold_lookups)
    all_emb_bytes = np.array(all_emb_bytes)
    all_cold_bytes = np.array(all_cold_bytes)

    print(f"\n  --- Embedding Lookup Statistics (batch_size={BATCH_SIZE}) ---")
    print(f"  Total lookups/batch:  mean={all_emb_lookups.mean():.0f}, "
          f"min={all_emb_lookups.min()}, max={all_emb_lookups.max()}")
    print(f"  Cold lookups/batch:   mean={all_cold_lookups.mean():.0f}, "
          f"min={all_cold_lookups.min()}, max={all_cold_lookups.max()}")
    print(f"  Cold fraction:        {all_cold_lookups.mean()/all_emb_lookups.mean()*100:.1f}%")
    print(f"  Embedding data/batch: mean={all_emb_bytes.mean()/1024/1024:.2f} MB "
          f"({all_emb_bytes.mean()/1024:.0f} KB)")
    print(f"  Cold data/batch:      mean={all_cold_bytes.mean()/1024/1024:.2f} MB "
          f"({all_cold_bytes.mean()/1024:.0f} KB)")
    print(f"  Each row: {EMB_DIM} × 4 bytes = {EMB_DIM*4} bytes (fp32)")

    # Per-table breakdown
    print(f"\n  Per-table cold lookups (mean per batch):")
    per_table_cold = {}
    for t in TABLES:
        cold_per_batch = []
        for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
            idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
            cold_mask = ~is_hot[t][idx]
            cold_per_batch.append(cold_mask.sum().item())
        per_table_cold[t] = np.array(cold_per_batch)
        print(f"    T{t:>2} ({sd[ek[t]].shape[0]:>10,} rows): "
              f"mean={per_table_cold[t].mean():.0f} cold lookups/batch, "
              f"{per_table_cold[t].mean()*EMB_DIM*4/1024:.1f} KB")

    # ================================================================
    # [3] Write cold embeddings to SSD
    # ================================================================
    print(f"\n[3] Writing cold embeddings to SSD...")
    cold_offsets = {}
    cold_indices = {}
    total_cold_bytes_disk = 0

    for t in TABLES:
        ci = torch.where(~is_hot[t])[0]
        cold_indices[t] = ci
        cold_offsets[t] = (total_cold_bytes_disk, len(ci))
        total_cold_bytes_disk += len(ci) * EMB_DIM * 4

    print(f"  Total cold on disk: {total_cold_bytes_disk/1024/1024:.0f} MB")

    with open(SSD_FILE, 'wb') as f:
        for t in TABLES:
            f.write(sd[ek[t]][cold_indices[t]].numpy().tobytes())

    # Build mapping
    orig_to_cold_pos = {}
    for t in TABLES:
        mapping = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
        mapping[cold_indices[t]] = torch.arange(len(cold_indices[t]))
        orig_to_cold_pos[t] = mapping

    # ================================================================
    # [4] Benchmark: fp32 DRAM baseline
    # ================================================================
    print(f"\n[4] fp32 DRAM baseline...")
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        dlrm.emb_l[t].weight.data = sd[k].clone()

    # Warmup
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches[:5]:
            dlrm(X, lS_o, lS_i)

    scores, targets, times = [], [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            t0 = time.perf_counter()
            Z = dlrm(X, lS_o, lS_i)
            times.append(time.perf_counter() - t0)
            scores.append(Z.numpy().ravel())
            targets.append(T.numpy().ravel())

    auc_fp32 = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    times_fp32 = np.array(times) * 1000
    print(f"  AUC={auc_fp32:.6f}, mean={times_fp32.mean():.2f}ms, "
          f"p50={np.percentile(times_fp32,50):.2f}ms, p99={np.percentile(times_fp32,99):.2f}ms")

    # ================================================================
    # [5] Benchmark: SSD cold reads
    # ================================================================
    print(f"\n[5] SSD cold (real disk reads)...")

    # Set up: hot rows in DRAM, cold rows zeroed
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        w = sd[k].clone()
        if t in TABLES:
            w[cold_indices[t]] = 0.0
        dlrm.emb_l[t].weight.data = w

    # Drop caches
    os.system('sync')
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as f:
            f.write('3')
    except:
        pass

    row_bytes = EMB_DIM * 4
    fd = os.open(SSD_FILE, os.O_RDONLY)

    scores, targets, times_ssd_list = [], [], []
    read_times = []

    with torch.no_grad():
        for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
            # Read cold rows from SSD
            t_read_start = time.perf_counter()
            for t in TABLES:
                idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
                for si in range(len(idx)):
                    orig_idx = idx[si].item()
                    cold_pos = orig_to_cold_pos[t][orig_idx].item()
                    if cold_pos >= 0:
                        byte_offset = cold_offsets[t][0] + cold_pos * row_bytes
                        os.lseek(fd, byte_offset, os.SEEK_SET)
                        raw = os.read(fd, row_bytes)
                        row_data = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                        dlrm.emb_l[t].weight.data[orig_idx] = row_data
            t_read = time.perf_counter() - t_read_start
            read_times.append(t_read)

            # Forward
            t0 = time.perf_counter()
            Z = dlrm(X, lS_o, lS_i)
            t_fwd = time.perf_counter() - t0

            times_ssd_list.append((t_read + t_fwd) * 1000)
            scores.append(Z.numpy().ravel())
            targets.append(T.numpy().ravel())

            # Zero cold rows back
            for t in TABLES:
                idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
                for si in range(len(idx)):
                    if orig_to_cold_pos[t][idx[si].item()].item() >= 0:
                        dlrm.emb_l[t].weight.data[idx[si].item()] = 0.0

            if bi % 50 == 0:
                print(f"    batch {bi}/{len(test_batches)}: read={t_read*1000:.1f}ms fwd={t_fwd*1000:.1f}ms")

    os.close(fd)
    auc_ssd = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    times_ssd = np.array(times_ssd_list)
    read_times_ms = np.array(read_times) * 1000
    print(f"  AUC={auc_ssd:.6f}, mean={times_ssd.mean():.1f}ms, "
          f"p50={np.percentile(times_ssd,50):.1f}ms, p99={np.percentile(times_ssd,99):.1f}ms")
    print(f"  SSD read: mean={read_times_ms.mean():.1f}ms, p50={np.percentile(read_times_ms,50):.1f}ms")

    # ================================================================
    # [6] Benchmark: DC PCA-sort 4-bit (in-memory)
    # ================================================================
    print(f"\n[6] DC PCA-sort 4-bit (1% hot, in-memory)...")

    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        if t in TABLES:
            modified_w, ratio = apply_dc_pca_4bit(sd[k], freq_counts[t], hf=0.01)
            dlrm.emb_l[t].weight.data = modified_w
            print(f"    T{t}: ratio={ratio:.0f}x")
        else:
            dlrm.emb_l[t].weight.data = sd[k].clone()

    # Warmup
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches[:5]:
            dlrm(X, lS_o, lS_i)

    scores, targets, times_dc = [], [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            t0 = time.perf_counter()
            Z = dlrm(X, lS_o, lS_i)
            times_dc.append((time.perf_counter() - t0) * 1000)
            scores.append(Z.numpy().ravel())
            targets.append(T.numpy().ravel())

    auc_dc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    times_dc = np.array(times_dc)
    print(f"  AUC={auc_dc:.6f}, mean={times_dc.mean():.2f}ms, "
          f"p50={np.percentile(times_dc,50):.2f}ms, p99={np.percentile(times_dc,99):.2f}ms")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY — batch_size={BATCH_SIZE}")
    print(f"{'='*70}")

    print(f"\n  Batch size analysis:")
    print(f"    Batch size:           {BATCH_SIZE} samples")
    print(f"    Total emb lookups:    {all_emb_lookups.mean():.0f}/batch ({all_emb_lookups.mean()*row_bytes/1024/1024:.2f} MB)")
    print(f"    Cold emb lookups:     {all_cold_lookups.mean():.0f}/batch ({all_cold_bytes.mean()/1024/1024:.2f} MB)")
    print(f"    Cold fraction:        {all_cold_lookups.mean()/all_emb_lookups.mean()*100:.1f}%")
    print(f"    Each row:             {EMB_DIM}×fp32 = {row_bytes} bytes")

    print(f"\n  Latency comparison:")
    print(f"    {'Method':<30} {'Mean':>8} {'p50':>8} {'p99':>8} {'AUC':>10}")
    print(f"    {'-'*66}")
    print(f"    {'fp32 DRAM':<30} {times_fp32.mean():>7.1f}ms {np.percentile(times_fp32,50):>7.1f}ms "
          f"{np.percentile(times_fp32,99):>7.1f}ms {auc_fp32:>10.6f}")
    print(f"    {'SSD cold':<30} {times_ssd.mean():>7.1f}ms {np.percentile(times_ssd,50):>7.1f}ms "
          f"{np.percentile(times_ssd,99):>7.1f}ms {auc_ssd:>10.6f}")
    print(f"    {'DC PCA-sort 4bit (1% hot)':<30} {times_dc.mean():>7.1f}ms {np.percentile(times_dc,50):>7.1f}ms "
          f"{np.percentile(times_dc,99):>7.1f}ms {auc_dc:>10.6f}")

    ssd_slowdown = times_ssd.mean() / times_fp32.mean()
    dc_speedup = times_fp32.mean() / times_dc.mean()
    print(f"\n    SSD is {ssd_slowdown:.0f}x slower than DRAM")
    print(f"    DC is {dc_speedup:.2f}x vs DRAM at {(auc_dc-auc_fp32)/auc_fp32*100:+.4f}% AUC loss")

    # Save
    results = {
        'batch_size': BATCH_SIZE,
        'num_test_batches': len(test_batches),
        'batch_stats': {
            'total_emb_lookups_mean': float(all_emb_lookups.mean()),
            'cold_emb_lookups_mean': float(all_cold_lookups.mean()),
            'cold_fraction': float(all_cold_lookups.mean() / all_emb_lookups.mean()),
            'emb_data_per_batch_mb': float(all_emb_bytes.mean() / 1024 / 1024),
            'cold_data_per_batch_mb': float(all_cold_bytes.mean() / 1024 / 1024),
            'row_bytes': row_bytes,
        },
        'fp32_dram': {
            'auc': float(auc_fp32),
            'mean_ms': float(times_fp32.mean()),
            'p50_ms': float(np.percentile(times_fp32, 50)),
            'p99_ms': float(np.percentile(times_fp32, 99)),
        },
        'ssd_cold': {
            'auc': float(auc_ssd),
            'mean_ms': float(times_ssd.mean()),
            'p50_ms': float(np.percentile(times_ssd, 50)),
            'p99_ms': float(np.percentile(times_ssd, 99)),
            'read_mean_ms': float(read_times_ms.mean()),
            'read_p50_ms': float(np.percentile(read_times_ms, 50)),
        },
        'dc_pca_4bit': {
            'auc': float(auc_dc),
            'mean_ms': float(times_dc.mean()),
            'p50_ms': float(np.percentile(times_dc, 50)),
            'p99_ms': float(np.percentile(times_dc, 99)),
            'hf': 0.01,
        },
        'ssd_slowdown_vs_dram': float(ssd_slowdown),
        'disk': 'Micron MTFDDAK480TDS (SATA SSD)',
    }
    with open(f'{OUT_DIR}/ssd_bs2048.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {OUT_DIR}/ssd_bs2048.json")

    # Cleanup
    os.unlink(SSD_FILE)


if __name__ == '__main__':
    main()
