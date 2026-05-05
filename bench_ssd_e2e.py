#!/usr/bin/env python3
"""
End-to-end batch latency: sorted SSD cold reads + hot embedding + MLP.
Compare with fp32 DRAM baseline and DC block-mean.
Drops page cache before each strategy.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
from codec_ondemand_benchmark import load_model_and_data, MODEL_PATH, HOTCOLD_DIR

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
EMB_DIM = 16
ROW_BYTES = EMB_DIM * 4
SSD_FILE = '/tmp/cold_emb_e2e.bin'
OUT_DIR = 'results/ssd_benchmark'
os.makedirs(OUT_DIR, exist_ok=True)


def drop_cache():
    os.system('sync')
    os.system('sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"')


def pca_sort_order(cold_w):
    sample_size = min(50000, len(cold_w))
    pca = PCA(n_components=1, random_state=42)
    if sample_size < len(cold_w):
        idx = np.random.RandomState(42).choice(len(cold_w), sample_size, replace=False)
        pca.fit(cold_w[idx])
    else:
        pca.fit(cold_w)
    return np.argsort(pca.transform(cold_w).ravel())


def apply_dc_pca_4bit(weight, freq, hf=0.01):
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
    BS = 16
    n_blocks = (n_cold + BS - 1) // BS
    padded = np.zeros((n_blocks * BS, dim))
    padded[:n_cold] = cold_w_sorted
    blocks = padded.reshape(n_blocks, BS, dim)
    # Single scalar DC
    means = blocks.mean(axis=(1, 2))
    mn, mx = means.min(), means.max()
    s = (mx - mn) / 15.0 if mx != mn else 1.0
    q = np.clip(np.round((means - mn) / s), 0, 15)
    deq = q * s + mn
    recon = np.repeat(deq[:, None, None], BS, axis=1)
    recon = np.repeat(recon, dim, axis=2).reshape(-1, dim)[:n_cold]
    unsort = np.argsort(order)
    result = weight.clone()
    result[cold_idx] = torch.from_numpy(recon[unsort]).float()
    return result


def main():
    print("=" * 70)
    print("END-TO-END BATCH LATENCY: SSD sorted + hot + MLP")
    print("=" * 70)

    print("\n[1] Loading...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    torch.set_num_threads(32)
    is_hot = {t: torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True) for t in TABLES}

    # Profile freq
    freq_counts = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, lS_o, lS_i, T in test_batches:
        for t in TABLES:
            idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
            freq_counts[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))

    # Write cold file
    print("\n[2] Writing cold embeddings to SSD...")
    cold_indices = {}
    cold_file_start = {}
    orig_to_cold_pos = {}
    offset = 0
    for t in TABLES:
        ci = torch.where(~is_hot[t])[0]
        cold_indices[t] = ci
        cold_file_start[t] = offset
        mapping = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
        mapping[ci] = torch.arange(len(ci))
        orig_to_cold_pos[t] = mapping
        offset += len(ci) * ROW_BYTES

    with open(SSD_FILE, 'wb') as f:
        for t in TABLES:
            f.write(sd[ek[t]][cold_indices[t]].numpy().tobytes())
    print(f"  Written {offset/1024/1024:.0f} MB")

    # ================================================================
    # [A] fp32 DRAM baseline — full inference
    # ================================================================
    print("\n[A] fp32 DRAM baseline...")
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        dlrm.emb_l[t].weight.data = sd[k].clone()

    # Warmup
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches[:10]:
            dlrm(X, lS_o, lS_i)

    scores, targets, times_dram = [], [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            t0 = time.perf_counter()
            Z = dlrm(X, lS_o, lS_i)
            times_dram.append(time.perf_counter() - t0)
            scores.append(Z.numpy().ravel())
            targets.append(T.numpy().ravel())
    auc_dram = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    times_dram = np.array(times_dram) * 1000
    print(f"  AUC={auc_dram:.6f}, mean={times_dram.mean():.2f}ms, "
          f"p50={np.percentile(times_dram,50):.2f}ms, p99={np.percentile(times_dram,99):.2f}ms")

    # ================================================================
    # [B] SSD sorted reads + inference — end to end
    # ================================================================
    print("\n[B] SSD sorted reads + hot embed + MLP (end-to-end)...")

    # Set up: hot in DRAM, cold zeroed
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        w = sd[k].clone()
        if t in TABLES:
            w[cold_indices[t]] = 0.0
        dlrm.emb_l[t].weight.data = w

    drop_cache()
    fd = os.open(SSD_FILE, os.O_RDONLY)

    scores, targets = [], []
    times_ssd_total = []
    times_ssd_read = []
    times_ssd_fwd = []

    with torch.no_grad():
        for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
            # Step 1: Collect cold row offsets, sort them, read
            t_read_start = time.perf_counter()

            read_list = []  # (table, orig_idx, cold_pos, file_offset)
            for t in TABLES:
                idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
                for si in range(len(idx)):
                    orig_idx = idx[si].item()
                    cold_pos = orig_to_cold_pos[t][orig_idx].item()
                    if cold_pos >= 0:
                        file_off = cold_file_start[t] + cold_pos * ROW_BYTES
                        read_list.append((t, orig_idx, file_off))

            # Sort by file offset
            read_list.sort(key=lambda x: x[2])

            # Read in sorted order and fill embedding weights
            for t, orig_idx, file_off in read_list:
                os.lseek(fd, file_off, os.SEEK_SET)
                raw = os.read(fd, ROW_BYTES)
                dlrm.emb_l[t].weight.data[orig_idx] = torch.frombuffer(bytearray(raw), dtype=torch.float32)

            t_read = time.perf_counter() - t_read_start

            # Step 2: Forward pass (hot embed + MLP)
            t_fwd_start = time.perf_counter()
            Z = dlrm(X, lS_o, lS_i)
            t_fwd = time.perf_counter() - t_fwd_start

            times_ssd_read.append(t_read * 1000)
            times_ssd_fwd.append(t_fwd * 1000)
            times_ssd_total.append((t_read + t_fwd) * 1000)
            scores.append(Z.numpy().ravel())
            targets.append(T.numpy().ravel())

            # Zero cold rows back
            for t, orig_idx, _ in read_list:
                dlrm.emb_l[t].weight.data[orig_idx] = 0.0

            if bi % 200 == 0:
                print(f"    batch {bi}/{len(test_batches)}: "
                      f"read={t_read*1000:.1f}ms fwd={t_fwd*1000:.1f}ms total={((t_read+t_fwd)*1000):.1f}ms")

    os.close(fd)
    auc_ssd = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    times_ssd_total = np.array(times_ssd_total)
    times_ssd_read = np.array(times_ssd_read)
    times_ssd_fwd = np.array(times_ssd_fwd)
    print(f"  AUC={auc_ssd:.6f}")
    print(f"  Total:  mean={times_ssd_total.mean():.1f}ms, p50={np.percentile(times_ssd_total,50):.1f}ms, "
          f"p99={np.percentile(times_ssd_total,99):.1f}ms")
    print(f"  Read:   mean={times_ssd_read.mean():.1f}ms, p50={np.percentile(times_ssd_read,50):.1f}ms")
    print(f"  Fwd:    mean={times_ssd_fwd.mean():.1f}ms, p50={np.percentile(times_ssd_fwd,50):.1f}ms")

    # ================================================================
    # [C] DC PCA-sort 4-bit (in-memory, no I/O)
    # ================================================================
    print("\n[C] DC PCA-sort scalar 4-bit (1% hot, in-memory)...")
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        if t in TABLES:
            dlrm.emb_l[t].weight.data = apply_dc_pca_4bit(sd[k], freq_counts[t], hf=0.01)
        else:
            dlrm.emb_l[t].weight.data = sd[k].clone()

    # Warmup
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches[:10]:
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
    print("SUMMARY — End-to-End Batch Latency")
    print(f"{'='*70}")
    print(f"\n  {'Method':<35} {'Mean':>8} {'p50':>8} {'p99':>8} {'AUC':>10} {'Memory':>10}")
    print(f"  {'-'*82}")
    print(f"  {'fp32 DRAM':<35} {times_dram.mean():>7.1f}ms {np.percentile(times_dram,50):>7.1f}ms "
          f"{np.percentile(times_dram,99):>7.1f}ms {auc_dram:>10.6f} {'2061 MB':>10}")
    print(f"  {'SSD sorted + hot + MLP':<35} {times_ssd_total.mean():>7.1f}ms {np.percentile(times_ssd_total,50):>7.1f}ms "
          f"{np.percentile(times_ssd_total,99):>7.1f}ms {auc_ssd:>10.6f} {'~50 MB':>10}")
    print(f"    {'(read only)':<35} {times_ssd_read.mean():>7.1f}ms")
    print(f"    {'(fwd only)':<35} {times_ssd_fwd.mean():>7.1f}ms")
    print(f"  {'DC PCA-sort 4-bit (1% hot)':<35} {times_dc.mean():>7.1f}ms {np.percentile(times_dc,50):>7.1f}ms "
          f"{np.percentile(times_dc,99):>7.1f}ms {auc_dc:>10.6f} {'10.9 MB':>10}")

    ssd_overhead = times_ssd_total.mean() / times_dram.mean()
    dc_speedup = times_dram.mean() / times_dc.mean()
    print(f"\n  SSD sorted e2e vs DRAM: {ssd_overhead:.2f}x")
    print(f"  DC vs DRAM: {dc_speedup:.2f}x")
    print(f"  SSD read fraction: {times_ssd_read.mean()/times_ssd_total.mean()*100:.0f}% of total batch time")

    # Save
    results = {
        'fp32_dram': {
            'auc': float(auc_dram), 'mean_ms': float(times_dram.mean()),
            'p50_ms': float(np.percentile(times_dram, 50)),
            'p99_ms': float(np.percentile(times_dram, 99)),
        },
        'ssd_sorted': {
            'auc': float(auc_ssd),
            'total_mean_ms': float(times_ssd_total.mean()),
            'total_p50_ms': float(np.percentile(times_ssd_total, 50)),
            'total_p99_ms': float(np.percentile(times_ssd_total, 99)),
            'read_mean_ms': float(times_ssd_read.mean()),
            'read_p50_ms': float(np.percentile(times_ssd_read, 50)),
            'fwd_mean_ms': float(times_ssd_fwd.mean()),
        },
        'dc_pca_4bit': {
            'auc': float(auc_dc), 'mean_ms': float(times_dc.mean()),
            'p50_ms': float(np.percentile(times_dc, 50)),
            'p99_ms': float(np.percentile(times_dc, 99)),
        },
        'ssd_overhead_vs_dram': float(ssd_overhead),
        'disk': 'Micron MTFDDAK480TDS (SATA SSD)',
    }
    with open(f'{OUT_DIR}/e2e_sorted_ssd.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {OUT_DIR}/e2e_sorted_ssd.json")

    os.unlink(SSD_FILE)

if __name__ == '__main__':
    main()
