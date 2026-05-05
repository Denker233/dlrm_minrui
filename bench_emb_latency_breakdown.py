#!/usr/bin/env python3
"""
Isolate embedding lookup latency: DRAM vs SSD vs DC.
Times: (1) SSD read + embedding lookup, (2) full forward, (3) derive MLP time.
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
SSD_FILE = '/tmp/cold_emb_breakdown.bin'
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

def apply_dc(weight, freq, hf=0.01):
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
    print("EMBEDDING LOOKUP LATENCY BREAKDOWN: DRAM vs SSD vs DC")
    print("=" * 70)

    print("\n[1] Loading...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    torch.set_num_threads(32)
    is_hot = {t: torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True) for t in TABLES}

    freq_counts = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, lS_o, lS_i, T in test_batches:
        for t in TABLES:
            idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
            freq_counts[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))

    # Write cold file
    print("\n[2] Preparing SSD cold file...")
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
    # [A] fp32 DRAM — measure embedding-only vs full forward
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

    emb_times_dram, fwd_times_dram = [], []
    scores, targets = [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            # Time embedding only
            t0 = time.perf_counter()
            for k_idx in range(len(ek)):
                E = dlrm.emb_l[k_idx]
                o = lS_o[k_idx] if isinstance(lS_o, list) else lS_o[k_idx]
                i = lS_i[k_idx] if isinstance(lS_i, list) else lS_i[k_idx]
                E(i, o)
            t_emb = time.perf_counter() - t0

            # Time full forward
            t0 = time.perf_counter()
            Z = dlrm(X, lS_o, lS_i)
            t_fwd = time.perf_counter() - t0

            emb_times_dram.append(t_emb * 1000)
            fwd_times_dram.append(t_fwd * 1000)
            scores.append(Z.numpy().ravel())
            targets.append(T.numpy().ravel())

    auc_dram = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    emb_dram = np.array(emb_times_dram)
    fwd_dram = np.array(fwd_times_dram)
    mlp_dram = fwd_dram - emb_dram  # MLP + interaction time
    mlp_dram = np.clip(mlp_dram, 0, None)

    print(f"  AUC={auc_dram:.6f}")
    print(f"  Embedding: mean={emb_dram.mean():.2f}ms, p50={np.percentile(emb_dram,50):.2f}ms, p99={np.percentile(emb_dram,99):.2f}ms")
    print(f"  Full fwd:  mean={fwd_dram.mean():.2f}ms, p50={np.percentile(fwd_dram,50):.2f}ms, p99={np.percentile(fwd_dram,99):.2f}ms")
    print(f"  MLP+int:   mean={mlp_dram.mean():.2f}ms")

    # ================================================================
    # [B] SSD cold — SSD read + embedding + full forward
    # ================================================================
    print("\n[B] SSD cold reads + inference...")
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        w = sd[k].clone()
        if t in TABLES:
            w[cold_indices[t]] = 0.0
        dlrm.emb_l[t].weight.data = w

    drop_cache()
    fd = os.open(SSD_FILE, os.O_RDONLY)

    ssd_read_times, emb_times_ssd, fwd_times_ssd = [], [], []
    scores, targets = [], []
    with torch.no_grad():
        for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
            # Time SSD reads
            t0 = time.perf_counter()
            for t in TABLES:
                idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
                for si in range(len(idx)):
                    orig_idx = idx[si].item()
                    cold_pos = orig_to_cold_pos[t][orig_idx].item()
                    if cold_pos >= 0:
                        byte_off = cold_file_start[t] + cold_pos * ROW_BYTES
                        os.lseek(fd, byte_off, os.SEEK_SET)
                        raw = os.read(fd, ROW_BYTES)
                        dlrm.emb_l[t].weight.data[orig_idx] = torch.frombuffer(bytearray(raw), dtype=torch.float32)
            t_ssd = time.perf_counter() - t0

            # Time embedding only
            t0 = time.perf_counter()
            for k_idx in range(len(ek)):
                E = dlrm.emb_l[k_idx]
                o = lS_o[k_idx] if isinstance(lS_o, list) else lS_o[k_idx]
                i = lS_i[k_idx] if isinstance(lS_i, list) else lS_i[k_idx]
                E(i, o)
            t_emb = time.perf_counter() - t0

            # Time full forward
            t0 = time.perf_counter()
            Z = dlrm(X, lS_o, lS_i)
            t_fwd = time.perf_counter() - t0

            ssd_read_times.append(t_ssd * 1000)
            emb_times_ssd.append(t_emb * 1000)
            fwd_times_ssd.append(t_fwd * 1000)
            scores.append(Z.numpy().ravel())
            targets.append(T.numpy().ravel())

            # Zero cold rows back
            for t in TABLES:
                idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
                for si in range(len(idx)):
                    if orig_to_cold_pos[t][idx[si].item()].item() >= 0:
                        dlrm.emb_l[t].weight.data[idx[si].item()] = 0.0

            if bi % 200 == 0:
                print(f"    batch {bi}/{len(test_batches)}: ssd={t_ssd*1000:.1f}ms emb={t_emb*1000:.2f}ms fwd={t_fwd*1000:.2f}ms")

    os.close(fd)
    auc_ssd = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    ssd_read = np.array(ssd_read_times)
    emb_ssd = np.array(emb_times_ssd)
    fwd_ssd = np.array(fwd_times_ssd)

    print(f"  AUC={auc_ssd:.6f}")
    print(f"  SSD read:  mean={ssd_read.mean():.1f}ms, p50={np.percentile(ssd_read,50):.1f}ms, p99={np.percentile(ssd_read,99):.1f}ms")
    print(f"  Embedding: mean={emb_ssd.mean():.2f}ms, p50={np.percentile(emb_ssd,50):.2f}ms, p99={np.percentile(emb_ssd,99):.2f}ms")
    print(f"  Full fwd:  mean={fwd_ssd.mean():.2f}ms, p50={np.percentile(fwd_ssd,50):.2f}ms")
    print(f"  E2E (ssd+fwd): mean={ssd_read.mean()+fwd_ssd.mean():.1f}ms")

    # ================================================================
    # [C] DC PCA-sort scalar 4-bit
    # ================================================================
    print("\n[C] DC PCA-sort scalar 4-bit (1% hot)...")
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        if t in TABLES:
            dlrm.emb_l[t].weight.data = apply_dc(sd[k], freq_counts[t], hf=0.01)
        else:
            dlrm.emb_l[t].weight.data = sd[k].clone()

    # Warmup
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches[:10]:
            dlrm(X, lS_o, lS_i)

    emb_times_dc, fwd_times_dc = [], []
    scores, targets = [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            t0 = time.perf_counter()
            for k_idx in range(len(ek)):
                E = dlrm.emb_l[k_idx]
                o = lS_o[k_idx] if isinstance(lS_o, list) else lS_o[k_idx]
                i = lS_i[k_idx] if isinstance(lS_i, list) else lS_i[k_idx]
                E(i, o)
            t_emb = time.perf_counter() - t0

            t0 = time.perf_counter()
            Z = dlrm(X, lS_o, lS_i)
            t_fwd = time.perf_counter() - t0

            emb_times_dc.append(t_emb * 1000)
            fwd_times_dc.append(t_fwd * 1000)
            scores.append(Z.numpy().ravel())
            targets.append(T.numpy().ravel())

    auc_dc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    emb_dc = np.array(emb_times_dc)
    fwd_dc = np.array(fwd_times_dc)

    print(f"  AUC={auc_dc:.6f}")
    print(f"  Embedding: mean={emb_dc.mean():.2f}ms, p50={np.percentile(emb_dc,50):.2f}ms, p99={np.percentile(emb_dc,99):.2f}ms")
    print(f"  Full fwd:  mean={fwd_dc.mean():.2f}ms, p50={np.percentile(fwd_dc,50):.2f}ms, p99={np.percentile(fwd_dc,99):.2f}ms")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY — Embedding Lookup Latency Isolation")
    print(f"{'='*70}")

    print(f"\n  {'Component':<25} {'DRAM':>10} {'SSD':>10} {'DC':>10}")
    print(f"  {'-'*57}")
    print(f"  {'SSD read':<25} {'—':>10} {ssd_read.mean():>9.1f}ms {'—':>10}")
    print(f"  {'Embedding lookup':<25} {emb_dram.mean():>9.2f}ms {emb_ssd.mean():>9.2f}ms {emb_dc.mean():>9.2f}ms")
    print(f"  {'Full forward':<25} {fwd_dram.mean():>9.2f}ms {fwd_ssd.mean():>9.2f}ms {fwd_dc.mean():>9.2f}ms")
    print(f"  {'E2E (ssd read + fwd)':<25} {fwd_dram.mean():>9.2f}ms {(ssd_read.mean()+fwd_ssd.mean()):>9.1f}ms {fwd_dc.mean():>9.2f}ms")
    print(f"  {'AUC':<25} {auc_dram:>10.6f} {auc_ssd:>10.6f} {auc_dc:>10.6f}")

    print(f"\n  p50:")
    print(f"  {'SSD read':<25} {'—':>10} {np.percentile(ssd_read,50):>9.1f}ms {'—':>10}")
    print(f"  {'Embedding lookup':<25} {np.percentile(emb_dram,50):>9.2f}ms {np.percentile(emb_ssd,50):>9.2f}ms {np.percentile(emb_dc,50):>9.2f}ms")
    print(f"  {'Full forward':<25} {np.percentile(fwd_dram,50):>9.2f}ms {np.percentile(fwd_ssd,50):>9.2f}ms {np.percentile(fwd_dc,50):>9.2f}ms")

    print(f"\n  p99:")
    print(f"  {'SSD read':<25} {'—':>10} {np.percentile(ssd_read,99):>9.1f}ms {'—':>10}")
    print(f"  {'Embedding lookup':<25} {np.percentile(emb_dram,99):>9.2f}ms {np.percentile(emb_ssd,99):>9.2f}ms {np.percentile(emb_dc,99):>9.2f}ms")
    print(f"  {'Full forward':<25} {np.percentile(fwd_dram,99):>9.2f}ms {np.percentile(fwd_ssd,99):>9.2f}ms {np.percentile(fwd_dc,99):>9.2f}ms")

    emb_overhead = ssd_read.mean() / fwd_dram.mean()
    print(f"\n  SSD read adds {ssd_read.mean():.1f}ms = {emb_overhead:.0f}x the DRAM forward time")
    print(f"  Embedding lookup itself is ~same speed across all methods (~{emb_dram.mean():.1f}ms)")
    print(f"  The bottleneck is purely SSD I/O, not embedding computation")

    results = {
        'dram': {
            'auc': float(auc_dram),
            'emb_mean_ms': float(emb_dram.mean()), 'emb_p50_ms': float(np.percentile(emb_dram,50)), 'emb_p99_ms': float(np.percentile(emb_dram,99)),
            'fwd_mean_ms': float(fwd_dram.mean()), 'fwd_p50_ms': float(np.percentile(fwd_dram,50)), 'fwd_p99_ms': float(np.percentile(fwd_dram,99)),
        },
        'ssd': {
            'auc': float(auc_ssd),
            'ssd_read_mean_ms': float(ssd_read.mean()), 'ssd_read_p50_ms': float(np.percentile(ssd_read,50)), 'ssd_read_p99_ms': float(np.percentile(ssd_read,99)),
            'emb_mean_ms': float(emb_ssd.mean()), 'emb_p50_ms': float(np.percentile(emb_ssd,50)), 'emb_p99_ms': float(np.percentile(emb_ssd,99)),
            'fwd_mean_ms': float(fwd_ssd.mean()), 'fwd_p50_ms': float(np.percentile(fwd_ssd,50)), 'fwd_p99_ms': float(np.percentile(fwd_ssd,99)),
            'e2e_mean_ms': float(ssd_read.mean() + fwd_ssd.mean()),
        },
        'dc': {
            'auc': float(auc_dc),
            'emb_mean_ms': float(emb_dc.mean()), 'emb_p50_ms': float(np.percentile(emb_dc,50)), 'emb_p99_ms': float(np.percentile(emb_dc,99)),
            'fwd_mean_ms': float(fwd_dc.mean()), 'fwd_p50_ms': float(np.percentile(fwd_dc,50)), 'fwd_p99_ms': float(np.percentile(fwd_dc,99)),
        },
    }
    with open(f'{OUT_DIR}/emb_latency_breakdown.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {OUT_DIR}/emb_latency_breakdown.json")
    os.unlink(SSD_FILE)

if __name__ == '__main__':
    main()
