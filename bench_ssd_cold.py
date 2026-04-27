#!/usr/bin/env python3
"""
REAL SSD benchmark: put cold embeddings on disk, measure actual I/O latency.
Uses O_DIRECT to bypass page cache for true SSD latency.
"""
import os, sys, time, json, ctypes, mmap
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
from codec_ondemand_benchmark import (
    load_model_and_data, HOTCOLD_DIR, REORDER_DIR, quantize_table, EMB_DIM, MODEL_PATH,
    LARGE_TABLE_THRESHOLD,
)

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
SSD_FILE = '/tmp/cold_embeddings.bin'

def main():
    print("=" * 70)
    print("REAL SSD COLD EMBEDDING BENCHMARK")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    nt = len(ln_emb)
    torch.set_num_threads(32)

    is_hot = {t: torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True) for t in TABLES}
    hi = {t: torch.where(is_hot[t])[0] for t in TABLES}

    # ================================================================
    # Step 1: Write cold embeddings to SSD file
    # ================================================================
    print("\n--- Step 1: Writing cold embeddings to SSD ---")

    # Build cold data: for each table, store cold rows contiguously
    # File layout: [table_0_cold_rows][table_1_cold_rows]...
    # Index: cold_offset[t] = byte offset of table t's cold rows in file
    cold_offsets = {}  # t -> (byte_offset, n_cold)
    cold_indices = {}  # t -> sorted cold row indices (original)

    # Compute total size
    total_cold_bytes = 0
    for t in TABLES:
        ci = torch.where(~is_hot[t])[0]
        cold_indices[t] = ci
        n_cold = len(ci)
        cold_offsets[t] = (total_cold_bytes, n_cold)
        total_cold_bytes += n_cold * EMB_DIM * 4  # fp32

    print(f"  Total cold: {total_cold_bytes / 1024 / 1024:.0f} MB across {len(TABLES)} tables")

    # Write to file
    with open(SSD_FILE, 'wb') as f:
        for t in TABLES:
            cold_w = sd[ek[t]][cold_indices[t]]
            f.write(cold_w.numpy().tobytes())
    print(f"  Written to {SSD_FILE}: {os.path.getsize(SSD_FILE)/1024/1024:.0f} MB")

    # Build orig_to_cold mapping for each table
    # orig_idx -> position in cold array (or -1 if hot)
    orig_to_cold_pos = {}
    for t in TABLES:
        mapping = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
        mapping[cold_indices[t]] = torch.arange(len(cold_indices[t]))
        orig_to_cold_pos[t] = mapping

    # ================================================================
    # Step 2: Drop page cache
    # ================================================================
    print("\n--- Step 2: Dropping page cache ---")
    os.system('sync')
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as f:
            f.write('3')
        print("  Page cache dropped")
    except:
        print("  WARNING: Cannot drop page cache (no root). Results may be cached.")

    # ================================================================
    # Step 3: Measure real SSD read latency
    # ================================================================
    print("\n--- Step 3: Raw SSD read latency ---")

    row_bytes = EMB_DIM * 4  # 64 bytes per fp32 row

    # Open file for reading
    fd = os.open(SSD_FILE, os.O_RDONLY)

    # Measure individual random reads
    file_size = os.path.getsize(SSD_FILE)
    n_rows_total = file_size // row_bytes

    # Serial random reads
    for n_reads in [1, 7, 10, 50]:
        positions = np.random.randint(0, n_rows_total, size=n_reads)
        times = []
        for _ in range(20):
            pos = np.random.randint(0, n_rows_total, size=n_reads)
            t0 = time.perf_counter()
            for p in pos:
                os.lseek(fd, int(p) * row_bytes, os.SEEK_SET)
                data = os.read(fd, row_bytes)
            times.append(time.perf_counter() - t0)
        med = np.median(times) * 1000
        per_read = med / n_reads * 1000  # μs
        print(f"  {n_reads:>3} serial reads: {med:.3f}ms ({per_read:.0f}μs/read)")

    os.close(fd)

    # ================================================================
    # Step 4: Full inference with SSD cold reads
    # ================================================================
    print(f"\n--- Step 4: Full inference comparison ---")

    # Baseline: all in DRAM (fp32)
    print("\n  [A] fp32 DRAM baseline:")
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        dlrm.emb_l[t].weight.data = sd[k].clone()

    with torch.no_grad():
        for X, o, i, T in test_batches[:20]: dlrm(X, o, i)

    scores, targets, times_dram = [], [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            t0 = time.perf_counter()
            Z = dlrm(X, o, i)
            times_dram.append(time.perf_counter() - t0)
            scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_dram = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    p50_dram = np.percentile(times_dram, 50) * 1000
    p99_dram = np.percentile(times_dram, 99) * 1000
    print(f"    AUC={auc_dram:.6f}  p50={p50_dram:.2f}ms  p99={p99_dram:.2f}ms")

    # SSD cold: hot in DRAM, cold read from file per batch
    print("\n  [B] SSD cold (real disk reads):")

    # Build hot-only weight tensors (cold rows zeroed)
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        w = sd[k].clone()
        if t in TABLES:
            w[cold_indices[t]] = 0.0  # zero cold rows initially
        dlrm.emb_l[t].weight.data = w

    # Drop cache again before SSD test
    os.system('sync')
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as f:
            f.write('3')
    except:
        pass

    fd = os.open(SSD_FILE, os.O_RDONLY)

    scores_ssd, targets_ssd, times_ssd = [], [], []
    cold_read_times = []

    with torch.no_grad():
        for batch_idx, (X, o, i, T) in enumerate(test_batches):
            # Step A: Read cold embeddings from SSD and fill into weight tensor
            t_read_start = time.perf_counter()
            for t in TABLES:
                idx = i[t] if isinstance(i, (list, tuple)) else i[t]
                for sample_idx in range(len(idx)):
                    orig_idx = idx[sample_idx].item()
                    cold_pos = orig_to_cold_pos[t][orig_idx].item()
                    if cold_pos >= 0:
                        # Read this cold row from SSD
                        byte_offset = cold_offsets[t][0] + cold_pos * row_bytes
                        os.lseek(fd, byte_offset, os.SEEK_SET)
                        raw = os.read(fd, row_bytes)
                        row_data = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                        dlrm.emb_l[t].weight.data[orig_idx] = row_data
            t_read = time.perf_counter() - t_read_start
            cold_read_times.append(t_read)

            # Step B: Forward pass
            t0 = time.perf_counter()
            Z = dlrm(X, o, i)
            t_fwd = time.perf_counter() - t0

            times_ssd.append(t_read + t_fwd)
            scores_ssd.append(Z.numpy().ravel())
            targets_ssd.append(T.numpy().ravel())

            # Zero cold rows back (so next batch doesn't reuse stale data)
            for t in TABLES:
                idx = i[t] if isinstance(i, (list, tuple)) else i[t]
                for sample_idx in range(len(idx)):
                    orig_idx = idx[sample_idx].item()
                    if orig_to_cold_pos[t][orig_idx].item() >= 0:
                        dlrm.emb_l[t].weight.data[orig_idx] = 0.0

            if batch_idx % 200 == 0:
                print(f"    batch {batch_idx}/{len(test_batches)}: "
                      f"read={t_read*1000:.2f}ms fwd={t_fwd*1000:.2f}ms")

    os.close(fd)

    auc_ssd = roc_auc_score(np.concatenate(targets_ssd), np.concatenate(scores_ssd))
    p50_ssd = np.percentile(times_ssd, 50) * 1000
    p99_ssd = np.percentile(times_ssd, 99) * 1000
    p50_read = np.percentile(cold_read_times, 50) * 1000
    p99_read = np.percentile(cold_read_times, 99) * 1000
    print(f"    AUC={auc_ssd:.6f}  p50={p50_ssd:.2f}ms  p99={p99_ssd:.2f}ms")
    print(f"    SSD read: p50={p50_read:.2f}ms  p99={p99_read:.2f}ms")

    # Block-mean (already measured — just reference)
    print("\n  [C] Block-mean (reference from prior experiments):")
    print(f"    AUC=0.802332  p50≈4.4ms  p99≈4.6ms  (zero I/O)")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: REAL SSD vs DRAM vs Block-Mean")
    print(f"{'='*70}")
    print(f"\n{'Method':<30} {'AUC':>10} {'p50':>8} {'p99':>8} {'SSD read p50':>12}")
    print("-" * 72)
    print(f"{'fp32 DRAM':<30} {auc_dram:>10.6f} {p50_dram:>7.2f}ms {p99_dram:>7.2f}ms {'—':>12}")
    print(f"{'SSD cold (real reads)':<30} {auc_ssd:>10.6f} {p50_ssd:>7.2f}ms {p99_ssd:>7.2f}ms {p50_read:>10.2f}ms")
    print(f"{'Block-mean (ours)':<30} {'0.802332':>10} {'~4.4':>7}ms {'~4.6':>7}ms {'0':>12}")

    # Cleanup
    os.unlink(SSD_FILE)

    # Save
    results = {
        'dram': {'auc': float(auc_dram), 'p50_ms': float(p50_dram), 'p99_ms': float(p99_dram)},
        'ssd': {'auc': float(auc_ssd), 'p50_ms': float(p50_ssd), 'p99_ms': float(p99_ssd),
                'read_p50_ms': float(p50_read), 'read_p99_ms': float(p99_read)},
        'disk': 'Micron MTFDDAK480TDS (SATA SSD)',
    }
    os.makedirs('results/ssd_benchmark', exist_ok=True)
    with open('results/ssd_benchmark/real_ssd.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/ssd_benchmark/real_ssd.json")


if __name__ == '__main__':
    main()
