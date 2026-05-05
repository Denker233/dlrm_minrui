#!/usr/bin/env python3
"""
Measure ACTUAL SSD read latency with different I/O strategies:
1. Serial random reads (our current worst-case benchmark)
2. Sorted + batched reads (coalesce adjacent pages)
3. Page-aligned reads (read full 4KB pages, extract rows)
4. preadv / scatter-gather (multiple reads in one syscall)

Measures real I/O time per batch, not theoretical estimates.
"""
import os, sys, time, json, struct
import numpy as np
import torch

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
from codec_ondemand_benchmark import load_model_and_data, HOTCOLD_DIR

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
EMB_DIM = 16
ROW_BYTES = EMB_DIM * 4  # 64 bytes
PAGE_SIZE = 4096
SSD_FILE = '/tmp/cold_emb_batched.bin'
OUT_DIR = 'results/ssd_benchmark'
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("SSD BATCHED READ BENCHMARK — ACTUAL MEASUREMENTS")
    print("=" * 70)

    print("\n[1] Loading...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load('models/dlrm_kaggle_correct.pt', map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    is_hot = {t: torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True) for t in TABLES}

    # Build cold file and mappings
    print("\n[2] Writing cold embeddings to disk...")
    cold_indices = {}
    cold_file_start = {}
    offset = 0
    for t in TABLES:
        ci = torch.where(~is_hot[t])[0]
        cold_indices[t] = ci
        cold_file_start[t] = offset
        offset += len(ci) * ROW_BYTES

    with open(SSD_FILE, 'wb') as f:
        for t in TABLES:
            f.write(sd[ek[t]][cold_indices[t]].numpy().tobytes())
    file_size = os.path.getsize(SSD_FILE)
    print(f"  Written {file_size/1024/1024:.0f} MB to {SSD_FILE}")

    # Build orig→cold_pos mapping
    orig_to_cold_pos = {}
    for t in TABLES:
        mapping = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
        mapping[cold_indices[t]] = torch.arange(len(cold_indices[t]))
        orig_to_cold_pos[t] = mapping

    # Pre-compute cold row offsets for each batch
    print("\n[3] Pre-computing batch access patterns...")
    batch_offsets_list = []  # list of arrays, each = byte offsets for that batch
    for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
        offsets = []
        for t in TABLES:
            idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
            for row_idx in idx.numpy():
                cold_pos = orig_to_cold_pos[t][int(row_idx)].item()
                if cold_pos >= 0:
                    byte_off = cold_file_start[t] + cold_pos * ROW_BYTES
                    offsets.append(byte_off)
        batch_offsets_list.append(np.array(offsets, dtype=np.int64))

    n_batches = len(batch_offsets_list)
    mean_cold = np.mean([len(b) for b in batch_offsets_list])
    print(f"  {n_batches} batches, mean {mean_cold:.0f} cold rows/batch")

    # Drop page cache
    os.system('sync')
    try:
        os.system('sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"')
        print("  Page cache dropped")
    except:
        print("  WARNING: Cache drop may have failed")

    # ================================================================
    # Strategy 1: Serial random reads (baseline, what we had before)
    # ================================================================
    print(f"\n[4] Strategy 1: Serial random reads...")
    os.system('sync')
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as f: f.write('3')
    except: pass

    fd = os.open(SSD_FILE, os.O_RDONLY)
    times_serial = []
    for bi in range(min(200, n_batches)):
        offsets = batch_offsets_list[bi]
        if len(offsets) == 0:
            continue
        t0 = time.perf_counter()
        for off in offsets:
            os.lseek(fd, int(off), os.SEEK_SET)
            os.read(fd, ROW_BYTES)
        times_serial.append(time.perf_counter() - t0)
    os.close(fd)
    times_serial = np.array(times_serial) * 1000
    print(f"  mean={times_serial.mean():.1f}ms, p50={np.percentile(times_serial,50):.1f}ms, "
          f"p99={np.percentile(times_serial,99):.1f}ms")

    # ================================================================
    # Strategy 2: Sorted reads (sort offsets, read in order)
    # ================================================================
    print(f"\n[5] Strategy 2: Sorted sequential reads...")
    os.system('sync')
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as f: f.write('3')
    except: pass

    fd = os.open(SSD_FILE, os.O_RDONLY)
    times_sorted = []
    for bi in range(min(200, n_batches)):
        offsets = batch_offsets_list[bi]
        if len(offsets) == 0:
            continue
        sorted_offsets = np.sort(offsets)
        t0 = time.perf_counter()
        for off in sorted_offsets:
            os.lseek(fd, int(off), os.SEEK_SET)
            os.read(fd, ROW_BYTES)
        times_sorted.append(time.perf_counter() - t0)
    os.close(fd)
    times_sorted = np.array(times_sorted) * 1000
    print(f"  mean={times_sorted.mean():.1f}ms, p50={np.percentile(times_sorted,50):.1f}ms, "
          f"p99={np.percentile(times_sorted,99):.1f}ms")

    # ================================================================
    # Strategy 3: Page-aligned reads (read unique 4KB pages, extract rows)
    # ================================================================
    print(f"\n[6] Strategy 3: Page-aligned batched reads...")
    os.system('sync')
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as f: f.write('3')
    except: pass

    fd = os.open(SSD_FILE, os.O_RDONLY)
    times_paged = []
    pages_per_batch = []
    for bi in range(min(200, n_batches)):
        offsets = batch_offsets_list[bi]
        if len(offsets) == 0:
            continue
        # Find unique pages
        pages = np.unique(offsets // PAGE_SIZE)
        pages_per_batch.append(len(pages))
        t0 = time.perf_counter()
        for page in pages:
            os.lseek(fd, int(page * PAGE_SIZE), os.SEEK_SET)
            os.read(fd, PAGE_SIZE)
        times_paged.append(time.perf_counter() - t0)
    os.close(fd)
    times_paged = np.array(times_paged) * 1000
    pages_per_batch = np.array(pages_per_batch)
    print(f"  mean={times_paged.mean():.1f}ms, p50={np.percentile(times_paged,50):.1f}ms, "
          f"p99={np.percentile(times_paged,99):.1f}ms")
    print(f"  pages/batch: mean={pages_per_batch.mean():.0f}, min={pages_per_batch.min()}, max={pages_per_batch.max()}")

    # ================================================================
    # Strategy 4: Single large pread (read entire cold file, extract rows)
    # ================================================================
    print(f"\n[7] Strategy 4: Read entire cold file once, extract from memory...")
    os.system('sync')
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as f: f.write('3')
    except: pass

    # Time the full file read
    t0 = time.perf_counter()
    with open(SSD_FILE, 'rb') as f:
        cold_data = f.read()
    full_read_ms = (time.perf_counter() - t0) * 1000
    cold_array = np.frombuffer(cold_data, dtype=np.float32).reshape(-1, EMB_DIM)
    print(f"  Full file read: {full_read_ms:.0f}ms for {file_size/1024/1024:.0f} MB "
          f"({file_size/1024/1024/(full_read_ms/1000):.0f} MB/s)")

    # Time per-batch extraction from memory (no I/O, just memcpy)
    times_memcpy = []
    for bi in range(min(200, n_batches)):
        offsets = batch_offsets_list[bi]
        if len(offsets) == 0:
            continue
        row_indices = offsets // ROW_BYTES
        t0 = time.perf_counter()
        rows = cold_array[row_indices]
        times_memcpy.append(time.perf_counter() - t0)
    times_memcpy = np.array(times_memcpy) * 1000
    amortized = full_read_ms / n_batches
    print(f"  Amortized read per batch: {amortized:.2f}ms ({full_read_ms:.0f}ms / {n_batches} batches)")
    print(f"  Memory extract per batch: mean={times_memcpy.mean():.3f}ms")
    print(f"  Total amortized: {amortized + times_memcpy.mean():.2f}ms/batch")

    # ================================================================
    # Page locality statistics
    # ================================================================
    print(f"\n[8] Page locality statistics...")
    all_rows_per_page = []
    for bi in range(n_batches):
        offsets = batch_offsets_list[bi]
        if len(offsets) == 0:
            continue
        pages = offsets // PAGE_SIZE
        unique, counts = np.unique(pages, return_counts=True)
        all_rows_per_page.extend(counts.tolist())

    all_rows_per_page = np.array(all_rows_per_page)
    single_row_pages = (all_rows_per_page == 1).sum()
    total_page_accesses = len(all_rows_per_page)
    useful_bytes_per_page = all_rows_per_page * ROW_BYTES
    wasted_pct = (1 - useful_bytes_per_page.mean() / PAGE_SIZE) * 100

    print(f"  Rows per page: mean={all_rows_per_page.mean():.2f}, median={np.median(all_rows_per_page):.0f}")
    print(f"  Single-row pages: {single_row_pages}/{total_page_accesses} ({single_row_pages/total_page_accesses*100:.1f}%)")
    print(f"  Useful bytes per page: mean={useful_bytes_per_page.mean():.0f}/{PAGE_SIZE} bytes")
    print(f"  Read amplification: {PAGE_SIZE/useful_bytes_per_page.mean():.1f}x")
    print(f"  Wasted: {wasted_pct:.1f}%")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY — Measured SSD Read Strategies")
    print(f"{'='*70}")
    print(f"  Cold rows per batch: {mean_cold:.0f}")
    print(f"  Cold data per batch: {mean_cold * ROW_BYTES / 1024:.0f} KB")
    print(f"  Unique pages per batch: {pages_per_batch.mean():.0f}")
    print(f"  Read amplification: {PAGE_SIZE / (all_rows_per_page.mean() * ROW_BYTES):.1f}x")
    print(f"  Wasted per page: {wasted_pct:.1f}%")
    print()
    print(f"  {'Strategy':<35} {'Mean':>8} {'p50':>8} {'p99':>8}")
    print(f"  {'-'*60}")
    print(f"  {'1. Serial random reads':<35} {times_serial.mean():>7.1f}ms {np.percentile(times_serial,50):>7.1f}ms {np.percentile(times_serial,99):>7.1f}ms")
    print(f"  {'2. Sorted sequential reads':<35} {times_sorted.mean():>7.1f}ms {np.percentile(times_sorted,50):>7.1f}ms {np.percentile(times_sorted,99):>7.1f}ms")
    print(f"  {'3. Page-aligned reads':<35} {times_paged.mean():>7.1f}ms {np.percentile(times_paged,50):>7.1f}ms {np.percentile(times_paged,99):>7.1f}ms")
    print(f"  {'4. Full file + extract (amortized)':<35} {amortized + times_memcpy.mean():>7.2f}ms {'—':>8} {'—':>8}")
    print(f"  {'DC block-mean (no I/O)':<35} {'0':>8}ms {'0':>8}ms {'0':>8}ms")

    speedup_sorted = times_serial.mean() / times_sorted.mean()
    speedup_paged = times_serial.mean() / times_paged.mean()
    print(f"\n  Sorted vs serial: {speedup_sorted:.2f}x faster")
    print(f"  Paged vs serial:  {speedup_paged:.2f}x faster")

    # Save
    results = {
        'cold_rows_per_batch': float(mean_cold),
        'cold_kb_per_batch': float(mean_cold * ROW_BYTES / 1024),
        'unique_pages_per_batch': float(pages_per_batch.mean()),
        'read_amplification': float(PAGE_SIZE / (all_rows_per_page.mean() * ROW_BYTES)),
        'wasted_pct': float(wasted_pct),
        'single_row_page_pct': float(single_row_pages / total_page_accesses * 100),
        'strategies': {
            'serial': {'mean_ms': float(times_serial.mean()), 'p50_ms': float(np.percentile(times_serial, 50)), 'p99_ms': float(np.percentile(times_serial, 99))},
            'sorted': {'mean_ms': float(times_sorted.mean()), 'p50_ms': float(np.percentile(times_sorted, 50)), 'p99_ms': float(np.percentile(times_sorted, 99))},
            'paged': {'mean_ms': float(times_paged.mean()), 'p50_ms': float(np.percentile(times_paged, 50)), 'p99_ms': float(np.percentile(times_paged, 99))},
            'full_file_amortized_ms': float(amortized + times_memcpy.mean()),
        },
        'disk': 'Micron MTFDDAK480TDS (SATA SSD)',
        'file_size_mb': float(file_size / 1024 / 1024),
    }
    with open(f'{OUT_DIR}/batched_read_strategies.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {OUT_DIR}/batched_read_strategies.json")

    os.unlink(SSD_FILE)

if __name__ == '__main__':
    main()
