#!/usr/bin/env python3
"""
Measure actual page locality of cold embedding accesses.
For each batch, check how many distinct 4KB pages the cold rows land on,
and how much of each page is actually used (vs wasted).
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
from codec_ondemand_benchmark import load_model_and_data, HOTCOLD_DIR

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
EMB_DIM = 16
ROW_BYTES = EMB_DIM * 4  # 64 bytes per row (fp32)
PAGE_SIZE = 4096  # 4KB page
ROWS_PER_PAGE = PAGE_SIZE // ROW_BYTES  # 64 rows per page

OUT_DIR = 'results/ssd_benchmark'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("PAGE LOCALITY ANALYSIS FOR COLD EMBEDDING ACCESSES")
print("=" * 70)

print("\nLoading model and data...")
dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
sd = torch.load('models/dlrm_kaggle_correct.pt', map_location='cpu', weights_only=False)['state_dict']
ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
            key=lambda x: int(x.split('.')[1]))
test_batches = list(test_ld)

is_hot = {t: torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True) for t in TABLES}

# Build cold layout: if cold rows were stored contiguously on disk,
# compute byte offsets for each cold row
cold_disk_offset = {}  # {table: {orig_row_idx: byte_offset_in_file}}
cold_file_offset = {}  # {table: start_offset_in_combined_file}
current_offset = 0
for t in TABLES:
    cold_idx = torch.where(~is_hot[t])[0]
    cold_file_offset[t] = current_offset
    cold_disk_offset[t] = {}
    for i, idx in enumerate(cold_idx.numpy()):
        cold_disk_offset[t][int(idx)] = current_offset + i * ROW_BYTES
    current_offset += len(cold_idx) * ROW_BYTES

total_cold_file_size = current_offset
total_cold_rows = sum(len(torch.where(~is_hot[t])[0]) for t in TABLES)
total_pages = (total_cold_file_size + PAGE_SIZE - 1) // PAGE_SIZE

print(f"\nCold file layout:")
print(f"  Total cold rows: {total_cold_rows:,}")
print(f"  Row size: {ROW_BYTES} bytes")
print(f"  Rows per 4KB page: {ROWS_PER_PAGE}")
print(f"  Total cold file: {total_cold_file_size/1024/1024:.0f} MB")
print(f"  Total 4KB pages: {total_pages:,}")

# Analyze page locality per batch
print(f"\nAnalyzing {len(test_batches)} batches...")

batch_stats = []
all_unique_pages = []
all_cold_rows = []
all_useful_bytes = []
all_wasted_bytes = []
all_rows_per_page = []

for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
    # Collect all cold row disk offsets for this batch
    cold_offsets = []
    for t in TABLES:
        idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
        for row_idx in idx.numpy():
            row_idx = int(row_idx)
            if row_idx in cold_disk_offset[t]:
                cold_offsets.append(cold_disk_offset[t][row_idx])

    if len(cold_offsets) == 0:
        continue

    cold_offsets = np.array(cold_offsets)
    n_cold = len(cold_offsets)

    # Which pages do these offsets land on?
    pages = cold_offsets // PAGE_SIZE
    unique_pages = np.unique(pages)
    n_unique_pages = len(unique_pages)

    # How many rows per page?
    page_counts = np.bincount(pages.astype(np.int64) - pages.min())
    page_counts = page_counts[page_counts > 0]

    # Bytes analysis
    useful_bytes = n_cold * ROW_BYTES
    read_bytes = n_unique_pages * PAGE_SIZE
    wasted_pct = (1 - useful_bytes / read_bytes) * 100 if read_bytes > 0 else 0

    all_unique_pages.append(n_unique_pages)
    all_cold_rows.append(n_cold)
    all_useful_bytes.append(useful_bytes)
    all_wasted_bytes.append(read_bytes - useful_bytes)
    all_rows_per_page.extend(page_counts.tolist())

    batch_stats.append({
        'n_cold': n_cold,
        'n_unique_pages': n_unique_pages,
        'useful_bytes': useful_bytes,
        'read_bytes': read_bytes,
        'wasted_pct': wasted_pct,
        'avg_rows_per_page': n_cold / n_unique_pages,
        'max_rows_per_page': int(page_counts.max()),
    })

all_unique_pages = np.array(all_unique_pages)
all_cold_rows = np.array(all_cold_rows)
all_useful_bytes = np.array(all_useful_bytes)
all_wasted_bytes = np.array(all_wasted_bytes)
all_rows_per_page = np.array(all_rows_per_page)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")

print(f"\n  Per-batch statistics ({len(batch_stats)} batches):")
print(f"    Cold rows/batch:        mean={all_cold_rows.mean():.0f}, "
      f"min={all_cold_rows.min()}, max={all_cold_rows.max()}")
print(f"    Unique pages/batch:     mean={all_unique_pages.mean():.0f}, "
      f"min={all_unique_pages.min()}, max={all_unique_pages.max()}")
print(f"    Rows per page:          mean={all_cold_rows.mean()/all_unique_pages.mean():.2f}, "
      f"median={np.median(all_rows_per_page):.0f}, max={all_rows_per_page.max()}")

# Page sharing: how many pages have >1 row from the same batch?
shared_pages = (all_rows_per_page > 1).sum()
total_page_accesses = len(all_rows_per_page)
print(f"    Pages with >1 row:      {shared_pages}/{total_page_accesses} ({shared_pages/total_page_accesses*100:.1f}%)")
print(f"    Pages with exactly 1:   {(all_rows_per_page == 1).sum()}/{total_page_accesses} "
      f"({(all_rows_per_page == 1).sum()/total_page_accesses*100:.1f}%)")

mean_useful = all_useful_bytes.mean()
mean_read = all_unique_pages.mean() * PAGE_SIZE
wasted_pct = (1 - mean_useful / mean_read) * 100
print(f"\n  I/O efficiency:")
print(f"    Useful data/batch:      {mean_useful/1024:.1f} KB ({all_cold_rows.mean():.0f} × {ROW_BYTES} bytes)")
print(f"    Pages read/batch:       {all_unique_pages.mean():.0f} × 4KB = {mean_read/1024:.0f} KB")
print(f"    Read amplification:     {mean_read/mean_useful:.1f}x")
print(f"    Wasted:                 {wasted_pct:.1f}%")

# Coalescing potential: if we sort offsets, how many sequential runs?
print(f"\n  Coalescing analysis (last batch sample):")
if len(cold_offsets) > 0:
    sorted_offsets = np.sort(cold_offsets)
    # A "run" is a set of consecutive pages
    sorted_pages = sorted_offsets // PAGE_SIZE
    unique_sorted = np.unique(sorted_pages)
    diffs = np.diff(unique_sorted)
    n_sequential = (diffs == 1).sum()
    n_runs = 1 + (diffs > 1).sum()  # number of disjoint sequential runs
    avg_run_len = len(unique_sorted) / n_runs

    print(f"    Unique pages: {len(unique_sorted)}")
    print(f"    Sequential page pairs: {n_sequential}/{len(diffs)} ({n_sequential/max(len(diffs),1)*100:.1f}%)")
    print(f"    Disjoint runs: {n_runs}")
    print(f"    Avg run length: {avg_run_len:.1f} pages")
    print(f"    If merged into runs: {n_runs} I/Os instead of {len(unique_sorted)}")

# NVMe latency estimate
print(f"\n  Estimated NVMe latency:")
mean_pages = all_unique_pages.mean()
for qd in [1, 4, 16, 64]:
    rounds = np.ceil(mean_pages / qd)
    latency_ms = rounds * 0.01  # 10μs per round
    print(f"    QD={qd:>2}: {rounds:.0f} rounds × 10μs = {latency_ms:.1f}ms")

print(f"\n  Comparison:")
print(f"    SATA serial (our benchmark):  {all_unique_pages.mean() * 0.05:.0f}ms")
print(f"    NVMe QD=64:                   {np.ceil(all_unique_pages.mean()/64) * 0.01:.1f}ms")
print(f"    DC block-mean:                0ms")

# Save results
output = {
    'row_bytes': ROW_BYTES,
    'page_size': PAGE_SIZE,
    'rows_per_page': ROWS_PER_PAGE,
    'total_cold_rows': int(total_cold_rows),
    'total_cold_mb': total_cold_file_size / 1024 / 1024,
    'total_pages': int(total_pages),
    'per_batch': {
        'cold_rows_mean': float(all_cold_rows.mean()),
        'unique_pages_mean': float(all_unique_pages.mean()),
        'rows_per_page_mean': float(all_cold_rows.mean() / all_unique_pages.mean()),
        'rows_per_page_median': float(np.median(all_rows_per_page)),
        'pct_single_row_pages': float((all_rows_per_page == 1).sum() / len(all_rows_per_page) * 100),
        'useful_kb_mean': float(mean_useful / 1024),
        'read_kb_mean': float(mean_read / 1024),
        'read_amplification': float(mean_read / mean_useful),
        'wasted_pct': float(wasted_pct),
    },
}
with open(f'{OUT_DIR}/page_locality.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved {OUT_DIR}/page_locality.json")
