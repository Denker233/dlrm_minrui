#!/usr/bin/env python3
"""Generate a synthetic cold-embedding file + access offsets matching the
SATA baseline in NVME_BENCHMARK_INSTRUCTIONS.md, so bench_iouring can be run
without the trained model present.

Usage: generate_nvme_workload.py [out_dir]
Then:  ./bench_iouring <out_dir>/cold_iouring.bin <out_dir>/cold_offsets.bin 1660 256 5
"""
import numpy as np, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme0"
COLD_FILE = os.path.join(OUT, "cold_iouring.bin")
OFFSETS_FILE = os.path.join(OUT, "cold_offsets.bin")

TOTAL_COLD_ROWS = 33_378_031   # measured Criteo Kaggle cold rows
ROW_BYTES = 64                 # 16 dims x fp32
N_COLD_PER_BATCH = 1660        # measured cold lookups per batch

size = TOTAL_COLD_ROWS * ROW_BYTES
print(f"Generating {size/1024/1024:.0f} MB cold file ({TOTAL_COLD_ROWS:,} rows x {ROW_BYTES} B)...")
rng = np.random.default_rng(0)
with open(COLD_FILE, "wb") as f:
    chunk_rows = 1_000_000
    for i in range(0, TOTAL_COLD_ROWS, chunk_rows):
        n = min(chunk_rows, TOTAL_COLD_ROWS - i)
        f.write(rng.standard_normal(n * ROW_BYTES // 4, dtype=np.float32).tobytes())
print(f"Wrote {os.path.getsize(COLD_FILE)/1024/1024:.0f} MB -> {COLD_FILE}")

offsets = rng.integers(0, TOTAL_COLD_ROWS, size=N_COLD_PER_BATCH).astype(np.int64) * ROW_BYTES
offsets.tofile(OFFSETS_FILE)
print(f"Wrote {len(offsets)} offsets -> {OFFSETS_FILE}")
