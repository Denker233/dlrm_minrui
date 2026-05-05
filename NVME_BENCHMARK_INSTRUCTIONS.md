# NVMe Cold Embedding Benchmark Instructions

## Goal
Measure whether NVMe + io_uring can make SSD-based cold embedding serving fast enough (under 10ms SLA). Compare with SATA SSD results from dlrm-90 machine.

## SATA Baseline Results (Micron MTFDDAK480TDS, dlrm-90)

```
=== io_uring Batched Read (2,316 cold rows × 64 bytes) ===
Serial pread (QD=1):  117.4ms
io_uring QD=1:        116.4ms  (1.0x)
io_uring QD=2:         59.2ms  (2.0x)
io_uring QD=4:         34.0ms  (3.5x)
io_uring QD=8:         22.0ms  (5.3x)
io_uring QD=16:        17.0ms  (6.9x)
io_uring QD=32:        14.9ms  (7.9x)  ← SATA NCQ limit
io_uring QD=64:        15.0ms  (7.9x)
io_uring QD=128:       13.6ms  (8.7x)
io_uring QD=256:       14.0ms  (8.4x)

Page-aligned QD=32:    13.2ms  (8.9x, 1,004 pages)

fio 4K random QD=1:     6,818 IOPS (142μs)
fio 4K random QD=32:   79,526 IOPS (402μs)
fio sequential:        492 MB/s

=== Embedding Latency Breakdown ===
DRAM embedding lookup:  1.46ms
DRAM full forward:      4.83ms
SSD read (serial):      104.4ms
SSD full e2e:           111.6ms
DC block-mean forward:  4.40ms  (0ms I/O, 10.9 MB DRAM)

=== Access Pattern ===
Cold rows per batch:    1,660 (at 4.3% hot fraction)
Cold data needed:       104 KB (1,660 × 64 bytes)
Unique 4KB pages:       882
SSD data actually read: 3,528 KB (34x read amplification)
Single-row pages:       91.7%
Wasted bytes:           97.1%
Cold file size:         1,969 MB (33M cold rows)

=== Overhead Breakdown ===
Python loop:            0.2ms (0%)
Syscalls (lseek+read):  1.8ms (2%)
SSD I/O:                102.3ms (98%)
```

## Step 1: System Info
```bash
nvme list
lsblk -d -o NAME,SIZE,MODEL,TRAN
cat /sys/block/nvme0n1/queue/scheduler 2>/dev/null
uname -r
```

## Step 2: fio Raw IOPS Baseline
```bash
# Install fio if needed
sudo apt-get install -y fio

# Create 2GB test file on NVMe
dd if=/dev/zero of=/tmp/fio_test.bin bs=1M count=2048

# 4K random read at different queue depths
for qd in 1 4 8 16 32 64 128 256; do
  echo "=== QD=$qd ==="
  fio --name=test --filename=/tmp/fio_test.bin \
    --rw=randread --bs=4k --ioengine=libaio --direct=1 \
    --iodepth=$qd --numjobs=1 --runtime=10 --time_based \
    --output-format=json 2>/dev/null | python3 -c "
import sys,json; d=json.load(sys.stdin)['jobs'][0]['read']
print(f'  QD=$qd: {d[\"iops\"]:.0f} IOPS, {d[\"lat_ns\"][\"mean\"]/1000:.0f}us, {d[\"bw\"]/1024:.0f} MB/s')"
done

# 512B random read (matches our 64-byte row size, minimum O_DIRECT)
for qd in 1 32 128; do
  fio --name=test --filename=/tmp/fio_test.bin \
    --rw=randread --bs=512 --ioengine=libaio --direct=1 \
    --iodepth=$qd --numjobs=1 --runtime=10 --time_based \
    --output-format=json 2>/dev/null | python3 -c "
import sys,json; d=json.load(sys.stdin)['jobs'][0]['read']
print(f'  512B QD=$qd: {d[\"iops\"]:.0f} IOPS, {d[\"lat_ns\"][\"mean\"]/1000:.0f}us')"
done

# Sequential bandwidth
fio --name=test --filename=/tmp/fio_test.bin \
  --rw=read --bs=128k --ioengine=libaio --direct=1 \
  --iodepth=32 --numjobs=1 --runtime=10 --time_based \
  --output-format=json 2>/dev/null | python3 -c "
import sys,json; d=json.load(sys.stdin)['jobs'][0]['read']
print(f'  Sequential: {d[\"bw\"]/1024:.0f} MB/s')"

rm /tmp/fio_test.bin
```

## Step 3: Build io_uring Benchmark
```bash
# Install liburing if needed
sudo apt-get install -y liburing-dev

# Build from repo
cd dlrm_minrui
gcc -O2 -o bench_iouring bench_iouring.c -luring
```

## Step 4: Generate Synthetic Workload
If the model/dataset is NOT on this machine, generate a synthetic workload that matches our measured access pattern:

```python
#!/usr/bin/env python3
"""Generate synthetic cold embedding file + access offsets matching SATA benchmark."""
import numpy as np
import os

COLD_FILE = '/tmp/cold_iouring.bin'
OFFSETS_FILE = '/tmp/cold_offsets.bin'

# Parameters matching Criteo Kaggle cold embeddings
TOTAL_COLD_ROWS = 33_378_031  # 33M cold rows
ROW_BYTES = 64                 # 16 dims × 4 bytes fp32
FILE_SIZE = TOTAL_COLD_ROWS * ROW_BYTES  # ~2 GB
N_COLD_PER_BATCH = 1660       # measured cold lookups per batch

print(f"Generating {FILE_SIZE/1024/1024:.0f} MB cold file ({TOTAL_COLD_ROWS:,} rows × {ROW_BYTES} bytes)...")

# Write cold file in chunks (2GB total)
with open(COLD_FILE, 'wb') as f:
    chunk_rows = 1_000_000
    chunk_bytes = chunk_rows * ROW_BYTES
    for i in range(0, TOTAL_COLD_ROWS, chunk_rows):
        n = min(chunk_rows, TOTAL_COLD_ROWS - i)
        f.write(np.random.randn(n * ROW_BYTES // 4).astype(np.float32).tobytes())
        if i % 5_000_000 == 0:
            print(f"  {i/TOTAL_COLD_ROWS*100:.0f}%")

print(f"Written {os.path.getsize(COLD_FILE)/1024/1024:.0f} MB")

# Generate random offsets (simulating power-law access pattern)
offsets = np.random.randint(0, TOTAL_COLD_ROWS, size=N_COLD_PER_BATCH).astype(np.int64) * ROW_BYTES
offsets.tofile(OFFSETS_FILE)
print(f"Generated {len(offsets)} offsets → {OFFSETS_FILE}")
print(f"\nRun: ./bench_iouring {COLD_FILE} {OFFSETS_FILE} {N_COLD_PER_BATCH} 256 5")
```

Save as `generate_nvme_workload.py` and run:
```bash
python3 generate_nvme_workload.py
```

## Step 4b: Use Real Model Data (if available)
If the model and dataset are on this machine:
```bash
# Generate real cold file and offsets
python3 bench_ssd_systems.py  # this writes cold file + runs the full benchmark
```

## Step 5: Run io_uring Benchmark
```bash
# Drop page cache before each run
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Run with QD up to 256, 5 rounds each
./bench_iouring /tmp/cold_iouring.bin /tmp/cold_offsets.bin 1660 256 5
```

## Step 6: Report Results
Please report:
1. NVMe model and specs (from `nvme list`)
2. fio results at each QD
3. io_uring benchmark output (full table)
4. Whether any QD achieves < 5ms (matching DRAM forward time)
5. Whether any QD achieves < 10ms (SLA target)

## Key Questions
1. At what QD does NVMe saturate?
2. Does NVMe + io_uring get 1,660 cold reads under 10ms?
3. What's the best-case NVMe latency for our access pattern?
4. Even at best-case NVMe, how does it compare to DC's 0ms I/O?
5. Considering NVMe SSD costs ($100-300/TB) vs DRAM savings from DC compression (2GB → 11MB), which is more cost-effective?
