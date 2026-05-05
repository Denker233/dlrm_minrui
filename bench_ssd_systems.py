#!/usr/bin/env python3
"""
Systems-level SSD benchmark for storage expert mentor.
Measures: fio raw IOPS, O_DIRECT, syscall overhead, mmap, io scheduler.
"""
import os, sys, time, json, subprocess, ctypes, mmap as mmap_mod
import numpy as np
import torch

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
from codec_ondemand_benchmark import load_model_and_data, MODEL_PATH, HOTCOLD_DIR

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
EMB_DIM = 16
ROW_BYTES = EMB_DIM * 4  # 64 bytes
PAGE_SIZE = 4096
SSD_FILE = '/tmp/cold_emb_systems.bin'
OUT_DIR = 'results/ssd_benchmark'
os.makedirs(OUT_DIR, exist_ok=True)

def drop_cache():
    os.system('sync')
    os.system('sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"')

def main():
    print("=" * 70)
    print("SYSTEMS-LEVEL SSD BENCHMARK")
    print("=" * 70)

    # System info
    print("\n[0] System info...")
    sched = open('/sys/block/sda/queue/scheduler').read().strip()
    nr_req = open('/sys/block/sda/queue/nr_requests').read().strip()
    print(f"  Disk: Micron MTFDDAK480TDS (SATA SSD)")
    print(f"  I/O scheduler: {sched}")
    print(f"  Queue depth: {nr_req}")
    print(f"  Row size: {ROW_BYTES} bytes (= 1 cache line)")
    print(f"  Page size: {PAGE_SIZE} bytes")
    print(f"  Rows per page: {PAGE_SIZE // ROW_BYTES}")

    # ================================================================
    # [1] fio raw SSD performance
    # ================================================================
    print(f"\n[1] fio raw SSD IOPS (ground truth)...")

    # Create test file
    print("  Creating 2GB test file...")
    subprocess.run(['dd', 'if=/dev/zero', f'of={SSD_FILE}', 'bs=1M', 'count=2048'],
                   capture_output=True)

    fio_results = {}

    # 4K random read, QD=1
    print("\n  --- 4K random read, QD=1 ---")
    r = subprocess.run([
        'fio', '--name=test', f'--filename={SSD_FILE}',
        '--rw=randread', '--bs=4k', '--ioengine=libaio', '--direct=1',
        '--iodepth=1', '--numjobs=1', '--runtime=10', '--time_based',
        '--output-format=json'
    ], capture_output=True, text=True)
    fio_out = json.loads(r.stdout)
    iops_4k_qd1 = fio_out['jobs'][0]['read']['iops']
    lat_4k_qd1 = fio_out['jobs'][0]['read']['lat_ns']['mean'] / 1000  # μs
    print(f"  IOPS: {iops_4k_qd1:.0f}, latency: {lat_4k_qd1:.0f}μs")
    fio_results['4k_qd1'] = {'iops': iops_4k_qd1, 'lat_us': lat_4k_qd1}

    # 4K random read, QD=32 (SATA NCQ max)
    print("\n  --- 4K random read, QD=32 ---")
    r = subprocess.run([
        'fio', '--name=test', f'--filename={SSD_FILE}',
        '--rw=randread', '--bs=4k', '--ioengine=libaio', '--direct=1',
        '--iodepth=32', '--numjobs=1', '--runtime=10', '--time_based',
        '--output-format=json'
    ], capture_output=True, text=True)
    fio_out = json.loads(r.stdout)
    iops_4k_qd32 = fio_out['jobs'][0]['read']['iops']
    lat_4k_qd32 = fio_out['jobs'][0]['read']['lat_ns']['mean'] / 1000
    print(f"  IOPS: {iops_4k_qd32:.0f}, latency: {lat_4k_qd32:.0f}μs")
    fio_results['4k_qd32'] = {'iops': iops_4k_qd32, 'lat_us': lat_4k_qd32}

    # 512B random read (min O_DIRECT size; SSD reads 4KB internally for 64B anyway), QD=1
    print("\n  --- 512B random read, QD=1 (proxy for 64B) ---")
    r = subprocess.run([
        'fio', '--name=test', f'--filename={SSD_FILE}',
        '--rw=randread', '--bs=512', '--ioengine=libaio', '--direct=1',
        '--iodepth=1', '--numjobs=1', '--runtime=10', '--time_based',
        '--output-format=json'
    ], capture_output=True, text=True)
    fio_out = json.loads(r.stdout)
    iops_512_qd1 = fio_out['jobs'][0]['read']['iops']
    lat_512_qd1 = fio_out['jobs'][0]['read']['lat_ns']['mean'] / 1000
    print(f"  IOPS: {iops_512_qd1:.0f}, latency: {lat_512_qd1:.0f}μs")
    fio_results['512b_qd1'] = {'iops': iops_512_qd1, 'lat_us': lat_512_qd1}

    # 512B random read, QD=32
    print("\n  --- 512B random read, QD=32 ---")
    r = subprocess.run([
        'fio', '--name=test', f'--filename={SSD_FILE}',
        '--rw=randread', '--bs=512', '--ioengine=libaio', '--direct=1',
        '--iodepth=32', '--numjobs=1', '--runtime=10', '--time_based',
        '--output-format=json'
    ], capture_output=True, text=True)
    fio_out = json.loads(r.stdout)
    iops_512_qd32 = fio_out['jobs'][0]['read']['iops']
    lat_512_qd32 = fio_out['jobs'][0]['read']['lat_ns']['mean'] / 1000
    print(f"  IOPS: {iops_512_qd32:.0f}, latency: {lat_512_qd32:.0f}μs")
    fio_results['512b_qd32'] = {'iops': iops_512_qd32, 'lat_us': lat_512_qd32}

    # Use 512B as proxy for 64B (SSD reads at least 512B internally)
    iops_64_qd1 = iops_512_qd1
    lat_64_qd1 = lat_512_qd1
    iops_64_qd32 = iops_512_qd32
    lat_64_qd32 = lat_512_qd32

    # Sequential read bandwidth
    print("\n  --- Sequential read, 128K ---")
    r = subprocess.run([
        'fio', '--name=test', f'--filename={SSD_FILE}',
        '--rw=read', '--bs=128k', '--ioengine=libaio', '--direct=1',
        '--iodepth=32', '--numjobs=1', '--runtime=10', '--time_based',
        '--output-format=json'
    ], capture_output=True, text=True)
    fio_out = json.loads(r.stdout)
    bw_seq = fio_out['jobs'][0]['read']['bw'] / 1024  # MB/s
    print(f"  Bandwidth: {bw_seq:.0f} MB/s")
    fio_results['seq_128k'] = {'bw_mbs': bw_seq}

    # Theoretical minimum for our workload
    theoretical_min_qd1 = 1660 / iops_64_qd1 * 1000  # ms
    theoretical_min_qd32 = 1660 / iops_64_qd32 * 1000
    print(f"\n  Theoretical minimum for 1,660 reads:")
    print(f"    QD=1:  1660 / {iops_64_qd1:.0f} IOPS = {theoretical_min_qd1:.1f}ms")
    print(f"    QD=32: 1660 / {iops_64_qd32:.0f} IOPS = {theoretical_min_qd32:.1f}ms")

    os.unlink(SSD_FILE)

    # ================================================================
    # [2] Actual embedding reads: buffered vs O_DIRECT vs mmap
    # ================================================================
    print(f"\n[2] Writing real cold embeddings...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    is_hot = {t: torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True) for t in TABLES}

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

    # Write aligned for O_DIRECT (must be 512-byte aligned)
    with open(SSD_FILE, 'wb') as f:
        for t in TABLES:
            f.write(sd[ek[t]][cold_indices[t]].numpy().tobytes())
    file_size = os.path.getsize(SSD_FILE)
    print(f"  Written {file_size/1024/1024:.0f} MB")

    # Pre-compute batch offsets
    batch_offsets = []
    for bi, (X, lS_o, lS_i, T) in enumerate(test_batches[:200]):
        offsets = []
        for t in TABLES:
            idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
            for row_idx in idx.numpy():
                cold_pos = orig_to_cold_pos[t][int(row_idx)].item()
                if cold_pos >= 0:
                    offsets.append(cold_file_start[t] + cold_pos * ROW_BYTES)
        batch_offsets.append(np.array(offsets, dtype=np.int64))

    N_BATCHES = len(batch_offsets)
    mean_cold = np.mean([len(b) for b in batch_offsets])
    print(f"  {N_BATCHES} batches, mean {mean_cold:.0f} cold rows")

    # ================================================================
    # [2a] Buffered read (lseek + read, our standard approach)
    # ================================================================
    print(f"\n[2a] Buffered read (lseek + read)...")
    drop_cache()
    fd = os.open(SSD_FILE, os.O_RDONLY)
    times_buf = []
    for offsets in batch_offsets:
        t0 = time.perf_counter()
        for off in offsets:
            os.lseek(fd, int(off), os.SEEK_SET)
            os.read(fd, ROW_BYTES)
        times_buf.append(time.perf_counter() - t0)
    os.close(fd)
    times_buf = np.array(times_buf) * 1000
    print(f"  mean={times_buf.mean():.1f}ms, p50={np.percentile(times_buf,50):.1f}ms, p99={np.percentile(times_buf,99):.1f}ms")

    # ================================================================
    # [2b] O_DIRECT read (bypass page cache)
    # ================================================================
    print(f"\n[2b] O_DIRECT read (bypass page cache)...")
    drop_cache()
    try:
        fd = os.open(SSD_FILE, os.O_RDONLY | os.O_DIRECT)
        # O_DIRECT requires aligned buffers and aligned offsets
        # Our 64-byte rows aren't 512-aligned, so we read 4K pages
        times_direct = []
        for offsets in batch_offsets:
            pages = np.unique(offsets // PAGE_SIZE)
            buf = ctypes.create_string_buffer(PAGE_SIZE)
            t0 = time.perf_counter()
            for page in pages:
                os.lseek(fd, int(page * PAGE_SIZE), os.SEEK_SET)
                os.read(fd, PAGE_SIZE)
            times_direct.append(time.perf_counter() - t0)
        os.close(fd)
        times_direct = np.array(times_direct) * 1000
        print(f"  mean={times_direct.mean():.1f}ms, p50={np.percentile(times_direct,50):.1f}ms, p99={np.percentile(times_direct,99):.1f}ms")
    except OSError as e:
        print(f"  O_DIRECT failed: {e}")
        times_direct = np.array([0])

    # ================================================================
    # [2c] mmap read (OS manages paging)
    # ================================================================
    print(f"\n[2c] mmap read...")
    drop_cache()
    fd = os.open(SSD_FILE, os.O_RDONLY)
    mm = mmap_mod.mmap(fd, 0, access=mmap_mod.ACCESS_READ)
    times_mmap = []
    for offsets in batch_offsets:
        t0 = time.perf_counter()
        for off in offsets:
            mm.seek(int(off))
            mm.read(ROW_BYTES)
        times_mmap.append(time.perf_counter() - t0)
    mm.close()
    os.close(fd)
    times_mmap = np.array(times_mmap) * 1000
    print(f"  mean={times_mmap.mean():.1f}ms, p50={np.percentile(times_mmap,50):.1f}ms, p99={np.percentile(times_mmap,99):.1f}ms")

    # ================================================================
    # [2d] pread (no lseek, one syscall per read)
    # ================================================================
    print(f"\n[2d] pread (no lseek needed)...")
    drop_cache()
    fd = os.open(SSD_FILE, os.O_RDONLY)
    times_pread = []
    for offsets in batch_offsets:
        t0 = time.perf_counter()
        for off in offsets:
            os.pread(fd, ROW_BYTES, int(off))
        times_pread.append(time.perf_counter() - t0)
    os.close(fd)
    times_pread = np.array(times_pread) * 1000
    print(f"  mean={times_pread.mean():.1f}ms, p50={np.percentile(times_pread,50):.1f}ms, p99={np.percentile(times_pread,99):.1f}ms")

    # ================================================================
    # [3] Syscall overhead measurement
    # ================================================================
    print(f"\n[3] Syscall overhead measurement...")

    # [3a] Measure pure syscall overhead (read from /dev/zero — no SSD)
    print("\n  [3a] Pure syscall overhead (read 64B from /dev/zero)...")
    fd_zero = os.open('/dev/zero', os.O_RDONLY)
    n_calls = 1660
    times_syscall = []
    for _ in range(100):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            os.read(fd_zero, ROW_BYTES)
        times_syscall.append(time.perf_counter() - t0)
    os.close(fd_zero)
    times_syscall = np.array(times_syscall) * 1000
    per_syscall_us = times_syscall.mean() / n_calls * 1000
    print(f"  {n_calls} reads: mean={times_syscall.mean():.2f}ms ({per_syscall_us:.1f}μs/syscall)")

    # [3b] Measure lseek + read overhead (from /dev/zero)
    print("\n  [3b] lseek + read overhead (from /dev/zero)...")
    fd_zero = os.open('/dev/zero', os.O_RDONLY)
    times_lseek_read = []
    for _ in range(100):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            os.lseek(fd_zero, 0, os.SEEK_SET)
            os.read(fd_zero, ROW_BYTES)
        times_lseek_read.append(time.perf_counter() - t0)
    os.close(fd_zero)
    times_lseek_read = np.array(times_lseek_read) * 1000
    per_lseek_read_us = times_lseek_read.mean() / n_calls * 1000
    print(f"  {n_calls} lseek+reads: mean={times_lseek_read.mean():.2f}ms ({per_lseek_read_us:.1f}μs/call)")

    # [3c] Python overhead (loop + function call, no syscall)
    print("\n  [3c] Pure Python loop overhead (no syscall)...")
    dummy = bytearray(64)
    times_python = []
    for _ in range(100):
        offsets = np.random.randint(0, 100000000, size=n_calls)
        t0 = time.perf_counter()
        for off in offsets:
            x = int(off)  # simulate offset computation
        times_python.append(time.perf_counter() - t0)
    times_python = np.array(times_python) * 1000
    per_python_us = times_python.mean() / n_calls * 1000
    print(f"  {n_calls} iterations: mean={times_python.mean():.2f}ms ({per_python_us:.1f}μs/iter)")

    # [3d] SSD read minus syscall overhead
    ssd_only = times_buf.mean() - times_lseek_read.mean()
    print(f"\n  [3d] SSD-only time (buffered read - syscall overhead):")
    print(f"  Total buffered read:  {times_buf.mean():.1f}ms")
    print(f"  Syscall overhead:     {times_lseek_read.mean():.1f}ms")
    print(f"  SSD-only:             {ssd_only:.1f}ms")
    print(f"  Syscall fraction:     {times_lseek_read.mean()/times_buf.mean()*100:.1f}% of total")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY — Systems-Level SSD Analysis")
    print(f"{'='*70}")

    print(f"\n  --- Raw SSD Performance (fio) ---")
    print(f"  {'Access Pattern':<30} {'IOPS':>10} {'Latency':>12} {'1660 reads':>12}")
    print(f"  {'-'*66}")
    print(f"  {'64B random, QD=1':<30} {iops_64_qd1:>10.0f} {lat_64_qd1:>10.0f}μs {1660/iops_64_qd1*1000:>10.1f}ms")
    print(f"  {'64B random, QD=32':<30} {iops_64_qd32:>10.0f} {lat_64_qd32:>10.0f}μs {1660/iops_64_qd32*1000:>10.1f}ms")
    print(f"  {'4K random, QD=1':<30} {iops_4k_qd1:>10.0f} {lat_4k_qd1:>10.0f}μs {882/iops_4k_qd1*1000:>10.1f}ms")
    print(f"  {'4K random, QD=32':<30} {iops_4k_qd32:>10.0f} {lat_4k_qd32:>10.0f}μs {882/iops_4k_qd32*1000:>10.1f}ms")
    print(f"  {'Sequential 128K, QD=32':<30} {'':>10} {bw_seq:>10.0f} MB/s")

    print(f"\n  --- Read Strategy Comparison (200 batches, cache dropped) ---")
    print(f"  {'Strategy':<30} {'Mean':>8} {'p50':>8} {'p99':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Buffered (lseek+read)':<30} {times_buf.mean():>7.1f}ms {np.percentile(times_buf,50):>7.1f}ms {np.percentile(times_buf,99):>7.1f}ms")
    if times_direct.mean() > 0:
        print(f"  {'O_DIRECT (4K pages)':<30} {times_direct.mean():>7.1f}ms {np.percentile(times_direct,50):>7.1f}ms {np.percentile(times_direct,99):>7.1f}ms")
    print(f"  {'mmap':<30} {times_mmap.mean():>7.1f}ms {np.percentile(times_mmap,50):>7.1f}ms {np.percentile(times_mmap,99):>7.1f}ms")
    print(f"  {'pread (no lseek)':<30} {times_pread.mean():>7.1f}ms {np.percentile(times_pread,50):>7.1f}ms {np.percentile(times_pread,99):>7.1f}ms")

    print(f"\n  --- Overhead Breakdown ---")
    print(f"  Python loop (no syscall):     {times_python.mean():.2f}ms ({per_python_us:.1f}μs × 1660)")
    print(f"  Syscall overhead (lseek+read): {times_lseek_read.mean():.2f}ms ({per_lseek_read_us:.1f}μs × 1660)")
    print(f"  SSD I/O time:                 {ssd_only:.1f}ms")
    print(f"  Total measured:               {times_buf.mean():.1f}ms")
    print(f"  Breakdown: {times_python.mean()/times_buf.mean()*100:.0f}% Python + "
          f"{(times_lseek_read.mean()-times_python.mean())/times_buf.mean()*100:.0f}% syscall + "
          f"{ssd_only/times_buf.mean()*100:.0f}% SSD")

    # Save
    results = {
        'system': {
            'disk': 'Micron MTFDDAK480TDS (SATA SSD)',
            'io_scheduler': sched,
            'queue_depth': int(nr_req),
        },
        'fio': fio_results,
        'read_strategies': {
            'buffered': {'mean_ms': float(times_buf.mean()), 'p50_ms': float(np.percentile(times_buf,50)), 'p99_ms': float(np.percentile(times_buf,99))},
            'o_direct': {'mean_ms': float(times_direct.mean()), 'p50_ms': float(np.percentile(times_direct,50))},
            'mmap': {'mean_ms': float(times_mmap.mean()), 'p50_ms': float(np.percentile(times_mmap,50)), 'p99_ms': float(np.percentile(times_mmap,99))},
            'pread': {'mean_ms': float(times_pread.mean()), 'p50_ms': float(np.percentile(times_pread,50)), 'p99_ms': float(np.percentile(times_pread,99))},
        },
        'overhead': {
            'python_loop_ms': float(times_python.mean()),
            'per_python_us': float(per_python_us),
            'syscall_ms': float(times_lseek_read.mean()),
            'per_syscall_us': float(per_lseek_read_us),
            'ssd_only_ms': float(ssd_only),
        },
    }
    with open(f'{OUT_DIR}/systems_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {OUT_DIR}/systems_analysis.json")

    os.unlink(SSD_FILE)

if __name__ == '__main__':
    main()
