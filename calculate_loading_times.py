#!/usr/bin/env python3
"""
Calculate model loading times for Kaggle (measured) and industrial scale (projected).
Uses measured decode throughput + SSD/network bandwidth specs.
"""
import json

# ================================================================
# Measured from Kaggle experiment
# ================================================================
KAGGLE_FP32_MB = 2058          # fp32 embedding size
KAGGLE_UINT8_MB = 492          # uint8 size (decoded)
KAGGLE_COMP_KB = 312           # CRF=30 single-frame compressed
KAGGLE_COMP_MB = KAGGLE_COMP_KB / 1024

# Measured times (ms)
KAGGLE_READ_UNCOMP_MS = 782    # read 2GB raw from SSD (cached)
KAGGLE_READ_COMP_MS = 0.1      # read 312KB from SSD
KAGGLE_DECODE_POOL_MS = 494    # decode 8 single-frames parallel

# Derived rates
SSD_READ_GBPS = KAGGLE_FP32_MB / 1024 / (KAGGLE_READ_UNCOMP_MS / 1000)  # ~2.6 GB/s (cached)
DECODE_GBPS = KAGGLE_UINT8_MB / 1024 / (KAGGLE_DECODE_POOL_MS / 1000)    # ~1.0 GB/s (single-frame)

# For 1080p multi-frame layout: 5-8 GB/s from earlier benchmark
DECODE_1080P_GBPS = 5.6  # measured for 20 small 1080p frames

# Network specs
IB_HDR_GBPS = 25       # InfiniBand HDR
IB_NDR_GBPS = 50       # InfiniBand NDR
ETH_100G_GBPS = 12.5   # 100GbE
ETH_10G_GBPS = 1.25    # 10GbE

# SSD specs
SSD_PCIE4_GBPS = 5.0   # NVMe PCIe 4.0 (cold read)
SSD_PCIE5_GBPS = 12.0  # NVMe PCIe 5.0

# Compression ratios
CRF30_RATIO = 6466
CRF35_RATIO = 9502
CRF51_RATIO = 12437
ZSTD_RATIO = 27

print("=" * 90)
print("MODEL LOADING TIME ANALYSIS: KAGGLE (MEASURED) + INDUSTRIAL SCALE (PROJECTED)")
print("=" * 90)

print(f"\nMeasured rates from Kaggle ({KAGGLE_FP32_MB}MB fp32):")
print(f"  SSD read (cached):            {SSD_READ_GBPS:.1f} GB/s")
print(f"  H.265 decode (single-frame):  {DECODE_GBPS:.2f} GB/s (decoded output rate)")
print(f"  H.265 decode (1080p frames):  {DECODE_1080P_GBPS:.1f} GB/s (decoded output rate)")
print(f"  Compression ratio (CRF=30):   {CRF30_RATIO}x vs fp32")

# ================================================================
# Scenario definitions
# ================================================================
model_sizes_gb = {
    'Kaggle':           2,
    'Terabyte':         5.5,
    'Medium prod':      50,
    'Large prod':       100,
    'Meta-scale':       1000,
}

# For each model size, compute times for different loading scenarios
print(f"\n{'='*90}")
print("SCENARIO 1: LOCAL LOADING (SSD → CPU DRAM)")
print(f"{'='*90}")
print(f"\n{'Model':<16} {'Size':>6} {'Compressed':>12} | {'Read uncomp':>12} {'Read+Decode':>14} {'Speedup':>8}")
print(f"{'':<16} {'(GB)':>6} {'(CRF=30)':>12} | {'(SSD→DRAM)':>12} {'(compressed)':>14} {'':>8}")
print("-" * 90)

ssd_speed = SSD_PCIE4_GBPS
results = {}

for name, size_gb in model_sizes_gb.items():
    comp_mb = size_gb * 1024 / CRF30_RATIO
    uint8_gb = size_gb / 4  # fp32→uint8

    # Uncompressed: read fp32 from SSD
    t_uncomp_s = size_gb / ssd_speed

    # Compressed: read tiny file + decode uint8
    t_read_comp_s = comp_mb / 1024 / ssd_speed  # negligible
    t_decode_s = uint8_gb / DECODE_GBPS          # single-frame decode
    t_comp_s = t_read_comp_s + t_decode_s

    speedup = t_uncomp_s / t_comp_s

    if name == 'Kaggle':
        # Use measured values
        t_uncomp_s = KAGGLE_READ_UNCOMP_MS / 1000
        t_comp_s = (KAGGLE_READ_COMP_MS + KAGGLE_DECODE_POOL_MS) / 1000
        speedup = t_uncomp_s / t_comp_s

    comp_str = f"{comp_mb:.1f} MB" if comp_mb >= 1 else f"{comp_mb*1024:.0f} KB"

    print(f"{name:<16} {size_gb:>5.1f}  {comp_str:>12} | {t_uncomp_s:>11.2f}s {t_comp_s:>13.2f}s {speedup:>7.1f}x")

    results[name] = {
        'size_gb': size_gb, 'compressed_mb': comp_mb,
        'load_uncomp_s': t_uncomp_s, 'load_comp_s': t_comp_s,
        'speedup': speedup,
    }

print(f"\n  Note: SSD = NVMe PCIe 4.0 ({ssd_speed} GB/s)")
print(f"  Note: Decode = H.265 pool parallel ({DECODE_GBPS:.2f} GB/s decoded output)")

# With 1080p multi-frame (faster decode, larger files)
print(f"\n  Alternative: 1080p multi-frame layout (decode at {DECODE_1080P_GBPS} GB/s, 1260x compression)")
for name, size_gb in model_sizes_gb.items():
    comp_mb_1080p = size_gb * 1024 / 1260  # 1080p ratio
    uint8_gb = size_gb / 4
    t_uncomp_s = size_gb / ssd_speed
    t_read_s = comp_mb_1080p / 1024 / ssd_speed
    t_decode_s = uint8_gb / DECODE_1080P_GBPS
    t_comp_s = t_read_s + t_decode_s
    speedup = t_uncomp_s / t_comp_s
    comp_str = f"{comp_mb_1080p:.1f} MB"
    print(f"  {name:<14} {comp_str:>10} | uncomp {t_uncomp_s:.2f}s | comp {t_comp_s:.2f}s | {speedup:.1f}x")


# ================================================================
# SCENARIO 2: NETWORK TRANSFER
# ================================================================
print(f"\n{'='*90}")
print("SCENARIO 2: NETWORK TRANSFER (Remote → CPU DRAM)")
print(f"{'='*90}")

networks = {
    '10GbE':    ETH_10G_GBPS,
    '100GbE':   ETH_100G_GBPS,
    'IB HDR':   IB_HDR_GBPS,
    'IB NDR':   IB_NDR_GBPS,
}

for net_name, net_gbps in networks.items():
    print(f"\n--- {net_name} ({net_gbps} GB/s) ---")
    print(f"{'Model':<16} {'Uncomp xfer':>12} {'Comp xfer':>12} {'+ Decode':>10} {'Total comp':>12} {'Speedup':>8} {'Traffic':>12}")
    print("-" * 90)

    for name, size_gb in model_sizes_gb.items():
        comp_mb = size_gb * 1024 / CRF30_RATIO
        uint8_gb = size_gb / 4

        # Uncompressed transfer
        t_xfer_uncomp = size_gb / net_gbps

        # Compressed: transfer + decode
        t_xfer_comp = comp_mb / 1024 / net_gbps
        t_decode = uint8_gb / DECODE_GBPS
        t_total_comp = t_xfer_comp + t_decode

        speedup = t_xfer_uncomp / t_total_comp
        traffic_reduction = size_gb * 1024 / comp_mb

        comp_str = f"{comp_mb:.1f}MB" if comp_mb >= 1 else f"{comp_mb*1024:.0f}KB"

        # Format times
        def fmt_time(s):
            if s < 0.001: return f"{s*1e6:.0f}µs"
            if s < 1: return f"{s*1000:.0f}ms"
            if s < 60: return f"{s:.1f}s"
            return f"{s/60:.1f}min"

        print(f"{name:<16} {fmt_time(t_xfer_uncomp):>12} {fmt_time(t_xfer_comp):>12} "
              f"{fmt_time(t_decode):>10} {fmt_time(t_total_comp):>12} {speedup:>7.1f}x "
              f"{comp_str:>10}→{fmt_time(t_xfer_comp):>6}")


# ================================================================
# SCENARIO 3: FLEET DEPLOYMENT (1000 servers)
# ================================================================
print(f"\n{'='*90}")
print("SCENARIO 3: FLEET MODEL UPDATE (push to N servers)")
print(f"{'='*90}")

N_SERVERS = [10, 100, 1000]
for name, size_gb in [('Kaggle', 2), ('Large prod', 100), ('Meta-scale', 1000)]:
    comp_mb = size_gb * 1024 / CRF30_RATIO
    print(f"\n  {name} ({size_gb}GB fp32, {comp_mb:.1f}MB compressed):")
    for n in N_SERVERS:
        total_uncomp_tb = size_gb * n / 1024
        total_comp_gb = comp_mb * n / 1024
        print(f"    {n:>5} servers: {total_uncomp_tb:.1f}TB uncomp → {total_comp_gb:.2f}GB comp "
              f"({size_gb*1024/comp_mb:.0f}x less traffic)")


# ================================================================
# SCENARIO 4: CHECKPOINT DURING TRAINING
# ================================================================
print(f"\n{'='*90}")
print("SCENARIO 4: CHECKPOINT SAVE/LOAD DURING TRAINING")
print(f"{'='*90}")
print(f"\n{'Model':<16} {'FP32 write':>12} {'Comp write':>12} {'Comp read':>12} {'+ Decode':>10} {'Ckpt/min':>10}")
print("-" * 80)

for name, size_gb in model_sizes_gb.items():
    comp_mb = size_gb * 1024 / CRF30_RATIO
    uint8_gb = size_gb / 4

    # Write: encode time is ~1s per GB of input (from Kaggle: 14s for 492MB input → ~0.035 GB/s encode)
    # Actually encode is offline one-time, checkpoint write is just the compressed bytes
    t_write_uncomp = size_gb / ssd_speed
    t_write_comp = comp_mb / 1024 / ssd_speed
    t_read_comp = t_write_comp
    t_decode = uint8_gb / DECODE_GBPS

    # How many checkpoints per minute possible?
    ckpt_per_min_uncomp = 60 / t_write_uncomp if t_write_uncomp > 0 else float('inf')
    ckpt_per_min_comp = 60 / t_write_comp if t_write_comp > 0 else float('inf')

    comp_str = f"{comp_mb:.1f}MB" if comp_mb >= 1 else f"{comp_mb*1024:.0f}KB"

    print(f"{name:<16} {fmt_time(t_write_uncomp):>12} {fmt_time(t_write_comp):>12} "
          f"{fmt_time(t_read_comp):>12} {fmt_time(t_decode):>10} "
          f"{ckpt_per_min_comp:>9.0f}")

print(f"\n  Note: 'Comp write' = write pre-compressed embeddings to disk")
print(f"  Note: Encoding is a separate offline/background step")


# ================================================================
# SUMMARY TABLE
# ================================================================
print(f"\n{'='*90}")
print("SUMMARY: KEY NUMBERS FOR THE PAPER")
print(f"{'='*90}")

print(f"\n{'Metric':<40} {'Kaggle (2GB)':>15} {'100GB prod':>15} {'1TB scale':>15}")
print("-" * 90)

for label, size_gb in [('Kaggle (2GB)', 2), ('100GB prod', 100), ('1TB scale', 1000)]:
    comp_mb = size_gb * 1024 / CRF30_RATIO
    uint8_gb = size_gb / 4
    t_load_uncomp = size_gb / ssd_speed
    t_load_comp = comp_mb / 1024 / ssd_speed + uint8_gb / DECODE_GBPS
    t_net_uncomp_10g = size_gb / ETH_10G_GBPS
    t_net_comp_10g = comp_mb / 1024 / ETH_10G_GBPS

    col = f"{comp_mb:.1f}MB" if comp_mb >= 1 else f"{comp_mb*1024:.0f}KB"
    if label == 'Kaggle (2GB)':
        col = f"312KB"

    # Print as a column
    pass

print(f"{'Embedding size (fp32)':<40} {'2 GB':>15} {'100 GB':>15} {'1 TB':>15}")
print(f"{'Compressed size (CRF=30)':<40} {'312 KB':>15} {'15.8 MB':>15} {'158 MB':>15}")
print(f"{'Compression ratio':<40} {'6,466x':>15} {'6,466x':>15} {'6,466x':>15}")

t1 = 2/ssd_speed; t2 = 100/ssd_speed; t3 = 1000/ssd_speed
print(f"{'SSD load (uncomp, 5 GB/s)':<40} {fmt_time(t1):>15} {fmt_time(t2):>15} {fmt_time(t3):>15}")

d1 = 0.5/DECODE_GBPS; d2 = 25/DECODE_GBPS; d3 = 250/DECODE_GBPS
print(f"{'SSD load (comp, read+decode)':<40} {fmt_time(d1):>15} {fmt_time(d2):>15} {fmt_time(d3):>15}")

s1 = t1/d1; s2 = t2/d2; s3 = t3/d3
print(f"{'Local loading speedup':<40} {s1:>14.1f}x {s2:>14.1f}x {s3:>14.1f}x")

n1 = 2/ETH_10G_GBPS; n2 = 100/ETH_10G_GBPS; n3 = 1000/ETH_10G_GBPS
c1 = 312/1024/1024/ETH_10G_GBPS; c2 = 15.8/1024/ETH_10G_GBPS; c3 = 158/1024/ETH_10G_GBPS
print(f"{'10GbE transfer (uncomp)':<40} {fmt_time(n1):>15} {fmt_time(n2):>15} {fmt_time(n3):>15}")
print(f"{'10GbE transfer (comp, xfer only)':<40} {fmt_time(c1):>15} {fmt_time(c2):>15} {fmt_time(c3):>15}")
print(f"{'10GbE speedup (xfer only)':<40} {n1/c1:>14.0f}x {n2/c2:>14.0f}x {n3/c3:>14.0f}x")

fleet = 1000
print(f"{'Fleet update traffic ({fleet} servers)':<40} {2*fleet/1024:.0f} TB{' ':>9} {100*fleet/1024:.0f} TB{' ':>8} {1000*fleet/1024:.0f} TB")
cf1 = 312*fleet/1024/1024; cf2 = 15.8*fleet/1024; cf3 = 158*fleet/1024
print(f"{'Fleet compressed traffic':<40} {cf1:.1f} MB{' ':>9} {cf2:.1f} GB{' ':>8} {cf3:.0f} GB")


# Save
with open('results/crf_pareto/loading_analysis.json', 'w') as f:
    json.dump({
        'measured_rates': {
            'ssd_cached_gbps': SSD_READ_GBPS,
            'decode_single_frame_gbps': DECODE_GBPS,
            'decode_1080p_gbps': DECODE_1080P_GBPS,
        },
        'kaggle_measured': {
            'read_uncomp_ms': KAGGLE_READ_UNCOMP_MS,
            'read_comp_ms': KAGGLE_READ_COMP_MS,
            'decode_pool_ms': KAGGLE_DECODE_POOL_MS,
            'total_comp_ms': KAGGLE_READ_COMP_MS + KAGGLE_DECODE_POOL_MS,
        },
    }, f, indent=2)

print(f"\nSaved to results/crf_pareto/loading_analysis.json")
