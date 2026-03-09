# Experimental Summary: DLRM Embedding Compression

## Setup
- **Model**: DLRM with 26 embedding tables (8 large, >50K rows)
- **Dataset**: Kaggle/Criteo (2GB embeddings, 3.3M test samples)
- **Hardware**: CPU-only (40 cores), 256GB RAM
- **Hot/Cold Split**: 4.3% hot (in fp32), 95.7% cold (compressed)
- **Frame size**: 129,600 rows per frame (1920x1080, 4x4 tiles)

## 1. Compression Ratio Comparison (Lossless)

All compressors tested on same quantized uint8 cold data, 8 tables combined.

| Compressor | Ratio (vs uint8) | Ratio (vs fp32) | Size (MB) | Decode (ms/frame) |
|-----------|------------------|-----------------|-----------|-------------------|
| LZ4       | 3.3x             | 13.3x           | 155       | 1.42              |
| Snappy    | 3.9x             | 15.6x           | 133       | 2.32              |
| Zstd-3    | 6.8x             | 27.0x           | 77        | 2.37              |
| Zstd-9    | 7.6x             | 30.3x           | 68        | 1.79              |
| **Zstd-19** | **9.4x**       | **37.6x**       | **55**    | **1.13**          |
| H.265     | 9.9x             | 39.1x           | 53        | 40.1 (lossless)   |

**Finding**: Zstd-19 achieves 95% of H.265 compression with 35x faster decode.
H.265 lossless decode is extremely slow (40ms vs 1.1ms for Zstd-19).

## 2. Lossy H.265 CRF Sweep

Video codecs uniquely support lossy compression via CRF parameter.

| CRF | Compression | AUC Delta | Max Error | Decode (ms) |
|-----|-------------|-----------|-----------|-------------|
| 0   | 9.9x        | -0.000001 | 0/255     | 40.1        |
| 10  | 20.9x       | -0.000034 | 7/255     | 28.8        |
| 18  | 225.5x      | -0.000100 | 19/255    | 11.9        |
| 23  | 434.1x      | -0.000190 | 28/255    | 10.5        |
| 28  | 596.1x      | -0.000316 | 56/255    | 9.7         |

**Finding**: CRF=18 achieves 225x compression with only 0.01% AUC loss.
This is a unique capability of video codecs that general-purpose compressors cannot match.

## 3. End-to-End Inference (Measured)

| Config | AUC | Batch Latency | Overhead | Storage |
|--------|-----|---------------|----------|---------|
| Baseline (fp32) | 0.802497 | 4.77ms | --- | 2,061MB |
| mmap (uint8) | 0.802481 | 5.72ms | +20% | 539MB |
| Zstd-19 cache=16 | 0.802496 | 6.13ms | +29% | 55MB |
| Zstd-3 cache=16 | 0.802496 | 6.50ms | +36% | 77MB |
| H.265 CRF=0 cache=16 | 0.802496 | 6.19ms | +30% | 53MB |

**Cache statistics** (cache=16 frames):
- Hit rate: 99.84% (22 misses in 13,812 accesses)
- Actual decode: 0.010ms/batch (negligible)
- Scan overhead: 2.2ms/batch (Python-level cold index scanning)

## 4. mmap Baseline

| Config | AUC | Batch Latency | RSS | Disk |
|--------|-----|---------------|-----|------|
| Baseline (fp32) | 0.802497 | 6.40ms | 19,476MB | 2,061MB |
| uint8 in-memory | 0.802481 | 6.62ms | 18,052MB | 539MB |
| mmap (uint8) | 0.802481 | 5.72ms | 17,769MB | 539MB |

**Finding**: mmap uint8 is 11% faster than fp32 baseline due to smaller data footprint.
Simple quantization already provides 4x compression with negligible AUC loss.

## 5. Reordering Analysis

Frequency-based reordering of cold rows: average improvement across 8 tables.

| Compressor | Random | Natural | Reordered | Improvement |
|-----------|--------|---------|-----------|-------------|
| LZ4       | 2.9x   | 2.9x    | 2.9x      | 1.01x       |
| Zstd-3    | 5.7x   | 5.9x    | 5.8x      | 1.02x       |
| Zstd-19   | 8.0x   | 8.3x    | 8.2x      | 1.02x       |
| H.265     | 8.7x   | 8.8x    | 8.8x      | 1.01x       |

**Finding**: Reordering provides only 1-2% compression improvement.
Its real value is **cache locality** — co-locating co-accessed rows in the same
frame to maximize LRU cache hit rates.

## Key Takeaways

### What H.265 DOES offer:
1. **Lossy compression**: CRF=18 gives 225x compression with 0.01% AUC loss
   - No general-purpose compressor can do this
   - Useful for extreme memory-constrained scenarios
2. **Slightly better lossless ratio**: 9.9x vs 9.4x for Zstd-19 (marginal)

### What H.265 does NOT offer:
1. **Speed**: 35x slower decode than Zstd-19 for similar compression
2. **Practical advantage at cache=16**: Both achieve 99.84% cache hit rate,
   so the decode speed difference is mostly irrelevant

## 6. Cache Size Sweep (Zstd-19)

| Cache Size | Batch Latency | p99 Latency | Hit Rate | Misses |
|-----------|---------------|-------------|----------|--------|
| 1         | 12.79ms       | 20.09ms     | 85.86%   | 1,953  |
| 2         | 7.49ms        | 20.18ms     | 98.08%   | 265    |
| **4**     | **6.84ms**    | **10.48ms** | **99.84%** | **22** |
| 8         | 6.68ms        | 10.34ms     | 99.84%   | 22     |
| 16        | 6.69ms        | 10.45ms     | 99.84%   | 22     |
| 32        | 6.48ms        | 10.20ms     | 99.84%   | 22     |
| 64        | 6.55ms        | 9.00ms      | 99.84%   | 22     |

Baseline: 4.16ms mean, 5.58ms p99

**Finding**: Cache=4 is the sweet spot. Hit rate saturates at 99.84% from cache=4.
Only 22 frame misses across all 1,599 batches (13,812 frame accesses).
Cache memory cost: 4 frames * 129,600 rows * 16 bytes * 4 (fp32) = 33MB.

### Recommended approach for production:
1. **Hot/cold split** with 4.3% hot threshold
2. **Zstd-19** for lossless cold compression (9.4x ratio, 1.13ms decode)
3. **LRU frame cache** size=16 (99.84% hit rate)
4. **Frequency-based reordering** for cache locality (not compression)
5. Total: **37x memory reduction**, **29% latency overhead**, **<0.01% AUC loss**

### When to use H.265 instead:
- Need >10x compression (use CRF=18 for 225x)
- Can tolerate 0.01% AUC loss
- Memory is more constrained than compute
