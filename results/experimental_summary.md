# Experimental Summary: DLRM Embedding Compression

## Setup
- **Model**: DLRM with 26 embedding tables (8 large, >50K rows)
- **Dataset**: Kaggle/Criteo (2GB embeddings, 3.3M test samples)
- **Hardware**: CPU-only (40 cores), 256GB RAM
- **Hot/Cold Split**: 4.3% hot (in fp32), 95.7% cold (compressed)
- **Frame size**: 129,600 rows per frame (1920x1080, 4x4 tiles)

## 1. Main Result: 1.77x Inference Speedup

The codec + C++ `fast_forward` approach achieves a **1.77x speedup** over the fp32 baseline,
while reducing disk storage by 37x.

| Config | AUC | Batch Latency | Speedup | Storage |
|--------|-----|---------------|---------|---------|
| Baseline (fp32) | 0.802497 | 4.26ms | 1.0x | 2,061MB |
| **Codec full_cpp** | **0.802496** | **2.41ms** | **1.77x** | **53MB** |

### Why is it faster?

| Component | Baseline (ms) | Codec (ms) | Speedup |
|-----------|---------------|------------|---------|
| Embedding lookup | 1.48 | 0.22 | **6.7x** |
| Interact | 1.15 | 0.57 | 2.0x |
| MLP | 1.57 | 1.58 | 1.0x |

The speedup comes from **memory footprint reduction**:
1. **Cold weights as uint8** — 4x smaller per row → dramatically better CPU cache utilization
2. **Compact hot weights** — only 4.3% of rows stored as fp32 → fits in L2/L3 cache
3. **All-C++ fast_forward** — single C++ call for all 26 tables, no Python loop overhead
4. **Bitmap-rank O(1) dispatch** — hot/cold classification with single popcount instruction
5. **AVX512 dequantization** — `accum_q8_d16` dequantizes uint8→fp32 and accumulates in one pass

Pre-decoded cold frames for the test set: 22 frames = ~45MB uint8 (vs 2,061MB fp32 original).
The working set is so small that it fits entirely in CPU cache, explaining the 6.7x embedding
lookup speedup.

## 2. Compression Ratio Comparison (Lossless)

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
H.265 lossless decode is slower (40ms vs 1.1ms for Zstd-19), but this is amortized by caching.

## 3. Lossy H.265 CRF Sweep

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

## 4. Cache Size Sensitivity (H.265 + fast_forward)

All cache variants achieve the same ~2.4ms batch latency (fast_forward speedup).
Total wall-clock time varies based on decode overhead for cache misses.

| Cache Size | BLat (ms) | Hit Rate | Total Time (s) | Frames Decoded |
|-----------|-----------|----------|-----------------|----------------|
| pre-decode | 2.41 | 100% | 31.0 | 22 |
| 1 | 2.48 | 46.3% | 80.1 | 7,422 |
| 2 | 2.56 | 92.5% | 38.2 | 1,038 |
| **4** | **2.43** | **99.8%** | **31.5** | **22** |
| 8 | 2.43 | 99.8% | 31.4 | 22 |
| 16 | 2.47 | 99.8% | 31.5 | 22 |
| 64 | 2.49 | 99.8% | 31.7 | 22 |

**Finding**: Cache=4 is the sweet spot. Hit rate saturates at 99.8% from cache=4.
Only 22 unique frames needed across all 1,599 batches.
With cache >= 4, total time matches pre-decode approach.

## 5. mmap Baseline

| Config | AUC | Batch Latency | RSS | Disk |
|--------|-----|---------------|-----|------|
| Baseline (fp32) | 0.802497 | 6.40ms | 19,476MB | 2,061MB |
| uint8 in-memory | 0.802481 | 6.62ms | 18,052MB | 539MB |
| mmap (uint8) | 0.802481 | 5.72ms | 17,769MB | 539MB |

**Finding**: mmap uint8 is 11% faster than fp32 baseline due to smaller data footprint.
However, mmap still uses 539MB disk. The codec approach achieves 53MB disk AND 1.77x
faster inference by using the optimized C++ path.

## 6. Reordering Analysis

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

## 7. Per-Table Compression Analysis

| Table | Cold Rows | Raw MB | Entropy (bits/byte) | Zstd-19 | H.265 |
|-------|-----------|--------|---------------------|---------|-------|
| 2     | 9.6M      | 154.1  | 0.29                | 23.9x   | 27.9x |
| 11    | 8.0M      | 127.4  | 0.69                | 11.9x   | 12.0x |
| 15    | 5.3M      | 84.3   | 0.79                | 10.6x   | 10.5x |
| 3     | 2.2M      | 34.7   | 1.33                | 5.6x    | 6.0x  |
| 20    | 6.7M      | 107.9  | 1.39                | 5.2x    | 5.6x  |
| 23    | 0.3M      | 4.5    | 2.00                | 3.7x    | 3.9x  |
| 25    | 0.1M      | 2.2    | 2.69                | 2.9x    | 2.8x  |
| 9     | 0.1M      | 1.4    | 3.91                | 2.0x    | 1.9x  |

**Correlation**: Entropy vs Zstd-19 ratio: r = -0.763.

## 8. C++ Pipeline Optimization (On-Demand Path)

For streaming/online scenarios where pre-decoding is not possible:

| Method | Mean (ms) | Speedup |
|--------|-----------|---------|
| Python scan + Python gather + writeback | 2.017 | 1.0x |
| C++ scan + Python gather + writeback | 1.593 | 1.27x |
| C++ scan + C++ gather + Python writeback | 0.693 | 2.91x |
| **C++ scan + C++ scatter (fused)** | **0.740** | **2.73x** |

## Key Takeaways

### Primary result: 1.77x inference speedup + 37x storage reduction
- Hot/cold split + uint8 quantization + C++ fast_forward = 2.41ms vs 4.26ms baseline
- Disk: 53MB (H.265) or 55MB (Zstd-19) vs 2,061MB fp32
- AUC loss: <0.01% (from uint8 quantization)
- The speedup comes from **memory footprint reduction** enabling better CPU cache utilization

### What video codecs uniquely offer:
1. **Lossy compression**: CRF=18 → 225x compression with 0.01% AUC loss
   - No general-purpose compressor can do this
   - Trades minimal accuracy for extreme compression
2. **Spatial exploitation**: 5% better lossless ratio than Zstd-19 (9.9x vs 9.4x)

### Compressor-agnostic framework:
- The speedup comes from the C++ fast_forward path with uint8 cold storage
- This architecture works with ANY compressor (H.265, Zstd, LZ4, etc.)
- Choice of compressor only affects disk size and decode latency on cache misses
- With cache >= 4 (99.8% hit rate), compressor choice is largely irrelevant for latency
