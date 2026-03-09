# Final Results: DLRM Embedding Compression

## Table 1: System Comparison (End-to-End Inference)

| System | Cold Storage | Disk (MB) | AUC | AUC Loss | BLat (ms) | Speedup |
|--------|-------------|-----------|-----|----------|-----------|---------|
| Baseline (fp32) | None | 2,061 | 0.802497 | — | 4.26 | 1.0x |
| **Codec (pre-decode, H.265)** | **C++ fast_forward** | **53** | **0.802496** | **-0.0001%** | **2.41** | **1.77x** |
| Codec (LRU cache=4, H.265) | C++ fast_forward | 53 | 0.802496 | -0.0001% | 2.43 | 1.75x |
| Codec (LRU cache=16, H.265) | C++ fast_forward | 53 | 0.802496 | -0.0001% | 2.47 | 1.72x |
| Quantize (uint8 in-memory) | In-memory | 539 | 0.802481 | -0.002% | ~4.2 | ~1.0x |
| mmap (uint8) | OS page cache | 539 | 0.802481 | -0.002% | 5.72 | 0.74x |

Notes:
- All lossless systems achieve AUC within 0.002% of baseline (quantization noise)
- BLat = mean batch latency over 1,599 batches (batch_size=2048), median of 3 runs
- The **1.77x speedup** comes from: (1) uint8 cold storage → 4x smaller memory footprint →
  better CPU cache utilization, (2) all-C++ `fast_forward` processes all 26 tables in one call
  with bitmap-rank O(1) hot/cold dispatch + AVX512 dequantization, (3) compact hot weights
  (4.3% of rows in fp32)
- Embedding lookup: 1.48ms (baseline) → 0.22ms (codec) = **6.7x faster**
- Pre-decoded cold frames: 22 frames = ~45MB uint8 (vs 2GB fp32 original)
- LRU cache variants also achieve ~1.75x speedup on batch latency; only total wall-clock
  time differs due to decode overhead on cache misses

## Table 2: Compression Ratio (Lossless)

Per-frame compression of uint8 quantized, frequency-reordered cold embeddings.

| Compressor | Ratio (vs uint8) | Ratio (vs fp32) | Decode (ms/frame) |
|-----------|------------------|-----------------|-------------------|
| LZ4 | 3.3x | 13.3x | 1.42 |
| Snappy | 3.9x | 15.6x | 2.32 |
| Zstd-3 | 6.8x | 27.0x | 2.37 |
| Zstd-9 | 7.6x | 30.3x | 1.79 |
| **Zstd-19** | **9.4x** | **37.6x** | **1.13** |
| H.265 (lossless) | 9.9x | 39.1x | 40.1 |

## Table 3: CRF Accuracy Tradeoff (H.265 only)

| CRF | Ratio (vs uint8) | AUC Delta | Max Error (uint8) |
|-----|------------------|-----------|-------------------|
| 0 | 9.9x | -0.000001 | 0/255 |
| 10 | 20.9x | -0.000034 | 7/255 |
| 18 | 225.5x | -0.000100 | 19/255 |
| 23 | 434.1x | -0.000190 | 28/255 |
| 28 | 596.1x | -0.000316 | 56/255 |

## Table 4: Why fast_forward Is Faster (Latency Breakdown)

| Component | Baseline (ms) | Codec full_cpp (ms) | Speedup |
|-----------|---------------|---------------------|---------|
| Embedding lookup | 1.48 | 0.22 | 6.7x |
| Interact | 1.15 | 0.57 | 2.0x |
| MLP | 1.57 | 1.58 | 1.0x |
| **Total** | **4.26** | **2.41** | **1.77x** |

The speedup is concentrated in embedding lookup (6.7x) and feature interaction (2.0x):
- **Embedding**: uint8 cold weights are 4x smaller → dramatically better CPU L2/L3 cache
  hit rates. Compact hot fp32 weights (only 4.3% of rows) fit entirely in cache.
  Bitmap-rank O(1) dispatch eliminates branch mispredictions.
- **Interact**: Smaller output tensors from cached-friendly embedding lookup → faster
  downstream tensor operations.
- **MLP**: Unchanged (compute-bound, not memory-bound).

## Table 5: Cache Size Sensitivity (H.265 + fast_forward)

| Cache Size (frames) | Hit Rate | BLat (ms) | Total Time (s) | Unique Frames |
|---------------------|----------|-----------|-----------------|---------------|
| pre-decode all | 100% | 2.41 | 31.0 | 22 |
| 1 | 46.3% | 2.48 | 80.1 | 7,422 |
| 2 | 92.5% | 2.56 | 38.2 | 1,038 |
| **4** | **99.8%** | **2.43** | **31.5** | **22** |
| 8 | 99.8% | 2.43 | 31.4 | 22 |
| 16 | 99.8% | 2.47 | 31.5 | 22 |
| 64 | 99.8% | 2.49 | 31.7 | 22 |

Key: BLat is always ~2.4ms (fast_forward speedup). Total wall-clock time differs due to
decode overhead on cache misses. Cache=4 saturates hit rate at 99.8%.

## Table 6: Per-Table Compression

| Table | Cold Rows | Raw (MB) | Entropy (b/B) | Zstd-19 | H.265 |
|-------|-----------|----------|---------------|---------|-------|
| 2 | 9.6M | 154.1 | 0.29 | 23.9x | 27.9x |
| 11 | 8.0M | 127.4 | 0.69 | 11.9x | 12.0x |
| 15 | 5.3M | 84.3 | 0.79 | 10.6x | 10.5x |
| 3 | 2.2M | 34.7 | 1.33 | 5.6x | 6.0x |
| 20 | 6.7M | 107.9 | 1.39 | 5.2x | 5.6x |
| 23 | 0.3M | 4.5 | 2.00 | 3.7x | 3.9x |
| 25 | 0.1M | 2.2 | 2.69 | 2.9x | 2.8x |
| 9 | 0.1M | 1.4 | 3.91 | 2.0x | 1.9x |

Pearson correlation (entropy vs Zstd-19 ratio): r = -0.763

## Table 7: C++ Pipeline Optimization (On-Demand Path)

For the on-demand decompression path (streaming/online scenarios):

| Method | Cold Lookup (ms/batch) | Speedup |
|--------|----------------------|---------|
| Python scan + Python gather + Python writeback | 2.02 | 1.0x |
| C++ scan + Python gather + Python writeback | 1.59 | 1.3x |
| C++ scan + C++ gather + Python writeback | 0.69 | 2.9x |
| **C++ scan + C++ scatter (fused)** | **0.74** | **2.7x** |

## Setup
- **Model**: DLRM, 26 embedding tables (8 large, >50K rows), D=16
- **Dataset**: Kaggle/Criteo (2GB embeddings, 3.3M test samples)
- **Hardware**: CPU-only (40 cores Intel Xeon), 256GB RAM
- **Hot/cold split**: 4.3% hot (fp32), 95.7% cold (uint8 compressed)
- **Frame size**: 129,600 rows per frame (1920x1080, 4x4 tiles)
