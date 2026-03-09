# Final Results: DLRM Embedding Compression

## Table 1: System Comparison

| System | Cold Storage | Disk (MB) | AUC | AUC Loss | BLat (ms) | Overhead |
|--------|-------------|-----------|-----|----------|-----------|----------|
| Baseline (fp32) | None | 2,061 | 0.802497 | — | 4.21 | — |
| Quantize (uint8) | In-memory | 539 | 0.802481 | -0.002% | ~4.2 | ~0% |
| mmap (uint8) | OS page cache | 539 | 0.802481 | -0.002% | 5.72 | +36% |
| **Zstd-19 + LRU** | **C++ fused** | **55** | **0.802496** | **-0.0001%** | **5.35** | **+27%** |
| Zstd-3 + LRU | C++ fused | 77 | 0.802496 | -0.0001% | 5.55 | +32% |
| H.265 CRF=0 + LRU | Python + cache | 53 | 0.802496 | -0.0001% | 6.19 | +47% |
| H.265 CRF=18 + LRU | Lossy | 4.3 | 0.802397 | -0.012% | ~6.0 | ~43% |

Notes:
- All lossless systems achieve AUC within 0.002% of baseline (quantization noise)
- BLat = mean batch latency over 1,599 batches (batch_size=2048)
- LRU cache = 16 frames, 99.84% hit rate (22 misses in 13,812 frame accesses)
- Zstd-19 uses C++ fused scan+scatter pipeline; H.265 uses Python decode path

## Table 2: Compression Ratio

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

## Table 4: C++ Pipeline Optimization

| Method | Cold Lookup (ms/batch) | Speedup |
|--------|----------------------|---------|
| Python scan + Python gather + Python writeback | 2.02 | 1.0x |
| C++ scan + Python gather + Python writeback | 1.59 | 1.3x |
| C++ scan + C++ gather + Python writeback | 0.69 | 2.9x |
| **C++ scan + C++ scatter (fused)** | **0.74** | **2.7x** |

## Table 5: Cache Size Sensitivity (Zstd-19)

| Cache Size (frames) | Hit Rate | BLat (ms) | p99 (ms) |
|---------------------|----------|-----------|----------|
| 1 | 85.86% | 12.79 | 20.09 |
| 2 | 98.08% | 7.49 | 20.18 |
| **4** | **99.84%** | **6.84** | **10.48** |
| 8 | 99.84% | 6.68 | 10.34 |
| 16 | 99.84% | 6.69 | 10.45 |
| 64 | 99.84% | 6.55 | 9.00 |

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

## Setup
- **Model**: DLRM, 26 embedding tables (8 large, >50K rows), D=16
- **Dataset**: Kaggle/Criteo (2GB embeddings, 3.3M test samples)
- **Hardware**: CPU-only (40 cores Intel Xeon), 256GB RAM
- **Hot/cold split**: 4.3% hot (fp32), 95.7% cold (uint8 compressed)
- **Frame size**: 129,600 rows per frame (1920×1080, 4×4 tiles)
