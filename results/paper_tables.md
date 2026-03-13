# Paper-Ready Tables

## Table 1: Compression Method Comparison (Kaggle, D=16)

| Method | Type | Ratio | AUC Loss (%) | Retraining |
|--------|------|-------|-------------|------------|
| INT8 quantization | Quant | 4.0x | 0.002 | No |
| INT4 quantization | Quant | 8.0x | 0.190 | No |
| PQ (M=2, K=256) | Quant | 31.9x | 0.559 | No |
| SVD (rank=4) | Low-rank | 4.0x | 0.101 | No |
| Row pruning 99% | Pruning | 100.0x | 0.181 | No |
| Zstd-19 + uint8 | Lossless | 132.7x | 0.002 | No |
| CAFE+ (paper) | Sketch | ~1000x | ~0.75 | **Yes** |
| **H.265 CRF=0 (ours)** | **Codec** | **130.6x** | **0.002** | **No** |
| **H.265 CRF=18+freq (ours)** | **Codec** | **1359.8x** | **0.037** | **No** |

## Table 2: Compression Method Comparison (Terabyte, D=64)

| Method | Ratio | AUC Loss (%) |
|--------|-------|-------------|
| INT8 quantization | 4.0x | 0.001 |
| INT4 quantization | 8.0x | 0.094 |
| PQ (M=2, K=256) | 126.5x | 0.763 |
| Row pruning 99% | 100.0x | 0.019 |
| Zstd-19 + uint8 | 14.2x | 0.001 |
| H.265 CRF=0 (ours) | 13.7x | 0.001 |
| **H.265 CRF=18+freq (ours)** | **68.6x** | **0.050** |

## Table 3: CRF Quality-Compression Tradeoff (Kaggle)

| CRF | Compressed (MB) | Ratio (vs fp32) | AUC Loss (%) |
|-----|-----------------|-----------------|--------------|
| 0 (lossless) | 16.4 | 125.6x | 0.002 |
| 10 | 1.5 | 1382.5x | 0.021 |
| 18 | 0.8 | 2667.5x | 0.089 |
| 23 | 0.7 | 2826.9x | 0.192 |
| 28 | 0.7 | 2874.1x | 0.508 |

Note: These ratios are for cold-only encoding (full table sort).
Exp 2 ratios (1359.8x at CRF=18) are for large-table cold rows only.

## Table 3b: CRF Quality-Compression Tradeoff (Terabyte)

| CRF | Compressed (MB) | Ratio (vs fp32) | AUC Loss (%) |
|-----|-----------------|-----------------|--------------|
| 0 (lossless) | 499.8 | 11.0x | 0.001 |
| 10 | 164.6 | 33.5x | 0.021 |
| 18 | 22.3 | 247.0x | 0.088 |
| 23 | 4.5 | 1213.7x | 0.184 |
| 28 | 2.3 | 2385.6x | 0.407 |

Storage (hot + compressed): CRF=18 → 259.6 MB (21.3x reduction from 5.5 GB).

## Table 4: Runtime Memory Breakdown

### Kaggle (D=16)
| Component | Baseline | Our System | % |
|-----------|----------|------------|---|
| Embeddings (fp32) | 2,058 MB | — | — |
| Hot embeddings (fp32) | — | 88.5 MB | 34% |
| Compressed cold | — | 0.8 MB | 0% |
| Decoded cache (20 frames) | — | 39.6 MB | 15% |
| Mapping + bitmap | — | 134.6 MB | 51% |
| **Total** | **2,058 MB** | **263 MB** | — |
| **Reduction** | — | **7.8x** | — |

### Terabyte (D=64)
| Component | Baseline | Our System | % |
|-----------|----------|------------|---|
| Embeddings (fp32) | 5,520 MB | — | — |
| Hot embeddings (fp32) | — | 237 MB | 37% |
| Compressed cold (CRF=18) | — | 22 MB | 3% |
| Decoded cache (147 frames) | — | 291 MB | 45% |
| Mapping + bitmap | — | 90 MB | 14% |
| **Total** | **5,520 MB** | **641 MB** | — |
| **Reduction** | — | **8.6x** | — |

## Table 5: Zero-Out vs Lossy Compression

| Method | Kaggle Ratio | Kaggle AUC Loss | Terabyte Ratio | Terabyte AUC Loss |
|--------|-------------|-----------------|----------------|-------------------|
| Zero 90% | 10x | 0.008% | 10x | 0.001% |
| Zero 95% | 20x | 0.031% | 20x | 0.002% |
| Zero 99% | 100x | 0.181% | 100x | 0.019% |
| Zero 99.9% | 1000x | 0.694% | 1000x | 0.181% |
| **H.265 CRF=18+freq** | **1360x** | **0.037%** | **69x** | **0.050%** |

At 1000x compression, H.265 CRF=18 loses 18.8x less AUC than zero-out (Kaggle).

## Table 6: Error Steering (MSE by frequency bucket, Kaggle Table 2)

| Ordering | Top 1% | 1-10% | 10-50% | 50-100% | Overall |
|----------|--------|-------|--------|---------|---------|
| Random | 1.156 | 0.037 | 0.015 | 0.026 | 0.034 |
| Natural | 0.790 | 0.034 | 0.012 | 0.028 | 0.030 |
| **Freq sort (ours)** | **0.559** | **0.034** | **0.012** | **0.024** | **0.026** |
| Freq / Random | **2.1x** | 1.1x | 1.2x | 1.1x | 1.3x |

## Table 7: Frame Access Concentration

| Dataset | Total Frames | Accessed Frames | Reduction |
|---------|-------------|-----------------|-----------|
| Kaggle | 254 | 20 | 92% |
| Terabyte | 671 | 147 | 78% |

## Table 8: Deployment Scenarios

| Scenario | RAM | Baseline Instances | Our Instances | Improvement |
|----------|-----|-------------------|---------------|-------------|
| Edge (4 GB) | 4 GB | 0 (doesn't fit) | 15 (Kaggle) | Enables deployment |
| Laptop (16 GB) | 16 GB | 7 | 61 | 8.7x |
| Server (256 GB) | 256 GB | 128 | 985 | 7.7x |
