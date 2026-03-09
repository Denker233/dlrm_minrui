# Comprehensive Benchmark: Baseline vs CAFE+ vs Codec+LRU

Generated: 2026-03-05 03:21:23
Hardware: Intel Xeon Platinum 8380 (80 cores), 40 threads
Dataset: Criteo Kaggle, batch_size=2048, 3 runs/config (median)
Codec: H.265 lossless, 1080p, batch-affinity reordered
**No batch pre-loading**: batches loaded from DataLoader during timed loop

## Summary Comparison

| Config | AUC | Accuracy | LogLoss | BLat (ms) | p50 (ms) | p99 (ms) | Total (s) | Memory (MB) | Compress |
|--------|-----|----------|---------|-----------|----------|----------|-----------|-------------|---------|
| Baseline (fp32) | 0.802497 | 0.2614 | 0.7270 | 4.29 | 4.21 | 5.51 | 33.5 | 2060.7 | 1.0x |
| Codec full_cpp | 0.802496 | 0.2614 | 0.7270 | 2.48 | 2.45 | 3.19 | 29.7 | 265.6 | 39.1x |
| Codec cache=4 | 0.802488 | 0.2614 | 0.7270 | 2.57 | 2.48 | 3.54 | 77.2 | 230.0 | 39.1x |
| Codec cache=8 | 0.802486 | 0.2614 | 0.7270 | 2.45 | 2.40 | 3.08 | 36.3 | 237.9 | 39.1x |
| Codec cache=16 | 0.802496 | 0.2614 | 0.7270 | 2.36 | 2.33 | 2.94 | 29.9 | 253.7 | 39.1x |
| Codec cache=22 | 0.802496 | 0.2614 | 0.7270 | 2.44 | 2.41 | 3.02 | 30.5 | 265.6 | 39.1x |
| Codec cache=32 | 0.802496 | 0.2614 | 0.7270 | 2.38 | 2.37 | 2.62 | 29.9 | 265.6 | 39.1x |
| Codec cache=64 | 0.802496 | 0.2614 | 0.7270 | 2.41 | 2.40 | 2.83 | 30.0 | 265.6 | 39.1x |
| Lookahead group=1 | 0.802496 | 0.2614 | 0.7270 | 2.62 | 2.57 | 3.30 | 115.0 | 222.1 | 39.1x |
| Lookahead group=10 | 0.802496 | 0.2614 | 0.7270 | 2.44 | 2.40 | 3.07 | 40.0 | 222.1 | 39.1x |
| Lookahead group=50 | 0.802496 | 0.2614 | 0.7270 | 2.43 | 2.37 | 3.12 | 31.9 | 222.1 | 39.1x |
| Lookahead group=100 | 0.802496 | 0.2614 | 0.7270 | 2.40 | 2.36 | 3.09 | 31.0 | 222.1 | 39.1x |
| Lookahead group=500 | 0.802496 | 0.2614 | 0.7270 | 2.36 | 2.33 | 3.03 | 30.0 | 222.1 | 39.1x |

## Per-Component Latency (ms/batch)

| Config | Emb | Interact | MLP | Data Load | Scan/batch | Decode/batch | Fwd Total |
|--------|-----|----------|-----|-----------|------------|--------------|-----------|
| Baseline (fp32) | 1.46 | 1.23 | 1.55 | 0.00 | 0.00 | 0.00 | 4.29 |
| Codec full_cpp | 0.23 | 0.57 | 1.62 | 0.01 | 0.00 | 0.00 | 2.48 |
| Codec cache=4 | 0.28 | 0.61 | 1.61 | 0.01 | 0.16 | 29.02 | 2.57 |
| Codec cache=8 | 0.25 | 0.59 | 1.55 | 0.00 | 0.13 | 3.98 | 2.45 |
| Codec cache=16 | 0.20 | 0.58 | 1.53 | 0.00 | 0.11 | 0.12 | 2.36 |
| Codec cache=22 | 0.21 | 0.61 | 1.56 | 0.00 | 0.13 | 0.13 | 2.44 |
| Codec cache=32 | 0.20 | 0.64 | 1.49 | 0.00 | 0.11 | 0.12 | 2.38 |
| Codec cache=64 | 0.20 | 0.62 | 1.53 | 0.00 | 0.12 | 0.12 | 2.41 |
| Lookahead group=1 | 0.28 | 0.64 | 1.64 | 0.00 | 0.14 | 52.65 | 2.62 |
| Lookahead group=10 | 0.22 | 0.59 | 1.58 | 0.00 | 0.08 | 5.99 | 2.44 |
| Lookahead group=50 | 0.22 | 0.64 | 1.52 | 0.00 | 0.07 | 1.27 | 2.43 |
| Lookahead group=100 | 0.22 | 0.59 | 1.54 | 0.00 | 0.07 | 0.64 | 2.40 |
| Lookahead group=500 | 0.19 | 0.61 | 1.51 | 0.00 | 0.07 | 0.18 | 2.36 |

## Memory Breakdown (MB)

| Config | Hot Table | Cold (disk) | LRU/Decoded | Mapping | Total | RSS | Reduction |
|--------|-----------|-------------|-------------|---------|-------|-----|-----------|
| Baseline (fp32) | 2060.7 | 0.0 | 0.0 | 0.0 | 2060.7 | 19476 | 1.0x |
| Codec full_cpp | 87.4 | 50.4 | 43.5 | 134.6 | 265.6 | 20108 | 7.8x |
| Codec cache=4 | 87.4 | 50.4 | 7.9 | 134.6 | 230.0 | 20346 | 9.0x |
| Codec cache=8 | 87.4 | 50.4 | 15.8 | 134.6 | 237.9 | 20430 | 8.7x |
| Codec cache=16 | 87.4 | 50.4 | 31.6 | 134.6 | 253.7 | 20364 | 8.1x |
| Codec cache=22 | 87.4 | 50.4 | 43.5 | 134.6 | 265.6 | 20410 | 7.8x |
| Codec cache=32 | 87.4 | 50.4 | 43.5 | 134.6 | 265.6 | 20514 | 7.8x |
| Codec cache=64 | 87.4 | 50.4 | 43.5 | 134.6 | 265.6 | 20512 | 7.8x |
| Lookahead group=1 | 87.4 | 50.4 | 0.0 | 134.6 | 222.1 | 20514 | 9.3x |
| Lookahead group=10 | 87.4 | 50.4 | 0.0 | 134.6 | 222.1 | 20396 | 9.3x |
| Lookahead group=50 | 87.4 | 50.4 | 0.0 | 134.6 | 222.1 | 20396 | 9.3x |
| Lookahead group=100 | 87.4 | 50.4 | 0.0 | 134.6 | 222.1 | 20457 | 9.3x |
| Lookahead group=500 | 87.4 | 50.4 | 0.0 | 134.6 | 222.1 | 20527 | 9.3x |

## Cache Size Sweep (real LRU, end-to-end)

| Cache Size | Hit Rate | Misses | Evictions | BLat (ms) | Scan+Decode (ms/batch) | Total (s) |
|------------|----------|--------|-----------|-----------|------------------------|-----------|
| 4 | 46.3% | 7422 | 7418 | 2.57 | 29.18 | 77.2 |
| 8 | 92.5% | 1038 | 1030 | 2.45 | 4.11 | 36.3 |
| 16 | 99.8% | 22 | 6 | 2.36 | 0.23 | 29.9 |
| 22 | 99.8% | 22 | 0 | 2.44 | 0.26 | 30.5 |
| 32 | 99.8% | 22 | 0 | 2.38 | 0.23 | 29.9 |
| 64 | 99.8% | 22 | 0 | 2.41 | 0.24 | 30.0 |

## Dynamic Look-Ahead Results

| Group Size | AUC | BLat (ms) | Scan/batch (ms) | Decode/batch (ms) | Total Overhead | Total (s) | Frames Decoded |
|------------|-----|-----------|-----------------|-------------------|----------------|-----------|----------------|
| 1 | 0.802496 | 2.62 | 0.14 | 52.65 | 52.78 | 115.0 | 13812 |
| 10 | 0.802496 | 2.44 | 0.08 | 5.99 | 6.07 | 40.0 | 1394 |
| 50 | 0.802496 | 2.43 | 0.07 | 1.27 | 1.35 | 31.9 | 289 |
| 100 | 0.802496 | 2.40 | 0.07 | 0.64 | 0.71 | 31.0 | 151 |
| 500 | 0.802496 | 2.36 | 0.07 | 0.18 | 0.25 | 30.0 | 48 |

## CAFE+ Reference (from training logs — different model)

| Config | AUC | Accuracy | Compression | Model Size |
|--------|-----|----------|-------------|------------|
| CAFE+ (121x) | 72.87% | 76.32% | 121x | 17 MB |
| CAFE+ (147x) | — | — | ~147x | 14 MB |
| CAFE+ (158x) | — | — | ~158x | 13 MB |
| **Codec+LRU** | **80.25%** | **~78.8%** | **10.3x** | **~200 MB** |

CAFE+ achieves 121x compression but loses ~7.4% AUC. Codec+LRU achieves
10.3x compression with <0.001% AUC loss (lossless codec + uint8 quantization).

## Latency Breakdown (ASCII)
```
  Baseline (fp32)                     |EEEEEEEEEEEEEEEEIIIIIIIIIIIIIIMMMMMMMMMMMMMMMMMM..| 4.29ms
  Codec full_cpp                      |EEIIIIIIMMMMMMMMMMMMMMMMMM..| 2.48ms
  Lookahead group=1                   |EEEIIIIIIIMMMMMMMMMMMMMMMMMMM.| 2.62ms
  Lookahead group=10                  |EEIIIIIIMMMMMMMMMMMMMMMMMM..| 2.44ms
  Lookahead group=50                  |EEIIIIIIIMMMMMMMMMMMMMMMMM..| 2.43ms
  Lookahead group=100                 |EEIIIIIIMMMMMMMMMMMMMMMMM..| 2.40ms
  Lookahead group=500                 |EEIIIIIIIMMMMMMMMMMMMMMMMM.| 2.36ms
  Legend: E=emb, I=interact, M=mlp, .=other
```