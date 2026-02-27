# Embedding Row Reordering: Comprehensive Comparison

Date: 2026-02-26
Dataset: Kaggle/Criteo, 26 tables, emb_dim=16
Hardware: CPU, 40 threads
Runs per experiment: 3 (averaged)

---

## Experiment 1: Reordering-Only (No Codec, Full fp32 Embeddings)

Tests pure CPU cache locality impact of physically rearranging embedding table rows.
All tables remain uncompressed fp32 (2061MB total). Only the row order changes.

| Method | AUC | Avg Time | ±Std | Mean Lat | P50 Lat | P99 Lat | Speedup |
|---|---|---|---|---|---|---|---|
| Baseline (original order) | 0.802497 | 8.63s | 0.37s | 5.37ms | 5.27ms | 7.49ms | 1.00x |
| Frequency-sort (most accessed first) | 0.802497 | 7.37s | 0.62s | 4.58ms | 4.23ms | 6.48ms | 1.17x |
| Batch-affinity (hot first + cold sorted) | 0.802497 | 6.67s | 0.53s | 4.15ms | 4.00ms | 5.26ms | 1.29x |
| Rec-AD Louvain (hot first + cold community) | 0.802497 | 6.69s | 0.37s | 4.16ms | 3.97ms | 4.93ms | 1.29x |

### Delta vs Baseline

| Method | Time Delta | Lat Delta | P99 Delta |
|---|---|---|---|
| Frequency-sort | -1.26s (-14.6%) | -0.79ms | -1.01ms |
| Batch-affinity | -1.96s (-22.7%) | -1.22ms | -2.23ms |
| Rec-AD Louvain | -1.94s (-22.5%) | -1.21ms | -2.56ms |

### Analysis

- Row reordering alone yields 1.29x speedup from CPU cache locality improvement.
- Batch-affinity and Rec-AD produce nearly identical results because most cold
  indices appear in only ~1 batch (avg batches/idx = 1.0 for the 5 largest tables),
  leaving no co-occurrence structure for Rec-AD's community detection to exploit.
- Frequency-sort helps (1.17x) but batch-affinity is better because it also places
  co-accessed cold rows adjacent, not just hot rows first.
- P99 latency improves dramatically (7.49ms -> 4.93ms) due to fewer cache misses.

### Per-Table Co-occurrence Statistics

| Table | Cold Rows | Active Cold | Avg Batches/Idx | Louvain Communities | Largest Community |
|---|---|---|---|---|---|
| 2 | 9.6M | 449K | 1.0 | N/A (batch-affinity fallback) | N/A |
| 3 | 2.2M | 271K | 1.3 | 262,832 | 25 |
| 9 | 89K | 41K | 8.2 | 26,024 | 2,777 |
| 11 | 8.0M | 429K | 1.0 | N/A (batch-affinity fallback) | N/A |
| 15 | 5.3M | 401K | 1.0 | N/A (batch-affinity fallback) | N/A |
| 20 | 6.7M | 415K | 1.0 | N/A (batch-affinity fallback) | N/A |
| 23 | 281K | 73K | 3.6 | 41,978 | 5,529 |
| 25 | 138K | 49K | 3.9 | 27,037 | 4,460 |

---

## Experiment 2: Hot Embedding Reordering (Codec Pipeline, 1080p, full_cpp)

Tests whether reordering hot embedding rows improves lookup latency within the
compressed embedding pipeline. Cold rows use existing batch-affinity reordering
with H.265 codec. Only the hot row order varies.

| Method | AUC | Avg Time | ±Std | Mean Lat | P50 Lat | P99 Lat | Speedup |
|---|---|---|---|---|---|---|---|
| Original hot order (codec baseline) | 0.802489 | 3.86s | 0.18s | 2.38ms | 2.30ms | 3.65ms | 1.00x |
| Frequency-sorted hot | 0.802489 | 3.97s | 0.11s | 2.45ms | 2.35ms | 3.34ms | 0.97x |
| Batch-affinity hot | 0.802489 | 4.05s | 0.14s | 2.50ms | 2.48ms | 3.75ms | 0.95x |

### Delta vs Original Hot Order

| Method | Time Delta | Lat Delta | P99 Delta |
|---|---|---|---|
| Frequency-sorted hot | +0.11s (+2.8%) | +0.07ms | -0.31ms |
| Batch-affinity hot | +0.19s (+4.9%) | +0.12ms | +0.10ms |

### Analysis

- Hot reordering does NOT help in the codec pipeline. Slightly slower overall.
- Root cause: hot embeddings are already compact (~22MB across 8 tables, fits in
  L3 cache), so row layout within that small tensor has minimal cache impact.
- Reordered configs lose access to bitmap-rank mode (which uses hardware popcount
  for O(1) lookups). The mapping tensor fallback has slightly higher overhead,
  which outweighs any cache locality benefit.
- The MLP dominates the forward pass (~65% of latency), making hot embedding
  lookup optimization a marginal contributor.

### Note on bitmap compatibility

Bitmap-rank mode computes hot compact indices via hardware popcount, which
implicitly assumes hot_weight rows are sorted by original embedding index.
Reordering hot rows requires falling back to the mapping tensor code path,
which is slightly slower. A bitmap-aware hot reorder would need to modify the
C++ extension to support an indirection layer, adding complexity for no gain.

---

## Summary

| Technique | Context | Speedup | Verdict |
|---|---|---|---|
| Row reordering (cold) | Full fp32 tables | **1.29x** | Significant win from cache locality |
| Batch-affinity vs Rec-AD | Full fp32 tables | Identical | Rec-AD adds complexity for no gain on this dataset |
| Frequency sort | Full fp32 tables | **1.17x** | Helps but weaker than batch-affinity |
| Hot row reordering | Codec pipeline | **0.95-0.97x** | No benefit; hot tensor already fits in cache |
| Cold row reordering | Codec pipeline | **2.2x** (from prior results) | Enables efficient frame-based caching |
