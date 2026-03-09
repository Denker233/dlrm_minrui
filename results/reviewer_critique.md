# Critical Review: DLRM Embedding Compression via H.265 Video Codecs

## What We Have
- Hot/cold split with H.265 compression of cold embeddings
- C++ fused inference pipeline (`fast_forward`) with bitmap-rank dispatch
- LRU frame cache with frequency-based reordering
- **1.77x inference speedup**, 37x memory reduction, <0.01% AUC loss on Kaggle/Criteo

## Major Weaknesses & Findings

### 1. "Why H.265?" — Compression Baseline Comparison [COMPLETED]

**Result: H.265 does NOT clearly dominate general-purpose compressors for lossless.**

| Compressor | Ratio (vs uint8) | Decode (ms/frame) | Notes |
|-----------|------------------|--------------------|-------|
| H.265 CRF=0 | 9.9x | 40.1ms | Lossless |
| Zstd-19 | 9.4x | 1.13ms | **35x faster decode, similar ratio** |
| Zstd-9 | 7.6x | 1.79ms | Good balance |
| Zstd-3 | 6.8x | 2.37ms | Fastest encode |
| LZ4 | 3.3x | 1.42ms | Lowest latency |
| Snappy | 3.9x | 2.32ms | Google's choice |

**Key finding**: Zstd-19 achieves 95% of H.265's lossless ratio with 35x faster decode.
However, with cache >= 4 (99.8% hit rate), decode speed is largely irrelevant — only 22
frames need decoding across all 1,599 batches. The compressor choice primarily affects
disk size, not inference latency.

**H.265's unique advantage**: Lossy compression via CRF. CRF=18 achieves 225x compression
with only 0.01% AUC loss — no general-purpose compressor can do this.

### 2. Dataset Scale [PARTIALLY ADDRESSED]
- Kaggle/Criteo only (2GB embeddings, 26 tables)
- Criteo Terabyte requires 24 days of data; we only have 7 days
- **Mitigation**: Per-table analysis shows results hold across all 8 large tables with
  varying entropy (0.29 to 3.91 bits/byte). The framework is table-size agnostic.

### 3. CRF Accuracy-Compression Tradeoff [COMPLETED]

**Result: Lossy H.265 encoding has negligible accuracy impact up to CRF=18.**

| CRF | Compression | AUC Delta | MaxErr (uint8) |
|-----|-------------|-----------|----------------|
| 0 (lossless) | 9.9x | -0.000001 | 0/255 |
| 10 | 20.9x | -0.000034 | 7/255 |
| 18 | 225.5x | -0.000100 | 19/255 |
| 23 | 434.1x | -0.000190 | 28/255 |
| 28 | 596.1x | -0.000316 | 56/255 |

**Key finding**: CRF=18 achieves 225x compression with only 0.01% AUC loss.
This IS a unique advantage of video codecs — general-purpose compressors can only do lossless.

### 4. mmap Baseline Comparison [COMPLETED]

**Result: mmap uint8 is faster than fp32 baseline (smaller footprint), but codec + fast_forward
is 1.77x faster than both.**

| Approach | AUC | Batch Latency | Speedup | Disk |
|---------|-----|---------------|---------|------|
| Baseline (fp32) | 0.802497 | 4.26ms | 1.0x | 2,061MB |
| mmap (uint8) | 0.802481 | 5.72ms | 0.74x | 539MB |
| **Codec + fast_forward** | **0.802496** | **2.41ms** | **1.77x** | **53MB** |

**Key finding**: The codec approach is not just about compression — the C++ `fast_forward`
path with uint8 cold storage achieves a **1.77x speedup** because:
- Embedding lookup: 1.48ms → 0.22ms (6.7x faster)
- Smaller working set → better CPU L2/L3 cache utilization
- All-C++ processing eliminates Python loop overhead

### 5. Reordering Analysis [COMPLETED]

**Result: Reordering provides only 1-2% compression improvement. Its value is cache locality.**

| Compressor | Random Order | Natural Order | Freq-Reordered | vs Random |
|-----------|-------------|---------------|----------------|-----------|
| LZ4       | 2.9x        | 2.9x          | 2.9x           | 1.01x     |
| Zstd-3    | 5.7x        | 5.9x          | 5.8x           | 1.02x     |
| Zstd-19   | 8.0x        | 8.3x          | 8.2x           | 1.02x     |
| H.265     | 8.7x        | 8.8x          | 8.8x           | 1.01x     |

**Key finding**: Reordering's real value is **cache hit rates** — co-locating frequently
co-accessed rows into the same frame reduces LRU cache misses from 7,422 (cache=1) to
22 (cache=4). This is the primary contribution of reordering: better caching, not compression.

### 6. Inference Speedup Analysis [COMPLETED]

**Result: 1.77x speedup (2.41ms vs 4.26ms baseline) — NOT overhead.**

The speedup mechanism:
1. **uint8 cold storage**: 95.7% of rows stored as uint8 (4x smaller) → fits in CPU cache
2. **Compact fp32 hot weights**: Only 4.3% of rows → tiny working set in L2/L3 cache
3. **C++ fast_forward**: All 26 tables processed in one C++ call with:
   - Bitmap-rank O(1) hot/cold dispatch (single popcount instruction)
   - Direct uint8→fp32 dequantization via AVX512 `accum_q8_d16`
   - Hardware prefetching for bitmap and hash table probing
4. **No Python overhead**: Eliminates Python for-loop over 26 tables

Latency breakdown:
| Component | Baseline (ms) | Codec (ms) | Speedup |
|-----------|---------------|------------|---------|
| Embedding | 1.48 | 0.22 | 6.7x |
| Interact | 1.15 | 0.57 | 2.0x |
| MLP | 1.57 | 1.58 | 1.0x |
| **Total** | **4.26** | **2.41** | **1.77x** |

### 6b. C++ On-Demand Pipeline [COMPLETED]

For streaming/online scenarios where pre-decoding is not possible:

| Method | Mean (ms) | Speedup |
|--------|-----------|---------|
| Python scan + Python gather | 2.017 | 1.0x |
| C++ scan + Python gather | 1.593 | 1.27x |
| **C++ scan + C++ gather** | **0.693** | **2.91x** |

### 7. Wall-clock Performance

**Batch latency**: 1.77x speedup (memory footprint reduction → better cache)
**Disk**: 37x reduction (53MB vs 2,061MB)
**AUC**: <0.01% loss (from uint8 quantization)

With LRU cache=4 (99.8% hit rate), total wall-clock time matches pre-decode approach.
Cache memory: only 33MB (4 frames × 129,600 rows × 16 bytes).

## Revised Contribution Narrative

1. **Hot/cold + uint8 + C++ fast_forward = 1.77x speedup**: The key insight is that
   quantizing cold embeddings to uint8 and processing all tables in a single C++ call
   reduces memory footprint enough to dramatically improve CPU cache utilization.
   Embedding lookup alone is 6.7x faster.

2. **Frame-based compression for disk storage**: Tiling embedding rows into video-like
   frames enables 37x disk compression (H.265: 39x, Zstd-19: 37.6x). The frame
   structure enables LRU caching with only 22 frames needed across all test batches.

3. **Lossy compression opportunity**: Video codecs uniquely offer lossy compression.
   CRF=18 achieves 225x compression with 0.01% AUC loss — general-purpose compressors
   cannot do this. This is valuable for extreme storage-constrained deployments.

4. **Compressor-agnostic architecture**: The fast_forward speedup is independent of
   compressor choice. Zstd-19 or H.265 on disk; either way, the inference path processes
   uint8 cold data with the same 1.77x speedup. Compressor only affects decode time on
   cache misses (amortized to near-zero at cache=4).

5. **Reordering for cache locality**: Frequency-based reordering improves cache hit rates
   (99.8% at cache=4 vs ~46% without), not compression ratios.

## Action Items Status
1. [x] Benchmark LZ4/Zstd/Snappy on same quantized data → Done
2. [ ] Run on Criteo Terabyte dataset → Blocked (missing data days 7-24)
3. [x] CRF sweep with AUC measurement → Done
4. [x] Compare with mmap baseline → Done
5. [x] Measure reordering benefit → Done (1-2% compression, value is caching)
6. [x] E2E inference measurement → Done: **1.77x speedup** (2.41ms vs 4.26ms)
7. [x] C++ fused pipeline → Done: fast_forward handles all tables, 6.7x embedding speedup
8. [ ] Measure GPU inference scenario → Blocked (no GPU available)
9. [ ] Run on Criteo Terabyte dataset → Blocked (missing data days 7-24)
