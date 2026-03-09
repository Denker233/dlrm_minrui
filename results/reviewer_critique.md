# Critical Review: DLRM Embedding Compression via H.265 Video Codecs

## What We Have
- Hot/cold split with H.265 compression of cold embeddings
- C++ fused tiling pipeline (116x faster than Python)
- LRU frame cache with batch-affinity reordering
- 7.8x memory reduction, 1.7x forward pass speedup on Kaggle/Criteo

## Major Weaknesses & Findings

### 1. "Why H.265?" — Compression Baseline Comparison [COMPLETED]

**Result: H.265 does NOT clearly dominate general-purpose compressors.**

| Compressor | Ratio (vs uint8) | Decode (ms/frame) | Notes |
|-----------|------------------|--------------------|-------|
| H.265 CRF=0 | 9.9x | 40.1ms | Lossless |
| Zstd-19 | 9.4x | 1.13ms | **35x faster decode, similar ratio** |
| Zstd-9 | 7.6x | 1.79ms | Good balance |
| Zstd-3 | 6.8x | 2.37ms | Fastest encode |
| LZ4 | 3.3x | 1.42ms | Lowest latency |
| Snappy | 3.9x | 2.32ms | Google's choice |

**Key finding**: Zstd-19 achieves 95% of H.265's compression ratio with 35x faster decode.
This means the video codec's spatial exploitation provides marginal compression benefit
over a well-tuned byte-stream compressor on frequency-sorted embedding data.

**However**: The reordering technique benefits ALL compressors equally. This is actually
an argument FOR reordering as the primary contribution, not the codec choice.

### 2. Dataset Scale [PARTIALLY ADDRESSED]
- Kaggle/Criteo only (2GB embeddings, 26 tables)
- Criteo Terabyte requires 24 days of data; we only have 7 days
- **Status**: Cannot run Terabyte without downloading remaining data
- **Mitigation**: Focus on per-table scaling analysis; show results hold across all 8 large tables

### 3. CRF Accuracy-Compression Tradeoff [COMPLETED]

**Result: Lossy H.265 encoding has negligible accuracy impact up to CRF=18.**

| CRF | Compression | AUC Delta | MaxErr (uint8) | Decode (ms) |
|-----|-------------|-----------|----------------|-------------|
| 0 (lossless) | 9.9x | -0.000001 | 0/255 | 40.1ms |
| 10 | 20.9x | -0.000034 | 7/255 | 28.8ms |
| 18 | 225.5x | -0.000100 | 19/255 | 11.9ms |
| 23 | 434.1x | -0.000190 | 28/255 | 10.5ms |
| 28 | 596.1x | -0.000316 | 56/255 | 9.7ms |

**Key finding**: CRF=18 achieves 225x compression with only 0.01% AUC loss.
This IS a unique advantage of video codecs — general-purpose compressors can only do lossless.
The lossy capability trades minimal accuracy for massive compression improvement.

### 4. mmap Baseline Comparison [COMPLETED]

**Result: mmap uint8 is actually faster than fp32 baseline.**

| Approach | AUC | Batch Latency | RSS | Disk |
|---------|-----|---------------|-----|------|
| Baseline (fp32) | 0.802497 | 6.40ms | 19,476MB | 2,061MB |
| mmap (uint8) | 0.802481 | 5.72ms | 17,769MB | 539MB |
| H.265 cache=16 | 0.802481 | ~6.3ms | ~256MB | 53MB |
| Zstd-19 cache=16 | 0.802481 | ~6.1ms | ~258MB | 55MB |

**Key finding**: mmap gives you 4x size reduction for free (from quantization) with
NO latency overhead. The codec approach reduces disk footprint further (55MB vs 539MB)
but the in-memory working set is similar (~256MB vs 539MB) once you add hot partition,
LRU cache, and index mappings.

The compelling case for codecs is when the full uint8 table doesn't fit in memory.

### 5. Reordering as Primary Contribution
- Reordering by access frequency creates spatial locality
- Benefits ALL compressors (not just H.265)
- The reordering contribution is codec-agnostic — this is the real innovation
- Should be positioned as "embedding layout optimization" with codec as one application

### 6. Wall-clock Speedup
- Forward pass: 1.7x faster (codec cache=16)
- End-to-end: limited by data loading (78% of wall time)
- **Honest framing**: Memory reduction is the primary benefit, not latency

## Revised Contribution Narrative

Instead of "H.265 is the best compressor for embeddings", the story should be:

1. **Embedding layout optimization**: Frequency-sorted reordering creates spatial locality
   that improves compression ratio for ANY compressor by 1.8-2.4x
2. **Hot/cold partitioning**: Explicit hot/cold split with LRU frame caching provides
   predictable memory budgets — key for production deployment
3. **Lossy compression opportunity**: Video codecs uniquely offer lossy compression
   (CRF=18: 225x compression, 0.01% AUC loss) — general-purpose compressors cannot
4. **Compressor choice**: For lossless, Zstd-19 dominates (35x faster decode, 95% of
   H.265 ratio). For extreme compression, lossy H.265 is unique.

## Action Items Status
1. [x] Benchmark LZ4/Zstd/Snappy on same quantized data → Done
2. [ ] Run on Criteo Terabyte dataset → Blocked (missing data days 7-24)
3. [x] CRF sweep with AUC measurement → Done
4. [x] Compare with mmap baseline → Done
5. [ ] Formalize reordering algorithm → TODO
6. [ ] Measure GPU inference scenario → TODO
