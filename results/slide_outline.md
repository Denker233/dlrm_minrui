# Presentation Outline: DLRM Embedding Compression via H.265 Video Codecs

## Slide 1: Problem & Motivation
- DLRM embedding tables dominate memory (2GB+ for Kaggle/Criteo)
- Most rows are rarely accessed (power-law distribution)
- Graph: access frequency CDF showing the 4.3% hot / 95.7% cold split

## Slide 2: Approach: Video Codec Compression
- Hot/cold split → keep hot in fp32, compress cold as H.265 frames
- Diagram: architecture overview (hot table in L3 cache, cold in compressed frames, LRU decode cache)

## Slide 3: Addressing Concern: "Tiling is slow due to non-contiguous copies"
- Bar chart: tiling cost (<0.1ms) vs decode cost (3-4ms) vs forward pass (2-4ms)
- Tiling is <3% of the pipeline — decode dominates, not memory copies
- C++ fused tiling is 35-100x faster than Python

## Slide 4: Embedding Lookup Speedup
- Bar chart: baseline emb lookup vs codec emb lookup (5-10x faster)
- Why: hot table (87MB) fits in L3 cache, cold rows served from decoded cache

## Slide 5: End-to-End Forward Pass Speedup
- Bar chart across batch sizes: 1.37x (bs=4096) to 1.85x (bs=256)
- Show that data loading dominates wall clock (pie chart: 78% data loading)

## Slide 6: Memory Reduction
- Stacked bar: baseline (2061MB) vs codec (265MB) = 7.8x reduction
- Breakdown: hot table + compressed disk + LRU cache + mappings

## Slide 7: Decode Optimization: Frame Granularity
- Line plot from granularity benchmark: decode time vs frame resolution
- Second line: compression ratio vs frame resolution
- Show 960x544 sweet spot (3.9x faster decode, better compression)

## Slide 8: Decode Optimization: Codec Tuning
- Bar chart: decode time for lossless vs CRF=10/18/23 vs H.264 vs FFV1
- Key insight: lossy CRF=18 → 2x faster decode, negligible error
- WPP already provides 5.5x benefit (show wpp=0 vs wpp=1)

## Slide 9: Cache Efficiency
- Line plot: cache hit rate vs cache size (frames) for different resolutions
- Batch-affinity reordering enables 99%+ hit rate with cache=2

## Slide 10: Summary Table
- Columns: Config, Memory, Decode/batch, Forward speedup, AUC delta
- Compare: baseline, codec+cache=16, codec+cache=2
