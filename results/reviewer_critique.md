# Critical Review: DLRM Embedding Compression via H.265 Video Codecs

## What We Have
- Hot/cold split with H.265 compression of cold embeddings
- C++ fused tiling pipeline (116x faster than Python)
- LRU frame cache with batch-affinity reordering
- 7.8x memory reduction, 1.7x forward pass speedup on Kaggle/Criteo

## Major Weaknesses

### 1. "Why H.265?" — Missing compression baseline comparison
- Need direct comparison: same hot/cold split, same quantization, but with LZ4/Zstd/Snappy
- If LZ4 gets similar compression ratio with 10x faster decode, the whole paper falls apart
- Also missing: comparison with existing embedding compression methods

### 2. Dataset is too small / single
- Kaggle/Criteo is only 2GB of embeddings
- Need Criteo Terabyte (~100GB embeddings) or production-scale
- One dataset = no generalization argument

### 3. Wall-clock speedup is marginal (1.05-1.12x)
- Forward pass is 1.7x but wall-clock only 1.12x (data loading = 78%)
- Need to isolate forward pass or show GPU scenario

### 4. No accuracy impact analysis for lossy
- CRF sweep: decode speed measured but not AUC impact
- Need full Pareto frontier: accuracy vs memory vs latency

### 5. No comparison with right baselines
- Missing: TT-Rec, mixed-dimension, compositional embeddings, QAT
- Missing: simple mmap + OS page cache
- Missing: SSD-based embedding (Meta's approach)

### 6. Reordering as contribution
- Useful trick but not standalone contribution
- Need to formalize as optimization problem
- Need to show generalization beyond DLRM

## Action Items
1. [ ] Benchmark LZ4/Zstd/Snappy on same quantized data
2. [ ] Run on Criteo Terabyte dataset
3. [ ] CRF sweep with AUC measurement
4. [ ] Compare with mmap baseline
5. [ ] Formalize reordering algorithm
6. [ ] Measure GPU inference scenario
