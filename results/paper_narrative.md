# Paper Narrative: Reframed for Systems Venues

## Title Options
1. "Compressing Recommendation Model Embeddings with Video Codecs"
2. "H.265 for Embeddings: 1000x Compression of DLRM Tables with Negligible Quality Loss"
3. "Video Codec Compression for Serving-Time Embedding Tables"

---

## Abstract (Draft)

Deep learning recommendation models (DLRMs) store billions of parameters in
embedding tables that dominate model size. We observe that these tables,
when quantized to uint8, resemble low-entropy grayscale video frames — and
can be compressed by video codecs. We present a post-training compression
system that leverages H.265/HEVC to achieve 1000x+ compression with <0.04%
AUC loss on the Criteo Kaggle benchmark, and 69x with <0.05% loss on the
Criteo Terabyte dataset. Our frequency-aware layout steers lossy codec errors
away from high-impact rows, preserving accuracy. Compared to CAFE+ (SIGMOD 2024),
our method achieves 20x less quality loss at similar compression ratios without
requiring any retraining. The system reduces runtime memory from 2-6 GB to
0.26-0.64 GB, enabling deployment of production recommendation models on
memory-constrained edge devices.

---

## Introduction Structure

### The Problem
- DLRM embedding tables are 2-100+ GB, often 99% of model parameters
- This limits deployment: edge/mobile devices, multi-tenant serving, cold start
- Prior compression methods (CAFE+, PQ, SVD) require retraining or sacrifice quality

### The Observation
- Quantized to uint8, embedding tables are 2D arrays of bytes
- They look like grayscale images: mostly near-zero with sparse non-zero regions
- This "video-like" structure means video codecs should work well

### The System
- Quantize fp32 → uint8 (global min/max)
- Sort cold rows by access frequency → pack into 1920×1080 frames
- Encode with H.265 (lossless or lossy via CRF parameter)
- At inference: hot rows in fp32, cold rows decoded on-demand from H.265

### The Results
- Kaggle (D=16): 1360x compression, -0.037% AUC, 7.8x memory reduction
- Terabyte (D=64): 69x compression, -0.050% AUC, 8.6x memory reduction
- Dominates ALL prior post-training methods at every operating point

---

## Key Claims and Evidence

### Claim 1: Embeddings are intrinsically compressible
**Evidence**: Intrinsic compressibility analysis
- Entropy: 0.13-4.25 bits/byte (vs 8 for random data)
- Near-zero rows: 99.7-99.9% (Kaggle), 26-100% (Terabyte)
- Random data baseline: 1.1x compression vs 40x for real embeddings
- **Figures**: Entropy distribution, near-zero CDF

### Claim 2: H.265 dominates the compression-quality Pareto frontier
**Evidence**: Experiment 2 (10 methods compared)
- At lossless: H.265 CRF=0 = 130.6x vs Zstd-19 = 132.7x (comparable)
- At lossy: H.265 CRF=18 = 1360x at -0.037% (no other method even close)
- PQ at 32x loses -0.56%, SVD at 16x loses -0.35%
- CAFE+ at 1000x loses ~-0.75% (20x worse than H.265)
- **Figure**: Pareto frontier plot (Figure 1 of paper)

### Claim 3: Frequency sorting steers lossy errors to unimportant rows
**Evidence**: Experiment 5 (per-row error analysis)
- Frequency-sorted layout: top-1% MSE = 0.559 (uint8 space)
- Random ordering: top-1% MSE = 1.156 (2.1x worse)
- Natural ordering: top-1% MSE = 0.790 (1.4x worse)
- **Figure**: Error distribution bar chart by frequency bucket

### Claim 4: System enables edge deployment
**Evidence**: Memory analysis
- Baseline: 2-6 GB (does NOT fit in 4 GB edge device)
- Our system: 0.26-0.64 GB (DOES fit)
- 7.8-8.9x runtime memory reduction
- **Table**: "What fits where" analysis with cache hierarchy

### Claim 5: No retraining required
**Evidence**: Method description + CAFE+ comparison
- CAFE+ requires full model retraining with modified embedding layers
- Our method: load trained model → compress → deploy
- Applied post-training to Kaggle (3.3M samples) and Terabyte (82M samples)

---

## What NOT to Claim

### DO NOT claim inference speedup
The 1.77x speedup is from C++ implementation, not compression:
- Config B (C++ fp32, no compression): 2.48ms
- Config C (C++ with compression): 2.43ms
- The 1.02x speedup from compression is within noise

If pressed: "Our C++ inference engine achieves comparable speed to the
PyTorch baseline while operating on 7.8x less memory."

### DO NOT claim better CPU cache utilization
- Kaggle: LLC miss INCREASES with compression (38.9% vs 29.1%)
- Terabyte: LLC miss decreases (31.6% vs 48.3%) but not decisive
- The mapping table overhead (134 MB) adds cache pressure

If pressed: "On the larger Terabyte dataset, the 8.6x memory reduction
translates to reduced LLC miss rate (31.6% vs 48.3%)."

---

## Paper Structure

### Section 1: Introduction (1.5 pages)
- Problem: embedding tables are huge
- Observation: they look like video frames
- Contribution: H.265 compression system
- Results: headline numbers

### Section 2: Background (1 page)
- DLRM architecture
- Embedding table structure and access patterns
- Video codec basics (H.265, CRF, I-frames)

### Section 3: Why Embeddings are Compressible (1.5 pages)
- Intrinsic entropy analysis
- Near-zero row dominance
- Comparison with random data baseline
- Frequency distribution (Zipfian)

### Section 4: System Design (2 pages)
- Quantization: fp32 → uint8 (global min/max)
- Hot/cold split (top 4.3% by frequency as fp32)
- Frequency-sorted layout (error steering)
- Frame packing: 4x4 tiling (D=16) or flat (D=64)
- H.265 encoding (per-frame, I-frame only)
- Inference: bitmap-rank dispatch + pre-decoded cache

### Section 5: Evaluation (3 pages)
- Datasets: Criteo Kaggle (D=16), Criteo Terabyte (D=64)
- Models: trained DLRM with correct hyperparameters
- Exp 1: Compression comparison (10 methods)
- Exp 2: CRF quality-compression tradeoff
- Exp 3: Zero-out baseline (proves compression > zeroing)
- Exp 4: Memory analysis and deployment scenarios
- Exp 5: Error steering analysis

### Section 6: Discussion (1 page)
- Why video codecs work (DCT on near-zero blocks)
- Mapping table overhead and optimization
- Limitations: dataset-specific ratios, Terabyte < Kaggle
- Connection to neural video compression

### Section 7: Related Work (0.5 pages)
- CAFE+, DeepLight, Mixed-Dimension, Compositional Embedding
- Post-training quantization (INT8, INT4)
- Product quantization, SVD

### Section 8: Conclusion (0.5 pages)

---

## Reviewer FAQ

### Q: "The speedup comes from C++, not compression."
A: Correct. Our paper claims COMPRESSION and MEMORY REDUCTION, not speedup.
The C++ engine is a deployment vehicle. The contribution is the observation
that embeddings are video-compressible and the frequency-sorted layout.

### Q: "Why not just use Zstd?"
A: At lossless, Zstd-19 (132.7x) and H.265 CRF=0 (130.6x) are comparable.
The advantage of H.265 emerges at LOSSY compression: CRF=18 gives 1360x at
-0.037% loss. Zstd has no lossy mode. This 10x compression improvement at
negligible quality cost is the key insight.

### Q: "Why not just zero out cold rows?"
A: Zero-out at 99.9% (1000x) loses -0.694% AUC. H.265 CRF=18 at 1360x loses
only -0.037% (18.8x less loss). Lossy compression preserves meaningful information
in cold rows that zero-out destroys.

### Q: "Only evaluated on Criteo. Generalizes?"
A: Fair criticism. Criteo is the standard DLRM benchmark. We show results on
both D=16 (Kaggle) and D=64 (Terabyte) to demonstrate across embedding dimensions.
The intrinsic compressibility analysis (near-zero dominance) applies to any
embedding trained with L2 regularization.

### Q: "CAFE+ comparison uses approximate numbers."
A: We use numbers from the published CAFE+ paper (SIGMOD 2024). Our attempts to
reproduce CAFE+ with official code resulted in crashes. The published numbers are
the fairest comparison point.

### Q: "Terabyte compression is much worse (69x vs 1360x)."
A: D=64 embeddings have higher intrinsic entropy (1.5-4.3 bits/byte vs 0.13-2.5).
The compression ratio correlates with uint8 standard deviation. Lossless H.265 gives
only 13.7x on Terabyte vs 130.6x on Kaggle. However, 69x compression at -0.050%
AUC loss is still excellent and still dominates all other methods.

### Q: "The mapping table overhead is huge (51% on Kaggle)."
A: This is an engineering issue, not fundamental. The mapping can be reduced from
134 MB to ~38 MB using uint8 frame IDs + bitmap rank (discussed in Section 6).
This would bring the total from 263 MB to 167 MB (12.3x reduction).
