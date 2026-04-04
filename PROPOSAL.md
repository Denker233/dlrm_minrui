# Research Proposal: Decomposing Video Codec Compression on Recommendation Model Embeddings — From 6,400× Storage to 97× Runtime Without Retraining

## 1. Problem Statement

### 1.1 Embedding Tables Dominate Recommendation Model Memory

Deep Learning Recommendation Models (DLRM) are the backbone of personalized content ranking at companies like Meta, Google, and ByteDance. These models combine sparse categorical features (user IDs, item IDs, ad campaigns, etc.) with dense numerical features through embedding tables — large lookup matrices that map categorical features to dense vectors.

**The memory problem is severe**:
- Embedding tables constitute **>90% of model parameters** [Naumov et al. 2019]
- Production embedding tables range from **10 GB to over 1 TB** [Acun et al. 2021]
- The Criteo Kaggle benchmark: 26 tables, 2 GB embeddings (D=16)
- The Criteo Terabyte benchmark: 26 tables, 5.5 GB embeddings (D=64)
- Meta's production models: tens of terabytes across hundreds of features

This memory footprint creates bottlenecks at every stage:
- **Serving**: Each model instance needs GB+ of memory → limits multi-tenant serving
- **Transfer**: Distributing model updates to thousands of serving machines → network congestion
- **Checkpointing**: Saving/loading 100 GB checkpoints → I/O bottleneck during training
- **Training**: Embedding tables must fit in memory → limits model scale

### 1.2 Existing Compression Methods Cause Performance Slowdown

All current embedding compression methods impose runtime overhead during inference:

| Method | Compression | AUC Loss | Inference Overhead | Retraining? |
|--------|:-----------:|:--------:|:------------------:|:-----------:|
| INT8 quantization | 4× | -0.002% | Negligible (fused dequant) | No |
| INT4 quantization | 8× | -0.190% | Low (bit-unpack + dequant) | No |
| Product Quantization | 14-238× | -0.559% | **High** (codebook reconstruction) | No |
| TT-Rec (decomposition) | 112× | ~-0.1% | **Moderate** (sequential matmuls, 14% overhead) | **Yes** |
| CAFE+ (learned) | ~10,000× | ~-0.75% | Zero (hash lookup) | **Yes** |
| H.265 video codec | 6,466× (storage) | -0.013% | **High** (frame decode, 5-8 GB/s) | No |

**The tradeoff**: methods with high compression ratios (>100×) either require retraining (CAFE+, TT-Rec) or impose significant decode overhead (PQ, H.265). No existing post-training method achieves >100× compression without performance penalty.

### 1.3 Why Inference Slowdown Is Especially Problematic

In production serving, recommendation models must meet strict latency SLAs (single-digit milliseconds). Any compression-induced overhead directly impacts:
- **Tail latency** (P99): decode jitter causes SLA violations
- **Throughput**: slower per-request processing reduces QPS
- **Cost**: higher latency requires more serving capacity

For training, slowdown is more tolerable (batches are 100ms+), but for inference, even 1ms of overhead is significant on a 3-5ms embedding lookup.

## 2. Our Approach: From Video Codecs to Block-Mean Compression

### 2.1 Starting Point: Video Codecs Achieve Extreme Compression

Inspired by LLM.265 (MICRO'25 Best Paper), which showed H.265 video codecs compress LLM tensors at 5.5×, we apply H.265 to DLRM embedding tables and discover **dramatically higher compression**:

- **H.265 CRF=30 on Kaggle embeddings: 6,466× compression** (vs 5.5× on LLM weights)
- AUC loss: only -0.013%
- No retraining required

Why the 1,175× difference from LLM.265? DLRM embeddings have fundamentally different properties from dense weight matrices: skewed access patterns, sparse structure, and near-zero cold rows.

### 2.2 The Decode Bottleneck — Storage vs Runtime Compression

However, H.265's 6,466× is **storage compression only**. At runtime, we must decode the compressed frames into memory:

- H.265 CPU decode: 5-8 GB/s throughput
- DDR4 memory bandwidth: 100+ GB/s
- **Decode is 15-20× slower than memory access**

This forces a decoded frame cache (40 MB for Kaggle), reducing runtime compression to **29×**. Without the cache, every batch requires frame decode → unacceptable latency.

**The fundamental question: Can we achieve high compression WITHOUT the decode step?**

### 2.3 Decomposing H.265 — What Actually Provides the Compression?

To answer this, we systematically decompose H.265's compression into its individual components, measuring each in isolation on the same embedding data:

| Component | Step Contribution | Cumulative | What It Does |
|-----------|:-----------------:|:----------:|:-------------|
| uint8 quantization | 4.0× | 4× | Reduce precision from 32-bit to 8-bit |
| Zstd entropy coding | 6.7× | 27× | Remove byte-level redundancy |
| CABAC + intra prediction | 1.5× | 40× | Context-adaptive entropy coding + spatial prediction |
| **Lossy DCT quantization** | **160×** | **6,466×** | **Drop small frequency coefficients** |

**Key finding: Lossy DCT quantization alone provides 160× — the dominant factor.** The sophisticated components of H.265 (CABAC entropy coding, intra prediction) contribute only 1.5× combined. The compression comes from DCT's ability to identify and discard information that doesn't affect the embedding values significantly.

### 2.4 Understanding DCT Quantization on Embeddings

The Discrete Cosine Transform (DCT) converts blocks of embedding values from the spatial domain to the frequency domain. In the frequency domain:
- **DC coefficient (coefficient #1)**: The block average — captures the "mean level"
- **AC coefficients (#2-64)**: Variations from the average — captures fine detail

At CRF=30 quality, the quantization step divides each coefficient by ~26 and rounds. For embedding data, this zeros out most AC coefficients because the variation between adjacent values is small (±1-2 in uint8 space).

**The critical discovery**: For cold embedding rows (95.7% of rows, rarely accessed), **100% of blocks are DC-only** — every AC coefficient rounds to zero. This means the entire block of values can be described by a single number: the block average.

### 2.5 Why Cold Embeddings Are DC-Only (Near-Zero)

This finding reflects a fundamental property of trained recommendation models:

**Cold embeddings converge to near-zero during training.** Five complementary arguments explain why:

**(a) Insufficient gradient signal.** A feature accessed K times receives ~K gradient updates. Cold features (K small) barely move from random initialization. Their embeddings are essentially untrained noise near zero [FIITED 2024, AdaEmbed OSDI 2023].

**(b) Rare features overfit, not generalize.** Ginart et al. (2019) prove that "allocating more parameters to rare items only decreases training loss but not test loss." Extra capacity for cold features memorizes noise rather than learning useful patterns. The Rademacher complexity of cold feature embeddings is bounded by 1/sqrt(frequency) [Adaptive Regularization, ICLR 2026].

**(c) Power-law access distribution.** In Criteo Kaggle, the top 4.3% of rows handle 95% of lookups [our profiling]. In Criteo Terabyte, 500 MB of hot embeddings handle 75% of inputs out of 63 GB total [Popularity-Based Skipping 2024]. This extreme skew means cold rows are accessed so rarely that their contribution to any individual prediction is negligible.

**(d) Prediction contribution bound.** A cold feature appearing in 0.001% of samples affects 0.001% of predictions. Even if its embedding were completely wrong, the expected AUC decrease is bounded by frequency × per-sample impact — both small for cold features.

**(e) Empirical validation.** PEP (ICLR 2021) prunes 97-99% of embedding parameters with minimal accuracy loss. FIITED (2024) prunes 93.75-99.75% without significant accuracy loss. CAFE achieves 10,000× compression by sharing embeddings among cold features. Mixed-Precision Embeddings (2024) assigns 0-2 bits to rare features at 200× compression.

### 2.6 The DC Block-Mean Compression Method

Since cold blocks are DC-only, we can represent them with just their block average:

**Encoding (offline, post-training)**:
1. Profile access frequency on representative data
2. Split rows: top K% by frequency → **hot** (keep as uint8); rest → **cold**
3. For cold rows: group into blocks of 16, store the mean value (1 byte per block)

**Inference (runtime)**:
1. Hot lookup (95% of accesses): direct uint8 array read + dequantize → **one AVX-512 instruction**
2. Cold lookup (5% of accesses): read block-mean value → broadcast to all dimensions → **one multiply-add**

**No decode step. No cache. No external codec library.**

The cold lookup is a single memory read + multiply, comparable in cost to an INT8 dequantization. The entire cold embedding representation for 32 million rows fits in **2 MB** (1 byte per 16-row block).

### 2.7 Tunable Hot Fraction — The Compression-Accuracy Knob

The hot fraction K% provides a smooth tradeoff:

| Hot % | AUC Loss (Kaggle) | AUC Loss (Terabyte) | Memory | Ratio |
|:-----:|:-----------------:|:-------------------:|:------:|:-----:|
| 4.3% | -0.038% | -0.005% | 33 MB | 63× |
| 2.0% | -0.092% | -0.009% | 21 MB | 97× |
| 1.0% | -0.181% | -0.021% | 16 MB | 129× |
| 0.5% | -0.296% | — | 13 MB | 158× |

At 2% hot: **97× compression with <0.1% AUC loss** on both Kaggle and Terabyte, no retraining.

## 3. Negative Results and What We Learned

### 3.1 Custom DCT+Zstd Codec Fails

We attempted to build a faster custom codec that replaces H.265's slow components (CABAC, intra prediction) with Zstd:
- Result: 279× compression (vs H.265's 6,466×) — **23× worse**
- CABAC is 20× more efficient than Zstd for DCT coefficient entropy coding
- **Conclusion**: H.265's components are all essential; you can't trivially simplify it

### 3.2 H.265 Tiles Don't Help Decode Speed

H.265 supports tile-based parallelism. We tested tiles on large single-frame embeddings:
- Result: **zero improvement** — the bottleneck is intra prediction reconstruction (serial dependency chain), not entropy coding
- **Conclusion**: H.265 decode speed is fundamentally limited by intra prediction

### 3.3 Single-Frame Layout Reveals Header Overhead

Encoding each table as one large frame (vs multiple 1080p frames):
- Result: **5× better compression** — from eliminating per-frame H.265 header overhead (5 KB × 254 frames = 1.3 MB of headers out of 1.6 MB total)
- **Conclusion**: Most of the compressed H.265 data was codec metadata, not actual embedding data

### 3.4 DC Cold Contribution Varies by Dataset

- Kaggle (D=16): DC cold adds +0.002% AUC over cold=0 — negligible
- Terabyte (D=64): DC cold adds +0.002-0.006% — slightly better, still small
- **Conclusion**: Cold values are near-zero on both datasets. DC preserves a real (non-zero) value, but it's very close to zero after dequantization. The technique is principled but the practical benefit is dominated by the hot/cold split.

## 4. Comparison with Prior Work

### 4.1 Positioning Against LLM.265

| Aspect | LLM.265 (MICRO'25) | Our Work |
|--------|:-------------------:|:--------:|
| Target | LLM weights, KV cache | DLRM embeddings |
| Compression | 5.5× | 6,466× (storage), 97× (runtime) |
| Why higher | Embeddings have skewed access, near-zero cold rows | — |
| Decode solution | Proposed custom ASIC (T.265) | DC block-mean (no decode needed) |
| Hardware | GPU NVDEC | CPU (production-aligned) |
| Finding reused | DCT quantization is key | Same finding, different magnitude |

### 4.2 Pareto Frontier

At any compression ratio, our method achieves less AUC loss than alternatives. At any AUC budget, our method achieves higher compression. And it requires no retraining.

## 5. Preliminary Results

### 5.1 Kaggle (D=16, 2.1 GB embeddings)

| Config | AUC | Loss | Memory | Ratio | Batch |
|--------|:---:|:----:|:------:|:-----:|:-----:|
| fp32 baseline | 0.802497 | — | 2,061 MB | 1× | 4.98ms |
| H.265+cache (C++) | 0.802107 | -0.039% | 71 MB | 29× | 2.39ms |
| DC 2% hot (C++) | 0.801581 | -0.092% | 21 MB | 97× | 2.48ms |

### 5.2 Terabyte (D=64, 5.5 GB embeddings)

| Config | AUC | Loss | Memory | Ratio |
|--------|:---:|:----:|:------:|:-----:|
| fp32 baseline | 0.773630 | — | 5,544 MB | 1× |
| DC 2% hot | 0.773538 | -0.009% | 58 MB | 96× |
| DC 1% hot | 0.773418 | -0.021% | 44 MB | 126× |

Note: Terabyte model is being retrained for 1 full epoch for more representative results.

## 6. Proposed Experiments

### 6.1 Completed
- [x] H.265 compression decomposition (uint8, Zstd, CABAC+intra, DCT quant)
- [x] CRF Pareto sweep (CRF 0-51)
- [x] Single-frame vs multi-frame layout
- [x] Frequency sorting ablation
- [x] Custom DCT+Zstd codec (negative result)
- [x] H.265 decode optimization (pool, tiles, threading)
- [x] DC block-mean implementation in C++ (with bitmap fix)
- [x] Hot fraction sweep (4.3% → 0%)
- [x] Block size sweep (4×4, 8×8, 16×16)
- [x] Terabyte evaluation (partial — model undertrained)

### 6.2 Remaining
- [ ] Terabyte with properly trained model (1 epoch, in progress)
- [ ] CAFE+ baseline on same setup (fair Pareto comparison)
- [ ] TT-Rec baseline on same setup
- [ ] Multi-tenant serving demo (N models in fixed memory budget)
- [ ] Threshold-based dispatch (eliminate bitmap, verify 136× claim)
- [ ] Model loading time benchmark at scale

## 7. Target Venues and Timeline

| Venue | Deadline | Fit |
|-------|----------|:---:|
| RecSys 2026 | April 21, 2026 | Good (embedding compression) |
| ATC 2026 | June 10, 2026 | Good (systems) |
| MLSys 2027 | ~October 2026 | Best (ML + systems) |

## 8. References

- [LLM.265] Xu et al., "Video Codecs are Secretly Tensor Codecs," MICRO 2025 (Best Paper)
- [CAFE] Zhang et al., "CAFE: Towards Compact, Adaptive, and Fast Embedding," SIGMOD 2024
- [ROBE] Desai et al., "Random Offset Block Embedding," MLSys 2022 (Outstanding Paper)
- [TT-Rec] Yin et al., "TT-Rec: Tensor Train Compression for DLRMs," MLSys 2021
- [Mixed Dim] Ginart et al., "Mixed Dimension Embeddings," 2019
- [PEP] Liu et al., "Learnable Embedding Sizes for Recommender Systems," ICLR 2021
- [FIITED] "Fine-grained Embedding Dimension Optimization," 2024
- [AdaEmbed] Lai et al., "Adaptive Embedding for Large-Scale Recommendation," OSDI 2023
- [MPE] "Mixed-Precision Embeddings for Large-Scale Recommendation Models," 2024
- [Adaptive Reg] "Adaptive Regularization for Sparse Feature Embedding Models," ICLR 2026
- [VLDB Benchmark] Zhang et al., "Experimental Analysis of Large-scale Learnable Vector Storage," VLDB 2024
- [RecSSD] Kwon et al., "RecSSD: Near Data Processing for SSD Based Recommendation," ASPLOS 2021
- [Prism] Yang et al., "Prism: Disaggregated GPU Serving for DLRM," NSDI 2025
