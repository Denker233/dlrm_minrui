# Research Proposal: Bypassing the Entropy Decode Bottleneck for Embedding Table Compression via DCT Decomposition Analysis

## 1. Problem Statement

### 1.1 Embedding Tables Dominate Recommendation Model Size

Deep Learning Recommendation Models (DLRM) rely on embedding tables to convert sparse categorical features (user IDs, item IDs, ad categories) into dense vectors. These tables are enormous:

- **Criteo Kaggle**: 26 tables, 33.7M rows × D=16, **2 GB** (fp32)
- **Criteo Terabyte**: 26 tables, 22.6M rows × D=64, **5.5 GB** (fp32)
- **Production (Meta)**: 100s of tables, billions of rows, **100 GB – 1 TB+**

Embedding tables constitute **>90% of model parameters** while the dense MLP layers are only tens of MB. This creates three deployment bottlenecks:

1. **Memory**: serving a single model requires GB-scale DRAM for embeddings alone. Multi-tenant serving (hundreds of models per machine) is memory-bound.
2. **Storage/Transfer**: checkpointing, model distribution to serving fleet (1000+ servers), and cross-datacenter replication transfer enormous data volumes.
3. **Loading latency**: cold-start model loading from SSD takes seconds to minutes for large models.

### 1.2 Existing Compression Methods Impose Decode Overhead

Prior embedding compression methods fall into two categories:

**Model-changing methods** (require retraining):
- CAFE/CAFE+ (SIGMOD 2024): hash-based hot/cold split, up to 10,000× compression, but ~0.75% AUC loss and requires full retraining
- TT-Rec (MLSys 2021): tensor-train decomposition, 112× compression, requires matrix multiplication per lookup (14% overhead)
- ROBE (MLSys 2022): random offset block embedding, 1000× compression, requires retraining

**Model-preserving methods** (post-training, no retraining):
- INT8/INT4 quantization: 4-8× compression, negligible overhead (fused dequant)
- Product quantization: 14-238× compression, high decode overhead (codebook lookup + reconstruction)
- Pruning: 100× compression, but sparse storage/lookup overhead

**The critical gap**: Methods achieving >100× compression either require expensive retraining (CAFE+, ROBE) or impose significant decode/reconstruction overhead at inference time (PQ, TT-Rec). No existing post-training method achieves >100× compression without decode overhead.

### 1.3 Video Codec Compression: Extreme Ratio but Decode Bottleneck

LLM.265 (MICRO 2025, Best Paper) showed that H.265 video codecs can compress ML tensors at 5.5×. We find they achieve **6,466× on DLRM embedding tables** — orders of magnitude better — because cold embedding data is highly compressible.

However, the decode bottleneck is severe:
- H.265 CABAC entropy decoding: **serial, 1 bit per cycle** — cannot use SIMD/AVX-512
- Measured decode throughput: 1-6 GB/s (vs 100+ GB/s DRAM bandwidth)
- Requires 40 MB decoded frame cache at runtime → only 29× runtime compression
- Needs FFmpeg library dependency

**LLM.265 proposed custom ASIC hardware** to solve this bottleneck. We solve it in software.

## 2. Why Cold Embeddings Can Be Aggressively Compressed

### 2.1 Power Law Access Distribution

Embedding table accesses follow an extreme Zipfian distribution:
- **Top 4.3% of rows** handle **95% of all lookups** (our measurement on Criteo Kaggle)
- Top 0.14% of rows account for 90% of gradient updates (TT-Rec, MLSys 2021)
- Top 6.8% of rows account for 76% of accesses (FAE, VLDB 2022)

This means 95.7% of embedding rows ("cold" rows) are rarely accessed during inference.

### 2.2 Four Reasons Cold Embeddings Are Near-Zero

Literature and our experiments converge on four mutually reinforcing mechanisms:

**Mechanism 1: Insufficient gradient updates.** Cold rows are updated k << K times during training. Under SGD convergence theory, the expected distance from initialization shrinks as O(1/√k). Cold rows remain near their initialization (typically near zero). EMBark (RecSys 2024) documents the "quality disparity" between frequently and rarely updated embeddings.

**Mechanism 2: Regularization decay.** L2 regularization / weight decay pushes rarely-updated parameters exponentially toward zero between updates. Adaptive regularization theory (arXiv 2025) proves that optimal regularization strength scales inversely with frequency: λ_i = μ₀/m_i. Cold features get the strongest regularization push toward zero.

**Mechanism 3: Loss dominated by frequent items.** The gradient of the loss function is dominated by hot items (which appear in most batches). Cold items contribute negligibly to the total loss. The model optimizes primarily for hot items.

**Mechanism 4: No distinguishing information.** The model sees each cold item too few times to learn anything unique about it. Multiple independent works (ROBE, CAFE, FDH) demonstrate that forcing cold features to share embeddings causes no information loss — there was no distinguishing information to lose.

### 2.3 Empirical Verification

Our experiments confirm this directly:

| Hot fraction | Cold rows | AUC loss from zeroing cold | Dataset |
|:--:|:--:|:--:|---|
| 4.3% | 32.3M | -0.005% | Kaggle (D=16) |
| 2.0% | 33.0M | -0.009% | Kaggle (D=16) |
| 4.3% | 20.8M | -0.005% | Terabyte (D=64) |
| 2.0% | 21.3M | -0.009% | Terabyte (D=64) |

Zeroing out 98% of embedding rows costs only 0.009% AUC. This validates the literature: cold embeddings carry negligible information after training.

## 3. Our Approach: From Video Codecs to DCT Decomposition

### 3.1 Starting Point: H.265 Achieves 6,466× on Embeddings

We encode cold embedding rows as H.265 video frames:
1. Quantize fp32 → uint8 (global scale/zero-point per table)
2. Tile embedding rows into grayscale video frames (4×4 tiles)
3. Encode with H.265 (CRF=30, medium preset)

Result: **6,466× compression** on Kaggle with -0.013% AUC loss. But runtime requires a 40 MB decoded frame cache (only 29× runtime compression).

### 3.2 The Journey: Decomposing WHY Codecs Work

We isolate each H.265 component's contribution by measuring compression with and without each stage:

| Component | Step contribution | Cumulative | How measured |
|---|:--:|:--:|---|
| uint8 quantization | 4.0× | 4× | fp32 / uint8 |
| Zstd entropy coding | 6.7× | 27× | uint8 / Zstd(uint8) |
| CABAC + intra prediction | 1.5× | 40× | Zstd / H.265-lossless |
| **Lossy DCT quantization** | **160×** | **6,466×** | H.265-lossless / H.265-CRF=30 |

**Key finding: lossy DCT quantization provides 160× — the dominant compression factor.** CABAC (the serial entropy decoder that bottlenecks decode speed) contributes only 1.5×.

This means: **the compression comes from a component (DCT quantization) that doesn't require serial entropy decoding. The bottleneck component (CABAC) barely contributes.**

### 3.3 What DCT Quantization Does on Embeddings

DCT (Discrete Cosine Transform) converts spatial values into frequency coefficients. Quantization rounds small coefficients to zero, keeping only significant ones.

For embedding data, we discover: **100% of cold blocks are DC-only** at CRF=30-equivalent quality. This means every block of 16 cold rows has only ONE non-zero DCT coefficient — the DC coefficient, which represents the block average.

Why? Cold embedding values cluster around the uint8 zero-point (e.g., 88 for table 2). Within each block, values vary by ±1-2. After DCT, these tiny variations become near-zero AC coefficients that the quantization step eliminates. Only the DC (average) survives.

### 3.4 The Simplification: DC = Block Average = Trivial Lookup

Since cold blocks are 100% DC-only, the "DCT-domain lookup" simplifies to:

```
For each cold row:
    block_id = row_index / 16
    output = (DC_value[block_id] × step / block_size - zero_point) × scale
```

**One multiply-add per cold lookup. No DCT. No entropy decode. No frame decode. No cache.**

This uses AVX-512: broadcast the scalar result to all 16 (or 64) dimensions in one instruction. The lookup is **10-100× faster than memory bandwidth** because it computes from a single stored byte rather than reading 16-64 bytes from a decoded cache.

### 3.5 Why Custom Codecs Can't Match H.265

We attempted to build a simpler codec (DCT + quantize + Zstd, removing CABAC and intra prediction) to get faster decode with similar compression. It failed:

- Custom DCT+Zstd: 279× compression (vs H.265's 6,466×) — 23× worse
- CABAC is 20× more efficient than Zstd for sparse DCT coefficient entropy coding
- Intra prediction contributes 12-22% additional compression at CRF=30

**This validates that H.265's components are all essential for STORAGE compression.** But for RUNTIME, our DC approach bypasses all of them.

## 4. System Design

### 4.1 Architecture

```
OFFLINE (one-time):
  1. Profile access frequency on training/validation data
  2. Select top K% rows as "hot" (K = 2-4.3%)
  3. Quantize hot rows to uint8 (per-table scale/zero-point)
  4. For cold rows: compute DC block averages (1 byte per block of 16 rows)
  5. Optionally: H.265 encode all rows for storage/transfer (6,466×)

RUNTIME (per batch):
  For each embedding index:
    if index < threshold:       // hot (simple comparison, no bitmap)
        output += dequant_avx512(hot_uint8[index])   // 3 AVX-512 instructions
    else:                       // cold
        block_id = (index - threshold) / 16
        output += DC_value[block_id] × constant      // 1 multiply + 1 AVX-512 broadcast
```

### 4.2 Memory Breakdown

| Component | Kaggle (D=16) | Terabyte (D=64) |
|---|:--:|:--:|
| Hot embeddings (uint8, 2%) | 10.3 MB | 27.6 MB |
| DC cold values (uint8, 16 rows/block) | 2.0 MB | 1.3 MB |
| Small tables (fp32) | 2.9 MB | 24.1 MB |
| Bitmap index | 6.0 MB | 6.0 MB |
| **Total** | **21.2 MB (97×)** | **59.0 MB (94×)** |
| **Without bitmap (reordered)** | **15.2 MB (136×)** | **53.0 MB (105×)** |

### 4.3 Dual-Format Storage

For maximum storage compression, we use H.265 single-frame-per-table encoding:

| Format | Use case | Compression | Decode needed? |
|---|---|:--:|:--:|
| H.265 (single-frame) | Storage, transfer, checkpoint | **6,466×** | Yes (one-time) |
| DC block-mean (uint8) | Runtime serving | **97-136×** | **No** |

Store H.265 for network transfer (312 KB for 2 GB of embeddings). Convert to DC format on first load. Serve from DC format with zero decode overhead.

## 5. Evaluation Summary

### 5.1 Compression vs AUC (Pareto Curve)

| Method | Compression | AUC Loss | Retraining | Decode overhead |
|---|:--:|:--:|:--:|:--:|
| INT8 | 4× | -0.002% | No | Negligible (fused) |
| Pruning 99% | 100× | -0.181% | No | Sparse lookup |
| TT-Rec | 112× | ~0.1% | **Yes** | Matrix multiply |
| CAFE+ | ~10,000× | ~0.75% | **Yes** | None (hash lookup) |
| H.265 + cache | 29× runtime | -0.039% | No | 40 MB cache |
| **Ours (DC 2% hot)** | **97×** | **-0.092%** | **No** | **None (1 multiply)** |

### 5.2 Cross-Dataset Generalization

| Dataset | D | Total embeddings | DC 2% hot | AUC Loss |
|---|:--:|:--:|:--:|:--:|
| Criteo Kaggle | 16 | 2.1 GB | 21.2 MB (97×) | -0.092% |
| Criteo Terabyte | 64 | 5.5 GB | 59.0 MB (94×) | -0.009% |

### 5.3 Inference Performance

| Config | Batch latency | Embedding time | Startup |
|---|:--:|:--:|:--:|
| fp32 baseline (Python) | 4.98 ms | ~3.0 ms | 0 |
| H.265 + cache (C++) | 2.39 ms | 0.27 ms | 893 ms (decode) |
| **DC 2% hot (C++)** | **2.48 ms** | **0.25 ms** | **0 ms** |

No performance degradation from DC compression. The embedding lookup (hot uint8 + DC cold) is 0.25 ms — dominated by the MLP (2.2 ms).

## 6. Key Contributions

1. **First decomposition of video codec compression on recommendation embeddings**, revealing that lossy DCT quantization provides 160× (the dominant factor) while CABAC entropy coding contributes only 1.5×.

2. **Discovery that cold embedding blocks are 100% DC-only**: trained DLRM models learn near-zero values for cold items, making block averaging a sufficient approximation. This connects DLRM training dynamics to compression theory.

3. **Software solution to the entropy decode bottleneck**: LLM.265 proposed custom hardware. We show that for embeddings, the codec can be bypassed entirely — the DC block-mean gives 97× runtime compression with zero decode overhead, using only a single AVX-512 instruction per cold lookup.

4. **97× post-training compression with <0.1% AUC loss** on two datasets (Kaggle D=16, Terabyte D=64), without retraining, without external codec dependencies, with zero startup latency.

## 7. Target Venue and Timeline

**Target: MLSys 2027** (expected deadline ~October 2026)

| Task | Timeline | Status |
|---|---|:--:|
| Kaggle experiments | Complete | ✅ |
| Terabyte experiments (D=64) | Complete (retrain pending) | ✅ |
| C++ implementation with AVX-512 | Complete | ✅ |
| H.265 decomposition analysis | Complete | ✅ |
| CAFE+ baseline (same setup) | Needed | ❌ |
| TT-Rec baseline (same setup) | Needed | ❌ |
| Multi-tenant serving demo | Needed | ❌ |
| Paper writing | Needed | ❌ |

## References

- [LLM.265] Xu et al., "Video Codecs are Secretly Tensor Codecs," MICRO 2025 (Best Paper)
- [CAFE] Zhu et al., "CAFE: Towards Compact, Adaptive, and Fast Embedding," SIGMOD 2024
- [ROBE] Desai et al., "Random Offset Block Embedding," MLSys 2022 (Outstanding Paper)
- [TT-Rec] Yin et al., "TT-Rec: Tensor Train Compression for DLRMs," MLSys 2021
- [FAE] Mahajan et al., "Accelerating Recommendation Training by Leveraging Popular Choices," VLDB 2022
- [EMBark] NVIDIA, "Embedding Optimization for Training Large-Scale DLRMs," RecSys 2024
- [Adaptive Reg] "Adaptive Regularization for Sparse Embedding Models," arXiv 2025
- [DeepLight] Deng et al., "Deep Lightweight Feature Interactions," WSDM 2021
- [Slipstream] "Efficient Training via Popularity-Based Skipping," arXiv 2024
