# Comprehensive Results for Paper

## CAFE+ Paper Numbers (SIGMOD 2024, Table 3/Figure 7)

Source: Zhang et al., "CAFE: Towards Compact, Adaptive, and Fast Embedding for Large-scale Recommendation Models", SIGMOD 2024.

### Criteo Kaggle (baseline AUC = 0.8025)

| Method | Compress Rate | Approx Ratio | AUC (approx) | AUC Delta |
|--------|--------------|-------------|-------------|-----------|
| Full Embedding | 1.0 | 1x | 0.8025 | 0.0% |
| CAFE | 0.1 | 10x | ~0.800 | ~-0.25% |
| CAFE | 0.01 | 100x | ~0.798 | ~-0.45% |
| CAFE | 0.001 | 1000x | ~0.795 | ~-0.75% |
| CAFE | 0.0001 | 10000x | ~0.787 | ~-1.55% |
| Hash | 0.01 | 100x | ~0.793 | ~-0.95% |
| Hash | 0.001 | 1000x | ~0.785 | ~-1.75% |
| QR | 0.1 | 10x | ~0.799 | ~-0.35% |
| QR | 0.01 | 100x | ~0.793 | ~-0.95% |
| AdaEmbed | 0.1 | 10x | ~0.800 | ~-0.25% |
| AdaEmbed | 0.01 | 100x | ~0.795 | ~-0.75% |

Note: These are approximate values read from CAFE paper figures. The paper reports
CAFE consistently outperforms Hash, QR, and AdaEmbed at all compression rates.

### Our Results vs CAFE+ (Kaggle)

| Method | Ratio | AUC Delta | Type |
|--------|-------|-----------|------|
| **H.265 CRF=18 + freq** | **1369x** | **-0.038%** | Post-training, lossy |
| **H.265 CRF=0 (tiled)** | **145x** | **-0.002%** | Post-training, lossless |
| Zstd-19 | 133x | -0.002% | Post-training, lossless |
| Prune 99% | 100x | -0.181% | Post-training |
| CAFE 1000x (paper) | 1000x | ~-0.75% | Training-time |
| CAFE 100x (paper) | 100x | ~-0.45% | Training-time |
| INT4 | 8x | -0.190% | Post-training |

**Key insight**: H.265 CRF=18+freq achieves 1369x compression with only 0.038% AUC loss.
CAFE at 1000x loses ~0.75% (20x more AUC loss at similar compression).

### Our Results vs CAFE+ (Terabyte)

| Method | Ratio | AUC Delta | Type |
|--------|-------|-----------|------|
| **H.265 CRF=0** | **13.7x** | **-0.0006%** | Post-training, lossless |
| Zstd-19 | 14.2x | -0.0006% | Post-training, lossless |
| H.265 CRF=18 (natural) | 68.6x | -0.093% | Post-training, lossy |
| Prune 99% | 100x | -0.019% | Post-training |
| INT4 | 8x | -0.094% | Post-training |

## Key Advantages of Our Approach vs CAFE+

1. **Post-training**: No retraining required. CAFE requires full retraining with modified embedding layers.
2. **Lower AUC loss**: 0.038% vs 0.75% at comparable compression.
3. **Inference speedup**: 1.77x (Kaggle), 1.44x (Terabyte) from C++ fused hot/cold dispatch.
4. **Adjustable**: CRF parameter allows smooth tradeoff between compression and accuracy.
5. **Composable**: Can be combined with any training recipe; applied after training is done.

## Key Disadvantage vs CAFE+

1. **No training-time optimization**: CAFE learns to allocate capacity where it matters.
2. **Comparison fairness**: CAFE is training-time (learns optimal allocation), ours is post-training (compresses what exists). Apples-to-oranges.
