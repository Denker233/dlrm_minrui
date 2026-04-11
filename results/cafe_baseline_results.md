# CAFE Baseline Results on Criteo Kaggle

## Experiment Setup
- **Dataset**: Criteo Kaggle (45.8M samples: 39.3M train / 6.5M test)
- **Model**: DLRM with embedding_dim=16 (CAFE's implementation)
- **Training**: 1 epoch, batch_size=128, lr=0.1, SGD optimizer
- **Device**: CPU (no GPU available; CAFE uses `--cafe_use_freq True` for CPU compatibility)
- **Date**: 2026-03-27

## Results

| Method | Compress Rate | Compression | Best Test AUC (%) | AUC Loss vs Full (%) | Final Train Loss |
|--------|--------------|-------------|-------------------|---------------------|-----------------|
| Full Embedding | 1.0 | 1x | **80.125** | 0.000 | 0.4354 |
| CAFE | 0.01 | 100x | 79.998 | -0.127 | 0.4365 |
| CAFE | 0.001 | 1000x | 79.309 | -0.816 | 0.4419 |
| CAFE | 0.0001 | 10000x | 77.777 | -2.348 | 0.4541 |
| Q-R Trick | 0.01 | 100x | 79.093 | -1.032 | 0.4436 |
| Q-R Trick | 0.005 | 200x | 79.119 | -1.006 | 0.4436 |
| Hash Embedding | 0.01 | 100x | 78.865 | -1.260 | 0.4458 |
| Hash Embedding | 0.001 | 1000x | 77.050 | -3.075 | 0.4594 |
| Hash Embedding | 0.0001 | 10000x | 75.067 | -5.058 | 0.4736 |

Note: "Best Test AUC" is the peak AUC observed during training (standard practice).

## Comparison with Our DC (H.265 Video Codec) Method

| Method | Compression | AUC (%) | AUC Loss (%) |
|--------|------------|---------|-------------|
| Our Full (DLRM) | 1x | 80.261 | 0.000 |
| **Our DC CRF=18+freq** | **1360x** | **80.224** | **-0.037** |
| CAFE 1000x | 1000x | 79.309 | -0.816* |
| CAFE 10000x | 10000x | 77.777 | -2.348* |
| Hash 1000x | 1000x | 77.050 | -3.075* |

*AUC loss is relative to CAFE's own full baseline (80.125%), not ours.

## Key Takeaways

1. **DC dominates the Pareto frontier**: At 1360x compression, DC loses only 0.037% AUC.
   CAFE at 1000x loses 0.816% -- that is **22x more AUC degradation** at similar compression.

2. **CAFE degrades rapidly beyond 1000x**: At 10000x, CAFE loses 2.35% AUC.
   By contrast, our method achieves 1360x with negligible quality loss.

3. **Hash embedding is weakest**: At 1000x, hash loses 3.08% AUC (83x worse than DC).

4. **Q-R trick is limited**: Q-R cannot achieve high compression ratios (max ~200x in CAFE config).

5. **CAFE's full baseline is slightly lower than ours**: 80.125% vs 80.261%.
   This is expected since CAFE uses a slightly different DLRM architecture and training recipe.

## CAFE Hyperparameters Used
From CAFE's official `tasks/criteo.json`:
- compress_rate=0.01: threshold=100, hash_rate=0.3
- compress_rate=0.001: threshold=500, hash_rate=0.2
- compress_rate=0.0001: threshold=500, hash_rate=0.1
- All CAFE runs used `--cafe_use_freq True` (frequency-based hot detection, standard option)

## Raw Log Files
All experiment logs are at: `/home/cc/expr/CAFE/ArtifactEvaluation/board/criteo_cpu/*/stdouterr.log`
