# Previous Results Summary (Pre-decoded frames in memory, width=16 frame geometry)

Date: 2026-02-24
Note: These results use pre-decoded uint8 frames in memory (NOT true on-demand decode)
Note: Frame geometry was width=16, height=4096 — codec EXPANDED data (0.7x ratio on uint8)
Note: The 2.83x "compression" is entirely from fp32→uint8 quantization, NOT from H.265

## Baselines

| Config | AUC | Time(s) | Mean Lat(ms) | p50(ms) | p99(ms) | RSS(MB) | Memory |
|---|---|---|---|---|---|---|---|
| A_baseline | 0.768613 | 54.37 | 4.53 | 4.37 | 6.74 | 44316 | 2060.7MB fp32 |
| B_hotcold | 0.768613 | 53.54 | 5.32 | 5.35 | 7.83 | 44404 | 87.4MB hot |

## C: Unordered Codec (no reorder, no prefetch)

| Config | AUC | Time(s) | Mean Lat(ms) | p50(ms) | p99(ms) | Emb Lat(ms) | Hit% | Demand/Batch | Demand Time(ms) | RSS(MB) | Compressed(MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C_unordered_cache25 | 0.768613 | 366.18 | 187.36 | 184.03 | 248.89 | 180.16 | 5.1% | 73.1 | 137777 | 45215 | 696.3 |
| C_unordered_cache50 | 0.768613 | 306.87 | 163.39 | 159.96 | 223.11 | 156.39 | 10.7% | 68.8 | 131017 | 45257 | 696.3 |
| C_unordered_cache100 | 0.768613 | 253.30 | 126.41 | 121.55 | 184.02 | 119.63 | 16.8% | 64.1 | 124290 | 45418 | 696.3 |
| C_unordered_cache200 | 0.768613 | 233.48 | 110.94 | 109.02 | 147.52 | 104.30 | 29.1% | 54.6 | 107343 | 45549 | 696.3 |

## D: Reordered Codec (batch-affinity ordering, no prefetch)

| Config | AUC | Time(s) | Mean Lat(ms) | p50(ms) | p99(ms) | Emb Lat(ms) | Hit% | Demand/Batch | Demand Time(ms) | RSS(MB) | Compressed(MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D_reordered_cache25 | 0.768613 | 60.98 | 14.59 | 13.39 | 23.66 | 9.69 | 64.2% | 2.71 | 4734 | 45338 | 696.3 |
| D_reordered_cache50 | 0.768613 | 53.24 | 10.65 | 9.84 | 16.71 | 5.97 | 93.7% | 0.48 | 947 | 45240 | 696.3 |
| D_reordered_cache100 | 0.768613 | 53.29 | 9.94 | 9.85 | 11.77 | 5.19 | 99.5% | 0.04 | 115 | 45193 | 696.3 |
| D_reordered_cache200 | 0.768613 | 55.17 | 10.32 | 10.33 | 12.42 | 5.38 | 99.5% | 0.04 | 114 | 45328 | 696.3 |

## E1: Last-Batch Predictor (reordered, cache=100)

| Config | AUC | Time(s) | Mean Lat(ms) | p50(ms) | p99(ms) | Emb Lat(ms) | Hit% | Demand/Batch | Demand Time(ms) | RSS(MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| E1_last_depth1 | 0.768613 | 53.31 | 10.29 | 10.30 | 12.56 | 5.45 | 99.5% | 0.04 | 113 | 45178 |
| E1_last_depth3 | 0.768613 | 53.21 | 10.07 | 9.92 | 11.84 | 5.28 | 99.5% | 0.04 | 112 | 45211 |

## E2: EMA Predictor (reordered, cache=100)

| Config | AUC | Time(s) | Mean Lat(ms) | p50(ms) | p99(ms) | Emb Lat(ms) | Hit% | Demand/Batch | Demand Time(ms) | RSS(MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| E2_ema_depth1 | 0.768613 | 53.98 | 10.07 | 9.93 | 11.88 | 5.27 | 99.5% | 0.04 | 112 | 45094 |
| E2_ema_depth3 | 0.768613 | 54.63 | 10.23 | 9.95 | 12.71 | 5.38 | 99.5% | 0.04 | 109 | 45187 |
| E2_ema_depth5 | 0.768613 | 52.89 | 10.14 | 10.13 | 12.36 | 5.27 | 99.5% | 0.04 | 109 | 45187 |

## E3: Markov Predictor (reordered, cache=100)

| Config | AUC | Time(s) | Mean Lat(ms) | p50(ms) | p99(ms) | Emb Lat(ms) | Hit% | Demand/Batch | Demand Time(ms) | RSS(MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| E3_markov_depth1 | 0.768613 | 53.70 | 9.99 | 9.83 | 12.07 | 5.18 | 99.7% | 0.02 | 63 | 45096 |
| E3_markov_depth3 | 0.768613 | 56.26 | 10.15 | 9.88 | 12.66 | 5.30 | 99.7% | 0.02 | 62 | 45128 |
| E3_markov_depth5 | 0.768613 | 55.39 | 10.20 | 10.18 | 12.84 | 5.33 | 99.7% | 0.02 | 61 | 45272 |

## E4: Oracle Predictor (reordered, cache=100)

| Config | AUC | Time(s) | Mean Lat(ms) | p50(ms) | p99(ms) | Emb Lat(ms) | Hit% | Demand/Batch | Demand Time(ms) | RSS(MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| E4_oracle_depth1 | 0.768613 | 53.01 | 10.18 | 10.22 | 12.07 | 5.36 | 99.9% | 0.01 | 33 | 45219 |
| E4_oracle_depth3 | 0.768613 | 53.69 | 10.03 | 9.87 | 11.80 | 5.22 | 99.9% | 0.01 | 31 | 45390 |
| E4_oracle_depth5 | 0.768613 | 53.36 | 10.23 | 10.24 | 12.06 | 5.41 | 99.9% | 0.01 | 32 | 45258 |
| E4_oracle_depth10 | 0.768613 | 54.12 | 10.01 | 9.81 | 12.19 | 5.17 | 99.9% | 0.01 | 32 | 45342 |

## E5: Markov d=3 Cache Size Sweep

| Config | AUC | Time(s) | Mean Lat(ms) | p50(ms) | p99(ms) | Emb Lat(ms) | Hit% | Demand/Batch | Demand Time(ms) | RSS(MB) | Cache Budget |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E5_markov_d3_cache25 | 0.768613 | 63.21 | 14.46 | 13.54 | 23.80 | 9.55 | 64.2% | 2.71 | 4775 | 45264 | 6.2MB |
| E5_markov_d3_cache50 | 0.768613 | 56.29 | 10.82 | 10.22 | 16.92 | 6.01 | 93.9% | 0.46 | 900 | 45360 | 12.5MB |
| E5_markov_d3_cache100 | 0.768613 | 56.25 | 10.19 | 9.89 | 12.79 | 5.35 | 99.7% | 0.02 | 61 | 45315 | 25.0MB |
| E5_markov_d3_cache200 | 0.768613 | 55.00 | 10.05 | 10.19 | 12.08 | 5.21 | 99.7% | 0.02 | 62 | 45367 | 50.0MB |
| E5_markov_d3_cache500 | 0.768613 | 55.71 | 10.06 | 9.89 | 12.49 | 5.29 | 99.7% | 0.02 | 61 | 45369 | 125.0MB |

## Sorted by Total Time

| Rank | Config | Time(s) | Hit% | Cache | Predictor | RSS(MB) |
|---|---|---|---|---|---|---|
| 1 | E2_ema_depth5 | 52.89 | 99.5% | 100 | ema d=5 | 45187 |
| 2 | E4_oracle_depth1 | 53.01 | 99.9% | 100 | oracle d=1 | 45219 |
| 3 | E1_last_depth3 | 53.21 | 99.5% | 100 | last d=3 | 45211 |
| 4 | D_reordered_cache50 | 53.24 | 93.7% | 50 | none | 45240 |
| 5 | D_reordered_cache100 | 53.29 | 99.5% | 100 | none | 45193 |
| 6 | E1_last_depth1 | 53.31 | 99.5% | 100 | last d=1 | 45178 |
| 7 | E4_oracle_depth5 | 53.36 | 99.9% | 100 | oracle d=5 | 45258 |
| 8 | B_hotcold | 53.54 | - | - | - | 44404 |
| 9 | E4_oracle_depth3 | 53.69 | 99.9% | 100 | oracle d=3 | 45390 |
| 10 | E3_markov_depth1 | 53.70 | 99.7% | 100 | markov d=1 | 45096 |
| 11 | E2_ema_depth1 | 53.98 | 99.5% | 100 | ema d=1 | 45094 |
| 12 | E4_oracle_depth10 | 54.12 | 99.9% | 100 | oracle d=10 | 45342 |
| 13 | A_baseline | 54.37 | - | - | no codec | 44316 |
| 14 | E2_ema_depth3 | 54.63 | 99.5% | 100 | ema d=3 | 45187 |
| 15 | E5_markov_d3_cache200 | 55.00 | 99.7% | 200 | markov d=3 | 45367 |
| 16 | D_reordered_cache200 | 55.17 | 99.5% | 200 | none | 45328 |
| 17 | E3_markov_depth5 | 55.39 | 99.7% | 100 | markov d=5 | 45272 |
| 18 | E5_markov_d3_cache500 | 55.71 | 99.7% | 500 | markov d=3 | 45369 |
| 19 | E5_markov_d3_cache100 | 56.25 | 99.7% | 100 | markov d=3 | 45315 |
| 20 | E3_markov_depth3 | 56.26 | 99.7% | 100 | markov d=3 | 45128 |
| 21 | E5_markov_d3_cache50 | 56.29 | 93.9% | 50 | markov d=3 | 45360 |
| 22 | D_reordered_cache25 | 60.98 | 64.2% | 25 | none | 45338 |
| 23 | E5_markov_d3_cache25 | 63.21 | 64.2% | 25 | markov d=3 | 45264 |
| 24 | C_unordered_cache200 | 233.48 | 29.1% | 200 | none | 45549 |
| 25 | C_unordered_cache100 | 253.30 | 16.8% | 100 | none | 45418 |
| 26 | C_unordered_cache50 | 306.87 | 10.7% | 50 | none | 45257 |
| 27 | C_unordered_cache25 | 366.18 | 5.1% | 25 | none | 45215 |

## Compression Details (ALL configs use same compressed data)

| Metric | Value |
|---|---|
| Original cold fp32 | 1970.4MB |
| After uint8 quantization | ~492.6MB |
| After H.265 CRF=0 (lossless) | 696.3MB |
| H.265 ratio (uint8→mp4) | 0.71x (EXPANSION) |
| Reported "compression ratio" | 2.83x (fp32→mp4, all from quantization) |
| Frame geometry | width=16, height=4096 (BAD for codec) |

## Issues Identified
1. Frame geometry width=16 is too narrow for H.265 CTU blocks (64x64) — codec expands data
2. All frames pre-decoded to uint8 in memory (~696MB) — not true on-demand
3. v10 used 1920x1080/2560x1440/3840x2160 frames with good compression ratios
