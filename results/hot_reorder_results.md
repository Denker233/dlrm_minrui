# Hot Embedding Reordering Results (Codec Pipeline)

Date: 2026-02-26 17:16:58
Config: 1080p, full_cpp, quantize_hot, 40 threads (bitmap only for original)
Runs per experiment: 3

Method                         AUC  AvgTime   ±Std  MeanLat   P50Lat   P99Lat
-----------------------------------------------------------------------------
original_hot              0.802489    3.86s  0.18s    2.38ms    2.30ms    3.65ms (1.000x)
freq_hot                  0.802489    3.97s  0.11s    2.45ms    2.35ms    3.34ms (0.973x)
batch_affinity_hot        0.802489    4.05s  0.14s    2.50ms    2.48ms    3.75ms (0.954x)


## Delta vs Original Hot Order

### freq_hot
  AUC delta:    +0.000000
  Time delta:   +0.11s (+2.8%)
  Lat delta:    +0.07ms

### batch_affinity_hot
  AUC delta:    +0.000000
  Time delta:   +0.19s (+4.8%)
  Lat delta:    +0.12ms
