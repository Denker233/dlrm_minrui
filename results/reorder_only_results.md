# Reordering-Only Benchmark Results

Date: 2026-02-26 16:55:24
Runs per experiment: 3

Method                         AUC  AvgTime   ±Std  MeanLat   P50Lat   P99Lat
-----------------------------------------------------------------------------
baseline                  0.802497    8.63s  0.37s    5.37ms    5.27ms    7.49ms (1.000x)
frequency_sort            0.802497    7.37s  0.62s    4.58ms    4.23ms    6.48ms (1.171x)
batch_affinity            0.802497    6.67s  0.53s    4.15ms    4.00ms    5.26ms (1.294x)
recad_louvain             0.802497    6.69s  0.37s    4.16ms    3.97ms    4.93ms (1.290x)


## Delta vs Baseline

### frequency_sort
  AUC delta:    +0.000000
  Time delta:   -1.26s (-14.6%)
  Lat delta:    -0.79ms

### batch_affinity
  AUC delta:    +0.000000
  Time delta:   -1.96s (-22.7%)
  Lat delta:    -1.22ms

### recad_louvain
  AUC delta:    +0.000000
  Time delta:   -1.94s (-22.5%)
  Lat delta:    -1.21ms
