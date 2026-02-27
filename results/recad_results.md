# Rec-AD vs Batch-Affinity Reordering Comparison

Date: 2026-02-26 07:38:03
Baseline AUC: 0.802497, Time: 7.94s

Config                               Method                AUC Time(s)  MeanLat   P99Lat   Hit%  TotalMB  Reduc
---------------------------------------------------------------------------------------------------------------
orig_1080p_bitmap_fullcpp            batch-affinity   0.802489    4.20    2.58ms    4.66ms  0.0%     200MB  10.3x
orig_1080p_fullcpp                   batch-affinity   0.802489    3.83    2.36ms    3.12ms  0.0%     194MB  10.6x
recad_1080p_bitmap_fullcpp           recad-louvain    0.802489    4.07    2.51ms    3.70ms  0.0%     200MB  10.3x
recad_1080p_fullcpp                  recad-louvain    0.802489    3.82    2.35ms    3.78ms  0.0%     194MB  10.6x


## Delta Analysis (Rec-AD vs Batch-Affinity)

### 1080p_fullcpp
  AUC delta:     +0.000000
  Time delta:    -0.01s (-0.3%)
  Mean lat delta: -0.01ms
  Hit rate delta: +0.000

### 1080p_bitmap_fullcpp
  AUC delta:     +0.000000
  Time delta:    -0.13s (-3.2%)
  Mean lat delta: -0.07ms
  Hit rate delta: +0.000
