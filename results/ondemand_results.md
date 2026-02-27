# On-Demand Codec Results

Date: 2026-02-26 04:20:18
Baseline AUC: 0.802497, Time: 8.78s

Config                                    AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------------
1080p_bitmap_fullcpp                 0.802489     3.8  0.0%    22M     0M    44M    200M  10.3x  41455M    2.4ms
1080p_fullcpp                        0.802489     4.0  0.0%    22M     0M    44M    194M  10.6x  41275M    2.5ms
480p_bitmap_fullcpp                  0.802489     4.1  0.0%    22M     0M    34M    190M  10.8x  41534M    2.5ms
4K_fullcpp                           0.802489     3.9  0.0%    22M     0M    63M    214M   9.6x  41554M    2.4ms
A_baseline_t32                       0.802497     8.3  0.0%  2061M     0M     0M   2061M   1.0x      0M    5.2ms
A_baseline_t80                       0.802497     8.8  0.0%  2061M     0M     0M   2061M   1.0x  42984M    5.5ms