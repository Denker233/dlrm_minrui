# On-Demand Codec Results

Date: 2026-02-25 15:14:32
Baseline AUC: 0.768613, Time: 11.92s

Config                                    AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------------
1080p_g32_q8hot                      0.768619    10.1 99.9%    22M     0M    44M    194M  10.6x  41584M    6.3ms
1080p_g32_q8hot_bitmap               0.768619    11.4 99.9%    22M     0M    44M     71M  28.9x  41822M    7.0ms
1080p_g32_q8hot_bitmap_fullcpp       0.768619     6.3  0.0%    22M     0M    44M    200M  10.3x  41516M    3.9ms
1080p_g32_q8hot_fullcpp              0.768619     5.6  0.0%    22M     0M    44M    194M  10.6x  41275M    3.4ms
480p_g64_q8hot_bitmap_fullcpp        0.768619     5.1  0.0%    22M     0M    34M    190M  10.8x  41561M    3.2ms
4K_g8_q8hot_fullcpp                  0.768619     5.3  0.0%    22M     0M    63M    214M   9.6x  41593M    3.3ms
A_baseline                           0.768613    11.9  0.0%  2061M     0M     0M   2061M   1.0x  43036M    7.4ms