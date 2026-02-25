# On-Demand Codec Results

Date: 2026-02-25 16:12:12
Baseline AUC: 0.768613, Time: 7.99s

Config                                    AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------------
1080p_g32_q8hot                      0.768619    10.1 99.9%    22M     0M    44M    194M  10.6x  41731M    6.3ms
1080p_g32_q8hot_bitmap               0.768619    11.0 99.9%    22M     0M    44M     71M  28.9x  41959M    6.8ms
1080p_g32_q8hot_bitmap_fullcpp       0.768619     4.3  0.0%    22M     0M    44M    200M  10.3x  41428M    2.6ms
1080p_g32_q8hot_fullcpp              0.768619     4.3  0.0%    22M     0M    44M    194M  10.6x  41277M    2.6ms
480p_g64_q8hot_bitmap_fullcpp        0.768619     4.1  0.0%    22M     0M    34M    190M  10.8x  41643M    2.5ms
4K_g8_q8hot_fullcpp                  0.768619     4.4  0.0%    22M     0M    63M    214M   9.6x  41641M    2.7ms
A_baseline                           0.768613     8.0  0.0%  2061M     0M     0M   2061M   1.0x  43037M    5.0ms