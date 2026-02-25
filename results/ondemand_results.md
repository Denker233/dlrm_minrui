# On-Demand Codec Results

Date: 2026-02-25 15:35:43
Baseline AUC: 0.768613, Time: 7.96s

Config                                    AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------------
1080p_g32_q8hot                      0.768619    10.4 99.9%    22M     0M    44M    194M  10.6x  41520M    6.4ms
1080p_g32_q8hot_bitmap               0.768619    10.9 99.9%    22M     0M    44M     71M  28.9x  41822M    6.7ms
1080p_g32_q8hot_bitmap_fullcpp       0.768619     3.9  0.0%    22M     0M    44M    200M  10.3x  41546M    2.4ms
1080p_g32_q8hot_fullcpp              0.768619     4.1  0.0%    22M     0M    44M    194M  10.6x  41283M    2.5ms
480p_g64_q8hot_bitmap_fullcpp        0.768619     4.1  0.0%    22M     0M    34M    190M  10.8x  41571M    2.5ms
4K_g8_q8hot_fullcpp                  0.768619     4.2  0.0%    22M     0M    63M    214M   9.6x  41641M    2.6ms
A_baseline                           0.768613     8.0  0.0%  2061M     0M     0M   2061M   1.0x  42985M    4.9ms