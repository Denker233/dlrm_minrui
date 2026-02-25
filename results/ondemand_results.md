# On-Demand Codec Results

Date: 2026-02-25 14:57:09
Baseline AUC: 0.768613, Time: 50.55s

Config                                    AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------------
1080p_g32_q8hot                      0.768619    46.7 99.9%    22M     0M    44M    194M  10.6x  40133M    6.5ms
1080p_g32_q8hot_bitmap               0.768619    48.0 99.9%    22M     0M    44M     71M  28.9x  40361M    7.0ms
1080p_g32_q8hot_bitmap_fullcpp       0.768619    55.4  0.0%    22M     0M    44M    200M  10.3x  39890M    3.9ms
1080p_g32_q8hot_fullcpp              0.768619    53.3  0.0%    22M     0M    44M    194M  10.6x  39768M    3.6ms
480p_g64_q8hot_bitmap_fullcpp        0.768619    52.9  0.0%    22M     0M    34M    190M  10.8x  40171M    3.7ms
4K_g8_q8hot_fullcpp                  0.768619    49.2  0.0%    22M     0M    63M    214M   9.6x  40007M    3.7ms
A_baseline                           0.768613    50.6  0.0%  2061M     0M     0M   2061M   1.0x  41268M    4.4ms