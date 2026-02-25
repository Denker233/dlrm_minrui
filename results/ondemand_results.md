# On-Demand Codec Results

Date: 2026-02-25 13:48:16
Baseline AUC: 0.768613, Time: 51.34s

Config                              AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------
1080p_g32_fp32hot_bitmap       0.768613    59.6 99.9%    87M     0M    44M    137M  15.0x  39950M    7.2ms
1080p_g32_q8hot                0.768619    45.4 99.9%    22M     0M    44M    194M  10.6x  39953M    6.1ms
1080p_g32_q8hot_bitmap         0.768619    47.0 99.9%    22M     0M    44M     71M  28.9x  40163M    6.7ms
480p_g64_q8hot_bitmap          0.768619    50.7 99.7%    22M     0M    19M     47M  44.2x  40163M    7.7ms
4K_g8_fp32hot_bitmap           0.768613    51.0 100.0%    87M     0M    63M    157M  13.1x  40034M    6.8ms
A_baseline                     0.768613    51.3  0.0%  2061M     0M     0M   2061M   1.0x  41283M    4.5ms