# On-Demand Codec Results

Date: 2026-02-25 13:35:52
Baseline AUC: 0.768613, Time: 51.77s

Config                              AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------
1080p_g32_q8hot                0.768619    47.6 99.9%    22M     0M    44M    194M  10.6x  39901M    6.3ms
1080p_g32_q8hot_bitmap         0.768619    54.3 99.9%    22M     0M    44M     71M  28.9x  40030M    7.0ms
480p_g64_q8hot_bitmap          0.768619    50.9 99.7%    22M     0M    19M     47M  44.2x  40163M    7.9ms
4K_g8_q8hot_bitmap             0.768619    55.9 100.0%    22M     0M    63M     91M  22.6x  39917M    7.1ms
4K_g8_q8hot_hash               0.768619    46.5 100.0%    22M     0M    63M    110M  18.8x  40163M    6.5ms
A_baseline                     0.768613    51.8  0.0%  2061M     0M     0M   2061M   1.0x  41310M    4.3ms