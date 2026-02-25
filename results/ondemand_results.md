# On-Demand Codec Results

Date: 2026-02-25 12:22:03
Baseline AUC: 0.768613, Time: 51.02s

Config                              AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------
1080p_g32_disk_fp32hot         0.768613    48.3 99.8%    87M     0M    44M    260M   7.9x  40016M    6.4ms
1080p_g32_disk_q8hot           0.768619    50.3 99.8%    22M     0M    44M    194M  10.6x  39976M    6.6ms
1080p_g32_disk_q8hot_hash      0.768619    49.0 99.8%    22M     0M    44M     90M  22.9x  40202M    7.2ms
4K_g8_disk_fp32hot             0.768613    56.4 99.9%    87M     0M    63M    279M   7.4x  39895M    6.8ms
4K_g8_disk_q8hot               0.768619    53.6 99.9%    22M     0M    63M    214M   9.6x  39828M    6.8ms
4K_g8_disk_q8hot_hash          0.768619    47.9 99.9%    22M     0M    63M    110M  18.8x  40202M    7.1ms
A_baseline                     0.768613    51.0  0.0%  2061M     0M     0M   2061M   1.0x  41283M    4.3ms