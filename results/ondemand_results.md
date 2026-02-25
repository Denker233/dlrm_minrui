# On-Demand Codec Results

Date: 2026-02-25 11:29:30
Baseline AUC: 0.768613, Time: 60.92s

Config                              AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------
1080p_g32_disk_fp32hot         0.768613    47.9 99.8%    87M     0M    44M    260M   7.9x  40010M    6.3ms
1080p_g32_disk_q8hot           0.768619    48.1 99.8%    22M     0M    44M    194M  10.6x  39940M    6.4ms
4K_g8_disk_fp32hot             0.768613    49.5 99.9%    87M     0M    63M    279M   7.4x  39795M    6.7ms
4K_g8_disk_q8hot               0.768619    47.3 99.9%    22M     0M    63M    214M   9.6x  39854M    6.3ms
A_baseline                     0.768613    60.9  0.0%  2061M     0M     0M   2061M   1.0x  41276M    5.0ms