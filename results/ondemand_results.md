# On-Demand Codec Results

Date: 2026-02-25 09:34:35
Baseline AUC: 0.768613, Time: 54.83s

Config                              AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------
1080p_g32_disk_fp32hot         0.768613    52.5 99.8%    87M     0M    44M    260M   7.9x  39884M    8.0ms
1080p_g32_disk_q8hot           0.768619    58.5 99.8%    22M     0M    44M    194M  10.6x  39880M    8.1ms
4K_g8_disk_fp32hot             0.768613    63.4 99.9%    87M     0M    63M    279M   7.4x  39876M    8.2ms
4K_g8_disk_q8hot               0.768619    63.0 99.9%    22M     0M    63M    214M   9.6x  39744M    7.9ms
A_baseline                     0.768613    54.8  0.0%  2061M     0M     0M   2061M   1.0x  41274M    4.6ms