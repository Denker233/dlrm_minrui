# On-Demand Codec Results

Date: 2026-02-25 12:00:37
Baseline AUC: 0.768613, Time: 52.06s

Config                              AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------
1080p_g32_disk_fp32hot         0.768613    57.4 99.8%    87M     0M    44M    156M  13.2x  40154M    7.5ms
1080p_g32_disk_q8hot           0.768619    59.4 99.8%    22M     0M    44M     90M  22.9x  40154M    7.8ms
4K_g8_disk_fp32hot             0.768613    49.8 99.9%    87M     0M    63M    175M  11.7x  39903M    7.4ms
4K_g8_disk_q8hot               0.768619    63.1 99.9%    22M     0M    63M    110M  18.8x  40080M    7.5ms
A_baseline                     0.768613    52.1  0.0%  2061M     0M     0M   2061M   1.0x  41279M    4.7ms