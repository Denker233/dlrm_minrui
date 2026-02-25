# On-Demand Codec Results

Date: 2026-02-25 07:46:14
Baseline AUC: 0.768613, Time: 58.48s

Config                              AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------
4K_g10_disk_fp32hot            0.768613    48.3 99.9%    87M     0M    63M    151M  13.7x  40009M    7.9ms
4K_g5_disk_fp32hot             0.768613   804.9  0.0%    87M     0M    40M    127M  16.2x  39936M  465.5ms
4K_g7_disk_fp32hot             0.768613   780.2  0.0%    87M     0M    55M    143M  14.4x  39926M  451.1ms
4K_g8_disk_fp32hot             0.768613    53.0 99.9%    87M     0M    63M    151M  13.7x  39923M    8.4ms
4K_g8_disk_q8hot               0.768619    54.6 99.9%    22M     0M    63M     85M  24.2x  39820M    8.0ms
4K_g8_inmem_fp32hot            0.768613    49.5 99.9%    87M   194M    63M    344M   6.0x  40091M    7.8ms
4K_pertable1_inmem             0.768613    49.7 99.9%    87M   194M   253M    534M   3.9x  40300M    7.8ms
A_baseline                     0.768613    58.5  0.0%  2061M     0M     0M   2061M   1.0x  41290M    5.2ms