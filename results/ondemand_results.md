# On-Demand Codec Results

Date: 2026-02-25 01:13:18
Baseline AUC: 0.768613, Time: 55.80s

Config                              AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------
4K_nopred_cache1               0.768613    57.3 99.9%    87M   194M   253M    534M   3.9x  40234M    8.8ms
4K_nopred_cache10              0.768613    47.8 99.9%    87M   194M   253M    534M   3.9x  40313M    8.1ms
4K_nopred_cache3               0.768613    51.2 99.9%    87M   194M   253M    534M   3.9x  40320M    8.5ms
4K_nopred_cache5               0.768613    50.8 99.9%    87M   194M   253M    534M   3.9x  40326M    8.3ms
4K_nopred_cache7               0.768613    49.2 99.9%    87M   194M   253M    534M   3.9x  40209M    8.2ms
A_baseline                     0.768613    55.8  0.0%  2061M     0M     0M   2061M   1.0x  41281M    4.6ms