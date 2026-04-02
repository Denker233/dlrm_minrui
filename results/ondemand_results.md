# On-Demand Codec Results

Date: 2026-03-19 15:46:27
Baseline AUC: 0.802497, Time: 8.73s

Config                                    AUC Time(s)   Hit%    Hot   Cold    LRU   Total  Reduc     RSS    BLat
----------------------------------------------------------------------------------------------------------------
1080p_bitmap_fullcpp                 0.802057     4.0  0.0%    22M    50M    39M    123M  16.7x  41430M    2.5ms
1080p_bitmap_lru16                   0.802057     4.0  0.0%    22M    50M    31M    115M  17.9x  41660M    2.5ms
1080p_bitmap_lru20                   0.802057     3.7  0.0%    22M    50M    37M    121M  17.0x  41666M    2.3ms
1080p_bitmap_lru8                    0.802074     3.7  0.0%    22M    50M    16M    100M  20.6x  41518M    2.3ms
1080p_crf18_bitmap_fullcpp           0.802198     3.8  0.0%    22M     3M    39M     76M  27.2x  41673M    2.3ms
1080p_crf18_bitmap_lru20             0.802198     3.7  0.0%    22M     3M    37M     74M  28.0x  41673M    2.3ms
1080p_crf18_bitmap_lru20_disk        0.802198     4.1  0.0%    22M     0M    37M     71M  29.1x  41697M    2.5ms
1080p_fullcpp                        0.802057     3.9  0.0%    22M    50M    40M    241M   8.6x  41051M    2.4ms
480p_bitmap_fullcpp                  0.802057     3.8  0.0%    22M    58M    33M    125M  16.5x  41424M    2.4ms
4K_fullcpp                           0.802057     3.7  0.0%    22M    49M    63M    263M   7.8x  41337M    2.3ms
A_baseline_t32                       0.802497     7.5  0.0%  2061M     0M     0M   2061M   1.0x      0M    4.6ms
A_baseline_t80                       0.802497     8.7  0.0%  2061M     0M     0M   2061M   1.0x  42750M    5.4ms