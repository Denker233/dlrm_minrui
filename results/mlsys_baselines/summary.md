# MLSys Reviewer Experiments


## KAGGLE

### Experiment 1: Fair Baseline Decomposition (W1)

| Config | Batch (ms) | Emb (ms) | AUC |
|--------|-----------|---------|-----|
| config_A | 4.44 | 1.39 | 0.802497 |
| config_B | 2.45 | 0.27 | 0.802497 |
| config_C | 2.46 | 0.23 | 0.802496 |

- A→B (C++ benefit): 1.81x
- B→C (compression benefit): 0.99x
- A→C (total): 1.80x

### Experiment 4: CPU Cache Profiling

| Config | Lat (ms) | L1 miss% | LLC miss% | IPC |
|--------|---------|---------|----------|-----|
| config_A | 4.33 | 8.0542 | 37.7564 | 0.348 |
| config_B | 2.42 | 8.1654 | 35.3475 | 0.377 |
| config_C | 2.54 | 8.1848 | 39.5781 | 0.375 |


## TERABYTE

### Experiment 1: Fair Baseline Decomposition (W1)

| Config | Batch (ms) | Emb (ms) | AUC |
|--------|-----------|---------|-----|
| config_A | 5.28 | 1.48 | 0.768818 |
| config_B | 3.63 | 0.38 | 0.768818 |
| config_C | 3.57 | 0.36 | 0.768818 |

- A→B (C++ benefit): 1.45x
- B→C (compression benefit): 1.02x
- A→C (total): 1.48x

### Experiment 4: CPU Cache Profiling

| Config | Lat (ms) | L1 miss% | LLC miss% | IPC |
|--------|---------|---------|----------|-----|
| config_A | 5.63 | 9.0307 | 36.0533 | 0.387 |
| config_B | 3.82 | 9.2984 | 48.31 | 0.416 |
| config_C | 4.12 | 9.1083 | 31.6373 | 0.41 |

