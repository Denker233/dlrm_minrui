# GPU DC Inference — A100 80GB, batch 2048, D=64, 24-day model

Anchors: all 5 AUC anchors reproduced the CPU 24-day values exactly (delta < 1e-6).
Timing: 20 warmup + 200 timed batches, torch.cuda.synchronize() around every wall
timestamp; CUDA events for gpu/gather stages. AUC at 2000 batches.
SSD/NVMe not involved — all configs fully GPU-resident.

| Config | Hot % | Storage MB | Idx MB | Total MB | Models/80GB | AUC | e2e mean/p50/p99 ms | gpu ms | gather ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| fp32 | — | 13050 | 0 | 13083 | 6 | 0.798717 | 1.49 / 1.48 / 1.85 | 1.25 | 0.37 |
| int8 | — | 3263 | 0 | 3295 | 24 | 0.798712 | 1.77 / 1.78 / 1.98 | 1.54 | 0.67 |
| dc_scalar_bs16 | 4.3% | 144 | 410 | 586 | 139 | 0.798090 | 4.63 / 4.62 / 4.82 | 4.39 | 3.51 |
| dc_perdim_bs256 | 4.3% | 151 | 410 | 593 | 138 | 0.798373 | 4.71 / 4.70 / 4.90 | 4.47 | 3.63 |
| dc_scalar_bs16 | 2% | 68 | 409 | 510 | 160 | 0.797892 | 4.79 / 4.79 / 4.84 | 4.55 | 3.68 |
| dc_perdim_bs256 | 2% | 75 | 410 | 517 | 158 | 0.798308 | 5.30 / 5.30 / 5.49 | 5.06 | 4.19 |
| dc_scalar_bs16 | 1% | 36 | 410 | 478 | 171 | 0.797366 | 4.97 / 4.97 / 5.07 | 4.73 | 3.86 |
| dc_perdim_bs256 | 1% | 42 | 410 | 484 | 169 | 0.798121 | 5.55 / 5.54 / 5.63 | 5.31 | 4.44 |
| dc_scalar_bs16 | 0.5% | 19 | 410 | 461 | 177 | 0.796844 | 4.98 / 4.98 / 5.04 | 4.74 | 3.87 |
| dc_perdim_bs256 | 0.5% | 25 | 411 | 469 | 174 | 0.797800 | 5.56 / 5.55 / 5.64 | 5.32 | 4.45 |
