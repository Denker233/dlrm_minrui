# Benchmark v26 — Emb Timing Fix Results

Date: 2026-02-26
Model: dlrm_kaggle_correct.pt (lr=0.1, bs=128, 1 epoch)
Dataset: Kaggle/Criteo, 1599 test batches, batch_size=2048
CPU: Intel Xeon Platinum 8380, 80 cores
Threads: 40 (optimized) for compressed configs

## Summary

| Config | AUC | Time | BLat | emb | interact | mlp | Memory | Compress |
|--------|-----|------|------|-----|----------|-----|--------|----------|
| Baseline (80t) | 0.802497 | 8.78s | 5.46ms | 2.20ms | 1.49ms | 1.72ms | 2061MB | 1.0x |
| Baseline (40t) | 0.802497 | 8.34s | 5.18ms | 2.07ms | 1.40ms | 1.67ms | 2061MB | 1.0x |
| 1080p_fullcpp | 0.802489 | 4.05s | 2.50ms | 0.38ms | 0.62ms | 1.56ms | 194MB | 10.6x |
| 1080p_bitmap_fullcpp | 0.802489 | 3.82s | 2.36ms | 0.51ms | 0.57ms | 1.55ms | 200MB | 10.3x |
| 480p_bitmap_fullcpp | 0.802489 | 4.08s | 2.52ms | 0.54ms | 0.60ms | 1.60ms | 190MB | 10.8x |
| 4K_fullcpp | 0.802489 | 3.91s | 2.42ms | 0.35ms | 0.62ms | 1.56ms | 214MB | 9.6x |

## Key Findings

- **Accuracy preserved**: All compressed configs achieve AUC=0.802489 vs baseline 0.802497 (delta < 0.001%)
- **Embedding lookup 5x faster**: 0.35-0.54ms (compressed) vs 2.07-2.20ms (baseline)
- **Interact 2.4x faster**: 0.57-0.62ms (compressed) vs 1.40-1.49ms (baseline), due to stacked tensor output
- **MLP unchanged**: ~1.56ms across all configs (expected, since MLP weights are not compressed)
- **10.3-10.8x memory reduction**: 190-214MB vs 2061MB baseline
- **2.2x total speedup**: 3.82-4.08s vs 8.34-8.78s baseline

## Forward Pass Breakdown

### Baseline (80 threads)
- emb: 2.20ms (40.3%)
- interact: 1.49ms (27.3%)
- mlp: 1.72ms (31.5%)
- other: 0.05ms (0.9%)

### Best Compressed (1080p_bitmap_fullcpp, 40 threads)
- emb: 0.51ms (21.6%)
- interact: 0.57ms (24.2%)
- mlp: 1.55ms (65.7%)
- other: -0.27ms (timing overlap)

## Configuration Details

- Hot/cold split: 80% access coverage -> ~4.3% hot rows kept in fp32, ~95.7% cold rows compressed
- Cold encoding: H.265/HEVC ALL-INTRA, CRF=0 (lossless quantized), batch-affinity reordered frames
- Cold lookup: Full C++ path with pre-decoded frames registered in C++ extension
- Hot quantization: fp16 for hot embeddings
- Bitmap-rank: Succinct data structure for O(1) hot/cold discrimination (6MB)
- Cold mapping: int32 merged mapping (128.6MB) or bitmap-rank + mmap (134.6MB)
