# Terabyte: like-for-like comparison, all measured on this node

Model: `models/dlrm_terabyte_4day.pt` (D=64, 4 days, 1 epoch, 48.5M embedding rows).
Batch 2048. AUC on 20,480,000 test samples. Forward time = bottom MLP + embedding
gather + interaction + top MLP, 100 timed batches. Nothing extrapolated, nothing
taken from another machine.

## Part A - PyTorch implementation (every config gathers from its real layout)

| config | hot% | mem MB | ratio | AUC | dAUC% | mean ms | p50 | p99 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| fp32 | - | 11846.2 | 1x | 0.788249 | +0.0000 | 9.70 | 9.53 | 13.75 |
| INT8 | - | 2961.5 | 4x | 0.788266 | +0.0017 | 8.64 | 8.63 | 8.86 |
| INT4 | - | 1484.1 | 8x | 0.785858 | -0.2390 | 10.04 | 9.92 | 13.61 |
| zero (hot uint8) | 4.3 | 139.5 | 85x | 0.787740 | -0.0509 | 10.80 | 10.73 | 12.92 |
| DC freq-sort 4bit | 4.3 | 140.8 | 84x | 0.787752 | -0.0497 | 10.73 | 10.62 | 13.37 |
| DC value-sort 4bit | 4.3 | 140.8 | 84x | 0.787908 | -0.0341 | 12.68 | 12.35 | 16.78 |
| DC pca-sort 4bit | 4.3 | 140.8 | 84x | 0.787911 | -0.0338 | 12.05 | 12.03 | 12.55 |
| SSD cold (io_uring) | 4.3 | 139.5 | 85x | 0.788277 | +0.0028 | 15.11 | 15.12 | 18.26 |
| zero (hot uint8) | 2.0 | 71.5 | 166x | 0.787411 | -0.0838 | 10.80 | 10.73 | 11.85 |
| DC freq-sort 4bit | 2.0 | 72.9 | 162x | 0.787431 | -0.0818 | 12.24 | 12.09 | 14.12 |
| DC value-sort 4bit | 2.0 | 72.9 | 162x | 0.787703 | -0.0546 | 12.84 | 12.70 | 17.08 |
| DC pca-sort 4bit | 2.0 | 72.9 | 162x | 0.787704 | -0.0545 | 13.09 | 12.97 | 14.54 |
| SSD cold (io_uring) | 2.0 | 71.5 | 166x | 0.788277 | +0.0028 | 14.16 | 14.75 | 17.78 |
| zero (hot uint8) | 1.0 | 41.9 | 282x | 0.786797 | -0.1451 | 9.99 | 9.98 | 10.18 |
| DC freq-sort 4bit | 1.0 | 43.4 | 273x | 0.786835 | -0.1414 | 12.52 | 12.48 | 13.34 |
| DC value-sort 4bit | 1.0 | 43.4 | 273x | 0.787303 | -0.0946 | 12.34 | 12.32 | 12.95 |
| DC pca-sort 4bit | 1.0 | 43.4 | 273x | 0.787301 | -0.0947 | 12.99 | 12.79 | 16.56 |
| SSD cold (io_uring) | 1.0 | 41.9 | 282x | 0.788278 | +0.0029 | 14.36 | 14.35 | 14.89 |
| zero (hot uint8) | 0.5 | 27.2 | 436x | 0.786081 | -0.2168 | 10.80 | 10.70 | 13.66 |
| DC freq-sort 4bit | 0.5 | 28.6 | 414x | 0.786141 | -0.2107 | 12.69 | 12.70 | 13.24 |
| DC value-sort 4bit | 0.5 | 28.6 | 414x | 0.786885 | -0.1364 | 12.92 | 12.89 | 13.43 |
| DC pca-sort 4bit | 0.5 | 28.6 | 414x | 0.786882 | -0.1366 | 12.44 | 12.39 | 13.25 |
| SSD cold (io_uring) | 0.5 | 27.2 | 436x | 0.788278 | +0.0029 | 16.22 | 17.05 | 17.62 |

### What Part A shows

1. **INT8 is nearly free and genuinely fast.** +0.0017% AUC (within noise of lossless) at
   4x, and 8.64 ms vs fp32's 9.70 -- *faster* than the baseline, because uint8 rows are 4x
   less memory traffic and the dequantise is cheap. This is the baseline to beat.
2. **INT4 costs real accuracy**: -0.2390% at 8x, and it is slower than fp32 (10.04 ms)
   because of nibble unpacking.
3. **DC reaches far higher ratios**: 84x at 4.3% hot for -0.034%, up to 414x at 0.5% hot
   for -0.136%.
4. **value-sort and pca-sort are indistinguishable on Terabyte** (-0.0341 vs -0.0338 at
   4.3%; -0.1364 vs -0.1366 at 0.5%). The 17-26% PCA advantage the README reports on
   Kaggle does NOT replicate at D=64. Worth stating as a negative result.
5. **DC's block means beat zeroing** consistently but modestly: at 0.5% hot, -0.136%
   (value-sort) vs -0.217% (zero).
6. **SSD is lossless** (+0.003% = noise) at every hot fraction, as it must be -- it stores
   exact fp32 cold rows -- but it is the slowest path, 14.2-16.2 ms.
7. **In PyTorch, DC is SLOWER than fp32** (12.3-13.1 vs 9.70 ms). See Part B: this is an
   artefact of the implementation, not of the method.

## Part B - C-fused implementation (`libembfwd.so`)

Same model, data, batch size and forward structure. The embedding gather for **every**
config runs through fused C -- including fp32, so the baseline is not handicapped
relative to the compressed paths. AUC here is on 4,096,000 samples (Part A used
20,480,000), so compare deltas within a part, not absolute AUC across parts.

| config | hot% | mem MB | ratio | AUC | dAUC% | mean ms | p99 |
|---|--:|--:|--:|--:|--:|--:|--:|
| fp32 (C fused) | - | 11846.2 | 1x | 0.789235 | +0.0000 | 11.10 | 14.12 |
| INT8 (C fused) | - | 2961.5 | 4x | 0.789252 | +0.0017 | 7.40 | 11.47 |
| DC value-sort (C fused) | 4.3 | 140.8 | 84x | 0.788960 | -0.0275 | 8.17 | 10.07 |
| SSD cold (C fused) | 4.3 | 139.5 | 85x | 0.789261 | +0.0026 | 9.03 | 10.73 |
| DC value-sort (C fused) | 2.0 | 72.9 | 162x | 0.788746 | -0.0489 | 8.35 | 10.00 |
| SSD cold (C fused) | 2.0 | 71.5 | 166x | 0.789261 | +0.0027 | 9.40 | 11.42 |
| DC value-sort (C fused) | 1.0 | 43.4 | 273x | 0.788361 | -0.0874 | 7.41 | 8.72 |
| SSD cold (C fused) | 1.0 | 41.9 | 282x | 0.789262 | +0.0027 | 9.35 | 10.87 |
| DC value-sort (C fused) | 0.5 | 28.6 | 414x | 0.787966 | -0.1268 | 7.42 | 11.36 |
| SSD cold (C fused) | 0.5 | 27.2 | 436x | 0.789262 | +0.0027 | 11.47 | 14.61 |

## Part C - what fusing in C actually bought

| config | PyTorch | C fused | speedup |
|---|--:|--:|--:|
| fp32 | 9.70 | 11.10 | **0.87x** |
| INT8 | 8.64 | 7.40 | 1.17x |
| DC value-sort 4.3% | 12.68 | 8.17 | 1.55x |
| DC value-sort 2.0% | 12.84 | 8.35 | 1.54x |
| DC value-sort 1.0% | 12.34 | 7.41 | 1.66x |
| DC value-sort 0.5% | 12.92 | 7.42 | 1.74x |
| SSD cold 4.3% | 15.11 | 9.03 | 1.67x |
| SSD cold 2.0% | 14.16 | 9.40 | 1.51x |
| SSD cold 1.0% | 14.36 | 9.35 | 1.54x |
| SSD cold 0.5% | 16.22 | 11.47 | 1.41x |

Three findings, two of which contradict what we expected going in:

1. **fp32 got SLOWER in C (0.87x).** PyTorch's `index_select` is a well-tuned contiguous
   row gather; a naive per-row `memcpy` loop does not beat it. There is nothing to fuse in
   a path that is already one operation. So for fp32 the best implementation is PyTorch's,
   and that is the number the baseline should use.
2. **DC gains 1.55-1.74x**, as expected: its PyTorch form was ~18 tensor ops per table
   (~470 per batch) with a `nonzero()` scan and an `index_put_` per branch, each writing
   to DRAM and being read back. Fused C does one pass, one branch per row, no temporaries.
3. **SSD gains 1.41-1.67x -- nearly as much as DC.** We predicted SSD would gain much less,
   because its device I/O is already C and therefore irreducible. That was wrong: the
   Python-side planning and assembly was a larger share of the SSD path than the device
   time. The prediction was corrected by the measurement.

## Headline: best implementation of each method, this machine, batch 2048

| method | best mean ms | memory | ratio | dAUC% | extra hardware |
|---|--:|--:|--:|--:|---|
| fp32 (PyTorch) | 9.70 | 11,846 MB | 1x | - | - |
| SSD cold, optimised NVMe (C) | 9.35 | 41.9 MB | 282x | +0.003 (lossless) | **2 NVMe + 8 I/O threads** |
| INT8 (C) | **7.40** | 2,961 MB | 4x | +0.002 | - |
| **DC value-sort 4-bit, 1% hot (C)** | **7.41** | **43.4 MB** | **273x** | **-0.087** | - |

The defensible claim from these numbers:

> DC matches INT8's forward latency (7.41 vs 7.40 ms) while using **68x less memory**
> (43 MB vs 2,961 MB), at a cost of 0.087% AUC. It is 24% faster than the fp32 baseline
> and 21% faster than cold embeddings served from two optimised NVMe drives -- without
> the drives or the 8 I/O threads that path requires.

Note this is a *weaker and more honest* claim than "SSD is infeasible at 91 ms". The
optimised NVMe path is entirely viable at 9.35 ms and is lossless; DC's advantage is that
it is faster, needs no storage hardware, and frees the I/O cores.
