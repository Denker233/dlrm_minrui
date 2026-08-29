# Bypassing the Entropy Decode Bottleneck for DLRM Embedding Compression

We decompose why H.265 video codecs achieve 6,466x storage compression on DLRM embedding tables and discover that lossy DCT quantization provides 160x (the dominant factor) while CABAC entropy coding contributes only 1.5x. This reveals that cold embedding blocks are 100% DC-only — the block average is sufficient. We bypass the serial entropy decode bottleneck entirely with a **DC block-mean + value-sort + 4-bit** approach: **341x compression on Terabyte (10.5 GB → 31 MB), -0.032% AUC loss; 248x on Kaggle (2.1 GB → 8.3 MB), -0.115% AUC loss. Zero decode overhead, no retraining.**


> **⚠️ August 2026 re-measurement.** The Terabyte model these tables were built on was
> undertrained (baseline AUC 0.768820). It has been retrained for a full epoch on 4 days
> (baseline AUC **0.789235**), and every Terabyte number below has been re-measured on a
> single machine with a single code path. **See [Re-measured Results](#re-measured-results-august-2026)
> for the current numbers and four corrections to the claims in this README, and
> **[Final Results](#final-results--full-24-day-terabyte-mlperf-configuration-august-2026)**
> for the definitive 24-day MLPerf-configuration numbers.**

## Key Results

| Method | Dataset | Runtime Compression | AUC Loss | Decode | Retraining |
|---|---|:--:|:--:|:--:|:--:|
| fp32 baseline | Kaggle | 1x (2,061 MB) | — | — | — |
| H.265 + cache | Kaggle | 29x (71 MB) | -0.039% | 40 MB cache | No |
| DC freq-sort (old) | Kaggle | 72x (29 MB) | -0.036% | 0 | No |
| **DC value-sort 4-bit** | **Kaggle** | **248x (8.3 MB)** | **-0.115%** | **0** | **No** |
| fp32 baseline | Terabyte | 1x (10.5 GB) | — | — | — |
| **DC value-sort 4-bit** | **Terabyte** | **341x (31 MB)** | **-0.032%** | **0** | **No** |

### Compression Decomposition (Novel Finding)

```
uint8 quantization:        4x    
Zstd entropy coding:       6.7x  (cumulative 27x)
CABAC + intra prediction:  1.5x  (cumulative 40x)  ← modest
Lossy DCT quantization:    160x  (cumulative 6,466x) ← DOMINANT
```

### Value-Sort + 4-bit: Up to 2.5x Less AUC Loss, Validated Across Datasets

Sorting cold rows by their mean embedding value (instead of access frequency) before computing DC block means reduces AUC loss by 1.4-2.5x at the same compression ratio. Using 4-bit quantization for block means (16 levels) gives an additional ~10% ratio boost at negligible AUC cost.

**Criteo Kaggle (D=16, baseline AUC 0.80250):**

| Hot % | Method | Ratio | AUC Loss | vs Zero |
|:--:|---|:--:|:--:|:--:|
| 4.3% | DC freq-sort 8-bit | 72x | -0.036% | — |
| 4.3% | **DC value-sort 4-bit** | **74x** | **-0.017%** | 2.2x less loss than zero |
| 2.0% | DC freq-sort 8-bit | 121x | -0.087% | — |
| 2.0% | **DC value-sort 4-bit** | **129x** | **-0.039%** | 2.3x less loss than zero |
| 1.0% | DC freq-sort 8-bit | 174x | -0.173% | — |
| 1.0% | **DC value-sort 4-bit** | **189x** | **-0.072%** | 2.5x less loss than zero |
| 0.5% | DC freq-sort 8-bit | 221x | -0.282% | — |
| 0.5% | **DC value-sort 4-bit** | **248x** | **-0.115%** | 2.6x less loss than zero |

**Criteo Terabyte (D=64, baseline AUC 0.76882):**

| Hot % | Method | Ratio | AUC Loss | vs Zero |
|:--:|---|:--:|:--:|:--:|
| 4.3% | DC value-sort 4-bit | 81x | -0.001% | 1.4x less loss than zero |
| 2.0% | DC value-sort 4-bit | 150x | -0.006% | 1.4x less loss than zero |
| 1.0% | DC value-sort 4-bit | 239x | -0.014% | 1.4x less loss than zero |
| **0.5%** | **DC value-sort 4-bit** | **341x** | **-0.032%** | **1.4x less loss than zero** |

**Key observations:**
- Value-sort benefit is stronger on Kaggle (2.1-2.5x) than Terabyte (1.3-1.4x), likely because D=16 makes the block mean a tighter approximation than D=64
- 4-bit vs 8-bit DC means: negligible AUC difference on both datasets (<0.001%), but ~1% ratio improvement
- Terabyte at 0.5% hot: **341x compression with only -0.032% AUC loss** (10.5 GB → 31 MB)
- **No latency penalty**: DC value-sort serves at identical QPS to freq-sort (~28,800 QPS on Kaggle)

**Why it works:** Inspired by AV1 video codec's intra prediction — adjacent pixels are spatially correlated, so prediction is accurate. Value-sorting creates the same property for embeddings: adjacent rows have similar values, so the block mean is a much better approximation (15-28% lower within-block MSE).

**Cost:** O(N log N) one-time sort at deployment (~4 seconds for all tables). Zero inference overhead — same O(1) lookup path.

### AV1 vs H.265 Codec Comparison (New Finding)

AV1 (dav1d decoder, AVX-512) dominates H.265 across the entire Pareto frontier:

| Codec | CRF | Storage Compression | AUC Loss | Decode Speed |
|:--:|:--:|:--:|:--:|:--:|
| H.265 | 45 | 8,368x | -0.034% | 6.2 GB/s (AVX2) |
| **AV1** | **30** | **75,236x** | **-0.023%** | **6.4 GB/s (AVX-512)** |

At matched AUC (-0.035%), AV1 achieves 9x higher compression than H.265. Both achieve identical decode throughput with codec context pooling. Software H.265 decode is limited to AVX2 (0 AVX-512 instructions in libavcodec); dav1d has 2,363 AVX-512 instructions.

### Cross-Dataset Results (DC Value-Sort 4-bit)

| Dataset | D | fp32 Size | Hot % | Compressed | AUC Loss | Ratio |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Criteo Kaggle | 16 | 2.1 GB | 4.3% | 27.8 MB | -0.017% | 74x |
| Criteo Kaggle | 16 | 2.1 GB | 1.0% | 10.9 MB | -0.072% | 189x |
| Criteo Kaggle | 16 | 2.1 GB | 0.5% | 8.3 MB | -0.115% | 248x |
| Criteo Terabyte | 64 | 10.5 GB | 4.3% | 130 MB | -0.001% | 81x |
| Criteo Terabyte | 64 | 10.5 GB | 1.0% | 44 MB | -0.014% | 239x |
| **Criteo Terabyte** | **64** | **10.5 GB** | **0.5%** | **31 MB** | **-0.032%** | **341x** |


---


## Final Results — full 24-day Terabyte, MLPerf configuration (August 2026)

Model: D=64, days 0-22 train (4.19B samples) / **day 23 test** (MLPerf convention),
1 epoch, `max-ind-range=10M`, batch 2048. Trained on an A100 (16.2 h); an independent
CPU run reproduced the final test accuracy to every printed digit. Baseline AUC
**0.798717**. Full details: `results/perdim_24day_analysis.md`.

### Headline: per-dimension block means on the fully-trained model

| method | mem | ratio | ΔAUC% |
|---|--:|--:|--:|
| fp32 baseline | 13.1 GB | 1x | — |
| INT8 whole-table | 3.3 GB | 4x | −0.0005 |
| INT4 whole-table | 1.6 GB | 8x | −0.3419 |
| DC scalar block=16 @1% | 48 MB | 273x | −0.1351 |
| **DC per-dim block=256 @1%** | **53 MB** | **248x** | **−0.0596** |
| **DC per-dim block=256 @0.5%** | **36 MB** | **359x** | **−0.0917** |

Per-dim dominates scalar at every hot fraction (1.8–2.0x less loss); INT4 is dominated
on every axis; INT8 remains the strong lossless 4x baseline.

### Compression loss vs training progress (9 checkpoints, one run)

DC loss grows monotonically as cold rows accumulate real signal — scalar 2.5x and
per-dim 3.5x from 12%→100% trained — then **flattens at convergence** (it tracks model
convergence, it does not run away). The method ranking (per-dim > scalar, 2.1–3.4x) is
stable at every point. Compressibility is therefore partly a function of training
budget: report it with the training configuration attached.
Raw: `results/dc_vs_training_progress.json`.

### Serving latency across three hardware generations

| hardware | fp32 | INT8 | DC per-dim @1% | takeaway |
|---|--:|--:|--:|---|
| 2015 Xeon (Haswell, C-fused) | 9.70 ms | 7.40 | 7.41 | DC matches INT8, beats fp32 |
| 2021 EPYC 7763 (idle, 24T) | 9.48 ms | 10.97 | 10.07 | **DC at fp32 parity; INT8 loses its edge** |
| A100 80GB (GPU-resident) | 1.49 ms | 1.77 | 5.55 | fp32 fastest (HBM); **DC fits ~170 models/GPU vs 6 — 28x tenancy** |

GPU implementation validated against CPU AUC anchors to <1e-6. On GPU, DC's footprint
is dominated by its index maps (410 MB vs 19–151 MB of data) — threshold-based
dispatch (dropping the maps) is the natural next step and would multiply GPU tenancy
again. All GPU/EPYC raw data: `results/a100/`.

---

## Re-measured Results (August 2026)

Everything in this section was measured on one machine (2× Xeon E5-2670 v3, 48T, 503 GB,
2× Intel P3600 NVMe), one model (`models/dlrm_terabyte_4day.pt`, D=64, 4 days, 1 epoch),
batch 2048, full forward time (bottom MLP + embedding gather + interaction + top MLP).
Nothing is extrapolated and nothing is taken from another machine.
Raw data: `results/unified_comparison.md`, `results/perdim_analysis.md`,
`results/sort_key_analysis.md`, `results/nvme_benchmark.md`.

### Headline: per-dimension block means

| method | mem MB | ratio | ΔAUC% | fwd ms | extra hardware |
|---|--:|--:|--:|--:|---|
| fp32 (uncompressed) | 11,846 | 1x | — | 7.79 | — |
| INT8 whole-table | 2,962 | 4x | +0.0017 | 6.90 | — |
| INT4 whole-table | 1,484 | 8x | −0.2553 | 7.55 | — |
| SSD cold, optimised NVMe @1% | 41.9 | 282x | +0.0027 | 9.35 | 2 NVMe + 8 I/O threads |
| DC scalar block=16 @1% | 43.4 | 273x | −0.0874 | 6.82 | — |
| **DC per-dim block=256 @1%** | **47.7** | **249x** | **−0.0301** | **7.31** | — |
| **DC per-dim block=256 @0.5%** | **32.9** | **360x** | **−0.0455** | **7.28** | — |

**Per-dimension block means replace scalar block means as the recommended method.**
They are Pareto-dominant: at 0.5% hot they use 24% *less* memory than scalar at 1% hot
(32.9 vs 43.4 MB) **and** have 48% less AUC loss (−0.0455 vs −0.0874), at no latency cost.

Full sweep:

| hot% | variant | B/row | mem MB | ratio | ΔAUC% | fwd ms |
|--:|---|--:|--:|--:|--:|--:|
| 4.3 | scalar block=16 | 0.031 | 140.84 | 84x | −0.0275 | 8.03 |
| 4.3 | per-dim block=256 | 0.125 | 144.98 | 82x | −0.0115 | 8.09 |
| 2.0 | scalar block=16 | 0.031 | 72.91 | 162x | −0.0489 | 7.58 |
| 2.0 | per-dim block=256 | 0.125 | 77.15 | 154x | −0.0175 | 7.48 |
| 1.0 | scalar block=16 | 0.031 | 43.37 | 273x | −0.0874 | 6.82 |
| 1.0 | per-dim block=256 | 0.125 | 47.66 | 249x | −0.0301 | 7.31 |
| 0.5 | scalar block=16 | 0.031 | 28.60 | 414x | −0.1268 | 7.29 |
| 0.5 | per-dim block=256 | 0.125 | 32.91 | 360x | −0.0455 | 7.28 |

### Why per-dimension wins: the error decomposition

Scalar DC replaces a block with one number. Its squared error decomposes exactly:

```
Σ_{r∈B} Σ_d (w[r,d] − μ_B)²  =  Σ_r Σ_d (w[r,d] − μ_r)²   ← within-row, ordering-INDEPENDENT
                              +  D · Σ_r (μ_r − μ_B)²      ← between-row, the only orderable term
```

Measured on real tables, the within-row term is **99.7–100.0% of the error at D=64**
(97.2–99.6% at D=16). No scalar — per block *or* per row — can represent it. Per-dimension
means are the only variant that attacks it.

Two consequences: **sorting by row mean is exactly optimal** for the between-row term
(value-sort is the optimum, not a heuristic), and **keeping a per-row mean is pointless**
— 16x the storage to remove 0.3% of the error.

### Four corrections to the claims above

**1. "SSD cold serving is infeasible (91.3 ms)" — not supported.**
That figure is a SATA drive with serial `pread`. An optimised NVMe path (O_DIRECT 512 B
sectors, `IORING_SETUP_IOPOLL`, registered files/buffers, 8 threads NUMA-pinned, striped
across 2 drives) does the same 1,660 cold-row batch in **1.79 ms** — or **0.39 ms** with a
sector-packed layout, a 24x improvement over the naive io_uring number. End-to-end it is
**9.35 ms** vs DC's 7.14 ms. The defensible claim is: *DC is faster and needs no storage
hardware and no I/O cores*, not that SSD is unusable. See `results/nvme_benchmark.md`.

**2. "PCA sort beats value sort by 17–26%" — does not replicate.**
On Terabyte, value-sort −0.0341% vs pca-sort −0.0338% at 4.3% hot: indistinguishable.
PCA-sort was *worse* than value-sort on all 8 tables tested, Kaggle included. On Kaggle
corr(PC1, row mean) = 0.93–0.99 — the two orderings are nearly the same permutation, and
both act on a term worth <3% of the error, so a 17–26% gap has no mechanism.
**Recommendation: drop PCA-sort.** See `results/sort_key_analysis.md`.

**3. INT8 is a much stronger baseline than the prior-work table suggests.**
Measured here: **+0.0017% AUC at 4x, and 6.90 ms — faster than fp32**. DC's advantage over
INT8 is memory (62x less), not accuracy or speed. **INT4 is not a competitor**: −0.2553%
at 8x, dominated by per-dim DC on every axis (31–45x less memory *and* better AUC).
(INT4 here uses one global scale per table; grouped scales would narrow the accuracy gap
but not the memory gap.)

**4. Scalar block means barely beat zeroing the cold rows.**
`zero` −0.1451%, `scalar block mean` −0.0946% (1.5x better), `per-dim block=256` −0.0301%
(4.8x better). The scalar method is much closer to the zeroing floor than the framing
implies; per-dim moves it decisively away.

### Caveat: ~99% of the table is still at random initialisation

Median row norm is **0.00147**. DLRM initialises embeddings as `U(±√(1/n))`, which for
n=10M predicts **0.00146** for a 64-dim row. Cold rows are not "trained toward zero" —
they are *untrained*. This is why DC works so well here, and it is the main
external-validity question: with the full 24-day dataset the same `max_ind_range`-capped
10M slots receive ~6x more updates, so DC would be discarding real signal rather than
initialisation noise. Table *sizes* would not change (the largest table is already
0.09% below the 10M cap), so compression ratios are structurally unaffected — only the
embedding content is. Full 24-day validation is planned; see `SETUP_PLAN.md` §4.

---

## Setup

### GitHub Codespaces Setup

```bash
git clone -b codecs_display https://github.com/Denker233/dlrm_minrui
cd dlrm_minrui
chmod +x *.sh
./set_env.sh
source dlrm_env/bin/activate
export TMPDIR=$PWD/dlrm_env
./install_req.sh
```

Download and prepare the Criteo Kaggle Display Advertising dataset:

```bash
cd input/
wget https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz \
  && tar -xzvf criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz \
  && mv train.txt train_original.txt \
  && mv test.txt test_original.txt
cd ..
```

Split the training data (required):

```bash
python3 input/train_split.py 1
```

### Prerequisites

- Python 3.10+, PyTorch 2.x
- FFmpeg libraries: `sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev pkg-config`
- PyAV: `pip install av`
- scikit-learn, psutil, matplotlib: `pip install scikit-learn psutil matplotlib`

### Build C++ Extension

```bash
python3 setup_compressed_emb.py build_ext --inplace
```

This builds `compressed_emb.cpython-310-x86_64-linux-gnu.so` with FFmpeg + AVX-512 support.

### Trained Model

Place the trained checkpoint at `models/dlrm_kaggle_correct.pt`. To train from scratch:

```bash
python3 dlrm_s_pytorch.py --arch-sparse-feature-size=16 \
  --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
  --data-generation=dataset --data-set=kaggle \
  --raw-data-file=./input/train.txt \
  --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
  --loss-function=bce --round-targets=True \
  --mini-batch-size=128 --nepochs=1 \
  --save-model=models/dlrm_kaggle_correct.pt
```

## Running: Compression → Inference (End-to-End)

### Option A: Full E2E Demo (single script)

Walks through the entire pipeline: load model → profile → hot/cold split → quantize → tile → H.265 encode → store compressed frames in RAM → on-demand decode during inference → compare with baseline.

```bash
# Lossless compression (first run encodes from scratch)
python3 demo_e2e_pipeline.py --crf 0 --resolution 1080p

# Lossy (sweet spot: 1360x compression, 0.037% AUC loss)
python3 demo_e2e_pipeline.py --crf 18 --resolution 1080p

# Skip encoding on subsequent runs (load existing compressed frames)
python3 demo_e2e_pipeline.py --crf 18 --compressed-dir results/demo_crf18_1080p

# All options
python3 demo_e2e_pipeline.py --help
#   --crf             H.265 quality (0=lossless, 18=lossy sweet spot)
#   --resolution      1080p or 4K
#   --num-batches     Limit test batches (0=all)
#   --cache-size      LRU frame cache size (default 20)
#   --compressed-dir  Load pre-compressed frames from this dir (skip encoding)
```

### Option B: Step-by-Step (separate scripts)

> **Prerequisite:** Run the demo first (`python3 demo_e2e_pipeline.py --crf 18`) to generate profiling data in `results/hotcold/` and `results/reorder/`. The step-by-step scripts load this data and will fail without it.

**Step 1: Profile + Compress + Run on-demand inference**

```bash
python3 codec_ondemand_benchmark.py
```

This runs the full pipeline with in-memory H.265 decode, LRU frame cache, Markov prefetching, and measures AUC/latency at multiple resolutions.

**Step 2: Standalone microbenchmark (tiling operations only)**

```bash
python3 benchmark_frame_packing.py
```

Benchmarks individual operations (tiling, untiling, gather, fused encode) per frame in isolation without running the full model.

Per-frame results (1080p, 129,600 rows/frame):

| Operation | Python | C++ | Speedup |
|-----------|--------|-----|---------|
| Tile (rows → frame) | 2.71 ms | 0.06 ms | 43x |
| Untile (frame → rows) | 2.85 ms | 0.04 ms | 75x |
| Gather K=100 rows | 2.94 ms | 0.003 ms | 1,069x |
| Gather K=1,000 rows | 2.88 ms | 0.022 ms | 133x |
| Gather+dequant K=100 | 2.94 ms | 0.004 ms | 749x |
| Gather+dequant K=1,000 | 2.88 ms | 0.022 ms | 133x |
| Fused encode pipeline | 4.25 ms | 0.096 ms | 44x |
| Total per frame (encode + decode K=100) | 7.19 ms | 0.10 ms | 72x |
| Memory allocations | 16 MB | 2 MB | 8x less |

At 4K (518,400 rows/frame), C++ selective gather is up to 3,693x faster than Python full-untile for K=10 rows.

**Step 3: Compare Python vs C++ overhead during inference**

```bash
# First run (encodes H.265 from scratch)
python3 benchmark_python_vs_cpp_inference.py --num-batches 500 --resolution 1080p

# Subsequent runs (load existing compressed frames, skip encoding)
python3 benchmark_python_vs_cpp_inference.py --num-batches 500 --compressed-dir results/demo_crf0_1080p

# Pre-decode all frames instead of LRU cache (experiment 4 only)
python3 benchmark_python_vs_cpp_inference.py --num-batches 500 --cache-size 0
```

Runs 4 experiments on the same compressed model:

| Experiment | What it tests |
|-----------|---------------|
| Baseline | Full fp32, no compression (reference) |
| Python compressed | Pure Python: PyAV in-memory H.265 decode, reshape+transpose untiling, numpy dequant. **Decodes all frames every batch (no cache)** to measure raw per-operation overhead |
| C++ compressed | C++ in-memory H.265 decode, fused gather+dequant from tiled frame, merged mapping. **Decodes all frames every batch (no cache)** to measure raw per-operation overhead |
| Full C++ fast_forward | Single C++ call per batch + LRU frame cache (default 20), on-demand H.265 decode on miss |

## How It Works

### Pipeline Overview

```
Trained DLRM Model (26 embedding tables, 2GB fp32)
  │
  ├── Phase 1: Profile access patterns
  │     → count per-row access frequency
  │
  ├── Phase 2: Hot/Cold split (top 4.3% by frequency → hot)
  │     → Hot: top 4.3% of rows by access frequency → keep as fp32
  │     → Cold: remaining 95.7% → sort by frequency (most accessed first)
  │
  ├── Phase 3: Compress cold embeddings
  │     → Quantize fp32 → uint8 (global min/max per table)
  │     → Tile each 16-element row into a 4×4 pixel block
  │     → Pack tiles into 1080p video frames (129,600 rows/frame)
  │     → Encode each frame independently with H.265 (CRF=0 lossless or CRF=18 lossy)
  │     → Store all compressed frames in RAM (~0.8 MB for Kaggle)
  │
  └── Phase 4: Inference with compressed model
        → Hot lookup: direct fp32 gather from compact tensor
        → Cold lookup: decode single H.265 frame from RAM → gather rows → dequantize
        → LRU frame cache: most-accessed frames stay decoded in memory
        → Frequency reordering ensures cache hit rate >95%
```

### Why Frequency Reordering + Cache Work Together

```
Without reordering:              With frequency reordering:

Cold rows randomly spread        Cold rows sorted by frequency
across 20 frames                 across 20 frames

  frame 0: [rare, rare, ...]      frame 0: [freq=500, freq=480, ...]  ← always cached
  frame 1: [hot, rare, ...]       frame 1: [freq=200, freq=190, ...]  ← usually cached
  ...                             ...
  frame 19: [hot, rare, ...]      frame 19: [freq=0, freq=0, ...]     ← never needed

Cache hit rate: ~60%              Cache hit rate: ~95%+
Many H.265 decodes per batch      Rare H.265 decodes
```

Reordering doesn't change compression ratio (same data, just reordered), but it concentrates most-accessed cold rows into fewer frames, making the LRU cache much more effective. This also steers H.265 compression errors away from frequently-accessed rows (1.6x less AUC loss vs random ordering).

### In-Memory Compressed Storage

All compressed H.265 frames are held in RAM as raw bytes (~0.8 MB total for Kaggle). On a cache miss, the needed frame is decoded directly from these in-memory bytes — no disk I/O during inference. Each frame is encoded as a standalone ALL-INTRA H.265 bitstream (no inter-frame dependencies), so any single frame can be decoded independently without touching the others.

### Python vs C++ Implementation

The key difference is how cold embeddings are gathered from tiled video frames during inference.

**Python** must untile the entire frame to get any rows:

```python
# To get K=100 rows from a 129,600-row frame:
# Step 1: decode H.265 from in-memory bytes
container = av.open(io.BytesIO(compressed_bytes))
frame_np = next(container.decode(video=0)).to_ndarray(format='gray')

# Step 2: untile ENTIRE frame — O(N), 2MB copy
grid = frame.reshape(tiles_col, 4, tiles_per_row, 4)
all_rows = grid.transpose(0, 2, 1, 3).reshape(-1, 16)  # forces full copy

# Step 3: index K rows
selected = all_rows[row_indices]

# Step 4: dequantize (3 intermediate allocations)
result = (selected.astype(np.float32) - zp) * scale
```

**C++** decodes from memory and computes tile coordinates to read only the needed rows:

```cpp
// Decode H.265 from in-memory bytes (libavcodec, zero-copy)
torch::Tensor frame = decode_h265_frame_from_bytes(compressed_tensor);

// O(K) work, not O(N). No untile, no intermediate buffers.
at::parallel_for(0, K, 128, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
        int64_t r = indices[i];
        int64_t ty = r / tiles_per_row;  // compute tile position
        int64_t tx = r % tiles_per_row;
        for (int ly = 0; ly < 4; ly++) {
            const uint8_t* pixel = src + (ty*4+ly) * width + tx*4;
            float* out = dst + i*16 + ly*4;
            for (int lx = 0; lx < 4; lx++)
                out[lx] = (float(pixel[lx]) - zero_point) * scale;  // fused
        }
    }
});
```

**Why C++ is faster:**

| Factor | Python | C++ |
|--------|--------|-----|
| H.265 decode | PyAV (Python wrapper) | libavcodec direct (C) |
| Gather K from N rows | Untile all N (O(N)) | Read only K (O(K)) |
| Transpose in tiling | Forces full-array copy | No transpose needed |
| Dequantization | 3 passes, 5 allocations | 1 pass, 0 allocations |
| Hot forward + pooling | Separate gather, then scatter_add | Fused accumulate in one loop |
| Parallelism | Single-threaded (GIL) | `at::parallel_for` across cores |

### Measured Inference Overhead (Kaggle, 1080p)

**Cache sweep** (`benchmark_full_comparison.py`, 1599 batches, CRF=0 lossless):

With LRU frame cache, frequency reordering ensures >99% cache hit rate — H.265 decodes are rare after warmup:

```
Experiment              Latency     vs Baseline    AUC        Cache Hit
Baseline (fp32)          4.29 ms       —           0.802497      —
C++ + LRU cache=16       2.36 ms      -45%         0.802496    99.8%
C++ + LRU cache=8        2.45 ms      -43%         0.802486    92.5%
Full C++ (pre-decoded)   2.48 ms      -42%         0.802496      —
```

**Python vs C++ raw overhead** (`benchmark_python_vs_cpp_inference.py`, 500 batches, CRF=0 lossless, OS page cache dropped between experiments):

Python and C++ compressed experiments decode all frames every batch (no cache) to measure raw per-operation cost. Full C++ uses LRU cache (default 20).

```
Experiment              Latency     vs Baseline    AUC        Cache Hit
Baseline (fp32)          4.63 ms       —           0.804736      —
Python compressed       96.16 ms    +1,978%        0.803617      —
C++ compressed          59.54 ms    +1,186%        0.803617      —
Full C++ + cache=20      2.55 ms      -45%         0.803617    99.7%
```

Per-operation breakdown, Python vs C++ compressed (ms/batch, decode all frames):

```
Operation                    Python       C++
H.265 decode (in-memory)     56.15      50.16      ~1x (both use libavcodec)
Frame untile + gather        24.25        —        (fused in C++)
Dequantization                1.21        —        (fused in C++)
Fused gather+dequant           —        51.23      includes decode in fused path
Mapping + hot/cold split      1.26       0.62       2.0x
Scatter + pooling             4.18       0.37      11.3x
Total embedding              92.92      56.07       1.7x
```

H.265 decode dominates when decoding all frames every batch (~53ms, same cost for Python and C++). The C++ advantage comes from fused gather+dequant directly from tiled frames (O(K) vs Python's O(N) full untile). With LRU cache (>99% hit rate), H.265 decodes are rare and the full C++ path is **45% faster than the uncompressed baseline** because it uses compact hot tensors + merged int32 mapping instead of full `nn.EmbeddingBag`.

## File Guide

### Core Implementation

| File | Description |
|------|-------------|
| `csrc/compressed_emb.cpp` | C++ PyTorch extension (3,846 lines). Frame tiling/untiling, fused gather+quantize+tile, AVX-512 SIMD, hot/cold embedding lookup, H.265 encode/decode via libavcodec |
| `setup_compressed_emb.py` | Build config for the C++ extension, links FFmpeg libraries |
| `codec_ondemand_benchmark.py` | Full pipeline: profiling, hot/cold split, H.265 encode, in-memory on-demand decode with LRU cache + Markov prefetching |

### Demo & Benchmarks

| File | Description |
|------|-------------|
| `demo_e2e_pipeline.py` | **End-to-end demo.** Load model → profile → compress → inference → compare with baseline. Supports `--compressed-dir` to skip encoding on repeat runs |
| `benchmark_python_vs_cpp_inference.py` | **Python vs C++ during inference.** Measures per-operation overhead (H.265 decode, untiling, gather, dequant, pooling) across 4 experiments. Supports `--compressed-dir` |
| `benchmark_frame_packing.py` | **Standalone microbenchmark.** Tests tiling, untiling, selective gather, fused encode per frame in isolation (no model needed) |
| `benchmark_full_comparison.py` | **Cache sweep benchmark.** Compares LRU cache sizes (8/16/32 frames) with full pre-decode and measures cache hit rates |

### Experiment Scripts

| File | Description | Runtime |
|------|-------------|---------|
| `experiment_mlsys_baselines.py` | Compare H.265 vs INT8, INT4, PQ, SVD, pruning, Zstd | ~2 hours |
| `experiment_crf_and_entropy.py` | CRF sweep + per-table entropy analysis | ~1 hour |
| `experiment_crf_cache_memory.py` | Memory breakdown + cache sizing | ~1 hour |
| `experiment_intrinsic_compressibility.py` | Why embeddings compress well: entropy, sparsity, ordering effects | ~3 hours |

### Figure Generation

| File | Output Directory | Figures |
|------|-----------------|---------|
| `generate_pareto_plots.py` | `results/paper_figures/` | Pareto frontiers, CRF tradeoffs, memory breakdown, error steering |
| `generate_interesting_figures.py` | `results/interesting_figures/` | Compression paradox, entropy vs compression, frame concentration, mapping optimization |
| `generate_reordering_figures.py` | `results/interesting_figures/` | Reordering AUC benefit, ratio no-effect, punchline triptych, Pareto shift |

## Detailed Results

### DC Block-Mean Approach — Kaggle (D=16, baseline AUC 0.80250)

**DC value-sort 4-bit (best):**

| Hot % | AUC | AUC Loss | Memory | Ratio | vs Zero | vs Freq-sort |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 4.3% | 0.802332 | -0.017% | 27.8 MB | 74x | 2.2x less loss | 2.1x less loss |
| 2.0% | 0.802104 | -0.039% | 16.0 MB | 129x | 2.3x less loss | 2.2x less loss |
| **1.0%** | **0.801778** | **-0.072%** | **10.9 MB** | **189x** | **2.5x less loss** | **2.4x less loss** |
| 0.5% | 0.801351 | -0.115% | 8.3 MB | 248x | 2.6x less loss | — |

**DC freq-sort 8-bit (old baseline):**

| Hot % | AUC | AUC Loss | Memory | Ratio |
|:--:|:--:|:--:|:--:|:--:|
| 4.3% | 0.802135 | -0.036% | 28.8 MB | 72x |
| 2.0% | 0.801624 | -0.087% | 17.0 MB | 121x |
| 1.0% | 0.800766 | -0.173% | 11.9 MB | 174x |

### DC Block-Mean Approach — Terabyte (D=64, baseline AUC 0.76882)

> **Superseded** — baseline AUC is now 0.789235 after full-epoch retraining.
> See [Re-measured Results](#re-measured-results-august-2026).


**DC value-sort 4-bit:**

| Hot % | AUC | AUC Loss | Ratio | vs Zero | vs Freq-sort |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 4.3% | 0.768804 | -0.001% | 81x | 1.4x less loss | 1.4x less loss |
| 2.0% | 0.768759 | -0.006% | 150x | 1.4x less loss | 1.3x less loss |
| 1.0% | 0.768682 | -0.014% | 239x | 1.4x less loss | 1.4x less loss |
| **0.5%** | **0.768495** | **-0.032%** | **341x** | **1.4x less loss** | **1.4x less loss** |

**DC freq-sort 4-bit:**

| Hot % | AUC | AUC Loss | Ratio |
|:--:|:--:|:--:|:--:|
| 4.3% | 0.768799 | -0.002% | 81x |
| 2.0% | 0.768739 | -0.008% | 150x |
| 1.0% | 0.768629 | -0.019% | 239x |
| 0.5% | 0.768378 | -0.044% | 341x |

### H.265 vs AV1 Storage Compression (Kaggle)

| Codec | CRF | Storage Size | Ratio (vs fp32) | AUC Loss |
|:--:|:--:|:--:|:--:|:--:|
| H.265 | 0 (lossless) | 50.1 MB | 39x | -0.0001% |
| H.265 | 18 | 9.7 MB | 203x | -0.003% |
| H.265 | 35 | 0.25 MB | 7,913x | -0.022% |
| H.265 | 45 | 0.24 MB | 8,368x | -0.034% |
| **AV1** | **10** | **0.35 MB** | **5,647x** | **-0.009%** |
| **AV1** | **30** | **0.03 MB** | **75,236x** | **-0.023%** |
| **AV1** | **50** | **0.009 MB** | **214,122x** | **-0.036%** |

### Comparison with Prior Methods

| Method | Runtime Compression | AUC Loss | Retraining | Decode overhead |
|---|:--:|:--:|:--:|---|
| INT8 | 4x | -0.002% | No | Negligible (fused) |
| PQ (1x8) | 36x | -0.004% | No | Codebook lookup |
| Pruning 99% | 100x | -0.181% | No | Sparse lookup |
| TT-Rec | 112x | ~-0.1% | **Yes** | Matrix multiply (14%) |
| CAFE (100x, freq) | 100x | -0.127% | **Yes** | None (hash lookup) |
| CAFE (1000x, freq) | 1000x | -0.816% | **Yes** | None (hash lookup) |
| H.265 + cache | 29x | -0.039% | No | 40MB cache, 893ms startup |
| **DC value-sort 4-bit (ours, 1% hot)** | **189x** | **-0.072%** | **No** | **None (O(1) lookup)** |
| **DC value-sort 4-bit (ours, 0.5% hot)** | **248x** | **-0.115%** | **No** | **None (O(1) lookup)** |
| **DC value-sort 4-bit (Terabyte, 0.5%)** | **341x** | **-0.032%** | **No** | **None (O(1) lookup)** |

### Memory Breakdown (DC value-sort 4-bit, 1% hot, Kaggle)

| Component | Size |
|---|:--:|
| Hot embeddings (uint8) | 5.2 MB |
| DC cold values (4-bit) | 1.0 MB |
| Bitmap-rank index | 3.0 MB |
| Small tables (fp32) | 2.9 MB |
| **Total** | **~10.9 MB (189x)** |

### Memory Breakdown (DC value-sort 4-bit, 0.5% hot, Terabyte)

| Component | Size |
|---|:--:|
| Hot embeddings (uint8) | 14.5 MB |
| DC cold values (4-bit) | 8.1 MB |
| Bitmap-rank index | 8.0 MB |
| Small tables (fp32) | 0.5 MB |
| **Total** | **~31 MB (341x)** |

### SSD Cold Embedding Latency (batch_size=2048)

> **Superseded** — these are SATA numbers with serial `pread`. Optimised NVMe reaches
> 1.79 ms for the same work. See [correction 1](#four-corrections-to-the-claims-above).


Putting cold embeddings on SSD instead of DRAM is infeasible for serving:

| Method | Mean | p50 | p99 | AUC | Memory |
|---|:--:|:--:|:--:|:--:|:--:|
| fp32 DRAM | 5.3ms | 5.4ms | 6.3ms | 0.802497 | 2,061 MB |
| **SSD cold** | **91.3ms** | **90.0ms** | **104.4ms** | 0.802497 | 1,970 MB disk |
| DC PCA-sort 4-bit (1% hot) | 4.9ms | 4.8ms | 5.7ms | 0.802301 | 11 MB |

Per batch: 53,241 embedding lookups (3.25 MB), of which 1,660 are cold (0.10 MB). Despite the cold data being only 104 KB per batch, the 1,660 **random** 64-byte SSD reads cost 83.5ms — **17x slower than DRAM**. SSD bandwidth is not the bottleneck; random access latency is.

### PCA Sort: Intelligent Compression Agent Finding

> **Does not replicate** — PCA-sort was worse than value-sort on every table tested.
> See [correction 2](#four-corrections-to-the-claims-above).


A contextual bandit agent evaluated 4 sort methods with real AUC (not proxy metrics). **PCA sort** (sort cold rows by first principal component score) beats all alternatives at every hot fraction:

| Hot % | PCA sort | Value sort | Freq sort | Zero |
|:--:|:--:|:--:|:--:|:--:|
| 0.5% | **-0.053%** | -0.064% | -0.337% | -0.368% |
| 1.0% | **-0.024%** | -0.033% | -0.207% | -0.226% |
| 2.0% | **-0.012%** | -0.013% | -0.105% | -0.113% |
| 4.3% | **-0.007%** | -0.010% | -0.043% | -0.046% |

PCA sort gives 17-26% less AUC loss than value sort by sorting along the direction of maximum variance rather than the unweighted row mean. The agent also found that optimal block size varies by table (T2→bs128, T9→bs8, T23→bs4), but with only 8 tables the learned policy couldn't outperform a uniform baseline — motivating cross-dataset training.

### Single-Scalar DC vs Per-Dimension DC (PCA sort, 4-bit)

Two DC block-mean variants trade off compression ratio vs AUC quality:

- **Scalar**: each block of 16 rows → 1 scalar (mean of all 256 values). Higher compression, more AUC loss.
- **Per-dim**: each block of 16 rows → 16 values (mean per dimension). Lower compression, less AUC loss.

| Hot % | Mode | AUC Loss | Size | Ratio |
|:--:|---|:--:|:--:|:--:|
| 1.0% | **Scalar** | -0.089% | 10.9 MB | **189x** |
| 1.0% | Per-dim | -0.024% | 25.8 MB | 80x |
| 0.5% | **Scalar** | -0.143% | 8.3 MB | **248x** |
| 0.5% | Per-dim | -0.053% | 23.3 MB | 88x |
| 4.3% | Scalar | -0.020% | 27.8 MB | 74x |
| 4.3% | Per-dim | -0.007% | 42.2 MB | 49x |

The difference is entirely in DC cold storage: 1 MB (scalar) vs 16 MB (per-dim). Hot embeddings, bitmap index, and small tables are identical. Both are valid Pareto-optimal operating points — scalar for maximum compression, per-dim for minimum AUC loss.

### Why Cold Embeddings Can Be Compressed

Four reinforcing mechanisms from the literature:
1. **Insufficient gradient updates**: Cold rows updated k << K times, never converge
2. **Regularization decay**: Weight decay pushes rarely-updated params to zero (λ_i ∝ 1/frequency)
3. **Loss dominated by frequent items**: Gradient dominated by hot items
4. **No distinguishing information**: Model can't learn anything unique about items seen 0-5 times

## Results Directory

```
results/
├── FINAL_RESULTS.md              # Consolidated summary
├── comprehensive_analysis.md     # Full analysis with reviewer critique
├── paper_tables.md               # Paper-ready tables
├── paper_narrative.md            # Suggested paper structure
├── memory_analysis.md            # Runtime memory breakdown
├── paper_figures/                # 10 paper figures (PNG)
├── interesting_figures/          # 12 analysis figures (PNG)
├── crf_and_entropy/              # CRF sweep + entropy data (JSON)
├── crf_cache_memory/             # Memory breakdown data (JSON)
├── mlsys_baselines/              # Baseline comparison data (JSON)
└── intrinsic_compressibility/    # Why embeddings compress (JSON + MD)
```
