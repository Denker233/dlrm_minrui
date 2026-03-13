# H.265 Video Codec Compression for DLRM Embedding Tables

Post-training compression of DLRM embedding tables using H.265 video codecs. Achieves 1360x compression on cold embeddings with only 0.037% AUC loss on Kaggle — no retraining required.

## Setup

### GitHub Codespaces Setup

```bash
git clone https://github.com/Denker233/dlrm_minrui
cd dlrm_minrui
chmod +x *.sh
./set_env.sh
source dlrm_env/bin/activate
export TMPDIR=$PWD/dlrm_env
./install_req.sh
```

Download and prepare the Criteo Kaggle Display Advertising dataset:

```bash
wget https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz \
  && tar -xzvf criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz \
  && mv train.txt train_original.txt \
  && mv test.txt test_original.txt
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
cd ~/dlrm_minrui
python3 setup_compressed_emb.py build_ext --inplace
```

This builds `compressed_emb.cpython-310-x86_64-linux-gnu.so` with FFmpeg + AVX-512 support.

### Dataset

Download the Kaggle Criteo dataset and place in `input/`:

```bash
cd input/
wget https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
tar -xzvf criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
cd ..
```

Or symlink from an existing location:

```bash
ln -sf /path/to/train.txt input/train.txt
ln -sf /path/to/kaggleAdDisplayChallenge_processed.npz input/kaggleAdDisplayChallenge_processed.npz
```

The `.npz` preprocessed file is created automatically on first run (~15 min) and reused on subsequent runs (~30s).

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
| Tile (rows → frame) | 2.71 ms | 0.25 ms | 11x |
| Untile (frame → rows) | 2.92 ms | 0.04 ms | 82x |
| Gather K=100 rows | 2.85 ms | 0.004 ms | 716x |
| Gather K=1,000 rows | 2.88 ms | 0.023 ms | 124x |
| Gather+dequant K=100 | 2.88 ms | 0.005 ms | 540x |
| Fused encode pipeline | 5.72 ms | 0.098 ms | 59x |
| Memory allocations | 16 MB | 2 MB | 8x less |

At 4K (518,400 rows/frame), C++ selective gather is up to 5,547x faster than Python full-untile for K=10 rows.

**Step 3: Compare Python vs C++ overhead during inference**

```bash
# First run (encodes H.265 from scratch)
python3 benchmark_python_vs_cpp_inference.py --num-batches 500 --resolution 1080p

# Subsequent runs (load existing compressed frames, skip encoding)
python3 benchmark_python_vs_cpp_inference.py --num-batches 500 --compressed-dir results/demo_crf0_1080p
```

Runs 4 experiments on the same compressed model:

| Experiment | What it tests |
|-----------|---------------|
| Baseline | Full fp32, no compression (reference) |
| Python compressed | Pure Python: PyAV in-memory H.265 decode, reshape+transpose untiling, numpy dequant |
| C++ compressed | C++ in-memory H.265 decode, fused gather+dequant from tiled frame, merged mapping |
| Full C++ fast_forward | Single C++ call per batch, all 26 tables, pre-decoded frames, zero Python loop |

## How It Works

### Pipeline Overview

```
Trained DLRM Model (26 embedding tables, 2GB fp32)
  │
  ├── Phase 1: Profile access patterns (all training batches)
  │     → count per-row access frequency
  │
  ├── Phase 2: Hot/Cold split
  │     → Hot: top rows covering 80% accesses → keep as fp32
  │     → Cold: remaining rows → sort by frequency (most accessed first)
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

Reordering doesn't change compression ratio (same data, just reordered), but it concentrates most-accessed cold rows into fewer frames, making the LRU cache much more effective. This also steers H.265 compression errors away from frequently-accessed rows (3.9x less AUC loss vs random ordering).

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

### Measured Inference Overhead (Kaggle, 1080p, 500 batches)

```
Experiment              Latency     vs Baseline    AUC
Baseline (fp32)          5.67 ms       —           0.804736
Python compressed      762.16 ms    +13,338%       0.804733
C++ compressed          14.94 ms      +163%        0.804733
Full C++ fast_forward    3.01 ms       -47%        0.804733
```

Per-operation breakdown (ms/batch):

```
Operation                    Python       C++     Speedup
Mapping + hot/cold split      1.31      0.69        1.9x
H.265 decode (memory)          —          —      (cached, rare)
Frame untiling              722.93      0.00    (fused in C++)
Row gather                    2.80      0.00    (fused in C++)
Dequantization               11.15      0.00    (fused in C++)
Fused gather+dequant           —        7.09       104x
Sum pooling                   4.61      0.00    (fused in C++)
```

The full C++ path (`fast_forward`) is actually **47% faster than the uncompressed baseline** because it uses compact hot tensors + merged int32 mapping instead of full `nn.EmbeddingBag`.

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

### Experiment Scripts

| File | Description | Runtime |
|------|-------------|---------|
| `experiment_mlsys_baselines.py` | Compare H.265 vs INT8, INT4, PQ, SVD, pruning, Zstd | ~2 hours |
| `experiment_crf_and_entropy.py` | CRF sweep + per-table entropy analysis | ~1 hour |
| `experiment_crf_cache_memory.py` | Memory breakdown + cache sizing for Kaggle & Terabyte | ~1 hour |
| `experiment_intrinsic_compressibility.py` | Why embeddings compress well: entropy, sparsity, ordering effects | ~3 hours |

### Figure Generation

| File | Output Directory | Figures |
|------|-----------------|---------|
| `generate_pareto_plots.py` | `results/paper_figures/` | Pareto frontiers, CRF tradeoffs, memory breakdown, error steering |
| `generate_interesting_figures.py` | `results/interesting_figures/` | Compression paradox, entropy vs compression, frame concentration, mapping optimization |
| `generate_reordering_figures.py` | `results/interesting_figures/` | Reordering AUC benefit, ratio no-effect, punchline triptych, Pareto shift |

## Key Results

### Headline Numbers

| Metric | Kaggle (D=16) | Terabyte (D=64) |
|--------|--------------|-----------------|
| Storage compression | 23.0x (89 MB) | 21.3x (260 MB) |
| Runtime memory | 7.8x (263 MB) | 8.6x (641 MB) |
| AUC loss (CRF=18) | -0.037% | -0.050% |
| Cold uint8 ratio | 1360x | 59x |

### Comparison with Prior Methods (Kaggle)

| Method | Ratio | AUC Loss | Retraining |
|--------|-------|----------|------------|
| INT8 | 4x | 0.002% | No |
| PQ M=2 | 32x | 0.559% | No |
| Prune 99% | 100x | 0.181% | No |
| Zstd-19+uint8 | 133x | 0.002% | No |
| CAFE+ (~1000x) | ~1000x | ~0.750% | **Yes** |
| **H.265 CRF=18+freq (ours)** | **1360x** | **0.037%** | **No** |

### Runtime Memory Breakdown (Kaggle CRF=18)

| Component | Size | % |
|-----------|------|---|
| Hot embeddings (fp32) | 88.5 MB | 34% |
| Compressed cold (H.265, in RAM) | 0.8 MB | 0% |
| Decoded frame cache (20 frames) | 39.6 MB | 15% |
| Mapping tables (int32) | 134.6 MB | 51% |
| **Total** | **263 MB** | — |

The mapping overhead dominates and can be reduced to ~7 MB with bitmap rank indexing.

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
