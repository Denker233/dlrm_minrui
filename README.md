# H.265 Video Codec Compression for DLRM Embedding Tables

Post-training compression of DLRM embedding tables using H.265 video codecs. Achieves 1360x compression on cold embeddings with only 0.037% AUC loss on Kaggle — no retraining required.

## Quick Start

```bash
cd ~/expr/dlrm_minrui

# Run the Python vs C++ frame packing comparison
python3 benchmark_frame_packing.py

# Generate all figures
python3 generate_pareto_plots.py
python3 generate_interesting_figures.py
python3 generate_reordering_figures.py
```

## Prerequisites

- Python 3.10+, PyTorch 2.x (with CUDA optional)
- FFmpeg libraries: `libavcodec`, `libavformat`, `libavutil`, `libswscale`
- PyAV: `pip install av`
- The C++ extension is pre-built (`compressed_emb.cpython-310-x86_64-linux-gnu.so`). To rebuild:

```bash
python3 setup_compressed_emb.py build_ext --inplace
```

## How It Works

### Pipeline

```
Embedding Table (N, 16) fp32
  → Quantize to uint8 (global min/max)
  → Reshape into 4×4 tiles packed into 1080p/4K video frames
  → Encode with H.265 (CRF=0 lossless or CRF=18 lossy)
  → Store compressed bitstream

At inference:
  → Decode frame (PyAV / libavcodec)
  → Gather specific rows from tiled frame
  → Dequantize to fp32
```

### Hot/Cold Split

- Profile access frequencies across batches
- Top rows covering 80% of accesses → **hot** (kept in fp32)
- Remaining rows → **cold** (quantized to uint8, H.265 compressed)
- Frequency sorting: cold rows ordered by access frequency before packing into frames

### Why Video Codecs?

Embedding tables quantized to uint8 are low-entropy "images":
- Kaggle (D=16): 0.13–2.5 bits/byte entropy, 99.7–99.9% near-zero rows
- H.265's DCT handles near-zero blocks extremely well
- Random data baseline: only 1.1x compression (proves it's the data structure, not codec magic)

## File Guide

### Core Implementation

| File | Description |
|------|-------------|
| `csrc/compressed_emb.cpp` | C++ PyTorch extension (3,846 lines). Frame tiling/untiling, fused gather+quantize+tile, AVX-512 SIMD, hot/cold embedding lookup, H.265 decode via libavcodec |
| `setup_compressed_emb.py` | Build config for the C++ extension, links FFmpeg libraries |
| `prefetch_benchmark_v10_h265.py` | H.265 encode/decode pipeline with hot/cold split |
| `codec_ondemand_benchmark.py` | On-demand frame-by-frame decode with LRU cache |

### Python vs C++ Comparison

| File | Description |
|------|-------------|
| `benchmark_frame_packing.py` | **Main comparison benchmark.** Tests tiling, untiling, selective gather, fused encode, decode pipeline, memory analysis. C++ achieves 9–80x speedup on tiling, 100–1400x on selective gather |

The Python implementation uses `reshape + transpose + reshape` (which forces a memory copy due to non-contiguous layout). The C++ implementation writes tiles directly via `memcpy` in a single parallel pass with `at::parallel_for`.

**Python (reshape + transpose):**
```python
# Tile: (N, 16) → (H, W)
tiles = rows.reshape(tiles_col, tiles_row, 4, 4)
frame = tiles.transpose(0, 2, 1, 3).reshape(H, W)

# Untile: (H, W) → (N, 16)
grid = frame.reshape(tiles_col, 4, tiles_row, 4)
rows = grid.transpose(0, 2, 1, 3).reshape(N, 16)
```

**C++ (direct memcpy, parallel):**
```cpp
// Tile: each row r → tile at (ty, tx) in frame
at::parallel_for(0, N, 512, [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; r++) {
        int64_t ty = r / tiles_per_row;
        int64_t tx = r % tiles_per_row;
        for (int ly = 0; ly < 4; ly++)
            memcpy(dst + (ty*4+ly)*width + tx*4, src + r*16 + ly*4, 4);
    }
});
```

**Benchmark results (1080p, 129K rows):**

| Operation | Python | C++ | Speedup |
|-----------|--------|-----|---------|
| Tile (rows → frame) | 2.70 ms | 0.30 ms | 9x |
| Untile (frame → rows) | 2.92 ms | 0.04 ms | 73x |
| Gather K=100 rows | 2.95 ms | 0.003 ms | 1,122x |
| Gather K=1000 rows | 2.96 ms | 0.021 ms | 138x |
| Fused encode pipeline | 3.92 ms | 0.098 ms | 40x |
| Memory allocations | 16 MB | 2 MB | 8x less |

### Experiment Scripts

| File | Description | Runtime |
|------|-------------|---------|
| `experiment_mlsys_baselines.py` | Compare H.265 vs INT8, INT4, PQ, SVD, pruning, Zstd | ~2 hours |
| `experiment_crf_and_entropy.py` | CRF sweep + per-table entropy analysis | ~1 hour |
| `experiment_crf_cache_memory.py` | Memory breakdown + cache sizing for Kaggle & Terabyte | ~1 hour |
| `experiment_intrinsic_compressibility.py` | Why embeddings compress well: entropy, sparsity, ordering effects | ~3 hours |

All experiments require the trained model (`models/dlrm_kaggle_correct.pt`) and dataset (`~/input/`).

### Figure Generation

| File | Output Directory | Figures |
|------|-----------------|---------|
| `generate_pareto_plots.py` | `results/paper_figures/` | Pareto frontiers, CRF tradeoffs, memory breakdown, error steering, speedup attribution |
| `generate_interesting_figures.py` | `results/interesting_figures/` | Compression paradox, entropy vs compression, frame concentration, mapping optimization, deployment scaling |
| `generate_reordering_figures.py` | `results/interesting_figures/` | Reordering AUC benefit, ratio no-effect, punchline triptych, advantage scaling, Pareto shift |

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

### Frequency Sorting (Error Steering)

Reordering cold rows by access frequency before frame packing:
- Does NOT change compression ratio (~same data, just reordered)
- Reduces AUC loss by 3.9x vs random ordering (Kaggle CRF=18)
- Mechanism: most-accessed cold rows placed in first frames where H.265 quality is highest

### Runtime Memory Breakdown (Kaggle CRF=18)

| Component | Size | % |
|-----------|------|---|
| Hot embeddings (fp32) | 88.5 MB | 34% |
| Compressed cold (H.265) | 0.8 MB | 0% |
| Decoded cache (20 frames) | 39.6 MB | 15% |
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
