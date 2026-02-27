# Frame Packing & Codec Pipeline Optimization Results

## Problem
Copying non-contiguous embeddings into video frames for H.265 encoding is costly.
The Python pipeline requires 4-5 intermediate memory copies (gather → quantize → pad → tile → tobytes),
with numpy's `transpose()` forcing a full-frame copy on every tile/untile operation.
Additionally, the encode/decode paths use subprocess ffmpeg and PyAV which add Python overhead.

## Solution: Full C++ Codec Pipeline

### Phase 1: Fused Frame Packing (35-100x faster tiling)
9 C++ functions in `csrc/compressed_emb.cpp` that replace the Python tiling pipeline:

| Function | Description |
|----------|-------------|
| `tile_rows_to_frame` | Tile uint8 rows → 2D frame (replaces reshape+transpose+reshape) |
| `untile_frame_to_rows` | Untile 2D frame → rows (inverse) |
| `gather_from_tiled_frame` | Gather K specific rows from tiled frame (skip full untile) |
| `gather_dequant_from_tiled_frame` | Gather + dequantize in one pass |
| `fused_gather_quantize_tile` | Scattered fp32 → quantize → tiled frame (single frame) |
| `fused_quantize_tile` | Contiguous fp32 → quantize → tiled frame |
| `fused_quantize_tile_multiframe` | Tile pre-quantized uint8 across multiple frames |
| `fused_quantize_tile_multiframe_fp32` | fp32 → quantize → tile across multiple frames |
| `fused_gather_quantize_tile_multiframe` | Scattered fp32 → quantize → tile (multi-frame) |

### Phase 2: Tiled-Layout Cold Storage (46% faster cache misses)
Instead of decode → untile → store rows, keep decoded frames in tiled (H,W) layout and
use C++ `gather_dequant_from_tiled_frame` for lookups. Eliminates untile at cache population time.

### Phase 3: Direct C++ H.265 Encode/Decode via libavcodec
Replaces subprocess ffmpeg and PyAV with direct libavcodec calls from C++:

| Function | Description |
|----------|-------------|
| `decode_h265_frame_from_file` | Direct libavcodec decode from file → torch tensor |
| `decode_h265_frame_from_bytes` | Direct decode from in-memory bytes |
| `decode_h265_gather_dequant` | Fused: decode + gather + dequant in one C++ call |
| `encode_h265_frame` | Direct libx265 encode to file or memory |
| `encode_h265_frame_to_file` | Convenience: encode frame to H.265 file |
| `batch_encode_h265_frames` | Parallel multi-frame encode via std::thread |

### Phase 4: Parallel Batch Decode (2.6-4.7x faster multi-frame)
When a batch needs rows from N frames, decode all N frames in parallel threads:

| Function | Description |
|----------|-------------|
| `batch_decode_frames` | Parallel multi-frame decode via std::thread |
| `batch_decode_gather_dequant` | Parallel decode + vectorized gather + dequant |

All functions use:
- `at::parallel_for` for multi-threaded gather/tiling operations
- AVX-512 SIMD for quantization (16 fp32 values per instruction)
- Direct pointer arithmetic (no intermediate numpy/torch allocations)
- `std::thread` for parallel I/O-bound decode/encode operations

## Benchmark Results

### Single-Frame Tiling Operations (1080p, 129,600 rows)
| Operation | Python | C++ | Speedup |
|-----------|--------|-----|---------|
| Tile rows → frame | 2.68ms | 0.076ms | **35x** |
| Untile frame → rows | 2.85ms | 0.029ms | **97x** |
| Full encode (gather+quant+tile) | 5.57ms | 0.080ms | **70x** |

### Selective Gather from Tiled Frame (1080p)
| K rows gathered | Python (full untile) | C++ gather | Speedup |
|----------------|---------------------|------------|---------|
| 10 | 2.84ms | 0.003ms | **911x** |
| 100 | 2.86ms | 0.004ms | **685x** |
| 1,000 | 2.87ms | 0.014ms | **199x** |
| 10,000 | 3.11ms | 0.020ms | **158x** |

### H.265 Encode: C++ vs subprocess ffmpeg (1080p)
| Method | Single frame | 5 frames | Speedup |
|--------|-------------|----------|---------|
| subprocess ffmpeg | 230ms | 1,110ms (serial) | baseline |
| C++ encode (file) | 83ms | 215ms (parallel) | **2.8x / 5.2x** |

### H.265 Decode: Serial vs Parallel Batch (1080p)
| Frames | Serial | Parallel batch | Speedup |
|--------|--------|---------------|---------|
| 1 | 19ms | 19ms | 1.0x |
| 2 | 37ms | 19ms | **1.9x** |
| 3 | 53ms | 19ms | **2.8x** |
| 5 | 92ms | 19ms | **4.7x** |
| 8 | 143ms | 40ms | **3.6x** |
| 10 | 189ms | 48ms | **3.9x** |

### Full Decode Pipeline: Batch Decode + Gather + Dequant (1080p)
| Miss frames | Python path | C++ batch fused | Speedup |
|-------------|-------------|----------------|---------|
| 1 frame, 100 rows | 134ms | 134ms | 1.0x |
| 3 frames, 300 rows | 404ms | 146ms | **2.8x** |
| 5 frames, 1000 rows | 521ms | 141ms | **3.7x** |

### Full Encode Pipeline (1080p)
| Method | 500K rows (4 frames) | Speedup |
|--------|---------------------|---------|
| Python tile + subprocess ffmpeg | 856ms | baseline |
| C++ fused tile + parallel encode | 203ms | **4.2x** |

### Real Data: All 8 Large Tables (32.3M cold rows, 254 frames)

#### C++ Encode Performance
| Table | Cold Rows | Frames | C++ Tile+Encode | Compression |
|-------|-----------|--------|----------------|-------------|
| 2 | 9.6M | 75 | 1,886ms | 27.9x |
| 3 | 2.2M | 17 | 454ms | 6.0x |
| 11 | 8.0M | 62 | 1,269ms | 12.0x |
| 15 | 5.3M | 41 | 847ms | 10.5x |
| 20 | 6.7M | 53 | 1,126ms | 5.6x |
| **Total** | **32.3M** | **254** | **5.8s** | - |

#### Batch Decode Performance (1000 random cold rows per table)
| Table | Frames decoded | Batch decode time |
|-------|---------------|------------------|
| 2 | 75 | 59ms |
| 3 | 17 | 83ms |
| 11 | 62 | 67ms |
| 15 | 41 | 62ms |
| 20 | 53 | 92ms |

### Tiling Memory Savings (unchanged)
| Table | Python Alloc | C++ Alloc | Saved |
|-------|-------------|-----------|-------|
| 2 | 1,032 MB | 148 MB | 883 MB (86%) |
| 11 | 853 MB | 123 MB | 730 MB (86%) |
| 20 | 724 MB | 105 MB | 619 MB (86%) |
| 15 | 564 MB | 81 MB | 483 MB (86%) |
| **Total** | **3,468 MB** | **503 MB** | **2,965 MB (86%)** |

## Key Insights

### Tiling Bottleneck (Phase 1)
The Python path creates 4-5 intermediate buffers per table:
1. `w[cold_idx]` — scatter gather creates new tensor
2. `((w / s).round() + zp).clamp(0, 255).to(uint8)` — 3 intermediate fp32 tensors
3. `np.zeros(padded, ...)` — padding buffer
4. `.transpose(0, 2, 1, 3).reshape(H, W)` — non-contiguous transpose forces copy
5. `.tobytes()` — serialization copy

C++ fused path: reads scattered fp32 → quantizes with AVX-512 → writes tiled bytes directly.
**Zero intermediate allocations.**

### Tiled Storage (Phase 2)
Untiling decoded frames is wasted work when we only need a few rows.
Keeping frames in tiled (H,W) layout and using C++ `gather_from_tiled_frame`
saves the untile step. For K<100 lookups, tiled gather is as fast as row-indexed.

### H.265 Codec Dominates (Phases 3-4)
H.265 encode (~80ms/frame) and decode (~18ms/frame) dominate the pipeline.
Parallel batch decode is the biggest win: for 5 cache-miss frames, latency
drops from 92ms (serial) to 19ms (parallel) = **4.7x speedup**.
The C++ direct encode is 2.8x faster than subprocess ffmpeg per frame,
and with parallel encoding achieves **5.2x total speedup**.

## Correctness
- All C++ outputs exactly match Python (verified per-pixel)
- Quantization MAE identical to Python: ~0.000190
- Lossless H.265 encode/decode roundtrip verified: MAE = 0.0000
- Batch decode results match serial decode results

## Files Modified/Created
- `csrc/compressed_emb.cpp` — C++ extension with 18 new functions
- `setup_compressed_emb.py` — Updated to link against libavcodec/libavformat/libavutil/libswscale
- `codec_ondemand_benchmark.py` — Integrated C++ tiling, tiled storage, C++ encode/decode, batch decode
- `benchmark_frame_packing.py` — Microbenchmark for tiling operations
- `benchmark_real_tables.py` — Full-table benchmark with real DLRM data
- `benchmark_encode_methods.py` — Encode method comparison (subprocess vs PyAV vs C++)
- `benchmark_decode_optimizations.py` — Decode-side optimization analysis
- `benchmark_batch_decode.py` — Batch decode parallelism benchmark
- `benchmark_full_pipeline.py` — End-to-end pipeline benchmark
- `test_integration.py` — 6 integration tests for C++ frame packing
- `test_tiled_storage.py` — 3 tests for tiled storage pipeline
