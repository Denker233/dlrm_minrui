# Frame Packing Optimization Results

## Problem
Copying non-contiguous embeddings into video frames for H.265 encoding is costly.
The Python pipeline requires 4-5 intermediate memory copies (gather → quantize → pad → tile → tobytes),
with numpy's `transpose()` forcing a full-frame copy on every tile/untile operation.

## Solution: C++ Fused Frame Packing
Implemented 8 C++ functions in `csrc/compressed_emb.cpp` that replace the Python pipeline:

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

All functions use:
- `at::parallel_for` for multi-threaded processing
- AVX-512 SIMD for quantization (16 fp32 values in one instruction)
- Direct pointer arithmetic (no intermediate numpy/torch allocations)

## Benchmark Results

### Single-Frame Operations (1080p, 129,600 rows)
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

### Decode Pipeline (4K, with dequantization)
| K rows needed | Python (untile+dequant) | C++ (fused) | Speedup |
|--------------|------------------------|-------------|---------|
| 100 | 11.29ms | 0.004ms | **2,958x** |
| 1,000 | 11.62ms | 0.014ms | **809x** |
| 10,000 | 11.54ms | 0.019ms | **606x** |

### Real Data: All 8 Large Tables (32.3M cold rows, 254 frames)
| Table | Cold Rows | Frames | Python | C++ (scatter) | Speedup |
|-------|-----------|--------|--------|---------------|---------|
| 2 | 9.6M | 75 | 528ms | 16.8ms | **31x** |
| 11 | 8.0M | 62 | 438ms | 12.7ms | **34x** |
| 20 | 6.7M | 53 | 367ms | 11.4ms | **32x** |
| 15 | 5.3M | 41 | 310ms | 8.3ms | **37x** |
| 3 | 2.2M | 17 | 125ms | 2.0ms | **63x** |
| **Total** | **32.3M** | **254** | **1,791ms** | **51.6ms** | **34.7x** |

### Memory Savings
| Table | Python Alloc | C++ Alloc | Saved |
|-------|-------------|-----------|-------|
| 2 | 1,032 MB | 148 MB | 883 MB (86%) |
| 11 | 853 MB | 123 MB | 730 MB (86%) |
| 20 | 724 MB | 105 MB | 619 MB (86%) |
| 15 | 564 MB | 81 MB | 483 MB (86%) |
| **Total** | **3,468 MB** | **503 MB** | **2,965 MB (86%)** |

## Key Insight
The Python path creates 4-5 intermediate buffers per table due to:
1. `w[cold_idx]` — scatter gather creates new tensor
2. `((w / s).round() + zp).clamp(0, 255).to(uint8)` — 3 intermediate fp32 tensors
3. `np.zeros(padded, ...)` — padding buffer
4. `.transpose(0, 2, 1, 3).reshape(H, W)` — transpose makes non-contiguous, reshape copies
5. `.tobytes()` — serialization copy

The C++ fused path reads scattered fp32 values directly from the embedding table,
quantizes with AVX-512, and writes tiled uint8 bytes to the frame buffer — **zero intermediate allocations**.

## Correctness
- All C++ outputs exactly match Python (verified per-pixel)
- Quantization MAE identical to Python: ~0.000190
- Max Python-C++ dequantized difference: 0.000000

## Files Modified
- `csrc/compressed_emb.cpp` — new frame packing C++ functions
- `codec_ondemand_benchmark.py` — integrated C++ tiling in encode/decode paths
- `benchmark_frame_packing.py` — microbenchmark for individual operations
- `benchmark_real_tables.py` — full-table benchmark with real data
