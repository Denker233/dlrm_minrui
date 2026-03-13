# Memory Analysis: "What Fits Where"

## 1. System Configurations

### Kaggle (D=16, 26 tables, 8 large tables >50K rows)
| Component | Baseline | Our System |
|-----------|----------|------------|
| Full embeddings (fp32) | 2,058 MB | — |
| Hot embeddings (fp32) | — | 88.5 MB |
| Cold compressed (H.265 CRF=18) | — | 0.8 MB |
| Decoded frame cache (20 frames) | — | 39.6 MB |
| Mapping + bitmap | — | 134.6 MB |
| **Total runtime** | **2,058 MB** | **263 MB** |
| **Reduction** | — | **7.8x** |
| Storage (disk) | 2,058 MB | 89 MB (23x) |

### Terabyte (D=64, 26 tables, 8 large tables >50K rows)
| Component | Baseline | Our System |
|-----------|----------|------------|
| Full embeddings (fp32) | 5,520 MB | — |
| Hot embeddings (fp32) | — | 237 MB |
| Cold uint8 (pre-decoded all 147 frames) | — | 291 MB |
| Compressed cold (H.265 CRF=0) | — | ~430 MB |
| Mapping + bitmap | — | 90 MB |
| **Total runtime (pre-decoded)** | **5,520 MB** | **~618 MB** |
| **Reduction** | — | **~8.9x** |
| Storage (H.265 CRF=0) | 5,520 MB | 667 MB (8.3x) |

## 2. Cache Hierarchy Analysis

### Typical Server CPU Cache Sizes
| Level | Size | What Fits |
|-------|------|-----------|
| L1d | 32-48 KB/core | 500-750 embedding rows (D=64, fp32) |
| L2 | 256 KB-1 MB/core | 4K-16K rows |
| L3 (LLC) | 10-35 MB shared | See below |
| L3 (large, e.g. Xeon) | 35-100 MB | See below |
| RAM (edge/laptop) | 4-16 GB | |
| RAM (server) | 64-512 GB | |

### What Fits in LLC (30 MB typical server)

**Baseline (fp32)**: NOTHING meaningful fits.
- Kaggle: 2,058 MB → 0.0015 of table fits in LLC
- Terabyte: 5,520 MB → 0.0005 of table fits in LLC
- Random access pattern = constant LLC misses

**Our system**:
- Hot top-1% rows: ~15-24 KB → fits entirely in LLC
- Per-table hot embedding bitmap: ~0.5-5 MB → fits in LLC
- The LLC effectively becomes a "hot row cache"

### Key Insight: Memory Footprint vs Working Set

The working set (rows actually accessed per batch) is much smaller than
the total memory footprint:

| | Kaggle | Terabyte |
|--|--------|----------|
| Rows accessed per batch | ~20K | ~20K |
| Distinct rows across test set | ~3.6M (10%) | ~5.6M (25%) |
| Hot rows (fp32 in memory) | 1.45M | 0.97M |
| Hot row memory | 88.5 MB | 237 MB |
| Cold row lookups per batch | ~2.1K | ~1.3K |
| Cold frames needed per batch | ~20 (all) | ~131 of 147 |

## 3. Mapping Table Overhead Analysis

The mapping table is the DOMINANT cost (51% on Kaggle, 15% on Terabyte).

### Current Design
- int32 per row in each large table
- Stores: hot row position (if hot) or packed(frame_id, offset) (if cold)
- Kaggle: 35.6M rows × 4B = 134.6 MB
- Terabyte: 22.6M rows × 4B = 86.2 MB + 4 MB bitmap = 90.2 MB

### Optimization: uint8 Frame ID + Bitmap Rank

**Idea**: For cold rows, only store frame_id (uint8 if ≤256 frames).
Position within frame computed via bitmap rank at runtime.

| Dataset | Current Mapping | Optimized | Savings |
|---------|----------------|-----------|---------|
| Kaggle | 134.6 MB | ~38 MB | 72% |
| Terabyte | 90.2 MB | ~27 MB | 70% |

**With optimized mapping**:
- Kaggle total: 88.5 + 0.8 + 39.6 + 38 = **167 MB** (12.3x reduction)
- Terabyte total: 237 + 430 + 291 + 27 = **~555 MB** (9.9x reduction)

### Alternative: Eliminate Mapping via Original ID Ordering

If cold rows are stored in original row ID order (not frequency-sorted):
- Frame assignment is deterministic: frame = bitmap_rank(row_id) // rpf
- No mapping needed at all
- **But**: all frames become "accessed" (no concentration)
- Kaggle: 254 frames (vs 20 accessed) → 503 MB decoded cache
- Terabyte: 671 frames (vs 147 accessed) → 1,329 MB decoded cache
- Net WORSE due to larger decoded cache

**Conclusion**: Frequency sorting + mapping is justified. The 134 MB mapping
enables 215 MB savings in decoded cache (254→20 frames).

## 4. Deployment Scenarios

### Edge Device (4 GB RAM, no GPU)
| | Baseline | Our System | Fits? |
|--|----------|------------|-------|
| Kaggle | 2,058 MB | 263 MB | Baseline: NO, Ours: YES |
| Terabyte | 5,520 MB | ~618 MB | Baseline: NO, Ours: YES |

### Laptop (16 GB RAM)
| | Baseline | Our System | Instances |
|--|----------|------------|-----------|
| Kaggle | 2 GB | 0.26 GB | 7 → 61 (9x more) |
| Terabyte | 5.5 GB | 0.62 GB | 2 → 25 (12x more) |

### Server (256 GB RAM)
| | Baseline | Our System | Instances |
|--|----------|------------|-----------|
| Kaggle | 2 GB | 0.26 GB | 128 → 985 (7.7x more) |
| Terabyte | 5.5 GB | 0.62 GB | 46 → 413 (9x more) |

### Storage (SSD/Network)
| | Baseline | CRF=0 (lossless) | CRF=18 (lossy) |
|--|----------|-------------------|-----------------|
| Kaggle | 2,058 MB | 104 MB (20x) | 89 MB (23x) |
| Terabyte | 5,520 MB | 667 MB (8.3x) | TBD |

## 5. Cost Analysis

At $0.10/GB/month (cloud RAM pricing):
| | Baseline | Our System | Monthly Savings (1000 instances) |
|--|----------|------------|--------------------------------|
| Kaggle | $206/inst | $26/inst | **$180K** |
| Terabyte | $552/inst | $62/inst | **$490K** |

## 6. LLC Miss Rate Measurement (from Experiment 4)

### Kaggle
| Config | LLC Miss % | Explanation |
|--------|-----------|-------------|
| A: PyTorch fp32 | 39.2% | 2 GB random access → terrible cache |
| B: C++ fp32 | 29.1% | Better loop order, but still 2 GB |
| C: C++ hot/cold | 38.9% | 263 MB, but mapping adds indirection |

**Surprise**: Config C has HIGHER LLC miss than Config B.
- Cause: Mapping table lookups (134 MB) add cache pressure
- The bitmap-rank dispatch adds an extra random memory access per row
- With optimized mapping (38 MB), this should improve

### Terabyte
| Config | LLC Miss % | Explanation |
|--------|-----------|-------------|
| A: PyTorch fp32 | 36.1% | 5.5 GB → poor cache |
| B: C++ fp32 | 48.3% | Higher miss rate with C++ (more efficient → less compute between loads) |
| C: C++ hot/cold | 31.6% | 618 MB → better cache utilization |

**Terabyte shows the expected behavior**: Compression REDUCES LLC misses
(31.6% vs 48.3%). This is because the memory reduction is larger (8.9x vs 7.8x)
and the mapping overhead is relatively smaller (90 MB vs 135 MB).
