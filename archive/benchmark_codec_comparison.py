#!/usr/bin/env python3
"""
Benchmark: H.265 vs LZ4 vs Zstd vs Snappy for embedding compression.

Compares general-purpose compressors against H.265 on the SAME quantized uint8
embedding data, measuring:
  1. Compression ratio (vs raw uint8)
  2. Compress time
  3. Decompress time (critical path)
  4. Random-access decode cost (decompress + gather specific rows)

Tests both per-frame granularity (same as H.265) and per-row-group granularity.
"""

import os, sys, time, json, gc
import numpy as np
import torch
import lz4.frame
import zstandard as zstd
import snappy

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR = "results"
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
ONDEMAND_DIR = os.path.join(RESULTS_DIR, "ondemand")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "codec_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TILE_H, TILE_W = 4, 4
FRAME_W, FRAME_H = 1920, 1080
ROWS_PER_FRAME = (FRAME_W // TILE_W) * (FRAME_H // TILE_H)  # 129600

# Tables with significant cold data
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]


def load_cold_data():
    """Load quantized cold embeddings for all large tables."""
    print("Loading model and cold data...")

    # Load model state dict
    ckpt_path = "./models/dlrm_kaggle_correct.pt"
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    elif 'model_state_dict' in sd:
        sd = sd['model_state_dict']

    tables = {}
    for t_idx in LARGE_TABLES:
        cold_idx_path = os.path.join(HOTCOLD_DIR, f"cold_indices_{t_idx}.pt")
        if not os.path.exists(cold_idx_path):
            continue

        cold_indices = torch.load(cold_idx_path, weights_only=False)
        if cold_indices.numel() == 0:
            continue

        # Get cold embeddings
        weight = sd[f'emb_l.{t_idx}.weight']
        cold_weight = weight[cold_indices]  # (N, 16) fp32

        # Quantize to uint8
        mn, mx = cold_weight.min().item(), cold_weight.max().item()
        scale = (mx - mn) / 255.0
        if scale == 0:
            scale = 1.0
        zp = round(-mn / scale)
        quant = ((cold_weight / scale).round() + zp).clamp(0, 255).to(torch.uint8)

        # Load reordered order if available
        order_path = os.path.join(REORDER_DIR, f"cold_order_{t_idx}.npy")
        if os.path.exists(order_path):
            order = np.load(order_path)
            quant = quant[torch.from_numpy(order.astype(np.int64))]

        tables[t_idx] = {
            'quant': quant.numpy(),  # (N, 16) uint8
            'scale': scale,
            'zp': zp,
            'n_rows': quant.shape[0],
        }
        print(f"  Table {t_idx}: {quant.shape[0]:,} cold rows, {quant.nbytes/1e6:.1f}MB uint8")

    return tables


def tile_to_frame(rows_uint8):
    """Tile embedding rows into a 2D frame for H.265 encoding."""
    n_rows = rows_uint8.shape[0]
    # Pad to fill frame
    if n_rows < ROWS_PER_FRAME:
        padded = np.zeros((ROWS_PER_FRAME, 16), dtype=np.uint8)
        padded[:n_rows] = rows_uint8
        rows_uint8 = padded

    tiles_w = FRAME_W // TILE_W  # 480
    tiles_h = FRAME_H // TILE_H  # 270
    frame = rows_uint8.reshape(tiles_h, tiles_w, TILE_H, TILE_W)
    frame = frame.transpose(0, 2, 1, 3).reshape(FRAME_H, FRAME_W)
    return frame


def split_into_frames(quant_data):
    """Split quantized data into frame-sized chunks."""
    n_rows = quant_data.shape[0]
    n_frames = (n_rows + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
    frames = []
    for i in range(n_frames):
        start = i * ROWS_PER_FRAME
        end = min(start + ROWS_PER_FRAME, n_rows)
        chunk = quant_data[start:end]
        frames.append(chunk)
    return frames


# ============================================================
# COMPRESSOR IMPLEMENTATIONS
# ============================================================

class H265Compressor:
    """H.265 lossless via C++ extension."""
    name = "H.265 (lossless)"

    def __init__(self):
        try:
            import compressed_emb as _C
            self._C = _C
        except ImportError:
            self._C = None

    def compress_frame(self, rows_uint8):
        """Compress a frame-worth of rows."""
        frame = tile_to_frame(rows_uint8)
        frame_t = torch.from_numpy(frame)
        tmp_path = "/tmp/_bench_frame.h265"
        self._C.encode_h265_frame_to_file(frame_t, tmp_path)
        with open(tmp_path, 'rb') as f:
            data = f.read()
        os.remove(tmp_path)
        return data

    def decompress_frame(self, compressed_bytes, n_rows=None):
        """Decompress to get back the frame."""
        tmp_path = "/tmp/_bench_frame.h265"
        with open(tmp_path, 'wb') as f:
            f.write(compressed_bytes)
        frame_t = self._C.decode_h265_frame_from_file(tmp_path)
        os.remove(tmp_path)
        return frame_t.numpy()

    def decompress_frame_to_rows(self, compressed_bytes, n_rows):
        """Decompress and untile to get embedding rows."""
        frame = self.decompress_frame(compressed_bytes)
        # Untile
        tiles_w = FRAME_W // TILE_W
        tiles_h = FRAME_H // TILE_H
        rows = frame.reshape(tiles_h, TILE_H, tiles_w, TILE_W)
        rows = rows.transpose(0, 2, 1, 3).reshape(-1, 16)
        return rows[:n_rows]


class LZ4Compressor:
    """LZ4 frame compression."""
    name = "LZ4"

    def compress_frame(self, rows_uint8):
        return lz4.frame.compress(rows_uint8.tobytes())

    def decompress_frame(self, compressed_bytes, n_rows=None):
        raw = lz4.frame.decompress(compressed_bytes)
        return np.frombuffer(raw, dtype=np.uint8)

    def decompress_frame_to_rows(self, compressed_bytes, n_rows):
        raw = lz4.frame.decompress(compressed_bytes)
        return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 16)[:n_rows]


class LZ4HCCompressor:
    """LZ4 HC (high compression) mode."""
    name = "LZ4-HC"

    def compress_frame(self, rows_uint8):
        return lz4.frame.compress(rows_uint8.tobytes(),
                                  compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)

    def decompress_frame(self, compressed_bytes, n_rows=None):
        raw = lz4.frame.decompress(compressed_bytes)
        return np.frombuffer(raw, dtype=np.uint8)

    def decompress_frame_to_rows(self, compressed_bytes, n_rows):
        raw = lz4.frame.decompress(compressed_bytes)
        return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 16)[:n_rows]


class ZstdCompressor:
    """Zstandard compression at default level."""
    name = "Zstd (level 3)"

    def __init__(self, level=3):
        self.level = level
        self.name = f"Zstd (level {level})"
        self.cctx = zstd.ZstdCompressor(level=level)
        self.dctx = zstd.ZstdDecompressor()

    def compress_frame(self, rows_uint8):
        return self.cctx.compress(rows_uint8.tobytes())

    def decompress_frame(self, compressed_bytes, n_rows=None):
        raw = self.dctx.decompress(compressed_bytes)
        return np.frombuffer(raw, dtype=np.uint8)

    def decompress_frame_to_rows(self, compressed_bytes, n_rows):
        raw = self.dctx.decompress(compressed_bytes)
        return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 16)[:n_rows]


class ZstdMaxCompressor(ZstdCompressor):
    """Zstandard at max compression."""
    def __init__(self):
        super().__init__(level=19)


class SnappyCompressor:
    """Snappy compression."""
    name = "Snappy"

    def compress_frame(self, rows_uint8):
        return snappy.compress(rows_uint8.tobytes())

    def decompress_frame(self, compressed_bytes, n_rows=None):
        raw = snappy.decompress(compressed_bytes)
        return np.frombuffer(raw, dtype=np.uint8)

    def decompress_frame_to_rows(self, compressed_bytes, n_rows):
        raw = snappy.decompress(compressed_bytes)
        return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 16)[:n_rows]


class NoCompressor:
    """Raw uint8 baseline (no compression)."""
    name = "Raw uint8 (no compression)"

    def compress_frame(self, rows_uint8):
        return rows_uint8.tobytes()

    def decompress_frame(self, compressed_bytes, n_rows=None):
        return np.frombuffer(compressed_bytes, dtype=np.uint8)

    def decompress_frame_to_rows(self, compressed_bytes, n_rows):
        return np.frombuffer(compressed_bytes, dtype=np.uint8).reshape(-1, 16)[:n_rows]


# ============================================================
# H.265 with tiled data (tests if tiling helps compression)
# ============================================================

class H265TiledCompressor:
    """H.265 on tiled frame layout (the layout we actually use)."""
    name = "H.265 tiled (lossless)"

    def __init__(self):
        import compressed_emb as _C
        self._C = _C

    def compress_frame(self, rows_uint8):
        frame = tile_to_frame(rows_uint8)
        frame_t = torch.from_numpy(frame)
        tmp_path = "/tmp/_bench_tiled.h265"
        self._C.encode_h265_frame_to_file(frame_t, tmp_path)
        with open(tmp_path, 'rb') as f:
            data = f.read()
        os.remove(tmp_path)
        return data

    def decompress_frame_to_rows(self, compressed_bytes, n_rows):
        tmp_path = "/tmp/_bench_tiled.h265"
        with open(tmp_path, 'wb') as f:
            f.write(compressed_bytes)
        frame_t = self._C.decode_h265_frame_from_file(tmp_path)
        os.remove(tmp_path)
        frame = frame_t.numpy()
        tiles_w = FRAME_W // TILE_W
        tiles_h = FRAME_H // TILE_H
        rows = frame.reshape(tiles_h, TILE_H, tiles_w, TILE_W)
        rows = rows.transpose(0, 2, 1, 3).reshape(-1, 16)
        return rows[:n_rows]


class LZ4TiledCompressor:
    """LZ4 on the TILED frame layout (to test if tiling helps LZ4 too)."""
    name = "LZ4 (tiled layout)"

    def compress_frame(self, rows_uint8):
        frame = tile_to_frame(rows_uint8)
        return lz4.frame.compress(frame.tobytes())

    def decompress_frame_to_rows(self, compressed_bytes, n_rows):
        raw = lz4.frame.decompress(compressed_bytes)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(FRAME_H, FRAME_W)
        tiles_w = FRAME_W // TILE_W
        tiles_h = FRAME_H // TILE_H
        rows = frame.reshape(tiles_h, TILE_H, tiles_w, TILE_W)
        rows = rows.transpose(0, 2, 1, 3).reshape(-1, 16)
        return rows[:n_rows]


class ZstdTiledCompressor:
    """Zstd on the TILED frame layout."""
    name = "Zstd (tiled layout)"

    def __init__(self):
        self.cctx = zstd.ZstdCompressor(level=3)
        self.dctx = zstd.ZstdDecompressor()

    def compress_frame(self, rows_uint8):
        frame = tile_to_frame(rows_uint8)
        return self.cctx.compress(frame.tobytes())

    def decompress_frame_to_rows(self, compressed_bytes, n_rows):
        raw = self.dctx.decompress(compressed_bytes)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(FRAME_H, FRAME_W)
        tiles_w = FRAME_W // TILE_W
        tiles_h = FRAME_H // TILE_H
        rows = frame.reshape(tiles_h, TILE_H, tiles_w, TILE_W)
        rows = rows.transpose(0, 2, 1, 3).reshape(-1, 16)
        return rows[:n_rows]


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_compressor(compressor, frames_data, n_warmup=3, n_iters=20):
    """Benchmark a single compressor on all frames."""
    results = {
        'name': compressor.name,
        'compress_times_ms': [],
        'decompress_times_ms': [],
        'compressed_sizes': [],
        'raw_sizes': [],
    }

    # Pre-compress all frames
    compressed_frames = []
    for i, (rows, n_rows) in enumerate(frames_data):
        t0 = time.time()
        compressed = compressor.compress_frame(rows)
        t1 = time.time()
        compressed_frames.append(compressed)
        results['compressed_sizes'].append(len(compressed))
        results['raw_sizes'].append(len(rows.tobytes()))
        results['compress_times_ms'].append((t1 - t0) * 1000)

    # Warmup decompression
    for _ in range(n_warmup):
        for compressed, (rows, n_rows) in zip(compressed_frames, frames_data):
            _ = compressor.decompress_frame_to_rows(compressed, n_rows)

    # Timed decompression
    per_frame_times = []
    for _ in range(n_iters):
        for compressed, (rows, n_rows) in zip(compressed_frames, frames_data):
            t0 = time.time()
            _ = compressor.decompress_frame_to_rows(compressed, n_rows)
            t1 = time.time()
            per_frame_times.append((t1 - t0) * 1000)

    n_frames = len(compressed_frames)
    results['decompress_times_ms'] = per_frame_times

    total_raw = sum(results['raw_sizes'])
    total_compressed = sum(results['compressed_sizes'])
    ratio = total_raw / total_compressed if total_compressed > 0 else 0

    avg_decompress = np.mean(per_frame_times)
    p50_decompress = np.median(per_frame_times)
    avg_compress = np.mean(results['compress_times_ms'])

    return {
        'name': compressor.name,
        'total_raw_mb': total_raw / 1e6,
        'total_compressed_mb': total_compressed / 1e6,
        'compression_ratio': ratio,
        'ratio_vs_fp32': ratio * 4,
        'avg_compress_ms': avg_compress,
        'avg_decompress_ms': avg_decompress,
        'p50_decompress_ms': p50_decompress,
        'n_frames': n_frames,
    }


def benchmark_h265_existing(tables):
    """Benchmark H.265 using existing .h265 files on disk (realistic)."""
    try:
        import compressed_emb as _C
    except ImportError:
        print("  SKIP: compressed_emb not available")
        return None

    total_raw = 0
    total_compressed = 0
    decomp_times = []

    for t_idx, info in tables.items():
        frame_dir = os.path.join(ONDEMAND_DIR, '1080p', f'table_{t_idx}')
        if not os.path.exists(frame_dir):
            continue

        files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.h265')])
        n_rows = info['n_rows']
        total_raw += n_rows * 16

        for fname in files:
            fpath = os.path.join(frame_dir, fname)
            total_compressed += os.path.getsize(fpath)

        # Time decoding: warmup
        for _ in range(2):
            for fname in files[:3]:
                fpath = os.path.join(frame_dir, fname)
                _ = _C.decode_h265_frame_from_file(fpath)

        # Timed decoding
        for _ in range(10):
            for fname in files:
                fpath = os.path.join(frame_dir, fname)
                t0 = time.time()
                _ = _C.decode_h265_frame_from_file(fpath)
                t1 = time.time()
                decomp_times.append((t1 - t0) * 1000)

    ratio = total_raw / total_compressed if total_compressed > 0 else 0
    return {
        'name': 'H.265 lossless (existing files)',
        'total_raw_mb': total_raw / 1e6,
        'total_compressed_mb': total_compressed / 1e6,
        'compression_ratio': ratio,
        'ratio_vs_fp32': ratio * 4,
        'avg_compress_ms': 83.0,  # from previous measurement
        'avg_decompress_ms': np.mean(decomp_times),
        'p50_decompress_ms': np.median(decomp_times),
        'n_frames': len(decomp_times) // 10,
    }


def run_benchmark():
    tables = load_cold_data()

    # Prepare frame-sized chunks from largest table (table 2) for fair comparison
    test_table = 2
    if test_table not in tables:
        test_table = list(tables.keys())[0]

    info = tables[test_table]
    quant = info['quant']
    print(f"\nBenchmarking on table {test_table}: {info['n_rows']:,} rows")

    frame_chunks = split_into_frames(quant)
    frames_data = [(chunk, len(chunk)) for chunk in frame_chunks]
    print(f"  Split into {len(frames_data)} frames of {ROWS_PER_FRAME:,} rows each")
    print(f"  Raw uint8 per frame: {ROWS_PER_FRAME * 16 / 1e6:.2f} MB")

    # Also prepare ALL tables combined
    all_frames = []
    for t_idx in sorted(tables.keys()):
        chunks = split_into_frames(tables[t_idx]['quant'])
        all_frames.extend([(c, len(c)) for c in chunks])
    print(f"  Total frames across all tables: {len(all_frames)}")

    compressors = [
        NoCompressor(),
        LZ4Compressor(),
        LZ4HCCompressor(),
        ZstdCompressor(level=3),
        ZstdCompressor(level=9),
        ZstdMaxCompressor(),
        SnappyCompressor(),
    ]

    # Try tiled-layout variants
    tiled_compressors = [
        LZ4TiledCompressor(),
        ZstdTiledCompressor(),
    ]

    # Try H.265
    try:
        h265_tiled = H265TiledCompressor()
        tiled_compressors.append(h265_tiled)
    except:
        pass

    # ---- Single table benchmark (table 2) ----
    print(f"\n{'='*80}")
    print(f"SINGLE TABLE BENCHMARK (Table {test_table}, {len(frames_data)} frames)")
    print(f"{'='*80}\n")

    print("--- Row-major layout (flat uint8) ---")
    single_results = []
    for comp in compressors:
        print(f"  Testing {comp.name}...")
        try:
            r = benchmark_compressor(comp, frames_data)
            single_results.append(r)
            print(f"    Ratio: {r['compression_ratio']:.1f}x (vs fp32: {r['ratio_vs_fp32']:.1f}x) | "
                  f"Decompress: {r['avg_decompress_ms']:.3f}ms/frame | "
                  f"Compress: {r['avg_compress_ms']:.1f}ms/frame | "
                  f"Size: {r['total_compressed_mb']:.1f}MB")
        except Exception as e:
            print(f"    FAILED: {e}")

    print("\n--- Tiled layout (2D frame, same as H.265 uses) ---")
    tiled_results = []
    for comp in tiled_compressors:
        print(f"  Testing {comp.name}...")
        try:
            r = benchmark_compressor(comp, frames_data, n_iters=10)
            tiled_results.append(r)
            print(f"    Ratio: {r['compression_ratio']:.1f}x (vs fp32: {r['ratio_vs_fp32']:.1f}x) | "
                  f"Decompress: {r['avg_decompress_ms']:.3f}ms/frame | "
                  f"Compress: {r['avg_compress_ms']:.1f}ms/frame | "
                  f"Size: {r['total_compressed_mb']:.1f}MB")
        except Exception as e:
            print(f"    FAILED: {e}")

    # ---- H.265 from existing files ----
    print(f"\n  Testing H.265 from existing files (all tables)...")
    h265_result = benchmark_h265_existing(tables)
    if h265_result:
        print(f"    Ratio: {h265_result['compression_ratio']:.1f}x (vs fp32: {h265_result['ratio_vs_fp32']:.1f}x) | "
              f"Decompress: {h265_result['avg_decompress_ms']:.3f}ms/frame | "
              f"Size: {h265_result['total_compressed_mb']:.1f}MB")

    # ---- All tables benchmark ----
    print(f"\n{'='*80}")
    print(f"ALL TABLES BENCHMARK ({len(all_frames)} frames)")
    print(f"{'='*80}\n")

    all_results = []
    for comp in compressors:
        if isinstance(comp, H265Compressor):
            continue  # too slow for all frames
        print(f"  Testing {comp.name}...")
        try:
            r = benchmark_compressor(comp, all_frames, n_iters=5)
            all_results.append(r)
            print(f"    Ratio: {r['compression_ratio']:.1f}x (vs fp32: {r['ratio_vs_fp32']:.1f}x) | "
                  f"Decompress: {r['avg_decompress_ms']:.3f}ms/frame | "
                  f"Size: {r['total_compressed_mb']:.1f}MB")
        except Exception as e:
            print(f"    FAILED: {e}")

    if h265_result:
        all_results.append(h265_result)

    # ---- Row-group granularity benchmark ----
    print(f"\n{'='*80}")
    print(f"ROW-GROUP GRANULARITY BENCHMARK (varying chunk sizes)")
    print(f"{'='*80}\n")

    chunk_sizes = [512, 1024, 4096, 16384, ROWS_PER_FRAME]
    granularity_results = {}

    test_data = quant[:min(len(quant), ROWS_PER_FRAME * 5)]  # first 5 frames worth

    for chunk_size in chunk_sizes:
        n_chunks = (len(test_data) + chunk_size - 1) // chunk_size
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(test_data))
            chunks.append((test_data[start:end], end - start))

        print(f"\n  Chunk size: {chunk_size:,} rows ({chunk_size*16/1024:.0f} KB raw)")

        for comp in [LZ4Compressor(), ZstdCompressor(level=3), SnappyCompressor()]:
            try:
                r = benchmark_compressor(comp, chunks, n_iters=10)
                key = (comp.name, chunk_size)
                granularity_results[key] = r
                print(f"    {comp.name:20s}: ratio={r['compression_ratio']:.1f}x, "
                      f"decomp={r['avg_decompress_ms']:.3f}ms/chunk")
            except Exception as e:
                print(f"    {comp.name:20s}: FAILED: {e}")

    # ---- Summary ----
    print(f"\n{'='*80}")
    print(f"SUMMARY: Sorted by decompress time (fastest first)")
    print(f"{'='*80}\n")

    all_sorted = single_results + tiled_results
    if h265_result:
        all_sorted.append(h265_result)
    all_sorted.sort(key=lambda x: x['avg_decompress_ms'])

    print(f"{'Compressor':<35s} | {'Ratio':>6s} | {'vs fp32':>7s} | "
          f"{'Decomp/frame':>13s} | {'Comp/frame':>11s} | {'Size':>8s}")
    print("-" * 110)
    for r in all_sorted:
        print(f"{r['name']:<35s} | {r['compression_ratio']:>5.1f}x | {r['ratio_vs_fp32']:>6.1f}x | "
              f"{r['avg_decompress_ms']:>10.3f} ms | {r['avg_compress_ms']:>8.1f} ms | "
              f"{r['total_compressed_mb']:>6.1f} MB")

    # ---- KEY QUESTION: Does H.265 provide better compression? ----
    print(f"\n{'='*80}")
    print("KEY FINDING: Does H.265 provide better compression than LZ4/Zstd?")
    print(f"{'='*80}\n")

    h265_ratio = None
    for r in all_sorted:
        if 'H.265' in r['name'] and 'tiled' in r['name']:
            h265_ratio = r['compression_ratio']
            break
    if h265_ratio is None and h265_result:
        h265_ratio = h265_result['compression_ratio']

    if h265_ratio:
        for r in all_sorted:
            if 'H.265' not in r['name']:
                better = "BETTER" if r['compression_ratio'] >= h265_ratio else "WORSE"
                faster = r['avg_decompress_ms'] < (h265_result['avg_decompress_ms'] if h265_result else 999)
                print(f"  {r['name']:<35s}: {r['compression_ratio']:.1f}x vs H.265 {h265_ratio:.1f}x "
                      f"({better} compression, {'FASTER' if faster else 'SLOWER'} decode)")

    # Save results
    output = {
        'single_table': single_results,
        'tiled_layout': tiled_results,
        'all_tables': all_results,
        'h265_existing': h265_result,
        'granularity': {f"{k[0]}_{k[1]}": v for k, v in granularity_results.items()},
        'metadata': {
            'test_table': test_table,
            'n_rows': info['n_rows'],
            'rows_per_frame': ROWS_PER_FRAME,
            'raw_frame_bytes': ROWS_PER_FRAME * 16,
        }
    }

    json_path = os.path.join(OUTPUT_DIR, 'codec_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    run_benchmark()
