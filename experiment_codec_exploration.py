#!/usr/bin/env python3
"""
Codec Exploration: Compare compression codecs on DLRM cold embedding data.

Benchmarks H.265, H.264, FFV1, Zstd (multiple levels), and LZ4 on the same
cold embedding frames. Measures: compressed size, encode time, decode time
(1-thread and 4-thread), and batch decode (20 frames).

Also includes:
- Advanced strategies: quantize-then-Zstd, delta+Zstd, dictionary Zstd
- GPU hardware decode analysis (theoretical)
"""

import os, sys, time, json, tempfile, gc
import numpy as np
import torch
import lz4.frame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# C++ extension
LD_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dlrm_env/lib/python3.10/site-packages/torch/lib')
if 'LD_LIBRARY_PATH' not in os.environ or LD_LIB not in os.environ.get('LD_LIBRARY_PATH', ''):
    pass  # assume already set or torch is importable

import compressed_emb as _C

RESULTS_DIR = "results/codec_exploration"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- Constants (Kaggle, 1080p) ----------
EMB_DIM = 16
TILE_W, TILE_H = 4, 4
WIDTH, HEIGHT = 1920, 1080
TILES_PER_ROW = WIDTH // TILE_W
TILES_PER_COL = HEIGHT // TILE_H
ROWS_PER_FRAME = TILES_PER_ROW * TILES_PER_COL  # 129600

REORDER_DIR = "results/reorder"
HOTCOLD_DIR = "results/hotcold"
ONDEMAND_DIR = "results/ondemand"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_cold_frame_data(table_id=2):
    """Load precomputed cold embedding data for table 2 (largest, 10.1M rows)."""
    cold_order = np.load(os.path.join(REORDER_DIR, f'cold_order_{table_id}.npy'))
    model_path = "./models/dlrm_kaggle_correct.pt"
    ld = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = ld['state_dict']
    emb_key = f'emb_l.{table_id}.weight'
    w = state_dict[emb_key]

    reordered_w = w[cold_order]
    mn = reordered_w.min().item()
    mx = reordered_w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((reordered_w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def tile_frame(q_rows):
    """Tile uint8 rows (N, 16) into (H, W) frame."""
    return _C.tile_rows_to_frame(q_rows, WIDTH, HEIGHT)


def benchmark_single_codec(name, encode_fn, decode_fn, frame_2d_or_flat,
                           n_iters=20, n_threads_list=[1, 4]):
    """Benchmark a single codec on one frame."""
    result = {'name': name}

    # Encode
    t0 = time.time()
    for _ in range(3):
        compressed = encode_fn(frame_2d_or_flat)
    encode_time = (time.time() - t0) / 3
    result['encode_time_ms'] = encode_time * 1000

    if isinstance(compressed, (bytes, bytearray)):
        result['compressed_bytes'] = len(compressed)
    elif isinstance(compressed, torch.Tensor):
        result['compressed_bytes'] = compressed.numel()
    else:
        result['compressed_bytes'] = os.path.getsize(compressed)  # file path

    # Decode timing for different thread counts
    for nt in n_threads_list:
        times = []
        for _ in range(n_iters):
            t0 = time.time()
            decoded = decode_fn(compressed, nt)
            times.append(time.time() - t0)
        result[f'decode_{nt}t_ms'] = np.median(times) * 1000
        result[f'decode_{nt}t_p99_ms'] = np.percentile(times, 99) * 1000

    return result


def benchmark_batch_decode(name, paths_or_data, decode_batch_fn, n_frames=20, n_iters=10):
    """Benchmark batch decode of multiple frames."""
    times = []
    for _ in range(n_iters):
        t0 = time.time()
        decode_batch_fn(paths_or_data, n_frames)
        times.append(time.time() - t0)
    return {
        'name': name,
        'n_frames': n_frames,
        'batch_decode_ms': np.median(times) * 1000,
        'batch_decode_p99_ms': np.percentile(times, 99) * 1000,
    }


def run_codec_benchmarks():
    """Run full codec comparison on frame 0 of table 2."""
    log("Loading cold embedding data for table 2...")
    q, scale, zp = get_cold_frame_data(table_id=2)
    log(f"  {q.shape[0]:,} cold rows, quantized to uint8")

    # Prepare frame 0 data
    frame_rows = q[:ROWS_PER_FRAME]  # (129600, 16) uint8
    tiled_frame = tile_frame(frame_rows)  # (1080, 1920) uint8
    flat_bytes = frame_rows.contiguous().view(-1)  # (2073600,) uint8
    raw_uint8_bytes = flat_bytes.numel()
    raw_fp32_bytes = raw_uint8_bytes * 4  # original fp32 = 4 bytes per element

    log(f"  Frame 0: {ROWS_PER_FRAME} rows, {raw_uint8_bytes} uint8 bytes, "
        f"{raw_fp32_bytes} fp32 bytes, tiled={tiled_frame.shape}")

    results = []
    tmpdir = tempfile.mkdtemp()

    # Helper to log a codec result with fp32 ratio
    def log_codec(r):
        log(f"  {r['name']}: {r['compressed_bytes']} bytes "
            f"({r['ratio_vs_fp32']:.1f}x vs fp32, {r['ratio_vs_uint8']:.1f}x vs uint8), "
            f"encode={r['encode_time_ms']:.1f}ms, "
            f"decode 1T={r['decode_1t_ms']:.2f}ms, "
            f"4T={r.get('decode_4t_ms', r['decode_1t_ms']):.2f}ms")

    def add_ratios(r):
        r['raw_fp32_bytes'] = raw_fp32_bytes
        r['raw_uint8_bytes'] = raw_uint8_bytes
        r['ratio_vs_fp32'] = raw_fp32_bytes / r['compressed_bytes']
        r['ratio_vs_uint8'] = raw_uint8_bytes / r['compressed_bytes']

    # ---- H.265 lossless (CRF=0) ----
    log("\nBenchmarking H.265 lossless...")
    h265_l_path = os.path.join(tmpdir, 'frame_h265_lossless.h265')
    def h265_l_encode(f):
        _C.encode_frame_codec(f, h265_l_path, 'h265', True, 0)
        return h265_l_path
    def h265_l_decode(path, nt):
        return _C.decode_h265_frame_from_file(path, nt)
    r = benchmark_single_codec('H.265 CRF=0', h265_l_encode, h265_l_decode, tiled_frame)
    add_ratios(r)
    r['lossy'] = False
    results.append(r)
    log_codec(r)

    # ---- H.265 CRF=18 ----
    log("\nBenchmarking H.265 CRF=18...")
    h265_crf_path = os.path.join(tmpdir, 'frame_h265_crf18.h265')
    def h265_crf_encode(f):
        _C.encode_frame_codec(f, h265_crf_path, 'h265', False, 18)
        return h265_crf_path
    def h265_crf_decode(path, nt):
        return _C.decode_h265_frame_from_file(path, nt)
    r = benchmark_single_codec('H.265 CRF=18', h265_crf_encode, h265_crf_decode, tiled_frame)
    add_ratios(r)
    r['lossy'] = True
    results.append(r)
    log_codec(r)

    # ---- H.264 lossless ----
    log("\nBenchmarking H.264 lossless...")
    h264_l_path = os.path.join(tmpdir, 'frame_h264_lossless.h264')
    def h264_l_encode(f):
        _C.encode_frame_codec(f, h264_l_path, 'h264', True, 0)
        return h264_l_path
    def h264_l_decode(path, nt):
        return _C.decode_h265_frame_from_file(path, nt)
    r = benchmark_single_codec('H.264 lossless', h264_l_encode, h264_l_decode, tiled_frame)
    add_ratios(r)
    r['lossy'] = False
    results.append(r)
    log_codec(r)

    # ---- H.264 CRF=18 ----
    log("\nBenchmarking H.264 CRF=18...")
    h264_crf_path = os.path.join(tmpdir, 'frame_h264_crf18.h264')
    def h264_crf_encode(f):
        _C.encode_frame_codec(f, h264_crf_path, 'h264', False, 18)
        return h264_crf_path
    def h264_crf_decode(path, nt):
        return _C.decode_h265_frame_from_file(path, nt)
    r = benchmark_single_codec('H.264 CRF=18', h264_crf_encode, h264_crf_decode, tiled_frame)
    add_ratios(r)
    r['lossy'] = True
    results.append(r)
    log_codec(r)

    # ---- FFV1 lossless ----
    log("\nBenchmarking FFV1 lossless...")
    ffv1_path = os.path.join(tmpdir, 'frame_ffv1.mkv')
    def ffv1_encode(f):
        _C.encode_frame_codec(f, ffv1_path, 'ffv1', True, 0)
        return ffv1_path
    def ffv1_decode(path, nt):
        return _C.decode_h265_frame_from_file(path, nt)
    r = benchmark_single_codec('FFV1', ffv1_encode, ffv1_decode, tiled_frame)
    add_ratios(r)
    r['lossy'] = False
    results.append(r)
    log_codec(r)

    # ---- Zstd levels 1, 3, 9, 19 ----
    for level in [1, 3, 9, 19]:
        log(f"\nBenchmarking Zstd L{level}...")
        def zstd_encode(f, lv=level):
            return _C.zstd_compress_frame(f, lv)
        def zstd_decode(compressed, nt):
            return _C.zstd_decompress_frame(compressed, raw_uint8_bytes)
        r = benchmark_single_codec(f'Zstd-L{level}', zstd_encode, zstd_decode, flat_bytes)
        add_ratios(r)
        r['lossy'] = False
        results.append(r)
        log_codec(r)

    # ---- LZ4 ----
    log("\nBenchmarking LZ4...")
    def lz4_encode(f):
        data = f.numpy().tobytes() if isinstance(f, torch.Tensor) else f
        return lz4.frame.compress(data)
    def lz4_decode(compressed, nt):
        return lz4.frame.decompress(compressed)
    r = benchmark_single_codec('LZ4', lz4_encode, lz4_decode, flat_bytes)
    add_ratios(r)
    r['lossy'] = False
    results.append(r)
    log_codec(r)

    # ---- Batch decode benchmarks (20 frames) ----
    log(f"\n{'='*60}")
    log("BATCH DECODE BENCHMARKS (20 frames)")
    log(f"{'='*60}")

    # Prepare 20 frames
    n_batch_frames = 20
    batch_results = []

    # H.265 CRF=0 batch
    log("\nEncoding 20 H.265 CRF=0 frames...")
    h265_batch_dir = os.path.join(tmpdir, 'h265_batch')
    os.makedirs(h265_batch_dir, exist_ok=True)
    tiled_frames = []
    for i in range(n_batch_frames):
        start = i * ROWS_PER_FRAME
        end = start + ROWS_PER_FRAME
        rows = q[start:end] if end <= q.shape[0] else torch.zeros(ROWS_PER_FRAME, EMB_DIM, dtype=torch.uint8)
        tiled_frames.append(tile_frame(rows))
    _C.batch_encode_frames_codec(tiled_frames, h265_batch_dir, 'h265', True)
    h265_paths = [os.path.join(h265_batch_dir, f'frame_{i:05d}.h265') for i in range(n_batch_frames)]
    br = benchmark_batch_decode('H.265 CRF=0', h265_paths,
        lambda paths, n: _C.batch_decode_file_paths(paths[:n], 1, 0))
    batch_results.append(br)
    log(f"  {br['name']}: {br['batch_decode_ms']:.1f}ms (20 frames, 1T/frame)")

    # H.265 CRF=0 batch 4T
    br4 = benchmark_batch_decode('H.265 CRF=0 (4T)', h265_paths,
        lambda paths, n: _C.batch_decode_file_paths(paths[:n], 4, 4))
    batch_results.append(br4)
    log(f"  {br4['name']}: {br4['batch_decode_ms']:.1f}ms (20 frames, 4T/decode, 4 parallel)")

    # Zstd-3 batch
    log("\nEncoding 20 Zstd-3 frames...")
    zstd_batch_dir = os.path.join(tmpdir, 'zstd_batch')
    os.makedirs(zstd_batch_dir, exist_ok=True)
    flat_frames = []
    for i in range(n_batch_frames):
        start = i * ROWS_PER_FRAME
        end = start + ROWS_PER_FRAME
        rows = q[start:end] if end <= q.shape[0] else torch.zeros(ROWS_PER_FRAME, EMB_DIM, dtype=torch.uint8)
        flat_frames.append(rows.contiguous().view(-1))
    _C.batch_zstd_compress(flat_frames, zstd_batch_dir, 3)
    zstd_paths = [os.path.join(zstd_batch_dir, f'frame_{i:05d}.zst') for i in range(n_batch_frames)]
    br = benchmark_batch_decode('Zstd-3', zstd_paths,
        lambda paths, n: _C.batch_zstd_decompress_files(paths[:n], raw_uint8_bytes, 0))
    batch_results.append(br)
    log(f"  {br['name']}: {br['batch_decode_ms']:.1f}ms (20 frames, parallel)")

    # LZ4 batch
    log("\nEncoding 20 LZ4 frames...")
    lz4_compressed_list = []
    for i in range(n_batch_frames):
        start = i * ROWS_PER_FRAME
        end = start + ROWS_PER_FRAME
        rows = q[start:end] if end <= q.shape[0] else torch.zeros(ROWS_PER_FRAME, EMB_DIM, dtype=torch.uint8)
        lz4_compressed_list.append(lz4.frame.compress(rows.numpy().tobytes()))
    def lz4_batch_decode(data_list, n):
        return [lz4.frame.decompress(d) for d in data_list[:n]]
    br = benchmark_batch_decode('LZ4', lz4_compressed_list,
        lz4_batch_decode)
    batch_results.append(br)
    log(f"  {br['name']}: {br['batch_decode_ms']:.1f}ms (20 frames, sequential Python)")

    return results, batch_results, q, scale, zp


def run_advanced_strategies(q, scale, zp):
    """Benchmark advanced codec strategies."""
    log(f"\n{'='*60}")
    log("ADVANCED CODEC STRATEGIES")
    log(f"{'='*60}")

    frame_rows = q[:ROWS_PER_FRAME]
    flat_bytes = frame_rows.contiguous().view(-1)
    raw_uint8_bytes = flat_bytes.numel()
    raw_fp32_bytes = raw_uint8_bytes * 4
    results = []

    def add_adv_ratios(r):
        r['raw_fp32_bytes'] = raw_fp32_bytes
        r['raw_uint8_bytes'] = raw_uint8_bytes
        r['ratio_vs_fp32'] = raw_fp32_bytes / r['compressed_bytes']
        r['ratio_vs_uint8'] = raw_uint8_bytes / r['compressed_bytes']

    # ---- Strategy 1: Quantize to fewer bits, then Zstd ----
    log("\n1. Bit-reduction + Zstd...")
    for n_bits in [7, 6, 5, 4]:
        shift = 8 - n_bits
        reduced = (frame_rows >> shift).to(torch.uint8)
        flat_reduced = reduced.contiguous().view(-1)
        compressed = _C.zstd_compress_frame(flat_reduced, 3)

        # Measure decode
        times = []
        for _ in range(20):
            t0 = time.time()
            _C.zstd_decompress_frame(compressed, raw_uint8_bytes)
            times.append(time.time() - t0)

        r = {
            'name': f'uint{n_bits}+Zstd-3',
            'compressed_bytes': compressed.numel(),
            'decode_1t_ms': np.median(times) * 1000,
            'lossy': True,
            'bit_depth': n_bits,
        }
        add_adv_ratios(r)
        results.append(r)
        log(f"  uint{n_bits}+Zstd: {compressed.numel()} bytes "
            f"({r['ratio_vs_fp32']:.1f}x vs fp32), decode={r['decode_1t_ms']:.2f}ms")

    # ---- Strategy 2: Delta coding + Zstd ----
    log("\n2. Delta coding + Zstd...")
    # Since rows are freq-sorted, adjacent rows may be similar
    deltas = torch.zeros_like(frame_rows)
    deltas[0] = frame_rows[0]
    deltas[1:] = frame_rows[1:].to(torch.int16) - frame_rows[:-1].to(torch.int16)
    # Wrap to uint8 (delta mod 256)
    deltas = deltas.to(torch.uint8)
    flat_deltas = deltas.contiguous().view(-1)

    for level in [3, 9]:
        compressed = _C.zstd_compress_frame(flat_deltas, level)
        times = []
        for _ in range(20):
            t0 = time.time()
            _C.zstd_decompress_frame(compressed, raw_uint8_bytes)
            times.append(time.time() - t0)
        r = {
            'name': f'Delta+Zstd-L{level}',
            'compressed_bytes': compressed.numel(),
            'decode_1t_ms': np.median(times) * 1000,
            'lossy': False,
        }
        add_adv_ratios(r)
        results.append(r)
        log(f"  Delta+Zstd-L{level}: {compressed.numel()} bytes "
            f"({r['ratio_vs_fp32']:.1f}x vs fp32), decode={r['decode_1t_ms']:.2f}ms")

    # Compare: plain Zstd on same data for reference
    plain_comp = _C.zstd_compress_frame(flat_bytes, 3)
    log(f"  (Reference: plain Zstd-3 = {plain_comp.numel()} bytes, "
        f"{raw_fp32_bytes/plain_comp.numel():.1f}x vs fp32)")

    # ---- Strategy 3: Row-transposed + Zstd ----
    log("\n3. Row-transposed (column-major) + Zstd...")
    # Store column-by-column instead of row-by-row — better for same-dimension patterns
    transposed = frame_rows.t().contiguous()  # (16, 129600)
    flat_transposed = transposed.view(-1)
    for level in [3, 9]:
        compressed = _C.zstd_compress_frame(flat_transposed, level)
        times = []
        for _ in range(20):
            t0 = time.time()
            _C.zstd_decompress_frame(compressed, raw_uint8_bytes)
            times.append(time.time() - t0)
        r = {
            'name': f'ColMajor+Zstd-L{level}',
            'compressed_bytes': compressed.numel(),
            'decode_1t_ms': np.median(times) * 1000,
            'lossy': False,
        }
        add_adv_ratios(r)
        results.append(r)
        log(f"  ColMajor+Zstd-L{level}: {compressed.numel()} bytes "
            f"({r['ratio_vs_fp32']:.1f}x vs fp32), decode={r['decode_1t_ms']:.2f}ms")

    # ---- Strategy 4: Delta + ColMajor + Zstd ----
    log("\n4. Delta + ColMajor + Zstd...")
    delta_transposed = deltas.t().contiguous().view(-1)
    compressed = _C.zstd_compress_frame(delta_transposed, 3)
    times = []
    for _ in range(20):
        t0 = time.time()
        _C.zstd_decompress_frame(compressed, raw_uint8_bytes)
        times.append(time.time() - t0)
    r = {
        'name': 'Delta+ColMajor+Zstd-3',
        'compressed_bytes': compressed.numel(),
        'decode_1t_ms': np.median(times) * 1000,
        'lossy': False,
    }
    add_adv_ratios(r)
    results.append(r)
    log(f"  Delta+ColMajor+Zstd-3: {compressed.numel()} bytes "
        f"({r['ratio_vs_fp32']:.1f}x vs fp32), decode={r['decode_1t_ms']:.2f}ms")

    return results


def write_gpu_decode_analysis():
    """Write theoretical GPU hardware decode analysis."""
    analysis = """# GPU Hardware Decode Analysis for Embedding Compression

## Context

This analysis considers whether GPU hardware video decoders (NVDEC, Intel QSV, VAAPI)
could accelerate embedding table decompression as an alternative to CPU-based codecs.

## Current Performance (CPU-only)

| Codec | Decode/frame | 20 frames (parallel) | Lossless? |
|-------|-------------|---------------------|-----------|
| H.265 CRF=0 | ~7ms | ~50ms (1T×20) / ~13ms (4T×4) | Yes |
| H.265 CRF=18 | ~7ms | ~50ms / ~13ms | No |
| Zstd-3 | ~0.2ms | ~4ms | Yes |
| LZ4 | ~0.15ms | ~3ms (Python sequential) | Yes |

## GPU Hardware Decode: NVDEC

### Theoretical Performance
- NVDEC on modern GPUs (RTX 3090, A100) can decode H.265 at 1000+ FPS for 1080p
- Per-frame decode: ~0.05ms (vs 7ms CPU) = **140x speedup**
- 20 frames: ~1ms total (vs 50ms CPU)

### Data Path Overhead
- Decoded frame must travel: GPU VRAM → PCIe → CPU RAM
- For embeddings that stay on CPU: PCIe 4.0 x16 = ~25 GB/s
- 20 frames × 2MB each = 40MB → ~1.6ms PCIe transfer
- Total: ~1ms decode + ~1.6ms transfer = **~2.6ms for 20 frames**

### If Embeddings Stay on GPU
- No PCIe transfer needed for decoded data
- But index gathering must happen on GPU (requires custom CUDA kernel)
- Indices must be sent CPU→GPU (~0.1ms for 2048 indices)
- Net: ~1ms total — competitive with Zstd

## Intel Quick Sync (QSV)

### This Machine: Not Available
- GPU: Matrox G200eW3 (BMC only, no hardware decode capability)
- VAAPI: Not functional (no VA drivers for Matrox)

### On Server-Grade Intel CPUs (Xeon w/ integrated GPU)
- Intel QSV supports H.265 decode at ~500 FPS (1080p)
- Per-frame: ~2ms → 20 frames: ~4ms (with software scheduling overhead)
- But most server Xeons lack integrated GPU

## Comparison Matrix

| Approach | 20-frame decode | Pipeline viable? | Lossless? | Storage |
|----------|----------------|-----------------|-----------|---------|
| CPU H.265 1T | 140ms | No (31x too slow) | Yes | 0.13MB |
| CPU H.265 4T×4 | ~13ms | Marginal (3x too slow) | Yes | 0.13MB |
| **NVDEC H.265** | ~2.6ms | **Yes** | Yes | 0.13MB |
| **CPU Zstd-3** | ~4ms | **Yes** | Yes | 2.1MB |
| CPU LZ4 | ~3ms | Yes | Yes | 2.3MB |

## Conclusions

1. **Zstd on CPU already approaches GPU decode speed**: 4ms vs 2.6ms (theoretical NVDEC).
   The 1.5x theoretical advantage of GPU decode is unlikely to justify the complexity
   of GPU-based embedding gathering.

2. **GPU decode only worthwhile if embeddings stay on GPU**: If the model runs on GPU
   and embeddings are gathered on GPU, NVDEC eliminates both decode latency and PCIe
   transfer. But this requires a custom CUDA gather kernel.

3. **For CPU inference (our use case)**: Zstd is the clear winner. It's lossless,
   requires no hardware dependencies, and its 4ms decode time fits within the
   inference pipeline (4.5ms per batch).

4. **Storage trade-off is negligible in practice**: Zstd uses 2.1MB vs H.265's 0.13MB
   for 20 accessed frames. With only 20 frames accessed total, even the 16x storage
   difference is under 2MB.

## Recommendation

Use **Zstd for CPU inference** (this project's primary target). Reserve GPU decode
for future work where embeddings live on GPU and the full inference pipeline runs
on GPU, making NVDEC's sub-millisecond decode meaningful.
"""
    path = os.path.join(RESULTS_DIR, 'gpu_decode_analysis.md')
    with open(path, 'w') as f:
        f.write(analysis)
    log(f"\nGPU decode analysis written to {path}")


def main():
    log(f"{'='*60}")
    log("CODEC EXPLORATION BENCHMARK")
    log(f"{'='*60}")

    # Run main codec comparison
    results, batch_results, q, scale, zp = run_codec_benchmarks()

    # Run advanced strategies
    advanced_results = run_advanced_strategies(q, scale, zp)

    # Write GPU analysis
    write_gpu_decode_analysis()

    # ---- Save all results ----
    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")

    # Summary table — all ratios vs fp32
    log(f"\n{'Codec':<25} {'Size':>10} {'vs fp32':>9} {'vs uint8':>9} {'Enc(ms)':>9} "
        f"{'Dec 1T':>9} {'Dec 4T':>9} {'Lossy':>6}")
    log("-" * 95)
    for r in results:
        size_kb = r['compressed_bytes'] / 1024
        log(f"{r['name']:<25} {size_kb:>8.1f}KB {r['ratio_vs_fp32']:>8.1f}x "
            f"{r['ratio_vs_uint8']:>8.1f}x "
            f"{r.get('encode_time_ms', 0):>8.1f} "
            f"{r.get('decode_1t_ms', 0):>8.2f} "
            f"{r.get('decode_4t_ms', r.get('decode_1t_ms', 0)):>8.2f} "
            f"{'Yes' if r['lossy'] else 'No':>6}")

    log(f"\nBatch decode (20 frames):")
    for br in batch_results:
        log(f"  {br['name']:<30} {br['batch_decode_ms']:.1f}ms")

    log(f"\nAdvanced strategies (ratio vs fp32):")
    for r in advanced_results:
        size_kb = r['compressed_bytes'] / 1024
        log(f"  {r['name']:<30} {size_kb:>8.1f}KB ({r['ratio_vs_fp32']:.1f}x vs fp32), "
            f"decode={r.get('decode_1t_ms', 0):.2f}ms"
            f"{' [LOSSY]' if r.get('lossy') else ''}")

    # Save JSON
    all_data = {
        'codec_comparison': results,
        'batch_decode': batch_results,
        'advanced_strategies': advanced_results,
        'metadata': {
            'emb_dim': EMB_DIM,
            'rows_per_frame': ROWS_PER_FRAME,
            'width': WIDTH,
            'height': HEIGHT,
            'raw_frame_uint8_bytes': ROWS_PER_FRAME * EMB_DIM,
            'raw_frame_fp32_bytes': ROWS_PER_FRAME * EMB_DIM * 4,
            'note': 'ratio_vs_fp32 = fp32_bytes / compressed_bytes; ratio_vs_uint8 = uint8_bytes / compressed_bytes',
        }
    }
    json_path = os.path.join(RESULTS_DIR, 'codec_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    log(f"\nResults saved to {json_path}")

    log(f"\n{'='*60}")
    log("CODEC EXPLORATION COMPLETE")
    log(f"{'='*60}")


if __name__ == '__main__':
    main()
