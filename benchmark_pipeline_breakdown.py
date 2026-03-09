#!/usr/bin/env python3
"""
Detailed timing breakdown of compression and decompression pipelines.

Measures every sub-step independently to identify bottlenecks:

ENCODE (Compression):
  1. Gather: collect non-contiguous cold embeddings from full table (scattered fp32 reads)
  2. Quantize: compute min/max, then fp32 -> uint8 with scale/zero_point
  3. Tile: arrange (N,16) uint8 rows into (H,W) 2D frame (4x4 spatial tiling)
  4. Codec encode: H.264/H.265 compress the tiled frame
  5. Fused gather+quantize+tile: single C++ pass (for comparison)

DECODE (Decompression):
  1. File I/O: read compressed bytes from disk
  2. Codec decode: decompress to (H,W) uint8 frame
  3. Copy to tensor: memcpy decoded frame to torch tensor
  4. Untile: reshape (H,W) frame back to (N,16) rows
  5. Gather from tiled: extract K specific rows from tiled frame (skip full untile)
  6. Dequantize: uint8 -> fp32 with scale/zero_point
  7. Fused gather+dequant from tiled: single C++ pass (for comparison)
"""

import os
import sys
import time
import torch
import numpy as np
import tempfile
import shutil

# Load C++ extension
try:
    import compressed_emb as _C
    HAS_CPP = True
    print(f"C++ extension loaded: {_C.__file__}")
except ImportError:
    HAS_CPP = False
    print("WARNING: C++ extension not available")
    sys.exit(1)


def time_fn(fn, warmup=3, repeat=20, label=""):
    """Time a function with warmup and multiple repeats, return median in microseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        result = fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # ns -> us
    times.sort()
    median = times[len(times) // 2]
    return median, result


def load_real_data():
    """Load real DLRM embedding table and cold indices."""
    model_path = "./models/dlrm_kaggle_correct.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, using synthetic data")
        return None, None, None

    print(f"Loading model from {model_path}...")
    sd = torch.load(model_path, map_location='cpu', weights_only=False)['state_dict']

    # Find largest table
    table_sizes = {}
    for k, v in sd.items():
        if 'emb_l' in k and 'weight' in k:
            tid = int(k.split('.')[1])
            table_sizes[tid] = v.shape[0]

    # Use table 2 (largest, 10.1M rows)
    target_table = 2
    weight_key = f'emb_l.{target_table}.weight'
    weight = sd[weight_key].clone()  # (N, 16) fp32
    print(f"Table {target_table}: {weight.shape[0]:,} rows x {weight.shape[1]} dims")

    # Load cold indices
    cold_path = f"results/hotcold/cold_indices_{target_table}.pt"
    if os.path.exists(cold_path):
        cold_indices = torch.load(cold_path, map_location='cpu', weights_only=True)
        print(f"Cold indices: {cold_indices.shape[0]:,} rows")
    else:
        # Use last 90% as cold
        n = weight.shape[0]
        cold_indices = torch.arange(int(n * 0.1), n, dtype=torch.int64)
        print(f"Synthetic cold indices: {cold_indices.shape[0]:,} rows")

    return weight, cold_indices, target_table


def benchmark_encode_breakdown(weight, cold_indices, table_id):
    """Detailed breakdown of the encode (compression) pipeline."""
    print("\n" + "=" * 80)
    print("ENCODE (COMPRESSION) PIPELINE BREAKDOWN")
    print("=" * 80)

    N_cold = cold_indices.shape[0]
    D = weight.shape[1]
    width, height = 1920, 1080
    tiles_per_row = width // 4  # 480
    rows_per_frame = tiles_per_row * (height // 4)  # 129,600

    # Use first frame's worth of cold rows for single-frame benchmarks
    N_frame = min(N_cold, rows_per_frame)
    frame_cold_indices = cold_indices[:N_frame]

    print(f"\nTable {table_id}: {weight.shape[0]:,} total rows, {N_cold:,} cold rows")
    print(f"Frame: {width}x{height}, {rows_per_frame:,} rows/frame, D={D}")
    print(f"Benchmarking with {N_frame:,} cold rows (1 frame)")
    print()

    # -------------------------------------------------------------------------
    # Step 1: GATHER - collect scattered fp32 embeddings
    # -------------------------------------------------------------------------
    # The cold indices point to non-contiguous rows in the full embedding table.
    # This step measures the cost of gathering them into a contiguous buffer.

    def gather_scattered():
        return weight[frame_cold_indices]  # fancy indexing -> contiguous copy

    t_gather, gathered_fp32 = time_fn(gather_scattered, label="gather")
    print(f"  1. GATHER (scattered fp32 -> contiguous fp32)")
    print(f"     {N_frame:,} rows x {D} dims = {N_frame * D * 4 / 1e6:.1f} MB")
    print(f"     Time: {t_gather:.1f} us ({t_gather/1000:.3f} ms)")
    bw = (N_frame * D * 4) / (t_gather / 1e6) / 1e9  # GB/s
    print(f"     Effective bandwidth: {bw:.1f} GB/s (scattered reads)")

    # Step 1b: contiguous gather for comparison
    contiguous_rows = weight[:N_frame].clone()  # already contiguous
    def gather_contiguous():
        return weight[:N_frame].clone()
    t_contig, _ = time_fn(gather_contiguous, label="contiguous gather")
    print(f"     [Contiguous baseline: {t_contig:.1f} us ({t_contig/1000:.3f} ms)]")
    print(f"     Scatter overhead: {t_gather/t_contig:.2f}x")

    # -------------------------------------------------------------------------
    # Step 2: QUANTIZE - fp32 -> uint8 with min/max scale
    # -------------------------------------------------------------------------
    def quantize_python():
        """Python quantization (what the Python pipeline does)."""
        fmin = gathered_fp32.min().item()
        fmax = gathered_fp32.max().item()
        scale = (fmax - fmin) / 255.0
        if scale == 0:
            scale = 1.0
        zp = round(-fmin / scale)
        q = ((gathered_fp32 / scale).round() + zp).clamp(0, 255).to(torch.uint8)
        return q, scale, zp

    t_quant_py, (q_py, scale, zp) = time_fn(quantize_python, label="quantize_python")
    print(f"\n  2. QUANTIZE (fp32 -> uint8)")
    print(f"     Python path: {t_quant_py:.1f} us ({t_quant_py/1000:.3f} ms)")

    # Step 2b: Just min/max computation
    def compute_minmax():
        return gathered_fp32.min().item(), gathered_fp32.max().item()
    t_minmax, _ = time_fn(compute_minmax, label="minmax")
    print(f"     - min/max scan: {t_minmax:.1f} us ({t_minmax/1000:.3f} ms)")

    # Step 2c: Just the scale+round+clamp
    fmin = gathered_fp32.min().item()
    fmax = gathered_fp32.max().item()
    scale_val = (fmax - fmin) / 255.0
    zp_val = round(-fmin / scale_val)
    def apply_quant():
        return ((gathered_fp32 / scale_val).round() + zp_val).clamp(0, 255).to(torch.uint8)
    t_apply, q_result = time_fn(apply_quant, label="apply_quant")
    print(f"     - scale+round+clamp+cast: {t_apply:.1f} us ({t_apply/1000:.3f} ms)")

    # -------------------------------------------------------------------------
    # Step 3: TILE - arrange (N,16) uint8 rows into (H,W) 2D frame
    # -------------------------------------------------------------------------
    q_uint8 = q_result  # (N_frame, 16) uint8

    # Python tiling
    def tile_python():
        n = q_uint8.shape[0]
        q_np = q_uint8.numpy()
        # Pad to rows_per_frame
        padded = np.zeros((rows_per_frame, 16), dtype=np.uint8)
        padded[:n] = q_np
        # Reshape to tiles: (tile_rows, 4, tile_cols, 4) then transpose
        tr = height // 4
        tc = width // 4
        frame = padded.reshape(tr, tc, 4, 4).transpose(0, 2, 1, 3).reshape(height, width)
        return frame

    t_tile_py, frame_py = time_fn(tile_python, label="tile_python")
    print(f"\n  3. TILE (rows -> 2D frame)")
    print(f"     Python (pad+reshape+transpose+reshape): {t_tile_py:.1f} us ({t_tile_py/1000:.3f} ms)")

    # C++ tiling
    def tile_cpp():
        return _C.tile_rows_to_frame(q_uint8, width, height)

    t_tile_cpp, frame_cpp = time_fn(tile_cpp, label="tile_cpp")
    print(f"     C++ tile_rows_to_frame: {t_tile_cpp:.1f} us ({t_tile_cpp/1000:.3f} ms)")
    print(f"     Speedup: {t_tile_py/t_tile_cpp:.1f}x")

    # -------------------------------------------------------------------------
    # Step 4: CODEC ENCODE - compress tiled frame
    # -------------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="codec_bench_")
    frame_tensor = frame_cpp  # (H, W) uint8

    for codec in ['h264', 'h265']:
        ext = '.h264' if codec == 'h264' else '.h265'
        fpath = os.path.join(tmpdir, f"test{ext}")

        def encode_codec(c=codec, p=fpath):
            return _C.encode_frame_codec(frame_tensor, p, c, True, 0)

        t_enc, _ = time_fn(encode_codec, warmup=2, repeat=10, label=f"encode_{codec}")
        fsize = os.path.getsize(fpath)
        raw_size = height * width
        print(f"\n  4. CODEC ENCODE ({codec.upper()})")
        print(f"     Time: {t_enc:.1f} us ({t_enc/1000:.3f} ms)")
        print(f"     Compressed: {fsize:,} bytes ({fsize/1024:.1f} KB)")
        print(f"     Ratio: {raw_size/fsize:.1f}x")

    # -------------------------------------------------------------------------
    # Step 5: FUSED gather+quantize+tile (C++ single pass)
    # -------------------------------------------------------------------------
    def fused_gqt():
        return _C.fused_gather_quantize_tile(weight, frame_cold_indices, width, height)

    t_fused, (fused_frame, fused_s, fused_zp) = time_fn(fused_gqt, label="fused_gqt")
    total_separate = t_gather + t_quant_py + t_tile_py
    print(f"\n  5. FUSED gather+quantize+tile (C++ single pass)")
    print(f"     Time: {t_fused:.1f} us ({t_fused/1000:.3f} ms)")
    print(f"     vs separate Python steps: {total_separate:.1f} us ({total_separate/1000:.3f} ms)")
    print(f"     Speedup: {total_separate/t_fused:.1f}x")

    # Also measure fused for contiguous input (no gather needed)
    cold_rows_fp32 = weight[frame_cold_indices].clone()
    def fused_qt():
        return _C.fused_quantize_tile(cold_rows_fp32, width, height)

    t_fused_qt, _ = time_fn(fused_qt, label="fused_qt")
    print(f"     Fused quantize+tile (contiguous input): {t_fused_qt:.1f} us ({t_fused_qt/1000:.3f} ms)")
    print(f"     Gather overhead in fused: {t_fused - t_fused_qt:.1f} us ({(t_fused - t_fused_qt)/1000:.3f} ms)")

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("ENCODE SUMMARY (1 frame, {:,} rows)".format(N_frame))
    print("-" * 80)
    print(f"{'Step':<45} {'Time (us)':>10} {'Time (ms)':>10} {'% Total':>8}")
    print("-" * 80)

    # Python path total
    t_enc_h264 = time_fn(lambda: _C.encode_frame_codec(frame_tensor, os.path.join(tmpdir, "t.h264"), "h264", True, 0),
                         warmup=2, repeat=10)[0]
    t_enc_h265 = time_fn(lambda: _C.encode_frame_codec(frame_tensor, os.path.join(tmpdir, "t.h265"), "h265", True, 0),
                         warmup=2, repeat=10)[0]

    py_total = t_gather + t_quant_py + t_tile_py + t_enc_h264
    steps_py = [
        ("1. Gather (scattered fp32)", t_gather),
        ("2. Quantize (fp32 -> uint8)", t_quant_py),
        ("   2a. min/max scan", t_minmax),
        ("   2b. scale+round+clamp+cast", t_apply),
        ("3. Tile (rows -> frame)", t_tile_py),
        ("4. H.264 encode", t_enc_h264),
    ]
    for label, t in steps_py:
        pct = t / py_total * 100
        print(f"  {label:<43} {t:>10.1f} {t/1000:>10.3f} {pct:>7.1f}%")
    print(f"  {'TOTAL (Python path + H.264)':<43} {py_total:>10.1f} {py_total/1000:>10.3f} {'100.0%':>8}")

    print()
    # C++ fused path
    cpp_total = t_fused + t_enc_h264
    steps_cpp = [
        ("1-3. Fused gather+quantize+tile (C++)", t_fused),
        ("  - gather overhead (vs contiguous)", t_fused - t_fused_qt),
        ("  - quantize+tile (contiguous)", t_fused_qt),
        ("4. H.264 encode", t_enc_h264),
    ]
    for label, t in steps_cpp:
        pct = t / cpp_total * 100
        print(f"  {label:<43} {t:>10.1f} {t/1000:>10.3f} {pct:>7.1f}%")
    print(f"  {'TOTAL (C++ fused + H.264)':<43} {cpp_total:>10.1f} {cpp_total/1000:>10.3f} {'100.0%':>8}")

    print()
    h265_total = t_fused + t_enc_h265
    print(f"  {'TOTAL (C++ fused + H.265)':<43} {h265_total:>10.1f} {h265_total/1000:>10.3f}")

    shutil.rmtree(tmpdir)
    return frame_tensor, scale_val, zp_val


def benchmark_decode_breakdown(weight, cold_indices, table_id, frame_tensor, scale, zp):
    """Detailed breakdown of the decode (decompression) pipeline."""
    print("\n" + "=" * 80)
    print("DECODE (DECOMPRESSION) PIPELINE BREAKDOWN")
    print("=" * 80)

    N_cold = cold_indices.shape[0]
    width, height = 1920, 1080
    tiles_per_row = width // 4
    rows_per_frame = tiles_per_row * (height // 4)
    N_frame = min(N_cold, rows_per_frame)

    # Encode a frame to disk for decode benchmarks
    tmpdir = tempfile.mkdtemp(prefix="decode_bench_")

    for codec in ['h264', 'h265']:
        ext = '.h264' if codec == 'h264' else '.h265'
        fpath = os.path.join(tmpdir, f"frame_00000{ext}")
        _C.encode_frame_codec(frame_tensor, fpath, codec, True, 0)

    print(f"\nFrame: {width}x{height} = {width*height:,} bytes")
    print(f"Testing with K rows gathered from decoded frame")
    print()

    for codec in ['h264', 'h265']:
        ext = '.h264' if codec == 'h264' else '.h265'
        fpath = os.path.join(tmpdir, f"frame_00000{ext}")
        fsize = os.path.getsize(fpath)

        print(f"\n--- Codec: {codec.upper()} (file: {fsize:,} bytes = {fsize/1024:.1f} KB) ---")

        # -----------------------------------------------------------------
        # Step 1: FILE I/O - read compressed bytes from disk
        # -----------------------------------------------------------------
        def read_file():
            with open(fpath, 'rb') as f:
                return f.read()

        t_io, raw_bytes = time_fn(read_file, label="file_io")
        print(f"\n  1. FILE I/O (read {fsize:,} bytes)")
        print(f"     Time: {t_io:.1f} us ({t_io/1000:.3f} ms)")

        # -----------------------------------------------------------------
        # Step 2: CODEC DECODE (file -> (H,W) uint8 tensor)
        # This includes file I/O + codec decode + memcpy to tensor
        # -----------------------------------------------------------------
        def decode_from_file():
            return _C.decode_h265_frame_from_file(fpath)

        t_decode, decoded_frame = time_fn(decode_from_file, warmup=3, repeat=15,
                                           label="decode_from_file")
        print(f"\n  2. CODEC DECODE (file -> tensor, includes I/O + decode + memcpy)")
        print(f"     Time: {t_decode:.1f} us ({t_decode/1000:.3f} ms)")
        print(f"     Pure codec time (decode - I/O): ~{(t_decode - t_io):.1f} us ({(t_decode - t_io)/1000:.3f} ms)")

        # Step 2b: Decode from in-memory bytes (no file I/O)
        compressed_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
        def decode_from_memory():
            return _C.decode_h265_frame_from_bytes(compressed_tensor)

        t_decode_mem, _ = time_fn(decode_from_memory, warmup=3, repeat=15,
                                   label="decode_from_bytes")
        print(f"     From memory (no I/O): {t_decode_mem:.1f} us ({t_decode_mem/1000:.3f} ms)")

        # -----------------------------------------------------------------
        # Step 3: UNTILE - (H,W) frame back to (N,16) rows
        # -----------------------------------------------------------------
        # Python untile
        def untile_python():
            f_np = decoded_frame.numpy()
            tr = height // 4
            tc = width // 4
            rows = f_np.reshape(tr, 4, tc, 4).transpose(0, 2, 1, 3).reshape(rows_per_frame, 16)
            return rows

        t_untile_py, _ = time_fn(untile_python, label="untile_python")
        print(f"\n  3. UNTILE (frame -> rows)")
        print(f"     Python (reshape+transpose+reshape): {t_untile_py:.1f} us ({t_untile_py/1000:.3f} ms)")

        # C++ untile
        def untile_cpp():
            return _C.untile_frame_to_rows(decoded_frame, rows_per_frame)

        t_untile_cpp, untiled_rows = time_fn(untile_cpp, label="untile_cpp")
        print(f"     C++ untile_frame_to_rows: {t_untile_cpp:.1f} us ({t_untile_cpp/1000:.3f} ms)")
        print(f"     Speedup: {t_untile_py/t_untile_cpp:.1f}x")

        # -----------------------------------------------------------------
        # Step 4: GATHER from tiled frame (selective row extraction)
        # -----------------------------------------------------------------
        for K in [10, 100, 1000, 10000]:
            row_indices = torch.randperm(rows_per_frame)[:K].to(torch.int64).sort().values

            # C++ gather (from tiled frame, no untile needed)
            def gather_tiled():
                return _C.gather_from_tiled_frame(decoded_frame, row_indices, tiles_per_row)

            t_gather, _ = time_fn(gather_tiled, label=f"gather_K{K}")

            # Python path: untile everything then index
            def gather_python_full():
                rows = untile_cpp()  # use C++ untile for fair Python-path comparison
                return rows[row_indices]

            t_gather_py, _ = time_fn(gather_python_full, label=f"gather_py_K{K}")

            if K == 10:
                print(f"\n  4. GATHER from tiled frame (selective rows)")
            print(f"     K={K:>6}: C++ gather={t_gather:>8.1f}us ({t_gather/1000:.3f}ms), "
                  f"Python(untile+index)={t_gather_py:>8.1f}us ({t_gather_py/1000:.3f}ms), "
                  f"speedup={t_gather_py/t_gather:.1f}x")

        # -----------------------------------------------------------------
        # Step 5: DEQUANTIZE - uint8 -> fp32
        # -----------------------------------------------------------------
        K = 1000
        row_indices = torch.randperm(rows_per_frame)[:K].to(torch.int64).sort().values
        gathered_uint8 = _C.gather_from_tiled_frame(decoded_frame, row_indices, tiles_per_row)

        def dequant_python():
            return (gathered_uint8.float() - zp) * scale

        t_dequant_py, _ = time_fn(dequant_python, label="dequant_python")
        print(f"\n  5. DEQUANTIZE (uint8 -> fp32, K={K})")
        print(f"     Python (cast + sub + mul): {t_dequant_py:.1f} us ({t_dequant_py/1000:.3f} ms)")

        # -----------------------------------------------------------------
        # Step 6: FUSED gather+dequant from tiled frame (C++)
        # -----------------------------------------------------------------
        def fused_gather_dequant():
            return _C.gather_dequant_from_tiled_frame(
                decoded_frame, row_indices, tiles_per_row, scale, int(zp))

        t_fused_gd, _ = time_fn(fused_gather_dequant, label="fused_gather_dequant")
        t_separate = time_fn(lambda: _C.gather_from_tiled_frame(decoded_frame, row_indices, tiles_per_row))[0] + t_dequant_py
        print(f"\n  6. FUSED gather+dequant from tiled (C++, K={K})")
        print(f"     Fused: {t_fused_gd:.1f} us ({t_fused_gd/1000:.3f} ms)")
        print(f"     Separate (gather + dequant): {t_separate:.1f} us ({t_separate/1000:.3f} ms)")

        # -----------------------------------------------------------------
        # Step 7: FULL DECODE PIPELINE (fused: decode+gather+dequant)
        # -----------------------------------------------------------------
        def full_fused():
            return _C.decode_h265_gather_dequant(
                fpath, row_indices, tiles_per_row, scale, int(zp))

        t_full_fused, _ = time_fn(full_fused, warmup=3, repeat=15, label="full_fused")
        t_full_separate = t_decode + t_fused_gd
        print(f"\n  7. FULL PIPELINE: decode + gather + dequant (K={K})")
        print(f"     Fused C++ (single call): {t_full_fused:.1f} us ({t_full_fused/1000:.3f} ms)")
        print(f"     Sum of parts: decode={t_decode/1000:.3f}ms + gather+dequant={t_fused_gd/1000:.3f}ms = {t_full_separate/1000:.3f}ms")

        # -----------------------------------------------------------------
        # Summary
        # -----------------------------------------------------------------
        print(f"\n  {'─' * 70}")
        print(f"  DECODE SUMMARY ({codec.upper()}, K=1000 rows from 1 frame)")
        print(f"  {'─' * 70}")
        print(f"  {'Step':<45} {'Time (us)':>10} {'Time (ms)':>10} {'%':>6}")
        print(f"  {'─' * 70}")

        total = t_decode + t_fused_gd
        steps = [
            (f"1. File I/O ({fsize:,}B)", t_io),
            ("2. Codec decode (pure)", t_decode - t_io),
            ("3. Memcpy to tensor", 0),  # included in decode
            (f"4+5. Gather+dequant (K={K})", t_fused_gd),
        ]
        for label, t in steps:
            pct = t / total * 100 if total > 0 else 0
            print(f"    {label:<43} {t:>10.1f} {t/1000:>10.3f} {pct:>5.1f}%")
        print(f"    {'TOTAL':<43} {total:>10.1f} {total/1000:>10.3f} 100.0%")

    # Also test multi-K scaling of gather+dequant
    print(f"\n  {'─' * 70}")
    print(f"  GATHER+DEQUANT SCALING (from H.264 decoded frame)")
    print(f"  {'─' * 70}")
    fpath_h264 = os.path.join(tmpdir, "frame_00000.h264")
    decoded = _C.decode_h265_frame_from_file(fpath_h264)
    for K in [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 129600]:
        if K > rows_per_frame:
            continue
        row_indices = torch.randperm(rows_per_frame)[:K].to(torch.int64).sort().values

        t_gd, _ = time_fn(
            lambda ri=row_indices: _C.gather_dequant_from_tiled_frame(
                decoded, ri, tiles_per_row, scale, int(zp)),
            label=f"gd_K{K}")
        us_per_row = t_gd / K
        print(f"    K={K:>6}: {t_gd:>8.1f} us ({t_gd/1000:.4f} ms), {us_per_row:.3f} us/row")

    shutil.rmtree(tmpdir)


def benchmark_multiframe_encode(weight, cold_indices, table_id):
    """Benchmark encoding all cold rows across multiple frames."""
    print("\n" + "=" * 80)
    print("MULTI-FRAME ENCODE (Full Table Compression)")
    print("=" * 80)

    N_cold = cold_indices.shape[0]
    width, height = 1920, 1080
    rows_per_frame = (width // 4) * (height // 4)
    num_frames = (N_cold + rows_per_frame - 1) // rows_per_frame

    print(f"\nTable {table_id}: {N_cold:,} cold rows -> {num_frames} frames")
    print()

    tmpdir = tempfile.mkdtemp(prefix="multiframe_bench_")

    # Step 1: Gather all cold rows
    def gather_all():
        return weight[cold_indices]
    t_gather_all, cold_fp32 = time_fn(gather_all, warmup=2, repeat=5, label="gather_all")
    print(f"  1. Gather {N_cold:,} scattered rows: {t_gather_all/1000:.3f} ms")

    # Step 2: Quantize all
    def quantize_all():
        fmin = cold_fp32.min().item()
        fmax = cold_fp32.max().item()
        s = (fmax - fmin) / 255.0
        if s == 0: s = 1.0
        zp = round(-fmin / s)
        q = ((cold_fp32 / s).round() + zp).clamp(0, 255).to(torch.uint8)
        return q, s, zp
    t_quant_all, (q_all, s_all, zp_all) = time_fn(quantize_all, warmup=2, repeat=5, label="quant_all")
    print(f"  2. Quantize {N_cold:,} rows: {t_quant_all/1000:.3f} ms")

    # Step 3: Tile into frames
    def tile_all_frames():
        frames = []
        for f in range(num_frames):
            start = f * rows_per_frame
            end = min(start + rows_per_frame, N_cold)
            chunk = q_all[start:end]
            frame = _C.tile_rows_to_frame(chunk, width, height)
            frames.append(frame)
        return frames
    t_tile_all, tiled_frames = time_fn(tile_all_frames, warmup=2, repeat=5, label="tile_all")
    print(f"  3. Tile into {num_frames} frames: {t_tile_all/1000:.3f} ms")

    # Step 4: Encode all frames (H.264, parallel)
    def encode_all_h264():
        return _C.batch_encode_frames_codec(tiled_frames, tmpdir, "h264", True)
    t_enc_h264, total_bytes_h264 = time_fn(encode_all_h264, warmup=1, repeat=5, label="enc_h264")
    print(f"  4. H.264 encode {num_frames} frames (parallel): {t_enc_h264/1000:.3f} ms ({total_bytes_h264/1024:.0f} KB)")

    # Clean up h264 files before h265
    for f_name in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f_name))

    def encode_all_h265():
        return _C.batch_encode_frames_codec(tiled_frames, tmpdir, "h265", True)
    t_enc_h265, total_bytes_h265 = time_fn(encode_all_h265, warmup=1, repeat=3, label="enc_h265")
    print(f"  4. H.265 encode {num_frames} frames (parallel): {t_enc_h265/1000:.3f} ms ({total_bytes_h265/1024:.0f} KB)")

    # Step 5: Fused C++ path
    def fused_all_h264():
        fr, s_t, zp_t = _C.fused_gather_quantize_tile(weight, cold_indices[:rows_per_frame], width, height)
        return fr
    t_fused_one, _ = time_fn(fused_all_h264, warmup=2, repeat=5, label="fused_one")

    # Multi-frame fused
    if hasattr(_C, 'fused_gather_quantize_tile_multiframe'):
        def fused_multi():
            return _C.fused_gather_quantize_tile_multiframe(weight, cold_indices, width, height)
        t_fused_multi, fused_frames = time_fn(fused_multi, warmup=2, repeat=5, label="fused_multi")
        print(f"  5. Fused gather+quant+tile {num_frames} frames (C++): {t_fused_multi/1000:.3f} ms")
    else:
        t_fused_multi = t_fused_one * num_frames  # estimate
        print(f"  5. (fused_gather_quantize_tile_multiframe not available)")

    # Summary
    print(f"\n  {'─' * 70}")
    print(f"  MULTI-FRAME ENCODE SUMMARY (Table {table_id}, {N_cold:,} rows, {num_frames} frames)")
    print(f"  {'─' * 70}")

    py_total = t_gather_all + t_quant_all + t_tile_all + t_enc_h264
    print(f"  {'Step':<45} {'H.264 (ms)':>10} {'%':>6}")
    print(f"  {'─' * 70}")
    print(f"    {'1. Gather scattered rows':<43} {t_gather_all/1000:>10.2f} {t_gather_all/py_total*100:>5.1f}%")
    print(f"    {'2. Quantize fp32->uint8':<43} {t_quant_all/1000:>10.2f} {t_quant_all/py_total*100:>5.1f}%")
    print(f"    {'3. Tile into frames':<43} {t_tile_all/1000:>10.2f} {t_tile_all/py_total*100:>5.1f}%")
    print(f"    {'4. Codec encode (parallel)':<43} {t_enc_h264/1000:>10.2f} {t_enc_h264/py_total*100:>5.1f}%")
    print(f"    {'TOTAL':<43} {py_total/1000:>10.2f} 100.0%")

    shutil.rmtree(tmpdir)


def main():
    print("=" * 80)
    print("CODEC PIPELINE DETAILED TIMING BREAKDOWN")
    print("=" * 80)
    print(f"Torch threads: {torch.get_num_threads()}")
    print(f"AVX-512: likely available (Xeon Platinum)")

    weight, cold_indices, table_id = load_real_data()
    if weight is None:
        print("No real data available, exiting")
        return

    frame_tensor, scale, zp = benchmark_encode_breakdown(weight, cold_indices, table_id)
    benchmark_decode_breakdown(weight, cold_indices, table_id, frame_tensor, scale, zp)
    benchmark_multiframe_encode(weight, cold_indices, table_id)


if __name__ == "__main__":
    main()
