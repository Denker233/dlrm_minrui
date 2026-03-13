#!/usr/bin/env python3
"""
Benchmark: Frame packing/unpacking overhead — Python vs C++ implementations.

Measures the cost of:
1. Tiling: rows (N, 16) → tiled frame (H, W)
2. Untiling: tiled frame (H, W) → rows (N, 16)
3. Selective gather from tiled frame (K << N rows)
4. Fused gather + quantize + tile (scattered fp32 → tiled frame)
5. Fused quantize + tile (contiguous fp32 → tiled frame)
6. End-to-end encode pipeline: quantize + tile + pipe to ffmpeg

Uses real embedding data from the DLRM cold tables when available,
falls back to synthetic data otherwise.
"""

import os, sys, time, io, subprocess, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import compressed_emb as _C
    HAS_CPP = True
    print("C++ extension loaded successfully")
except ImportError as e:
    HAS_CPP = False
    print(f"WARNING: C++ extension not available: {e}")

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False
    print("WARNING: PyAV not available")


# ============================================================
# Configuration
# ============================================================
EMB_DIM = 16
TILE_H = 4
TILE_W = 4
WARMUP = 3
ITERS = 20

RESOLUTIONS = {
    '1080p': (1920, 1080),
    '4K':    (3840, 2160),
}

HOTCOLD_DIR = "results/hotcold"
MODEL_PATH = "./models/dlrm_kaggle_correct.pt"


# ============================================================
# Python reference implementations (from codec_ondemand_benchmark.py)
# ============================================================
def py_rows_to_tiled_frame(emb_rows, width, height):
    """Python: (N, 16) uint8 → (H, W) uint8 tiled frame."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    tiles = emb_rows[:rows_per_frame].reshape(tiles_per_col, tiles_per_row, TILE_H, TILE_W)
    frame = tiles.transpose(0, 2, 1, 3).reshape(height, width)
    return frame


def py_tiled_frame_to_rows(frame, width, height):
    """Python: (H, W) uint8 → (N, 16) uint8."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    grid = frame.reshape(tiles_per_col, TILE_H, tiles_per_row, TILE_W)
    rows = grid.transpose(0, 2, 1, 3).reshape(rows_per_frame, EMB_DIM)
    return rows


def py_quantize(w_fp32):
    """Python: fp32 → uint8 + scale + zp."""
    mn = w_fp32.min().item()
    mx = w_fp32.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((w_fp32 / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def py_gather_from_tiled(frame_np, row_indices, tiles_per_row):
    """Python: gather K rows from tiled frame (slow, per-row)."""
    width = frame_np.shape[1]
    K = len(row_indices)
    out = np.empty((K, 16), dtype=np.uint8)
    for k, r in enumerate(row_indices):
        ty = r // tiles_per_row
        tx = r % tiles_per_row
        for ly in range(4):
            out[k, ly*4:ly*4+4] = frame_np[ty*4+ly, tx*4:tx*4+4]
    return out


def py_full_untile_then_gather(frame_np, row_indices, width, height):
    """Python: untile entire frame, then index specific rows."""
    all_rows = py_tiled_frame_to_rows(frame_np, width, height)
    return all_rows[row_indices]


# ============================================================
# Benchmark helpers
# ============================================================
def bench(fn, warmup=WARMUP, iters=ITERS, label=""):
    """Run function, return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    med = np.median(times)
    mn = np.min(times)
    mx = np.max(times)
    print(f"  {label:55s}  median={med:8.3f}ms  min={mn:8.3f}ms  max={mx:8.3f}ms")
    return med, result


def load_real_data():
    """Try to load real cold embedding data from saved tables."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        state = ld["state_dict"]
        # Table 2 is the largest
        key = "emb_l.2.weight"
        if key in state:
            w = state[key]  # (N, 16) fp32
            cold_idx_path = os.path.join(HOTCOLD_DIR, "cold_indices_2.pt")
            if os.path.exists(cold_idx_path):
                cold_idx = torch.load(cold_idx_path, weights_only=False)
                print(f"Loaded real data: table 2 ({w.shape[0]} rows), "
                      f"{len(cold_idx)} cold indices")
                return w, cold_idx
            return w, None
    except Exception as e:
        print(f"Could not load real data: {e}")
    return None, None


# ============================================================
# Main benchmarks
# ============================================================
def benchmark_tiling(res_name, width, height, num_rows=None):
    """Benchmark: rows → tiled frame and tiled frame → rows."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col

    if num_rows is None:
        num_rows = rows_per_frame

    print(f"\n{'='*80}")
    print(f"TILING BENCHMARK: {res_name} ({width}x{height}), "
          f"{num_rows:,} rows, {rows_per_frame:,} rows/frame")
    print(f"{'='*80}")

    # Create test data
    emb_uint8_np = np.random.randint(0, 256, (num_rows, 16), dtype=np.uint8)
    emb_uint8_t = torch.from_numpy(emb_uint8_np.copy())

    # === Tiling: rows → frame ===
    print("\n--- Tile: rows (N, 16) → frame (H, W) ---")

    py_time, py_frame = bench(
        lambda: py_rows_to_tiled_frame(emb_uint8_np, width, height),
        label="Python (reshape+transpose+reshape)"
    )

    if HAS_CPP:
        cpp_time, cpp_frame_t = bench(
            lambda: _C.tile_rows_to_frame(emb_uint8_t, width, height),
            label="C++ tile_rows_to_frame"
        )
        # Verify correctness
        cpp_frame_np = cpp_frame_t.numpy()
        if np.array_equal(py_frame, cpp_frame_np):
            print(f"  ✓ C++ matches Python. Speedup: {py_time/cpp_time:.2f}x")
        else:
            diff = np.sum(py_frame != cpp_frame_np)
            print(f"  ✗ MISMATCH: {diff} pixels differ!")

    # === Untiling: frame → rows ===
    print("\n--- Untile: frame (H, W) → rows (N, 16) ---")

    frame_np = py_frame.copy()
    frame_t = torch.from_numpy(frame_np.copy())

    py_time, py_rows = bench(
        lambda: py_tiled_frame_to_rows(frame_np, width, height),
        label="Python (reshape+transpose+reshape)"
    )

    if HAS_CPP:
        cpp_time, cpp_rows_t = bench(
            lambda: _C.untile_frame_to_rows(frame_t, num_rows),
            label="C++ untile_frame_to_rows"
        )
        cpp_rows_np = cpp_rows_t.numpy()
        if np.array_equal(py_rows[:num_rows], cpp_rows_np):
            print(f"  ✓ C++ matches Python. Speedup: {py_time/cpp_time:.2f}x")
        else:
            diff = np.sum(py_rows[:num_rows] != cpp_rows_np)
            print(f"  ✗ MISMATCH: {diff} values differ!")

    return py_frame


def benchmark_selective_gather(res_name, width, height, frame_np, gather_sizes=[10, 100, 1000, 10000]):
    """Benchmark: gather K rows from tiled frame (K << N)."""
    tiles_per_row = width // TILE_W
    rows_per_frame = tiles_per_row * (height // TILE_H)
    frame_t = torch.from_numpy(frame_np.copy())

    print(f"\n{'='*80}")
    print(f"SELECTIVE GATHER from tiled frame: {res_name}")
    print(f"{'='*80}")

    for K in gather_sizes:
        if K > rows_per_frame:
            continue
        indices = np.sort(np.random.choice(rows_per_frame, K, replace=False))
        indices_t = torch.from_numpy(indices.copy()).long()

        print(f"\n--- Gather K={K:,} rows from {rows_per_frame:,} total ---")

        # Method 1: Full untile + index
        full_time, full_result = bench(
            lambda: py_full_untile_then_gather(frame_np, indices, width, height),
            label="Python: full untile + index"
        )

        # Method 2: Python per-row gather from tiled
        if K <= 1000:
            per_row_time, per_row_result = bench(
                lambda: py_gather_from_tiled(frame_np, indices, tiles_per_row),
                label="Python: per-row tiled gather"
            )
        else:
            per_row_time = float('inf')

        if HAS_CPP:
            # Method 3: C++ untile entire frame + index
            cpp_untile_time, _ = bench(
                lambda: _C.untile_frame_to_rows(frame_t, rows_per_frame)[indices_t],
                label="C++ untile_frame_to_rows + index"
            )

            # Method 4: C++ selective gather (best for small K)
            cpp_gather_time, cpp_gathered = bench(
                lambda: _C.gather_from_tiled_frame(frame_t, indices_t, tiles_per_row),
                label="C++ gather_from_tiled_frame"
            )

            # Method 5: C++ gather + dequantize fused
            scale, zp = 0.01, 128
            cpp_gd_time, cpp_gd = bench(
                lambda: _C.gather_dequant_from_tiled_frame(
                    frame_t, indices_t, tiles_per_row, scale, zp),
                label="C++ gather_dequant_from_tiled_frame"
            )

            # Verify
            cpp_np = cpp_gathered.numpy()
            if np.array_equal(full_result, cpp_np):
                print(f"  ✓ C++ gather correct. vs full untile: {full_time/cpp_gather_time:.2f}x")
            else:
                diff = np.sum(full_result != cpp_np)
                print(f"  ✗ MISMATCH: {diff} values differ!")


def benchmark_fused_encode(res_name, width, height, weight_fp32, cold_indices):
    """Benchmark: fused gather + quantize + tile vs step-by-step Python."""
    tiles_per_row = width // TILE_W
    rows_per_frame = tiles_per_row * (height // TILE_H)

    # Limit to one frame worth of rows
    n_cold = min(len(cold_indices), rows_per_frame)
    cold_idx = cold_indices[:n_cold].long()

    print(f"\n{'='*80}")
    print(f"FUSED ENCODE BENCHMARK: {res_name} ({width}x{height}), {n_cold:,} cold rows")
    print(f"{'='*80}")

    # === Python step-by-step ===
    print("\n--- Step-by-step Python encode ---")

    def py_stepwise():
        # Step 1: Gather cold rows
        cold_w = weight_fp32[cold_idx]
        # Step 2: Quantize
        q, s, zp = py_quantize(cold_w)
        q_np = q.numpy()
        # Step 3: Pad
        padded = np.zeros((rows_per_frame, EMB_DIM), dtype=np.uint8)
        padded[:n_cold] = q_np
        # Step 4: Tile
        frame = py_rows_to_tiled_frame(padded, width, height)
        return frame, s, zp

    py_time, (py_frame, py_s, py_zp) = bench(py_stepwise, label="Python: gather + quantize + pad + tile")

    # === Python: gather then quantize then tile (skip pad for timing) ===
    def py_no_pad():
        cold_w = weight_fp32[cold_idx]
        q, s, zp = py_quantize(cold_w)
        q_np = q.numpy()
        frame = py_rows_to_tiled_frame(q_np, width, height)
        return frame, s, zp

    py_nopad_time, _ = bench(py_no_pad, label="Python: gather + quantize + tile (no explicit pad)")

    if HAS_CPP:
        print("\n--- C++ fused encode ---")

        # Method 1: C++ fused gather + quantize + tile (from scattered fp32)
        cpp_fused_time, (cpp_frame, cpp_s_t, cpp_zp_t) = bench(
            lambda: _C.fused_gather_quantize_tile(weight_fp32, cold_idx, width, height),
            label="C++ fused_gather_quantize_tile"
        )

        # Method 2: C++ fused quantize + tile (from contiguous fp32)
        cold_w_contiguous = weight_fp32[cold_idx].contiguous()
        cpp_qt_time, _ = bench(
            lambda: _C.fused_quantize_tile(cold_w_contiguous, width, height),
            label="C++ fused_quantize_tile (contiguous input)"
        )

        # Method 3: C++ tile only (pre-quantized uint8)
        q_pre, _, _ = py_quantize(cold_w_contiguous)
        padded_t = torch.zeros(rows_per_frame, EMB_DIM, dtype=torch.uint8)
        padded_t[:n_cold] = q_pre
        cpp_tile_time, _ = bench(
            lambda: _C.tile_rows_to_frame(padded_t, width, height),
            label="C++ tile_rows_to_frame (pre-quantized)"
        )

        # Method 4: C++ multi-frame tile (pre-quantized)
        cpp_mf_time, _ = bench(
            lambda: _C.fused_quantize_tile_multiframe(padded_t, width, height),
            label="C++ fused_quantize_tile_multiframe"
        )

        print(f"\n--- Speedups ---")
        print(f"  Full pipeline (py step-by-step vs C++ fused):  {py_time/cpp_fused_time:.2f}x")
        print(f"  Quantize+tile (py vs C++ fused):                {py_nopad_time/cpp_qt_time:.2f}x")
        print(f"  Tile only (py vs C++):                          "
              f"(see tiling benchmark above)")

        # Correctness: compare quantized values
        # Note: exact match unlikely due to different rounding, check approximate
        cpp_frame_np = cpp_frame.numpy()
        cpp_s = cpp_s_t.item()
        cpp_zp = cpp_zp_t.item()
        # Untile both and compare
        py_rows = py_tiled_frame_to_rows(py_frame, width, height)[:n_cold]
        cpp_rows = py_tiled_frame_to_rows(cpp_frame_np, width, height)[:n_cold]
        # Dequantize both
        py_deq = (py_rows.astype(np.float32) - py_zp) * py_s
        cpp_deq = (cpp_rows.astype(np.float32) - cpp_zp) * cpp_s
        # Compare original fp32 values
        orig = weight_fp32[cold_idx].numpy()
        py_err = np.abs(py_deq - orig).mean()
        cpp_err = np.abs(cpp_deq - orig).mean()
        print(f"\n--- Quantization quality ---")
        print(f"  Python mean abs error:  {py_err:.6f}")
        print(f"  C++ mean abs error:     {cpp_err:.6f}")


def benchmark_decode_pipeline(res_name, width, height, frame_np):
    """Benchmark the decode side: H.265 decode → untile → dequantize."""
    tiles_per_row = width // TILE_W
    rows_per_frame = tiles_per_row * (height // TILE_H)
    frame_t = torch.from_numpy(frame_np.copy())

    print(f"\n{'='*80}")
    print(f"DECODE PIPELINE BENCHMARK: {res_name}")
    print(f"{'='*80}")

    scale, zp = 0.01, 128

    # Full untile + dequantize
    print("\n--- Full frame untile + dequantize ---")

    def py_decode():
        rows = py_tiled_frame_to_rows(frame_np, width, height)
        fp32 = (rows.astype(np.float32) - zp) * scale
        return fp32

    py_time, _ = bench(py_decode, label="Python: untile + dequant (full frame)")

    if HAS_CPP:
        def cpp_decode():
            rows_t = _C.untile_frame_to_rows(frame_t, rows_per_frame)
            # Dequantize using gather_dequant_uint8 (full frame)
            all_idx = torch.arange(rows_per_frame, dtype=torch.long)
            return _C.gather_dequant_uint8(rows_t, all_idx, scale, zp)

        cpp_time, _ = bench(cpp_decode, label="C++ untile + gather_dequant_uint8")

        # Fused gather+dequant from tiled frame (for specific rows)
        for K in [100, 1000, 10000]:
            if K > rows_per_frame:
                continue
            indices = torch.from_numpy(
                np.sort(np.random.choice(rows_per_frame, K, replace=False))
            ).long()

            def cpp_fused_gd(idx=indices):
                return _C.gather_dequant_from_tiled_frame(
                    frame_t, idx, tiles_per_row, scale, zp)

            cpp_gd_time, _ = bench(
                cpp_fused_gd,
                label=f"C++ gather_dequant_from_tiled_frame (K={K})"
            )

            # Compare: untile whole frame then index
            def py_untile_idx(idx=indices):
                rows = py_tiled_frame_to_rows(frame_np, width, height)
                sel = rows[idx.numpy()]
                fp32 = (sel.astype(np.float32) - zp) * scale
                return fp32

            py_idx_time, _ = bench(
                py_untile_idx,
                label=f"Python: full untile + index + dequant (K={K})"
            )
            print(f"  → Speedup for K={K}: {py_idx_time/cpp_gd_time:.2f}x")


def benchmark_memory_copies():
    """Measure the number of memory allocations and total bytes copied."""
    print(f"\n{'='*80}")
    print(f"MEMORY COPY ANALYSIS")
    print(f"{'='*80}")

    width, height = 1920, 1080
    tiles_per_row = width // TILE_W
    rows_per_frame = tiles_per_row * (height // TILE_H)  # 129600

    # Simulated data
    N = rows_per_frame
    emb_fp32 = torch.randn(N, 16)
    data_bytes = N * 16

    print(f"\n  Frame: {width}x{height}, {N:,} rows, {data_bytes/1024/1024:.1f}MB raw (fp32)")

    # Python pipeline: measure each step
    print("\n--- Python encode pipeline memory trace ---")

    t0 = time.perf_counter()
    # Step 1: Quantize
    q, s, zp = py_quantize(emb_fp32)  # Creates intermediate tensors + uint8 output
    t1 = time.perf_counter()
    print(f"  1. Quantize fp32→uint8:    {(t1-t0)*1000:8.3f}ms  "
          f"output: {q.numel()*q.element_size()/1024/1024:.1f}MB "
          f"(+intermediates ~{N*16*4/1024/1024:.0f}MB)")

    t0 = time.perf_counter()
    q_np = q.numpy()  # Zero-copy if contiguous
    t1 = time.perf_counter()
    print(f"  2. .numpy() conversion:    {(t1-t0)*1000:8.3f}ms  (zero-copy)")

    t0 = time.perf_counter()
    padded = np.zeros((rows_per_frame, EMB_DIM), dtype=np.uint8)
    padded[:N] = q_np
    t1 = time.perf_counter()
    print(f"  3. Pad + copy:             {(t1-t0)*1000:8.3f}ms  "
          f"alloc: {padded.nbytes/1024/1024:.1f}MB")

    t0 = time.perf_counter()
    frame = py_rows_to_tiled_frame(padded, width, height)
    t1 = time.perf_counter()
    print(f"  4. Tile (transpose copy):  {(t1-t0)*1000:8.3f}ms  "
          f"alloc: {frame.nbytes/1024/1024:.1f}MB")

    t0 = time.perf_counter()
    frame_bytes = frame.tobytes()
    t1 = time.perf_counter()
    print(f"  5. .tobytes():             {(t1-t0)*1000:8.3f}ms  "
          f"alloc: {len(frame_bytes)/1024/1024:.1f}MB")

    total_alloc = (N*16*4 + q.numel() + padded.nbytes + frame.nbytes + len(frame_bytes))
    print(f"\n  Total allocations: ~{total_alloc/1024/1024:.0f}MB  "
          f"({total_alloc / (N*16):.1f}x the raw uint8 data)")

    if HAS_CPP:
        print("\n--- C++ fused encode memory trace ---")
        t0 = time.perf_counter()
        results = _C.fused_quantize_tile(emb_fp32.contiguous(), width, height)
        t1 = time.perf_counter()
        cpp_frame = results[0]
        cpp_alloc = cpp_frame.numel()  # Just the output frame
        print(f"  1. fused_quantize_tile:    {(t1-t0)*1000:8.3f}ms  "
              f"output: {cpp_alloc/1024/1024:.1f}MB")
        print(f"\n  Total allocations: ~{cpp_alloc/1024/1024:.0f}MB  "
              f"({cpp_alloc / (N*16):.1f}x the raw uint8 data)")
        print(f"  Memory saved: ~{(total_alloc - cpp_alloc)/1024/1024:.0f}MB "
              f"({(1 - cpp_alloc/total_alloc)*100:.0f}% reduction)")


def benchmark_end_to_end_with_codec(res_name, width, height, frame_np):
    """Benchmark full encode+decode cycle including H.265 codec."""
    if not HAS_AV:
        print("Skipping codec benchmark (no PyAV)")
        return

    print(f"\n{'='*80}")
    print(f"END-TO-END CODEC BENCHMARK: {res_name}")
    print(f"{'='*80}")

    tiles_per_row = width // TILE_W
    rows_per_frame = tiles_per_row * (height // TILE_H)
    frame_t = torch.from_numpy(frame_np.copy())

    # Encode to H.265 using PyAV
    print("\n--- Encode single frame to H.265 ---")

    def encode_pyav(frame_data):
        output = io.BytesIO()
        container = av.open(output, mode='w', format='matroska')
        stream = container.add_stream('libx265', rate=1)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'gray'
        stream.options = {
            'preset': 'ultrafast',
            'x265-params': 'lossless=1:log-level=error'
        }
        avframe = av.VideoFrame.from_ndarray(frame_data, format='gray')
        for packet in stream.encode(avframe):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return output.getvalue()

    # Python: tile + encode
    def py_tile_encode():
        frame_2d = py_rows_to_tiled_frame(
            np.random.randint(0, 256, (rows_per_frame, 16), dtype=np.uint8),
            width, height
        )
        return encode_pyav(frame_2d)

    py_enc_time, compressed = bench(py_tile_encode, iters=5, warmup=1,
                                     label="Python: tile + PyAV encode")

    if HAS_CPP:
        def cpp_tile_encode():
            rows = torch.randint(0, 256, (rows_per_frame, 16), dtype=torch.uint8)
            frame_2d = _C.tile_rows_to_frame(rows, width, height)
            return encode_pyav(np.ascontiguousarray(frame_2d.numpy()))

        try:
            cpp_enc_time, _ = bench(cpp_tile_encode, iters=5, warmup=1,
                                     label="C++ tile + PyAV encode")
            print(f"  Tile overhead reduction: {(py_enc_time - cpp_enc_time):.1f}ms saved")
        except Exception as e:
            print(f"  C++ tile + PyAV encode: SKIPPED ({e})")

    # Decode
    print(f"\n--- Decode single frame from H.265 ({len(compressed)/1024:.1f}KB) ---")

    def decode_pyav_untile_py():
        container = av.open(io.BytesIO(compressed))
        frame = next(container.decode(video=0))
        arr = frame.to_ndarray(format='gray')
        container.close()
        rows = py_tiled_frame_to_rows(arr, width, height)
        return rows

    py_dec_time, _ = bench(decode_pyav_untile_py, iters=5, warmup=1,
                            label="Python: PyAV decode + untile")

    if HAS_CPP:
        def decode_pyav_untile_cpp():
            container = av.open(io.BytesIO(compressed))
            frame = next(container.decode(video=0))
            arr = frame.to_ndarray(format='gray')
            container.close()
            arr_t = torch.from_numpy(arr)
            rows = _C.untile_frame_to_rows(arr_t, rows_per_frame)
            return rows

        cpp_dec_time, _ = bench(decode_pyav_untile_cpp, iters=5, warmup=1,
                                 label="C++ PyAV decode + untile")
        print(f"  Untile overhead reduction: {(py_dec_time - cpp_dec_time):.1f}ms saved")

        # Selective decode: only gather needed rows
        for K in [100, 1000]:
            indices = torch.from_numpy(
                np.sort(np.random.choice(rows_per_frame, K, replace=False))
            ).long()

            def decode_gather_only(idx=indices):
                container = av.open(io.BytesIO(compressed))
                frame = next(container.decode(video=0))
                arr = frame.to_ndarray(format='gray')
                container.close()
                arr_t = torch.from_numpy(arr)
                return _C.gather_from_tiled_frame(arr_t, idx, tiles_per_row)

            gath_time, _ = bench(decode_gather_only, iters=5, warmup=1,
                                  label=f"C++ PyAV decode + gather K={K} (skip untile)")
            print(f"  → vs full untile: {py_dec_time/gath_time:.2f}x faster")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Frame Packing/Unpacking Benchmark: Python vs C++")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print("=" * 80)

    # Load real data if available
    weight_fp32, cold_indices = load_real_data()

    for res_name, (width, height) in RESOLUTIONS.items():
        # Basic tiling
        frame_np = benchmark_tiling(res_name, width, height)

        # Selective gather
        benchmark_selective_gather(res_name, width, height, frame_np,
                                   gather_sizes=[10, 100, 1000, 10000, 50000])

        # Decode pipeline
        benchmark_decode_pipeline(res_name, width, height, frame_np)

        # Fused encode
        if weight_fp32 is not None and cold_indices is not None:
            benchmark_fused_encode(res_name, width, height, weight_fp32, cold_indices)
        else:
            # Use synthetic data
            rows_per_frame = (width // TILE_W) * (height // TILE_H)
            synth_weight = torch.randn(rows_per_frame + 10000, 16)
            synth_cold = torch.arange(10000, 10000 + rows_per_frame).long()
            benchmark_fused_encode(res_name, width, height, synth_weight, synth_cold)

        # End-to-end with codec
        benchmark_end_to_end_with_codec(res_name, width, height, frame_np)

    # Memory analysis
    benchmark_memory_copies()

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print("=" * 80)
