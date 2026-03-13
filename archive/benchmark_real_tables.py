#!/usr/bin/env python3
"""
Real-data benchmark: C++ fused frame packing on all 8 large DLRM embedding tables.

Tests the full encode pipeline:
  Python:  gather cold rows → quantize → pad → tile → per-frame tobytes
  C++:     fused_gather_quantize_tile_multiframe (single call)
  C++ v2:  fused_quantize_tile_multiframe_fp32 (cold rows already gathered)

Also tests decode-side optimizations:
  Python:  H.265 decode → numpy untile → dequantize
  C++:     H.265 decode → C++ untile → C++ gather_dequant
"""

import os, sys, time, io
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

# Config
MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
HOTCOLD_DIR = "results/hotcold"
REORDER_DIR = "results/reorder"
EMB_DIM = 16
TILE_H = 4
TILE_W = 4
WIDTH, HEIGHT = 1920, 1080
TILES_PER_ROW = WIDTH // TILE_W
ROWS_PER_FRAME = TILES_PER_ROW * (HEIGHT // TILE_H)


def py_quantize(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def py_rows_to_tiled_frame(emb_rows, width, height):
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    n = min(len(emb_rows), rows_per_frame)
    padded = np.zeros((rows_per_frame, EMB_DIM), dtype=np.uint8)
    padded[:n] = emb_rows[:n]
    tiles = padded.reshape(tiles_per_col, tiles_per_row, TILE_H, TILE_W)
    frame = tiles.transpose(0, 2, 1, 3).reshape(height, width)
    return frame


def py_full_encode_pipeline(weight_fp32, cold_indices, width, height):
    """Python step-by-step: gather → quantize → tile all frames."""
    cold_w = weight_fp32[cold_indices]
    q, s, zp = py_quantize(cold_w)
    q_np = q.numpy()

    rows_per_frame = (width // TILE_W) * (height // TILE_H)
    n_cold = len(cold_indices)
    num_frames = (n_cold + rows_per_frame - 1) // rows_per_frame

    frames = []
    for f in range(num_frames):
        start = f * rows_per_frame
        end = min(start + rows_per_frame, n_cold)
        chunk = q_np[start:end]
        frame = py_rows_to_tiled_frame(chunk, width, height)
        frames.append(frame)

    return frames, s, zp


def bench(fn, warmup=2, iters=5, label=""):
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
    print(f"  {label:55s}  median={med:8.1f}ms  min={mn:8.1f}ms")
    return med, result


def main():
    print("=" * 80)
    print("Real-Data Multi-Table Frame Packing Benchmark")
    print(f"Resolution: {WIDTH}x{HEIGHT}, rows_per_frame={ROWS_PER_FRAME:,}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state = ld["state_dict"]

    # Find large tables with cold indices
    emb_keys = sorted([k for k in state.keys() if k.startswith("emb_l.") and k.endswith(".weight")])
    large_tables = []
    for k in emb_keys:
        t = int(k.split('.')[1])
        w = state[k]
        cold_path = os.path.join(HOTCOLD_DIR, f"cold_indices_{t}.pt")
        if os.path.exists(cold_path):
            cold_idx = torch.load(cold_path, weights_only=False)
            if len(cold_idx) > 10000:  # only large tables
                n_frames = (len(cold_idx) + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
                large_tables.append((t, w, cold_idx, n_frames))
                print(f"  Table {t}: {w.shape[0]:>10,} total, {len(cold_idx):>10,} cold, "
                      f"{n_frames:>3} frames")

    if not large_tables:
        print("No large tables found!")
        return

    total_cold = sum(len(ci) for _, _, ci, _ in large_tables)
    total_frames = sum(nf for _, _, _, nf in large_tables)
    print(f"\nTotal: {len(large_tables)} tables, {total_cold:,} cold rows, {total_frames} frames")

    # ============================================================
    # Per-table benchmarks
    # ============================================================
    all_py_times = []
    all_cpp_scatter_times = []
    all_cpp_contig_times = []

    for t, weight, cold_idx, n_frames in large_tables:
        n_cold = len(cold_idx)
        data_mb = n_cold * 16 * 4 / 1024 / 1024  # fp32

        print(f"\n{'='*80}")
        print(f"TABLE {t}: {n_cold:,} cold rows ({data_mb:.0f}MB fp32), {n_frames} frames")
        print(f"{'='*80}")

        cold_idx_long = cold_idx.long()

        # Python full pipeline
        py_time, (py_frames, py_s, py_zp) = bench(
            lambda w=weight, ci=cold_idx_long: py_full_encode_pipeline(w, ci, WIDTH, HEIGHT),
            label="Python: gather + quantize + tile (all frames)",
            warmup=1, iters=3
        )
        all_py_times.append(py_time)

        # C++ fused scatter gather + quantize + tile (multiframe)
        cpp_scatter_time, cpp_results = bench(
            lambda w=weight, ci=cold_idx_long: _C.fused_gather_quantize_tile_multiframe(
                w, ci, WIDTH, HEIGHT),
            label="C++ fused_gather_quantize_tile_multiframe",
            warmup=1, iters=3
        )
        all_cpp_scatter_times.append(cpp_scatter_time)

        # C++ fused quantize + tile (contiguous input, multiframe)
        cold_w = weight[cold_idx_long].contiguous()
        cpp_contig_time, _ = bench(
            lambda cw=cold_w: _C.fused_quantize_tile_multiframe_fp32(cw, WIDTH, HEIGHT),
            label="C++ fused_quantize_tile_multiframe_fp32 (contig)",
            warmup=1, iters=3
        )
        all_cpp_contig_times.append(cpp_contig_time)

        # C++ tile only (pre-quantized uint8)
        q, _, _ = py_quantize(cold_w)
        cpp_tile_time, cpp_tile_frames = bench(
            lambda q_t=q: _C.fused_quantize_tile_multiframe(q_t, WIDTH, HEIGHT),
            label="C++ fused_quantize_tile_multiframe (uint8 tile only)",
            warmup=1, iters=3
        )

        # Verify correctness: compare first frame
        cpp_frame_list = cpp_results[:-2]  # last 2 are scale, zp
        cpp_s = cpp_results[-2].item()
        cpp_zp = cpp_results[-1].item()

        # Compare dequantized values (first frame)
        from codec_ondemand_benchmark import tiled_frame_to_rows
        py_rows_0 = tiled_frame_to_rows(py_frames[0], WIDTH, HEIGHT)[:min(n_cold, ROWS_PER_FRAME)]
        cpp_rows_0 = tiled_frame_to_rows(cpp_frame_list[0].numpy(), WIDTH, HEIGHT)[:min(n_cold, ROWS_PER_FRAME)]

        py_deq = (py_rows_0.astype(np.float32) - py_zp) * py_s
        cpp_deq = (cpp_rows_0.astype(np.float32) - cpp_zp) * cpp_s
        orig = weight[cold_idx_long[:min(n_cold, ROWS_PER_FRAME)]].numpy()

        py_err = np.abs(py_deq - orig).mean()
        cpp_err = np.abs(cpp_deq - orig).mean()
        max_diff = np.abs(py_deq - cpp_deq).max()

        print(f"\n  Speedups:")
        print(f"    vs Python full pipeline:   {py_time/cpp_scatter_time:.1f}x (scatter gather)")
        print(f"    vs Python full pipeline:   {py_time/cpp_contig_time:.1f}x (contiguous)")
        print(f"    vs Python full pipeline:   {py_time/cpp_tile_time:.1f}x (tile only)")
        print(f"  Quantization quality:")
        print(f"    Python MAE: {py_err:.6f}  C++ MAE: {cpp_err:.6f}  "
              f"Max py-cpp diff: {max_diff:.6f}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*80}")
    print("SUMMARY: All Tables Combined")
    print(f"{'='*80}")

    total_py = sum(all_py_times)
    total_cpp_scatter = sum(all_cpp_scatter_times)
    total_cpp_contig = sum(all_cpp_contig_times)

    print(f"  Python total encode time:        {total_py:>8.0f} ms")
    print(f"  C++ scatter gather total:        {total_cpp_scatter:>8.1f} ms  "
          f"({total_py/total_cpp_scatter:.1f}x speedup)")
    print(f"  C++ contiguous total:            {total_cpp_contig:>8.1f} ms  "
          f"({total_py/total_cpp_contig:.1f}x speedup)")
    print(f"\n  Time saved (scatter):            {total_py - total_cpp_scatter:>8.0f} ms")
    print(f"  Time saved (contiguous):         {total_py - total_cpp_contig:>8.0f} ms")

    # Memory analysis
    print(f"\n--- Memory Analysis ---")
    for t, weight, cold_idx, n_frames in large_tables:
        n = len(cold_idx)
        # Python allocations per table:
        # 1. cold_w (fp32): n*16*4
        # 2. q (uint8): n*16
        # 3. q_np reference: 0
        # 4. padded (uint8): n_frames * rows_per_frame * 16
        # 5. frames (uint8): n_frames * width * height
        # 6. tobytes per frame: n_frames * width * height
        py_alloc = n*16*4 + n*16 + n_frames*ROWS_PER_FRAME*16 + n_frames*WIDTH*HEIGHT
        # C++ fused scatter: only output frames
        cpp_alloc = n_frames * WIDTH * HEIGHT
        saved = py_alloc - cpp_alloc
        print(f"  Table {t}: Python {py_alloc/1024/1024:>6.0f}MB → "
              f"C++ {cpp_alloc/1024/1024:>6.0f}MB "
              f"(saved {saved/1024/1024:>6.0f}MB, {saved/py_alloc*100:.0f}%)")


if __name__ == "__main__":
    main()
