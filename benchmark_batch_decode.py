#!/usr/bin/env python3
"""
Benchmark batch decode pipeline: parallel multi-frame H.265 decode + vectorized gather.

Tests the key inference scenario: a batch of queries needs embeddings from
multiple H.265 compressed frames. Compare:
1. Serial decode (current): decode frames one-by-one
2. Parallel batch decode (new): decode all needed frames simultaneously
3. Fused batch decode+gather: C++ does everything in one call
"""

import os, sys, time, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C
import av

EMB_DIM = 16
WIDTH, HEIGHT = 1920, 1080
ROWS_PER_FRAME = (WIDTH // 4) * (HEIGHT // 4)
TILES_PER_ROW = WIDTH // 4


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
    p99 = np.percentile(times, 99)
    print(f"  {label:60s}  med={med:8.2f}ms  min={mn:8.2f}ms  p99={p99:8.2f}ms")
    return med, result


def create_test_frames(tmpdir, num_frames):
    """Create test H.265 compressed frames."""
    frame_dir = os.path.join(tmpdir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)

    all_data = []
    for i in range(num_frames):
        data = torch.randint(0, 256, (ROWS_PER_FRAME, 16), dtype=torch.uint8)
        all_data.append(data)
        frame = _C.tile_rows_to_frame(data, WIDTH, HEIGHT).numpy()
        fpath = os.path.join(frame_dir, f'frame_{i:05d}.h265')
        container = av.open(fpath, mode='w', format='matroska')
        stream = container.add_stream('libx265', rate=1)
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = 'gray'
        stream.options = {'preset': 'ultrafast', 'x265-params': 'lossless=1:log-level=error'}
        avframe = av.VideoFrame.from_ndarray(frame, format='gray')
        for pkt in stream.encode(avframe):
            container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

    return frame_dir, all_data


def benchmark_decode_parallelism():
    """Benchmark serial vs parallel frame decode for different frame counts."""
    print("\n" + "=" * 80)
    print("BATCH DECODE: Serial vs Parallel for N Frames")
    print("=" * 80)

    NUM_FRAMES = 10
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir, all_data = create_test_frames(tmpdir, NUM_FRAMES)
        print(f"  Created {NUM_FRAMES} test frames at {WIDTH}x{HEIGHT}")

        for n_frames in [1, 2, 3, 5, 8, 10]:
            print(f"\n--- {n_frames} frames ---")
            fids = torch.arange(n_frames, dtype=torch.long)

            # Serial: C++ decode one-by-one
            def serial_decode():
                frames = []
                for fid in range(n_frames):
                    path = os.path.join(frame_dir, f'frame_{fid:05d}.h265')
                    frames.append(_C.decode_h265_frame_from_file(path))
                return frames
            t_serial, _ = bench(serial_decode, label=f"Serial C++ decode ({n_frames} frames)")

            # Parallel: C++ batch decode
            def parallel_decode():
                return _C.batch_decode_frames(frame_dir, fids)
            t_parallel, _ = bench(parallel_decode, label=f"Parallel C++ batch decode ({n_frames} frames)")

            # Serial PyAV decode
            def serial_pyav():
                frames = []
                for fid in range(n_frames):
                    path = os.path.join(frame_dir, f'frame_{fid:05d}.h265')
                    c = av.open(path)
                    f = next(c.decode(video=0))
                    frames.append(f.to_ndarray(format='gray'))
                    c.close()
                return frames
            t_pyav, _ = bench(serial_pyav, label=f"Serial PyAV decode ({n_frames} frames)")

            print(f"  → Parallel speedup vs serial C++: {t_serial/t_parallel:.2f}x")
            print(f"  → Parallel speedup vs serial PyAV: {t_pyav/t_parallel:.2f}x")
            print(f"  → Per-frame: serial={t_serial/n_frames:.1f}ms, "
                  f"parallel={t_parallel/n_frames:.1f}ms")


def benchmark_full_pipeline():
    """Benchmark the full lookup pipeline: decode + gather + dequant."""
    print("\n" + "=" * 80)
    print("FULL PIPELINE: Decode + Gather + Dequant (realistic batch scenario)")
    print("=" * 80)

    NUM_FRAMES = 8
    SCALE, ZP = 0.01, 128
    K_PER_FRAME = 200  # typical: ~200 cold rows per frame per batch

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir, all_data = create_test_frames(tmpdir, NUM_FRAMES)

        for n_miss_frames in [1, 2, 3, 5]:
            miss_fids = list(range(n_miss_frames))
            # Create indices scattered across miss frames
            all_indices = []
            for fid in miss_fids:
                offsets = np.sort(np.random.choice(ROWS_PER_FRAME, K_PER_FRAME, replace=False))
                all_indices.extend(fid * ROWS_PER_FRAME + offsets)
            indices_t = torch.tensor(all_indices, dtype=torch.long)
            miss_fids_t = torch.tensor(miss_fids, dtype=torch.long)

            print(f"\n--- {n_miss_frames} cache-miss frames, {len(all_indices)} cold indices ---")

            # Method A: Serial PyAV decode + Python gather + dequant
            def method_pyav_serial():
                results = torch.zeros(len(all_indices), EMB_DIM)
                frame_cache = {}
                for fid in miss_fids:
                    path = os.path.join(frame_dir, f'frame_{fid:05d}.h265')
                    c = av.open(path)
                    f = next(c.decode(video=0))
                    arr = f.to_ndarray(format='gray')
                    c.close()
                    frame_cache[fid] = torch.from_numpy(arr)
                for i, idx in enumerate(all_indices):
                    fid = idx // ROWS_PER_FRAME
                    row = idx % ROWS_PER_FRAME
                    tiled = frame_cache[fid]
                    ty = row // TILES_PER_ROW
                    tx = row % TILES_PER_ROW
                    for ly in range(4):
                        for lx in range(4):
                            val = tiled[ty*4+ly, tx*4+lx].item()
                            results[i, ly*4+lx] = (val - ZP) * SCALE
                return results

            # Method B: Serial C++ decode + C++ gather_dequant
            def method_cpp_serial():
                results = torch.zeros(len(all_indices), EMB_DIM)
                for fid in miss_fids:
                    path = os.path.join(frame_dir, f'frame_{fid:05d}.h265')
                    tiled = _C.decode_h265_frame_from_file(path)
                    mask = []
                    offsets = []
                    for i, idx in enumerate(all_indices):
                        if idx // ROWS_PER_FRAME == fid:
                            mask.append(i)
                            offsets.append(idx % ROWS_PER_FRAME)
                    if offsets:
                        offsets_t = torch.tensor(offsets, dtype=torch.long)
                        gathered = _C.gather_dequant_from_tiled_frame(
                            tiled, offsets_t, TILES_PER_ROW, SCALE, ZP)
                        for j, mi in enumerate(mask):
                            results[mi] = gathered[j]
                return results

            # Method C: C++ batch decode + C++ gather_dequant (per frame)
            def method_batch_decode_gather():
                decoded = _C.batch_decode_frames(frame_dir, miss_fids_t)
                results = torch.zeros(len(all_indices), EMB_DIM)
                fid_to_frame = {fid: decoded[i] for i, fid in enumerate(miss_fids)}
                for fid in miss_fids:
                    mask = []
                    offsets = []
                    for i, idx in enumerate(all_indices):
                        if idx // ROWS_PER_FRAME == fid:
                            mask.append(i)
                            offsets.append(idx % ROWS_PER_FRAME)
                    if offsets:
                        offsets_t = torch.tensor(offsets, dtype=torch.long)
                        gathered = _C.gather_dequant_from_tiled_frame(
                            fid_to_frame[fid], offsets_t, TILES_PER_ROW, SCALE, ZP)
                        for j, mi in enumerate(mask):
                            results[mi] = gathered[j]
                return results

            # Method D: C++ fully fused batch decode + gather + dequant
            def method_fully_fused():
                return _C.batch_decode_gather_dequant(
                    frame_dir, miss_fids_t, indices_t,
                    ROWS_PER_FRAME, TILES_PER_ROW, SCALE, ZP)

            t_b, _ = bench(method_cpp_serial, label=f"Serial C++ decode + gather_dequant",
                          warmup=1, iters=3)
            t_c, _ = bench(method_batch_decode_gather, label=f"Batch decode + per-frame gather",
                          warmup=1, iters=3)
            t_d, _ = bench(method_fully_fused, label=f"Fully fused batch_decode_gather_dequant",
                          warmup=1, iters=3)

            print(f"  → Batch+gather vs serial: {t_b/t_c:.2f}x")
            print(f"  → Fully fused vs serial:  {t_b/t_d:.2f}x")


def benchmark_with_real_data():
    """Benchmark with real compressed .mp4 files from reorder directory."""
    print("\n" + "=" * 80)
    print("REAL DATA: Batch decode from compressed .mp4 files")
    print("=" * 80)

    REORDER_DIR = "results/reorder"
    if not os.path.exists(REORDER_DIR):
        print("  No reorder directory found, skipping")
        return

    mp4_files = sorted([f for f in os.listdir(REORDER_DIR) if f.endswith('.mp4')])[:3]
    if not mp4_files:
        print("  No .mp4 files found")
        return

    for mp4_file in mp4_files:
        fpath = os.path.join(REORDER_DIR, mp4_file)
        fsize = os.path.getsize(fpath) / 1024 / 1024
        print(f"\n--- {mp4_file} ({fsize:.1f}MB) ---")

        # Get frame info
        container = av.open(fpath)
        n_frames = container.streams.video[0].frames
        frame = next(container.decode(video=0))
        h, w = frame.height, frame.width
        container.close()

        tpr = w // 4
        rpf = tpr * (h // 4)
        print(f"  {n_frames} frames, {w}x{h}, {rpf:,} rows/frame")

        # Benchmark: serial vs parallel decode for first 3 frames
        n_test = min(3, n_frames)

        # Serial PyAV
        def serial_pyav():
            frames = []
            container = av.open(fpath)
            for i, frame in enumerate(container.decode(video=0)):
                if i >= n_test:
                    break
                frames.append(frame.to_ndarray(format='gray'))
            container.close()
            return frames

        t_pyav, _ = bench(serial_pyav, warmup=1, iters=3,
                          label=f"Serial PyAV decode ({n_test} frames from multi-frame .mp4)")

        print(f"  Per-frame: {t_pyav/n_test:.1f}ms")


if __name__ == "__main__":
    print("=" * 80)
    print("Batch Decode Pipeline Benchmark")
    print(f"Resolution: {WIDTH}x{HEIGHT}, rows_per_frame={ROWS_PER_FRAME:,}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print("=" * 80)

    benchmark_decode_parallelism()
    benchmark_full_pipeline()
    benchmark_with_real_data()

    print("\n" + "=" * 80)
    print("BATCH DECODE BENCHMARK COMPLETE")
    print("=" * 80)
