#!/usr/bin/env python3
"""
Benchmark different encoding methods for embedding frames:
1. subprocess ffmpeg (current) with .tobytes()
2. PyAV in-process encoding (avoids subprocess + tobytes)
3. C++ tiled frame → PyAV encode (combines C++ tile + PyAV)
4. Direct buffer write via memoryview

Tests both single-frame and multi-frame scenarios.
"""

import os, sys, time, io, subprocess, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compressed_emb as _C

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False
    print("PyAV not available!")
    sys.exit(1)

EMB_DIM = 16
WIDTH, HEIGHT = 1920, 1080
TILES_PER_ROW = WIDTH // 4
ROWS_PER_FRAME = TILES_PER_ROW * (HEIGHT // 4)


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


def method_subprocess_ffmpeg(frame_np, frame_path, width, height):
    """Current method: subprocess ffmpeg with .tobytes() pipe."""
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'gray', '-s', f'{width}x{height}',
        '-r', '1', '-i', 'pipe:0',
        '-c:v', 'libx265', '-preset', 'ultrafast', '-pix_fmt', 'gray',
        '-x265-params', 'lossless=1:log-level=error',
        '-f', 'matroska', frame_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.stdin.write(frame_np.tobytes())
    proc.stdin.close()
    proc.wait()
    return os.path.getsize(frame_path)


def method_pyav_encode(frame_np, width, height):
    """PyAV in-process encoding (no subprocess, no tobytes)."""
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
    # from_ndarray avoids tobytes by using numpy buffer protocol
    avframe = av.VideoFrame.from_ndarray(frame_np, format='gray')
    for packet in stream.encode(avframe):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return output.getvalue()


def method_pyav_encode_to_file(frame_np, frame_path, width, height):
    """PyAV in-process encoding directly to file."""
    container = av.open(frame_path, mode='w', format='matroska')
    stream = container.add_stream('libx265', rate=1)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'gray'
    stream.options = {
        'preset': 'ultrafast',
        'x265-params': 'lossless=1:log-level=error'
    }
    avframe = av.VideoFrame.from_ndarray(frame_np, format='gray')
    for packet in stream.encode(avframe):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return os.path.getsize(frame_path)


def method_subprocess_memoryview(frame_np, frame_path, width, height):
    """Subprocess ffmpeg with memoryview (avoids tobytes copy)."""
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'gray', '-s', f'{width}x{height}',
        '-r', '1', '-i', 'pipe:0',
        '-c:v', 'libx265', '-preset', 'ultrafast', '-pix_fmt', 'gray',
        '-x265-params', 'lossless=1:log-level=error',
        '-f', 'matroska', frame_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Use contiguous buffer and memoryview
    contiguous = np.ascontiguousarray(frame_np)
    proc.stdin.write(memoryview(contiguous))
    proc.stdin.close()
    proc.wait()
    return os.path.getsize(frame_path)


def main():
    print("=" * 80)
    print("Encode Method Benchmark")
    print(f"Frame: {WIDTH}x{HEIGHT}, {ROWS_PER_FRAME:,} rows/frame")
    print("=" * 80)

    # Generate test data
    emb_uint8 = torch.randint(0, 256, (ROWS_PER_FRAME, 16), dtype=torch.uint8)
    frame_t = _C.tile_rows_to_frame(emb_uint8, WIDTH, HEIGHT)
    frame_np = frame_t.numpy()
    frame_np_copy = frame_np.copy()  # ensure contiguous

    print(f"Frame data: {frame_np.nbytes/1024/1024:.1f}MB, "
          f"contiguous={frame_np.flags['C_CONTIGUOUS']}")

    with tempfile.TemporaryDirectory() as tmpdir:
        fp1 = os.path.join(tmpdir, "test1.mkv")
        fp2 = os.path.join(tmpdir, "test2.mkv")
        fp3 = os.path.join(tmpdir, "test3.mkv")

        # === Breakdown: just the tobytes() call ===
        print("\n--- tobytes() overhead alone ---")
        bench(lambda: frame_np.tobytes(), label=".tobytes() on (H,W) uint8 frame")
        bench(lambda: memoryview(frame_np_copy), label="memoryview() on contiguous frame")

        # === Single frame encoding ===
        print("\n--- Single frame encoding ---")

        t1, _ = bench(
            lambda: method_subprocess_ffmpeg(frame_np_copy, fp1, WIDTH, HEIGHT),
            label="subprocess ffmpeg + .tobytes()",
            warmup=1, iters=3
        )

        t2, compressed_data = bench(
            lambda: method_pyav_encode(frame_np_copy, WIDTH, HEIGHT),
            label="PyAV in-memory encode",
            warmup=1, iters=3
        )

        t3, _ = bench(
            lambda: method_pyav_encode_to_file(frame_np_copy, fp2, WIDTH, HEIGHT),
            label="PyAV encode to file",
            warmup=1, iters=3
        )

        t4, _ = bench(
            lambda: method_subprocess_memoryview(frame_np_copy, fp3, WIDTH, HEIGHT),
            label="subprocess ffmpeg + memoryview",
            warmup=1, iters=3
        )

        # Verify compression sizes match
        print(f"\n  Compressed sizes: "
              f"ffmpeg={os.path.getsize(fp1)/1024:.0f}KB, "
              f"pyav_mem={len(compressed_data)/1024:.0f}KB, "
              f"pyav_file={os.path.getsize(fp2)/1024:.0f}KB")

        # === Full pipeline: C++ tile + encode ===
        print("\n--- Full pipeline: tile + encode ---")

        def pipeline_subprocess():
            frame = _C.tile_rows_to_frame(emb_uint8, WIDTH, HEIGHT).numpy()
            return method_subprocess_ffmpeg(frame, fp1, WIDTH, HEIGHT)

        def pipeline_pyav():
            frame = _C.tile_rows_to_frame(emb_uint8, WIDTH, HEIGHT).numpy()
            return method_pyav_encode(frame, WIDTH, HEIGHT)

        bench(pipeline_subprocess, label="C++ tile + subprocess ffmpeg",
              warmup=1, iters=3)
        bench(pipeline_pyav, label="C++ tile + PyAV in-memory",
              warmup=1, iters=3)

        # === Multi-frame (10 frames) ===
        print("\n--- Multi-frame encoding (10 frames) ---")
        N = ROWS_PER_FRAME * 10
        big_data = torch.randint(0, 256, (N, 16), dtype=torch.uint8)

        def multi_ffmpeg():
            frames = _C.fused_quantize_tile_multiframe(big_data, WIDTH, HEIGHT)
            total = 0
            for i, f in enumerate(frames):
                fp = os.path.join(tmpdir, f"multi_{i}.mkv")
                total += method_subprocess_ffmpeg(f.numpy(), fp, WIDTH, HEIGHT)
            return total

        def multi_pyav():
            frames = _C.fused_quantize_tile_multiframe(big_data, WIDTH, HEIGHT)
            total = 0
            for f in frames:
                data = method_pyav_encode(f.numpy(), WIDTH, HEIGHT)
                total += len(data)
            return total

        t_ffmpeg, _ = bench(multi_ffmpeg, label="10 frames: C++ tile + subprocess ffmpeg",
                            warmup=1, iters=2)
        t_pyav, _ = bench(multi_pyav, label="10 frames: C++ tile + PyAV in-memory",
                          warmup=1, iters=2)

        print(f"\n  Per-frame average: ffmpeg={t_ffmpeg/10:.1f}ms, pyav={t_pyav/10:.1f}ms")

    # === PyAV decode benchmark ===
    print("\n--- Decode methods ---")
    compressed = method_pyav_encode(frame_np_copy, WIDTH, HEIGHT)
    print(f"  Compressed size: {len(compressed)/1024:.1f}KB")

    def decode_pyav_untile_py():
        container = av.open(io.BytesIO(compressed))
        frame = next(container.decode(video=0))
        arr = frame.to_ndarray(format='gray')
        container.close()
        from codec_ondemand_benchmark import tiled_frame_to_rows
        return tiled_frame_to_rows(arr, WIDTH, HEIGHT)

    def decode_pyav_untile_cpp():
        container = av.open(io.BytesIO(compressed))
        frame = next(container.decode(video=0))
        arr = frame.to_ndarray(format='gray')
        container.close()
        return _C.untile_frame_to_rows(torch.from_numpy(arr), ROWS_PER_FRAME)

    def decode_pyav_gather_100():
        container = av.open(io.BytesIO(compressed))
        frame = next(container.decode(video=0))
        arr = frame.to_ndarray(format='gray')
        container.close()
        idx = torch.arange(100).long()
        return _C.gather_from_tiled_frame(torch.from_numpy(arr), idx, TILES_PER_ROW)

    bench(decode_pyav_untile_py, label="Decode + Python untile", warmup=1, iters=5)
    bench(decode_pyav_untile_cpp, label="Decode + C++ untile", warmup=1, iters=5)
    bench(decode_pyav_gather_100, label="Decode + C++ gather K=100", warmup=1, iters=5)


if __name__ == "__main__":
    main()
