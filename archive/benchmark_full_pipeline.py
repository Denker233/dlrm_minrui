#!/usr/bin/env python3
"""
Full Pipeline Benchmark: End-to-end H.265 embedding codec with all C++ optimizations.

Measures the complete encode and decode paths:
ENCODE: fp32 embeddings → quantize → tile → H.265 compress → files
DECODE: files → H.265 decode → tiled frame → gather+dequant → fp32 embeddings

Compares:
- Original Python path (subprocess ffmpeg, Python tiling/untiling)
- C++ optimized path (C++ tile/encode/decode/gather, tiled storage, batch decode)
"""

import os, sys, time, tempfile, subprocess, io
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C
import av

EMB_DIM = 16
TILE_W, TILE_H = 4, 4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    print(f"  {label:60s}  med={med:8.1f}ms  min={mn:8.1f}ms")
    return med, result


def benchmark_encode_pipeline(width, height, num_cold_rows):
    """Benchmark the full encode pipeline."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    num_frames = (num_cold_rows + rows_per_frame - 1) // rows_per_frame

    print(f"\n{'='*80}")
    print(f"ENCODE PIPELINE: {num_cold_rows:,} rows → {num_frames} frames ({width}x{height})")
    print(f"{'='*80}")

    # Generate test data
    fp32_data = torch.randn(num_cold_rows, EMB_DIM)

    # Quantize
    mn = fp32_data.min().item()
    mx = fp32_data.max().item()
    scale = (mx - mn) / 255.0
    zp = round(-mn / scale)
    q_uint8 = ((fp32_data / scale).round() + zp).clamp(0, 255).to(torch.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ===== Method A: Original Python pipeline =====
        print("\n--- Method A: Python tile + subprocess ffmpeg encode ---")

        def encode_python():
            outdir = os.path.join(tmpdir, 'py')
            os.makedirs(outdir, exist_ok=True)
            # Pad
            padded_rows = num_frames * rows_per_frame
            padded = np.zeros((padded_rows, EMB_DIM), dtype=np.uint8)
            padded[:num_cold_rows] = q_uint8.numpy()
            total = 0
            for i in range(num_frames):
                chunk = padded[i*rows_per_frame:(i+1)*rows_per_frame]
                # Python tiling (reshape+transpose+reshape)
                tiles = chunk.reshape(tiles_per_col, tiles_per_row, TILE_H, TILE_W)
                frame = tiles.transpose(0, 2, 1, 3).reshape(height, width)
                # subprocess ffmpeg encode
                fpath = os.path.join(outdir, f'frame_{i:05d}.h265')
                cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
                       '-s', f'{width}x{height}', '-r', '1', '-i', 'pipe:0',
                       '-c:v', 'libx265', '-preset', 'ultrafast', '-pix_fmt', 'gray',
                       '-x265-params', 'lossless=1:log-level=error',
                       '-f', 'matroska', fpath]
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                p.stdin.write(frame.tobytes())
                p.stdin.close()
                p.wait()
                total += os.path.getsize(fpath)
            return total

        t_py, bytes_py = bench(encode_python, warmup=0, iters=1,
                                label=f"Python tile + subprocess ffmpeg ({num_frames} frames)")

        # ===== Method B: C++ fused tile + C++ batch encode =====
        print("\n--- Method B: C++ fused tile + C++ batch encode ---")

        def encode_cpp():
            outdir = os.path.join(tmpdir, 'cpp')
            os.makedirs(outdir, exist_ok=True)
            # Pad
            padded_rows = num_frames * rows_per_frame
            if q_uint8.shape[0] < padded_rows:
                padded = torch.zeros(padded_rows, EMB_DIM, dtype=torch.uint8)
                padded[:num_cold_rows] = q_uint8
            else:
                padded = q_uint8
            # C++ fused tile (all frames in single parallel pass)
            tiled_frames = _C.fused_quantize_tile_multiframe(padded, width, height)
            # C++ batch encode (all frames in parallel via libx265)
            total = _C.batch_encode_h265_frames(tiled_frames, outdir, True)
            return total

        t_cpp, bytes_cpp = bench(encode_cpp, warmup=0, iters=1,
                                  label=f"C++ tile + C++ batch encode ({num_frames} frames)")

        print(f"\n  Encode speedup: {t_py/t_cpp:.1f}x")
        print(f"  Python: {t_py:.0f}ms, C++: {t_cpp:.0f}ms")
        print(f"  Compressed: Python={bytes_py/1024:.0f}KB, C++={bytes_cpp/1024:.0f}KB")
        return t_py, t_cpp


def benchmark_decode_pipeline(width, height, num_cold_rows):
    """Benchmark the full decode pipeline (cache miss scenario)."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    num_frames = (num_cold_rows + rows_per_frame - 1) // rows_per_frame

    print(f"\n{'='*80}")
    print(f"DECODE PIPELINE: {num_frames} frames → lookup K rows ({width}x{height})")
    print(f"{'='*80}")

    # Create test compressed data
    fp32_data = torch.randn(num_cold_rows, EMB_DIM)
    mn, mx = fp32_data.min().item(), fp32_data.max().item()
    scale = (mx - mn) / 255.0
    zp = round(-mn / scale)
    q_uint8 = ((fp32_data / scale).round() + zp).clamp(0, 255).to(torch.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Encode frames
        frame_dir = os.path.join(tmpdir, 'frames')
        os.makedirs(frame_dir)
        padded_rows = num_frames * rows_per_frame
        if q_uint8.shape[0] < padded_rows:
            padded = torch.zeros(padded_rows, EMB_DIM, dtype=torch.uint8)
            padded[:num_cold_rows] = q_uint8
        else:
            padded = q_uint8
        tiled_frames = _C.fused_quantize_tile_multiframe(padded, width, height)
        _C.batch_encode_h265_frames(tiled_frames, frame_dir, True)

        # Load compressed bytes from disk into RAM for in-memory decode
        compressed_bytes = {}
        for fid in range(num_frames):
            fpath = os.path.join(frame_dir, f'frame_{fid:05d}.h265')
            with open(fpath, 'rb') as fh:
                compressed_bytes[fid] = fh.read()

        # Test different lookup scenarios
        for n_miss, K_per_frame in [(1, 100), (1, 1000), (3, 100), (3, 1000), (5, 200)]:
            n_miss = min(n_miss, num_frames)
            miss_fids = list(range(n_miss))
            total_K = n_miss * K_per_frame

            # Create indices
            all_indices = []
            for fid in miss_fids:
                offsets = np.sort(np.random.choice(rows_per_frame, K_per_frame, replace=False))
                all_indices.extend(fid * rows_per_frame + offsets)
            indices_t = torch.tensor(all_indices, dtype=torch.long)
            miss_fids_t = torch.tensor(miss_fids, dtype=torch.long)

            print(f"\n--- {n_miss} miss frames, {total_K} cold rows ---")

            # Method A: Python decode path (PyAV from in-memory bytes + untile + numpy gather)
            def decode_python():
                results = np.zeros((total_K, EMB_DIM), dtype=np.float32)
                for fid in miss_fids:
                    data = compressed_bytes[fid]
                    container = av.open(io.BytesIO(data))
                    frame = next(container.decode(video=0))
                    arr = frame.to_ndarray(format='gray')
                    container.close()
                    # Full untile
                    grid = arr.reshape(tiles_per_col, TILE_H, tiles_per_row, TILE_W)
                    rows = grid.transpose(0, 2, 1, 3).reshape(rows_per_frame, EMB_DIM)
                    # Gather
                    for i, idx in enumerate(all_indices):
                        if idx // rows_per_frame == fid:
                            row_in_frame = idx % rows_per_frame
                            results[i] = (rows[row_in_frame].astype(np.float32) - zp) * scale
                return results

            # Method B: C++ optimized (batch decode from in-memory bytes + tiled gather + dequant)
            def decode_cpp_batch():
                # Note: batch_decode_gather_dequant still uses file path; decode frames from bytes individually
                results = torch.zeros(total_K, EMB_DIM)
                decoded_frames = {}
                for fid in miss_fids:
                    data = compressed_bytes[fid]
                    compressed_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                    decoded_frames[fid] = _C.decode_h265_frame_from_bytes(compressed_t)
                for fid in miss_fids:
                    mask = []
                    offsets_list = []
                    for i, idx in enumerate(all_indices):
                        if idx // rows_per_frame == fid:
                            mask.append(i)
                            offsets_list.append(idx % rows_per_frame)
                    if offsets_list:
                        offsets_t = torch.tensor(offsets_list, dtype=torch.long)
                        gathered = _C.gather_dequant_from_tiled_frame(
                            decoded_frames[fid], offsets_t, tiles_per_row, scale, zp)
                        for j, mi in enumerate(mask):
                            results[mi] = gathered[j]
                return results

            # Method C: C++ decode from in-memory bytes + C++ tiled gather (serial decode)
            def decode_cpp_serial():
                results = torch.zeros(total_K, EMB_DIM)
                for fid in miss_fids:
                    data = compressed_bytes[fid]
                    compressed_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                    tiled = _C.decode_h265_frame_from_bytes(compressed_t)
                    mask = []
                    offsets = []
                    for i, idx in enumerate(all_indices):
                        if idx // rows_per_frame == fid:
                            mask.append(i)
                            offsets.append(idx % rows_per_frame)
                    if offsets:
                        offsets_t = torch.tensor(offsets, dtype=torch.long)
                        gathered = _C.gather_dequant_from_tiled_frame(
                            tiled, offsets_t, tiles_per_row, scale, zp)
                        for j, mi in enumerate(mask):
                            results[mi] = gathered[j]
                return results

            t_py, _ = bench(decode_python, warmup=1, iters=3,
                           label=f"Python (PyAV memory + untile + numpy)")
            t_cpp_s, _ = bench(decode_cpp_serial, warmup=1, iters=3,
                              label=f"C++ serial decode (memory) + tiled gather")
            t_cpp_b, _ = bench(decode_cpp_batch, warmup=1, iters=3,
                              label=f"C++ batch decode (memory) + fused gather")

            print(f"  → Serial C++ vs Python:  {t_py/t_cpp_s:.1f}x")
            print(f"  → Batch C++ vs Python:   {t_py/t_cpp_b:.1f}x")
            if n_miss > 1:
                print(f"  → Batch vs serial C++:   {t_cpp_s/t_cpp_b:.1f}x")


def benchmark_with_real_tables():
    """Benchmark using actual DLRM table data."""
    print(f"\n{'='*80}")
    print(f"REAL DATA: Full pipeline with DLRM embedding tables")
    print(f"{'='*80}")

    model_path = "./models/dlrm_kaggle_correct.pt"
    hotcold_dir = "results/hotcold"

    if not os.path.exists(model_path) or not os.path.exists(hotcold_dir):
        print("  Model or hot/cold data not available, skipping")
        return

    # Load model
    import dlrm_s_pytorch as dlrm_s
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(model, dict):
        log("  Cannot load model state dict directly, skipping")
        return

    # Find large tables
    large_tables = []
    for i, emb in enumerate(model.emb_l):
        n = emb.num_embeddings
        if n > 50000:
            cold_path = os.path.join(hotcold_dir, f'cold_mask_{i}.pt')
            if os.path.exists(cold_path):
                cold_mask = torch.load(cold_path, map_location='cpu')
                n_cold = cold_mask.sum().item()
                large_tables.append((i, n, n_cold, emb.weight.data))

    if not large_tables:
        print("  No large tables found")
        return

    WIDTH, HEIGHT = 1920, 1080
    RPF = (WIDTH // 4) * (HEIGHT // 4)

    for tid, n_emb, n_cold, weight in large_tables[:3]:
        n_frames = (n_cold + RPF - 1) // RPF
        print(f"\n--- Table {tid}: {n_emb:,} embeddings, {n_cold:,} cold, {n_frames} frames ---")

        # Extract cold weights
        cold_path = os.path.join(hotcold_dir, f'cold_mask_{tid}.pt')
        cold_mask = torch.load(cold_path, map_location='cpu')
        cold_idx = torch.where(cold_mask)[0]
        cold_weight = weight[cold_idx[:n_cold]]

        # Quantize
        mn, mx = cold_weight.min().item(), cold_weight.max().item()
        s = (mx - mn) / 255.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        q = ((cold_weight / s).round() + zp).clamp(0, 255).to(torch.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Encode
            frame_dir = os.path.join(tmpdir, f'table_{tid}')
            os.makedirs(frame_dir)

            # C++ encode
            padded_rows = n_frames * RPF
            padded = torch.zeros(padded_rows, EMB_DIM, dtype=torch.uint8)
            padded[:n_cold] = q
            t0 = time.perf_counter()
            tiled = _C.fused_quantize_tile_multiframe(padded, WIDTH, HEIGHT)
            total_bytes = _C.batch_encode_h265_frames(tiled, frame_dir, True)
            t_encode = (time.perf_counter() - t0) * 1000

            raw_bytes = n_cold * EMB_DIM
            ratio = raw_bytes / total_bytes if total_bytes > 0 else 0
            print(f"  Encode: {t_encode:.0f}ms "
                  f"({raw_bytes/1024/1024:.1f}MB → {total_bytes/1024/1024:.1f}MB, "
                  f"{ratio:.2f}x ratio)")

            # Load compressed bytes into RAM for in-memory decode
            compressed_bytes = {}
            for fid in range(n_frames):
                fpath = os.path.join(frame_dir, f'frame_{fid:05d}.h265')
                with open(fpath, 'rb') as fh:
                    compressed_bytes[fid] = fh.read()

            # Decode benchmarks (from in-memory bytes)
            test_indices = torch.from_numpy(
                np.sort(np.random.choice(n_cold, min(1000, n_cold), replace=False))
            ).long()
            miss_frames = torch.unique(test_indices // RPF).long()

            t0 = time.perf_counter()
            # Decode from in-memory bytes + gather + dequant
            results = torch.zeros(len(test_indices), EMB_DIM)
            for fid in miss_frames.tolist():
                data = compressed_bytes[fid]
                compressed_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                tiled = _C.decode_h265_frame_from_bytes(compressed_t)
                mask = []
                offsets = []
                for i, idx in enumerate(test_indices.tolist()):
                    if idx // RPF == fid:
                        mask.append(i)
                        offsets.append(idx % RPF)
                if offsets:
                    offsets_t = torch.tensor(offsets, dtype=torch.long)
                    gathered = _C.gather_dequant_from_tiled_frame(
                        tiled, offsets_t, WIDTH // 4, s, zp)
                    for j, mi in enumerate(mask):
                        results[mi] = gathered[j]
            t_decode = (time.perf_counter() - t0) * 1000

            print(f"  Decode: {t_decode:.1f}ms for {len(test_indices)} rows "
                  f"from {len(miss_frames)} frames (in-memory decode)")


if __name__ == "__main__":
    print("=" * 80)
    print("Full Pipeline Benchmark: All C++ Optimizations")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print("=" * 80)

    # 1080p benchmarks
    benchmark_encode_pipeline(1920, 1080, 500000)   # ~4 frames
    benchmark_decode_pipeline(1920, 1080, 500000)

    # Real DLRM data
    benchmark_with_real_tables()

    print(f"\n{'='*80}")
    print("FULL PIPELINE BENCHMARK COMPLETE")
    print("=" * 80)
