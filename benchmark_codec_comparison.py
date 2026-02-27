#!/usr/bin/env python3
"""
Multi-Codec Benchmark: Compare H.265 vs H.264 vs FFV1 for embedding compression.

Measures encode speed, decode speed, compression ratio, and end-to-end latency
for each codec. Shows the tradeoff between decode latency and compression ratio,
which is critical for on-demand embedding access patterns.
"""

import os, sys, time, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C

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
    print(f"  {label:60s}  med={med:8.2f}ms  min={mn:8.2f}ms")
    return med, result


def benchmark_single_frame_encode(width, height):
    """Benchmark single-frame encode for each codec."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col

    print(f"\n{'='*80}")
    print(f"SINGLE-FRAME ENCODE: {width}x{height} ({rows_per_frame:,} embeddings)")
    print(f"{'='*80}")

    # Generate structured test data (quantized normal distribution, like real embeddings)
    fp32_data = torch.randn(rows_per_frame, EMB_DIM)
    mn, mx = fp32_data.min().item(), fp32_data.max().item()
    s = (mx - mn) / 255.0
    zp = round(-mn / s)
    data = ((fp32_data / s).round() + zp).clamp(0, 255).to(torch.uint8)
    frame = _C.tile_rows_to_frame(data, width, height)
    raw_bytes = rows_per_frame * EMB_DIM

    codecs = ["h265", "h264", "ffv1"]
    ext_map = {"h265": ".h265", "h264": ".mkv", "ffv1": ".mkv"}
    results = {}

    for codec_name in codecs:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, f'test_frame{ext_map[codec_name]}')

            def encode_fn(c=codec_name, p=fpath):
                _C.encode_frame_codec(frame, p, c, True, 0)
                return os.path.getsize(p)

            t_enc, comp_bytes = bench(encode_fn, warmup=1, iters=5,
                                       label=f"{codec_name} encode (lossless)")
            ratio = raw_bytes / comp_bytes if comp_bytes > 0 else 0
            results[codec_name] = {
                'encode_ms': t_enc,
                'compressed_bytes': comp_bytes,
                'ratio': ratio,
            }
            print(f"    → {comp_bytes/1024:.1f}KB, {ratio:.2f}x compression")

    return results


def benchmark_single_frame_decode(width, height):
    """Benchmark single-frame decode for each codec."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col

    print(f"\n{'='*80}")
    print(f"SINGLE-FRAME DECODE: {width}x{height} ({rows_per_frame:,} embeddings)")
    print(f"{'='*80}")

    # Generate structured test data
    fp32_data = torch.randn(rows_per_frame, EMB_DIM)
    mn, mx = fp32_data.min().item(), fp32_data.max().item()
    s = (mx - mn) / 255.0
    zp = round(-mn / s)
    data = ((fp32_data / s).round() + zp).clamp(0, 255).to(torch.uint8)
    frame = _C.tile_rows_to_frame(data, width, height)

    codecs = ["h265", "h264", "ffv1"]
    ext_map = {"h265": ".h265", "h264": ".mkv", "ffv1": ".mkv"}
    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Encode all codecs (use separate dirs to avoid name collision)
        paths = {}
        for codec_name in codecs:
            codec_dir = os.path.join(tmpdir, codec_name)
            os.makedirs(codec_dir)
            fpath = os.path.join(codec_dir, f'test_frame{ext_map[codec_name]}')
            _C.encode_frame_codec(frame, fpath, codec_name, True, 0)
            paths[codec_name] = fpath

        # Benchmark decode
        for codec_name in codecs:
            fpath = paths[codec_name]
            fsize = os.path.getsize(fpath)

            def decode_fn(p=fpath):
                return _C.decode_h265_frame_from_file(p)

            t_dec, decoded = bench(decode_fn, warmup=2, iters=10,
                                    label=f"{codec_name} decode")

            # Verify correctness
            if codec_name != "h264":
                # H.264 uses YUV420P so lossless only for Y plane
                mae = torch.abs(decoded.float() - frame.float()).mean().item()
                print(f"    → MAE={mae:.4f}, {fsize/1024:.1f}KB")
            else:
                # H.264 YUV420P: just check Y channel (luma) is close
                mae = torch.abs(decoded.float() - frame.float()).mean().item()
                print(f"    → MAE={mae:.4f} (YUV420P), {fsize/1024:.1f}KB")

            results[codec_name] = {
                'decode_ms': t_dec,
                'file_size': fsize,
                'mae': mae,
            }

    return results


def benchmark_batch_decode(width, height, num_frames_list=[1, 3, 5, 8]):
    """Benchmark parallel batch decode for each codec."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col

    max_frames = max(num_frames_list)

    print(f"\n{'='*80}")
    print(f"BATCH DECODE: {width}x{height}, up to {max_frames} frames")
    print(f"{'='*80}")

    # Generate structured test data (quantized normal distribution)
    frames = []
    for i in range(max_frames):
        fp32_data = torch.randn(rows_per_frame, EMB_DIM)
        mn, mx = fp32_data.min().item(), fp32_data.max().item()
        s = (mx - mn) / 255.0
        zp = round(-mn / s)
        data = ((fp32_data / s).round() + zp).clamp(0, 255).to(torch.uint8)
        frames.append(_C.tile_rows_to_frame(data, width, height))

    codecs = ["h265", "h264", "ffv1"]
    ext_map = {"h265": ".h265", "h264": ".mkv", "ffv1": ".mkv"}
    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Encode all frames for each codec
        for codec_name in codecs:
            codec_dir = os.path.join(tmpdir, codec_name)
            os.makedirs(codec_dir)

            ext = ext_map[codec_name]

            for i, frame in enumerate(frames):
                fpath = os.path.join(codec_dir, f'frame_{i:05d}{ext}')
                _C.encode_frame_codec(frame, fpath, codec_name, True, 0)

            # Benchmark serial and parallel decode for different frame counts
            results[codec_name] = {}
            for n_frames in num_frames_list:
                fids = torch.arange(n_frames, dtype=torch.long)

                # Serial decode
                def serial_fn(n=n_frames, d=codec_dir, e=ext):
                    decoded = []
                    for i in range(n):
                        fpath = os.path.join(d, f'frame_{i:05d}{e}')
                        decoded.append(_C.decode_h265_frame_from_file(fpath))
                    return decoded

                # For batch_decode_frames, create .h265-named links
                batch_h265_dir = os.path.join(tmpdir, f'{codec_name}_h265')
                os.makedirs(batch_h265_dir, exist_ok=True)
                for i in range(max_frames):
                    src = os.path.join(codec_dir, f'frame_{i:05d}{ext}')
                    dst = os.path.join(batch_h265_dir, f'frame_{i:05d}.h265')
                    if not os.path.exists(dst):
                        os.link(src, dst)

                def parallel_fn(n=n_frames, d=batch_h265_dir):
                    return _C.batch_decode_frames(d, torch.arange(n, dtype=torch.long))

                print(f"\n--- {codec_name}, {n_frames} frames ---")
                t_serial, _ = bench(serial_fn, warmup=1, iters=5,
                                     label=f"Serial decode ({n_frames} frames)")
                t_parallel, _ = bench(parallel_fn, warmup=1, iters=5,
                                       label=f"Parallel batch decode ({n_frames} frames)")

                speedup = t_serial / t_parallel if t_parallel > 0 else 0
                per_frame = t_parallel / n_frames
                print(f"    → Speedup: {speedup:.2f}x, per-frame: {per_frame:.1f}ms")

                results[codec_name][n_frames] = {
                    'serial_ms': t_serial,
                    'parallel_ms': t_parallel,
                    'speedup': speedup,
                    'per_frame_ms': per_frame,
                }

    return results


def benchmark_gather_pipeline(width, height, K_values=[100, 500, 1000]):
    """Benchmark full decode+gather+dequant pipeline for each codec."""
    tiles_per_row = width // TILE_W
    tiles_per_col = height // TILE_H
    rows_per_frame = tiles_per_row * tiles_per_col
    num_frames = 5

    print(f"\n{'='*80}")
    print(f"DECODE+GATHER+DEQUANT PIPELINE: {width}x{height}, {num_frames} frames")
    print(f"{'='*80}")

    # Generate structured data
    SCALE, ZP = 0.01, 128
    frames = []
    for i in range(num_frames):
        fp32_data = torch.randn(rows_per_frame, EMB_DIM)
        data = ((fp32_data / 0.01).round() + 128).clamp(0, 255).to(torch.uint8)
        frames.append(_C.tile_rows_to_frame(data, width, height))

    codecs = ["h265", "h264", "ffv1"]
    ext_map = {"h265": ".h265", "h264": ".mkv", "ffv1": ".mkv"}

    with tempfile.TemporaryDirectory() as tmpdir:
        for codec_name in codecs:
            codec_dir = os.path.join(tmpdir, codec_name)
            os.makedirs(codec_dir)

            ext = ext_map[codec_name]

            for i, frame in enumerate(frames):
                fpath = os.path.join(codec_dir, f'frame_{i:05d}{ext}')
                _C.encode_frame_codec(frame, fpath, codec_name, True, 0)

            # Create .h265-named links for batch_decode_gather_dequant
            h265_dir = os.path.join(tmpdir, f'{codec_name}_h265')
            os.makedirs(h265_dir, exist_ok=True)
            for i in range(num_frames):
                src = os.path.join(codec_dir, f'frame_{i:05d}{ext}')
                dst = os.path.join(h265_dir, f'frame_{i:05d}.h265')
                if not os.path.exists(dst):
                    os.link(src, dst)

            for K in K_values:
                # Random indices spread across all frames
                n_miss = min(3, num_frames)
                miss_fids = list(range(n_miss))
                K_per_frame = K // n_miss

                all_indices = []
                for fid in miss_fids:
                    offsets = np.sort(np.random.choice(rows_per_frame, K_per_frame, replace=False))
                    all_indices.extend(fid * rows_per_frame + offsets)
                indices_t = torch.tensor(all_indices, dtype=torch.long)
                miss_fids_t = torch.tensor(miss_fids, dtype=torch.long)

                def fused_fn(d=h265_dir, mf=miss_fids_t, idx=indices_t):
                    return _C.batch_decode_gather_dequant(
                        d, mf, idx, rows_per_frame, tiles_per_row, SCALE, ZP)

                print(f"\n--- {codec_name}, {n_miss} frames, {K} rows ---")
                t_fused, _ = bench(fused_fn, warmup=1, iters=5,
                                    label=f"batch_decode_gather_dequant")
                per_row_us = t_fused * 1000 / K
                print(f"    → {per_row_us:.1f} us/row")


def benchmark_real_tables():
    """Benchmark with real DLRM embedding table data."""
    print(f"\n{'='*80}")
    print(f"REAL DATA: Multi-codec comparison with DLRM embeddings")
    print(f"{'='*80}")

    model_path = "./models/dlrm_kaggle_correct.pt"
    hotcold_dir = "results/hotcold"
    reorder_dir = "results/reorder"

    if not os.path.exists(model_path) or not os.path.exists(hotcold_dir):
        print("  Model or hot/cold data not available, skipping")
        return

    # Load model weights
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        sd = state['state_dict']
    elif isinstance(state, dict):
        sd = state
    else:
        sd = state.state_dict()

    # Find large tables with cold data
    WIDTH, HEIGHT = 1920, 1080
    RPF = (WIDTH // 4) * (HEIGHT // 4)
    codecs = ["h265", "h264", "ffv1"]

    # Get embedding keys and find large tables
    emb_keys = sorted([k for k in sd.keys() if 'emb_l' in k and 'weight' in k],
                       key=lambda k: int(k.split('.')[1]))

    large_tables = []
    for emb_key in emb_keys:
        t_idx = int(emb_key.split('.')[1])
        weight = sd[emb_key]
        n_emb = weight.shape[0]
        if n_emb < 50000:
            continue
        cold_path = os.path.join(hotcold_dir, f'cold_indices_{t_idx}.pt')
        if not os.path.exists(cold_path):
            continue
        large_tables.append((t_idx, emb_key, weight, cold_path))

    log(f"Found {len(large_tables)} large tables with cold data")

    for t_idx, emb_key, weight, cold_path in large_tables[:3]:
        n_emb = weight.shape[0]

        cold_indices = torch.load(cold_path, map_location='cpu', weights_only=True)
        n_cold = len(cold_indices)
        cold_weight = weight[cold_indices]

        # Quantize
        mn, mx = cold_weight.min().item(), cold_weight.max().item()
        s = (mx - mn) / 255.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        q = ((cold_weight / s).round() + zp).clamp(0, 255).to(torch.uint8)

        n_frames = (n_cold + RPF - 1) // RPF
        print(f"\n--- Table {t_idx}: {n_emb:,} embeddings, {n_cold:,} cold, "
              f"{n_frames} frames ---")

        # Pad
        padded_rows = n_frames * RPF
        padded = torch.zeros(padded_rows, EMB_DIM, dtype=torch.uint8)
        padded[:n_cold] = q

        # Tile all frames
        tiled_frames = _C.fused_quantize_tile_multiframe(padded, WIDTH, HEIGHT)

        raw_bytes = n_cold * EMB_DIM

        for codec_name in codecs:
            with tempfile.TemporaryDirectory() as tmpdir:
                frame_dir = os.path.join(tmpdir, codec_name)
                os.makedirs(frame_dir)

                # Encode
                t0 = time.perf_counter()
                total_bytes = _C.batch_encode_frames_codec(
                    tiled_frames, frame_dir, codec_name, True)
                t_encode = (time.perf_counter() - t0) * 1000

                ratio = raw_bytes / total_bytes if total_bytes > 0 else 0

                # Create .h265 links for decode function
                ext_map = {"h265": ".h265", "h264": ".h264", "ffv1": ".mkv"}
                ext = ext_map[codec_name]
                h265_dir = os.path.join(tmpdir, f'{codec_name}_h265')
                os.makedirs(h265_dir)
                for i in range(n_frames):
                    src = os.path.join(frame_dir, f'frame_{i:05d}{ext}')
                    dst = os.path.join(h265_dir, f'frame_{i:05d}.h265')
                    if os.path.exists(src):
                        os.link(src, dst)

                # Decode benchmark (1000 random rows)
                test_K = min(1000, n_cold)
                test_indices = torch.from_numpy(
                    np.sort(np.random.choice(n_cold, test_K, replace=False))
                ).long()
                miss_frames = torch.unique(test_indices // RPF).long()

                # Warmup
                try:
                    _C.batch_decode_gather_dequant(
                        h265_dir, miss_frames, test_indices,
                        RPF, WIDTH // 4, s, zp)
                except Exception:
                    pass

                t0 = time.perf_counter()
                _C.batch_decode_gather_dequant(
                    h265_dir, miss_frames, test_indices,
                    RPF, WIDTH // 4, s, zp)
                t_decode = (time.perf_counter() - t0) * 1000

                print(f"  {codec_name:5s}: encode={t_encode:7.0f}ms, "
                      f"decode={t_decode:6.1f}ms ({len(miss_frames)} frames), "
                      f"ratio={ratio:5.2f}x ({total_bytes/1024/1024:.1f}MB)")


def print_summary(enc_results, dec_results):
    """Print comparison summary table."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: Codec Comparison (1080p)")
    print(f"{'='*80}")

    print(f"\n{'Codec':<8} {'Encode':>10} {'Decode':>10} {'Ratio':>8} {'Size':>8}")
    print("-" * 50)

    for codec_name in ["h265", "h264", "ffv1"]:
        enc = enc_results.get(codec_name, {})
        dec = dec_results.get(codec_name, {})
        enc_ms = enc.get('encode_ms', 0)
        dec_ms = dec.get('decode_ms', 0)
        ratio = enc.get('ratio', 0)
        size_kb = enc.get('compressed_bytes', 0) / 1024
        print(f"{codec_name:<8} {enc_ms:>8.1f}ms {dec_ms:>8.1f}ms {ratio:>7.2f}x {size_kb:>6.0f}KB")


if __name__ == "__main__":
    print("=" * 80)
    print("Multi-Codec Benchmark: H.265 vs H.264 vs FFV1")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print("=" * 80)

    # 1080p benchmarks
    enc_results = benchmark_single_frame_encode(1920, 1080)
    dec_results = benchmark_single_frame_decode(1920, 1080)

    print_summary(enc_results, dec_results)

    # Batch decode comparison
    benchmark_batch_decode(1920, 1080, [1, 3, 5])

    # Full pipeline
    benchmark_gather_pipeline(1920, 1080, [100, 1000])

    # Real data
    benchmark_real_tables()

    print(f"\n{'='*80}")
    print("MULTI-CODEC BENCHMARK COMPLETE")
    print("=" * 80)
