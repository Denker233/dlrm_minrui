#!/usr/bin/env python3
"""
Benchmark H.265 tiles for faster single-frame decode.
Tiles split a frame into independent regions with separate CABAC streams.
"""
import os, sys, time, json, subprocess, tempfile, shutil
import numpy as np
import torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
import compressed_emb as _C
from codec_ondemand_benchmark import quantize_table, REORDER_DIR

TABLES = [2, 11, 20, 15]  # largest tables only (most bottlenecked)
SF_DIMS = {
    2: (3840, 40400), 11: (3840, 33304), 20: (1920, 56200), 15: (1920, 43556),
}
EMB_DIM = 16
MODEL_PATH = 'models/dlrm_kaggle_correct.pt'

def encode_with_tiles(q_np, width, height, outpath, crf=30, tile_cols=1, tile_rows=1):
    """Encode with H.265 tiles for parallel decode."""
    rpf = (width * height) // EMB_DIM
    n = q_np.shape[0]
    if n < rpf:
        padded = np.zeros((rpf, EMB_DIM), dtype=np.uint8)
        padded[:n] = q_np
        q_np = padded

    q_t = torch.from_numpy(q_np[:rpf])
    tiled = _C.fused_quantize_tile_multiframe(q_t, width, height)
    raw = tiled[0].numpy().tobytes()

    # Build x265 params with tiles
    x265 = (f'keyint=1:min-keyint=1:crf={crf}:log-level=error:'
            f'no-deblock=1:no-sao=1')
    if tile_cols > 1 or tile_rows > 1:
        # x265 uses --num-tile-columns and --num-tile-rows (0-indexed count)
        # In x265-params format: use columns and rows directly
        x265 += f':tile-columns={tile_cols}:tile-rows={tile_rows}'

    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
           '-s', f'{width}x{height}', '-r', '1', '-i', 'pipe:0',
           '-c:v', 'libx265', '-preset', 'medium', '-pix_fmt', 'gray',
           '-x265-params', x265, '-f', 'matroska', outpath]
    r = subprocess.run(cmd, input=raw, capture_output=True, timeout=120)
    if r.returncode != 0:
        print(f"  ffmpeg error: {r.stderr.decode()[-200:]}")
    return os.path.getsize(outpath) if os.path.exists(outpath) else 0


def bench_decode(path, num_threads=1, n_reps=3, warmup=1):
    for _ in range(warmup):
        _C.decode_h265_frame_from_file(path, num_threads, True, False, False)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _C.decode_h265_frame_from_file(path, num_threads, True, False, False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times), min(times)


def bench_batch(paths, num_threads_per=1, max_parallel=0, n_reps=3, warmup=1):
    if max_parallel <= 0:
        max_parallel = len(paths)
    for _ in range(warmup):
        _C.batch_decode_file_paths(paths, num_threads_per, max_parallel, True, False, False)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _C.batch_decode_file_paths(paths, num_threads_per, max_parallel, True, False, False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times), min(times)


def main():
    torch.set_num_threads(1)
    print("=" * 70)
    print("H.265 TILES: Can tiles speed up single-frame decode?")
    print("=" * 70)

    # Load model and cold data
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))

    tmpdir = tempfile.mkdtemp(prefix='tiles_bench_')
    print(f"Temp: {tmpdir}")

    # Tile configurations to test
    tile_configs = [
        (1, 1, "no tiles"),
        (2, 1, "2 cols"),
        (4, 1, "4 cols"),
        (2, 2, "2×2"),
        (4, 2, "4×2"),
        (4, 4, "4×4"),
        (8, 4, "8×4"),
        (8, 8, "8×8"),
    ]

    # ================================================================
    # Test on table 2 (largest: 3840×40400 = 148MB)
    # ================================================================
    t = 2
    w, h = SF_DIMS[t]
    decoded_mb = w * h / 1024 / 1024
    cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
    cold_w = sd[ek[t]][cold_order]
    q, s, zp = quantize_table(cold_w)
    q_np = q.numpy()

    print(f"\n{'='*70}")
    print(f"TABLE 2: {w}×{h} = {decoded_mb:.0f}MB decoded")
    print(f"{'='*70}")

    # Encode with different tile configs
    print(f"\n--- Encoding ---")
    print(f"{'Config':>12} {'Comp KB':>10} {'Ratio':>8} {'Encode':>8}")
    encoded_paths = {}

    for tc, tr, label in tile_configs:
        outpath = os.path.join(tmpdir, f't{t}_tc{tc}_tr{tr}.h265')
        t0 = time.perf_counter()
        comp = encode_with_tiles(q_np, w, h, outpath, crf=30, tile_cols=tc, tile_rows=tr)
        enc_time = time.perf_counter() - t0
        ratio = q_np.size / comp if comp > 0 else 0
        print(f"{label:>12} {comp/1024:>9.1f} {ratio:>7.0f}x {enc_time:>7.1f}s")
        if comp > 0:
            encoded_paths[(tc, tr)] = outpath

    # Decode benchmark
    print(f"\n--- Decode: Single frame, varying threads ---")
    print(f"{'Config':>12} {'1T':>10} {'4T':>10} {'8T':>10} {'16T':>10} {'Best':>10}")

    for tc, tr, label in tile_configs:
        key = (tc, tr)
        if key not in encoded_paths:
            continue
        path = encoded_paths[key]
        results = {}
        for nt in [1, 4, 8, 16]:
            med, mn = bench_decode(path, nt, n_reps=3)
            results[nt] = med

        best_nt = min(results, key=results.get)
        best = results[best_nt]
        tp = decoded_mb / (best / 1000)
        print(f"{label:>12} {results[1]:>9.0f}ms {results[4]:>9.0f}ms "
              f"{results[8]:>9.0f}ms {results[16]:>9.0f}ms "
              f"{best:>7.0f}ms@{best_nt}T ({tp:.0f}MB/s)")

    # ================================================================
    # Test batch decode: all 4 large tables with tiles
    # ================================================================
    print(f"\n{'='*70}")
    print(f"BATCH: 4 largest tables, best tile config")
    print(f"{'='*70}")

    # Encode all 4 tables with a few tile configs
    best_tile_configs = [(1, 1, "no tiles"), (4, 1, "4 cols"), (4, 4, "4×4"), (8, 4, "8×4")]

    for tc, tr, label in best_tile_configs:
        all_paths = []
        total_comp = 0
        total_decoded = 0
        for t in TABLES:
            w, h = SF_DIMS[t]
            cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
            cold_w = sd[ek[t]][cold_order]
            q, s, zp = quantize_table(cold_w)
            outpath = os.path.join(tmpdir, f'batch_t{t}_tc{tc}_tr{tr}.h265')
            comp = encode_with_tiles(q.numpy(), w, h, outpath, crf=30, tile_cols=tc, tile_rows=tr)
            all_paths.append(outpath)
            total_comp += comp
            total_decoded += w * h

        total_decoded_mb = total_decoded / 1024 / 1024

        # Decode all 4 in parallel with different thread configs
        print(f"\n  {label} ({total_comp/1024:.0f}KB compressed, {total_decoded_mb:.0f}MB decoded):")
        for tpf, mp in [(1, 4), (4, 4), (8, 4), (16, 4)]:
            med, mn = bench_batch(all_paths, tpf, mp, n_reps=3)
            tp = total_decoded_mb / (med / 1000)
            print(f"    {tpf}T×{mp}par: {med:.0f}ms ({tp:.0f} MB/s)")

    # ================================================================
    # Full 8 tables with best config
    # ================================================================
    print(f"\n{'='*70}")
    print(f"FULL: All 8 tables, tiles vs no-tiles")
    print(f"{'='*70}")

    ALL_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
    ALL_DIMS = {
        2: (3840, 40400), 3: (1920, 17568), 9: (1920, 744), 11: (3840, 33304),
        15: (1920, 43556), 20: (1920, 56200), 23: (1920, 2284), 25: (1920, 1140),
    }

    for tc, tr, label in [(1, 1, "no tiles"), (4, 1, "4 cols"), (4, 4, "4×4")]:
        all_paths = []
        total_comp = 0
        total_decoded = 0
        for t in ALL_TABLES:
            w, h = ALL_DIMS[t]
            cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
            cold_w = sd[ek[t]][cold_order]
            q, s, zp = quantize_table(cold_w)
            outpath = os.path.join(tmpdir, f'full_t{t}_tc{tc}_tr{tr}.h265')
            comp = encode_with_tiles(q.numpy(), w, h, outpath, crf=30, tile_cols=tc, tile_rows=tr)
            all_paths.append(outpath)
            total_comp += comp
            total_decoded += w * h

        total_decoded_mb = total_decoded / 1024 / 1024
        print(f"\n  {label}: {total_comp/1024:.0f}KB compressed, {total_decoded_mb:.0f}MB decoded")

        for tpf, mp in [(1, 8), (4, 8), (8, 8), (16, 8)]:
            med, mn = bench_batch(all_paths, tpf, mp, n_reps=3)
            tp = total_decoded_mb / (med / 1000)
            total_t = tpf * min(mp, 8)
            print(f"    {tpf}T×{mp}par ({total_t}T): {med:.0f}ms ({tp:.0f} MB/s)")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nDone.")


if __name__ == '__main__':
    main()
