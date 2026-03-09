#!/usr/bin/env python3
"""
Benchmark: Optimal frame granularity for decode time vs cache size tradeoff.

Tests different frame resolutions to find the sweet spot where:
- Each frame decodes fast enough
- Cache granularity is fine enough for small cache sizes
- Compression ratio remains reasonable
"""

import os, sys, time, json, tempfile, shutil
import numpy as np
import torch
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C

# ============================================================
# CONFIG
# ============================================================
EMB_DIM = 16
TILE_H = TILE_W = 4
TEST_BATCH_SIZE = 2048
HOTCOLD_DIR = "results/hotcold"
REORDER_DIR = "results/reorder"
ONDEMAND_DIR = "results/ondemand"

# Frame resolutions to test (width, height) - all multiples of 4
RESOLUTIONS = [
    (1920, 1080),  # 129,600 rows/frame (current)
    (960, 544),    # 32,640 rows/frame
    (480, 272),    # 8,160 rows/frame
    (256, 256),    # 4,096 rows/frame
    (128, 128),    # 1,024 rows/frame
    (64, 64),      # 256 rows/frame
]

CACHE_SIZES = [2, 4, 8, 16, 32, 64]
DECODE_ITERS = 20  # decode iterations for timing

def rows_per_frame(w, h):
    return (w // TILE_W) * (h // TILE_H)


def log(msg):
    print(msg, flush=True)


# ============================================================
# PART 1: Measure actual decode time at each resolution
# ============================================================
def measure_decode_times(cold_data_uint8, table_id=2):
    """Encode cold data at each resolution, measure decode time per frame."""

    results = {}
    num_rows = cold_data_uint8.shape[0]
    log(f"\n{'='*60}")
    log(f"PART 1: Decode time vs frame resolution (table {table_id}, {num_rows} cold rows)")
    log(f"{'='*60}")

    for width, height in RESOLUTIONS:
        rpf = rows_per_frame(width, height)
        num_frames = (num_rows + rpf - 1) // rpf
        log(f"\n--- Resolution {width}x{height} ({rpf} rows/frame, {num_frames} frames) ---")

        # Tile rows into frames
        tiled_frames = []
        for fid in range(min(num_frames, 5)):  # encode up to 5 frames for timing
            start_row = fid * rpf
            end_row = min(start_row + rpf, num_rows)
            chunk = cold_data_uint8[start_row:end_row]

            # Pad to full frame if needed
            if chunk.shape[0] < rpf:
                pad = torch.zeros(rpf - chunk.shape[0], EMB_DIM, dtype=torch.uint8)
                chunk = torch.cat([chunk, pad], dim=0)

            # Tile: each row (16 values) -> 4x4 pixel block
            frame = _C.tile_rows_to_frame(chunk, width, height)
            tiled_frames.append(frame)

        # Encode all frames to temp dir
        tmpdir = tempfile.mkdtemp(prefix=f'decode_bench_{width}x{height}_')
        try:
            total_bytes = _C.batch_encode_h265_frames(tiled_frames, tmpdir, True)
            avg_bytes = total_bytes / len(tiled_frames)
            compression_ratio = (rpf * EMB_DIM) / max(1, avg_bytes)

            log(f"  Encoded {len(tiled_frames)} frames, avg {avg_bytes/1024:.1f} KB/frame, "
                f"ratio {compression_ratio:.1f}x")

            # Measure decode time
            decode_times = []
            for iteration in range(DECODE_ITERS):
                for fid in range(len(tiled_frames)):
                    fpath = os.path.join(tmpdir, f'frame_{fid:05d}.h265')
                    t0 = time.perf_counter()
                    decoded = _C.decode_h265_frame_from_file(fpath)
                    dt = (time.perf_counter() - t0) * 1000
                    if iteration >= 2:  # skip warmup
                        decode_times.append(dt)

            avg_decode_ms = np.mean(decode_times)
            p50_decode_ms = np.percentile(decode_times, 50)
            p99_decode_ms = np.percentile(decode_times, 99)

            log(f"  Decode: avg={avg_decode_ms:.2f}ms, p50={p50_decode_ms:.2f}ms, "
                f"p99={p99_decode_ms:.2f}ms")

            results[(width, height)] = {
                'width': width, 'height': height,
                'rows_per_frame': rpf,
                'num_frames': num_frames,
                'avg_bytes_per_frame': avg_bytes,
                'compression_ratio': compression_ratio,
                'avg_decode_ms': avg_decode_ms,
                'p50_decode_ms': p50_decode_ms,
                'p99_decode_ms': p99_decode_ms,
                'total_compressed_mb': num_frames * avg_bytes / 1024 / 1024,
                'total_uncompressed_mb': num_rows * EMB_DIM / 1024 / 1024,
            }

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return results


# ============================================================
# PART 2: Simulate LRU cache at each resolution
# ============================================================
def simulate_cache(batch_cold_accesses, num_cold_rows, decode_results):
    """Simulate LRU cache for each resolution and cache size.

    batch_cold_accesses: list of sets, each set contains cold row indices accessed in that batch
    """
    log(f"\n{'='*60}")
    log(f"PART 2: LRU cache simulation ({len(batch_cold_accesses)} batches)")
    log(f"{'='*60}")

    sim_results = {}

    for (width, height), dr in decode_results.items():
        rpf = dr['rows_per_frame']
        decode_ms = dr['avg_decode_ms']

        log(f"\n--- {width}x{height} ({rpf} rows/frame, decode={decode_ms:.2f}ms) ---")

        for cache_cap in CACHE_SIZES:
            cache = OrderedDict()
            hits = 0
            misses = 0
            per_batch_miss_cost = []

            for batch_idx, cold_rows in enumerate(batch_cold_accesses):
                batch_misses = 0
                # Map cold rows to frame IDs
                frame_ids = set()
                for row in cold_rows:
                    frame_ids.add(row // rpf)

                for fid in frame_ids:
                    if fid in cache:
                        cache.move_to_end(fid)
                        hits += 1
                    else:
                        misses += 1
                        batch_misses += 1
                        cache[fid] = True
                        while len(cache) > cache_cap:
                            cache.popitem(last=False)

                per_batch_miss_cost.append(batch_misses * decode_ms)

            total = hits + misses
            hit_rate = hits / total if total > 0 else 1.0
            avg_decode_per_batch = np.mean(per_batch_miss_cost)
            p99_decode_per_batch = np.percentile(per_batch_miss_cost, 99)
            cache_mem_mb = cache_cap * rpf * EMB_DIM / 1024 / 1024

            key = (width, height, cache_cap)
            sim_results[key] = {
                'width': width, 'height': height,
                'rows_per_frame': rpf,
                'cache_capacity': cache_cap,
                'hit_rate': hit_rate,
                'total_misses': misses,
                'avg_decode_ms_per_batch': avg_decode_per_batch,
                'p99_decode_ms_per_batch': p99_decode_per_batch,
                'cache_mem_mb': cache_mem_mb,
                'decode_ms_per_frame': decode_ms,
                'compressed_mb': dr['total_compressed_mb'],
            }

            log(f"  cache={cache_cap:3d}: hit={hit_rate:.4f}, misses={misses}, "
                f"avg_decode={avg_decode_per_batch:.2f}ms/batch, "
                f"cache_mem={cache_mem_mb:.1f}MB")

    return sim_results


# ============================================================
# PART 3: Find optimal configuration
# ============================================================
def find_optimal(decode_results, sim_results):
    """Find configurations that minimize memory while keeping decode overhead low."""
    log(f"\n{'='*60}")
    log(f"PART 3: Optimal configurations")
    log(f"{'='*60}")

    # Target: decode overhead < 1ms/batch (negligible vs 2-4ms forward pass)
    TARGET_DECODE_MS = 1.0

    log(f"\nAll configs with avg decode < {TARGET_DECODE_MS}ms/batch:")
    log(f"{'Resolution':>12s} | {'RPF':>7s} | {'Cache':>5s} | {'Hit%':>6s} | "
        f"{'Dec/batch':>10s} | {'Cache MB':>9s} | {'Disk MB':>8s} | {'Total MB':>9s}")
    log("-" * 90)

    candidates = []
    for key, sr in sorted(sim_results.items()):
        if sr['avg_decode_ms_per_batch'] < TARGET_DECODE_MS:
            total_mb = sr['cache_mem_mb'] + sr['compressed_mb']
            candidates.append((key, sr, total_mb))

    # Sort by total memory
    candidates.sort(key=lambda x: x[2])

    for key, sr, total_mb in candidates:
        w, h, cap = key
        log(f"  {w}x{h:>4d} | {sr['rows_per_frame']:>7d} | {cap:>5d} | "
            f"{sr['hit_rate']*100:>5.1f}% | "
            f"{sr['avg_decode_ms_per_batch']:>8.2f}ms | "
            f"{sr['cache_mem_mb']:>7.1f}MB | "
            f"{sr['compressed_mb']:>6.1f}MB | "
            f"{total_mb:>7.1f}MB")

    # Find Pareto-optimal (minimize memory, constrained by decode overhead)
    log(f"\nPareto-optimal (lowest memory for each resolution):")
    best_per_res = {}
    for key, sr, total_mb in candidates:
        w, h, cap = key
        res = (w, h)
        if res not in best_per_res or total_mb < best_per_res[res][2]:
            best_per_res[res] = (key, sr, total_mb)

    for res in sorted(best_per_res.keys(), key=lambda r: best_per_res[r][2]):
        key, sr, total_mb = best_per_res[res]
        w, h, cap = key
        log(f"  {w}x{h}: cache={cap}, hit={sr['hit_rate']*100:.1f}%, "
            f"decode={sr['avg_decode_ms_per_batch']:.2f}ms/batch, "
            f"mem={total_mb:.1f}MB (cache={sr['cache_mem_mb']:.1f}MB + disk={sr['compressed_mb']:.1f}MB)")

    return candidates


# ============================================================
# MAIN
# ============================================================
def main():
    log("Loading cold data and access patterns...")

    # Load model and data to get batch access patterns
    from benchmark_full_comparison import load_model_and_data
    dlrm, test_ld, ln_emb, state_dict = load_model_and_data()

    # Load hot/cold split data
    test_table = 2  # largest table
    cold_indices = torch.load(os.path.join(HOTCOLD_DIR, f'cold_indices_{test_table}.pt'),
                               map_location='cpu', weights_only=True)
    hot_indices = torch.load(os.path.join(HOTCOLD_DIR, f'hot_indices_{test_table}.pt'),
                              map_location='cpu', weights_only=True)
    is_hot_t = torch.load(os.path.join(HOTCOLD_DIR, f'is_hot_{test_table}.pt'),
                           map_location='cpu', weights_only=True)

    # Load reorder mapping
    reorder_map_path = os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{test_table}.npy')
    if os.path.exists(reorder_map_path):
        o2c = torch.from_numpy(np.load(reorder_map_path).copy()).long()
    else:
        o2c = torch.load(os.path.join(REORDER_DIR, f'orig_to_cold_reordered_{test_table}.pt'),
                          map_location='cpu', weights_only=True).long()

    # Get cold embedding data
    emb_key = f'emb_l.{test_table}.weight'
    full_emb = state_dict[emb_key]
    cold_emb = full_emb[cold_indices].float()

    # Quantize to uint8
    vmin, vmax = cold_emb.min(), cold_emb.max()
    scale = (vmax - vmin) / 255.0
    zp = vmin
    cold_uint8 = ((cold_emb - zp) / scale).clamp(0, 255).to(torch.uint8)

    # Reorder according to batch-affinity ordering
    num_cold = cold_indices.shape[0]
    max_cold_idx = o2c[cold_indices].max().item() + 1
    reordered_uint8 = torch.zeros(max_cold_idx, EMB_DIM, dtype=torch.uint8)
    for i in range(num_cold):
        orig_idx = cold_indices[i].item()
        cold_pos = o2c[orig_idx].item()
        if cold_pos >= 0:
            reordered_uint8[cold_pos] = cold_uint8[i]

    log(f"Cold data: {reordered_uint8.shape[0]} rows, {reordered_uint8.numel()/1024/1024:.1f}MB uint8")

    # Collect batch access patterns (cold row indices in reordered space)
    log("Scanning batch access patterns...")
    batch_cold_accesses = []
    num_batches = 0

    for inputBatch in test_ld:
        X, lS_o, lS_i, T = inputBatch
        if isinstance(lS_i, (list, tuple)):
            indices = lS_i[test_table]
        elif lS_i.dim() == 2:
            indices = lS_i[test_table]
        else:
            indices = lS_i

        # Find cold indices and map to reordered positions
        cold_mask = ~is_hot_t[indices]
        if cold_mask.any():
            cold_orig = indices[cold_mask]
            cold_reordered = o2c[cold_orig]
            valid = cold_reordered >= 0
            if valid.any():
                batch_cold_accesses.append(set(cold_reordered[valid].tolist()))
            else:
                batch_cold_accesses.append(set())
        else:
            batch_cold_accesses.append(set())
        num_batches += 1

    cold_batches = sum(1 for s in batch_cold_accesses if len(s) > 0)
    log(f"Collected {num_batches} batches, {cold_batches} have cold accesses for table {test_table}")

    # PART 1: Measure decode times
    decode_results = measure_decode_times(reordered_uint8, test_table)

    # PART 2: Simulate cache
    sim_results = simulate_cache(batch_cold_accesses, reordered_uint8.shape[0], decode_results)

    # PART 3: Find optimal
    candidates = find_optimal(decode_results, sim_results)

    # Summary table
    log(f"\n{'='*60}")
    log(f"SUMMARY: Decode time per frame")
    log(f"{'='*60}")
    log(f"{'Resolution':>12s} | {'Rows/Frame':>10s} | {'#Frames':>7s} | "
        f"{'Decode ms':>10s} | {'Comp Ratio':>10s} | {'Disk MB':>8s}")
    log("-" * 75)
    for (w, h), dr in sorted(decode_results.items(), key=lambda x: -x[1]['rows_per_frame']):
        log(f"  {w}x{h:>4d} | {dr['rows_per_frame']:>10d} | {dr['num_frames']:>7d} | "
            f"{dr['avg_decode_ms']:>8.2f}ms | {dr['compression_ratio']:>8.1f}x | "
            f"{dr['total_compressed_mb']:>6.1f}MB")


if __name__ == '__main__':
    main()
