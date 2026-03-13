#!/usr/bin/env python3
"""
Cache size sweep: measure actual E2E latency for different LRU cache sizes.

Tests cache sizes 1, 2, 4, 8, 16, 32, 64 with Zstd-19 compressed cold storage.
For each: run full inference, measure batch latency, cache hit rate, total time.
"""

import os, sys, time, json
import numpy as np
import torch
from collections import OrderedDict
import zstandard as zstd

sys.path.insert(0, '.')

RESULTS_DIR = "results"
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "codec_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMB_DIM = 16
ROWS_PER_FRAME = 129600
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return 0


class ZstdFrameCache:
    """Zstd compressed frames with LRU cache."""

    def __init__(self, compressed_frames, frame_n_rows, scale, zp, cache_size=16):
        self.compressed_frames = compressed_frames
        self.frame_n_rows = frame_n_rows
        self.scale = scale
        self.zp = zp
        self.cache_size = cache_size
        self.dctx = zstd.ZstdDecompressor()
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get_frame(self, frame_id):
        if frame_id in self.cache:
            self.cache.move_to_end(frame_id)
            self.hits += 1
            return self.cache[frame_id]
        self.misses += 1
        raw = self.dctx.decompress(self.compressed_frames[frame_id])
        n_rows = self.frame_n_rows[frame_id]
        uint8_data = np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, EMB_DIM)
        fp32_data = (uint8_data.astype(np.float32) - self.zp) * self.scale
        decoded = torch.from_numpy(fp32_data)
        self.cache[frame_id] = decoded
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return decoded

    def reset_stats(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


def main():
    from benchmark_full_comparison import load_model_and_data, compute_metrics, run_inference

    dlrm, test_ld, ln_emb, state_dict = load_model_and_data()

    # ---- Baseline ----
    log(f"\n{'='*60}")
    log("BASELINE")
    log(f"{'='*60}")

    baseline = run_inference(dlrm, test_ld, tag="baseline")
    log(f"  AUC={baseline['auc']:.6f}, BLat={baseline['mean_lat_ms']:.2f}ms")

    # ---- Build compressed cold storage ----
    log(f"\n{'='*60}")
    log("BUILDING ZSTD-19 COLD STORAGE")
    log(f"{'='*60}")

    table_info = {}
    for t_idx in LARGE_TABLES:
        cold_idx_path = os.path.join(HOTCOLD_DIR, f"cold_indices_{t_idx}.pt")
        if not os.path.exists(cold_idx_path):
            continue
        cold_idx = torch.load(cold_idx_path, weights_only=False)
        if cold_idx.numel() == 0:
            continue

        key = f'emb_l.{t_idx}.weight'
        weight = state_dict[key]
        cold_weight = weight[cold_idx]

        mn, mx = cold_weight.min().item(), cold_weight.max().item()
        scale = (mx - mn) / 255.0
        if scale == 0: scale = 1.0
        zp = round(-mn / scale)
        quant = ((cold_weight / scale).round() + zp).clamp(0, 255).to(torch.uint8).numpy()

        # Apply reordering
        order_path = os.path.join(REORDER_DIR, f"cold_order_{t_idx}.npy")
        if os.path.exists(order_path):
            order = np.load(order_path)
            quant = quant[order]
            cold_idx = cold_idx[torch.from_numpy(order.astype(np.int64))]

        # Build mappings
        is_hot = torch.ones(weight.shape[0], dtype=torch.bool)
        is_hot[cold_idx] = False
        orig_to_cold = torch.full((weight.shape[0],), -1, dtype=torch.int32)
        orig_to_cold[cold_idx] = torch.arange(len(cold_idx), dtype=torch.int32)

        # Compress frames
        n_rows = quant.shape[0]
        n_frames = (n_rows + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
        cctx = zstd.ZstdCompressor(level=19)
        compressed_frames = []
        frame_n_rows = []
        for i in range(n_frames):
            start = i * ROWS_PER_FRAME
            end = min(start + ROWS_PER_FRAME, n_rows)
            compressed_frames.append(cctx.compress(quant[start:end].tobytes()))
            frame_n_rows.append(end - start)

        table_info[t_idx] = {
            'cold_idx': cold_idx,
            'is_hot': is_hot,
            'orig_to_cold': orig_to_cold,
            'compressed_frames': compressed_frames,
            'frame_n_rows': frame_n_rows,
            'scale': scale,
            'zp': zp,
            'n_frames': n_frames,
        }
        log(f"  Table {t_idx}: {len(cold_idx):,} cold, {n_frames} frames")

    # ---- Cache size sweep ----
    cache_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = {'baseline': baseline}

    for cache_size in cache_sizes:
        log(f"\n{'='*60}")
        log(f"CACHE SIZE = {cache_size}")
        log(f"{'='*60}")

        # Build caches
        caches = {}
        for t_idx, info in table_info.items():
            caches[t_idx] = ZstdFrameCache(
                info['compressed_frames'], info['frame_n_rows'],
                info['scale'], info['zp'], cache_size=cache_size
            )

        # Restore model + zero cold
        for t_idx, info in table_info.items():
            key = f'emb_l.{t_idx}.weight'
            hot_weight = state_dict[key].clone()
            hot_weight[info['cold_idx']] = 0
            dlrm.emb_l[t_idx].weight.data = hot_weight

        # Run inference
        max_samples = 2000 * 2048
        all_scores = np.empty(max_samples, dtype=np.float32)
        all_targets = np.empty(max_samples, dtype=np.float32)
        sample_idx = 0
        blats = []

        t_total_start = time.time()
        with torch.no_grad():
            for inputBatch in test_ld:
                X, lS_o, lS_i, T = inputBatch[0], inputBatch[1], inputBatch[2], inputBatch[3]
                bt0 = time.time()

                # Fill cold embeddings
                for t_idx, info in table_info.items():
                    cache = caches[t_idx]
                    if isinstance(lS_i, list) or isinstance(lS_i, tuple):
                        indices = lS_i[t_idx]
                    elif lS_i.dim() == 2:
                        indices = lS_i[t_idx]
                    else:
                        indices = lS_i

                    cold_mask = ~info['is_hot'][indices]
                    if not cold_mask.any():
                        continue

                    cold_orig = indices[cold_mask]
                    cold_reordered = info['orig_to_cold'][cold_orig].long()
                    frame_ids = (cold_reordered // ROWS_PER_FRAME).unique().tolist()

                    frame_ids_all = cold_reordered // ROWS_PER_FRAME
                    rows_in_frame = cold_reordered % ROWS_PER_FRAME
                    gathered = torch.zeros(len(cold_orig), EMB_DIM)

                    # Process each frame: decode + gather immediately
                    for fid in frame_ids:
                        frame_data = cache.get_frame(fid)
                        fmask = (frame_ids_all == fid)
                        if fmask.any():
                            gathered[fmask] = frame_data[rows_in_frame[fmask]]

                    dlrm.emb_l[t_idx].weight.data[cold_orig] = gathered

                Z = dlrm(X, lS_o, lS_i)
                blats.append(time.time() - bt0)

                z_np = Z.detach().cpu().numpy().ravel()
                t_np = T.detach().cpu().numpy().ravel()
                bs = z_np.shape[0]
                all_scores[sample_idx:sample_idx+bs] = z_np
                all_targets[sample_idx:sample_idx+bs] = t_np
                sample_idx += bs

        total_time = time.time() - t_total_start
        scores, targets = all_scores[:sample_idx], all_targets[:sample_idx]
        auc, ll, acc = compute_metrics(scores, targets)

        total_hits = sum(c.hits for c in caches.values())
        total_misses = sum(c.misses for c in caches.values())
        hit_rate = total_hits / max(total_hits + total_misses, 1)

        log(f"  AUC={auc:.6f} (delta={auc - baseline['auc']:+.6f})")
        log(f"  BLat={np.mean(blats)*1000:.2f}ms (p50={np.percentile(blats,50)*1000:.2f}ms)")
        log(f"  Total={total_time:.1f}s")
        log(f"  Cache: hits={total_hits}, misses={total_misses}, rate={hit_rate:.4f}")
        log(f"  RSS={get_rss_mb():.0f}MB")

        results[f'cache_{cache_size}'] = {
            'cache_size': cache_size,
            'auc': auc,
            'auc_delta': auc - baseline['auc'],
            'mean_lat_ms': np.mean(blats) * 1000,
            'p50_lat_ms': np.percentile(blats, 50) * 1000,
            'p99_lat_ms': np.percentile(blats, 99) * 1000,
            'total_time': total_time,
            'cache_hits': total_hits,
            'cache_misses': total_misses,
            'hit_rate': hit_rate,
            'rss_mb': get_rss_mb(),
        }

        # Restore
        for t_idx in table_info:
            key = f'emb_l.{t_idx}.weight'
            dlrm.emb_l[t_idx].weight.data = state_dict[key].clone()

    # ---- Summary ----
    log(f"\n{'='*60}")
    log("CACHE SIZE SWEEP SUMMARY")
    log(f"{'='*60}\n")

    log(f"  {'Cache':>6s} | {'AUC':>10s} | {'BLat':>8s} | {'p99':>8s} | "
        f"{'Hits':>8s} | {'Misses':>8s} | {'Hit%':>8s}")
    log(f"  {'-'*70}")

    log(f"  {'base':>6s} | {baseline['auc']:>10.6f} | {baseline['mean_lat_ms']:>6.2f}ms | "
        f"{baseline['p99_lat_ms']:>6.2f}ms | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>8s}")

    for cs in cache_sizes:
        r = results[f'cache_{cs}']
        log(f"  {cs:>6d} | {r['auc']:>10.6f} | {r['mean_lat_ms']:>6.2f}ms | "
            f"{r['p99_lat_ms']:>6.2f}ms | {r['cache_hits']:>8d} | "
            f"{r['cache_misses']:>8d} | {r['hit_rate']:>7.2%}")

    # Save
    json_path = os.path.join(OUTPUT_DIR, 'cache_sweep.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
