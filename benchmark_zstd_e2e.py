#!/usr/bin/env python3
"""
End-to-end inference with Zstd-compressed cold embeddings + LRU cache.

Runs actual DLRM inference using:
1. Baseline (fp32, all in memory)
2. H.265 compressed cold with LRU cache (existing approach)
3. Zstd-19 compressed cold with LRU cache
4. Zstd-3 compressed cold with LRU cache

For each, measures: AUC, batch latency, total time, RSS.
"""

import os, sys, time, json, gc
import numpy as np
import torch
from collections import OrderedDict
import zstandard as zstd

sys.path.insert(0, '.')

RESULTS_DIR = "results"
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
ONDEMAND_DIR = os.path.join(RESULTS_DIR, "ondemand")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "codec_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMB_DIM = 16
ROWS_PER_FRAME = 129600  # (1920/4) * (1080/4)
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
CACHE_SIZE = 16


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return 0


class ZstdColdStorage:
    """Cold embedding storage using Zstd compression with LRU frame cache."""

    def __init__(self, t_idx, cold_idx, weight, scale, zp, level=19, cache_size=CACHE_SIZE):
        self.t_idx = t_idx
        self.cold_idx = cold_idx
        self.n_cold = len(cold_idx)
        self.scale = scale
        self.zp = zp
        self.emb_dim = weight.shape[1]
        self.cache_size = cache_size

        # Quantize cold rows
        quant = ((weight[cold_idx] / scale).round() + zp).clamp(0, 255).to(torch.uint8).numpy()

        # Apply reordering
        order_path = os.path.join(REORDER_DIR, f"cold_order_{t_idx}.npy")
        if os.path.exists(order_path):
            order = np.load(order_path)
            quant = quant[order]
            self.cold_idx = cold_idx[torch.from_numpy(order.astype(np.int64))]

        # Build orig_idx -> cold_reordered_idx mapping
        self.orig_to_cold = torch.full((weight.shape[0],), -1, dtype=torch.int32)
        self.orig_to_cold[self.cold_idx] = torch.arange(self.n_cold, dtype=torch.int32)

        # Build is_hot boolean mask
        self.is_hot = torch.ones(weight.shape[0], dtype=torch.bool)
        self.is_hot[self.cold_idx] = False

        # Compress frames
        self.n_frames = (self.n_cold + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
        cctx = zstd.ZstdCompressor(level=level)
        self.compressed_frames = []
        self.frame_n_rows = []
        total_compressed = 0
        for i in range(self.n_frames):
            start = i * ROWS_PER_FRAME
            end = min(start + ROWS_PER_FRAME, self.n_cold)
            frame_data = quant[start:end]
            compressed = cctx.compress(frame_data.tobytes())
            self.compressed_frames.append(compressed)
            self.frame_n_rows.append(end - start)
            total_compressed += len(compressed)

        self.total_compressed_bytes = total_compressed
        self.total_raw_bytes = self.n_cold * self.emb_dim

        # LRU cache: frame_id -> decoded fp32 tensor
        self.cache = OrderedDict()
        self.dctx = zstd.ZstdDecompressor()
        self.cache_hits = 0
        self.cache_misses = 0

    def decode_frame(self, frame_id):
        """Decode a compressed frame to fp32 tensor."""
        raw = self.dctx.decompress(self.compressed_frames[frame_id])
        n_rows = self.frame_n_rows[frame_id]
        uint8_data = np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, self.emb_dim)
        fp32_data = (uint8_data.astype(np.float32) - self.zp) * self.scale
        return torch.from_numpy(fp32_data)

    def get_frame(self, frame_id):
        """Get frame from cache, decode on miss."""
        if frame_id in self.cache:
            self.cache.move_to_end(frame_id)
            self.cache_hits += 1
            return self.cache[frame_id]
        # Cache miss
        self.cache_misses += 1
        decoded = self.decode_frame(frame_id)
        self.cache[frame_id] = decoded
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return decoded

    def lookup(self, indices):
        """Look up embeddings for given original indices.
        Returns fp32 embeddings for cold indices, None-filled for hot."""
        cold_mask = ~self.is_hot[indices]
        if not cold_mask.any():
            return None, cold_mask

        cold_orig_indices = indices[cold_mask]
        cold_reordered = self.orig_to_cold[cold_orig_indices]

        # Determine needed frames
        frame_ids = (cold_reordered // ROWS_PER_FRAME).unique().tolist()

        # Ensure frames are cached
        for fid in frame_ids:
            self.get_frame(fid)

        # Vectorized gather from cached frames
        frame_ids_all = cold_reordered // ROWS_PER_FRAME
        rows_in_frame = cold_reordered % ROWS_PER_FRAME
        n = cold_mask.sum().item()
        result = torch.zeros(n, self.emb_dim)

        # Group by frame for efficient gather
        for fid in frame_ids:
            frame_data = self.cache[fid]
            mask_fid = (frame_ids_all == fid)
            if mask_fid.any():
                row_indices = rows_in_frame[mask_fid]
                result[mask_fid] = frame_data[row_indices]

        return result, cold_mask


def main():
    from benchmark_full_comparison import load_model_and_data, compute_metrics, run_inference

    dlrm, test_ld, ln_emb, state_dict = load_model_and_data()

    # ---- Baseline ----
    log(f"\n{'='*60}")
    log("BASELINE (fp32, all in memory)")
    log(f"{'='*60}")

    baseline = run_inference(dlrm, test_ld, tag="baseline")
    log(f"  AUC={baseline['auc']:.6f}, BLat={baseline['mean_lat_ms']:.2f}ms, "
        f"Total={baseline['total_time']:.1f}s, RSS={baseline['rss_mb']:.0f}MB")

    # ---- Build Zstd cold storage for all large tables ----
    log(f"\n{'='*60}")
    log("BUILDING ZSTD COLD STORAGE")
    log(f"{'='*60}")

    cold_storages = {}
    for t_idx in LARGE_TABLES:
        cold_idx_path = os.path.join(HOTCOLD_DIR, f"cold_indices_{t_idx}.pt")
        hot_idx_path = os.path.join(HOTCOLD_DIR, f"hot_indices_{t_idx}.pt")
        if not os.path.exists(cold_idx_path):
            continue
        cold_idx = torch.load(cold_idx_path, weights_only=False)
        hot_idx = torch.load(hot_idx_path, weights_only=False)
        if cold_idx.numel() == 0:
            continue

        key = f'emb_l.{t_idx}.weight'
        weight = state_dict[key]

        mn, mx = weight[cold_idx].min().item(), weight[cold_idx].max().item()
        scale = (mx - mn) / 255.0
        if scale == 0: scale = 1.0
        zp = round(-mn / scale)

        for level_name, level in [('Zstd-19', 19), ('Zstd-3', 3)]:
            storage = ZstdColdStorage(t_idx, cold_idx, weight, scale, zp,
                                      level=level, cache_size=CACHE_SIZE)
            if level_name not in cold_storages:
                cold_storages[level_name] = {}
            cold_storages[level_name][t_idx] = storage
            if level == 19:
                log(f"  Table {t_idx}: {storage.n_cold:,} cold, {storage.n_frames} frames, "
                    f"compressed={storage.total_compressed_bytes/1e6:.1f}MB "
                    f"(ratio={storage.total_raw_bytes/storage.total_compressed_bytes:.1f}x)")

    # Try to use C++ scan
    try:
        import compressed_emb as _C
        has_cpp_scan = True
        log("C++ scan_needed_frames available")
    except ImportError:
        has_cpp_scan = False

    # ---- Run inference with Zstd cold storage ----
    for comp_name in ['Zstd-19', 'Zstd-3']:
        log(f"\n{'='*60}")
        log(f"{comp_name} + LRU cache={CACHE_SIZE}")
        log(f"{'='*60}")

        storages = cold_storages[comp_name]
        comp_tables = sorted(storages.keys())

        # Build C++ scan inputs (once, not per batch)
        if has_cpp_scan:
            is_hot_list = [storages[t].is_hot for t in comp_tables]
            o2c_map_list = [storages[t].orig_to_cold for t in comp_tables]

        # Replace cold embeddings with quantized versions in the model
        for t_idx, storage in storages.items():
            key = f'emb_l.{t_idx}.weight'
            hot_weight = state_dict[key].clone()
            hot_weight[storage.cold_idx] = 0
            dlrm.emb_l[t_idx].weight.data = hot_weight

        # Custom inference with Zstd cold lookup
        max_samples = 2000 * 2048
        all_scores = np.empty(max_samples, dtype=np.float32)
        all_targets = np.empty(max_samples, dtype=np.float32)
        sample_idx = 0
        blats = []
        scan_times = []
        decode_times = []

        # Reset cache stats
        for s in storages.values():
            s.cache.clear()
            s.cache_hits = 0
            s.cache_misses = 0

        t_total_start = time.time()
        with torch.no_grad():
            for inputBatch in test_ld:
                X, lS_o, lS_i, T = inputBatch[0], inputBatch[1], inputBatch[2], inputBatch[3]
                bt0 = time.time()

                t_scan = time.time()

                if has_cpp_scan:
                    # C++ fused scan: find needed frames for all tables at once
                    lS_i_for_scan = []
                    for t_idx in comp_tables:
                        if isinstance(lS_i, list) or isinstance(lS_i, tuple):
                            lS_i_for_scan.append(lS_i[t_idx].long())
                        elif lS_i.dim() == 2:
                            lS_i_for_scan.append(lS_i[t_idx].long())
                        else:
                            lS_i_for_scan.append(lS_i.long())

                    frame_lists = _C.scan_needed_frames(
                        lS_i_for_scan, is_hot_list, o2c_map_list, ROWS_PER_FRAME
                    )

                    # Process each table
                    for k, t_idx in enumerate(comp_tables):
                        storage = storages[t_idx]
                        needed_frames = frame_lists[k]
                        if needed_frames.numel() == 0:
                            continue

                        frame_ids = needed_frames.tolist()

                        # Pre-decode needed frames
                        t_dec = time.time()
                        for fid in frame_ids:
                            storage.get_frame(fid)
                        decode_times.append(time.time() - t_dec)

                        # Get cold indices for this table
                        if isinstance(lS_i, list) or isinstance(lS_i, tuple):
                            indices = lS_i[t_idx]
                        elif lS_i.dim() == 2:
                            indices = lS_i[t_idx]
                        else:
                            indices = lS_i

                        cold_mask = ~storage.is_hot[indices]
                        if not cold_mask.any():
                            continue

                        cold_orig = indices[cold_mask]
                        cold_reordered = storage.orig_to_cold[cold_orig].long()

                        # Vectorized gather
                        frame_ids_all = cold_reordered // ROWS_PER_FRAME
                        rows_in_frame = cold_reordered % ROWS_PER_FRAME
                        gathered = torch.zeros(len(cold_orig), storage.emb_dim)
                        for fid in frame_ids:
                            fmask = (frame_ids_all == fid)
                            if fmask.any():
                                gathered[fmask] = storage.cache[fid][rows_in_frame[fmask]]
                        dlrm.emb_l[t_idx].weight.data[cold_orig] = gathered

                else:
                    # Python scan fallback
                    for t_idx, storage in storages.items():
                        if isinstance(lS_i, list) or isinstance(lS_i, tuple):
                            indices = lS_i[t_idx]
                        elif lS_i.dim() == 2:
                            indices = lS_i[t_idx]
                        else:
                            indices = lS_i

                        cold_mask = ~storage.is_hot[indices]
                        if not cold_mask.any():
                            continue

                        cold_orig = indices[cold_mask]
                        cold_reordered = storage.orig_to_cold[cold_orig].long()
                        frame_ids = (cold_reordered // ROWS_PER_FRAME).unique().tolist()

                        t_dec = time.time()
                        for fid in frame_ids:
                            storage.get_frame(fid)
                        decode_times.append(time.time() - t_dec)

                        frame_ids_all = cold_reordered // ROWS_PER_FRAME
                        rows_in_frame = cold_reordered % ROWS_PER_FRAME
                        gathered = torch.zeros(len(cold_orig), storage.emb_dim)
                        for fid in frame_ids:
                            fmask = (frame_ids_all == fid)
                            if fmask.any():
                                gathered[fmask] = storage.cache[fid][rows_in_frame[fmask]]
                        dlrm.emb_l[t_idx].weight.data[cold_orig] = gathered

                scan_times.append(time.time() - t_scan)

                # Forward pass
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

        total_hits = sum(s.cache_hits for s in storages.values())
        total_misses = sum(s.cache_misses for s in storages.values())
        hit_rate = total_hits / max(total_hits + total_misses, 1)

        total_compressed = sum(s.total_compressed_bytes for s in storages.values())

        scan_method = "C++ fused" if has_cpp_scan else "Python"
        log(f"  AUC={auc:.6f} (delta={auc - baseline['auc']:+.6f})")
        log(f"  BLat={np.mean(blats)*1000:.2f}ms (p50={np.percentile(blats,50)*1000:.2f}ms)")
        log(f"  Total={total_time:.1f}s")
        log(f"  Scan ({scan_method})+decode={np.mean(scan_times)*1000:.2f}ms/batch, "
            f"decode_only={np.mean(decode_times)*1000:.3f}ms/batch")
        log(f"  Cache: hits={total_hits}, misses={total_misses}, rate={hit_rate:.4f}")
        log(f"  Compressed: {total_compressed/1e6:.1f}MB")
        log(f"  RSS={get_rss_mb():.0f}MB")

        # Restore model weights
        for t_idx in storages:
            key = f'emb_l.{t_idx}.weight'
            dlrm.emb_l[t_idx].weight.data = state_dict[key].clone()

    # ---- Summary ----
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    log(f"  Baseline: AUC={baseline['auc']:.6f}, BLat={baseline['mean_lat_ms']:.2f}ms")
    log(f"  See above for Zstd-19 and Zstd-3 results")


if __name__ == '__main__':
    main()
