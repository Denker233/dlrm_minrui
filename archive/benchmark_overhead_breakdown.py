#!/usr/bin/env python3
"""
Fine-grained breakdown of per-batch overhead in the C++ fused cold lookup path.

Measures each sub-step:
1. Building lS_i_for_scan (Python list comprehension)
2. C++ scan_needed_frames
3. Cache ensure (decode on miss)
4. Building cached_frames_list
5. C++ gather_cold_embeddings
6. Writeback to model weights
"""

import os, sys, time, json
import numpy as np
import torch
from collections import OrderedDict
import zstandard as zstd

sys.path.insert(0, '.')
import compressed_emb as _C

RESULTS_DIR = "results"
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
REORDER_DIR = os.path.join(RESULTS_DIR, "reorder")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "codec_comparison")

EMB_DIM = 16
ROWS_PER_FRAME = 129600
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
CACHE_SIZE = 16
MODEL_PATH = "./models/dlrm_kaggle_correct.pt"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class TableStorage:
    def __init__(self, t_idx, cold_idx, weight, scale, zp, level=19):
        self.t_idx = t_idx
        self.scale = scale
        self.zp = zp
        self.emb_dim = weight.shape[1]
        self.n_total = weight.shape[0]

        quant = ((weight[cold_idx] / scale).round() + zp).clamp(0, 255).to(torch.uint8).numpy()
        order_path = os.path.join(REORDER_DIR, f"cold_order_{t_idx}.npy")
        if os.path.exists(order_path):
            order = np.load(order_path)
            quant = quant[order]
            cold_idx = cold_idx[torch.from_numpy(order.astype(np.int64))]

        self.cold_idx = cold_idx
        self.n_cold = len(cold_idx)
        self.orig_to_cold = torch.full((self.n_total,), -1, dtype=torch.int32)
        self.orig_to_cold[self.cold_idx] = torch.arange(self.n_cold, dtype=torch.int32)
        self.is_hot = torch.ones(self.n_total, dtype=torch.bool)
        self.is_hot[self.cold_idx] = False

        self.n_frames = (self.n_cold + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
        cctx = zstd.ZstdCompressor(level=level)
        self.compressed_frames = []
        self.frame_n_rows = []
        for i in range(self.n_frames):
            start = i * ROWS_PER_FRAME
            end = min(start + ROWS_PER_FRAME, self.n_cold)
            self.compressed_frames.append(cctx.compress(quant[start:end].tobytes()))
            self.frame_n_rows.append(end - start)

        self.dctx = zstd.ZstdDecompressor()
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def decode_frame(self, fid):
        raw = self.dctx.decompress(self.compressed_frames[fid])
        n_rows = self.frame_n_rows[fid]
        uint8_data = np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, self.emb_dim)
        fp32_data = (uint8_data.astype(np.float32) - self.zp) * self.scale
        return torch.from_numpy(fp32_data).contiguous()

    def get_frame(self, fid):
        if fid in self.cache:
            self.cache.move_to_end(fid)
            self.cache_hits += 1
            return self.cache[fid]
        self.cache_misses += 1
        decoded = self.decode_frame(fid)
        self.cache[fid] = decoded
        if len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)
        return decoded


def main():
    log("Loading model...")
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']

    log("Building cold storage...")
    storages = {}
    for t_idx in LARGE_TABLES:
        cold_idx = torch.load(os.path.join(HOTCOLD_DIR, f"cold_indices_{t_idx}.pt"), weights_only=False)
        if cold_idx.numel() == 0:
            continue
        weight = sd[f'emb_l.{t_idx}.weight']
        mn, mx = weight[cold_idx].min().item(), weight[cold_idx].max().item()
        scale = (mx - mn) / 255.0
        if scale == 0: scale = 1.0
        zp = round(-mn / scale)
        storages[t_idx] = TableStorage(t_idx, cold_idx, weight, scale, zp)

    comp_tables = sorted(storages.keys())
    is_hot_list = [storages[t].is_hot for t in comp_tables]
    o2c_map_list = [storages[t].orig_to_cold for t in comp_tables]

    # Pre-allocate empty frame placeholder
    empty_frame = torch.empty(0, dtype=torch.float32)

    log("Loading test data...")
    from benchmark_full_comparison import load_model_and_data
    dlrm, test_ld, _, _ = load_model_and_data()

    # Set up model weights
    for t_idx, storage in storages.items():
        hot_weight = sd[f'emb_l.{t_idx}.weight'].clone()
        hot_weight[storage.cold_idx] = 0
        dlrm.emb_l[t_idx].weight.data = hot_weight

    # Warm up caches
    log("Warming up...")
    with torch.no_grad():
        for i, batch in enumerate(test_ld):
            if i >= 50:
                break
            lS_i = batch[2]
            lS_i_for_scan = []
            for t_idx in comp_tables:
                if isinstance(lS_i, (list, tuple)):
                    lS_i_for_scan.append(lS_i[t_idx].long())
                elif lS_i.dim() == 2:
                    lS_i_for_scan.append(lS_i[t_idx].long())
                else:
                    lS_i_for_scan.append(lS_i.long())

            frame_lists = _C.scan_needed_frames(lS_i_for_scan, is_hot_list, o2c_map_list, ROWS_PER_FRAME)
            for k, t_idx in enumerate(comp_tables):
                needed = frame_lists[k]
                if needed.numel() > 0:
                    for fid in needed.tolist():
                        storages[t_idx].get_frame(fid)

    # Benchmark with fine-grained timing
    log("Benchmarking overhead breakdown...")
    t_build_lsi = []
    t_scan = []
    t_decode = []
    t_build_frames = []
    t_gather = []
    t_writeback = []

    with torch.no_grad():
        for batch in test_ld:
            X, lS_o, lS_i, T = batch[0], batch[1], batch[2], batch[3]

            # 1. Build lS_i_for_scan
            t0 = time.time()
            lS_i_for_scan = []
            for t_idx in comp_tables:
                if isinstance(lS_i, (list, tuple)):
                    lS_i_for_scan.append(lS_i[t_idx].long())
                elif lS_i.dim() == 2:
                    lS_i_for_scan.append(lS_i[t_idx].long())
                else:
                    lS_i_for_scan.append(lS_i.long())
            t_build_lsi.append(time.time() - t0)

            # 2. C++ scan
            t0 = time.time()
            frame_lists = _C.scan_needed_frames(lS_i_for_scan, is_hot_list, o2c_map_list, ROWS_PER_FRAME)
            t_scan.append(time.time() - t0)

            # 3. Decode on miss
            t0 = time.time()
            for k, t_idx in enumerate(comp_tables):
                needed = frame_lists[k]
                if needed.numel() > 0:
                    for fid in needed.tolist():
                        storages[t_idx].get_frame(fid)
            t_decode.append(time.time() - t0)

            # 4. Build cached_frames_list
            t0 = time.time()
            cached_frames_list = []
            frame_offsets_list = []
            for t_idx in comp_tables:
                storage = storages[t_idx]
                frames = []
                for fid in range(storage.n_frames):
                    if fid in storage.cache:
                        frames.append(storage.cache[fid])
                    else:
                        frames.append(empty_frame)
                cached_frames_list.append(frames)
                frame_offsets_list.append(0)
            t_build_frames.append(time.time() - t0)

            # 5. C++ gather
            t0 = time.time()
            gather_results = _C.gather_cold_embeddings(
                lS_i_for_scan, is_hot_list, o2c_map_list,
                cached_frames_list, frame_offsets_list,
                ROWS_PER_FRAME, EMB_DIM
            )
            t_gather.append(time.time() - t0)

            # 6. Writeback
            t0 = time.time()
            for k, t_idx in enumerate(comp_tables):
                gathered = gather_results[2*k]
                orig_indices = gather_results[2*k+1]
                if gathered.size(0) > 0:
                    dlrm.emb_l[t_idx].weight.data[orig_indices] = gathered
            t_writeback.append(time.time() - t0)

    log(f"\n{'='*60}")
    log("OVERHEAD BREAKDOWN (per batch, ms)")
    log(f"{'='*60}")

    steps = [
        ("1. Build lS_i list", t_build_lsi),
        ("2. C++ scan", t_scan),
        ("3. Decode (cache miss)", t_decode),
        ("4. Build frames list", t_build_frames),
        ("5. C++ gather", t_gather),
        ("6. Writeback", t_writeback),
    ]

    total_mean = 0
    for label, times in steps:
        mean_us = np.mean(times) * 1e6
        p50_us = np.percentile(times, 50) * 1e6
        total_mean += np.mean(times) * 1000
        log(f"  {label:<25s}: {mean_us:>7.1f} us (p50={p50_us:.1f} us)")

    log(f"  {'TOTAL':<25s}: {total_mean:>7.1f} us")

    results = {}
    for label, times in steps:
        key = label.split(". ")[1].replace(" ", "_").lower()
        results[key] = {
            'mean_us': np.mean(times) * 1e6,
            'p50_us': np.percentile(times, 50) * 1e6,
            'p99_us': np.percentile(times, 99) * 1e6,
        }

    json_path = os.path.join(OUTPUT_DIR, 'overhead_breakdown.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved to {json_path}")


if __name__ == '__main__':
    main()
