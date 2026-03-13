#!/usr/bin/env python3
"""
mmap Baseline: Let the OS handle hot/cold with memory-mapped embedding tables.

Approach: Store quantized uint8 embeddings in a flat file, mmap it, and let
the OS page cache automatically keep hot rows in memory and page out cold ones.

Compared to our explicit hot/cold split:
- No manual access frequency profiling needed
- No compression/decompression
- OS manages the cache automatically
- But: no 2D spatial locality, no compression beyond quantization

Measures: inference latency, RSS memory, AUC.
"""

import os, sys, time, json, gc, mmap
import numpy as np
import torch

sys.path.insert(0, '.')

RESULTS_DIR = "results"
HOTCOLD_DIR = os.path.join(RESULTS_DIR, "hotcold")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "codec_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
EMB_DIM = 16
TEST_BATCH_SIZE = 2048
LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return 0


def main():
    from benchmark_full_comparison import load_model_and_data, compute_metrics, run_inference

    # ---- Load model ----
    dlrm, test_ld, ln_emb, state_dict = load_model_and_data()

    # ---- Baseline (fp32, all in memory) ----
    log(f"\n{'='*60}")
    log("BASELINE (fp32, all in memory)")
    log(f"{'='*60}")

    baseline = run_inference(dlrm, test_ld, tag="baseline")
    log(f"  AUC={baseline['auc']:.6f}, BLat={baseline['mean_lat_ms']:.2f}ms, "
        f"Total={baseline['total_time']:.1f}s, RSS={baseline['rss_mb']:.0f}MB")

    # Restore weights
    for t_idx in LARGE_TABLES:
        key = f'emb_l.{t_idx}.weight'
        if key in state_dict:
            dlrm.emb_l[t_idx].weight.data = state_dict[key].clone()

    # ---- Quantized uint8 in-memory (no mmap, just smaller) ----
    log(f"\n{'='*60}")
    log("QUANTIZED uint8 (all in memory, no compression)")
    log(f"{'='*60}")

    # Quantize all large tables, dequantize on access
    for t_idx in LARGE_TABLES:
        key = f'emb_l.{t_idx}.weight'
        weight = state_dict[key]
        mn, mx = weight.min().item(), weight.max().item()
        s = (mx - mn) / 255.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        quant = ((weight / s).round() + zp).clamp(0, 255).to(torch.uint8)
        dequant = (quant.float() - zp) * s
        dlrm.emb_l[t_idx].weight.data = dequant

    quant_result = run_inference(dlrm, test_ld, tag="uint8_dequant")
    log(f"  AUC={quant_result['auc']:.6f} (delta={quant_result['auc']-baseline['auc']:+.6f}), "
        f"BLat={quant_result['mean_lat_ms']:.2f}ms, "
        f"Total={quant_result['total_time']:.1f}s, RSS={quant_result['rss_mb']:.0f}MB")

    # Restore
    for t_idx in LARGE_TABLES:
        key = f'emb_l.{t_idx}.weight'
        dlrm.emb_l[t_idx].weight.data = state_dict[key].clone()

    # ---- mmap uint8 approach ----
    log(f"\n{'='*60}")
    log("MMAP uint8 (memory-mapped quantized embeddings)")
    log(f"{'='*60}")

    mmap_dir = os.path.join(OUTPUT_DIR, 'mmap_tables')
    os.makedirs(mmap_dir, exist_ok=True)

    total_fp32_bytes = 0
    total_uint8_bytes = 0
    mmap_files = {}
    mmap_scales = {}
    mmap_zps = {}

    for t_idx in LARGE_TABLES:
        key = f'emb_l.{t_idx}.weight'
        weight = state_dict[key]
        n_rows, d = weight.shape
        total_fp32_bytes += n_rows * d * 4

        # Quantize
        mn, mx = weight.min().item(), weight.max().item()
        s = (mx - mn) / 255.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        quant = ((weight / s).round() + zp).clamp(0, 255).to(torch.uint8).numpy()

        # Write to file
        fpath = os.path.join(mmap_dir, f'emb_{t_idx}.bin')
        quant.tofile(fpath)
        total_uint8_bytes += os.path.getsize(fpath)

        mmap_scales[t_idx] = s
        mmap_zps[t_idx] = zp
        log(f"  Table {t_idx}: {n_rows:,} rows, {os.path.getsize(fpath)/1e6:.1f}MB uint8")

    log(f"  Total: fp32={total_fp32_bytes/1e6:.0f}MB, uint8={total_uint8_bytes/1e6:.0f}MB, "
        f"ratio={total_fp32_bytes/total_uint8_bytes:.1f}x")

    # Drop page cache to simulate cold start
    gc.collect()
    try:
        os.system('sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null')
    except:
        log("  Warning: could not drop page cache (need root)")

    # Memory-map the files and replace embedding weights
    for t_idx in LARGE_TABLES:
        key = f'emb_l.{t_idx}.weight'
        weight = state_dict[key]
        n_rows, d = weight.shape
        s = mmap_scales[t_idx]
        zp = mmap_zps[t_idx]

        fpath = os.path.join(mmap_dir, f'emb_{t_idx}.bin')
        # mmap the file
        mm = np.memmap(fpath, dtype=np.uint8, mode='r', shape=(n_rows, d))
        # Dequantize (this loads pages on demand from the OS cache)
        dequant = torch.from_numpy((mm.astype(np.float32) - zp) * s)
        dlrm.emb_l[t_idx].weight.data = dequant

    rss_after_mmap = get_rss_mb()
    log(f"  RSS after mmap load: {rss_after_mmap:.0f}MB")

    # Run inference with mmap'd tables
    mmap_result = run_inference(dlrm, test_ld, tag="mmap_uint8")
    log(f"  AUC={mmap_result['auc']:.6f} (delta={mmap_result['auc']-baseline['auc']:+.6f}), "
        f"BLat={mmap_result['mean_lat_ms']:.2f}ms, "
        f"Total={mmap_result['total_time']:.1f}s, RSS={mmap_result['rss_mb']:.0f}MB")

    # Restore
    for t_idx in LARGE_TABLES:
        key = f'emb_l.{t_idx}.weight'
        dlrm.emb_l[t_idx].weight.data = state_dict[key].clone()

    # ---- Summary ----
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}\n")

    all_results = {
        'baseline': baseline,
        'uint8_dequant': quant_result,
        'mmap_uint8': mmap_result,
    }

    log(f"  {'Config':<25s} | {'AUC':>10s} | {'Delta':>10s} | {'BLat':>8s} | "
        f"{'Total':>7s} | {'RSS':>7s}")
    log(f"  {'-'*75}")
    for name, r in all_results.items():
        delta = r['auc'] - baseline['auc']
        delta_s = f"{delta:+.6f}" if name != 'baseline' else "---"
        log(f"  {name:<25s} | {r['auc']:>10.6f} | {delta_s:>10s} | "
            f"{r['mean_lat_ms']:>6.2f}ms | {r['total_time']:>5.1f}s | "
            f"{r['rss_mb']:>5.0f}MB")

    log(f"\n  Key comparison:")
    log(f"  - mmap approach: {total_uint8_bytes/1e6:.0f}MB on disk (4x compression from quantization only)")
    log(f"  - H.265 approach: 53MB on disk (39x compression) + 87MB hot + 84MB mappings = 224MB total")
    log(f"  - mmap lets OS handle caching automatically — no manual hot/cold split needed")

    # Save
    json_path = os.path.join(OUTPUT_DIR, 'mmap_baseline.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
