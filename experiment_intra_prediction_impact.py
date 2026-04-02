#!/usr/bin/env python3
"""
Measure exact impact of H.265 intra prediction on embedding data compression.

Encodes the 8 large Kaggle tables (cold, quantized, freq-sorted, tiled into 1080p
grayscale frames) under several x265 configurations that progressively strip away
intra prediction tools, plus a Zstd-3 baseline for raw entropy.

Configurations tested:
  A) Full H.265 CRF=30 medium no-deblock:no-sao  (our production config)
  B) DC-only intra prediction  (disable angular/planar modes)
  C) Minimal intra: DC-only + no-strong-intra-smoothing + no-rect + no-amp + no-early-skip
  D) QP=0 (near-lossless) + DC-only  — to see intra impact without quantization
  E) Zstd-3 on tiled frame data  (entropy-only baseline, no spatial prediction)
  F) Zstd-3 on flat (un-tiled) uint8 data  (pure entropy baseline)
"""

import os, sys, time, json, subprocess, tempfile, shutil, gc
import numpy as np

os.chdir('/home/cc/expr/dlrm_minrui')
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')

import torch
import compressed_emb as _C

# ============================================================
# Constants
# ============================================================
MODEL_PATH = "./models/dlrm_kaggle_correct.pt"
REORDER_DIR = "results/reorder"
EMB_DIM = 16
WIDTH = 1920
HEIGHT = 1080
TILE_H = 4
TILE_W = 4
RPF = (WIDTH * HEIGHT) // EMB_DIM  # 129600 rows per frame

LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]

# ============================================================
# Encoding configurations
# ============================================================
# Each config: (label, dict of encode params)
# "mode" is either "h265" or "zstd_tiled" or "zstd_flat"
CONFIGS = [
    ("A: Full H.265 (CRF=30 medium no-deblock:no-sao)", {
        "mode": "h265",
        "preset": "medium",
        "x265_params": "keyint=1:min-keyint=1:crf=30:log-level=error:no-deblock=1:no-sao=1",
    }),
    ("B: DC-only intra (CRF=30 medium)", {
        "mode": "h265",
        "preset": "medium",
        # Force only DC intra prediction: constrained-intra + no strong smoothing
        # x265 doesn't have a single "dc-only" flag, but we can force it via:
        # - rdpenalty=2 heavily penalizes non-DC modes (2 = max penalty)
        # There's no direct "intra-pred-mode=dc" in x265; closest is rdpenalty
        "x265_params": "keyint=1:min-keyint=1:crf=30:log-level=error:no-deblock=1:no-sao=1"
                       ":no-strong-intra-smoothing=1:constrained-intra=1:rdpenalty=2",
    }),
    ("C: Minimal intra (CRF=30 + rdpenalty + no-rect/amp/early-skip)", {
        "mode": "h265",
        "preset": "medium",
        # Strip as much intra-prediction sophistication as possible:
        # rdpenalty=2: heavy penalty for complex intra modes
        # no-strong-intra-smoothing: disable smoothing filter on large blocks
        # constrained-intra: forbid using reconstructed inter samples for intra
        # no-rect: disable rectangular motion partitions (CU split)
        # no-amp: disable asymmetric motion partitions
        # no-early-skip: disable early CU skip decisions
        # ctu=16: force small CTU to limit prediction reach
        "x265_params": "keyint=1:min-keyint=1:crf=30:log-level=error:no-deblock=1:no-sao=1"
                       ":no-strong-intra-smoothing=1:constrained-intra=1:rdpenalty=2"
                       ":no-rect=1:no-amp=1:no-early-skip=1:ctu=16",
    }),
    ("D: CRF=30 ultrafast (baseline comparison)", {
        "mode": "h265",
        "preset": "ultrafast",
        "x265_params": "keyint=1:min-keyint=1:crf=30:log-level=error",
    }),
    ("E: Lossless H.265 (CRF=0)", {
        "mode": "h265",
        "preset": "ultrafast",
        "x265_params": "keyint=1:min-keyint=1:lossless=1:log-level=error",
    }),
    ("F: Zstd-3 on TILED frames", {
        "mode": "zstd_tiled",
        "level": 3,
    }),
    ("G: Zstd-3 on FLAT uint8 data", {
        "mode": "zstd_flat",
        "level": 3,
    }),
    ("H: Zstd-19 on TILED frames", {
        "mode": "zstd_tiled",
        "level": 19,
    }),
]


def quantize_table(w):
    """Global quantization: single scale/zero-point for entire table."""
    mn = w.min().item()
    mx = w.max().item()
    s = (mx - mn) / 255.0
    if s == 0:
        s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def encode_h265_frames(tiled_frames, tmpdir, preset, x265_params):
    """Encode a list of tiled frame tensors via ffmpeg. Returns total compressed bytes."""
    total = 0
    for i, frame_t in enumerate(tiled_frames):
        frame_2d = frame_t.numpy()
        frame_path = os.path.join(tmpdir, f'frame_{i:05d}.mkv')
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', f'{WIDTH}x{HEIGHT}',
            '-r', '1',
            '-i', 'pipe:0',
            '-c:v', 'libx265',
            '-preset', preset,
            '-pix_fmt', 'gray',
            '-x265-params', x265_params,
            '-f', 'matroska',
            frame_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.stdin.write(frame_2d.tobytes())
        proc.stdin.close()
        proc.wait()
        if os.path.exists(frame_path):
            total += os.path.getsize(frame_path)
    return total


def encode_zstd_tiled_frames(tiled_frames, level):
    """Zstd-compress each tiled frame (as flat bytes). Returns total compressed bytes."""
    total = 0
    for frame_t in tiled_frames:
        flat = frame_t.contiguous().view(-1)
        compressed = _C.zstd_compress_frame(flat, level)
        total += compressed.numel()
    return total


def encode_zstd_flat(q_uint8, rpf, level):
    """Zstd-compress flat uint8 data per-frame (no tiling). Returns total compressed bytes."""
    num_rows = q_uint8.shape[0]
    num_frames = max(1, (num_rows + rpf - 1) // rpf)
    padded_rows = num_frames * rpf
    if q_uint8.shape[0] < padded_rows:
        padded = torch.zeros(padded_rows, EMB_DIM, dtype=torch.uint8)
        padded[:q_uint8.shape[0]] = q_uint8
        q_uint8 = padded

    total = 0
    for i in range(num_frames):
        start = i * rpf
        end = start + rpf
        flat = q_uint8[start:end].contiguous().view(-1)
        compressed = _C.zstd_compress_frame(flat, level)
        total += compressed.numel()
    return total


def main():
    print("=" * 80)
    print("EXPERIMENT: Impact of H.265 Intra Prediction on Embedding Compression")
    print("=" * 80)
    print(f"Resolution: {WIDTH}x{HEIGHT}, EMB_DIM={EMB_DIM}, RPF={RPF}")
    print(f"Tables: {LARGE_TABLES}")
    print()

    # ---- Load model weights ----
    print("Loading model weights...")
    t0 = time.time()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    print(f"  Loaded in {time.time()-t0:.1f}s, {len(emb_keys)} tables")

    # ---- Load cold row counts and orderings ----
    cold_num_rows = {}
    cold_orders = {}
    for t in LARGE_TABLES:
        with open(os.path.join(REORDER_DIR, f'num_cold_{t}.txt')) as f:
            cold_num_rows[t] = int(f.read().strip())
        cold_orders[t] = np.load(os.path.join(REORDER_DIR, f'cold_order_{t}.npy'))

    print("\nCold rows per table:")
    for t in LARGE_TABLES:
        n = cold_num_rows[t]
        nf = max(1, (n + RPF - 1) // RPF)
        fp32_mb = n * EMB_DIM * 4 / 1024 / 1024
        uint8_mb = n * EMB_DIM / 1024 / 1024
        print(f"  Table {t:2d}: {n:>10,} rows, {nf:>3d} frames, "
              f"fp32={fp32_mb:>7.1f} MB, uint8={uint8_mb:>7.1f} MB")

    total_fp32_bytes = sum(cold_num_rows[t] * EMB_DIM * 4 for t in LARGE_TABLES)
    total_uint8_bytes = sum(cold_num_rows[t] * EMB_DIM for t in LARGE_TABLES)
    print(f"\n  TOTAL: fp32={total_fp32_bytes/1024/1024:.1f} MB, "
          f"uint8={total_uint8_bytes/1024/1024:.1f} MB")

    # ---- Quantize and tile all tables (once) ----
    print("\nQuantizing and tiling all tables...")
    table_q = {}       # table -> uint8 tensor (n_cold, 16)
    table_tiled = {}   # table -> list of (H, W) uint8 tensors
    table_nframes = {} # table -> int

    for t in LARGE_TABLES:
        n_cold = cold_num_rows[t]
        cold_order = cold_orders[t]

        # Load reordered cold weights and quantize
        reordered_w = state_dict[emb_keys[t]][cold_order[:n_cold]]
        q, s, zp = quantize_table(reordered_w)
        table_q[t] = q

        # Tile into frames
        num_frames = max(1, (n_cold + RPF - 1) // RPF)
        table_nframes[t] = num_frames
        padded_rows = num_frames * RPF
        if q.shape[0] < padded_rows:
            padded = torch.zeros(padded_rows, EMB_DIM, dtype=torch.uint8)
            padded[:q.shape[0]] = q
            q_padded = padded
        else:
            q_padded = q

        tiled_frames = _C.fused_quantize_tile_multiframe(q_padded, WIDTH, HEIGHT)
        table_tiled[t] = tiled_frames
        print(f"  Table {t}: {num_frames} frames tiled ({q_padded.shape[0]:,} rows)")

        del reordered_w
        gc.collect()

    # Free model weights — not needed anymore
    del state_dict
    gc.collect()

    # ---- Run each configuration ----
    results = {}

    for cfg_label, cfg_params in CONFIGS:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg_label}")
        print(f"{'='*70}")

        mode = cfg_params["mode"]
        per_table = {}
        total_compressed = 0

        for t in LARGE_TABLES:
            n_cold = cold_num_rows[t]
            t0 = time.time()

            if mode == "h265":
                # Create temp dir for this table's frames
                tmpdir = tempfile.mkdtemp(prefix=f"intra_t{t}_")
                compressed_bytes = encode_h265_frames(
                    table_tiled[t], tmpdir,
                    cfg_params["preset"],
                    cfg_params["x265_params"]
                )
                shutil.rmtree(tmpdir, ignore_errors=True)

            elif mode == "zstd_tiled":
                compressed_bytes = encode_zstd_tiled_frames(
                    table_tiled[t], cfg_params["level"]
                )

            elif mode == "zstd_flat":
                compressed_bytes = encode_zstd_flat(
                    table_q[t], RPF, cfg_params["level"]
                )

            elapsed = time.time() - t0
            raw_uint8 = n_cold * EMB_DIM
            raw_fp32 = n_cold * EMB_DIM * 4

            per_table[t] = {
                "n_cold": n_cold,
                "compressed_bytes": compressed_bytes,
                "uint8_ratio": raw_uint8 / compressed_bytes if compressed_bytes > 0 else 0,
                "fp32_ratio": raw_fp32 / compressed_bytes if compressed_bytes > 0 else 0,
                "encode_time": elapsed,
            }
            total_compressed += compressed_bytes

            print(f"  Table {t:2d}: {compressed_bytes:>10,} bytes "
                  f"({raw_uint8/compressed_bytes:>7.1f}x uint8, "
                  f"{raw_fp32/compressed_bytes:>8.1f}x fp32) "
                  f"[{elapsed:.1f}s]")

        total_uint8_ratio = total_uint8_bytes / total_compressed if total_compressed > 0 else 0
        total_fp32_ratio = total_fp32_bytes / total_compressed if total_compressed > 0 else 0
        total_enc_time = sum(v["encode_time"] for v in per_table.values())

        results[cfg_label] = {
            "total_compressed": total_compressed,
            "total_uint8_ratio": total_uint8_ratio,
            "total_fp32_ratio": total_fp32_ratio,
            "total_encode_time": total_enc_time,
            "per_table": per_table,
        }

        print(f"\n  TOTAL: {total_compressed:,} bytes = {total_compressed/1024/1024:.2f} MB")
        print(f"  Ratios: {total_uint8_ratio:.1f}x vs uint8, {total_fp32_ratio:.1f}x vs fp32")
        print(f"  Encode time: {total_enc_time:.1f}s")

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n\n" + "=" * 100)
    print("SUMMARY: Impact of Intra Prediction on H.265 Embedding Compression")
    print("=" * 100)

    header = f"{'Configuration':<60s} {'Size MB':>8s} {'uint8x':>8s} {'fp32x':>8s} {'Time':>6s}"
    print(header)
    print("-" * len(header))

    baseline_size = None
    for cfg_label, _ in CONFIGS:
        r = results[cfg_label]
        size_mb = r["total_compressed"] / 1024 / 1024
        if baseline_size is None:
            baseline_size = r["total_compressed"]
        overhead = r["total_compressed"] / baseline_size if baseline_size else 1.0
        print(f"{cfg_label:<60s} {size_mb:>8.2f} {r['total_uint8_ratio']:>8.1f} "
              f"{r['total_fp32_ratio']:>8.1f} {r['total_encode_time']:>5.1f}s")

    # ---- Per-table breakdown for selected configs ----
    print("\n\n" + "=" * 100)
    print("PER-TABLE COMPARISON: Full H.265 vs DC-only vs Zstd-3")
    print("=" * 100)

    configs_to_compare = [c[0] for c in CONFIGS if any(
        tag in c[0] for tag in ["A:", "B:", "C:", "F:", "G:"])]

    header2 = f"{'Table':>6s} {'n_cold':>10s}"
    for c in configs_to_compare:
        tag = c.split(":")[0]
        header2 += f"  {tag+' bytes':>12s} {tag+' uint8x':>10s}"
    print(header2)
    print("-" * len(header2))

    for t in LARGE_TABLES:
        line = f"{t:>6d} {cold_num_rows[t]:>10,}"
        for c in configs_to_compare:
            pt = results[c]["per_table"][t]
            line += f"  {pt['compressed_bytes']:>12,} {pt['uint8_ratio']:>10.1f}"
        print(line)

    # ---- Relative size increase from disabling intra ----
    print("\n\n" + "=" * 100)
    print("RELATIVE SIZE vs FULL H.265 (config A)")
    print("=" * 100)

    a_key = CONFIGS[0][0]
    a_total = results[a_key]["total_compressed"]

    header3 = f"{'Configuration':<60s} {'Rel. Size':>10s} {'Delta %':>8s}"
    print(header3)
    print("-" * len(header3))

    for cfg_label, _ in CONFIGS:
        r = results[cfg_label]
        rel = r["total_compressed"] / a_total
        delta_pct = (r["total_compressed"] - a_total) / a_total * 100
        marker = ""
        if cfg_label == a_key:
            marker = " (baseline)"
        print(f"{cfg_label:<60s} {rel:>9.3f}x {delta_pct:>+7.1f}%{marker}")

    # ---- Key insights ----
    print("\n\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    a = results[CONFIGS[0][0]]
    b = results[CONFIGS[1][0]]
    c = results[CONFIGS[2][0]]
    f_tiled = results[CONFIGS[5][0]]
    g_flat = results[CONFIGS[6][0]]

    intra_impact = (b["total_compressed"] - a["total_compressed"]) / a["total_compressed"] * 100
    full_strip_impact = (c["total_compressed"] - a["total_compressed"]) / a["total_compressed"] * 100
    h265_vs_zstd = a["total_compressed"] / f_tiled["total_compressed"]
    tiling_impact = (f_tiled["total_compressed"] - g_flat["total_compressed"]) / g_flat["total_compressed"] * 100

    print(f"1. Disabling angular+planar intra (DC-only) increases size by: {intra_impact:+.1f}%")
    print(f"2. Fully stripping intra sophistication increases size by:     {full_strip_impact:+.1f}%")
    print(f"3. H.265 full vs Zstd-3 on tiled frames:                      {h265_vs_zstd:.3f}x")
    print(f"   (H.265 is {'smaller' if h265_vs_zstd < 1 else 'larger'} than Zstd)")
    print(f"4. Tiling impact on Zstd: tiled vs flat = {tiling_impact:+.1f}%")
    print(f"5. Total compression chain benefit of intra prediction modes:")
    print(f"   Full H.265 CRF=30: {a['total_fp32_ratio']:.1f}x vs fp32")
    print(f"   Zstd-3 flat:       {g_flat['total_fp32_ratio']:.1f}x vs fp32")
    print(f"   Ratio of ratios:   {a['total_fp32_ratio']/g_flat['total_fp32_ratio']:.2f}x")

    # Save results
    out_path = "results/intra_prediction_impact.json"
    # Convert per_table keys to strings for JSON
    json_results = {}
    for k, v in results.items():
        jv = dict(v)
        jv["per_table"] = {str(tk): tv for tk, tv in v["per_table"].items()}
        json_results[k] = jv
    with open(out_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
