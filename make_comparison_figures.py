#!/usr/bin/env python3
"""Generate presentation figures for pipeline breakdown and end-to-end comparison."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "results/figures"
import os
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 13,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Colors
C_EMB = '#3498db'
C_INTERACT = '#e67e22'
C_MLP = '#9b59b6'
C_SCAN = '#2ecc71'
C_DECODE = '#e74c3c'
C_OTHER = '#95a5a6'
C_TILE = '#1abc9c'
C_QUANT = '#f39c12'
C_ENCODE = '#c0392b'


# ============================================================
# Figure 1: Compression (encode) pipeline breakdown
# ============================================================
def fig_encode_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Per-frame encode breakdown (C++ fused)
    # From re-measured data: fused_gather_quantize_tile = 0.098ms, H.265 encode = 83ms
    steps = ['Gather\n(non-contig)', 'Quantize\n(fp32→uint8)', 'Tile\n(→frame)', 'H.265\nEncode']
    times_ms = [0.012, 0.040, 0.046, 83.0]
    colors = [C_EMB, C_QUANT, C_TILE, C_ENCODE]

    bars = ax1.bar(steps, times_ms, color=colors, edgecolor='white', width=0.6)
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Compression Pipeline per Frame\n(C++ fused, 1920×1080)')
    ax1.set_yscale('log')
    ax1.set_ylim(0.005, 200)

    total = sum(times_ms)
    for bar, t in zip(bars, times_ms):
        pct = t / total * 100
        ax1.text(bar.get_x() + bar.get_width()/2, t * 1.4,
                f'{t:.3f}ms\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')

    # Right: Per-table encode time (real data, all 8 tables)
    tables = ['T2\n9.6M', 'T11\n8.0M', 'T20\n6.7M', 'T15\n5.3M', 'T3\n2.2M']
    encode_s = [1.886, 1.269, 1.126, 0.847, 0.454]
    ratios = [27.9, 12.0, 5.6, 10.5, 6.0]

    x = np.arange(len(tables))
    bars2 = ax2.bar(x, encode_s, color=C_ENCODE, edgecolor='white', width=0.6, alpha=0.8)
    ax2.set_ylabel('Encode Time (seconds)', color=C_ENCODE)
    ax2.set_title('C++ Parallel Encode per Table\n(tile + H.265, all frames)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tables)

    # Add compression ratio on secondary axis
    ax2b = ax2.twinx()
    ax2b.plot(x, ratios, 'D-', color=C_EMB, markersize=8, linewidth=2, label='Compression ratio')
    ax2b.set_ylabel('Compression Ratio (×)', color=C_EMB)
    ax2b.legend(loc='upper right')

    for bar, t in zip(bars2, encode_s):
        ax2.text(bar.get_x() + bar.get_width()/2, t + 0.05,
                f'{t:.1f}s', ha='center', fontsize=10, fontweight='bold', color=C_ENCODE)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'encode_breakdown.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Figure 2: Decompression (decode) pipeline breakdown
# ============================================================
def fig_decode_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Per-frame decode breakdown
    steps = ['H.265\nDecode', 'Untile\n(C++)', 'Gather+\nDequant', 'Scan\n(C++ fused)']
    times_ms = [3.5, 0.029, 0.014, 0.05]
    colors = [C_DECODE, C_TILE, C_EMB, C_SCAN]

    bars = ax1.bar(steps, times_ms, color=colors, edgecolor='white', width=0.6)
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Decompression Pipeline per Frame\n(C++ fused, 1920×1080)')
    ax1.set_yscale('log')
    ax1.set_ylim(0.005, 20)

    total = sum(times_ms)
    for bar, t in zip(bars, times_ms):
        pct = t / total * 100
        ax1.text(bar.get_x() + bar.get_width()/2, t * 1.5,
                f'{t:.3f}ms\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')

    # Right: Parallel batch decode scaling
    n_frames = [1, 2, 3, 5, 8, 10]
    serial_ms = [19, 37, 53, 92, 143, 189]
    parallel_ms = [19, 19, 19, 19, 40, 48]

    ax2.plot(n_frames, serial_ms, 'o-', color=C_DECODE, linewidth=2, markersize=8, label='Serial decode')
    ax2.plot(n_frames, parallel_ms, 's-', color=C_SCAN, linewidth=2, markersize=8, label='Parallel batch decode')
    ax2.set_xlabel('Number of frames to decode')
    ax2.set_ylabel('Total decode time (ms)')
    ax2.set_title('Parallel Batch Decode Speedup')
    ax2.legend()

    # Annotate speedups
    for i, n in enumerate(n_frames):
        if n > 1:
            speedup = serial_ms[i] / parallel_ms[i]
            ax2.annotate(f'{speedup:.1f}x', xy=(n, parallel_ms[i]),
                        xytext=(5, -15), textcoords='offset points',
                        fontweight='bold', fontsize=10, color=C_SCAN)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'decode_breakdown.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Figure 3: End-to-end latency comparison (stacked bar)
# ============================================================
def fig_e2e_latency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Per-batch latency breakdown (stacked bar)
    configs = ['Baseline\n(fp32)', 'Codec\nfull_cpp', 'Codec\ncache=16', 'Lookahead\ng=500']
    emb_ms =      [1.46, 0.23, 0.20, 0.19]
    interact_ms = [1.23, 0.57, 0.58, 0.61]
    mlp_ms =      [1.55, 1.62, 1.53, 1.51]
    scan_dec_ms = [0.00, 0.00, 0.23, 0.25]
    other_ms =    [0.05, 0.06, 0.02, 0.00]  # residual

    x = np.arange(len(configs))
    w = 0.55

    bottom = np.zeros(len(configs))
    components = [
        (emb_ms, 'Embedding lookup', C_EMB),
        (interact_ms, 'Feature interaction', C_INTERACT),
        (mlp_ms, 'MLP forward', C_MLP),
        (scan_dec_ms, 'Scan + Decode', C_DECODE),
    ]

    for vals, label, color in components:
        ax1.bar(x, vals, w, bottom=bottom, label=label, color=color, edgecolor='white')
        bottom += np.array(vals)

    ax1.set_ylabel('Per-batch latency (ms)')
    ax1.set_title('Forward Pass Latency Breakdown')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend(loc='upper right', fontsize=10)

    # Add total labels
    totals = [sum(t) for t in zip(emb_ms, interact_ms, mlp_ms, scan_dec_ms, other_ms)]
    for i, total in enumerate(totals):
        ax1.text(i, total + 0.1, f'{total:.2f}ms', ha='center', fontweight='bold', fontsize=11)

    # Speedup annotations
    baseline_total = totals[0]
    for i in range(1, len(totals)):
        speedup = baseline_total / totals[i]
        ax1.text(i, totals[i] + 0.35, f'{speedup:.2f}x', ha='center', fontsize=10, color=C_SCAN)

    # Right: Total inference time (wall clock)
    configs_full = ['Baseline', 'full_cpp', 'cache=16', 'LA g=500']
    total_time_s = [33.5, 29.7, 29.9, 30.0]
    colors_bar = [C_OTHER, C_EMB, C_SCAN, C_DECODE]

    bars = ax2.bar(configs_full, total_time_s, color=colors_bar, edgecolor='white', width=0.55, alpha=0.85)
    ax2.set_ylabel('Total inference time (seconds)')
    ax2.set_title('Wall-Clock Time (1,599 batches × 2048)')
    ax2.set_ylim(0, 40)

    for bar, t in zip(bars, total_time_s):
        ax2.text(bar.get_x() + bar.get_width()/2, t + 0.5,
                f'{t:.1f}s', ha='center', fontweight='bold', fontsize=12)
        speedup = total_time_s[0] / t
        if speedup > 1.01:
            ax2.text(bar.get_x() + bar.get_width()/2, t - 2,
                    f'{speedup:.2f}x', ha='center', fontsize=11, color='white', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'e2e_latency.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Figure 4: Memory usage and compression ratio comparison
# ============================================================
def fig_memory_compression():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Memory usage stacked bar
    configs = ['Baseline\n(fp32)', 'Codec\nfull_cpp', 'Codec\ncache=16', 'Lookahead\ng=500']

    # Memory breakdown (MB)
    # Baseline: all embeddings in fp32
    # Codec: hot (fp32) + cold compressed (disk) + LRU cache (uint8) + mappings
    hot_mb =      [0, 87.4, 87.4, 87.4]
    cold_disk_mb = [0, 50.4, 50.4, 50.4]
    lru_cache_mb = [0, 43.5, 31.6, 0]  # full_cpp has all frames decoded, cache=16 has 16 frames
    mapping_mb =  [0, 84.3, 84.3, 84.3]
    baseline_mb = [2060.7, 0, 0, 0]

    x = np.arange(len(configs))
    w = 0.55

    ax1.bar(x, baseline_mb, w, label='Baseline embeddings (fp32)', color=C_DECODE, edgecolor='white')

    bottom = np.zeros(len(configs))
    stack = [
        (hot_mb, 'Hot embeddings (fp32)', C_EMB),
        (cold_disk_mb, 'Cold compressed (disk)', C_ENCODE),
        (lru_cache_mb, 'LRU cache (decoded)', C_SCAN),
        (mapping_mb, 'Index mappings', C_OTHER),
    ]
    for vals, label, color in stack:
        ax1.bar(x, vals, w, bottom=bottom, label=label, color=color, edgecolor='white')
        bottom += np.array(vals)

    totals = [2060.7, 265.6, 253.7, 222.1]
    for i, t in enumerate(totals):
        ax1.text(i, max(t, bottom[i]) + 30, f'{t:.0f} MB', ha='center', fontweight='bold', fontsize=11)

    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('Memory Usage Breakdown')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 2400)

    # Add compression ratio
    for i in range(1, len(totals)):
        ratio = totals[0] / totals[i]
        ax1.text(i, totals[i] + 100, f'{ratio:.1f}× smaller', ha='center',
                fontsize=10, color=C_SCAN, fontweight='bold')

    # Right: Performance vs Memory scatter
    configs_all = [
        ('Baseline', 4.29, 2060.7),
        ('full_cpp', 2.48, 265.6),
        ('cache=4', 2.57, 230.0),
        ('cache=8', 2.45, 237.9),
        ('cache=16', 2.36, 253.7),
        ('cache=32', 2.38, 265.6),
        ('LA g=10', 2.44, 222.1),
        ('LA g=50', 2.43, 222.1),
        ('LA g=100', 2.40, 222.1),
        ('LA g=500', 2.36, 222.1),
    ]

    for name, lat, mem in configs_all:
        color = C_DECODE if name == 'Baseline' else (C_SCAN if 'LA' in name else C_EMB)
        marker = 'o' if name == 'Baseline' else ('D' if 'LA' in name else 's')
        size = 120 if name == 'Baseline' else 80
        ax2.scatter(mem, lat, s=size, c=color, marker=marker, edgecolors='black', linewidth=0.5, zorder=5)
        # Label
        offset = (8, 5) if name != 'Baseline' else (-60, 8)
        ax2.annotate(name, (mem, lat), xytext=offset, textcoords='offset points', fontsize=8)

    ax2.set_xlabel('Memory Usage (MB)')
    ax2.set_ylabel('Mean Batch Latency (ms)')
    ax2.set_title('Latency vs Memory Trade-off')

    # Draw arrow showing improvement direction
    ax2.annotate('Better →\n↓', xy=(300, 2.2), fontsize=12, color='gray',
                ha='center', fontweight='bold')

    # Highlight the Pareto front
    ax2.axhline(y=4.29, color=C_DECODE, linestyle='--', alpha=0.3, label='Baseline latency')
    ax2.axvline(x=2060.7, color=C_DECODE, linestyle='--', alpha=0.3, label='Baseline memory')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'memory_compression.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Figure 5: Batch size sensitivity
# ============================================================
def fig_batchsize_sweep():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    batch_sizes = [256, 512, 1024, 2048, 4096]

    # Forward latency
    baseline_lat = [2.16, 2.45, 3.04, 4.29, 6.39]
    cache16_lat =  [1.09, 1.32, 1.75, 2.36, 4.26]
    la500_lat =    [1.03, 1.28, 1.71, 2.35, 3.40]

    ax1.plot(batch_sizes, baseline_lat, 'o-', color=C_DECODE, linewidth=2, markersize=8, label='Baseline (fp32)')
    ax1.plot(batch_sizes, cache16_lat, 's-', color=C_EMB, linewidth=2, markersize=8, label='Codec cache=16')
    ax1.plot(batch_sizes, la500_lat, 'D-', color=C_SCAN, linewidth=2, markersize=8, label='Lookahead g=500')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Mean Batch Latency (ms)')
    ax1.set_title('Forward Pass Latency vs Batch Size')
    ax1.legend()

    # Annotate speedups
    for i, bs in enumerate(batch_sizes):
        speedup = baseline_lat[i] / la500_lat[i]
        ax1.annotate(f'{speedup:.2f}x', xy=(bs, la500_lat[i]),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=C_SCAN)

    # Wall clock time
    baseline_wc = [57.3, 42.2, 35.4, 33.5, 31.1]
    cache16_wc =  [43.9, 35.6, 31.9, 29.9, 29.7]
    la500_wc =    [42.0, 34.5, 31.3, 30.0, 28.7]

    ax2.plot(batch_sizes, baseline_wc, 'o-', color=C_DECODE, linewidth=2, markersize=8, label='Baseline (fp32)')
    ax2.plot(batch_sizes, cache16_wc, 's-', color=C_EMB, linewidth=2, markersize=8, label='Codec cache=16')
    ax2.plot(batch_sizes, la500_wc, 'D-', color=C_SCAN, linewidth=2, markersize=8, label='Lookahead g=500')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Total Inference Time (seconds)')
    ax2.set_title('Wall-Clock Time vs Batch Size')
    ax2.legend()

    for i, bs in enumerate(batch_sizes):
        speedup = baseline_wc[i] / la500_wc[i]
        ax2.annotate(f'{speedup:.2f}x', xy=(bs, la500_wc[i]),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=C_SCAN)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'batchsize_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Figure 6: Cache hit rate vs decode overhead
# ============================================================
def fig_cache_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Cache hit rate and decode cost vs cache size
    cache_sizes = [4, 8, 16, 22, 32, 64]
    hit_rates = [46.3, 92.5, 99.8, 99.8, 99.8, 99.8]
    decode_ms = [29.02, 3.98, 0.12, 0.13, 0.12, 0.12]

    ax1.plot(cache_sizes, hit_rates, 'o-', color=C_EMB, linewidth=2, markersize=8, label='Cache hit rate (%)')
    ax1.set_xlabel('LRU Cache Size (frames)')
    ax1.set_ylabel('Cache Hit Rate (%)', color=C_EMB)
    ax1.set_ylim(0, 105)

    ax1b = ax1.twinx()
    ax1b.plot(cache_sizes, decode_ms, 's-', color=C_DECODE, linewidth=2, markersize=8, label='Decode overhead')
    ax1b.set_ylabel('Decode Overhead (ms/batch)', color=C_DECODE)
    ax1b.set_yscale('log')

    ax1.set_title('Cache Hit Rate & Decode Cost vs Cache Size')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # Right: Lookahead group size vs decode overhead
    group_sizes = [1, 10, 50, 100, 500]
    la_decode_ms = [52.65, 5.99, 1.27, 0.64, 0.18]
    la_frames = [13812, 1394, 289, 151, 48]

    ax2.plot(group_sizes, la_decode_ms, 'D-', color=C_DECODE, linewidth=2, markersize=8, label='Decode (ms/batch)')
    ax2.set_xlabel('Look-Ahead Group Size')
    ax2.set_ylabel('Decode Overhead (ms/batch)', color=C_DECODE)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2b = ax2.twinx()
    ax2b.plot(group_sizes, la_frames, 'o-', color=C_SCAN, linewidth=2, markersize=8, label='Frames decoded')
    ax2b.set_ylabel('Total Frames Decoded', color=C_SCAN)
    ax2b.set_yscale('log')

    ax2.set_title('Look-Ahead: Group Size vs Decode Cost')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'cache_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def fig_batchsize_sweep_no_la():
    """Batch size sweep with only Baseline vs Codec cache=16 (no lookahead)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    batch_sizes = [256, 512, 1024, 2048, 4096]

    # Forward latency
    baseline_lat = [2.16, 2.45, 3.04, 4.29, 6.39]
    cache16_lat =  [1.09, 1.32, 1.75, 2.36, 4.26]

    ax1.plot(batch_sizes, baseline_lat, 'o-', color=C_DECODE, linewidth=2, markersize=8, label='Baseline (fp32)')
    ax1.plot(batch_sizes, cache16_lat, 's-', color=C_EMB, linewidth=2, markersize=8, label='Codec + LRU cache')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Mean Batch Latency (ms)')
    ax1.set_title('Forward Pass Latency vs Batch Size')
    ax1.legend()

    for i, bs in enumerate(batch_sizes):
        speedup = baseline_lat[i] / cache16_lat[i]
        ax1.annotate(f'{speedup:.2f}x', xy=(bs, cache16_lat[i]),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=C_EMB)

    # Wall clock time
    baseline_wc = [57.3, 42.2, 35.4, 33.5, 31.1]
    cache16_wc =  [43.9, 35.6, 31.9, 29.9, 29.7]

    ax2.plot(batch_sizes, baseline_wc, 'o-', color=C_DECODE, linewidth=2, markersize=8, label='Baseline (fp32)')
    ax2.plot(batch_sizes, cache16_wc, 's-', color=C_EMB, linewidth=2, markersize=8, label='Codec + LRU cache')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Total Inference Time (seconds)')
    ax2.set_title('Wall-Clock Time vs Batch Size')
    ax2.legend()

    for i, bs in enumerate(batch_sizes):
        speedup = baseline_wc[i] / cache16_wc[i]
        ax2.annotate(f'{speedup:.2f}x', xy=(bs, cache16_wc[i]),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=C_EMB)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'batchsize_sweep_no_la.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    fig_encode_breakdown()
    fig_decode_breakdown()
    fig_e2e_latency()
    fig_memory_compression()
    fig_batchsize_sweep()
    fig_cache_analysis()
    fig_batchsize_sweep_no_la()
    print(f"\nAll figures saved to {OUTDIR}/")
