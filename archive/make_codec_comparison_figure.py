#!/usr/bin/env python3
"""Generate the critical H.265 vs general-purpose compressor figure."""

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


def fig_codec_comparison():
    """Main figure: Compression ratio vs Decode speed for all compressors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Data from benchmark
    compressors = {
        'LZ4':      {'ratio': 3.3, 'decode_ms': 1.422, 'size_mb': 155.3, 'color': '#2ecc71', 'marker': 's'},
        'Snappy':   {'ratio': 3.9, 'decode_ms': 2.321, 'size_mb': 132.8, 'color': '#27ae60', 'marker': '^'},
        'Zstd-3':   {'ratio': 6.8, 'decode_ms': 2.370, 'size_mb': 76.5,  'color': '#3498db', 'marker': 'D'},
        'Zstd-9':   {'ratio': 7.6, 'decode_ms': 1.793, 'size_mb': 68.3,  'color': '#2980b9', 'marker': 'D'},
        'Zstd-19':  {'ratio': 9.4, 'decode_ms': 1.132, 'size_mb': 55.0,  'color': '#1a5276', 'marker': 'D'},
        'H.265':    {'ratio': 9.8, 'decode_ms': 4.972, 'size_mb': 52.8,  'color': '#e74c3c', 'marker': 'o'},
    }

    # Left: Scatter plot - Compression ratio vs Decode speed
    for name, d in compressors.items():
        ax1.scatter(d['decode_ms'], d['ratio'], s=200, c=d['color'],
                   marker=d['marker'], edgecolors='black', linewidth=0.8, zorder=5)
        # Label
        offset = (10, 5)
        if name == 'H.265':
            offset = (10, -15)
        elif name == 'LZ4':
            offset = (10, -10)
        elif name == 'Zstd-19':
            offset = (-80, 5)
        ax1.annotate(name, (d['decode_ms'], d['ratio']),
                    xytext=offset, textcoords='offset points',
                    fontsize=11, fontweight='bold')

    # Draw Pareto frontier
    ax1.annotate('Better\n(faster + smaller)', xy=(0.5, 10.5),
                fontsize=11, color='gray', ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'),
                xytext=(2.5, 11.5))

    ax1.set_xlabel('Decode Time per Frame (ms)', fontsize=13)
    ax1.set_ylabel('Compression Ratio (vs uint8)', fontsize=13)
    ax1.set_title('Compression Ratio vs Decode Speed\n(129,600 embedding rows per frame)')
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 12)

    # Highlight: H.265 is in the worst quadrant
    ax1.axhline(y=9.8, color='#e74c3c', linestyle='--', alpha=0.2)
    ax1.axvline(x=4.972, color='#e74c3c', linestyle='--', alpha=0.2)
    ax1.fill_between([0, 4.972], [9.8, 9.8], [12, 12], alpha=0.05, color='#2ecc71',
                     label='Better than H.265 on both axes')
    ax1.legend(loc='lower right', fontsize=10)

    # Right: Bar chart - decode time comparison with speedup labels
    names = ['H.265', 'Zstd-3', 'Zstd-9', 'Zstd-19', 'LZ4', 'Snappy']
    times = [compressors[n]['decode_ms'] for n in names]
    ratios = [compressors[n]['ratio'] for n in names]
    colors = [compressors[n]['color'] for n in names]

    x = np.arange(len(names))
    bars = ax2.bar(x, times, color=colors, edgecolor='white', width=0.6)

    # Add decode time labels
    for i, (bar, t, r) in enumerate(zip(bars, times, ratios)):
        ax2.text(bar.get_x() + bar.get_width()/2, t + 0.1,
                f'{t:.2f}ms\n({r:.1f}x)',
                ha='center', fontsize=10, fontweight='bold')
        if i > 0:
            speedup = times[0] / t
            ax2.text(bar.get_x() + bar.get_width()/2, t/2,
                    f'{speedup:.1f}x\nfaster',
                    ha='center', fontsize=9, color='white', fontweight='bold')

    ax2.set_ylabel('Decode Time per Frame (ms)')
    ax2.set_title('Decode Speed: H.265 vs General-Purpose Compressors')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)
    ax2.set_ylim(0, 6.5)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'codec_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def fig_codec_tradeoff():
    """Detailed tradeoff analysis: what happens when we switch to Zstd."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Memory breakdown with different compressors
    configs = ['Baseline\n(fp32)', 'H.265\nlossless', 'Zstd-3', 'Zstd-19']

    hot_mb =    [0, 87.4, 87.4, 87.4]
    cold_mb =   [0, 52.8, 76.5, 55.0]
    cache_mb =  [0, 31.6, 31.6, 31.6]  # cache=16 decoded frames
    mapping_mb =[0, 84.3, 84.3, 84.3]
    baseline_mb=[2060.7, 0, 0, 0]

    x = np.arange(len(configs))
    w = 0.55

    ax1.bar(x, baseline_mb, w, label='Baseline (fp32)', color='#e74c3c', edgecolor='white')

    bottom = np.zeros(len(configs))
    stack = [
        (hot_mb, 'Hot embeddings', '#3498db'),
        (cold_mb, 'Cold compressed', '#c0392b'),
        (cache_mb, 'LRU cache', '#2ecc71'),
        (mapping_mb, 'Index mappings', '#95a5a6'),
    ]
    for vals, label, color in stack:
        ax1.bar(x, vals, w, bottom=bottom, label=label, color=color, edgecolor='white')
        bottom += np.array(vals)

    totals = [2060.7, 256.1, 279.8, 258.3]
    for i, t in enumerate(totals):
        ax1.text(i, max(t, bottom[i]) + 30, f'{t:.0f} MB', ha='center', fontweight='bold', fontsize=11)
    for i in range(1, len(totals)):
        ratio = totals[0] / totals[i]
        ax1.text(i, totals[i] + 100, f'{ratio:.1f}x smaller', ha='center',
                fontsize=10, color='#2ecc71', fontweight='bold')

    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('Memory: H.265 vs Zstd')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 2400)

    # Right: Cache miss penalty comparison
    cache_sizes = [4, 8, 16, 32]
    # miss rates: cache=4: 53.7%, cache=8: 7.5%, cache=16: 0.2%, cache=32: 0.2%
    miss_rates = [0.537, 0.075, 0.002, 0.002]
    avg_frames = 7

    h265_overhead = [m * avg_frames * 4.972 for m in miss_rates]
    zstd3_overhead = [m * avg_frames * 2.370 for m in miss_rates]
    zstd19_overhead = [m * avg_frames * 1.132 for m in miss_rates]
    lz4_overhead = [m * avg_frames * 1.422 for m in miss_rates]

    x2 = np.arange(len(cache_sizes))
    w2 = 0.2
    ax2.bar(x2 - 1.5*w2, h265_overhead, w2, label='H.265', color='#e74c3c', edgecolor='white')
    ax2.bar(x2 - 0.5*w2, zstd3_overhead, w2, label='Zstd-3', color='#3498db', edgecolor='white')
    ax2.bar(x2 + 0.5*w2, zstd19_overhead, w2, label='Zstd-19', color='#1a5276', edgecolor='white')
    ax2.bar(x2 + 1.5*w2, lz4_overhead, w2, label='LZ4', color='#2ecc71', edgecolor='white')

    ax2.set_xlabel('LRU Cache Size (frames)')
    ax2.set_ylabel('Decode Overhead per Batch (ms)')
    ax2.set_title('Cache Miss Penalty by Compressor')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cache_sizes)
    ax2.legend()
    ax2.set_yscale('log')
    ax2.set_ylim(0.01, 30)

    # Annotate speedups for cache=4
    for i, (h, z3, z19, l) in enumerate(zip(h265_overhead, zstd3_overhead, zstd19_overhead, lz4_overhead)):
        if i == 0:  # cache=4
            ax2.text(x2[i] - 1.5*w2, h + 1, f'{h:.1f}ms', ha='center', fontsize=8, rotation=90)
            ax2.text(x2[i] - 0.5*w2, z3 + 0.5, f'{z3:.1f}ms', ha='center', fontsize=8, rotation=90)
            ax2.text(x2[i] + 0.5*w2, z19 + 0.3, f'{z19:.1f}ms', ha='center', fontsize=8, rotation=90)
            ax2.text(x2[i] + 1.5*w2, l + 0.3, f'{l:.1f}ms', ha='center', fontsize=8, rotation=90)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'codec_tradeoff.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def fig_key_finding():
    """Single panel: the key finding for slides."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pareto plot
    compressors = [
        ('LZ4',    3.3, 1.422, '#2ecc71', 's', 150),
        ('Snappy', 3.9, 2.321, '#27ae60', '^', 150),
        ('Zstd-3', 6.8, 2.370, '#3498db', 'D', 180),
        ('Zstd-9', 7.6, 1.793, '#2980b9', 'D', 180),
        ('Zstd-19',9.4, 1.132, '#1a5276', 'D', 220),
        ('H.265',  9.8, 4.972, '#e74c3c', 'o', 250),
    ]

    for name, ratio, decode, color, marker, size in compressors:
        ax.scatter(decode, ratio, s=size, c=color, marker=marker,
                  edgecolors='black', linewidth=1, zorder=5)
        offset = (12, 0)
        if name == 'H.265':
            offset = (12, -8)
        elif name == 'LZ4':
            offset = (12, -8)
        elif name == 'Zstd-19':
            offset = (-85, -5)
        elif name == 'Zstd-9':
            offset = (-80, 5)
        ax.annotate(name, (decode, ratio),
                   xytext=offset, textcoords='offset points',
                   fontsize=13, fontweight='bold', color=color)

    # Draw arrow from H.265 to Zstd-19
    ax.annotate('', xy=(1.132, 9.4), xytext=(4.972, 9.8),
               arrowprops=dict(arrowstyle='->', color='black', lw=2, ls='--'))
    ax.text(3.0, 10.2, '4.4x faster decode\nsimilar compression',
           ha='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Decode Time per Frame (ms)', fontsize=14)
    ax.set_ylabel('Compression Ratio (vs quantized uint8)', fontsize=14)
    ax.set_title('H.265 vs General-Purpose Compressors\nfor Embedding Table Compression', fontsize=15)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 12)

    # Pareto frontier annotation
    ax.fill_between([0, 4.972], [9.8, 9.8], [12, 12], alpha=0.08, color='#2ecc71')
    ax.text(2.0, 11.3, 'Strictly dominates H.265', fontsize=11,
           color='#27ae60', fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='#eafaf1', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'codec_key_finding.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    fig_codec_comparison()
    fig_codec_tradeoff()
    fig_key_finding()
    print(f"\nAll codec comparison figures saved to {OUTDIR}/")
