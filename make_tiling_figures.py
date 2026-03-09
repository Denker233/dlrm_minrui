#!/usr/bin/env python3
"""Generate presentation figures for tiling performance and pipeline breakdown."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "results/figures"
import os
os.makedirs(OUTDIR, exist_ok=True)

# Color scheme
C_PYTHON = '#e74c3c'   # red
C_CPP = '#2ecc71'      # green
C_FUSED = '#3498db'    # blue
C_DECODE = '#e67e22'   # orange
C_FWD = '#9b59b6'      # purple
C_OTHER = '#95a5a6'    # gray

plt.rcParams.update({
    'font.size': 14,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# ============================================================
# Figure 1: Python vs C++ tiling operations (bar chart)
# ============================================================
def fig1_tiling_speedup():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Absolute times (re-measured 2026-03-05)
    ops = ['Tile only\n(uint8→frame)', 'Untile only\n(frame→uint8)', 'Full Pipeline\n(gather+quant+tile)']
    python_ms = [2.70, 2.92, 11.39]
    cpp_ms = [0.05, 0.03, 0.10]

    x = np.arange(len(ops))
    w = 0.35
    bars_py = ax1.bar(x - w/2, python_ms, w, label='Python (NumPy)', color=C_PYTHON, edgecolor='white')
    bars_cpp = ax1.bar(x + w/2, cpp_ms, w, label='C++ (fused, parallel)', color=C_CPP, edgecolor='white')

    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Tiling Operation Latency')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ops)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_ylim(0.01, 20)

    # Add speedup labels
    for i in range(len(ops)):
        speedup = python_ms[i] / cpp_ms[i]
        ax1.annotate(f'{speedup:.0f}x', xy=(x[i] + w/2, cpp_ms[i]),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontweight='bold', fontsize=12, color=C_CPP)

    # Right: Selective gather (varying K)
    K_vals = [10, 100, 1000, 10000]
    py_gather = [2.84, 2.86, 2.87, 3.11]  # Python always untiles full frame
    cpp_gather = [0.003, 0.004, 0.014, 0.020]

    ax2.plot(K_vals, py_gather, 'o-', color=C_PYTHON, linewidth=2, markersize=8,
             label='Python (full untile + index)')
    ax2.plot(K_vals, cpp_gather, 's-', color=C_CPP, linewidth=2, markersize=8,
             label='C++ (selective gather)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('K (rows to extract)')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Extract K Rows from Tiled Frame')
    ax2.legend(fontsize=11)

    # Annotate speedups
    for i, k in enumerate(K_vals):
        speedup = py_gather[i] / cpp_gather[i]
        ax2.annotate(f'{speedup:.0f}x', xy=(k, cpp_gather[i]),
                    xytext=(10, -5), textcoords='offset points',
                    fontweight='bold', fontsize=11, color=C_CPP)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'tiling_speedup.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Figure 2: Pipeline breakdown - tiling is NOT the bottleneck
# ============================================================
def fig2_pipeline_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Per-cache-miss breakdown (single frame decode path)
    components = ['H.265\nDecode', 'Untile', 'Gather+\nDequant', 'Cache\nInsert']
    times_ms = [3.5, 0.029, 0.014, 0.005]  # representative values
    colors = [C_DECODE, C_CPP, C_FUSED, C_OTHER]

    bars = ax1.bar(components, times_ms, color=colors, edgecolor='white', width=0.6)
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Cache Miss Pipeline Breakdown')
    ax1.set_yscale('log')
    ax1.set_ylim(0.001, 10)

    # Add percentage labels
    total = sum(times_ms)
    for bar, t in zip(bars, times_ms):
        pct = t / total * 100
        ax1.text(bar.get_x() + bar.get_width()/2, t * 1.3,
                f'{t:.3f}ms\n({pct:.1f}%)', ha='center', fontsize=11, fontweight='bold')

    # Right: Where does batch time go? (forward pass breakdown)
    labels = ['Data\nLoading', 'Embedding\nLookup', 'MLP\nForward', 'Scan +\nDecode', 'Tiling\n(amortized)']
    # From real benchmark: bs=2048
    times = [52.0, 0.25, 1.9, 0.15, 0.003]
    colors2 = [C_OTHER, C_FUSED, C_FWD, C_DECODE, C_CPP]

    bars2 = ax1_right = ax2.bar(labels, times, color=colors2, edgecolor='white', width=0.6)
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Per-Batch Time Breakdown (bs=2048)')
    ax2.set_yscale('log')
    ax2.set_ylim(0.001, 200)

    total2 = sum(times)
    for bar, t in zip(bars2, times):
        pct = t / total2 * 100
        if pct >= 0.1:
            ax2.text(bar.get_x() + bar.get_width()/2, t * 1.3,
                    f'{t:.2f}ms\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'pipeline_breakdown.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Figure 3: WHY C++ is faster (conceptual diagram as bar chart)
# ============================================================
def fig3_why_cpp_faster():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.2, 1]})

    # Left: Side-by-side breakdown
    steps = ['Gather\n(non-contig)', 'Quantize\n(fp32→uint8)', 'Tile\n(→frame)']
    python_ms = [2.54, 5.45, 2.70]   # + 0.70 interpreter overhead
    cpp_ms =    [0.012, 0.040, 0.046]

    x = np.arange(len(steps))
    w = 0.35
    bars_py = ax1.bar(x - w/2, python_ms, w, label='Python (NumPy)', color=C_PYTHON, edgecolor='white')
    bars_cpp = ax1.bar(x + w/2, cpp_ms, w, label='C++ (fused)', color=C_CPP, edgecolor='white')

    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Pipeline Step Breakdown per Frame\n(1920×1080 frame = 129,600 embedding rows)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(steps)
    ax1.legend(fontsize=12)
    ax1.set_yscale('log')
    ax1.set_ylim(0.005, 20)

    # Add speedup labels
    for i in range(len(steps)):
        speedup = python_ms[i] / cpp_ms[i]
        ax1.annotate(f'{speedup:.0f}x', xy=(x[i] + w/2, cpp_ms[i]),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontweight='bold', fontsize=13, color=C_CPP)
        # Python bar label
        ax1.annotate(f'{python_ms[i]:.2f}ms', xy=(x[i] - w/2, python_ms[i]),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=10, color=C_PYTHON)

    # Add interpreter overhead note
    ax1.annotate('+0.70ms interpreter overhead',
                xy=(0, 2.54), xytext=(-0.3, 0.5),
                fontsize=9, color='gray', style='italic')

    # Right: Stacked totals with breakdown
    interp_ms = 0.70
    py_total = sum(python_ms) + interp_ms
    cpp_total = sum(cpp_ms)

    step_names = ['Gather\n(non-contig)', 'Quantize\n(fp32→uint8)', 'Tile\n(transpose)']
    colors_steps = ['#e67e22', '#e74c3c', '#c0392b']  # orange, red, dark red
    cpp_colors = ['#27ae60', '#2ecc71', '#82e0aa']     # dark green, green, light green

    # Python stacked bar with section labels
    py_bottom = 0
    for i in range(len(step_names)):
        ax2.bar(0, python_ms[i], bottom=py_bottom, width=0.5,
                color=colors_steps[i], edgecolor='white')
        # Label each section
        mid_y = py_bottom + python_ms[i] / 2
        ax2.text(0, mid_y, f'{step_names[i]}\n{python_ms[i]:.2f}ms',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        py_bottom += python_ms[i]

    # Interpreter overhead
    ax2.bar(0, interp_ms, bottom=py_bottom, width=0.5,
            color=C_OTHER, edgecolor='white', alpha=0.6)
    ax2.text(0, py_bottom + interp_ms / 2, f'Interp.\n{interp_ms:.2f}ms',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # C++ stacked bar - too small to label inside, use a side annotation
    cpp_bottom = 0
    for i in range(len(step_names)):
        ax2.bar(1, cpp_ms[i], bottom=cpp_bottom, width=0.5,
                color=cpp_colors[i], edgecolor='white')
        cpp_bottom += cpp_ms[i]

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Python (NumPy)\nsingle-threaded', 'C++ (fused)\nmulti-threaded + AVX-512'])
    ax2.set_ylabel('Total time (ms)')
    ax2.set_title(f'Total Pipeline Cost\nPython: {py_total:.1f}ms  vs  C++: {cpp_total:.3f}ms')

    # Annotate totals
    ax2.annotate(f'{py_total:.1f}ms', xy=(0, py_total), xytext=(0, 5),
                textcoords='offset points', ha='center', fontweight='bold',
                fontsize=14, color=C_PYTHON)
    ax2.annotate(f'{cpp_total:.3f}ms', xy=(1, cpp_total), xytext=(0, 5),
                textcoords='offset points', ha='center', fontweight='bold',
                fontsize=14, color=C_CPP)

    # C++ breakdown annotation (too small to fit inside bar)
    ax2.annotate(f'Gather: {cpp_ms[0]:.3f}ms\nQuantize: {cpp_ms[1]:.3f}ms\nTile: {cpp_ms[2]:.3f}ms',
                xy=(1.25, cpp_total), xytext=(1.5, 3),
                arrowprops=dict(arrowstyle='->', color=C_CPP, lw=1.5),
                fontsize=10, color=C_CPP,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor=C_CPP))

    # Speedup arrow
    ax2.annotate(f'{py_total/cpp_total:.0f}x faster',
                xy=(1, cpp_total + 0.3), xytext=(1.5, 7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen'))

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'why_cpp_faster.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Figure 4: Memory allocation comparison
# ============================================================
def fig4_memory_alloc():
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = ['Python\n(tile + encode)', 'C++ Fused\n(tile + encode)']
    mem_mb = [3468, 503]
    colors = [C_PYTHON, C_CPP]

    bars = ax.bar(methods, mem_mb, color=colors, edgecolor='white', width=0.5)
    ax.set_ylabel('Peak Memory Allocation (MB)')
    ax.set_title('Temporary Memory During Tiling + Encoding\n(4 large tables, 32.3M rows)')

    for bar, m in zip(bars, mem_mb):
        ax.text(bar.get_x() + bar.get_width()/2, m + 50,
               f'{m:,} MB', ha='center', fontweight='bold', fontsize=14)

    # Savings annotation
    saved = mem_mb[0] - mem_mb[1]
    ax.annotate(f'{saved/mem_mb[0]*100:.0f}% less memory\n({saved:,} MB saved)',
               xy=(1, mem_mb[1]), xytext=(1.3, 2000),
               arrowprops=dict(arrowstyle='->', color='black', lw=2),
               fontsize=13, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'memory_alloc.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Figure 5: Combined - tiling cost in context of full pipeline
# ============================================================
def fig5_tiling_in_context():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked bar: decode path components for a cache miss
    configs = ['Cache Miss\n(decode needed)', 'Cache Hit\n(no decode)']

    # Cache miss path
    decode_ms = 3.5
    untile_ms = 0.029
    gather_ms = 0.014
    dequant_ms = 0.005
    lookup_ms = 0.01

    # Cache hit path
    hit_lookup_ms = 0.25  # embedding lookup from cached data

    miss_components = [decode_ms, untile_ms, gather_ms + dequant_ms, lookup_ms]
    hit_components = [0, 0, 0, hit_lookup_ms]

    labels = ['H.265 Decode', 'Untile (C++)', 'Gather + Dequant', 'Embedding Lookup']
    colors = [C_DECODE, C_CPP, C_FUSED, C_FWD]

    bottom_miss = np.zeros(1)
    bottom_hit = np.zeros(1)

    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.barh(0, miss_components[i], left=bottom_miss, height=0.4,
                color=color, edgecolor='white', label=label)
        bottom_miss += miss_components[i]

        ax.barh(1, hit_components[i], left=bottom_hit, height=0.4,
                color=color, edgecolor='white')
        bottom_hit += hit_components[i]

    ax.set_yticks([0, 1])
    ax.set_yticklabels(configs)
    ax.set_xlabel('Time (ms)')
    ax.set_title('Tiling Cost vs Decode Cost per Cold Embedding Access')
    ax.legend(loc='upper right', fontsize=11)

    # Annotation: tiling is <1% of cache miss
    tiling_total = untile_ms + gather_ms + dequant_ms
    miss_total = sum(miss_components)
    ax.annotate(f'Tiling: {tiling_total:.3f}ms ({tiling_total/miss_total*100:.1f}% of miss cost)\n'
               f'Decode: {decode_ms:.1f}ms ({decode_ms/miss_total*100:.0f}% of miss cost)',
               xy=(2.5, 0.3), fontsize=12,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'tiling_in_context.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    fig1_tiling_speedup()
    fig2_pipeline_breakdown()
    fig3_why_cpp_faster()
    fig4_memory_alloc()
    fig5_tiling_in_context()
    print(f"\nAll figures saved to {OUTDIR}/")
