#!/usr/bin/env python3
"""
Generate presentation figures for embedding reordering research.
Produces two sets:
  - Concise: 4 key figures for quick overview
  - Detailed: 7 figures with full analysis
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette
C_BASELINE = '#8B8B8B'
C_FREQ = '#5B9BD5'
C_BATCH = '#ED7D31'
C_RECAD = '#70AD47'
C_CODEC = '#4472C4'
C_CODEC2 = '#9DC3E6'
C_HOT_ORIG = '#4472C4'
C_HOT_FREQ = '#5B9BD5'
C_HOT_BATCH = '#ED7D31'

# ── Load data ──────────────────────────────────────────────────────────
with open('results/reorder_only_results.json') as f:
    reorder = json.load(f)
with open('results/hot_reorder_results.json') as f:
    hot_reorder = json.load(f)
with open('results/ondemand_results.json') as f:
    ondemand = json.load(f)
with open('results/profiling/profiling_summary.json') as f:
    profiling = json.load(f)

# ── Helper ─────────────────────────────────────────────────────────────
def add_bar_labels(ax, bars, fmt='{:.2f}', offset=0.02, fontsize=9):
    ymax = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + ymax*offset,
                fmt.format(h), ha='center', va='bottom', fontsize=fontsize)


# ========================================================================
# CONCISE FIGURES (4 figures)
# ========================================================================

def fig1_reorder_speedup():
    """Bar chart: reordering methods speedup over baseline."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    methods = ['Baseline\n(original)', 'Frequency\nsort', 'Batch-affinity\n(ours)', 'Rec-AD\n(Louvain)']
    times = [reorder['baseline']['avg_time'], reorder['frequency_sort']['avg_time'],
             reorder['batch_affinity']['avg_time'], reorder['recad_louvain']['avg_time']]
    stds = [reorder['baseline']['std_time'], reorder['frequency_sort']['std_time'],
            reorder['batch_affinity']['std_time'], reorder['recad_louvain']['std_time']]
    colors = [C_BASELINE, C_FREQ, C_BATCH, C_RECAD]
    speedups = [times[0]/t for t in times]

    bars = ax.bar(methods, times, color=colors, edgecolor='white', linewidth=1.2,
                  yerr=stds, capsize=5, error_kw={'linewidth': 1.5})

    # Add speedup annotations
    for i, (bar, sp) in enumerate(zip(bars, speedups)):
        h = bar.get_height() + stds[i]
        label = '1.00x' if i == 0 else f'{sp:.2f}x'
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Inference time (s)')
    ax.set_title('Row Reordering: CPU Cache Locality Impact\n(Full fp32 embeddings, no compression)')
    ax.set_ylim(0, max(times) * 1.3)
    ax.axhline(y=times[0], color=C_BASELINE, linestyle='--', alpha=0.3, linewidth=1)

    fig.tight_layout()
    fig.savefig('results/fig1_reorder_speedup.png')
    fig.savefig('results/fig1_reorder_speedup.pdf')
    print('  Saved fig1_reorder_speedup')
    plt.close(fig)


def fig2_end_to_end_system():
    """Grouped bar: baseline vs codec pipeline (time + memory)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: inference time comparison
    configs = ['Baseline\nfp32', '1080p\n+bitmap', '4K\nfullcpp', '480p\n+bitmap']
    time_vals = [
        ondemand['A_baseline_t80']['total_time'],
        ondemand['1080p_bitmap_fullcpp']['total_time'],
        ondemand['4K_fullcpp']['total_time'],
        ondemand['480p_bitmap_fullcpp']['total_time'],
    ]
    colors_t = [C_BASELINE, C_CODEC, C_CODEC2, '#B4C7E7']
    speedups = [time_vals[0]/t for t in time_vals]

    bars1 = ax1.bar(configs, time_vals, color=colors_t, edgecolor='white', linewidth=1.2)
    for i, (bar, sp) in enumerate(zip(bars1, speedups)):
        label = '1.00x' if i == 0 else f'{sp:.2f}x'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Inference time (s)')
    ax1.set_title('Inference Latency')
    ax1.set_ylim(0, max(time_vals) * 1.25)

    # Right: memory comparison
    baseline_mem = ondemand['A_baseline_t80']['emb_memory_mb']
    codec_mems = {
        '1080p+bm': ondemand['1080p_bitmap_fullcpp']['total_mem_mb'],
        '4K': ondemand['4K_fullcpp']['total_mem_mb'],
        '480p+bm': ondemand['480p_bitmap_fullcpp']['total_mem_mb'],
    }

    mem_configs = ['Baseline\nfp32', '1080p\n+bitmap', '4K\nfullcpp', '480p\n+bitmap']
    mem_vals = [baseline_mem, codec_mems['1080p+bm'], codec_mems['4K'], codec_mems['480p+bm']]
    ratios = [1.0] + [baseline_mem/m for m in list(codec_mems.values())]

    bars2 = ax2.bar(mem_configs, mem_vals, color=colors_t, edgecolor='white', linewidth=1.2)
    for i, (bar, r) in enumerate(zip(bars2, ratios)):
        label = f'{mem_vals[i]:.0f}MB' if i == 0 else f'{mem_vals[i]:.0f}MB\n({r:.0f}x smaller)'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 label, ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('Embedding memory (MB)')
    ax2.set_title('Memory Footprint')
    ax2.set_ylim(0, max(mem_vals) * 1.25)

    fig.suptitle('H.265 Codec Embedding Compression: End-to-End Results', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('results/fig2_end_to_end.png')
    fig.savefig('results/fig2_end_to_end.pdf')
    print('  Saved fig2_end_to_end')
    plt.close(fig)


def fig3_latency_breakdown():
    """Stacked bar: embedding vs MLP latency for baseline and codec."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Baseline: from profiling summary (avg_emb_ms + avg_mlp_ms ~= total - overhead)
    baseline_total = ondemand['A_baseline_t80']['mean_lat_ms']
    baseline_emb = profiling['avg_emb_ms']
    baseline_mlp = profiling['avg_mlp_ms']
    baseline_other = baseline_total - baseline_emb - baseline_mlp

    # Codec 1080p bitmap: estimate emb from difference
    codec_total = ondemand['1080p_bitmap_fullcpp']['mean_lat_ms']
    # In codec, MLP is same, embedding is much faster
    codec_mlp = baseline_mlp  # MLP doesn't change
    codec_other = baseline_other  # interaction overhead similar
    codec_emb = codec_total - codec_mlp - codec_other
    if codec_emb < 0:
        codec_emb = 0.1
        codec_other = codec_total - codec_mlp - codec_emb

    configs = ['Baseline fp32', 'H.265 Codec\n(1080p+bitmap)']
    emb_vals = [baseline_emb, codec_emb]
    mlp_vals = [baseline_mlp, codec_mlp]
    other_vals = [max(0, baseline_other), max(0, codec_other)]

    x = np.arange(len(configs))
    w = 0.45

    b1 = ax.bar(x, emb_vals, w, label='Embedding lookup', color='#ED7D31')
    b2 = ax.bar(x, mlp_vals, w, bottom=emb_vals, label='MLP forward', color='#4472C4')
    b3 = ax.bar(x, other_vals, w, bottom=[e+m for e,m in zip(emb_vals, mlp_vals)],
                label='Interaction + other', color='#A5A5A5')

    # Add percentage labels
    for i in range(len(configs)):
        total = emb_vals[i] + mlp_vals[i] + other_vals[i]
        # Emb label
        if emb_vals[i] > 0.3:
            ax.text(x[i], emb_vals[i]/2, f'{emb_vals[i]:.1f}ms\n({emb_vals[i]/total*100:.0f}%)',
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        # MLP label
        ax.text(x[i], emb_vals[i] + mlp_vals[i]/2,
                f'{mlp_vals[i]:.1f}ms\n({mlp_vals[i]/total*100:.0f}%)',
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.set_ylabel('Per-batch latency (ms)')
    ax.set_title('Forward Pass Latency Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(baseline_total, codec_total) * 1.15)

    fig.tight_layout()
    fig.savefig('results/fig3_latency_breakdown.png')
    fig.savefig('results/fig3_latency_breakdown.pdf')
    print('  Saved fig3_latency_breakdown')
    plt.close(fig)


def fig4_hot_reorder_result():
    """Bar chart: hot reordering shows no benefit in codec pipeline."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    methods = ['Original\n(+bitmap)', 'Frequency\nsorted', 'Batch-affinity\nsorted']
    times = [hot_reorder['original_hot']['avg_time'],
             hot_reorder['freq_hot']['avg_time'],
             hot_reorder['batch_affinity_hot']['avg_time']]
    stds = [hot_reorder['original_hot']['std_time'],
            hot_reorder['freq_hot']['std_time'],
            hot_reorder['batch_affinity_hot']['std_time']]
    colors = [C_HOT_ORIG, C_HOT_FREQ, C_HOT_BATCH]

    bars = ax.bar(methods, times, color=colors, edgecolor='white', linewidth=1.2,
                  yerr=stds, capsize=5, error_kw={'linewidth': 1.5})

    speedups = [times[0]/t for t in times]
    for i, (bar, sp) in enumerate(zip(bars, speedups)):
        h = bar.get_height() + stds[i]
        label = '1.00x' if i == 0 else f'{sp:.2f}x'
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add annotation about hot size
    ax.annotate(f'Hot embeddings: {hot_reorder["original_hot"]["hot_mb"]:.0f}MB\n(fits in L3 cache)',
                xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_ylabel('Inference time (s)')
    ax.set_title('Hot Embedding Reordering: No Benefit\n(Codec pipeline, 1080p, H.265)')
    ax.set_ylim(0, max(times) * 1.25)
    ax.axhline(y=times[0], color=C_HOT_ORIG, linestyle='--', alpha=0.3, linewidth=1)

    fig.tight_layout()
    fig.savefig('results/fig4_hot_reorder.png')
    fig.savefig('results/fig4_hot_reorder.pdf')
    print('  Saved fig4_hot_reorder')
    plt.close(fig)


# ========================================================================
# DETAILED FIGURES (additional 3 figures)
# ========================================================================

def fig5_p99_comparison():
    """P99 tail latency comparison across all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: reorder-only P99
    methods = ['Baseline', 'Freq-sort', 'Batch-aff.', 'Rec-AD']
    p99s = [reorder['baseline']['p99_lat_ms'], reorder['frequency_sort']['p99_lat_ms'],
            reorder['batch_affinity']['p99_lat_ms'], reorder['recad_louvain']['p99_lat_ms']]
    colors = [C_BASELINE, C_FREQ, C_BATCH, C_RECAD]

    bars1 = ax1.bar(methods, p99s, color=colors, edgecolor='white', linewidth=1.2)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{bar.get_height():.1f}ms', ha='center', va='bottom', fontsize=9)

    ax1.set_ylabel('P99 latency (ms)')
    ax1.set_title('Reordering-Only (fp32)')
    ax1.set_ylim(0, max(p99s) * 1.2)

    # Arrow showing improvement
    ax1.annotate('', xy=(3, p99s[3]), xytext=(0, p99s[0]),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(1.5, (p99s[0]+p99s[3])/2 + 0.3,
             f'-{p99s[0]-p99s[3]:.1f}ms\n(-{(p99s[0]-p99s[3])/p99s[0]*100:.0f}%)',
             ha='center', fontsize=10, color='red', fontweight='bold')

    # Right: codec pipeline P99
    codec_methods = ['Baseline\nfp32', '1080p\nfullcpp', '1080p\n+bitmap', '4K\nfullcpp']
    codec_p99s = [
        ondemand['A_baseline_t80']['p99_lat_ms'],
        ondemand['1080p_fullcpp']['p99_lat_ms'],
        ondemand['1080p_bitmap_fullcpp']['p99_lat_ms'],
        ondemand['4K_fullcpp']['p99_lat_ms'],
    ]
    colors2 = [C_BASELINE, C_CODEC, C_CODEC, C_CODEC2]

    bars2 = ax2.bar(codec_methods, codec_p99s, color=colors2, edgecolor='white', linewidth=1.2)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{bar.get_height():.1f}ms', ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('P99 latency (ms)')
    ax2.set_title('Codec Pipeline')
    ax2.set_ylim(0, max(codec_p99s) * 1.2)

    fig.suptitle('P99 Tail Latency Comparison', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('results/fig5_p99_comparison.png')
    fig.savefig('results/fig5_p99_comparison.pdf')
    print('  Saved fig5_p99_comparison')
    plt.close(fig)


def fig6_memory_breakdown():
    """Stacked bar: memory breakdown for codec configs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = ['Baseline\nfp32', '1080p\nfullcpp', '1080p\n+bitmap', '4K\nfullcpp', '480p\n+bitmap']

    # Memory components
    baseline_emb = ondemand['A_baseline_t80']['emb_memory_mb']

    hot = [baseline_emb,
           ondemand['1080p_fullcpp']['hot_mb'],
           ondemand['1080p_bitmap_fullcpp']['hot_mb'],
           ondemand['4K_fullcpp']['hot_mb'],
           ondemand['480p_bitmap_fullcpp']['hot_mb']]

    mapping = [0,
               ondemand['1080p_fullcpp']['mapping_mb'],
               ondemand['1080p_bitmap_fullcpp']['mapping_mb'],
               ondemand['4K_fullcpp']['mapping_mb'],
               ondemand['480p_bitmap_fullcpp']['mapping_mb']]

    compressed = [0,
                  ondemand['1080p_fullcpp']['compressed_mb'],
                  ondemand['1080p_bitmap_fullcpp']['compressed_mb'],
                  ondemand['4K_fullcpp']['compressed_mb'],
                  ondemand['480p_bitmap_fullcpp']['compressed_mb']]

    lru = [0,
           ondemand['1080p_fullcpp']['lru_mb'],
           ondemand['1080p_bitmap_fullcpp']['lru_mb'],
           ondemand['4K_fullcpp']['lru_mb'],
           ondemand['480p_bitmap_fullcpp']['lru_mb']]

    x = np.arange(len(configs))
    w = 0.55

    b1 = ax.bar(x, hot, w, label='Hot embeddings (fp32)', color='#ED7D31')
    b2 = ax.bar(x, mapping, w, bottom=hot, label='Index mapping', color='#A5A5A5')
    b3 = ax.bar(x, compressed, w, bottom=[h+m for h,m in zip(hot, mapping)],
                label='Compressed cold (H.265)', color='#4472C4')
    b4 = ax.bar(x, lru, w, bottom=[h+m+c for h,m,c in zip(hot, mapping, compressed)],
                label='LRU frame cache', color='#70AD47')

    # Total labels
    totals = [h+m+c+l for h,m,c,l in zip(hot, mapping, compressed, lru)]
    for i, t in enumerate(totals):
        ax.text(x[i], t + 20, f'{t:.0f}MB', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Breakdown by Component')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, max(totals) * 1.15)

    fig.tight_layout()
    fig.savefig('results/fig6_memory_breakdown.png')
    fig.savefig('results/fig6_memory_breakdown.pdf')
    print('  Saved fig6_memory_breakdown')
    plt.close(fig)


def fig7_cooccurrence_sparsity():
    """Horizontal bar: per-table co-occurrence density showing why Rec-AD fails."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Per-table data from the comparison markdown
    tables = ['Table 2', 'Table 3', 'Table 9', 'Table 11',
              'Table 15', 'Table 20', 'Table 23', 'Table 25']
    cold_rows = [9.6e6, 2.2e6, 89e3, 8.0e6, 5.3e6, 6.7e6, 281e3, 138e3]
    active_cold = [449e3, 271e3, 41e3, 429e3, 401e3, 415e3, 73e3, 49e3]
    avg_batches = [1.0, 1.3, 8.2, 1.0, 1.0, 1.0, 3.6, 3.9]
    communities = [0, 262832, 26024, 0, 0, 0, 41978, 27037]  # 0 = fallback

    # Left: avg batches per cold index
    colors_bars = ['#D9534F' if b <= 1.0 else '#5CB85C' if b >= 3.0 else '#F0AD4E' for b in avg_batches]
    bars = ax1.barh(tables, avg_batches, color=colors_bars, edgecolor='white', linewidth=1.2)

    for bar, val in zip(bars, avg_batches):
        ax1.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}', ha='left', va='center', fontsize=10)

    ax1.set_xlabel('Avg batches per cold index')
    ax1.set_title('Co-occurrence Density')
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.text(1.1, 7.5, 'no co-occurrence\n(single-batch indices)', fontsize=8, color='red', style='italic')
    ax1.set_xlim(0, max(avg_batches) * 1.3)
    ax1.invert_yaxis()

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#D9534F', label='Fallback to batch-affinity'),
        mpatches.Patch(facecolor='#F0AD4E', label='Sparse co-occurrence'),
        mpatches.Patch(facecolor='#5CB85C', label='Rich co-occurrence'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # Right: active cold rows (log scale)
    bars2 = ax1_r = ax2.barh(tables, [a/1000 for a in active_cold],
                              color='#4472C4', edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars2, active_cold):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f'{val/1000:.0f}K', ha='left', va='center', fontsize=10)

    ax2.set_xlabel('Active cold rows (thousands)')
    ax2.set_title('Cold Row Count')
    ax2.set_xlim(0, max(active_cold)/1000 * 1.25)
    ax2.invert_yaxis()

    fig.suptitle('Why Rec-AD Community Detection = Batch-Affinity on Kaggle/Criteo',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('results/fig7_cooccurrence_sparsity.png')
    fig.savefig('results/fig7_cooccurrence_sparsity.pdf')
    print('  Saved fig7_cooccurrence_sparsity')
    plt.close(fig)


# ========================================================================
# COMPOSITE / OVERVIEW FIGURE
# ========================================================================

def fig0_overview_composite():
    """Single 2x2 composite figure combining the 4 concise results."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Reordering speedup ──
    ax1 = fig.add_subplot(gs[0, 0])
    methods = ['Baseline', 'Freq-sort', 'Batch-aff.\n(ours)', 'Rec-AD']
    times = [reorder['baseline']['avg_time'], reorder['frequency_sort']['avg_time'],
             reorder['batch_affinity']['avg_time'], reorder['recad_louvain']['avg_time']]
    stds = [reorder['baseline']['std_time'], reorder['frequency_sort']['std_time'],
            reorder['batch_affinity']['std_time'], reorder['recad_louvain']['std_time']]
    colors = [C_BASELINE, C_FREQ, C_BATCH, C_RECAD]
    speedups = [times[0]/t for t in times]

    bars = ax1.bar(methods, times, color=colors, edgecolor='white', linewidth=1,
                   yerr=stds, capsize=4, error_kw={'linewidth': 1.2})
    for i, (bar, sp) in enumerate(zip(bars, speedups)):
        label = '1.00x' if i == 0 else f'{sp:.2f}x'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.15,
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('(a) Cold Row Reordering (fp32, no compression)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(times) * 1.3)

    # ── Panel B: End-to-end codec results ──
    ax2 = fig.add_subplot(gs[0, 1])
    configs = ['Baseline\nfp32', '1080p\n+bitmap', '4K', '480p\n+bitmap']
    time_vals = [
        ondemand['A_baseline_t80']['total_time'],
        ondemand['1080p_bitmap_fullcpp']['total_time'],
        ondemand['4K_fullcpp']['total_time'],
        ondemand['480p_bitmap_fullcpp']['total_time'],
    ]
    mem_vals = [
        ondemand['A_baseline_t80']['emb_memory_mb'],
        ondemand['1080p_bitmap_fullcpp']['total_mem_mb'],
        ondemand['4K_fullcpp']['total_mem_mb'],
        ondemand['480p_bitmap_fullcpp']['total_mem_mb'],
    ]
    colors2 = [C_BASELINE, C_CODEC, C_CODEC2, '#B4C7E7']

    bars2 = ax2.bar(configs, time_vals, color=colors2, edgecolor='white', linewidth=1)
    speedups2 = [time_vals[0]/t for t in time_vals]
    for i, (bar, sp) in enumerate(zip(bars2, speedups2)):
        label = '1.00x' if i == 0 else f'{sp:.2f}x'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add memory reduction as secondary annotation
    for i in range(1, len(configs)):
        ratio = mem_vals[0] / mem_vals[i]
        ax2.text(bars2[i].get_x() + bars2[i].get_width()/2, 0.3,
                 f'{ratio:.0f}x mem\nreduction', ha='center', va='bottom',
                 fontsize=8, color='white', fontweight='bold')

    ax2.set_ylabel('Time (s)')
    ax2.set_title('(b) H.265 Codec Pipeline Results', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(time_vals) * 1.25)

    # ── Panel C: Hot reordering (no benefit) ──
    ax3 = fig.add_subplot(gs[1, 0])
    hot_methods = ['Original\n(+bitmap)', 'Freq-sort', 'Batch-aff.']
    hot_times = [hot_reorder['original_hot']['avg_time'],
                 hot_reorder['freq_hot']['avg_time'],
                 hot_reorder['batch_affinity_hot']['avg_time']]
    hot_stds = [hot_reorder['original_hot']['std_time'],
                hot_reorder['freq_hot']['std_time'],
                hot_reorder['batch_affinity_hot']['std_time']]
    hot_colors = [C_HOT_ORIG, C_HOT_FREQ, C_HOT_BATCH]

    bars3 = ax3.bar(hot_methods, hot_times, color=hot_colors, edgecolor='white', linewidth=1,
                    yerr=hot_stds, capsize=4, error_kw={'linewidth': 1.2})
    hot_sp = [hot_times[0]/t for t in hot_times]
    for i, (bar, sp) in enumerate(zip(bars3, hot_sp)):
        label = '1.00x' if i == 0 else f'{sp:.2f}x'
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + hot_stds[i] + 0.05,
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.annotate(f'Hot emb: {hot_reorder["original_hot"]["hot_mb"]:.0f}MB (fits in L3)',
                 xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                 fontsize=8, style='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax3.set_ylabel('Time (s)')
    ax3.set_title('(c) Hot Embedding Reordering (no benefit)', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, max(hot_times) * 1.3)

    # ── Panel D: Co-occurrence sparsity ──
    ax4 = fig.add_subplot(gs[1, 1])
    tables = ['T2', 'T3', 'T9', 'T11', 'T15', 'T20', 'T23', 'T25']
    avg_batches = [1.0, 1.3, 8.2, 1.0, 1.0, 1.0, 3.6, 3.9]
    bar_colors = ['#D9534F' if b <= 1.0 else '#5CB85C' if b >= 3.0 else '#F0AD4E' for b in avg_batches]

    bars4 = ax4.bar(tables, avg_batches, color=bar_colors, edgecolor='white', linewidth=1)
    for bar, val in zip(bars4, avg_batches):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax4.set_ylabel('Avg batches per cold index')
    ax4.set_title('(d) Why Rec-AD = Batch-Affinity', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(avg_batches) * 1.3)

    legend_elements = [
        mpatches.Patch(facecolor='#D9534F', label='Single-batch (no graph)'),
        mpatches.Patch(facecolor='#F0AD4E', label='Sparse graph'),
        mpatches.Patch(facecolor='#5CB85C', label='Rich graph'),
    ]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.savefig('results/fig0_overview_composite.png')
    fig.savefig('results/fig0_overview_composite.pdf')
    print('  Saved fig0_overview_composite')
    plt.close(fig)


# ========================================================================
# Main
# ========================================================================
if __name__ == '__main__':
    print('Generating CONCISE figures (4):')
    fig1_reorder_speedup()
    fig2_end_to_end_system()
    fig3_latency_breakdown()
    fig4_hot_reorder_result()

    print('\nGenerating DETAILED figures (3 additional):')
    fig5_p99_comparison()
    fig6_memory_breakdown()
    fig7_cooccurrence_sparsity()

    print('\nGenerating COMPOSITE overview:')
    fig0_overview_composite()

    print('\nAll figures saved to results/')
    print('  Concise:  fig1-fig4 (.png + .pdf)')
    print('  Detailed: fig5-fig7 (.png + .pdf)')
    print('  Overview: fig0_overview_composite (.png + .pdf)')
