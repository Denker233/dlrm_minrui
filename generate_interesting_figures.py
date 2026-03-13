#!/usr/bin/env python3
"""Generate interesting/insightful figures for the H.265 DLRM compression paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("results/interesting_figures", exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'figure.facecolor': 'white',
})

# ==========================================================================
# FIGURE 1: "The Compression Paradox" — metadata costs more than the data
# ==========================================================================
def fig_compression_paradox():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]})

    # Left: Kaggle storage breakdown (what's on disk)
    ax = axes[0]
    labels = ['Hot emb\n(fp32)', 'Cold compressed\n(H.265 CRF=18)']
    sizes = [88.5, 0.8]
    colors = ['#E91E63', '#4CAF50']
    explode = (0, 0.15)
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       explode=explode, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 11})
    autotexts[1].set_fontweight('bold')
    autotexts[1].set_fontsize(13)
    ax.set_title('Storage: 89 MB total\n(23x compression)', fontsize=14, fontweight='bold')

    # Right: Kaggle runtime memory breakdown
    ax = axes[1]
    components = ['Hot emb\n(fp32)', 'Mapping\ntables', 'Decoded\ncache', 'Compressed\ncold']
    mem = [88.5, 134.6, 39.6, 0.8]
    colors_rt = ['#E91E63', '#FF9800', '#2196F3', '#4CAF50']
    bars = ax.barh(components, mem, color=colors_rt, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, mem):
        xpos = val + 3
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f'{val:.1f} MB ({val/263.4*100:.0f}%)',
                va='center', fontsize=11, fontweight='bold')

    # Highlight the paradox
    ax.axvline(x=0.8, color='#4CAF50', linestyle='--', alpha=0.5)
    ax.annotate('Mapping is 168x\nlarger than the\ndata it indexes!',
                xy=(134.6, 1), xytext=(100, 2.8),
                fontsize=12, color='#FF9800', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2))
    ax.set_xlabel('Memory (MB)', fontsize=13)
    ax.set_title('Runtime Memory: 263 MB total\n(7.8x reduction)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 180)

    plt.suptitle('The Compression Paradox: Metadata > Data',
                 fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('results/interesting_figures/compression_paradox.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: compression_paradox.png")


# ==========================================================================
# FIGURE 2: Entropy predicts compression — per-table scatter
# ==========================================================================
def fig_entropy_vs_compression():
    # Terabyte tables: entropy vs CRF=18 ratio
    tb_tables = {
        'T0':  {'entropy': 1.490, 'crf18': 50.9,  'rows': 5.45, 'near_zp': 100.0},
        'T9':  {'entropy': 1.962, 'crf18': 24.5,  'rows': 3.95, 'near_zp': 100.0},
        'T21': {'entropy': 1.576, 'crf18': 41.8,  'rows': 4.94, 'near_zp': 100.0},
        'T19': {'entropy': 2.748, 'crf18': 7.8,   'rows': 5.73, 'near_zp': 83.7},
        'T20': {'entropy': 3.055, 'crf18': 5.8,   'rows': 1.79, 'near_zp': 66.2},
        'T10': {'entropy': 3.396, 'crf18': 4.6,   'rows': 0.40, 'near_zp': 51.5},
        'T11': {'entropy': 4.225, 'crf18': 3.0,   'rows': 0.14, 'near_zp': 28.2},
        'T22': {'entropy': 4.254, 'crf18': 2.9,   'rows': 0.20, 'near_zp': 26.9},
    }

    # Kaggle tables (from intrinsic compressibility, approximate from logs)
    kaggle_tables = {
        'T2':  {'entropy': 0.13, 'crf18': 1409, 'rows': 10.1},
        'T3':  {'entropy': 0.35, 'crf18': 800,  'rows': 2.2},
        'T11': {'entropy': 0.52, 'crf18': 500,  'rows': 8.4},
        'T20': {'entropy': 0.71, 'crf18': 350,  'rows': 7.0},
        'T15': {'entropy': 0.44, 'crf18': 600,  'rows': 5.5},
        'T25': {'entropy': 1.20, 'crf18': 100,  'rows': 0.14},
        'T23': {'entropy': 2.00, 'crf18': 50,   'rows': 0.29},
        'T9':  {'entropy': 2.52, 'crf18': 30,   'rows': 0.09},
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    # Terabyte points
    for name, d in tb_tables.items():
        s = max(50, d['rows'] * 30)
        ax.scatter(d['entropy'], d['crf18'], s=s, c='#2196F3', alpha=0.8,
                   edgecolors='black', linewidth=0.5, zorder=5)
        offset = (8, 5) if d['crf18'] > 10 else (8, -12)
        ax.annotate(name, (d['entropy'], d['crf18']),
                    textcoords='offset points', xytext=offset,
                    fontsize=8, color='#2196F3')

    # Kaggle points
    for name, d in kaggle_tables.items():
        s = max(50, d['rows'] * 15)
        ax.scatter(d['entropy'], d['crf18'], s=s, c='#E91E63', alpha=0.8,
                   edgecolors='black', linewidth=0.5, zorder=5)
        offset = (8, 5)
        ax.annotate(name, (d['entropy'], d['crf18']),
                    textcoords='offset points', xytext=offset,
                    fontsize=8, color='#E91E63')

    # Fit line (log scale)
    all_ent = [d['entropy'] for d in list(tb_tables.values()) + list(kaggle_tables.values())]
    all_rat = [d['crf18'] for d in list(tb_tables.values()) + list(kaggle_tables.values())]
    z = np.polyfit(all_ent, np.log10(all_rat), 1)
    x_fit = np.linspace(0.1, 4.5, 100)
    y_fit = 10 ** np.polyval(z, x_fit)
    ax.plot(x_fit, y_fit, 'k--', alpha=0.3, linewidth=2, label=f'Fit: log(ratio) ~ {z[0]:.1f}·entropy')

    ax.set_yscale('log')
    ax.set_xlabel('uint8 Entropy (bits/byte)', fontsize=14)
    ax.set_ylabel('H.265 CRF=18 Compression Ratio', fontsize=14)
    ax.set_title('Lower Entropy → Higher Compression\n(bubble size ~ table rows)',
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    legend_elements = [
        plt.scatter([], [], c='#E91E63', s=100, edgecolors='black', linewidth=0.5, label='Kaggle (D=16)'),
        plt.scatter([], [], c='#2196F3', s=100, edgecolors='black', linewidth=0.5, label='Terabyte (D=64)'),
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc='upper right')

    # Add annotation for the key insight
    ax.annotate('Kaggle D=16: extremely\nlow entropy → 1000x+ compression',
                xy=(0.3, 800), xytext=(1.5, 600),
                fontsize=11, color='#E91E63', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5))

    ax.annotate('Terabyte D=64: higher\nentropy → modest compression',
                xy=(3.5, 4), xytext=(2.5, 15),
                fontsize=11, color='#2196F3', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5))

    plt.tight_layout()
    plt.savefig('results/interesting_figures/entropy_vs_compression.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: entropy_vs_compression.png")


# ==========================================================================
# FIGURE 3: Frame access concentration — freq sort magic
# ==========================================================================
def fig_frame_concentration():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Kaggle per-table: accessed vs total frames
    tables_k  = ['T2', 'T3', 'T9', 'T11', 'T15', 'T20', 'T23', 'T25']
    total_k   = [75,    17,   1,    62,    41,    53,    3,     2]
    accessed_k = [4,    2,    1,    4,     3,     4,     1,     1]

    ax = axes[0]
    x = np.arange(len(tables_k))
    w = 0.35
    b1 = ax.bar(x - w/2, total_k, w, color='#BBDEFB', edgecolor='#1565C0',
                label='Total frames', linewidth=1)
    b2 = ax.bar(x + w/2, accessed_k, w, color='#E91E63', edgecolor='#880E4F',
                label='Accessed frames', linewidth=1)

    # Add percentage labels
    for i, (tot, acc) in enumerate(zip(total_k, accessed_k)):
        pct = acc / tot * 100
        ax.text(i + w/2, acc + 1, f'{pct:.0f}%', ha='center', fontsize=9,
                fontweight='bold', color='#E91E63')

    ax.set_xticks(x)
    ax.set_xticklabels(tables_k)
    ax.set_ylabel('Number of Frames', fontsize=12)
    ax.set_title('Kaggle: Only 20/254 frames accessed (92% reduction)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: "What if" memory comparison
    ax = axes[1]
    scenarios = ['All frames\ndecoded', 'Accessed only\n(freq sort)', 'LRU cache\n(size=4)']
    # All 254 frames × 1.98 MB/frame ≈ 503 MB decoded
    # 20 frames × 1.98 MB = 39.6 MB
    # LRU=4 × 8 tables × 1.98 MB ≈ 63 MB (but actually 7.9 MB with per-table cache=4)
    decoded_mem = [502.6, 39.6, 7.9]
    colors = ['#F44336', '#4CAF50', '#2196F3']

    bars = ax.bar(scenarios, decoded_mem, color=colors, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, decoded_mem):
        ax.text(bar.get_x() + bar.get_width()/2, val + 10,
                f'{val:.0f} MB', ha='center', fontsize=12, fontweight='bold')

    # Add savings annotation
    ax.annotate(f'12.7x\nsaved', xy=(1, 39.6), xytext=(0.5, 250),
                fontsize=13, fontweight='bold', color='#4CAF50',
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))

    ax.set_ylabel('Decoded Cache Memory (MB)', fontsize=12)
    ax.set_title('Frequency Sorting Concentrates Access',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 580)

    plt.suptitle('Frame Access Concentration: The Power of Frequency Sorting',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/interesting_figures/frame_concentration.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: frame_concentration.png")


# ==========================================================================
# FIGURE 4: Mapping optimization — current vs bitmap rank
# ==========================================================================
def fig_mapping_optimization():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Current memory breakdown vs optimized
    ax = axes[0]
    configs = ['Current\n(263 MB)', 'Bitmap Rank\n(~135 MB)', 'Theoretical\nMinimum']
    hot =     [88.5, 88.5, 88.5]
    compressed = [0.8, 0.8, 0.8]
    decoded = [39.6, 39.6, 39.6]
    mapping = [134.6, 7.0, 0]  # bitmap rank = ~7 MB, theoretical = 0

    x = np.arange(len(configs))
    w = 0.5
    b1 = ax.bar(x, hot, w, color='#E91E63', label='Hot embeddings (fp32)')
    b2 = ax.bar(x, compressed, w, bottom=hot, color='#4CAF50', label='Compressed cold')
    b3 = ax.bar(x, decoded, w, bottom=[h+c for h,c in zip(hot,compressed)],
                color='#2196F3', label='Decoded cache')
    b4 = ax.bar(x, mapping, w, bottom=[h+c+d for h,c,d in zip(hot,compressed,decoded)],
                color='#FF9800', label='Mapping overhead')

    totals = [h+c+d+m for h,c,d,m in zip(hot,compressed,decoded,mapping)]
    for i, t in enumerate(totals):
        ax.text(i, t + 5, f'{t:.0f} MB\n({2058/t:.1f}x)', ha='center',
                fontsize=11, fontweight='bold')

    # Savings arrow
    ax.annotate('', xy=(1, totals[1]+3), xycoords='data',
                xytext=(0, totals[0]+3), textcoords='data',
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2.5))
    ax.text(0.5, 220, '−128 MB\n(49% less)', ha='center', fontsize=11,
            fontweight='bold', color='#4CAF50')

    ax.set_ylabel('Runtime Memory (MB)', fontsize=12)
    ax.set_title('Memory Breakdown: Current vs Optimized', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 310)

    # Right: What the mapping contains
    ax = axes[1]
    # 35.6M rows × 4B = 134.6 MB int32 mapping
    # vs bitmap: 35.6M bits = 4.5 MB + 2.2 MB rank index = 6.7 MB
    approaches = ['int32 per row\n(current)', 'Bitmap + rank\n(proposed)']
    int32_data = [134.6, 0]
    bitmap_data = [0, 4.5]
    rank_idx = [0, 2.2]

    b1 = ax.bar(approaches, int32_data, 0.5, color='#FF9800', label='int32 lookup array')
    b2 = ax.bar(approaches, bitmap_data, 0.5, bottom=int32_data,
                color='#2196F3', label='Hot/cold bitmap (1 bit/row)')
    b3 = ax.bar(approaches, rank_idx, 0.5,
                bottom=[a+b for a,b in zip(int32_data, bitmap_data)],
                color='#9C27B0', label='Rank index (4B/64 rows)')

    ax.text(0, 134.6 + 3, '134.6 MB\n(4B × 35.6M rows)', ha='center',
            fontsize=11, fontweight='bold', color='#FF9800')
    ax.text(1, 6.7 + 3, '6.7 MB\n(19x smaller)', ha='center',
            fontsize=11, fontweight='bold', color='#4CAF50')

    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('Mapping Data Structure Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 165)

    plt.suptitle('Optimization Opportunity: Bitmap Rank Replaces int32 Mapping',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/interesting_figures/mapping_optimization.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: mapping_optimization.png")


# ==========================================================================
# FIGURE 5: Kaggle vs Terabyte — why embedding dimension matters
# ==========================================================================
def fig_dimension_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Compression ratio at different CRFs
    ax = axes[0]
    crfs = [0, 10, 18, 23, 28]
    kaggle_ratio  = [145.1, 489.7, 1368.8, 1972.9, 2405.6]  # fp32 ratio, cold only
    terabyte_ratio = [11.0, 33.5, 247.0, 1213.7, 2385.6]     # fp32 ratio, cold only

    ax.plot(crfs, kaggle_ratio, 'o-', color='#E91E63', linewidth=2.5, markersize=10,
            label='Kaggle (D=16)')
    ax.plot(crfs, terabyte_ratio, 's-', color='#2196F3', linewidth=2.5, markersize=10,
            label='Terabyte (D=64)')

    ax.set_yscale('log')
    ax.set_xlabel('CRF Value', fontsize=12)
    ax.set_ylabel('Cold Compression Ratio (vs fp32)', fontsize=12)
    ax.set_title('Compression Ratio', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Annotate the gap at CRF=18
    ax.annotate(f'5.5x gap', xy=(18, 500), fontsize=11, fontweight='bold',
                color='gray', ha='center')
    ax.annotate('', xy=(18, 1368), xycoords='data',
                xytext=(18, 247), textcoords='data',
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))

    # Panel 2: AUC loss at different CRFs
    ax = axes[1]
    kaggle_auc   = [0.0016, 0.0114, 0.0383, 0.0933, 0.2049]   # in %
    terabyte_auc = [0.0009, 0.0209, 0.0875, 0.1844, 0.4067]   # in %

    ax.plot(crfs, kaggle_auc, 'o-', color='#E91E63', linewidth=2.5, markersize=10,
            label='Kaggle (D=16)')
    ax.plot(crfs, terabyte_auc, 's-', color='#2196F3', linewidth=2.5, markersize=10,
            label='Terabyte (D=64)')

    ax.set_yscale('log')
    ax.set_xlabel('CRF Value', fontsize=12)
    ax.set_ylabel('|AUC Loss| (%)', fontsize=12)
    ax.set_title('Quality Degradation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 3: Entropy distribution
    ax = axes[2]
    kaggle_ent = [0.13, 0.35, 0.44, 0.52, 0.71, 1.20, 2.00, 2.52]
    terabyte_ent = [1.49, 1.58, 1.96, 2.75, 3.05, 3.40, 4.22, 4.25]

    bp1 = ax.boxplot([kaggle_ent], positions=[1], widths=0.6,
                      patch_artist=True, boxprops=dict(facecolor='#E91E63', alpha=0.5))
    bp2 = ax.boxplot([terabyte_ent], positions=[2], widths=0.6,
                      patch_artist=True, boxprops=dict(facecolor='#2196F3', alpha=0.5))

    # Overlay individual points
    ax.scatter(np.ones(len(kaggle_ent)) + np.random.normal(0, 0.05, len(kaggle_ent)),
               kaggle_ent, c='#E91E63', s=60, zorder=5, edgecolors='black', linewidth=0.5)
    ax.scatter(2*np.ones(len(terabyte_ent)) + np.random.normal(0, 0.05, len(terabyte_ent)),
               terabyte_ent, c='#2196F3', s=60, zorder=5, edgecolors='black', linewidth=0.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Kaggle\n(D=16)', 'Terabyte\n(D=64)'], fontsize=11)
    ax.set_ylabel('uint8 Entropy (bits/byte)', fontsize=12)
    ax.set_title('Intrinsic Entropy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax.annotate('Low entropy\n= free compression', xy=(1, 0.5), xytext=(1.4, 0.8),
                fontsize=10, color='#E91E63', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1))

    plt.suptitle('Kaggle (D=16) vs Terabyte (D=64): Why Embedding Dimension Matters',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('results/interesting_figures/dimension_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: dimension_comparison.png")


# ==========================================================================
# FIGURE 6: The "20x less loss" Pareto domination (cleaner version)
# ==========================================================================
def fig_pareto_domination():
    fig, ax = plt.subplots(figsize=(10, 7))

    # All methods
    methods = [
        # (name, ratio, auc_loss_pct, marker, color, size)
        ('INT8',           4.0,   0.002, 's', '#2196F3', 80),
        ('INT4',           8.0,   0.190, 's', '#2196F3', 80),
        ('PQ M=2',         31.9,  0.559, 'D', '#FF9800', 80),
        ('SVD r=4',        4.0,   0.101, 'v', '#9C27B0', 80),
        ('Prune 99%',      100,   0.181, '^', '#795548', 80),
        ('Zstd-19+uint8',  132.7, 0.002, 'P', '#607D8B', 100),
        ('H.265 CRF=0',    130.6, 0.002, 'o', '#4CAF50', 100),
    ]

    for name, ratio, loss, marker, color, s in methods:
        ax.scatter(ratio, loss, marker=marker, c=color, s=s,
                   edgecolors='black', linewidth=0.5, zorder=5, alpha=0.8)
        # Label select points
        if name in ('Zstd-19+uint8', 'Prune 99%', 'PQ M=2'):
            ax.annotate(name, (ratio, loss), textcoords='offset points',
                       xytext=(10, 5), fontsize=9, color=color)

    # CAFE+ with connecting line
    cafe_x = [10, 100, 1000]
    cafe_y = [0.25, 0.45, 0.75]
    ax.plot(cafe_x, cafe_y, 'X--', color='#F44336', markersize=12, linewidth=1.5,
            label='CAFE+ (requires retraining)', alpha=0.8)
    ax.annotate('CAFE+ 1000x', (1000, 0.75), textcoords='offset points',
               xytext=(-80, 10), fontsize=10, color='#F44336', fontweight='bold')

    # Our best point — BIG
    ax.scatter([1360], [0.037], marker='*', c='#E91E63', s=500, zorder=10,
              edgecolors='black', linewidth=1)
    ax.annotate('H.265 CRF=18+freq\n1360x, 0.037% loss',
               (1360, 0.037), textcoords='offset points',
               xytext=(-180, -40), fontsize=12, color='#E91E63', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#E91E63', lw=2))

    # Draw the "20x less loss" comparison
    ax.annotate('', xy=(1360, 0.037), xycoords='data',
                xytext=(1000, 0.75), textcoords='data',
                arrowprops=dict(arrowstyle='<->', color='black', lw=2, linestyle='--'))
    ax.text(900, 0.15, '20x less loss\nat higher compression\n(no retraining!)',
            fontsize=12, fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compression Ratio (cold embeddings vs fp32)', fontsize=14)
    ax.set_ylabel('|AUC Loss| (%)', fontsize=14)
    ax.set_title('H.265 Dominates the Pareto Frontier',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # "Better" arrow
    ax.annotate('', xy=(0.95, 0.05), xycoords='axes fraction',
               xytext=(0.80, 0.20), textcoords='axes fraction',
               arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(0.88, 0.12, 'Better', transform=ax.transAxes,
           fontsize=12, color='green', ha='center', style='italic')

    ax.set_xlim(2, 3000)
    ax.set_ylim(0.001, 1.5)
    plt.tight_layout()
    plt.savefig('results/interesting_figures/pareto_domination.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: pareto_domination.png")


# ==========================================================================
# FIGURE 7: Deployment scenarios — what fits where
# ==========================================================================
def fig_deployment():
    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = ['Edge\n(1 GB RAM)', 'Laptop\n(16 GB RAM)', 'Server\n(256 GB RAM)']
    baseline_instances = [0, 7, 124]  # 2058 MB baseline
    current_instances  = [3, 60, 973]  # 263 MB current
    optimized_instances = [7, 118, 1896]  # 135 MB optimized

    x = np.arange(len(scenarios))
    w = 0.25

    b1 = ax.bar(x - w, baseline_instances, w, color='#F44336', label='Baseline (2058 MB)', alpha=0.85)
    b2 = ax.bar(x, current_instances, w, color='#FF9800', label='Current (263 MB)', alpha=0.85)
    b3 = ax.bar(x + w, optimized_instances, w, color='#4CAF50', label='Bitmap-optimized (135 MB)', alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 15,
                        f'{int(height)}', ha='center', fontsize=10, fontweight='bold')

    # Special annotation for edge
    ax.annotate("Baseline\ncan't fit!", xy=(0 - w, 0), xytext=(-0.8, 200),
                fontsize=11, fontweight='bold', color='#F44336',
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.set_ylabel('Model Instances', fontsize=13)
    ax.set_title('Deployment Scaling: Model Instances per Device',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/interesting_figures/deployment_scaling.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: deployment_scaling.png")


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    fig_compression_paradox()
    fig_entropy_vs_compression()
    fig_frame_concentration()
    fig_mapping_optimization()
    fig_dimension_comparison()
    fig_pareto_domination()
    fig_deployment()
    print("\nAll 7 figures saved to results/interesting_figures/")
