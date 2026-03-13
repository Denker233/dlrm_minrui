#!/usr/bin/env python3
"""Generate all paper figures from experimental data.

Figures:
1. Kaggle Pareto frontier (compression ratio vs AUC loss)
2. Terabyte Pareto frontier
3. CRF vs AUC/ratio dual-axis
4. Memory breakdown (Kaggle + Terabyte)
5. Error steering (per-bucket MSE)
6. Speedup attribution (stacked bar)
7. Frame access concentration (freq sort benefit)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results/paper_figures", exist_ok=True)

# Color palette
C_QUANT   = '#2196F3'
C_PQ      = '#FF9800'
C_SVD     = '#9C27B0'
C_PRUNE   = '#795548'
C_ZSTD    = '#607D8B'
C_H265L   = '#4CAF50'
C_H265    = '#E91E63'
C_CAFE    = '#F44336'
C_HASH    = '#FF5722'

# ============================================================
# DATA: Kaggle (D=16, 26 tables, baseline AUC=0.802497)
# ============================================================
kaggle_baseline = 0.802497

# From Experiment 2 (mlsys_baselines.log)
kaggle_exp2 = [
    ('INT8',                4.0,    -0.000016, 's', C_QUANT, 'Quantization'),
    ('INT4',                8.0,    -0.001899, 's', C_QUANT, 'Quantization'),
    ('PQ M=2',              31.9,   -0.005594, 'D', C_PQ,    'Product Quant.'),
    ('PQ M=4',              16.0,   -0.005040, 'D', C_PQ,    'Product Quant.'),
    ('SVD r=1',             16.0,   -0.003513, 'v', C_SVD,   'SVD'),
    ('SVD r=2',             8.0,    -0.002658, 'v', C_SVD,   'SVD'),
    ('SVD r=4',             4.0,    -0.001008, 'v', C_SVD,   'SVD'),
    ('SVD r=8',             2.0,    -0.000242, 'v', C_SVD,   'SVD'),
    ('Prune 90%',           10.0,   -0.000081, '^', C_PRUNE, 'Row Pruning'),
    ('Prune 95%',           20.0,   -0.000306, '^', C_PRUNE, 'Row Pruning'),
    ('Prune 99%',           100.0,  -0.001813, '^', C_PRUNE, 'Row Pruning'),
    ('Zstd-19',             132.7,  -0.000016, 'P', C_ZSTD,  'Zstd-19'),
    ('H.265 CRF=0',        130.6,  -0.000016, 'o', C_H265L, 'H.265 Lossless'),
    ('H.265 CRF=18 nat.',   1186.0, -0.000700, 'o', C_H265L, 'H.265 Natural'),
    ('H.265 CRF=18+freq',  1359.8, -0.000372, '*', C_H265,  'H.265+Freq (Ours)'),
]

# CRF sweep (from crf_and_entropy, tiled+freq, cold only)
kaggle_crf = [
    ('CRF=0',   145.1,   -0.000016),
    ('CRF=10',  489.7,   -0.000114),
    ('CRF=18',  1368.8,  -0.000383),
    ('CRF=23',  1972.9,  -0.000933),
    ('CRF=28',  2405.6,  -0.002049),
    ('CRF=33',  2661.1,  -0.005236),
]

# CAFE+ paper numbers (SIGMOD 2024, approximate from Figure 7)
kaggle_cafe = [
    ('CAFE+ 10x',     10.0,   -0.0025,  'X', C_CAFE, 'CAFE+'),
    ('CAFE+ 100x',    100.0,  -0.0045,  'X', C_CAFE, 'CAFE+'),
    ('CAFE+ 1000x',   1000.0, -0.0175,  'X', C_CAFE, 'CAFE+'),
]

# ============================================================
# DATA: Terabyte (D=64, 26 tables, baseline AUC=0.768818)
# ============================================================
terabyte_baseline = 0.768818

terabyte_exp2 = [
    ('INT8',                4.0,    -0.000006, 's', C_QUANT, 'Quantization'),
    ('INT4',                8.0,    -0.000937, 's', C_QUANT, 'Quantization'),
    ('PQ M=2',              126.5,  -0.007629, 'D', C_PQ,    'Product Quant.'),
    ('PQ M=4',              63.6,   -0.007394, 'D', C_PQ,    'Product Quant.'),
    ('SVD r=1',             64.0,   -0.010002, 'v', C_SVD,   'SVD'),
    ('SVD r=4',             16.0,   -0.004786, 'v', C_SVD,   'SVD'),
    ('SVD r=8',             8.0,    -0.003330, 'v', C_SVD,   'SVD'),
    ('Prune 90%',           10.0,   -0.000005, '^', C_PRUNE, 'Row Pruning'),
    ('Prune 95%',           20.0,   -0.000016, '^', C_PRUNE, 'Row Pruning'),
    ('Prune 99%',           100.0,  -0.000193, '^', C_PRUNE, 'Row Pruning'),
    ('Zstd-19',             14.2,   -0.000006, 'P', C_ZSTD,  'Zstd-19'),
    ('H.265 CRF=0',        13.7,   -0.000006, 'o', C_H265L, 'H.265 Lossless'),
    ('H.265 CRF=18 nat.',   68.6,   -0.000934, 'o', C_H265L, 'H.265 Natural'),
    ('H.265 CRF=18+freq',  68.6,   -0.000503, '*', C_H265,  'H.265+Freq (Ours)'),
]

# Terabyte CRF sweep (from crf_cache_memory, full table sort, Phase 5 AUC)
terabyte_crf = [
    ('CRF=0',   11.0,    -0.000009),
    ('CRF=10',  33.5,    -0.000209),
    ('CRF=18',  247.0,   -0.000875),
    ('CRF=23',  1213.7,  -0.001844),
    ('CRF=28',  2385.6,  -0.004067),
]


# ============================================================
# FIGURE 1: Pareto frontier
# ============================================================
def plot_pareto(data_list, cafe_list, title, filename, annotate_ours=True):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Track categories for legend
    cat_handles = {}

    for name, ratio, delta, marker, color, cat in data_list + cafe_list:
        s = 250 if 'Ours' in cat else 150 if 'CAFE' in cat else 80
        zorder = 10 if 'Ours' in cat else 8 if 'CAFE' in cat else 5
        alpha = 1.0 if 'Ours' in cat else 0.85
        h = ax.scatter(ratio, abs(delta) * 100, marker=marker, c=color, s=s,
                       edgecolors='black', linewidth=0.5, zorder=zorder, alpha=alpha)
        if cat not in cat_handles:
            cat_handles[cat] = h

    # Compute Pareto frontier
    all_pts = [(r, abs(d) * 100, n) for n, r, d, *_ in data_list + cafe_list]
    sorted_pts = sorted(all_pts, key=lambda x: x[0])
    pareto = []
    min_loss = float('inf')
    for r, l, n in sorted_pts:
        if l < min_loss:
            min_loss = l
            pareto.append((r, l, n))

    if len(pareto) > 1:
        pr, pl = zip(*[(p[0], p[1]) for p in pareto])
        ax.plot(pr, pl, 'k--', alpha=0.4, linewidth=2, label='Pareto frontier')

    # Annotations
    for name, ratio, delta, *_ in data_list + cafe_list:
        loss = abs(delta) * 100
        if 'freq' in name and 'CRF=18' in name:
            ax.annotate(f'{name}\n({ratio:.0f}x, {loss:.3f}%)',
                       (ratio, loss), textcoords="offset points",
                       xytext=(15, 15), fontsize=8, color=C_H265,
                       fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=C_H265, lw=1))
        elif 'CAFE' in name and '1000' in name:
            ax.annotate(name, (ratio, loss),
                       textcoords="offset points", xytext=(10, -15),
                       fontsize=8, color=C_CAFE,
                       arrowprops=dict(arrowstyle='->', color=C_CAFE, lw=0.8))
        elif 'Zstd' in name:
            ax.annotate(name, (ratio, loss),
                       textcoords="offset points", xytext=(-50, -15),
                       fontsize=8, color=C_ZSTD)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compression Ratio (vs fp32)', fontsize=13)
    ax.set_ylabel('|AUC Loss| (%)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # "Better" arrow
    ax.annotate('', xy=(0.95, 0.05), xycoords='axes fraction',
               xytext=(0.78, 0.22), textcoords='axes fraction',
               arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(0.87, 0.13, 'Better', transform=ax.transAxes,
           fontsize=11, color='green', ha='center', style='italic')

    # Legend
    order = ['Quantization', 'Product Quant.', 'SVD', 'Row Pruning',
             'Zstd-19', 'H.265 Lossless', 'H.265 Natural', 'H.265+Freq (Ours)', 'CAFE+']
    handles = [cat_handles[c] for c in order if c in cat_handles]
    labels = [c for c in order if c in cat_handles]
    ax.legend(handles, labels, loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ============================================================
# FIGURE 2: CRF vs AUC/Compression tradeoff
# ============================================================
def plot_crf_tradeoff(filename):
    crfs = [c[0] for c in kaggle_crf]
    ratios = [c[1] for c in kaggle_crf]
    deltas = [abs(c[2]) * 100 for c in kaggle_crf]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel('CRF Value', fontsize=13)
    ax1.set_ylabel('|AUC Loss| (%)', fontsize=13, color=C_H265)
    l1, = ax1.plot([int(c.split('=')[1]) for c in crfs], deltas,
                    'o-', color=C_H265, linewidth=2.5, markersize=10, label='AUC Loss')
    ax1.tick_params(axis='y', labelcolor=C_H265)
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Compression Ratio (vs fp32)', fontsize=13, color='#2196F3')
    l2, = ax2.plot([int(c.split('=')[1]) for c in crfs], ratios,
                    's--', color='#2196F3', linewidth=2.5, markersize=10, label='Compression Ratio')
    ax2.tick_params(axis='y', labelcolor='#2196F3')

    # Sweet spot
    ax1.axvline(x=18, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax1.annotate('Sweet spot: CRF=18\n1369x comp., 0.038% loss',
                xy=(18, deltas[2]), xytext=(22, deltas[2] * 3),
                fontsize=10, fontweight='bold', color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax1.set_title('Kaggle: CRF Quality-Compression Tradeoff\n(4x4 tiling + frequency sort)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend([l1, l2], ['AUC Loss', 'Compression Ratio'],
              loc='center left', fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_crf_tradeoff_terabyte(filename):
    crfs_str = [c[0] for c in terabyte_crf]
    crfs = [int(c.split('=')[1]) for c in crfs_str]
    ratios = [c[1] for c in terabyte_crf]
    deltas = [abs(c[2]) * 100 for c in terabyte_crf]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('CRF Value', fontsize=13)
    ax1.set_ylabel('|AUC Loss| (%)', fontsize=13, color=C_H265)
    l1, = ax1.plot(crfs, deltas, 'o-', color=C_H265, linewidth=2.5, markersize=10, label='AUC Loss')
    ax1.tick_params(axis='y', labelcolor=C_H265)
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Compression Ratio (vs fp32)', fontsize=13, color='#2196F3')
    l2, = ax2.plot(crfs, ratios, 's--', color='#2196F3', linewidth=2.5, markersize=10,
                    label='Compression Ratio')
    ax2.tick_params(axis='y', labelcolor='#2196F3')

    ax1.axvline(x=18, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax1.annotate(f'CRF=18\n247x, 0.088% loss',
                xy=(18, deltas[2]), xytext=(22, deltas[2] * 3),
                fontsize=10, fontweight='bold', color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax1.set_title('Terabyte: CRF Quality-Compression Tradeoff (D=64, flat layout)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend([l1, l2], ['AUC Loss', 'Compression Ratio'], loc='center left', fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ============================================================
# FIGURE 3: Memory breakdown
# ============================================================
def plot_memory(filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Kaggle ---
    ax = axes[0]
    configs = ['Baseline\n(fp32)', 'C++ fp32\n(Config B)', 'Hot/Cold\nuint8', 'H.265\nCRF=18']
    hot =     [2058, 2058, 88.5,  88.5]
    cold =    [0,    0,    492.3, 0.8]
    decoded = [0,    0,    0,     39.6]
    mapping = [0,    0,    134.6, 134.6]

    x = np.arange(len(configs))
    w = 0.55
    b1 = ax.bar(x, hot, w, color='#E91E63', label='Hot/Full Emb (fp32)')
    b2 = ax.bar(x, cold, w, bottom=hot, color='#2196F3', label='Cold (uint8/compressed)')
    b3 = ax.bar(x, decoded, w, bottom=[h+c for h,c in zip(hot,cold)],
                color='#4CAF50', label='Decoded Cache')
    b4 = ax.bar(x, mapping, w, bottom=[h+c+d for h,c,d in zip(hot,cold,decoded)],
                color='#FF9800', label='Mapping Tables')

    totals = [h+c+d+m for h,c,d,m in zip(hot,cold,decoded,mapping)]
    for i, t in enumerate(totals):
        ax.text(i, t + 30, f'{t:.0f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('Kaggle (D=16)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 2300)

    # --- Terabyte ---
    ax = axes[1]
    configs_tb = ['Baseline\n(fp32)', 'C++ fp32\n(Config B)', 'Hot/Cold\nuint8']
    # Terabyte: 5519 MB fp32 for large tables. 237 MB hot. 1320 MB cold uint8. 86+4=90 MB mapping.
    hot_tb     = [5520, 5520, 237.3]
    cold_tb    = [0,    0,    1320.5]
    decoded_tb = [0,    0,    0]
    mapping_tb = [0,    0,    90.2]

    x2 = np.arange(len(configs_tb))
    ax.bar(x2, hot_tb, w, color='#E91E63', label='Hot/Full Emb (fp32)')
    ax.bar(x2, cold_tb, w, bottom=hot_tb, color='#2196F3', label='Cold (uint8)')
    ax.bar(x2, mapping_tb, w,
           bottom=[h+c for h,c in zip(hot_tb, cold_tb)],
           color='#FF9800', label='Mapping Tables')

    totals_tb = [h+c+m for h,c,m in zip(hot_tb, cold_tb, mapping_tb)]
    for i, t in enumerate(totals_tb):
        ax.text(i, t + 50, f'{t:.0f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_title('Terabyte (D=64)', fontsize=13, fontweight='bold')
    ax.set_xticks(x2)
    ax.set_xticklabels(configs_tb, fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 6200)

    plt.suptitle('Runtime Memory Breakdown', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ============================================================
# FIGURE 4: Error steering
# ============================================================
def plot_error_steering(filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Table 2 (Kaggle, 10.1M rows, low entropy)
    buckets = ['Top 1%', '1-10%', '10-50%', '50-100%']
    t2_rand = [1.1564, 0.0365, 0.0146, 0.0261]
    t2_nat  = [0.7904, 0.0342, 0.0124, 0.0275]
    t2_freq = [0.5590, 0.0341, 0.0123, 0.0241]

    x = np.arange(len(buckets))
    w = 0.25
    ax = axes[0]
    ax.bar(x - w, t2_rand, w, label='Random', color='#F44336', alpha=0.85)
    ax.bar(x,     t2_nat,  w, label='Natural', color='#FF9800', alpha=0.85)
    ax.bar(x + w, t2_freq, w, label='Freq Sort (ours)', color='#4CAF50', alpha=0.85)
    ax.set_ylabel('Per-Row MSE (uint8)', fontsize=12)
    ax.set_title('Kaggle Table 2 (10.1M rows, D=16)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.annotate('2.1x less error', xy=(0 + w, 0.57), xytext=(1.2, 0.95),
               fontsize=10, color='#4CAF50', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

    # Table 20 (Kaggle, 7.0M rows)
    t20_rand = [1.2008, 0.0367, 0.0088, 0.0173]
    t20_nat  = [0.8042, 0.0357, 0.0069, 0.0186]
    t20_freq = [0.5831, 0.0346, 0.0069, 0.0155]

    ax = axes[1]
    ax.bar(x - w, t20_rand, w, label='Random', color='#F44336', alpha=0.85)
    ax.bar(x,     t20_nat,  w, label='Natural', color='#FF9800', alpha=0.85)
    ax.bar(x + w, t20_freq, w, label='Freq Sort (ours)', color='#4CAF50', alpha=0.85)
    ax.set_title('Kaggle Table 20 (7.0M rows, D=16)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.annotate('2.1x less error', xy=(0 + w, 0.60), xytext=(1.2, 0.95),
               fontsize=10, color='#4CAF50', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

    plt.suptitle('H.265 CRF=18: Error Distribution by Row Access Frequency',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ============================================================
# FIGURE 5: Speedup attribution
# ============================================================
def plot_speedup_attribution(filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Kaggle
    ax = axes[0]
    configs = ['Config A\nPyTorch fp32', 'Config B\nC++ fp32', 'Config C\nC++ hot/cold']
    emb =     [1.34, 0.23, 0.23]
    interact =[1.23, 0.57, 0.57]
    mlp =     [1.55, 1.62, 1.62]
    other =   [0.13, 0.06, 0.01]

    x = np.arange(len(configs))
    w = 0.5
    ax.bar(x, emb, w, color='#E91E63', label='Embedding Lookup')
    ax.bar(x, interact, w, bottom=emb, color='#2196F3', label='Interaction')
    ax.bar(x, mlp, w, bottom=[e+i for e,i in zip(emb,interact)], color='#4CAF50', label='MLP')
    ax.bar(x, other, w, bottom=[e+i+m for e,i,m in zip(emb,interact,mlp)],
           color='#FF9800', label='Other')

    totals = [e+i+m+o for e,i,m,o in zip(emb,interact,mlp,other)]
    for i, t in enumerate(totals):
        ax.text(i, t + 0.1, f'{t:.2f}ms', ha='center', fontsize=10, fontweight='bold')

    # Speedup arrows
    ax.annotate('', xy=(1, totals[1]+0.3), xycoords='data',
               xytext=(0, totals[0]+0.3), textcoords='data',
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.5, totals[0]+0.5, '1.71x\n(C++ opt)', ha='center', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(2, totals[2]+0.3), xycoords='data',
               xytext=(1, totals[1]+0.3), textcoords='data',
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(1.5, totals[1]+0.5, '1.02x\n(compr.)', ha='center', fontsize=9, color='gray')

    ax.set_ylabel('Batch Latency (ms)', fontsize=12)
    ax.set_title('Kaggle (D=16)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 5.5)

    # Terabyte
    ax = axes[1]
    configs_tb = ['Config A\nPyTorch fp32', 'Config B\nC++ fp32', 'Config C\nC++ hot/cold']
    # Terabyte (from Exp 1)
    batch_a, batch_b, batch_c = 5.82, 4.09, 4.32

    ax.bar([0], [batch_a], w, color='#E91E63')
    ax.bar([1], [batch_b], w, color='#2196F3')
    ax.bar([2], [batch_c], w, color='#4CAF50')

    for i, t in enumerate([batch_a, batch_b, batch_c]):
        ax.text(i, t + 0.1, f'{t:.2f}ms', ha='center', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(1, batch_b+0.3), xycoords='data',
               xytext=(0, batch_a+0.3), textcoords='data',
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.5, batch_a+0.5, '1.42x\n(C++ opt)', ha='center', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(2, batch_c+0.3), xycoords='data',
               xytext=(1, batch_b+0.3), textcoords='data',
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(1.5, batch_b+0.5, '0.95x\n(overhead)', ha='center', fontsize=9, color='red')

    ax.set_title('Terabyte (D=64)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_tb, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 7.5)

    plt.suptitle('Speedup Attribution: C++ Optimization vs Compression',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ============================================================
# FIGURE 6: Zero-out vs H.265 comparison
# ============================================================
def plot_zeroout_vs_h265(filename):
    fig, ax = plt.subplots(figsize=(9, 6))

    # Kaggle zero-out (Exp 3)
    zero_ratios = [2, 5, 10, 20, 100, 200, 1000]
    zero_loss_k = [0.0, 0.0012, 0.0081, 0.0306, 0.1813, 0.2957, 0.6942]  # % AUC

    # Terabyte zero-out
    zero_loss_t = [0.0, 0.0002, 0.0005, 0.0016, 0.0193, 0.0452, 0.1808]

    ax.plot(zero_ratios, zero_loss_k, 'o-', color='#F44336', linewidth=2, markersize=8,
            label='Zero-out (Kaggle)')
    ax.plot(zero_ratios, zero_loss_t, 's-', color='#FF9800', linewidth=2, markersize=8,
            label='Zero-out (Terabyte)')

    # H.265 CRF=18+freq points
    ax.scatter([1360], [0.037], marker='*', c=C_H265, s=300, zorder=10,
              edgecolors='black', linewidth=0.5, label='H.265 CRF=18+freq (Kaggle)')
    ax.scatter([68.6], [0.050], marker='*', c='#4CAF50', s=300, zorder=10,
              edgecolors='black', linewidth=0.5, label='H.265 CRF=18+freq (Terabyte)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compression Ratio', fontsize=13)
    ax.set_ylabel('|AUC Loss| (%)', fontsize=13)
    ax.set_title('Zero-Out vs H.265: Lossy Compression Preserves More Information',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='upper left')

    # Annotate H.265 advantage
    ax.annotate('18.8x less loss\nat higher compression',
               xy=(1360, 0.037), xytext=(300, 0.15),
               fontsize=10, color=C_H265, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=C_H265, lw=1.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Fig 1: Kaggle Pareto
    plot_pareto(kaggle_exp2, kaggle_cafe,
               "Kaggle: Compression Ratio vs AUC Loss (All Methods)",
               "results/paper_figures/kaggle_pareto.png")

    # Fig 2: Terabyte Pareto
    plot_pareto(terabyte_exp2, [],
               "Terabyte: Compression Ratio vs AUC Loss (All Methods)",
               "results/paper_figures/terabyte_pareto.png")

    # Fig 3: CRF tradeoff (Kaggle + Terabyte)
    plot_crf_tradeoff("results/paper_figures/kaggle_crf_tradeoff.png")
    plot_crf_tradeoff_terabyte("results/paper_figures/terabyte_crf_tradeoff.png")

    # Fig 4: Memory breakdown
    plot_memory("results/paper_figures/memory_breakdown.png")

    # Fig 5: Error steering
    plot_error_steering("results/paper_figures/error_steering.png")

    # Fig 6: Speedup attribution
    plot_speedup_attribution("results/paper_figures/speedup_attribution.png")

    # Fig 7: Zero-out vs H.265
    plot_zeroout_vs_h265("results/paper_figures/zeroout_vs_h265.png")

    print("\nAll 7 figures saved to results/paper_figures/")
