#!/usr/bin/env python3
"""Figures showing the benefit of frequency-based reordering."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results/interesting_figures", exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'figure.facecolor': 'white',
})

# ==========================================================================
# DATA: CRF × Ordering (from intrinsic_compressibility.md, Experiment 1b)
# ==========================================================================

# Kaggle AUC deltas (absolute values, in %)
crfs = [0, 10, 18, 23, 28]

kaggle_auc_random  = [0.0016, 0.0406, 0.1446, 0.4071, 0.9444]
kaggle_auc_natural = [0.0016, 0.0250, 0.0700, 0.1676, 0.3645]
kaggle_auc_freq    = [0.0016, 0.0126, 0.0372, 0.0796, 0.1758]

terabyte_auc_random  = [0.0006, 0.0296, 0.1258, 0.2544, 0.5057]
terabyte_auc_natural = [0.0006, 0.0235, 0.0934, 0.1781, 0.3371]
terabyte_auc_freq    = [0.0006, 0.0152, 0.0503, 0.1003, 0.2142]

# Kaggle Table 2 compression ratios (from Experiment 1)
kaggle_ratio_natural = [46.4, 199.4, 466.8, 602.5, 672.0]
kaggle_ratio_random  = [44.5, 182.8, 473.5, 623.1, 683.4]
kaggle_ratio_freq    = [65.6, 213.7, 511.8, 613.5, 665.3]

# Terabyte Table 0 compression ratios
terabyte_ratio_natural = [5.0, 8.2, 352.4, 674.6, 698.4]
terabyte_ratio_random  = [5.0, 8.2, 348.6, 675.8, 700.4]
terabyte_ratio_freq    = [5.0, 8.3, 332.1, 681.4, 698.5]

# MSE by frequency bucket (Kaggle, CRF=18)
buckets = ['Top 1%', '1-10%', '10-50%', '50-100%']
t2_rand = [1.156, 0.037, 0.015, 0.026]
t2_nat  = [0.790, 0.034, 0.012, 0.028]
t2_freq = [0.559, 0.034, 0.012, 0.024]

C_RAND = '#F44336'
C_NAT  = '#FF9800'
C_FREQ = '#4CAF50'


# ==========================================================================
# FIGURE 1: The main story — AUC loss across CRF for 3 orderings
# ==========================================================================
def fig_auc_vs_ordering():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Kaggle
    ax = axes[0]
    ax.plot(crfs, kaggle_auc_random, 'o--', color=C_RAND, linewidth=2, markersize=9,
            label='Random order')
    ax.plot(crfs, kaggle_auc_natural, 's--', color=C_NAT, linewidth=2, markersize=9,
            label='Natural order')
    ax.plot(crfs, kaggle_auc_freq, '*-', color=C_FREQ, linewidth=2.5, markersize=14,
            label='Frequency sort (ours)')

    ax.set_yscale('log')
    ax.set_xlabel('CRF Value', fontsize=13)
    ax.set_ylabel('|AUC Loss| (%)', fontsize=13)
    ax.set_title('Kaggle (D=16)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    # Annotate the gap at CRF=18
    ax.annotate('', xy=(18, kaggle_auc_freq[2]), xycoords='data',
                xytext=(18, kaggle_auc_random[2]), textcoords='data',
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(19.5, 0.07, '3.9x\nbetter', fontsize=12, fontweight='bold',
            color='black', va='center')

    # Annotate CRF=28
    ax.annotate('', xy=(28, kaggle_auc_freq[4]), xycoords='data',
                xytext=(28, kaggle_auc_random[4]), textcoords='data',
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(26, 0.45, '5.4x', fontsize=11, fontweight='bold', color='black')

    # Terabyte
    ax = axes[1]
    ax.plot(crfs, terabyte_auc_random, 'o--', color=C_RAND, linewidth=2, markersize=9,
            label='Random order')
    ax.plot(crfs, terabyte_auc_natural, 's--', color=C_NAT, linewidth=2, markersize=9,
            label='Natural order')
    ax.plot(crfs, terabyte_auc_freq, '*-', color=C_FREQ, linewidth=2.5, markersize=14,
            label='Frequency sort (ours)')

    ax.set_yscale('log')
    ax.set_xlabel('CRF Value', fontsize=13)
    ax.set_ylabel('|AUC Loss| (%)', fontsize=13)
    ax.set_title('Terabyte (D=64)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    # Annotate at CRF=18
    ax.annotate('', xy=(18, terabyte_auc_freq[2]), xycoords='data',
                xytext=(18, terabyte_auc_random[2]), textcoords='data',
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(19.5, 0.07, '2.5x\nbetter', fontsize=12, fontweight='bold',
            color='black', va='center')

    plt.suptitle('Frequency Sorting Dramatically Reduces AUC Loss\n(same data, same compression)',
                 fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('results/interesting_figures/reorder_auc_benefit.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: reorder_auc_benefit.png")


# ==========================================================================
# FIGURE 2: Compression ratio is barely affected — the contrast
# ==========================================================================
def fig_ratio_vs_ordering():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Kaggle Table 2
    ax = axes[0]
    ax.plot(crfs, kaggle_ratio_random, 'o--', color=C_RAND, linewidth=2, markersize=9,
            label='Random order')
    ax.plot(crfs, kaggle_ratio_natural, 's--', color=C_NAT, linewidth=2, markersize=9,
            label='Natural order')
    ax.plot(crfs, kaggle_ratio_freq, '*-', color=C_FREQ, linewidth=2.5, markersize=14,
            label='Frequency sort')

    ax.set_xlabel('CRF Value', fontsize=13)
    ax.set_ylabel('Compression Ratio', fontsize=13)
    ax.set_title('Kaggle Table 2 (10.1M rows)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Shade the "within 10%" band
    ratio_mean = [(a+b+c)/3 for a,b,c in zip(kaggle_ratio_random, kaggle_ratio_natural, kaggle_ratio_freq)]
    ax.fill_between(crfs, [r*0.9 for r in ratio_mean], [r*1.1 for r in ratio_mean],
                    alpha=0.1, color='gray')
    ax.text(14, 550, 'All within\n~10%', fontsize=12, fontweight='bold',
            color='gray', ha='center', style='italic')

    # Terabyte Table 0
    ax = axes[1]
    ax.plot(crfs, terabyte_ratio_random, 'o--', color=C_RAND, linewidth=2, markersize=9,
            label='Random order')
    ax.plot(crfs, terabyte_ratio_natural, 's--', color=C_NAT, linewidth=2, markersize=9,
            label='Natural order')
    ax.plot(crfs, terabyte_ratio_freq, '*-', color=C_FREQ, linewidth=2.5, markersize=14,
            label='Frequency sort')

    ax.set_xlabel('CRF Value', fontsize=13)
    ax.set_ylabel('Compression Ratio', fontsize=13)
    ax.set_title('Terabyte Table 0 (5.4M rows)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ratio_mean_tb = [(a+b+c)/3 for a,b,c in zip(terabyte_ratio_random, terabyte_ratio_natural, terabyte_ratio_freq)]
    ax.fill_between(crfs, [r*0.9 for r in ratio_mean_tb], [r*1.1 for r in ratio_mean_tb],
                    alpha=0.1, color='gray')
    ax.text(14, 400, 'All within\n~6%', fontsize=12, fontweight='bold',
            color='gray', ha='center', style='italic')

    plt.suptitle('Compression Ratio is Nearly Identical Across Orderings\n(reordering does NOT improve compression)',
                 fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('results/interesting_figures/reorder_ratio_noeffect.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: reorder_ratio_noeffect.png")


# ==========================================================================
# FIGURE 3: Combined "same ratio, less loss" — the punchline
# ==========================================================================
def fig_reorder_punchline():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Kaggle AUC loss at CRF=18 — bar chart
    ax = axes[0]
    orderings = ['Random', 'Natural', 'Freq Sort\n(ours)']
    auc_losses = [kaggle_auc_random[2], kaggle_auc_natural[2], kaggle_auc_freq[2]]
    colors = [C_RAND, C_NAT, C_FREQ]
    bars = ax.bar(orderings, auc_losses, color=colors, edgecolor='white', linewidth=2, width=0.6)

    for bar, val in zip(bars, auc_losses):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f'{val:.3f}%', ha='center', fontsize=12, fontweight='bold')

    # Add ratio annotations
    ax.annotate(f'{auc_losses[0]/auc_losses[2]:.1f}x worse',
                xy=(0, auc_losses[0]), xytext=(0.7, auc_losses[0] + 0.02),
                fontsize=11, color=C_RAND,
                arrowprops=dict(arrowstyle='->', color=C_RAND, lw=1.5))

    ax.set_ylabel('|AUC Loss| (%)', fontsize=13)
    ax.set_title('Kaggle CRF=18: AUC Loss', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Compression ratio at CRF=18 — bar chart (showing they're the same)
    ax = axes[1]
    ratios = [kaggle_ratio_random[2], kaggle_ratio_natural[2], kaggle_ratio_freq[2]]
    bars = ax.bar(orderings, ratios, color=colors, edgecolor='white', linewidth=2, width=0.6)

    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, val + 10,
                f'{val:.0f}x', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('Compression Ratio', fontsize=13)
    ax.set_title('Kaggle CRF=18: Compression Ratio', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 600)

    # Add "~same" annotation
    mean_r = np.mean(ratios)
    ax.axhline(y=mean_r, color='gray', linestyle='--', alpha=0.5)
    ax.text(2.4, mean_r + 15, '~same', fontsize=12, fontweight='bold',
            color='gray', style='italic')

    # Panel 3: MSE by frequency bucket — where the magic happens
    ax = axes[2]
    x = np.arange(len(buckets))
    w = 0.25
    ax.bar(x - w, t2_rand, w, label='Random', color=C_RAND, alpha=0.85)
    ax.bar(x,     t2_nat,  w, label='Natural', color=C_NAT, alpha=0.85)
    ax.bar(x + w, t2_freq, w, label='Freq Sort', color=C_FREQ, alpha=0.85)

    ax.set_ylabel('Per-Row MSE (uint8)', fontsize=12)
    ax.set_title('Where Error Goes (Table 2)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate the key: top 1% error drops
    ax.annotate(f'2.1x less\nerror here', xy=(0 + w, 0.57), xytext=(1.2, 0.9),
                fontsize=11, color=C_FREQ, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_FREQ, lw=1.5))

    ax.annotate('Tail rows:\nsame error', xy=(3, 0.028), xytext=(2.5, 0.5),
                fontsize=10, color='gray', style='italic',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    plt.suptitle('The Reordering Punchline: Same Compression, Much Better Accuracy',
                 fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('results/interesting_figures/reorder_punchline.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: reorder_punchline.png")


# ==========================================================================
# FIGURE 4: AUC advantage grows with CRF (the scaling story)
# ==========================================================================
def fig_advantage_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Compute advantage ratios
    kaggle_advantage_vs_random = [r/f if f > 0 else 1 for r, f in zip(kaggle_auc_random, kaggle_auc_freq)]
    kaggle_advantage_vs_natural = [n/f if f > 0 else 1 for n, f in zip(kaggle_auc_natural, kaggle_auc_freq)]
    terabyte_advantage_vs_random = [r/f if f > 0 else 1 for r, f in zip(terabyte_auc_random, terabyte_auc_freq)]
    terabyte_advantage_vs_natural = [n/f if f > 0 else 1 for n, f in zip(terabyte_auc_natural, terabyte_auc_freq)]

    # Skip CRF=0 (lossless, all orderings are same)
    crfs_lossy = crfs[1:]

    ax = axes[0]
    ax.plot(crfs_lossy, kaggle_advantage_vs_random[1:], 'o-', color=C_RAND, linewidth=2.5,
            markersize=10, label='vs Random')
    ax.plot(crfs_lossy, kaggle_advantage_vs_natural[1:], 's-', color=C_NAT, linewidth=2.5,
            markersize=10, label='vs Natural')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('CRF Value', fontsize=13)
    ax.set_ylabel('Advantage (x times less AUC loss)', fontsize=13)
    ax.set_title('Kaggle (D=16)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate peak
    peak_k = max(kaggle_advantage_vs_random[1:])
    ax.annotate(f'{peak_k:.1f}x', xy=(28, peak_k), xytext=(25, peak_k + 0.5),
                fontsize=13, fontweight='bold', color=C_RAND)

    ax = axes[1]
    ax.plot(crfs_lossy, terabyte_advantage_vs_random[1:], 'o-', color=C_RAND, linewidth=2.5,
            markersize=10, label='vs Random')
    ax.plot(crfs_lossy, terabyte_advantage_vs_natural[1:], 's-', color=C_NAT, linewidth=2.5,
            markersize=10, label='vs Natural')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('CRF Value', fontsize=13)
    ax.set_ylabel('Advantage (x times less AUC loss)', fontsize=13)
    ax.set_title('Terabyte (D=64)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    peak_t = max(terabyte_advantage_vs_random[1:])
    ax.annotate(f'{peak_t:.1f}x', xy=(28, peak_t), xytext=(25, peak_t + 0.2),
                fontsize=13, fontweight='bold', color=C_RAND)

    plt.suptitle('Freq Sort Advantage Grows with Lossy Compression\n(more aggressive CRF → bigger benefit)',
                 fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('results/interesting_figures/reorder_advantage_scaling.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: reorder_advantage_scaling.png")


# ==========================================================================
# FIGURE 5: Pareto — same ratio axis, different AUC for each ordering
# ==========================================================================
def fig_reorder_pareto():
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each CRF point for each ordering (Kaggle full-table)
    # Use Exp 2 numbers for the aggregate ratios
    # Random
    ax.plot(kaggle_ratio_random, kaggle_auc_random, 'o--', color=C_RAND,
            linewidth=2, markersize=10, label='Random order', zorder=5)
    # Natural
    ax.plot(kaggle_ratio_natural, kaggle_auc_natural, 's--', color=C_NAT,
            linewidth=2, markersize=10, label='Natural order', zorder=5)
    # Freq sort
    ax.plot(kaggle_ratio_freq, kaggle_auc_freq, '*-', color=C_FREQ,
            linewidth=2.5, markersize=16, label='Frequency sort (ours)', zorder=6)

    # Label CRF values on freq line
    for i, crf in enumerate(crfs):
        if crf > 0:
            ax.annotate(f'CRF={crf}', (kaggle_ratio_freq[i], kaggle_auc_freq[i]),
                       textcoords='offset points', xytext=(12, -5),
                       fontsize=9, color=C_FREQ)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compression Ratio (Table 2)', fontsize=14)
    ax.set_ylabel('|AUC Loss| (%)', fontsize=14)
    ax.set_title('Kaggle: Reordering Shifts the Pareto Curve Down\n(same x-axis, lower y-axis = better)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    # "Better" arrow
    ax.annotate('', xy=(0.95, 0.05), xycoords='axes fraction',
               xytext=(0.80, 0.20), textcoords='axes fraction',
               arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(0.88, 0.12, 'Better', transform=ax.transAxes,
           fontsize=12, color='green', ha='center', style='italic')

    # Annotate the vertical gap at CRF=18
    r18 = kaggle_ratio_freq[2]
    ax.annotate('', xy=(r18, kaggle_auc_freq[2]), xycoords='data',
                xytext=(r18*0.95, kaggle_auc_random[2]), textcoords='data',
                arrowprops=dict(arrowstyle='<->', color='black', lw=2, linestyle='--'))
    ax.text(r18*1.3, 0.065, '3.9x less\nloss at\nsame ratio',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig('results/interesting_figures/reorder_pareto_shift.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: reorder_pareto_shift.png")


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    fig_auc_vs_ordering()
    fig_ratio_vs_ordering()
    fig_reorder_punchline()
    fig_advantage_scaling()
    fig_reorder_pareto()
    print("\nAll 5 reordering figures saved to results/interesting_figures/")
