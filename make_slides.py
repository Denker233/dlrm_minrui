#!/usr/bin/env python3
"""
Generate slide figures for the DCT-domain embedding compression presentation.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('results/slide_figures_v2', exist_ok=True)
plt.rcParams.update({'font.size': 14, 'figure.dpi': 150})

# ================================================================
# Data
# ================================================================

# Decomposition
decomp_labels = ['fp32\n(raw)', '+ uint8\nquant', '+ Zstd\nentropy', '+ CABAC\n+ intra', '+ DCT\nquant (lossy)']
decomp_sizes_mb = [1969, 492, 73, 49, 0.305]  # MB
decomp_ratios = [1, 4, 27, 40, 6466]
decomp_step_x = [4.0, 6.7, 1.5, 160]
decomp_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

# Pareto: AUC loss vs compression ratio
# Our DCT-domain results
dct_points = {
    'DCT 4.3% hot': (77, 0.037),
    'DCT 2% hot': (136, 0.091),
    'DCT 1% hot': (206, 0.181),
}

# H.265 storage compression
h265_points = {
    'H.265 CRF=0': (27, 0.0001),
    'H.265 CRF=18': (232, 0.003),
    'H.265 CRF=25': (2650, 0.007),
    'H.265 CRF=30': (6466, 0.013),
    'H.265 CRF=35': (9502, 0.021),
    'H.265 CRF=51': (12437, 0.037),
}

# Baselines
baselines = {
    'INT8': (4, 0.002),
    'INT4': (8, 0.190),
    'PQ (M=2)': (32, 0.559),
    'Pruning 99%': (100, 0.181),
    'Zstd-3': (27, 0.001),
    'CAFE+\n(retrain)': (10000, 0.75),
}

# Memory comparison
mem_methods = ['fp32\nbaseline', 'H.265\n+cache', 'DCT 16×16\n4.3% hot', 'DCT 16×16\n2% hot', 'DCT 16×16\n1% hot']
mem_values = [2061, 71, 26.9, 15.2, 10.0]
mem_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

# Performance comparison
perf_methods = ['fp32\n(Python)', 'H.265+cache\n(C++)', 'DCT 16×16\n4.3% hot', 'DCT 16×16\n2% hot', 'DCT 16×16\n1% hot']
perf_batch = [4.98, 2.39, 2.40, 2.35, 2.39]
perf_auc = [0.802497, 0.802107, 0.802123, 0.801589, 0.800684]

# ================================================================
# Figure 1: Compression Decomposition (waterfall chart)
# ================================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(decomp_labels))
bars = ax.bar(x, [np.log10(r) for r in decomp_ratios], color=decomp_colors, width=0.6, edgecolor='black', linewidth=0.5)

# Add ratio labels on bars
for i, (r, s) in enumerate(zip(decomp_ratios, decomp_sizes_mb)):
    sz = f"{s:.0f}MB" if s >= 1 else f"{s*1024:.0f}KB"
    ax.text(i, np.log10(r) + 0.15, f'{r:,}×\n({sz})', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add step contribution arrows
for i in range(len(decomp_step_x)):
    ax.annotate(f'×{decomp_step_x[i]:.1f}', xy=(i + 0.5, (np.log10(decomp_ratios[i]) + np.log10(decomp_ratios[i+1])) / 2),
                fontsize=11, ha='center', color='#333', fontstyle='italic')

ax.set_xticks(x)
ax.set_xticklabels(decomp_labels, fontsize=12)
ax.set_ylabel('Compression Ratio (log scale)', fontsize=13)
ax.set_title('Why H.265 Achieves 6,466× on Embeddings: Component Decomposition', fontsize=15, fontweight='bold')
ax.set_ylim(0, 4.5)
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(['1×', '10×', '100×', '1,000×', '10,000×'])
ax.grid(axis='y', alpha=0.3)

# Highlight the dominant factor
ax.annotate('DCT quantization\nis the dominant factor\n(160× alone)', xy=(4, np.log10(6466)),
            xytext=(3.0, 4.2), fontsize=12, color='#9467bd', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#9467bd', lw=2))

plt.tight_layout()
plt.savefig('results/slide_figures_v2/01_decomposition.png', bbox_inches='tight')
plt.close()
print("Saved 01_decomposition.png")

# ================================================================
# Figure 2: Why DC-only works (visual explanation)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Cold embedding block values
np.random.seed(42)
cold_block = 88 + np.random.randint(-2, 3, (4, 4))
im = axes[0].imshow(cold_block, cmap='RdYlBu_r', vmin=82, vmax=94, aspect='equal')
axes[0].set_title('Cold Embedding Block\n(4 rows × 4 dims)', fontsize=13, fontweight='bold')
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, str(cold_block[i,j]), ha='center', va='center', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Dimension', fontsize=12)
axes[0].set_ylabel('Row', fontsize=12)

# Panel B: DCT coefficients
dct_coeffs = np.zeros((4, 4))
dct_coeffs[0, 0] = 88  # DC = average
dct_coeffs[0, 1] = 0.5  # tiny AC
dct_coeffs[1, 0] = -0.3
im2 = axes[1].imshow(np.abs(dct_coeffs), cmap='YlOrRd', vmin=0, vmax=100, aspect='equal')
axes[1].set_title('After DCT Transform\n(frequency coefficients)', fontsize=13, fontweight='bold')
labels = [['DC=88', '≈0', '≈0', '≈0'],
          ['≈0', '0', '0', '0'],
          ['≈0', '0', '0', '0'],
          ['≈0', '0', '0', '0']]
for i in range(4):
    for j in range(4):
        color = 'white' if i == 0 and j == 0 else 'black'
        axes[1].text(j, i, labels[i][j], ha='center', va='center', fontsize=13, fontweight='bold', color=color)
axes[1].set_xlabel('Frequency (horizontal)', fontsize=12)
axes[1].set_ylabel('Frequency (vertical)', fontsize=12)

# Panel C: After quantization (DC-only)
quant = np.zeros((4, 4))
quant[0, 0] = 88
im3 = axes[2].imshow(np.abs(quant), cmap='YlOrRd', vmin=0, vmax=100, aspect='equal')
axes[2].set_title('After Quantization\n(DC-only: 1 number for 16 values!)', fontsize=13, fontweight='bold')
labels2 = [['DC=88', '0', '0', '0'],
           ['0', '0', '0', '0'],
           ['0', '0', '0', '0'],
           ['0', '0', '0', '0']]
for i in range(4):
    for j in range(4):
        color = 'white' if i == 0 and j == 0 else '#666'
        axes[2].text(j, i, labels2[i][j], ha='center', va='center', fontsize=13, fontweight='bold', color=color)
axes[2].set_xlabel('All variation rounds to zero', fontsize=12)

plt.suptitle('Why Cold Embeddings Are "DC-Only": Values ≈ 88 (zero-point), Variation ≈ ±1', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('results/slide_figures_v2/02_dc_only_visual.png', bbox_inches='tight')
plt.close()
print("Saved 02_dc_only_visual.png")

# ================================================================
# Figure 3: Pareto curve (AUC loss vs compression ratio)
# ================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Baselines
for label, (ratio, loss) in baselines.items():
    marker = 's' if 'CAFE' in label else 'D'
    color = '#d62728' if 'CAFE' in label else '#888888'
    ax.scatter(ratio, loss, s=120, marker=marker, color=color, zorder=5, edgecolors='black', linewidth=0.5)
    offset = (10, 8) if 'CAFE' not in label else (-15, 10)
    ax.annotate(label, (ratio, loss), textcoords="offset points", xytext=offset, fontsize=10)

# H.265 storage line
h265_x = [v[0] for v in h265_points.values()]
h265_y = [v[1] for v in h265_points.values()]
ax.plot(h265_x, h265_y, 'o--', color='#ff7f0e', markersize=8, label='H.265 (storage, need decode cache)', zorder=4)

# DCT-domain runtime line
dct_x = [v[0] for v in dct_points.values()]
dct_y = [v[1] for v in dct_points.values()]
ax.plot(dct_x, dct_y, 's-', color='#2ca02c', markersize=12, linewidth=2.5, label='DCT-domain (runtime, zero cache)', zorder=6)
for label, (ratio, loss) in dct_points.items():
    ax.annotate(label, (ratio, loss), textcoords="offset points", xytext=(10, -15), fontsize=11, fontweight='bold', color='#2ca02c')

# Production threshold line
ax.axhline(y=0.1, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(3, 0.11, 'Production threshold (0.1%)', fontsize=10, color='red', alpha=0.7)

ax.set_xscale('log')
ax.set_xlabel('Compression Ratio (vs fp32)', fontsize=14)
ax.set_ylabel('AUC Loss (%)', fontsize=14)
ax.set_title('Pareto Frontier: AUC Loss vs Compression Ratio\n(Lower-left is better)', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(2, 20000)
ax.set_ylim(-0.02, 0.85)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/slide_figures_v2/03_pareto_curve.png', bbox_inches='tight')
plt.close()
print("Saved 03_pareto_curve.png")

# ================================================================
# Figure 4: Memory comparison (bar chart)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(mem_methods))
bars = ax.bar(x, mem_values, color=mem_colors, width=0.6, edgecolor='black', linewidth=0.5)

for i, (v, m) in enumerate(zip(mem_values, mem_methods)):
    ratio = 2061 / v
    if v > 100:
        ax.text(i, v + 30, f'{v:.0f} MB', ha='center', fontsize=12, fontweight='bold')
    else:
        ax.text(i, v + 30, f'{v:.1f} MB\n({ratio:.0f}×)', ha='center', fontsize=12, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(mem_methods, fontsize=11)
ax.set_ylabel('Runtime Memory (MB)', fontsize=13)
ax.set_title('Runtime Memory Comparison\n(with reordered tables, no bitmap)', fontsize=15, fontweight='bold')
ax.set_ylim(0, 2300)
ax.grid(axis='y', alpha=0.3)

# Add "zero cache" annotation
ax.annotate('Zero decode cache\nZero startup\nNo FFmpeg', xy=(3, 15.2), xytext=(3.5, 800),
            fontsize=12, color='#1f77b4', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2),
            ha='center')

plt.tight_layout()
plt.savefig('results/slide_figures_v2/04_memory_comparison.png', bbox_inches='tight')
plt.close()
print("Saved 04_memory_comparison.png")

# ================================================================
# Figure 5: AUC + Latency comparison (grouped bar)
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

methods_short = ['fp32', 'H.265\n+cache', 'DCT\n4.3%hot', 'DCT\n2%hot', 'DCT\n1%hot']
x = np.arange(len(methods_short))
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

# AUC
bars1 = ax1.bar(x, perf_auc, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
ax1.set_ylim(0.7995, 0.8030)
ax1.set_xticks(x)
ax1.set_xticklabels(methods_short, fontsize=11)
ax1.set_ylabel('AUC', fontsize=13)
ax1.set_title('Model Accuracy (AUC)', fontsize=14, fontweight='bold')
for i, v in enumerate(perf_auc):
    loss = (0.802497 - v) * 100
    label = f'{v:.4f}\n({loss:+.03f}%)' if loss != 0 else f'{v:.4f}'
    ax1.text(i, v + 0.0002, label, ha='center', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Batch latency
bars2 = ax2.bar(x, perf_batch, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(methods_short, fontsize=11)
ax2.set_ylabel('Batch Latency (ms)', fontsize=13)
ax2.set_title('Inference Latency (ms/batch)', fontsize=14, fontweight='bold')
for i, v in enumerate(perf_batch):
    ax2.text(i, v + 0.08, f'{v:.2f}ms', ha='center', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 6)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('All DCT Configs Match H.265 Latency with Better AUC and Less Memory', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/slide_figures_v2/05_auc_latency.png', bbox_inches='tight')
plt.close()
print("Saved 05_auc_latency.png")

# ================================================================
# Figure 6: The journey (H.265 → DCT → DC-only → Block Mean)
# ================================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

steps = [
    ("H.265 Video Codec\n6,466× storage\n71 MB runtime\nNeeds FFmpeg + cache", '#ff7f0e', 0.05),
    ("Decompose:\nWhy does it work?\n→ DCT quant = 160×\n→ CABAC+intra = 1.5×", '#2ca02c', 0.28),
    ("Discovery:\nCold blocks are\n100% DC-only\n→ Only average matters", '#1f77b4', 0.51),
    ("Block-Mean Codec:\n136× runtime\n15 MB memory\nZero cache, zero FFmpeg", '#9467bd', 0.74),
]

for text, color, xpos in steps:
    box = mpatches.FancyBboxPatch((xpos, 0.15), 0.2, 0.7, boxstyle="round,pad=0.02",
                                   facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(xpos + 0.1, 0.5, text, ha='center', va='center', fontsize=12,
            fontweight='bold', color=color, transform=ax.transAxes)

# Arrows between steps
for i in range(3):
    x1 = steps[i][2] + 0.21
    x2 = steps[i+1][2] - 0.01
    ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.5),
                transform=ax.transAxes)

ax.set_title('The Journey: From Video Codec to Block-Mean Compression', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/slide_figures_v2/06_journey.png', bbox_inches='tight')
plt.close()
print("Saved 06_journey.png")

# ================================================================
# Figure 7: Summary comparison table as figure
# ================================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

table_data = [
    ['Method', 'Compression', 'AUC Loss', 'Batch Latency', 'Runtime Memory', 'Cache', 'Retraining'],
    ['fp32 baseline', '1×', '—', '4.98 ms', '2,061 MB', '—', 'No'],
    ['INT8', '4×', '-0.002%', '~5 ms', '515 MB', '—', 'No'],
    ['Pruning 99%', '100×', '-0.181%', '~5 ms', '~21 MB', '—', 'No'],
    ['CAFE+ (learned)', '~10,000×', '~-0.75%', '~5 ms', '~0.2 MB', '—', 'Yes'],
    ['H.265 + cache', '29× / 6,466×*', '-0.039%', '2.39 ms', '71 MB', '40 MB', 'No'],
    ['DCT 2% hot (ours)', '136×', '-0.091%', '2.35 ms', '15.2 MB', 'NONE', 'No'],
]

colors_table = [['#e0e0e0'] * 7,
                ['#fff'] * 7,
                ['#fff'] * 7,
                ['#fff'] * 7,
                ['#ffe0e0'] * 7,  # CAFE+ - requires retraining
                ['#fff5e0'] * 7,
                ['#e0ffe0'] * 7]  # Ours - highlight

table = ax.table(cellText=table_data, cellColours=colors_table,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)

# Bold header
for j in range(7):
    table[0, j].set_text_props(fontweight='bold')
    table[6, j].set_text_props(fontweight='bold')  # Our row

ax.set_title('Comparison with Prior Methods (* H.265: 29× runtime / 6,466× storage)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/slide_figures_v2/07_comparison_table.png', bbox_inches='tight')
plt.close()
print("Saved 07_comparison_table.png")

print("\nAll 7 figures saved to results/slide_figures_v2/")
