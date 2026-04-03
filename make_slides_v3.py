#!/usr/bin/env python3
"""
Generate slide figures v3: "DC" labels, plus DCT vs DC-only explanation figure.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('results/slide_figures_v3', exist_ok=True)
plt.rcParams.update({'font.size': 14, 'figure.dpi': 150})

# ================================================================
# Data
# ================================================================
decomp_labels = ['fp32\n(raw)', '+ uint8\nquant', '+ Zstd\nentropy', '+ CABAC\n+ intra', '+ DCT\nquant (lossy)']
decomp_ratios = [1, 4, 27, 40, 6466]
decomp_sizes_mb = [1969, 492, 73, 49, 0.305]
decomp_step_x = [4.0, 6.7, 1.5, 160]
decomp_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

dct_points = {'DC 4.3% hot': (77, 0.037), 'DC 2% hot': (136, 0.091), 'DC 1% hot': (206, 0.181)}
h265_points = {'H.265 CRF=0': (27, 0.0001), 'H.265 CRF=18': (232, 0.003), 'H.265 CRF=25': (2650, 0.007),
               'H.265 CRF=30': (6466, 0.013), 'H.265 CRF=35': (9502, 0.021), 'H.265 CRF=51': (12437, 0.037)}
baselines = {'INT8': (4, 0.002), 'INT4': (8, 0.190), 'PQ (M=2)': (32, 0.559),
             'Pruning 99%': (100, 0.181), 'Zstd-3': (27, 0.001), 'CAFE+\n(retrain)': (10000, 0.75)}

perf_methods = ['fp32\n(Python)', 'H.265+cache\n(C++)', 'DC\n4.3% hot', 'DC\n2% hot', 'DC\n1% hot']
perf_batch = [4.98, 2.39, 2.40, 2.35, 2.39]
perf_auc = [0.802497, 0.802107, 0.802123, 0.801589, 0.800684]
mem_methods = ['fp32\nbaseline', 'H.265\n+cache', 'DC\n4.3% hot', 'DC\n2% hot', 'DC\n1% hot']
mem_values = [2061, 71, 26.9, 15.2, 10.0]
colors5 = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']


# ================================================================
# Figure 1: Decomposition (same as before)
# ================================================================
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(decomp_labels))
ax.bar(x, [np.log10(r) for r in decomp_ratios], color=decomp_colors, width=0.6, edgecolor='black', linewidth=0.5)
for i, (r, s) in enumerate(zip(decomp_ratios, decomp_sizes_mb)):
    sz = f"{s:.0f}MB" if s >= 1 else f"{s*1024:.0f}KB"
    ax.text(i, np.log10(r) + 0.15, f'{r:,}×\n({sz})', ha='center', fontsize=12, fontweight='bold')
for i in range(len(decomp_step_x)):
    ax.annotate(f'×{decomp_step_x[i]:.1f}', xy=(i+0.5, (np.log10(decomp_ratios[i])+np.log10(decomp_ratios[i+1]))/2),
                fontsize=11, ha='center', color='#333', fontstyle='italic')
ax.set_xticks(x); ax.set_xticklabels(decomp_labels, fontsize=12)
ax.set_ylabel('Compression Ratio (log scale)', fontsize=13)
ax.set_title('Why H.265 Achieves 6,466× on Embeddings: Component Decomposition', fontsize=15, fontweight='bold')
ax.set_ylim(0, 4.5); ax.set_yticks([0,1,2,3,4]); ax.set_yticklabels(['1×','10×','100×','1,000×','10,000×'])
ax.grid(axis='y', alpha=0.3)
ax.annotate('DCT quantization\nis the dominant factor\n(160× alone)', xy=(4, np.log10(6466)),
            xytext=(3.0, 4.2), fontsize=12, color='#9467bd', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#9467bd', lw=2))
plt.tight_layout(); plt.savefig('results/slide_figures_v3/01_decomposition.png', bbox_inches='tight'); plt.close()
print("Saved 01_decomposition.png")


# ================================================================
# Figure 2: DCT vs DC-only explanation (NEW — the key explanation slide)
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# --- Top row: FULL DCT (what H.265 does) ---
np.random.seed(42)

# Top-left: Original block (high entropy — hot-like data)
hot_block = np.array([[127,131,129,137,124,127,136,122],
                       [125,127,124,128,126,125,129,127],
                       [130,133,128,135,129,130,134,126],
                       [122,124,135,133,128,124,135,118],
                       [128,130,127,131,129,128,132,130],
                       [125,127,124,128,126,125,129,127],
                       [130,133,128,140,129,130,138,126],
                       [122,124,140,133,128,124,135,118]], dtype=float)

im = axes[0,0].imshow(hot_block, cmap='RdYlBu_r', vmin=115, vmax=145, aspect='equal')
axes[0,0].set_title('High-Entropy Block\n(e.g., hot rows)', fontsize=13, fontweight='bold')
axes[0,0].set_ylabel('Full DCT\n(H.265 approach)', fontsize=13, fontweight='bold', color='#ff7f0e')
for i in range(8):
    for j in range(8):
        axes[0,0].text(j, i, f'{int(hot_block[i,j])}', ha='center', va='center', fontsize=8)

# Top-middle: DCT coefficients (many non-zero)
from scipy.fft import dctn, idctn
hot_dct = dctn(hot_block.reshape(1,8,8), axes=(-2,-1), type=2, norm='ortho')[0]
hot_dct_q = np.round(hot_dct / 8)  # quantize
n_nonzero = np.count_nonzero(hot_dct_q)
im2 = axes[0,1].imshow(np.log1p(np.abs(hot_dct_q)), cmap='YlOrRd', vmin=0, vmax=4, aspect='equal')
axes[0,1].set_title(f'DCT Coefficients\n({n_nonzero}/64 non-zero)', fontsize=13, fontweight='bold')
for i in range(8):
    for j in range(8):
        v = int(hot_dct_q[i,j])
        color = 'white' if abs(v) > 5 else 'black'
        axes[0,1].text(j, i, f'{v}', ha='center', va='center', fontsize=8, color=color)

# Top-right: Reconstruction (good quality, needs all coefficients)
hot_recon = idctn((hot_dct_q * 8).reshape(1,8,8), axes=(-2,-1), type=2, norm='ortho')[0]
hot_recon = np.clip(np.round(hot_recon), 0, 255)
im3 = axes[0,2].imshow(hot_recon, cmap='RdYlBu_r', vmin=115, vmax=145, aspect='equal')
err = np.abs(hot_recon - hot_block).max()
axes[0,2].set_title(f'Reconstructed\n(max error={err:.0f}, needs {n_nonzero} coefficients)', fontsize=13, fontweight='bold')
for i in range(8):
    for j in range(8):
        axes[0,2].text(j, i, f'{int(hot_recon[i,j])}', ha='center', va='center', fontsize=8)

# --- Bottom row: DC-ONLY (what works for cold embeddings) ---

# Bottom-left: Original cold block (near-uniform)
cold_block = np.array([[88,87,89,88,87,88,88,89],
                        [88,88,87,88,89,88,87,88],
                        [87,88,88,89,88,87,88,88],
                        [88,89,88,88,87,88,88,87],
                        [88,87,89,88,88,88,87,89],
                        [87,88,88,89,88,87,88,88],
                        [88,88,87,88,89,88,88,87],
                        [89,88,88,87,88,89,88,88]], dtype=float)

im4 = axes[1,0].imshow(cold_block, cmap='RdYlBu_r', vmin=82, vmax=94, aspect='equal')
axes[1,0].set_title('Low-Entropy Block\n(cold rows, near zero-point)', fontsize=13, fontweight='bold')
axes[1,0].set_ylabel('DC-Only\n(our approach)', fontsize=13, fontweight='bold', color='#2ca02c')
for i in range(8):
    for j in range(8):
        axes[1,0].text(j, i, f'{int(cold_block[i,j])}', ha='center', va='center', fontsize=9)

# Bottom-middle: DCT coefficients (only DC non-zero)
cold_dct = dctn(cold_block.reshape(1,8,8), axes=(-2,-1), type=2, norm='ortho')[0]
cold_dct_q = np.round(cold_dct / 32)  # same step as our method
n_nz_cold = np.count_nonzero(cold_dct_q)
im5 = axes[1,1].imshow(np.log1p(np.abs(cold_dct_q)), cmap='YlOrRd', vmin=0, vmax=4, aspect='equal')
axes[1,1].set_title(f'DCT Coefficients\n(only DC survives = {n_nz_cold}/64 non-zero)', fontsize=13, fontweight='bold')
for i in range(8):
    for j in range(8):
        v = int(cold_dct_q[i,j])
        if i == 0 and j == 0:
            axes[1,1].text(j, i, f'DC={v}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        else:
            axes[1,1].text(j, i, '0', ha='center', va='center', fontsize=9, color='#999')

# Bottom-right: DC-only reconstruction
dc_val = cold_dct_q[0, 0] * 32 / 8  # DC * step / block_size
cold_recon = np.full((8, 8), np.clip(np.round(dc_val), 0, 255))
err_cold = np.abs(cold_recon - cold_block).max()
im6 = axes[1,2].imshow(cold_recon, cmap='RdYlBu_r', vmin=82, vmax=94, aspect='equal')
axes[1,2].set_title(f'DC-Only Reconstruction\n(max error={err_cold:.0f}, just 1 number!)', fontsize=13, fontweight='bold')
for i in range(8):
    for j in range(8):
        axes[1,2].text(j, i, f'{int(cold_recon[i,j])}', ha='center', va='center', fontsize=9,
                       fontweight='bold', color='#2ca02c')

# Add big arrow and explanation between rows
fig.text(0.5, 0.50, '↑ High entropy: needs many coefficients (full DCT decode required)\n'
         '↓ Low entropy (cold embeddings): DC alone is sufficient (just store the average)',
         ha='center', fontsize=13, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='#333', linewidth=2))

plt.suptitle('DCT vs DC-Only: Why Cold Embeddings Only Need the Block Average', fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.subplots_adjust(hspace=0.55)
plt.savefig('results/slide_figures_v3/02_dct_vs_dc_only.png', bbox_inches='tight')
plt.close()
print("Saved 02_dct_vs_dc_only.png")


# ================================================================
# Figure 3: Pareto curve (DC labels)
# ================================================================
fig, ax = plt.subplots(figsize=(12, 7))
for label, (ratio, loss) in baselines.items():
    marker = 's' if 'CAFE' in label else 'D'
    color = '#d62728' if 'CAFE' in label else '#888888'
    ax.scatter(ratio, loss, s=120, marker=marker, color=color, zorder=5, edgecolors='black', linewidth=0.5)
    offset = (10, 8) if 'CAFE' not in label else (-15, 10)
    ax.annotate(label, (ratio, loss), textcoords="offset points", xytext=offset, fontsize=10)

h265_x = [v[0] for v in h265_points.values()]; h265_y = [v[1] for v in h265_points.values()]
ax.plot(h265_x, h265_y, 'o--', color='#ff7f0e', markersize=8, label='H.265 (storage only, needs decode cache)', zorder=4)

dct_x = [v[0] for v in dct_points.values()]; dct_y = [v[1] for v in dct_points.values()]
ax.plot(dct_x, dct_y, 's-', color='#2ca02c', markersize=12, linewidth=2.5, label='DC block-mean (runtime, zero cache)', zorder=6)
for label, (ratio, loss) in dct_points.items():
    ax.annotate(label, (ratio, loss), textcoords="offset points", xytext=(10, -15),
                fontsize=11, fontweight='bold', color='#2ca02c')

ax.axhline(y=0.1, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(3, 0.11, 'Production threshold (0.1%)', fontsize=10, color='red', alpha=0.7)
ax.set_xscale('log'); ax.set_xlabel('Compression Ratio (vs fp32)', fontsize=14)
ax.set_ylabel('AUC Loss (%)', fontsize=14)
ax.set_title('Pareto Frontier: AUC Loss vs Compression Ratio\n(lower-left is better)', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper left'); ax.set_xlim(2, 20000); ax.set_ylim(-0.02, 0.85)
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('results/slide_figures_v3/03_pareto_curve.png', bbox_inches='tight'); plt.close()
print("Saved 03_pareto_curve.png")


# ================================================================
# Figure 4: Memory comparison (DC labels)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(mem_methods))
ax.bar(x, mem_values, color=colors5, width=0.6, edgecolor='black', linewidth=0.5)
for i, v in enumerate(mem_values):
    ratio = 2061 / v
    if v > 100:
        ax.text(i, v+30, f'{v:.0f} MB', ha='center', fontsize=12, fontweight='bold')
    else:
        ax.text(i, v+30, f'{v:.1f} MB\n({ratio:.0f}×)', ha='center', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(mem_methods, fontsize=11)
ax.set_ylabel('Runtime Memory (MB)', fontsize=13)
ax.set_title('Runtime Memory: DC Block-Mean Eliminates Decode Cache', fontsize=15, fontweight='bold')
ax.set_ylim(0, 2300); ax.grid(axis='y', alpha=0.3)
ax.annotate('Zero decode cache\nZero startup\nNo FFmpeg', xy=(3, 15.2), xytext=(3.5, 800),
            fontsize=12, color='#1f77b4', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2), ha='center')
plt.tight_layout(); plt.savefig('results/slide_figures_v3/04_memory_comparison.png', bbox_inches='tight'); plt.close()
print("Saved 04_memory_comparison.png")


# ================================================================
# Figure 5: AUC + Latency (DC labels)
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(perf_methods))

bars1 = ax1.bar(x, perf_auc, color=colors5, width=0.6, edgecolor='black', linewidth=0.5)
ax1.set_ylim(0.7995, 0.8030); ax1.set_xticks(x); ax1.set_xticklabels(perf_methods, fontsize=11)
ax1.set_ylabel('AUC', fontsize=13); ax1.set_title('Model Accuracy (AUC)', fontsize=14, fontweight='bold')
for i, v in enumerate(perf_auc):
    loss = (0.802497 - v) * 100
    label = f'{v:.4f}\n({loss:+.03f}%)' if loss != 0 else f'{v:.4f}'
    ax1.text(i, v + 0.0002, label, ha='center', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

bars2 = ax2.bar(x, perf_batch, color=colors5, width=0.6, edgecolor='black', linewidth=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(perf_methods, fontsize=11)
ax2.set_ylabel('Batch Latency (ms)', fontsize=13)
ax2.set_title('Inference Latency (ms/batch)', fontsize=14, fontweight='bold')
for i, v in enumerate(perf_batch):
    ax2.text(i, v+0.08, f'{v:.2f}ms', ha='center', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 6); ax2.grid(axis='y', alpha=0.3)

plt.suptitle('DC Block-Mean Matches H.265 Latency With Less Memory', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('results/slide_figures_v3/05_auc_latency.png', bbox_inches='tight'); plt.close()
print("Saved 05_auc_latency.png")


# ================================================================
# Figure 6: Journey diagram (updated labels)
# ================================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')
steps = [
    ("H.265 Video Codec\n6,466× storage\n71 MB runtime\nNeeds FFmpeg + cache", '#ff7f0e', 0.05),
    ("Decompose:\nWhy does it work?\n→ DCT quant = 160×\n→ CABAC+intra = 1.5×", '#2ca02c', 0.28),
    ("Discovery:\nCold blocks are\n100% DC-only\n→ Only average matters", '#1f77b4', 0.51),
    ("DC Block-Mean:\n136× runtime\n15 MB memory\nZero cache, zero FFmpeg", '#9467bd', 0.74),
]
for text, color, xpos in steps:
    box = mpatches.FancyBboxPatch((xpos, 0.15), 0.2, 0.7, boxstyle="round,pad=0.02",
                                   facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(xpos+0.1, 0.5, text, ha='center', va='center', fontsize=12,
            fontweight='bold', color=color, transform=ax.transAxes)
for i in range(3):
    x1 = steps[i][2]+0.21; x2 = steps[i+1][2]-0.01
    ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.5), transform=ax.transAxes)
ax.set_title('The Journey: From Video Codec to DC Block-Mean Compression', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout(); plt.savefig('results/slide_figures_v3/06_journey.png', bbox_inches='tight'); plt.close()
print("Saved 06_journey.png")


# ================================================================
# Figure 7: Comparison table (DC labels)
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
    ['DC 2% hot (ours)', '136×', '-0.091%', '2.35 ms', '15.2 MB', 'NONE', 'No'],
]
colors_t = [['#e0e0e0']*7, ['#fff']*7, ['#fff']*7, ['#fff']*7,
            ['#ffe0e0']*7, ['#fff5e0']*7, ['#e0ffe0']*7]
table = ax.table(cellText=table_data, cellColours=colors_t, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 1.8)
for j in range(7):
    table[0,j].set_text_props(fontweight='bold')
    table[6,j].set_text_props(fontweight='bold')
ax.set_title('Comparison With Prior Methods (* H.265: 29× runtime / 6,466× storage)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout(); plt.savefig('results/slide_figures_v3/07_comparison_table.png', bbox_inches='tight'); plt.close()
print("Saved 07_comparison_table.png")


print("\nAll 7 figures saved to results/slide_figures_v3/")
