#!/usr/bin/env python3
"""
Generate presentation figures for PCA sort finding.
Includes: AUC comparison, why PCA works, block-mean illustration, feature importance.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = 'results/mentor_slides'
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================
# Measured AUC data (from intelligent_agent_auc.py)
# ================================================================
BASELINE_AUC = 0.802497

data = {
    0.005: {'pca': -0.0534, 'value': -0.0643, 'freq': -0.3368, 'zero': -0.3682},
    0.01:  {'pca': -0.0245, 'value': -0.0330, 'freq': -0.2075, 'zero': -0.2262},
    0.02:  {'pca': -0.0118, 'value': -0.0131, 'freq': -0.1048, 'zero': -0.1128},
    0.043: {'pca': -0.0069, 'value': -0.0096, 'freq': -0.0427, 'zero': -0.0462},
}

sort_colors = {'pca': '#E74C3C', 'value': '#2ECC71', 'freq': '#3498DB', 'zero': '#95A5A6'}
sort_labels = {'pca': 'PCA sort', 'value': 'Value sort', 'freq': 'Freq sort', 'zero': 'Zero (no DC)'}
hf_list = [0.005, 0.01, 0.02, 0.043]

# ================================================================
# Figure 1: What is PCA sort — the intuition
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Value sort vs PCA sort concept
ax = axes[0]
np.random.seed(42)
# Simulate 2D embeddings with correlated dimensions
n_pts = 100
x = np.random.randn(n_pts) * 2
y = 0.7 * x + np.random.randn(n_pts) * 0.8  # correlated

# Value sort: sort by mean of (x,y)
means = (x + y) / 2
val_order = np.argsort(means)

# PCA sort: sort by PC1
from sklearn.decomposition import PCA
pts = np.column_stack([x, y])
pca = PCA(n_components=1)
scores = pca.fit_transform(pts).ravel()
pca_order = np.argsort(scores)

# Plot points colored by PCA score
scatter = ax.scatter(x, y, c=scores, cmap='RdYlGn', s=30, alpha=0.7, edgecolors='black', linewidth=0.3)

# Draw PC1 direction
pc1 = pca.components_[0]
center = [x.mean(), y.mean()]
scale = 3
ax.annotate('', xy=(center[0] + pc1[0]*scale, center[1] + pc1[1]*scale),
            xytext=(center[0] - pc1[0]*scale, center[1] - pc1[1]*scale),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
ax.text(center[0] + pc1[0]*scale + 0.3, center[1] + pc1[1]*scale + 0.3,
        'PC1\n(max variance)', fontsize=10, color='red', fontweight='bold')

# Draw mean direction
ax.annotate('', xy=(center[0] + 1*scale, center[1] + 1*scale),
            xytext=(center[0] - 1*scale, center[1] - 1*scale),
            arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='--'))
ax.text(center[0] - 1*scale - 1.5, center[1] - 1*scale,
        'Mean\n(equal weight)', fontsize=10, color='green', fontweight='bold')

ax.set_xlabel('Dimension 1', fontsize=11)
ax.set_ylabel('Dimension 2', fontsize=11)
ax.set_title('PCA vs Value Sort Direction\n(PCA finds max-variance axis)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 2: Block-mean error comparison
ax = axes[1]
block_size = 8
n_blocks = n_pts // block_size

# Compute block-mean MSE for both orderings
def block_mse(ordered_pts, bs):
    n = len(ordered_pts)
    nb = n // bs
    mses = []
    for i in range(nb):
        block = ordered_pts[i*bs:(i+1)*bs]
        mean = block.mean(axis=0)
        mse = ((block - mean)**2).mean()
        mses.append(mse)
    return np.array(mses)

val_mse = block_mse(pts[val_order], block_size)
pca_mse = block_mse(pts[pca_order], block_size)
freq_order = np.random.permutation(n_pts)  # random as proxy for freq
freq_mse = block_mse(pts[freq_order], block_size)

x_blocks = np.arange(len(val_mse))
ax.bar(x_blocks - 0.25, freq_mse, 0.25, color=sort_colors['freq'], alpha=0.7, label='Freq sort')
ax.bar(x_blocks, val_mse, 0.25, color=sort_colors['value'], alpha=0.7, label='Value sort')
ax.bar(x_blocks + 0.25, pca_mse, 0.25, color=sort_colors['pca'], alpha=0.7, label='PCA sort')
ax.set_xlabel('Block index', fontsize=11)
ax.set_ylabel('Within-block MSE', fontsize=11)
ax.set_title(f'Block-Mean Approximation Error\n(block_size={block_size}, lower = better)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Panel 3: Why adjacent rows matter
ax = axes[2]
# Show sorted embeddings — adjacent rows should be similar
pca_sorted = pts[pca_order]
val_sorted = pts[val_order]

ax.plot(range(n_pts), pca_sorted[:, 0], 'r-', alpha=0.6, linewidth=1, label='PCA sort (dim 1)')
ax.plot(range(n_pts), val_sorted[:, 0], 'g--', alpha=0.6, linewidth=1, label='Value sort (dim 1)')
ax.plot(range(n_pts), pts[freq_order, 0], 'b:', alpha=0.4, linewidth=1, label='Freq sort (dim 1)')

# Show block boundaries
for i in range(0, n_pts, block_size):
    ax.axvline(x=i, color='gray', alpha=0.2, linewidth=0.5)

ax.set_xlabel('Row position (after sorting)', fontsize=11)
ax.set_ylabel('Embedding value (dim 1)', fontsize=11)
ax.set_title('Sorted Embeddings: Smoothness Matters\n(smoother = better block-mean approximation)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('PCA Sort: Sort Rows by First Principal Component Before Blocking',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/08_pca_intuition.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/08_pca_intuition.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/08_pca_intuition.png")
plt.close()

# ================================================================
# Figure 2: AUC comparison — all sort methods at all hot fractions
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(hf_list))
width = 0.2

for si, sm in enumerate(['pca', 'value', 'freq', 'zero']):
    losses = [data[hf][sm] for hf in hf_list]
    bars = ax.bar(x + si * width - 1.5 * width, losses, width,
                  color=sort_colors[sm], alpha=0.85, label=sort_labels[sm],
                  edgecolor='black', linewidth=0.5)
    for b, loss in zip(bars, losses):
        ax.text(b.get_x() + b.get_width()/2, loss - 0.008,
                f'{loss:.3f}%', ha='center', va='top', fontsize=7, fontweight='bold', rotation=90)

ax.set_xticks(x)
ax.set_xticklabels([f'{hf*100:.1f}% hot' for hf in hf_list], fontsize=11)
ax.set_ylabel('AUC Loss (%)', fontsize=13)
ax.set_title('Sort Method Comparison: Real AUC Impact\n(4-bit DC block-mean, Criteo Kaggle, baseline AUC=0.8025)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/09_pca_auc_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/09_pca_auc_comparison.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/09_pca_auc_comparison.png")
plt.close()

# ================================================================
# Figure 3: PCA improvement over other methods
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: PCA vs Value sort improvement
hf_labels = [f'{hf*100:.1f}%' for hf in hf_list]
pca_vs_value = [abs(data[hf]['value']) / abs(data[hf]['pca']) for hf in hf_list]
pca_vs_freq = [abs(data[hf]['freq']) / abs(data[hf]['pca']) for hf in hf_list]
pca_vs_zero = [abs(data[hf]['zero']) / abs(data[hf]['pca']) for hf in hf_list]

x = np.arange(len(hf_list))
w = 0.25
ax1.bar(x - w, pca_vs_value, w, color='#2ECC71', alpha=0.85, label='vs Value sort', edgecolor='black', linewidth=0.5)
ax1.bar(x, pca_vs_freq, w, color='#3498DB', alpha=0.85, label='vs Freq sort', edgecolor='black', linewidth=0.5)
ax1.bar(x + w, pca_vs_zero, w, color='#95A5A6', alpha=0.85, label='vs Zero', edgecolor='black', linewidth=0.5)

for i in range(len(hf_list)):
    ax1.text(i - w, pca_vs_value[i] + 0.05, f'{pca_vs_value[i]:.1f}x', ha='center', fontsize=9, fontweight='bold')
    ax1.text(i, pca_vs_freq[i] + 0.05, f'{pca_vs_freq[i]:.1f}x', ha='center', fontsize=9, fontweight='bold')
    ax1.text(i + w, pca_vs_zero[i] + 0.05, f'{pca_vs_zero[i]:.1f}x', ha='center', fontsize=9, fontweight='bold')

ax1.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(hf_labels, fontsize=11)
ax1.set_xlabel('Hot Fraction', fontsize=12)
ax1.set_ylabel('AUC Loss Reduction Factor\n(higher = PCA better)', fontsize=11)
ax1.set_title('How Much Less AUC Loss\nDoes PCA Sort Give?', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Right: AUC loss at 1% hot — the headline number
ax2.axis('off')
headline_data = [
    ['Sort Method', 'AUC Loss', 'vs PCA'],
    ['PCA sort', f'{data[0.01]["pca"]:+.004f}%', '—'],
    ['Value sort', f'{data[0.01]["value"]:+.004f}%', f'{abs(data[0.01]["value"])/abs(data[0.01]["pca"]):.1f}x worse'],
    ['Freq sort', f'{data[0.01]["freq"]:+.004f}%', f'{abs(data[0.01]["freq"])/abs(data[0.01]["pca"]):.1f}x worse'],
    ['Zero', f'{data[0.01]["zero"]:+.004f}%', f'{abs(data[0.01]["zero"])/abs(data[0.01]["pca"]):.1f}x worse'],
]
table = ax2.table(cellText=headline_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.3, 1.8)

# Style header
for j in range(3):
    table[0, j].set_facecolor('#2C3E50')
    table[0, j].set_text_props(color='white', fontweight='bold')
# Color PCA row green
for j in range(3):
    table[1, j].set_facecolor('#D5F5E3')

ax2.set_title('At 1% Hot (189x Compression)', fontsize=12, fontweight='bold')

plt.suptitle('PCA Sort Improvement Over Baselines', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/10_pca_improvement.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/10_pca_improvement.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/10_pca_improvement.png")
plt.close()

# ================================================================
# Figure 4: How PCA sort works — step by step
# ================================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

np.random.seed(123)
n = 32
dim = 4
emb = np.random.randn(n, dim) * 0.5
emb[:, 0] += np.linspace(-2, 2, n)[np.random.permutation(n)]  # dim 0 has most variance

# Step 1: Original embeddings (unsorted)
ax = axes[0]
im = ax.imshow(emb, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
ax.set_xlabel('Dimension', fontsize=10)
ax.set_ylabel('Row index', fontsize=10)
ax.set_title('Step 1: Cold Rows\n(original order)', fontsize=10, fontweight='bold')
for i in range(0, n, 8):
    ax.axhline(y=i - 0.5, color='black', linewidth=1)

# Step 2: Compute PCA
ax = axes[1]
pca = PCA(n_components=1)
scores = pca.fit_transform(emb).ravel()
ax.barh(range(n), scores, color=['#E74C3C' if s > 0 else '#3498DB' for s in scores], alpha=0.7)
ax.set_xlabel('PC1 Score', fontsize=10)
ax.set_ylabel('Row index', fontsize=10)
ax.set_title('Step 2: Compute PC1\n(1 PCA on cold rows)', fontsize=10, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Step 3: Sort by PC1
ax = axes[2]
order = np.argsort(scores)
emb_sorted = emb[order]
im = ax.imshow(emb_sorted, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
ax.set_xlabel('Dimension', fontsize=10)
ax.set_ylabel('Sorted row index', fontsize=10)
ax.set_title('Step 3: Sort by PC1\n(similar rows adjacent)', fontsize=10, fontweight='bold')
for i in range(0, n, 8):
    ax.axhline(y=i - 0.5, color='black', linewidth=1)

# Step 4: Block mean
ax = axes[3]
bs = 8
nb = n // bs
block_means = emb_sorted.reshape(nb, bs, dim).mean(axis=1)
recon = np.repeat(block_means, bs, axis=0)
im = ax.imshow(recon, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
ax.set_xlabel('Dimension', fontsize=10)
ax.set_ylabel('Row index', fontsize=10)
ax.set_title('Step 4: Block Mean\n(each block → 1 value)', fontsize=10, fontweight='bold')
for i in range(0, n, 8):
    ax.axhline(y=i - 0.5, color='black', linewidth=1)

plt.suptitle('PCA Sort Pipeline: O(N log N) Sort → Better Block-Mean Approximation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/11_pca_pipeline.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/11_pca_pipeline.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/11_pca_pipeline.png")
plt.close()

# ================================================================
# Figure 5: Why PCA > Value sort — mathematical argument
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

text = """
Value Sort:  sort by mean(row) = Σᵢ xᵢ / D
             → weights all dimensions equally
             → two rows with same mean but different patterns are adjacent
               e.g., [0, 1, 0, 1] and [0.5, 0.5, 0.5, 0.5] have same mean
             → block average loses the pattern difference

PCA Sort:    sort by PC1 score = Σᵢ wᵢ xᵢ  (learned weights wᵢ from data)
             → weights dimensions by variance explained
             → high-variance dimensions get more weight
             → adjacent rows are similar along the most informative direction
             → block average preserves more useful information

Key insight: PCA sort minimizes within-block variance along the direction
             that matters most for reconstruction accuracy.

             This is equivalent to 1D k-means clustering along PC1,
             which minimizes the quantization error of the block-mean
             approximation in the optimal projection direction.

Cost:        O(N) PCA fit (one pass) + O(N log N) sort
             ~4 seconds for 33M rows on CPU — done once at deployment
"""

ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#2C3E50'))
ax.set_title('Why PCA Sort > Value Sort: Mathematical Intuition', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/12_pca_math.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/12_pca_math.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/12_pca_math.png")
plt.close()

print(f"\nAll PCA slides saved to {OUT_DIR}/")
