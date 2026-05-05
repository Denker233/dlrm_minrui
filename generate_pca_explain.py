#!/usr/bin/env python3
"""
Simple visual explanation of PCA sort for embedding compression.
Uses actual embedding data from the model.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

OUT_DIR = 'results/mentor_slides'
os.makedirs(OUT_DIR, exist_ok=True)

# Load real embedding data
sd = torch.load('models/dlrm_kaggle_correct.pt', map_location='cpu', weights_only=False)['state_dict']
# Use table 25 (142K rows, manageable size, interesting structure)
emb = sd['emb_l.25.weight'].numpy()
print(f"Table 25: {emb.shape[0]:,} rows × {emb.shape[1]} dims")

# Take a sample of 64 cold rows (skip the first few hot rows)
np.random.seed(42)
sample_idx = np.random.choice(range(1000, 5000), size=64, replace=False)
sample = emb[sample_idx].copy()

# ================================================================
# Figure: 4-panel explanation
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# ----------------------------------------------------------------
# Panel 1: The problem — unsorted rows in blocks
# ----------------------------------------------------------------
ax = axes[0, 0]
block_size = 8
n_blocks = len(sample) // block_size

# Show heatmap of unsorted rows
im = ax.imshow(sample, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax.set_xlabel('Embedding Dimension (D=16)', fontsize=11)
ax.set_ylabel('Row Index (unsorted)', fontsize=11)
ax.set_title('Step 1: Cold Rows (Original Order)\nRows within each block are dissimilar', fontsize=12, fontweight='bold')

# Draw block boundaries
for i in range(0, len(sample) + 1, block_size):
    ax.axhline(y=i - 0.5, color='black', linewidth=2)

# Label blocks
for i in range(n_blocks):
    ax.text(-1.5, i * block_size + block_size / 2 - 0.5, f'B{i}',
            fontsize=8, fontweight='bold', va='center', ha='center')

plt.colorbar(im, ax=ax, shrink=0.6, label='Value')

# ----------------------------------------------------------------
# Panel 2: PCA finds the best sorting direction
# ----------------------------------------------------------------
ax = axes[0, 1]

# Fit PCA
pca = PCA(n_components=2)
scores = pca.fit_transform(sample)

# Plot rows in PC1-PC2 space
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=scores[:, 0], cmap='RdYlGn',
                     s=40, edgecolors='black', linewidth=0.5, alpha=0.8)

# Draw PC1 axis (sorting direction)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')

# Arrow showing PC1 direction
xlim = ax.get_xlim()
ax.annotate('', xy=(xlim[1] * 0.9, 0), xytext=(xlim[0] * 0.9, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=3))
ax.text(0, ax.get_ylim()[1] * 0.85, f'PC1 direction\n(explains {pca.explained_variance_ratio_[0]*100:.0f}% of variance)',
        ha='center', fontsize=10, fontweight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel(f'PC1 Score (sort by this)', fontsize=11)
ax.set_ylabel(f'PC2 Score', fontsize=11)
ax.set_title('Step 2: PCA Finds Best Direction\nProject all rows onto PC1', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, shrink=0.6, label='PC1 Score')

# ----------------------------------------------------------------
# Panel 3: Sorted rows — similar rows now adjacent
# ----------------------------------------------------------------
ax = axes[1, 0]

# Sort by PC1
pc1_scores = scores[:, 0]
sort_order = np.argsort(pc1_scores)
sample_sorted = sample[sort_order]

im = ax.imshow(sample_sorted, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax.set_xlabel('Embedding Dimension (D=16)', fontsize=11)
ax.set_ylabel('Row Index (sorted by PC1)', fontsize=11)
ax.set_title('Step 3: Sort Rows by PC1 Score\nSimilar rows are now adjacent in each block', fontsize=12, fontweight='bold')

for i in range(0, len(sample) + 1, block_size):
    ax.axhline(y=i - 0.5, color='black', linewidth=2)

for i in range(n_blocks):
    ax.text(-1.5, i * block_size + block_size / 2 - 0.5, f'B{i}',
            fontsize=8, fontweight='bold', va='center', ha='center')

plt.colorbar(im, ax=ax, shrink=0.6, label='Value')

# ----------------------------------------------------------------
# Panel 4: Block mean error comparison
# ----------------------------------------------------------------
ax = axes[1, 1]

def compute_block_mse(rows, bs):
    """Compute per-block MSE when replacing rows with block mean."""
    n = len(rows)
    nb = n // bs
    mses = []
    for i in range(nb):
        block = rows[i*bs:(i+1)*bs]
        mean = block.mean(axis=0)
        mse = ((block - mean) ** 2).mean()
        mses.append(mse)
    return np.array(mses)

# Three orderings
mse_unsorted = compute_block_mse(sample, block_size)

# Value sort (by row mean)
val_order = np.argsort(sample.mean(axis=1))
mse_value = compute_block_mse(sample[val_order], block_size)

# PCA sort
mse_pca = compute_block_mse(sample_sorted, block_size)

x = np.arange(n_blocks)
w = 0.25
bars1 = ax.bar(x - w, mse_unsorted, w, color='#95A5A6', alpha=0.85, label=f'Unsorted (mean MSE={mse_unsorted.mean():.6f})', edgecolor='black', linewidth=0.3)
bars2 = ax.bar(x, mse_value, w, color='#2ECC71', alpha=0.85, label=f'Value sort (mean MSE={mse_value.mean():.6f})', edgecolor='black', linewidth=0.3)
bars3 = ax.bar(x + w, mse_pca, w, color='#E74C3C', alpha=0.85, label=f'PCA sort (mean MSE={mse_pca.mean():.6f})', edgecolor='black', linewidth=0.3)

reduction_vs_unsorted = (1 - mse_pca.mean() / mse_unsorted.mean()) * 100
reduction_vs_value = (1 - mse_pca.mean() / mse_value.mean()) * 100

ax.set_xlabel('Block Index', fontsize=11)
ax.set_ylabel('Within-Block MSE (lower = better)', fontsize=11)
ax.set_title(f'Step 4: Block-Mean Error Comparison\n'
             f'PCA sort: {reduction_vs_unsorted:.0f}% less error than unsorted, '
             f'{reduction_vs_value:.0f}% less than value sort',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('PCA Sort: How It Works\n'
             'Sort embedding rows by first principal component before computing block means\n'
             '→ adjacent rows become similar → block mean is a better approximation → less AUC loss',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/13_pca_explained.png', dpi=100, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/13_pca_explained.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/13_pca_explained.png")
plt.close()

# ================================================================
# Figure 2: Simple 1D analogy
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Generate simple 1D example
np.random.seed(42)
values = np.random.randn(24) * 2
block_size = 4

# Panel 1: Unsorted
ax = axes[0]
colors = plt.cm.Set3(np.arange(len(values) // block_size).repeat(block_size) / (len(values) // block_size))
ax.bar(range(len(values)), values, color=colors, edgecolor='black', linewidth=0.5)
for i in range(0, len(values) + 1, block_size):
    ax.axvline(x=i - 0.5, color='red', linewidth=2, linestyle='--')
# Show block means
for i in range(len(values) // block_size):
    block = values[i*block_size:(i+1)*block_size]
    mean = block.mean()
    ax.hlines(mean, i*block_size - 0.5, (i+1)*block_size - 0.5, color='red', linewidth=3)
mse_un = sum((values[i] - values[i//block_size*block_size:i//block_size*block_size+block_size].mean())**2
             for i in range(len(values))) / len(values)
ax.set_title(f'Unsorted: Block Mean ≠ Row Values\nMSE = {mse_un:.2f}', fontsize=11, fontweight='bold')
ax.set_xlabel('Row index', fontsize=10)
ax.set_ylabel('Value', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Panel 2: Sorted
ax = axes[1]
sorted_vals = np.sort(values)
colors_sorted = plt.cm.Set3(np.arange(len(sorted_vals) // block_size).repeat(block_size) / (len(sorted_vals) // block_size))
ax.bar(range(len(sorted_vals)), sorted_vals, color=colors_sorted, edgecolor='black', linewidth=0.5)
for i in range(0, len(sorted_vals) + 1, block_size):
    ax.axvline(x=i - 0.5, color='red', linewidth=2, linestyle='--')
for i in range(len(sorted_vals) // block_size):
    block = sorted_vals[i*block_size:(i+1)*block_size]
    mean = block.mean()
    ax.hlines(mean, i*block_size - 0.5, (i+1)*block_size - 0.5, color='red', linewidth=3)
mse_sorted = sum((sorted_vals[i] - sorted_vals[i//block_size*block_size:i//block_size*block_size+block_size].mean())**2
                  for i in range(len(sorted_vals))) / len(sorted_vals)
ax.set_title(f'Sorted: Block Mean ≈ Row Values\nMSE = {mse_sorted:.2f} ({(1-mse_sorted/mse_un)*100:.0f}% reduction)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Row index (sorted)', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Panel 3: The key insight text
ax = axes[2]
ax.axis('off')
insight_text = """
WHY SORTING HELPS:

Block-mean replaces each row
with its block's average.

If rows in a block are similar,
the average is close to each row
→ small error

If rows are different,
the average is far from each row
→ large error

SORTING groups similar rows
into the same block.

PCA sort is BETTER than
value sort because it sorts
along the direction where
rows vary MOST.

Value sort: mean(row) = Σxᵢ/16
  → equal weight per dimension

PCA sort: PC1·row = Σwᵢxᵢ
  → learned weights from data
  → high-variance dims count more
"""
ax.text(0.1, 0.95, insight_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#F0F8FF', edgecolor='#2C3E50'))

plt.suptitle('PCA Sort Intuition: Sorting Makes Block Averages More Accurate',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/14_pca_simple.png', dpi=100, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/14_pca_simple.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/14_pca_simple.png")
plt.close()

print("Done.")
