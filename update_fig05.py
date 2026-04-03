#!/usr/bin/env python3
"""Update figure 05: AUC + Latency with corrected DC numbers."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('results/slide_figures_v3', exist_ok=True)
plt.rcParams.update({'font.size': 14, 'figure.dpi': 150})

# Corrected data (Python-verified AUC, C++ batch latency)
methods = ['fp32\nbaseline', 'H.265\n+cache', 'DC\n4.3% hot', 'DC\n2% hot', 'DC\n1% hot']
auc_vals = [0.802497, 0.802107, 0.802136, 0.801643, 0.800829]
batch_ms = [4.98, 2.39, 2.49, 2.75, 2.82]
mem_mb =   [2061, 71, 32.9, 21.2, 16.0]
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: AUC
bars1 = ax1.bar(range(len(methods)), auc_vals, color=colors, width=0.6,
                edgecolor='black', linewidth=0.5)
ax1.set_ylim(0.7995, 0.8030)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=11)
ax1.set_ylabel('AUC', fontsize=13)
ax1.set_title('Model Accuracy (AUC)', fontsize=14, fontweight='bold')
for i, v in enumerate(auc_vals):
    loss = (0.802497 - v) * 100
    label = f'{v:.4f}\n({loss:+.03f}%)' if loss > 0.0005 else f'{v:.4f}'
    ax1.text(i, v + 0.0002, label, ha='center', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Panel 2: Batch Latency
bars2 = ax2.bar(range(len(methods)), batch_ms, color=colors, width=0.6,
                edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, fontsize=11)
ax2.set_ylabel('Batch Latency (ms)', fontsize=13)
ax2.set_title('Inference Latency', fontsize=14, fontweight='bold')
for i, v in enumerate(batch_ms):
    ax2.text(i, v + 0.08, f'{v:.2f}ms', ha='center', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 6)
ax2.grid(axis='y', alpha=0.3)

# Panel 3: Memory
bars3 = ax3.bar(range(len(methods)), mem_mb, color=colors, width=0.6,
                edgecolor='black', linewidth=0.5)
ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels(methods, fontsize=11)
ax3.set_ylabel('Runtime Memory (MB)', fontsize=13)
ax3.set_title('Runtime Memory', fontsize=14, fontweight='bold')
for i, v in enumerate(mem_mb):
    ratio = 2061 / v
    if v > 100:
        ax3.text(i, v + 30, f'{v:.0f}MB', ha='center', fontsize=11, fontweight='bold')
    else:
        ax3.text(i, v + 30, f'{v:.1f}MB\n({ratio:.0f}×)', ha='center', fontsize=11, fontweight='bold')
ax3.set_ylim(0, 2300)
ax3.grid(axis='y', alpha=0.3)

plt.suptitle('DC Block-Mean: Comparable Latency and AUC, 3× Less Memory Than H.265+Cache',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/slide_figures_v3/05_auc_latency.png', bbox_inches='tight')
plt.close()
print("Saved 05_auc_latency.png")
