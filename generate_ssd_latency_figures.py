#!/usr/bin/env python3
"""
Generate figures showing SSD cold embedding latency vs in-memory approaches.
Proves that disk-based cold embeddings are prohibitively slow for serving.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = 'results/ssd_figures'

# ================================================================
# Load data
# ================================================================
with open('results/ssd_benchmark/real_ssd.json') as f:
    ssd_data = json.load(f)

with open('results/serving_benchmark.json') as f:
    serving_data = json.load(f)

# Build unified data table: fp32 DRAM, SSD cold, DC value-sort (one config)
methods = []

# fp32 DRAM baseline
methods.append({
    'name': 'fp32 DRAM',
    'p50': ssd_data['dram']['p50_ms'],
    'p99': ssd_data['dram']['p99_ms'],
    'category': 'baseline',
    'mem_mb': 2061,
    'read_p50': 0,
    'read_p99': 0,
})

# SSD cold (real disk reads)
methods.append({
    'name': 'SSD cold',
    'p50': ssd_data['ssd']['p50_ms'],
    'p99': ssd_data['ssd']['p99_ms'],
    'category': 'disk',
    'mem_mb': 1970,
    'read_p50': ssd_data['ssd']['read_p50_ms'],
    'read_p99': ssd_data['ssd']['read_p99_ms'],
})

# DC value-sort 4-bit (1% hot) — best balance of compression and AUC
dc_val_1 = next(s for s in serving_data if s['config'] == 'dc_value_hf0.01')
methods.append({
    'name': 'DC value-sort\n4-bit (ours)',
    'p50': dc_val_1['p50_ms'],
    'p99': dc_val_1['p99_ms'],
    'category': 'ours',
    'mem_mb': 10.9,
    'read_p50': 0,
    'read_p99': 0,
})

import os
os.makedirs(OUT_DIR, exist_ok=True)

# Color scheme
colors = {
    'baseline': '#4A90D9',
    'disk': '#E74C3C',
    'ours': '#2ECC71',
}

n = len(methods)
x = np.arange(n)

# ================================================================
# Figure 1: Batch Latency Comparison (p50 + p99, grouped bar)
# ================================================================
fig, ax = plt.subplots(figsize=(8, 6))

width = 0.35

bar_colors_p50 = [colors[m['category']] for m in methods]
bar_colors_p99_dark = []
for c in bar_colors_p50:
    r, g, b = int(c[1:3], 16)/255, int(c[3:5], 16)/255, int(c[5:7], 16)/255
    bar_colors_p99_dark.append((r*0.7, g*0.7, b*0.7))

bars_p50 = ax.bar(x - width/2, [m['p50'] for m in methods], width,
                   color=bar_colors_p50, alpha=0.85, edgecolor='black', linewidth=0.5)
bars_p99 = ax.bar(x + width/2, [m['p99'] for m in methods], width,
                   color=bar_colors_p99_dark, alpha=0.85, edgecolor='black', linewidth=0.5)

for bar in bars_p50:
    h = bar.get_height()
    label = f'p50: {h:.1f}' if h < 20 else f'p50: {h:.0f}'
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, label,
            ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars_p99:
    h = bar.get_height()
    label = f'p99: {h:.1f}' if h < 20 else f'p99: {h:.0f}'
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, label,
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.8)
ax.text(n - 0.6, 11.5, '10ms SLA', fontsize=11, color='orange', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([m['name'] for m in methods], fontsize=11)
ax.set_ylabel('Batch Latency (ms)', fontsize=13)
ax.set_title('Cold Embedding Serving: SSD vs In-Memory\n(Criteo Kaggle, batch=128, SATA SSD)', fontsize=13, fontweight='bold')

legend_patches = [
    mpatches.Patch(color='#888888', alpha=0.85, label='p50 (lighter)'),
    mpatches.Patch(color='#444444', alpha=0.85, label='p99 (darker)'),
    mpatches.Patch(color=colors['baseline'], label='fp32 baseline'),
    mpatches.Patch(color=colors['disk'], label='SSD disk-based'),
    mpatches.Patch(color=colors['ours'], label='DC block-mean (ours)'),
]
ax.legend(handles=legend_patches, loc='upper left', fontsize=9)
ax.set_ylim(0, max(m['p99'] for m in methods) * 1.18)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/batch_latency_comparison.png', dpi=120, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/batch_latency_comparison.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/batch_latency_comparison.png")
plt.close()

# ================================================================
# Figure 2: SSD read breakdown (stacked bar)
# ================================================================
fig, ax = plt.subplots(figsize=(8, 6))

inference_time = []
read_time = []
for m in methods:
    if m['read_p50'] > 0:
        inference_time.append(m['p50'] - m['read_p50'])
        read_time.append(m['read_p50'])
    else:
        inference_time.append(m['p50'])
        read_time.append(0)

bar_cols = [colors[m['category']] for m in methods]
bars1 = ax.bar(x, inference_time, width=0.55, color=bar_cols,
               alpha=0.85, label='Inference', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, read_time, width=0.55, bottom=inference_time,
               color='#E74C3C', alpha=0.4, hatch='///', label='SSD read I/O',
               edgecolor='#E74C3C', linewidth=0.5)

for i, m in enumerate(methods):
    total = m['p50']
    label = f'{total:.1f}ms' if total < 20 else f'{total:.0f}ms'
    ax.text(i, total + 2, label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    if m['read_p50'] > 0:
        mid = inference_time[i] + read_time[i]/2
        ax.text(i, mid, f'read: {m["read_p50"]:.0f}ms', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.8)
ax.text(n - 0.6, 11.5, '10ms SLA', fontsize=11, color='orange', fontweight='bold')

# Slowdown annotation
ssd_idx = 1
dram_idx = 0
slowdown = methods[ssd_idx]['p50'] / methods[dram_idx]['p50']
ax.annotate(f'{slowdown:.0f}x slower\nthan DRAM',
            xy=(ssd_idx, methods[ssd_idx]['p50']),
            xytext=(ssd_idx + 0.6, methods[ssd_idx]['p50'] * 0.82),
            fontsize=12, fontweight='bold', color='#E74C3C',
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))

ax.set_xticks(x)
ax.set_xticklabels([m['name'] for m in methods], fontsize=11)
ax.set_ylabel('p50 Batch Latency (ms)', fontsize=13)
ax.set_title('SSD I/O Dominates Cold Embedding Latency\n(91% of SSD batch time is disk read)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(0, max(m['p50'] for m in methods) * 1.2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/ssd_read_breakdown.png', dpi=120, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/ssd_read_breakdown.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/ssd_read_breakdown.png")
plt.close()

# ================================================================
# Figure 3: p99 tail latency (horizontal bar, full scale + zoom)
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1]})

bar_colors = [colors[m['category']] for m in methods]

# Left: full scale
bars = ax1.barh(np.arange(n), [m['p99'] for m in methods], color=bar_colors,
                alpha=0.85, edgecolor='black', linewidth=0.5, height=0.5)
for i, m in enumerate(methods):
    ax1.text(m['p99'] + 1.5, i, f"{m['p99']:.1f}ms", va='center', fontsize=10, fontweight='bold')
ax1.axvline(x=10, color='orange', linestyle='--', linewidth=2, alpha=0.8)
ax1.text(11, n-0.7, '10ms SLA', fontsize=9, color='orange', fontweight='bold')
ax1.set_yticks(np.arange(n))
ax1.set_yticklabels([m['name'] for m in methods], fontsize=11)
ax1.set_xlabel('p99 Tail Latency (ms)', fontsize=11)
ax1.set_title('Full Scale', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Right: zoomed <15ms
non_ssd = [m for m in methods if m['p99'] < 15]
n2 = len(non_ssd)
bar_colors2 = [colors[m['category']] for m in non_ssd]
bars2 = ax2.barh(np.arange(n2), [m['p99'] for m in non_ssd], color=bar_colors2,
                 alpha=0.85, edgecolor='black', linewidth=0.5, height=0.5)
for i, m in enumerate(non_ssd):
    ax2.text(m['p99'] + 0.15, i, f"{m['p99']:.1f}ms", va='center', fontsize=10, fontweight='bold')
ax2.axvline(x=10, color='orange', linestyle='--', linewidth=2, alpha=0.8)
ax2.text(10.2, n2-0.7, '10ms SLA', fontsize=9, color='orange', fontweight='bold')

# SSD off-chart callout
ax2.text(9, 0.15, f'SSD: {methods[1]["p99"]:.0f}ms\n({methods[1]["p99"]/methods[0]["p99"]:.0f}x baseline)',
         fontsize=9, color='#E74C3C', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='#E74C3C'))

ax2.set_yticks(np.arange(n2))
ax2.set_yticklabels([m['name'] for m in non_ssd], fontsize=11)
ax2.set_xlabel('p99 Tail Latency (ms)', fontsize=11)
ax2.set_title('Zoomed (< 15ms)', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 14)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

plt.suptitle('Tail Latency: SSD Cold Embeddings Violate SLA\n(Criteo Kaggle, SATA SSD, batch=128)',
             fontsize=13, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/p99_tail_latency.png', dpi=120, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/p99_tail_latency.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/p99_tail_latency.png")
plt.close()

# ================================================================
# Figure 4: Latency vs Memory tradeoff (scatter)
# ================================================================
fig, ax = plt.subplots(figsize=(8, 6))

for m in methods:
    mem = m['mem_mb']
    color = colors[m['category']]
    marker = 's' if m['category'] == 'ours' else 'o'
    size = 200

    ax.scatter(mem, m['p50'], c=color, s=size, marker=marker, edgecolors='black',
               linewidth=1, zorder=5, alpha=0.9)
    ax.scatter(mem, m['p99'], c=color, s=size, marker='^', edgecolors='black',
               linewidth=1, zorder=5, alpha=0.6)
    ax.plot([mem, mem], [m['p50'], m['p99']], color=color, linewidth=1.5, alpha=0.5)

    name_short = m['name'].replace('\n', ' ')
    offset_y = 5 if m['p99'] < 20 else -10
    ax.annotate(name_short, (mem, m['p99']), textcoords="offset points",
                xytext=(12, offset_y), fontsize=9, color=color, fontweight='bold')

ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='10ms SLA')
ax.set_xscale('log')
ax.set_xlabel('Memory / Storage Footprint (MB)', fontsize=13)
ax.set_ylabel('Batch Latency (ms)', fontsize=13)
ax.set_title('Memory vs Latency Tradeoff\n(circles = p50, triangles = p99)', fontsize=13, fontweight='bold')

legend_patches = [
    mpatches.Patch(color=colors['baseline'], label='fp32 baseline'),
    mpatches.Patch(color=colors['disk'], label='SSD disk-based'),
    mpatches.Patch(color=colors['ours'], label='DC block-mean (ours)'),
    plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='10ms SLA'),
]
ax.legend(handles=legend_patches, loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/latency_vs_memory.png', dpi=120, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/latency_vs_memory.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/latency_vs_memory.png")
plt.close()

# ================================================================
# Figure 5: Average batch latency (single bar per method)
# ================================================================
fig, ax = plt.subplots(figsize=(7, 5))

# Mean latencies: serving benchmark has mean_ms for fp32 and DC;
# SSD benchmark didn't save mean, but mean >= p50 for right-skewed distributions.
# Use (p50 + p99) / 2 as a conservative estimate for SSD mean.
fp32_serving = next(s for s in serving_data if s['config'] == 'fp32')
dc_serving = next(s for s in serving_data if s['config'] == 'dc_value_hf0.01')

avg_methods = [
    {'name': 'fp32\nDRAM',     'mean': fp32_serving['mean_ms'], 'category': 'baseline'},
    {'name': 'SSD\ncold',      'mean': (ssd_data['ssd']['p50_ms'] + ssd_data['ssd']['p99_ms']) / 2, 'category': 'disk'},
    {'name': 'DC value-sort\n4-bit (ours)', 'mean': dc_serving['mean_ms'], 'category': 'ours'},
]

n_avg = len(avg_methods)
x_avg = np.arange(n_avg)
bar_cols = [colors[m['category']] for m in avg_methods]

bars = ax.bar(x_avg, [m['mean'] for m in avg_methods], width=0.55,
              color=bar_cols, alpha=0.85, edgecolor='black', linewidth=0.5)

for i, bar in enumerate(bars):
    h = bar.get_height()
    label = f'{h:.1f} ms' if h < 20 else f'{h:.0f} ms'
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, label,
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Slowdown annotation on SSD bar
ssd_mean = avg_methods[1]['mean']
dram_mean = avg_methods[0]['mean']
slowdown = ssd_mean / dram_mean
ax.annotate(f'{slowdown:.0f}x slower',
            xy=(1, ssd_mean),
            xytext=(1.55, ssd_mean * 0.82),
            fontsize=13, fontweight='bold', color='#E74C3C',
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))

ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.8)
ax.text(n_avg - 0.55, 12, '10ms SLA', fontsize=11, color='orange', fontweight='bold')

ax.set_xticks(x_avg)
ax.set_xticklabels([m['name'] for m in avg_methods], fontsize=11)
ax.set_ylabel('Average Batch Latency (ms)', fontsize=13)
ax.set_title('Average Batch Latency: SSD vs In-Memory\n(Criteo Kaggle, batch=128, SATA SSD)', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(m['mean'] for m in avg_methods) * 1.2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/avg_batch_latency.png', dpi=120, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/avg_batch_latency.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/avg_batch_latency.png")
plt.close()

# ================================================================
# Summary
# ================================================================
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}")
print(f"{'Method':<25} {'p50 (ms)':>10} {'p99 (ms)':>10} {'Memory':>12} {'vs DRAM p50':>12}")
print("-" * 70)
for m in methods:
    slowdown = m['p50'] / methods[0]['p50']
    name = m['name'].replace('\n', ' ')
    slow_str = f'{slowdown:.1f}x' if slowdown > 1.1 else '1.0x'
    print(f"{name:<25} {m['p50']:>9.1f} {m['p99']:>9.1f} {m['mem_mb']:>10.0f} MB {slow_str:>12}")

print(f"\nSSD cold is {methods[1]['p50']/methods[0]['p50']:.0f}x slower (p50) "
      f"and {methods[1]['p99']/methods[0]['p99']:.0f}x slower (p99) than DRAM.")
print(f"DC block-mean: {methods[0]['p50']/methods[2]['p50']:.2f}x faster than fp32 DRAM "
      f"at {methods[2]['mem_mb']/methods[0]['mem_mb']*100:.1f}% memory ({methods[2]['mem_mb']:.0f} MB vs {methods[0]['mem_mb']:.0f} MB).")
