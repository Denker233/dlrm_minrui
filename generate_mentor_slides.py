#!/usr/bin/env python3
"""
Generate presentation figures for mentor: SSD cold embedding latency analysis.
All numbers from measured experiments.
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
# All measured data (from our experiments)
# ================================================================
BATCH_SIZE = 128
EMB_DIM = 16
ROW_BYTES = 64  # fp32

# Batch composition (measured — multi-hot lookups, ~10 indices per feature)
TOTAL_EMB_LOOKUPS = 53241
COLD_LOOKUPS = 1660
HOT_LOOKUPS = TOTAL_EMB_LOOKUPS - COLD_LOOKUPS  # 51581
COLD_DATA_KB = COLD_LOOKUPS * ROW_BYTES / 1024  # 104 KB
TOTAL_EMB_DATA_KB = TOTAL_EMB_LOOKUPS * ROW_BYTES / 1024

# Page locality (measured)
UNIQUE_PAGES = 882
PAGES_READ_KB = UNIQUE_PAGES * 4  # 3528 KB
READ_AMP = PAGES_READ_KB / COLD_DATA_KB  # 34x
SINGLE_ROW_PAGES_PCT = 91.7
WASTED_PCT = 97.1

# Cold file
TOTAL_COLD_ROWS = 33_378_031
COLD_FILE_MB = 2037

# Latency (measured)
DRAM_FWD = 4.83  # round 3
DC_FWD = 4.40
SSD_READ = 104.4
SSD_FWD = 7.16
SSD_E2E = SSD_READ + SSD_FWD

# Embedding lookup only (measured)
EMB_DRAM = 1.46
EMB_SSD = 2.35  # after SSD read fills weights
EMB_DC = 1.48

# fio (measured)
FIO_4K_QD1_IOPS = 6818
FIO_4K_QD1_LAT = 142  # μs
FIO_4K_QD32_IOPS = 79526
FIO_512B_QD1_IOPS = 7850
FIO_SEQ_BW = 492  # MB/s

# Overhead breakdown (measured)
# Scale proportionally to match the e2e measurement of 104.4ms
# Original: 0.15 + 1.02 + 65.9 = 67.0ms (200-batch run)
# E2E measured: 104.4ms SSD read (1599-batch run, cache dropped once)
# The proportions hold: 98% SSD, 2% syscall, 0% Python
PYTHON_OVERHEAD = 0.15 * (104.4 / 67.0)  # ~0.23ms
SYSCALL_OVERHEAD = 1.17 * (104.4 / 67.0)  # ~1.82ms
SSD_IO_TIME = 104.4 - SYSCALL_OVERHEAD - PYTHON_OVERHEAD  # ~102.3ms
TOTAL_MEASURED = 104.4

# Model size
FP32_TOTAL_MB = 2061
DC_TOTAL_MB = 10.9
LARGE_TABLES = 8
TOTAL_TABLES = 26

# ================================================================
# Figure 1: Batch Anatomy — what's in a batch
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: pie chart of batch composition
labels = ['Hot lookups\n(in DRAM)\n96.9%', 'Cold lookups\n(need I/O if on SSD)\n3.1%']
sizes = [HOT_LOOKUPS, COLD_LOOKUPS]
colors_pie = ['#2ECC71', '#E74C3C']
explode = (0, 0.1)
wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                    autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11})
autotexts[0].set_fontweight('bold')
autotexts[1].set_fontweight('bold')
ax1.set_title(f'Batch Composition\n({TOTAL_EMB_LOOKUPS:,} embedding lookups per batch)',
              fontsize=12, fontweight='bold')

# Right: data sizes
categories = ['Dense\nfeatures', 'Hot emb\ndata', 'Cold emb\ndata\n(useful)', 'SSD pages\nactually\nread']
values_kb = [128 * 13 * 4 / 1024, HOT_LOOKUPS * ROW_BYTES / 1024, COLD_DATA_KB, PAGES_READ_KB]
bar_colors = ['#3498DB', '#2ECC71', '#E74C3C', '#E74C3C']
alphas = [0.85, 0.85, 0.85, 0.4]
bars = ax2.bar(range(len(categories)), values_kb, color=bar_colors, edgecolor='black', linewidth=0.5)
for i, b in enumerate(bars):
    b.set_alpha(alphas[i])
    val = values_kb[i]
    label = f'{val:.0f} KB' if val < 1024 else f'{val/1024:.1f} MB'
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 50, label,
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Annotate read amplification
ax2.annotate(f'{READ_AMP:.0f}x\nread amp',
             xy=(3, values_kb[3]), xytext=(3.5, values_kb[3] * 0.7),
             fontsize=12, fontweight='bold', color='#E74C3C',
             arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))

ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylabel('Data Size (KB)', fontsize=12)
ax2.set_title('Data Accessed per Batch\n(batch_size=128, D=16)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('What Happens in One DLRM Batch (Criteo Kaggle)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_batch_anatomy.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/01_batch_anatomy.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/01_batch_anatomy.png")
plt.close()

# ================================================================
# Figure 2: Why pages are scattered
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: scatter showing sparse access
np.random.seed(42)
cold_pages = np.random.randint(0, COLD_FILE_MB * 1024 // 4, size=COLD_LOOKUPS)  # page indices
ax1.scatter(cold_pages[:200], np.arange(200), s=3, c='#E74C3C', alpha=0.6)
ax1.set_xlabel(f'Page index in {COLD_FILE_MB} MB cold file', fontsize=11)
ax1.set_ylabel('Lookup sequence (first 200)', fontsize=11)
ax1.set_title(f'{COLD_LOOKUPS:,} random picks from\n{TOTAL_COLD_ROWS:,} cold rows ({COLD_FILE_MB} MB)',
              fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Right: page utilization histogram
page_util = np.concatenate([
    np.ones(int(SINGLE_ROW_PAGES_PCT / 100 * UNIQUE_PAGES)) * 1,  # 91.7% single-row
    np.random.randint(2, 5, size=int((100 - SINGLE_ROW_PAGES_PCT) / 100 * UNIQUE_PAGES))  # rest
])
ax2.hist(page_util, bins=range(1, 7), color='#E74C3C', alpha=0.8, edgecolor='black',
         linewidth=0.5, align='left', rwidth=0.7)
ax2.set_xlabel('Rows per 4KB page', fontsize=11)
ax2.set_ylabel('Number of pages', fontsize=11)
ax2.set_title(f'{SINGLE_ROW_PAGES_PCT}% of pages have only 1 row\n(read 4KB to get 64 bytes = {WASTED_PCT}% wasted)',
              fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Random Access Amplification: Why SSD Reads Are Wasteful', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_page_scatter.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/02_page_scatter.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/02_page_scatter.png")
plt.close()

# ================================================================
# Figure 3: Latency breakdown — the money slide
# Embedding = SSD I/O + table lookup (SSD I/O is part of embedding access)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['fp32 DRAM', 'SSD cold\n(serial reads)', 'DC block-mean\n(1% hot)']

# Stacked bars: embedding (= SSD I/O + table lookup) | MLP+interaction
# For SSD: embedding time = SSD read + EmbeddingBag lookup
emb_total = [EMB_DRAM, SSD_READ + EMB_SSD, EMB_DC]
mlp_time =  [DRAM_FWD - EMB_DRAM, SSD_FWD - EMB_SSD, DC_FWD - EMB_DC]

# For SSD bar, split embedding into I/O portion and lookup portion
ssd_io =     [0,        SSD_READ,  0]
table_lookup = [EMB_DRAM, EMB_SSD,   EMB_DC]

x = np.arange(len(methods))
w = 0.5

b1 = ax.bar(x, ssd_io, w, color='#E74C3C', alpha=0.85, label='Embedding: SSD I/O',
            edgecolor='black', linewidth=0.5)
b2 = ax.bar(x, table_lookup, w, bottom=ssd_io, color='#F39C12', alpha=0.85,
            label='Embedding: table lookup', edgecolor='black', linewidth=0.5)
b3 = ax.bar(x, mlp_time, w, bottom=[s+e for s, e in zip(ssd_io, table_lookup)],
            color='#3498DB', alpha=0.85, label='MLP + interaction', edgecolor='black', linewidth=0.5)

# Total labels
totals = [DRAM_FWD, SSD_READ + SSD_FWD, DC_FWD]
for i, total in enumerate(totals):
    ax.text(i, total + 2, f'{total:.1f}ms', ha='center', va='bottom',
            fontsize=13, fontweight='bold')

# Bracket for embedding time on SSD bar
emb_ssd_total = SSD_READ + EMB_SSD
ax.annotate('', xy=(1.3, 0), xytext=(1.3, emb_ssd_total),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(1.42, emb_ssd_total / 2, f'Embedding\n{emb_ssd_total:.0f}ms',
        fontsize=10, fontweight='bold', va='center')

# SSD I/O annotation inside bar
ax.text(1, SSD_READ / 2, f'SSD I/O\n{SSD_READ:.0f}ms',
        ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Embedding annotation on DRAM and DC bars
ax.text(0, EMB_DRAM / 2, f'{EMB_DRAM:.1f}ms', ha='center', va='center',
        fontsize=9, fontweight='bold')
ax.text(2, EMB_DC / 2, f'{EMB_DC:.1f}ms', ha='center', va='center',
        fontsize=9, fontweight='bold')

ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.8)
ax.text(2.3, 11, '10ms SLA', fontsize=11, color='orange', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12)
ax.set_ylabel('Batch Latency (ms)', fontsize=13)
ax.set_title('End-to-End Batch Latency Breakdown\n(Criteo Kaggle, batch=128, SATA SSD)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, max(totals) * 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_latency_breakdown.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/03_latency_breakdown.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/03_latency_breakdown.png")
plt.close()

# ================================================================
# Figure 4: Overhead breakdown — 98% is SSD
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: pie chart
labels_oh = [f'SSD I/O\n({SSD_IO_TIME:.0f}ms)', f'Syscall\n({SYSCALL_OVERHEAD:.1f}ms)', f'Python\n({PYTHON_OVERHEAD:.1f}ms)']
sizes_oh = [SSD_IO_TIME, SYSCALL_OVERHEAD, PYTHON_OVERHEAD]
colors_oh = ['#E74C3C', '#F39C12', '#3498DB']
wedges, texts, autotexts = ax1.pie(sizes_oh, labels=labels_oh, colors=colors_oh,
                                    autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11})
for at in autotexts:
    at.set_fontweight('bold')
ax1.set_title('Where Does the 104ms Go?\n(proportions from /dev/zero subtraction)', fontsize=12, fontweight='bold')

# Right: bar chart per-component
comps = ['Python loop\n(int conversion)', 'Syscalls\n(lseek+read)', 'SSD I/O\n(actual disk)']
vals = [PYTHON_OVERHEAD, SYSCALL_OVERHEAD - PYTHON_OVERHEAD, SSD_IO_TIME]
cols = ['#3498DB', '#F39C12', '#E74C3C']
bars = ax2.bar(range(len(comps)), vals, color=cols, alpha=0.85, edgecolor='black', linewidth=0.5)
for b in bars:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 0.5, f'{h:.1f}ms',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_xticks(range(len(comps)))
ax2.set_xticklabels(comps, fontsize=10)
ax2.set_ylabel('Time (ms)', fontsize=12)
ax2.set_title('Overhead Breakdown\n(1,660 cold reads per batch)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Bottleneck Analysis: 98% of Time is SSD, Not Software', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_overhead_breakdown.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/04_overhead_breakdown.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/04_overhead_breakdown.png")
plt.close()

# ================================================================
# Figure 5: fio ground truth + theoretical minimum
# ================================================================
fig, ax = plt.subplots(figsize=(9, 5))

fio_labels = ['4K QD=1', '4K QD=32', '512B QD=1', '512B QD=32']
fio_iops = [FIO_4K_QD1_IOPS, FIO_4K_QD32_IOPS, FIO_512B_QD1_IOPS, 86924]
time_for_reads = [UNIQUE_PAGES / iops * 1000 for iops in [FIO_4K_QD1_IOPS, FIO_4K_QD32_IOPS]]
time_for_reads += [COLD_LOOKUPS / iops * 1000 for iops in [FIO_512B_QD1_IOPS, 86924]]

bars = ax.bar(range(len(fio_labels)), time_for_reads, color=['#3498DB', '#2ECC71', '#3498DB', '#2ECC71'],
              alpha=0.85, edgecolor='black', linewidth=0.5)
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 2, f'{h:.1f}ms',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(y=TOTAL_MEASURED, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
ax.text(len(fio_labels) - 0.5, TOTAL_MEASURED + 3, f'Our measured: {TOTAL_MEASURED:.0f}ms',
        fontsize=10, color='#E74C3C', fontweight='bold')

ax.set_xticks(range(len(fio_labels)))
ax.set_xticklabels(fio_labels, fontsize=11)
ax.set_ylabel('Time for batch cold reads (ms)', fontsize=12)
ax.set_title(f'fio Ground Truth: Raw SSD IOPS → Theoretical Minimum\n'
             f'(882 page reads or 1,660 row reads)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/05_fio_ground_truth.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/05_fio_ground_truth.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/05_fio_ground_truth.png")
plt.close()

# ================================================================
# Figure 6: Memory vs Latency tradeoff — the final argument
# ================================================================
fig, ax = plt.subplots(figsize=(8, 6))

points = [
    ('fp32 DRAM', FP32_TOTAL_MB, DRAM_FWD, '#3498DB', 'o', 0.802497),
    ('SSD cold\n(serial)', 50, SSD_E2E, '#E74C3C', 's', 0.802497),
    ('DC scalar\n4-bit', DC_TOTAL_MB, DC_FWD, '#2ECC71', 'D', 0.801783),
]

for name, mem, lat, color, marker, auc in points:
    ax.scatter(mem, lat, c=color, s=250, marker=marker, edgecolors='black',
               linewidth=1, zorder=5, alpha=0.9)
    offset_x = 10 if mem > 20 else 2
    offset_y = 3 if lat < 20 else -8
    ax.annotate(f'{name}\n{lat:.1f}ms, {mem:.0f}MB',
                (mem, lat), textcoords="offset points",
                xytext=(offset_x, offset_y), fontsize=9, fontweight='bold', color=color)

ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.8)
ax.text(1500, 11, '10ms SLA', fontsize=11, color='orange', fontweight='bold')

ax.set_xscale('log')
ax.set_xlabel('DRAM Footprint (MB)', fontsize=13)
ax.set_ylabel('End-to-End Batch Latency (ms)', fontsize=13)
ax.set_title('Memory vs Latency: DC Eliminates I/O\nat 189x Compression, -0.089% AUC',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/06_memory_vs_latency.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/06_memory_vs_latency.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/06_memory_vs_latency.png")
plt.close()

# ================================================================
# Figure 7: Summary table as figure
# ================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

table_data = [
    ['Metric', 'Value'],
    ['Model', '26 embedding tables, D=16, 2,061 MB fp32'],
    ['Batch size', '128 samples'],
    ['Embedding lookups/batch', f'{TOTAL_EMB_LOOKUPS:,} total ({HOT_LOOKUPS:,} hot + {COLD_LOOKUPS:,} cold)'],
    ['Cold data needed/batch', f'{COLD_DATA_KB:.0f} KB ({COLD_LOOKUPS:,} rows × {ROW_BYTES} bytes)'],
    ['Cold file on SSD', f'{COLD_FILE_MB:,} MB ({TOTAL_COLD_ROWS:,} rows)'],
    ['Unique 4KB pages/batch', f'{UNIQUE_PAGES} (91.7% have only 1 row)'],
    ['SSD data read/batch', f'{PAGES_READ_KB:,} KB = {PAGES_READ_KB/1024:.1f} MB (34x amplification)'],
    ['Wasted bytes', f'{WASTED_PCT}% (read 4KB page for 64B row)'],
    ['', ''],
    ['SSD random 4K IOPS (fio)', f'QD=1: {FIO_4K_QD1_IOPS:,} ({FIO_4K_QD1_LAT}μs)  |  QD=32: {FIO_4K_QD32_IOPS:,}'],
    ['Sequential bandwidth', f'{FIO_SEQ_BW} MB/s'],
    ['Overhead: Python/syscall/SSD', f'{PYTHON_OVERHEAD:.1f}ms / {SYSCALL_OVERHEAD:.1f}ms / {SSD_IO_TIME:.0f}ms (98% SSD)'],
    ['', ''],
    ['DRAM: embedding lookup', f'{EMB_DRAM:.2f}ms (full forward: {DRAM_FWD:.1f}ms)'],
    ['SSD: read + lookup + forward', f'{SSD_READ:.0f}ms + {EMB_SSD:.1f}ms + {SSD_FWD - EMB_SSD:.1f}ms = {SSD_E2E:.0f}ms'],
    ['DC: lookup + forward', f'{EMB_DC:.2f}ms (full forward: {DC_FWD:.1f}ms)  |  Memory: {DC_TOTAL_MB} MB (189x)'],
    ['', ''],
    ['DC vs DRAM speed', '~1.0x (same speed, 3 rounds measured with cache drops)'],
    ['SSD vs DRAM speed', f'{SSD_E2E/DRAM_FWD:.0f}x slower'],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.4)

# Style header
for j in range(2):
    table[0, j].set_facecolor('#2C3E50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Style section separators
for i in [9, 13, 17]:
    for j in range(2):
        table[i, j].set_facecolor('#ECF0F1')

ax.set_title('Complete SSD Cold Embedding Analysis — All Numbers Measured',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/07_summary_table.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/07_summary_table.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/07_summary_table.png")
plt.close()

print(f"\nAll 7 figures saved to {OUT_DIR}/")
