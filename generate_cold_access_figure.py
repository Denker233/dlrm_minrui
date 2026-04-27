#!/usr/bin/env python3
"""
Count how many samples per batch touch cold embeddings and how many cold
lookups occur per batch. Generate distribution graphs.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

from codec_ondemand_benchmark import load_model_and_data, HOTCOLD_DIR

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
OUT_DIR = 'results/ssd_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Load model and data
print("Loading model and test data...")
dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
test_batches = list(test_ld)
print(f"  {len(test_batches)} test batches, batch_size=128")

# Load hot/cold masks
is_hot = {}
for t in TABLES:
    is_hot[t] = torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True)

# Count per-batch cold access statistics
print("Counting cold accesses per batch...")

cold_lookups_per_batch = []      # total cold row lookups per batch (across all tables)
cold_samples_per_batch = []      # number of samples that touch >= 1 cold row
cold_tables_per_batch = []       # number of tables with cold accesses per batch
per_table_cold_per_batch = {t: [] for t in TABLES}  # cold lookups per table per batch

for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_batches):
    batch_cold_total = 0
    batch_tables_with_cold = 0

    # Track which samples touch cold rows (across all tables)
    batch_size = T.shape[0]
    sample_touches_cold = np.zeros(batch_size, dtype=bool)

    for t in TABLES:
        # Get indices for this table
        if isinstance(lS_i, list):
            indices = lS_i[t]
        else:
            indices = lS_i[t]

        if isinstance(lS_o, list):
            offsets = lS_o[t]
        else:
            offsets = lS_o[t]

        # Check which indices are cold
        cold_mask = ~is_hot[t][indices]
        n_cold = cold_mask.sum().item()
        batch_cold_total += n_cold
        per_table_cold_per_batch[t].append(n_cold)

        if n_cold > 0:
            batch_tables_with_cold += 1

            # Map cold lookups back to samples
            # offsets[i] to offsets[i+1] are the indices for sample i
            offsets_np = offsets.numpy()
            cold_mask_np = cold_mask.numpy()
            for s in range(batch_size):
                start = offsets_np[s]
                end = offsets_np[s + 1] if s + 1 < len(offsets_np) else len(indices)
                if cold_mask_np[start:end].any():
                    sample_touches_cold[s] = True

    cold_lookups_per_batch.append(batch_cold_total)
    cold_samples_per_batch.append(sample_touches_cold.sum())
    cold_tables_per_batch.append(batch_tables_with_cold)

cold_lookups_per_batch = np.array(cold_lookups_per_batch)
cold_samples_per_batch = np.array(cold_samples_per_batch)
cold_tables_per_batch = np.array(cold_tables_per_batch)

print(f"\n--- Cold Access Statistics ({len(test_batches)} batches) ---")
print(f"  Cold lookups/batch:  mean={cold_lookups_per_batch.mean():.0f}, "
      f"min={cold_lookups_per_batch.min()}, max={cold_lookups_per_batch.max()}, "
      f"p50={np.percentile(cold_lookups_per_batch, 50):.0f}, "
      f"p99={np.percentile(cold_lookups_per_batch, 99):.0f}")
print(f"  Samples touching cold/batch: mean={cold_samples_per_batch.mean():.0f}/{batch_size} "
      f"({cold_samples_per_batch.mean()/batch_size*100:.1f}%), "
      f"min={cold_samples_per_batch.min()}, max={cold_samples_per_batch.max()}")
print(f"  Tables with cold/batch: mean={cold_tables_per_batch.mean():.1f}/{len(TABLES)}")

for t in TABLES:
    arr = np.array(per_table_cold_per_batch[t])
    pct_batches_with_cold = (arr > 0).sum() / len(arr) * 100
    print(f"  Table {t:>2}: cold lookups mean={arr.mean():.0f}, "
          f"batches with cold={pct_batches_with_cold:.0f}%")

# ================================================================
# Figure: Combined cold access visualization
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Samples touching cold per batch (histogram)
ax = axes[0]
ax.hist(cold_samples_per_batch, bins=30, color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axvline(cold_samples_per_batch.mean(), color='black', linestyle='--', linewidth=2,
           label=f'mean = {cold_samples_per_batch.mean():.0f}/{batch_size}')
ax.set_xlabel('Samples Touching Cold Rows', fontsize=12)
ax.set_ylabel('Number of Batches', fontsize=12)
ax.set_title(f'Samples with Cold Access per Batch\n({cold_samples_per_batch.mean()/batch_size*100:.0f}% of samples touch cold rows)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Panel 2: Total cold lookups per batch (histogram)
ax = axes[1]
ax.hist(cold_lookups_per_batch, bins=30, color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axvline(cold_lookups_per_batch.mean(), color='black', linestyle='--', linewidth=2,
           label=f'mean = {cold_lookups_per_batch.mean():.0f}')
ax.set_xlabel('Cold Row Lookups per Batch', fontsize=12)
ax.set_ylabel('Number of Batches', fontsize=12)
ax.set_title(f'Cold Embedding Lookups per Batch\n(each = 1 SSD read if on disk)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Panel 3: Per-table cold lookups (box plot)
ax = axes[2]
table_data = [np.array(per_table_cold_per_batch[t]) for t in TABLES]
table_labels = [f'T{t}' for t in TABLES]
bp = ax.boxplot(table_data, labels=table_labels, patch_artist=True,
                boxprops=dict(facecolor='#E74C3C', alpha=0.6),
                medianprops=dict(color='black', linewidth=2))
ax.set_xlabel('Embedding Table', fontsize=12)
ax.set_ylabel('Cold Lookups per Batch', fontsize=12)
ax.set_title('Cold Lookups by Table\n(per batch distribution)', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Cold Embedding Access Patterns (Criteo Kaggle, 4.3% hot, batch=128)',
             fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cold_access_distribution.png', dpi=120, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/cold_access_distribution.pdf', bbox_inches='tight')
print(f"\nSaved {OUT_DIR}/cold_access_distribution.png")
plt.close()

# ================================================================
# Figure 2: SSD I/O implications
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5))

# If each cold lookup = 1 random SSD read at ~0.1ms (4KB aligned) or ~0.05ms (sequential)
# SATA SSD random read IOPS ~ 80-100K → ~0.01ms/read best case
# But our bench shows 84ms for a batch → ~0.05ms/read effective
read_latency_us = 50  # ~50μs per random read (measured from bench_ssd_cold.py)

estimated_ssd_time = cold_lookups_per_batch * read_latency_us / 1000  # ms

ax.hist(estimated_ssd_time, bins=30, color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.5,
        label='SSD read time per batch')
ax.axvline(estimated_ssd_time.mean(), color='black', linestyle='--', linewidth=2,
           label=f'mean = {estimated_ssd_time.mean():.0f}ms')
ax.axvline(10, color='orange', linestyle='--', linewidth=2, alpha=0.8,
           label='10ms SLA')

pct_over_sla = (estimated_ssd_time > 10).sum() / len(estimated_ssd_time) * 100
ax.set_xlabel('Estimated SSD Read Time per Batch (ms)', fontsize=12)
ax.set_ylabel('Number of Batches', fontsize=12)
ax.set_title(f'SSD Read Latency from Cold Accesses\n'
             f'({pct_over_sla:.0f}% of batches exceed 10ms SLA from disk I/O alone)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cold_access_ssd_impact.png', dpi=120, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/cold_access_ssd_impact.pdf', bbox_inches='tight')
print(f"Saved {OUT_DIR}/cold_access_ssd_impact.png")
plt.close()

# Save stats
stats = {
    'num_batches': len(test_batches),
    'batch_size': int(batch_size),
    'hot_fraction': 0.043,
    'cold_lookups_per_batch': {
        'mean': float(cold_lookups_per_batch.mean()),
        'min': int(cold_lookups_per_batch.min()),
        'max': int(cold_lookups_per_batch.max()),
        'p50': float(np.percentile(cold_lookups_per_batch, 50)),
        'p99': float(np.percentile(cold_lookups_per_batch, 99)),
    },
    'cold_samples_per_batch': {
        'mean': float(cold_samples_per_batch.mean()),
        'pct_of_batch': float(cold_samples_per_batch.mean() / batch_size * 100),
        'min': int(cold_samples_per_batch.min()),
        'max': int(cold_samples_per_batch.max()),
    },
    'cold_tables_per_batch_mean': float(cold_tables_per_batch.mean()),
    'per_table': {str(t): {
        'mean_cold_lookups': float(np.array(per_table_cold_per_batch[t]).mean()),
        'pct_batches_with_cold': float((np.array(per_table_cold_per_batch[t]) > 0).sum() / len(test_batches) * 100),
    } for t in TABLES},
}
with open(f'{OUT_DIR}/cold_access_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print(f"Saved {OUT_DIR}/cold_access_stats.json")
