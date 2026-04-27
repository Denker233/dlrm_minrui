#!/usr/bin/env python3
"""
Intelligent Compression Agent v3 — Block Size Selection per Table.

Key idea: different tables have different optimal block sizes for DC block-mean.
- Small blocks (4-8): fine-grained means, less quality loss, less compression
- Large blocks (64-256): coarse means, more compression, more quality loss
- Tables with uniform cold rows → large blocks OK
- Tables with high-variance cold rows → need small blocks

The agent learns which table properties predict the optimal block size.
Uses PCA sort (best from v2) and real AUC measurements.
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

from codec_ondemand_benchmark import load_model_and_data, MODEL_PATH

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
EMB_DIM = 16
OUT_DIR = 'results/intelligent_agent'
os.makedirs(OUT_DIR, exist_ok=True)

# Action space: block_size is the key variable
BLOCK_SIZES = [4, 8, 16, 32, 64, 128, 256]
HOT_FRACTIONS = [0.01, 0.043]
SORT_METHOD = 'pca'  # Fixed to PCA (best from v2)


def pca_sort_order(cold_w):
    """Sort cold rows by first principal component score."""
    sample_size = min(50000, len(cold_w))
    if sample_size < len(cold_w):
        idx = np.random.RandomState(42).choice(len(cold_w), sample_size, replace=False)
        pca = PCA(n_components=1, random_state=42)
        pca.fit(cold_w[idx])
    else:
        pca = PCA(n_components=1, random_state=42)
        pca.fit(cold_w)
    return np.argsort(pca.transform(cold_w).ravel())


def apply_dc_config(weight, freq, hf, block_size):
    """Apply DC block-mean with given block_size. Uses PCA sort, 4-bit quantization."""
    w = weight.numpy().copy()
    n_rows, dim = w.shape
    n_hot = max(1, int(n_rows * hf))
    n_cold = n_rows - n_hot

    if n_cold == 0:
        return weight.clone(), 1.0

    freq_np = freq.numpy() if isinstance(freq, torch.Tensor) else freq
    sorted_by_freq = np.argsort(-freq_np)
    cold_idx = sorted_by_freq[n_hot:]
    cold_w = w[cold_idx]

    # PCA sort
    order = pca_sort_order(cold_w)
    cold_w_sorted = cold_w[order]

    # DC block-mean with variable block_size
    n_blocks = (n_cold + block_size - 1) // block_size
    padded = np.zeros((n_blocks * block_size, dim))
    padded[:n_cold] = cold_w_sorted
    blocks = padded.reshape(n_blocks, block_size, dim)
    block_means = blocks.mean(axis=1)

    # 4-bit quantization of block means
    n_levels = 16
    bm_min, bm_max = block_means.min(), block_means.max()
    bm_scale = (bm_max - bm_min) / (n_levels - 1) if bm_max > bm_min else 1.0
    bm_q = np.clip(np.round((block_means - bm_min) / bm_scale), 0, n_levels - 1)
    bm_deq = bm_q * bm_scale + bm_min

    # Reconstruct
    reconstructed = np.repeat(bm_deq, block_size, axis=0)[:n_cold]

    # Un-sort
    unsort = np.argsort(order)
    reconstructed_unsorted = reconstructed[unsort]

    result = weight.clone()
    result[cold_idx] = torch.from_numpy(reconstructed_unsorted).float()

    # Compression ratio
    orig_bytes = n_rows * dim * 4
    hot_bytes = n_hot * dim * 1
    cold_bytes = n_blocks * dim * 0.5  # 4-bit
    map_bytes = n_rows * 0.5
    ratio = orig_bytes / (hot_bytes + cold_bytes + map_bytes)

    return result, ratio


def extract_features(weight, freq):
    """Extract table features."""
    w = weight.numpy().astype(np.float64)
    n_rows, dim = w.shape
    freq_np = freq.numpy().astype(np.float64)

    row_norms = np.linalg.norm(w, axis=1)
    row_means = w.mean(axis=1)
    row_vars = w.var(axis=1)

    # Sort by freq, get cold rows
    sorted_by_freq = np.argsort(-freq_np)

    features = {
        'log_num_rows': np.log10(n_rows + 1),
        'l2_norm_mean': row_norms.mean(),
        'l2_norm_std': row_norms.std(),
        'l2_norm_cv': row_norms.std() / (row_norms.mean() + 1e-8),
        'row_mean_std': row_means.std(),
        'row_var_mean': row_vars.mean(),
        'row_var_std': row_vars.std(),
        'freq_skewness': scipy_stats.skew(freq_np),
        'freq_kurtosis': scipy_stats.kurtosis(freq_np),
        'freq_gini': _gini(freq_np),
        'pct_zero_freq': (freq_np == 0).sum() / len(freq_np),
    }

    # Cold-specific features (at hf=4.3%)
    n_hot = max(1, int(n_rows * 0.043))
    cold_idx = sorted_by_freq[n_hot:]
    cold_w = w[cold_idx]
    cold_norms = np.linalg.norm(cold_w, axis=1)
    cold_means = cold_w.mean(axis=1)

    features['cold_l2_mean'] = cold_norms.mean()
    features['cold_l2_std'] = cold_norms.std()
    features['cold_mean_std'] = cold_means.std()
    # Smoothness after PCA sort: how similar are adjacent rows?
    if len(cold_w) > 1000:
        sample = np.random.RandomState(42).choice(len(cold_w), 1000, replace=False)
        sample_w = cold_w[sample]
    else:
        sample_w = cold_w
    pca_order = pca_sort_order(sample_w)
    sorted_sample = sample_w[pca_order]
    diffs = np.diff(sorted_sample, axis=0)
    features['pca_sorted_diff_mean'] = np.abs(diffs).mean()
    features['pca_sorted_diff_std'] = np.abs(diffs).std()

    # Effective dimensionality
    sample_size = min(5000, n_rows)
    idx = np.random.RandomState(42).choice(n_rows, sample_size, replace=False)
    try:
        _, s, _ = np.linalg.svd(w[idx] - w[idx].mean(0), full_matrices=False)
        s2 = (s**2) / (s**2).sum()
        features['effective_dim'] = 1.0 / (s2**2).sum()
        features['top1_var'] = s2[0]
    except:
        features['effective_dim'] = dim
        features['top1_var'] = 1.0 / dim

    return features


def _gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if n == 0 or x.sum() == 0: return 0
    return (2 * np.sum(np.arange(1, n+1) * x) - (n+1) * x.sum()) / (n * x.sum())


def main():
    print("=" * 70)
    print("INTELLIGENT AGENT v3 — BLOCK SIZE SELECTION (Real AUC)")
    print("=" * 70)

    # Load
    print("\n[1] Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    torch.set_num_threads(32)

    # Profile frequencies
    print("  Profiling...")
    freq_counts = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, lS_o, lS_i, T in test_batches:
        for t in TABLES:
            idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
            freq_counts[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))

    # Baseline
    print("\n[2] Baseline AUC...")
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        dlrm.emb_l[t].weight.data = sd[k].clone()
    scores, targets = [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            Z = dlrm(X, lS_o, lS_i)
            scores.append(Z.detach().numpy().ravel())
            targets.append(T.numpy().ravel())
    baseline_auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"  Baseline AUC = {baseline_auc:.6f}")

    # Features
    print("\n[3] Extracting table features...")
    table_features = {}
    for t in TABLES:
        table_features[t] = extract_features(sd[ek[t]], freq_counts[t])
        print(f"  T{t}: {sd[ek[t]].shape[0]:>10,} rows, eff_dim={table_features[t]['effective_dim']:.1f}, "
              f"cold_l2={table_features[t]['cold_l2_mean']:.5f}, "
              f"pca_diff={table_features[t]['pca_sorted_diff_mean']:.5f}")

    # ================================================================
    # [4] Per-table block size sweep with real AUC
    # ================================================================
    n_configs = len(TABLES) * len(BLOCK_SIZES) * len(HOT_FRACTIONS)
    print(f"\n[4] Per-table block size sweep: {len(TABLES)} tables × {len(BLOCK_SIZES)} blocks × "
          f"{len(HOT_FRACTIONS)} hf = {n_configs} configs (real AUC)...")

    per_table_results = {t: [] for t in TABLES}
    eval_count = 0

    for t in TABLES:
        for hf in HOT_FRACTIONS:
            for bs in BLOCK_SIZES:
                # Modify only table t, rest at baseline
                for k in ek:
                    ti = int(k.split('.')[1])
                    dlrm.emb_l[ti] = nn.EmbeddingBag(int(ln_emb[ti]), EMB_DIM, mode='sum', sparse=True)
                    if ti == t:
                        modified_w, ratio = apply_dc_config(sd[k], freq_counts[t], hf, bs)
                        dlrm.emb_l[ti].weight.data = modified_w
                    else:
                        dlrm.emb_l[ti].weight.data = sd[k].clone()

                scores, targets = [], []
                with torch.no_grad():
                    for X, lS_o, lS_i, T in test_batches:
                        Z = dlrm(X, lS_o, lS_i)
                        scores.append(Z.detach().numpy().ravel())
                        targets.append(T.numpy().ravel())
                auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))

                per_table_results[t].append({
                    'hf': hf, 'block_size': bs, 'auc': auc,
                    'auc_delta': auc - baseline_auc,
                    'auc_loss_pct': (auc - baseline_auc) / baseline_auc * 100,
                    'ratio': ratio,
                })

                eval_count += 1
                if eval_count % 14 == 0 or eval_count == n_configs:
                    print(f"  [{eval_count}/{n_configs}] T{t} hf={hf} bs={bs}: "
                          f"AUC={auc:.6f} ({(auc-baseline_auc)/baseline_auc*100:+.4f}%) ratio={ratio:.0f}x")

    # Print best block size per table
    print(f"\n  Best block size per table (hf=0.01):")
    for t in TABLES:
        results_hf1 = [r for r in per_table_results[t] if r['hf'] == 0.01]
        best = max(results_hf1, key=lambda x: x['auc'])
        worst = min(results_hf1, key=lambda x: x['auc'])
        print(f"    T{t:>2}: best bs={best['block_size']:>3} (loss {best['auc_loss_pct']:+.4f}%), "
              f"worst bs={worst['block_size']:>3} (loss {worst['auc_loss_pct']:+.4f}%), "
              f"spread={abs(best['auc_loss_pct'] - worst['auc_loss_pct']):.4f}%")

    # ================================================================
    # [5] Train agent: table features → optimal block size
    # ================================================================
    print(f"\n[5] Training agent...")

    X_train, y_train = [], []
    feat_names = None
    for t in TABLES:
        for r in per_table_results[t]:
            feat_vec = list(table_features[t].values()) + [r['hf'], np.log10(r['hf']), np.log2(r['block_size'])]
            X_train.append(feat_vec)
            y_train.append(r['auc_loss_pct'])
            if feat_names is None:
                feat_names = list(table_features[t].keys()) + ['hf', 'log_hf', 'log2_block_size']

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_scaled, y_train)
    r2 = model.score(X_scaled, y_train)
    print(f"  R² = {r2:.4f} on {len(y_train)} observations")

    importances = dict(zip(feat_names, model.feature_importances_))
    print("\n  Feature importances:")
    for name, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {name:<25} {imp:.4f}")

    # Agent picks best block_size per table
    print(f"\n  Agent's per-table block size (hf=0.01):")
    agent_configs = {}
    uniform_auc_loss = 0
    agent_auc_loss = 0

    # Uniform: best single block_size across all tables
    best_uniform_bs = None
    best_uniform_total = -np.inf
    for bs in BLOCK_SIZES:
        total = sum(max((r['auc'] for r in per_table_results[t] if r['block_size'] == bs and r['hf'] == 0.01), default=0)
                    for t in TABLES)
        if total > best_uniform_total:
            best_uniform_total = total
            best_uniform_bs = bs

    for t in TABLES:
        # Agent picks: predict AUC for each block size, pick best
        best_bs = None
        best_pred_auc = -np.inf
        for bs in BLOCK_SIZES:
            feat_vec = list(table_features[t].values()) + [0.01, np.log10(0.01), np.log2(bs)]
            x = scaler.transform([feat_vec])
            pred_loss = model.predict(x)[0]
            if -pred_loss > best_pred_auc:  # less negative = better
                best_pred_auc = -pred_loss
                best_bs = bs
        agent_configs[t] = best_bs

        # Actual AUC for agent's pick
        agent_r = next(r for r in per_table_results[t] if r['block_size'] == best_bs and r['hf'] == 0.01)
        uniform_r = next(r for r in per_table_results[t] if r['block_size'] == best_uniform_bs and r['hf'] == 0.01)
        oracle_r = max((r for r in per_table_results[t] if r['hf'] == 0.01), key=lambda x: x['auc'])

        agent_auc_loss += agent_r['auc_loss_pct']
        uniform_auc_loss += uniform_r['auc_loss_pct']

        print(f"    T{t:>2}: agent=bs{best_bs:>3} ({agent_r['auc_loss_pct']:+.4f}%), "
              f"uniform=bs{best_uniform_bs} ({uniform_r['auc_loss_pct']:+.4f}%), "
              f"oracle=bs{oracle_r['block_size']:>3} ({oracle_r['auc_loss_pct']:+.4f}%)")

    print(f"\n  Total AUC loss (sum over 8 tables, hf=0.01):")
    print(f"    Uniform (bs={best_uniform_bs}): {uniform_auc_loss:+.4f}%")
    print(f"    Agent (per-table):    {agent_auc_loss:+.4f}%")
    improvement = uniform_auc_loss - agent_auc_loss
    print(f"    Agent saves:          {improvement:.4f}% total AUC loss")

    # ================================================================
    # [6] Figures
    # ================================================================
    print(f"\n[6] Generating figures...")

    # Fig 1: Block size vs AUC loss per table (the key figure)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for idx, t in enumerate(TABLES):
        ax = axes[idx]
        for hf in HOT_FRACTIONS:
            results = [r for r in per_table_results[t] if r['hf'] == hf]
            results.sort(key=lambda x: x['block_size'])
            bs_list = [r['block_size'] for r in results]
            losses = [r['auc_loss_pct'] for r in results]
            color = '#E74C3C' if hf == 0.01 else '#3498DB'
            ax.plot(bs_list, losses, 'o-', color=color, linewidth=2, markersize=6,
                    label=f'hf={hf}', alpha=0.85)

        # Mark agent's choice
        if t in agent_configs:
            agent_bs = agent_configs[t]
            agent_r = next(r for r in per_table_results[t] if r['block_size'] == agent_bs and r['hf'] == 0.01)
            ax.scatter([agent_bs], [agent_r['auc_loss_pct']], c='gold', s=200, marker='*',
                       zorder=10, edgecolors='black', linewidth=1, label='Agent pick')

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Block Size', fontsize=9)
        ax.set_ylabel('AUC Loss (%)', fontsize=9)
        n_rows = sd[ek[t]].shape[0]
        ax.set_title(f'Table {t} ({n_rows:,} rows)\neff_dim={table_features[t]["effective_dim"]:.1f}',
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(BLOCK_SIZES)
        ax.set_xticklabels([str(b) for b in BLOCK_SIZES], fontsize=8)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Block Size vs AUC Loss per Table (PCA sort, 4-bit DC)\n'
                 'Different tables have different optimal block sizes',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/blocksize_per_table.png', dpi=120, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/blocksize_per_table.pdf', bbox_inches='tight')
    print(f"  Saved {OUT_DIR}/blocksize_per_table.png")
    plt.close()

    # Fig 2: Block size vs ratio vs AUC (Pareto, all tables overlaid)
    fig, ax = plt.subplots(figsize=(8, 6))
    table_colors = plt.cm.Set2(np.linspace(0, 1, len(TABLES)))
    for idx, t in enumerate(TABLES):
        results = [r for r in per_table_results[t] if r['hf'] == 0.01]
        results.sort(key=lambda x: x['block_size'])
        ratios = [r['ratio'] for r in results]
        losses = [-r['auc_loss_pct'] for r in results]
        ax.plot(ratios, losses, 'o-', color=table_colors[idx], linewidth=1.5,
                markersize=5, label=f'T{t}', alpha=0.8)
        # Label block sizes on first and last point
        ax.annotate(f'bs={results[0]["block_size"]}', (ratios[0], losses[0]),
                    fontsize=7, color=table_colors[idx])
        ax.annotate(f'bs={results[-1]["block_size"]}', (ratios[-1], losses[-1]),
                    fontsize=7, color=table_colors[idx])

    ax.set_xlabel('Compression Ratio', fontsize=12)
    ax.set_ylabel('AUC Loss (%)', fontsize=12)
    ax.set_title('Block Size Pareto: Ratio vs AUC Loss per Table\n(hf=1%, PCA sort)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/blocksize_pareto.png', dpi=120, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/blocksize_pareto.pdf', bbox_inches='tight')
    print(f"  Saved {OUT_DIR}/blocksize_pareto.png")
    plt.close()

    # Fig 3: Feature importance for block size prediction
    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_imp]
    vals = [f[1] for f in sorted_imp]
    colors_fi = ['#E74C3C' if 'block' in n or 'hf' in n else '#2ECC71' for n in names]
    ax.barh(range(len(names)), vals, color=colors_fi, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('What Determines Optimal Block Size?\n(red = action features, green = table features)',
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/blocksize_importance.png', dpi=120, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/blocksize_importance.pdf', bbox_inches='tight')
    print(f"  Saved {OUT_DIR}/blocksize_importance.png")
    plt.close()

    # Fig 4: Agent vs uniform comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TABLES))
    width = 0.3

    uniform_losses = []
    agent_losses = []
    oracle_losses = []
    for t in TABLES:
        ur = next(r for r in per_table_results[t] if r['block_size'] == best_uniform_bs and r['hf'] == 0.01)
        uniform_losses.append(ur['auc_loss_pct'])
        ar = next(r for r in per_table_results[t] if r['block_size'] == agent_configs[t] and r['hf'] == 0.01)
        agent_losses.append(ar['auc_loss_pct'])
        orc = max((r for r in per_table_results[t] if r['hf'] == 0.01), key=lambda x: x['auc'])
        oracle_losses.append(orc['auc_loss_pct'])

    ax.bar(x - width, uniform_losses, width, color='#3498DB', alpha=0.85, label=f'Uniform (bs={best_uniform_bs})',
           edgecolor='black', linewidth=0.5)
    ax.bar(x, agent_losses, width, color='#E74C3C', alpha=0.85, label='Agent (per-table bs)',
           edgecolor='black', linewidth=0.5)
    ax.bar(x + width, oracle_losses, width, color='#F39C12', alpha=0.85, label='Oracle',
           edgecolor='black', linewidth=0.5)

    # Annotate agent's block size choice
    for i, t in enumerate(TABLES):
        ax.text(i, agent_losses[i] - 0.002, f'bs={agent_configs[t]}', ha='center', va='top',
                fontsize=8, fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f'T{t}' for t in TABLES], fontsize=11)
    ax.set_ylabel('AUC Loss (%)', fontsize=12)
    ax.set_title('Per-Table Block Size: Agent vs Uniform vs Oracle\n(hf=1%, PCA sort, 4-bit DC)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/blocksize_agent_vs_uniform.png', dpi=120, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/blocksize_agent_vs_uniform.pdf', bbox_inches='tight')
    print(f"  Saved {OUT_DIR}/blocksize_agent_vs_uniform.png")
    plt.close()

    # Save results
    output = {
        'baseline_auc': baseline_auc,
        'best_uniform_block_size': best_uniform_bs,
        'agent_block_sizes': {str(t): agent_configs[t] for t in TABLES},
        'uniform_total_loss_pct': uniform_auc_loss,
        'agent_total_loss_pct': agent_auc_loss,
        'agent_improvement_pct': improvement,
        'model_r2': r2,
        'feature_importances': {k: float(v) for k, v in importances.items()},
        'per_table_results': {str(t): [{
            'hf': r['hf'], 'block_size': r['block_size'],
            'auc': r['auc'], 'auc_loss_pct': r['auc_loss_pct'], 'ratio': r['ratio'],
        } for r in per_table_results[t]] for t in TABLES},
    }
    with open(f'{OUT_DIR}/blocksize_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {OUT_DIR}/blocksize_results.json")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline AUC: {baseline_auc:.6f}")
    print(f"  Uniform (bs={best_uniform_bs}): {uniform_auc_loss:+.4f}% total loss")
    print(f"  Agent (per-table):     {agent_auc_loss:+.4f}% total loss")
    print(f"  Agent saves: {improvement:.4f}% AUC loss")
    print(f"  Model R² = {r2:.4f}")
    print(f"\n  Per-table optimal block sizes:")
    for t in TABLES:
        oracle = max((r for r in per_table_results[t] if r['hf'] == 0.01), key=lambda x: x['auc'])
        print(f"    T{t:>2} ({sd[ek[t]].shape[0]:>10,} rows): "
              f"agent=bs{agent_configs[t]:>3}, oracle=bs{oracle['block_size']:>3}")


if __name__ == '__main__':
    main()
