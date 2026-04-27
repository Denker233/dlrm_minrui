#!/usr/bin/env python3
"""
Intelligent Compression Agent v2 — uses ACTUAL AUC (not MSE proxy).
Includes PCA sort (first principal component) as a sorting method.

Runs full DLRM inference for each config to get real AUC numbers.
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

# Action space
HOT_FRACTIONS = [0.005, 0.01, 0.02, 0.043]
SORT_METHODS = ['freq', 'value', 'pca', 'zero']
NBITS = [4]  # 4-bit DC means (8-bit gives negligible AUC diff, saves time)

def pca_sort_order(cold_w):
    """Sort cold rows by first principal component score."""
    sample_size = min(50000, len(cold_w))
    if sample_size < len(cold_w):
        sample_idx = np.random.RandomState(42).choice(len(cold_w), sample_size, replace=False)
        pca = PCA(n_components=1, random_state=42)
        pca.fit(cold_w[sample_idx])
    else:
        pca = PCA(n_components=1, random_state=42)
        pca.fit(cold_w)
    scores = pca.transform(cold_w).ravel()
    return np.argsort(scores)


def apply_dc_config(weight, freq, hf, sort_method, nbits):
    """
    Apply DC block-mean compression to a table and return modified weights.
    Returns (modified_weight, ratio, cold_frac).
    """
    w = weight.numpy().copy()
    n_rows, dim = w.shape
    n_hot = max(1, int(n_rows * hf))
    n_cold = n_rows - n_hot

    if n_cold == 0:
        return weight.clone(), 1.0, 0.0

    freq_np = freq.numpy() if isinstance(freq, torch.Tensor) else freq
    sorted_by_freq = np.argsort(-freq_np)
    hot_idx = sorted_by_freq[:n_hot]
    cold_idx = sorted_by_freq[n_hot:]

    cold_w = w[cold_idx]

    if sort_method == 'zero':
        # Zero all cold rows
        result = weight.clone()
        result[cold_idx] = 0.0
        orig_bytes = n_rows * dim * 4
        hot_bytes = n_hot * dim * 1 + n_rows * 0.5
        return result, orig_bytes / hot_bytes, n_cold / n_rows

    # Sort cold rows
    if sort_method == 'value':
        order = np.argsort(cold_w.mean(axis=1))
    elif sort_method == 'pca':
        order = pca_sort_order(cold_w)
    else:  # freq — already sorted by decreasing frequency
        order = np.arange(n_cold)

    cold_w_sorted = cold_w[order]

    # DC block-mean
    block_size = 16
    n_blocks = (n_cold + block_size - 1) // block_size

    padded = np.zeros((n_blocks * block_size, dim))
    padded[:n_cold] = cold_w_sorted
    blocks = padded.reshape(n_blocks, block_size, dim)
    block_means = blocks.mean(axis=1)

    # Quantize to nbits
    n_levels = 2 ** nbits
    bm_min, bm_max = block_means.min(), block_means.max()
    bm_scale = (bm_max - bm_min) / (n_levels - 1) if bm_max > bm_min else 1.0
    bm_q = np.clip(np.round((block_means - bm_min) / bm_scale), 0, n_levels - 1)
    bm_deq = bm_q * bm_scale + bm_min

    # Reconstruct
    reconstructed = np.repeat(bm_deq, block_size, axis=0)[:n_cold]

    # Un-sort: map back to original cold positions
    unsort = np.argsort(order)
    reconstructed_unsorted = reconstructed[unsort]

    # Build result
    result = weight.clone()
    result[cold_idx] = torch.from_numpy(reconstructed_unsorted).float()

    # Compression ratio
    orig_bytes = n_rows * dim * 4
    hot_bytes = n_hot * dim * 1
    cold_bytes = n_blocks * dim * (nbits / 8)
    map_bytes = n_rows * 0.5
    ratio = orig_bytes / (hot_bytes + cold_bytes + map_bytes)

    return result, ratio, n_cold / n_rows


def evaluate_auc(dlrm, test_batches, ln_emb, sd, ek, config_per_table, freq_counts):
    """
    Run full inference with given per-table configs and return AUC.
    config_per_table: dict {table_id: (hf, sort, nbits)} or single tuple for all.
    """
    # Apply configs
    for k in ek:
        t = int(k.split('.')[1])
        if t in TABLES:
            cfg = config_per_table if isinstance(config_per_table, tuple) else config_per_table[t]
            hf, sort_method, nbits = cfg
            modified_w, _, _ = apply_dc_config(sd[k], freq_counts[t], hf, sort_method, nbits)
            dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
            dlrm.emb_l[t].weight.data = modified_w
        else:
            dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
            dlrm.emb_l[t].weight.data = sd[k].clone()

    scores_all, targets_all = [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            Z = dlrm(X, lS_o, lS_i)
            scores_all.append(Z.detach().numpy().ravel())
            targets_all.append(T.numpy().ravel())

    return roc_auc_score(np.concatenate(targets_all), np.concatenate(scores_all))


def extract_features(weight, freq):
    """Extract table features for the agent."""
    w = weight.numpy().astype(np.float64)
    n_rows, dim = w.shape
    freq_np = freq.numpy().astype(np.float64)

    row_norms = np.linalg.norm(w, axis=1)
    row_means = w.mean(axis=1)

    features = {
        'log_num_rows': np.log10(n_rows + 1),
        'l2_norm_mean': row_norms.mean(),
        'l2_norm_std': row_norms.std(),
        'row_mean_std': row_means.std(),
        'freq_skewness': scipy_stats.skew(freq_np),
        'freq_gini': _gini(freq_np),
        'pct_zero_freq': (freq_np == 0).sum() / len(freq_np),
    }

    # Effective dim
    sample = min(5000, n_rows)
    idx = np.random.RandomState(42).choice(n_rows, sample, replace=False)
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
    print("INTELLIGENT COMPRESSION AGENT v2 — REAL AUC")
    print("=" * 70)

    # Load
    print("\n[1] Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    torch.set_num_threads(32)
    print(f"  {len(test_batches)} test batches")

    # Profile frequencies
    print("  Profiling access frequencies...")
    freq_counts = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for bi, (X, lS_o, lS_i, T) in enumerate(test_batches):
        for t in TABLES:
            idx = lS_i[t] if isinstance(lS_i, list) else lS_i[t]
            freq_counts[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))

    # Baseline AUC
    print("\n[2] Measuring baseline AUC...")
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        dlrm.emb_l[t].weight.data = sd[k].clone()

    scores_all, targets_all = [], []
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_batches:
            Z = dlrm(X, lS_o, lS_i)
            scores_all.append(Z.detach().numpy().ravel())
            targets_all.append(T.numpy().ravel())
    baseline_auc = roc_auc_score(np.concatenate(targets_all), np.concatenate(scores_all))
    print(f"  Baseline AUC = {baseline_auc:.6f}")

    # Extract features
    print("\n[3] Extracting table features...")
    table_features = {}
    for t in TABLES:
        table_features[t] = extract_features(sd[ek[t]], freq_counts[t])
        print(f"  T{t}: {sd[ek[t]].shape[0]:>10,} rows, "
              f"eff_dim={table_features[t]['effective_dim']:.1f}, "
              f"gini={table_features[t]['freq_gini']:.3f}")

    # ================================================================
    # [4] Sweep: uniform configs (same config for all tables)
    # ================================================================
    configs = []
    for hf in HOT_FRACTIONS:
        for sm in SORT_METHODS:
            for nb in NBITS:
                configs.append((hf, sm, nb))

    print(f"\n[4] Evaluating {len(configs)} uniform configs (real AUC)...")
    uniform_results = []

    for ci, cfg in enumerate(configs):
        hf, sm, nb = cfg
        t0 = time.time()
        auc = evaluate_auc(dlrm, test_batches, ln_emb, sd, ek, cfg, freq_counts)
        elapsed = time.time() - t0

        # Compute avg ratio
        ratios = []
        for t in TABLES:
            _, ratio, _ = apply_dc_config(sd[ek[t]], freq_counts[t], hf, sm, nb)
            ratios.append(ratio)

        result = {
            'config': cfg,
            'auc': auc,
            'auc_loss': auc - baseline_auc,
            'auc_loss_pct': (auc - baseline_auc) / baseline_auc * 100,
            'avg_ratio': np.mean(ratios),
        }
        uniform_results.append(result)

        print(f"  [{ci+1}/{len(configs)}] hf={hf}, sort={sm}, nb={nb}: "
              f"AUC={auc:.6f} ({result['auc_loss_pct']:+.3f}%), "
              f"ratio={result['avg_ratio']:.0f}x [{elapsed:.1f}s]")

    # ================================================================
    # [5] Per-table sweep: change one table at a time
    # ================================================================
    print(f"\n[5] Per-table sensitivity (one table changed, rest at baseline)...")
    per_table_results = {t: [] for t in TABLES}

    # Use a subset of configs for per-table (to save time)
    per_table_configs = []
    for hf in [0.01, 0.043]:
        for sm in SORT_METHODS:
            per_table_configs.append((hf, sm, 4))

    total_evals = len(TABLES) * len(per_table_configs)
    eval_count = 0

    for t in TABLES:
        for cfg in per_table_configs:
            hf, sm, nb = cfg
            # All tables at baseline except table t
            config_map = {}
            for t2 in TABLES:
                if t2 == t:
                    config_map[t2] = cfg
                else:
                    config_map[t2] = (1.0, 'freq', 4)  # effectively no compression

            # Restore baseline for non-target tables
            for k in ek:
                ti = int(k.split('.')[1])
                dlrm.emb_l[ti] = nn.EmbeddingBag(int(ln_emb[ti]), EMB_DIM, mode='sum', sparse=True)
                if ti == t:
                    modified_w, ratio, _ = apply_dc_config(sd[k], freq_counts[t], hf, sm, nb)
                    dlrm.emb_l[ti].weight.data = modified_w
                else:
                    dlrm.emb_l[ti].weight.data = sd[k].clone()

            scores_all, targets_all = [], []
            with torch.no_grad():
                for X, lS_o, lS_i, T in test_batches:
                    Z = dlrm(X, lS_o, lS_i)
                    scores_all.append(Z.detach().numpy().ravel())
                    targets_all.append(T.numpy().ravel())
            auc = roc_auc_score(np.concatenate(targets_all), np.concatenate(scores_all))

            per_table_results[t].append({
                'config': cfg, 'auc': auc,
                'auc_delta': auc - baseline_auc,
                'ratio': ratio,
            })

            eval_count += 1
            if eval_count % 8 == 0:
                print(f"  [{eval_count}/{total_evals}] T{t} hf={hf} sort={sm}: "
                      f"AUC={auc:.6f} ({(auc-baseline_auc)*100:+.4f}%)")

    # ================================================================
    # [6] Train agent on real AUC
    # ================================================================
    print(f"\n[6] Training agent on real AUC data...")

    # Combine uniform + per-table data
    X_train, y_train = [], []

    # From uniform results: each is a data point
    for r in uniform_results:
        hf, sm, nb = r['config']
        # Use average features across tables
        avg_feat = {}
        for key in table_features[TABLES[0]]:
            avg_feat[key] = np.mean([table_features[t][key] for t in TABLES])
        feat_vec = list(avg_feat.values()) + [
            hf, np.log10(hf),
            {'freq': 0, 'value': 1, 'pca': 2, 'zero': 3}[sm],
        ]
        X_train.append(feat_vec)
        y_train.append(r['auc_loss_pct'])  # negative = worse

    # From per-table results
    for t in TABLES:
        for r in per_table_results[t]:
            hf, sm, nb = r['config']
            feat_vec = list(table_features[t].values()) + [
                hf, np.log10(hf),
                {'freq': 0, 'value': 1, 'pca': 2, 'zero': 3}[sm],
            ]
            X_train.append(feat_vec)
            y_train.append(r['auc_delta'] / baseline_auc * 100)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_scaled, y_train)
    r2 = model.score(X_scaled, y_train)
    print(f"  Model R² = {r2:.4f}")
    print(f"  Training points: {len(y_train)} ({len(uniform_results)} uniform + "
          f"{sum(len(v) for v in per_table_results.values())} per-table)")

    feat_names = list(table_features[TABLES[0]].keys()) + ['hf', 'log_hf', 'sort_method_id']
    importances = dict(zip(feat_names, model.feature_importances_))
    print("\n  Feature importances:")
    for name, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {name:<25} {imp:.4f}")

    # ================================================================
    # [7] Generate figures
    # ================================================================
    print(f"\n[7] Generating figures...")

    # Fig 1: Sort method comparison (the key AI finding)
    fig, axes = plt.subplots(1, len(HOT_FRACTIONS), figsize=(4*len(HOT_FRACTIONS), 5), sharey=True)
    if len(HOT_FRACTIONS) == 1:
        axes = [axes]

    sort_colors = {'freq': '#3498DB', 'value': '#2ECC71', 'pca': '#E74C3C', 'zero': '#95A5A6'}

    for ax, hf in zip(axes, HOT_FRACTIONS):
        for sm in SORT_METHODS:
            r = next((x for x in uniform_results if x['config'] == (hf, sm, 4)), None)
            if r:
                bar = ax.bar(sm, r['auc_loss_pct'], color=sort_colors[sm],
                             alpha=0.85, edgecolor='black', linewidth=0.5)
                ax.text(bar[0].get_x() + bar[0].get_width()/2, r['auc_loss_pct'] - 0.005,
                        f"{r['auc_loss_pct']:.3f}%", ha='center', va='top', fontsize=8, fontweight='bold')

        ax.set_title(f'hot={hf*100:.1f}%', fontsize=11, fontweight='bold')
        ax.set_ylabel('AUC Loss (%)' if ax == axes[0] else '', fontsize=11)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Sort Method Comparison: Actual AUC Impact\n(PCA sort vs Value sort vs Freq sort vs Zero)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/sort_method_auc.png', dpi=120, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/sort_method_auc.pdf', bbox_inches='tight')
    print(f"  Saved {OUT_DIR}/sort_method_auc.png")
    plt.close()

    # Fig 2: Per-table sensitivity — which tables benefit most from intelligent config
    fig, ax = plt.subplots(figsize=(10, 6))

    table_labels = [f'T{t}' for t in TABLES]
    x = np.arange(len(TABLES))
    width = 0.2

    for si, sm in enumerate(['freq', 'value', 'pca', 'zero']):
        deltas = []
        for t in TABLES:
            r = next((x for x in per_table_results[t] if x['config'][1] == sm and x['config'][0] == 0.01), None)
            if r:
                deltas.append(r['auc_delta'] / baseline_auc * 100)
            else:
                deltas.append(0)
        ax.bar(x + si * width - 1.5*width, deltas, width,
               color=sort_colors[sm], alpha=0.85, label=sm, edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(table_labels, fontsize=11)
    ax.set_ylabel('AUC Loss (%) when compressing this table alone', fontsize=11)
    ax.set_title('Per-Table Sensitivity: Which Tables Need Intelligent Config Selection\n(1% hot, 4-bit DC)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, title='Sort method')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/per_table_sensitivity.png', dpi=120, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/per_table_sensitivity.pdf', bbox_inches='tight')
    print(f"  Saved {OUT_DIR}/per_table_sensitivity.png")
    plt.close()

    # Fig 3: Pareto frontier — ratio vs AUC loss, colored by sort method
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in uniform_results:
        hf, sm, nb = r['config']
        color = sort_colors[sm]
        marker = {0.005: 'v', 0.01: 's', 0.02: 'D', 0.043: 'o'}[hf]
        size = 150
        ax.scatter(r['avg_ratio'], -r['auc_loss_pct'], c=color, marker=marker, s=size,
                   edgecolors='black', linewidth=0.5, zorder=5, alpha=0.85)

    # Pareto frontier
    pareto_points = []
    for r in uniform_results:
        dominated = False
        for r2 in uniform_results:
            if r2['avg_ratio'] >= r['avg_ratio'] and r2['auc_loss_pct'] >= r['auc_loss_pct'] and r2 != r:
                dominated = True
                break
        if not dominated:
            pareto_points.append(r)

    pareto_points.sort(key=lambda x: x['avg_ratio'])
    if pareto_points:
        ax.plot([p['avg_ratio'] for p in pareto_points],
                [-p['auc_loss_pct'] for p in pareto_points],
                'k--', alpha=0.5, linewidth=1, label='Pareto frontier')

    # Legend
    sort_patches = [mpatches.Patch(color=sort_colors[sm], label=sm) for sm in SORT_METHODS]
    marker_legend = [plt.scatter([], [], marker=m, c='gray', s=100, label=f'hf={hf}')
                     for hf, m in [(0.005, 'v'), (0.01, 's'), (0.02, 'D'), (0.043, 'o')]]
    ax.legend(handles=sort_patches + marker_legend, fontsize=9, loc='upper right')

    ax.set_xlabel('Compression Ratio', fontsize=12)
    ax.set_ylabel('AUC Loss (%)', fontsize=12)
    ax.set_title('Compression Pareto Frontier by Sort Method\n(Actual AUC, Criteo Kaggle)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/pareto_by_sort.png', dpi=120, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/pareto_by_sort.pdf', bbox_inches='tight')
    print(f"  Saved {OUT_DIR}/pareto_by_sort.png")
    plt.close()

    # Fig 4: Feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_imp]
    vals = [f[1] for f in sorted_imp]
    ax.barh(range(len(names)), vals, color='#2ECC71', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('What the Agent Learned (Real AUC)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/feature_importance_auc.png', dpi=120, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/feature_importance_auc.pdf', bbox_inches='tight')
    print(f"  Saved {OUT_DIR}/feature_importance_auc.png")
    plt.close()

    # ================================================================
    # Save results
    # ================================================================
    output = {
        'baseline_auc': baseline_auc,
        'uniform_results': [{
            'hf': r['config'][0], 'sort': r['config'][1], 'nbits': r['config'][2],
            'auc': r['auc'], 'auc_loss_pct': r['auc_loss_pct'], 'avg_ratio': r['avg_ratio'],
        } for r in uniform_results],
        'per_table_results': {str(t): [{
            'hf': r['config'][0], 'sort': r['config'][1],
            'auc': r['auc'], 'auc_delta_pct': r['auc_delta'] / baseline_auc * 100,
            'ratio': r['ratio'],
        } for r in per_table_results[t]] for t in TABLES},
        'model_r2': r2,
        'feature_importances': {k: float(v) for k, v in importances.items()},
        'table_features': {str(t): {k: float(v) for k, v in table_features[t].items()} for t in TABLES},
    }
    with open(f'{OUT_DIR}/agent_auc_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {OUT_DIR}/agent_auc_results.json")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — Sort Method Ranking by AUC (uniform, 4-bit DC)")
    print(f"{'='*70}")
    print(f"  Baseline AUC: {baseline_auc:.6f}")
    print(f"\n  {'HF':<8} {'Sort':<8} {'AUC':>10} {'Loss%':>10} {'Ratio':>8}")
    print(f"  {'-'*46}")
    for r in sorted(uniform_results, key=lambda x: (x['config'][0], -x['auc'])):
        hf, sm, nb = r['config']
        print(f"  {hf:<8} {sm:<8} {r['auc']:>10.6f} {r['auc_loss_pct']:>+9.4f}% {r['avg_ratio']:>7.0f}x")

    # Find best sort method at each hf
    print(f"\n  Best sort method per hot fraction:")
    for hf in HOT_FRACTIONS:
        hf_results = [r for r in uniform_results if r['config'][0] == hf]
        best = max(hf_results, key=lambda x: x['auc'])
        print(f"    hf={hf}: {best['config'][1]} (AUC loss {best['auc_loss_pct']:+.4f}%)")


if __name__ == '__main__':
    main()
