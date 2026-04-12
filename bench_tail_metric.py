#!/usr/bin/env python3
"""
Tail-metric experiment: does DC preserve long-tail differentiation better than pruning?

Methodology:
  - Profile access frequency on test_ld (same as bench_kaggle_dc.py)
  - For each test sample, compute "coldness" = number of large-table features
    (out of 8) whose row is in the cold tier at 2% hot fraction
  - Coldness ranges 0 (all features hot, "head sample") to 8 (all features cold,
    "deep tail sample")
  - Run inference with three configs:
      (1) fp32 baseline
      (2) DC rpb=16 @ 2% hot (our headline)
      (3) Pruning 98% sp (matched compression)
  - Collect per-sample scores and labels
  - Compute AUC stratified by coldness bucket for each config
  - Compare: does pruning collapse on high-coldness samples while DC holds?
"""
import os, sys, json, gc
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

MODEL_PATH = 'models/dlrm_kaggle_correct.pt'
LARGE_THRESHOLD = 50000
EMB_DIM = 16
HOT_FRACTION = 0.02  # 2% hot fraction for the matched comparison
ROWS_PER_BLOCK = 16  # DC rpb=16

def quantize_table(w):
    mn = w.min().item(); mx = w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp

def load_kaggle():
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net
    from codec_ondemand_benchmark import create_args
    args = create_args()
    print("[KG] Loading dataset...")
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    ln_emb = np.array(train_data.counts)
    m_spa = args.arch_sparse_feature_size
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + args.arch_mlp_top, dtype=int, sep="-")
    dlrm = DLRM_Net(m_spa, ln_emb, ln_bot, ln_top,
                    arch_interaction_op="dot", arch_interaction_itself=False,
                    sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    return dlrm, test_ld, ln_emb

def main():
    print("=" * 80)
    print("TAIL-METRIC EXPERIMENT: stratified AUC by coldness")
    print("=" * 80)

    dlrm, test_ld, ln_emb = load_kaggle()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]
    torch.set_num_threads(32)
    print(f"D={EMB_DIM}, {nt} tables, {len(TABLES)} large: {TABLES}")

    print("Caching test batches...")
    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    print(f"Cached {len(test_batches)} test batches")

    print("Profiling frequencies on test_ld...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    sorted_idx = {t: torch.argsort(freq[t], descending=True) for t in TABLES}

    # Build hot-set mask per large table at 2% hot fraction
    print(f"Building hot-set mask at {HOT_FRACTION*100:.1f}% hot...")
    is_hot = {}
    for t in TABLES:
        n = int(ln_emb[t])
        n_hot = int(n * HOT_FRACTION)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[sorted_idx[t][:n_hot]] = True
        is_hot[t] = mask
        print(f"  Table {t}: {n} rows, {n_hot} hot ({n_hot/n*100:.2f}%)")

    # Compute coldness per test sample: count of large-table features whose row is cold
    print("Computing per-sample coldness...")
    coldness_per_batch = []
    labels_per_batch = []
    for X, o, i, T in test_batches:
        batch_size = T.shape[0]
        coldness = torch.zeros(batch_size, dtype=torch.int32)
        for t in TABLES:
            idx = (i[t] if isinstance(i, (list, tuple)) else i[t]).long()
            # Each sample has possibly multiple indices per table (bag of embeddings).
            # To get a per-sample flag, we check if ANY of the indices for this sample
            # in this table is cold. For simplicity, use offsets to split indices.
            offsets = o[t] if isinstance(o, (list, tuple)) else o[t]
            # offsets has length batch_size, pointing to start of each sample's indices
            # For the last sample, the end is len(idx)
            off_np = offsets.long().numpy() if hasattr(offsets, 'numpy') else np.array(offsets)
            idx_np = idx.numpy()
            sample_cold = np.zeros(batch_size, dtype=bool)
            for s in range(batch_size):
                start = int(off_np[s])
                end = int(off_np[s+1]) if s+1 < batch_size else len(idx_np)
                if end > start:
                    # Sample is cold in table t if any of its rows is cold
                    sample_rows = idx_np[start:end]
                    any_cold = not is_hot[t][sample_rows].any().item()
                    if any_cold:
                        sample_cold[s] = True
            coldness += torch.from_numpy(sample_cold.astype(np.int32))
        coldness_per_batch.append(coldness.numpy())
        labels_per_batch.append(T.numpy().ravel())
    all_coldness = np.concatenate(coldness_per_batch)
    all_labels = np.concatenate(labels_per_batch)
    print(f"Coldness distribution (over {len(all_coldness)} test samples):")
    for c in range(9):
        n = int((all_coldness == c).sum())
        pct = n / len(all_coldness) * 100
        print(f"  coldness={c}: {n:8d} samples ({pct:5.2f}%)")

    def run_config(modifier_name, modifier_fn):
        """modifier_fn(w, t) modifies weights in place for table t."""
        with torch.no_grad():
            for k in ek:
                t = int(k.split('.')[1])
                dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
                w = sd[k].clone()
                if t in TABLES:
                    w = modifier_fn(w, t)
                dlrm.emb_l[t].weight.data = w
        scores_list = []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                Z = dlrm(X, o, i); scores_list.append(Z.numpy().ravel())
        scores = np.concatenate(scores_list)
        # Overall AUC
        overall = roc_auc_score(all_labels, scores)
        # Stratified AUC
        strat = {}
        for c in range(9):
            mask = (all_coldness == c)
            n = int(mask.sum())
            if n > 100:  # need enough positives for meaningful AUC
                pos = int(all_labels[mask].sum())
                neg = n - pos
                if pos > 10 and neg > 10:
                    strat[c] = {
                        "n": n, "pos": pos, "neg": neg,
                        "auc": float(roc_auc_score(all_labels[mask], scores[mask])),
                    }
                else:
                    strat[c] = {"n": n, "pos": pos, "neg": neg, "auc": None}
            else:
                strat[c] = {"n": n, "auc": None}
        return {"name": modifier_name, "overall_auc": overall, "strat": strat, "scores": scores}

    def baseline_mod(w, t):
        return w  # unchanged

    def dc_mod(w, t):
        n = int(ln_emb[t]); n_hot = int(n * HOT_FRACTION)
        hot_idx = sorted_idx[t][:n_hot]
        cold_idx = sorted_idx[t][n_hot:]
        # DC rpb=16 on cold
        cold_w = w[cold_idx]
        q, s, zp = quantize_table(cold_w)
        nc = len(cold_idx); rpb = ROWS_PER_BLOCK
        n_pad = ((nc + rpb - 1) // rpb) * rpb
        q_np = q.numpy()
        if n_pad > nc:
            p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8)
            p[:nc] = q_np; q_np = p
        nb = n_pad // rpb
        block_means = q_np.reshape(nb, rpb, EMB_DIM).astype(np.float32).mean(axis=(1, 2), keepdims=True)
        recon = np.broadcast_to(block_means, (nb, rpb, EMB_DIM)).reshape(n_pad, EMB_DIM)[:nc]
        recon = np.clip(np.round(recon), 0, 255).astype(np.uint8)
        w[cold_idx] = (torch.from_numpy(recon.copy()).float() - zp) * s
        # Hot quantize
        if n_hot > 0:
            q_h, s_h, zp_h = quantize_table(w[hot_idx])
            w[hot_idx] = (q_h.float() - zp_h) * s_h
        return w

    def prune_mod(w, t):
        n = int(ln_emb[t]); n_hot = int(n * HOT_FRACTION)
        hot_idx = sorted_idx[t][:n_hot]
        cold_idx = sorted_idx[t][n_hot:]
        w[cold_idx] = 0.0
        if n_hot > 0:
            q_h, s_h, zp_h = quantize_table(w[hot_idx])
            w[hot_idx] = (q_h.float() - zp_h) * s_h
        return w

    print("\nRunning baseline (fp32)...")
    baseline = run_config("baseline", baseline_mod)
    print(f"  Overall AUC: {baseline['overall_auc']:.6f}")

    print("\nRunning DC rpb=16 @ 2% hot...")
    dc = run_config("DC rpb=16 @ 2% hot", dc_mod)
    print(f"  Overall AUC: {dc['overall_auc']:.6f}")

    print("\nRunning Pruning 98% sp (uint8 kept)...")
    pr = run_config("Pruning 98% sp", prune_mod)
    print(f"  Overall AUC: {pr['overall_auc']:.6f}")

    print("\n" + "=" * 90)
    print("STRATIFIED AUC BY COLDNESS")
    print("=" * 90)
    print(f"{'Coldness':>9} {'N samples':>10} {'Baseline':>12} {'DC rpb=16':>12} {'Pruning':>12} "
          f"{'DC Δ bps':>11} {'Pr Δ bps':>11} {'DC−Pr bps':>11}")
    print("-" * 90)
    for c in range(9):
        b = baseline["strat"][c]
        if b.get("auc") is None:
            continue
        d = dc["strat"][c]
        p = pr["strat"][c]
        if d.get("auc") is None or p.get("auc") is None:
            continue
        dc_delta = (d["auc"] - b["auc"]) * 10000
        pr_delta = (p["auc"] - b["auc"]) * 10000
        gap = (d["auc"] - p["auc"]) * 10000
        print(f"{c:>9d} {b['n']:>10d} {b['auc']:>12.6f} {d['auc']:>12.6f} {p['auc']:>12.6f} "
              f"{dc_delta:>+10.2f} {pr_delta:>+10.2f} {gap:>+10.2f}")

    # Also: compute variance of scores on cold-only samples
    print("\n" + "=" * 90)
    print("PREDICTION SCORE VARIANCE BY COLDNESS (measures differentiation)")
    print("=" * 90)
    print(f"{'Coldness':>9} {'N':>10} {'σ(baseline)':>14} {'σ(DC)':>14} {'σ(Pruning)':>14} "
          f"{'DC/Pr σ ratio':>15}")
    print("-" * 90)
    for c in range(9):
        mask = (all_coldness == c)
        n = int(mask.sum())
        if n < 100:
            continue
        std_b = float(baseline["scores"][mask].std())
        std_d = float(dc["scores"][mask].std())
        std_p = float(pr["scores"][mask].std())
        ratio = std_d / std_p if std_p > 0 else float('inf')
        print(f"{c:>9d} {n:>10d} {std_b:>14.6f} {std_d:>14.6f} {std_p:>14.6f} {ratio:>14.3f}")

    # Save
    out = {
        "baseline_auc": baseline["overall_auc"],
        "dc_auc": dc["overall_auc"],
        "pr_auc": pr["overall_auc"],
        "stratified": {
            c: {
                "n": int(baseline["strat"][c].get("n", 0)),
                "baseline_auc": baseline["strat"][c].get("auc"),
                "dc_auc": dc["strat"][c].get("auc"),
                "pr_auc": pr["strat"][c].get("auc"),
            } for c in range(9) if baseline["strat"][c].get("auc") is not None
        },
    }
    with open("results/dct_domain/tail_metric_stratified.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/dct_domain/tail_metric_stratified.json")

if __name__ == '__main__':
    main()
