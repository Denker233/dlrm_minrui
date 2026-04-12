#!/usr/bin/env python3
"""
Extended tail-metric experiment: compare rpb=1 (per-row DC) against rpb=16
(per-block-16 DC) and pruning to see if per-row DC provides meaningful
tail-quality differentiation that block DC doesn't.

Key question: does per-row DC create measurable variance in cold-sample
predictions beyond what pruning does?
"""
import os, sys, json
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

MODEL_PATH = 'models/dlrm_kaggle_correct.pt'
LARGE_THRESHOLD = 50000
EMB_DIM = 16
HOT_FRACTION = 0.02

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
    print("TAIL-METRIC V2: rpb=1 vs rpb=16 vs pruning")
    print("=" * 80)

    dlrm, test_ld, ln_emb = load_kaggle()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]
    torch.set_num_threads(32)

    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    print(f"Cached {len(test_batches)} test batches")

    print("Profiling frequencies...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    sorted_idx = {t: torch.argsort(freq[t], descending=True) for t in TABLES}

    # Build hot-set mask
    is_hot = {}
    for t in TABLES:
        n = int(ln_emb[t])
        n_hot = int(n * HOT_FRACTION)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[sorted_idx[t][:n_hot]] = True
        is_hot[t] = mask

    # Compute coldness
    print("Computing per-sample coldness...")
    coldness_per_batch = []
    labels_per_batch = []
    for X, o, i, T in test_batches:
        batch_size = T.shape[0]
        coldness = torch.zeros(batch_size, dtype=torch.int32)
        for t in TABLES:
            idx = (i[t] if isinstance(i, (list, tuple)) else i[t]).long()
            offsets = o[t] if isinstance(o, (list, tuple)) else o[t]
            off_np = offsets.long().numpy() if hasattr(offsets, 'numpy') else np.array(offsets)
            idx_np = idx.numpy()
            sample_cold = np.zeros(batch_size, dtype=bool)
            for s in range(batch_size):
                start = int(off_np[s])
                end = int(off_np[s+1]) if s+1 < batch_size else len(idx_np)
                if end > start:
                    sample_rows = idx_np[start:end]
                    any_cold = not is_hot[t][sample_rows].any().item()
                    if any_cold:
                        sample_cold[s] = True
            coldness += torch.from_numpy(sample_cold.astype(np.int32))
        coldness_per_batch.append(coldness.numpy())
        labels_per_batch.append(T.numpy().ravel())
    all_coldness = np.concatenate(coldness_per_batch)
    all_labels = np.concatenate(labels_per_batch)

    def run_config(name, modifier_fn):
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
        overall = roc_auc_score(all_labels, scores)
        strat = {}
        for c in range(9):
            mask = (all_coldness == c)
            n = int(mask.sum())
            if n > 100:
                pos = int(all_labels[mask].sum())
                neg = n - pos
                if pos > 10 and neg > 10:
                    strat[c] = {
                        "n": n, "pos": pos, "neg": neg,
                        "auc": float(roc_auc_score(all_labels[mask], scores[mask])),
                        "std": float(scores[mask].std()),
                    }
        return {"name": name, "overall": overall, "strat": strat, "scores": scores}

    def baseline_mod(w, t):
        return w

    def make_dc_mod(rpb):
        def mod(w, t):
            n = int(ln_emb[t]); n_hot = int(n * HOT_FRACTION)
            hot_idx = sorted_idx[t][:n_hot]
            cold_idx = sorted_idx[t][n_hot:]
            cold_w = w[cold_idx]
            q, s, zp = quantize_table(cold_w)
            nc = len(cold_idx)
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
            if n_hot > 0:
                q_h, s_h, zp_h = quantize_table(w[hot_idx])
                w[hot_idx] = (q_h.float() - zp_h) * s_h
            return w
        return mod

    def prune_mod(w, t):
        n = int(ln_emb[t]); n_hot = int(n * HOT_FRACTION)
        hot_idx = sorted_idx[t][:n_hot]
        cold_idx = sorted_idx[t][n_hot:]
        w[cold_idx] = 0.0
        if n_hot > 0:
            q_h, s_h, zp_h = quantize_table(w[hot_idx])
            w[hot_idx] = (q_h.float() - zp_h) * s_h
        return w

    print("\nRunning baseline...")
    baseline = run_config("baseline", baseline_mod)
    print(f"  Overall AUC: {baseline['overall']:.6f}")

    print("\nRunning DC rpb=1 @ 2% hot (per-row DC)...")
    dc1 = run_config("DC rpb=1", make_dc_mod(1))
    print(f"  Overall AUC: {dc1['overall']:.6f}")

    print("\nRunning DC rpb=16 @ 2% hot...")
    dc16 = run_config("DC rpb=16", make_dc_mod(16))
    print(f"  Overall AUC: {dc16['overall']:.6f}")

    print("\nRunning Pruning 98% sp...")
    pr = run_config("Pruning", prune_mod)
    print(f"  Overall AUC: {pr['overall']:.6f}")

    print("\n" + "=" * 100)
    print("STRATIFIED AUC (basis points delta from baseline)")
    print("=" * 100)
    print(f"{'Coldness':>9} {'N':>9} {'baseline':>10} {'DC-1 Δbps':>12} {'DC-16 Δbps':>12} "
          f"{'Pr Δbps':>10} {'DC1-Pr':>9} {'DC16-Pr':>9}")
    print("-" * 100)
    for c in range(9):
        b = baseline["strat"].get(c, {}).get("auc")
        if b is None:
            continue
        d1 = dc1["strat"][c]["auc"]
        d16 = dc16["strat"][c]["auc"]
        p = pr["strat"][c]["auc"]
        n = baseline["strat"][c]["n"]
        print(f"{c:>9d} {n:>9d} {b:>10.6f} {(d1-b)*10000:>+11.2f} {(d16-b)*10000:>+11.2f} "
              f"{(p-b)*10000:>+9.2f} {(d1-p)*10000:>+8.2f} {(d16-p)*10000:>+8.2f}")

    print("\n" + "=" * 100)
    print("SCORE VARIANCE BY COLDNESS (measures row-level differentiation)")
    print("=" * 100)
    print(f"{'Coldness':>9} {'σ(baseline)':>14} {'σ(DC-1)':>12} {'σ(DC-16)':>12} {'σ(Pruning)':>12} "
          f"{'DC1/Pr':>9} {'DC16/Pr':>9}")
    print("-" * 100)
    for c in range(9):
        b = baseline["strat"].get(c, {}).get("std")
        if b is None:
            continue
        std1 = dc1["strat"][c]["std"]
        std16 = dc16["strat"][c]["std"]
        stdp = pr["strat"][c]["std"]
        r1 = std1 / stdp if stdp > 0 else float('inf')
        r16 = std16 / stdp if stdp > 0 else float('inf')
        print(f"{c:>9d} {b:>14.6f} {std1:>12.6f} {std16:>12.6f} {stdp:>12.6f} {r1:>9.4f} {r16:>9.4f}")

    # Save
    with open("results/dct_domain/tail_metric_v2.json", "w") as f:
        json.dump({
            "baseline_auc": baseline["overall"],
            "dc1_auc": dc1["overall"],
            "dc16_auc": dc16["overall"],
            "pr_auc": pr["overall"],
            "baseline_strat": {c: v for c, v in baseline["strat"].items()},
            "dc1_strat": {c: v for c, v in dc1["strat"].items()},
            "dc16_strat": {c: v for c, v in dc16["strat"].items()},
            "pr_strat": {c: v for c, v in pr["strat"].items()},
        }, f, indent=2, default=str)
    print("\nSaved results/dct_domain/tail_metric_v2.json")

if __name__ == '__main__':
    main()
