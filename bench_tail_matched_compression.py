#!/usr/bin/env python3
"""
Tail-metric at MATCHED compression ratio.

Compare:
  - DC rpb=1  @ 2.0% hot (42.3x)   vs pruning 92% sp  (42.9x)   — low compression
  - DC rpb=16 @ 2.0% hot (107.5x)  vs pruning 97.5% sp (104.2x) — high compression

Question: at the same compression ratio, does DC preserve more tail-AUC than pruning?
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
    print("TAIL METRIC AT MATCHED COMPRESSION RATIO")
    print("=" * 80)

    dlrm, test_ld, ln_emb = load_kaggle()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]
    torch.set_num_threads(32)

    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]

    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    sorted_idx = {t: torch.argsort(freq[t], descending=True) for t in TABLES}

    # Compute coldness based on 2% hot fraction boundary (used for DC)
    # Note: we use 2% hot as the "coldness reference" even for pruning at 92% sp,
    # because the question is "how well does each method preserve accuracy on
    # the bottom 98% of rows" — the reference is the hot/cold boundary of DC.
    HOT_REF = 0.02
    is_hot_ref = {}
    for t in TABLES:
        n = int(ln_emb[t])
        n_hot = int(n * HOT_REF)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[sorted_idx[t][:n_hot]] = True
        is_hot_ref[t] = mask

    print("Computing per-sample coldness (reference: 2% hot threshold)...")
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
                    any_cold = not is_hot_ref[t][sample_rows].any().item()
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
                if pos > 10 and (n - pos) > 10:
                    strat[c] = {
                        "n": n,
                        "auc": float(roc_auc_score(all_labels[mask], scores[mask])),
                        "std": float(scores[mask].std()),
                    }
        return {"name": name, "overall": overall, "strat": strat, "scores": scores}

    def baseline_mod(w, t):
        return w

    def dc_mod(rpb, hot_frac):
        def mod(w, t):
            n = int(ln_emb[t]); n_hot = int(n * hot_frac)
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

    def prune_mod(sp_pct):
        def mod(w, t):
            n = int(ln_emb[t])
            keep_n = max(1, int(round(n * (1 - sp_pct / 100.0))))
            keep_idx = sorted_idx[t][:keep_n]
            drop_idx = sorted_idx[t][keep_n:]
            w[drop_idx] = 0.0
            if len(keep_idx) > 0:
                q_h, s_h, zp_h = quantize_table(w[keep_idx])
                w[keep_idx] = (q_h.float() - zp_h) * s_h
            return w
        return mod

    print("\nRunning baseline...")
    baseline = run_config("baseline", baseline_mod)
    print(f"  Overall AUC: {baseline['overall']:.6f}")

    # Pair 1: low-compression (42x)
    print("\nPair 1: ~42x compression")
    print("  DC rpb=1 @ 2.0% hot ...")
    dc1 = run_config("DC rpb=1 @ 2% hot", dc_mod(1, 0.02))
    print(f"    Overall AUC: {dc1['overall']:.6f}  (expected ~0.802108)")
    print("  Pruning 92% sp ...")
    pr92 = run_config("Pruning 92% sp", prune_mod(92))
    print(f"    Overall AUC: {pr92['overall']:.6f}  (expected ~0.802344)")

    # Pair 2: high-compression (~105x)
    print("\nPair 2: ~105x compression")
    print("  DC rpb=16 @ 2.0% hot ...")
    dc16 = run_config("DC rpb=16 @ 2% hot", dc_mod(16, 0.02))
    print(f"    Overall AUC: {dc16['overall']:.6f}  (expected ~0.801614)")
    print("  Pruning 97.5% sp ...")
    pr975 = run_config("Pruning 97.5% sp", prune_mod(97.5))
    print(f"    Overall AUC: {pr975['overall']:.6f}  (expected ~0.801783)")

    print("\n" + "=" * 100)
    print("PAIR 1: ~42x COMPRESSION — DC rpb=1 (2% hot) vs Pruning 92% sp")
    print("=" * 100)
    print(f"{'Coldness':>9} {'N':>9} {'baseline':>10} {'DC Δbps':>10} {'Pr Δbps':>10} {'DC−Pr':>8}")
    for c in range(9):
        if c not in baseline["strat"]: continue
        b = baseline["strat"][c]["auc"]; n = baseline["strat"][c]["n"]
        d = dc1["strat"][c]["auc"]
        p = pr92["strat"][c]["auc"]
        print(f"{c:>9d} {n:>9d} {b:>10.6f} {(d-b)*10000:>+9.2f} {(p-b)*10000:>+9.2f} "
              f"{(d-p)*10000:>+7.2f}")

    print("\n" + "=" * 100)
    print("PAIR 2: ~105x COMPRESSION — DC rpb=16 (2% hot) vs Pruning 97.5% sp")
    print("=" * 100)
    print(f"{'Coldness':>9} {'N':>9} {'baseline':>10} {'DC Δbps':>10} {'Pr Δbps':>10} {'DC−Pr':>8}")
    for c in range(9):
        if c not in baseline["strat"]: continue
        b = baseline["strat"][c]["auc"]; n = baseline["strat"][c]["n"]
        d = dc16["strat"][c]["auc"]
        p = pr975["strat"][c]["auc"]
        print(f"{c:>9d} {n:>9d} {b:>10.6f} {(d-b)*10000:>+9.2f} {(p-b)*10000:>+9.2f} "
              f"{(d-p)*10000:>+7.2f}")

    print("\n" + "=" * 100)
    print("SCORE VARIANCE (matched compression)")
    print("=" * 100)
    print(f"{'Coldness':>9} {'σ base':>11} {'σ DC-1':>11} {'σ Pr-92':>11} {'DC1/Pr92':>10} "
          f"{'σ DC-16':>11} {'σ Pr-97.5':>11} {'DC16/Pr97':>10}")
    for c in range(9):
        if c not in baseline["strat"]: continue
        sb = baseline["strat"][c]["std"]
        s1 = dc1["strat"][c]["std"]
        sp92 = pr92["strat"][c]["std"]
        s16 = dc16["strat"][c]["std"]
        sp975 = pr975["strat"][c]["std"]
        r1 = s1 / sp92 if sp92 > 0 else 0
        r16 = s16 / sp975 if sp975 > 0 else 0
        print(f"{c:>9d} {sb:>11.6f} {s1:>11.6f} {sp92:>11.6f} {r1:>9.4f}  "
              f"{s16:>11.6f} {sp975:>11.6f} {r16:>9.4f}")

    with open("results/dct_domain/tail_matched_compression.json", "w") as f:
        json.dump({
            "baseline": {"auc": baseline["overall"], "strat": baseline["strat"]},
            "dc1": {"auc": dc1["overall"], "strat": dc1["strat"]},
            "pr92": {"auc": pr92["overall"], "strat": pr92["strat"]},
            "dc16": {"auc": dc16["overall"], "strat": dc16["strat"]},
            "pr975": {"auc": pr975["overall"], "strat": pr975["strat"]},
        }, f, indent=2, default=str)
    print("\nSaved results/dct_domain/tail_matched_compression.json")

if __name__ == '__main__':
    main()
