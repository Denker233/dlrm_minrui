#!/usr/bin/env python3
"""D=64 tail-metric experiment at matched compression ratios."""
import os, sys, json
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

MODEL_PATH = 'models/dlrm_kaggle_d64.pt'
LARGE_THRESHOLD = 50000
EMB_DIM = 64

def quantize_table(w):
    mn = w.min().item(); mx = w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp

def load_kaggle_d64():
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net
    class Args: pass
    args = Args()
    args.arch_sparse_feature_size = 64
    args.arch_mlp_bot = "13-512-256-64"
    args.arch_mlp_top = "512-256-1"
    args.arch_interaction_op = "dot"
    args.arch_interaction_itself = False
    args.data_generation = "dataset"
    args.data_set = "kaggle"
    args.raw_data_file = "./input/train.txt"
    args.processed_data_file = "./input/kaggleAdDisplayChallenge_processed.npz"
    args.loss_function = "bce"
    args.max_ind_range = -1
    args.test_mini_batch_size = 16384
    args.test_num_workers = 0
    args.num_workers = 0
    args.mlperf_logging = False
    args.memory_map = False
    args.data_randomize = "total"
    args.data_trace_enable_padding = False
    args.data_sub_sample_rate = 0.0
    args.num_indices_per_lookup = 10
    args.num_indices_per_lookup_fixed = False
    args.mini_batch_size = 128
    args.round_targets = True
    args.mlperf_bin_loader = False
    args.mlperf_bin_shuffle = False
    args.dataset_multiprocessing = False

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
    print("KAGGLE D=64 TAIL METRIC")
    print("=" * 80)
    dlrm, test_ld, ln_emb = load_kaggle_d64()
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

    HOT_REF = 0.02
    is_hot_ref = {}
    for t in TABLES:
        n = int(ln_emb[t])
        n_hot = int(n * HOT_REF)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[sorted_idx[t][:n_hot]] = True
        is_hot_ref[t] = mask

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
        return {"name": name, "overall": overall, "strat": strat}

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

    # Run all DC configs and pruning at multiple sparsities
    configs = []
    print("\n DC rpb=1 @ 4.3% hot")
    configs.append(run_config("DC rpb=1 @ 4.3%", dc_mod(1, 0.043)))
    print(f"  overall: {configs[-1]['overall']:.6f}")

    print("\n DC rpb=1 @ 2% hot")
    configs.append(run_config("DC rpb=1 @ 2%", dc_mod(1, 0.02)))
    print(f"  overall: {configs[-1]['overall']:.6f}")

    print("\n DC rpb=16 @ 2% hot")
    configs.append(run_config("DC rpb=16 @ 2%", dc_mod(16, 0.02)))
    print(f"  overall: {configs[-1]['overall']:.6f}")

    print("\n Pruning 95.7% sp")
    configs.append(run_config("Prune 95.7%", prune_mod(95.7)))
    print(f"  overall: {configs[-1]['overall']:.6f}")

    print("\n Pruning 98% sp")
    configs.append(run_config("Prune 98%", prune_mod(98)))
    print(f"  overall: {configs[-1]['overall']:.6f}")

    print("\n Pruning 92% sp")
    configs.append(run_config("Prune 92%", prune_mod(92)))
    print(f"  overall: {configs[-1]['overall']:.6f}")

    # Print stratified AUC
    print("\n" + "=" * 100)
    print("STRATIFIED AUC (all configs), basis-point delta from baseline")
    print("=" * 100)
    names = [c["name"] for c in configs]
    header = f"{'Cold':>5} {'N':>9} " + " ".join(f"{n:>14}" for n in names)
    print(header)
    for c in range(9):
        if c not in baseline["strat"]:
            continue
        b = baseline["strat"][c]["auc"]
        n = baseline["strat"][c]["n"]
        line = f"{c:>5d} {n:>9d} "
        for cfg in configs:
            if c in cfg["strat"]:
                delta = (cfg["strat"][c]["auc"] - b) * 10000
                line += f"{delta:>+14.2f}"
            else:
                line += " " * 14
        print(line)

    with open("results/dct_domain/kaggle_d64_tail.json", "w") as f:
        json.dump({
            "baseline_auc": baseline["overall"],
            "baseline_strat": baseline["strat"],
            "configs": [{"name": c["name"], "overall": c["overall"], "strat": c["strat"]} for c in configs],
        }, f, indent=2, default=str)
    print("\nSaved results/dct_domain/kaggle_d64_tail.json")

if __name__ == '__main__':
    main()
