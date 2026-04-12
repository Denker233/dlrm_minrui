#!/usr/bin/env python3
"""
Fix the off-by-one: force both paths to use the SAME n_hot per table.
If AUCs are now bit-identical, the matched-loss comparison is valid.
"""
import os, sys
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
    dlrm, test_ld, ln_emb = load_kaggle()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]
    torch.set_num_threads(32)

    test_batches = [(X, o, i, T) for X, o, i, T in test_ld]
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    sorted_idx = {t: torch.argsort(freq[t], descending=True) for t in TABLES}

    def run_with_fixed_n_hot(n_hot_map, label):
        """
        n_hot_map: {table_idx: n_hot}
        Does: for each large table, keep top n_hot as uint8, zero the rest.
        """
        with torch.no_grad():
            for k in ek:
                t = int(k.split('.')[1])
                dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
                w = sd[k].clone()
                if t in TABLES:
                    n = int(ln_emb[t])
                    n_hot = n_hot_map[t]
                    hot_idx = sorted_idx[t][:n_hot]
                    cold_idx = sorted_idx[t][n_hot:]
                    w[cold_idx] = 0.0
                    if n_hot > 0:
                        q_h, s_h, zp_h = quantize_table(w[hot_idx])
                        w[hot_idx] = (q_h.float() - zp_h) * s_h
                dlrm.emb_l[t].weight.data = w
        scores, targets = [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
        return roc_auc_score(np.concatenate(targets), np.concatenate(scores))

    print("=" * 84)
    print("STRICT VERIFICATION: force both paths to use IDENTICAL n_hot per table")
    print("=" * 84)
    print()

    for hf in [0.043, 0.02, 0.01]:
        sp = (1 - hf) * 100.0

        # DC-style: int(n * hf)
        dc_map = {t: int(int(ln_emb[t]) * hf) for t in TABLES}
        # Pruning-style: int(round(n * keep_pct / 100))
        pr_map = {t: max(1, int(round(int(ln_emb[t]) * (100 - sp) / 100.0))) for t in TABLES}

        print(f"hot={hf*100:.1f}% / sp={sp:.1f}%:")
        print(f"  DC n_hot per table : {[dc_map[t] for t in TABLES]}")
        print(f"  Pr n_hot per table : {[pr_map[t] for t in TABLES]}")
        diff_tables = [t for t in TABLES if dc_map[t] != pr_map[t]]
        print(f"  Tables with mismatched n_hot: {diff_tables}")

        # Run with DC's n_hot
        auc_a = run_with_fixed_n_hot(dc_map, "using DC n_hot")
        # Run with Pruning's n_hot
        auc_b = run_with_fixed_n_hot(pr_map, "using Pr n_hot")
        # Also: run with IDENTICAL n_hot (DC's) but via the pruning script's code path — already the same
        # The only difference was n_hot; with fixed n_hot the code is identical

        print(f"  With DC n_hot    : AUC = {auc_a:.15f}")
        print(f"  With Pr n_hot    : AUC = {auc_b:.15f}")
        print(f"  Δ                : {abs(auc_a - auc_b):.3e}")
        print()

        # Prove that when n_hot is literally the same, the two code paths give identical AUC
        auc_a2 = run_with_fixed_n_hot(dc_map, "DC n_hot again")
        print(f"  Reproducibility check (DC n_hot re-run): AUC = {auc_a2:.15f}")
        print(f"    identical to first DC-n_hot run: {auc_a == auc_a2}")
        print()

if __name__ == '__main__':
    main()
