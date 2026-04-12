#!/usr/bin/env python3
"""
Run BOTH code paths (DC-sweep 'zero' mode and pruning-sweep 'uint8_kept' variant)
in a single process. At matched hot fraction == 1 - sparsity, they should produce
BIT-IDENTICAL AUCs because they perform the same operation:
  - Keep top K% of rows by frequency, quantize them to uint8
  - Set the remaining rows to 0

Any difference would be a real discrepancy. No difference = verification.
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
    print("VERIFICATION: DC-zero path vs pruning-uint8 path at matched hot fraction")
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
    test_batches = []
    for X, lS_o, lS_i, T in test_ld:
        test_batches.append((X, lS_o, lS_i, T))
    print(f"Cached {len(test_batches)} test batches")

    print("Profiling frequencies on test_ld...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    sorted_idx = {t: torch.argsort(freq[t], descending=True) for t in TABLES}

    print("Baseline AUC (unmodified fp32)...")
    scores, targets = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_base = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"  Baseline: {auc_base:.10f}")
    print()

    def run_dc_zero_path(hf):
        """Exact copy of bench_kaggle_dc_blocksize.py 'zero' mode path."""
        with torch.no_grad():
            for k in ek:
                t = int(k.split('.')[1])
                dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
                w = sd[k].clone()
                if t in TABLES:
                    n = int(ln_emb[t]); n_hot = int(n * hf)
                    hot_idx = sorted_idx[t][:n_hot]
                    cold_idx = sorted_idx[t][n_hot:]
                    # 'zero' mode: set cold to 0
                    w[cold_idx] = 0.0
                    # Hot: uint8 quantize
                    if n_hot > 0:
                        q_h, s_h, zp_h = quantize_table(w[hot_idx])
                        w[hot_idx] = (q_h.float() - zp_h) * s_h
                dlrm.emb_l[t].weight.data = w
        scores, targets = [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
        return roc_auc_score(np.concatenate(targets), np.concatenate(scores))

    def run_pruning_uint8_path(sparsity_pct):
        """Exact copy of bench_kaggle_pruning_sweep.py 'uint8_kept' variant path."""
        with torch.no_grad():
            for k in ek:
                t = int(k.split('.')[1])
                dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
                w = sd[k].clone()
                if t in TABLES:
                    n = int(ln_emb[t])
                    keep_n = max(1, int(round(n * (1 - sparsity_pct / 100.0))))
                    keep_idx = sorted_idx[t][:keep_n]
                    drop_idx = sorted_idx[t][keep_n:]
                    w[drop_idx] = 0.0
                    if len(keep_idx) > 0:
                        q_h, s_h, zp_h = quantize_table(w[keep_idx])
                        w[keep_idx] = (q_h.float() - zp_h) * s_h
                dlrm.emb_l[t].weight.data = w
        scores, targets = [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
        return roc_auc_score(np.concatenate(targets), np.concatenate(scores))

    # Also check: are n_hot and n_kept the same at matched points?
    print("Row-count cross-check:")
    print(f"{'Hot%':>6} {'Table':>6} {'n_hot (DC)':>12} {'n_kept (Pr)':>12} {'Same?':>7}")
    for hf in [0.043, 0.02, 0.01]:
        sp = (1 - hf) * 100.0
        for t in TABLES[:3]:
            n = int(ln_emb[t])
            n_hot = int(n * hf)
            n_kept = max(1, int(round(n * (1 - sp / 100.0))))
            same = "✓" if n_hot == n_kept else "✗"
            print(f"{hf*100:>5.1f}% {t:>6d} {n_hot:>12d} {n_kept:>12d} {same:>7}")
    print()

    # Run matched comparisons
    print("=" * 80)
    print("RUN BOTH PATHS AT MATCHED HOT FRACTION")
    print("=" * 80)
    print(f"{'Hot%':>6} {'Sparsity':>9} {'DC-zero AUC':>18} {'Pruning AUC':>18} {'|Δ|':>12}")
    for hf in [0.043, 0.02, 0.01]:
        sp = (1 - hf) * 100.0
        auc_dc = run_dc_zero_path(hf)
        auc_pr = run_pruning_uint8_path(sp)
        diff = abs(auc_dc - auc_pr)
        flag = "✓ BIT-MATCH" if diff < 1e-9 else f"DIFFER"
        print(f"{hf*100:>5.1f}% {sp:>8.2f}% {auc_dc:>18.15f} {auc_pr:>18.15f} {diff:>12.3e}  {flag}")
    print()
    print("If DC-zero AUC == Pruning AUC to float precision, the two paths are")
    print("computing bit-identical operations and the matched-loss comparison is valid.")

if __name__ == '__main__':
    main()
