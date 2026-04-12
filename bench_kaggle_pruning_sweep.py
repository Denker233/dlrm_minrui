#!/usr/bin/env python3
"""
Frequency-pruning sweep on Criteo Kaggle for matched-loss comparison with DC.

For each sparsity P, zero out the bottom P% of cold rows (by frequency).
Measure AUC and compression ratio. Two variants:
  (a) fp32 keep — kept rows stored as fp32 (standard pruning baseline)
  (b) uint8 keep — kept rows stored as uint8 (apples-to-apples with our DC method)

This enables: "at AUC loss X, what compression does pruning need?" — the
matched-loss comparison the reviewer asked for.
"""
import os, sys, gc, json
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
    print("[KG] Loading model checkpoint...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    return dlrm, test_ld, train_ld, ln_emb

def main():
    print("=" * 70)
    print("KAGGLE PRUNING SWEEP (for matched-loss comparison with DC)")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_kaggle()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]
    torch.set_num_threads(32)

    print(f"D={EMB_DIM}, {nt} tables, {len(TABLES)} large")
    total_emb_mb = sum(sd[ek[i]].numel() * 4 for i in range(nt)) / 1024 / 1024
    print(f"Total fp32 embedding: {total_emb_mb:.0f} MB")

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

    print("Baseline AUC...")
    scores, targets = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_base = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"  Baseline fp32: {auc_base:.6f}")

    def compute_pruning(sparsity_pct, quantize_kept):
        """Zero out bottom sparsity_pct% of rows by frequency.
        quantize_kept=True → uint8 quantize the kept rows; False → keep fp32."""
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
                    if quantize_kept and len(keep_idx) > 0:
                        q_h, s_h, zp_h = quantize_table(w[keep_idx])
                        w[keep_idx] = (q_h.float() - zp_h) * s_h
                dlrm.emb_l[t].weight.data = w

        scores, targets = [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
        return roc_auc_score(np.concatenate(targets), np.concatenate(scores))

    # Memory accounting
    small_mb = sum(int(ln_emb[t]) * EMB_DIM * 4 for t in range(nt) if t not in TABLES) / 1024 / 1024

    sparsities = [50, 80, 90, 92, 94, 95, 96, 97, 97.5, 98, 98.5, 99, 99.3, 99.5, 99.7, 99.9]

    results = {"baseline_auc": auc_base, "total_emb_mb": total_emb_mb,
               "small_mb": small_mb, "rows": []}

    for variant in ["fp32_kept", "uint8_kept"]:
        bytes_per_row = 4 if variant == "fp32_kept" else 1
        quantize = (variant == "uint8_kept")
        print(f"\n{'='*84}")
        print(f"Variant: {variant} (kept rows stored as {'fp32' if not quantize else 'uint8'})")
        print(f"{'-'*84}")
        print(f"{'Sparsity':>9} {'Keep%':>6} {'AUC':>10} {'Loss':>9} {'Kept MB':>9} {'Total':>8} {'Ratio':>7}")

        for sp in sparsities:
            auc = compute_pruning(sp, quantize_kept=quantize)
            loss = (auc_base - auc) * 100
            keep_pct = 100.0 - sp
            # Memory: kept rows × D × bytes_per_row
            n_total_large = sum(int(ln_emb[t]) for t in TABLES)
            n_kept = sum(max(1, int(round(int(ln_emb[t]) * keep_pct / 100))) for t in TABLES)
            kept_mb = n_kept * EMB_DIM * bytes_per_row / 1024 / 1024
            # bitmap for which rows are kept (1 bit/row)
            bitmap_mb = n_total_large / 8 / 1024 / 1024
            total_mb = kept_mb + small_mb + bitmap_mb
            ratio = total_emb_mb / total_mb
            print(f"{sp:>8.1f}% {keep_pct:>5.2f}% {auc:>10.6f} {loss:>+8.4f}% "
                  f"{kept_mb:>8.2f} {total_mb:>7.1f} {ratio:>6.0f}x")
            results["rows"].append({
                "variant": variant, "sparsity_pct": sp, "auc": auc, "loss_pct": loss,
                "kept_mb": kept_mb, "bitmap_mb": bitmap_mb, "total_mb": total_mb,
                "ratio": ratio,
            })
        gc.collect()

    os.makedirs("results/dct_domain", exist_ok=True)
    with open("results/dct_domain/kaggle_pruning_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results/dct_domain/kaggle_pruning_sweep.json")
    print("Done.")

if __name__ == '__main__':
    main()
