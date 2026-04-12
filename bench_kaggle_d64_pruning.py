#!/usr/bin/env python3
"""Kaggle D=64 pruning sweep — apples-to-apples baseline for DC at D=64."""
import os, sys, gc, json
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

    print("[D=64] Loading dataset...")
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
    print("[D=64] Loading checkpoint...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    return dlrm, test_ld, ln_emb

def main():
    print("=" * 70)
    print("KAGGLE D=64 PRUNING SWEEP")
    print("=" * 70)

    dlrm, test_ld, ln_emb = load_kaggle_d64()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]
    torch.set_num_threads(32)

    total_emb_mb = sum(sd[ek[i]].numel() * 4 for i in range(nt)) / 1024 / 1024
    print(f"D={EMB_DIM}, {nt} tables, {len(TABLES)} large")
    print(f"Total fp32 embedding: {total_emb_mb:.0f} MB")

    test_batches = [(X, lS_o, lS_i, T) for X, lS_o, lS_i, T in test_ld]
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    sorted_idx = {t: torch.argsort(freq[t], descending=True) for t in TABLES}

    scores, targets = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_base = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"Baseline fp32 AUC: {auc_base:.6f}")

    def compute_pruning(sp, quantize_kept):
        with torch.no_grad():
            for k in ek:
                t = int(k.split('.')[1])
                dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
                w = sd[k].clone()
                if t in TABLES:
                    n = int(ln_emb[t])
                    keep_n = max(1, int(round(n * (1 - sp / 100.0))))
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

    small_mb = sum(int(ln_emb[t]) * EMB_DIM * 4 for t in range(nt) if t not in TABLES) / 1024 / 1024
    n_total_large = sum(int(ln_emb[t]) for t in TABLES)
    bitmap_mb = n_total_large / 8 / 1024 / 1024

    sparsities = [50, 80, 90, 92, 94, 95, 96, 97, 97.5, 98, 98.5, 99, 99.3, 99.5, 99.7, 99.9]
    results = {"baseline_auc": auc_base, "total_emb_mb": total_emb_mb,
               "small_mb": small_mb, "bitmap_mb": bitmap_mb, "rows": []}

    for variant in ["uint8_kept"]:  # skip fp32_kept to save time; it's never the fair comparison
        bytes_per_row = 1
        quantize = True
        print(f"\n{'='*84}\nVariant: {variant}\n{'='*84}")
        print(f"{'Sparsity':>9} {'Keep%':>6} {'AUC':>10} {'Loss':>9} {'Kept MB':>9} {'Total':>8} {'Ratio':>7}")
        for sp in sparsities:
            auc = compute_pruning(sp, quantize_kept=quantize)
            loss = (auc_base - auc) * 100
            keep_pct = 100.0 - sp
            n_kept = sum(max(1, int(round(int(ln_emb[t]) * keep_pct / 100))) for t in TABLES)
            kept_mb = n_kept * EMB_DIM * bytes_per_row / 1024 / 1024
            total_mb = kept_mb + small_mb + bitmap_mb
            ratio = total_emb_mb / total_mb
            print(f"{sp:>8.1f}% {keep_pct:>5.2f}% {auc:>10.6f} {loss:>+8.4f}% "
                  f"{kept_mb:>8.2f} {total_mb:>7.1f} {ratio:>6.0f}x")
            results["rows"].append({"variant": variant, "sparsity_pct": sp, "auc": auc,
                                    "loss_pct": loss, "kept_mb": kept_mb,
                                    "bitmap_mb": bitmap_mb, "total_mb": total_mb, "ratio": ratio})
        gc.collect()

    os.makedirs("results/dct_domain", exist_ok=True)
    with open("results/dct_domain/kaggle_d64_pruning.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/dct_domain/kaggle_d64_pruning.json")

if __name__ == '__main__':
    main()
