#!/usr/bin/env python3
"""
Kaggle DC block-size sweep: rpb ∈ {1, 4, 8, 16} rows per block, plus zero baseline.
Same uint8 framework as bench_kaggle_dc.py — apples-to-apples.
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
    print("KAGGLE DC BLOCK-SIZE SWEEP: rpb ∈ {1, 4, 8, 16}")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_kaggle()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]
    torch.set_num_threads(32)

    print(f"D={EMB_DIM}, {nt} tables, {len(TABLES)} large: {TABLES}")
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

    def compute_auc(cold_mode, hf, rpb=1):
        """cold_mode: 'zero' or 'dc'. rpb=1 → per-row DC; rpb=4 → 4 rows/block; ..."""
        with torch.no_grad():
            for k in ek:
                t = int(k.split('.')[1])
                dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
                w = sd[k].clone()
                if t in TABLES:
                    n = int(ln_emb[t]); n_hot = int(n * hf)
                    hot_idx = sorted_idx[t][:n_hot]
                    cold_idx = sorted_idx[t][n_hot:]

                    if cold_mode == 'zero':
                        w[cold_idx] = 0.0
                    elif cold_mode == 'dc':
                        cold_w = w[cold_idx]
                        q, s, zp = quantize_table(cold_w)
                        nc = len(cold_idx)
                        n_pad = ((nc + rpb - 1) // rpb) * rpb
                        q_np = q.numpy()
                        if n_pad > nc:
                            p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8)
                            p[:nc] = q_np; q_np = p
                        nb = n_pad // rpb
                        # 1 byte per block: avg of (rpb rows × D dims)
                        block_means = q_np.reshape(nb, rpb, EMB_DIM).astype(np.float32).mean(axis=(1, 2), keepdims=True)
                        recon = np.broadcast_to(block_means, (nb, rpb, EMB_DIM)).reshape(n_pad, EMB_DIM)[:nc]
                        recon = np.clip(np.round(recon), 0, 255).astype(np.uint8)
                        w[cold_idx] = (torch.from_numpy(recon.copy()).float() - zp) * s

                    if n_hot > 0:
                        q_h, s_h, zp_h = quantize_table(w[hot_idx])
                        w[hot_idx] = (q_h.float() - zp_h) * s_h

                dlrm.emb_l[t].weight.data = w

        scores, targets = [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
        return roc_auc_score(np.concatenate(targets), np.concatenate(scores))

    hot_fractions = [0.043, 0.02, 0.01]
    total_cold_rows = {hf: sum(int(ln_emb[t]) - int(int(ln_emb[t]) * hf) for t in TABLES) for hf in hot_fractions}
    total_hot_rows = {hf: sum(int(int(ln_emb[t]) * hf) for t in TABLES) for hf in hot_fractions}
    small_mb = sum(int(ln_emb[t]) * EMB_DIM * 4 for t in range(nt) if t not in TABLES) / 1024 / 1024

    results = {"baseline_auc": auc_base, "total_emb_mb": total_emb_mb, "rows": []}

    print(f"\n{'='*92}")
    print(f"{'Hot%':>5} {'Mode':>12} {'AUC':>10} {'Loss':>10} {'HotMB':>7} {'ColdMB':>8} {'Total':>8} {'Ratio':>7}")
    print("-" * 92)

    for hf in hot_fractions:
        hot_mb = total_hot_rows[hf] * EMB_DIM / 1024 / 1024
        n_cold = total_cold_rows[hf]

        # zero baseline
        auc = compute_auc('zero', hf)
        loss = (auc_base - auc) * 100
        cold_mb = 0.0
        total_mb = hot_mb + small_mb + 6.0 + cold_mb
        ratio = total_emb_mb / total_mb
        print(f"{hf*100:>4.1f}% {'zero':>12} {auc:>10.6f} {loss:>+9.4f}% "
              f"{hot_mb:>6.1f} {cold_mb:>7.2f} {total_mb:>7.1f} {ratio:>6.0f}x")
        results["rows"].append({"hot_fraction": hf, "mode": "zero", "rpb": 0,
                                "auc": auc, "loss_pct": loss, "hot_mb": hot_mb,
                                "cold_mb": cold_mb, "total_mb": total_mb, "ratio": ratio})

        # DC at different block sizes
        for rpb in [1, 4, 8, 16]:
            auc = compute_auc('dc', hf, rpb=rpb)
            loss = (auc_base - auc) * 100
            cold_mb = (n_cold / rpb) / 1024 / 1024  # 1 byte per block of rpb rows
            total_mb = hot_mb + small_mb + 6.0 + cold_mb
            ratio = total_emb_mb / total_mb
            mode_name = f"DC rpb={rpb}"
            print(f"{hf*100:>4.1f}% {mode_name:>12} {auc:>10.6f} {loss:>+9.4f}% "
                  f"{hot_mb:>6.1f} {cold_mb:>7.2f} {total_mb:>7.1f} {ratio:>6.0f}x")
            results["rows"].append({"hot_fraction": hf, "mode": "dc", "rpb": rpb,
                                    "auc": auc, "loss_pct": loss, "hot_mb": hot_mb,
                                    "cold_mb": cold_mb, "total_mb": total_mb, "ratio": ratio})
        print()
        gc.collect()

    os.makedirs("results/dct_domain", exist_ok=True)
    with open("results/dct_domain/kaggle_dc_blocksize_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results/dct_domain/kaggle_dc_blocksize_sweep.json")
    print("Done.")

if __name__ == '__main__':
    main()
