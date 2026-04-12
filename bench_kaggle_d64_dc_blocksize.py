#!/usr/bin/env python3
"""
Kaggle D=64 DC block-size sweep (parallel to bench_kaggle_dc_blocksize.py for D=16).
Tests whether DC's per-row magnitude advantage scales with larger D.
"""
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
    """Load Kaggle DLRM at D=64 (architecture matches what dlrm_s_pytorch.py expects)."""
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net

    # Build Kaggle args manually for D=64
    class Args: pass
    args = Args()
    args.arch_sparse_feature_size = 64
    args.arch_mlp_bot = "13-512-256-64"  # D=64 bottom MLP
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
    print("[D=64] Loading model checkpoint...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    print(f"[D=64] Model loaded. {len(ln_emb)} tables, D={m_spa}")
    return dlrm, test_ld, ln_emb

def main():
    print("=" * 70)
    print("KAGGLE D=64 DC BLOCK-SIZE SWEEP")
    print("=" * 70)

    dlrm, test_ld, ln_emb = load_kaggle_d64()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]
    torch.set_num_threads(32)

    total_emb_mb = sum(sd[ek[i]].numel() * 4 for i in range(nt)) / 1024 / 1024
    print(f"D={EMB_DIM}, {nt} tables, {len(TABLES)} large: {TABLES}")
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
    n_total_large = sum(int(ln_emb[t]) for t in TABLES)
    BITMAP_MB = n_total_large / 8 / 1024 / 1024
    SMALL_MB = sum(int(ln_emb[t]) * EMB_DIM * 4 for t in range(nt) if t not in TABLES) / 1024 / 1024

    results = {"baseline_auc": auc_base, "total_emb_mb": total_emb_mb,
               "bitmap_mb": BITMAP_MB, "small_mb": SMALL_MB, "rows": []}

    print(f"\nbitmap_mb = {BITMAP_MB:.4f}, small_mb = {SMALL_MB:.4f}")
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
        total_mb = hot_mb + cold_mb + SMALL_MB + BITMAP_MB
        ratio = total_emb_mb / total_mb
        print(f"{hf*100:>4.1f}% {'zero':>12} {auc:>10.6f} {loss:>+9.4f}% "
              f"{hot_mb:>6.1f} {cold_mb:>7.2f} {total_mb:>7.1f} {ratio:>6.0f}x")
        results["rows"].append({"hot_fraction": hf, "mode": "zero", "rpb": 0,
                                "auc": auc, "loss_pct": loss, "hot_mb": hot_mb,
                                "cold_mb": cold_mb, "total_mb": total_mb, "ratio": ratio})

        for rpb in [1, 4, 8, 16]:
            auc = compute_auc('dc', hf, rpb=rpb)
            loss = (auc_base - auc) * 100
            cold_mb = (n_cold / rpb) / 1024 / 1024
            total_mb = hot_mb + cold_mb + SMALL_MB + BITMAP_MB
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
    with open("results/dct_domain/kaggle_d64_dc_blocksize.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results/dct_domain/kaggle_d64_dc_blocksize.json")

if __name__ == '__main__':
    main()
