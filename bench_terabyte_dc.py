#!/usr/bin/env python3
"""
Terabyte DC experiment: does DC carry meaningful information with D=64?
Compare cold=0 vs cold=DC vs cold=uint8 at different hot fractions.
Uses Python EmbeddingBag (no C++ needed for AUC verification).
"""
import os, sys, time, json, gc
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

# Terabyte config
os.environ['CRITEO_DAYS'] = '4'
MODEL_PATH = 'models/dlrm_terabyte_4day.pt'
LARGE_THRESHOLD = 50000
EMB_DIM = 64

def quantize_table(w):
    mn = w.min().item(); mx = w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp

def load_terabyte():
    """Load Terabyte model and data (4 days)."""
    import dlrm_data_pytorch as dp
    from dlrm_s_pytorch import DLRM_Net
    from codec_ondemand_benchmark import create_args

    # Start from Kaggle args, override for Terabyte
    args = create_args()
    args.arch_sparse_feature_size = 64
    args.arch_mlp_bot = "13-512-256-64"
    args.data_set = "terabyte"
    args.raw_data_file = "/home/cc/input/terabyte/day"
    args.processed_data_file = "/home/cc/input/terabyte/terabyte_processed.npz"
    args.memory_map = True
    args.mini_batch_size = 2048
    args.test_mini_batch_size = 1024

    print("[TB] Loading dataset...")
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    ln_emb = np.array(train_data.counts)
    m_spa = args.arch_sparse_feature_size
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-") if isinstance(args.arch_mlp_bot, str) else np.array(args.arch_mlp_bot)
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[-1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top_str = str(num_int) + "-512-512-256-1"
    ln_top = np.fromstring(ln_top_str, dtype=int, sep="-")

    dlrm = DLRM_Net(m_spa, ln_emb, ln_bot, ln_top,
                     arch_interaction_op="dot", arch_interaction_itself=False,
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")
    print("[TB] Loading model checkpoint...")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    print(f"[TB] Model loaded. {len(ln_emb)} tables, D={m_spa}")
    return dlrm, test_ld, train_ld, ln_emb

def main():
    print("=" * 70)
    print("TERABYTE DC EXPERIMENT: D=64")
    print("=" * 70)

    # Load model
    print("Loading model...")
    dlrm, test_ld, train_ld, ln_emb = load_terabyte()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]
    torch.set_num_threads(32)

    print(f"D={EMB_DIM}, {nt} tables, {len(TABLES)} large: {TABLES}")
    total_emb_mb = sum(sd[ek[i]].numel() * 4 for i in range(nt)) / 1024 / 1024
    print(f"Total embedding: {total_emb_mb:.0f} MB")

    # Cache test batches (use subset if too many)
    print("Caching test batches...")
    test_batches = []
    for X, lS_o, lS_i, T in test_ld:
        test_batches.append((X, lS_o, lS_i, T))
        if len(test_batches) >= 500:  # limit for speed
            break
    print(f"Cached {len(test_batches)} test batches")

    # Profile access frequency
    print("Profiling frequencies...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    sorted_idx = {t: torch.argsort(freq[t], descending=True) for t in TABLES}

    # Baseline AUC
    print("Baseline AUC...")
    scores, targets = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_base = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"  Baseline fp32: {auc_base:.6f}")

    def compute_auc(cold_mode, hf):
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
                        # DC = mean of each block of 16 rows
                        nc = len(cold_idx); rpb = 16
                        n_pad = ((nc + rpb - 1) // rpb) * rpb
                        q_np = q.numpy()
                        if n_pad > nc:
                            p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8)
                            p[:nc] = q_np; q_np = p
                        # Per-row mean (DC of 1-row block): mean of D values per row
                        # For D=64, this gives 1 value per row
                        row_means = q_np.astype(np.float32).mean(axis=1, keepdims=True)
                        recon = np.broadcast_to(row_means, (n_pad, EMB_DIM))[:nc]
                        recon = np.clip(np.round(recon), 0, 255).astype(np.uint8)
                        w[cold_idx] = (torch.from_numpy(recon.copy()).float() - zp) * s
                    elif cold_mode == 'dc16':
                        # Block of 16 rows: 1 value for all 16 rows × 64 dims
                        cold_w = w[cold_idx]
                        q, s, zp = quantize_table(cold_w)
                        nc = len(cold_idx); rpb = 16
                        n_pad = ((nc + rpb - 1) // rpb) * rpb
                        q_np = q.numpy()
                        if n_pad > nc:
                            p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8)
                            p[:nc] = q_np; q_np = p
                        nb = n_pad // rpb
                        block_means = q_np.reshape(nb, rpb, EMB_DIM).astype(np.float32).mean(axis=(1,2), keepdims=True)
                        recon = np.broadcast_to(block_means, (nb, rpb, EMB_DIM)).reshape(n_pad, EMB_DIM)[:nc]
                        recon = np.clip(np.round(recon), 0, 255).astype(np.uint8)
                        w[cold_idx] = (torch.from_numpy(recon.copy()).float() - zp) * s
                    elif cold_mode == 'perdim16':
                        # Per-dimension mean of 16-row blocks (16 values per block)
                        cold_w = w[cold_idx]
                        q, s, zp = quantize_table(cold_w)
                        nc = len(cold_idx); rpb = 16
                        n_pad = ((nc + rpb - 1) // rpb) * rpb
                        q_np = q.numpy()
                        if n_pad > nc:
                            p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8)
                            p[:nc] = q_np; q_np = p
                        nb = n_pad // rpb
                        dim_means = q_np.reshape(nb, rpb, EMB_DIM).astype(np.float32).mean(axis=1)  # (nb, D)
                        recon = np.repeat(np.clip(np.round(dim_means), 0, 255).astype(np.uint8), rpb, axis=0)[:nc]
                        w[cold_idx] = (torch.from_numpy(recon.copy()).float() - zp) * s
                    elif cold_mode == 'uint8':
                        cold_w = w[cold_idx]
                        q, s, zp = quantize_table(cold_w)
                        w[cold_idx] = (q.float() - zp) * s

                    # Quantize hot to uint8
                    if n_hot > 0:
                        q_h, s_h, zp_h = quantize_table(w[hot_idx])
                        w[hot_idx] = (q_h.float() - zp_h) * s_h

                dlrm.emb_l[t].weight.data = w

        scores, targets = [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
        return roc_auc_score(np.concatenate(targets), np.concatenate(scores))

    # Memory calculation
    total_cold_rows = {hf: sum(int(ln_emb[t]) - int(int(ln_emb[t]) * hf) for t in TABLES) for hf in [0.043, 0.02, 0.01]}
    total_hot_rows = {hf: sum(int(int(ln_emb[t]) * hf) for t in TABLES) for hf in [0.043, 0.02, 0.01]}

    print(f"\n{'='*80}")
    print(f"{'Hot%':>5} {'Mode':>12} {'AUC':>10} {'Loss':>8} {'Hot MB':>8} {'Cold':>12} {'Total':>8} {'Ratio':>6}")
    print("-" * 80)

    for hf in [0.043, 0.02, 0.01]:
        hot_mb = total_hot_rows[hf] * EMB_DIM / 1024 / 1024  # uint8
        n_cold = total_cold_rows[hf]
        small_mb = sum(int(ln_emb[t]) * EMB_DIM * 4 for t in range(nt) if t not in TABLES) / 1024 / 1024

        for mode in ['zero', 'dc16', 'dc', 'perdim16', 'uint8']:
            auc = compute_auc(mode, hf)
            loss = (auc_base - auc) * 100

            if mode == 'zero':
                cold_mb = 0; cold_desc = "0 MB"
            elif mode == 'dc16':
                cold_mb = (n_cold / 16) / 1024 / 1024  # 1 byte per 16 rows
                cold_desc = f"{cold_mb:.1f}MB (1B/16r)"
            elif mode == 'dc':
                cold_mb = n_cold / 1024 / 1024  # 1 byte per row
                cold_desc = f"{cold_mb:.1f}MB (1B/r)"
            elif mode == 'perdim16':
                cold_mb = (n_cold / 16) * EMB_DIM / 1024 / 1024  # D bytes per 16 rows
                cold_desc = f"{cold_mb:.1f}MB ({EMB_DIM}B/16r)"
            elif mode == 'uint8':
                cold_mb = n_cold * EMB_DIM / 1024 / 1024
                cold_desc = f"{cold_mb:.1f}MB (full)"

            total_mb = hot_mb + small_mb + 6.0 + cold_mb  # +6 bitmap
            ratio = total_emb_mb / total_mb

            print(f"{hf*100:>4.1f}% {mode:>12} {auc:>10.6f} {loss:>+7.3f}% "
                  f"{hot_mb:>7.1f} {cold_desc:>12} {total_mb:>7.1f} {ratio:>5.0f}x")
        print()
        gc.collect()

    print("Done.")

if __name__ == '__main__':
    main()
