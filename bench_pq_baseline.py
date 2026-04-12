#!/usr/bin/env python3
"""
Product Quantization (PQ) baseline for DLRM embedding compression.

Compares PQ/OPQ at various M/nbits settings against DC value-sort.
Uses faiss for PQ training and encoding.

For each config:
1. Train PQ on cold embeddings
2. Encode cold rows → PQ codes
3. Decode PQ codes → reconstructed embeddings
4. Replace in model, measure AUC
5. Compute compression ratio
"""
import os, sys, time, json
import numpy as np
import torch
import faiss
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
from sklearn.metrics import roc_auc_score
from codec_ondemand_benchmark import load_model_and_data, REORDER_DIR

LARGE_TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
EMB_DIM = 16
HOT_FRACTIONS = [0.043, 0.02, 0.01, 0.005]

# PQ configs: (M, nbits) where M = number of subquantizers
# D=16, so M must divide 16: M in {1, 2, 4, 8, 16}
# nbits = bits per subquantizer code: typically 4 or 8
# Storage per row = M * nbits / 8 bytes
PQ_CONFIGS = [
    # (M, nbits, label)
    (16, 8, 'PQ16x8'),   # 16 bytes/row = same as fp32/4 = uint8
    (8, 8, 'PQ8x8'),     # 8 bytes/row = 2x compression vs uint8
    (4, 8, 'PQ4x8'),     # 4 bytes/row = 4x vs uint8
    (2, 8, 'PQ2x8'),     # 2 bytes/row = 8x vs uint8
    (1, 8, 'PQ1x8'),     # 1 byte/row = 16x vs uint8
    (8, 4, 'PQ8x4'),     # 4 bytes/row
    (4, 4, 'PQ4x4'),     # 2 bytes/row
    (2, 4, 'PQ2x4'),     # 1 byte/row
    (1, 4, 'PQ1x4'),     # 0.5 bytes/row
    (2, 2, 'PQ2x2'),     # 0.5 bytes/row
    (1, 2, 'PQ1x2'),     # 0.25 bytes/row
]


def run_auc(dlrm, test_ld):
    all_s, all_l = [], []
    with torch.no_grad():
        for X, o, i, T in test_ld:
            Z = dlrm(X, o, i)
            all_s.append(Z.cpu().numpy().flatten())
            all_l.append(T.numpy().flatten())
    return roc_auc_score(np.concatenate(all_l), np.concatenate(all_s))


def pq_compress_table(cold_fp32_np, M, nbits, train_sample_size=100000):
    """
    Train PQ on cold embeddings and encode/decode.
    Returns: (reconstructed_fp32, bytes_per_row, train_time, encode_time)
    """
    d = cold_fp32_np.shape[1]
    n = cold_fp32_np.shape[0]

    # Train PQ on a sample
    if n > train_sample_size:
        train_idx = np.random.choice(n, train_sample_size, replace=False)
        train_data = cold_fp32_np[train_idx].copy()
    else:
        train_data = cold_fp32_np.copy()

    # Ensure contiguous float32
    train_data = np.ascontiguousarray(train_data, dtype=np.float32)

    t0 = time.time()
    pq = faiss.ProductQuantizer(d, M, nbits)
    pq.train(train_data)
    train_time = time.time() - t0

    # Encode all cold rows
    cold_contig = np.ascontiguousarray(cold_fp32_np, dtype=np.float32)
    t0 = time.time()
    codes = pq.compute_codes(cold_contig)
    encode_time = time.time() - t0

    # Decode (reconstruct)
    t0 = time.time()
    reconstructed = pq.decode(codes)
    decode_time = time.time() - t0

    bytes_per_row = codes.shape[1]  # M * ceil(nbits/8)

    return reconstructed, bytes_per_row, train_time, encode_time, decode_time


def main():
    print("=" * 70)
    print("PQ BASELINE: Product Quantization vs DC value-sort")
    print("=" * 70)

    dlrm, test_ld, _, ln_emb = load_model_and_data()
    sd = dlrm.state_dict()
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))

    baseline_auc = run_auc(dlrm, test_ld)
    print(f"Baseline AUC: {baseline_auc:.6f}")

    # Profile access frequencies
    print("Profiling access frequencies...")
    freq = {}
    for X, lS_o, lS_i, T in test_ld:
        for t in LARGE_TABLES:
            idx = lS_i[t].flatten() if isinstance(lS_i, (list, tuple)) else lS_i[t].flatten()
            if t not in freq:
                freq[t] = torch.zeros(int(ln_emb[t]), dtype=torch.long)
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))

    orig_w = {t: sd[ek[t]].clone() for t in LARGE_TABLES}
    total_fp32_mb = sum(sd[ek[i]].shape[0] * EMB_DIM * 4 for i in range(26)) / 1024**2
    small_uint8_mb = sum(sd[ek[i]].shape[0] * EMB_DIM * 1 for i in range(26)
                         if i not in LARGE_TABLES) / 1024**2

    results = {'baseline_auc': baseline_auc, 'configs': []}

    print(f"\n{'method':>14} {'hf':>5} {'ratio':>8} {'AUC':>10} {'delta%':>10} {'bytes/row':>10}")
    print("-" * 70)

    for hf in HOT_FRACTIONS:
        # Compute hot/cold split
        cold_idx = {}
        for t in LARGE_TABLES:
            n = int(ln_emb[t])
            n_hot = max(1, int(n * hf))
            sorted_idx = freq[t].argsort(descending=True)
            cold_idx[t] = sorted_idx[n_hot:]

        hot_mb = sum((int(ln_emb[t]) - len(cold_idx[t])) * EMB_DIM * 1
                     for t in LARGE_TABLES) / 1024**2
        bitmap_mb = sum(int(ln_emb[t]) for t in LARGE_TABLES) / 8 / 1024**2
        n_cold_total = sum(len(cold_idx[t]) for t in LARGE_TABLES)

        # --- DC value-sort reference ---
        BS = 16
        for t in LARGE_TABLES:
            w = orig_w[t].clone()
            cold_w = w[cold_idx[t]]
            row_means = cold_w.mean(dim=1)
            order = torch.argsort(row_means)
            cold_sorted = cold_w[order]
            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(len(order))

            nc = len(cold_idx[t])
            nb = (nc + BS - 1) // BS
            padded = torch.zeros(nb * BS, EMB_DIM)
            padded[:nc] = cold_sorted
            means = padded.reshape(nb, BS, EMB_DIM).mean(dim=(1, 2))

            # 4-bit quantize means
            n_levels = 16
            mn, mx = means.min().item(), means.max().item()
            s = (mx - mn) / (n_levels - 1) if mx != mn else 1.0
            q_means = ((means - mn) / s).round().clamp(0, n_levels - 1) * s + mn

            recon_cold = q_means.unsqueeze(1).unsqueeze(2).expand(nb, BS, EMB_DIM).reshape(-1, EMB_DIM)[:nc]
            recon_cold = recon_cold[inv_order]
            w[cold_idx[t]] = recon_cold
            dlrm.emb_l[t].weight.data = w

        dc_auc = run_auc(dlrm, test_ld)
        dc_delta = dc_auc - baseline_auc
        for t in LARGE_TABLES:
            dlrm.emb_l[t].weight.data = orig_w[t].clone()

        n_blocks = sum((len(cold_idx[t]) + BS - 1) // BS for t in LARGE_TABLES)
        dc_cold_bytes = n_blocks * 4 / 8  # 4 bits per block
        dc_cold_mb = dc_cold_bytes / 1024**2
        dc_total = hot_mb + dc_cold_mb + small_uint8_mb + bitmap_mb
        dc_ratio = total_fp32_mb / dc_total

        print(f"{'DC-val-4bit':>14} {hf:>5.1%} {dc_ratio:>8.0f}x {dc_auc:>10.6f} {dc_delta*100:>+10.4f} {'0.5(blk)':>10}")

        results['configs'].append({
            'method': 'DC-val-4bit', 'hf': hf, 'ratio': dc_ratio,
            'auc': dc_auc, 'delta': dc_delta, 'bytes_per_row': 0.5/BS,
        })

        # --- PQ configs ---
        for M, nbits, label in PQ_CONFIGS:
            if M > EMB_DIM:
                continue

            for t in LARGE_TABLES:
                w = orig_w[t].clone()
                cold_fp32 = w[cold_idx[t]].numpy()

                recon, bpr, tt, et, dt = pq_compress_table(cold_fp32, M, nbits)
                w[cold_idx[t]] = torch.from_numpy(recon)
                dlrm.emb_l[t].weight.data = w

            auc = run_auc(dlrm, test_ld)
            delta = auc - baseline_auc
            for t in LARGE_TABLES:
                dlrm.emb_l[t].weight.data = orig_w[t].clone()

            pq_cold_mb = n_cold_total * bpr / 1024**2
            pq_total = hot_mb + pq_cold_mb + small_uint8_mb + bitmap_mb
            pq_ratio = total_fp32_mb / pq_total

            print(f"{label:>14} {hf:>5.1%} {pq_ratio:>8.0f}x {auc:>10.6f} {delta*100:>+10.4f} {bpr:>10}")

            results['configs'].append({
                'method': label, 'hf': hf, 'ratio': pq_ratio,
                'auc': auc, 'delta': delta, 'bytes_per_row': bpr,
            })

        # --- Zero reference ---
        for t in LARGE_TABLES:
            w = orig_w[t].clone()
            w[cold_idx[t]] = 0.0
            dlrm.emb_l[t].weight.data = w
        zero_auc = run_auc(dlrm, test_ld)
        zero_delta = zero_auc - baseline_auc
        for t in LARGE_TABLES:
            dlrm.emb_l[t].weight.data = orig_w[t].clone()

        zero_total = hot_mb + small_uint8_mb + bitmap_mb
        zero_ratio = total_fp32_mb / zero_total

        print(f"{'Zero':>14} {hf:>5.1%} {zero_ratio:>8.0f}x {zero_auc:>10.6f} {zero_delta*100:>+10.4f} {'0':>10}")
        results['configs'].append({
            'method': 'Zero', 'hf': hf, 'ratio': zero_ratio,
            'auc': zero_auc, 'delta': zero_delta, 'bytes_per_row': 0,
        })

        print()

    # Save
    out_path = 'results/pq_baseline.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
