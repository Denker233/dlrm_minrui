#!/usr/bin/env python3
"""
Head-to-head batch latency: DC vs frequency pruning on Kaggle D=16.

Measures wall-clock inference time per batch for each method at matched
configurations. Since both methods produce modified EmbeddingBag weights
that go through the same forward pass in Python, we expect them to be
roughly equal; the goal is to confirm this empirically and quantify any
cache/memory-pressure effects.

Protocol: 50 warmup batches + 150 measured batches, report mean ± std.
"""
import os, sys, json, time, gc
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

MODEL_PATH = 'models/dlrm_kaggle_correct.pt'
LARGE_THRESHOLD = 50000
EMB_DIM = 16
WARMUP = 50
MEASURE = 150

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
    print("DC vs PRUNING BATCH LATENCY (Kaggle D=16)")
    print("=" * 80)

    dlrm, test_ld, ln_emb = load_kaggle()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]
    torch.set_num_threads(32)

    print(f"D={EMB_DIM}, {nt} tables, {len(TABLES)} large")
    print(f"Threads: {torch.get_num_threads()}")
    print(f"Warmup: {WARMUP} batches, Measure: {MEASURE} batches")

    print("Caching test batches...")
    test_batches = []
    for X, lS_o, lS_i, T in test_ld:
        test_batches.append((X, lS_o, lS_i, T))
        if len(test_batches) >= WARMUP + MEASURE:
            break
    print(f"Cached {len(test_batches)} test batches")

    print("Profiling frequencies...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    # Use all test_ld batches for frequency profiling (not just the subset)
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    sorted_idx = {t: torch.argsort(freq[t], descending=True) for t in TABLES}

    def apply_config(mode, hf=None, rpb=None, sp=None):
        """Apply a compression configuration to the model's embedding tables."""
        with torch.no_grad():
            for k in ek:
                t = int(k.split('.')[1])
                dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
                w = sd[k].clone()
                if t in TABLES:
                    if mode == 'fp32':
                        pass  # unchanged
                    elif mode == 'uint8_all':
                        q, s, zp = quantize_table(w)
                        w = (q.float() - zp) * s
                    elif mode == 'pruning':
                        n = int(ln_emb[t])
                        keep_n = max(1, int(round(n * (1 - sp / 100.0))))
                        keep_idx = sorted_idx[t][:keep_n]
                        drop_idx = sorted_idx[t][keep_n:]
                        w[drop_idx] = 0.0
                        if len(keep_idx) > 0:
                            q_h, s_h, zp_h = quantize_table(w[keep_idx])
                            w[keep_idx] = (q_h.float() - zp_h) * s_h
                    elif mode == 'dc':
                        n = int(ln_emb[t]); n_hot = int(n * hf)
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
                    elif mode == 'zero':
                        n = int(ln_emb[t]); n_hot = int(n * hf)
                        hot_idx = sorted_idx[t][:n_hot]
                        cold_idx = sorted_idx[t][n_hot:]
                        w[cold_idx] = 0.0
                        if n_hot > 0:
                            q_h, s_h, zp_h = quantize_table(w[hot_idx])
                            w[hot_idx] = (q_h.float() - zp_h) * s_h
                dlrm.emb_l[t].weight.data = w

    def measure_latency(name):
        """Run warmup + measured batches, return mean and std per-batch latency in ms."""
        # Warmup
        with torch.no_grad():
            for b in range(min(WARMUP, len(test_batches))):
                X, o, i, T = test_batches[b]
                _ = dlrm(X, o, i)
        # Measure
        times = []
        auc_scores = []
        auc_labels = []
        with torch.no_grad():
            for b in range(WARMUP, min(WARMUP + MEASURE, len(test_batches))):
                X, o, i, T = test_batches[b]
                t_start = time.perf_counter()
                Z = dlrm(X, o, i)
                t_end = time.perf_counter()
                times.append((t_end - t_start) * 1000)  # ms
                auc_scores.append(Z.numpy().ravel())
                auc_labels.append(T.numpy().ravel())
        times = np.array(times)
        auc_val = roc_auc_score(np.concatenate(auc_labels), np.concatenate(auc_scores))
        return {
            "name": name,
            "mean_ms": float(times.mean()),
            "median_ms": float(np.median(times)),
            "std_ms": float(times.std()),
            "p5_ms": float(np.percentile(times, 5)),
            "p95_ms": float(np.percentile(times, 95)),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "auc": float(auc_val),
        }

    configs = [
        ("fp32 baseline",          dict(mode='fp32')),
        ("uint8 all rows",         dict(mode='uint8_all')),
        ("pruning 92% sp",         dict(mode='pruning', sp=92)),
        ("pruning 95% sp",         dict(mode='pruning', sp=95)),
        ("pruning 97% sp",         dict(mode='pruning', sp=97)),
        ("pruning 98% sp",         dict(mode='pruning', sp=98)),
        ("pruning 99% sp",         dict(mode='pruning', sp=99)),
        ("DC rpb=1 @ 4.3% hot",    dict(mode='dc', hf=0.043, rpb=1)),
        ("DC rpb=4 @ 4.3% hot",    dict(mode='dc', hf=0.043, rpb=4)),
        ("DC rpb=16 @ 4.3% hot",   dict(mode='dc', hf=0.043, rpb=16)),
        ("DC rpb=1 @ 2.0% hot",    dict(mode='dc', hf=0.02, rpb=1)),
        ("DC rpb=16 @ 2.0% hot",   dict(mode='dc', hf=0.02, rpb=16)),
        ("DC rpb=1 @ 1.0% hot",    dict(mode='dc', hf=0.01, rpb=1)),
        ("DC rpb=16 @ 1.0% hot",   dict(mode='dc', hf=0.01, rpb=16)),
        ("zero @ 2.0% hot",        dict(mode='zero', hf=0.02)),
    ]

    results = []
    for name, kwargs in configs:
        print(f"\nRunning {name} ...")
        apply_config(**kwargs)
        gc.collect()
        r = measure_latency(name)
        print(f"  mean: {r['mean_ms']:.3f} ms  (median: {r['median_ms']:.3f} ms, "
              f"std: {r['std_ms']:.3f}, p5-p95: {r['p5_ms']:.3f}-{r['p95_ms']:.3f})  "
              f"AUC: {r['auc']:.6f}")
        results.append(r)

    print("\n" + "=" * 90)
    print("SUMMARY (sorted by mean latency)")
    print("=" * 90)
    print(f"{'Config':>25} {'Mean ms':>10} {'Median':>10} {'Std':>8} {'p5-p95':>18} {'AUC':>10}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x["mean_ms"]):
        print(f"{r['name']:>25} {r['mean_ms']:>9.3f}  {r['median_ms']:>9.3f}  {r['std_ms']:>7.3f}  "
              f"{r['p5_ms']:>8.3f}-{r['p95_ms']:>7.3f}  {r['auc']:>10.6f}")

    os.makedirs("results/dct_domain", exist_ok=True)
    with open("results/dct_domain/dc_vs_pruning_latency.json", "w") as f:
        json.dump({
            "model": MODEL_PATH, "D": EMB_DIM, "warmup": WARMUP, "measure": MEASURE,
            "threads": torch.get_num_threads(),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved results/dct_domain/dc_vs_pruning_latency.json")

if __name__ == '__main__':
    main()
