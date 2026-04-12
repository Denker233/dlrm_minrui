#!/usr/bin/env python3
"""
V2 latency benchmark: run each config 3 times in RANDOMIZED order,
aggregate all measurements per config. Eliminates ordering confound.
"""
import os, sys, json, time, gc, random
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

MODEL_PATH = 'models/dlrm_kaggle_correct.pt'
LARGE_THRESHOLD = 50000
EMB_DIM = 16
WARMUP = 30       # per run
MEASURE = 100     # per run
N_RUNS = 3        # times each config is measured

random.seed(42)

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
    print("DC vs PRUNING BATCH LATENCY V2 (randomized order, 3 runs each)")
    print("=" * 80)

    dlrm, test_ld, ln_emb = load_kaggle()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]
    torch.set_num_threads(32)

    test_batches = []
    for X, lS_o, lS_i, T in test_ld:
        test_batches.append((X, lS_o, lS_i, T))
        if len(test_batches) >= WARMUP + MEASURE:
            break
    print(f"Cached {len(test_batches)} test batches, {N_RUNS} runs × {len(test_batches)} each")

    print("Profiling frequencies...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    sorted_idx = {t: torch.argsort(freq[t], descending=True) for t in TABLES}

    def apply_config(mode, hf=None, rpb=None, sp=None):
        with torch.no_grad():
            for k in ek:
                t = int(k.split('.')[1])
                dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
                w = sd[k].clone()
                if t in TABLES:
                    if mode == 'fp32':
                        pass
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
                dlrm.emb_l[t].weight.data = w

    def measure_once():
        # Warmup
        with torch.no_grad():
            for b in range(WARMUP):
                X, o, i, T = test_batches[b]
                _ = dlrm(X, o, i)
        times = []
        with torch.no_grad():
            for b in range(WARMUP, WARMUP + MEASURE):
                X, o, i, T = test_batches[b]
                t0 = time.perf_counter()
                _ = dlrm(X, o, i)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
        return times

    configs = [
        ("fp32",                dict(mode='fp32')),
        ("uint8_all",           dict(mode='uint8_all')),
        ("pruning_92sp",        dict(mode='pruning', sp=92)),
        ("pruning_95sp",        dict(mode='pruning', sp=95)),
        ("pruning_98sp",        dict(mode='pruning', sp=98)),
        ("pruning_99sp",        dict(mode='pruning', sp=99)),
        ("DC_rpb1_hot4p3",      dict(mode='dc', hf=0.043, rpb=1)),
        ("DC_rpb16_hot4p3",     dict(mode='dc', hf=0.043, rpb=16)),
        ("DC_rpb1_hot2p0",      dict(mode='dc', hf=0.02, rpb=1)),
        ("DC_rpb16_hot2p0",     dict(mode='dc', hf=0.02, rpb=16)),
        ("DC_rpb1_hot1p0",      dict(mode='dc', hf=0.01, rpb=1)),
        ("DC_rpb16_hot1p0",     dict(mode='dc', hf=0.01, rpb=16)),
    ]

    all_measurements = {name: [] for name, _ in configs}

    # Run N_RUNS times in shuffled order
    for run_idx in range(N_RUNS):
        run_order = configs[:]
        random.shuffle(run_order)
        print(f"\n--- Run {run_idx+1}/{N_RUNS} (shuffled order) ---")
        for name, kwargs in run_order:
            apply_config(**kwargs)
            gc.collect()
            times = measure_once()
            all_measurements[name].extend(times)
            print(f"  {name:>20}: mean={np.mean(times):.3f}ms median={np.median(times):.3f}ms")

    print("\n" + "=" * 100)
    print(f"AGGREGATED RESULTS ({N_RUNS} runs × {MEASURE} batches = {N_RUNS*MEASURE} samples per config)")
    print("=" * 100)
    print(f"{'Config':>22} {'Mean ms':>10} {'Median':>10} {'Std':>8} {'p5':>8} {'p95':>8} "
          f"{'Min':>8} {'Max':>8}")
    print("-" * 100)

    summary = []
    for name, _ in sorted(configs, key=lambda c: np.mean(all_measurements[c[0]])):
        times = np.array(all_measurements[name])
        row = {
            "name": name, "n": len(times),
            "mean_ms": float(times.mean()), "median_ms": float(np.median(times)),
            "std_ms": float(times.std()),
            "p5_ms": float(np.percentile(times, 5)), "p95_ms": float(np.percentile(times, 95)),
            "min_ms": float(times.min()), "max_ms": float(times.max()),
        }
        summary.append(row)
        print(f"{name:>22} {row['mean_ms']:>9.3f}  {row['median_ms']:>9.3f}  "
              f"{row['std_ms']:>7.3f}  {row['p5_ms']:>7.3f}  {row['p95_ms']:>7.3f}  "
              f"{row['min_ms']:>7.3f}  {row['max_ms']:>7.3f}")

    # Pairwise comparison: pruning vs DC at matched compression
    print("\n" + "=" * 70)
    print("PAIRWISE COMPARISON AT MATCHED OPERATING POINT")
    print("=" * 70)
    def lookup(name):
        times = np.array(all_measurements[name])
        return np.mean(times), np.median(times), np.std(times)
    pairs = [
        ("pruning_92sp", "DC_rpb1_hot4p3"),   # ~43x
        ("pruning_95sp", "DC_rpb16_hot4p3"),  # ~63x
        ("pruning_98sp", "DC_rpb16_hot2p0"),  # ~105-120x
        ("pruning_99sp", "DC_rpb16_hot1p0"),  # ~147-170x
    ]
    for a, b in pairs:
        ma, med_a, sa = lookup(a)
        mb, med_b, sb = lookup(b)
        diff_pct = (mb - ma) / ma * 100
        print(f"  {a:>20}: {ma:.3f}ms  vs  {b:>20}: {mb:.3f}ms  "
              f"→ DC is {diff_pct:+.1f}% slower")

    with open("results/dct_domain/dc_vs_pruning_latency_v2.json", "w") as f:
        json.dump({"n_runs": N_RUNS, "measure_per_run": MEASURE, "summary": summary}, f, indent=2)
    print("\nSaved results/dct_domain/dc_vs_pruning_latency_v2.json")

if __name__ == '__main__':
    main()
