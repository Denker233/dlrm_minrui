#!/usr/bin/env python3
"""
V3 latency benchmark: setup is fully decoupled from timing.

Key differences vs v2:
  1. EmbeddingBag instances are allocated ONCE at startup. The measurement
     loop never reallocates anything.
  2. All 12 config weight tensors are pre-built upfront and stored. The
     measurement loop only does a pointer swap (weight.data = prebuilt[name][t]).
     No numpy preprocessing, no quantize/dequantize, no allocations between
     apply_config and the timed region.
  3. Denormals flushed (torch.set_flush_denormal) to rule out subnormal effects.
  4. Larger warmup (60), more samples per run (200), more runs (5), trimmed
     mean reported alongside mean/median to handle system-noise outliers.

This isolates "what bytes sit in the weight tensor" from "what cache/allocator
state was left behind by the apply step", which was the dominant confound in v2.
"""
import os, sys, json, time, gc, random
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

MODEL_PATH = 'models/dlrm_kaggle_correct.pt'
LARGE_THRESHOLD = 50000
EMB_DIM = 16
WARMUP = 60       # per run
MEASURE = 200     # per run
N_RUNS = 5        # times each config is measured
TRIM_PCT = 5      # trim N% from each tail before computing trimmed mean

random.seed(42)
torch.set_flush_denormal(True)


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


def build_weight(mode, t, n, w_orig, sorted_idx_t, hf=None, rpb=None, sp=None):
    """Return a fresh fp32 weight tensor for table t under the given mode."""
    w = w_orig.clone()
    if mode == 'fp32':
        return w
    if mode == 'uint8_all':
        q, s, zp = quantize_table(w)
        return (q.float() - zp) * s
    if mode == 'pruning':
        keep_n = max(1, int(round(n * (1 - sp / 100.0))))
        keep_idx = sorted_idx_t[:keep_n]
        drop_idx = sorted_idx_t[keep_n:]
        w[drop_idx] = 0.0
        if len(keep_idx) > 0:
            q_h, s_h, zp_h = quantize_table(w[keep_idx])
            w[keep_idx] = (q_h.float() - zp_h) * s_h
        return w
    if mode == 'dc':
        n_hot = int(n * hf)
        hot_idx = sorted_idx_t[:n_hot]
        cold_idx = sorted_idx_t[n_hot:]
        cold_w = w[cold_idx]
        q, s, zp = quantize_table(cold_w)
        nc = len(cold_idx)
        n_pad = ((nc + rpb - 1) // rpb) * rpb
        q_np = q.numpy()
        if n_pad > nc:
            p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8)
            p[:nc] = q_np
            q_np = p
        nb = n_pad // rpb
        block_means = q_np.reshape(nb, rpb, EMB_DIM).astype(np.float32).mean(axis=(1, 2), keepdims=True)
        recon = np.broadcast_to(block_means, (nb, rpb, EMB_DIM)).reshape(n_pad, EMB_DIM)[:nc]
        recon = np.clip(np.round(recon), 0, 255).astype(np.uint8)
        w[cold_idx] = (torch.from_numpy(recon.copy()).float() - zp) * s
        if n_hot > 0:
            q_h, s_h, zp_h = quantize_table(w[hot_idx])
            w[hot_idx] = (q_h.float() - zp_h) * s_h
        return w
    raise ValueError(f"unknown mode {mode}")


def main():
    print("=" * 80)
    print("DC vs PRUNING BATCH LATENCY V3")
    print("  - setup decoupled from timing (pre-built weight tensors)")
    print("  - denormals flushed")
    print(f"  - {N_RUNS} runs x {MEASURE} batches per config (randomized order)")
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
    print(f"Cached {len(test_batches)} test batches "
          f"({WARMUP} warmup + {MEASURE} measure)")

    print("Profiling frequencies on cached test batches...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    sorted_idx = {t: torch.argsort(freq[t], descending=True) for t in TABLES}

    # Re-instantiate EmbeddingBag instances ONCE. After this point, no
    # EmbeddingBag is ever reallocated; only weight.data is rebound.
    print("Allocating EmbeddingBag instances (one-time)...")
    for k in ek:
        t = int(k.split('.')[1])
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        dlrm.emb_l[t].weight.data = sd[k].clone()

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

    # Pre-build all weight tensors for every (config, large-table) pair.
    # Small tables are unchanged across configs and stay as the originals.
    print(f"Pre-building {len(configs)} configs x {len(TABLES)} large tables...")
    prebuilt = {}  # name -> {t: fp32 tensor}
    for name, kwargs in configs:
        prebuilt[name] = {}
        for k in ek:
            t = int(k.split('.')[1])
            if t not in TABLES:
                continue
            n = int(ln_emb[t])
            w_orig = sd[k]
            prebuilt[name][t] = build_weight(
                t=t, n=n, w_orig=w_orig, sorted_idx_t=sorted_idx[t], **kwargs
            )
        print(f"  built {name}")

    # Free the source state dict — no longer needed.
    del sd
    gc.collect()

    def apply_prebuilt(name):
        # Pure pointer swap: no allocations, no memcpy, no compute.
        for t, w in prebuilt[name].items():
            dlrm.emb_l[t].weight.data = w

    def measure_once():
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

    all_measurements = {name: [] for name, _ in configs}
    per_run_means = {name: [] for name, _ in configs}

    for run_idx in range(N_RUNS):
        run_order = configs[:]
        random.shuffle(run_order)
        print(f"\n--- Run {run_idx+1}/{N_RUNS} (shuffled) ---")
        for name, _ in run_order:
            apply_prebuilt(name)
            gc.collect()
            times = measure_once()
            all_measurements[name].extend(times)
            per_run_means[name].append(float(np.mean(times)))
            print(f"  {name:>20}: mean={np.mean(times):.3f}ms median={np.median(times):.3f}ms")

    def trimmed_mean(arr, pct):
        a = np.sort(arr)
        k = int(len(a) * pct / 100)
        return float(a[k:len(a) - k].mean()) if k > 0 else float(a.mean())

    print("\n" + "=" * 110)
    print(f"AGGREGATED ({N_RUNS} runs x {MEASURE} batches = {N_RUNS*MEASURE} samples per config)")
    print("=" * 110)
    print(f"{'Config':>22} {'Mean':>9} {'Trim5%':>9} {'Median':>9} "
          f"{'Std':>8} {'p5':>8} {'p95':>8} {'Min':>8} {'Max':>8} {'RunStd':>8}")
    print("-" * 110)

    summary = []
    for name, _ in sorted(configs, key=lambda c: trimmed_mean(all_measurements[c[0]], TRIM_PCT)):
        times = np.array(all_measurements[name])
        runs = np.array(per_run_means[name])
        row = {
            "name": name, "n": len(times),
            "mean_ms": float(times.mean()),
            "trimmed_mean_ms": trimmed_mean(times, TRIM_PCT),
            "median_ms": float(np.median(times)),
            "std_ms": float(times.std()),
            "p5_ms": float(np.percentile(times, 5)),
            "p95_ms": float(np.percentile(times, 95)),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "per_run_means": runs.tolist(),
            "across_run_std_ms": float(runs.std()),
        }
        summary.append(row)
        print(f"{name:>22} {row['mean_ms']:>8.3f}  {row['trimmed_mean_ms']:>8.3f}  "
              f"{row['median_ms']:>8.3f}  {row['std_ms']:>7.3f}  "
              f"{row['p5_ms']:>7.3f}  {row['p95_ms']:>7.3f}  "
              f"{row['min_ms']:>7.3f}  {row['max_ms']:>7.3f}  "
              f"{row['across_run_std_ms']:>7.3f}")

    # Pairwise comparison at matched compression operating points.
    print("\n" + "=" * 90)
    print("PAIRWISE: pruning vs DC at matched compression (Welch's t on 1000 samples each)")
    print("=" * 90)

    def stats(name):
        t = np.array(all_measurements[name])
        return t, trimmed_mean(t, TRIM_PCT), float(np.median(t)), float(t.std())

    pairs = [
        ("pruning_92sp", "DC_rpb1_hot4p3"),    # ~43x
        ("pruning_95sp", "DC_rpb16_hot4p3"),   # ~63x
        ("pruning_98sp", "DC_rpb16_hot2p0"),   # ~105-120x
        ("pruning_99sp", "DC_rpb16_hot1p0"),   # ~147-170x
    ]
    for a, b in pairs:
        ta, tma, meda, sa = stats(a)
        tb, tmb, medb, sb = stats(b)
        # Welch's t on the trimmed means (use medians as robust estimate)
        gap = tmb - tma
        gap_pct = gap / tma * 100
        # Pooled SE for unequal variances
        se = float(np.sqrt(sa**2 / len(ta) + sb**2 / len(tb)))
        z = gap / se if se > 0 else float('inf')
        sig = "**" if abs(z) > 2.58 else "*" if abs(z) > 1.96 else "ns"
        print(f"  {a:>20} trim={tma:.3f}ms  vs  {b:>20} trim={tmb:.3f}ms  "
              f"-> Δ={gap:+.3f}ms ({gap_pct:+.1f}%)  z={z:+.2f}  {sig}")

    os.makedirs("results/dct_domain", exist_ok=True)
    out_path = "results/dct_domain/dc_vs_pruning_latency_v3.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_runs": N_RUNS,
            "measure_per_run": MEASURE,
            "warmup_per_run": WARMUP,
            "trim_pct": TRIM_PCT,
            "denormals_flushed": True,
            "decoupled_setup": True,
            "summary": summary,
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
