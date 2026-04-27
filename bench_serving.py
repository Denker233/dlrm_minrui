#!/usr/bin/env python3
"""
End-to-end serving benchmark: batch latency, QPS, memory (RSS).

Compares: fp32 → uint8 → DC freq-sort → DC value-sort → PQ1x8 → Zero
at hot fractions 4.3%, 1%, 0.5%.

Methodology:
- Pre-build all weight tensors (no allocation during timing)
- Denormals flushed
- 60 warmup + 500 measured batches per config
- Reports: mean, p50, p99 latency; QPS; RSS delta
"""
import os, sys, time, json, gc, psutil
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')

MODEL_PATH = 'models/dlrm_kaggle_correct.pt'
LARGE_THRESHOLD = 50000
EMB_DIM = 16
BS = 16  # DC block size
WARMUP = 60
MEASURE = 500
BATCH_SIZE = 128

torch.set_flush_denormal(True)
torch.set_num_threads(32)

def quantize_table(w):
    mn, mx = w.min().item(), w.max().item()
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
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    ln_emb = np.array(train_data.counts)
    m_spa = args.arch_sparse_feature_size
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[-1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + args.arch_mlp_top, dtype=int, sep="-")
    dlrm = DLRM_Net(m_spa, ln_emb, ln_bot, ln_top,
                     arch_interaction_op="dot", arch_interaction_itself=False,
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")
    ld = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    return dlrm, test_ld, ln_emb

def build_dc_weight(w_orig, sorted_idx, hf, value_sort=False):
    """Build DC block-mean weight tensor with 4-bit quantized means."""
    n = w_orig.shape[0]
    w = w_orig.clone()
    n_hot = max(1, int(n * hf))
    hot_idx = sorted_idx[:n_hot]
    cold_idx = sorted_idx[n_hot:]
    cold_w = w[cold_idx]
    nc = len(cold_idx)

    if value_sort:
        row_means = cold_w.mean(dim=1)
        order = torch.argsort(row_means)
        cold_w = cold_w[order]
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(nc)
    else:
        inv_order = None

    # Compute block means
    nb = (nc + BS - 1) // BS
    padded = torch.zeros(nb * BS, EMB_DIM)
    padded[:nc] = cold_w
    means = padded.reshape(nb, BS, EMB_DIM).mean(dim=(1, 2))

    # 4-bit quantize
    mn, mx = means.min().item(), means.max().item()
    if mx != mn:
        s = (mx - mn) / 15.0
        q = ((means - mn) / s).round().clamp(0, 15)
        means = q * s + mn

    recon = means.unsqueeze(1).unsqueeze(2).expand(nb, BS, EMB_DIM).reshape(-1, EMB_DIM)[:nc]
    if inv_order is not None:
        recon = recon[inv_order]
    w[cold_idx] = recon

    # Hot rows: uint8 dequantized
    if n_hot > 0:
        q_h, s_h, zp_h = quantize_table(w[hot_idx])
        w[hot_idx] = (q_h.float() - zp_h) * s_h
    return w

def build_pq_weight(w_orig, sorted_idx, hf):
    """Build PQ1x8 weight tensor."""
    import faiss
    n = w_orig.shape[0]
    w = w_orig.clone()
    n_hot = max(1, int(n * hf))
    hot_idx = sorted_idx[:n_hot]
    cold_idx = sorted_idx[n_hot:]

    cold_fp32 = w[cold_idx].numpy().copy()
    cold_fp32 = np.ascontiguousarray(cold_fp32, dtype=np.float32)

    # Train PQ on sample
    sample_n = min(100000, len(cold_fp32))
    sample = cold_fp32[np.random.choice(len(cold_fp32), sample_n, replace=False)]
    pq = faiss.ProductQuantizer(EMB_DIM, 1, 8)
    pq.train(sample)
    codes = pq.compute_codes(cold_fp32)
    recon = pq.decode(codes)
    w[cold_idx] = torch.from_numpy(recon)

    if n_hot > 0:
        q_h, s_h, zp_h = quantize_table(w[hot_idx])
        w[hot_idx] = (q_h.float() - zp_h) * s_h
    return w

def get_rss_mb():
    return psutil.Process().memory_info().rss / 1024**2

def main():
    print("=" * 70)
    print("SERVING BENCHMARK: batch latency, QPS, memory")
    print(f"  batch_size={BATCH_SIZE}, warmup={WARMUP}, measure={MEASURE}")
    print(f"  threads={torch.get_num_threads()}")
    print("=" * 70)

    dlrm, test_ld, ln_emb = load_kaggle()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    TABLES = [i for i in range(nt) if ln_emb[i] > LARGE_THRESHOLD]

    # Pre-collect test batches
    test_batches = []
    for X, lS_o, lS_i, T in test_ld:
        test_batches.append((X, lS_o, lS_i, T))
    print(f"Loaded {len(test_batches)} test batches")

    # Profile access frequencies
    print("Profiling...")
    freq = {}
    for X, lS_o, lS_i, T in test_batches:
        for t in TABLES:
            idx = lS_i[t].flatten() if isinstance(lS_i, (list, tuple)) else lS_i[t].flatten()
            if t not in freq:
                freq[t] = torch.zeros(int(ln_emb[t]), dtype=torch.long)
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))

    sorted_idx = {t: freq[t].argsort(descending=True) for t in TABLES}
    orig_w = {t: sd[ek[t]].clone() for t in range(nt)}

    # Build all configs
    configs = [
        ('fp32', {}),
        ('uint8_all', {}),
    ]
    for hf in [0.043, 0.01, 0.005]:
        configs.extend([
            (f'zero_hf{hf}', {'hf': hf}),
            (f'dc_freq_hf{hf}', {'hf': hf, 'value_sort': False}),
            (f'dc_value_hf{hf}', {'hf': hf, 'value_sort': True}),
            (f'pq1x8_hf{hf}', {'hf': hf}),
        ])

    print(f"\nPre-building {len(configs)} weight configs...")
    prebuilt = {}
    for name, params in configs:
        weights = {}
        for t in range(nt):
            w = orig_w[t].clone()
            if name == 'fp32':
                weights[t] = w
            elif name == 'uint8_all':
                q, s, zp = quantize_table(w)
                weights[t] = (q.float() - zp) * s
            elif name.startswith('zero'):
                hf = params['hf']
                if t in TABLES:
                    n_hot = max(1, int(int(ln_emb[t]) * hf))
                    w[sorted_idx[t][n_hot:]] = 0.0
                    q_h, s_h, zp_h = quantize_table(w[sorted_idx[t][:n_hot]])
                    w[sorted_idx[t][:n_hot]] = (q_h.float() - zp_h) * s_h
                else:
                    q, s, zp = quantize_table(w)
                    w = (q.float() - zp) * s
                weights[t] = w
            elif name.startswith('dc_'):
                hf = params['hf']
                vs = params['value_sort']
                if t in TABLES:
                    weights[t] = build_dc_weight(w, sorted_idx[t], hf, vs)
                else:
                    q, s, zp = quantize_table(w)
                    weights[t] = (q.float() - zp) * s
            elif name.startswith('pq'):
                hf = params['hf']
                if t in TABLES:
                    weights[t] = build_pq_weight(w, sorted_idx[t], hf)
                else:
                    q, s, zp = quantize_table(w)
                    weights[t] = (q.float() - zp) * s
        prebuilt[name] = weights
        print(f"  {name}: built")

    # Measure
    print(f"\n{'config':>25} {'mean':>8} {'p50':>8} {'p99':>8} {'QPS':>8} {'RSS MB':>8}")
    print("-" * 75)

    results = []

    for name, params in configs:
        # Apply weights
        for t in range(nt):
            dlrm.emb_l[t].weight.data = prebuilt[name][t]

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        rss_before = get_rss_mb()

        # Warmup
        with torch.no_grad():
            for bi in range(WARMUP):
                X, lS_o, lS_i, T = test_batches[bi % len(test_batches)]
                dlrm(X, lS_o, lS_i)

        # Measure
        latencies = []
        with torch.no_grad():
            for bi in range(MEASURE):
                X, lS_o, lS_i, T = test_batches[bi % len(test_batches)]
                t0 = time.perf_counter()
                dlrm(X, lS_o, lS_i)
                latencies.append((time.perf_counter() - t0) * 1000)

        rss_after = get_rss_mb()

        lat = np.array(latencies)
        mean_ms = lat.mean()
        p50 = np.percentile(lat, 50)
        p99 = np.percentile(lat, 99)
        qps = BATCH_SIZE * 1000.0 / mean_ms

        print(f"{name:>25} {mean_ms:>7.2f}ms {p50:>7.2f}ms {p99:>7.2f}ms {qps:>7.0f} {rss_after:>7.0f}")

        results.append({
            'config': name,
            'mean_ms': mean_ms, 'p50_ms': p50, 'p99_ms': p99,
            'qps': qps, 'rss_mb': rss_after,
            'params': params,
        })

    # Restore fp32
    for t in range(nt):
        dlrm.emb_l[t].weight.data = orig_w[t]

    out_path = 'results/serving_benchmark.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    print("Done.")

if __name__ == '__main__':
    main()
