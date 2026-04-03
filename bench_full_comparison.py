#!/usr/bin/env python3
"""
Full comparison: fp32 baseline vs all compression methods.
Same model, same test set, same machine, back-to-back.
Metrics: AUC, batch latency, embedding latency, memory, compression ratio.
"""
import os, sys, time, json, gc
import numpy as np
from scipy.fft import dctn
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/cc/expr/dlrm_minrui')
os.chdir('/home/cc/expr/dlrm_minrui')
import compressed_emb as _C
from codec_ondemand_benchmark import (
    load_model_and_data, CompressedEmbeddingBag, EMB_DIM, MODEL_PATH,
    HOTCOLD_DIR, REORDER_DIR, ONDEMAND_DIR, quantize_table, LARGE_TABLE_THRESHOLD,
)

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
BLOCK = 8; TILE = 4; RPB = 4


def measure(dlrm, test_batches, nt, label, use_cpp=False):
    """Run full inference, return metrics dict."""
    # Warmup
    with torch.no_grad():
        for X, o, i, T in test_batches[:20]:
            dlrm(X, o, i)

    scores, targets = [], []
    batch_times = []
    emb_times = []

    with torch.no_grad():
        for X, o, i, T in test_batches:
            if use_cpp:
                li = torch.stack(i) if isinstance(i, (list,tuple)) else (i if i.dim()==2 else i.view(nt,-1))
                lo = torch.stack(o) if isinstance(o, (list,tuple)) else (o if o.dim()==2 else o.view(nt,-1))
                t0 = time.perf_counter()
                _C.fast_forward(li, lo)
                emb_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            Z = dlrm(X, o, i)
            batch_times.append(time.perf_counter() - t0)
            scores.append(Z.numpy().ravel())
            targets.append(T.numpy().ravel())

    auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    r = {
        'auc': float(auc),
        'batch_p50_ms': float(np.median(batch_times) * 1000),
        'batch_p99_ms': float(np.percentile(batch_times, 99) * 1000),
        'batch_mean_ms': float(np.mean(batch_times) * 1000),
        'total_s': float(sum(batch_times)),
    }
    if emb_times:
        r['emb_p50_ms'] = float(np.median(emb_times) * 1000)
    return r


def setup_compressed_tables(dlrm, sd, ek, ln_emb, nt, is_hot, hi):
    """Setup CompressedEmbeddingBag + register in C++."""
    for t in TABLES:
        hot_w = sd[ek[t]][hi[t]].clone()
        o2h = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
        o2h[hi[t]] = torch.arange(len(hi[t]))
        o2c = torch.load(f'{REORDER_DIR}/orig_to_cold_reordered_{t}.pt', weights_only=True)
        dlrm.emb_l[t] = CompressedEmbeddingBag(
            hot_weight=hot_w, is_hot=is_hot[t], orig_to_hot=o2h,
            orig_to_cold_reordered=o2c, cold_cache=None,
            num_embeddings=int(ln_emb[t]), embedding_dim=EMB_DIM, quantize_hot=False)

    tk, ws, ms, sc_l, zp_l = [], [], [], [], []
    for i in range(nt):
        E = dlrm.emb_l[i]
        if isinstance(E, CompressedEmbeddingBag):
            tk.append(1); ws.append(E.hot_weight); ms.append(E.mapping)
            sc_l.append(0.0); zp_l.append(0)
        else:
            tk.append(0); ws.append(E.weight)
            ms.append(torch.empty(0, dtype=torch.int32))
            sc_l.append(0.0); zp_l.append(0)
    _C.register_tables(tk, ws, ms, sc_l, zp_l, False, True)


def apply_cpp(nt):
    def fn(lS_o, lS_i, emb_l, v_W_l):
        li = torch.stack(lS_i) if isinstance(lS_i, (list,tuple)) else (lS_i if lS_i.dim()==2 else lS_i.view(nt,-1))
        lo = torch.stack(lS_o) if isinstance(lS_o, (list,tuple)) else (lS_o if lS_o.dim()==2 else lS_o.view(nt,-1))
        return _C.fast_forward(li, lo)[-1]
    return fn


def reset_model(dlrm, sd, ek, ln_emb, nt):
    dlrm.apply_emb = dlrm.__class__.apply_emb.__get__(dlrm)
    for t in range(nt):
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        dlrm.emb_l[t].weight.data = sd[ek[t]].clone()
    gc.collect()


def main():
    print("=" * 80)
    print("FULL COMPARISON: All Methods, All Metrics")
    print("=" * 80)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    nt = len(ln_emb)
    torch.set_num_threads(32)

    is_hot = {t: torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True) for t in TABLES}
    hi = {t: torch.where(is_hot[t])[0] for t in TABLES}

    all_results = {}

    # ================================================================
    # 1. fp32 Baseline (Python EmbeddingBag)
    # ================================================================
    print("\n--- 1. fp32 Baseline (Python) ---")
    reset_model(dlrm, sd, ek, ln_emb, nt)
    r = measure(dlrm, test_batches, nt, "fp32", use_cpp=False)
    r['memory_mb'] = 2060.7
    r['cold_mb'] = 0
    r['hot_mb'] = 2060.7
    r['cache_mb'] = 0
    r['startup_ms'] = 0
    r['needs_ffmpeg'] = False
    r['retraining'] = False
    all_results['fp32_baseline'] = r
    print(f"  AUC={r['auc']:.6f}  batch={r['batch_p50_ms']:.2f}ms  mem={r['memory_mb']:.0f}MB")

    # ================================================================
    # 2. uint8 only (no codec, C++ fast_forward)
    # ================================================================
    print("\n--- 2. uint8 quantized (C++ fast_forward) ---")
    # Replace cold with uint8 dequantized
    for k in ek:
        t_idx = int(k.split('.')[1])
        dlrm.emb_l[t_idx] = nn.EmbeddingBag(int(ln_emb[t_idx]), EMB_DIM, mode='sum', sparse=True)
        w = sd[k].clone()
        if t_idx in TABLES:
            cold_order = np.load(f'{REORDER_DIR}/cold_order_{t_idx}.npy')
            q, s, zp = quantize_table(w[cold_order])
            w[cold_order] = (q.float() - zp) * s
        dlrm.emb_l[t_idx].weight.data = w

    r = measure(dlrm, test_batches, nt, "uint8", use_cpp=False)
    r['memory_mb'] = 2060.7  # still fp32 in memory (just quantized values)
    all_results['uint8_only'] = r
    print(f"  AUC={r['auc']:.6f}  batch={r['batch_p50_ms']:.2f}ms")
    reset_model(dlrm, sd, ek, ln_emb, nt)

    # ================================================================
    # 3. H.265 CRF=30 + decoded frame cache (C++)
    # ================================================================
    print("\n--- 3. H.265 CRF=30 + 40MB cache (C++) ---")
    setup_compressed_tables(dlrm, sd, ek, ln_emb, nt, is_hot, hi)

    dir_1080p = os.path.join(ONDEMAND_DIR, '1080p_crf30_nofilter')
    t_decode_start = time.perf_counter()
    cache_mb = 0
    for t in TABLES:
        frame_dir = os.path.join(dir_1080p, f'table_{t}')
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.h265')])
        paths = [os.path.join(frame_dir, f) for f in frame_files]
        if not paths: continue
        cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
        _, s, zp = quantize_table(sd[ek[t]][cold_order])
        decoded_frames = _C.batch_decode_fast(paths, 2, len(paths), True, False, False)
        rpf = 1920 * 1080 // EMB_DIM
        all_rows = torch.cat([_C.untile_frame_to_rows(f, rpf) for f in decoded_frames])[:len(cold_order)]
        _C.register_cold_flat(t, all_rows, float(s), float(zp), len(cold_order))
        cache_mb += all_rows.numel() / 1024 / 1024
    t_decode = (time.perf_counter() - t_decode_start) * 1000

    dlrm.apply_emb = apply_cpp(nt)
    r = measure(dlrm, test_batches, nt, "H.265+cache", use_cpp=True)
    r['memory_mb'] = 22.1 + 2.9 + 6.0 + 0.3 + cache_mb  # hot + small + bitmap + compressed + cache
    r['hot_mb'] = 22.1
    r['cold_mb'] = 0.3  # compressed H.265 in memory
    r['cache_mb'] = cache_mb
    r['startup_ms'] = t_decode
    r['needs_ffmpeg'] = True
    r['retraining'] = False
    all_results['h265_cache'] = r
    print(f"  AUC={r['auc']:.6f}  batch={r['batch_p50_ms']:.2f}ms  emb={r['emb_p50_ms']:.3f}ms  "
          f"mem={r['memory_mb']:.0f}MB  startup={t_decode:.0f}ms")
    reset_model(dlrm, sd, ek, ln_emb, nt)

    # ================================================================
    # 4. DCT-domain 8×8 (current, C++)
    # ================================================================
    print("\n--- 4. DCT-domain 8×8 blocks, step=32 (C++) ---")
    setup_compressed_tables(dlrm, sd, ek, ln_emb, nt, is_hot, hi)

    step = 32
    dct_cold_kb = 0
    for t in TABLES:
        cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
        q, s, zp = quantize_table(sd[ek[t]][cold_order])
        q_np = q.numpy()
        n = q_np.shape[0]; n_pad = ((n+RPB-1)//RPB)*RPB
        if n_pad > n:
            p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8); p[:n] = q_np; q_np = p
        nb = n_pad // RPB
        rows = q_np.reshape(nb, RPB, EMB_DIM).astype(np.float32)
        blocks = np.zeros((nb, BLOCK, BLOCK), dtype=np.float32)
        for r_i in range(RPB):
            tiles = rows[:, r_i, :].reshape(nb, TILE, TILE)
            blocks[:, (r_i//2)*TILE:(r_i//2)*TILE+TILE, (r_i%2)*TILE:(r_i%2)*TILE+TILE] = tiles
        dct = dctn(blocks, axes=(-2,-1), type=2, norm='ortho')
        dc = np.round(dct[:, 0, 0] / step).astype(np.int16)
        # Skip AC loop — register with empty AC (DC-only)
        ac_d = np.zeros(0, dtype=np.int16)
        ac_p = np.zeros(0, dtype=np.uint8)
        ac_o = np.zeros(nb + 1, dtype=np.int64)
        _C.register_cold_dct(t, torch.from_numpy(dc), torch.from_numpy(ac_d),
            torch.from_numpy(ac_p), torch.from_numpy(ac_o),
            float(step), float(s), float(zp), n)
        dct_cold_kb += nb / 1024  # uint8 in C++

    dlrm.apply_emb = apply_cpp(nt)
    r = measure(dlrm, test_batches, nt, "DCT 8×8", use_cpp=True)
    r['memory_mb'] = 22.1 + 2.9 + 6.0 + dct_cold_kb / 1024
    r['hot_mb'] = 22.1
    r['cold_mb'] = dct_cold_kb / 1024
    r['cache_mb'] = 0
    r['startup_ms'] = 0
    r['needs_ffmpeg'] = False
    r['retraining'] = False
    all_results['dct_8x8'] = r
    print(f"  AUC={r['auc']:.6f}  batch={r['batch_p50_ms']:.2f}ms  emb={r['emb_p50_ms']:.3f}ms  "
          f"mem={r['memory_mb']:.1f}MB")
    reset_model(dlrm, sd, ek, ln_emb, nt)

    # ================================================================
    # 5. DCT-domain 4×4 (1 row/block, best AUC)
    # Build full table with DC-only reconstruction, measure via Python
    # ================================================================
    print("\n--- 5. DCT-domain 4×4 blocks, step=32 (Python reconstruct) ---")
    dct_cold_kb_4x4 = 0
    for k in ek:
        t_idx = int(k.split('.')[1])
        dlrm.emb_l[t_idx] = nn.EmbeddingBag(int(ln_emb[t_idx]), EMB_DIM, mode='sum', sparse=True)
        w = sd[k].clone()
        if t_idx in TABLES:
            cold_order = np.load(f'{REORDER_DIR}/cold_order_{t_idx}.npy')
            q, s, zp = quantize_table(w[cold_order])
            q_np = q.numpy()
            n_cold = len(cold_order)
            # 4×4 DCT: each row is one block
            blocks_4 = q_np.reshape(n_cold, 4, 4).astype(np.float32)
            dct_4 = dctn(blocks_4, axes=(-2,-1), type=2, norm='ortho')
            dc_4 = np.round(dct_4[:, 0, 0] / step).astype(np.int16)
            # DC-only reconstruct: 1/4 weight for 4×4 ortho
            dc_val = (dc_4 * step * 0.25).astype(np.float32)
            recon = np.broadcast_to(dc_val[:, np.newaxis], (n_cold, EMB_DIM)).copy()
            recon = np.clip(np.round(recon), 0, 255).astype(np.uint8)
            decoded_fp32 = (torch.from_numpy(recon).float() - zp) * s
            w[cold_order] = decoded_fp32
            dct_cold_kb_4x4 += n_cold / 1024  # 1 byte per row (uint8 DC)
        dlrm.emb_l[t_idx].weight.data = w

    r = measure(dlrm, test_batches, nt, "DCT 4×4", use_cpp=False)
    r['memory_mb'] = 22.1 + 2.9 + 6.0 + dct_cold_kb_4x4 / 1024  # raw DC
    # With sparse: 99.9% default → ~183 KB
    r['memory_mb_sparse'] = 22.1 + 2.9 + 6.0 + 0.18
    r['hot_mb'] = 22.1
    r['cold_mb'] = dct_cold_kb_4x4 / 1024
    r['cold_sparse_kb'] = 183
    r['cache_mb'] = 0
    r['startup_ms'] = 0
    r['needs_ffmpeg'] = False
    r['retraining'] = False
    all_results['dct_4x4'] = r
    print(f"  AUC={r['auc']:.6f}  batch={r['batch_p50_ms']:.2f}ms  "
          f"mem={r['memory_mb']:.1f}MB (sparse={r['memory_mb_sparse']:.1f}MB)")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*80}")
    print("FULL COMPARISON TABLE")
    print(f"{'='*80}")

    auc_base = all_results['fp32_baseline']['auc']

    print(f"\n{'Method':<28} {'AUC':>10} {'Loss%':>8} {'Batch':>8} {'Emb':>7} "
          f"{'Memory':>8} {'Ratio':>6} {'Cache':>6} {'Start':>7} {'FFmpeg':>6}")
    print("-" * 105)

    rows = [
        ('fp32 baseline (Python)', 'fp32_baseline'),
        ('uint8 quant only', 'uint8_only'),
        ('H.265+cache (C++)', 'h265_cache'),
        ('DCT 8×8 DC-only (C++)', 'dct_8x8'),
        ('DCT 4×4 DC-only (Python)', 'dct_4x4'),
    ]

    for label, key in rows:
        r = all_results[key]
        loss = (auc_base - r['auc']) * 100
        mem = r.get('memory_mb_sparse', r['memory_mb'])
        ratio = 2060.7 / mem
        emb = f"{r['emb_p50_ms']:.3f}" if 'emb_p50_ms' in r else "—"
        cache = f"{r.get('cache_mb', 0):.0f}MB" if r.get('cache_mb', 0) > 0 else "0"
        start = f"{r.get('startup_ms', 0):.0f}ms" if r.get('startup_ms', 0) > 0 else "0"
        ffmpeg = "Yes" if r.get('needs_ffmpeg', False) else "No"
        print(f"{label:<28} {r['auc']:>10.6f} {loss:>+7.3f}% {r['batch_p50_ms']:>7.2f} "
              f"{emb:>7} {mem:>7.1f}MB {ratio:>5.0f}x {cache:>6} {start:>7} {ffmpeg:>6}")

    # Save
    os.makedirs('results/dct_domain', exist_ok=True)
    with open('results/dct_domain/full_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/dct_domain/full_comparison.json")


if __name__ == '__main__':
    main()
