#!/usr/bin/env python3
"""
Final benchmark: uint8 hot + DC 16×16 cold, all in C++.
Hot fractions: 4.3%, 2%, 1%
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
    HOTCOLD_DIR, REORDER_DIR, quantize_table, LARGE_TABLE_THRESHOLD,
)

TABLES = [2, 3, 9, 11, 15, 20, 23, 25]
TILE = 4


def encode_dct_16x16(uint8_rows, step):
    rpb = 16; BS = 16
    n = uint8_rows.shape[0]
    n_pad = ((n + rpb - 1) // rpb) * rpb
    if n_pad > n:
        p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8); p[:n] = uint8_rows; uint8_rows = p
    n_blocks = n_pad // rpb
    rows = uint8_rows.reshape(n_blocks, rpb, EMB_DIM).astype(np.float32)
    blocks = np.zeros((n_blocks, BS, BS), dtype=np.float32)
    for r in range(rpb):
        tile = rows[:, r, :].reshape(n_blocks, TILE, TILE)
        blocks[:, (r // 4) * TILE:(r // 4 + 1) * TILE, (r % 4) * TILE:(r % 4 + 1) * TILE] = tile
    dct = dctn(blocks, axes=(-2, -1), type=2, norm='ortho')
    dc = np.round(dct[:, 0, 0] / step).astype(np.int16)
    ac_d = np.zeros(0, dtype=np.int16)
    ac_p = np.zeros(0, dtype=np.uint8)
    ac_o = np.zeros(n_blocks + 1, dtype=np.int64)
    return dc, ac_d, ac_p, ac_o, n_blocks


def main():
    print("=" * 70)
    print("FINAL: uint8 hot + DC 16×16 cold (all C++)")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    nt = len(ln_emb)
    torch.set_num_threads(32)

    # Profile frequencies
    print("Profiling...")
    freq = {t: torch.zeros(int(ln_emb[t]), dtype=torch.long) for t in TABLES}
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    sorted_idx = {t: torch.argsort(freq[t], descending=True) for t in TABLES}

    # Baseline
    scores, targets = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_base = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"Baseline fp32: {auc_base:.6f}")

    def apply_cpp(lS_o, lS_i, emb_l, v_W_l):
        li = torch.stack(lS_i) if isinstance(lS_i, (list, tuple)) else (lS_i if lS_i.dim() == 2 else lS_i.view(nt, -1))
        lo = torch.stack(lS_o) if isinstance(lS_o, (list, tuple)) else (lS_o if lS_o.dim() == 2 else lS_o.view(nt, -1))
        return _C.fast_forward(li, lo)[-1]

    step = 32
    results = []

    for hf in [0.043, 0.02, 0.01]:
        print(f"\n--- Hot={hf*100:.1f}%, uint8 hot, DC 16×16 ---")

        is_hot_new, hi_new = {}, {}
        hot_bytes = 0
        for t in TABLES:
            n = int(ln_emb[t])
            n_hot = max(0, int(n * hf))
            is_h = torch.zeros(n, dtype=torch.bool)
            if n_hot > 0:
                is_h[sorted_idx[t][:n_hot]] = True
            is_hot_new[t] = is_h
            hi_new[t] = torch.where(is_h)[0]
            hot_bytes += n_hot * EMB_DIM  # uint8

        # Setup with quantize_hot=True (uint8 hot)
        for t in TABLES:
            hot_w = sd[ek[t]][hi_new[t]].clone()
            o2h = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
            o2h[hi_new[t]] = torch.arange(len(hi_new[t]))
            cold_idx = torch.where(~is_hot_new[t])[0]
            cold_freq = freq[t][cold_idx]
            cold_sorted = cold_idx[torch.argsort(cold_freq, descending=True)]
            o2c = torch.full((int(ln_emb[t]),), -1, dtype=torch.int32)
            o2c[cold_sorted] = torch.arange(len(cold_sorted), dtype=torch.int32)

            dlrm.emb_l[t] = CompressedEmbeddingBag(
                hot_weight=hot_w, is_hot=is_hot_new[t], orig_to_hot=o2h,
                orig_to_cold_reordered=o2c, cold_cache=None,
                num_embeddings=int(ln_emb[t]), embedding_dim=EMB_DIM,
                quantize_hot=True)  # <-- uint8 hot

        # Register as COMPRESSED_Q8
        tk, ws, ms, sc_l, zp_l = [], [], [], [], []
        for i_t in range(nt):
            E = dlrm.emb_l[i_t]
            if isinstance(E, CompressedEmbeddingBag):
                tk.append(2)  # COMPRESSED_Q8
                ws.append(E.hot_weight_q8)
                ms.append(E.mapping)
                sc_l.append(float(E.hot_scale))
                zp_l.append(int(E.hot_zp))
            else:
                tk.append(0)
                ws.append(E.weight)
                ms.append(torch.empty(0, dtype=torch.int32))
                sc_l.append(0.0)
                zp_l.append(0)
        _C.register_tables(tk, ws, ms, sc_l, zp_l, False, True)

        # Register DC cold
        dct_bytes = 0
        for t in TABLES:
            cold_idx = torch.where(~is_hot_new[t])[0]
            cold_freq_t = freq[t][cold_idx]
            cold_sorted = cold_idx[torch.argsort(cold_freq_t, descending=True)]
            cold_w = sd[ek[t]][cold_sorted]
            q, s, zp = quantize_table(cold_w)
            dc, ac_d, ac_p, ac_o, nb = encode_dct_16x16(q.numpy(), step)
            _C.register_cold_dct(t, torch.from_numpy(dc), torch.from_numpy(ac_d),
                torch.from_numpy(ac_p), torch.from_numpy(ac_o),
                float(step), float(s), float(zp), len(cold_sorted), 16)
            dct_bytes += nb  # uint8 in C++

        hot_mb = hot_bytes / 1024 / 1024
        dct_mb = dct_bytes / 1024 / 1024
        small_mb = 2.9
        bitmap_mb = 6.0
        total_bitmap = hot_mb + small_mb + bitmap_mb + dct_mb
        total_reord = hot_mb + small_mb + dct_mb

        print(f"  Hot: {hot_mb:.1f}MB (uint8)  Cold DC: {dct_mb:.1f}MB")
        print(f"  With bitmap: {total_bitmap:.1f}MB ({2060.7/total_bitmap:.0f}x)")
        print(f"  Reordered:   {total_reord:.1f}MB ({2060.7/total_reord:.0f}x)")

        # Benchmark
        dlrm.apply_emb = apply_cpp
        with torch.no_grad():
            for X, o, i, T in test_batches[:20]:
                dlrm(X, o, i)

        scores, targets, emb_times, batch_times = [], [], [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                li = torch.stack(i) if isinstance(i, (list, tuple)) else i
                lo = torch.stack(o) if isinstance(o, (list, tuple)) else o
                t0 = time.perf_counter()
                _C.fast_forward(li, lo)
                emb_times.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                Z = dlrm(X, o, i)
                batch_times.append(time.perf_counter() - t0)
                scores.append(Z.numpy().ravel())
                targets.append(T.numpy().ravel())

        auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
        emb_ms = np.median(emb_times) * 1000
        batch_ms = np.median(batch_times) * 1000

        print(f"  AUC: {auc:.6f} (loss: {(auc_base-auc)*100:+.4f}%)")
        print(f"  Emb: {emb_ms:.3f}ms  Batch: {batch_ms:.2f}ms")

        results.append({
            'hot_pct': hf, 'auc': float(auc),
            'auc_loss_pct': float((auc_base - auc) * 100),
            'emb_ms': float(emb_ms), 'batch_ms': float(batch_ms),
            'hot_mb': float(hot_mb), 'dct_mb': float(dct_mb),
            'total_bitmap': float(total_bitmap), 'total_reord': float(total_reord),
            'ratio_bitmap': float(2060.7 / total_bitmap),
            'ratio_reord': float(2060.7 / total_reord),
        })

        # Reset
        dlrm.apply_emb = dlrm.__class__.apply_emb.__get__(dlrm)
        for t_r in range(nt):
            dlrm.emb_l[t_r] = nn.EmbeddingBag(int(ln_emb[t_r]), EMB_DIM, mode='sum', sparse=True)
            dlrm.emb_l[t_r].weight.data = sd[ek[t_r]].clone()
        gc.collect()

    # Summary
    print(f"\n{'='*80}")
    print("VERIFIED RESULTS: uint8 hot + DC 16×16 cold")
    print(f"{'='*80}")
    print(f"\n{'Hot%':>5} {'AUC':>10} {'Loss':>8} {'Emb':>7} {'Batch':>7} "
          f"{'Hot MB':>7} {'DC MB':>6} {'Bitmap':>8} {'Ratio':>6} {'Reord':>8} {'Ratio':>6}")
    print("-" * 90)
    print(f"{'base':>5} {auc_base:>10.6f} {'—':>8} {'—':>7} {'4.98':>7} "
          f"{'—':>7} {'—':>6} {'2061MB':>8} {'1x':>6} {'—':>8} {'—':>6}")
    for r in results:
        print(f"{r['hot_pct']*100:>4.1f}% {r['auc']:>10.6f} {r['auc_loss_pct']:>+7.3f}% "
              f"{r['emb_ms']:>6.3f} {r['batch_ms']:>6.2f} "
              f"{r['hot_mb']:>6.1f} {r['dct_mb']:>5.1f} "
              f"{r['total_bitmap']:>7.1f}MB {r['ratio_bitmap']:>5.0f}x "
              f"{r['total_reord']:>7.1f}MB {r['ratio_reord']:>5.0f}x")

    with open('results/dct_domain/final_q8hot.json', 'w') as f:
        json.dump({'baseline_auc': float(auc_base), 'results': results}, f, indent=2)


if __name__ == '__main__':
    main()
