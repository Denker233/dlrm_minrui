#!/usr/bin/env python3
"""
Sweep hot fraction: 4.3% (current), 2%, 1%, 0.5%, 0% (all DCT).
For each: measure AUC, batch latency, memory breakdown.
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
BLOCK=8; TILE=4; RPB=4

def tile_and_dct(uint8_rows, step_size):
    n = uint8_rows.shape[0]
    n_pad = ((n + RPB - 1) // RPB) * RPB
    if n_pad > n:
        p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8); p[:n] = uint8_rows; uint8_rows = p
    n_blocks = n_pad // RPB
    rows = uint8_rows.reshape(n_blocks, RPB, EMB_DIM).astype(np.float32)
    blocks = np.zeros((n_blocks, BLOCK, BLOCK), dtype=np.float32)
    for r in range(RPB):
        tiles = rows[:, r, :].reshape(n_blocks, TILE, TILE)
        blocks[:, (r//2)*TILE:(r//2)*TILE+TILE, (r%2)*TILE:(r%2)*TILE+TILE] = tiles
    dct = dctn(blocks, axes=(-2,-1), type=2, norm='ortho')
    quantized = np.round(dct / step_size).astype(np.int16)
    dc = quantized[:, 0, 0].copy()
    ac_data, ac_pos, ac_off = [], [], [0]
    for bi in range(n_blocks):
        q = quantized[bi]
        for u in range(8):
            for v in range(8):
                if (u,v)==(0,0): continue
                if q[u,v] != 0: ac_pos.append(u*8+v); ac_data.append(q[u,v])
        ac_off.append(len(ac_data))
    return dc, np.array(ac_data, dtype=np.int16), np.array(ac_pos, dtype=np.uint8), np.array(ac_off, dtype=np.int64), n_blocks


def main():
    print("=" * 70)
    print("HOT FRACTION SWEEP: AUC vs Memory vs Latency")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    nt = len(ln_emb)
    torch.set_num_threads(32)

    # Load access frequency profiles to determine hot rows at different thresholds
    # Profile: count how many times each row is accessed across test batches
    print("Profiling access frequencies...")
    freq = {}
    for t in TABLES:
        freq[t] = torch.zeros(int(ln_emb[t]), dtype=torch.long)
    for X, o, i, T in test_batches:
        for t in TABLES:
            idx = i[t] if isinstance(i, (list, tuple)) else i[t]
            freq[t].scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))

    # For each table, sort by frequency
    sorted_indices = {}
    for t in TABLES:
        sorted_indices[t] = torch.argsort(freq[t], descending=True)

    # Baseline AUC
    print("Baseline AUC...")
    for t in range(nt):
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        dlrm.emb_l[t].weight.data = sd[ek[t]].clone()
    scores, targets = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_base = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"  Baseline: {auc_base:.6f}")

    # Test different hot fractions
    hot_fractions = [0.043, 0.02, 0.01, 0.005, 0.0]
    step = 32
    results = []

    for hf in hot_fractions:
        print(f"\n{'='*60}")
        print(f"HOT FRACTION = {hf*100:.1f}%")
        print(f"{'='*60}")

        # Determine hot rows for each table
        is_hot_new = {}
        hi_new = {}
        n_hot_total = 0
        n_cold_total = 0
        hot_mem = 0

        for t in TABLES:
            n = int(ln_emb[t])
            n_hot = max(0, int(n * hf))
            # Top n_hot by frequency
            is_h = torch.zeros(n, dtype=torch.bool)
            if n_hot > 0:
                is_h[sorted_indices[t][:n_hot]] = True
            is_hot_new[t] = is_h
            hi_new[t] = torch.where(is_h)[0]
            n_hot_total += n_hot
            n_cold = n - n_hot
            n_cold_total += n_cold
            hot_mem += n_hot * EMB_DIM  # uint8

        print(f"  Hot rows: {n_hot_total:,} ({hot_mem/1024/1024:.1f} MB uint8)")
        print(f"  Cold rows: {n_cold_total:,}")

        if hf == 0:
            # All DCT, no hot
            # Register all tables as STANDARD (small tables) but override large ones
            for t in range(nt):
                dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
                dlrm.emb_l[t].weight.data = sd[ek[t]].clone()

            # For 0% hot: we need a different approach
            # Use CompressedEmbeddingBag with empty hot weight
            for t in TABLES:
                # All rows are cold
                hot_w = torch.zeros(0, EMB_DIM, dtype=torch.float32)
                is_h = torch.zeros(int(ln_emb[t]), dtype=torch.bool)
                o2h = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
                # Cold mapping: identity (all rows are cold, rank = natural order)
                o2c = torch.arange(int(ln_emb[t]), dtype=torch.int32)

                dlrm.emb_l[t] = CompressedEmbeddingBag(
                    hot_weight=hot_w, is_hot=is_h, orig_to_hot=o2h,
                    orig_to_cold_reordered=o2c, cold_cache=None,
                    num_embeddings=int(ln_emb[t]), embedding_dim=EMB_DIM, quantize_hot=False)

            tk, ws, ms, sc_l, zp_l = [], [], [], [], []
            for i_t in range(nt):
                E = dlrm.emb_l[i_t]
                if isinstance(E, CompressedEmbeddingBag):
                    tk.append(1); ws.append(torch.zeros(1, EMB_DIM))  # dummy
                    ms.append(E.mapping); sc_l.append(0.0); zp_l.append(0)
                else:
                    tk.append(0); ws.append(E.weight)
                    ms.append(torch.empty(0, dtype=torch.int32)); sc_l.append(0.0); zp_l.append(0)
            _C.register_tables(tk, ws, ms, sc_l, zp_l, False, True)

            # Register DCT for ALL rows
            dct_mem = 0
            for t in TABLES:
                q, s, zp = quantize_table(sd[ek[t]])
                dc, ac_d, ac_p, ac_o, nb = tile_and_dct(q.numpy(), step)
                _C.register_cold_dct(t, torch.from_numpy(dc), torch.from_numpy(ac_d),
                    torch.from_numpy(ac_p), torch.from_numpy(ac_o),
                    float(step), float(s), float(zp), int(ln_emb[t]))
                dct_mem += len(dc)  # uint8 in C++
            print(f"  DCT cold: {dct_mem/1024:.0f} KB")

        else:
            # Normal hot/cold split with custom hot fraction
            for t in TABLES:
                hot_w = sd[ek[t]][hi_new[t]].clone()
                o2h = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
                o2h[hi_new[t]] = torch.arange(len(hi_new[t]))

                # Cold: all non-hot rows, sorted by frequency for DCT
                cold_idx = torch.where(~is_hot_new[t])[0]
                # Sort cold by frequency (most accessed first)
                cold_freq = freq[t][cold_idx]
                cold_sorted = cold_idx[torch.argsort(cold_freq, descending=True)]
                # Build cold mapping
                o2c = torch.full((int(ln_emb[t]),), -1, dtype=torch.int32)
                o2c[cold_sorted] = torch.arange(len(cold_sorted), dtype=torch.int32)

                dlrm.emb_l[t] = CompressedEmbeddingBag(
                    hot_weight=hot_w, is_hot=is_hot_new[t], orig_to_hot=o2h,
                    orig_to_cold_reordered=o2c, cold_cache=None,
                    num_embeddings=int(ln_emb[t]), embedding_dim=EMB_DIM, quantize_hot=False)

            tk, ws, ms, sc_l, zp_l = [], [], [], [], []
            for i_t in range(nt):
                E = dlrm.emb_l[i_t]
                if isinstance(E, CompressedEmbeddingBag):
                    tk.append(1); ws.append(E.hot_weight); ms.append(E.mapping)
                    sc_l.append(0.0); zp_l.append(0)
                else:
                    tk.append(0); ws.append(E.weight)
                    ms.append(torch.empty(0, dtype=torch.int32)); sc_l.append(0.0); zp_l.append(0)
            _C.register_tables(tk, ws, ms, sc_l, zp_l, False, True)

            # Register DCT for cold rows (frequency-sorted)
            dct_mem = 0
            for t in TABLES:
                cold_idx = torch.where(~is_hot_new[t])[0]
                cold_freq_t = freq[t][cold_idx]
                cold_sorted = cold_idx[torch.argsort(cold_freq_t, descending=True)]
                cold_w = sd[ek[t]][cold_sorted]
                q, s, zp = quantize_table(cold_w)
                dc, ac_d, ac_p, ac_o, nb = tile_and_dct(q.numpy(), step)
                _C.register_cold_dct(t, torch.from_numpy(dc), torch.from_numpy(ac_d),
                    torch.from_numpy(ac_p), torch.from_numpy(ac_o),
                    float(step), float(s), float(zp), len(cold_sorted))
                dct_mem += len(dc)  # uint8 in C++
            print(f"  DCT cold: {dct_mem/1024:.0f} KB")

        # Apply C++ fast_forward
        def apply_cpp(lS_o, lS_i, emb_l, v_W_l):
            li = torch.stack(lS_i) if isinstance(lS_i, (list,tuple)) else (lS_i if lS_i.dim()==2 else lS_i.view(nt,-1))
            lo = torch.stack(lS_o) if isinstance(lS_o, (list,tuple)) else (lS_o if lS_o.dim()==2 else lS_o.view(nt,-1))
            return _C.fast_forward(li, lo)[-1]
        dlrm.apply_emb = apply_cpp

        # Warmup
        with torch.no_grad():
            for X, o, i, T in test_batches[:20]:
                dlrm(X, o, i)

        # Benchmark
        scores, targets, batch_times, emb_times = [], [], [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                li = torch.stack(i) if isinstance(i, (list,tuple)) else i
                lo = torch.stack(o) if isinstance(o, (list,tuple)) else o

                t0 = time.perf_counter()
                _C.fast_forward(li, lo)
                emb_times.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                Z = dlrm(X, o, i)
                batch_times.append(time.perf_counter() - t0)
                scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())

        auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
        emb_ms = np.median(emb_times) * 1000
        batch_ms = np.median(batch_times) * 1000

        small_mem = 2.9  # MB, fixed
        total_mem = hot_mem / 1024 / 1024 + small_mem + dct_mem / 1024 / 1024
        # Add bitmap (6 MB) since we're not using reordered tables in this test
        total_mem += 6.0

        print(f"\n  Results:")
        print(f"    AUC:      {auc:.6f} (loss: {(auc_base-auc)*100:+.4f}%)")
        print(f"    Emb:      {emb_ms:.3f}ms")
        print(f"    Batch:    {batch_ms:.2f}ms")
        print(f"    Hot mem:  {hot_mem/1024/1024:.1f}MB")
        print(f"    DCT mem:  {dct_mem/1024:.0f}KB")
        print(f"    Total:    {total_mem:.1f}MB ({2060.7/total_mem:.0f}x)")

        results.append({
            'hot_fraction': hf,
            'auc': float(auc),
            'auc_loss': float(auc_base - auc),
            'emb_ms': float(emb_ms),
            'batch_ms': float(batch_ms),
            'hot_mb': hot_mem / 1024 / 1024,
            'dct_kb': dct_mem / 1024,
            'total_mb': total_mem,
            'ratio': 2060.7 / total_mem,
        })

        # Reset
        dlrm.apply_emb = dlrm.__class__.apply_emb.__get__(dlrm)
        for t in range(nt):
            dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
            dlrm.emb_l[t].weight.data = sd[ek[t]].clone()
        gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Hot%':>5} {'AUC':>10} {'Loss':>10} {'Emb ms':>8} {'Batch ms':>10} "
          f"{'Hot MB':>8} {'DCT KB':>8} {'Total':>8} {'Ratio':>6}")
    print("-" * 82)
    for r in results:
        print(f"{r['hot_fraction']*100:>4.1f}% {r['auc']:>10.6f} {r['auc_loss']*100:>+9.4f}% "
              f"{r['emb_ms']:>7.3f} {r['batch_ms']:>9.2f}ms "
              f"{r['hot_mb']:>7.1f} {r['dct_kb']:>7.0f} {r['total_mb']:>7.1f}MB {r['ratio']:>5.0f}x")

    os.makedirs('results/dct_domain', exist_ok=True)
    with open('results/dct_domain/hot_fraction_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
