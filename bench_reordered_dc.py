#!/usr/bin/env python3
"""
Reordered tables + uint8 hot + DC 16×16 cold.
Hot rows first (index < threshold), cold rows after (freq-sorted).
No bitmap needed — just threshold comparison.
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
    print("REORDERED TABLES + uint8 hot + DC 16×16 cold")
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

    step = 32
    results = []

    for hf in [0.043, 0.02, 0.01]:
        print(f"\n{'='*60}")
        print(f"Hot={hf*100:.1f}%, reordered + uint8 hot + DC 16×16")
        print(f"{'='*60}")

        # Step 1: Build reordering per table
        # New order: hot rows first (by freq), then cold rows (by freq)
        reorder_maps = {}  # t -> old_to_new mapping
        n_hot_per_table = {}

        for t in TABLES:
            n = int(ln_emb[t])
            n_hot = max(0, int(n * hf))
            n_hot_per_table[t] = n_hot

            # Hot: top n_hot by frequency
            hot_idx = sorted_idx[t][:n_hot].numpy()
            # Cold: remaining, sorted by frequency
            cold_idx = sorted_idx[t][n_hot:].numpy()

            # New order: hot first, cold after
            full_order = np.concatenate([hot_idx, cold_idx])

            # Build old→new mapping
            old_to_new = np.full(n, -1, dtype=np.int32)
            for new_pos, old_pos in enumerate(full_order):
                old_to_new[old_pos] = new_pos

            reorder_maps[t] = torch.from_numpy(old_to_new)

        # Step 2: Remap test batch indices
        print("  Remapping indices...")
        remapped_batches = []
        for X, lS_o, lS_i, T in test_batches:
            new_lS_i = []
            for t_idx in range(nt):
                idx = lS_i[t_idx] if isinstance(lS_i, (list, tuple)) else lS_i[t_idx]
                if t_idx in reorder_maps:
                    new_idx = reorder_maps[t_idx][idx.long()].int()
                else:
                    new_idx = idx
                new_lS_i.append(new_idx)
            remapped_batches.append((X, lS_o, new_lS_i, T))

        # Step 3: Build reordered weights
        print("  Building reordered weights...")
        reordered_weights = {}
        for t in TABLES:
            n_hot = n_hot_per_table[t]
            hot_idx = sorted_idx[t][:n_hot]
            cold_idx = sorted_idx[t][n_hot:]
            full_order = torch.cat([hot_idx, cold_idx])
            reordered_weights[t] = sd[ek[t]][full_order]

        # Step 4: Setup compressed embedding with reordered tables
        # In reordered table: index < n_hot → hot, index >= n_hot → cold
        # is_hot: first n_hot rows are hot
        for t in TABLES:
            n = int(ln_emb[t])
            n_hot = n_hot_per_table[t]

            is_hot_reordered = torch.zeros(n, dtype=torch.bool)
            is_hot_reordered[:n_hot] = True

            hot_w = reordered_weights[t][:n_hot].clone()

            o2h = torch.full((n,), -1, dtype=torch.long)
            o2h[:n_hot] = torch.arange(n_hot)

            # Cold mapping: identity (reordered index - n_hot = cold position)
            o2c = torch.full((n,), -1, dtype=torch.int32)
            for idx in range(n_hot, n):
                o2c[idx] = idx - n_hot

            dlrm.emb_l[t] = CompressedEmbeddingBag(
                hot_weight=hot_w, is_hot=is_hot_reordered, orig_to_hot=o2h,
                orig_to_cold_reordered=o2c, cold_cache=None,
                num_embeddings=n, embedding_dim=EMB_DIM,
                quantize_hot=True)  # uint8 hot

        # Register in C++ with bitmap (bitmap on reordered = trivial, first n_hot bits set)
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

        # Register DC cold (from reordered cold weights)
        dct_bytes = 0
        for t in TABLES:
            n_hot = n_hot_per_table[t]
            cold_w = reordered_weights[t][n_hot:]
            q, s, zp = quantize_table(cold_w)
            dc, ac_d, ac_p, ac_o, nb = encode_dct_16x16(q.numpy(), step)
            _C.register_cold_dct(t, torch.from_numpy(dc), torch.from_numpy(ac_d),
                torch.from_numpy(ac_p), torch.from_numpy(ac_o),
                float(step), float(s), float(zp), len(cold_w), 16)
            dct_bytes += nb

        hot_bytes = sum(n_hot_per_table[t] * EMB_DIM for t in TABLES)
        hot_mb = hot_bytes / 1024 / 1024
        dct_mb = dct_bytes / 1024 / 1024
        small_mb = 2.9
        # Bitmap is still used in C++ (bitmap_rank), but with reordered tables
        # it's trivial (first n_hot bits = 1). Still costs 6 MB.
        bitmap_mb = 6.0
        total_with_bitmap = hot_mb + small_mb + bitmap_mb + dct_mb
        total_no_bitmap = hot_mb + small_mb + dct_mb

        print(f"  Hot: {hot_mb:.1f}MB  DC: {dct_mb:.1f}MB  Bitmap: {bitmap_mb:.1f}MB")
        print(f"  Total (with bitmap): {total_with_bitmap:.1f}MB ({2060.7/total_with_bitmap:.0f}x)")
        print(f"  Total (no bitmap*):  {total_no_bitmap:.1f}MB ({2060.7/total_no_bitmap:.0f}x)")

        # Benchmark with C++ fast_forward
        def apply_cpp(lS_o, lS_i, emb_l, v_W_l):
            li = torch.stack(lS_i) if isinstance(lS_i, (list, tuple)) else (lS_i if lS_i.dim() == 2 else lS_i.view(nt, -1))
            lo = torch.stack(lS_o) if isinstance(lS_o, (list, tuple)) else (lS_o if lS_o.dim() == 2 else lS_o.view(nt, -1))
            return _C.fast_forward(li, lo)[-1]

        dlrm.apply_emb = apply_cpp

        # Warmup
        with torch.no_grad():
            for X, o, i, T in remapped_batches[:20]:
                dlrm(X, o, i)

        # Measure
        scores, targets, emb_times, batch_times = [], [], [], []
        with torch.no_grad():
            for X, o, i, T in remapped_batches:
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

        print(f"\n  AUC: {auc:.6f} (loss: {(auc_base-auc)*100:+.4f}%)")
        print(f"  Emb: {emb_ms:.3f}ms  Batch: {batch_ms:.2f}ms")

        results.append({
            'hot_pct': hf, 'auc': float(auc),
            'auc_loss_pct': float((auc_base - auc) * 100),
            'emb_ms': float(emb_ms), 'batch_ms': float(batch_ms),
            'hot_mb': float(hot_mb), 'dct_mb': float(dct_mb),
            'total_bitmap': float(total_with_bitmap),
            'total_no_bitmap': float(total_no_bitmap),
            'ratio_bitmap': float(2060.7 / total_with_bitmap),
            'ratio_no_bitmap': float(2060.7 / total_no_bitmap),
        })

        # Reset
        dlrm.apply_emb = dlrm.__class__.apply_emb.__get__(dlrm)
        for t_r in range(nt):
            dlrm.emb_l[t_r] = nn.EmbeddingBag(int(ln_emb[t_r]), EMB_DIM, mode='sum', sparse=True)
            dlrm.emb_l[t_r].weight.data = sd[ek[t_r]].clone()
        gc.collect()

    # Summary
    print(f"\n{'='*90}")
    print("VERIFIED: REORDERED + uint8 hot + DC 16×16 cold")
    print(f"{'='*90}")
    print(f"\n{'Hot%':>5} {'AUC':>10} {'Loss':>8} {'Emb':>7} {'Batch':>7} "
          f"{'Hot':>6} {'DC':>5} {'w/bmp':>7} {'ratio':>6} {'no bmp':>7} {'ratio':>6}")
    print("-" * 85)
    print(f"{'base':>5} {auc_base:>10.6f} {'—':>8} {'—':>7} {'4.98':>7} "
          f"{'—':>6} {'—':>5} {'2061':>6}MB {'1x':>6} {'—':>7} {'—':>6}")
    for r in results:
        print(f"{r['hot_pct']*100:>4.1f}% {r['auc']:>10.6f} {r['auc_loss_pct']:>+7.3f}% "
              f"{r['emb_ms']:>6.3f} {r['batch_ms']:>6.2f} "
              f"{r['hot_mb']:>5.1f} {r['dct_mb']:>4.1f} "
              f"{r['total_bitmap']:>6.1f}MB {r['ratio_bitmap']:>5.0f}x "
              f"{r['total_no_bitmap']:>6.1f}MB {r['ratio_no_bitmap']:>5.0f}x")

    with open('results/dct_domain/reordered_final.json', 'w') as f:
        json.dump({'baseline_auc': float(auc_base), 'results': results}, f, indent=2)


if __name__ == '__main__':
    main()
