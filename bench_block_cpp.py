#!/usr/bin/env python3
"""
C++ benchmark: 4×4 vs 8×8 vs 16×16 block sizes, all through fast_forward.
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
TILE = 4

def encode_dct(uint8_rows, step, block_size):
    """Encode uint8 rows into DCT DC coefficients for given block size."""
    n = uint8_rows.shape[0]
    if block_size == 4:
        rpb = 1
    elif block_size == 8:
        rpb = 4
    else:
        rpb = 16

    n_pad = ((n + rpb - 1) // rpb) * rpb
    if n_pad > n:
        p = np.zeros((n_pad, EMB_DIM), dtype=np.uint8); p[:n] = uint8_rows; uint8_rows = p

    n_blocks = n_pad // rpb
    rows = uint8_rows.reshape(n_blocks, rpb, EMB_DIM).astype(np.float32)
    blocks = np.zeros((n_blocks, block_size, block_size), dtype=np.float32)

    for r in range(rpb):
        tile = rows[:, r, :].reshape(n_blocks, TILE, TILE)
        if block_size == 4:
            blocks[:, :, :] = tile
        elif block_size == 8:
            rs, cs = (r // 2) * TILE, (r % 2) * TILE
            blocks[:, rs:rs+TILE, cs:cs+TILE] = tile
        else:  # 16
            rs, cs = (r // 4) * TILE, (r % 4) * TILE
            blocks[:, rs:rs+TILE, cs:cs+TILE] = tile

    dct = dctn(blocks, axes=(-2, -1), type=2, norm='ortho')
    dc = np.round(dct[:, 0, 0] / step).astype(np.int16)

    # Empty AC (DC-only)
    ac_d = np.zeros(0, dtype=np.int16)
    ac_p = np.zeros(0, dtype=np.uint8)
    ac_o = np.zeros(n_blocks + 1, dtype=np.int64)

    return dc, ac_d, ac_p, ac_o, n_blocks, rpb


def setup_and_register(dlrm, sd, ek, ln_emb, nt, is_hot, hi, step, block_size):
    """Setup compressed tables + register DCT cold with given block size."""
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
            ms.append(torch.empty(0, dtype=torch.int32)); sc_l.append(0.0); zp_l.append(0)
    _C.register_tables(tk, ws, ms, sc_l, zp_l, False, True)

    dct_mem = 0
    for t in TABLES:
        cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
        q, s, zp = quantize_table(sd[ek[t]][cold_order])
        dc, ac_d, ac_p, ac_o, nb, rpb = encode_dct(q.numpy(), step, block_size)
        _C.register_cold_dct(t, torch.from_numpy(dc), torch.from_numpy(ac_d),
            torch.from_numpy(ac_p), torch.from_numpy(ac_o),
            float(step), float(s), float(zp), len(cold_order), block_size)
        dct_mem += nb  # uint8 in C++

    return dct_mem


def main():
    print("=" * 70)
    print("C++ BLOCK SIZE COMPARISON: 4×4 vs 8×8 vs 16×16")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    nt = len(ln_emb)
    torch.set_num_threads(32)

    is_hot = {t: torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True) for t in TABLES}
    hi = {t: torch.where(is_hot[t])[0] for t in TABLES}

    def apply_cpp(lS_o, lS_i, emb_l, v_W_l):
        li = torch.stack(lS_i) if isinstance(lS_i, (list,tuple)) else (lS_i if lS_i.dim()==2 else lS_i.view(nt,-1))
        lo = torch.stack(lS_o) if isinstance(lS_o, (list,tuple)) else (lS_o if lS_o.dim()==2 else lS_o.view(nt,-1))
        return _C.fast_forward(li, lo)[-1]

    # Baseline
    print("\nBaseline...")
    scores, targets = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_base = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"  fp32: {auc_base:.6f}")

    # Also measure H.265+cache for comparison
    print("\nH.265+cache...")
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
            tk.append(1); ws.append(E.hot_weight); ms.append(E.mapping); sc_l.append(0.0); zp_l.append(0)
        else:
            tk.append(0); ws.append(E.weight); ms.append(torch.empty(0, dtype=torch.int32)); sc_l.append(0.0); zp_l.append(0)
    _C.register_tables(tk, ws, ms, sc_l, zp_l, False, True)

    dir_1080p = os.path.join(ONDEMAND_DIR, '1080p_crf30_nofilter')
    for t in TABLES:
        frame_dir = os.path.join(dir_1080p, f'table_{t}')
        paths = [os.path.join(frame_dir, f) for f in sorted(os.listdir(frame_dir)) if f.endswith('.h265')]
        cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
        _, s, zp = quantize_table(sd[ek[t]][cold_order])
        decoded = _C.batch_decode_fast(paths, 2, len(paths), True, False, False)
        rpf = 1920 * 1080 // EMB_DIM
        rows = torch.cat([_C.untile_frame_to_rows(f, rpf) for f in decoded])[:len(cold_order)]
        _C.register_cold_flat(t, rows, float(s), float(zp), len(cold_order))

    dlrm.apply_emb = apply_cpp
    with torch.no_grad():
        for X, o, i, T in test_batches[:20]: dlrm(X, o, i)
    scores, targets, emb_times, batch_times = [], [], [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            li = torch.stack(i) if isinstance(i, (list,tuple)) else i
            lo = torch.stack(o) if isinstance(o, (list,tuple)) else o
            t0 = time.perf_counter(); _C.fast_forward(li, lo); emb_times.append(time.perf_counter()-t0)
            t0 = time.perf_counter(); Z = dlrm(X, o, i); batch_times.append(time.perf_counter()-t0)
            scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_h265 = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"  H.265+cache: AUC={auc_h265:.6f}  emb={np.median(emb_times)*1000:.3f}ms  batch={np.median(batch_times)*1000:.2f}ms")

    # Reset
    dlrm.apply_emb = dlrm.__class__.apply_emb.__get__(dlrm)
    for t in range(nt):
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        dlrm.emb_l[t].weight.data = sd[ek[t]].clone()
    gc.collect()

    # Test each block size
    step = 32
    results = {}
    for bs_label, block_size in [("4×4", 4), ("8×8", 8), ("16×16", 16)]:
        print(f"\n--- DCT {bs_label} (C++, step={step}) ---")
        dct_bytes = setup_and_register(dlrm, sd, ek, ln_emb, nt, is_hot, hi, step, block_size)
        dct_kb = dct_bytes / 1024

        dlrm.apply_emb = apply_cpp
        with torch.no_grad():
            for X, o, i, T in test_batches[:20]: dlrm(X, o, i)

        scores, targets, emb_times, batch_times = [], [], [], []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                li = torch.stack(i) if isinstance(i, (list,tuple)) else i
                lo = torch.stack(o) if isinstance(o, (list,tuple)) else o
                t0 = time.perf_counter(); _C.fast_forward(li, lo); emb_times.append(time.perf_counter()-t0)
                t0 = time.perf_counter(); Z = dlrm(X, o, i); batch_times.append(time.perf_counter()-t0)
                scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())

        auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
        emb_ms = np.median(emb_times) * 1000
        batch_ms = np.median(batch_times) * 1000
        mem_mb = 22.1 + 2.9 + 6.0 + dct_kb / 1024

        print(f"  AUC={auc:.6f} loss={auc_base-auc:.6f} ({(auc_base-auc)*100:+.4f}%)")
        print(f"  Emb={emb_ms:.3f}ms  Batch={batch_ms:.2f}ms")
        print(f"  Cold DC={dct_kb:.0f}KB  Total={mem_mb:.1f}MB ({2060.7/mem_mb:.0f}x)")

        results[bs_label] = {
            'auc': float(auc), 'auc_loss': float(auc_base-auc),
            'emb_ms': float(emb_ms), 'batch_ms': float(batch_ms),
            'dct_kb': float(dct_kb), 'mem_mb': float(mem_mb),
            'ratio': float(2060.7 / mem_mb),
        }

        # Reset
        dlrm.apply_emb = dlrm.__class__.apply_emb.__get__(dlrm)
        for t in range(nt):
            dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
            dlrm.emb_l[t].weight.data = sd[ek[t]].clone()
        gc.collect()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (all C++, same machine, back-to-back)")
    print(f"{'='*80}")
    print(f"\n{'Method':<25} {'AUC':>10} {'Loss%':>8} {'Emb ms':>8} {'Batch ms':>9} {'Cold KB':>8} {'Mem':>8} {'Ratio':>6}")
    print("-" * 90)
    print(f"{'fp32 baseline':<25} {auc_base:>10.6f} {'—':>8} {'—':>8} {'—':>9} {'—':>8} {'2061MB':>8} {'1x':>6}")
    print(f"{'H.265+cache':<25} {auc_h265:>10.6f} {(auc_base-auc_h265)*100:>+7.3f}% "
          f"{np.median(emb_times)*1000:>7.3f} {np.median(batch_times)*1000:>8.2f} {'—':>8} {'~71MB':>8} {'29x':>6}")
    for label in ["4×4", "8×8", "16×16"]:
        r = results[label]
        print(f"{'DCT '+label:<25} {r['auc']:>10.6f} {r['auc_loss']*100:>+7.3f}% "
              f"{r['emb_ms']:>7.3f} {r['batch_ms']:>8.2f} {r['dct_kb']:>7.0f} {r['mem_mb']:>7.1f}MB {r['ratio']:>5.0f}x")

    with open('results/dct_domain/block_size_cpp.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
