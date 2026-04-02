#!/usr/bin/env python3
"""
Benchmark C++ DCT-domain embedding lookup: register DCT coefficients,
run inference through fast_forward, measure speed and AUC.
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
BLOCK = 8; TILE = 4; RPB = 4

def tile_and_dct(uint8_rows, step_size):
    """Tile rows into 8×8 blocks, DCT, quantize. Return DC values + sparse AC."""
    n = uint8_rows.shape[0]
    n_pad = ((n + RPB - 1) // RPB) * RPB
    if n_pad > n:
        padded = np.zeros((n_pad, EMB_DIM), dtype=np.uint8)
        padded[:n] = uint8_rows
        uint8_rows = padded

    n_blocks = n_pad // RPB
    rows = uint8_rows.reshape(n_blocks, RPB, EMB_DIM).astype(np.float32)

    blocks = np.zeros((n_blocks, BLOCK, BLOCK), dtype=np.float32)
    for r in range(RPB):
        tiles = rows[:, r, :].reshape(n_blocks, TILE, TILE)
        rs, cs = (r // 2) * TILE, (r % 2) * TILE
        blocks[:, rs:rs+TILE, cs:cs+TILE] = tiles

    dct_coeffs = dctn(blocks, axes=(-2, -1), type=2, norm='ortho')
    quantized = np.round(dct_coeffs / step_size).astype(np.int16)

    # Extract DC values
    dc_values = quantized[:, 0, 0].copy()

    # Extract sparse AC (non-DC non-zero)
    ac_data_list = []
    ac_pos_list = []
    ac_offsets = [0]

    for bi in range(n_blocks):
        q = quantized[bi]
        for u in range(BLOCK):
            for v in range(BLOCK):
                if u == 0 and v == 0:
                    continue  # skip DC
                if q[u, v] != 0:
                    ac_pos_list.append(u * BLOCK + v)
                    ac_data_list.append(q[u, v])
        ac_offsets.append(len(ac_data_list))

    ac_data = np.array(ac_data_list, dtype=np.int16) if ac_data_list else np.zeros(0, dtype=np.int16)
    ac_pos = np.array(ac_pos_list, dtype=np.uint8) if ac_pos_list else np.zeros(0, dtype=np.uint8)
    ac_offsets = np.array(ac_offsets, dtype=np.int64)

    return dc_values, ac_data, ac_pos, ac_offsets, n_blocks


def main():
    print("=" * 70)
    print("C++ DCT-DOMAIN EMBEDDING LOOKUP BENCHMARK")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
    ek = sorted([k for k in sd if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    test_batches = list(test_ld)
    nt = len(ln_emb)
    lt = [i for i in range(nt) if ln_emb[i] > LARGE_TABLE_THRESHOLD]
    torch.set_num_threads(32)

    is_hot = {t: torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', weights_only=True) for t in TABLES}
    hi = {t: torch.where(is_hot[t])[0] for t in TABLES}

    # ================================================================
    # Baseline AUC (fp32)
    # ================================================================
    print("\nBaseline AUC...")
    scores, targets = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            Z = dlrm(X, o, i); scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_base = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    print(f"  fp32 baseline: {auc_base:.6f}")

    # ================================================================
    # Test each step size
    # ================================================================
    for step_size in [8, 16, 32]:
        print(f"\n{'='*70}")
        print(f"STEP SIZE = {step_size}")
        print(f"{'='*70}")

        # Setup: register hot tables as COMPRESSED_Q8, cold as DCT-domain
        for t in TABLES:
            hot_w = sd[ek[t]][hi[t]].clone()
            o2h = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
            o2h[hi[t]] = torch.arange(len(hi[t]))
            o2c = torch.load(f'{REORDER_DIR}/orig_to_cold_reordered_{t}.pt', weights_only=True)

            dlrm.emb_l[t] = CompressedEmbeddingBag(
                hot_weight=hot_w, is_hot=is_hot[t], orig_to_hot=o2h,
                orig_to_cold_reordered=o2c, cold_cache=None,
                num_embeddings=int(ln_emb[t]), embedding_dim=EMB_DIM, quantize_hot=False)

        # Register tables in C++
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

        # Encode cold data as DCT and register
        total_dc_bytes = 0
        total_ac_bytes = 0
        print("  Encoding + registering DCT cold data...")
        for t in TABLES:
            cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
            w = sd[ek[t]][cold_order]
            q, s, zp = quantize_table(w)
            q_np = q.numpy()
            n_cold = len(cold_order)

            dc, ac_data, ac_pos, ac_off, n_blocks = tile_and_dct(q_np, step_size)
            total_dc_bytes += dc.nbytes
            total_ac_bytes += ac_data.nbytes + ac_pos.nbytes

            n_ac = len(ac_data)
            dc_only_pct = (np.diff(ac_off) == 0).sum() / n_blocks * 100

            _C.register_cold_dct(
                t,
                torch.from_numpy(dc),
                torch.from_numpy(ac_data),
                torch.from_numpy(ac_pos),
                torch.from_numpy(ac_off),
                float(step_size), float(s), float(zp), n_cold)

            print(f"    table {t}: {n_blocks} blocks, DC-only={dc_only_pct:.0f}%, "
                  f"{n_ac} AC coeffs, DC={dc.nbytes/1024:.1f}KB")

        total_mem_kb = (total_dc_bytes + total_ac_bytes) / 1024
        total_fp32_mb = sum(len(np.load(f'{REORDER_DIR}/cold_order_{t}.npy')) * EMB_DIM * 4
                           for t in TABLES) / 1024 / 1024
        print(f"  Total DCT memory: {total_mem_kb:.0f}KB ({total_fp32_mb*1024/total_mem_kb:.0f}x vs fp32)")

        # Setup fast_forward
        def apply_dct(lS_o, lS_i, emb_l, v_W_l):
            if isinstance(lS_i, (list, tuple)):
                li = torch.stack(lS_i)
            else:
                li = lS_i if lS_i.dim() == 2 else lS_i.view(nt, -1)
            if isinstance(lS_o, (list, tuple)):
                lo = torch.stack(lS_o)
            else:
                lo = lS_o if lS_o.dim() == 2 else lS_o.view(nt, -1)
            return _C.fast_forward(li, lo)[-1]

        dlrm.apply_emb = apply_dct

        # Warmup
        for X, o, i, T in test_batches[:5]:
            with torch.no_grad():
                dlrm(X, o, i)

        # Speed benchmark
        print("  Speed benchmark (1599 batches)...")
        t0 = time.perf_counter()
        scores, targets = [], []
        batch_times = []
        with torch.no_grad():
            for X, o, i, T in test_batches:
                tb = time.perf_counter()
                Z = dlrm(X, o, i)
                batch_times.append(time.perf_counter() - tb)
                scores.append(Z.numpy().ravel())
                targets.append(T.numpy().ravel())
        total_time = time.perf_counter() - t0
        auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))

        med_ms = np.median(batch_times) * 1000
        p99_ms = np.percentile(batch_times, 99) * 1000
        mean_ms = np.mean(batch_times) * 1000

        print(f"\n  Results (step={step_size}):")
        print(f"    AUC: {auc:.6f} (loss: {auc_base-auc:.6f}, {(auc_base-auc)*100:+.4f}%)")
        print(f"    Total: {total_time:.2f}s ({mean_ms:.2f}ms/batch, p50={med_ms:.2f}, p99={p99_ms:.2f})")
        print(f"    Cold memory: {total_mem_kb:.0f}KB ({total_fp32_mb*1024/total_mem_kb:.0f}x vs fp32)")
        print(f"    Cache: NONE (pure DCT-domain lookup)")

        # Reset for next iteration
        dlrm.apply_emb = dlrm.__class__.apply_emb.__get__(dlrm)
        for t in range(nt):
            dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
            dlrm.emb_l[t].weight.data = sd[ek[t]].clone()
        gc.collect()

    # ================================================================
    # Comparison
    # ================================================================
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Method':<35} {'AUC':>10} {'Loss':>10} {'Batch ms':>10} {'Cold mem':>10}")
    print(f"  {'-'*75}")
    print(f"  {'fp32 baseline':<35} {auc_base:>10.6f} {'—':>10} {'—':>10} {'2058 MB':>10}")
    print(f"  Compare DCT-domain results above with H.265 + 40MB cache:")
    print(f"  {'H.265 CRF=30 + cache':<35} {'0.802368':>10} {'-0.013%':>10} {'~3.0ms':>10} {'76 MB':>10}")


if __name__ == '__main__':
    main()
