#!/usr/bin/env python3
"""
Head-to-head: DCT-domain vs H.265+cache vs fp32 baseline.
Same script, same data, same machine, back-to-back.
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


def run_benchmark(dlrm, test_batches, nt, label, n_warmup=20, n_measure=1599):
    """Run inference, measure latency and AUC."""
    # Warmup
    with torch.no_grad():
        for X, o, i, T in test_batches[:n_warmup]:
            dlrm(X, o, i)

    # Measure
    scores, targets = [], []
    emb_times, full_times = [], []
    with torch.no_grad():
        for X, o, i, T in test_batches[:n_measure]:
            li = torch.stack(i) if isinstance(i, (list,tuple)) else (i if i.dim()==2 else i.view(nt,-1))
            lo = torch.stack(o) if isinstance(o, (list,tuple)) else (o if o.dim()==2 else o.view(nt,-1))

            # Embedding only
            t0 = time.perf_counter()
            _C.fast_forward(li, lo)
            emb_times.append(time.perf_counter() - t0)

            # Full batch
            t0 = time.perf_counter()
            Z = dlrm(X, o, i)
            full_times.append(time.perf_counter() - t0)

            scores.append(Z.numpy().ravel())
            targets.append(T.numpy().ravel())

    auc = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    emb_ms = np.median(emb_times) * 1000
    full_ms = np.median(full_times) * 1000
    mlp_ms = full_ms - emb_ms

    print(f"  {label}:")
    print(f"    AUC:       {auc:.6f}")
    print(f"    Embedding: {emb_ms:.3f}ms")
    print(f"    MLP:       {mlp_ms:.3f}ms")
    print(f"    Full batch:{full_ms:.3f}ms (p99={np.percentile(full_times,99)*1000:.2f}ms)")
    return auc, emb_ms, full_ms


def main():
    print("=" * 70)
    print("HEAD-TO-HEAD: DCT-domain vs H.265+cache vs fp32 baseline")
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

    def setup_compressed(mode='dct'):
        """Setup CompressedEmbeddingBag for all large tables."""
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

    def apply_cpp(lS_o, lS_i, emb_l, v_W_l):
        li = torch.stack(lS_i) if isinstance(lS_i, (list,tuple)) else (lS_i if lS_i.dim()==2 else lS_i.view(nt,-1))
        lo = torch.stack(lS_o) if isinstance(lS_o, (list,tuple)) else (lS_o if lS_o.dim()==2 else lS_o.view(nt,-1))
        return _C.fast_forward(li, lo)[-1]

    results = {}

    # ================================================================
    # Config 1: fp32 baseline (Python EmbeddingBag)
    # ================================================================
    print("\n--- Config 1: fp32 baseline (Python) ---")
    for t in range(nt):
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True)
        dlrm.emb_l[t].weight.data = sd[ek[t]].clone()
    dlrm.apply_emb = dlrm.__class__.apply_emb.__get__(dlrm)

    # Can't use fast_forward for fp32 baseline, measure full batch only
    with torch.no_grad():
        for X, o, i, T in test_batches[:20]: dlrm(X, o, i)  # warmup
    scores, targets, full_times = [], [], []
    with torch.no_grad():
        for X, o, i, T in test_batches:
            t0 = time.perf_counter()
            Z = dlrm(X, o, i)
            full_times.append(time.perf_counter() - t0)
            scores.append(Z.numpy().ravel()); targets.append(T.numpy().ravel())
    auc_base = roc_auc_score(np.concatenate(targets), np.concatenate(scores))
    base_ms = np.median(full_times) * 1000
    print(f"  fp32 baseline: AUC={auc_base:.6f}, batch={base_ms:.2f}ms")
    results['fp32'] = {'auc': float(auc_base), 'batch_ms': float(base_ms)}

    # ================================================================
    # Config 2: H.265 CRF=30 + decoded frame cache (C++)
    # ================================================================
    print("\n--- Config 2: H.265 CRF=30 + 40MB cache (C++) ---")
    setup_compressed('h265')

    # Decode and register cold frames from 1080p
    dir_1080p = os.path.join(ONDEMAND_DIR, '1080p_crf30_nofilter')
    for t in TABLES:
        frame_dir = os.path.join(dir_1080p, f'table_{t}')
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.h265')])
        paths = [os.path.join(frame_dir, f) for f in frame_files]
        if not paths: continue

        cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
        n_cold = len(cold_order)
        _, s, zp = quantize_table(sd[ek[t]][cold_order])

        # Decode all frames
        decoded_frames = _C.batch_decode_fast(paths, 2, len(paths), True, False, False)
        # Untile and concatenate
        rpf = 1920 * 1080 // EMB_DIM
        all_rows = []
        for frame in decoded_frames:
            rows = _C.untile_frame_to_rows(frame, rpf)
            all_rows.append(rows)
        cold_data = torch.cat(all_rows, dim=0)[:n_cold]

        _C.register_cold_flat(t, cold_data, float(s), float(zp), n_cold)

    dlrm.apply_emb = apply_cpp
    auc_h265, emb_h265, full_h265 = run_benchmark(dlrm, test_batches, nt, "H.265+cache")
    results['h265_cache'] = {'auc': float(auc_h265), 'emb_ms': float(emb_h265), 'batch_ms': float(full_h265)}

    # ================================================================
    # Config 3: DCT-domain (step=32, zero cache, C++)
    # ================================================================
    print("\n--- Config 3: DCT-domain step=32 (zero cache, C++) ---")
    setup_compressed('dct')

    step = 32
    for t in TABLES:
        cold_order = np.load(f'{REORDER_DIR}/cold_order_{t}.npy')
        q, s, zp = quantize_table(sd[ek[t]][cold_order])
        dc, ac_d, ac_p, ac_o, nb = tile_and_dct(q.numpy(), step)
        _C.register_cold_dct(t, torch.from_numpy(dc), torch.from_numpy(ac_d),
            torch.from_numpy(ac_p), torch.from_numpy(ac_o), float(step), float(s), float(zp), len(cold_order))

    dlrm.apply_emb = apply_cpp
    auc_dct, emb_dct, full_dct = run_benchmark(dlrm, test_batches, nt, "DCT-domain")
    results['dct_domain'] = {'auc': float(auc_dct), 'emb_ms': float(emb_dct), 'batch_ms': float(full_dct)}

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY (same machine, back-to-back)")
    print(f"{'='*70}")
    print(f"\n{'Config':<30} {'AUC':>10} {'Loss':>10} {'Emb ms':>8} {'Batch ms':>10} {'Memory':>10}")
    print("-" * 82)
    print(f"{'fp32 baseline (Python)':<30} {auc_base:>10.6f} {'—':>10} {'—':>8} {base_ms:>9.2f}ms {'2061 MB':>10}")
    print(f"{'H.265+cache (C++)':<30} {auc_h265:>10.6f} {(auc_base-auc_h265)*100:>+9.4f}% "
          f"{emb_h265:>7.3f} {full_h265:>9.2f}ms {'71 MB':>10}")
    print(f"{'DCT-domain (C++)':<30} {auc_dct:>10.6f} {(auc_base-auc_dct)*100:>+9.4f}% "
          f"{emb_dct:>7.3f} {full_dct:>9.2f}ms {'25 MB*':>10}")
    print(f"\n  * DCT memory with reordered tables + sparse DC optimization")

    os.makedirs('results/dct_domain', exist_ok=True)
    with open('results/dct_domain/head_to_head.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
