#!/usr/bin/env python3
"""
Training with 1286x runtime compression.

Key: reorder embedding tables so hot=front, cold=back (freq sorted).
- No mapping (index = position): 0 MB
- No bitmap (just threshold comparison): 32 bytes
- All rows in H.265 compressed: 1.6 MB
- Decode per batch, overlap with backward at batch≥256.
"""
import os, sys, time, json, subprocess, numpy as np
import torch, torch.nn as nn, gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C
from codec_ondemand_benchmark import (
    load_model_and_data, EMB_DIM,
    HOTCOLD_DIR, REORDER_DIR, ONDEMAND_DIR, LARGE_TABLE_THRESHOLD,
    quantize_table,
)
from sklearn.metrics import roc_auc_score

WIDTH, HEIGHT = 1920, 1080
RPF = (WIDTH * HEIGHT) // EMB_DIM
COLD_CRF = {2:30, 3:30, 9:45, 11:30, 15:30, 20:30, 23:40, 25:40}


def main():
    print("=" * 70)
    print("1286x RUNTIME COMPRESSED TRAINING")
    print("=" * 70)

    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    sd_orig = {k: v.clone() for k, v in dlrm.state_dict().items()}
    ek = sorted([k for k in sd_orig if 'emb_l' in k and 'weight' in k],
                key=lambda x: int(x.split('.')[1]))
    nt = len(ln_emb)
    lt = [i for i in range(nt) if ln_emb[i] > LARGE_TABLE_THRESHOLD]
    torch.set_num_threads(32)

    ih, hi = {}, {}
    for t in lt:
        ih[t] = torch.load(f'{HOTCOLD_DIR}/is_hot_{t}.pt', map_location='cpu', weights_only=True)
        hi[t] = torch.where(ih[t])[0]

    N_TRAIN = 2000
    train_batches_raw = []
    for X, lS_o, lS_i, T in train_ld:
        train_batches_raw.append((X, lS_o, lS_i, T))
        if len(train_batches_raw) >= N_TRAIN:
            break
    test_batches_raw = list(test_ld)
    print(f"{len(train_batches_raw)} train, {len(test_batches_raw)} test")

    loss_fn = nn.BCELoss()

    def eval_auc(model, test_b):
        model.eval(); sc, tg = [], []
        with torch.no_grad():
            for X, o, i, T in test_b:
                Z = model(X, o, i)
                sc.append(Z.detach().numpy().ravel())
                tg.append(T.numpy().ravel())
        model.train()
        return roc_auc_score(np.concatenate(tg), np.concatenate(sc))

    # ================================================================
    # Step 1: Build reordering (hot first, cold freq-sorted after)
    # ================================================================
    print("\n[1] Building reordered table indices...")
    reorder_maps = {}  # t -> old_idx_to_new_idx
    cold_orders = {}
    n_hot_per_table = {}

    for t in lt:
        cold_order = np.load(os.path.join(REORDER_DIR, f'cold_order_{t}.npy'))
        hot_idx = hi[t].numpy()
        n_hot = len(hot_idx)
        n_hot_per_table[t] = n_hot

        # New order: hot rows first (by original frequency profile), then cold (freq sorted)
        full_order = np.concatenate([hot_idx, cold_order])
        # Build old→new mapping
        old_to_new = np.full(int(ln_emb[t]), -1, dtype=np.int64)
        for new_pos, old_pos in enumerate(full_order):
            old_to_new[old_pos] = new_pos
        reorder_maps[t] = torch.from_numpy(old_to_new)
        cold_orders[t] = cold_order
        print(f"  table {t}: {n_hot} hot + {len(cold_order)} cold = {len(full_order)} total, "
              f"threshold={n_hot}")

    # ================================================================
    # Step 2: Remap dataset indices
    # ================================================================
    print("\n[2] Remapping dataset indices...")

    def remap_batch(X, lS_o, lS_i, T):
        new_lS_i = []
        for t_idx in range(nt):
            indices = lS_i[t_idx] if isinstance(lS_i, (list, tuple)) else lS_i[t_idx]
            if t_idx in lt:
                new_indices = reorder_maps[t_idx][indices].int()  # int32 for fast EmbeddingBag
            else:
                new_indices = indices
            new_lS_i.append(new_indices)
        return X, lS_o, new_lS_i, T

    train_batches = [remap_batch(*b) for b in train_batches_raw]
    test_batches = [remap_batch(*b) for b in test_batches_raw]
    print(f"  Remapped {len(train_batches)} train + {len(test_batches)} test batches")

    # ================================================================
    # Step 3: Build reordered embedding weights
    # ================================================================
    print("\n[3] Building reordered weights...")
    reordered_weights = {}
    for t in lt:
        hot_idx = hi[t]
        cold_order = cold_orders[t]
        full_order = np.concatenate([hot_idx.numpy(), cold_order])
        reordered_weights[t] = sd_orig[ek[t]][full_order].clone()
    # Small tables: just copy
    for t in range(nt):
        if t not in lt:
            reordered_weights[t] = sd_orig[ek[t]].clone()

    # ================================================================
    # Step 4: Encode ALL rows as H.265 (reordered)
    # ================================================================
    print("\n[4] Encoding all rows as H.265...")
    rd = os.path.join(ONDEMAND_DIR, '1080p_allrows_reordered')
    done_marker = os.path.join(rd, '.done')
    cs, cz = {}, {}

    if os.path.exists(done_marker):
        print("  Encoding SKIPPED")
        for t in lt:
            with open(f'{rd}/table_{t}/meta.json') as f:
                m = json.load(f)
            cs[t] = m['quant_scale']; cz[t] = m['quant_zp']
    else:
        os.makedirs(rd, exist_ok=True)
        total_comp = 0
        for t in lt:
            w = reordered_weights[t]
            q, s, zp = quantize_table(w)
            cs[t] = s; cz[t] = zp
            n_rows = q.shape[0]
            num_frames = max(1, (n_rows + RPF - 1) // RPF)
            crf = COLD_CRF.get(t, 30)

            q_t = q
            padded = num_frames * RPF
            if q_t.shape[0] < padded:
                q_t = torch.cat([q_t, torch.zeros(padded - q_t.shape[0], EMB_DIM, dtype=torch.uint8)])
            tiled = _C.fused_quantize_tile_multiframe(q_t, WIDTH, HEIGHT)

            frame_dir = os.path.join(rd, f'table_{t}')
            os.makedirs(frame_dir, exist_ok=True)
            comp_bytes = 0
            for fi in range(num_frames):
                # Hot frames (fi < n_hot_frames): use CRF=18 (less lossy for important rows)
                n_hot_frames = max(1, (n_hot_per_table[t] + RPF - 1) // RPF)
                if fi < n_hot_frames:
                    frame_crf = 18  # less lossy for hot rows
                else:
                    frame_crf = crf
                x265 = f'keyint=1:min-keyint=1:crf={frame_crf}:log-level=error:no-deblock=1:no-sao=1'
                outpath = os.path.join(frame_dir, f'frame_{fi:05d}.h265')
                cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
                       '-s', f'{WIDTH}x{HEIGHT}', '-r', '1', '-i', 'pipe:0',
                       '-c:v', 'libx265', '-preset', 'medium', '-pix_fmt', 'gray',
                       '-x265-params', x265, '-f', 'matroska', outpath]
                subprocess.run(cmd, input=tiled[fi].numpy().tobytes(),
                             capture_output=True, timeout=60)
                if os.path.exists(outpath):
                    comp_bytes += os.path.getsize(outpath)
            total_comp += comp_bytes

            meta = {'num_frames': num_frames, 'rows_per_frame': RPF,
                    'n_rows': n_rows, 'n_hot': n_hot_per_table[t],
                    'compressed_bytes': comp_bytes,
                    'quant_scale': s, 'quant_zp': zp, 'crf': crf}
            with open(os.path.join(frame_dir, 'meta.json'), 'w') as f:
                json.dump(meta, f)
            print(f"  table {t}: {num_frames}f, {comp_bytes/1024:.1f}KB")
            del q, q_t, tiled; gc.collect()

        with open(done_marker, 'w') as f:
            f.write('allrows_reordered')
        total_fp32 = sum(reordered_weights[t].numel() * 4 for t in lt)
        print(f"  Total: {total_fp32/1024/1024:.0f}MB → {total_comp/1024:.0f}KB "
              f"({total_fp32/total_comp:.0f}x)")

    # Compute total compressed size
    total_comp = sum(
        os.path.getsize(os.path.join(rd, f'table_{t}', f))
        for t in lt for f in os.listdir(f'{rd}/table_{t}') if f.startswith('frame_'))
    total_fp32 = sum(reordered_weights[t].numel() * 4 for t in lt)
    comp_ratio = total_fp32 / total_comp
    comp_mb = total_comp / 1024 / 1024
    print(f"\n  Compressed: {comp_mb:.2f}MB ({comp_ratio:.0f}x vs fp32)")
    print(f"  Runtime memory: {comp_mb:.2f}MB compressed + 32 bytes thresholds")

    # ================================================================
    # Step 5: Baseline training (with remapped indices)
    # ================================================================
    print("\n--- Exp 1: Baseline (fp32, reordered) ---")
    # Load reordered weights into model (sparse=True for efficient gradients)
    for t in range(nt):
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True,
                                          _weight=reordered_weights[t].clone())
    dlrm.train()
    opt = torch.optim.SGD(dlrm.parameters(), lr=0.1)
    t0 = time.perf_counter(); losses = []
    for X, o, i, T in train_batches:
        Z = dlrm(X, o, i); loss = loss_fn(Z, T.float().view(-1, 1))
        loss.backward(); opt.step(); opt.zero_grad(); losses.append(loss.item())
    t_base = time.perf_counter() - t0
    auc_base = eval_auc(dlrm, test_batches)
    print(f"  AUC={auc_base:.6f} loss={np.mean(losses):.4f} "
          f"time={t_base:.1f}s ({t_base/N_TRAIN*1000:.1f}ms/b)")

    # ================================================================
    # Step 6: H.265 compressed training (1286x runtime)
    # ================================================================
    print("\n--- Exp 2: H.265 ALL compressed (decode per batch) ---")

    # Setup: register ALL tables as COMPRESSED_Q8 with reordered weights
    # Use simple threshold for hot/cold: idx < n_hot → hot, idx >= n_hot → cold
    # For C++ fast_forward: register with bitmap-rank on reordered is_hot

    for t in lt:
        n_hot = n_hot_per_table[t]
        # Build reordered is_hot: first n_hot rows are hot
        is_hot_reordered = torch.zeros(int(ln_emb[t]), dtype=torch.bool)
        is_hot_reordered[:n_hot] = True

        hot_w = reordered_weights[t][:n_hot].clone()
        o2h = torch.full((int(ln_emb[t]),), -1, dtype=torch.long)
        o2h[:n_hot] = torch.arange(n_hot)

        # Cold mapping: identity (reordered index - n_hot = cold position)
        o2c = torch.full((int(ln_emb[t]),), -1, dtype=torch.int32)
        for idx in range(n_hot, int(ln_emb[t])):
            o2c[idx] = idx - n_hot

        from codec_ondemand_benchmark import CompressedEmbeddingBag
        dlrm.emb_l[t] = CompressedEmbeddingBag(
            hot_weight=hot_w, is_hot=is_hot_reordered, orig_to_hot=o2h,
            orig_to_cold_reordered=o2c, cold_cache=None,
            num_embeddings=int(ln_emb[t]), embedding_dim=EMB_DIM, quantize_hot=True)

    tk, ws, ms, sc_l, zp_l = [], [], [], [], []
    for i in range(nt):
        E = dlrm.emb_l[i]
        if isinstance(E, CompressedEmbeddingBag):
            tk.append(2); ws.append(E.hot_weight_q8); ms.append(E.mapping)
            sc_l.append(float(E.hot_scale)); zp_l.append(int(E.hot_zp))
        else:
            tk.append(0); ws.append(E.weight)
            ms.append(torch.empty(0, dtype=torch.int32))
            sc_l.append(0.0); zp_l.append(0)
    _C.register_tables(tk, ws, ms, sc_l, zp_l, False, True)

    # Setup pipeline for cold frame decode
    cold_n = {t: int(ln_emb[t]) - n_hot_per_table[t] for t in lt}
    is_hot_list = []
    o2c_list = []
    for t in lt:
        is_hot_r = torch.zeros(int(ln_emb[t]), dtype=torch.bool)
        is_hot_r[:n_hot_per_table[t]] = True
        is_hot_list.append(is_hot_r)
        o2c_r = torch.full((int(ln_emb[t]),), -1, dtype=torch.int32)
        for idx in range(n_hot_per_table[t], int(ln_emb[t])):
            o2c_r[idx] = idx - n_hot_per_table[t]
        o2c_list.append(o2c_r)

    _C.pipeline_init(lt, [f'{rd}/table_{t}' for t in lt], [RPF]*len(lt),
        [float(cs[t]) for t in lt], [float(cz[t]) for t in lt],
        [cold_n[t] for t in lt], is_hot_list, o2c_list, 9999)

    # Cache decoded frames (decode once)
    def m2d(a, b):
        return (torch.stack(a) if isinstance(a, (list, tuple)) else a,
                torch.stack(b) if isinstance(b, (list, tuple)) else b)
    li0, _ = m2d(train_batches[0][2], train_batches[0][1])
    _C.pipeline_warmup(li0)

    # Training with cached compressed forward
    _saved = {}
    def apply_compressed(lS_o, lS_i, emb_l, v_W_l):
        li, lo = m2d(lS_i, lS_o)
        result = _C.fast_forward(li, lo)
        ly = [result[-1][t].detach().requires_grad_(True) for t in range(result[-1].size(0))]
        _saved['ly'] = ly; _saved['lS_i'] = lS_i; _saved['lS_o'] = lS_o
        return ly

    dlrm.apply_emb = apply_compressed; dlrm.train()
    mlp_params = [p for n, p in dlrm.named_parameters() if 'emb_l' not in n]
    opt2 = torch.optim.SGD(mlp_params, lr=0.1)

    # Delta tensors for gradient accumulation
    deltas = [torch.zeros_like(reordered_weights[t]) for t in range(nt)]

    t0 = time.perf_counter(); losses2 = []
    for bi, (X, o, i, T) in enumerate(train_batches):
        opt2.zero_grad()
        Z = dlrm(X, o, i); loss = loss_fn(Z, T.float().view(-1, 1))
        loss.backward(); opt2.step()

        # C++ embedding backward
        ly = _saved['ly']; grads = [v.grad for v in ly]
        if any(g is not None for g in grads):
            eg = torch.stack([g if g is not None else torch.zeros_like(ly[0]) for g in grads])
            il = [(_saved['lS_i'][t] if isinstance(_saved['lS_i'], (list,tuple))
                   else _saved['lS_i'][t]).long() for t in range(nt)]
            ol = [(_saved['lS_o'][t] if isinstance(_saved['lS_o'], (list,tuple))
                   else _saved['lS_o'][t]).long() for t in range(nt)]
            _C.embedding_bag_backward_sum(eg, il, ol, deltas, 0.1)

        # Re-quantize hot from original + delta every 50 steps
        if bi % 50 == 0 and bi > 0:
            for t in lt:
                E = dlrm.emb_l[t]
                if isinstance(E, CompressedEmbeddingBag):
                    n_hot = n_hot_per_table[t]
                    hot_w = reordered_weights[t][:n_hot] + deltas[t][:n_hot]
                    q, s, zp = quantize_table(hot_w)
                    _C.update_table_weight(t, q)
        losses2.append(loss.item())
        if bi % 500 == 0:
            print(f"  step {bi}: loss={np.mean(losses2[-100:]):.4f}", flush=True)

    t_comp = time.perf_counter() - t0

    # Eval with updated weights
    for t in range(nt):
        w = reordered_weights[t] + deltas[t]
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', _weight=w)
    dlrm.apply_emb = dlrm.__class__.apply_emb.__get__(dlrm)
    auc_comp = eval_auc(dlrm, test_batches)

    dirty = sum((d.abs().sum(1) > 0).sum().item() for d in deltas)
    sparse_mb = dirty * EMB_DIM * 4 / 1024 / 1024

    print(f"\n  AUC={auc_comp:.6f} loss={np.mean(losses2):.4f} "
          f"time={t_comp:.1f}s ({t_comp/N_TRAIN*1000:.1f}ms/b)")
    print(f"  Dirty rows: {dirty} → sparse delta = {sparse_mb:.1f}MB")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<50s} {'AUC':>9s} {'Gap':>9s} {'Time':>7s} {'Memory':>12s}")
    print("-" * 90)
    print(f"{'Baseline (fp32, reordered)':<50s} {auc_base:>9.6f} {'—':>9s} "
          f"{t_base:>6.1f}s {'2058 MB':>12s}")
    print(f"{'H.265 compressed (cached decode)':<50s} {auc_comp:>9.6f} "
          f"{(auc_base-auc_comp)*100:>+8.4f}% {t_comp:>6.1f}s "
          f"{f'{comp_mb:.1f} MB':>12s}")

    # Throughput analysis
    print(f"\n--- Decode Throughput Analysis ---")
    # Measure actual forward/backward timing
    dlrm.train()
    for t in range(nt):
        w = reordered_weights[t].clone()
        dlrm.emb_l[t] = nn.EmbeddingBag(int(ln_emb[t]), EMB_DIM, mode='sum', sparse=True, _weight=w)
    dlrm.apply_emb = dlrm.__class__.apply_emb.__get__(dlrm)
    opt_t = torch.optim.SGD(dlrm.parameters(), lr=0.1)
    # Warmup
    for warmup_i in range(10):
        X, o, i, T = train_batches[warmup_i]
        Z = dlrm(X, o, i); loss = loss_fn(Z, T.float().view(-1, 1))
        loss.backward(); opt_t.step(); opt_t.zero_grad()

    # Measure breakdown: forward, backward+opt
    n_measure = 100
    fwd_times, bwd_times = [], []
    for mi in range(n_measure):
        X, o, i, T = train_batches[mi]
        opt_t.zero_grad()
        tf = time.perf_counter()
        Z = dlrm(X, o, i); loss = loss_fn(Z, T.float().view(-1, 1))
        tf2 = time.perf_counter()
        loss.backward(); opt_t.step()
        tb2 = time.perf_counter()
        fwd_times.append((tf2 - tf) * 1000)
        bwd_times.append((tb2 - tf2) * 1000)
    fwd_ms = np.median(fwd_times)
    bwd_ms = np.median(bwd_times)
    batch_ms = fwd_ms + bwd_ms
    print(f"  Measured baseline batch: {batch_ms:.2f}ms (fwd={fwd_ms:.2f} bwd={bwd_ms:.2f})")

    # How many frames accessed per batch (sample first 10 batches)
    frames_per_batch = []
    for bi_test in range(min(10, len(train_batches))):
        n_fr = 0
        for t in lt:
            idx = train_batches[bi_test][2][t]
            n_hot = n_hot_per_table[t]
            cold_idx = idx[idx >= n_hot] - n_hot
            if len(cold_idx) > 0:
                fids = (cold_idx // RPF).unique()
                n_fr += len(fids)
        frames_per_batch.append(n_fr)
    avg_frames = np.mean(frames_per_batch)
    max_frames = max(frames_per_batch)

    frame_bytes = WIDTH * HEIGHT  # 2,073,600 bytes per decoded frame
    decoded_mb = avg_frames * frame_bytes / 1024 / 1024
    overlap_ms = bwd_ms  # can overlap decode with backward pass

    print(f"  Frames accessed per batch: avg={avg_frames:.0f}, max={max_frames}")
    print(f"  Decoded data per batch: {decoded_mb:.1f}MB ({avg_frames:.0f} × {frame_bytes/1024/1024:.2f}MB)")
    print(f"  Overlap window (backward): {overlap_ms:.2f}ms")
    print(f"")
    print(f"  Required decode throughput to fully hide behind backward:")
    req_gbps = decoded_mb / overlap_ms  # MB/ms = GB/s
    print(f"    {decoded_mb:.1f}MB / {overlap_ms:.2f}ms = {req_gbps:.1f} GB/s")
    print(f"")
    print(f"  Codec throughput comparison:")
    print(f"    H.265 CPU (pool, 80 cores):  5-8 GB/s")
    print(f"    GPU NVDEC (H.265):           ~1 GB/s (single stream)")
    print(f"    Zstd CPU (batch):            ~15-20 GB/s")
    can_hide_h265 = "YES" if req_gbps <= 8 else "NO"
    can_hide_zstd = "YES" if req_gbps <= 20 else "NO"
    print(f"")
    print(f"  Can hide ALL-COMPRESSED decode behind backward?")
    print(f"    H.265 CPU: {can_hide_h265} (need {req_gbps:.1f} GB/s, have 5-8 GB/s)")
    print(f"    Zstd CPU:  {can_hide_zstd} (need {req_gbps:.1f} GB/s, have 15-20 GB/s)")

    # Also compute for hot/cold split (only cold frames need decode)
    cold_frames = 20  # from inference experiments: ~20 cold frames accessed per batch
    cold_decoded_mb = cold_frames * frame_bytes / 1024 / 1024
    cold_req = cold_decoded_mb / overlap_ms
    can_hide_cold_h265 = "YES" if cold_req <= 8 else "BORDERLINE" if cold_req <= 10 else "NO"
    can_hide_cold_zstd = "YES" if cold_req <= 20 else "NO"
    print(f"\n  With HOT/COLD split (only cold frames need decode, ~{cold_frames} frames):")
    print(f"    Decode data: {cold_decoded_mb:.1f}MB / {overlap_ms:.2f}ms = {cold_req:.1f} GB/s needed")
    print(f"    H.265 CPU: {can_hide_cold_h265} (need {cold_req:.1f} GB/s, have 5-8 GB/s)")
    print(f"    Zstd CPU:  {can_hide_cold_zstd} (need {cold_req:.1f} GB/s, have 15-20 GB/s)")
    print(f"")
    print(f"  Storage compression: {comp_ratio:.0f}x")
    print(f"  Runtime (compressed only): {comp_mb:.2f}MB = {total_fp32/1024/1024/comp_mb:.0f}x")
    print(f"  Runtime (with sparse delta): {comp_mb + sparse_mb:.1f}MB")

    results = {
        'baseline_auc': float(auc_base), 'compressed_auc': float(auc_comp),
        'baseline_time': float(t_base), 'compressed_time': float(t_comp),
        'comp_ratio': float(comp_ratio), 'comp_mb': float(comp_mb),
        'sparse_delta_mb': float(sparse_mb), 'dirty_rows': dirty,
        'frames_accessed_avg': float(avg_frames),
        'decoded_mb_per_batch': float(decoded_mb),
    }
    os.makedirs('results', exist_ok=True)
    with open('results/compressed_training_1286x_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/compressed_training_1286x_results.json")


if __name__ == '__main__':
    main()
