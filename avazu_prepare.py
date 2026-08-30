#!/usr/bin/env python3
"""Avazu_x1 (FuxiCTR split) -> npz for DLRM training + DC sweep.

Input: 22 categorical fields, globally integer-encoded. We re-map each field to a
0-based per-field index space and record cardinalities. No dense features.
Output: /mnt/nvme1/avazu/avazu_x1.npz with X_cat_{train,valid,test} (int32),
y_{train,valid,test} (int8), counts (int64).
"""
import numpy as np, pandas as pd, time
D = '/mnt/nvme1/avazu'
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

parts = {}
for split in ('train','valid','test'):
    log(f"reading {split}.csv ...")
    df = pd.read_csv(f'{D}/{split}.csv', dtype=np.int32)
    parts[split] = df
    log(f"  {split}: {len(df):,} rows, ctr={df['label'].mean():.4f}")

feats = [f'feat_{i}' for i in range(1,23)]
log("building per-field vocabularies over ALL splits ...")
counts = []
for f in feats:
    allv = pd.concat([parts[s][f] for s in ('train','valid','test')])
    uniq = np.sort(allv.unique())
    counts.append(len(uniq))
    # remap to 0..card-1 (train-order-independent: sorted id order)
    lut = pd.Series(np.arange(len(uniq), dtype=np.int32), index=uniq)
    for s in ('train','valid','test'):
        parts[s][f] = lut[parts[s][f].values].values
log(f"cardinalities: {counts}")
log(f"total rows across fields: {sum(counts):,}  -> fp32 D=16 table = {sum(counts)*16*4/2**20:.1f} MB")

out = {}
for s in ('train','valid','test'):
    out[f'X_cat_{s}'] = parts[s][feats].values.astype(np.int32)
    out[f'y_{s}'] = parts[s]['label'].values.astype(np.int8)
out['counts'] = np.array(counts, dtype=np.int64)
np.savez(f'{D}/avazu_x1.npz', **out)
log(f"saved {D}/avazu_x1.npz")
