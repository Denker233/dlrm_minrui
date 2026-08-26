#!/usr/bin/env python3
"""Measure baseline AUC of the trained Terabyte checkpoint.

The training log only reported accuracy, which is uninformative at a 3.2% positive
rate (the majority-class baseline is 96.795%).  AUC is what the paper quotes.

Usage: eval_terabyte_auc.py [max_batches]   (0 = full test set)
"""
import sys, time, numpy as np, torch
from sklearn.metrics import roc_auc_score
from bench_terabyte_dc import load_terabyte, MODEL_PATH

max_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 0
t0 = time.time()
dlrm, test_ld, train_ld, ln_emb = load_terabyte()
print(f"[{time.time()-t0:.0f}s] loaders ready", flush=True)

sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
dlrm.load_state_dict(sd)
dlrm.eval()
print(f"[{time.time()-t0:.0f}s] weights loaded", flush=True)

scores, labels, n = [], [], 0
with torch.no_grad():
    for X, o, i, T in test_ld:
        scores.append(dlrm(X, o, i).numpy().flatten())
        labels.append(T.numpy().flatten())
        n += 1
        if n % 200 == 0:
            print(f"[{time.time()-t0:7.0f}s] {n} batches, {sum(len(s) for s in scores):,} samples", flush=True)
        if max_batches and n >= max_batches:
            break

y = np.concatenate(labels); p = np.concatenate(scores)
auc = roc_auc_score(y, p)
print(f"\nsamples      : {len(y):,}")
print(f"positive rate: {y.mean()*100:.3f}%")
print(f"BASELINE AUC : {auc:.6f}")
print(f"README quotes: 0.768820  (previous, undertrained model)")
print(f"elapsed      : {time.time()-t0:.0f}s")
