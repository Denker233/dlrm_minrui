#!/usr/bin/env python3
"""Quick benchmark: test thread count impact on DLRM baseline forward pass.
Loads model and data using codec_ondemand_benchmark infrastructure."""

import sys, os, time, gc
sys.path.insert(0, '/home/cc/expr/dlrm_minrui')

import numpy as np
import torch

# Import model loading from benchmark
os.chdir('/home/cc/expr/dlrm_minrui')
from codec_ondemand_benchmark import load_model_and_data, MODEL_PATH

print("Loading model and data...")
dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
dlrm.eval()

# Pre-cache test batches
print("Pre-caching test batches...")
test_batches = []
for X, lS_o, lS_i, T in test_ld:
    test_batches.append((X, lS_o, lS_i, T))
print(f"Loaded {len(test_batches)} batches, batch_size={test_batches[0][0].shape[0]}")

NUM_WARMUP = 100
NUM_TEST = 500

thread_counts = [8, 16, 24, 32, 40, 56, 80]

print(f"\nTesting BASELINE forward pass ({NUM_TEST} batches after {NUM_WARMUP} warmup)")
print(f"{'='*80}")

for n_threads in thread_counts:
    torch.set_num_threads(n_threads)
    gc.collect()

    # Reset timing counters
    dlrm.time_look_up = 0
    dlrm.time_interact = 0
    dlrm.time_mlp = 0

    # Warmup
    with torch.no_grad():
        for i in range(NUM_WARMUP):
            X, lS_o, lS_i, T = test_batches[i % len(test_batches)]
            dlrm.sequential_forward(X, lS_o, lS_i)

    dlrm.time_look_up = 0
    dlrm.time_interact = 0
    dlrm.time_mlp = 0

    # Test
    lats = []
    with torch.no_grad():
        for i in range(NUM_TEST):
            X, lS_o, lS_i, T = test_batches[i % len(test_batches)]
            t0 = time.time()
            dlrm.sequential_forward(X, lS_o, lS_i)
            lats.append(time.time() - t0)

    avg_lat = np.mean(lats) * 1000
    p50_lat = np.percentile(lats, 50) * 1000
    p99_lat = np.percentile(lats, 99) * 1000
    emb_ms = dlrm.time_look_up / NUM_TEST * 1000
    interact_ms = dlrm.time_interact / NUM_TEST * 1000
    mlp_ms = dlrm.time_mlp / NUM_TEST * 1000

    print(f"Threads={n_threads:3d}: avg={avg_lat:.2f}ms p50={p50_lat:.2f}ms p99={p99_lat:.2f}ms "
          f"| emb={emb_ms:.2f} interact={interact_ms:.2f} mlp={mlp_ms:.2f}")

# Restore
torch.set_num_threads(80)
print(f"\n{'='*80}")
print("Done. Restored torch.set_num_threads(80)")
