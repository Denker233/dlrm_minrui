#!/usr/bin/env python3
"""
Analyze what actually happened in the benchmark
"""

print("="*80)
print("ANALYZING TRAINING BENCHMARK RESULTS")
print("="*80)
print()

# Parse the times
original_times = [379, 388, 386]
compressed_times = [385, 377, 382]

import numpy as np
from scipy import stats

orig_times = np.array(original_times)
comp_times = np.array(compressed_times)

orig_mean = np.mean(orig_times)
comp_mean = np.mean(comp_times)
overhead = ((comp_mean - orig_mean) / orig_mean) * 100

print("TIMING RESULTS:")
print(f"  Original:   {orig_mean:.1f}s ± {np.std(orig_times, ddof=1):.1f}s")
print(f"  Compressed: {comp_mean:.1f}s ± {np.std(comp_times, ddof=1):.1f}s")
print(f"  Overhead:   {overhead:+.2f}%")
print()

t_stat, p_value = stats.ttest_rel(comp_times, orig_times)
print(f"Statistical test: p={p_value:.4f}")
print()

print("="*80)
print("CRITICAL ISSUES DETECTED")
print("="*80)
print()

print("❌ ISSUE 1: NO TRAINING OCCURRED!")
print("  Evidence:")
print("    - Loss is 0.483387 for ALL runs (original and compressed)")
print("    - Accuracy is 76.865% for ALL runs")
print("    - These match the loaded checkpoint exactly")
print()
print("  What happened:")
print("    - Model loaded from checkpoint ✓")
print("    - Training loop did NOT execute")
print("    - Only evaluation/testing ran")
print()

print("❌ ISSUE 2: TIMING BREAKDOWN IS ZERO")
print("  Evidence:")
print("    - MLP time: 0")
print("    - Embedding time: 0")  
print("    - Interaction time: 0")
print()
print("  What this means:")
print("    - Detailed timing not being captured")
print("    - Only total wall-clock time measured")
print()

print("="*80)
print("WHAT THE TIMES ACTUALLY MEASURED")
print("="*80)
print()

print("The ~380s includes:")
print("  1. Data loading (~140s)")
print("  2. Model loading (~5s)")
print("  3. Testing/evaluation (~240s)")
print("  4. NO actual training!")
print()

print("="*80)
print("NEXT STEPS")
print("="*80)
print()

print("To fix this, we need to:")
print()
print("1. Check if --num-batches is working correctly")
print("2. Verify training actually runs (loss should change!)")
print("3. Add print statements to confirm training loop executes")
print("4. Check if --nepochs=2 is actually doing 2 epochs")
print()

print("Let me check the log files for training output...")
print()

# Check if there are any training iteration logs
import os
import glob

log_files = glob.glob('training_benchmark_*/original_iter1.log')
if log_files:
    print(f"Checking {log_files[0]}...")
    with open(log_files[0], 'r') as f:
        content = f.read()
    
    # Look for training iteration output
    if 'it' in content and 'loss' in content:
        print("✓ Found training iteration logs")
    else:
        print("✗ No training iteration logs found")
        print("  Training loop likely didn't execute")
    
    # Count how many times "it" appears (training iterations)
    it_count = content.count('it')
    print(f"  Training iterations logged: {it_count}")

print()

