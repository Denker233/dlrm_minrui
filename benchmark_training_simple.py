#!/usr/bin/env python3
"""
Training benchmark - modified from working inference code
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from datetime import datetime

print("="*80)
print("TRAINING BENCHMARK")
print("="*80)
print()

# Configuration
NUM_RUNS = 3
NUM_EPOCHS = 2
BATCH_SIZE = 2048
MAX_BATCHES = 50

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'training_benchmark_{timestamp}'
os.makedirs(results_dir, exist_ok=True)

print(f"Configuration:")
print(f"  Runs: {NUM_RUNS}")
print(f"  Epochs per run: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Batches per epoch: {MAX_BATCHES}")
print(f"  Results dir: {results_dir}")
print()

def clear_caches():
    """Clear system caches"""
    print("  Clearing caches...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.system('sync')
    os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1')
    time.sleep(2)
    print("  ✓ Caches cleared")

# Load training data
if os.path.exists('./processed_data/train_data.pt'):
    print("Loading training data...")
    X_train = torch.load('./processed_data/train_data.pt')
    y_train = torch.load('./processed_data/train_label.pt')
    print(f"  Samples: {len(X_train)}")
else:
    print("Creating synthetic training data...")
    n_samples = MAX_BATCHES * BATCH_SIZE
    X_train = torch.randn(n_samples, 39)
    y_train = torch.randint(0, 2, (n_samples,))
    os.makedirs('./processed_data', exist_ok=True)
    torch.save(X_train, './processed_data/train_data.pt')
    torch.save(y_train, './processed_data/train_label.pt')
    print(f"  Created {len(X_train)} samples")

print()

# Create DataLoader
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def train_model(model_path, model_name):
    """Train model and measure time"""
    
    print(f"Loading {model_name}...")
    
    # Use the same loading code as your inference benchmark
    dlrm = torch.load(model_path, map_location='cpu')
    
    # Get model from checkpoint
    if hasattr(dlrm, 'state_dict'):
        # It's already a model
        model = dlrm
    else:
        # It's a checkpoint dict
        from dlrm_s_pytorch import DLRM_Net
        # This should match your inference code
        model = dlrm  # Assuming it's already the model object
    
    model.eval()  # Start in eval mode, switch to train mode in loop
    
    # Setup optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    epoch_times = []
    epoch_losses = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}")
        
        model.train()  # Switch to training mode
        
        epoch_start = time.time()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            if batch_idx >= MAX_BATCHES:
                break
            
            # Split features (same as inference)
            X_int = X[:, :13]
            X_cat = X[:, 13:].long()
            
            # Forward pass
            optimizer.zero_grad()
            output = model(X_int, X_cat)
            
            # Loss
            loss = criterion(output, y.float())
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {batch_idx+1}/{MAX_BATCHES}: Loss={loss.item():.4f}", end='\r')
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        
        epoch_times.append(epoch_time)
        epoch_losses.append(avg_loss)
        
        print(f"\n    Time: {epoch_time:.1f}s, Loss: {avg_loss:.4f}")
    
    return np.array(epoch_times), np.array(epoch_losses)

# Benchmark both models
all_original_times = []
all_compressed_times = []

log_file = open(f'{results_dir}/benchmark_log.txt', 'w')

for run in range(1, NUM_RUNS + 1):
    print("="*80)
    print(f"RUN {run}/{NUM_RUNS}")
    print("="*80)
    print()
    
    # Original model
    print("ORIGINAL MODEL")
    print("-" * 40)
    clear_caches()
    
    orig_times, orig_losses = train_model(
        './models/dlrm_kaggle_quick.pt',
        'ORIGINAL'
    )
    all_original_times.extend(orig_times)
    
    print()
    print(f"  Run {run} Original: {orig_times} seconds")
    log_file.write(f"Run {run} Original: {orig_times}\n")
    print()
    
    # Compressed model
    print("COMPRESSED MODEL")
    print("-" * 40)
    clear_caches()
    
    comp_times, comp_losses = train_model(
        './models/dlrm_kaggle_quick_compressed.pt',
        'COMPRESSED'
    )
    all_compressed_times.extend(comp_times)
    
    print()
    print(f"  Run {run} Compressed: {comp_times} seconds")
    log_file.write(f"Run {run} Compressed: {comp_times}\n")
    print()

# Final analysis
print()
print("="*80)
print("FINAL RESULTS")
print("="*80)
print()

orig_times = np.array(all_original_times)
comp_times = np.array(all_compressed_times)

orig_mean = np.mean(orig_times)
orig_std = np.std(orig_times, ddof=1)
comp_mean = np.mean(comp_times)
comp_std = np.std(comp_times, ddof=1)

print(f"Original model:")
print(f"  Times: {orig_times}")
print(f"  Mean: {orig_mean:.1f}s ± {orig_std:.1f}s")
print()

print(f"Compressed model:")
print(f"  Times: {comp_times}")
print(f"  Mean: {comp_mean:.1f}s ± {comp_std:.1f}s")
print()

overhead = ((comp_mean - orig_mean) / orig_mean) * 100
print(f"Training Overhead: {overhead:.2f}%")
print()

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_rel(comp_times, orig_times)
print(f"Statistical Test:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"  Result: Statistically significant (p < 0.05)")
else:
    print(f"  Result: Not statistically significant (p >= 0.05)")
print()

# Save summary
log_file.write("\n" + "="*80 + "\n")
log_file.write("FINAL RESULTS\n")
log_file.write("="*80 + "\n\n")
log_file.write(f"Original: {orig_mean:.1f}s ± {orig_std:.1f}s\n")
log_file.write(f"Compressed: {comp_mean:.1f}s ± {comp_std:.1f}s\n")
log_file.write(f"Overhead: {overhead:.2f}%\n")
log_file.write(f"p-value: {p_value:.4f}\n")

log_file.close()

print(f"Results saved to: {results_dir}/")
print()

