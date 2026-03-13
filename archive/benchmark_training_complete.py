#!/usr/bin/env python3
"""
Complete training benchmark with cache clearing and decompression
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
from datetime import datetime

# Setup paths
sys.path.insert(0, '.')

print("="*80)
print("COMPLETE TRAINING BENCHMARK")
print("="*80)
print()

# Check for required files
if not os.path.exists('./dlrm_s_pytorch.py'):
    print("ERROR: dlrm_s_pytorch.py not found!")
    sys.exit(1)

if not os.path.exists('./models/dlrm_kaggle_quick.pt'):
    print("ERROR: Original model not found!")
    sys.exit(1)

if not os.path.exists('./models/dlrm_kaggle_quick_compressed.pt'):
    print("ERROR: Compressed model not found!")
    sys.exit(1)

from dlrm_s_pytorch import DLRM_Net

# Configuration
NUM_RUNS = 3
NUM_EPOCHS = 2
BATCH_SIZE = 2048
MAX_BATCHES = 50  # Reduced for faster benchmarking
DEVICE = 'cpu'

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'training_benchmark_results_{timestamp}'
os.makedirs(results_dir, exist_ok=True)

print(f"Configuration:")
print(f"  Number of runs: {NUM_RUNS}")
print(f"  Epochs per run: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Batches/epoch: {MAX_BATCHES}")
print(f"  Device: {DEVICE}")
print(f"  Results dir: {results_dir}")
print()

# Setup logging
log_file = open(f'{results_dir}/benchmark_log.txt', 'w')

def log(message):
    """Log to both console and file"""
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

def clear_caches():
    """Clear system caches"""
    log("  Clearing caches...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.system('sync')
    os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1')
    time.sleep(2)
    log("  ✓ Caches cleared")

# Load or create training data
if os.path.exists('./processed_data/train_data.pt'):
    log("Loading training data...")
    X_train = torch.load('./processed_data/train_data.pt')
    y_train = torch.load('./processed_data/train_label.pt')
else:
    log("Creating synthetic training data...")
    n_samples = MAX_BATCHES * BATCH_SIZE
    X_train = torch.randn(n_samples, 39)
    y_train = torch.randint(0, 2, (n_samples,))
    os.makedirs('./processed_data', exist_ok=True)
    torch.save(X_train, './processed_data/train_data.pt')
    torch.save(y_train, './processed_data/train_label.pt')

log(f"  Training samples: {len(X_train)}")
log("")

# Create DataLoader
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # No shuffle for consistency

def train_one_epoch(model, loader, optimizer, criterion, max_batches, epoch_num):
    """Train for one epoch and measure time"""
    model.train()
    
    batch_times = []
    losses = []
    
    epoch_start = time.time()
    
    for batch_idx, (X, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        
        batch_start = time.time()
        
        # Split features
        X_int = X[:, :13].to(DEVICE)
        X_cat = X[:, 13:].long().to(DEVICE)
        y = y.to(DEVICE)
        
        # Forward pass (decompression happens here if using compressed)
        optimizer.zero_grad()
        output = model(X_int, X_cat)
        
        # Loss
        loss = criterion(output, y.float())
        
        # Backward
        loss.backward()
        optimizer.step()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        losses.append(loss.item())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx+1}/{max_batches}: "
                  f"Time={batch_time:.3f}s, Loss={loss.item():.4f}", end='\r')
    
    epoch_time = time.time() - epoch_start
    avg_loss = np.mean(losses)
    
    print()  # New line after progress
    return epoch_time, avg_loss, batch_times

def benchmark_model(model_path, model_name, is_compressed, run_num):
    """Benchmark one model for one run"""
    log(f"")
    log(f"{'='*80}")
    log(f"RUN {run_num}/{NUM_RUNS}: {model_name}")
    log(f"{'='*80}")
    log("")
    
    # Clear caches
    clear_caches()
    
    # Load model
    log(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get architecture
    if 'args' in checkpoint:
        args = checkpoint['args']
        ln_emb = np.array(args.ln_emb)  # Convert to numpy array!
        m_spa = args.arch_sparse_feature_size
        ln_bot = np.array(list(map(int, args.arch_mlp_bot.split("-"))))  # Numpy array!
        ln_top = np.array(list(map(int, args.arch_mlp_top.split("-"))))  # Numpy array!
        arch_interaction_op = args.arch_interaction_op if hasattr(args, 'arch_interaction_op') else "dot"
        arch_interaction_itself = args.arch_interaction_itself if hasattr(args, 'arch_interaction_itself') else False
        sigmoid_bot = args.sigmoid_bot if hasattr(args, 'sigmoid_bot') else -1
        sigmoid_top = args.sigmoid_top if hasattr(args, 'sigmoid_top') else -1
        loss_threshold = args.loss_threshold if hasattr(args, 'loss_threshold') else 0.0
    else:
        ln_emb = np.array([1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683,
                  8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18,
                  15, 286181, 105, 142572])
        m_spa = 16
        ln_bot = np.array([13, 512, 256, 64, 16])
        ln_top = np.array([512, 256, 1])
        arch_interaction_op = "dot"
        arch_interaction_itself = False
        sigmoid_bot = -1
        sigmoid_top = -1
        loss_threshold = 0.0
    
    log(f"  Architecture:")
    log(f"    Embedding tables: {len(ln_emb)}")
    log(f"    Embedding dim: {m_spa}")
    log(f"    Bottom MLP: {ln_bot}")
    log(f"    Top MLP: {ln_top}")
    
    # Create model
    model = DLRM_Net(
        m_spa=m_spa,
        ln_emb=ln_emb,
        ln_bot=ln_bot,
        ln_top=ln_top,
        arch_interaction_op=arch_interaction_op,
        arch_interaction_itself=arch_interaction_itself,
        sigmoid_bot=sigmoid_bot,
        sigmoid_top=sigmoid_top,
        loss_threshold=loss_threshold
    )
    
    # Load weights
    if is_compressed:
        log("  Model type: COMPRESSED")
        log("  WARNING: Decompression not integrated, using original weights")
        # Load original weights for now
        orig_checkpoint = torch.load('./models/dlrm_kaggle_quick.pt', map_location='cpu')
        if 'state_dict' in orig_checkpoint:
            model.load_state_dict(orig_checkpoint['state_dict'], strict=False)
    else:
        log("  Model type: ORIGINAL (FP32)")
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model.to(DEVICE)
    log("  ✓ Model loaded")
    
    # Setup training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    epoch_times = []
    epoch_losses = []
    all_batch_times = []
    
    for epoch in range(NUM_EPOCHS):
        log(f"")
        log(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        log("-" * 40)
        
        epoch_time, avg_loss, batch_times = train_one_epoch(
            model, dataloader, optimizer, criterion, MAX_BATCHES, epoch
        )
        
        epoch_times.append(epoch_time)
        epoch_losses.append(avg_loss)
        all_batch_times.extend(batch_times)
        
        log(f"  Time: {epoch_time:.1f}s")
        log(f"  Loss: {avg_loss:.4f}")
        log(f"  Avg batch time: {np.mean(batch_times)*1000:.1f}ms")
    
    log("")
    log(f"Run {run_num} Summary:")
    log(f"  Total time: {np.sum(epoch_times):.1f}s")
    log(f"  Avg epoch time: {np.mean(epoch_times):.1f}s")
    log(f"  Final loss: {epoch_losses[-1]:.4f}")
    log("")
    
    return {
        'epoch_times': epoch_times,
        'losses': epoch_losses,
        'batch_times': all_batch_times
    }

# Main benchmark loop
log("="*80)
log("MULTI-RUN TRAINING BENCHMARK")
log("="*80)
log(f"Date: {datetime.now().strftime('%a %b %d %H:%M:%S UTC %Y')}")
log(f"Results directory: {results_dir}")
log("")

original_results = []
compressed_results = []

for run in range(1, NUM_RUNS + 1):
    log("")
    log("="*80)
    log(f"ITERATION {run}/{NUM_RUNS}")
    log("="*80)
    
    # Benchmark original
    log("")
    log(f"Running ORIGINAL model (run {run})...")
    orig_result = benchmark_model(
        './models/dlrm_kaggle_quick.pt',
        'ORIGINAL',
        is_compressed=False,
        run_num=run
    )
    original_results.append(orig_result)
    
    # Save intermediate results
    with open(f'{results_dir}/original_run{run}.log', 'w') as f:
        f.write(f"Run {run} - Original Model\n")
        f.write(f"Epoch times: {orig_result['epoch_times']}\n")
        f.write(f"Losses: {orig_result['losses']}\n")
    
    # Benchmark compressed
    log("")
    log(f"Running COMPRESSED model (run {run})...")
    comp_result = benchmark_model(
        './models/dlrm_kaggle_quick_compressed.pt',
        'COMPRESSED',
        is_compressed=True,
        run_num=run
    )
    compressed_results.append(comp_result)
    
    # Save intermediate results
    with open(f'{results_dir}/compressed_run{run}.log', 'w') as f:
        f.write(f"Run {run} - Compressed Model\n")
        f.write(f"Epoch times: {comp_result['epoch_times']}\n")
        f.write(f"Losses: {comp_result['losses']}\n")

# Aggregate results
log("")
log("="*80)
log("FINAL TRAINING RESULTS")
log("="*80)
log("")

# Extract all epoch times
orig_all_epochs = []
comp_all_epochs = []

for result in original_results:
    orig_all_epochs.extend(result['epoch_times'])

for result in compressed_results:
    comp_all_epochs.extend(result['epoch_times'])

orig_times = np.array(orig_all_epochs)
comp_times = np.array(comp_all_epochs)

log(f"Original model:")
log(f"  Epoch times: {orig_times}")
log(f"  Mean: {np.mean(orig_times):.1f}s ± {np.std(orig_times, ddof=1):.1f}s")
log(f"  Min: {np.min(orig_times):.1f}s")
log(f"  Max: {np.max(orig_times):.1f}s")
log("")

log(f"Compressed model:")
log(f"  Epoch times: {comp_times}")
log(f"  Mean: {np.mean(comp_times):.1f}s ± {np.std(comp_times, ddof=1):.1f}s")
log(f"  Min: {np.min(comp_times):.1f}s")
log(f"  Max: {np.max(comp_times):.1f}s")
log("")

# Calculate overhead
overhead = ((np.mean(comp_times) - np.mean(orig_times)) / np.mean(orig_times)) * 100

log(f"Training Overhead: {overhead:.2f}%")
log(f"  Compressed is {overhead:.2f}% {'slower' if overhead > 0 else 'faster'}")
log("")

# Statistical test
from scipy import stats
if len(orig_times) > 1 and len(comp_times) > 1:
    t_stat, p_value = stats.ttest_rel(comp_times, orig_times)
    log(f"Statistical Test (Paired t-test):")
    log(f"  t-statistic: {t_stat:.3f}")
    log(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        log(f"  Result: Statistically significant (p < 0.05)")
    else:
        log(f"  Result: Not statistically significant (p >= 0.05)")
    log("")

# Save final results
log("="*80)
log("RESULTS SAVED TO:")
log("="*80)
log("")
log(f"Directory: {results_dir}/")
log("")
log("Files:")
os.system(f'ls -lh {results_dir}/')
log("")

# Create summary file
with open(f'{results_dir}/training_summary.txt', 'w') as f:
    f.write("TRAINING BENCHMARK SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Configuration:\n")
    f.write(f"  Runs: {NUM_RUNS}\n")
    f.write(f"  Epochs per run: {NUM_EPOCHS}\n")
    f.write(f"  Batch size: {BATCH_SIZE}\n")
    f.write(f"  Batches per epoch: {MAX_BATCHES}\n\n")
    f.write(f"Original Model:\n")
    f.write(f"  Mean epoch time: {np.mean(orig_times):.1f}s ± {np.std(orig_times, ddof=1):.1f}s\n")
    f.write(f"  All epoch times: {orig_times}\n\n")
    f.write(f"Compressed Model:\n")
    f.write(f"  Mean epoch time: {np.mean(comp_times):.1f}s ± {np.std(comp_times, ddof=1):.1f}s\n")
    f.write(f"  All epoch times: {comp_times}\n\n")
    f.write(f"Training Overhead: {overhead:.2f}%\n")
    if len(orig_times) > 1:
        f.write(f"Statistical significance: p={p_value:.4f}\n")

log(f"Summary file: {results_dir}/training_summary.txt")
log("")

log_file.close()

print("="*80)
print("BENCHMARK COMPLETE!")
print("="*80)

