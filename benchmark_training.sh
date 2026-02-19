#!/bin/bash

echo "========================================="
echo "DLRM Training Benchmark"
echo "========================================="
echo "Start time: $(date)"
echo ""

ORIGINAL_MODEL="./models/dlrm_kaggle_quick.pt"
COMPRESSED_MODEL="./models/dlrm_kaggle_quick_compressed.pt"
DECOMPRESSED_MODEL="./models/dlrm_kaggle_quick_decompressed.pt"

# Configuration
NUM_RUNS=3
NUM_EPOCHS=2
BATCH_SIZE=2048
NUM_BATCHES=100  # Limit batches for faster benchmark

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="training_benchmark_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Configuration:"
echo "  Runs: $NUM_RUNS"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Batches per epoch: $NUM_BATCHES"
echo "  Results dir: $RESULTS_DIR"
echo ""

# Log to file as well as console
exec > >(tee -a "$RESULTS_DIR/benchmark_log.txt")
exec 2>&1

# Check models exist
if [ ! -f "$ORIGINAL_MODEL" ]; then
    echo "ERROR: Original model not found!"
    exit 1
fi

if [ ! -f "$COMPRESSED_MODEL" ]; then
    echo "ERROR: Compressed model not found!"
    exit 1
fi

# Function to clear caches
clear_caches() {
    echo "Clearing caches..."
    sync
    if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
        echo "  ✓ Caches cleared"
    else
        echo "  ⚠ Could not clear caches (no sudo access)"
        echo "  Sleeping 5 seconds to let system settle..."
        sleep 5
    fi
    sleep 2
    echo ""
}

# ========================================
# Decompress model first (one time only)
# ========================================
echo "========================================="
echo "STEP 0: Decompression (One-time)"
echo "========================================="
echo ""

if [ -f "$DECOMPRESSED_MODEL" ]; then
    echo "Decompressed model already exists, removing old version..."
    rm "$DECOMPRESSED_MODEL"
fi

echo "Clearing caches before decompression..."
clear_caches

echo "Decompressing model..."
DECOMP_START=$(date +%s)

python << 'PYTHON'
import torch
import numpy as np
import subprocess
import tempfile
import os
import time

print("Loading compressed model...")
compressed = torch.load('./models/dlrm_kaggle_quick_compressed.pt', map_location='cpu')
original = torch.load('./models/dlrm_kaggle_quick.pt', map_location='cpu')

print("Decompressing embedding tables...")
decompressed = original.copy()
compressed_tables = compressed['compressed_tables']

def decompress_table(compressed_data, metadata):
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(compressed_data)
        video_path = f.name
    
    try:
        cmd = [
            'ffmpeg', '-loglevel', 'quiet',
            '-i', video_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            'pipe:1'
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        raw_pixels = np.frombuffer(result.stdout, dtype=np.uint8)
        
        num_embeddings, embedding_dim = metadata['shape']
        total_elements = num_embeddings * embedding_dim
        quantized = raw_pixels[:total_elements].reshape(num_embeddings, embedding_dim)
        
        scale = metadata['quant_params']['scale']
        zero_point = metadata['quant_params']['zero_point']
        dequantized = (quantized.astype(np.float32) - zero_point) * scale
        
        return dequantized
    finally:
        os.unlink(video_path)

start_time = time.time()
for table_idx in sorted(compressed_tables.keys()):
    table_data = compressed_tables[table_idx]
    weights = decompress_table(table_data['data'], table_data['metadata'])
    decompressed['state_dict'][f'emb_l.{table_idx}.weight'] = torch.from_numpy(weights)
    print(f"  Decompressed table {table_idx}: {weights.shape}")

decomp_time = time.time() - start_time

print(f"\nDecompression took {decomp_time:.2f} seconds")
print("Saving decompressed model...")
torch.save(decompressed, './models/dlrm_kaggle_quick_decompressed.pt')
print("✓ Saved!")
PYTHON

DECOMP_END=$(date +%s)
DECOMP_TIME=$((DECOMP_END - DECOMP_START))

echo ""
echo "Decompression complete: ${DECOMP_TIME}s"
echo ""

# Save decompression time
echo "Decompression time: ${DECOMP_TIME}s" > "$RESULTS_DIR/decompression_time.txt"

# ========================================
# Training Benchmark Loop
# ========================================

echo "========================================="
echo "MULTI-RUN TRAINING BENCHMARK"
echo "========================================="
echo "Date: $(date)"
echo "Results directory: $RESULTS_DIR"
echo ""

# Arrays to store results
declare -a ORIGINAL_TIMES
declare -a COMPRESSED_TIMES

for RUN in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "========================================="
    echo "ITERATION $RUN/$NUM_RUNS"
    echo "========================================="
    echo ""
    
    # ========================================
    # Train Original Model
    # ========================================
    echo "Training ORIGINAL model (iteration $RUN)..."
    echo ""
    
    clear_caches
    
    echo "Starting training..."
    START_TIME=$(date +%s)
    
    python dlrm_s_pytorch.py \
        --arch-sparse-feature-size=16 \
        --arch-mlp-bot="13-512-256-64-16" \
        --arch-mlp-top="512-256-1" \
        --data-generation=dataset \
        --data-set=kaggle \
        --raw-data-file=./input/train.txt \
        --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
        --loss-function=bce \
        --round-targets=True \
        --learning-rate=0.01 \
        --mini-batch-size=$BATCH_SIZE \
        --nepochs=$NUM_EPOCHS \
        --num-batches=$NUM_BATCHES \
        --num-workers=0 \
        --load-model="$ORIGINAL_MODEL" \
        --print-freq=20 \
        2>&1 | tee "$RESULTS_DIR/original_iter${RUN}.log"
    
    END_TIME=$(date +%s)
    ORIG_TIME=$((END_TIME - START_TIME))
    ORIGINAL_TIMES+=($ORIG_TIME)
    
    # Extract final loss/accuracy if available
    ORIG_LOSS=$(grep -oP "loss \K[\d.]+" "$RESULTS_DIR/original_iter${RUN}.log" | tail -1)
    ORIG_ACC=$(grep -oP "accuracy \K[\d.]+" "$RESULTS_DIR/original_iter${RUN}.log" | tail -1)
    
    echo ""
    echo "  Time: ${ORIG_TIME}s"
    echo "  Loss: ${ORIG_LOSS}"
    echo "  Accuracy: ${ORIG_ACC}%"
    echo ""
    
    # ========================================
    # Train Compressed Model (Decompressed)
    # ========================================
    echo "Training COMPRESSED model (iteration $RUN)..."
    echo ""
    
    clear_caches
    
    echo "Starting training..."
    START_TIME=$(date +%s)
    
    python dlrm_s_pytorch.py \
        --arch-sparse-feature-size=16 \
        --arch-mlp-bot="13-512-256-64-16" \
        --arch-mlp-top="512-256-1" \
        --data-generation=dataset \
        --data-set=kaggle \
        --raw-data-file=./input/train.txt \
        --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
        --loss-function=bce \
        --round-targets=True \
        --learning-rate=0.01 \
        --mini-batch-size=$BATCH_SIZE \
        --nepochs=$NUM_EPOCHS \
        --num-batches=$NUM_BATCHES \
        --num-workers=0 \
        --load-model="$DECOMPRESSED_MODEL" \
        --print-freq=20 \
        2>&1 | tee "$RESULTS_DIR/compressed_iter${RUN}.log"
    
    END_TIME=$(date +%s)
    COMP_TIME=$((END_TIME - START_TIME))
    COMPRESSED_TIMES+=($COMP_TIME)
    
    # Extract final loss/accuracy if available
    COMP_LOSS=$(grep -oP "loss \K[\d.]+" "$RESULTS_DIR/compressed_iter${RUN}.log" | tail -1)
    COMP_ACC=$(grep -oP "accuracy \K[\d.]+" "$RESULTS_DIR/compressed_iter${RUN}.log" | tail -1)
    
    echo ""
    echo "  Time: ${COMP_TIME}s"
    echo "  Loss: ${COMP_LOSS}"
    echo "  Accuracy: ${COMP_ACC}%"
    echo ""
done

# ========================================
# Final Analysis
# ========================================
echo ""
echo "========================================="
echo "FINAL RESULTS"
echo "========================================="
echo ""

echo "Original times:   ${ORIGINAL_TIMES[@]}"
echo "Compressed times: ${COMPRESSED_TIMES[@]}"
echo ""

# Calculate statistics with Python
python3 << PYTHON
import numpy as np
from scipy import stats
import os

orig_times = np.array([${ORIGINAL_TIMES[@]}])
comp_times = np.array([${COMPRESSED_TIMES[@]}])

orig_mean = np.mean(orig_times)
orig_std = np.std(orig_times, ddof=1)
comp_mean = np.mean(comp_times)
comp_std = np.std(comp_times, ddof=1)

overhead = ((comp_mean - orig_mean) / orig_mean) * 100

print("="*80)
print("TRAINING PERFORMANCE SUMMARY")
print("="*80)
print()

print(f"Original Model:")
print(f"  Times:  {orig_times}")
print(f"  Mean:   {orig_mean:.1f}s ± {orig_std:.1f}s")
print(f"  Min:    {np.min(orig_times):.1f}s")
print(f"  Max:    {np.max(orig_times):.1f}s")
print()

print(f"Compressed Model:")
print(f"  Times:  {comp_times}")
print(f"  Mean:   {comp_mean:.1f}s ± {comp_std:.1f}s")
print(f"  Min:    {np.min(comp_times):.1f}s")
print(f"  Max:    {np.max(comp_times):.1f}s")
print()

print(f"Training Overhead: {overhead:+.2f}%")
print(f"  Compressed is {overhead:.2f}% {'slower' if overhead > 0 else 'faster'}")
print()

# Statistical test
if len(orig_times) > 1:
    t_stat, p_value = stats.ttest_rel(comp_times, orig_times)
    print(f"Statistical Test (Paired t-test):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Result: Statistically significant (p < 0.05)")
    else:
        print(f"  Result: Not statistically significant (p >= 0.05)")
    print()

# Verdict
print("="*80)
print("VERDICT")
print("="*80)
print()

if abs(overhead) < 2:
    print("✓ EXCELLENT: Training overhead is negligible (<2%)")
elif abs(overhead) < 5:
    print("✓ GOOD: Training overhead is minimal (<5%)")
elif abs(overhead) < 10:
    print("⚠ MODERATE: Training overhead is acceptable (<10%)")
else:
    print("⚠ HIGH: Training overhead is significant (>10%)")

print()
print("Compression Summary:")
print("  • Model size: 148x smaller (2061 MB → 14 MB)")
print(f"  • Training overhead: {overhead:+.1f}%")
print()

# Model sizes
orig_size = os.path.getsize("$ORIGINAL_MODEL") / (1024**2)
comp_size = os.path.getsize("$COMPRESSED_MODEL") / (1024**2)

print("="*80)
print("COMPLETE RESULTS SUMMARY")
print("="*80)
print()
print(f"Model Sizes:")
print(f"  Original:   {orig_size:.2f} MB")
print(f"  Compressed: {comp_size:.2f} MB")
print(f"  Ratio:      {orig_size/comp_size:.1f}x")
print()
print(f"Training Performance:")
print(f"  Original:   {orig_mean:.1f}s ± {orig_std:.1f}s")
print(f"  Compressed: {comp_mean:.1f}s ± {comp_std:.1f}s")
print(f"  Overhead:   {overhead:+.2f}%")
print()

# Save detailed summary
with open("$RESULTS_DIR/training_summary.txt", "w") as f:
    f.write("TRAINING BENCHMARK SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Date: $(date)\n")
    f.write(f"Results directory: $RESULTS_DIR\n\n")
    
    f.write(f"Configuration:\n")
    f.write(f"  Runs: $NUM_RUNS\n")
    f.write(f"  Epochs per run: $NUM_EPOCHS\n")
    f.write(f"  Batch size: $BATCH_SIZE\n")
    f.write(f"  Batches per epoch: $NUM_BATCHES\n\n")
    
    f.write(f"Model Sizes:\n")
    f.write(f"  Original:   {orig_size:.2f} MB\n")
    f.write(f"  Compressed: {comp_size:.2f} MB\n")
    f.write(f"  Ratio:      {orig_size/comp_size:.1f}x\n\n")
    
    f.write(f"Training Times:\n")
    f.write(f"  Original times:   {orig_times}\n")
    f.write(f"  Compressed times: {comp_times}\n\n")
    
    f.write(f"Statistics:\n")
    f.write(f"  Original:   {orig_mean:.1f}s ± {orig_std:.1f}s (min={np.min(orig_times):.1f}s, max={np.max(orig_times):.1f}s)\n")
    f.write(f"  Compressed: {comp_mean:.1f}s ± {comp_std:.1f}s (min={np.min(comp_times):.1f}s, max={np.max(comp_times):.1f}s)\n")
    f.write(f"  Overhead:   {overhead:+.2f}%\n\n")
    
    if len(orig_times) > 1:
        f.write(f"Statistical Test:\n")
        f.write(f"  t-statistic: {t_stat:.3f}\n")
        f.write(f"  p-value:     {p_value:.4f}\n")
        f.write(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)\n\n")
    
    f.write(f"Verdict:\n")
    if abs(overhead) < 2:
        f.write(f"  ✓ EXCELLENT: Training overhead is negligible (<2%)\n")
    elif abs(overhead) < 5:
        f.write(f"  ✓ GOOD: Training overhead is minimal (<5%)\n")
    elif abs(overhead) < 10:
        f.write(f"  ⚠ MODERATE: Training overhead is acceptable (<10%)\n")
    else:
        f.write(f"  ⚠ HIGH: Training overhead is significant (>10%)\n")

print(f"\nResults saved to: $RESULTS_DIR/")
print()

PYTHON

echo ""
echo "========================================="
echo "RESULTS SAVED"
echo "========================================="
echo ""
echo "Directory: $RESULTS_DIR/"
echo ""
echo "Files:"
ls -lh "$RESULTS_DIR/"
echo ""

echo "Summary file: $RESULTS_DIR/training_summary.txt"
echo "Full log:     $RESULTS_DIR/benchmark_log.txt"
echo ""

echo "========================================="
echo "BENCHMARK COMPLETE!"
echo "========================================="
echo "End time: $(date)"
echo ""

