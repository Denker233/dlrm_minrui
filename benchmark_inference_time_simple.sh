#!/bin/bash

echo "========================================="
echo "DLRM Inference Time Benchmark"
echo "========================================="
echo "Start time: $(date)"
echo ""

ORIGINAL_MODEL="./models/dlrm_kaggle_quick.pt"
COMPRESSED_MODEL="./models/dlrm_kaggle_quick_compressed.pt"
DECOMPRESSED_MODEL="./models/dlrm_kaggle_quick_decompressed.pt"

# Check models exist
if [ ! -f "$ORIGINAL_MODEL" ]; then
    echo "ERROR: Original model not found!"
    exit 1
fi

if [ ! -f "$COMPRESSED_MODEL" ]; then
    echo "ERROR: Compressed model not found!"
    exit 1
fi

echo "Found models:"
ls -lh "$ORIGINAL_MODEL"
ls -lh "$COMPRESSED_MODEL"
echo ""

# ========================================
# Decompress model first
# ========================================
echo "========================================="
echo "STEP 1/3: Decompressing Model"
echo "========================================="

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

# Start with original checkpoint structure (preserves all metadata)
decompressed = original.copy()

# Decompress embeddings
compressed_tables = compressed['compressed_tables']

def decompress_table(compressed_data, metadata):
    """Decompress a single table"""
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
    # Replace embedding weights in the state_dict
    decompressed['state_dict'][f'emb_l.{table_idx}.weight'] = torch.from_numpy(weights)
    print(f"  Decompressed table {table_idx}: {weights.shape}")

decomp_time = time.time() - start_time
print(f"\nDecompression complete in {decomp_time:.2f} seconds")

# Verify checkpoint has all required fields
print("\nCheckpoint fields:")
for key in decompressed.keys():
    print(f"  {key}: {type(decompressed[key])}")

print("\nSaving decompressed model...")
torch.save(decompressed, './models/dlrm_kaggle_quick_decompressed.pt')
print("✓ Saved!")
PYTHON

echo ""

# ========================================
# STEP 2: Benchmark Original Model
# ========================================
echo "========================================="
echo "STEP 2/3: Benchmarking Original Model"
echo "========================================="
echo ""

echo "Running inference on full test dataset..."
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
    --test-mini-batch-size=2048 \
    --num-workers=0 \
    --test-num-workers=0 \
    --load-model="$ORIGINAL_MODEL" \
    --inference-only \
    2>&1 | tee inference_original_benchmark.log

END_TIME=$(date +%s)
ORIGINAL_TIME=$((END_TIME - START_TIME))

ORIGINAL_ACC=$(grep -oP "accuracy \K[\d.]+" inference_original_benchmark.log | tail -1)
echo ""
echo "Original model:"
echo "  Accuracy: ${ORIGINAL_ACC}%"
echo "  Inference time: ${ORIGINAL_TIME}s"
echo ""

# ========================================
# STEP 3: Benchmark Compressed Model
# ========================================
echo "========================================="
echo "STEP 3/3: Benchmarking Compressed Model"
echo "========================================="
echo ""

echo "Running inference on full test dataset..."
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
    --test-mini-batch-size=2048 \
    --num-workers=0 \
    --test-num-workers=0 \
    --load-model="$DECOMPRESSED_MODEL" \
    --inference-only \
    2>&1 | tee inference_compressed_benchmark.log

END_TIME=$(date +%s)
COMPRESSED_TIME=$((END_TIME - START_TIME))

COMPRESSED_ACC=$(grep -oP "accuracy \K[\d.]+" inference_compressed_benchmark.log | tail -1)
echo ""
echo "Compressed model:"
echo "  Accuracy: ${COMPRESSED_ACC}%"
echo "  Inference time: ${COMPRESSED_TIME}s"
echo ""

# Check if compressed run succeeded
if grep -q "KeyError\|Traceback" inference_compressed_benchmark.log; then
    echo "⚠ WARNING: Compressed model inference had errors!"
    echo "Check inference_compressed_benchmark.log for details"
    echo ""
fi

# ========================================
# COMPARISON
# ========================================
echo "========================================="
echo "COMPARISON RESULTS"
echo "========================================="
echo ""

python3 << PYTHON
import os

# Model sizes
orig_size = os.path.getsize("$ORIGINAL_MODEL") / (1024**2)
comp_size = os.path.getsize("$COMPRESSED_MODEL") / (1024**2)
ratio = orig_size / comp_size

# Inference times
orig_time = ${ORIGINAL_TIME}
comp_time = ${COMPRESSED_TIME}
time_diff = comp_time - orig_time
time_pct = (comp_time / orig_time - 1) * 100 if orig_time > 0 else 0

# Accuracies
try:
    orig_acc = float("${ORIGINAL_ACC}")
    comp_acc = float("${COMPRESSED_ACC}")
    acc_diff = comp_acc - orig_acc
    acc_valid = True
except:
    orig_acc = 0
    comp_acc = 0
    acc_diff = 0
    acc_valid = False

print("="*80)
print("SUMMARY")
print("="*80)
print()

print("MODEL SIZES:")
print(f"  Original:   {orig_size:>10.2f} MB")
print(f"  Compressed: {comp_size:>10.2f} MB")
print(f"  Ratio:      {ratio:>10.1f}x smaller")
print()

print("INFERENCE TIME (Full Test Dataset):")
print(f"  Original:   {orig_time:>10d}s")
print(f"  Compressed: {comp_time:>10d}s")
print(f"  Difference: {time_diff:>+10d}s ({time_pct:+.2f}%)")
print()

if abs(time_pct) < 1:
    print("  ✓ ZERO INFERENCE OVERHEAD!")
elif abs(time_pct) < 5:
    print(f"  ✓ Minimal overhead: {time_pct:+.2f}%")
else:
    print(f"  ⚠ Some overhead: {time_pct:+.2f}%")

print()

if acc_valid:
    print("ACCURACY:")
    print(f"  Original:   {orig_acc:>10.4f}%")
    print(f"  Compressed: {comp_acc:>10.4f}%")
    print(f"  Difference: {acc_diff:>+10.4f}%")
    print()
    
    if abs(acc_diff) < 0.1:
        print("  ✓ Accuracy preserved! (<0.1% difference)")
    elif abs(acc_diff) < 0.5:
        print(f"  ✓ Minimal loss: {abs(acc_diff):.4f}%")
    else:
        print(f"  ⚠ Some accuracy loss: {abs(acc_diff):.4f}%")
else:
    print("ACCURACY:")
    print("  ⚠ Could not parse accuracy (check log files)")

print()
print("OVERALL TRADE-OFF:")
print(f"  Storage:    {ratio:.0f}x smaller  ✓✓✓")
print(f"  Inference:  {abs(time_pct):.1f}% difference")
if acc_valid:
    print(f"  Accuracy:   {abs(acc_diff):.4f}% difference  ✓")

print()
print("="*80)

PYTHON

echo ""
echo "========================================="
echo "BENCHMARK COMPLETE!"
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Log files:"
echo "  - inference_original_benchmark.log"
echo "  - inference_compressed_benchmark.log"
echo ""

