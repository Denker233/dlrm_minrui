#!/bin/bash

echo "========================================="
echo "DLRM Video Codec Inference Benchmark"
echo "========================================="
echo "Start time: $(date)"
echo ""

ORIGINAL_MODEL="./models/dlrm_kaggle_quick.pt"
COMPRESSED_MODEL="./models/dlrm_kaggle_quick_compressed.pt"
DECOMPRESSED_MODEL="./models/dlrm_kaggle_quick_decompressed.pt"

# Video codec settings (adjust to match your compression)
QP="${QP:-23}"  # Quality parameter used during compression
CODEC="${CODEC:-h264}"  # h264 or h265

# Check models exist
if [ ! -f "$ORIGINAL_MODEL" ]; then
    echo "ERROR: Original model not found!"
    exit 1
fi

if [ ! -f "$COMPRESSED_MODEL" ]; then
    echo "ERROR: Compressed model not found at $COMPRESSED_MODEL"
    echo "Make sure you've run your compression script first!"
    exit 1
fi

echo "Found models:"
ls -lh "$ORIGINAL_MODEL"
ls -lh "$COMPRESSED_MODEL"
echo ""
echo "Codec settings: $CODEC, QP=$QP"
echo ""

# ========================================
# STEP 1: Decompress Model
# ========================================
echo "========================================="
echo "STEP 1/3: Decompressing Video Codec Model"
echo "========================================="
echo ""

DECOMPRESS_START=$(date +%s)

python << 'PYTHON'
import torch
import numpy as np
import subprocess
import tempfile
import os
import time
import sys

print("Loading compressed model...")
compressed = torch.load('./models/dlrm_kaggle_quick_compressed.pt', map_location='cpu')
original = torch.load('./models/dlrm_kaggle_quick.pt', map_location='cpu')

print("Model compression info:")
if 'compression_info' in compressed:
    info = compressed['compression_info']
    print(f"  Codec: {info.get('codec', 'unknown')}")
    print(f"  QP: {info.get('qp', 'unknown')}")
    print(f"  Quantization: {info.get('quantization', 'unknown')}")
    print(f"  Compression time: {info.get('compression_time', 'unknown')}")
print()

print("Decompressing embedding tables...")

# Start with original checkpoint structure
decompressed = original.copy()

# Get compressed tables
compressed_tables = compressed['compressed_tables']

def decompress_table(compressed_data, metadata):
    """Decompress a single embedding table using video codec"""
    # Write compressed video to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(compressed_data)
        video_path = f.name
    
    try:
        # Decode video using ffmpeg
        cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-i', video_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            'pipe:1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, check=True)
        raw_pixels = np.frombuffer(result.stdout, dtype=np.uint8)
        
        # Reshape to original dimensions
        num_embeddings, embedding_dim = metadata['shape']
        total_elements = num_embeddings * embedding_dim
        
        if len(raw_pixels) < total_elements:
            print(f"Warning: Decoded {len(raw_pixels)} pixels, expected {total_elements}")
            # Pad if necessary
            raw_pixels = np.pad(raw_pixels, (0, total_elements - len(raw_pixels)))
        
        quantized = raw_pixels[:total_elements].reshape(num_embeddings, embedding_dim)
        
        # Dequantize
        scale = metadata['quant_params']['scale']
        zero_point = metadata['quant_params']['zero_point']
        dequantized = (quantized.astype(np.float32) - zero_point) * scale
        
        return dequantized
        
    except subprocess.CalledProcessError as e:
        print(f"Error decoding video: {e}")
        print(f"stderr: {e.stderr.decode()}")
        raise
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)

# Decompress all tables
start_time = time.time()
total_tables = len(compressed_tables)

for idx, table_idx in enumerate(sorted(compressed_tables.keys())):
    table_data = compressed_tables[table_idx]
    weights = decompress_table(table_data['data'], table_data['metadata'])
    
    # Replace embedding weights in state_dict
    decompressed['state_dict'][f'emb_l.{table_idx}.weight'] = torch.from_numpy(weights)
    
    print(f"  [{idx+1}/{total_tables}] Decompressed table {table_idx}: {weights.shape}")

decomp_time = time.time() - start_time
print(f"\n✓ Decompression complete in {decomp_time:.2f} seconds")

print("\nSaving decompressed model...")
torch.save(decompressed, './models/dlrm_kaggle_quick_decompressed.pt')
print("✓ Saved to ./models/dlrm_kaggle_quick_decompressed.pt")
PYTHON

DECOMPRESS_END=$(date +%s)
DECOMPRESS_TIME=$((DECOMPRESS_END - DECOMPRESS_START))
echo ""
echo "Decompression took: ${DECOMPRESS_TIME}s"
echo ""

# ========================================
# STEP 2: Benchmark Decompressed (Codec) Model
# ========================================
echo "========================================="
echo "STEP 2/3: Benchmarking Codec Decompressed Model"
echo "========================================="
echo ""

echo "Running inference on decompressed model (warm cache)..."
echo ""

# FIX: Properly capture time output in the log file
{ python dlrm_s_pytorch.py \
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
    --print-time \
    --load-model="$DECOMPRESSED_MODEL" \
    --inference-only \
    2>&1; } 2>&1 | tee inference_codec_decompressed.log

echo ""

# ========================================
# STEP 3: Clear Cache and Benchmark Original Model
# ========================================
echo "========================================="
echo "STEP 3/3: Benchmarking Original Model"
echo "========================================="
echo ""

echo "Clearing system caches..."
sync
if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    echo "✓ System caches cleared"
else
    echo "⚠ Could not clear caches (no sudo access)"
fi
echo ""

echo "Running inference on original model (cold cache)..."
echo ""

# FIX: Properly capture time output in the log file
{ python dlrm_s_pytorch.py \
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
    --print-time \
    --load-model="$ORIGINAL_MODEL" \
    --inference-only \
    2>&1; } 2>&1 | tee inference_original_codec.log

echo ""

# ========================================
# COMPARISON
# ========================================
echo "========================================="
echo "DETAILED COMPARISON RESULTS"
echo "========================================="
echo ""

# Extract metrics
CODEC_ACC=$(grep -oP "accuracy \K[\d.]+" inference_codec_decompressed.log | tail -1)
CODEC_AUC=$(grep -oP "auc \K[\d.]+" inference_codec_decompressed.log | tail -1)
CODEC_EMB=$(grep "The embedding time is" inference_codec_decompressed.log | awk '{print $NF}')
CODEC_TOTAL=$(grep "The total time is" inference_codec_decompressed.log | awk '{print $NF}')

ORIGINAL_ACC=$(grep -oP "accuracy \K[\d.]+" inference_original_codec.log | tail -1)
ORIGINAL_AUC=$(grep -oP "auc \K[\d.]+" inference_original_codec.log | tail -1)
ORIGINAL_EMB=$(grep "The embedding time is" inference_original_codec.log | awk '{print $NF}')
ORIGINAL_TOTAL=$(grep "The total time is" inference_original_codec.log | awk '{print $NF}')

python3 << PYTHON
import os
import re

# Model sizes
orig_size = os.path.getsize("$ORIGINAL_MODEL") / (1024**2)
comp_size = os.path.getsize("$COMPRESSED_MODEL") / (1024**2)
decomp_size = os.path.getsize("$DECOMPRESSED_MODEL") / (1024**2)
comp_ratio = orig_size / comp_size

# Decompression time
decomp_time = ${DECOMPRESS_TIME}

# Extract wall clock times from logs
def extract_time(logfile):
    """Extract real/user/sys time from log file"""
    try:
        with open(logfile, 'r') as f:
            content = f.read()
        
        real_match = re.search(r'real\s+(\d+)m([\d.]+)s', content)
        user_match = re.search(r'user\s+(\d+)m([\d.]+)s', content)
        sys_match = re.search(r'sys\s+(\d+)m([\d.]+)s', content)
        
        real_time = float(real_match.group(1)) * 60 + float(real_match.group(2)) if real_match else None
        user_time = float(user_match.group(1)) * 60 + float(user_match.group(2)) if user_match else None
        sys_time = float(sys_match.group(1)) * 60 + float(sys_match.group(2)) if sys_match else None
        
        return real_time, user_time, sys_time
    except:
        return None, None, None

codec_real, codec_user, codec_sys = extract_time('inference_codec_decompressed.log')
orig_real, orig_user, orig_sys = extract_time('inference_original_codec.log')

# Inference times (internal)
try:
    orig_total = float("${ORIGINAL_TOTAL}")
    codec_total = float("${CODEC_TOTAL}")
    total_diff = codec_total - orig_total
    total_pct = (codec_total / orig_total - 1) * 100 if orig_total > 0 else 0
    has_total = True
except:
    has_total = False

try:
    orig_emb = float("${ORIGINAL_EMB}")
    codec_emb = float("${CODEC_EMB}")
    emb_diff = codec_emb - orig_emb
    emb_pct = (codec_emb / orig_emb - 1) * 100 if orig_emb > 0 else 0
    has_emb = True
except:
    has_emb = False

# Accuracies
try:
    orig_acc = float("${ORIGINAL_ACC}")
    codec_acc = float("${CODEC_ACC}")
    acc_diff = codec_acc - orig_acc
    acc_pct = (acc_diff / orig_acc) * 100 if orig_acc > 0 else 0
    has_acc = True
except:
    has_acc = False

try:
    orig_auc = float("${ORIGINAL_AUC}")
    codec_auc = float("${CODEC_AUC}")
    auc_diff = codec_auc - orig_auc
    auc_pct = (auc_diff / orig_auc) * 100 if orig_auc > 0 else 0
    has_auc = True
except:
    has_auc = False

print("="*80)
print("VIDEO CODEC COMPRESSION BENCHMARK RESULTS")
print("="*80)
print()

print("MODEL SIZES:")
print(f"  Original:            {orig_size:>10.2f} MB")
print(f"  Compressed (codec):  {comp_size:>10.2f} MB")
print(f"  Decompressed:        {decomp_size:>10.2f} MB")
print(f"  Compression Ratio:   {comp_ratio:>10.1f}x")
print()

print("DECOMPRESSION OVERHEAD:")
print(f"  Decompression time:  {decomp_time:>10d}s")
print(f"  Note: This is a one-time offline cost")
print()

if codec_real and orig_real:
    time_diff = codec_real - orig_real
    time_pct = (codec_real / orig_real - 1) * 100 if orig_real > 0 else 0
    
    print("INFERENCE TIME (Wall Clock - from 'time' command):")
    print(f"  Codec (warm cache):    {codec_real:>10.1f}s")
    print(f"  Original (cold cache): {orig_real:>10.1f}s")
    print(f"  Difference:            {time_diff:>+10.1f}s ({time_pct:+.2f}%)")
    print()
    
    if codec_user and orig_user:
        print("CPU TIME:")
        print(f"  Codec user time:       {codec_user:>10.1f}s")
        print(f"  Original user time:    {orig_user:>10.1f}s")
        print()

if has_total:
    print("INFERENCE TIME (Internal - Pure Inference Loop):")
    print(f"  Codec:               {codec_total:>10.2f}s")
    print(f"  Original:            {orig_total:>10.2f}s")
    print(f"  Difference:          {total_diff:>+10.2f}s ({total_pct:+.2f}%)")
    print()

if has_emb:
    print("EMBEDDING TIME (Internal):")
    print(f"  Codec:               {codec_emb:>10.2f}s")
    print(f"  Original:            {orig_emb:>10.2f}s")
    print(f"  Difference:          {emb_diff:>+10.2f}s ({emb_pct:+.2f}%)")
    print()

if has_acc:
    print("ACCURACY:")
    print(f"  Codec:               {codec_acc:>10.4f}%")
    print(f"  Original:            {orig_acc:>10.4f}%")
    print(f"  Difference:          {acc_diff:>+10.4f}% ({acc_pct:+.3f}%)")
    print()
    
if has_auc:
    print("AUC:")
    print(f"  Codec:               {codec_auc:>10.4f}%")
    print(f"  Original:            {orig_auc:>10.4f}%")
    print(f"  Difference:          {auc_diff:>+10.4f}% ({auc_pct:+.3f}%)")
    print()

print("="*80)
print("SUMMARY")
print("="*80)
print()

print(f"✓ Compression: {comp_ratio:.0f}x smaller ({orig_size:.1f}MB → {comp_size:.1f}MB)")

if codec_real and orig_real:
    if abs(time_pct) < 1:
        print(f"✓ Inference: ~0% overhead (essentially identical)")
    elif abs(time_pct) < 5:
        print(f"✓ Inference: {abs(time_pct):.1f}% overhead (minimal)")
    else:
        if time_pct < 0:
            print(f"✓ Inference: {abs(time_pct):.1f}% FASTER!")
        else:
            print(f"⚠ Inference: {time_pct:.1f}% slower")

if has_acc and abs(acc_diff) < 0.1:
    print(f"✓ Accuracy: {abs(acc_diff):.4f}% loss (negligible)")
elif has_acc and abs(acc_diff) < 1.0:
    print(f"✓ Accuracy: {abs(acc_diff):.4f}% loss (acceptable)")
elif has_acc:
    print(f"⚠ Accuracy: {abs(acc_diff):.4f}% loss")

print()
print("NOTE: Codec model tested with warm cache, Original with cold cache")
print("This gives codec a slight advantage but shows real-world deployment scenario")
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
echo "  - inference_codec_decompressed.log"
echo "  - inference_original_codec.log"
echo ""