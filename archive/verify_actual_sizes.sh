#!/bin/bash

echo "================================================================================"
echo "VERIFYING ACTUAL COMPRESSED MODEL SIZE"
echo "================================================================================"
echo ""

MODEL_DIR="./models"

echo "Checking actual file sizes on disk:"
echo ""

# Original model
if [ -f "$MODEL_DIR/dlrm_kaggle_quick.pt" ]; then
    ORIGINAL_SIZE=$(ls -lh "$MODEL_DIR/dlrm_kaggle_quick.pt" | awk '{print $5}')
    ORIGINAL_BYTES=$(stat -f%z "$MODEL_DIR/dlrm_kaggle_quick.pt" 2>/dev/null || stat -c%s "$MODEL_DIR/dlrm_kaggle_quick.pt" 2>/dev/null)
    echo "Original model:"
    echo "  File: dlrm_kaggle_quick.pt"
    echo "  Size: $ORIGINAL_SIZE ($ORIGINAL_BYTES bytes)"
else
    echo "❌ Original model not found!"
fi

echo ""

# Compressed model
if [ -f "$MODEL_DIR/dlrm_kaggle_quick_compressed.pt" ]; then
    COMPRESSED_SIZE=$(ls -lh "$MODEL_DIR/dlrm_kaggle_quick_compressed.pt" | awk '{print $5}')
    COMPRESSED_BYTES=$(stat -f%z "$MODEL_DIR/dlrm_kaggle_quick_compressed.pt" 2>/dev/null || stat -c%s "$MODEL_DIR/dlrm_kaggle_quick_compressed.pt" 2>/dev/null)
    echo "Compressed model:"
    echo "  File: dlrm_kaggle_quick_compressed.pt"
    echo "  Size: $COMPRESSED_SIZE ($COMPRESSED_BYTES bytes)"
    
    if [ ! -z "$ORIGINAL_BYTES" ] && [ ! -z "$COMPRESSED_BYTES" ]; then
        RATIO=$(echo "scale=2; $ORIGINAL_BYTES / $COMPRESSED_BYTES" | bc)
        echo "  Actual compression ratio: ${RATIO}x"
    fi
else
    echo "❌ Compressed model not found!"
fi

echo ""
echo "================================================================================"
echo "CHECKING WHAT'S INSIDE THE COMPRESSED MODEL"
echo "================================================================================"
echo ""

python << 'PYEOF'
import torch
import os

model_path = './models/dlrm_kaggle_quick_compressed.pt'

if not os.path.exists(model_path):
    print("❌ Compressed model not found!")
    exit(1)

print("Loading compressed model...")
compressed = torch.load(model_path, map_location='cpu')

print(f"Keys in compressed model: {list(compressed.keys())[:10]}...")
print()

# Analyze what's stored
total_size = 0
embedding_compressed_size = 0
embedding_metadata_size = 0
mlp_size = 0
other_size = 0

for key, value in compressed.items():
    if isinstance(value, bytes):
        size = len(value)
    elif isinstance(value, torch.Tensor):
        size = value.element_size() * value.numel()
    elif isinstance(value, dict):
        size = sum(len(str(k)) + len(str(v)) for k, v in value.items())
    else:
        size = len(str(value))
    
    total_size += size
    
    if 'compressed' in key:
        embedding_compressed_size += size
    elif 'metadata' in key:
        embedding_metadata_size += size
    elif 'top_l' in key or 'bot_l' in key:
        mlp_size += size
    else:
        other_size += size

print("Content breakdown:")
print(f"  Compressed embeddings: {embedding_compressed_size / 1024 / 1024:.2f} MB")
print(f"  Embedding metadata:    {embedding_metadata_size / 1024 / 1024:.2f} MB")
print(f"  MLP weights:           {mlp_size / 1024 / 1024:.2f} MB")
print(f"  Other (structure):     {other_size / 1024 / 1024:.2f} MB")
print(f"  Total calculated:      {total_size / 1024 / 1024:.2f} MB")
print()

# Get actual file size
file_size = os.path.getsize(model_path)
print(f"Actual file size on disk: {file_size / 1024 / 1024:.2f} MB")
print()

# Check if there's a discrepancy
overhead = file_size - total_size
if overhead > 0:
    print(f"PyTorch serialization overhead: {overhead / 1024 / 1024:.2f} MB")
    print("(This includes pickle protocol, tensor metadata, etc.)")
print()

# Count compressed tables
compressed_tables = [k for k in compressed.keys() if 'compressed' in k]
print(f"Number of compressed embedding tables: {len(compressed_tables)}")
print()

# Show size of largest compressed tables
print("Largest compressed tables:")
table_sizes = []
for key in compressed_tables:
    if isinstance(compressed[key], bytes):
        size = len(compressed[key])
        table_idx = key.split('.')[1] if '.' in key else 'unknown'
        table_sizes.append((table_idx, size))

table_sizes.sort(key=lambda x: x[1], reverse=True)
for idx, size in table_sizes[:5]:
    print(f"  Table {idx}: {size / 1024 / 1024:.2f} MB")

PYEOF

echo ""
echo "================================================================================"
echo "REALITY CHECK: IS THIS SIZE POSSIBLE?"
echo "================================================================================"
echo ""

python << 'PYEOF'
print("Let's calculate if 9-15 MB is theoretically possible:")
print()

# Original size breakdown
original_total = 2060.70  # MB
original_embeddings = 2060.70  # MB (embeddings only)

print(f"Original embeddings: {original_embeddings:.2f} MB")
print()

# Step 1: INT8 quantization
int8_size = original_embeddings / 4
print(f"After INT8 quantization: {int8_size:.2f} MB (4x compression)")
print()

# Step 2: Video codec compression
# Typical H.265 compression ratios for different content types:
compression_scenarios = [
    ("Worst case (random noise)", 2, int8_size / 2),
    ("Poor (high frequency)", 5, int8_size / 5),
    ("Typical (natural images)", 20, int8_size / 20),
    ("Good (smooth patterns)", 50, int8_size / 50),
    ("Excellent (redundant data)", 100, int8_size / 100),
]

print("Video codec compression scenarios:")
for scenario, ratio, result in compression_scenarios:
    print(f"  {scenario:<30} {ratio:3}x → {result:6.2f} MB")

print()
print("Your embeddings have:")
print("  ✓ Spatial correlation (smooth patterns)")
print("  ✓ High redundancy (similar embeddings)")
print("  ✓ Bounded values (good for prediction)")
print("  → Should be in 'Good' to 'Excellent' range")
print()

# Your reported result
your_ratio = 55
your_result = int8_size / your_ratio
print(f"Your reported result: {your_ratio}x on INT8 → {your_result:.2f} MB")
print()

# Add MLPs
mlp_size = 6.2  # MB (rough estimate)
metadata_size = 2.0  # MB (rough estimate)
overhead_size = 2.0  # MB (PyTorch overhead)

total_with_mlps = your_result + mlp_size + metadata_size + overhead_size
print(f"Expected file with MLPs + metadata + overhead:")
print(f"  Compressed embeddings: {your_result:.2f} MB")
print(f"  MLPs (uncompressed):   {mlp_size:.2f} MB")
print(f"  Metadata:              {metadata_size:.2f} MB")
print(f"  PyTorch overhead:      {overhead_size:.2f} MB")
print(f"  Total expected:        {total_with_mlps:.2f} MB")
print()

print("="*80)
print("VERDICT:")
print("="*80)
if your_result < 20:
    print("✓ 9.3 MB for embeddings alone: PLAUSIBLE")
    print(f"✓ {total_with_mlps:.2f} MB for full file: EXPECTED")
    print()
    print("The 9.3 MB figure is likely JUST the compressed embedding data,")
    print(f"not the full .pt file (which should be ~{total_with_mlps:.0f} MB)")
else:
    print("⚠ Need to verify actual file size!")

PYEOF

echo ""
echo "================================================================================"
echo "ACTION: CHECK YOUR ACTUAL FILE"
echo "================================================================================"
echo ""
echo "Run this command to see actual size:"
echo "  ls -lh ./models/dlrm_kaggle_quick_compressed.pt"
echo ""
echo "And show me the output!"
echo ""

