#!/bin/bash

echo "========================================="
echo "DLRM Inference on Full Dataset"
echo "========================================="
echo "Start time: $(date)"
echo ""

ORIGINAL_MODEL="./models/dlrm_kaggle_quick.pt"
COMPRESSED_MODEL="./models/dlrm_kaggle_quick_compressed.pt"

# Check models exist
if [ ! -f "$ORIGINAL_MODEL" ]; then
    echo "ERROR: Original model not found at $ORIGINAL_MODEL"
    exit 1
fi

if [ ! -f "$COMPRESSED_MODEL" ]; then
    echo "ERROR: Compressed model not found at $COMPRESSED_MODEL"
    exit 1
fi

echo "Models found:"
ls -lh "$ORIGINAL_MODEL"
ls -lh "$COMPRESSED_MODEL"
echo ""

echo "========================================="
echo "STEP 1/3: Original Model Inference"
echo "========================================="

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
    --test-mini-batch-size=16384 \
    --test-num-workers=0 \
    --load-model="$ORIGINAL_MODEL" \
    --inference-only \
    2>&1 | tee inference_original_full.log

ORIGINAL_ACC=$(grep -oP "accuracy \K[\d.]+" inference_original_full.log | tail -1)
echo ""
echo "Original model accuracy: ${ORIGINAL_ACC}%"
echo ""

echo "========================================="
echo "STEP 2/3: Decompressing Model"
echo "========================================="

DECOMPRESSED_MODEL="./models/dlrm_kaggle_quick_decompressed.pt"

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
decompressed = {'state_dict': {}}

# Copy non-embedding layers
for key in original['state_dict'].keys():
    if 'emb_l' not in key:
        decompressed['state_dict'][key] = original['state_dict'][key]

# Decompress embeddings
compressed_tables = compressed['compressed_tables']

def decompress_table(compressed_data, metadata):
    """Decompress a single table"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(compressed_data)
        video_path = f.name
    
    try:
        width = metadata['width']
        height = metadata['height']
        
        cmd = [
            'ffmpeg', '-loglevel', 'quiet',
            '-i', video_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            'pipe:1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, check=True)
        raw_pixels = np.frombuffer(result.stdout, dtype=np.uint8)
        pixels = raw_pixels.reshape((height, width))
        
        # Untile if needed
        if metadata.get('tiled', False):
            tile_size = metadata.get('tile_size', 4)
            h, w = pixels.shape
            num_tiles_h = h // tile_size
            num_tiles_w = w // tile_size
            
            embeddings = []
            for i in range(num_tiles_h):
                for j in range(num_tiles_w):
                    tile = pixels[
                        i*tile_size:(i+1)*tile_size,
                        j*tile_size:(j+1)*tile_size
                    ]
                    embeddings.append(tile.flatten())
            
            embeddings = np.array(embeddings)
            embeddings = embeddings[:metadata['num_embeddings'], :metadata['embedding_dim']]
        else:
            embeddings = pixels.flatten()[:metadata['num_embeddings'] * metadata['embedding_dim']]
            embeddings = embeddings.reshape(metadata['num_embeddings'], metadata['embedding_dim'])
        
        # Dequantize
        scale = metadata['scale']
        zero_point = metadata['zero_point']
        dequantized = (embeddings.astype(np.float32) - zero_point) * scale
        
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
print(f"\nDecompression complete in {decomp_time:.2f} seconds")

# Save decompressed model
print("Saving decompressed model...")
torch.save(decompressed, './models/dlrm_kaggle_quick_decompressed.pt')
print("✓ Saved to ./models/dlrm_kaggle_quick_decompressed.pt")
PYTHON

if [ $? -ne 0 ]; then
    echo "ERROR: Decompression failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "STEP 3/3: Compressed Model Inference"
echo "========================================="

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
    --test-mini-batch-size=16384 \
    --test-num-workers=0 \
    --load-model="$DECOMPRESSED_MODEL" \
    --inference-only \
    2>&1 | tee inference_compressed_full.log

COMPRESSED_ACC=$(grep -oP "accuracy \K[\d.]+" inference_compressed_full.log | tail -1)
echo ""
echo "Compressed model accuracy: ${COMPRESSED_ACC}%"
echo ""

echo "========================================="
echo "RESULTS SUMMARY"
echo "========================================="
echo ""

python3 << PYTHON
import os

orig_size = os.path.getsize("$ORIGINAL_MODEL") / (1024**2)
comp_size = os.path.getsize("$COMPRESSED_MODEL") / (1024**2)
ratio = orig_size / comp_size

print("Model Sizes:")
print(f"  Original:   {orig_size:.2f} MB")
print(f"  Compressed: {comp_size:.2f} MB")
print(f"  Ratio:      {ratio:.1f}x smaller")
print()

# Parse accuracies
try:
    orig_acc = float("${ORIGINAL_ACC}")
    comp_acc = float("${COMPRESSED_ACC}")
    diff = comp_acc - orig_acc
    
    print("Accuracy:")
    print(f"  Original:   {orig_acc:.5f}%")
    print(f"  Compressed: {comp_acc:.5f}%")
    print(f"  Difference: {diff:+.5f}%")
    print()
    
    if abs(diff) < 0.1:
        print("✓ Accuracy preserved! (<0.1% difference)")
    elif abs(diff) < 0.5:
        print("✓ Minimal accuracy loss (<0.5%)")
    else:
        print("⚠ Some accuracy loss (>{abs(diff):.2f}%)")
except:
    print("Could not parse accuracies from logs")

PYTHON

echo ""
echo "========================================="
echo "COMPLETE!"
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Log files:"
echo "  - inference_original_full.log"
echo "  - inference_compressed_full.log"
echo ""

