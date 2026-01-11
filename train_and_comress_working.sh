#!/bin/bash

echo "========================================="
echo "DLRM Quick Training + Compression"
echo "========================================="
echo "Start time: $(date)"
echo ""

MODEL="./models/dlrm_kaggle_quick.pt"
COMPRESSED="./models/dlrm_kaggle_quick_compressed.pt"

mkdir -p ./models

echo "========================================="
echo "STEP 1/2: Quick Training (5000 batches)"
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
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --test-mini-batch-size=2048 \
    --print-freq=512 \
    --num-batches=5000 \
    --test-freq=5000 \
    --nepochs=1 \
    --num-workers=0 \
    --test-num-workers=0 \
    --save-model="$MODEL"

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

# Check if model was actually saved
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file was not created at $MODEL"
    echo "Training completed but save failed"
    exit 1
fi

echo ""
echo "Training complete!"
ls -lh "$MODEL"
echo ""

echo "========================================="
echo "STEP 2/2: Compressing Model"
echo "========================================="

python compress_with_details.py \
    --model "$MODEL" \
    --output "$COMPRESSED" \
    --codec hevc_qsv \
    --quality 20 \
    --quantization asymmetric

if [ $? -ne 0 ]; then
    echo "ERROR: Compression failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "COMPLETE!"
echo "========================================="
echo "End time: $(date)"
echo ""
ls -lh "$MODEL"
ls -lh "$COMPRESSED"

python3 << 'PYTHON'
import os
if os.path.exists("./models/dlrm_kaggle_quick.pt") and os.path.exists("./models/dlrm_kaggle_quick_compressed.pt"):
    orig_mb = os.path.getsize("./models/dlrm_kaggle_quick.pt") / (1024**2)
    comp_mb = os.path.getsize("./models/dlrm_kaggle_quick_compressed.pt") / (1024**2)
    ratio = orig_mb / comp_mb
    print(f"\nOriginal: {orig_mb:.2f} MB → Compressed: {comp_mb:.2f} MB ({ratio:.2f}x)")
PYTHON

echo ""
echo "✓ Done!"
