#!/bin/bash

DATA_DIR="$HOME/expr/dlrm_minrui/criteo_24days"

echo "========================================================================"
echo "INFERENCE SPEED TEST"
echo "========================================================================"
echo ""

# First train a small model
echo ">>> Training small model for inference test..."

python dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot=13-512-256-64-16 \
--arch-mlp-top=512-256-1 \
--data-generation=dataset \
--data-set=kaggle \
--learning-rate=0.005 \
--mini-batch-size=128 \
--num-batches=10000 \
--sketch-flag \
--compress-rate=0.007 \
--hash-rate=0.2 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="./inference_test_model.pt" \
2>&1 > train_for_inference.log

echo "✓ Model trained"
echo ""

# Now test inference speed
echo ">>> Testing inference speed..."
echo ""

python dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot=13-512-256-64-16 \
--arch-mlp-top=512-256-1 \
--data-generation=dataset \
--data-set=kaggle \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--sketch-flag \
--compress-rate=0.007 \
--hash-rate=0.2 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--load-model="./inference_test_model.pt" \
--inference-only \
2>&1 | tee inference_speed.log

echo ""
echo "========================================================================"
echo "INFERENCE RESULTS"
echo "========================================================================"
echo ""

# Extract timing
grep "SKETCH.*TIMING" inference_speed.log
grep "Testing" inference_speed.log

echo "========================================================================"
