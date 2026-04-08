#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./quick_run_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

DATA_DIR="$HOME/expr/dlrm_minrui/criteo_24days"

echo "Quick 1-epoch comparison (~5 hours total)"

# Baseline (1 epoch)
python dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.005 \
--mini-batch-size=128 \
--print-freq=5000 \
--print-time \
--test-freq=10000 \
--test-mini-batch-size=2048 \
--nepochs=1 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="$OUTPUT_DIR/baseline.pt" \
2>&1 | tee "$OUTPUT_DIR/train_baseline.log"

echo "Baseline final AUC:"
grep "accuracy.*auc" "$OUTPUT_DIR/train_baseline.log" | tail -1

# Sketch (1 epoch)
python dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.005 \
--mini-batch-size=128 \
--print-freq=5000 \
--print-time \
--test-freq=10000 \
--test-mini-batch-size=2048 \
--sketch-flag \
--compress-rate=0.001 \
--hash-rate=0.5 \
--nepochs=1 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="$OUTPUT_DIR/sketch.pt" \
2>&1 | tee "$OUTPUT_DIR/train_sketch.log"

echo "Sketch final AUC:"
grep "accuracy.*auc" "$OUTPUT_DIR/train_sketch.log" | tail -1

echo ""
echo "Comparison:"
echo "Baseline: $(du -h $OUTPUT_DIR/baseline.pt | cut -f1)"
echo "Sketch:   $(du -h $OUTPUT_DIR/sketch.pt | cut -f1)"

