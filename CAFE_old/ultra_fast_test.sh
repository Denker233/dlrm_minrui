#!/bin/bash

# Just 2000 batches each = 8-10 hours total

OUTPUT_DIR="./ultra_fast_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
DATA_DIR="$HOME/expr/dlrm_minrui/criteo_24days"

echo "ULTRA FAST TEST (2000 batches = ~8-10 hours)"

# Baseline
python dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--mini-batch-size=128 \
--print-freq=500 \
--test-freq=1000 \
--test-mini-batch-size=2048 \
--num-batches=2000 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="./models/baseline_ultra.pt" \
2>&1 | tee "$OUTPUT_DIR/baseline.log" &

# Sketch (run in parallel on another core)
python dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--mini-batch-size=128 \
--print-freq=500 \
--test-freq=1000 \
--test-mini-batch-size=2048 \
--sketch-flag \
--compress-rate=0.001 \
--hash-rate=0.5 \
--num-batches=2000 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="./models/sketch_ultra.pt" \
2>&1 | tee "$OUTPUT_DIR/sketch.log" &

wait

echo ""
echo "RESULTS (2000 batches each):"
echo "Baseline: $(grep 'accuracy.*auc' $OUTPUT_DIR/baseline.log | tail -1)"
echo "Sketch:   $(grep 'accuracy.*auc' $OUTPUT_DIR/sketch.log | tail -1)"

