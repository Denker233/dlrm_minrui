#!/bin/bash

set -e

DATA_DIR="$HOME/expr/dlrm_minrui/criteo_24days"
OUTPUT_DIR="./cafe_corrected_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "CAFE with CORRECTED PARAMETERS (in CAFE_old - working version)"
echo "========================================================================"
echo ""
echo "Key changes from previous runs:"
echo "  ✓ sketch-alpha: 1.0 → 1.0000005 (enables decay)"
echo "  ✓ hash-rate: 0.2 → 0.35 (better balance for ~180x compression)"
echo ""
echo "Target: ~180x compression with proper CAFE parameters"
echo "========================================================================"
echo ""

START_TIME=$(date +%s)

python dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot=13-512-256-64-16 \
--arch-mlp-top=512-256-1 \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.005 \
--mini-batch-size=128 \
--print-freq=5000 \
--test-freq=50000 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--nepochs=1 \
--sketch-flag \
--compress-rate=0.006 \
--hash-rate=0.35 \
--sketch-alpha=1.0000005 \
--sketch-threshold=1 \
--adjust-threshold=1 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="$OUTPUT_DIR/model.pt" \
2>&1 | tee "$OUTPUT_DIR/train.log"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Extract metrics
FINAL_ACC=$(grep "accuracy.*auc" "$OUTPUT_DIR/train.log" | tail -1 | grep -oP 'accuracy \K[0-9.]+')
FINAL_AUC=$(grep "accuracy.*auc" "$OUTPUT_DIR/train.log" | tail -1 | grep -oP 'auc \K[0-9.]+')
MODEL_SIZE=$(du -m "$OUTPUT_DIR/model.pt" | cut -f1)
ACTUAL_CR=$((2062 / MODEL_SIZE))

cat > "$OUTPUT_DIR/summary.txt" << SUMMARY
======================================================================
CAFE CORRECTED PARAMETERS RESULTS
======================================================================
Configuration:
  compress_rate      = 0.006
  hash_rate          = 0.35 (changed from 0.2)
  sketch_alpha       = 1.0000005 (changed from 1.0)
  sketch_threshold   = 1
  adjust_threshold   = 1

Results:
  Final Accuracy     = ${FINAL_ACC}%
  Final AUC          = ${FINAL_AUC}%
  Model Size         = ${MODEL_SIZE} MB
  Actual CR          = ${ACTUAL_CR}x
  Training Time      = $((ELAPSED / 60))m $((ELAPSED % 60))s

Comparison with Previous Runs (from CAFE_old):
  Previous (CR=121x, alpha=1.0, hash=0.2):  72.869% AUC (4.4% loss)
  Previous (CR=147x, alpha=1.0, hash=0.2):  70.520% AUC (6.7% loss)
  Previous (CR=158x, alpha=1.0, hash=0.2):  72.426% AUC (4.8% loss)
  
  This run (CR~167x, alpha=1.0000005, hash=0.35): ${FINAL_AUC}% AUC

Expected: ~2-3% AUC loss with corrected parameters
======================================================================
SUMMARY

cat "$OUTPUT_DIR/summary.txt"

echo ""
echo "✓ Training complete!"
echo "  Output: $OUTPUT_DIR"
