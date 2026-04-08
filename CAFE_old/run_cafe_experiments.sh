#!/bin/bash

set -e

DATA_DIR="$HOME/expr/dlrm_minrui/criteo_24days"
BASE_OUTPUT="./cafe_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_OUTPUT"

# Common training parameters
COMMON_PARAMS="
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
--cat-path=$DATA_DIR/sparse \
--dense-path=$DATA_DIR/dense \
--label-path=$DATA_DIR/label \
--count-path=$DATA_DIR/processed_count.bin
"

echo "========================================================================"
echo "CAFE COMPRESSION EXPERIMENTS - Targeting 150x-200x Compression"
echo "========================================================================"
echo ""
echo "This will run 3 experiments:"
echo "  1. Conservative: ~150x compression (compress_rate=0.007, hash_rate=0.2)"
echo "  2. Moderate:     ~180x compression (compress_rate=0.0055, hash_rate=0.2)"
echo "  3. Aggressive:   ~200x compression (compress_rate=0.005, hash_rate=0.2)"
echo ""
echo "Expected time: ~3-4 hours per experiment (~10-12 hours total)"
echo "Output directory: $BASE_OUTPUT"
echo "========================================================================"
echo ""

# Function to run one experiment
run_experiment() {
    local name=$1
    local compress_rate=$2
    local hash_rate=$3
    local expected_cr=$4
    
    echo ""
    echo "========================================================================"
    echo "EXPERIMENT: $name"
    echo "========================================================================"
    echo "Parameters:"
    echo "  compress_rate = $compress_rate"
    echo "  hash_rate     = $hash_rate"
    echo "  Expected CR   = ~${expected_cr}x"
    echo ""
    echo "Breakdown of compressed capacity:"
    echo "  Total capacity:      $(awk "BEGIN {printf \"%.3f%%\", $compress_rate * 100}")"
    echo "  Learned embeddings:  $(awk "BEGIN {printf \"%.3f%%\", $compress_rate * (1-$hash_rate) * 100}") ($(awk "BEGIN {printf \"%.0f%%\", (1-$hash_rate) * 100}") of compressed)"
    echo "  Hash table:          $(awk "BEGIN {printf \"%.3f%%\", $compress_rate * $hash_rate * 100}") ($(awk "BEGIN {printf \"%.0f%%\", $hash_rate * 100}") of compressed)"
    echo "========================================================================"
    echo ""
    
    OUTPUT_DIR="$BASE_OUTPUT/${name}"
    mkdir -p "$OUTPUT_DIR"
    
    START_TIME=$(date +%s)
    
    python dlrm_s_pytorch.py \
        $COMMON_PARAMS \
        --sketch-flag \
        --compress-rate=$compress_rate \
        --hash-rate=$hash_rate \
        --save-model="$OUTPUT_DIR/model.pt" \
        2>&1 | tee "$OUTPUT_DIR/train.log"
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    # Extract final metrics
    FINAL_ACC=$(grep "accuracy.*auc" "$OUTPUT_DIR/train.log" | tail -1 | grep -oP 'accuracy \K[0-9.]+')
    FINAL_AUC=$(grep "accuracy.*auc" "$OUTPUT_DIR/train.log" | tail -1 | grep -oP 'auc \K[0-9.]+')
    
    # Calculate actual compression ratio
    if [ -f "$OUTPUT_DIR/model.pt" ]; then
        MODEL_SIZE=$(du -m "$OUTPUT_DIR/model.pt" | cut -f1)
        ACTUAL_CR=$((2062 / MODEL_SIZE))
    else
        MODEL_SIZE="N/A"
        ACTUAL_CR="N/A"
    fi
    
    # Save summary
    cat > "$OUTPUT_DIR/summary.txt" << SUMMARY
======================================================================
EXPERIMENT SUMMARY: $name
======================================================================
Parameters:
  compress_rate   = $compress_rate
  hash_rate       = $hash_rate
  Expected CR     = ~${expected_cr}x
  
Results:
  Final Accuracy  = ${FINAL_ACC}%
  Final AUC       = ${FINAL_AUC}%
  Model Size      = ${MODEL_SIZE} MB
  Actual CR       = ${ACTUAL_CR}x
  Training Time   = $((ELAPSED / 60))m $((ELAPSED % 60))s

Capacity Breakdown:
  Total capacity:      $(awk "BEGIN {printf \"%.3f%%\", $compress_rate * 100}")
  Learned embeddings:  $(awk "BEGIN {printf \"%.3f%%\", $compress_rate * (1-$hash_rate) * 100}") ($(awk "BEGIN {printf \"%.0f%%\", (1-$hash_rate) * 100}") of compressed)
  Hash table:          $(awk "BEGIN {printf \"%.3f%%\", $compress_rate * $hash_rate * 100}") ($(awk "BEGIN {printf \"%.0f%%\", $hash_rate * 100}") of compressed)
======================================================================
SUMMARY
    
    echo ""
    echo "✓ Experiment $name completed!"
    echo "  Time: $((ELAPSED / 60))m $((ELAPSED % 60))s"
    echo "  Final AUC: ${FINAL_AUC}%"
    echo "  Actual CR: ${ACTUAL_CR}x"
    echo ""
}

# Run all three experiments
run_experiment "exp1_150x_conservative" 0.007 0.2 150
run_experiment "exp2_180x_moderate" 0.0055 0.2 180
run_experiment "exp3_200x_aggressive" 0.005 0.2 200

# Generate comparison report
echo ""
echo "========================================================================"
echo "GENERATING COMPARISON REPORT"
echo "========================================================================"

cat > "$BASE_OUTPUT/comparison_report.txt" << 'REPORT_HEADER'
======================================================================
CAFE COMPRESSION EXPERIMENTS - COMPARISON REPORT
======================================================================

REPORT_HEADER

for exp in exp1_150x_conservative exp2_180x_moderate exp3_200x_aggressive; do
    if [ -f "$BASE_OUTPUT/$exp/summary.txt" ]; then
        cat "$BASE_OUTPUT/$exp/summary.txt" >> "$BASE_OUTPUT/comparison_report.txt"
        echo "" >> "$BASE_OUTPUT/comparison_report.txt"
    fi
done

cat >> "$BASE_OUTPUT/comparison_report.txt" << 'REPORT_FOOTER'
======================================================================
KEY FINDINGS
======================================================================

Compare these results with:
1. Your baseline (uncompressed): 77.268% AUC
2. Your previous CAFE run (525x): 65.827% AUC (11.4% loss)
3. Your video codec compression: 148x-221x with 0.052% loss

Expected results:
- At ~180x compression: AUC loss should be ~1-2% (target: 76.0-76.5%)
- Much better than the 11.4% loss from compress_rate=0.001

======================================================================
REPORT_FOOTER

cat "$BASE_OUTPUT/comparison_report.txt"

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo "Output directory: $BASE_OUTPUT"
echo "Comparison report: $BASE_OUTPUT/comparison_report.txt"
echo "========================================================================"