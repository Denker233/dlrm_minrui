#!/bin/bash

# Train ONE model with CAFE+ architecture, then compress at multiple EXTREME quality levels
# This tests the limits of video codec compression

set -e

OUTPUT_BASE="./extreme_quality_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

echo "========================================================================"
echo "EXTREME QUALITY COMPRESSION TEST"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Train ONE model with CAFE+ architecture (1 epoch)"
echo "  2. Compress it at MULTIPLE extreme quality levels (Q15-Q51)"
echo "  3. Test compression limits of video codec"
echo ""
echo "Architecture:"
echo "  Bottom MLP: 13-512-256-64-16"
echo "  Top MLP:    512-256-1"
echo "  (Same as CAFE+ baseline for fair comparison)"
echo ""
echo "Output: $OUTPUT_BASE"
echo "========================================================================"
echo ""

# ============================================================================
# STEP 1: TRAIN ONE MODEL WITH CAFE+ ARCHITECTURE
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 1: TRAINING SINGLE MODEL (CAFE+ Architecture)"
echo "========================================================================"
echo ""

python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot=13-512-256-64-16 \
    --arch-mlp-top=512-256-1 \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=./input/train.txt \
    --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.005 \
    --mini-batch-size=128 \
    --nepochs=1 \
    --print-freq=5000 \
    --test-freq=50000 \
    --test-mini-batch-size=2048 \
    --num-workers=0 \
    --test-num-workers=8 \
    --mlperf-logging \
    --save-model="$OUTPUT_BASE/model_uncompressed.pt" \
    2>&1 | tee "$OUTPUT_BASE/train.log"

echo ""
echo "✓ Training complete!"

# Extract final metrics (same pattern as existing scripts)
FINAL_AUC=$(grep -oP '(?<!best )auc \K[0-9.]+' "$OUTPUT_BASE/train.log" | tail -1)
FINAL_ACC=$(grep -oP 'accuracy \K[0-9.]+(?= %)' "$OUTPUT_BASE/train.log" | tail -1)
FINAL_RECALL=$(grep -oP 'recall \K[0-9.]+' "$OUTPUT_BASE/train.log" | tail -1)
FINAL_PRECISION=$(grep -oP 'precision \K[0-9.]+' "$OUTPUT_BASE/train.log" | tail -1)
FINAL_F1=$(grep -oP 'f1 \K[0-9.]+' "$OUTPUT_BASE/train.log" | tail -1)
FINAL_AP=$(grep -oP 'ap \K[0-9.]+' "$OUTPUT_BASE/train.log" | tail -1)

echo ""
echo "Baseline Model Performance:"
echo "  Accuracy:  ${FINAL_ACC}%"
echo "  AUC:       ${FINAL_AUC}"
echo "  Recall:    ${FINAL_RECALL}"
echo "  Precision: ${FINAL_PRECISION}"
echo "  F1:        ${FINAL_F1}"
echo "  AP:        ${FINAL_AP}"
echo ""

# ============================================================================
# STEP 2: COMPRESS AT MULTIPLE EXTREME QUALITY LEVELS
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 2: COMPRESSING AT EXTREME QUALITY LEVELS"
echo "========================================================================"
echo ""

# Define quality levels to test
# Lower quality = higher compression
# H.265 CRF range: 0 (lossless) to 51 (worst quality)
QUALITIES=(15 18 20 23 25 28 30 33 35 38 40 43 45 48 51)

echo "Testing quality levels: ${QUALITIES[@]}"
echo "(Lower quality number = better quality, lower compression)"
echo "(Higher quality number = worse quality, higher compression)"
echo ""

for QUALITY in "${QUALITIES[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------"
    echo "Compressing with Quality=$QUALITY"
    echo "--------------------------------------------------------------------"
    
    OUTPUT_FILE="$OUTPUT_BASE/model_compressed_q${QUALITY}.pt"
    
    python compress_with_details.py \
        --model="$OUTPUT_BASE/model_uncompressed.pt" \
        --output="$OUTPUT_FILE" \
        --quality=$QUALITY \
        --codec=libx265 \
        2>&1 | tee "$OUTPUT_BASE/compress_q${QUALITY}.log"
    
    # Extract compression stats
    if [ -f "$OUTPUT_BASE/compress_q${QUALITY}.log" ]; then
        COMP_RATIO=$(grep "Average ratio:" "$OUTPUT_BASE/compress_q${QUALITY}.log" | grep -oP 'Average ratio:\s+\K[0-9.]+' || echo "N/A")
        ORIG_SIZE=$(du -m "$OUTPUT_BASE/model_uncompressed.pt" | cut -f1)
        COMP_SIZE=$(du -m "$OUTPUT_FILE" | cut -f1)
        
        echo ""
        echo "✓ Q${QUALITY} Compression complete:"
        echo "    Original:   ${ORIG_SIZE} MB"
        echo "    Compressed: ${COMP_SIZE} MB"
        echo "    Ratio:      ${COMP_RATIO}x"
    fi
done

echo ""
echo "========================================================================"
echo "COMPRESSION SUMMARY"
echo "========================================================================"
echo ""

printf "%-10s %-15s %-15s %-15s\n" "Quality" "Original (MB)" "Compressed (MB)" "Ratio"
printf "%-10s %-15s %-15s %-15s\n" "----------" "---------------" "---------------" "---------------"

ORIG_SIZE=$(du -m "$OUTPUT_BASE/model_uncompressed.pt" | cut -f1)

for QUALITY in "${QUALITIES[@]}"; do
    OUTPUT_FILE="$OUTPUT_BASE/model_compressed_q${QUALITY}.pt"
    if [ -f "$OUTPUT_FILE" ]; then
        COMP_SIZE=$(du -m "$OUTPUT_FILE" | cut -f1)
        COMP_RATIO=$(grep "Average ratio:" "$OUTPUT_BASE/compress_q${QUALITY}.log" | grep -oP 'Average ratio:\s+\K[0-9.]+' || echo "N/A")
        printf "%-10s %-15s %-15s %-15s\n" "Q${QUALITY}" "${ORIG_SIZE}" "${COMP_SIZE}" "${COMP_RATIO}x"
    fi
done

echo ""
echo "Baseline Model Metrics:"
echo "  Accuracy:  ${FINAL_ACC}%"
echo "  AUC:       ${FINAL_AUC}"
echo "  Recall:    ${FINAL_RECALL}"
echo "  Precision: ${FINAL_PRECISION}"
echo "  F1:        ${FINAL_F1}"
echo "  AP:        ${FINAL_AP}"
echo ""
echo "========================================================================"
echo "TRAINING AND COMPRESSION COMPLETE!"
echo "========================================================================"
echo ""
echo "Architecture: CAFE+ (13-512-256-64-16, 512-256-1)"
echo "Baseline AUC: ${FINAL_AUC}"
echo "Baseline Accuracy: ${FINAL_ACC}%"
echo ""
echo "Next step: Run test_extreme_compressed.sh to test all compressed models"
echo ""
echo "Usage:"
echo "  ./test_extreme_compressed.sh $OUTPUT_BASE"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo "  - model_uncompressed.pt          (original trained model)"
echo "  - model_compressed_q*.pt         (compressed versions)"
echo "  - compress_q*.log                (compression logs)"
echo "  - train.log                      (training log)"
echo "========================================================================"