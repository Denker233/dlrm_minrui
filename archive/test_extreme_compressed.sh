#!/bin/bash

# Test all extreme quality compressed models
# Decompresses and runs inference on each quality level

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_directory>"
    echo "Example: $0 ./extreme_quality_test_20260123_180000"
    exit 1
fi

EXPERIMENT_DIR=$1

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Directory $EXPERIMENT_DIR does not exist"
    exit 1
fi

echo "========================================================================"
echo "TESTING EXTREME QUALITY COMPRESSED MODELS"
echo "========================================================================"
echo ""
echo "Experiment directory: $EXPERIMENT_DIR"
echo ""
echo "This will:"
echo "  1. Decompress embedding tables from each compressed model"
echo "  2. Run inference on test dataset"
echo "  3. Compute AUC and accuracy for each quality level"
echo "  4. Generate comparison report"
echo ""
echo "========================================================================"
echo ""

# Find all compressed models
COMPRESSED_MODELS=($(ls $EXPERIMENT_DIR/model_compressed_q*.pt 2>/dev/null | sort -V))

if [ ${#COMPRESSED_MODELS[@]} -eq 0 ]; then
    echo "Error: No compressed models found in $EXPERIMENT_DIR"
    exit 1
fi

echo "Found ${#COMPRESSED_MODELS[@]} compressed models"
echo ""

# Create results file
RESULTS_FILE="$EXPERIMENT_DIR/extreme_quality_results.txt"
SUMMARY_FILE="$EXPERIMENT_DIR/extreme_quality_summary.csv"

# Initialize summary CSV
echo "Quality,Compression_Ratio,Decomp_Time_s,Orig_AUC,Comp_AUC,AUC_Loss,AUC_Loss_Pct,Orig_Acc,Comp_Acc,Acc_Loss" > "$SUMMARY_FILE"

# Get original model metrics
ORIGINAL_MODEL="$EXPERIMENT_DIR/model_uncompressed.pt"

if [ ! -f "$ORIGINAL_MODEL" ]; then
    echo "Error: Original model not found at $ORIGINAL_MODEL"
    exit 1
fi

# Test each compressed model
for COMPRESSED_MODEL in "${COMPRESSED_MODELS[@]}"; do
    # Extract quality from filename
    QUALITY=$(basename "$COMPRESSED_MODEL" | grep -oP 'q\K[0-9]+')
    
    echo ""
    echo "========================================================================"
    echo "TESTING: Quality=$QUALITY"
    echo "========================================================================"
    echo ""
    
    LOG_FILE="$EXPERIMENT_DIR/inference_q${QUALITY}.log"
    
    python decompress_infer_autoarch.py \
        --compressed-model="$COMPRESSED_MODEL" \
        --original-model="$ORIGINAL_MODEL" \
        --data-set=kaggle \
        --raw-data-file=./input/train.txt \
        --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
        --test-mini-batch-size=2048 \
        --test-num-workers=8 \
        2>&1 | tee "$LOG_FILE"
    
    # Extract results from log (same patterns as existing scripts)
    ORIG_AUC=$(grep "Original:" "$LOG_FILE" | grep "AUC" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    COMP_AUC=$(grep "Decompressed:" "$LOG_FILE" | grep "AUC" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    AUC_LOSS=$(grep "Loss:" "$LOG_FILE" | grep "AUC" | head -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
    AUC_LOSS_PCT=$(grep "Loss:" "$LOG_FILE" | grep "AUC" | head -1 | grep -oP '\([0-9]+\.[0-9]+%\)' | grep -oP '[0-9]+\.[0-9]+')
    
    ORIG_ACC=$(grep "Original:" "$LOG_FILE" | grep "Accuracy" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    COMP_ACC=$(grep "Decompressed:" "$LOG_FILE" | grep "Accuracy" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    ACC_LOSS=$(grep "Loss:" "$LOG_FILE" | grep "Accuracy" | head -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    
    ORIG_RECALL=$(grep "Original:" "$LOG_FILE" | grep "Recall" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    COMP_RECALL=$(grep "Decompressed:" "$LOG_FILE" | grep "Recall" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    
    ORIG_PRECISION=$(grep "Original:" "$LOG_FILE" | grep "Precision" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    COMP_PRECISION=$(grep "Decompressed:" "$LOG_FILE" | grep "Precision" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    
    COMP_RATIO=$(grep "Overall:" "$LOG_FILE" | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    DECOMP_TIME=$(grep "Time:" "$LOG_FILE" | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    
    # Add to summary CSV
    echo "${QUALITY},${COMP_RATIO},${DECOMP_TIME},${ORIG_AUC},${COMP_AUC},${AUC_LOSS},${AUC_LOSS_PCT},${ORIG_ACC},${COMP_ACC},${ACC_LOSS}" >> "$SUMMARY_FILE"
    
    echo ""
    echo "✓ Completed Q${QUALITY}"
    echo "  Compression: ${COMP_RATIO}x"
    echo "  AUC Loss:    ${AUC_LOSS} (${AUC_LOSS_PCT}%)"
    echo "  Acc Loss:    ${ACC_LOSS} pp"
    echo ""
done

# ============================================================================
# GENERATE FINAL REPORT
# ============================================================================

echo ""
echo "========================================================================"
echo "EXTREME QUALITY COMPRESSION RESULTS"
echo "========================================================================"
echo ""

{
    echo "========================================================================"
    echo "EXTREME QUALITY COMPRESSION RESULTS"
    echo "========================================================================"
    echo ""
    echo "Experiment: $EXPERIMENT_DIR"
    echo "Architecture: bot=13-512-256-64-16, top=512-256-1 (CAFE+)"
    echo "Dataset: Criteo Kaggle"
    echo "Epochs: 1"
    echo ""
    echo "========================================================================"
    echo "COMPRESSION-ACCURACY TRADE-OFF"
    echo "========================================================================"
    echo ""
    
    printf "%-10s %-15s %-15s %-15s %-15s\n" \
        "Quality" "Compression" "Orig AUC" "Comp AUC" "AUC Loss"
    printf "%-10s %-15s %-15s %-15s %-15s\n" \
        "----------" "---------------" "---------------" "---------------" "---------------"
    
    # Read and display results
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r quality ratio time orig_auc comp_auc loss loss_pct orig_acc comp_acc acc_loss; do
        printf "%-10s %-15s %-15s %-15s %-15s\n" \
            "Q${quality}" "${ratio}x" "${orig_auc}" "${comp_auc}" "${loss} (${loss_pct}%)"
    done
    
    echo ""
    echo "========================================================================"
    echo "ACCURACY PRESERVATION"
    echo "========================================================================"
    echo ""
    
    printf "%-10s %-15s %-15s %-15s\n" \
        "Quality" "Orig Acc" "Comp Acc" "Acc Loss"
    printf "%-10s %-15s %-15s %-15s\n" \
        "----------" "---------------" "---------------" "---------------"
    
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r quality ratio time orig_auc comp_auc loss loss_pct orig_acc comp_acc acc_loss; do
        printf "%-10s %-15s %-15s %-15s\n" \
            "Q${quality}" "${orig_acc}%" "${comp_acc}%" "${acc_loss} pp"
    done
    
    echo ""
    echo "========================================================================"
    echo "DECOMPRESSION TIME"
    echo "========================================================================"
    echo ""
    
    printf "%-10s %-15s %-15s\n" \
        "Quality" "Compression" "Decomp Time"
    printf "%-10s %-15s %-15s\n" \
        "----------" "---------------" "---------------"
    
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r quality ratio time orig_auc comp_auc loss loss_pct orig_acc comp_acc acc_loss; do
        printf "%-10s %-15s %-15s\n" \
            "Q${quality}" "${ratio}x" "${time}s"
    done
    
    echo ""
    echo "========================================================================"
    echo "KEY FINDINGS"
    echo "========================================================================"
    echo ""
    
    # Find best compression with acceptable loss (<5%)
    echo "Models with AUC loss < 5% (acceptable for production):"
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r quality ratio time orig_auc comp_auc loss loss_pct orig_acc comp_acc acc_loss; do
        if (( $(echo "$loss_pct < 5" | bc -l) )); then
            echo "  Q${quality}: ${ratio}x compression, ${loss_pct}% loss"
        fi
    done
    
    echo ""
    echo "Models with AUC loss 5-10% (acceptable for some use cases):"
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r quality ratio time orig_auc comp_auc loss loss_pct orig_acc comp_acc acc_loss; do
        if (( $(echo "$loss_pct >= 5 && $loss_pct < 10" | bc -l) )); then
            echo "  Q${quality}: ${ratio}x compression, ${loss_pct}% loss"
        fi
    done
    
    echo ""
    echo "Models with AUC loss > 10% (probably too lossy):"
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r quality ratio time orig_auc comp_auc loss loss_pct orig_acc comp_acc acc_loss; do
        if (( $(echo "$loss_pct >= 10" | bc -l) )); then
            echo "  Q${quality}: ${ratio}x compression, ${loss_pct}% loss"
        fi
    done
    
    echo ""
    echo "========================================================================"
    echo "COMPARISON TO CAFE BASELINE"
    echo "========================================================================"
    echo ""
    echo "CAFE (Reference Implementation):"
    echo "  Compression:     525x"
    echo "  Uncompressed:    0.77268 AUC"
    echo "  Compressed:      0.65827 AUC"
    echo "  Loss:            0.11441 (14.8%)"
    echo ""
    echo "Your Results (see table above)"
    echo ""
    echo "========================================================================"
    echo ""
    
} | tee "$RESULTS_FILE"

echo ""
echo "========================================================================"
echo "TESTING COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - Summary table:     $RESULTS_FILE"
echo "  - CSV data:          $SUMMARY_FILE"
echo "  - Individual logs:   $EXPERIMENT_DIR/inference_q*.log"
echo ""
echo "To visualize results:"
echo "  python plot_extreme_quality.py $SUMMARY_FILE"
echo ""
echo "========================================================================"