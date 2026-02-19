#!/bin/bash

# Batch test all compressed models
# This will decompress and run inference on all three quality settings

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_directory>"
    echo "Example: $0 ./codec_experiments_20260123_023329"
    exit 1
fi

EXP_DIR="$1"

if [ ! -d "$EXP_DIR" ]; then
    echo "Error: Directory $EXP_DIR does not exist!"
    exit 1
fi

echo "========================================================================"
echo "TESTING ALL COMPRESSED MODELS"
echo "========================================================================"
echo "Experiment directory: $EXP_DIR"
echo ""
echo "This will:"
echo "  1. Decompress embedding tables from compressed models"
echo "  2. Run inference on test dataset"
echo "  3. Compute AUC and compare to uncompressed baseline"
echo "  4. Compare results to CAFE baseline"
echo ""
echo "========================================================================"
echo ""

# Create results summary file
RESULTS_FILE="$EXP_DIR/compression_results.txt"
echo "=======================================================================" > "$RESULTS_FILE"
echo "COMPRESSION RESULTS SUMMARY" >> "$RESULTS_FILE"
echo "Generated: $(date)" >> "$RESULTS_FILE"
echo "=======================================================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Arrays to store results
declare -a EXPERIMENTS
declare -a QUALITIES
declare -a RATIOS
declare -a UNCOMP_AUCS
declare -a COMP_AUCS
declare -a AUC_LOSSES

# Test each compressed model
for compressed_path in "$EXP_DIR"/exp*/model_compressed_q*.pt; do
    if [ -f "$compressed_path" ]; then
        # Extract info
        exp_dir=$(dirname "$compressed_path")
        exp_name=$(basename "$exp_dir")
        compressed_name=$(basename "$compressed_path")
        quality=$(echo "$compressed_name" | grep -oP 'q\K[0-9]+')
        
        original_path="$exp_dir/model.pt"
        
        if [ ! -f "$original_path" ]; then
            echo "✗ Original model not found for $exp_name"
            continue
        fi
        
        echo ""
        echo "========================================================================"
        echo "TESTING: $exp_name (Quality=$quality)"
        echo "========================================================================"
        echo ""
        
        # Run decompression and inference with auto-architecture detection
        LOG_FILE="$exp_dir/inference_compressed_q${quality}.log"
        
        python decompress_infer_autoarch.py \
            --compressed-model "$compressed_path" \
            --original-model "$original_path" \
            --data-set=kaggle \
            --raw-data-file=./input/train.txt \
            --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
            --test-mini-batch-size=2048 \
            --test-num-workers=8 \
            2>&1 | tee "$LOG_FILE"
        
        # Extract results from log
        UNCOMP_AUC=$(grep "Original:" "$LOG_FILE" | grep "AUC" | grep -oP '[0-9]+\.[0-9]+' | head -1)
        COMP_AUC=$(grep "Decompressed:" "$LOG_FILE" | grep "AUC" | grep -oP '[0-9]+\.[0-9]+' | head -1)
        AUC_LOSS=$(grep "Loss:" "$LOG_FILE" | grep "AUC" | head -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
        COMP_RATIO=$(grep "Overall:" "$LOG_FILE" | grep -oP '[0-9]+\.[0-9]+' | tail -1)
        
        # Store results
        EXPERIMENTS+=("$exp_name")
        QUALITIES+=("$quality")
        RATIOS+=("${COMP_RATIO}x")
        UNCOMP_AUCS+=("$UNCOMP_AUC")
        COMP_AUCS+=("$COMP_AUC")
        AUC_LOSSES+=("$AUC_LOSS")
        
        echo ""
        echo "✓ Completed: $exp_name"
        echo "  Uncompressed AUC: $UNCOMP_AUC"
        echo "  Compressed AUC:   $COMP_AUC"
        echo "  AUC Loss:         $AUC_LOSS"
        echo "  Compression:      $COMP_RATIO"
    fi
done

# Generate summary report
echo ""
echo "========================================================================"
echo "FINAL RESULTS SUMMARY"
echo "========================================================================"
echo ""

# Print table header
printf "%-25s %-10s %-15s %-15s %-15s %-15s\n" \
    "Experiment" "Quality" "Compression" "Uncomp AUC" "Comp AUC" "AUC Loss"
printf "%-25s %-10s %-15s %-15s %-15s %-15s\n" \
    "-------------------------" "----------" "---------------" "---------------" "---------------" "---------------"

# Print results table
for i in "${!EXPERIMENTS[@]}"; do
    printf "%-25s %-10s %-15s %-15s %-15s %-15s\n" \
        "${EXPERIMENTS[$i]}" \
        "${QUALITIES[$i]}" \
        "${RATIOS[$i]}" \
        "${UNCOMP_AUCS[$i]}" \
        "${COMP_AUCS[$i]}" \
        "${AUC_LOSSES[$i]}"
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

# Save to file
echo "" >> "$RESULTS_FILE"
printf "%-25s %-10s %-15s %-15s %-15s %-15s\n" \
    "Experiment" "Quality" "Compression" "Uncomp AUC" "Comp AUC" "AUC Loss" >> "$RESULTS_FILE"
printf "%-25s %-10s %-15s %-15s %-15s %-15s\n" \
    "-------------------------" "----------" "---------------" "---------------" "---------------" "---------------" >> "$RESULTS_FILE"

for i in "${!EXPERIMENTS[@]}"; do
    printf "%-25s %-10s %-15s %-15s %-15s %-15s\n" \
        "${EXPERIMENTS[$i]}" \
        "${QUALITIES[$i]}" \
        "${RATIOS[$i]}" \
        "${UNCOMP_AUCS[$i]}" \
        "${COMP_AUCS[$i]}" \
        "${AUC_LOSSES[$i]}" >> "$RESULTS_FILE"
done

echo "" >> "$RESULTS_FILE"
echo "CAFE Baseline: 525x compression, 0.11441 AUC loss (14.8%)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "=======================================================================" >> "$RESULTS_FILE"

echo "========================================================================"
echo "ALL TESTS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Individual logs saved to:"
echo "  $EXP_DIR/exp*/inference_compressed_q*.log"
echo ""
echo "========================================================================"