# #!/bin/bash

# set -e

# BASE_OUTPUT="./codec_experiments_$(date +%Y%m%d_%H%M%S)"
# mkdir -p "$BASE_OUTPUT"

# echo "========================================================================"
# echo "CODEC COMPRESSION EXPERIMENTS"
# echo "========================================================================"
# echo ""
# echo "Will run 3 experiments:"
# echo "  1. Quality=18  (~150x compression, high quality)"
# echo "  2. Quality=23  (~180x compression, balanced)"  
# echo "  3. Quality=28  (~210x compression, high compression)"
# echo ""
# echo "Note: Using libx265 codec with CRF quality (18-28 range)"
# echo "Output: $BASE_OUTPUT"
# echo "========================================================================"
# echo ""

# # Function to run one experiment
# run_experiment() {
#     local name=$1
#     local quality=$2
    
#     echo ""
#     echo "========================================================================"
#     echo "EXPERIMENT: $name (Quality=$quality)"
#     echo "========================================================================"
    
#     OUTPUT_DIR="$BASE_OUTPUT/${name}"
#     mkdir -p "$OUTPUT_DIR"
    
#     # STEP 1: Train DLRM (your existing script)
#     echo ""
#     echo "Training DLRM..."
#     python dlrm_s_pytorch.py \
#         --arch-sparse-feature-size=16 \
#         --arch-mlp-bot=13-512-256-64-16 \
#         --arch-mlp-top=512-256-1 \
#         --data-generation=dataset \
#         --data-set=kaggle \
#         --raw-data-file=./input/train.txt \
#         --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#         --loss-function=bce \
#         --round-targets=True \
#         --learning-rate=0.005 \
#         --mini-batch-size=128 \
#         --nepochs=1 \
#         --print-freq=5000 \
#         --test-freq=50000 \
#         --test-mini-batch-size=2048 \
#         --num-workers=0 \
#         --test-num-workers=8 \
#         --save-model="$OUTPUT_DIR/model.pt" \
#         2>&1 | tee "$OUTPUT_DIR/train.log"
    
#     # Extract metrics
#     FINAL_ACC=$(grep "accuracy" "$OUTPUT_DIR/train.log" | grep -oP 'accuracy \K[0-9.]+' | tail -1)
#     FINAL_AUC=$(grep "auc" "$OUTPUT_DIR/train.log" | grep -oP 'auc \K[0-9.]+' | tail -1)
    
#     echo ""
#     echo "Training complete!"
#     echo "  Accuracy: ${FINAL_ACC}%"
#     echo "  AUC: ${FINAL_AUC}"
    
#     # STEP 2: Compress model (using your existing script)
#     echo ""
#     echo "Compressing model..."
#     python compress_with_details.py \
#         --model="$OUTPUT_DIR/model.pt" \
#         --output="$OUTPUT_DIR/model_compressed_q${quality}.pt" \
#         --quality=$quality \
#         --codec=libx265 \
#         2>&1 | tee "$OUTPUT_DIR/compress.log"
    
#     # Get compression ratio from their verbose output
#     COMP_RATIO=$(grep "Average ratio:" "$OUTPUT_DIR/compress.log" | grep -oP 'Average ratio:\s+\K[0-9.]+')
    
#     # Get file sizes
#     ORIG_SIZE=$(du -m "$OUTPUT_DIR/model.pt" | cut -f1)
#     COMP_SIZE=$(du -m "$OUTPUT_DIR/model_compressed_q${quality}.pt" | cut -f1)
    
#     echo ""
#     echo "Compression complete!"
#     echo "  Original: ${ORIG_SIZE} MB"
#     echo "  Compressed: ${COMP_SIZE} MB"
#     echo "  Ratio: ${COMP_RATIO}x"
    
#     # Save summary
#     cat > "$OUTPUT_DIR/summary.txt" << SUMMARY
# ======================================================================
# EXPERIMENT: $name
# ======================================================================
# Quality: $quality

# Results:
#   Accuracy:       ${FINAL_ACC}%
#   AUC:            ${FINAL_AUC}

# Compression:
#   Original size:  ${ORIG_SIZE} MB
#   Compressed:     ${COMP_SIZE} MB
#   Ratio:          ${COMP_RATIO}x
# ======================================================================
# SUMMARY
    
#     echo ""
#     echo "✓ Done!"
# }

# # Run experiments
# run_experiment "exp1_q18_highquality" 18
# run_experiment "exp2_q23_balanced" 23
# run_experiment "exp3_q28_highcompress" 28

# # Generate report
# echo ""
# echo "========================================================================"
# echo "COMPARISON REPORT"
# echo "========================================================================"

# for exp in exp1_q18_highquality exp2_q23_balanced exp3_q28_highcompress; do
#     if [ -f "$BASE_OUTPUT/$exp/summary.txt" ]; then
#         cat "$BASE_OUTPUT/$exp/summary.txt"
#         echo ""
#     fi
# done

# echo "CAFE Baseline:"
# echo "  Uncompressed:  77.268% AUC"
# echo "  CAFE 525x:     65.827% AUC (11.4% loss)"
# echo ""
# echo "Results saved to: $BASE_OUTPUT"
# echo "========================================================================"

#!/bin/bash

set -e

BASE_OUTPUT="./codec_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_OUTPUT"

echo "========================================================================"
echo "CODEC COMPRESSION EXPERIMENTS"
echo "========================================================================"
echo ""
echo "Will run 3 experiments:"
echo "  1. Quality=18  (~150x compression, high quality)"
echo "  2. Quality=23  (~180x compression, balanced)"  
echo "  3. Quality=28  (~210x compression, high compression)"
echo ""
echo "Note: Using libx265 codec with CRF quality (18-28 range)"
echo "Output: $BASE_OUTPUT"
echo "========================================================================"
echo ""

# Function to run one experiment
run_experiment() {
    local name=$1
    local quality=$2
    
    echo ""
    echo "========================================================================"
    echo "EXPERIMENT: $name (Quality=$quality)"
    echo "========================================================================"
    
    OUTPUT_DIR="$BASE_OUTPUT/${name}"
    mkdir -p "$OUTPUT_DIR"
    
    # STEP 1: Train DLRM
    echo ""
    echo "Training DLRM..."
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
        --save-model="$OUTPUT_DIR/model.pt" \
        2>&1 | tee "$OUTPUT_DIR/train.log"
    
    # Extract metrics - updated patterns for mlperf format
    # Format: "auc 0.1234, best auc 0.1234, accuracy 12.34 %, best accuracy 12.34 %"
    FINAL_AUC=$(grep -oP '(?<!best )auc \K[0-9.]+' "$OUTPUT_DIR/train.log" | tail -1)
    FINAL_ACC=$(grep -oP 'accuracy \K[0-9.]+(?= %)' "$OUTPUT_DIR/train.log" | tail -1)
    
    # Also extract other metrics
    FINAL_RECALL=$(grep -oP 'recall \K[0-9.]+' "$OUTPUT_DIR/train.log" | tail -1)
    FINAL_PRECISION=$(grep -oP 'precision \K[0-9.]+' "$OUTPUT_DIR/train.log" | tail -1)
    FINAL_F1=$(grep -oP 'f1 \K[0-9.]+' "$OUTPUT_DIR/train.log" | tail -1)
    FINAL_AP=$(grep -oP 'ap \K[0-9.]+' "$OUTPUT_DIR/train.log" | tail -1)
    
    echo ""
    echo "Training complete!"
    echo "  Accuracy:  ${FINAL_ACC}%"
    echo "  AUC:       ${FINAL_AUC}"
    echo "  Recall:    ${FINAL_RECALL}"
    echo "  Precision: ${FINAL_PRECISION}"
    echo "  F1:        ${FINAL_F1}"
    echo "  AP:        ${FINAL_AP}"
    
    # STEP 2: Compress model
    echo ""
    echo "Compressing model..."
    python compress_with_details.py \
        --model="$OUTPUT_DIR/model.pt" \
        --output="$OUTPUT_DIR/model_compressed_q${quality}.pt" \
        --quality=$quality \
        --codec=libx265 \
        2>&1 | tee "$OUTPUT_DIR/compress.log"
    
    # Get compression ratio from their verbose output
    COMP_RATIO=$(grep "Average ratio:" "$OUTPUT_DIR/compress.log" | grep -oP 'Average ratio:\s+\K[0-9.]+')
    
    # Get file sizes
    ORIG_SIZE=$(du -m "$OUTPUT_DIR/model.pt" | cut -f1)
    COMP_SIZE=$(du -m "$OUTPUT_DIR/model_compressed_q${quality}.pt" | cut -f1)
    
    echo ""
    echo "Compression complete!"
    echo "  Original: ${ORIG_SIZE} MB"
    echo "  Compressed: ${COMP_SIZE} MB"
    echo "  Ratio: ${COMP_RATIO}x"
    
    # Save detailed summary
    cat > "$OUTPUT_DIR/summary.txt" << SUMMARY
======================================================================
EXPERIMENT: $name
======================================================================
Quality: $quality

Training Results:
  Accuracy:       ${FINAL_ACC}%
  AUC:            ${FINAL_AUC}
  Recall:         ${FINAL_RECALL}
  Precision:      ${FINAL_PRECISION}
  F1 Score:       ${FINAL_F1}
  Average Precision: ${FINAL_AP}

Compression:
  Original size:  ${ORIG_SIZE} MB
  Compressed:     ${COMP_SIZE} MB
  Ratio:          ${COMP_RATIO}x

Accuracy Loss:
  N/A (baseline)
======================================================================
SUMMARY
    
    echo ""
    echo "✓ Done!"
}

# Run experiments
run_experiment "exp1_q18_highquality" 18
run_experiment "exp2_q23_balanced" 23
run_experiment "exp3_q28_highcompress" 28

# Generate comparison report
echo ""
echo "========================================================================"
echo "COMPARISON REPORT"
echo "========================================================================"
echo ""

# Create a nice comparison table
printf "%-20s %-12s %-12s %-12s %-15s\n" "Experiment" "Quality" "Compression" "Accuracy" "AUC"
printf "%-20s %-12s %-12s %-12s %-15s\n" "--------------------" "------------" "------------" "------------" "---------------"

for exp in exp1_q18_highquality exp2_q23_balanced exp3_q28_highcompress; do
    if [ -f "$BASE_OUTPUT/$exp/summary.txt" ]; then
        QUALITY=$(grep "Quality:" "$BASE_OUTPUT/$exp/summary.txt" | awk '{print $2}')
        RATIO=$(grep "Ratio:" "$BASE_OUTPUT/$exp/summary.txt" | awk '{print $2}')
        ACC=$(grep "Accuracy:" "$BASE_OUTPUT/$exp/summary.txt" | awk '{print $2}')
        AUC=$(grep "AUC:" "$BASE_OUTPUT/$exp/summary.txt" | awk '{print $2}')
        
        printf "%-20s %-12s %-12s %-12s %-15s\n" "$exp" "$QUALITY" "${RATIO}" "${ACC}" "${AUC}"
    fi
done

echo ""
echo "========================================================================"
echo "BASELINE COMPARISON"
echo "========================================================================"
echo ""
echo "CAFE (Reference):"
echo "  Uncompressed:  77.268% AUC"
echo "  CAFE 525x:     65.827% AUC (11.441 pp loss, 14.8% relative loss)"
echo ""
echo "Your Results Above ^"
echo ""
echo "Detailed summaries:"
echo "========================================================================"
echo ""

for exp in exp1_q18_highquality exp2_q23_balanced exp3_q28_highcompress; do
    if [ -f "$BASE_OUTPUT/$exp/summary.txt" ]; then
        cat "$BASE_OUTPUT/$exp/summary.txt"
        echo ""
    fi
done

echo "Results saved to: $BASE_OUTPUT"
echo "========================================================================"