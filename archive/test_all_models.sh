#!/bin/bash

set -e

RESULTS_DIR="./codec_experiments_20260120_221951"

# DLRM Configuration
ARCH_SPARSE_FEATURE_SIZE=16
ARCH_MLP_BOT="13-512-256-64-16"
ARCH_MLP_TOP="512-256-1"
BATCH_SIZE=2048
DATA_FILE="./input/kaggleAdDisplayChallenge_processed.npz"
RAW_DATA_FILE="./input/train.txt"

echo "========================================================================"
echo "CODEC COMPRESSION EXPERIMENTS - DECOMPRESSION & INFERENCE"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Embedding dim:     $ARCH_SPARSE_FEATURE_SIZE"
echo "  MLP architecture:  $ARCH_MLP_BOT → $ARCH_MLP_TOP"
echo "  Batch size:        $BATCH_SIZE"
echo ""
echo "========================================================================"
echo ""

# Function to clear cache
clear_cache() {
    echo "Clearing system cache..."
    sync
    if sudo -n true 2>/dev/null; then
        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
        echo "  ✓ Cache cleared (full)"
    else
        echo "  ⚠ Partial cache clear (run 'sudo -v' first for full clear)"
    fi
    sleep 2
}

# Function to test one model
test_model() {
    local exp_name=$1
    local quality=$2
    
    echo ""
    echo "========================================================================"
    echo "EXPERIMENT: $exp_name (Quality=$quality)"
    echo "========================================================================"
    
    local exp_dir="$RESULTS_DIR/$exp_name"
    local compressed_model="$exp_dir/model_compressed_q${quality}.pt"
    local decompressed_model="$exp_dir/model_decompressed_q${quality}.pt"
    
    if [ ! -f "$compressed_model" ]; then
        echo "ERROR: Compressed model not found: $compressed_model"
        return 1
    fi
    
    # Step 1: Decompress
    echo ""
    echo "[STEP 1/3] Decompression"
    echo "────────────────────────────────────────────────────────────────────────"
    
    python3 decompress_model.py \
        --compressed="$compressed_model" \
        --output="$decompressed_model" \
        2>&1 | tee "$exp_dir/decompress.log"
    
    if [ ! -f "$decompressed_model" ]; then
        echo "ERROR: Decompression failed!"
        return 1
    fi
    
    # Step 2: Clear cache
    echo ""
    echo "[STEP 2/3] Cache Clearing"
    echo "────────────────────────────────────────────────────────────────────────"
    clear_cache
    
    # Step 3: Inference
    echo ""
    echo "[STEP 3/3] Inference"
    echo "────────────────────────────────────────────────────────────────────────"
    
    python3 dlrm_s_pytorch.py \
        --arch-sparse-feature-size=$ARCH_SPARSE_FEATURE_SIZE \
        --arch-mlp-bot=$ARCH_MLP_BOT \
        --arch-mlp-top=$ARCH_MLP_TOP \
        --data-generation=dataset \
        --data-set=kaggle \
        --raw-data-file=$RAW_DATA_FILE \
        --processed-data-file=$DATA_FILE \
        --loss-function=bce \
        --round-targets=True \
        --test-mini-batch-size=$BATCH_SIZE \
        --num-workers=0 \
        --test-num-workers=8 \
        --load-model="$decompressed_model" \
        --inference-only \
        2>&1 | tee "$exp_dir/inference.log"
    
    # Extract metrics
    local accuracy=$(grep "accuracy" "$exp_dir/inference.log" | grep -oP 'accuracy \K[0-9.]+' | tail -1 || echo "N/A")
    local auc=$(grep "auc" "$exp_dir/inference.log" | grep -oP 'auc \K[0-9.]+' | tail -1 || echo "N/A")
    local comp_ratio=$(grep "Average ratio:" "$exp_dir/compress.log" | grep -oP 'Average ratio:\s+\K[0-9.]+' || echo "N/A")
    
    echo ""
    echo "✓ Results:"
    echo "  Accuracy: ${accuracy}%"
    [ "$auc" != "N/A" ] && echo "  AUC:      $auc"
    
    # Calculate loss
    local baseline="78.523"
    local acc_loss="N/A"
    
    if [ "$accuracy" != "N/A" ]; then
        acc_loss=$(python3 -c "print(f'{$baseline - $accuracy:.3f}')" 2>/dev/null || echo "N/A")
    fi
    
    # Save summary
    cat > "$exp_dir/results_summary.txt" << SUMMARY
Experiment: $exp_name
Quality: $quality
Compression Ratio: ${comp_ratio}x
Baseline Accuracy: ${baseline}%
Inference Accuracy: ${accuracy}%
Accuracy Loss: ${acc_loss}%
AUC: $auc
SUMMARY
    
    echo "✓✓ COMPLETE: $exp_name ✓✓"
}

# Test all three models
test_model "exp1_q18_highquality" 18
test_model "exp2_q23_balanced" 23
test_model "exp3_q28_highcompress" 28

# Final report
echo ""
echo "========================================================================"
echo "FINAL RESULTS"
echo "========================================================================"
echo ""

printf "%-25s | %-3s | %-10s | %-12s | %-12s | %-10s\n" \
    "Experiment" "Q" "Ratio" "Baseline" "Compressed" "Loss"
echo "─────────────────────────────────────────────────────────────────────────────────"

for exp_info in "exp1_q18_highquality:18" "exp2_q23_balanced:23" "exp3_q28_highcompress:28"; do
    IFS=':' read -r exp_name quality <<< "$exp_info"
    
    if [ -f "$RESULTS_DIR/$exp_name/results_summary.txt" ]; then
        ratio=$(grep "Compression Ratio:" "$RESULTS_DIR/$exp_name/results_summary.txt" | awk '{print $3}')
        baseline=$(grep "Baseline Accuracy:" "$RESULTS_DIR/$exp_name/results_summary.txt" | awk '{print $3}')
        infer=$(grep "Inference Accuracy:" "$RESULTS_DIR/$exp_name/results_summary.txt" | awk '{print $3}')
        loss=$(grep "Accuracy Loss:" "$RESULTS_DIR/$exp_name/results_summary.txt" | awk '{print $3}')
        
        printf "%-25s | %-3s | %-10s | %-12s | %-12s | %-10s\n" \
            "$exp_name" "$quality" "$ratio" "$baseline" "$infer" "$loss"
    fi
done

echo ""
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "CAFE Baseline: 525x compression, 11.4% AUC loss"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""
echo "All results saved to: $RESULTS_DIR/*/results_summary.txt"
echo ""
echo "========================================================================"
echo "✓ ALL TESTING COMPLETE!"
echo "========================================================================"
