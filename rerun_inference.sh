#!/bin/bash
set -e

RESULTS_DIR="./codec_experiments_20260120_221951"

clear_cache() {
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    sleep 2
}

run_inference() {
    local name=$1
    local q=$2
    
    echo ""
    echo "========================================================================"
    echo "$name (Quality=$q)"
    echo "========================================================================"
    
    local dir="$RESULTS_DIR/$name"
    local compressed="$dir/model_compressed_q${q}.pt"
    local decompressed="$dir/model_decompressed_q${q}.pt"
    
    # Decompress
    if [ ! -f "$decompressed" ]; then
        echo "[1/3] Decompressing..."
        python3 decompress_with_details.py \
            --compressed="$compressed" \
            --output="$decompressed"
    else
        echo "[1/3] Using existing decompressed model..."
    fi
    
    # Clear cache
    echo ""
    echo "[2/3] Clearing cache..."
    clear_cache
    
    # Inference
    echo ""
    echo "[3/3] Running inference..."
    python3 dlrm_s_pytorch.py \
        --arch-sparse-feature-size=16 \
        --arch-mlp-bot=13-512-256-64-16 \
        --arch-mlp-top=512-256-1 \
        --data-generation=dataset \
        --data-set=kaggle \
        --raw-data-file=./input/train.txt \
        --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
        --loss-function=bce \
        --round-targets=True \
        --test-mini-batch-size=2048 \
        --num-workers=0 \
        --test-num-workers=8 \
        --load-model="$decompressed" \
        --inference-only \
        2>&1 | tee "$dir/inference_q${q}.log"
    
    # Extract results
    local acc=$(grep "accuracy" "$dir/inference_q${q}.log" | grep -oP 'accuracy \K[0-9.]+' | tail -1)
    local auc=$(grep "auc" "$dir/inference_q${q}.log" | grep -oP 'auc \K[0-9.]+' | tail -1 || echo "N/A")
    local ratio=$(grep "Average ratio:" "$dir/compress.log" | grep -oP 'Average ratio:\s+\K[0-9.]+' 2>/dev/null || echo "?")
    local loss=$(python3 -c "print(f'{78.523-${acc:-0}:.3f}')" 2>/dev/null || echo "?")
    
    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "RESULTS:"
    echo "  Compression: ${ratio}x"
    echo "  Accuracy: ${acc}%"
    echo "  Loss: ${loss}%"
    [ "$auc" != "N/A" ] && echo "  AUC: ${auc}"
    echo "──────────────────────────────────────────────────────────────────────"
    
    # Save summary
    cat > "$dir/results_summary.txt" << SUMMARY
Experiment: $name
Quality: $q
Compression Ratio: ${ratio}x
Baseline Accuracy: 78.523%
Inference Accuracy: ${acc}%
Accuracy Loss: ${loss}%
AUC: ${auc}
SUMMARY
    
    echo "✓ Complete: $name"
}

echo "========================================================================"
echo "CODEC COMPRESSION - INFERENCE TESTING"
echo "========================================================================"

# Remove incomplete decompressed models
rm -f "$RESULTS_DIR"/*/model_decompressed_q*.pt

# Run all tests
run_inference "exp1_q18_highquality" 18
run_inference "exp2_q23_balanced" 23
run_inference "exp3_q28_highcompress" 28

# Summary
echo ""
echo "========================================================================"
echo "FINAL RESULTS"
echo "========================================================================"
echo ""

printf "%-25s | Q  | %-8s | %-10s | %-10s\n" "Experiment" "" "Ratio" "Accuracy" "Loss"
echo "────────────────────────────────────────────────────────────────────────"

for exp in "exp1_q18_highquality:18" "exp2_q23_balanced:23" "exp3_q28_highcompress:28"; do
    IFS=':' read -r name q <<< "$exp"
    dir="$RESULTS_DIR/$name"
    
    if [ -f "$dir/results_summary.txt" ]; then
        ratio=$(grep "Compression Ratio:" "$dir/results_summary.txt" | awk '{print $3}')
        acc=$(grep "Inference Accuracy:" "$dir/results_summary.txt" | awk '{print $3}')
        loss=$(grep "Accuracy Loss:" "$dir/results_summary.txt" | awk '{print $3}')
        
        printf "%-25s | %-2s | %6s | %8s | %8s\n" "$name" "$q" "$ratio" "$acc" "$loss"
    fi
done

echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "CAFE: 525x compression, 11.4% AUC loss"
echo "========================================================================"
