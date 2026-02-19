#!/bin/bash
set -e

RESULTS_DIR="./codec_experiments_20260120_221951"

clear_cache() {
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "  (cache clear skipped - no sudo)"
    sleep 2
}

test_model() {
    local name=$1
    local q=$2
    
    echo ""
    echo "========================================================================"
    echo "$name (Quality=$q)"
    echo "========================================================================"
    
    local dir="$RESULTS_DIR/$name"
    local compressed="$dir/model_compressed_q${q}.pt"
    local decompressed="$dir/model_decompressed_q${q}.pt"
    
    echo "[1/3] Decompressing..."
    python3 decompress_codec.py "$compressed" "$decompressed"
    
    echo ""
    echo "[2/3] Clearing cache..."
    clear_cache
    
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
        > "$dir/inference_q${q}.log" 2>&1
    
    # Extract results
    local acc=$(grep "accuracy" "$dir/inference_q${q}.log" | grep -oP 'accuracy \K[0-9.]+' | tail -1)
    local ratio=$(grep "Average ratio:" "$dir/compress.log" | grep -oP 'Average ratio:\s+\K[0-9.]+' 2>/dev/null || echo "?")
    local loss=$(python3 -c "print(f'{78.523-float(${acc:-0}):.3f}')" 2>/dev/null || echo "?")
    
    echo ""
    echo "Results:"
    echo "  Compression ratio: ${ratio}x"
    echo "  Accuracy: ${acc}%"
    echo "  Accuracy loss: ${loss}%"
    
    echo ""
    echo "✓ Complete: $name"
}

echo "========================================================================"
echo "CODEC COMPRESSION - INFERENCE TESTING"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Architecture: 16-dim, 13-512-256-64-16 → 512-256-1"
echo "  Batch size: 2048"
echo "  Baseline: 78.523%"
echo ""
echo "========================================================================"

# Run all tests
test_model "exp1_q18_highquality" 18
test_model "exp2_q23_balanced" 23
test_model "exp3_q28_highcompress" 28

# Final summary
echo ""
echo "========================================================================"
echo "FINAL RESULTS"
echo "========================================================================"
echo ""

printf "%-25s | %-3s | %-8s | %-10s | %-10s\n" "Experiment" "Q" "Ratio" "Accuracy" "Loss"
echo "────────────────────────────────────────────────────────────────────────"

for exp in "exp1_q18_highquality:18" "exp2_q23_balanced:23" "exp3_q28_highcompress:28"; do
    IFS=':' read -r name q <<< "$exp"
    dir="$RESULTS_DIR/$name"
    
    if [ -f "$dir/inference_q${q}.log" ]; then
        acc=$(grep "accuracy" "$dir/inference_q${q}.log" | grep -oP 'accuracy \K[0-9.]+' | tail -1 || echo "N/A")
        ratio=$(grep "Average ratio:" "$dir/compress.log" | grep -oP 'Average ratio:\s+\K[0-9.]+' 2>/dev/null || echo "?")
        loss=$(python3 -c "print(f'{78.523-float(${acc:-0}):.3f}')" 2>/dev/null || echo "?")
        
        printf "%-25s | %-3s | %6sx | %8s%% | %8s%%\n" "$name" "$q" "$ratio" "$acc" "$loss"
    fi
done

echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "CAFE Baseline: 525x compression, 11.4% AUC loss"
echo "========================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR/*/inference_q*.log"
echo ""
