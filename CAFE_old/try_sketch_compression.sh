#!/bin/bash

OUTPUT_DIR="./sketch_attempts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
mkdir -p "./models"
DATA_DIR="$HOME/expr/dlrm_minrui/criteo_24days"

echo "================================================================================"
echo "ATTEMPTING SKETCH COMPRESSION WITH DIFFERENT PARAMETERS"
echo "================================================================================"
echo ""

# Strategy: Try higher compress_rate = larger hot cache = might avoid bug

attempts=(
    "0.01:0.5:10"   # 10x higher compress rate
    "0.005:0.5:10"  # 5x higher compress rate  
    "0.002:0.5:10"  # 2x higher compress rate
    "0.001:0.3:10"  # Lower hash rate
    "0.001:0.5:100" # Higher sketch threshold
)

for params in "${attempts[@]}"; do
    IFS=':' read -r compress_rate hash_rate sketch_threshold <<< "$params"
    
    echo ""
    echo "================================================================================"
    echo "ATTEMPT: compress_rate=$compress_rate, hash_rate=$hash_rate, threshold=$sketch_threshold"
    echo "================================================================================"
    
    python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --loss-function=bce \
    --mini-batch-size=128 \
    --print-freq=100 \
    --test-freq=500 \
    --test-mini-batch-size=2048 \
    --test-num-workers=8 \
    --sketch-flag \
    --compress-rate=$compress_rate \
    --hash-rate=$hash_rate \
    --sketch-threshold=$sketch_threshold \
    --adjust-threshold=1 \
    --sketch-alpha=1.0 \
    --num-batches=500 \
    --cat-path="$DATA_DIR/sparse" \
    --dense-path="$DATA_DIR/dense" \
    --label-path="$DATA_DIR/label" \
    --count-path="$DATA_DIR/processed_count.bin" \
    --save-model="./models/sketch_test_${compress_rate}.pt" \
    2>&1 | tee "$OUTPUT_DIR/attempt_${compress_rate}_${hash_rate}_${sketch_threshold}.log"
    
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS with parameters: compress_rate=$compress_rate, hash_rate=$hash_rate"
        echo ""
        echo "Final result:"
        grep "accuracy.*auc" "$OUTPUT_DIR/attempt_${compress_rate}_${hash_rate}_${sketch_threshold}.log" | tail -1
        
        echo ""
        echo "Now running full 2000 batch test with working parameters..."
        
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
        --test-num-workers=8 \
        --sketch-flag \
        --compress-rate=$compress_rate \
        --hash-rate=$hash_rate \
        --sketch-threshold=$sketch_threshold \
        --num-batches=2000 \
        --cat-path="$DATA_DIR/sparse" \
        --dense-path="$DATA_DIR/dense" \
        --label-path="$DATA_DIR/label" \
        --count-path="$DATA_DIR/processed_count.bin" \
        --save-model="./models/sketch_working.pt" \
        2>&1 | tee "$OUTPUT_DIR/sketch_full.log"
        
        echo ""
        echo "Full test complete!"
        exit 0
    else
        echo "✗ FAILED with parameters: compress_rate=$compress_rate, hash_rate=$hash_rate"
        echo "Error in log: $OUTPUT_DIR/attempt_${compress_rate}_${hash_rate}_${sketch_threshold}.log"
    fi
done

echo ""
echo "================================================================================"
echo "All attempts failed. The sketch code has a bug that needs fixing."
echo "See logs in: $OUTPUT_DIR"
echo "================================================================================"

