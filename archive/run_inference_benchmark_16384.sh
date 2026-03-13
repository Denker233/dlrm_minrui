#!/bin/bash

echo "================================================================================"
echo "DLRM INFERENCE BENCHMARK - BATCH SIZE 16384"
echo "================================================================================"
echo ""

# Default paths
ORIGINAL_MODEL="./models/dlrm_kaggle_quick.pt"
COMPRESSED_MODEL="./models/dlrm_kaggle_quick_compressed.pt"
BATCH_SIZE=16384
NUM_ITERATIONS=100

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --original-model)
            ORIGINAL_MODEL="$2"
            shift 2
            ;;
        --compressed-model)
            COMPRESSED_MODEL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if models exist
if [ ! -f "$ORIGINAL_MODEL" ]; then
    echo "Error: Original model not found at $ORIGINAL_MODEL"
    exit 1
fi

if [ ! -f "$COMPRESSED_MODEL" ]; then
    echo "Error: Compressed model not found at $COMPRESSED_MODEL"
    exit 1
fi

echo "Configuration:"
echo "  Original model:   $ORIGINAL_MODEL"
echo "  Compressed model: $COMPRESSED_MODEL"
echo "  Batch size:       $BATCH_SIZE"
echo "  Iterations:       $NUM_ITERATIONS"
echo ""

# Run benchmark
python benchmark_inference_16384.py \
    --original-model "$ORIGINAL_MODEL" \
    --compressed-model "$COMPRESSED_MODEL" \
    --batch-size "$BATCH_SIZE" \
    --num-iterations "$NUM_ITERATIONS"

echo ""
echo "================================================================================"
echo "DONE!"
echo "================================================================================"

