#!/bin/bash

# Find most recent results directory
if [ -z "$1" ]; then
    RESULTS_DIR=$(ls -td training_benchmark_* 2>/dev/null | head -1)
    if [ -z "$RESULTS_DIR" ]; then
        echo "No training benchmark results found!"
        echo "Usage: ./view_training_results.sh [results_directory]"
        exit 1
    fi
    echo "Using most recent results: $RESULTS_DIR"
else
    RESULTS_DIR="$1"
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Directory not found: $RESULTS_DIR"
    exit 1
fi

echo "========================================="
echo "TRAINING BENCHMARK RESULTS"
echo "========================================="
echo ""

# Show summary
if [ -f "$RESULTS_DIR/training_summary.txt" ]; then
    cat "$RESULTS_DIR/training_summary.txt"
else
    echo "Summary file not found!"
fi

echo ""
echo "========================================="
echo "AVAILABLE FILES"
echo "========================================="
echo ""
ls -lh "$RESULTS_DIR/"

echo ""
echo "========================================="
echo "QUICK COMMANDS"
echo "========================================="
echo ""
echo "View full log:"
echo "  less $RESULTS_DIR/benchmark_log.txt"
echo ""
echo "View specific run:"
echo "  less $RESULTS_DIR/original_iter1.log"
echo "  less $RESULTS_DIR/compressed_iter1.log"
echo ""
echo "View summary:"
echo "  cat $RESULTS_DIR/training_summary.txt"
echo ""

