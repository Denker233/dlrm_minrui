#!/bin/bash

OUTPUT_DIR="./models/dqrm_comparison"

echo "================================================================================"
echo "TRAINING PROGRESS MONITOR"
echo "================================================================================"
echo ""

if [ ! -f "$OUTPUT_DIR/training_log.txt" ]; then
    echo "Training hasn't started yet or log file not found."
    echo "Run: ./train_and_compare_with_dqrm.sh"
    exit 1
fi

echo "Latest accuracy results:"
echo "--------------------------------------------------------------------------------"
grep "accuracy" $OUTPUT_DIR/training_log.txt | tail -10
echo ""

echo "Current epoch/batch:"
echo "--------------------------------------------------------------------------------"
grep "Finished" $OUTPUT_DIR/training_log.txt | tail -5
echo ""

echo "Estimated time remaining:"
echo "--------------------------------------------------------------------------------"
BATCHES_DONE=$(grep -c "Finished" $OUTPUT_DIR/training_log.txt)
TOTAL_BATCHES=$((5 * 6000))  # 5 epochs * ~6000 batches/epoch
PERCENT_DONE=$(echo "scale=1; $BATCHES_DONE * 100 / $TOTAL_BATCHES" | bc)
echo "Progress: $BATCHES_DONE / $TOTAL_BATCHES batches ($PERCENT_DONE%)"
echo ""

echo "To watch live progress:"
echo "  tail -f $OUTPUT_DIR/training_log.txt"
echo ""
