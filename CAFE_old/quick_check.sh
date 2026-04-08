#!/bin/bash

echo "================================================================================"
echo "TRAINING PROGRESS - QUICK CHECK"
echo "================================================================================"
echo ""

DIR="proper_run_20260115_223709"

if [ ! -d "$DIR" ]; then
    echo "Error: Directory not found"
    exit 1
fi

# Current phase
if [ -f "$DIR/train_sketch.log" ] && [ $(wc -l < "$DIR/train_sketch.log") -gt 10 ]; then
    PHASE="Sketch Training"
    LOG="$DIR/train_sketch.log"
else
    PHASE="Baseline Training"
    LOG="$DIR/train_baseline.log"
fi

echo "Current Phase: $PHASE"
echo ""

# Latest metrics
echo "Latest 5 Test Results:"
echo "----------------------------------------"
grep "Testing at" "$LOG" 2>/dev/null | tail -5 | while read line; do
    iter=$(echo "$line" | grep -oP '\d+/\d+' | head -1)
    acc=$(grep -A1 "$line" "$LOG" | tail -1 | grep -oP 'accuracy \K[\d.]+')
    auc=$(grep -A1 "$line" "$LOG" | tail -1 | grep -oP 'auc \K[\d.]+')
    if [ -n "$acc" ]; then
        echo "Iter $iter: Acc=${acc}%, AUC=${auc}%"
    fi
done

echo ""
echo "Current Status:"
echo "----------------------------------------"
LAST_ITER=$(grep -oP "Finished training it \K\d+" "$LOG" | tail -1)
TOTAL_ITER=$(grep -oP "Finished training it \d+/\K\d+" "$LOG" | head -1)
if [ -n "$LAST_ITER" ] && [ -n "$TOTAL_ITER" ]; then
    PROGRESS=$(echo "scale=2; $LAST_ITER * 100 / $TOTAL_ITER" | bc)
    echo "Progress: $LAST_ITER / $TOTAL_ITER iterations (${PROGRESS}%)"
fi

LAST_AUC=$(grep "auc" "$LOG" | tail -1 | grep -oP 'auc \K[\d.]+')
if [ -n "$LAST_AUC" ]; then
    echo "Latest AUC: ${LAST_AUC}%"
    
    if (( $(echo "$LAST_AUC < 55" | bc -l) )); then
        echo "Status: ⚠ Still learning (target: 65%)"
    elif (( $(echo "$LAST_AUC < 65" | bc -l) )); then
        echo "Status: ⏳ Making progress"
    else
        echo "Status: ✓ Good results!"
    fi
fi

echo ""
echo "================================================================================"
