#!/bin/bash

echo "Checking if training actually happened..."
echo ""

LOG="training_benchmark_20260115_224909/original_iter1.log"

if [ -f "$LOG" ]; then
    echo "Contents of $LOG:"
    echo "========================================="
    cat "$LOG"
    echo ""
    echo "========================================="
    echo ""
    
    # Check for training indicators
    if grep -q "Epoch" "$LOG"; then
        echo "✓ Found 'Epoch' in log"
    else
        echo "✗ No 'Epoch' markers found"
    fi
    
    if grep -q "batch" "$LOG" || grep -q "iteration" "$LOG"; then
        echo "✓ Found batch/iteration markers"
    else
        echo "✗ No batch/iteration markers"
    fi
    
    # Count occurrences
    echo ""
    echo "Log analysis:"
    echo "  'loss' mentions: $(grep -c 'loss' $LOG)"
    echo "  'batch' mentions: $(grep -c 'batch' $LOG)"
    echo "  'epoch' mentions: $(grep -c -i 'epoch' $LOG)"
else
    echo "Log file not found: $LOG"
fi

