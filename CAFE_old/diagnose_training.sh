#!/bin/bash

DIR="comparison_20260115_214502"

echo "=== TRAINING DIAGNOSTICS ==="
echo ""

echo "1. How many training iterations actually ran?"
grep "Finished training it" "$DIR/train_baseline.log" | wc -l
echo "   (Should be 5000)"

echo ""
echo "2. Did training loop run or just evaluate?"
grep -c "Training state: loss" "$DIR/train_baseline.log"

echo ""
echo "3. What was the training doing?"
head -50 "$DIR/train_baseline.log"

echo ""
echo "4. Model file sizes:"
ls -lh "$DIR"/*.pt

echo ""
echo "5. Check if --inference-only was accidentally set:"
grep "inference-only\|inference_only" "$DIR/train_baseline.log" | head -5

