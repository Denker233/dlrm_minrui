#!/bin/bash

RESULTS_FILE="benchmark_multi_run_results.txt"
RESULTS_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

echo "Running 3 iterations of each model..."
echo "Results will be saved to: $RESULTS_DIR/"
echo ""

# Save all output to both terminal and file
exec > >(tee "$RESULTS_DIR/$RESULTS_FILE") 2>&1

echo "========================================="
echo "MULTI-RUN BENCHMARK"
echo "========================================="
echo "Date: $(date)"
echo "Results directory: $RESULTS_DIR"
echo ""

ORIGINAL="./models/dlrm_kaggle_quick.pt"
DECOMPRESSED="./models/dlrm_kaggle_quick_decompressed.pt"

# Arrays to store times
ORIG_TIMES=()
COMP_TIMES=()

for i in {1..3}; do
    echo "========================================="
    echo "ITERATION $i/3"
    echo "========================================="
    
    # Clear caches
    echo "Clearing caches..."
    sync
    if sudo sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; then
        echo "  ✓ Caches cleared"
    else
        echo "  ⚠ No sudo - sleeping 30s"
        sleep 30
    fi
    echo ""
    
    # Original
    echo "Running ORIGINAL model (iteration $i)..."
    START=$(date +%s)
    python dlrm_s_pytorch.py \
        --arch-sparse-feature-size=16 \
        --arch-mlp-bot="13-512-256-64-16" \
        --arch-mlp-top="512-256-1" \
        --data-generation=dataset \
        --data-set=kaggle \
        --raw-data-file=./input/train.txt \
        --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
        --loss-function=bce \
        --round-targets=True \
        --test-mini-batch-size=2048 \
        --num-workers=0 \
        --test-num-workers=0 \
        --load-model="$ORIGINAL" \
        --inference-only > "$RESULTS_DIR/original_iter${i}.log" 2>&1
    END=$(date +%s)
    ORIG_TIME=$((END - START))
    ORIG_TIMES+=($ORIG_TIME)
    
    # Extract accuracy
    ORIG_ACC=$(grep -oP "accuracy \K[\d.]+" "$RESULTS_DIR/original_iter${i}.log" | tail -1)
    
    echo "  Time: ${ORIG_TIME}s"
    echo "  Accuracy: ${ORIG_ACC}%"
    echo ""
    
    # Clear caches again
    echo "Clearing caches..."
    sync
    if sudo sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; then
        echo "  ✓ Caches cleared"
    else
        echo "  ⚠ No sudo - sleeping 30s"
        sleep 30
    fi
    echo ""
    
    # Compressed
    echo "Running COMPRESSED model (iteration $i)..."
    START=$(date +%s)
    python dlrm_s_pytorch.py \
        --arch-sparse-feature-size=16 \
        --arch-mlp-bot="13-512-256-64-16" \
        --arch-mlp-top="512-256-1" \
        --data-generation=dataset \
        --data-set=kaggle \
        --raw-data-file=./input/train.txt \
        --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
        --loss-function=bce \
        --round-targets=True \
        --test-mini-batch-size=2048 \
        --num-workers=0 \
        --test-num-workers=0 \
        --load-model="$DECOMPRESSED" \
        --inference-only > "$RESULTS_DIR/compressed_iter${i}.log" 2>&1
    END=$(date +%s)
    COMP_TIME=$((END - START))
    COMP_TIMES+=($COMP_TIME)
    
    # Extract accuracy
    COMP_ACC=$(grep -oP "accuracy \K[\d.]+" "$RESULTS_DIR/compressed_iter${i}.log" | tail -1)
    
    echo "  Time: ${COMP_TIME}s"
    echo "  Accuracy: ${COMP_ACC}%"
    echo ""
done

echo "========================================="
echo "FINAL RESULTS"
echo "========================================="
echo ""
echo "Original times:   ${ORIG_TIMES[@]}"
echo "Compressed times: ${COMP_TIMES[@]}"
echo ""

python3 << PYTHON > "$RESULTS_DIR/statistics.txt"
import numpy as np

orig = np.array([${ORIG_TIMES[@]}])
comp = np.array([${COMP_TIMES[@]}])

print("="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print()

print(f"Original Model:")
print(f"  Mean:   {orig.mean():.2f}s")
print(f"  Std:    {orig.std():.2f}s")
print(f"  Min:    {orig.min()}s")
print(f"  Max:    {orig.max()}s")
print(f"  Values: {list(orig)}")
print()

print(f"Compressed Model:")
print(f"  Mean:   {comp.mean():.2f}s")
print(f"  Std:    {comp.std():.2f}s")
print(f"  Min:    {comp.min()}s")
print(f"  Max:    {comp.max()}s")
print(f"  Values: {list(comp)}")
print()

diff = comp.mean() - orig.mean()
pct = (comp.mean() / orig.mean() - 1) * 100

print(f"Difference:")
print(f"  Absolute: {diff:+.2f}s")
print(f"  Relative: {pct:+.2f}%")
print()

# Statistical significance (simple t-test)
from scipy import stats
if len(orig) > 1 and len(comp) > 1:
    t_stat, p_value = stats.ttest_ind(orig, comp)
    print(f"Statistical test (t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Result: Statistically significant difference (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")
else:
    print(f"Statistical test: Need more samples")

print()
print("="*80)
PYTHON

cat "$RESULTS_DIR/statistics.txt"

echo ""
echo "========================================="
echo "RESULTS SAVED TO:"
echo "========================================="
echo ""
echo "Directory: $RESULTS_DIR/"
echo ""
echo "Files:"
ls -lh "$RESULTS_DIR/"
echo ""
echo "Summary file: $RESULTS_DIR/$RESULTS_FILE"
echo "Statistics:   $RESULTS_DIR/statistics.txt"
echo ""

