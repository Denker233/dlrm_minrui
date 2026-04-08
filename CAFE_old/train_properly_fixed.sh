#!/bin/bash

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./proper_run_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

DATA_DIR="$HOME/expr/dlrm_minrui/criteo_24days"

TIMING_LOG="$OUTPUT_DIR/timing.log"
touch "$TIMING_LOG"

log_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$TIMING_LOG"
}

echo "================================================================================"
echo "FIXED DLRM TRAINING: Proper Hyperparameters"
echo "================================================================================"
echo "Key fixes:"
echo "  ✓ Learning rate: 0.1 → 0.005 (20x lower to avoid collapse)"
echo "  ✓ Epochs: 1 → 20 (learn patterns properly)"
echo "  ✓ Frequent testing to monitor AUC"
echo "  ✓ Expected AUC: 65-75% (vs broken 50%)"
echo ""
echo "Output: $OUTPUT_DIR"
echo "Expected time: 2-3 hours total"
echo "================================================================================"

SCRIPT_START=$(date +%s)
log_time "Training started"

#==============================================================================
# BASELINE TRAINING
#==============================================================================
echo ""
echo "================================================================================"
echo "PHASE 1: BASELINE TRAINING (No Compression)"
echo "================================================================================"
log_time "Baseline training started"

BASELINE_START=$(date +%s)

python dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.005 \
--mini-batch-size=128 \
--print-freq=50 \
--print-time \
--test-freq=111 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--nepochs=20 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="$OUTPUT_DIR/baseline.pt" \
2>&1 | tee "$OUTPUT_DIR/train_baseline.log"

BASELINE_END=$(date +%s)
BASELINE_TIME=$((BASELINE_END - BASELINE_START))

log_time "Baseline completed in $((BASELINE_TIME/60))m $((BASELINE_TIME%60))s"

echo ""
echo "✓ Baseline training complete!"
echo "  Duration: $((BASELINE_TIME/60))m $((BASELINE_TIME%60))s"
echo "  Final metrics:"
grep "accuracy.*auc" "$OUTPUT_DIR/train_baseline.log" | tail -1

#==============================================================================
# SKETCH TRAINING
#==============================================================================
echo ""
echo "================================================================================"
echo "PHASE 2: SKETCH TRAINING (221x Compression)"
echo "================================================================================"
log_time "Sketch training started"

SKETCH_START=$(date +%s)

python dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.005 \
--mini-batch-size=128 \
--print-freq=50 \
--print-time \
--test-freq=111 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--sketch-flag \
--compress-rate=0.001 \
--hash-rate=0.5 \
--nepochs=20 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="$OUTPUT_DIR/sketch.pt" \
2>&1 | tee "$OUTPUT_DIR/train_sketch.log"

SKETCH_END=$(date +%s)
SKETCH_TIME=$((SKETCH_END - SKETCH_START))

log_time "Sketch completed in $((SKETCH_TIME/60))m $((SKETCH_TIME%60))s"

echo ""
echo "✓ Sketch training complete!"
echo "  Duration: $((SKETCH_TIME/60))m $((SKETCH_TIME%60))s"
echo "  Final metrics:"
grep "accuracy.*auc" "$OUTPUT_DIR/train_sketch.log" | tail -1

#==============================================================================
# GENERATE REPORT
#==============================================================================
echo ""
echo "================================================================================"
echo "GENERATING COMPARISON REPORT"
echo "================================================================================"
log_time "Generating report"

cat > "$OUTPUT_DIR/generate_report.py" << 'PYEOF'
import re
import os
import sys

def extract_final_metrics(filename):
    """Extract final accuracy and AUC from training log"""
    try:
        with open(filename) as f:
            lines = f.readlines()
            # Find all lines with accuracy and auc
            for line in reversed(lines):
                if 'accuracy' in line.lower() and 'auc' in line.lower():
                    acc = re.search(r'accuracy\s+(\d+\.\d+)\s*%', line)
                    auc = re.search(r'auc\s+(\d+\.\d+)\s*%', line)
                    if acc and auc:
                        return float(acc.group(1)), float(auc.group(1))
    except:
        pass
    return None, None

def extract_learning_curve(filename):
    """Extract AUC progression during training"""
    aucs = []
    try:
        with open(filename) as f:
            for line in f:
                if 'accuracy' in line.lower() and 'auc' in line.lower():
                    match = re.search(r'auc\s+(\d+\.\d+)\s*%', line)
                    if match:
                        aucs.append(float(match.group(1)))
    except:
        pass
    return aucs

output_dir = sys.argv[1]
baseline_time = int(sys.argv[2])
sketch_time = int(sys.argv[3])

print("="*80)
print("DLRM TRAINING RESULTS - PROPER HYPERPARAMETERS")
print("="*80)
print()

# Get final metrics
base_acc, base_auc = extract_final_metrics(f'{output_dir}/train_baseline.log')
sketch_acc, sketch_auc = extract_final_metrics(f'{output_dir}/train_sketch.log')

# Get learning curves
base_aucs = extract_learning_curve(f'{output_dir}/train_baseline.log')
sketch_aucs = extract_learning_curve(f'{output_dir}/train_sketch.log')

print("BASELINE (No Compression)")
print("-"*80)
if base_acc and base_auc:
    print(f"Final Accuracy:  {base_acc:.3f}%")
    print(f"Final AUC:       {base_auc:.3f}%")
    
    if base_auc > 55:
        print("✓ Model is learning! (AUC > 55%)")
    else:
        print("⚠ Model may still be broken (AUC ≤ 55%)")
else:
    print("Could not extract metrics")

print(f"Training time:   {baseline_time//60}m {baseline_time%60}s")

if base_aucs:
    print(f"AUC progression: {base_aucs[0]:.1f}% → {base_aucs[-1]:.1f}%")

print()
print("SKETCH (221x Compression)")
print("-"*80)
if sketch_acc and sketch_auc:
    print(f"Final Accuracy:  {sketch_acc:.3f}%")
    print(f"Final AUC:       {sketch_auc:.3f}%")
    
    if sketch_auc > 55:
        print("✓ Model is learning! (AUC > 55%)")
    else:
        print("⚠ Model may still be broken (AUC ≤ 55%)")
else:
    print("Could not extract metrics")

print(f"Training time:   {sketch_time//60}m {sketch_time%60}s")

if sketch_aucs:
    print(f"AUC progression: {sketch_aucs[0]:.1f}% → {sketch_aucs[-1]:.1f}%")

# Model sizes
print()
print("MODEL SIZES")
print("-"*80)
try:
    base_size = os.path.getsize(f'{output_dir}/baseline.pt')
    sketch_size = os.path.getsize(f'{output_dir}/sketch.pt')
    print(f"Baseline:  {base_size/1024/1024:.1f} MB")
    print(f"Sketch:    {sketch_size/1024/1024:.1f} MB")
    print(f"Compression: {base_size/sketch_size:.1f}x")
except:
    print("Could not get model sizes")

# Comparison
if base_acc and sketch_acc and base_auc and sketch_auc:
    print()
    print("COMPARISON")
    print("-"*80)
    acc_loss = base_acc - sketch_acc
    auc_loss = base_auc - sketch_auc
    
    print(f"Accuracy loss: {acc_loss:+.3f}% ({acc_loss/base_acc*100:+.2f}% relative)")
    print(f"AUC loss:      {auc_loss:+.3f}%")
    
    time_overhead = (sketch_time - baseline_time) / baseline_time * 100
    print(f"Time overhead: {time_overhead:+.1f}%")
    
    print()
    print("ASSESSMENT:")
    if base_auc < 55 or sketch_auc < 55:
        print("  ⚠ Models still broken - AUC too low!")
        print("    Need even lower learning rate or more epochs")
    elif abs(auc_loss) < 2:
        print("  ✓✓✓ EXCELLENT! Compression has minimal impact on quality")
    elif abs(auc_loss) < 5:
        print("  ✓✓ GOOD! Acceptable quality trade-off")
    else:
        print("  ✗ Compression degrades quality significantly")
    
    print()
    print("DQRM COMPARISON:")
    print("  Method         Compression  Accuracy Loss")
    print("  " + "-"*45)
    print("  DQRM INT8      4x           ~0.1%")
    print("  DQRM PQ-64     64x          ~1.0%")
    if base_size and sketch_size:
        rel_loss = abs(acc_loss) / base_acc * 100 if base_acc > 0 else 0
        print(f"  Your Sketch    {base_size/sketch_size:.0f}x         {rel_loss:.3f}%")

print()
print("="*80)
print(f"Detailed logs: {output_dir}/")
print("="*80)
PYEOF

python3 "$OUTPUT_DIR/generate_report.py" "$OUTPUT_DIR" "$BASELINE_TIME" "$SKETCH_TIME" | tee "$OUTPUT_DIR/FINAL_REPORT.txt"

# Display report
cat "$OUTPUT_DIR/FINAL_REPORT.txt"

SCRIPT_END=$(date +%s)
TOTAL_TIME=$((SCRIPT_END - SCRIPT_START))
log_time "Complete! Total time: $((TOTAL_TIME/60))m $((TOTAL_TIME%60))s"

echo ""
echo "================================================================================"
echo "TRAINING COMPLETE!"
echo "================================================================================"
echo "Total time: $((TOTAL_TIME/60))m $((TOTAL_TIME%60))s"
echo ""
echo "Key files:"
echo "  Report:          $OUTPUT_DIR/FINAL_REPORT.txt"
echo "  Timing log:      $OUTPUT_DIR/timing.log"
echo "  Baseline model:  $OUTPUT_DIR/baseline.pt"
echo "  Sketch model:    $OUTPUT_DIR/sketch.pt"
echo "================================================================================"

