#!/bin/bash

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./final_1epoch_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

DATA_DIR="$HOME/expr/dlrm_minrui/criteo_24days"

TIMING_LOG="$OUTPUT_DIR/timing.log"
touch "$TIMING_LOG"

log_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$TIMING_LOG"
}

echo "================================================================================"
echo "FINAL COMPARISON: 1 EPOCH TRAINING (Standard Baseline)"
echo "================================================================================"
echo "Configuration:"
echo "  - Training: 1 EPOCH (~306k iterations)"
echo "  - Learning rate: 0.005"
echo "  - Expected time: ~5 hours total"
echo "  - Expected AUC: 75-76%"
echo ""
echo "Output: $OUTPUT_DIR"
echo "================================================================================"

SCRIPT_START=$(date +%s)
log_time "Training started"

#==============================================================================
# BASELINE TRAINING (1 EPOCH)
#==============================================================================
echo ""
echo ">>> BASELINE TRAINING (1 epoch)"
echo ">>> Estimated time: 2.5 hours"
echo ""

BASELINE_START=$(date +%s)
log_time "Baseline training started"

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
--print-freq=5000 \
--print-time \
--test-freq=50000 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--nepochs=1 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="$OUTPUT_DIR/baseline.pt" \
2>&1 | tee "$OUTPUT_DIR/train_baseline.log"

BASELINE_END=$(date +%s)
BASELINE_TIME=$((BASELINE_END - BASELINE_START))

log_time "Baseline completed: $((BASELINE_TIME/3600))h $((BASELINE_TIME%3600/60))m"

echo ""
echo "✓ Baseline complete!"
echo "  Time: $((BASELINE_TIME/60))m"
grep "accuracy.*auc" "$OUTPUT_DIR/train_baseline.log" | tail -1

#==============================================================================
# SKETCH TRAINING (1 EPOCH)
#==============================================================================
echo ""
echo ">>> SKETCH TRAINING (1 epoch)"
echo ">>> Estimated time: 2.5 hours"
echo ""

SKETCH_START=$(date +%s)
log_time "Sketch training started"

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
--print-freq=5000 \
--print-time \
--test-freq=50000 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--sketch-flag \
--compress-rate=0.001 \
--hash-rate=0.5 \
--nepochs=1 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="$OUTPUT_DIR/sketch.pt" \
2>&1 | tee "$OUTPUT_DIR/train_sketch.log"

SKETCH_END=$(date +%s)
SKETCH_TIME=$((SKETCH_END - SKETCH_START))

log_time "Sketch completed: $((SKETCH_TIME/3600))h $((SKETCH_TIME%3600/60))m"

echo ""
echo "✓ Sketch complete!"
echo "  Time: $((SKETCH_TIME/60))m"
grep "accuracy.*auc" "$OUTPUT_DIR/train_sketch.log" | tail -1

#==============================================================================
# GENERATE FINAL REPORT
#==============================================================================
echo ""
echo "================================================================================"
echo "GENERATING FINAL REPORT"
echo "================================================================================"

cat > "$OUTPUT_DIR/generate_report.py" << 'PYEOF'
import re
import os
import sys

def extract_final_metrics(filename):
    try:
        with open(filename) as f:
            lines = f.readlines()
            for line in reversed(lines):
                if 'accuracy' in line.lower() and 'auc' in line.lower():
                    acc = re.search(r'accuracy\s+(\d+\.\d+)\s*%', line)
                    auc = re.search(r'auc\s+(\d+\.\d+)\s*%', line)
                    if acc and auc:
                        return float(acc.group(1)), float(auc.group(1))
    except:
        pass
    return None, None

output_dir = sys.argv[1]
baseline_time = int(sys.argv[2])
sketch_time = int(sys.argv[3])

base_acc, base_auc = extract_final_metrics(f'{output_dir}/train_baseline.log')
sketch_acc, sketch_auc = extract_final_metrics(f'{output_dir}/train_sketch.log')

print("="*80)
print("DLRM BASELINE vs SKETCH COMPRESSION - FINAL RESULTS")
print("="*80)
print()
print("Training: 1 epoch (~306k iterations)")
print()

print("BASELINE (No Compression)")
print("-"*80)
if base_acc and base_auc:
    print(f"Accuracy:  {base_acc:.3f}%")
    print(f"AUC:       {base_auc:.3f}%")
else:
    print("Metrics not available")
print(f"Time:      {baseline_time//60}m")

try:
    base_size = os.path.getsize(f'{output_dir}/baseline.pt') / 1024 / 1024
    print(f"Size:      {base_size:.1f} MB")
except:
    pass

print()
print("SKETCH COMPRESSION")
print("-"*80)
if sketch_acc and sketch_auc:
    print(f"Accuracy:  {sketch_acc:.3f}%")
    print(f"AUC:       {sketch_auc:.3f}%")
else:
    print("Metrics not available")
print(f"Time:      {sketch_time//60}m")

try:
    sketch_size = os.path.getsize(f'{output_dir}/sketch.pt') / 1024 / 1024
    print(f"Size:      {sketch_size:.1f} MB")
except:
    pass

if base_acc and sketch_acc:
    print()
    print("COMPARISON")
    print("-"*80)
    acc_loss = base_acc - sketch_acc
    auc_loss = base_auc - sketch_auc
    
    print(f"Accuracy loss: {acc_loss:+.3f}% absolute")
    print(f"AUC loss:      {auc_loss:+.3f}%")
    
    try:
        compression = base_size / sketch_size
        print(f"Compression:   {compression:.1f}x")
    except:
        pass
    
    time_overhead = (sketch_time - baseline_time) / baseline_time * 100
    print(f"Time overhead: {time_overhead:+.1f}%")
    
    print()
    print("DQRM COMPARISON:")
    print("  Method         Compression  Accuracy Loss")
    print("  " + "-"*45)
    print("  DQRM INT8      4x           ~0.1%")
    print("  DQRM PQ-64     64x          ~1.0%")
    rel_loss = abs(acc_loss) / base_acc * 100 if base_acc > 0 else 0
    try:
        print(f"  Your Sketch    {compression:.0f}x         {rel_loss:.3f}%")
    except:
        print(f"  Your Sketch    ???x         {rel_loss:.3f}%")
    
    print()
    if abs(auc_loss) < 2:
        print("✓✓✓ EXCELLENT! Minimal quality loss at high compression")
    elif abs(auc_loss) < 5:
        print("✓✓ GOOD! Acceptable trade-off")
    else:
        print("⚠ Significant quality degradation")

print()
print("="*80)
PYEOF

python3 "$OUTPUT_DIR/generate_report.py" "$OUTPUT_DIR" "$BASELINE_TIME" "$SKETCH_TIME" | tee "$OUTPUT_DIR/FINAL_REPORT.txt"

cat "$OUTPUT_DIR/FINAL_REPORT.txt"

SCRIPT_END=$(date +%s)
TOTAL_TIME=$((SCRIPT_END - SCRIPT_START))
log_time "Complete! Total: $((TOTAL_TIME/3600))h $((TOTAL_TIME%3600/60))m"

echo ""
echo "================================================================================"
echo "TRAINING COMPLETE!"
echo "================================================================================"
echo "Total time: $((TOTAL_TIME/60))m"
echo "Report: $OUTPUT_DIR/FINAL_REPORT.txt"
echo "================================================================================"

