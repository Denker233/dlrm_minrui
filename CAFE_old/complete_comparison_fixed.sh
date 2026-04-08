#!/bin/bash

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./comparison_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"
mkdir -p "./models"

DATA_DIR="$HOME/expr/dlrm_minrui/criteo_24days"

# Create timing log
TIMING_LOG="$OUTPUT_DIR/timing.log"
touch "$TIMING_LOG"

log_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$TIMING_LOG"
}

echo "================================================================================"
echo "COMPLETE DLRM COMPARISON: BASELINE vs SKETCH COMPRESSION"
echo "================================================================================"
echo "Test configuration:"
echo "  - Training batches: 5000 (more batches for better accuracy)"
echo "  - Training batch size: 128"
echo "  - Inference batch size: 2048"
echo "  - Device: CPU"
echo "  - Estimated time: 20-25 hours total"
echo ""
echo "Output: $OUTPUT_DIR"
echo "================================================================================"
echo ""

SCRIPT_START=$(date +%s)
log_time "Script started"

#==============================================================================
# PHASE 1: TRAINING COMPARISON
#==============================================================================
echo ""
echo "================================================================================"
echo "PHASE 1: TRAINING COMPARISON (5000 batches each)"
echo "================================================================================"
log_time "Phase 1: Training started"

# Train Baseline
echo ""
echo ">>> Training BASELINE (No Compression)"
echo ">>> Estimated time: 12-15 hours..."
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
--learning-rate=0.1 \
--mini-batch-size=128 \
--print-freq=500 \
--print-time \
--test-freq=2000 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--num-batches=5000 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="$OUTPUT_DIR/baseline.pt" \
2>&1 | tee "$OUTPUT_DIR/train_baseline.log"

BASELINE_END=$(date +%s)
BASELINE_TRAIN_TIME=$((BASELINE_END - BASELINE_START))
log_time "Baseline training completed in $((BASELINE_TRAIN_TIME/3600))h $((BASELINE_TRAIN_TIME%3600/60))m"

echo ""
echo "✓ Baseline training complete!"
echo "  Duration: $((BASELINE_TRAIN_TIME/3600))h $((BASELINE_TRAIN_TIME%3600/60))m $((BASELINE_TRAIN_TIME%60))s"
echo "  Final result:"
grep "accuracy.*auc" "$OUTPUT_DIR/train_baseline.log" | tail -1

# Train Sketch
echo ""
echo ">>> Training SKETCH COMPRESSION (221x)"
echo ">>> Estimated time: 13-18 hours..."
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
--learning-rate=0.1 \
--mini-batch-size=128 \
--print-freq=500 \
--print-time \
--test-freq=2000 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--sketch-flag \
--compress-rate=0.001 \
--hash-rate=0.5 \
--sketch-threshold=1 \
--adjust-threshold=1 \
--sketch-alpha=1.0 \
--num-batches=5000 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="$OUTPUT_DIR/sketch.pt" \
2>&1 | tee "$OUTPUT_DIR/train_sketch.log"

SKETCH_END=$(date +%s)
SKETCH_TRAIN_TIME=$((SKETCH_END - SKETCH_START))
log_time "Sketch training completed in $((SKETCH_TRAIN_TIME/3600))h $((SKETCH_TRAIN_TIME%3600/60))m"

echo ""
echo "✓ Sketch training complete!"
echo "  Duration: $((SKETCH_TRAIN_TIME/3600))h $((SKETCH_TRAIN_TIME%3600/60))m $((SKETCH_TRAIN_TIME%60))s"
echo "  Final result:"
grep "accuracy.*auc" "$OUTPUT_DIR/train_sketch.log" | tail -1

#==============================================================================
# PHASE 2: INFERENCE COMPARISON
#==============================================================================
echo ""
echo "================================================================================"
echo "PHASE 2: INFERENCE COMPARISON (Batch Size 2048)"
echo "================================================================================"
log_time "Phase 2: Inference started"

# Check if models exist
if [ ! -f "$OUTPUT_DIR/baseline.pt" ]; then
    echo "ERROR: Baseline model not found!"
    exit 1
fi

if [ ! -f "$OUTPUT_DIR/sketch.pt" ]; then
    echo "ERROR: Sketch model not found!"
    exit 1
fi

# Baseline Inference
echo ""
echo ">>> Running baseline inference..."
BASELINE_INF_START=$(date +%s)
log_time "Baseline inference started"

python dlrm_s_pytorch.py \
--inference-only \
--load-model="$OUTPUT_DIR/baseline.pt" \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--mini-batch-size=2048 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--print-freq=500 \
--print-time \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
2>&1 | tee "$OUTPUT_DIR/inference_baseline.log"

BASELINE_INF_END=$(date +%s)
BASELINE_INF_TIME=$((BASELINE_INF_END - BASELINE_INF_START))
log_time "Baseline inference completed in $((BASELINE_INF_TIME/60))m"

# Sketch Inference  
echo ""
echo ">>> Running sketch inference..."
SKETCH_INF_START=$(date +%s)
log_time "Sketch inference started"

python dlrm_s_pytorch.py \
--inference-only \
--load-model="$OUTPUT_DIR/sketch.pt" \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--mini-batch-size=2048 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--sketch-flag \
--compress-rate=0.001 \
--hash-rate=0.5 \
--notinsert-test \
--print-freq=500 \
--print-time \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
2>&1 | tee "$OUTPUT_DIR/inference_sketch.log"

SKETCH_INF_END=$(date +%s)
SKETCH_INF_TIME=$((SKETCH_INF_END - SKETCH_INF_START))
log_time "Sketch inference completed in $((SKETCH_INF_TIME/60))m"

#==============================================================================
# PHASE 3: GENERATE REPORT
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
from datetime import timedelta

def parse_time(seconds):
    return str(timedelta(seconds=seconds))

def extract_metrics(filename):
    try:
        with open(filename) as f:
            for line in reversed(f.readlines()):
                if 'accuracy' in line.lower() and 'auc' in line.lower():
                    acc = re.search(r'accuracy\s+(\d+\.\d+)\s*%', line)
                    auc = re.search(r'auc\s+(\d+\.\d+)\s*%', line)
                    if acc and auc:
                        return float(acc.group(1)), float(auc.group(1))
    except:
        pass
    return None, None

def extract_time(filename):
    try:
        times = []
        with open(filename) as f:
            for line in f:
                if 'ms/it' in line and 'Finished' in line:
                    match = re.search(r'(\d+\.\d+)\s+ms/it', line)
                    if match:
                        t = float(match.group(1))
                        if t > 0:
                            times.append(t)
        if times:
            return sum(times[-500:]) / len(times[-500:])
    except:
        pass
    return None

# Get parameters
output_dir = sys.argv[1]
baseline_train = int(sys.argv[2])
sketch_train = int(sys.argv[3])
baseline_inf = int(sys.argv[4])
sketch_inf = int(sys.argv[5])

print("="*80)
print("DLRM BASELINE vs SKETCH COMPRESSION - FINAL REPORT")
print("="*80)
print()
print(f"Configuration: 5000 training batches, batch_size=128/2048")
print(f"Device: CPU")
print()

# Training
print("="*80)
print("TRAINING RESULTS")
print("="*80)
print()

base_acc, base_auc = extract_metrics(f'{output_dir}/train_baseline.log')
sketch_acc, sketch_auc = extract_metrics(f'{output_dir}/train_sketch.log')

print("Baseline:")
if base_acc and base_auc:
    print(f"  Accuracy:  {base_acc:.3f}%")
    print(f"  AUC:       {base_auc:.3f}%")
print(f"  Time:      {parse_time(baseline_train)}")

print()
print("Sketch (221x):")
if sketch_acc and sketch_auc:
    print(f"  Accuracy:  {sketch_acc:.3f}%")
    print(f"  AUC:       {sketch_auc:.3f}%")
print(f"  Time:      {parse_time(sketch_train)}")

if base_acc and sketch_acc:
    print()
    loss = base_acc - sketch_acc
    rel_loss = (loss / base_acc * 100) if base_acc > 0 else 0
    print(f"Accuracy loss: {loss:.3f}% abs, {rel_loss:.3f}% relative")

overhead = sketch_train - baseline_train
overhead_pct = (overhead / baseline_train * 100) if baseline_train > 0 else 0
print(f"Time overhead: {parse_time(overhead)} ({overhead_pct:.1f}%)")

# Inference
print()
print("="*80)
print("INFERENCE RESULTS")
print("="*80)
print()

base_inf_acc, base_inf_auc = extract_metrics(f'{output_dir}/inference_baseline.log')
sketch_inf_acc, sketch_inf_auc = extract_metrics(f'{output_dir}/inference_sketch.log')
base_time = extract_time(f'{output_dir}/inference_baseline.log')
sketch_time = extract_time(f'{output_dir}/inference_sketch.log')

print("Baseline:")
if base_inf_acc:
    print(f"  Accuracy:   {base_inf_acc:.3f}%")
if base_time:
    print(f"  Time/batch: {base_time:.2f} ms")
    print(f"  Throughput: {2048/base_time*1000:.0f} samples/sec")

print()
print("Sketch:")
if sketch_inf_acc:
    print(f"  Accuracy:   {sketch_inf_acc:.3f}%")
if sketch_time:
    print(f"  Time/batch: {sketch_time:.2f} ms")
    print(f"  Throughput: {2048/sketch_time*1000:.0f} samples/sec")

if base_time and sketch_time:
    print()
    overhead_ms = sketch_time - base_time
    overhead_pct = (overhead_ms / base_time * 100)
    print(f"Overhead: {overhead_ms:.2f} ms ({overhead_pct:.1f}%)")

# Summary
print()
print("="*80)
print("DQRM COMPARISON")
print("="*80)
print()
print("  Method         Compression  Accuracy Loss")
print("  " + "-"*45)
print("  DQRM INT8      4x           ~0.1%")
print("  DQRM PQ-64     64x          ~1.0%")
if base_acc and sketch_acc:
    print(f"  Your Sketch    221x         {rel_loss:.3f}%")
print()
print("="*80)
PYEOF

python3 "$OUTPUT_DIR/generate_report.py" "$OUTPUT_DIR" "$BASELINE_TRAIN_TIME" "$SKETCH_TRAIN_TIME" "$BASELINE_INF_TIME" "$SKETCH_INF_TIME" | tee "$OUTPUT_DIR/FINAL_REPORT.txt"

SCRIPT_END=$(date +%s)
TOTAL_TIME=$((SCRIPT_END - SCRIPT_START))
log_time "Script completed in $((TOTAL_TIME/3600))h $((TOTAL_TIME%3600/60))m"

echo ""
echo "================================================================================"
echo "COMPARISON COMPLETE!"
echo "================================================================================"
echo "Total time: $((TOTAL_TIME/3600))h $((TOTAL_TIME%3600/60))m"
echo "Report: $OUTPUT_DIR/FINAL_REPORT.txt"
echo ""

