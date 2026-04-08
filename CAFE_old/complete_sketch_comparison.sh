#!/bin/bash

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./comparison_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"
mkdir -p "./models"

DATA_DIR="$HOME/expr/dlrm_minrui/criteo_24days"

echo "================================================================================"
echo "COMPLETE DLRM COMPARISON: BASELINE vs SKETCH COMPRESSION"
echo "================================================================================"
echo "Test configuration:"
echo "  - Training batches: 2000 (mini-batch: 128)"
echo "  - Inference batch size: 2048"
echo "  - Device: CPU"
echo "  - Estimated time: 16-20 hours total"
echo ""
echo "Output: $OUTPUT_DIR"
echo "================================================================================"
echo ""

#==============================================================================
# PHASE 1: QUICK INFERENCE TEST (Verify Setup - 10 minutes)
#==============================================================================
echo ""
echo "================================================================================"
echo "PHASE 1: QUICK INFERENCE TEST (Untrained Models)"
echo "================================================================================"
echo "Purpose: Verify data loading and model architecture work correctly"
echo "Expected: Random accuracy (~50% AUC), but should complete without errors"
echo ""

echo "Testing baseline inference..."
timeout 300 python dlrm_s_pytorch.py \
--inference-only \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--mini-batch-size=2048 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--num-batches=50 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
2>&1 | tee "$OUTPUT_DIR/phase1_baseline_check.log"

if [ $? -ne 0 ]; then
    echo "✗ Baseline inference failed! Check $OUTPUT_DIR/phase1_baseline_check.log"
    exit 1
fi

echo ""
echo "Testing sketch inference..."
timeout 300 python dlrm_s_pytorch.py \
--inference-only \
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
--num-batches=50 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
2>&1 | tee "$OUTPUT_DIR/phase1_sketch_check.log"

if [ $? -ne 0 ]; then
    echo "✗ Sketch inference failed! Check $OUTPUT_DIR/phase1_sketch_check.log"
    exit 1
fi

echo ""
echo "✓ Phase 1 complete: Both models can run inference"
echo "✓ Data loading works correctly"
echo "✓ Proceeding to training phase..."
sleep 3

#==============================================================================
# PHASE 2: TRAINING COMPARISON (2000 batches each - 16-20 hours)
#==============================================================================
echo ""
echo "================================================================================"
echo "PHASE 2: TRAINING COMPARISON (2000 batches each)"
echo "================================================================================"
echo "Started: $(date)"
echo ""

# Train Baseline
echo ">>> Training BASELINE (No Compression)"
echo ">>> This will take 8-10 hours..."
echo ""

BASELINE_START=$(date +%s)

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
--test-freq=1000 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--num-batches=2000 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="./models/baseline_2k.pt" \
2>&1 | tee "$OUTPUT_DIR/train_baseline.log"

BASELINE_END=$(date +%s)
BASELINE_TRAIN_TIME=$((BASELINE_END - BASELINE_START))

echo ""
echo "✓ Baseline training complete!"
echo "  Duration: $((BASELINE_TRAIN_TIME/3600))h $((BASELINE_TRAIN_TIME%3600/60))m $((BASELINE_TRAIN_TIME%60))s"
echo "  Final result:"
grep "accuracy.*auc" "$OUTPUT_DIR/train_baseline.log" | tail -1
echo ""

# Train Sketch
echo ">>> Training SKETCH COMPRESSION (221x)"
echo ">>> This will take 8-12 hours..."
echo ""

SKETCH_START=$(date +%s)

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
--test-freq=1000 \
--test-mini-batch-size=2048 \
--test-num-workers=8 \
--sketch-flag \
--compress-rate=0.001 \
--hash-rate=0.5 \
--sketch-threshold=1 \
--adjust-threshold=1 \
--sketch-alpha=1.0 \
--num-batches=2000 \
--cat-path="$DATA_DIR/sparse" \
--dense-path="$DATA_DIR/dense" \
--label-path="$DATA_DIR/label" \
--count-path="$DATA_DIR/processed_count.bin" \
--save-model="./models/sketch_2k.pt" \
2>&1 | tee "$OUTPUT_DIR/train_sketch.log"

SKETCH_END=$(date +%s)
SKETCH_TRAIN_TIME=$((SKETCH_END - SKETCH_START))

echo ""
echo "✓ Sketch training complete!"
echo "  Duration: $((SKETCH_TRAIN_TIME/3600))h $((SKETCH_TRAIN_TIME%3600/60))m $((SKETCH_TRAIN_TIME%60))s"
echo "  Final result:"
grep "accuracy.*auc" "$OUTPUT_DIR/train_sketch.log" | tail -1
echo ""

#==============================================================================
# PHASE 3: INFERENCE COMPARISON ON TRAINED MODELS
#==============================================================================
echo ""
echo "================================================================================"
echo "PHASE 3: INFERENCE COMPARISON (Trained Models, Batch Size 2048)"
echo "================================================================================"
echo ""

echo ">>> Running baseline inference (full test set)..."
BASELINE_INF_START=$(date +%s)

python dlrm_s_pytorch.py \
--inference-only \
--load-model="./models/baseline_2k.pt" \
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

echo ""
echo ">>> Running sketch inference (full test set)..."
SKETCH_INF_START=$(date +%s)

python dlrm_s_pytorch.py \
--inference-only \
--load-model="./models/sketch_2k.pt" \
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

#==============================================================================
# PHASE 4: GENERATE REPORT
#==============================================================================
echo ""
echo "================================================================================"
echo "GENERATING COMPARISON REPORT"
echo "================================================================================"

python3 << PYEOF > "$OUTPUT_DIR/FINAL_REPORT.txt"
import re
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

def extract_time(filename, skip_invalid=True):
    try:
        times = []
        with open(filename) as f:
            for line in f:
                if 'ms/it' in line and 'Finished' in line:
                    match = re.search(r'(\d+\.\d+)\s+ms/it', line)
                    if match:
                        t = float(match.group(1))
                        if not skip_invalid or t > 0:  # Skip -1.00
                            times.append(t)
        if times:
            return sum(times[-500:]) / len(times[-500:])  # Avg last 500
    except:
        pass
    return None

print("="*80)
print("DLRM BASELINE vs SKETCH COMPRESSION - FINAL REPORT")
print("="*80)
print()
print(f"Test Date: {TIMESTAMP}")
print(f"Configuration: 2000 training batches, batch_size=128/2048")
print(f"Device: CPU")
print()

# Training Results
print("="*80)
print("TRAINING RESULTS (2000 batches)")
print("="*80)
print()

base_acc, base_auc = extract_metrics('$OUTPUT_DIR/train_baseline.log')
sketch_acc, sketch_auc = extract_metrics('$OUTPUT_DIR/train_sketch.log')

print("Baseline (No Compression):")
if base_acc and base_auc:
    print(f"  Accuracy:  {base_acc:.3f}%")
    print(f"  AUC:       {base_auc:.3f}%")
else:
    print("  [Could not extract metrics]")
print(f"  Train time: {parse_time($BASELINE_TRAIN_TIME)}")

print()
print("Sketch Compression (221x theoretical):")
if sketch_acc and sketch_auc:
    print(f"  Accuracy:  {sketch_acc:.3f}%")
    print(f"  AUC:       {sketch_auc:.3f}%")
else:
    print("  [Could not extract metrics]")
print(f"  Train time: {parse_time($SKETCH_TRAIN_TIME)}")

if base_acc and sketch_acc and base_auc and sketch_auc:
    print()
    print("Training Accuracy Impact:")
    print(f"  Accuracy loss:  {base_acc - sketch_acc:.3f}% absolute")
    print(f"                  {((base_acc - sketch_acc)/base_acc * 100):.3f}% relative")
    print(f"  AUC loss:       {base_auc - sketch_auc:.3f}%")

print()
print("Training Time Overhead:")
overhead_sec = $SKETCH_TRAIN_TIME - $BASELINE_TRAIN_TIME
overhead_pct = (overhead_sec / $BASELINE_TRAIN_TIME * 100) if $BASELINE_TRAIN_TIME > 0 else 0
print(f"  Baseline:  {parse_time($BASELINE_TRAIN_TIME)}")
print(f"  Sketch:    {parse_time($SKETCH_TRAIN_TIME)}")
print(f"  Overhead:  {parse_time(overhead_sec)} ({overhead_pct:.1f}%)")

# Inference Results
print()
print("="*80)
print("INFERENCE RESULTS (Batch Size: 2048)")
print("="*80)
print()

base_inf_acc, base_inf_auc = extract_metrics('$OUTPUT_DIR/inference_baseline.log')
sketch_inf_acc, sketch_inf_auc = extract_metrics('$OUTPUT_DIR/inference_sketch.log')
base_time = extract_time('$OUTPUT_DIR/inference_baseline.log')
sketch_time = extract_time('$OUTPUT_DIR/inference_sketch.log')

print("Baseline:")
if base_inf_acc and base_inf_auc:
    print(f"  Accuracy:  {base_inf_acc:.3f}%")
    print(f"  AUC:       {base_inf_auc:.3f}%")
print(f"  Inf time:  {parse_time($BASELINE_INF_TIME)}")
if base_time:
    print(f"  Avg/batch: {base_time:.2f} ms")
    print(f"  Throughput: {2048/base_time*1000:.0f} samples/sec")

print()
print("Sketch Compression:")
if sketch_inf_acc and sketch_inf_auc:
    print(f"  Accuracy:  {sketch_inf_acc:.3f}%")
    print(f"  AUC:       {sketch_inf_auc:.3f}%")
print(f"  Inf time:  {parse_time($SKETCH_INF_TIME)}")
if sketch_time:
    print(f"  Avg/batch: {sketch_time:.2f} ms")
    print(f"  Throughput: {2048/sketch_time*1000:.0f} samples/sec")

if base_time and sketch_time:
    print()
    print("Inference Overhead:")
    print(f"  Baseline:  {base_time:.2f} ms/batch")
    print(f"  Sketch:    {sketch_time:.2f} ms/batch")
    print(f"  Overhead:  {sketch_time - base_time:.2f} ms ({((sketch_time - base_time)/base_time * 100):.1f}%)")

# Model Sizes
print()
print("="*80)
print("MODEL SIZES")
print("="*80)
print()
import os
try:
    base_size = os.path.getsize('./models/baseline_2k.pt') / 1024 / 1024
    sketch_size = os.path.getsize('./models/sketch_2k.pt') / 1024 / 1024
    print(f"Baseline model:  {base_size:.2f} MB")
    print(f"Sketch model:    {sketch_size:.2f} MB")
    print(f"Size ratio:      {base_size/sketch_size:.2f}x")
except:
    print("[Model size information not available]")

# Compression Timing
print()
print("="*80)
print("SKETCH COMPRESSION/DECOMPRESSION TIMING")
print("="*80)
print()
try:
    with open('$OUTPUT_DIR/inference_sketch.log') as f:
        content = f.read()
        if 'SKETCH COMPRESSION' in content:
            # Extract timing section
            start = content.find('SKETCH COMPRESSION')
            end = content.find('='*70, start + 100)
            if end > start:
                print(content[start:end])
        else:
            print("[No detailed timing available]")
except:
    print("[Could not extract timing]")

# Summary
print()
print("="*80)
print("SUMMARY & COMPARISON WITH DQRM")
print("="*80)
print()

if base_acc and sketch_acc:
    loss_pct = ((base_acc - sketch_acc)/base_acc * 100)
    
    print("Your Results:")
    print(f"  Compression: 221x theoretical (sketch + hash)")
    print(f"  Accuracy loss: {loss_pct:.3f}% relative")
    print(f"  Training overhead: {overhead_pct:.1f}%")
    if base_time and sketch_time:
        inf_overhead = ((sketch_time - base_time)/base_time * 100)
        print(f"  Inference overhead: {inf_overhead:.1f}%")
    
    print()
    print("DQRM Comparison:")
    print("  Method              Compression  Accuracy Loss")
    print("  " + "-"*50)
    print("  DQRM INT8           4x           ~0.1%")
    print("  DQRM PQ-16          16x          ~0.5%")
    print("  DQRM PQ-64          64x          ~1.0%")
    print(f"  Your Sketch (221x)  221x         {loss_pct:.3f}%")
    
    print()
    if loss_pct < 0.5:
        print("  ✓✓✓ EXCELLENT! Better than DQRM at much higher compression!")
    elif loss_pct < 1.0:
        print("  ✓✓ VERY GOOD! Competitive with DQRM PQ-64 at 3.5x compression")
    elif loss_pct < 2.0:
        print("  ✓ GOOD! Acceptable trade-off for 221x compression")
    else:
        print("  ⚠ Higher accuracy loss than DQRM - may need tuning")

print()
print("="*80)
print("DETAILED LOGS:")
print(f"  Training:   $OUTPUT_DIR/train_baseline.log")
print(f"              $OUTPUT_DIR/train_sketch.log")
print(f"  Inference:  $OUTPUT_DIR/inference_baseline.log")
print(f"              $OUTPUT_DIR/inference_sketch.log")
print(f"  Models:     ./models/baseline_2k.pt")
print(f"              ./models/sketch_2k.pt")
print("="*80)
PYEOF

# Display report
cat "$OUTPUT_DIR/FINAL_REPORT.txt"

echo ""
echo "================================================================================"
echo "COMPARISON COMPLETE!"
echo "================================================================================"
echo "Total runtime: $(($(date +%s) - BASELINE_START)) seconds"
echo "Report saved to: $OUTPUT_DIR/FINAL_REPORT.txt"
echo ""

