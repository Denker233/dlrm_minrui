#!/bin/bash
# Orchestration: wait for D=64 model, then run all experiments in sequence
set -e

MODEL=./models/dlrm_kaggle_d64.pt
LOG_DIR=results/dct_domain
mkdir -p "$LOG_DIR"

echo "[$(date +%H:%M:%S)] Waiting for $MODEL ..."
while [ ! -f "$MODEL" ]; do
    sleep 30
done
# Wait until file size is stable (training has finished writing)
prev_size=0
while true; do
    size=$(stat -c %s "$MODEL" 2>/dev/null || echo 0)
    if [ "$size" -eq "$prev_size" ] && [ "$size" -gt 100000000 ]; then
        break
    fi
    prev_size=$size
    sleep 15
done
echo "[$(date +%H:%M:%S)] Model ready: $(ls -lh $MODEL | awk '{print $5}')"

echo "[$(date +%H:%M:%S)] Running DC block-size sweep..."
PYTHONUNBUFFERED=1 /usr/bin/python3 -u bench_kaggle_d64_dc_blocksize.py 2>&1 \
    | tee $LOG_DIR/d64_dc_sweep.log

echo "[$(date +%H:%M:%S)] Running pruning sweep..."
PYTHONUNBUFFERED=1 /usr/bin/python3 -u bench_kaggle_d64_pruning.py 2>&1 \
    | tee $LOG_DIR/d64_pruning.log

echo "[$(date +%H:%M:%S)] Running tail-metric experiment..."
PYTHONUNBUFFERED=1 /usr/bin/python3 -u bench_kaggle_d64_tail.py 2>&1 \
    | tee $LOG_DIR/d64_tail.log

echo "[$(date +%H:%M:%S)] All D=64 experiments complete."
