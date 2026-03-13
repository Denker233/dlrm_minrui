#!/bin/bash
# Train DLRM on Criteo Terabyte (4 days subset)
# Days 0-2: training, Day 3: test
cd /home/cc/expr/dlrm_minrui

export CRITEO_DAYS=4

python3 dlrm_s_pytorch.py \
    --arch-sparse-feature-size=64 \
    --arch-mlp-bot="13-512-256-64" \
    --arch-mlp-top="512-512-256-1" \
    --max-ind-range=10000000 \
    --data-generation=dataset \
    --data-set=terabyte \
    --raw-data-file=/home/cc/input/terabyte/day \
    --processed-data-file=/home/cc/input/terabyte/terabyte_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=2048 \
    --nepochs=1 \
    --num-workers=0 \
    --print-freq=2048 \
    --print-time \
    --test-freq=102400 \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --save-model="./models/dlrm_terabyte_4day.pt" \
    2>&1 | tee logs/terabyte_train_4day.log
