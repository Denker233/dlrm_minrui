#!/bin/bash
cd /home/cc/expr/dlrm_minrui
python3 dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/home/cc/input/train.txt \
    --processed-data-file=/home/cc/input/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --nepochs=1 \
    --num-workers=0 \
    --print-freq=8192 \
    --test-freq=50000 \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --save-model="./models/dlrm_kaggle_correct.pt" \
    2>&1 | tee logs/dlrm_train_correct.log
