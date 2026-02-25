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
    --learning-rate=0.01 \
    --mini-batch-size=2048 \
    --nepochs=1 \
    --num-workers=0 \
    --print-freq=1000 \
    --test-freq=10000 \
    --save-model="./models/dlrm_kaggle_1epoch_new.pt" \
    2>&1 | tee logs/dlrm_train_1epoch.log
