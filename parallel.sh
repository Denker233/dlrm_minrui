#!/bin/bash

# Base command-line options (excluding mini-batch and thread count)
COMMON_OPTIONS="--arch-sparse-feature-size=16 \
--arch-mlp-bot=13-512-256-64-16 \
--arch-mlp-top=512-256-1 \
--data-generation=dataset \
--data-set=kaggle \
--raw-data-file=./input/train.txt \
--dataset-multiprocessing \
--processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--mini-batch-size=128000 \
--print-freq=8192 \
--print-time \
--test-mini-batch-size=16384 \
--test-num-workers=16"

# Function to clear OS cache
clear_cache() {
    echo "Clearing OS cache..."
    sync
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
}

# Optional extra options
EXTRA_OPTIONS=${dlrm_extra_option:-""}

LOG_FILE="combined_output.log"
# > $LOG_FILE  # Clear previous log

THREAD_COUNTS=(5 10 20 25 32 40)

for THREADS in "${THREAD_COUNTS[@]}"; do
    echo "===============================================" | tee -a $LOG_FILE
    echo "Running with OMP_NUM_THREADS=$THREADS" | tee -a $LOG_FILE
    echo "===============================================" | tee -a $LOG_FILE

    echo -e "\n=== RUNNING dlrm_para_pytorch.py ===" | tee -a $LOG_FILE
    python3 dlrm_para_pytorch.py --thread-count=$THREADS $COMMON_OPTIONS $EXTRA_OPTIONS >> $LOG_FILE 2>&1

    echo -e "\n=== RUNNING dlrm_s_pytorch.py ===" | tee -a $LOG_FILE
    python3 dlrm_s_pytorch.py --thread-count=$THREADS $COMMON_OPTIONS $EXTRA_OPTIONS >> $LOG_FILE 2>&1
done

# COMMON_OPTIONS="--arch-sparse-feature-size=16 \
# --arch-mlp-bot=13-512-256-64-16 \
# --arch-mlp-top=512-256-1 \
# --data-generation=dataset \
# --data-set=kaggle \
# --raw-data-file=./input/train.txt \
# --dataset-multiprocessing \
# --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
# --loss-function=bce \
# --round-targets=True \
# --learning-rate=0.1 \
# --mini-batch-size=128 \
# --print-freq=8192 \
# --print-time \
# --test-mini-batch-size=16384 \
# --test-num-workers=16"

# for THREADS in "${THREAD_COUNTS[@]}"; do
#     echo "===============================================" | tee -a $LOG_FILE
#     echo "Running with OMP_NUM_THREADS=$THREADS" | tee -a $LOG_FILE
#     echo "===============================================" | tee -a $LOG_FILE

#     echo -e "\n=== RUNNING dlrm_para_pytorch.py ===" | tee -a $LOG_FILE
#     clear_cache
#     OMP_NUM_THREADS=$THREADS python3 dlrm_para_pytorch.py $COMMON_OPTIONS $EXTRA_OPTIONS >> $LOG_FILE 2>&1

#     echo -e "\n=== RUNNING dlrm_s_pytorch.py ===" | tee -a $LOG_FILE
#     clear_cache
#     OMP_NUM_THREADS=$THREADS python3 dlrm_s_pytorch.py $COMMON_OPTIONS $EXTRA_OPTIONS >> $LOG_FILE 2>&1
# done
# echo "Execution finished. Combined log saved to $LOG_FILE"
