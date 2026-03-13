#!/bin/bash

THREAD_COUNTS=(5 10 15 20 25 30 35 40)
BATCH_SIZES=(128 1280 12800)

for BATCH in "${BATCH_SIZES[@]}"; do
    for THREADS in "${THREAD_COUNTS[@]}"; do
        SCRIPT_NAME="run_dlrm_b${BATCH}_t${THREADS}.sh"
        cat <<EOF > "$SCRIPT_NAME"
#!/bin/bash
source dlrm_env/bin/activate
python3 dlrm_s_pytorch.py \\
    --thread-count=${THREADS} \\
    --mini-batch-size=${BATCH} \\
    --arch-sparse-feature-size=16 \\
    --arch-mlp-bot=13-512-256-64-16 \\
    --arch-mlp-top=512-256-1 \\
    --data-generation=dataset \\
    --data-set=kaggle \\
    --raw-data-file=./input/train.txt \\
    --dataset-multiprocessing \\
    --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \\
    --loss-function=bce \\
    --round-targets=True \\
    --learning-rate=0.1 \\
    --print-freq=8192 \\
    --print-time \\
    --test-mini-batch-size=16384 \\
    --test-num-workers=16 \\
    > output_b${BATCH}_t${THREADS}.log 2>&1
EOF
        chmod +x "$SCRIPT_NAME"
    done
done

echo "Scripts generated: run_dlrm_b*_t*.sh"
