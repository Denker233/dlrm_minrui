#!/bin/bash
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi


#change the dir to run thr throttling module
#!/bin/bash

# Set up throttle for different bandwidth values
bandwidths=(10 8 5 3 1 0.8 0.6 0.4)

for bw in "${bandwidths[@]}"; do
    # Apply bandwidth throttling
    echo "Setting bandwidth to ${bw} GB"
    sudo bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 ${bw}


    # Run benchmark for the current bandwidth
    echo "# Run DLRM (NUMA 1) ${bw}GB bandwidth" >> combined_output_numa1_10GB.log
    echo "# Run DLRM (NUMA 1) ${bw}GB bandwidth" >> numa_memory_consumption_10GB.log
    # echo "# Embedding table size 1.91 GB of memory used (NUMA 1)" >> combined_output_numa1.log
    numactl --physcpubind=10-19,30-39 --membind=1 python3 dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
    --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt \
    --dataset-multiprocessing --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce \
    --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=8192 --print-time \
    --test-mini-batch-size=16384 --test-num-workers=16 $dlrm_extra_option\
    >> combined_output_numa1_10GB.log 2>&1 &
    python_pid=$!
    bash ./input/memory_monitor.sh $python_pid
    echo "done" >> combined_output_numa1_10GB.log

    # echo "# Embedding table size 19.072 GB of memory used (NUMA 1)" >> combined_output_numa1.log
    # numactl --membind=1 python3 dlrm_s_pytorch.py \
    # --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
    # --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt \
    # --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce \
    # --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=8192 --print-time \
    # --test-mini-batch-size=16384 --test-num-workers=16 --enable-profiling $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log\
    # >> combined_output_numa1.log 2>&1
    # echo "done" >> combined_output_numa1.log

    # echo "# Embedding table size 28.61 GB of memory used (NUMA 1)" >> combined_output_numa1.log
    # numactl --membind=1 python3 dlrm_s_pytorch.py \
    # --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
    # --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt \
    # --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce \
    # --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=8192 --print-time \
    # --test-mini-batch-size=16384 --test-num-workers=16 --enable-profiling $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log\
    # >> combined_output_numa1.log 2>&1
    # echo "done" >> combined_output_numa1.log

    # echo "# Embedding table size 38.15 GB of memory used (NUMA 1)" >> combined_output_numa1.log
    # numactl --membind=1 python3 dlrm_s_pytorch.py \
    # --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
    # --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt \
    # --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce \
    # --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=8192 --print-time \
    # --test-mini-batch-size=16384 --test-num-workers=16 --enable-profiling $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log\
    # >> combined_output_numa1.log 2>&1
    # echo "done" >> combined_output_numa1.log
done




# for bw in "${bandwidths[@]}"; do
#     # Apply bandwidth throttling
#     echo "Setting bandwidth to ${bw} GB"
#     sudo bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 ${bw}

#     # Run benchmark for the current bandwidth
#     echo "# Run DLRM (NUMA 1) ${bw}GB bandwidth" >> combined_output_numa1.log
#     echo "# Embedding table size 1.91 GB of memory used (NUMA 1)" >> combined_output_numa1.log
#     numactl --membind=1 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
#     --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt \
#     --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce \
#     --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=8192 --print-time \
#     --test-mini-batch-size=16384 --test-num-workers=16 --enable-profiling $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log\
#      >> combined_output_numa1.log 2>&1

#     echo "# Embedding table size 19.072 GB of memory used (NUMA 1)" >> combined_output_numa1.log
#     numactl --membind=1 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
#     --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt \
#     --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce \
#     --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=8192 --print-time \
#     --test-mini-batch-size=16384 --test-num-workers=16 --enable-profiling $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log\
#      >> combined_output_numa1.log 2>&1

#     echo "# Embedding table size 28.61 GB of memory used (NUMA 1)" >> combined_output_numa1.log
#     numactl --membind=1 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
#     --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt \
#     --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce \
#     --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=8192 --print-time \
#     --test-mini-batch-size=16384 --test-num-workers=16 --enable-profiling $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log\
#     >> combined_output_numa1.log 2>&1

#     echo "# Embedding table size 38.15 GB of memory used (NUMA 1)" >> combined_output_numa1.log
#     numactl --membind=1 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-20000000-20000000-20000000-20000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
#     --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt \
#     --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce \
#     --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=8192 --print-time \
#     --test-mini-batch-size=16384 --test-num-workers=16 --enable-profiling $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log\
#     >> combined_output_numa1.log 2>&1
# done
