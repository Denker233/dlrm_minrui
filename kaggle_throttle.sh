# #!/bin/bash
# if [[ $# == 1 ]]; then
#     dlrm_extra_option=$1
# else
#     dlrm_extra_option=""
# fi


# #change the dir to run thr throttling module
# bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 10
# # Run on NUMA 1
# echo "# Run DLRM (NUMA 1) 10GB bandwidth" >> combined_output_numa1.log
# ./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1

# ############################################################
# bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 8
# # Run on NUMA 1
# echo "# Run DLRM (NUMA 1) 8GB bandwidth" >> combined_output_numa1.log
# ./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1

# ############################################################
# bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 5

# # Run on NUMA 1
# echo "# Run DLRM (NUMA 1) 5GB bandwidth" >> combined_output_numa1.log
# ./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1

# ############################################################
# bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 3

# # Run on NUMA 1
# echo "# Run DLRM (NUMA 1) 3GB bandwidth" >> combined_output_numa1.log
# ./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1

# ############################################################
# bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 1


# # Run on NUMA 1
# echo "# Run DLRM (NUMA 1) 1GB bandwidth" >> combined_output_numa0.log
# ./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1
frequencies=( 2.0 1.6 1.4 1 0.6 0.2)  # Frequencies in GHz

for freq in "${frequencies[@]}"; do
    # Apply CPU frequency to NUMA node 1 (CPUs 10-19, 30-39)
    echo "Setting CPU frequency of NUMA 1 CPUs (10-19, 30-39) to ${freq} GHz"
    sudo bash /users/mt1370/throttle/run_freq.sh 2.4 ${freq} 10



    # Run benchmark for the current CPU frequency
    echo "# Run DLRM (NUMA 1) at ${freq}GHz CPU frequency" >> combined_output_numa1_10GB.log
    echo "# Run DLRM (NUMA 1) at ${freq}GHz CPU frequency" >> numa_memory_consumption_10GB.log
    numactl --physcpubind=10-19,30-39 --membind=1 python3 dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
    --dataset-multiprocessing --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt \
    --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce \
    --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=8192 --print-time \
    --test-mini-batch-size=16384 --test-num-workers=16 --enable-profiling $dlrm_extra_option\
    >> combined_output_numa1_10GB.log 2>&1 &

    python_pid=$!
    bash ./input/memory_monitor.sh $python_pid
    echo "done" >> combined_output_numa1_10GB.log
done

