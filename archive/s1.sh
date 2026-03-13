#!/bin/bash
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi



bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 10
# Run on NUMA 0
# numactl --membind=0 python3 dlrm_s_pytorch.py \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" \
#   --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 \
#   --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 \
#   $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log >> combined_output_numa0.log 2>&1

# Run on NUMA 1
echo "# Run DLRM (NUMA 1) 10GB bandwidth" >> combined_output_numa1.log
./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1
# echo "# Run DLRM (NUMA 1) 10GB bandwidth with half memory" >> combined_output_numa1.log
# numactl --membind=1 python3 dlrm_s_pytorch.py\
#   --arch-embedding-size=1000-1000-10000-1000-1000-1000-1000-1000 \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-256-16" \
#   --arch-mlp-top="256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=64 \
#   --print-freq=4096 --print-time --test-mini-batch-size=163840 --test-num-workers=16 --dataset-multiprocessing\
#   --enable-profiling >> combined_output_numa1.log 2>&1

# numactl --membind=1 python3 dlrm_s_pytorch.py\
#   --arch-embedding-size=1000-1000-10000-1000-1000-1000-1000-1000 \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-256-16" \
#   --arch-mlp-top="256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=64 \
#   --print-freq=4096 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --dataset-multiprocessing\
#   --enable-profiling >> combined_output_numa1.log 2>&1

# numactl --physcpubind=1 --membind=1 python3 dlrm_s_pytorch.py \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" \
#   --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 \
#   --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 \
#   $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 1.91 GB of memory used (NUMA 1)" >> combined_output_numa1.log
############################################################
bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 8
# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" \
#   --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 \
#   --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 \
#   $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 19.072 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
echo "# Run DLRM (NUMA 1) 8GB bandwidth" >> combined_output_numa1.log
./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1
# echo "# Run DLRM (NUMA 1) 10GB bandwidth with half memory" >> combined_output_numa1.log
# numactl --membind=1 python3 dlrm_s_pytorch.py\
#   --arch-embedding-size=1000-1000-10000-1000-1000-1000-1000-1000 \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-256-16" \
#   --arch-mlp-top="256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=64 \
#   --print-freq=4096 --print-time --test-mini-batch-size=64 --test-num-workers=16 --dataset-multiprocessing\
#   --enable-profiling >> combined_output_numa1.log 2>&1
# numactl --physcpubind=1 --membind=1 python3 dlrm_s_pytorch.py \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" \
#   --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 \
#   --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 \
#   $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log >> combined_output_numa1.log 2>&1
# echo "# Embedding table size 19.072 GB of memory used (NUMA 1)" >> combined_output_numa1.log

bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 5
# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" \
#   --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 \
#   --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 \
#   $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 19.072 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
echo "# Run DLRM (NUMA 1) 5GB bandwidth" >> combined_output_numa1.log
./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1
# echo "# Run DLRM (NUMA 1) 10GB bandwidth with half memory" >> combined_output_numa1.log
# numactl --membind=1 python3 dlrm_s_pytorch.py\
#   --arch-embedding-size=1000-1000-10000-1000-1000-1000-1000-1000 \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-256-16" \
#   --arch-mlp-top="256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=64 \
#   --print-freq=4096 --print-time --test-mini-batch-size=64 --test-num-workers=16 --dataset-multiprocessing\
#   --enable-profiling >> combined_output_numa1.log 2>&1
# numactl --physcpubind=1 --membind=1 python3 dlrm_s_pytorch.py \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" \
#   --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 \
#   --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 \
#   $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log >> combined_output_numa1.log 2>&1
# echo "# Embedding table size 19.072 GB of memory used (NUMA 1)" >> combined_output_numa1.log

bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 3
# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" \
#   --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 \
#   --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 \
#   $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log >> combined_output_numa0.log 2>&1

# Run on NUMA 1
echo "# Run DLRM (NUMA 1) 3GB bandwidth" >> combined_output_numa1.log
./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1
# echo "# Run DLRM (NUMA 1) 10GB bandwidth with half memory" >> combined_output_numa1.log
# numactl --membind=1 python3 dlrm_s_pytorch.py \
#   --arch-embedding-size=1000-1000-10000-1000-1000-1000-1000-1000 \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-256-16" \
#   --arch-mlp-top="256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=64 \
#   --print-freq=4096 --print-time --test-mini-batch-size=64 --test-num-workers=16 --dataset-multiprocessing\
#   --enable-profiling >> combined_output_numa1.log 2>&1
# numactl --physcpubind=1 --membind=1 python3 dlrm_s_pytorch.py \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" \
#   --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 \
#   --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 \
#   $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 1.91 GB of memory used (NUMA 1)" >> combined_output_numa1.log

bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 1
# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" \
#   --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 \
#   --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 \
#   $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log >> combined_output_numa0.log 2>&1

# Run on NUMA 1
echo "# Run DLRM (NUMA 1) 1GB bandwidth" >> combined_output_numa0.log
./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1
# numactl --membind=1 python3 dlrm_s_pytorch.py \
#   --arch-embedding-size=1000-1000-10000-1000-1000-1000-1000-1000 \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-256-16" \
#   --arch-mlp-top="256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=64 \
#   --print-freq=4096 --print-time --test-mini-batch-size=64 --test-num-workers=16 --dataset-multiprocessing\
#   --enable-profiling >> combined_output_numa1.log 2>&1
# numactl --physcpubind=1 --membind=1 python3 dlrm_s_pytorch.py \
#   --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" \
#   --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle \
#   --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#   --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 \
#   --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 \
#   $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 1.91 GB of memory used (NUMA 1)" >> combined_output_numa1.log

