bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 10
# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 1.91 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 1.91 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 19.072 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 19.072 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 28.61 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=20000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 28.61 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-20000000-20000000-20000000-20000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 38.15 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=20000000-20000000-20000000-20000000-20000000-20000000-20000000-20000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 38.15 GB of memory used (NUMA 1)" >> combined_output_numa1.log

#########################################################################################
bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 7

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 1.91 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 1.91 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 19.072 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 19.072 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 28.61 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=20000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 28.61 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-20000000-20000000-20000000-20000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 38.15 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=20000000-20000000-20000000-20000000-20000000-20000000-20000000-20000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 38.15 GB of memory used (NUMA 1)" >> combined_output_numa1.log

#########################################################################################
bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 5

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 1.91 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 1.91 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 19.072 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 19.072 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 28.61 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=20000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 28.61 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-20000000-20000000-20000000-20000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 38.15 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=20000000-20000000-20000000-20000000-20000000-20000000-20000000-20000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 38.15 GB of memory used (NUMA 1)" >> combined_output_numa1.log


#########################################################################################
bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 3

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 1.91 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 1.91 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 19.072 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 19.072 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 28.61 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=20000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 28.61 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-20000000-20000000-20000000-20000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 38.15 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=20000000-20000000-20000000-20000000-20000000-20000000-20000000-20000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 38.15 GB of memory used (NUMA 1)" >> combined_output_numa1.log



#########################################################################################
bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 1

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 1.91 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 1.91 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 19.072 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 19.072 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 28.61 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=20000000-20000000-20000000-20000000-10000000-10000000-10000000-10000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 28.61 GB of memory used (NUMA 1)" >> combined_output_numa1.log

# Run on NUMA 0
# numactl --physcpubind=0 --membind=0 python3 dlrm_s_pytorch.py \
#     --arch-embedding-size=20000000-20000000-20000000-20000000-20000000-20000000-20000000-20000000 \
#     --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
#     --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
#     --print-freq=200 --print-time --enable-profiling >> combined_output_numa0.log 2>&1
# echo "# Embedding table size 38.15 GB of memory used (NUMA 0)" >> combined_output_numa0.log

# Run on NUMA 1
numactl --membind=1 python3 dlrm_s_pytorch.py \
    --arch-embedding-size=20000000-20000000-20000000-20000000-20000000-20000000-20000000-20000000 \
    --arch-sparse-feature-size=16 --arch-mlp-bot="512-512-16" --arch-mlp-top="1024-1024-1024-1" \
    --data-generation=random --mini-batch-size=256 --num-batches=1000 --num-indices-per-lookup=100 \
    --print-freq=200 --print-time --enable-profiling >> combined_output_numa1.log 2>&1
echo "# Embedding table size 38.15 GB of memory used (NUMA 1)" >> combined_output_numa1.log









