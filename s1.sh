#!/bin/bash
# Define the extra option passed to the script, if any
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=“”
fi
# GPU configuration
gpu=1
ngpus=“1"  # Adjust this value if you want to run on multiple GPUs, e.g., “1 2 4 8”
# Model parameters specific to your command
mb_size=2
data_size=6
numa_cmd=“”  # Empty since not using specific CPU binding for GPUs
dlrm_pt_bin=“python dlrm_s_pytorch.py”
# GPU Benchmarking
if [ $gpu = 1 ]; then
  echo “--------------------------------------------”
  echo “GPU Benchmarking - running on $ngpus GPUs”
  echo “--------------------------------------------”
  for _ng in $ngpus
  do
    # strong scaling, keeping batch size fixed
    _mb_size=$((mb_size*1))
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg=“CUDA_VISIBLE_DEVICES=$_gpus”
    echo “-------------------”
    echo “Using GPUS: “$_gpus
    echo “-------------------”
    # Running the PyTorch model on GPU
    outf=“model_GPU_PT_$_ng.log”
    outp=“dlrm_s_pytorch.prof”
    echo “-------------------------------”
    echo “Running PT (log file: $outf)”
    echo “-------------------------------”
    # Command to run your specific setup on GPU
    cmd=“$cuda_arg $dlrm_pt_bin --mini-batch-size=$_mb_size --data-size=$data_size --use-gpu $dlrm_extra_option > $outf”
    echo $cmd
    eval $cmd
    # Extract minimum iteration time from the log
    min=$(grep “iteration” $outf | awk ‘BEGIN{best=999999} {if (best > $7) best=$7} END{print best}’)
    echo “Min time per iteration = $min”
    # Move profiling file(s)
    mv $outp ${outf//“.log”/“.prof”}
    mv ${outp//“.prof”/“.json”} ${outf//“.log”/“.json”}
  done
fi
