#!/bin/bash
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi


#change the dir to run thr throttling module
bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 10
# Run on NUMA 1
echo "# Run DLRM (NUMA 1) 10GB bandwidth" >> combined_output_numa1.log
./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1

############################################################
bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 8
# Run on NUMA 1
echo "# Run DLRM (NUMA 1) 8GB bandwidth" >> combined_output_numa1.log
./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1

############################################################
bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 5

# Run on NUMA 1
echo "# Run DLRM (NUMA 1) 5GB bandwidth" >> combined_output_numa1.log
./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1

############################################################
bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 3

# Run on NUMA 1
echo "# Run DLRM (NUMA 1) 3GB bandwidth" >> combined_output_numa1.log
./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1

############################################################
bash /users/mt1370/expr/throttle/run_freq.sh 2.4 2.4 1


# Run on NUMA 1
echo "# Run DLRM (NUMA 1) 1GB bandwidth" >> combined_output_numa0.log
./bench/dlrm_s_criteo_kaggle.sh >> combined_output_numa1.log 2>&1

