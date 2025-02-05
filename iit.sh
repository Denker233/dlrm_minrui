# launch.sh

#!/bin/bash

# Retrieve the directory path where the path contains both the sample.py and launch.sh so that this bash script can be invoked from any directory
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# <Activate a Python environment>
cd ${BASEFOLDER}
source ./dlrm_env/bin/activate
# python3 dlrm_iit_pytorch.py --mini-batch-size=2 --data-size=6 --debug-mode
python3 dlrm_iit_pytorch.py \
    --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
    --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt \
    --dataset-multiprocessing --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce \
    --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=8192 --print-time \
    --test-mini-batch-size=16384 --test-num-workers=16
# python3 dlrm_iit_pytorch.py --mini-batch-size=2 --data-size=6 
