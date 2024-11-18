#!/bin/bash

mkdir -p dlrm_env

sudo apt update
sudo apt install -y python3.9-venv

python3 -m venv dlrm_env

source dlrm_env/bin/activate

export TMPDIR=$PWD/dlrm_env

#change the route to location of dlrm
cd "$PWD/dlrm_minrui"

pip3 install -r requirements.txt --no-cache-dir
pip3 install tensorboard mlperf_logging