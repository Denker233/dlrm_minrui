#!/bin/bash
set -e

cd ~/expr/dlrm_minrui
source dlrm_env/bin/activate

echo "=== Installing PyTorch binary ==="
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
python -c "import torch; print('PyTorch:', torch.__version__)"

echo -e "\n=== Installing dependencies ==="
pip install numpy ninja pyyaml cffi typing-extensions future six requests

echo -e "\n=== Cloning IPEX ==="
if [ ! -d "intel-extension-for-pytorch" ]; then
    git clone --recursive -b v1.12.300 https://github.com/intel/intel-extension-for-pytorch
fi
cd intel-extension-for-pytorch

export IPEX_PATH=$(pwd)
export IPEX_KERNEL_PATH=$IPEX_PATH/intel_extension_for_pytorch/csrc/aten/cpu/kernels

echo -e "\n=== Verifying kernel files ==="
ls -la $IPEX_KERNEL_PATH/EmbeddingBagKrnl.cpp

echo -e "\n=== Building IPEX (15-30 minutes) ==="
USE_CUDA=0 \
USE_NATIVE_ARCH=1 \
REL_WITH_DEB_INFO=1 \
MAX_JOBS=8 \
python setup.py install 2>&1 | tee ~/expr/dlrm_minrui/ipex_build.log

echo -e "\n=== Verifying installation ==="
python -c "import intel_extension_for_pytorch as ipex; print('✓ IPEX installed:', ipex.__version__)"

echo -e "\n=== Setup complete! ==="
echo "IPEX_PATH: $IPEX_PATH"
echo "IPEX_KERNEL_PATH: $IPEX_KERNEL_PATH"