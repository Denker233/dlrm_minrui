# launch.sh

#!/bin/bash

# Retrieve the directory path where the path contains both the sample.py and launch.sh so that this bash script can be invoked from any directory
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# <Activate a Python environment>
cd ${BASEFOLDER}
source ./dlrm_env/bin/activate
python3 dlrm_iit_pytorch.py --mini-batch-size=2 --data-size=6 --debug-mode
