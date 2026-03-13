#!/bin/bash
set -x

INSTALL_VTUNE(){
    yes | wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    yes | sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
    yes | sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
    yes | sudo apt-get update
    yes | sudo apt-get install -y libnss3-dev libgdk-pixbuf2.0-dev libgtk-3-dev libxss-dev
    yes | sudo apt-get install -y intel-oneapi-vtune

    sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
    sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'
}


SETUP_VTUNE(){
    local bashrc="$HOME/.bashrc"
    local vtune_bin_export="export VTUNE_BIN=/opt/intel/oneapi/vtune/latest/bin64"
    local vtune_amplxe_export="export VTUNE_AMPLXE=amplxe-cl"
    local vtune_path_export="export PATH=\$VTUNE_BIN:\$PATH"
    local echo_perf="sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'"
    local echo_kptr="sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'"
    local drop_cache="alias drop_cache='sudo sh -c \"echo 3 > /proc/sys/vm/drop_caches; sync\"'"
    local SUDO="alias SUDO='sudo'"

    # Check if the export statements already exist in ~/.bashrc
    if grep -q "$vtune_bin_export" "$bashrc" && \
       grep -q "$vtune_amplxe_export" "$bashrc" && \
       grep -q "$vtune_path_export" "$bashrc" && \
       grep -q "$echo_perf" "$bashrc" && \
       grep -q "$echo_kptr" "$bashrc" && \
       grep -q "$drop_cache" "$bashrc" && \
       grep -q "$SUDO" "$bashrc"; then
        echo "VTune environment variables and aliases already exist in $bashrc"
    else
        # Add things to ~/.bashrc if they don't exist
        echo -e "\n# Set up VTune environment variables" >> "$bashrc"
        echo "$vtune_bin_export" >> "$bashrc"
        echo "$vtune_amplxe_export" >> "$bashrc"
        echo "$vtune_path_export" >> "$bashrc"
        echo "$echo_perf" >> "$bashrc"
        echo "$echo_kptr" >> "$bashrc"
        echo "$drop_cache" >> "$bashrc"
        echo "$SUDO" >> "$bashrc"
        echo "VTune environment variables and aliases added to $bashrc"
    fi
}

INSTALL_SYSTEM_LIBS(){
    yes | sudo apt-get update
    yes | sudo apt install -y htop
    yes | sudo apt-get install -y liblapack-dev
    yes | sudo apt-get install -y libatlas-base-dev
    yes | sudo apt install -y numactl
    yes | sudo apt install -y libnuma-dev
    yes | sudo apt install -y libelf-dev
    yes | sudo apt-get install -y cpufrequtils

    yes | sudo apt install -y python-pip
    yes | sudo apt install -y python3-pip
    yes | sudo apt install python3-pip
    yes | sudo apt-get install -y libncurses-dev
    yes | sudo apt-get install -y git
    yes | sudo apt-get install -y software-properties-common
    yes | sudo apt-get install -y python3-software-properties
    yes | sudo apt-get install -y python-software-properties
    yes | sudo apt-get install -y unzip
    yes | sudo apt-get install -y python-setuptools python-dev build-essential
    yes | sudo easy_install -y pip
    yes | sudo apt install -y python-pip
    yes | sudo pip install zplot
    yes | sudo apt-get install -y numactl
    yes | sudo apt-get install -y libnuma-dev
    yes | sudo apt-get install -y cmake
    yes | sudo apt-get install -y build-essential
    yes | sudo apt-get install -y libboost-dev
    yes | sudo apt-get install -y libboost-thread-dev
    yes | sudo apt-get install -y libboost-system-dev
    yes | sudo apt-get install -y libboost-program-options-dev
    yes | sudo apt-get install -y libconfig-dev
    yes | sudo apt-get install -y uthash-dev
    yes | sudo apt-get install -y cscope
    yes | sudo apt-get install -y msr-tools
    yes | sudo apt-get install -y msrtool
    yes | sudo pip install -y psutil
    yes | sudo apt-get install -y libmpich-dev
    yes | sudo apt-get install -y libzstd-dev
    yes | sudo apt-get install -y liblz4-dev
    yes | sudo apt-get install -y libsnappy-dev
    yes | sudo apt-get install -y libssl-dev
    yes | sudo apt-get install -y libgflags-dev
    yes | sudo apt-get install -y zlib1g-dev
    yes | sudo apt-get install -y libbz2-dev
    yes | sudo apt-get install -y libevent-dev
    yes | sudo apt-get install -y systemd
    yes | sudo apt-get install -y libaio*
    yes | sudo apt-get install -y software-properties-common
    yes | sudo apt-get install -y libjemalloc-dev
    yes | pip install matplotlib
    yes | pip3 install future numpy torch scikit-learn pydot torchviz tensorboard packaging tqdm
}

INSTALL_SYSTEM_LIBS
INSTALL_VTUNE
sleep 5
SETUP_VTUNE
source ~/.bashrc
echo "The ~/.bashrc has been updated and sourced."
exit
