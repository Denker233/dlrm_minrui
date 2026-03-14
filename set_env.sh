#!/bin/bash
set -e

mkdir -p dlrm_env

# Install venv support if missing
sudo apt update
sudo apt install -y python3-venv

# Create virtual environment
python3 -m venv dlrm_env

echo "Virtual environment created at dlrm_env/"
echo "Run: source dlrm_env/bin/activate"
