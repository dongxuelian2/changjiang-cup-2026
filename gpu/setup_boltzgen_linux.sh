#!/usr/bin/env bash
# Isolated WSL/Linux environment for the upstream CUDA acceleration path.
# Requires a C compiler (Ubuntu build-essential) and a working download proxy/network.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/build/uv-cache-linux"
export UV_PYTHON_INSTALL_DIR="$PWD/build/python-linux"
uv_bin="${UV_BIN:-$PWD/build/linux-tools/uv-x86_64-unknown-linux-gnu/uv}"
"$uv_bin" venv build/env-boltzgen-linux --python 3.11
"$uv_bin" pip install --python build/env-boltzgen-linux/bin/python \
  'torch==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
"$uv_bin" pip install --python build/env-boltzgen-linux/bin/python \
  'boltzgen==0.3.2' 'pytorch-lightning==2.5.5' 'einx==0.3.0' \
  'numpy==2.0.2' 'numba==0.61.0' \
  'cuequivariance-ops-cu12==0.5.1' 'cuequivariance-ops-torch-cu12==0.5.1' \
  'cuequivariance-torch==0.5.1'
build/env-boltzgen-linux/bin/python -c \
  'import torch,boltzgen; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))'
