#!/usr/bin/env bash
# Local AlphaFold2-Multimer cross-check, not a hosted Colab session.
# Requires a C compiler for the accelerated JAX/Triton path.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/build/uv-cache-linux"
export UV_PYTHON_INSTALL_DIR="$PWD/build/python-linux"
uv_bin="${UV_BIN:-$PWD/build/linux-tools/uv-x86_64-unknown-linux-gnu/uv}"
"$uv_bin" venv build/env-colabfold-linux --python 3.11
"$uv_bin" pip install --python build/env-colabfold-linux/bin/python \
  'colabfold[alphafold]==1.6.3' 'jax[cuda12]==0.6.2'
# The research pipeline installs BoltzGen first; patch only optional dependency
# loading/WSL telemetry, never the predictor's forward calculation.
build/env-colabfold-linux/bin/python gpu/patch_linux_compat.py
