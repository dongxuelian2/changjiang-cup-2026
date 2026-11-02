#!/usr/bin/env bash
# Local Linux/WSL entry point. GPU jobs must run sequentially on the 8 GB device.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/build/huggingface"
export XDG_CACHE_HOME="$PWD/build/cache"
if [[ ! -x build/env-boltzgen-linux/bin/python ]]; then
  bash gpu/setup_boltzgen_linux.sh
fi
if [[ ! -f build/sources/2RE9.cif ]]; then
  build/env-boltzgen-linux/bin/python scripts/fetch_research_inputs.py
fi
if grep -qi microsoft /proc/version; then
  build/env-boltzgen-linux/bin/python gpu/patch_linux_compat.py
  export CUEQ_WSL_METADATA=1
fi
build/env-boltzgen-linux/bin/python scripts/run_boltzgen.py \
  --spec gpu/tl1a_boltzgen_beta.yaml --num-designs 4 --sequences-per-backbone 2 \
  --checkpoint diverse --sampling-steps 200 --seed 20261102 --use-kernels --log-sampling "$@"
