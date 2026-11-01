#!/usr/bin/env bash
# Unexecuted Linux entry point, prepared after native Windows resolution failed.
# Use a free/new allocation, never a paid resource or an occupied Colab session.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/build/huggingface"
export XDG_CACHE_HOME="$PWD/build/cache"
python3.11 scripts/fetch_research_inputs.py
python3.11 -m venv build/env-boltzgen-linux
build/env-boltzgen-linux/bin/python -m pip install 'boltzgen==0.3.2'
# Keep the initial job small. These explicit steps generate actual structures and
# sequences, without claiming the unrun ESMFold2 evaluation is official screening.
build/env-boltzgen-linux/bin/boltzgen run gpu/tl1a_boltzgen.yaml \
  --output build/tl1a_boltzgen --protocol protein-anything \
  --num_designs 32 --steps design inverse_folding
