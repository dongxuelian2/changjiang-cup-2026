#!/usr/bin/env bash
# Local ColabDesign experiment; no hosted Colab or PyRosetta dependency.
set -euo pipefail
cd "$(dirname "$0")/.."
revision=8fd167defd1b0f98e29f00584ff4f3987e6935dc
if [[ ! -d build/models/ColabDesign/.git ]]; then
  git clone https://github.com/sokrypton/ColabDesign.git build/models/ColabDesign
  git -C build/models/ColabDesign checkout --detach "$revision"
fi
[[ "$(git -C build/models/ColabDesign rev-parse HEAD)" == "$revision" ]]
if [[ ! -x build/env-colabfold-linux/bin/python ]]; then
  bash gpu/setup_colabfold_linux.sh
fi
export UV_CACHE_DIR="$PWD/build/uv-cache-linux"
uv_bin="${UV_BIN:-$PWD/build/linux-tools/uv-x86_64-unknown-linux-gnu/uv}"
"$uv_bin" pip install --python build/env-colabfold-linux/bin/python \
  -e build/models/ColabDesign 'jax[cuda12]==0.6.2' 'dm-haiku==0.0.17' \
  'chex==0.1.90' 'optax==0.2.5' 'joblib==1.6.0' 'cloudpickle==3.1.2'
for model in 3 1; do
  if [[ ! -f build/models/colabfold/params/params_model_${model}_multimer_v3.source.json ]]; then
    build/env-colabfold-linux/bin/python scripts/fetch_af2_model.py --model "$model"
  fi
done
