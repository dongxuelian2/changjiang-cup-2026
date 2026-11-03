#!/usr/bin/env bash
# Reuse the existing BoltzGen Torch environment; no new CUDA environment.
set -euo pipefail
cd "$(dirname "$0")/.."
py="$PWD/build/env-boltzgen-linux/bin/python"
uv="$PWD/build/linux-tools/uv-x86_64-unknown-linux-gnu/uv"
export UV_CACHE_DIR="$PWD/build/uv-cache-linux"
export UV_LINK_MODE=copy
test -x "$py"
mkdir -p build/sources build/models
if [[ ! -d build/models/LigandMPNN-main ]]; then
  curl -fL --retry 2 'https://codeload.github.com/dauparas/LigandMPNN/zip/26ec57ac976ade5379920dbd43c7f97a91cf82de' -o build/sources/ligandmpnn-main.zip
  "$py" - <<'PY'
from pathlib import Path
from zipfile import ZipFile
with ZipFile('build/sources/ligandmpnn-main.zip') as z:
    assert z.comment.decode() == '26ec57ac976ade5379920dbd43c7f97a91cf82de'
    source=z.namelist()[0].split('/')[0]
    z.extractall('build/models')
Path('build/models',source).rename('build/models/LigandMPNN-main')
PY
fi
mkdir -p build/models/LigandMPNN-main/model_params
if [[ ! -s build/models/LigandMPNN-main/model_params/ligandmpnn_v_32_010_25.pt ]]; then
  curl -fL --retry 2 'https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt' -o build/models/LigandMPNN-main/model_params/ligandmpnn_v_32_010_25.pt
fi
"$uv" pip install --python "$py" prody==2.6.1 ml-collections==0.1.1 dm-tree==0.1.9
# The old bundled OpenFold is run with this isolated array runtime. BoltzGen's
# installed NumPy stays unchanged; generation records the actual versions.
"$uv" pip install --python "$py" --target build/ligandmpnn-numpy1 --no-deps numpy==1.26.4
echo 'Run LigandMPNN with PYTHONPATH="$PWD/build/ligandmpnn-numpy1" and --device cuda.'
