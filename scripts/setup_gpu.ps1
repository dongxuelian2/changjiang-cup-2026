param([switch]$WithBoltz)
$ErrorActionPreference = 'Stop'
$researchRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $researchRoot
$env:UV_CACHE_DIR = Join-Path $researchRoot 'build/uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $researchRoot 'build/python'
$env:PYTHONIOENCODING = 'utf-8'

function Check-Exit {
    if ($LASTEXITCODE -ne 0) { throw "Previous command failed with exit $LASTEXITCODE" }
}

uv venv build/env-rfd --python 3.10
Check-Exit
uv pip install --python build/env-rfd/Scripts/python.exe 'torch==2.0.1+cu117' --index-url https://download.pytorch.org/whl/cu117
Check-Exit
uv pip install --python build/env-rfd/Scripts/python.exe 'numpy==1.26.4' 'scipy==1.15.3' 'biopython==1.85' 'e3nn==0.3.3' 'hydra-core==1.3.2' 'pyrsistent==0.20.0' 'icecream==2.1.8' 'opt-einsum-fx==0.1.4' 'decorator==5.1.1' 'tqdm==4.67.1' 'dgl==1.1.2+cu117' --find-links https://data.dgl.ai/wheels/cu117/repo.html
Check-Exit
build/env-rfd/Scripts/python.exe scripts/fetch_research_inputs.py
Check-Exit
build/env-rfd/Scripts/python.exe scripts/fetch_models.py
Check-Exit
build/env-rfd/Scripts/python.exe scripts/prepare_targets.py
Check-Exit

if ($WithBoltz) {
    uv venv build/env-boltz --python 3.10
    Check-Exit
    uv pip install --python build/env-boltz/Scripts/python.exe 'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
    Check-Exit
    uv pip install --python build/env-boltz/Scripts/python.exe 'boltz==2.2.1'
    Check-Exit
    build/env-boltz/Scripts/python.exe scripts/target_msa.py
    Check-Exit
}
