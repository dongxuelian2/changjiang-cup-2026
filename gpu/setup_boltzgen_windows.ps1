# Native Windows, upstream PyTorch implementation (--use_kernels=false).
# cuEquivariance is a declared dependency but is lazily imported only by kernels.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$env:UV_CACHE_DIR = Join-Path (Get-Location) 'build/uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) 'build/python'
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
uv venv build/env-boltzgen --python 3.11
Check-Exit
uv pip install --python build/env-boltzgen/Scripts/python.exe 'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
Check-Exit
uv pip install --python build/env-boltzgen/Scripts/python.exe --no-deps 'boltzgen==0.3.2'
Check-Exit
# Exclude only the three Linux-only acceleration dependencies and optional extras.
$researchDeps = build/env-boltzgen/Scripts/python.exe -c "import importlib.metadata as m; print('\n'.join(r for r in m.requires('boltzgen') if ';' not in r and not r.startswith(('cuequivariance', 'torch'))))"
Check-Exit
$researchDeps | Set-Content -LiteralPath build/boltzgen-requirements.txt -Encoding utf8
uv pip install --python build/env-boltzgen/Scripts/python.exe -r build/boltzgen-requirements.txt 'pytorch-lightning==2.5.0' 'einx==0.3.0' 'numpy==2.0.2' 'numba==0.61.0'
Check-Exit
build/env-boltzgen/Scripts/python.exe -c "import torch,boltzgen.model.models.boltz; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
Check-Exit
