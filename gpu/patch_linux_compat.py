"""Narrow adapters for WSL telemetry and loading only requested AF2 weights.

No model architecture, parameters, confidence calculation or kernel arithmetic
is changed. WSL uses default kernel configurations, without NVML-based tuning.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
site = ROOT / "build/env-boltzgen-linux/lib/python3.11/site-packages"
path = site / "cuequivariance_ops/triton/cache_manager.py"
source = path.read_text()
marker = "def get_gpu_information():\n"
guard = '''    if os.getenv("CUEQ_WSL_METADATA") == "1":
        import torch
        props = torch.cuda.get_device_properties(0)
        return {"name": props.name, "total_memory": math.ceil(props.total_memory/(1024**3)),
                "multi_processor_count": props.multi_processor_count,
                "power_limit": None, "clock_rate": None,
                "major": props.major, "minor": props.minor}
'''
if "CUEQ_WSL_METADATA" not in source:
    if marker not in source or "    def load_cache(self, fn_key: str) -> dict:\n" not in source:
        raise ValueError("unexpected cuEquivariance 0.5.1 source")
    source = source.replace(marker,marker+guard,1)
    source = source.replace("    def load_cache(self, fn_key: str) -> dict:\n", '''    def load_cache(self, fn_key: str) -> dict:
        if os.getenv("CUEQ_WSL_METADATA") == "1":
            self.gpu_cache[fn_key] = {"gpu_information": self.gpu_information}
            return self.gpu_cache[fn_key]
''',1)
    path.write_text(source)
path = ROOT / "build/env-colabfold-linux/lib/python3.11/site-packages/colabfold/alphafold/models.py"
record = {"cuequivariance_ops_0.5.1": "WSL unsupported NVML power/clock are unknown; use real Torch device properties and skip telemetry-based tuning cache; requires CUEQ_WSL_METADATA=1 and default configs"}
if path.exists():
    source = path.read_text()
    old = "    for model_number in model_build_order:\n"
    new = "    for model_number in (n for n in model_build_order if n in model_order or n in models_need_compilation):\n"
    if old in source:
        path.write_text(source.replace(old,new,1))
    elif new not in source:
        raise ValueError("unexpected ColabFold 1.6.3 source")
    record["colabfold_1.6.3"] = "load only requested parameter sets plus compilation model; forward arithmetic unchanged"
(ROOT / "build/linux_compatibility.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
