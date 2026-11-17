# Reproduce the local design pipeline

Generation and prediction run **sequentially** on the 8 GB GPU. No paid compute,
existing Colab session, competition account or formal submission is used.

The tested BoltzGen 0.3.2 Linux environment uses torch 2.7.1+cu128, Triton 3.3.1,
cuEquivariance 0.5.1, Lightning 2.5.5 and a C compiler. Native Windows uses the
public `use_kernels=false` path with torch 2.5.1+cu121; Linux-only acceleration
dependencies are omitted there. WSL's unsupported NVML power telemetry is bypassed
by `gpu/patch_linux_compat.py` using actual Torch device properties and default
kernel settings. Model weights, architecture and arithmetic are unchanged.
Curated BoltzGen sequence metrics use Windows `env-rfd` (BioPython 1.85,
NumPy 1.26.4); the Linux generator's parser is BioPython 1.88. AF gradient
outputs were imported with BioPython 1.84. Importers record the actual analysis
version and reject mixed-version appends; fresh Linux datasets record theirs.

From PowerShell, enter the existing Ubuntu distribution with `wsl.exe -d Ubuntu`.
The following commands are Linux commands inside this checkout. The setup expects
`uv` at `UV_BIN` or the already installed `build/linux-tools/.../uv` path; network
proxy variables, if needed, belong to the caller and are not committed.

```bash
cd /path/to/changjiang-cup-2026
bash gpu/setup_boltzgen_linux.sh
bash gpu/run_boltzgen.sh --spec gpu/tl1a_boltzgen_helix.yaml \
  --checkpoint diverse --sampling-steps 500 --seed 20261109 \
  --num-designs 4 --out build/next_terminal
build/env-boltzgen-linux/bin/python scripts/import_boltzgen.py \
  build/next_terminal --label tl1a_terminal_next --out build/next_terminal_data
build/env-boltzgen-linux/bin/python scripts/triage_candidates.py \
  build/next_terminal_data/candidates.json --generation-stage inverse_folding \
  --k 4 --one-per-backbone

bash gpu/setup_colabfold_linux.sh
build/env-colabfold-linux/bin/python scripts/fetch_af2_model.py --model 1
# Use IDs selected for actual independent evaluation, not an assumed binder hit.
build/env-colabfold-linux/bin/python scripts/run_af2.py \
  build/next_terminal_data/candidates.json --ids SELECTED_ID --model 1 \
  --out build/next_af2
build/env-colabfold-linux/bin/python scripts/import_af2.py \
  build/next_terminal_data/candidates.json build/next_af2
```

ColabFold 1.6.3 runs AF2-Multimer-v3 models 3 (previous round) and 1 (new holdout),
JAX 0.6.2, one seed/sample, three
recycles, target-only MSA 128/256, no template/initial guess/relaxation. It runs
locally, not on hosted Colab. The requested legacy kernel has no sm89 binary and
the actual successful backend is **XLA fallback**, recorded in provenance. Only
the requested genuine parameter files are fetched from the public archive by ranges;
ZIP CRC is checked. Loading unused model parameter sets is bypassed separately.

The executed gradient-design entry point is `scripts/run_afdesign.py`;
`gpu/setup_afdesign_linux.sh` pins ColabDesign and its JAX dependencies and obtains
models 3/1. The third trajectory can be reproduced with `--site terminal
--length 80 --seeds 20261107 --schedule curriculum --recycles 1
--context-neighbors 20 --clash-probes results/tl1a_afdesign_20261102/candidates.json
--out build/new_afdesign`. Its original negative probes are retained there.
`scripts/import_afdesign.py` imports only completed discrete-sequence predictions.
Optional local steric repair used OpenMM 8.3.1 and PDBFixer at
`5e658a4fb8d2b90d65ca8408c845cbf104ce7099`, checked out at
`build/models/PDBFixer` and installed in `build/env-boltz`, with
`scripts/refine_backbone.py`.
`scripts/redesign_mpnn.py` uses the cached soluble `v_48_020` weights and refuses
to extend any family already evaluated independently. `--redesign-interface`
also designs interface positions; `--cpu` runs small batches without taking GPU
memory. Local repairs can precede the IF import: import raw outputs, run
`refine_backbone.py`, then append with `--inherit-refined-backbones`, before
any independent evaluation of that family. The importer verifies that the
original IF backbone matches its atom14 source within 0.02 Å.

`gpu/setup_ligandmpnn_linux.sh` reuses the BoltzGen Torch environment, pins the
LigandMPNN source and one 10.5 MB public checkpoint, and supplies an isolated
NumPy 1.26 runtime. The tested path is CUDA; CPU attempts aborted in native
libraries. Run `scripts/redesign_ligandmpnn.py generate` with
`PYTHONPATH="$PWD/build/ligandmpnn-numpy1"`, then its `import` stage with the
dataset's original Windows BioPython environment. Full commands, versions,
fixed positions and scores are retained in
`results/tl1a_top7_20261103/generation/top7_ligandmpnn_native_context_gpu/run.json`.
No new sequence design is permitted on these independently screened families.

The configured native Boltz-2 2.2.1 environment uses torch 2.5.1+cu121. In PowerShell:

```powershell
build/env-boltz/Scripts/python.exe scripts/run_boltz.py build/next_terminal_data/candidates.json --ids SELECTED_ID --trimer --msa-depth 512 --with-binder-controls --per-record --out build/next_boltz
build/env-boltz/Scripts/python.exe scripts/import_boltz.py build/next_terminal_data/candidates.json build/next_boltz
build/env-boltz/Scripts/python.exe scripts/import_af2.py build/next_terminal_data/candidates.json build/next_af2
build/env-rfd/Scripts/python.exe scripts/triage_candidates.py build/next_terminal_data/candidates.json --complete --require-boltz --require-af2
```

Both predictors use the cached public target MSA (2,286 records); regenerate it
with `scripts/target_msa.py` if missing. The generation wrapper also expects the
existing `build/models/boltz2/boltz2_conf.ckpt` to resolve an upstream **unused**
folding-step configuration; only design/inverse-folding tasks execute there.
Boltz evaluation uses fresh seed 2026 per input with `--per-record`, 200 sampling
steps and three recycles. Original scores and per-pair ipSAE at PAE cutoffs 10/15 Å
remain separate by predictor/configuration. **Neither predictor reproduces the
unpublished official ESMFold2 pipeline**, and no protein–protein KD is inferred
from Boltz's small-molecule affinity head. The full public ESMFold2 weights exceed
this GPU's native 8 GB capacity before activations; it was not run.

