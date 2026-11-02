# Changjiang Cup 2026: TL1A minibinder research

Real de novo design against **TL1A**, with RFdiffusion/ProteinMPNN and BoltzGen,
independent Boltz-2/AF2-Multimer predictions, and the existing candidate-portfolio
optimizers. **No candidate currently passes the complete research screen.**
No measured KD, experimental binder or official admission score has been obtained.

[RULES.md](RULES.md) records the successfully retrieved November 1 rules and the
unpublished official ESMFold2/ipSAE configuration. [math/PROOF.md](math/PROOF.md)
separates mathematical proofs, synthetic examples, actual model observations and
biological hypotheses. Simulated data remain under `examples/synthetic/`.

## Target and evidence

| Priority target | Audited reference | Research implication |
|---|---|---|
| **TL1A**, minibinder | 2RE9, 2.1 Å homotrimer; observed UniProt 88–251; official construct 72–251 | Compact target; explicitly retain three protomers and check N133/N229 glycan anchors on each |
| IL-7Rα, minibinder | 3DI3 B, 2.9 Å; observed 37–229; I66T versus current UniProt | Domain arrangement, six annotated N-glycosylation sites and target-state uncertainty add difficulty |
| TNF-α, VHH | 1TNF, 2.6 Å trimer; observed 82–233; L219 versus native D219 | Compact, but this track needs a complete 100–150 aa VHH |

TL1A is a tractable initial choice, not a demonstrated superiority ranking.
Primary sources: [2RE9](https://www.rcsb.org/structure/2RE9),
[UniProt O95150](https://www.uniprot.org/uniprotkb/O95150/entry),
[TL1A biochemistry](https://pubmed.ncbi.nlm.nih.gov/19522538/),
[DcR3 target-site context](https://www.rcsb.org/structure/3K51),
[IL7 receptor study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2654238/),
[1TNF](https://www.rcsb.org/structure/1TNF).
No known same-target positive binder sequence/scaffold was supplied to generation
or used as an affinity-optimization starting point. This is not an exhaustive
novelty search or a claim about model training data.

`targets/` contains all seven official construct FASTAs and priority-target audits.
For 2RE9, **author residue +71 = UniProt**, whereas **mmCIF label_seq_id +70 =
UniProt**: the observed canonical interval is **18–181**, not 17–180. Generation
preflight verifies three exact 164 aa target chains. Independent predictions use
the full official 180 aa sequence in all three protomers. Missing experimental
72–87 coordinates are not invented. `targets/TL1A/surface.csv` records native
SASA, neighboring-subunit burial, glycan-anchor distance and receptor context.

## Executed experiments

| Curated data | Real generation | Independent sequence-to-structure evaluation |
|---|---|---|
| [RFdiffusion round](results/tl1a_20261101) | 28 sequences / 7 backbones | 18 Boltz-2 complexes / 12 sequences, plus 2 target-only controls |
| [BoltzGen round](results/tl1a_boltzgen_20261101) | 57 sequences / 19 backbones | 6 Boltz-2 complexes + 6 binder-alone controls; 10 AF2-Multimer complexes |

BoltzGen counts include actual atom14-generated sequences and two inverse-folded
variants per backbone. They are distinct sequences, not independent backbones.
Original CIF/NPZ outputs, normalized backbone complexes, FASTA/CSV, metrics and
resolved generation/prediction provenance are retained. `synthetic: false` means
real model execution, not validated binding. Weights, environments and logs stay
in ignored `build/`.

The RFdiffusion round exposed adjacent-subunit clashes, noncompact rods and
target-state prediction errors. Increasing target MSA depth from 64 to 512
improved a target-only beta-core RMSD from 6.63 to 0.45 Å. This is a diagnostic
observation, not a controlled causal estimate. Those binder families were not
expanded further.

BoltzGen has now actually run both natively on Windows and in **Ubuntu via WSL**
on the same **RTX 4060 Laptop 8 GB**. The first distal-site and beta-site batches
produced 21 sequences / 7 backbones. Binder-alone Boltz predictions for four
tested sequences have pLDDT 0.961–0.969 and fold RMSD 0.30–0.49 Å; this supports
folding, not binding. `tl1a_bg_a_1_if1` has Boltz binder-target ipTM 0.838 and
ipSAE(10) 0.436, but pose RMSD 35.55 Å. AF2 gives ipSAE(10) 0 and a different,
clashing pose. It is rejected. The other beta-site AF2 checks also lack reliable
interfaces. AF2's global ipTM around 0.74 in the first checks includes the native
target trimer and must not be treated as binder confidence.

An inter-protomer D175/F209 hypothesis generated 8 further backbones / 24 sequences.
All eight backbones lie 4.8–9.1 Å from a native glycan anchor, inside the declared
12 Å research envelope, and were rejected before expensive prediction. Binding-
type flags are not hard steric masks. A narrower R166/Q167/A168 distal-tip test
then actually ran 500 diffusion steps: 4 backbones / 12 sequences. Glycan-anchor
distances improved to 21.3–27.7 Å, but only two backbones had zero native-atom
overlaps. The optimizer selected `tl1a_bg_tip_2_gen` and `tl1a_bg_tip_0_if0` for
independent prediction. Both, plus two near-clash diagnostic sequences, failed
AF2 interface checks: ipSAE(10) 0–0.011, pose RMSD 22.9–26.3 Å and severe overlap
with native target atoms. These are failed research hypotheses, not binder hits.
Boltz independently rechecked both priority sequences with binder-alone controls:
ipSAE(10) is 0.012 / 0, with pose RMSD 33.90 / 77.50 Å, respectively for
`tl1a_bg_tip_0_if0` / `tl1a_bg_tip_2_gen`. The revised glycan-safe generation site
did not solve association. Further expansion of these families is unsupported.

All-atom clashes, beta-core alignment, symmetry-equivalent target-chain labels,
fold/pose RMSD, buried binder SASA, glycan-anchor distance, sequence composition
and hydrophobic-window proxies are evaluated separately. Glycan envelopes and
sequence heuristics are approximations, not explicit glycan conformer sampling,
measured solubility or aggregation assays. No completed 20-sequence experimental
shortlist is claimed. The [complete-case research screen](results/tl1a_boltzgen_20261101/evaluated_triage.json)
records zero accepted sequences and coverage separately from missing evaluations.

## Continue the configured local pipeline

Generation and prediction run **sequentially** on the 8 GB GPU. No paid compute,
existing Colab session, competition account or formal submission is used.

The tested BoltzGen 0.3.2 Linux environment uses torch 2.7.1+cu128, Triton 3.3.1,
cuEquivariance 0.5.1, Lightning 2.5.5 and a C compiler. Native Windows uses the
public `use_kernels=false` path with torch 2.5.1+cu121; Linux-only acceleration
dependencies are omitted there. WSL's unsupported NVML power telemetry is bypassed
by `gpu/patch_linux_compat.py` using actual Torch device properties and default
kernel settings. Model weights, architecture and arithmetic are unchanged.

From PowerShell, enter the existing Ubuntu distribution with `wsl.exe -d Ubuntu`.
The following commands are Linux commands inside this checkout. The setup expects
`uv` at `UV_BIN` or the already installed `build/linux-tools/.../uv` path; network
proxy variables, if needed, belong to the caller and are not committed.

```bash
cd /mnt/e/bio/changjiang-cup-2026
bash gpu/setup_boltzgen_linux.sh
bash gpu/run_boltzgen.sh --spec gpu/tl1a_boltzgen_tip.yaml \
  --checkpoint adherence --sampling-steps 500 --seed 20261104 \
  --num-designs 4 --out build/next_tip
build/env-boltzgen-linux/bin/python scripts/import_boltzgen.py \
  build/next_tip --label tl1a_tip_next --append
build/env-boltzgen-linux/bin/python scripts/triage_candidates.py \
  results/tl1a_boltzgen_20261101/candidates.json --k 4 --one-per-backbone

bash gpu/setup_colabfold_linux.sh
build/env-colabfold-linux/bin/python scripts/fetch_af2_model.py --model 3
# Use IDs selected for actual independent evaluation, not an assumed binder hit.
build/env-colabfold-linux/bin/python scripts/run_af2.py \
  results/tl1a_boltzgen_20261101/candidates.json --ids tl1a_bg_a_1_if1 \
  --out build/next_af2
build/env-colabfold-linux/bin/python scripts/import_af2.py \
  results/tl1a_boltzgen_20261101/candidates.json build/next_af2
```

ColabFold 1.6.3 runs AF2-Multimer-v3 model 3, JAX 0.6.2, one seed/sample, three
recycles, target-only MSA 128/256, no template/initial guess/relaxation. It runs
locally, not on hosted Colab. The requested legacy kernel has no sm89 binary and
the actual successful backend is **XLA fallback**, recorded in provenance. Only
the genuine model-3 parameter file is fetched from the public archive by ranges;
ZIP CRC is checked. Loading unused model parameter sets is bypassed separately.

The configured native Boltz-2 2.2.1 environment uses torch 2.5.1+cu121. In PowerShell:

```powershell
build/env-boltz/Scripts/python.exe scripts/run_boltz.py results/tl1a_boltzgen_20261101/candidates.json --ids tl1a_bg_a_1_if1 --trimer --msa-depth 512 --with-binder-controls --per-record --out build/next_boltz
build/env-boltz/Scripts/python.exe scripts/import_boltz.py results/tl1a_boltzgen_20261101/candidates.json build/next_boltz
build/env-rfd/Scripts/python.exe scripts/triage_candidates.py results/tl1a_boltzgen_20261101/candidates.json --complete --require-boltz --require-af2
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

## Mathematics used on real data

The existing NumPy-only CLI retains raw/singleton Top-K, submodular greedy,
safeguarded portfolio, exact enumeration, diversity and CSV/FASTA export:

```bash
python -m pip install -e .
python -m changjiang_select benchmark
python -m unittest discover -s tests -v
```

Actual metrics define uncalibrated, source-balanced rank-coverage views; missing
and constant metrics are omitted, and rank-identical views within one predictor
family are collapsed. Reports retain pairwise-complete Spearman sample counts,
shared-backbone warnings and leave-one-source-out sensitivity. Cross-model
correlations are descriptive on the small, deliberately selected overlap cohort;
they do not establish general model agreement or a relationship to KD.
On four overlapping sequences, interface ipSAE(10) ranks have Spearman rho -0.544,
while binder pLDDT ranks have rho 0.80. The sample is tiny and interface scores
have many zeros. BoltzGen and Boltz-2 are related models; independent execution
is not an assumption of statistically independent errors.

The cardinality greedy guarantee remains `1-1/e`. Optional one-per-backbone
exploration forms a truncated partition matroid, with a proved greedy `1/2`
bound and a matching synthetic example. This is a compute-allocation option,
not a competition constraint or a biological-success guarantee. On the interim
45-sequence table, 9 passed preliminary gates; at k=4 with that constraint,
portfolio/singleton/exact all give fitted rank coverage **0.890625**. None of this
turns the failed independent interfaces into a credible binder or a calibrated
success probability. The later six-sequence tip allocation is frozen in
[compute_priority](results/tl1a_boltzgen_20261101/compute_priority): at k=2,
portfolio/singleton/exact all give **0.85**, choosing one sequence per eligible
backbone. Its saved inputs reproduce the allocation before prediction feedback.
Ranks are cohort-relative; these two fitted objective values cannot be compared
as measures of binder quality.
Research-priority CSV/FASTA are not official submissions.
