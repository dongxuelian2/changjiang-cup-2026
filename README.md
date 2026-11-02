# Changjiang Cup 2026: TL1A minibinder research

Real de novo design against **TL1A**, with RFdiffusion/ProteinMPNN, BoltzGen and
local ColabDesign/AF2 gradient experiments,
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
| [AF2 gradient experiments](results/tl1a_afdesign_20261102) | 5 sequences / 3 new backbones, including 2 soluble-MPNN core variants | 3 held-out AF2 model-1 complexes; 3 Boltz-2 complexes + 3 binder-alone controls |
| [L235/Y238 BoltzGen round](results/tl1a_boltzgen_terminal_20261102) | 24 sequences / 8 new backbones, 90–100 aa, full native trimer | 3 AF2 model-1 complexes; 3 Boltz-2 complexes + 3 binder-alone controls |
| [Three-helix BoltzGen/MPNN round](results/tl1a_boltzgen_helix_20261102) | 14 sequences / 4 new 80 aa backbones, including 2 full-binder soluble-MPNN designs | 5 AF2 model-1 complexes; 4 Boltz-2 complexes + 4 binder-alone controls |
| [Alpha/beta pose and core tests](results/tl1a_pose_design_20261102) | 7 sequences / 2 new 72 aa backbones: 2 raw, 2 IF, 2 AF2 refinements and one supported 65 aa core follow-up | 7 AF2 model-1 + 2 model-3 complexes; 4 Boltz-2 complexes across two seeds + 2 binder-alone controls |

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

BoltzGen runs both natively on Windows and in **Ubuntu via WSL**, using the same
**RTX 4060 Laptop 8 GB**. Earlier distal/beta binders could refold confidently
in Boltz alone, while their complex poses disagreed with design and AF2. The
D175/F209 junction batch failed native glycan clearance; the R166/Q167/A168 tip
batch improved clearance but still failed association. Learned binding flags
are not hard steric constraints. These families are frozen. Detailed original
scores and observations remain in the result tables and `math/PROOF.md`.
AF2 global ipTM includes target-target contacts and is not binder confidence.

All-atom clashes, beta-core alignment, symmetry-equivalent target-chain labels,
fold/pose RMSD, buried binder SASA, glycan-anchor distance, sequence composition
and hydrophobic-window proxies are evaluated separately. Glycan envelopes and
sequence heuristics are approximations, not explicit glycan conformer sampling,
measured solubility or aggregation assays. No completed 20-sequence experimental
shortlist is claimed. The [complete-case research screen](results/tl1a_boltzgen_20261101/evaluated_triage.json)
records zero accepted sequences and coverage separately from missing evaluations.

The November 2 diagnosis found strong prediction disagreement at the old
Q167/A168 loop (median CA displacement 9.4/11.2 Å), versus 0.26/0.54 Å for the
new L235/Y238 hydrophobic patch. This measures prediction disagreement, not
experimental flexibility. The old tip designs also bury only 283–372 Å².

A new local ColabDesign/AF2 gradient experiment used random initial sequences,
without old binder scaffolds. Two trajectories docked inside neighboring native
protomers. The existing exact coverage solver then chose **38 native B/C
residues in five-residue fragments**, improving coverage of those actual clash
points over singleton Top-K. Including these fragments with complete protomer A
gave 202 native feature residues; all observed trimer atoms remained in the
geometric loss. This is cropped feature context, not a full-trimer AF2 generation
claim. A third new backbone avoided B/C clashes. Restrained OpenMM refinement
removed two remaining local A clashes, with target atoms immobile and binder
CA RMS movement 0.27 Å. Original outputs and the refinement are both retained.

The raw sequence and two interface-fixed soluble-MPNN core variants passed
generation geometry, but **all three failed independent validation**. Held-out
AF2 model 1 gives ipSAE(10)=0; Boltz-2 gives 0/.096/.106, with incompatible poses
and folds. No more variants of these families are generated. The initial
unranked cohort and the complete-case rejection report are retained. This shows
why repairing geometry and optimizing a template-conditioned generator score
does not establish a generalizable binder.

The new 500-step BoltzGen L235/Y238 round yielded two initially clash-free
backbones. One additional local repair removed three overlaps with CA RMS
movement 0.21 Å and an immobile target. The generation-only optimizer selected
one sequence from each of these three families; all failed the complete check.
`tl1a_bg_terminal_2_gen` retains its fold/pose in Boltz-2 (0.86/2.10 Å RMSD),
but ipSAE(10)=0 and AF2 disagrees strongly. `1_gen` reaches Boltz ipSAE .318,
but pose RMSD is 9.06 Å. These are diagnostics, not hits. See the frozen
allocation inputs and individual rejection reasons in that result directory.
The initial risk-rank objective preferred raw co-design sequences to inverse-
folded ones; this is an uncalibrated selection bias, not evidence of better
folding. Further tests must distinguish sequence refinement from association.

A subsequent **diverse** checkpoint run used 80 aa, three-helix conditioning
and per-position inverse-folding restrictions. Actual masks contain 57 helix
and 23 loop positions on every backbone; all eight IF sequences exclude C and
P/G at requested helix positions. Two small fixed-target repairs remove four
and one native overlaps, with CA RMS movements 0.24/0.15 Å. Inherited IF poses
are explicitly threaded backbones, with original CIFs and an original-backbone
identity check retained. They are not presented as sequence refolds.

The generation-stage filter allocated one IF sequence per eligible backbone.
Two maintain their folds across AF2/Boltz, but all three tested IF sequences
change association pose. Full-sequence soluble ProteinMPNN then designed two
sequences on the fourth backbone **before any member of that family was
independently screened**. Its CPU sequence-sampling loop took about five seconds; the
heavy prediction jobs run sequentially on the GPU. All measured scores,
source sequences and structures are retained in the table above.

`tl1a_bg_helix_2_mpnn2` has the strongest new independent AF2 interface signal
(ipSAE .278, pLDDT .844), but Boltz gives ipSAE .166; cross-model pose RMSD is
4.95 Å, with native-target overlaps and 10.69 Å glycan clearance. It is rejected,
despite reproducible folding. The initial two-sequence MPNN batch was frozen;
no more variants of these screened families are generated. The final
[complete-case report](results/tl1a_boltzgen_helix_20261102/evaluated_triage.json)
records four jointly evaluated sequences, one additional AF2-only rejection,
and zero accepted candidates.

2RE9 also lacks the **B:F110 aromatic ring**. It remains absent in experimental
native inputs; PDBFixer rebuilt it only in the explicitly labeled force-field
preparation, without adding missing residues. See the updated target manifest.

The latest round changes topology to one helix plus three beta strands. Actual
conditioning masks contain 18 HELIX, 24 SHEET and 30 LOOP positions. A new
six-anchor log-determinant objective constrains local pose observability;
AF2 optimization also penalizes native and predicted-target heavy-atom overlaps.
Neither the strict nor the staged AF2 sequence refinement retains an independent
model-1 interface signal. Full-rank geometric constraints do not make a desired
binding pose learnable by sequence gradients.

The **raw co-design** `tl1a_bg_ab72_0_gen` instead gives AF2 model-1/model-3
ipSAE(10) **.470/.530**, and Boltz seeds 2026/2027 **.486/.333**. Its apparent
cross-model disagreement is concentrated at the termini: full-chain AF2/Boltz
pose RMSD is 3.11 A, versus 1.68 A for residues 5–69. This motivated one actual
65-aa truncation, preserving all four Cys, with a parent-prediction-derived
reference pose and a small fixed-target repair. Fresh predictions use no
template or initial guess; no parent scores are inherited.

The truncated sequence has AF2/Boltz fold RMSDs .77/1.05 A and cross-model pose
RMSD 2.70 A, but Boltz ipSAE is **0/.025** across two seeds, versus AF2
**.4996/.441** across checkpoints. Parent seed 2027 also approaches a native
glycan anchor to **10.10 A**. Better geometric agreement therefore does not
establish a confident, glycan-safe association. Both jointly evaluated sequences
fail the [complete screen](results/tl1a_pose_design_20261102/evaluated_triage.json);
five further sequences are explicit early AF2 rejections, not complete cases.
This family is frozen after that single targeted follow-up. The next useful
direction is to preserve co-designed interface/Cys chemistry on new backbones
with larger glycan clearance, and test interface confidence across models and
seeds before expanding a family.

Raw Cys SG clusters are chemically inconsistent and remain visible in the
interface audit. Ordinary disulfide bonding cannot be inferred from these
unrelaxed coordinates. A single-IF output filename collision was also fixed;
original atom14 and IF CIF/NPZ files are retained separately and checked against
their actual sequences. See section 16 of [the research notes](math/PROOF.md).

## Continue the configured local pipeline

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
cd /mnt/e/bio/changjiang-cup-2026
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
