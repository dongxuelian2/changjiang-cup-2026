# TL1A design experiments

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
| [RFdiffusion round](../results/tl1a_20261101) | 28 sequences / 7 backbones | 18 Boltz-2 complexes / 12 sequences, plus 2 target-only controls |
| [BoltzGen round](../results/tl1a_boltzgen_20261101) | 57 sequences / 19 backbones | 6 Boltz-2 complexes + 6 binder-alone controls; 10 AF2-Multimer complexes |
| [AF2 gradient experiments](../results/tl1a_afdesign_20261102) | 5 sequences / 3 new backbones, including 2 soluble-MPNN core variants | 3 held-out AF2 model-1 complexes; 3 Boltz-2 complexes + 3 binder-alone controls |
| [L235/Y238 BoltzGen round](../results/tl1a_boltzgen_terminal_20261102) | 24 sequences / 8 new backbones, 90–100 aa, full native trimer | 3 AF2 model-1 complexes; 3 Boltz-2 complexes + 3 binder-alone controls |
| [Three-helix BoltzGen/MPNN round](../results/tl1a_boltzgen_helix_20261102) | 14 sequences / 4 new 80 aa backbones, including 2 full-binder soluble-MPNN designs | 5 AF2 model-1 complexes; 4 Boltz-2 complexes + 4 binder-alone controls |
| [Alpha/beta pose and core tests](../results/tl1a_pose_design_20261102) | 7 sequences / 2 new 72 aa backbones: 2 raw, 2 IF, 2 AF2 refinements and one supported 65 aa core follow-up | 7 AF2 model-1 + 2 model-3 complexes; 4 Boltz-2 complexes across two seeds + 2 binder-alone controls |
| [Two-link Cys-conditioned alpha/beta tests](../results/tl1a_cystine_20261102) | 6 sequences / 2 new 72 aa backbones at R156/M158: raw, IF and Cys-preserving MPNN | 6 AF2 model-1 complexes; 1 diagnostic Boltz-2 complex + 1 binder-alone control |
| [Three-link compact beta conditioning](../results/tl1a_cys6_20261103) | 4 sequences / 2 new 56 aa backbones at L235/Y238: raw and IF | 4 AF2 model-1 complexes; 1 diagnostic Boltz-2 complex + 1 binder-alone control |
| [Fold-preserving Top7 surface tests](../results/tl1a_top7_20261103) | 8 sequences / 2 generated poses on one generic 92-aa scaffold family: raw/IF, one MPNN, one AF2 surface refinement and a two-sequence LigandMPNN context ablation | 5 AF2 model-1 complexes; 2 Boltz-2 complexes + 2 binder-alone controls; 3 conditional-likelihood pose audits |
| [Binding-state / glycan-ensemble alpha3D tests](../results/tl1a_a3d_ensemble_20261103) | 5 valid sequences / 2 poses in a NEW generic 73-aa scaffold family; atomic-context LigandMPNN and 55-step binding-state AF2 design | 3 held-out AF2 model-1 complexes; 2 Boltz-2 complexes + 2 apo controls; original generic scaffold: 1 Boltz complex + 1 apo control |

BoltzGen counts include actual atom14-generated sequences and the recorded
inverse-folding batch (one or two sequences per backbone). Distinct sequences
on a shared backbone are correlated candidates.
Original CIF/NPZ outputs, normalized backbone complexes, FASTA/CSV, metrics and
resolved generation/prediction provenance are retained. `synthetic: false` means
real model execution, not validated binding. Weights, environments and logs stay
in ignored `build/`.

**New binding-state experiment (November 3).** No screened Top7/Cys/helix
family was extended. The generic experimental [alpha3D scaffold, 2A3D](https://www.rcsb.org/structure/2A3D)
supplied a new family: 26 packing/cap identities protected, 47 surface positions
redesigned. Two BoltzGen poses retain its fold within .15/.16 A. One atom14
output contains unknown residues and is retained as rejected provenance, not a
candidate. LigandMPNN parsed **366 modelled glycan heavy atoms** plus native
target sidechains. This is actual model execution, not simulated binding data.

Three native R156/M158/L235/Y238 sidechain states and six modelled
Man3GlcNAc2 roots use observed 2RE9 protein geometry and generic linked-Asn
glycan cores from [4BYH](https://www.rcsb.org/structure/4BYH). Internal glycan
bonds and attachment bonds are checked; the two source glycans number their
Man arms differently. These are **not measured TL1A glycans or populations**;
outer branches and unknown conformers remain unmodelled. A minimum three-state
marginal cover retains the worst single-atom-pair clash over **1,458** allowed
joint assignments / **20** marginals ([proof, section 20](../math/PROOF.md)).
It does not preserve summed energies or statistical free energies.

AF2 model 3 generation used an intrachain scaffold template and bound-pose
initial guess for 45 steps, then removed both for 10 more steps and a discrete
three-recycle prediction. The objective combines interface ipTM/contact terms,
pose/fold terms and external all-heavy-atom native-sidechain/glycan penalties.
Full terminal sidechains are supplied through the external loss, not inferred
to be visible through standard AF2 template features. This changed G29 to D
in the atomic-context sequence; no further variant was generated after holdout.

| Frozen design | AF2 / Boltz interface ipSAE10 | AF2 / Boltz binder pLDDT | Cross-model pose RMSD (A) | Boltz apo RMSD / pLDDT |
|---|---|---|---:|---|
| atomic-context LigandMPNN | 0 / 0 | .399 / .717 | 13.31 | 3.19 A / .805 |
| binding-state AF2, G29D | 0 / 0 | .415 / .740 | 7.51 | 2.08 A / .812 |

The IF baseline also has AF2 interface ipSAE zero. **Complete-screen survivors:
0/2 evaluated designs.** AF2/Boltz use the full 180-aa target trimer, no binder
templates/initial guesses, and the existing MSA/seed settings. All original
PAE and apo predictions are retained. The generic original-scaffold control
has Boltz apo pLDDT **.916** but RMSD **3.12 A** to its experimental reference;
its packing-position RMSD is **1.80 A** (helices 2.15 A), so the overall
deviation does not imply unfolding. Confidence alone does not establish full
reference-fold recovery.

The new bottleneck is explicit: soft multi-state penalties still leave an
infeasible predicted binding state. CPU auditing of the released generator
output finds **.47 A** minimum native-protein and **.41 A** minimum modelled-glycan
distances, agreeing with GPU losses within .002. Its apparent AF2/Boltz fold
consistency (1.58/1.81 A to the generated fold) also differs from agreement
with the original scaffold (3.28/3.56 A). Conversely the Boltz poses clear the
sampled glycan cores by 3.11/3.34 A while failing the conservative 12 A ND2
envelope; neither establishes tolerance of unknown outer glycan branches.
These failures do not prove physical nonbinding. Further work should use a
new scaffold with calibrated apo/reference agreement and enforce multistate
geometric feasibility before confidence optimization; this family is frozen.

A GPU numerical audit also found up to **.426 A** nearest-distance error and
three false <2 A clashes with default matrix multiplication on a real input.
Centered highest-precision multiplication reduces the error below **.00003 A**.
The aborted default-precision run is retained separately and never counted as
a candidate; it and the corrected trajectory have different initial sequences,
so they are not a controlled design ablation. See the real
[atomic/pose audits and complete triage](../results/tl1a_a3d_ensemble_20261103).
The executed entry points are `scripts/prepare_tl1a_ensemble.py`,
`scripts/redesign_ligandmpnn.py --context-state 0 --contexts 1`, and
`scripts/run_binding_state_design.py`; generation provenance retains the stages,
weights, initial sequence and checkpoint. Generation refuses screened families.

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
shortlist is claimed. The [complete-case research screen](../results/tl1a_boltzgen_20261101/evaluated_triage.json)
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
[complete-case report](../results/tl1a_boltzgen_helix_20261102/evaluated_triage.json)
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
fail the [complete screen](../results/tl1a_pose_design_20261102/evaluated_triage.json);
five further sequences are explicit early AF2 rejections, not complete cases.
This family is frozen after that single targeted follow-up. A subsequent audit
of its original PAE finds median reciprocal contact PAE increasing from
4.67/6.76 A to 16.28/11.59 A across the two seeds. The truncation's confidence
loss is broader than a few pairs crossing the scoring cutoff. A fixed-mask
smooth-summary stability bound and its limits are in section 17 of the research
notes; this diagnostic is not a replacement official score.

Raw Cys SG clusters are chemically inconsistent and remain visible in the
interface audit. Ordinary disulfide bonding cannot be inferred from these
unrelaxed coordinates. A single-IF output filename collision was also fixed;
original atom14 and IF CIF/NPZ files are retained separately and checked against
their actual sequences. See section 16 of [the research notes](../math/PROOF.md).

The next experiments explicitly condition **two internal SG links and four
fixed Cys** on two entirely new 72-aa alpha/beta backbones at R156/M158. One
raw, one IF and one Cys-preserving soluble-MPNN sequence per backbone form a
frozen six-sequence batch. All six give AF2 ipSAE(10)=0. The best-folding
`tl1a_bg_cysbeta_1_if_mpnn1` retains its backbone in AF2, Boltz complex and Boltz
alone (RMSDs .941/.910/.450 A), but AF2/Boltz cross-pose RMSD is **21.56 A**
despite cross-fold RMSD 1.055 A. Both interface scores are zero; Boltz approaches
a native glycan anchor to **9.58 A**. Actual reciprocal contact PAE has median
20.11 A. This is another fold-supported, association-unsupported sequence,
rejected by the [complete screen](../results/tl1a_cystine_20261102/evaluated_triage.json).
The other five are explicit AF2 rejections without Boltz evaluation.

Two new **56-aa, three-internal-link** backbones at L235/Y238 instead achieve
generated glycan clearance **17.49/17.92 A**. Their raw and IF sequences all
give AF2 ipSAE zero; the exact-six-Cys IF folds diverge by 20.59/12.87 A with
pLDDT .221/.288. A diagnostic Boltz run on the second IF sequence gives
complex/alone fold RMSD 4.86/3.07 A, ipSAE zero and glycan clearance 8.67 A.
Generated clearance did not persist in independent association. These families
are frozen. The
[input audit](../results/tl1a_cys6_20261103/input_feature_audit.json) verifies the
three specified pair features survive the actual masker as typed covalent
features, and the genuine adherence checkpoint enables that channel. Binary
bond features are zeroed upstream. Neither feature is a hard SG-distance
constraint or a verified cystine knot. Raw generation still emits extra Cys;
per-position IF restrictions do not constrain raw co-design decoding.
Actual SG distances remain in each dataset's `cystine_geometry.json`.

A different method preserves a **generic de novo fold**, using only the 92
observed, tag-free Top7 residues from [1QYS](https://www.rcsb.org/structure/1QYS).
The unresolved N-terminal residues and expression tail are excluded; MSE is
explicitly normalized to MET. This is not a known TL1A binder. The prepared
structure, source and modifications are in `targets/scaffolds/`. Seventy input
identities are held fixed; 22 exposed positions are redesigned. Separate actual
structure groups reveal internal scaffold and target geometry, with **zero
binder-target distance visibility**. Two generated poses share this fold and
are not independent scaffold families.

Fourteen initial overlaps all involve native **R156 sidechain** atoms and
binder residues 40–42, with no target backbone penetration. A strictly localized
repair removes them with native particles immobile, CA RMS movement .306 A and
maximum 1.950 A. The existing final steric/glycan and movement limits are
unchanged. Actual source sidechains can be incomplete; raw SASA uses emitted
atoms, while force-field atom completion is explicitly recorded. The second
pose is an early geometry rejection. An 85-step AF2 surface optimization clamps
the same 70 identities one-hot, but ends with 24 native backbone overlaps and
generator ipSAE zero; it is also rejected before independent allocation.

The [sidechain audit](../results/tl1a_top7_20261103/target_sidechains.json) shows why
CA agreement alone is insufficient. Across the two generated poses, R156 CA
deviation has median .090 A, versus local-backbone-fitted sidechain RMSD 2.96 A.
Backbone-only ProteinMPNN cannot observe a change confined to target sidechains.
The new [LigandMPNN](https://github.com/dauparas/LigandMPNN) experiment enables
observed fixed sidechains as atom context, with an off control using the same
checkpoint, seed, reference fold and 22 designable positions. Three residues
change. Both real sequences give AF2 ipSAE zero; their fold RMSDs are
1.51/1.22 A. The on sequence's Boltz fold/alone RMSDs are .917/.872 A, but
its pose RMSD is **84.56 A** and ipSAE zero. The IF baseline similarly retains
its fold but has AF2/Boltz ipSAE .015/0 and incompatible poses. The
[two complete cases](../results/tl1a_top7_20261103/evaluated_triage.json) are rejected;
three other sequences fail AF2, and three are early geometry rejections.

Finally, real LigandMPNN scoring on one frozen sequence **does prefer** its
design pose over rigid poses derived from the two independent predictions:
NLL 1.414/2.490/1.713 nats per designable residue, with identical decoding order
and unchanged internal fold. This is conditional sequence preference, not a
pose posterior or measured affinity. Original inputs/statistics remain in the
[pose audit](../results/tl1a_top7_20261103/pose_likelihood.json); sections 18–19 of
`math/PROOF.md` separate input identifiability and conditional likelihood from
association. More conditional-score optimization of these frozen families is
not supported. The next useful test needs an association-sensitive objective
and a larger compatible interface, treating native sidechain states and glycan
access together, on a genuinely new scaffold family.

