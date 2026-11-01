# Changjiang Cup 2026: TL1A minibinder research

This private research project now works on the competition's **real TL1A target**,
using de novo RFdiffusion backbones and ProteinMPNN sequences. The original
candidate-portfolio mathematics and solvers are reused for prediction triage.
There are **28 real sequences from seven backbones**, and **18 all-atom Boltz
complex predictions covering 12 distinct sequences**, plus two target-only
controls. **Zero evaluated candidates currently pass the complete research
screen.** No experimental binder, measured KD, or official admission score has
been obtained.

[RULES.md](RULES.md) records the verified November 1 targets/rules and the missing
official ESMFold2/ipSAE configuration. [math/PROOF.md](math/PROOF.md) separates proofs,
synthetic benchmarks, actual model observations and biological hypotheses.

## Why TL1A

| Priority target | Coordinate/sequence audit | Research implication |
|---|---|---|
| **TL1A**, minibinder | 2RE9: 2.1 Å, native homotrimer; all three chains match current UniProt; observed 88–251, missing 72–87; author numbering +71 | Compact rigid protomer, accessible external patches, compatible with backbone/sequence design; explicitly check trimer and N133/N229 glycans |
| IL-7Rα, minibinder | 3DI3 B: 2.9 Å, observed UniProt 37–229, I66T relative to current canonical sequence; receptor in IL7:IL7RA heterodimer | Two-domain geometry, six annotated N-glycosylation sites and target-state uncertainty add difficulty |
| TNF-α, VHH | 1TNF: 2.6 Å trimer, observed 82–233; L219 versus native D219 | Compact target, but the permitted format needs a complete VHH; a generic 60 aa minibinder would be ineligible |

This is a resource/scientific choice, not a demonstrated superiority ranking.
The subsequent predictions show substantial loop/state sensitivity; the native
beta scaffold must be distinguished from flexible regions when aligning models.
EGFR/TfR1 add larger glycosylated ectodomains and VHH design; EGFR also adds a pH
endpoint. GLP-1R adds agonism, which binding the supplied ectodomain alone cannot
establish. TNFR1 also requires VHH. We concentrate on TL1A before opening another track.

Primary evidence: [TL1A structure](https://www.rcsb.org/structure/2RE9),
[UniProt TL1A](https://www.uniprot.org/uniprotkb/O95150/entry),
[TL1A biochemistry](https://pubmed.ncbi.nlm.nih.gov/19522538/),
[DcR3 complex for target-site context](https://www.rcsb.org/structure/3K51),
[IL7 receptor structural study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2654238/),
[1TNF](https://www.rcsb.org/structure/1TNF).
The IL7 study found strong glycosylation effects for **natural IL7 binding**;
we do not transfer its quantitative effect to a designed binder.

## Actual data and limitations

`targets/` contains the seven exact official construct FASTAs, source hashes,
priority-target sequence/coordinate audits, residue mapping, and observed native
coordinates. `targets/TL1A/surface.csv` reports trimer SASA, burial by adjacent
subunits, glycan-anchor distances and receptor-context contacts. No missing region
has been filled with invented coordinates. Official full sequences remain intact.

Real generations, sequence/geometry measurements, FASTA/CSV and backbone poses are
in [results/tl1a_20261101](results/tl1a_20261101). `synthetic: false` means an actual
model produced the design; it does not mean binding has been demonstrated.
Generation provenance records upstream commits, checkpoint hashes, seeds, masks,
temperature, resolved settings and input coordinates. No same-target known-positive
binder sequence/scaffold was supplied. This is not an exhaustive novelty search.

The first two monomer-conditioned backbones produced eight distinct 60 aa sequences.
One family had 122 backbone clashes below 2 Å against adjacent native subunits,
despite RFdiffusion confidence 0.99089. The other had no such severe clashes but
only 23 CA contact pairs within 8 Å. A full-trimer design added four sequences;
two outer-site backbones added eight, but were long rods with no nonlocal CA
contacts. Radius-of-gyration guidance then produced eight 80 aa sequences with
75–78 nonlocal contacts. All 28 sequences and failed families are retained. Design poses
have binder backbone atoms only; they are **not independent sequence refolds**.

The actually executed independent evaluation uses Boltz-2 2.2.1, full official target 72–251,
target-only MSA, no templates/pocket constraints, seed 2026 and one sample. The
public target MSA has 2,286 records; both 64- and 512-depth configurations were
run. Batch runs seed the global RNG once, so file order affects subsequent draws;
the final single-candidate trimer run has its own seed 2026. A monomer prediction
is checked against the observed native trimer. These are
explicit approximation, not a recreation of official ESMFold2 screening. The
Boltz protein–small-molecule affinity head is not used to claim protein–protein KD.

| Actual diagnostic | Finding |
|---|---|
| Highest 64-depth monomer ipSAE | 0.7313 for `tl1a_outer_3_s4`, but its target beta-core RMSD is 6.89 Å and its backbone is a long rod; rejected |
| Target-only control, MSA 64 → 512 | Beta-core RMSD 6.63 → 0.45 Å; all-observed-residue RMSD 11.23 → 5.33 Å; remaining loop motion makes whole-target alignment misleading |
| Native-target-templated checks | Two binder folds match design within 0.70 / 0.93 Å, but association/native trimer or glycan context fails; these scores have a separate namespace |
| Full trimer, MSA 512, **no template** | `tl1a_receptor_0_s1`: target beta-core RMSD 0.326 Å, binder-only RMSD 0.702 Å, binder-target ipTM 0.387, ipSAE(10) 0.000, ipSAE(15) 0.005; no supported interface |

Native-context geometry is computed after aligning experimental **SHEET beta-core
residues**, while whole-target RMSD is preserved separately. The final trimer
result supports folding of the candidate, not binding. We stopped expanding these
families. `compute_priority.csv`/FASTA contain two **unvalidated variants for
another predictor**, selected from four still-unmeasured geometry survivors;
`evaluated_triage.json` contains no accepted evaluated candidate. These are not a
20-sequence competition submission.

## Run the real pipeline

The tested native Windows environment uses Python 3.10.16, torch 2.0.1+cu117,
DGL 1.1.2+cu117 and e3nn 0.3.3 for generation. An isolated Boltz environment uses
torch 2.5.1+cu121 and boltz 2.2.1. Both use the existing RTX 4060 8 GB; generation
and independent GPU evaluation run sequentially. Weights, environments, caches
and logs stay in ignored `build/`. No paid resource or existing Colab session is used.

```powershell
# Requires uv and network access; all downloads stay under this checkout's build/.
./scripts/setup_gpu.ps1 -WithBoltz

build/env-rfd/Scripts/python.exe scripts/run_rfdiffusion.py --out build/tl1a_receptor --backbones 2 --sequences 4
build/env-rfd/Scripts/python.exe scripts/run_rfdiffusion.py --context trimer --out build/tl1a_trimer --start 2 --backbones 1 --sequences 4
build/env-rfd/Scripts/python.exe scripts/run_rfdiffusion.py --site outer --out build/tl1a_outer --start 3 --backbones 2 --sequences 4
build/env-rfd/Scripts/python.exe scripts/run_rfdiffusion.py --site outer --compact --length 80 --out build/tl1a_compact --start 5 --backbones 2 --sequences 4
build/env-rfd/Scripts/python.exe scripts/evaluate_designs.py build/tl1a_receptor build/tl1a_trimer build/tl1a_outer build/tl1a_compact

# Choose an explicit prediction cohort, inspect its native trimer/glycan context.
build/env-boltz/Scripts/python.exe scripts/run_boltz.py results/tl1a_20261101/candidates.json --ids tl1a_receptor_0_s1 --out build/boltz_eval
build/env-boltz/Scripts/python.exe scripts/import_boltz.py results/tl1a_20261101/candidates.json build/boltz_eval
build/env-boltz/Scripts/python.exe scripts/run_boltz.py results/tl1a_20261101/candidates.json --ids tl1a_receptor_0_s1 --trimer --msa-depth 512 --out build/boltz_trimer_deepmsa
build/env-boltz/Scripts/python.exe scripts/import_boltz.py results/tl1a_20261101/candidates.json build/boltz_trimer_deepmsa
python.exe scripts/triage_candidates.py results/tl1a_20261101/candidates.json --k 2
```

The runner applies one recorded Windows compatibility patch: converting NumPy's
int32 sequence tensor to torch int64 in RFdiffusion. ProteinMPNN paths use forward
slashes and an explicit checkpoint directory. It verifies the actual generated
chain before sequence design. ProteinMPNN designs only that chain, omitting Cys,
with `v_48_020`, T=0.1, seed 2026; targets are fixed. This can be adjusted for later
experiments. Minibinder length is checked at 40–100 aa.

Generation imports rebuild the dataset; run independent prediction imports
afterward. Complete executed evaluation configurations and hashes are retained
in `results/tl1a_20261101/*evaluation.json`, with per-candidate context references.
Templated, monomer/trimer and MSA-depth experiments never share a score namespace.

**Current compute boundary:** native Windows dependency resolution for BoltzGen
0.3.2 fails because its required `cuequivariance-ops-cu12` has Linux wheels only;
there is no installed WSL distribution. The full public ESMFold2/ESMFold2-Fast
models include a 6B language-model backbone; their native GPU weights exceed
this 8 GB card before activations. Neither was run or substituted for official
screening. The next generator is prepared in [gpu/run_boltzgen.sh](gpu/run_boltzgen.sh)
and [its full-trimer specification](gpu/tl1a_boltzgen.yaml), for a supported free
Linux GPU allocation. That job is **unexecuted** and uses the original RCSB mmCIF
with polymer metadata. No sponsor account, credential or paid allocation was used.

## Reuse the mathematical tools

```sh
python -m pip install -e .
python scripts/triage_candidates.py results/tl1a_20261101/candidates.json --k 2
python -m changjiang_select select build/real_triage/eligible_candidates.json --scenarios build/real_triage/proxy_scenarios.json --k 2 --out build/real_selection
python -m changjiang_select select build/real_triage/eligible_candidates.json --scenarios build/real_triage/proxy_scenarios.json --method exact --k 2 --out build/real_exact
python -m unittest discover -s tests -v
```

The triage gate uses observed trimer clashes and an explicitly heuristic glycan
envelope. Oriented metric midranks define source-balanced coverage viewpoints;
missing official scores are not imputed. It reports Spearman correlations,
backbone dependence, leave-one-source-out sensitivity, and portfolio versus exact
optima. A documented exact-objective tie can prefer more backbone families for
exploration, preserving the fitted objective. These uncalibrated utilities allocate compute; they are not success
probabilities, official scores, KD estimates or a final experimental shortlist.
The selected two-column CSV/FASTA are research drafts, not an official upload.

The original NumPy-only import/validation, raw Top-K, singleton Top-K, greedy,
safeguarded portfolio, exact optimization, diversity and export CLI remain intact.
The fixed-scenario best-of-set objective is monotone submodular and greedy has
the classical `1-1/e` bound. Synthetic examples remain under `examples/synthetic/`:

```sh
python -m changjiang_select select examples/synthetic/candidates.csv --scenarios examples/synthetic/scenarios.json --k 3 --out build/toy
python -m changjiang_select benchmark
```

See math/PROOF.md for the original 90-instance synthetic benchmark and its limitations.
No synthetic scores or sequences are mixed with the real TL1A results.
