# Official rules checked 2026-11-01 (Asia/Shanghai)

Primary sources: [home](https://aiprotein.sjtu.edu.cn/),
[targets](https://aiprotein.sjtu.edu.cn/targets),
[rules](https://aiprotein.sjtu.edu.cn/rules). The public JavaScript application
was downloaded directly, including its target and rule text, rather than relying
only on an older search index. The site still calls the regulations interim;
later official notices prevail. `scripts/fetch_research_inputs.py` retrieves the
source and hashes into ignored `build/sources/downloads.json`. A fresh check on
2026-11-02 was unsuccessful (web retrieval failed; Windows and Ubuntu clients
received TLS EOF errors); the last successfully retrieved rule text remains
the November 1 version. This is not evidence that the rules have stayed unchanged.

## Confirmed tracks

| Target | UniProt | Inclusive range | Reference PDB/chains | Required format | Additional endpoint |
|---|---|---|---|---|---|
| EGFR | P00533-1 | Met1–Ser645 | 6ARU A | VHH | affinity at pH 6.0 |
| TNF-α | P01375 | Val77–Leu233 | 1TNF A/B/C | VHH | — |
| GLP-1R | P43220 | Ala21–Glu139 | 6X18 R | VHH | agonist function |
| TfR1/CD71 | P02786 | Cys89–Phe760 | 11QC A/B | VHH | — |
| TL1A/TNFSF15 | O95150 | Leu72–Leu251 | 2RE9 A/B/C | minibinder | — |
| TNFR1 | P19438-1 | Met1–Thr211 | 9S3I A | VHH | — |
| IL-7Rα/CD127 | P16871-1 | Met1–Gly236 | 3DI3 B | minibinder | — |

The accession and range define the target; the PDB is illustrative. Preserve
the full official target sequence even when a design uses only observed
coordinates. The ban on signal peptides applies to the **submitted binder**;
it does not authorize editing an official target range.

## Eligibility and submission

- At most **20 distinct de novo sequences per team per track**, plus a methodology
  summary. Teams may enter multiple independently ranked tracks.
- Minibinder: **40–100 aa**, one chain, one designed binding module.
- VHH: **100–150 aa**, one complete VHH variable domain including framework/CDRs.
- Only the 20 standard amino acids; mature binder only, without signal peptide,
  purification tag, Fc, or other fusion segments.
- Do not submit a known same-target experimentally positive sequence, or start
  affinity optimization from one. Target structures/literature may inform site
  selection; no such binder is supplied to this project's generator.
- The current website specifies CSV sequences and a design summary. Model weights,
  training data and code need not be released. Last successful submission before
  the deadline counts; submissions can be updated until then.
- A later sequence covering **>90%** of an earlier sequence at **100% identity**
  is a duplicate; antibodies are compared using CDRs only. First valid submission
  takes priority across teams. Exact duplicate detection in our existing importer
  is therefore useful but does **not** implement the full official novelty check.
- The site gives **2026-11-30** as registration/sequence deadline, then shortlist
  in mid-December, wet lab in late December, results at end of January 2027.
  Deadline hour/timezone is not specified in the public text.

## Admission and experimental ranking are different

The organizers provide 7,000 free measurements. Verified seed-team submissions
proceed to experiments. Open-registration sequences receive an individual score
and are admitted in descending score order:

| Published setting | Value |
|---|---|
| Structure prediction | **ESMFold2** |
| Ranking score | **ipSAE**, higher first |
| Seeds / samples | **1 / 1** |
| Templates | none |
| MSA | target only; none for designed binder |

Only the team's best experimental candidate counts within each track; **measured
KD ascending** determines the main ranking. Developability/cell activity bonuses
do not change it. Selected sequences and results, including failures, are planned
for public release; nonselected sequences are not.

## Reproducibility gaps — do not silently fill them

The public rules name the model but **do not pin** ESMFold2 checkpoint/variant,
weight revision, `esm`/Transformers version, numeric seed value, recycle/loop
count, diffusion steps, precision, target MSA database/date/depth/pairing,
oligomer stoichiometry, or treatment of target signal peptides/glycans. They do
not specify ipSAE revision, PAE cutoff, distance cutoff, directional aggregation,
or which chain pair(s) determine a trimer–binder score. Admission thresholds,
track allocation, assay antigen expression/glycoform and KD protocol are also
unpublished in the retrieved public text. These remain unresolved; the website
does not provide a downloadable reproducibility bundle.

[Biohub's implementation](https://github.com/Biohub/esm) is available as
`esm==3.4.1.post1`; its public ESMFold2 uses ESMC-6B and supports MSA/multichain
input. [Transformers support](https://huggingface.co/docs/transformers/model_doc/esmfold2)
starts at v5.16.0. **Neither is verified as the organizer's build**. ESM-2,
ESMFold v1, ESMFold2-Fast, Boltz and AlphaFold are not interchangeable with the
named official model. The local driver reports CUDA 12.9; the current upstream
BoltzGen documentation also adds a CUDA-13 ESMFold2 evaluation dependency.

[DunbrackLab/IPSAE](https://github.com/DunbrackLab/IPSAE) was obtained at
`6174cf9e71cb1bd660cc805856a18c4871a6dec3`. Its command is
`python ipsae.py PAE_FILE STRUCTURE_FILE PAE_CUTOFF DIST_CUTOFF`. This is a
**local analysis version**, not an official version declaration. Real PAE and
matching residue/chain order are required; backbone geometry and RFdiffusion
confidence cannot be converted into ipSAE. Any alternative predictor score must
retain its model/configuration namespace and must not be called an official score.

No competition account, password, registration, or final submission was used.
