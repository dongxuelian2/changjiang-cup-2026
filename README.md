# Changjiang Cup 2026: candidate portfolio research

Mathematical and algorithmic tools for selecting up to 20 distinct protein
designs per track in the [2026 Changjiang Cup](https://aiprotein.sjtu.edu.cn/).
The aim is to improve the team's best experimental candidate, accounting for
uncertain quality and correlated failures.

The current CPU implementation imports candidates, preserves model-specific
measurements, measures diversity, and compares portfolio algorithms against
exact small-instance optima. The checked-in examples and all reported numerical
experiments are **synthetic**. Official information and the current access
limitations are recorded in [RULES.md](RULES.md).

## Implemented

- CSV and JSON import with metric definitions, model versions, run provenance,
  target identity, structure references, binding-site residue IDs and metadata.
- Canonical amino-acid validation, unique candidate IDs, missing-score handling,
  and normalized duplicate-sequence detection within a target.
- Sequence edit similarity, epitope Jaccard similarity and group diversity summaries.
- Raw-metric Top-K, singleton-utility Top-K, marginal-gain greedy, a safeguarded
  portfolio method, and exhaustive optimization for small instances.
- Selected FASTA, complete selected records, individual rankings, marginal
  contributions, objective values and computable optimum upper bounds.
- Seeded synthetic benchmarks, held-out scenario evaluation, scenario-count
  sensitivity experiments and correctness tests. Runtime dependency: NumPy only.

## Mathematical method

For nonnegative scenario utilities `u[i,s]` and normalized nonnegative weights:

$$F(S)=\sum_s w_s\max\left(\{0\}\cup\{u_{is}:i\in S\}\right),\quad |S|\le k.$$

The objective is normalized, monotone and submodular. Greedy attains at least
`1-(1-1/k)^k >= 1-1/e` of its optimum. `portfolio` returns the better of greedy
and singleton Top-K on this objective, retaining that approximation bound and
the baseline's value. Joint scenarios can represent shared failures and fixed
screening/admission outcomes without candidate independence assumptions.

[MATH.md](MATH.md) contains the proofs, optimum upper bound, surrogate-error and
finite-sample bounds, counterexamples, and full experimental findings. Scenario
weights are supplied modelling assumptions; raw predictor scores are retained
separately. Calibration is needed to interpret them as experimental outcomes.

## Run

Python 3.10 or newer:

```sh
python -m pip install -e .
python -m changjiang_select validate examples/synthetic/candidates.csv
python -m changjiang_select select examples/synthetic/candidates.csv --scenarios examples/synthetic/scenarios.json --k 3 --out build/selection
```

The installed `cj-select` command is equivalent to `python -m changjiang_select`.
On Windows, use `python.exe` if another program shadows `python`.

```sh
# Baselines and exact optimum use the same objective for comparison.
python -m changjiang_select select examples/synthetic/candidates.json --scenarios examples/synthetic/scenarios.json --method raw-topk --metric toy.raw_score --k 3 --out build/raw
python -m changjiang_select select examples/synthetic/candidates.json --scenarios examples/synthetic/scenarios.json --method singleton-topk --k 3 --out build/singleton
python -m changjiang_select select examples/synthetic/candidates.json --scenarios examples/synthetic/scenarios.json --method exact --k 3 --out build/exact
python -m changjiang_select similarity examples/synthetic/candidates.json --out build/similarity.json
python -m changjiang_select benchmark
python -m unittest discover -s tests -v
```

`select` writes `selection.json`, `selected.fasta` and `ranking.csv`. The JSON
preserves selected candidates' raw measurements and metric definitions alongside
the decision report. Rankings use singleton utility for scenario methods and
the specified metric for raw Top-K; selection order is a separate column.
`leave_one_out_loss` is a selected candidate's contribution to the final set.

Use `--target ID` to select one target from a larger file. Repeated sequences
cause selection to stop; `--deduplicate` explicitly retains the first row per
sequence and reports all duplicate IDs. Review repeated measurements before
using this policy. A budget larger than the available distinct candidates is
rejected. The CLI caps submission portfolios at 20.

Generate a different toy dataset with:

```sh
python -m changjiang_select synthetic --seed 7 --n 12 --m 64 --regime clustered --out build/example
```

The checked-in eight-candidate example can be regenerated with `--n 8 --m 8`.
All generated experiment outputs belong in ignored `build/`.

## Candidate and scenario formats

See the complete [JSON example](examples/synthetic/candidates.json),
[CSV example](examples/synthetic/candidates.csv), its
[metric metadata](examples/synthetic/candidates.metrics.json), and
[scenario table](examples/synthetic/scenarios.json).

A candidate JSON file has `schema_version: 1`, boolean `synthetic`,
`metric_definitions`, and a nonempty `candidates` list. Each candidate requires:

| Field | Meaning |
|---|---|
| `id`, `sequence` | Unique candidate identifier and designed amino-acid sequence |
| `target_id`, `design_type` | Exact target construct/track identifier and format |
| `source_model`, `source_version`, `run_id` | Generation provenance; use `unknown` explicitly when needed |
| `metrics` | Metric-key to finite number or `null`; scores remain in native units |
| `structures` | Optional references such as `{"path":"poses/design.cif","role":"complex","chain_id":"B"}` |
| `epitope_residues` | Optional strings such as `A:125`; all must use the same target numbering |
| `risk_group`, `structural_group` | Optional descriptive family/fold labels |
| `metadata` | Optional arbitrary object for seed, scaffold, confidence intervals, or assay context |

Each metric definition requires `model`, `version`, `meaning`, `units` and
`direction` (`higher`, `lower`, or `unspecified`). Namespace metrics by model,
version and context when their semantics differ. JSON preserves extension
fields. Sequence case and whitespace are normalized, retaining `input_sequence`
when normalization changes the original. Ambiguous residues are rejected.

CSV uses the same scalar fields and columns `metric:<key>`. `structures`,
`epitope_residues` and `metadata` are JSON-valued cells. Dataset provenance and
metric definitions live in `<stem>.metrics.json`, or a file supplied with
`--metadata`. Empty metric cells mean missing values; raw Top-K requires a
complete explicitly chosen metric and a specified direction.

A separate scenario JSON contains `schema_version`, `synthetic`, `description`,
`candidate_ids`, an `N x M` `utilities` array and `M` `weights` summing to one.
Every value must be finite and nonnegative. IDs must match the candidate file
before target filtering or deduplication; rows are aligned by ID. Supply a
different joint table with `--evaluation-scenarios` to evaluate a selected set
without selecting on that table. Record the utility transformation, assumptions
and weight rationale in `description`.

Generic records can carry adapted BoltzGen, RFdiffusion, ProteinMPNN, BindCraft
and structure-evaluation outputs. Native output parsers and large models are
not installed. Structure paths are references, preserved without loading files;
geometric comparison of coordinates requires a future structural analysis step.
Sequence similarity is `1 - Levenshtein/max(lengths)`, not alignment identity.
Missing epitopes produce `null`, not evidence of a distinct binding site.

## Actual results

Ninety small instances were solved on CPU: three regimes, seeds `0..29`,
`N=12`, `k=3`, 64 fitted scenarios and 4096 independent held-out scenarios.
Exhaustive search evaluated all 220 sets for each instance.

| Synthetic regime | Singleton Top-K / exact (mean) | Portfolio / exact (mean) | Held-out singleton F | Held-out portfolio F |
|---|---:|---:|---:|---:|
| Aligned quality ordering | 1.0000 | 1.0000 | .6878 | .6878 |
| Independent failures | .9742 | .9958 | .9160 | .9173 |
| Family-correlated failures | .8674 | .9983 | .7706 | .8961 |

Correlated failures support portfolio selection: held-out improvement over
singleton Top-K was about **16.3%**, with 28 wins, 2 ties and no losses. Independent
failures give weak evidence of improvement: 10 wins, 7 ties and 13 losses, and
raw-metric Top-K achieved a slightly higher held-out mean `.9179`. Aligned
ordering makes simple ranking optimal. Increasing independent fitted scenarios
to 1024 reduced portfolio losses to 3/30, though the finite-sample trend was
not monotone.

Median small-instance portfolio selection took about `.05 ms`, versus `.84 ms`
for exhaustive search on this machine. A `N=120, M=256, k=20` CPU check took about
1 ms for portfolio selection and gave fitted/held-out values `.9786/.9689`,
versus singleton Top-K's `.8964/.8938`. These are toy utilities and indicative
timings, not measured binder performance. A tested misspecification example
raises fitted utility from `.8` to `.9` while lowering true utility from `1` to `0`.

Tests include all 4096 binary four-candidate, three-scenario matrices against an
independent integer optimum, diminishing returns, approximation bounds, a
suboptimal greedy instance, the baseline safeguard, data validation, and a CSV
to FASTA/JSON command-line round trip.

The next decision is how official screening and assay endpoints should define
joint outcomes and utilities. Target-specific calibration and correlated
failure modelling are likely to matter more than the small remaining solver gap.
