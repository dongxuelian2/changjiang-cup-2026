# Candidate portfolio mathematics

## 1. Objective and interpretation

For one confirmed track, let `V={1,...,N}` be distinct candidate sequences and
`k <= 20`. Scenario `s` assigns nonnegative, higher-is-better utility `u[i,s]`
to each candidate. Weights satisfy `w[s] >= 0` and `sum(w)=1`. Define

$$F(S)=\sum_{s=1}^M w_s\max\left(\{0\}\cup\{u_{is}:i\in S\}\right),
\qquad \max_{|S|\le k} F(S).$$

The maximum reflects ranking by the team's best experimental candidate.
Utilities must be defined from the actual track's experimental endpoint and
preferences. Raw confidence metrics from different predictors do not supply
these utilities automatically. Section 6 uses **synthetic modelling inputs**;
section 8 applies an explicitly uncalibrated rank objective to real model output.

A joint scenario can set `u[i,s]=a[i,s]*r[i,s]`, where `a` is an admission /
expression / assay-validity mask and `r` is utility conditional on those stages.
Shared masks can describe family-wide, epitope-wide, model-wide, or target-state
failures. No independence between candidates is required. Neither sequence
distance nor a group label alone establishes the strength of those dependencies.

With binary utility the objective is weighted coverage. Under a genuine joint
outcome law, it equals the probability of at least one successful candidate;
this interpretation requires calibrated outcomes and weights. More generally,
for bounded nonnegative random utilities,

$$\mathbb E[\max_{i\in S}U_i]
=\int_0^B\Pr(\exists i\in S:U_i>t)\,dt.$$

Thus maximizing the expected count of successes is a different decision problem.
The choice of utility scale also expresses risk preferences; it is not fixed by
the best-candidate ranking rule alone.

## 2. Proved structure and algorithms

**Proposition (normalized monotone submodularity).** For fixed utilities and
weights above, `F(empty)=0`, `F` is monotone, and it has diminishing returns.

**Proof.** Put `b_s(S)=max({0} union {u[i,s]: i in S})`. The marginal gain is

$$\Delta(i\mid S)=\sum_s w_s(u_{is}-b_s(S))_+.$$

It is nonnegative. If `A` is a subset of `B`, then `b_s(A) <= b_s(B)` in every
scenario, so `Delta(i|A) >= Delta(i|B)`. Summing with nonnegative weights proves
the claim. Correlation of the columns' joint outcomes does not enter the proof.

Greedy selects the largest current marginal gain, recomputing after each choice.
It uses `O(k*N*M)` arithmetic with the current NumPy implementation and `O(N*M)`
memory. Equal gains are broken by singleton utility, then input order. This
remains a valid greedy rule, including when the sampled objective is saturated.

**Theorem (classical cardinality guarantee).** For `1 <= k <= N`, greedy's
set `G_k` satisfies

$$F(G_k)\ge\alpha_k\,\mathrm{OPT},\qquad
\alpha_k=1-(1-1/k)^k\ge 1-1/e.$$

**Proof.** For any optimum `O`, submodularity and monotonicity give
`F(O)-F(G_t) <= sum_{i in O\G_t} Delta(i|G_t)`.
There are at most `k` terms; greedy's next gain is at least
`(OPT-F(G_t))/k`. Iterating the residual contraction for `k` steps proves the
bound. When `k=0`, the empty set is optimal. This is the
[Nemhauser–Wolsey–Fisher cardinality result (1978)](https://doi.org/10.1007/BF01588971),
not a new approximation theorem.

`singleton-topk` ranks `F({i})`, while `raw-topk` uses one explicitly named raw
metric and its direction. Neither updates marginal gains. `portfolio` returns
the better of greedy and singleton Top-K **on the fitted scenario objective**.
It inherits `alpha_k` and cannot be worse than that baseline on this objective.
For `k=2`, greedy already dominates singleton Top-K because both start with the
best singleton; the safeguard matters at larger budgets.

**Counterexamples retained as tests.** For `k=2`, four equal-weight scenarios,
and rows `(1,0,1,0), (1,1,0,0), (0,0,1,1)`, greedy obtains `3/4` while the
last two rows obtain `1`. Greedy need not always dominate singleton Top-K for
`k=3`: the six-row integer example in `test_portfolio_fallback_is_needed_for_some_instances`
gives `2.5` versus `8/3`. The wrapper preserves the better result.

Exhaustive search visits all `C(N,k)` size-`k` subsets. By monotonicity an optimum
of that size is optimal for the at-most-`k` problem. A combination-count limit
prevents accidentally starting a large exact search.

**Proposition (computable upper bound).** For any set `S`, with `d_(j)` the
marginal gains of *all* candidates sorted in descending order,

$$\mathrm{OPT}\le \min\left\{\sum_s w_s\max_i u_{is},\quad
F(S)+\sum_{j=1}^{k}d_{(j)}\right\}.$$

**Proof.** The first bound allows a different candidate for every scenario.
For the second, `F(O) <= F(S union O) <= F(S)+sum_{i in O\S}Delta(i|S)`;
there are at most `k` nonnegative terms. The CLI reports this upper bound and
`F(S)/bound` as a lower bound on the objective's approximation ratio. This is
an optimization bound, not an experimental confidence interval.

## 3. When simple ranking suffices

**Proposition (aligned scenarios).** If a candidate `i*` weakly dominates every
other candidate in every scenario, any nonempty set containing it is optimal.
In particular, if `u[i,s]=q[i]*z[s]` with `q,z >= 0`, Top-K on `q` is optimal
for every positive budget.

**Proof.** Its row attains the per-scenario maxima simultaneously. That attains
the first upper bound above. A quality-diversity penalty would add an unsupported
preference to this case. The aligned benchmark is a control for this result.

## 4. Model error and statistical estimation

**Proposition (surrogate regret bound).** Suppose the fitted objective `Fhat`
and the true decision objective `F` differ by at most `delta` on every feasible
set. If a method achieves `Fhat(G) >= alpha * max Fhat`, then

$$F(G)\ge\alpha\,\mathrm{OPT}_F-(1+\alpha)\delta.$$

**Proof.** Compare to a true optimum `O`: `F(G) >= Fhat(G)-delta >=
alpha*Fhat(O)-delta >= alpha*F(O)-(1+alpha)*delta`.

On a common scenario support, if both utility tables are in `[0,B]`, every
entry differs by at most `epsilon`, and weights are `w` versus `v`, then
`delta <= epsilon + (B/2)*||w-v||_1`. A maximum changes by at most `epsilon`;
the weight change for any vector in `[0,B]^M` is bounded by `B` times total
variation. This does not estimate unknown error or account for unmodelled
outcome mechanisms without additional evidence.

**Proposition (finite-sample bound under explicit assumptions).** Suppose joint
scenario vectors are IID from the true law and all utilities lie in `[0,B]`.
For `L=sum_{j=0}^k C(N,j)`, `M` equally weighted scenarios, and `eta in (0,1)`,
with probability at least `1-eta`, simultaneously for all feasible sets,

$$|\widehat F(S)-F(S)|\le B\sqrt{\frac{\log(2L/\eta)}{2M}}.$$

**Proof.** For a fixed set, its sampled maximum is IID in `[0,B]`; apply
[Hoeffding's inequality (1963)](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)
and union-bound over `L` sets. The bound includes sets
selected after examining the sample. Candidate outcomes within each scenario
may be correlated. Predictor ensembles are not automatically IID draws from
the experimental outcome law. The result explains why solving a small sampled
objective exactly need not improve unseen outcomes.

**Explicit misspecification example.** Fitted rows `A1=(.8,.8)`,
`A2=(.79,.79)`, `B=(0,1)` with equal weights and `k=2` give singleton Top-K
`.8`, portfolio `.9`. If the true one-scenario rewards are `(0,1,0)`, the
baseline gets `1` and portfolio gets `0`. The approximation theorem cannot
protect against a wrong joint model.

## 5. Where the assumptions can break

Admission masks must be fixed jointly across all possible portfolios. If the
organizer instead applies a portfolio-dependent quota, the effective utility
may depend on the submitted set. With one wet-lab slot, suppose `a` has reward
`1`, `b` reward `0`, and screening ranks `b` above `a`: `{a}` yields `1` but
`{a,b}` yields `0`. Monotonicity fails. The actual screening workflow must be
read before using an admission model for competition decisions.

Likewise, replacing the weighted average by a worst case over uncertain weights
does not preserve submodularity in general. Take `F1(S)=1[a in S]` and
`F2(S)=1[b in S]`. Their minimum is zero on each singleton and one on the pair,
so it violates diminishing returns. A robust optimizer needs its own analysis;
calling the current greedy algorithm “robust” would be incorrect.

## 6. CPU experiments actually run

Command: `python -m changjiang_select benchmark`. NumPy 2.2.5, Python 3.13.1,
Windows 11, AMD64 Family 25 Model 116. No GPU. Thirty seeds `0..29` per regime,
`N=12`, `k=3`, `M=64`; all `220` sets enumerated per instance. Evaluation uses
`4096` independent joint scenarios from the same specified toy law. The generator
defines family activation probabilities `(.72,.62,.55,.45)`, qualities uniform
on `[.75,1]`, and clustered per-candidate survival `.95`. The clustered raw score
adds a family-0 bias of `.18`. These are synthetic choices.

| Regime / method | Mean fitted F | Mean F / exact | Minimum F / exact | Mean held-out F |
|---|---:|---:|---:|---:|
| Aligned / all methods | .680494 | 1.000000 | 1.000000 | .687782 |
| Independent / raw Top-K | .919998 | .967960 | .883488 | .917899 |
| Independent / singleton Top-K | .925710 | .974237 | .925227 | .915957 |
| Independent / greedy | .946044 | .995710 | .973024 | .917517 |
| Independent / portfolio | .946124 | .995792 | .973024 | .917326 |
| Independent / exact | .950089 | 1.000000 | 1.000000 | .916570 |
| Clustered / raw Top-K | .685600 | .744213 | .589891 | .670778 |
| Clustered / singleton Top-K | .799350 | .867419 | .651584 | .770631 |
| Clustered / greedy and portfolio | .919824 | .998281 | .966556 | .896138 |
| Clustered / exact | .921447 | 1.000000 | 1.000000 | .898320 |

Portfolio versus singleton on held-out scenarios: clustered `28 wins / 2 ties /
0 losses`; independent `10 / 7 / 13`; aligned `0 / 30 / 0`. Clustered improvement
is `.125507` absolute, about `16.29%` relative to the singleton mean. Independent
improvement is only `.001369`, and the raw baseline's mean is higher. This does
not support replacing simple ranking universally. The family correlation and
score bias in this benchmark were deliberately constructed, not learned from
protein experiments.

Median selection times across the three regimes were approximately `.007 ms`
for singleton Top-K, `.049 ms` for portfolio, and `.84 ms` for exhaustive search.
These short measurements are indicative overhead measurements, not stable
hardware performance claims. Clustered portfolio's mean Jaccard overlap with
singleton Top-K was `.3633`, showing that similar utility improvements can
require materially different sets.

Independent regime, varying fitted scenario count with the **same held-out data
and candidates** for each seed:

| M | Mean held-out portfolio minus singleton | Wins / ties / losses |
|---:|---:|---:|
| 16 | .007105 | 15 / 7 / 8 |
| 64 | .001369 | 10 / 7 / 13 |
| 256 | .005174 | 12 / 9 / 9 |
| 1024 | .010222 | 15 / 12 / 3 |

Larger samples help in this toy study at `M=1024`, but improvements are not
monotone over these finite seeds. Statistical calibration and joint failure
modelling matter more than closing the remaining small optimization gap.

CPU scale check: `N=120, M=256, k=20`, seed `2026`; no exact search:

| Method | Fitted F | Held-out F (8192 scenarios) | Selection time (ms) |
|---|---:|---:|---:|
| Raw Top-K | .658011 | .717141 | .072 |
| Singleton Top-K | .896350 | .893796 | .071 |
| Greedy | .978627 | .968916 | 1.244 |
| Portfolio | .978627 | .968916 | .840 |

Tests also exhaust **all 4096 binary utility matrices** with four candidates,
three equal-weight scenarios and `k=2`, comparing against an independent integer
oracle and checking the `.75` bound and optimum upper bound. These finite checks
validate the implementation; the general guarantee comes from the proof above.

The CLI and synthetic generator have been exercised end to end with both CSV
and JSON. Tiny checked-in examples use `N=8, M=8`; the larger benchmark outputs
are generated in ignored `build/benchmark/`.

## 7. Primary protein-design sources and implications

- [ProteinMPNN documentation](https://github.com/dauparas/ProteinMPNN) defines its
  sequence score as a mean negative log probability over designed residues.
  Preserve the scoring mask, checkpoint and temperature; this is a sequence
  model score, not a binding success probability.
- [RFdiffusion (Watson et al., 2023)](https://www.nature.com/articles/s41586-023-06415-8)
  generates backbones conditioned on design specifications. Keep backbone and
  sequence-design provenance separately when the two stages use different models.
- [BindCraft (Pacesa et al., 2025)](https://www.nature.com/articles/s41586-025-09429-6)
  uses AlphaFold2 during design and subsequent prediction/filtering. Our inference:
  agreement among related AF-based outputs should not automatically be treated
  as independent evidence about experimental success.
- [BoltzGen documentation](https://github.com/HannesStark/boltzgen) already provides
  ranked designs and a final diversity budget. Future comparisons should retain
  the generating pipeline's own selection as an additional baseline.

These sources motivate provenance and independent evaluation. No target-specific
experimental probability calibration has been inferred from them in this stage.

## 8. Real TL1A designs: separating admission, affinity and triage

### Official screening: verified facts and code inspection

The November 1 public rules specify per-sequence ESMFold2/ipSAE ranking with
target-only MSA, one seed/sample and no template. They do not publish an exact
software/configuration bundle; see [RULES.md](../RULES.md). A public implementation
with a matching name is not evidence of matching organizer weights or defaults.

We inspected the actually downloaded `DunbrackLab/IPSAE` revision
`6174cf9e71cb1bd660cc805856a18c4871a6dec3`. For its core residue-normalized
ipSAE, let `J_i={j in other chain: PAE[i,j]<cutoff}`. It computes a row mean
of `1/(1+(PAE[i,j]/d0(|J_i|))^2)` over `J_i`, then the best alignment row.
It reports both directional values and a symmetric maximum. Thus low interchain
PAE, the number of accepted residues and the normalization affect the score;
changing cutoff can change both the population and its normalization.

**Specific to this revision**, coordinate distance cutoff contributes to contact
statistics, not the main ipSAE formula. pLDDT and hydropathy are also not terms in
that formula. Interface geometry can help diagnose physically implausible models
without being the official scalar score. For a trimer, never take an unrestricted
maximum over all chain pairs: a confident target–target interaction can swamp a
failed binder. Retain both directions and only binder–target pairs; official
stoichiometry and aggregation are still unknown. No PAE is inferred from a design
backbone or from RFdiffusion's own confidence.

### An actual failure found by adding native context

The first real batch used two 60-residue RFdiffusion backbones conditioned on
2RE9 A88–251, hotspots A187/A188, deterministic seeds 0/1, 50 diffusion steps,
and zero additional denoiser noise. ProteinMPNN `v_48_020`, seed 2026, T=0.1,
four samples/backbone produced **eight distinct real sequences**. Only the
generated chain was redesigned; the target remained fixed. These are not toy
data and have no wet-lab evidence.

| First-batch backbone | RFdiffusion binder confidence | CA contact pairs <8 Å | Binder-backbone / other-subunit atom pairs <2 Å | Closest glycan ND2 anchor |
|---|---:|---:|---:|---:|
| seed 0 | 0.99246 | 23 | 0 | 12.98 Å |
| seed 1 | 0.99089 | 253 | 122 | 5.01 Å |

The contact count is measured after aligning the target to the **full observed
trimer**. The second pose's apparent large interface comes partly from occupying
another subunit. Both confidence values are almost one. The wrong sequence-only
rule "best ProteinMPNN score wins" also favors the rejected family (its best
score is 1.0701, versus 1.1393 nats/residue for seed 0). This finite observation
motivated full-trimer conditioning for the next run. It does not estimate any
model's general false-positive rate.

### Uncalibrated compute-allocation objective on real measurements

`proxies.py` maps a complete nonconstant metric to its oriented midrank divided
by `N-1`. Missing official scores are omitted, never set to zero. Metrics retain
their raw units in the candidate file. Every source family (metric prefix) has
equal total weight, divided among its available viewpoints. This limits the
extra weight a model would receive merely by exposing more confidence heads.
It does not establish statistical independence, and geometry/RFdiffusion remain
related. Leave-one-metric and leave-one-family-out selections expose sensitivity.

The same existing `ScenarioObjective`, portfolio/greedy and exhaustive solvers
operate on these columns. Their submodular guarantee still holds because the
matrix is fixed and nonnegative. **The guarantee concerns rank coverage only**;
the weights are decision assumptions, not a joint experimental outcome law.
Percentiles are cohort-dependent and must be recomputed after explicit filtering;
they cannot be compared across cohorts as absolute biological utilities.

The initial geometry gate rejects any noncanonical/wrong-length sequence, any
unconditioned-subunit backbone clash below 2 Å, zero CA contacts within 8 Å, or
a backbone closer than 12 Å to an N133/N229 ND2 anchor. The 12 Å exclusion is a
heuristic for allocating prediction compute, **not** a glycan conformational
simulation or a calibrated assay threshold. It cannot rule out sidechain clashes
because design poses lack binder sidechains. All rejected records are retained.

On the first eight sequences this leaves four variants of one backbone. Their
geometry and generating-model confidence are identical, so those columns are
omitted as constant. A `k=2` run of the existing CLI gave portfolio/exact rank
coverage **0.916667 / 0.916667**. Alternative optimal sets exist; this agreement
does not establish better affinity or meaningful epitope diversity.

`triage_candidates.py` reports pairwise-complete Spearman correlations with
sample counts and nulls for constant columns. Sequence variants share backbone
measurements, so eight sequences are only two backbone families. We do not report
inferential p-values or success probabilities. Cross-model correlations and
prediction failures must be assessed before expanding a high-scoring family.

### Biological hypotheses and limits

- A rigid external TL1A surface away from glycans is a useful minibinder target;
  its superiority to other competition targets is a research choice, not a theorem.
- The native trimer should be a more relevant steric context than an isolated
  protomer. The assay construct/glycoform and immobilization remain unknown.
- Full-target, template-free Boltz prediction with target-only MSA provides a
  second model view. Its ipSAE would be **Boltz-derived**, not an official ESMFold2
  score; even a high value would not predict KD without experimental calibration.
- Selecting the best over many seeds and then quoting that seed alone induces
  optimistic selection bias. Keep the one-seed run, failures and any later seed
  ensemble separate. Compare models on the same sequences/configurations, and
  examine family-level rather than pseudo-replicated sequence-only agreement.

## 9. Expanded real experiment and failure analysis (2026-11-01)

**Actually run:** seven RFdiffusion backbones and four ProteinMPNN samples each:
20 sequences of length 60 and eight of length 80, all distinct. Conditioning was
monomer A187/A188 (seeds 0/1), full observed trimer A187/A188 (seed 2), outer
A110/A123 (seeds 3/4), then outer A110/A123 with binder-ROG guidance (seeds 5/6,
80 aa, weight 1, min_dist 8, guide_scale 2, quadratic decay). No known binder
sequence or scaffold was supplied.

### Why generation confidence and low hydropathy were insufficient

The outer-site 60 aa designs had CA radii of gyration 25.42/25.55 Å and **zero**
CA contacts between residues separated by at least eight sequence positions.
They are long rods, including extremely charged sequences. Treating lower
hydrophobic-window content as uniformly better solubility would reward these
designs. The original proxy objective did exactly what its columns requested,
without establishing a globular binding module. Adding compactness guidance
produced Rg 17.62/17.54 Å and 75/78 nonlocal contacts at 80 aa. This is observed
geometric improvement, not demonstrated binding or stability improvement.

The final research gate additionally prioritizes at least ten nonlocal CA
contacts, charged fraction <=0.65 and hydrophobic fraction >=0.15. These are
declared **exploration assumptions for globular binders**, not competition rules
or validated developability thresholds. A designed helix could in principle
bind through coupled folding; these criteria intentionally give it lower priority.

### Independent model results and native-state controls

Boltz-2 2.2.1 / torch 2.5.1+cu121 produced **18 actual all-atom complex predictions
covering 12 distinct candidates**, plus two target-only controls. Each complex
has one sample, seed argument 2026, three recycles and 200 diffusion steps.
Multi-input runs seed the batch RNG once; input order affects later draws.
They are not matched-noise per-sequence replicates. Single-input runs avoid that
batch-order ambiguity. No result is an official ESMFold2 score.

| Context | Candidate / control | Target all-CA / beta-core RMSD (Å) | ipTM / ipSAE(10) | Interpretation |
|---|---|---:|---:|---|
| No template, MSA 64, monomer | highest-ipSAE `outer_3_s4` | 13.46 / 6.89 | .861 / .731 | High scalar score with incorrect target core and rod-like design; reject |
| No template, target only, MSA 64 | public target control | 11.23 / 6.63 | — | This approximate configuration is a poor target predictor |
| No template, target only, MSA 512 | public target control | 5.33 / .445 | — | Native beta core recovered; flexible regions still differ |
| No template, MSA 512, monomer | `receptor_0_s1` | 4.83 / .490 | .738 / .359 | Incompatible native trimer/glycan placement |
| Native target template, monomer | `receptor_0_s1` | .515 / separately recorded | .791 / .556 | Binder-only RMSD .700 Å, but clashes/glycan incompatibility; not official-equivalent |
| Native target templates, trimer | `receptor_0_s1` | .425 / .233 | .186 / .000 | No supported interface even with preserved target state |
| **No template, MSA 512, trimer** | `receptor_0_s1` | **2.052 / .326** | **.387 / .000** | Binder-only RMSD .702 Å, native-context clearance, but no confident binding |

The final row's ipSAE(15) is .005. It uses only binder–target pairs; target–target
confidence is excluded. Its glycan-anchor distance is 14.85 Å and it has no
<2 Å clashes with adjacent native subunits. Thus a successfully refolded sequence
and geometrically possible placement still do not establish association.

The beta core is the union of experimental chain-A `SHEET` records in 2RE9,
remapped +71 to UniProt, fixed independently of the predictions. Whole-target
RMSD is preserved separately. All binder-pose/trimer/glycan geometry uses a
beta-core alignment (derived geometry v2); flexible loops must not define that
rigid frame. `import_boltz.py` retains both alignment measures and checks chain
sequence, PAE and pLDDT indexing. Template-assisted scores have their own namespace
and cannot replace no-template evidence. The depth comparison is finite evidence,
not a controlled estimate of MSA's causal effect: global batch RNG draws differ.

### Correlation, optimization and what remains unknown

On the selected 12-sequence, seven-backbone MSA-64 cohort:

| Raw / oriented ranking comparison | Spearman rho |
|---|---:|
| Better ProteinMPNN score versus Boltz ipSAE(10), direction corrected | .042 |
| RFdiffusion generating confidence versus Boltz ipSAE(10) | -.035 |
| Boltz ipTM versus Boltz ipSAE(10) | .895 |
| Better ProteinMPNN score versus Boltz binder pLDDT, direction corrected | .615 |

Related Boltz confidence heads strongly agree with each other; this is not
independent experimental evidence. Sequence-design quality is more associated
with the second model's fold confidence than with its interface score in this
small selected cohort. Shared backbones, selected inputs and model bias preclude
IID significance or KD-success estimates. Correlations remain descriptive.

After all measured checks, **none of the twelve independently evaluated
sequences passes the complete research gate**. The no-template context with the
largest available target state/depth takes priority; an exact zero ipSAE(10)
means no interface support at that local cutoff, not an official admission rule.
Four unmeasured variants remain eligible for further prediction, not wet lab.
At k=2, existing portfolio, singleton and exact solvers all give rank coverage
**.833333**. Portfolio chooses two variants of backbone 6, whereas an equally
optimal exact set covers two backbones. We retain the latter for exploratory
compute allocation: `tl1a_receptor_0_s2` and `tl1a_outer_6_s3`. This secondary
preference changes neither fitted objective value nor its guarantee; it is not
a proof of biological benefit. Their CSV/FASTA explicitly mean compute priority.

An earlier 20-sequence proxy run gave portfolio/exact .931818 versus singleton
.840909, but its initial geometry survivors included the rods later rejected.
That apparent optimization improvement therefore **does not establish candidate
quality improvement**. Better biological constraints mattered more than solver
precision. The final complete-case report deliberately contains no accepted set,
and stale nonempty shortlists are cleared when later evidence rejects them.

The current route has produced foldable hypotheses but no credible binder hit.
We stopped expanding these families. Native Windows resolution of BoltzGen 0.3.2
fails on mandatory Linux-only cuEquivariance CUDA wheels; WSL has no installed
distribution. Full current ESMFold2/Fast checkpoints include ESMC-6B, exceeding
8 GB in the native GPU path before activations. The prepared Linux BoltzGen
full-trimer job is unexecuted. Official checkpoint/MSA/ipSAE configuration and
experimental KD calibration remain unavailable. No paid compute, competition
account, or existing Colab session was used.


## 10. Budgeted context preservation: complementarity, not portfolio coverage

### Model and boundary of the claim

This section concerns **abstract context selection**, not selection of candidate
sequences. Let finite `V` be optional context units together with a mandatory
core `C`. Supplied motifs `H_j ⊆ V` have nonnegative rational weights `w_j`;
a motif contributes only if **all** its units are retained. Define

$$G(S)=\sum_j w_j\mathbf1[H_j\subseteq S],\qquad
 c(S)=\sum_{i\in S}c_i,\qquad
 \mathrm{OPT}(B)=\max_{C\subseteq S,\ c(S)\le B}G(S).$$

Costs are nonnegative rationals, and `c(C)≤B` is required. Empty motifs contribute
constants; duplicate motifs add their weights; zero costs are allowed. Conditioning
on the core replaces each motif by `H_j\C`, charges `c(C)` once, and preserves
already-completed motifs as a constant. Weights encode a declared **synergy
surrogate**, not physical fidelity, folding probability or affinity. Motifs and
weights must be supplied independently of the optimizer. No geometric interaction
radius, connectivity, contiguous sequence constraint, or uncertainty model is
silently assumed. Adding such constraints needs another analysis.

**Proposition (opposite curvature and hardness).** `G` is monotone supermodular.
Already for unit costs, empty core and rank-two motifs, its budget problem is
NP-hard. The candidate-portfolio `1−1/e` guarantee in section 2 does not apply.

**Proof.** For `i∉S`, the marginal is the sum of weights of motifs containing `i`
whose other units are all in `S`. This sum increases with `S`. One two-unit motif
has zero singleton gains but positive pair gain, violating submodularity. For a
simple graph, take its edges as unit-weight motifs and ask if a budget-`k` set
has reward at least `k(k−1)/2`. This is equivalent to a `k`-clique (the bound is
also valid for sets smaller than `k`). Thus the model includes densest-`k`-subgraph.

### What penalty tuning can and cannot solve

For any rational `λ≥0`, define

$$P(\lambda)=\max_{S\supseteq C}\{G(S)-\lambda c(S)\},\qquad
 U_\lambda(B)=P(\lambda)+\lambda B.$$

**Proposition (exact priced oracle; budget upper bound).** `P(λ)` is computable
by one maximum-flow/minimum-cut problem, and `OPT(B)≤U_λ(B)`. A feasible set
with `G(S)=U_λ(B)` is optimal for this surrogate.

**Proof.** Make a reward vertex for each motif with source capacity `w_j`, a
vertex for each optional unit with sink capacity `λc_i`, and an arc from each
reward vertex to each of its optional prerequisites. Give prerequisite arcs
capacity `W+1`, where `W=Σ_j w_j`. The cut putting all optional and reward vertices
on the sink side costs `W`, so a minimum cut cannot cut a prerequisite arc.
For a fixed source-side retained set, place every completed reward vertex on the
source side: the cut costs `W−G(S)+λc(S\C)`. Therefore its minimizing retained
set maximizes `G(S)−λc(S)`; the fixed core cost is subtracted afterward. This also
handles empty/core-only motifs. For any feasible `S`,
`G(S)≤P(λ)+λc(S)≤P(λ)+λB`, proving the bound and equality criterion.

**Counterexample (unbounded optimized-penalty gap, even rank two).** On the
complete graph `K_n`, `n≥3`, use unit node costs and edge rewards, `C=∅`, `B=2`.
Then `OPT=1`, whereas

$$\inf_{\lambda\ge0}U_\lambda(2)=n-1.$$

**Proof.** A size-`s` set has priced value `s(s−1)/2−λs`, strictly convex in `s`.
Every intermediate value is strictly below the chord joining sizes 0 and `n`;
only the empty or full set can maximize it, at any price. Hence
`P(λ)=max(0,n(n−1)/2−λn)`. The minimum of `2λ+P(λ)` occurs at
`λ=(n−1)/2`, giving `n−1`. Every two-unit motif itself fits the budget: the failure
is not caused by individually infeasible requirements. For `n=3`, the hard optimum
is 1 and the dual bound is 2. No denser price grid or min-cut tie-break returns a
pair in this example. A randomized choice of entire empty/full sets satisfies an
*expected* budget, not the hard budget on every realization.

### A rank-sensitive recovery result between tied priced optima

Now require **unit costs**. Suppose `C⊆A⊆T` are both *global* optimizers at the
**same** price `λ≥0`, and `|A|≤B≤|T|`, with integer `B`. Put
`m=|T\A|`, `k=B−|A|`, `a=G(A)`, `Δ=G(T)−G(A)`.
For a uniformly chosen size-`k` subset `R` of `T\A`,

$$L=\mathbb E G(A\cup R)=a+
 \sum_{\substack{H_j\subseteq T\\H_j\nsubseteq A}}
 w_j\frac{\binom{k}{r_j}}{\binom{m}{r_j}},\qquad r_j=|H_j\setminus A|,$$

where `binom(k,r)=0` for `r>k`. There is a deterministic feasible set with reward
at least `L`: reveal units one at a time, retaining the include/exclude branch
with larger conditional expectation, while keeping exactly `k` total additions.
The current expectation is a convex combination of these branches, so it never
decreases. Each branch's value is given by the same formula on the remaining
units. The final value is at least the initial expectation.

**Theorem (conditional rank certificate; rank-one exactness).** If `1≤d≤k≤m`
and every positive-weight motif counted in the sum has `r_j≤d`, this construction
returns a budget-feasible `S` satisfying

$$G(S)\ge L\ge a+\frac{\binom{k}{d}}{\binom{m}{d}}\Delta,
 \qquad \frac{G(S)}{\mathrm{OPT}(B)}\ge
 \frac{\binom{k-1}{d-1}}{\binom{m-1}{d-1}}$$

whenever `OPT(B)>0`. More informatively, report the instance-specific bracket
`G(S)≤OPT(B)≤U_λ(B)`. For residual rank one, the construction is exact.

**Proof.** The probability that all `r` specified additions are selected is
`binom(k,r)/binom(m,r)`, decreasing in `r`; this proves the expectation and its
rank lower bound. Equality of the two priced optima gives `Δ=λm` and therefore
`U_λ(B)=a+(k/m)Δ`. Write `p=binom(k,d)/binom(m,d)`, `q=k/m`. Since `a≥0` and
`p≤q`, `a+pΔ≥(p/q)(a+qΔ)=(p/q)U_λ(B)`. The desired ratio follows from
`U_λ(B)≥OPT(B)`, and `p/q=binom(k−1,d−1)/binom(m−1,d−1)`. For `d=1`,
`L=U_λ(B)` and feasibility forces equality. If `k=0` or `m=0`, `A` itself attains
the bound. If no positive residual motifs remain, `Δ=0` and every intermediate
set has reward `a`. These cases need no division. For `k<d` the rank-only bound
can be zero; use the exact expectation instead. The existence of a useful
same-price nested bracket is an assumption, not a guarantee of this routine.

**Sharpness of this dual-based ratio, not of actual solution quality.** Take all
`d`-subsets of `m` optional units as unit-weight motifs. At
`λ=binom(m,d)/m`, the empty and full sets are priced optima: for `s≥1`,
`binom(s,d)/s=binom(s−1,d−1)/d` is nondecreasing. Thus
`U_λ(k)=(k/m)binom(m,d)`, whereas every size-`k` set has reward `binom(k,d)`.
Their ratio is exactly the theorem's factor. The constructed set is actually
optimal here; the looseness is entirely in the dual bound. This includes the
clique obstruction above. One must not call the factor a tight performance ratio
for the rounding algorithm itself.

### Minimal implementation and falsifiable use

`math/context_motifs.py` implements exact-rational priced min-cut, a bound at a
**supplied** price, the exact conditional expectation, and its deterministic
cardinality rounding. `priced(price)` and `priced(price, maximal=True)` return
the inclusion-minimal and inclusion-maximal priced optima, allowing a bracket
check at that price. In the final residual network, source-reachable vertices
must belong to every source-side minimum cut; vertices that can reach the sink
must belong to none. The source-reachable set and the complement of the
sink-reachable set have no outgoing positive residual arc, hence define the
extremal minimum cuts. The routine does not search prices for a
bracket, enforce connectivity, or solve the general hard-budget problem.
For `N=|V|+number of
motifs+2` and `E=O(|V|+number of motifs+Σ_j|H_j|)`, the elementary Edmonds–Karp
oracle uses `O(N E²)` augmenting-path work (rational bit costs additional).
Rounding evaluates at most two expectations per optional unit; with set-membership
work it uses `O(|V|+m (|T|+Σ_j(1+|H_j|)))` set/arithmetic operations (rational and
binomial bit costs additional). This is proof-oriented
code for small inputs, not a protein-scale performance claim.

Run `python -m unittest discover -s tests -p test_context_motifs.py -v`.
Inputs are integer node labels, core, motifs, rational weights/costs, and a price
or cardinality bracket; outputs are retained sets, exact rewards and bounds.
Tests enumerate all 64 unweighted graphs on four nodes, compare priced solutions
with exhaustive search, and check seeded rational hypergraphs, conditioning,
zero costs, empty/duplicate motifs, clique gaps, complete-hypergraph sharpness,
and rank-one exactness. These finite checks validate code, not the general proof.
The existing candidate-portfolio implementation and tests are unchanged.

For the collaborator's later experiments, freeze the core, motif definition and
weights **before** selecting a context. Compare equal-budget contexts and retain
omitted-motif weight and residual ranks alongside independently measured context
errors. Mathematical predictions are falsifiable without protein data: a verified
same-price rank-one bracket must give zero surrogate optimization gap; complete
rank-two examples must have the stated penalty gap; the deterministic reward must
never fall below the exact restriction expectation. A separate empirical hypothesis
is that lower omitted-motif weight predicts fewer independently assessed context
failures at matched size. Failure of that association rejects the surrogate's
relevance, not the optimization proof. No structural experiments are run here.

**Novelty boundary and primary sources.** AND-preservation is weighted
[densest-`k`-subhypergraph](https://arxiv.org/abs/1605.04284), not a new problem;
maximum closure/min-cut is classical
[Picard (1976)](https://doi.org/10.1287/mnsc.22.11.1268), with an explicit
[closure construction in Hochbaum (2001)](https://hochbaum.ieor.berkeley.edu/html/pub/HPF-closure-Net2001.pdf).
The conditional-expectation argument is standard derandomization. The contribution
here is a separate context surrogate, an explicit failure diagnostic, and a
core-conditioned rank certificate with tested implementation—not a claim of
priority for these optimization ingredients or biological improvement.
Discontinuous fragments are also [prior art](https://arxiv.org/abs/2509.25479);
noncontiguity or multichain context alone is not claimed as novel.
[Kim et al. (2026)](https://arxiv.org/abs/2610.33195) study epitope excision with
contact/halo-based contiguous cropping. Those empirical findings motivate careful
context definitions; they are not universal sufficiency guarantees for this
abstract motif surrogate or evidence that it improves a real design.
