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
these utilities automatically. In this repository, all numerical utilities
and weights are **synthetic modelling inputs**.

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
