# Candidate selection and mathematical guarantees

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
[compute_priority](../results/tl1a_boltzgen_20261101/compute_priority): at k=2,
portfolio/singleton/exact all give **0.85**, choosing one sequence per eligible
backbone. Its saved inputs reproduce the allocation before prediction feedback.
Ranks are cohort-relative; these two fitted objective values cannot be compared
as measures of binder quality.
Research-priority CSV/FASTA are not official submissions.
