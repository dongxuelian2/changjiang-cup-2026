# Research workflow

This is a personal mathematical and computational protein-design project for the 2026 Changjiang Cup. Research progress comes before engineering ceremony. Read the existing mathematics and code, then work directly on a concrete problem; do not turn each experiment into a new project phase or report.

## Research priorities

- Attack problems that can improve real binder design: target-context reduction, geometric validity, model-selection bias, candidate screening, and efficient search. Prefer useful general results, counterexamples, or experimentally testable algorithms over elaborate infrastructure.
- Pursue promising ideas independently. When an approach fails, construct counterexamples, revise assumptions, or change direction rather than polishing an unproductive framework.
- Mathematical programs serve proofs and conjectures. Use Python for small symbolic, exhaustive, or numerical checks; use more efficient tools only when the problem requires them.
- Distinguish rigorous proofs, finite exhaustive checks, restricted searches, numerical observations, and conjectures. Keep precise assumptions, quantifiers, and counterexamples; do not describe a geometric certificate as evidence of experimental binding.
- Retain model score names, units, versions, and provenance. Synthetic outcome scenarios are modelling assumptions, not measured probabilities or binding affinities.

## Parallel research responsibilities

- **Mathematics stream (Natsu):** Focus on abstract mathematical formulation, proofs, constructions, counterexamples, complexity, approximation or robustness results, and small purely mathematical computations. Develop precise, reusable input/output contracts and predictions for the experiment stream to test.
- **CPU experiment stream (repository owner):** Continue the real-structure work: official target sequences and PDB interpretation, chains and numbering, oligomeric and glycosylation context, geometric implementations, baseline comparisons, and CPU benchmarks on actual structures.
- Avoid duplicate pipelines. The mathematics stream does not need to download/clean PDBs, engineer protein-structure workflows, run prediction models, or reproduce CPU structural benchmarks. The CPU stream can provide focused counterexamples or measurements that refine the mathematics.
- Keep the Colab/GPU-dependent binder generation and large structure-prediction runs deferred until those resources are available. Do not start paid compute without explicit authorization.
- For this collaboration, the mathematics stream works on a separate branch and **opens a PR for review without merging it**. The CPU stream may advance independently. Do not overwrite or rewrite the other stream's changes.

## Repository conventions

- Use `MATH.md` for mathematical statements, short complete proofs, constructions, and open questions. Keep `RULES.md` aligned with verified competition rules. The `README.md` should briefly state actual capabilities, verified results, and how to run the useful code; avoid defensive declarations and repeated process narration.
- Preserve the existing candidate-selection implementation and interfaces unless there is a concrete research reason to change them. Add only code, tests, and examples needed to support a result or avoid a real regression.
- Keep essential source code and small irreducible inputs under version control. Put generated structures, benchmark outputs, temporary files, caches, and logs into ignored `build/` or another ignored working area. Avoid uploading large model weights or binary datasets into Git.
- No self-imposed CI/CD, hash seals, certificate bundles, release packaging, sprawling configuration, dashboards, recurring status reports, archival folders, redundant independent checkers, or formal verification pipelines. Add such infrastructure only when specifically requested or essential to a concrete task.
- Do not build services, user interfaces, or framework abstractions merely to accommodate hypothetical future workflows. Reuse existing scripts and dependencies wherever possible.
- Validate important claims with the smallest meaningful test or calculation. Remove failed or superseded scaffolding while preserving useful mathematical findings and tests.
- Commit and push real work on the appropriate branch. Do not force-push or rewrite shared Git history. A PR is required for the Natsu mathematics handoff, not for every local research experiment.

## Current scientific boundary

- Target and submission constraints come from the competition's latest official rules, not from assumptions inferred from reference PDB structures.
- Geometric preservation, downstream prediction-score fidelity, experimental binder success, and measured `K_D` are separate claims requiring separate evidence.
- Prefer real data and clearly labelled benchmarks; never present synthetic candidates, assumed probabilities, or unrun calculations as experimental results.
