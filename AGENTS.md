# Research workflow

A personal mathematical and computational protein-design project for the 2026 Changjiang Cup. Work directly on the scientific problem; mathematics and experiments take priority over project ceremony.

## Mathematics and experiments

- Seek methods that improve real binder design: geometric context reduction, structural validity, prediction reliability, candidate selection, and search efficiency. Prefer general insight, useful counterexamples, and testable algorithms.
- Read existing results first, then pursue promising approaches freely. Change assumptions or direction when calculations or evidence reveal a better problem.
- Use mathematical derivations and concise proofs where possible. Small Python experiments, exhaustive searches, and other computational tools are welcome when they answer a concrete question.
- Distinguish proved results, exhaustive finite computations, limited searches, empirical findings, and conjectures. State assumptions and applicability precisely.
- Use verified target sequences, structures, and current official competition rules. Keep original model scores, units, versions, and provenance. Do not interpret synthetic scores as measured binding affinity.
- Add structural processing, model integrations, or heavier computation when they contribute to a specific research goal. GPU-dependent work can be resumed when resources become available; do not initiate paid compute without authorization.

## Repository conventions

- Keep mathematical findings and short proofs in `MATH.md`, current official rules in `RULES.md`, and a concise description of actual results and usage in `README.md`.
- Retain useful source code, small input examples, and necessary tests. Put temporary calculations, generated results, caches, and large artifacts in ignored `build/`.
- Reuse existing programs and libraries. Avoid unnecessary CI/CD, cryptographic seals, certificate bundles, repetitive reports, dashboards, sprawling configuration, and procedural scaffolding.
- Do not create extra documentation or infrastructure merely to record each research round. Remove superseded material while preserving the results needed for further work.
- Run relevant checks before committing. Keep shared Git history intact; no force pushes. Use branches and PRs when collaboration requires them, not as mandatory steps for every experiment.

## Scientific interpretation

- Geometric consistency, predicted interface quality, experimental binding success, and measured `K_D` are distinct properties.
- State exactly what each proof or experiment establishes. Clearly label synthetic examples and limitations of available data.
