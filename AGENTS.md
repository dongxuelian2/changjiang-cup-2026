# Research workflow

## Zero-defensive-prose rule — mandatory

Apply this rule to all repository-authored English prose: README, mathematical research notes, `PROOF.md`, LaTeX exposition, documentation, pull-request descriptions, and code comments. Enforce it when auditing historical material and when writing anything new.

**Prohibited rhetorical constructions (including grammatical variants):**
- Chinese examples to recognize and eliminate in translated source material: `不是……`, `并不是……`, `不构成……`, `还没有做……`, `尚未……`, `而不是……`, `并未……`, `不意味着……`, `不代表……`, `不能据此……`, `这并不证明……`, `我们不声称……`, `不保证……`, `仅仅是……`.
- English equivalents to eliminate from authored explanations: `not a ...`, `is not ...`, `does not ...`, `do not ...`, `has not ...`, `have not ...`, `not yet ...`, `rather than ...`, `instead of ...`, `does not constitute ...`, `does not establish ...`, `does not imply ...`, `should not be interpreted as ...`, `we do not claim ...`, `no claim is made ...`, `merely ...`, `only a preliminary ...`, `beyond the scope ...`, `not intended to ...`.
- Delete disclaimers, self-exculpation, premature qualification, repetitive status caveats, and defensive contrasts. Do not relocate them to a different paragraph or replace them with synonyms serving the same rhetorical purpose. Zero such constructions are acceptable in explanatory prose.

**Required replacement style:** Lead with the actual theorem, deliverable, tested behavior, evidence, applicability conditions, exact scope, and concrete open mathematical question, using affirmative statements. Examples:
- `We prove the classification for n >= N under hypotheses H1 and H2.`
- `Verified cases: n = 2,...,7. General-case proof target: ...`
- `CI checks: 16 passed. Proof: Proposition 4. Research priority: external review.`
- `Algorithm: FPT for disjoint ordinary cardinality quotas. Next target: robust mixed-status source-SCC accessibility.`

**Mathematical truth takes precedence over stylistic cleanup.** Preserve essential logical negation in theorems, inequalities, counterexamples, constraints, source quotations, and executable code where changing it would alter correctness. In ordinary explanation, express boundaries by positive quantified scope and precise next objectives. If a claim is unsupported, correct or remove it; never invent evidence to avoid a negative sentence.

Before opening or updating a PR, review every added or modified paragraph for prohibited rhetoric and its semantic equivalents. Record the direct result and evidence; keep mathematical limitations in compact, exact statements of domain and open problems.

## Mandatory English and research-integrity audit

- Write all repository-authored prose in English: README, AGENTS, Markdown proofs and notes, manuscripts, code comments/docstrings, and PR descriptions. Translate existing Chinese explanatory text, including nested directories and archived research. Preserve mathematics, definitions, assumptions, references, formulas, labels, and the original proof's logical strength. Retain foreign-language source titles and quotations only where necessary with English context.
- Audit every existing mathematical or research claim, not just the changed files: exact quantifiers, cases, dependency chains, proof completeness, counterexamples, citation accuracy, numerical methodology, and the limits of computational verification. Fill missing arguments and necessary definitions; correct false or imprecise claims. Distinguish conjectures, conditional results, finite checks, and fully proved theorems.
- If a gap remains, state its precise mathematical location and adjust the claim accordingly. Never hide uncertainty behind vague exposition or manufacture a proof to make a document look complete.
- Submit focused, reviewable PRs for each repository, explaining substantial corrections and verification. Check internal links, mathematical equivalence of translations, remaining Chinese prose, and existing tests. Preserve essential research content and avoid redundant logs, defensive boilerplate, or invented novelty claims.
- Enforce this policy for every future update; do not add new Chinese prose or mixed-language public documentation.

A personal mathematical and computational protein-design project for the 2026 Changjiang Cup. Work directly on the scientific problem; mathematics and experiments take priority over project ceremony.

## Mathematics and experiments

- Seek methods that improve real binder design: geometric context reduction, structural validity, prediction reliability, candidate selection, and search efficiency. Prefer general insight, useful counterexamples, and testable algorithms.
- Read existing results first, then pursue promising approaches freely. Change assumptions or direction when calculations or evidence reveal a better problem.
- Use mathematical derivations and concise proofs where possible. Small Python experiments, exhaustive searches, and other computational tools are welcome when they answer a concrete question.
- Distinguish proved results, exhaustive finite computations, limited searches, empirical findings, and conjectures. State assumptions and applicability precisely.
- Use verified target sequences, structures, and current official competition rules. Keep original model scores, units, versions, and provenance. Do not interpret synthetic scores as measured binding affinity.
- Add structural processing, model integrations, or heavier computation when they contribute to a specific research goal. GPU-dependent work can be resumed when resources become available; do not initiate paid compute without authorization.

## Repository conventions

- Keep mathematics and short proofs in `math/PROOF.md`, and put small proof-oriented programs in `math/` when needed. Keep real-structure inputs and engineering results in `targets/`, `scripts/`, `results/`, and `gpu/`, retaining their existing working paths. Use `RULES.md` for official rules and `README.md` for a concise project overview and links to both research streams.
- Retain useful source code, small input examples, and necessary tests. Put temporary calculations, generated results, caches, and large artifacts in ignored `build/`.
- Reuse existing programs and libraries. Avoid unnecessary CI/CD, cryptographic seals, certificate bundles, repetitive reports, dashboards, sprawling configuration, and procedural scaffolding.
- Do not create extra documentation or infrastructure merely to record each research round. Remove superseded material while preserving the results needed for further work.
- Run relevant checks before committing. Keep shared Git history intact; no force pushes. Use branches and PRs when collaboration requires them, not as mandatory steps for every experiment.

## Scientific interpretation

- Geometric consistency, predicted interface quality, experimental binding success, and measured `K_D` are distinct properties.
- State exactly what each proof or experiment establishes. Clearly label synthetic examples and limitations of available data.
