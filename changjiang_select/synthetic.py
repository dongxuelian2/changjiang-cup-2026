"""Toy data only: sequences are labels, not proposed biological designs."""

from pathlib import Path

import numpy as np

from .data import AA, save_candidates, save_csv
from .objective import ScenarioObjective, save_scenarios


def make_synthetic(seed=0, n=12, m=64, regime="clustered", evaluation_m=4096):
    if n < 4 or m < 1 or evaluation_m < 1:
        raise ValueError("synthetic generator requires n >= 4 and positive scenario counts")
    if regime not in ("clustered", "independent", "aligned"):
        raise ValueError("unknown synthetic regime")
    rng = np.random.default_rng(seed)
    group = np.arange(n) % 4
    quality = rng.uniform(0.75, 1.0, n)
    probability = np.array([0.72, 0.62, 0.55, 0.45])
    raw = quality * probability[group] if regime != "aligned" else quality.copy()
    # A common model bias exaggerates family 0 in the clustered regime.
    if regime == "clustered":
        raw = raw + 0.18 * (group == 0)

    def draw(count, draw_rng):
        if regime == "aligned":
            return quality[:, None] * draw_rng.uniform(0.4, 1.0, count)[None, :]
        if regime == "independent":
            return quality[:, None] * (draw_rng.random((n, count)) < probability[group, None])
        active = draw_rng.random((4, count)) < probability[:, None]
        individual = draw_rng.random((n, count)) < 0.95
        return quality[:, None] * active[group] * individual

    # Keep held-out scenarios, raw metrics, and sequences fixed when m changes.
    train = draw(m, np.random.default_rng(np.random.SeedSequence([seed, 1])))
    evaluation = draw(evaluation_m, np.random.default_rng(np.random.SeedSequence([seed, 2])))
    alphabet = np.array(sorted(AA))
    templates = [rng.choice(alphabet, 32) for _ in range(4)]
    rows, sequences = [], set()
    for i in range(n):
        while True:
            seq = templates[group[i]].copy()
            positions = rng.choice(32, 5, replace=False)
            seq[positions] = rng.choice(alphabet, 5)
            sequence = "".join(seq)
            if sequence not in sequences:
                sequences.add(sequence)
                break
        rows.append({"id": f"syn-{i:03d}", "sequence": sequence,
            "target_id": "SYNTHETIC-NOT-A-COMPETITION-TARGET", "design_type": "toy-label",
            "source_model": "synthetic-generator", "source_version": "1", "run_id": f"seed-{seed}",
            "metrics": {"toy.raw_score": float(raw[i])}, "structures": [],
            "epitope_residues": [f"synthetic-chain:{10 * group[i] + j}" for j in range(4)],
            "risk_group": f"family-{group[i]}", "structural_group": f"toy-fold-{group[i]}",
            "metadata": {"toy_quality": float(quality[i]), "regime": regime}})
    dataset = {"schema_version": 1, "synthetic": True,
        "description": "SYNTHETIC toy sequences and artificial metrics; not binder designs.",
        "metric_definitions": {"toy.raw_score": {"model": "synthetic-generator", "version": "1",
            "meaning": "Artificial individual ranking score, family-biased in clustered regime",
            "units": "arbitrary", "direction": "higher"}}, "candidates": rows}
    ids = tuple(row["id"] for row in rows)
    description = f"SYNTHETIC {regime}, seed={seed}; artificial joint outcomes, not calibrated probabilities"
    obj = ScenarioObjective(ids, train, np.full(m, 1 / m), description, True)
    heldout = ScenarioObjective(ids, evaluation, np.full(evaluation_m, 1 / evaluation_m),
        description + "; independent held-out scenario draws from the same toy law", True)
    return dataset, obj, heldout


def write_example(directory, seed=7, n=12, m=64, regime="clustered"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset, obj, _ = make_synthetic(seed, n, m, regime)
    save_candidates(dataset, directory / "candidates.json")
    save_csv(dataset, directory / "candidates.csv")
    save_scenarios(obj, directory / "scenarios.json")
