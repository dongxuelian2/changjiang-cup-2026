"""Descriptive score disagreement and uncalibrated rank-coverage scenarios.

This uses the existing fixed-scenario objective/solvers; it is not a KD model.
"""
import numpy as np

from .objective import ScenarioObjective


def source_family(key):
    # Alternative contexts/seeds of one predictor are not independent models.
    prefix = key.split(".")[0]
    return "boltz" if prefix == "boltz" or prefix.startswith("boltz_") else prefix


def midranks(values):
    """Average zero-based ranks; exact ties receive equal ranks."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ranks require a complete finite vector")
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start+1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranked[order[start:stop]] = (start+stop-1)/2
        start = stop
    return ranked


def score_correlations(rows, keys):
    """Pairwise-complete Spearman rho; report N and constant-score nulls.

    Correlation is descriptive. Related sequences/backbones are not IID samples.
    """
    result = []
    for i, first in enumerate(keys):
        for second in keys[i+1:]:
            pairs = [(r["metrics"].get(first), r["metrics"].get(second)) for r in rows]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            rho = None
            if len(pairs) >= 3:
                a, b = map(midranks, zip(*pairs))
                if np.ptp(a) and np.ptp(b):
                    rho = float(np.corrcoef(a, b)[0, 1])
            result.append({"first": first, "second": second, "n": len(pairs), "spearman_rho": rho})
    return result


def rank_objective(dataset, keys):
    """Each complete, nonconstant metric is one rank viewpoint.

    Percentile rank preserves its stated direction, not its magnitude. It is
    cohort-relative and is never a probability or comparable across batches.
    Constant/missing columns are omitted explicitly, without imputation.
    """
    rows, columns, used, skipped = dataset["candidates"], [], [], {}
    for key in keys:
        spec = dataset["metric_definitions"][key]
        values = [r["metrics"].get(key) for r in rows]
        if any(v is None for v in values):
            skipped[key] = "missing scores"
            continue
        if spec["direction"] not in ("higher", "lower"):
            raise ValueError("rank objective needs an explicit metric direction")
        values = np.asarray(values, dtype=float)
        if len(values) < 2 or np.ptp(values) == 0:
            skipped[key] = "constant scores"
            continue
        oriented = values if spec["direction"] == "higher" else -values
        column = midranks(oriented)/(len(values)-1)
        aliases = [prev for prev, existing in zip(used, columns)
                   if source_family(prev) == source_family(key) and np.array_equal(existing, column)]
        if aliases:
            skipped[key] = "rank-equivalent within source family to "+aliases[0]
            continue
        columns.append(column)
        used.append(key)
    if not columns:
        raise ValueError("no complete, nonconstant metrics; cannot invent a decision objective")
    # Give each source family (prefix) equal total weight, so reporting several
    # related Boltz confidence heads does not give that model extra votes.
    groups = [source_family(key) for key in used]
    weights = np.array([1/(len(set(groups))*groups.count(group)) for group in groups])
    description = ("UNCALIBRATED rank-coverage triage, not KD utility or success probability. "
                   "Equal total weight per source family, divided across its measured proxy viewpoints: " + ", ".join(used) +
                   ". Correlation persists across families; inspect leave-one-family-out sensitivity.")
    return ScenarioObjective(tuple(r["id"] for r in rows), np.column_stack(columns),
                             weights, description,
                             dataset["synthetic"]), used, skipped
