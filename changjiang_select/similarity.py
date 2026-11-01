"""Descriptive diversity; no claim that sequence distance measures failure risk."""

from collections import Counter
from itertools import combinations

import numpy as np


def sequence_similarity(a, b):
    """1 - unit-cost Levenshtein distance / max length; not alignment identity."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, aa in enumerate(a, 1):
        current = [i]
        for j, bb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (aa != bb)))
        previous = current
    return 1 - previous[-1] / len(a) if a else 1.0


def epitope_similarity(a, b):
    if not a or not b:
        return None  # unobserved is not a disjoint binding site
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def pairwise(rows):
    n = len(rows)
    sequence = np.eye(n)
    epitope = [[None] * n for _ in rows]
    for i in range(n):
        if rows[i]["epitope_residues"]:
            epitope[i][i] = 1.0
    for i, j in combinations(range(n), 2):
        if rows[i]["target_id"] != rows[j]["target_id"]:
            raise ValueError("pairwise comparison requires one target")
        sequence[i, j] = sequence[j, i] = sequence_similarity(rows[i]["sequence"], rows[j]["sequence"])
        epitope[i][j] = epitope[j][i] = epitope_similarity(rows[i]["epitope_residues"], rows[j]["epitope_residues"])
    return {"candidate_ids": [r["id"] for r in rows],
            "sequence_edit_similarity": sequence.tolist(), "epitope_jaccard": epitope}


def diversity(rows):
    pairs = list(combinations(rows, 2))
    distances = [1 - sequence_similarity(a["sequence"], b["sequence"]) for a, b in pairs]
    epitope = [epitope_similarity(a["epitope_residues"], b["epitope_residues"]) for a, b in pairs]
    observed = [v for v in epitope if v is not None]
    return {"mean_sequence_edit_distance": float(np.mean(distances)) if distances else None,
            "min_sequence_edit_distance": min(distances) if distances else None,
            "mean_epitope_jaccard": float(np.mean(observed)) if observed else None,
            "observed_epitope_pairs": len(observed),
            "risk_group_counts": dict(Counter(r.get("risk_group", "unknown") for r in rows)),
            "structural_group_counts": dict(Counter(r.get("structural_group", "unknown") for r in rows))}
