"""Deterministic cardinality algorithms for the same scenario objective."""

from itertools import combinations
from math import comb

import numpy as np


def check_budget(n, k):
    if type(k) is not int or k < 0 or k > n:
        raise ValueError("budget must be an integer between zero and candidate count")


def top_k(scores, k, *, higher=True):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Top-K scores must be a finite vector; missing values require a decision")
    check_budget(len(scores), k)
    return sorted(range(len(scores)), key=lambda i: (-scores[i] if higher else scores[i], i))[:k]


def singleton_top_k(obj, k):
    """Top-K expected individual utilities: a stronger baseline than raw score."""
    return top_k(obj.utilities @ obj.weights, k)


def greedy(obj, k):
    check_budget(len(obj.ids), k)
    chosen = []
    singletons = obj.utilities @ obj.weights
    for _ in range(k):
        gains = obj.marginal_gains(chosen)
        gains[chosen] = -np.inf
        # Exact equal-gain ties use individual utility, then input order. This is
        # especially useful after a finite scenario sample has been saturated.
        chosen.append(int(np.lexsort((np.arange(len(obj.ids)), -singletons, -gains))[0]))
    return chosen


def portfolio(obj, k):
    """Better of greedy and singleton Top-K; preserves both model guarantees."""
    g, t = greedy(obj, k), singleton_top_k(obj, k)
    return t if obj.value(t) > obj.value(g) else g


def exhaustive(obj, k, *, max_combinations=1_000_000):
    check_budget(len(obj.ids), k)
    count = comb(len(obj.ids), k)
    if count > max_combinations:
        raise ValueError(f"exact search needs {count} sets; limit is {max_combinations}")
    # Monotonicity means that an optimum of size k is also optimal for size <= k.
    best, value = [], -1.0
    for indices in combinations(range(len(obj.ids)), k):
        score = obj.value(indices)
        if score > value:
            best, value = list(indices), score
    return best


def approximation_factor(k):
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")
    return 1.0 if k == 0 else 1 - (1 - 1 / k) ** k


def upper_bound(obj, selected, k):
    """Valid upper bound on OPT, not a confidence interval for experiments."""
    selected = list(selected)
    check_budget(len(obj.ids), k)
    gains = obj.marginal_gains(selected)
    return min(float(obj.utilities.max(axis=0) @ obj.weights) if len(obj.ids) else 0.0,
               obj.value(selected) + float(np.sort(gains)[-k:].sum()) if k else 0.0)
