"""Same scenario objective, with cardinality or optional backbone-group budgets."""

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


def check_groups(obj, k, groups):
    check_budget(len(obj.ids), k)
    if groups is not None and (len(groups) != len(obj.ids) or len(set(groups)) < k):
        raise ValueError("one-per-group budget needs one label per candidate and at least k groups")


def singleton_top_k(obj, k, *, groups=None):
    """Top-K expected individual utilities: a stronger baseline than raw score."""
    check_groups(obj, k, groups)
    order = top_k(obj.utilities @ obj.weights, len(obj.ids))
    if groups is None:
        return order[:k]
    selected, used = [], set()
    for i in order:
        if len(selected) == k:
            break
        if groups[i] not in used:
            selected.append(i)
            used.add(groups[i])
    return selected


def greedy(obj, k, *, groups=None):
    """Cardinality greedy, or 1/2-approximate greedy for a truncated partition matroid."""
    check_groups(obj, k, groups)
    chosen = []
    singletons = obj.utilities @ obj.weights
    for _ in range(k):
        gains = obj.marginal_gains(chosen)
        gains[chosen] = -np.inf
        if groups is not None:
            used = {groups[i] for i in chosen}
            gains[[i for i,g in enumerate(groups) if g in used]] = -np.inf
        # Exact equal-gain ties use individual utility, then input order. This is
        # especially useful after a finite scenario sample has been saturated.
        chosen.append(int(np.lexsort((np.arange(len(obj.ids)), -singletons, -gains))[0]))
    return chosen


def portfolio(obj, k, *, groups=None):
    """Better of feasible greedy and singleton Top-K; retains the relevant greedy bound."""
    g, t = greedy(obj, k, groups=groups), singleton_top_k(obj, k, groups=groups)
    return t if obj.value(t) > obj.value(g) else g


def exhaustive(obj, k, *, max_combinations=1_000_000, groups=None):
    check_groups(obj, k, groups)
    count = comb(len(obj.ids), k)
    if count > max_combinations:
        raise ValueError(f"exact search needs {count} sets; limit is {max_combinations}")
    # Monotonicity means that an optimum of size k is also optimal for size <= k.
    best, value = [], -1.0
    for indices in combinations(range(len(obj.ids)), k):
        if groups is not None and len({groups[i] for i in indices}) != k:
            continue
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
