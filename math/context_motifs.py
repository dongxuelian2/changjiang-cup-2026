"""Exact-rational toy tools for PROOF.md §10; no structural or model inputs.

Nodes are integer labels 0..len(costs)-1. A motif is (iterable_of_nodes, weight).
Only int/Fraction nonnegative costs, weights and prices are accepted. The priced
solver is classical maximum closure, NOT an exact hard-budget solver.
"""

from collections import deque
from fractions import Fraction
from math import comb


def _rational(value):
    if type(value) not in (int, Fraction) or value < 0:
        raise ValueError("expected a nonnegative int or Fraction")
    return Fraction(value)


class MotifProblem:
    def __init__(self, costs, motifs, core=()):
        self.costs = tuple(map(_rational, costs))
        self.core = self._nodes(core)
        self.motifs = tuple((self._nodes(h), _rational(w)) for h, w in motifs)

    def _nodes(self, nodes):
        nodes = tuple(nodes)
        if any(type(i) is not int or not 0 <= i < len(self.costs) for i in nodes):
            raise ValueError("invalid node label")
        return frozenset(nodes)

    def value(self, selected):
        selected = self._nodes(selected)
        return sum((w for h, w in self.motifs if h <= selected), Fraction())

    def cost(self, selected):
        return sum((self.costs[i] for i in self._nodes(selected)), Fraction())

    def priced(self, price, *, maximal=False):
        """Return (selected superset of core, max_S F(S)-price*c(S)).

        Edmonds--Karp uses exact capacities; intended for small symbolic inputs.
        Source-reachable nodes give the inclusion-minimal optimizer. With
        maximal=True, complement of reverse sink-reachability gives the
        inclusion-maximal optimizer. Neither must fill a hard budget.
        """
        price = _rational(price)
        if type(maximal) is not bool:
            raise ValueError("maximal must be a bool")
        n, q = len(self.costs), len(self.motifs)
        source, sink = n + q, n + q + 1
        graph = [{} for _ in range(n + q + 2)]

        def edge(u, v, capacity):
            graph[u][v] = graph[u].get(v, Fraction()) + capacity
            graph[v].setdefault(u, Fraction())

        # Empty optional set has cut <= W, so W+1 strictly prevents cutting
        # a prerequisite arc. Core is contracted out, including its fixed cost.
        total = sum((w for _, w in self.motifs), Fraction())
        for j, (h, weight) in enumerate(self.motifs):
            edge(source, n + j, weight)
            for i in h - self.core:
                edge(n + j, i, total + 1)
        for i, cost in enumerate(self.costs):
            if i not in self.core:
                edge(i, sink, price * cost)

        while True:
            parent = {source: None}
            queue = deque([source])
            while queue and sink not in parent:
                u = queue.popleft()
                for v, capacity in graph[u].items():
                    if capacity > 0 and v not in parent:
                        parent[v] = u
                        queue.append(v)
            if sink not in parent:
                reachable = set(parent)
                if maximal:
                    to_sink, queue = {sink}, deque([sink])
                    while queue:
                        v = queue.popleft()
                        for u in graph[v]:
                            if graph[u][v] > 0 and u not in to_sink:
                                to_sink.add(u)
                                queue.append(u)
                    reachable = set(range(len(graph))) - to_sink
                selected = self.core | frozenset(i for i in range(n) if i in reachable)
                return selected, self.value(selected) - price * self.cost(selected)
            path, v = [], sink
            while v != source:
                u = parent[v]
                path.append((u, v))
                v = u
            flow = min(graph[u][v] for u, v in path)
            for u, v in path:
                graph[u][v] -= flow
                graph[v][u] += flow

    def upper_bound(self, budget, price):
        """Return F(S_price)-price*c(S_price)+price*budget >= hard OPT.

        This is one price's bound, not a claimed optimized dual. An infeasible
        mandatory core raises ValueError; zero costs and empty motifs are valid.
        """
        budget, price = _rational(budget), _rational(price)
        if self.cost(self.core) > budget:
            raise ValueError("mandatory core exceeds budget")
        return self.priced(price)[1] + price * budget

    def restriction_expectation(self, fixed, available, count):
        """Exact expected reward after a uniform count-subset of available.

        This helper is combinatorial; no cost guarantee for nonunit costs.
        Motifs not contained in fixed|available contribute zero.
        """
        fixed, available = self._nodes(fixed), self._nodes(available)
        if fixed & available or type(count) is not int or not 0 <= count <= len(available):
            raise ValueError("need disjoint sets and a valid subset count")
        total = Fraction()
        allowed = fixed | available
        for h, weight in self.motifs:
            if h <= allowed:
                r = len(h - fixed)
                if r <= count:
                    total += weight * Fraction(comb(count, r), comb(len(available), r))
        return total

    def round_cardinality(self, fixed, superset, budget):
        """Return a deterministic budget-size subset of superset containing fixed.

        Requires unit costs, core <= fixed <= superset and |fixed| <= budget <=
        |superset|. Reward is at least restriction_expectation(fixed,
        superset-fixed, budget-len(fixed)), by conditional expectations.
        This is not a global approximation guarantee or a connectivity solver.
        """
        fixed, superset = self._nodes(fixed), self._nodes(superset)
        if (any(c != 1 for c in self.costs) or not self.core <= fixed <= superset
                or type(budget) is not int or not len(fixed) <= budget <= len(superset)):
            raise ValueError("unit costs and core <= fixed <= budget <= superset required")
        selected, remaining = set(fixed), set(superset - fixed)
        while remaining:
            i = min(remaining)
            remaining.remove(i)
            count = budget - len(selected)
            if count == 0:
                break
            include = self.restriction_expectation(selected | {i}, remaining, count - 1)
            if count > len(remaining) or include >= self.restriction_expectation(selected, remaining, count):
                selected.add(i)
        return frozenset(selected)
