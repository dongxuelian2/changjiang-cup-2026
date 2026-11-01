"""Small exact mathematical checks only; no target files or model execution."""
from fractions import Fraction as Q
import importlib.util
from itertools import combinations, product
from math import comb
from pathlib import Path
import random
import unittest

SPEC = importlib.util.spec_from_file_location(
    "context_motifs", Path(__file__).resolve().parents[1] / "math/context_motifs.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MotifProblem = MODULE.MotifProblem


def subsets(nodes):
    nodes = tuple(nodes)
    for k in range(len(nodes) + 1):
        for s in combinations(nodes, k):
            yield frozenset(s)


class ContextMotifTests(unittest.TestCase):
    def test_every_four_node_unweighted_graph_priced_and_budget_bounds(self):
        edges = tuple(combinations(range(4), 2))
        sets = tuple(subsets(range(4)))
        for bits in product((0, 1), repeat=len(edges)):
            p = MotifProblem([1] * 4, zip(edges, bits))
            for price in (Q(i, 2) for i in range(7)):
                selected, value = p.priced(price)
                oracle = max(p.value(s) - price * len(s) for s in sets)
                self.assertEqual(value, oracle)
                self.assertEqual(value, p.value(selected) - price * len(selected))
                optima = [s for s in sets if p.value(s) - price * len(s) == oracle]
                largest, largest_value = p.priced(price, maximal=True)
                self.assertEqual(selected, frozenset.intersection(*optima))
                self.assertEqual(largest, frozenset.union(*optima))
                self.assertEqual(largest_value, value)
                for budget in range(5):
                    opt = max(p.value(s) for s in sets if len(s) <= budget)
                    self.assertGreaterEqual(p.upper_bound(budget, price), opt)
                    if len(selected) <= budget <= len(largest):
                        rounded = p.round_cardinality(selected, largest, budget)
                        expected = p.restriction_expectation(
                            selected, largest - selected, budget - len(selected))
                        self.assertGreaterEqual(p.value(rounded), expected)
                        self.assertLessEqual(p.value(rounded), opt)

    def test_seeded_rational_hypergraphs_core_zero_costs_empty_duplicate_motifs(self):
        rng = random.Random(20261101)
        for _ in range(60):
            n = rng.randrange(1, 7)
            core = frozenset(i for i in range(n) if rng.randrange(4) == 0)
            costs = [Q(rng.randrange(4), rng.randrange(1, 4)) for _ in range(n)]
            motifs = [(rng.sample(range(n), rng.randrange(n + 1)),
                       Q(rng.randrange(6), rng.randrange(1, 4))) for _ in range(8)]
            motifs += [((), Q(2)), ((0,), Q(1)), ((0,), Q(3))]
            p = MotifProblem(costs, motifs, core)
            sets = [s for s in subsets(range(n)) if core <= s]
            for price in (Q(0), Q(1, 3), Q(3, 2), Q(5)):
                selected, actual = p.priced(price)
                self.assertLessEqual(core, selected)
                self.assertEqual(actual, max(p.value(s) - price * p.cost(s) for s in sets))
                largest, upper = p.priced(price, maximal=True)
                optima = [s for s in sets if p.value(s) - price * p.cost(s) == actual]
                self.assertEqual(selected, frozenset.intersection(*optima))
                self.assertEqual(largest, frozenset.union(*optima))
                self.assertEqual(upper, actual)

    def test_clique_unsupported_budget_and_unbounded_dual_gap(self):
        for n in range(3, 9):
            p = MotifProblem([1] * n, [(h, 1) for h in combinations(range(n), 2)])
            price = Q(n - 1, 2)
            self.assertEqual(p.upper_bound(2, price), n - 1)
            scores = [p.value(s) - price * len(s) for s in subsets(range(n))]
            self.assertEqual(max(scores), 0)
            for s in subsets(range(n)):
                if 0 < len(s) < n:
                    self.assertLess(p.value(s) - price * len(s), 0)
            rounded = p.round_cardinality((), range(n), 2)
            self.assertEqual(p.value(rounded), 1)

    def test_conditioning_expectation_and_derandomization_against_enumeration(self):
        rng = random.Random(41)
        for _ in range(60):
            n = 7
            core = {0}
            p = MotifProblem([1] * n,
                [(rng.sample(range(n), rng.randrange(5)), Q(rng.randrange(7), 3))
                 for _ in range(10)], core)
            fixed, superset = {0, 1}, set(range(6))  # outside motifs must vanish
            for k in range(5):
                values = [p.value(fixed | set(s)) for s in combinations(superset - fixed, k)]
                expected = p.restriction_expectation(fixed, superset - fixed, k)
                self.assertEqual(expected, sum(values) / len(values))
                rounded = p.round_cardinality(fixed, superset, len(fixed) + k)
                self.assertEqual(len(rounded), len(fixed) + k)
                self.assertTrue(fixed <= rounded <= superset)
                self.assertGreaterEqual(p.value(rounded), expected)

    def test_rank_sensitive_dual_ratio_is_sharp_for_complete_hypergraphs(self):
        for m in range(2, 8):
            for d in range(1, m + 1):
                p = MotifProblem([1] * m, [(h, 1) for h in combinations(range(m), d)])
                price = Q(comb(m, d), m)
                self.assertEqual(p.priced(price)[1], 0)
                for k in range(d, m + 1):
                    lb = p.restriction_expectation((), range(m), k)
                    ub = p.upper_bound(k, price)
                    self.assertEqual(lb / ub, Q(comb(k - 1, d - 1), comb(m - 1, d - 1)))
                    self.assertEqual(p.value(p.round_cardinality((), range(m), k)), lb)

    def test_residual_rank_one_recovers_exact_budget_with_nonempty_core(self):
        p = MotifProblem([1] * 4, [({0}, 7)] + [({0, i}, 2) for i in range(1, 4)], {0})
        for budget in range(1, 5):
            selected = p.round_cardinality({0}, range(4), budget)
            self.assertEqual(p.value(selected), p.upper_bound(budget, 2))

    def test_invalid_inputs_and_empty_problem(self):
        self.assertEqual(MotifProblem([], [((), 2)]).upper_bound(0, 1), 2)
        self.assertEqual(MotifProblem([], []).round_cardinality((), (), 0), frozenset())
        for costs, motifs in [([-1], []), ([1.0], []), ([1], [({1}, 2)]),
                              ([1], [({0}, -1)]), ([1], [([0, False], 2)]),
                              ([1], [([0, 0.0], 2)])]:
            with self.assertRaises(ValueError):
                MotifProblem(costs, motifs)
        p = MotifProblem([1, 2], [({0, 1}, 3)], {0})
        for operation in (lambda: p.upper_bound(0, 1), lambda: p.priced(-1),
                          lambda: p.round_cardinality({0}, {0, 1}, 1),
                          lambda: p.restriction_expectation({0}, {0}, 0)):
            with self.assertRaises(ValueError):
                operation()


if __name__ == "__main__":
    unittest.main()
