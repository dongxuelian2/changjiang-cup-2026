import unittest

import numpy as np

from changjiang_select.proxies import midranks, rank_objective, score_correlations
from changjiang_select.optimize import exhaustive, portfolio


class ProxyTests(unittest.TestCase):
    def test_duplicate_boltz_context_does_not_add_an_independent_vote(self):
        d = {"synthetic": False, "metric_definitions": {
            key: {"direction": "higher"} for key in
            ("boltz.ipsae_10", "boltz.plddt", "boltz_trimer_deepmsa.ipsae_10", "geometry.contacts")},
            "candidates": [{"id": str(i), "metrics": {
                "boltz.ipsae_10": a, "boltz_trimer_deepmsa.ipsae_10": a,
                "boltz.plddt": b, "geometry.contacts": c}} for i, (a, b, c) in enumerate([(3, 1, 1), (2, 3, 2), (1, 2, 4)])]}
        first, _, _ = rank_objective(d, ["boltz.ipsae_10", "boltz.plddt", "geometry.contacts"])
        repeated, _, _ = rank_objective(d, list(d["metric_definitions"]))
        for subset in ([0], [1], [2], [0, 2], [0, 1, 2]):
            self.assertAlmostEqual(first.value(subset), repeated.value(subset))

    def test_ties_and_reversed_metric_units_do_not_change_objective(self):
        np.testing.assert_equal(midranks([3, 1, 1]), [2, .5, .5])
        d = {"synthetic": False, "metric_definitions": {
            "a": {"direction": "higher"}, "b": {"direction": "lower"}},
            "candidates": [{"id": str(i), "metrics": {"a": a, "b": b}}
                           for i, (a, b) in enumerate([(3, -30), (1, -10), (1, -10)])]}
        obj, keys, skipped = rank_objective(d, ["a", "b"])
        np.testing.assert_equal(obj.utilities[:, 0], obj.utilities[:, 1])
        self.assertEqual(obj.value(portfolio(obj, 1)), obj.value(exhaustive(obj, 1)))
        self.assertFalse(obj.synthetic)

    def test_missing_official_score_is_not_imputed_or_treated_as_zero(self):
        d = {"synthetic": False, "metric_definitions": {"official": {"direction": "higher"}},
             "candidates": [{"id": "x", "metrics": {"official": None}}]}
        with self.assertRaisesRegex(ValueError, "cannot invent"):
            rank_objective(d, ["official"])

    def test_pairwise_complete_correlations_and_constant_scores(self):
        rows = [{"metrics": {"a": a, "b": b, "c": 7}} for a, b in [(1, 4), (2, 3), (3, 1), (None, 9)]]
        values = score_correlations(rows, ["a", "b", "c"])
        self.assertEqual(values[0]["n"], 3)
        self.assertAlmostEqual(values[0]["spearman_rho"], -1)
        self.assertIsNone(values[1]["spearman_rho"])


if __name__ == "__main__":
    unittest.main()
