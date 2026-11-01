import copy
from itertools import combinations, product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from changjiang_select.benchmark import mismatch_counterexample
from changjiang_select.data import (deduplicate, duplicate_groups, load_candidates, metric_vector,
                                    save_candidates, save_csv, validate_dataset)
from changjiang_select.objective import ScenarioObjective, load_scenarios, save_scenarios
from changjiang_select.optimize import (approximation_factor, exhaustive, greedy, portfolio,
                                        singleton_top_k, top_k, upper_bound)
from changjiang_select.similarity import epitope_similarity, sequence_similarity
from changjiang_select.synthetic import make_synthetic

TEST_BUILD = Path(__file__).resolve().parents[1] / "build" / "tests"
TEST_BUILD.mkdir(parents=True, exist_ok=True)


class MathematicsTests(unittest.TestCase):
    def test_every_binary_four_candidate_three_scenario_instance(self):
        # Complete finite family: 2^12 utility matrices, k=2. Independent integer oracle.
        for bits in product((0, 1), repeat=12):
            u = np.array(bits).reshape(4, 3)
            obj = ScenarioObjective(tuple("abcd"), u, [1 / 3] * 3)
            oracle = max(sum(max(u[i, s], u[j, s]) for s in range(3)) for i, j in combinations(range(4), 2)) / 3
            exact, g, p = exhaustive(obj, 2), greedy(obj, 2), portfolio(obj, 2)
            self.assertAlmostEqual(obj.value(exact), oracle)
            self.assertGreaterEqual(obj.value(g) + 1e-12, 0.75 * oracle)
            self.assertGreaterEqual(obj.value(p) + 1e-12, obj.value(singleton_top_k(obj, 2)))
            self.assertGreaterEqual(upper_bound(obj, g, 2) + 1e-12, oracle)

    def test_diminishing_returns_over_all_nested_sets(self):
        rng = np.random.default_rng(23)
        obj = ScenarioObjective(tuple("abcde"), rng.random((5, 9)), np.full(9, 1 / 9))
        for labels in product((0, 1, 2), repeat=5):
            a = [i for i, label in enumerate(labels) if label == 1]
            b = [i for i, label in enumerate(labels) if label != 0]
            self.assertLessEqual(obj.value(a), obj.value(b) + 1e-12)
            for x, label in enumerate(labels):
                if label == 0:
                    self.assertGreaterEqual(obj.value(a + [x]) - obj.value(a) + 1e-12,
                                            obj.value(b + [x]) - obj.value(b))

    def test_greedy_can_be_suboptimal(self):
        obj = ScenarioObjective(tuple("abc"), [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]], [0.25] * 4)
        self.assertEqual(obj.value(greedy(obj, 2)), 0.75)
        self.assertEqual(obj.value(exhaustive(obj, 2)), 1)

    def test_aligned_order_topk_is_sufficient(self):
        for seed in range(10):
            dataset, obj, _ = make_synthetic(seed=seed, regime="aligned")
            raw, _ = metric_vector(dataset, "toy.raw_score")
            self.assertAlmostEqual(obj.value(top_k(raw, 3)), obj.value(exhaustive(obj, 3)))

    def test_portfolio_fallback_is_needed_for_some_instances(self):
        obj = ScenarioObjective(tuple("abcdef"),
            [[0, 2, 0, 1, 2, 2], [2, 2, 0, 0, 1, 2], [0, 3, 3, 3, 0, 0],
             [3, 2, 2, 1, 0, 1], [0, 0, 3, 0, 2, 1], [0, 0, 3, 2, 0, 1]], [1 / 6] * 6)
        self.assertAlmostEqual(obj.value(greedy(obj, 3)), 2.5)
        self.assertAlmostEqual(obj.value(portfolio(obj, 3)), 8 / 3)

    def test_sensitivity_uses_fixed_heldout_data(self):
        small_data, _, small_test = make_synthetic(seed=1, m=16)
        large_data, _, large_test = make_synthetic(seed=1, m=256)
        self.assertEqual(small_data, large_data)
        np.testing.assert_array_equal(small_test.utilities, large_test.utilities)

    def test_zero_full_budget_and_limits(self):
        obj = ScenarioObjective(("a", "b"), [[0, 0], [0, 0]], [0.5, 0.5])
        for method in (greedy, singleton_top_k, portfolio, exhaustive):
            self.assertEqual(method(obj, 0), [])
            self.assertEqual(set(method(obj, 2)), {0, 1})
            with self.assertRaises(ValueError):
                method(obj, 3)
        with self.assertRaises(ValueError):
            exhaustive(obj, 1, max_combinations=1)
        self.assertEqual(approximation_factor(0), 1)

    def test_bad_scenarios_and_id_alignment(self):
        for u, w in [([[-1]], [1]), ([[float("nan")]], [1]), ([[1]], [0.8]), ([[1]], [-1])]:
            with self.assertRaises(ValueError):
                ScenarioObjective(("a",), u, w)
        obj = ScenarioObjective(("a", "b"), [[1], [2]], [1])
        aligned = obj.align(["b", "a"])
        self.assertEqual(aligned.value([0]), 2)
        with self.assertRaises(ValueError):
            obj.align(["c"])
        with self.assertRaises(ValueError):
            obj.value([-1])

    def test_no_biological_guarantee_under_misspecification(self):
        result = mismatch_counterexample()
        self.assertGreater(result["fitted_portfolio"], result["fitted_baseline"])
        self.assertLess(result["true_portfolio"], result["true_baseline"])


class DataAndWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.dataset, self.obj, self.heldout = make_synthetic(seed=7)

    def test_json_csv_roundtrip_preserves_scores_semantics_and_metadata(self):
        with tempfile.TemporaryDirectory(dir=TEST_BUILD) as directory:
            for suffix, writer in (("json", save_candidates), ("csv", save_csv)):
                path = Path(directory) / ("candidates." + suffix)
                writer(self.dataset, path)
                result = load_candidates(path)
                self.assertEqual(result, self.dataset)
            scenario_path = Path(directory) / "scenarios.json"
            save_scenarios(self.obj, scenario_path)
            loaded = load_scenarios(scenario_path)
            np.testing.assert_array_equal(loaded.utilities, self.obj.utilities)

    def test_normalization_duplicates_and_missing_values(self):
        d = copy.deepcopy(self.dataset)
        d["candidates"][1]["sequence"] = " \n" + d["candidates"][0]["sequence"].lower()
        rows = validate_dataset(d)["candidates"]
        self.assertEqual(duplicate_groups(rows), [["syn-000", "syn-001"]])
        self.assertEqual(len(deduplicate(rows)), 11)
        d["candidates"][1]["metrics"]["toy.raw_score"] = None
        d = validate_dataset(d)
        with self.assertRaises(ValueError):
            metric_vector(d, "toy.raw_score")

    def test_invalid_amino_acid_metric_or_id(self):
        for key, value in (("sequence", "ABXZ"), ("id", "syn-001"), ("metrics", {"undefined": 1}),
                           ("metrics", {"toy.raw_score": float("inf")})):
            d = copy.deepcopy(self.dataset)
            d["candidates"][0][key] = value
            with self.assertRaises(ValueError):
                validate_dataset(d)

    def test_similarity_and_missing_epitope(self):
        self.assertEqual(sequence_similarity("ACD", "ACD"), 1)
        self.assertAlmostEqual(sequence_similarity("ACD", "AD"), 2 / 3)
        self.assertEqual(sequence_similarity("AAA", "CCC"), 0)
        self.assertEqual(sequence_similarity("ACD", "AD"), sequence_similarity("AD", "ACD"))
        self.assertIsNone(epitope_similarity([], ["A:1"]))
        self.assertAlmostEqual(epitope_similarity(["A:1", "A:2"], ["A:2", "A:3"]), 1 / 3)

    def test_cli_selection_end_to_end_from_csv(self):
        with tempfile.TemporaryDirectory(dir=TEST_BUILD) as directory:
            directory = Path(directory)
            save_csv(self.dataset, directory / "candidates.csv")
            save_scenarios(self.obj, directory / "scenarios.json")
            command = [sys.executable, "-m", "changjiang_select", "select", str(directory / "candidates.csv"),
                       "--scenarios", str(directory / "scenarios.json"), "--k", "3", "--out", str(directory / "out")]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            report = json.loads(completed.stdout)
            self.assertEqual(len(report["selected_ids"]), 3)
            selected = load_candidates(directory / "out" / "selection.json")
            self.assertEqual(len({r["sequence"] for r in selected["candidates"]}), 3)
            self.assertEqual((directory / "out" / "selected.fasta").read_text().count(">"), 3)
            self.assertGreaterEqual(report["optimum_upper_bound"] + 1e-12, report["objective_value"])
            self.assertGreaterEqual(report["objective_ratio_lower_bound"], 0)
            # A valid multi-target dataset must not be pooled by the selector.
            self.dataset["candidates"][0]["target_id"] = "another-target"
            save_csv(self.dataset, directory / "candidates.csv")
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("exactly one target", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
