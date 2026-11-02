"""Finite checks for exploration budgets across real-design backbone families."""
from itertools import combinations
import unittest
import numpy as np
from changjiang_select.objective import ScenarioObjective
from changjiang_select.optimize import exhaustive,greedy,portfolio,singleton_top_k


class PartitionSelectionTests(unittest.TestCase):
    def test_half_factor_is_tight_and_exact_can_choose_another_variant(self):
        obj=ScenarioObjective(("a1","a2","b1"),np.array([[1,0],[0,1],[1,0.]]),np.array([.5,.5]),"toy",True)
        groups=["a","a","b"]
        self.assertEqual(greedy(obj,2,groups=groups),[0,2])
        self.assertEqual(exhaustive(obj,2,groups=groups),[1,2])
        self.assertEqual(obj.value(greedy(obj,2,groups=groups)),.5)
        self.assertEqual(obj.value(exhaustive(obj,2,groups=groups)),1)

    def test_finite_oracle_feasibility_and_half_bound(self):
        rng=np.random.default_rng(20261102)
        groups=["a","a","b","b","c","c"]
        for _ in range(128):
            u=rng.integers(0,4,size=(6,3)).astype(float)
            obj=ScenarioObjective(tuple(map(str,range(6))),u,np.array([.2,.3,.5]),"toy",True)
            for k in (0,1,2,3):
                feasible=[s for s in combinations(range(6),k) if len({groups[i] for i in s})==k]
                optimum=max(obj.value(s) for s in feasible)
                exact=exhaustive(obj,k,groups=groups)
                g=greedy(obj,k,groups=groups)
                p=portfolio(obj,k,groups=groups)
                self.assertAlmostEqual(obj.value(exact),optimum)
                self.assertGreaterEqual(obj.value(g)+1e-12,.5*optimum)
                for s in (g,p,exact,singleton_top_k(obj,k,groups=groups)):
                    self.assertEqual(len(s),k)
                    self.assertEqual(len({groups[i] for i in s}),k)
                self.assertGreaterEqual(obj.value(p),obj.value(g))
