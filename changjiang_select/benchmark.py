"""Small exhaustive benchmarks and a CPU scale check; outputs go to build/."""

import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import metric_vector
from .objective import ScenarioObjective
from .optimize import approximation_factor, exhaustive, greedy, portfolio, singleton_top_k, top_k
from .synthetic import make_synthetic


def algorithms(dataset, obj, k, exact=True):
    raw, higher = metric_vector(dataset, "toy.raw_score")
    methods = {"raw_topk": lambda: top_k(raw, k, higher=higher),
               "singleton_topk": lambda: singleton_top_k(obj, k),
               "greedy": lambda: greedy(obj, k), "portfolio": lambda: portfolio(obj, k)}
    if exact:
        methods["exact"] = lambda: exhaustive(obj, k)
    result = {}
    for name, method in methods.items():
        start = perf_counter()
        selected = method()
        elapsed = perf_counter() - start
        result[name] = {"selected": selected, "value": obj.value(selected), "seconds": elapsed}
    return result


def mismatch_counterexample():
    # A2 is a valuable hedge in reality, although the fitted model says it is redundant.
    fitted = ScenarioObjective(("A1", "A2", "B"), [[0.8, 0.8], [0.79, 0.79], [0, 1]], [0.5, 0.5],
        "SYNTHETIC counterexample: fitted model", True)
    truth = ScenarioObjective(fitted.ids, [[0], [1], [0]], [1],
        "SYNTHETIC counterexample: different true outcome law", True)
    baseline, selected = singleton_top_k(fitted, 2), portfolio(fitted, 2)
    return {"baseline_ids": [fitted.ids[i] for i in baseline],
            "portfolio_ids": [fitted.ids[i] for i in selected],
            "fitted_baseline": fitted.value(baseline), "fitted_portfolio": fitted.value(selected),
            "true_baseline": truth.value(baseline), "true_portfolio": truth.value(selected)}


def run_benchmark(directory="build/benchmark", trials=30):
    if trials < 1:
        raise ValueError("trials must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records, summary = [], {}
    for regime in ("aligned", "independent", "clustered"):
        for seed in range(trials):
            dataset, obj, heldout = make_synthetic(seed=seed, regime=regime)
            results = algorithms(dataset, obj, 3)
            optimum = results["exact"]["value"]
            if results["greedy"]["value"] + 1e-12 < approximation_factor(3) * optimum:
                raise AssertionError("greedy bound violated")
            if results["portfolio"]["value"] + 1e-12 < results["singleton_topk"]["value"]:
                raise AssertionError("portfolio safeguard violated")
            base_set = set(results["singleton_topk"]["selected"])
            for method, result in results.items():
                selected = result["selected"]
                subset = set(selected)
                records.append({"regime": regime, "seed": seed, "method": method,
                    "objective": result["value"], "ratio_to_exact": result["value"] / optimum,
                    "heldout_objective": heldout.value(selected), "seconds": result["seconds"],
                    "jaccard_to_singleton": len(subset & base_set) / len(subset | base_set),
                    "selected_ids": ";".join(obj.ids[i] for i in selected)})
        regime_rows = [r for r in records if r["regime"] == regime]
        by_method = {}
        for method in results:
            group = [r for r in regime_rows if r["method"] == method]
            by_method[method] = {key: float(np.mean([r[key] for r in group])) for key in
                ("objective", "ratio_to_exact", "heldout_objective", "jaccard_to_singleton")}
            by_method[method]["minimum_ratio"] = min(r["ratio_to_exact"] for r in group)
            by_method[method]["median_ms"] = float(np.median([r["seconds"] for r in group]) * 1000)
        baseline = [r for r in regime_rows if r["method"] == "singleton_topk"]
        portfolios = [r for r in regime_rows if r["method"] == "portfolio"]
        delta = np.array([p["heldout_objective"] - b["heldout_objective"] for p, b in zip(portfolios, baseline)])
        by_method["heldout_portfolio_vs_singleton"] = {
            "wins": int((delta > 1e-12).sum()), "ties": int((abs(delta) <= 1e-12).sum()),
            "losses": int((delta < -1e-12).sum()), "mean_difference": float(delta.mean())}
        summary[regime] = by_method
    with (directory / "trials.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    dataset, obj, heldout = make_synthetic(seed=2026, n=120, m=256, evaluation_m=8192)
    scale = algorithms(dataset, obj, 20, exact=False)
    for result in scale.values():
        result["heldout_objective"] = heldout.value(result["selected"])
        result["selected_ids"] = [obj.ids[i] for i in result.pop("selected")]
    sensitivity = {}
    for m in (16, 64, 256, 1024):
        differences = []
        for seed in range(trials):
            dataset, obj, heldout = make_synthetic(seed=seed, m=m, regime="independent")
            differences.append(heldout.value(portfolio(obj, 3)) - heldout.value(singleton_top_k(obj, 3)))
        delta = np.asarray(differences)
        sensitivity[str(m)] = {"heldout_mean_portfolio_minus_singleton": float(delta.mean()),
            "wins": int((delta > 1e-12).sum()), "ties": int((abs(delta) <= 1e-12).sum()),
            "losses": int((delta < -1e-12).sum())}
    report = {"synthetic": True, "trials_per_regime": trials, "n": 12, "m": 64, "k": 3,
              "heldout_m": 4096, "seeds": [0, trials - 1], "greedy_factor_k3": approximation_factor(3),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform(), "processor": platform.processor()},
              "regimes": summary, "scale_n120_m256_k20": scale,
              "independent_scenario_count_sensitivity": sensitivity,
              "misspecification_counterexample": mismatch_counterexample()}
    (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
