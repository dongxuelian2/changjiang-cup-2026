"""Small file-based CLI; all first-stage work runs on CPU."""

import argparse
import csv
import json
from pathlib import Path

from .benchmark import run_benchmark
from .data import deduplicate, duplicate_groups, load_candidates, metric_vector
from .objective import load_scenarios
from .optimize import approximation_factor, exhaustive, greedy, portfolio, singleton_top_k, top_k, upper_bound
from .similarity import diversity, pairwise
from .synthetic import write_example


def dump_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def select(args):
    dataset = load_candidates(args.candidates, args.metadata)
    rows = dataset["candidates"]
    obj = load_scenarios(args.scenarios) if args.scenarios else None
    evaluation = load_scenarios(args.evaluation_scenarios) if args.evaluation_scenarios else None
    for scenarios in (obj, evaluation):
        if scenarios is not None:
            if set(scenarios.ids) != {r["id"] for r in rows}:
                raise ValueError("candidate and scenario IDs must match before filtering/deduplication")
            if scenarios.synthetic != dataset["synthetic"]:
                raise ValueError("candidate/scenario synthetic provenance mismatch")
    if args.target:
        rows = [r for r in rows if r["target_id"] == args.target]
    if not rows or len({r["target_id"] for r in rows}) != 1:
        raise ValueError("select exactly one target (use --target); target must have candidates")
    duplicates = duplicate_groups(rows)
    if duplicates and not args.deduplicate:
        raise ValueError(f"duplicate sequences: {duplicates}; review or explicitly use --deduplicate first")
    rows = deduplicate(rows) if args.deduplicate else rows
    dataset["candidates"] = rows
    ids = [r["id"] for r in rows]
    obj = obj.align(ids) if obj else None
    evaluation = evaluation.align(ids) if evaluation else None
    if args.k < 0 or args.k > 20:
        raise ValueError("submission budget must be between 0 and 20")
    if args.method == "raw-topk":
        if not args.metric:
            raise ValueError("raw-topk needs --metric with explicit metric semantics")
        scores, higher = metric_vector(dataset, args.metric)
        selected = top_k(scores, args.k, higher=higher)
        ranking = top_k(scores, len(rows), higher=higher)
    else:
        if obj is None:
            raise ValueError("scenario optimization needs --scenarios")
        method = {"singleton-topk": singleton_top_k, "greedy": greedy, "portfolio": portfolio,
                  "exact": exhaustive}[args.method]
        selected = method(obj, args.k)
        ranking = singleton_top_k(obj, len(rows))
    chosen = [rows[i] for i in selected]
    report = {"schema_version": 1, "synthetic": dataset["synthetic"], "method": args.method,
              "budget": args.k, "target_id": rows[0]["target_id"], "selected_ids": [ids[i] for i in selected],
              "duplicate_groups": duplicates, "deduplication_policy": "first" if args.deduplicate else None,
              "diversity": diversity(chosen)}
    if args.method == "raw-topk":
        report["ranking_metric"] = args.metric
    if obj:
        value, ub = obj.value(selected), upper_bound(obj, selected, args.k)
        report.update({"scenario_description": obj.description, "objective_value": value,
                       "optimum_upper_bound": ub,
                       "objective_ratio_lower_bound": value / ub if ub > 0 else 1.0})
        if args.method in ("greedy", "portfolio"):
            report["theoretical_greedy_factor"] = approximation_factor(args.k)
        if evaluation:
            report["evaluation_description"] = evaluation.description
            report["evaluation_objective_value"] = evaluation.value(selected)
    elif evaluation:
        report["evaluation_description"] = evaluation.description
        report["evaluation_objective_value"] = evaluation.value(selected)
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    dump_json({**dataset, "candidates": chosen, "selection": report}, directory / "selection.json")
    label = " synthetic=true" if dataset["synthetic"] else ""
    (directory / "selected.fasta").write_text("".join(f">{r['id']}{label}\n{r['sequence']}\n" for r in chosen), encoding="utf-8")
    metric_keys = list(dataset["metric_definitions"])
    with (directory / "ranking.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["synthetic", "individual_rank", "id", "selection_order", "singleton_utility",
                         "leave_one_out_loss", *["metric:" + key for key in metric_keys]])
        for rank, i in enumerate(ranking, 1):
            loss = obj.value(selected) - obj.value([j for j in selected if j != i]) if obj and i in selected else None
            writer.writerow([dataset["synthetic"], rank, ids[i], selected.index(i) + 1 if i in selected else "",
                             obj.value([i]) if obj else "", loss,
                             *[rows[i]["metrics"].get(key) for key in metric_keys]])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "similarity", "select"):
        sub = commands.add_parser(name)
        sub.add_argument("candidates")
        sub.add_argument("--metadata", help="CSV metadata JSON; default <stem>.metrics.json")
        if name == "similarity":
            sub.add_argument("--out", required=True)
            sub.add_argument("--max-pairs", type=int, default=100_000)
        if name == "select":
            sub.add_argument("--target")
            sub.add_argument("--scenarios")
            sub.add_argument("--evaluation-scenarios")
            sub.add_argument("--method", choices=["raw-topk", "singleton-topk", "greedy", "portfolio", "exact"], default="portfolio")
            sub.add_argument("--metric")
            sub.add_argument("--k", type=int, default=20)
            sub.add_argument("--deduplicate", action="store_true")
            sub.add_argument("--out", required=True)
    example = commands.add_parser("synthetic")
    example.add_argument("--out", default="build/example")
    example.add_argument("--seed", type=int, default=7)
    example.add_argument("--n", type=int, default=12)
    example.add_argument("--m", type=int, default=64)
    example.add_argument("--regime", choices=["clustered", "independent", "aligned"], default="clustered")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--out", default="build/benchmark")
    benchmark.add_argument("--trials", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        if args.command == "select":
            result = select(args)
        elif args.command == "synthetic":
            write_example(args.out, args.seed, args.n, args.m, args.regime)
            result = {"synthetic": True, "directory": args.out}
        elif args.command == "benchmark":
            result = run_benchmark(args.out, args.trials)
        else:
            dataset = load_candidates(args.candidates, args.metadata)
            rows = dataset["candidates"]
            if args.command == "validate":
                result = {"count": len(rows), "synthetic": dataset["synthetic"],
                          "targets": sorted({r["target_id"] for r in rows}),
                          "duplicate_groups": duplicate_groups(rows)}
            else:
                if len(rows) * (len(rows) - 1) // 2 > args.max_pairs:
                    raise ValueError("too many pairs; subset input or explicitly increase --max-pairs")
                result = pairwise(rows)
                result["synthetic"] = dataset["synthetic"]
                destination = Path(args.out)
                destination.parent.mkdir(parents=True, exist_ok=True)
                dump_json(result, destination)
                result = {"count": len(rows), "synthetic": dataset["synthetic"], "path": str(destination)}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
