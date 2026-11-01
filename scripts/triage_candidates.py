"""Reuse existing portfolio and exact solvers on real measured PROXY ranks."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from changjiang_select.data import load_candidates
from changjiang_select.objective import save_scenarios
from changjiang_select.optimize import exhaustive, portfolio, singleton_top_k
from changjiang_select.proxies import rank_objective, score_correlations


def main(args):
    dataset = load_candidates(args.candidates)
    if dataset["synthetic"]:
        raise ValueError("this entry point is for real model outputs")
    all_rows = dataset["candidates"]
    missing_evaluation = []
    if args.require_boltz:
        missing_evaluation = [r["id"] for r in all_rows if r["metrics"].get("boltz.ipsae_10") is None]
        all_rows = [r for r in all_rows if r["metrics"].get("boltz.ipsae_10") is not None]
    keys = ["mpnn.score", "rfd.final_binder_plddt", "geometry.CA_contact_pairs8",
            "geometry.glycan_ND2_min_BB_distance", "sequence.hydrophobic8_max"]
    # Additional independent predictor values enter only when actually present.
    keys += [k for k in dataset["metric_definitions"] if k.startswith("boltz.") and
             k.endswith(("iptm", "ipsae_10", "binder_plddt", "binder_CA_rmsd"))]
    rejected, rows = {}, []
    for row in all_rows:
        m, reasons = row["metrics"], []
        if row["design_type"] != "minibinder" or not 40 <= len(row["sequence"]) <= 100:
            reasons.append("wrong sequence format")
        if m["geometry.other_chain_BB_clash_pairs2"]:
            reasons.append("unconditioned trimer backbone clash below 2 A")
        if m["geometry.glycan_ND2_min_BB_distance"] < 12:
            reasons.append("backbone within heuristic 12 A glycan-anchor envelope")
        if m["geometry.CA_contact_pairs8"] == 0:
            reasons.append("no target CA contacts within 8 A")
        if m.get("geometry.binder_nonlocal_CA_pairs8", 10) < 10:
            reasons.append("fewer than ten nonlocal CA contacts: current research prioritizes compact globular binders")
        if m.get("sequence.charged_fraction", 0) > .65 or m.get("sequence.hydrophobic_fraction", 1) < .15:
            reasons.append("extreme composition outside current globular-binder research gate")
        context = next((prefix for prefix in ("boltz_trimer_deepmsa.", "boltz_trimer.", "boltz_deepmsa.", "boltz.")
                        if m.get(prefix+"target_CA_rmsd") is not None), "boltz.")
        if m.get(context+"target_beta_CA_rmsd", m.get(context+"target_CA_rmsd", 0)) > 3:
            reasons.append("independent prediction changes known target core by >3 A; native-context geometry is unreliable")
        else:
            if m.get(context+"other_chain_clash_pairs2", 0):
                reasons.append("independent all-atom prediction clashes with native trimer below 2 A")
            if m.get(context+"glycan_ND2_min_distance", 12) < 12:
                reasons.append("independent all-atom prediction inside heuristic glycan envelope")
        if m.get(context+"ipsae_10") == 0:
            reasons.append("no interface support at the local 10 A PAE cutoff; not an official admission threshold")
        if reasons:
            rejected[row["id"]] = reasons
        else:
            rows.append(row)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Never leave a stale nonempty shortlist after a later validation rejects it.
    (out / "research_shortlist.fasta").write_text("")
    (out / "research_shortlist.csv").write_text("name,sequence\n")
    if not rows:
        for filename in ("eligible_candidates.json", "proxy_scenarios.json"):
            (out / filename).unlink(missing_ok=True)
    report = {"synthetic": False, "purpose": "compute-allocation triage; not an experimental shortlist or final competition submission",
              "candidate_count": len(all_rows), "unique_backbones": len({r['structural_group'] for r in all_rows}),
              "not_in_evaluated_cohort": missing_evaluation,
              "gate": "research-only sequence/compactness, native trimer and 12 A glycan-envelope gates; independent target beta-core/PAE checks when available; not official admission thresholds",
              "gate_survivors": len(rows), "rejections": rejected,
              "correlations_all_sequences": score_correlations(all_rows, keys),
              "dependence_warning": "sequence variants share backbone-level scores; no inferential p-values, calibration or success probabilities"}
    for pair in report["correlations_all_sequences"]:
        directions = [dataset["metric_definitions"][key]["direction"] for key in (pair["first"], pair["second"])]
        pair["oriented_spearman_rho"] = None if pair["spearman_rho"] is None or "unspecified" in directions else (
            pair["spearman_rho"]*(-1 if directions.count("lower") == 1 else 1))
    if rows:
        working = copy.deepcopy(dataset)
        working["candidates"] = rows
        obj, used, skipped = rank_objective(working, keys)
        k = min(args.k, len(rows), 20)
        selected, baseline = portfolio(obj, k), singleton_top_k(obj, k)
        exact = exhaustive(obj, k)
        portfolio_ids = [rows[i]["id"] for i in selected]
        family_count = lambda indices: len({rows[i]["structural_group"] for i in indices})
        # A secondary exploration preference can resolve an EXACT objective tie
        # without changing its value or claiming calibrated biological utility.
        tie_break = None
        if obj.value(exact) == obj.value(selected) and family_count(exact) > family_count(selected):
            selected = exact
            tie_break = "equal proxy objective: prefer the already-computed exact set covering more backbone families"
        report.update({"budget": k, "used_metrics": used, "skipped_metrics": skipped,
            "objective": obj.description, "selected_ids": [rows[i]["id"] for i in selected],
            "fitted_proxy_objectives": {"portfolio": obj.value(selected), "singleton_topk": obj.value(baseline), "exact": obj.value(exact)},
            "portfolio_selected_ids_before_tie_break": portfolio_ids,
            "exact_selected_ids": [rows[i]["id"] for i in exact], "tie_break": tie_break,
            "source_balanced_weights": dict(zip(used, obj.weights.tolist())),
            "selected_backbones": sorted({rows[i]["structural_group"] for i in selected}),
            "leave_one_metric_out": [], "leave_one_family_out": []})
        for key in used:
            reduced, _, _ = rank_objective(working, [x for x in used if x != key])
            alt = portfolio(reduced, k)
            report["leave_one_metric_out"].append({"omitted": key, "selected_ids": [rows[i]["id"] for i in alt],
                "jaccard": len(set(alt)&set(selected))/len(set(alt)|set(selected))})
        for family in sorted({key.split(".")[0] for key in used}):
            remaining = [x for x in used if x.split(".")[0] != family]
            if not remaining:
                continue
            reduced, _, _ = rank_objective(working, remaining)
            alt = portfolio(reduced, k)
            report["leave_one_family_out"].append({"omitted": family, "selected_ids": [rows[i]["id"] for i in alt],
                "jaccard": len(set(alt)&set(selected))/len(set(alt)|set(selected))})
        save_scenarios(obj, out / "proxy_scenarios.json")
        # Existing optimizer exports remain directly reproducible from this table.
        from changjiang_select.data import save_candidates
        save_candidates(working, out / "eligible_candidates.json")
        (out / "research_shortlist.fasta").write_text("".join(f">{rows[i]['id']} unvalidated_compute_priority\n{rows[i]['sequence']}\n" for i in selected))
        with (out / "research_shortlist.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "sequence"])
            writer.writerows((rows[i]["id"], rows[i]["sequence"]) for i in selected)
    else:
        report["selected_ids"] = []
    (out / "triage.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report.get(k) for k in ("candidate_count", "unique_backbones", "gate_survivors", "selected_ids", "fitted_proxy_objectives")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidates")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--require-boltz", action="store_true", help="compare only the explicitly evaluated complete-case cohort")
    p.add_argument("--out", default="build/real_triage")
    main(p.parse_args())
