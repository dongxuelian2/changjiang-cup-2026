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
from changjiang_select.proxies import rank_objective, score_correlations, source_family


def main(args):
    dataset = load_candidates(args.candidates)
    if dataset["synthetic"]:
        raise ValueError("this entry point is for real model outputs")
    all_rows = dataset["candidates"]
    missing_evaluation = []
    contexts = ("boltz_trimer_deepmsa.", "boltz_trimer.", "boltz_deepmsa.", "boltz.")
    boltz_ids = {r["id"] for r in all_rows if any(
        r["metrics"].get(c+"ipsae_10") is not None for c in contexts)}
    af2_ids = {r["id"] for r in all_rows if r["metrics"].get("af2.ipsae_10") is not None}
    coverage = dict(generated_sequences=len(all_rows),
        generated_backbones=len({r["structural_group"] for r in all_rows}),
        boltz_evaluated_sequences=len(boltz_ids), af2_evaluated_sequences=len(af2_ids),
        both_predictors_evaluated_sequences=len(boltz_ids & af2_ids))
    if args.require_boltz:
        has_prediction = lambda r: any(r["metrics"].get(c+"ipsae_10") is not None for c in contexts)
        missing_evaluation = [r["id"] for r in all_rows if not has_prediction(r)]
        all_rows = [r for r in all_rows if has_prediction(r)]
    if args.require_af2:
        missing_evaluation += [r["id"] for r in all_rows if r["metrics"].get("af2.ipsae_10") is None]
        all_rows = [r for r in all_rows if r["metrics"].get("af2.ipsae_10") is not None]
    keys = ["mpnn.score", "rfd.final_binder_plddt", "geometry.CA_contact_pairs8",
            "geometry.glycan_ND2_min_BB_distance", "sequence.hydrophobic8_max"]
    keys = [k for k in keys if k in dataset["metric_definitions"]]
    # Additional independent predictor values enter only when actually present.
    keys += [k for k in dataset["metric_definitions"] if k.startswith(contexts) and
             k.endswith(("iptm", "ipsae_10", "binder_plddt", "binder_CA_rmsd"))]
    keys += [k for k in dataset["metric_definitions"] if k.startswith("af2.") and
             k.endswith(("ipsae_10", "binder_plddt", "binder_CA_rmsd"))]
    rejected, rows = {}, []
    for row in all_rows:
        m, reasons = row["metrics"], []
        if row["design_type"] != "minibinder" or not 40 <= len(row["sequence"]) <= 100:
            reasons.append("wrong sequence format")
        if m["geometry.other_chain_BB_clash_pairs2"]:
            reasons.append("unconditioned trimer backbone clash below 2 A")
        if m.get("geometry.native_all_chain_BB_clash_pairs2", 0):
            reasons.append("generated binder backbone clashes with an observed native target atom below 2 A")
        if m["geometry.glycan_ND2_min_BB_distance"] < 12:
            reasons.append("backbone within heuristic 12 A glycan-anchor envelope")
        if m["geometry.CA_contact_pairs8"] == 0:
            reasons.append("no target CA contacts within 8 A")
        if m.get("geometry.binder_nonlocal_CA_pairs8", 10) < 10:
            reasons.append("fewer than ten nonlocal CA contacts: current research prioritizes compact globular binders")
        if m.get("sequence.charged_fraction", 0) > .65 or m.get("sequence.hydrophobic_fraction", 1) < .15:
            reasons.append("extreme composition outside current globular-binder research gate")
        context = next((prefix for prefix in contexts
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
        if m.get("af2.ipsae_10") == 0:
            reasons.append("AF2 has no interface support at the local 10 A PAE cutoff; not an experimental binding conclusion")
        if args.complete:
            # Declared research consistency thresholds, NOT calibrated KD,
            # official cutoffs, or a guarantee of experimental binding.
            if context != "boltz_trimer_deepmsa.":
                reasons.append("complete check requires no-template full-trimer prediction with target MSA depth >64")
            if m.get("geometry.native_all_chain_BB_clash_pairs2") is None:
                reasons.append("complete check requires generated backbone sterics against all three native protomers")
            requirements = {"target_beta_CA_rmsd": ("<=", 1.0),
                            "binder_fold_CA_rmsd": ("<=", 2.0),
                            "binder_CA_rmsd": ("<=", 3.0),
                            "binder_plddt": (">=", .8), "iptm": (">=", .6),
                            "ipsae_10": (">=", .5), "binder_buried_sasa": (">=", 300),
                            "binder_alone_CA_rmsd": ("<=", 2.0), "binder_alone_plddt": (">=", .8),
                            "native_all_chain_clash_pairs2": ("<=", 0),
                            "predicted_clash_pairs2": ("<=", 0)}
            for key, (op, threshold) in requirements.items():
                value = m.get(context+key)
                if value is None or (value > threshold if op == "<=" else value < threshold):
                    reasons.append(f"complete check: {key} must be {op}{threshold}; got {value}")
            if args.require_af2:
                af2_requirements = {"ipsae_10": (">=", .5), "binder_plddt": (">=", .8),
                    "target_beta_CA_rmsd": ("<=",1), "binder_fold_CA_rmsd": ("<=",2),
                    "binder_CA_rmsd": ("<=",3), "boltz_pose_CA_rmsd": ("<=",3),
                    "native_all_chain_clash_pairs2": ("<=",0), "predicted_clash_pairs2": ("<=",0),
                    "glycan_ND2_min_distance": (">=",12), "binder_buried_sasa": (">=",300)}
                for key,(op,threshold) in af2_requirements.items():
                    value=m.get("af2."+key)
                    if value is None or (value>threshold if op=="<=" else value<threshold):
                        reasons.append(f"AF2 cross-check: {key} must be {op}{threshold}; got {value}")
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
    elif len(rows) == 1:
        (out / "proxy_scenarios.json").unlink(missing_ok=True)
    report = {"synthetic": False, "purpose": ("research consistency screen; not official screening or experimental validation" if args.complete else
              "compute-allocation triage; not an experimental shortlist or final competition submission"),
              "evaluation_coverage": coverage,
              "complete_checks_required": args.complete, "af2_required": args.require_af2,
              "one_per_backbone": args.one_per_backbone,
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
    if len(rows) == 1:
        selected = [0]
        report.update(budget=1, selected_ids=[rows[0]["id"]],
                      fitted_proxy_objectives=None,
                      selection_note="only one feasible sequence; no comparative objective or utility invented")
    elif rows:
        working = copy.deepcopy(dataset)
        working["candidates"] = rows
        obj, used, skipped = rank_objective(working, keys)
        k = min(args.k, len(rows), 20)
        groups = [r["structural_group"] for r in rows] if args.one_per_backbone else None
        if groups is not None:
            k = min(k,len(set(groups)))
        selected, baseline = portfolio(obj, k, groups=groups), singleton_top_k(obj, k, groups=groups)
        try:
            exact = exhaustive(obj, k, groups=groups)
            exact_note = None
        except ValueError as exc:
            exact, exact_note = None,str(exc)
        portfolio_ids = [rows[i]["id"] for i in selected]
        family_count = lambda indices: len({rows[i]["structural_group"] for i in indices})
        # A secondary exploration preference can resolve an EXACT objective tie
        # without changing its value or claiming calibrated biological utility.
        tie_break = None
        if exact is not None and obj.value(exact) == obj.value(selected) and family_count(exact) > family_count(selected):
            selected = exact
            tie_break = "equal proxy objective: prefer the already-computed exact set covering more backbone families"
        report.update({"budget": k, "used_metrics": used, "skipped_metrics": skipped,
            "objective": obj.description, "selected_ids": [rows[i]["id"] for i in selected],
            "fitted_proxy_objectives": {"portfolio": obj.value(selected), "singleton_topk": obj.value(baseline), "exact": obj.value(exact) if exact is not None else None},
            "exact_search_note": exact_note,
            "portfolio_selected_ids_before_tie_break": portfolio_ids,
            "exact_selected_ids": [rows[i]["id"] for i in exact] if exact is not None else None, "tie_break": tie_break,
            "source_balanced_weights": dict(zip(used, obj.weights.tolist())),
            "selected_backbones": sorted({rows[i]["structural_group"] for i in selected}),
            "leave_one_metric_out": [], "leave_one_family_out": []})
        for key in used:
            if len(used) == 1:
                report["leave_one_metric_out"].append({"omitted": key, "selected_ids": None,
                    "jaccard": None, "reason": "no comparative metrics remain"})
                continue
            reduced, _, _ = rank_objective(working, [x for x in used if x != key])
            alt = portfolio(reduced, k, groups=groups)
            report["leave_one_metric_out"].append({"omitted": key, "selected_ids": [rows[i]["id"] for i in alt],
                "jaccard": len(set(alt)&set(selected))/len(set(alt)|set(selected))})
        for family in sorted({source_family(key) for key in used}):
            remaining = [x for x in used if source_family(x) != family]
            if not remaining:
                continue
            reduced, _, _ = rank_objective(working, remaining)
            alt = portfolio(reduced, k, groups=groups)
            report["leave_one_family_out"].append({"omitted": family, "selected_ids": [rows[i]["id"] for i in alt],
                "jaccard": len(set(alt)&set(selected))/len(set(alt)|set(selected))})
        save_scenarios(obj, out / "proxy_scenarios.json")
    else:
        report["selected_ids"] = []
    if rows:
        # Existing optimizer exports remain directly reproducible from this table.
        from changjiang_select.data import save_candidates
        working = copy.deepcopy(dataset)
        working["candidates"] = rows
        save_candidates(working, out / "eligible_candidates.json")
        (out / "research_shortlist.fasta").write_text("".join(f">{rows[i]['id']} unvalidated_compute_priority\n{rows[i]['sequence']}\n" for i in selected))
        with (out / "research_shortlist.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "sequence"])
            writer.writerows((rows[i]["id"], rows[i]["sequence"]) for i in selected)
    (out / "triage.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report.get(k) for k in ("candidate_count", "unique_backbones", "gate_survivors", "selected_ids", "fitted_proxy_objectives")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidates")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--require-boltz", action="store_true", help="compare only the explicitly evaluated complete-case cohort")
    p.add_argument("--require-af2", action="store_true", help="require an actual AF2 cross-check; in complete mode require both models' interface consistency")
    p.add_argument("--complete", action="store_true", help="require full-trimer deep-MSA fold/pose/interface and all-chain steric consistency; research criteria only")
    p.add_argument("--one-per-backbone", action="store_true", help="exploratory compute allocation: at most one sequence per generated backbone")
    p.add_argument("--out", default="build/real_triage")
    main(p.parse_args())
