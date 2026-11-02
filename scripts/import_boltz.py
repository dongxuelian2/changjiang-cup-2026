"""Preserve independent predictions/PAE, run pinned ipSAE and compare to design.

All Boltz-derived values have their own namespace and are not official scores.
"""
import argparse
import hashlib
import json
from itertools import permutations
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, Superimposer, ShrakeRupley
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from changjiang_select.data import load_candidates, save_candidates, save_csv


def align_prediction(reference, predicted, designed, target_ids, binder_id, beta_core):
    """Remove homotrimer chain-label ambiguity for pose comparison only.

    Among permutations within 0.5 A of the best target-core fit, choose the
    smallest binder-pose RMSD. Never modifies predictor confidence or PAE.
    The target-only best fit and all alternatives are retained for inspection.
    """
    trials = []
    bca = np.array([r["CA"].coord for r in predicted[0][binder_id]])
    dca = np.array([r["CA"].coord for r in designed])
    for perm in permutations(target_ids):
        whole, core = Superimposer(), Superimposer()
        whole.set_atoms([r["CA"] for c in target_ids for r in reference[0][c]],
                        [predicted[0][c][n-71]["CA"] for c in perm for n in range(88, 252)])
        core.set_atoms([reference[0][c][n]["CA"] for c in target_ids for n in beta_core],
                       [predicted[0][c][n-71]["CA"] for c in perm for n in beta_core])
        rot, tran = core.rotran
        pose = float(np.sqrt(np.mean(np.sum((bca @ rot+tran-dca)**2, axis=1))))
        trials.append((whole, core, perm, pose))
    minimum = min(t[1].rms for t in trials)
    chosen = min((t for t in trials if t[1].rms <= minimum+0.5), key=lambda t: t[3])
    chosen[1].apply(list(predicted.get_atoms()))
    details = [{"permutation": list(t[2]), "target_beta_CA_rmsd": float(t[1].rms),
                "binder_pose_CA_rmsd": t[3]} for t in trials]
    return chosen[0], chosen[1], chosen[2], minimum, details


def ipsae(pae, cif, cutoff, binder_id="B", target_ids="A"):
    script = ROOT / "build/models/IPSAE/ipsae.py"
    completed = subprocess.run([sys.executable, str(script), str(pae), str(cif), str(cutoff), str(cutoff)],
                               capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-3000:])
    files = list(cif.parent.glob(f"*_{cutoff:02}_{cutoff:02}.txt"))
    if len(files) != 1:
        raise ValueError(f"inspect ipSAE output names in {cif.parent}: {completed.stdout[-1000:]}")
    lines = [line.split() for line in files[0].read_text().splitlines() if line.strip()]
    header = lines[0]
    rows = [dict(zip(header, line)) for line in lines[1:] if len(line) == len(header)]
    relevant = [r for r in rows if binder_id in (r["Chn1"], r["Chn2"]) and
                any(t in (r["Chn1"], r["Chn2"]) for t in target_ids)]
    maximum = [r for r in relevant if r["Type"] == "max"]
    if len(maximum) != len(target_ids):
        raise ValueError("missing binder-target score pair")
    return max(float(r["ipSAE"]) for r in maximum), relevant


def main(args):
    dataset = load_candidates(args.candidates)
    out = Path(args.candidates).resolve().parent
    predictions = (ROOT / args.predictions).resolve()
    meta = json.loads((predictions / "run.json").read_text())
    trimer = meta["target_stoichiometry"] == "trimer"
    templated = meta["template"]
    deep_msa = meta["msa_depth_cap"] > 64
    context_key = "boltz_trimer_template" if trimer and templated else (
        "boltz_template" if templated else ("boltz_trimer_deepmsa" if trimer and deep_msa else (
            "boltz_trimer" if trimer else ("boltz_deepmsa" if deep_msa else "boltz"))))
    prefix = context_key+"."
    target_ids, binder_id = ("ABC", "D") if trimer else ("A", "B")
    target_count = 180*len(target_ids)
    native_sequence = "".join((ROOT / "targets/TL1A/construct.fasta").read_text().splitlines()[1:])
    reference = PDBParser(QUIET=True).get_structure("ref", ROOT / "targets/TL1A/observed.pdb")
    target_manifest = json.loads((ROOT / "targets/manifest.json").read_text())
    beta_core = next(t for t in target_manifest["targets"] if t["target_id"] == "TL1A")["beta_core_uniprot"]
    defs = dataset["metric_definitions"]
    for key, meaning, units, direction in [
        ("iptm", "maximum directional binder-target pair ipTM; excludes target-target pairs", "fraction", "higher"),
        ("binder_plddt", "mean designed-chain token pLDDT", "fraction", "higher"),
        ("ipsae_10", "pinned ipSAE, PAE/distance cutoffs 10 A; maximum over binder-target directional pairs only", "score", "higher"),
        ("ipsae_15", "pinned ipSAE, PAE/distance cutoffs 15 A; maximum over binder-target directional pairs only", "score", "higher"),
        ("binder_CA_rmsd", "binder CA RMSD to design after experimental target beta-core alignment (pose agreement)", "angstrom", "lower"),
        ("binder_fold_CA_rmsd", "binder-only aligned CA RMSD to design (fold agreement)", "angstrom", "lower"),
        ("target_CA_rmsd", "predicted target A88-251 versus 2RE9 aligned CA RMSD", "angstrom", "lower"),
        ("target_beta_CA_rmsd", "aligned CA RMSD over experimental 2RE9 SHEET residues only; separate from flexible loops", "angstrom", "lower"),
        ("other_chain_clash_pairs2", "all-atom predicted binder vs observed native trimer B/C atom pairs <2 A", "count", "lower"),
        ("glycan_ND2_min_distance", "all-atom predicted binder vs six native N133/N229 ND2 anchors", "angstrom", "higher")]:
        defs[prefix+key] = {"model": "Boltz-2 / ipSAE" if "ipsae" in key else "Boltz-2 / geometry",
            "version": "boltz 2.2.1; ipSAE 6174cf9", "meaning": meaning+"; NOT official ESMFold2 screening",
            "units": units, "direction": direction, "target_stoichiometry": "trimer" if trimer else "monomer",
            "target_template": templated, "msa_depth_cap": meta["msa_depth_cap"]}
    for key, meaning, units, direction in [
        ("target_best_beta_CA_rmsd", "best target-only beta-core RMSD across chain permutations", "angstrom", "lower"),
        ("native_all_chain_clash_pairs2", "predicted binder / ALL observed native target chains heavy-atom pairs below 2 A", "count", "lower"),
        ("predicted_clash_pairs2", "binder / target heavy-atom pairs below 2 A in the prediction itself", "count", "lower"),
        ("binder_buried_sasa", "binder solvent area lost upon complex formation; Shrake-Rupley 100 points, probe 1.4 A", "angstrom^2", "higher")]:
        defs[prefix+key] = dict(model="Boltz-2 / geometry", version="boltz 2.2.1; derived geometry v3",
                               meaning=meaning+"; NOT official screening", units=units, direction=direction)
    if meta.get("binder_controls"):
        for key, meaning, units, direction in [
            ("binder_alone_plddt", "mean pLDDT of sequence-only binder-alone prediction, no MSA/template", "fraction", "higher"),
            ("binder_alone_CA_rmsd", "binder-alone CA RMSD to generated fold after binder-only alignment", "angstrom", "lower")]:
            defs[prefix+key] = dict(model="Boltz-2 binder-alone control", version="boltz 2.2.1",
                                   meaning=meaning, units=units, direction=direction)
    processed, missing = [], []
    requested = set(meta["candidate_ids"])
    if requested - {r["id"] for r in dataset["candidates"]}:
        raise ValueError("prediction metadata contains unknown candidate IDs")
    for row in dataset["candidates"]:
        if row["id"] not in requested:
            continue
        name = row["id"]
        files = list(predictions.glob(f"boltz_results_*/predictions/{name}/{name}_model_0.cif"))
        if not files:
            missing.append(name)
            continue
        if len(files) != 1:
            raise ValueError("ambiguous prediction files")
        cif = files[0]
        pae = cif.with_name(f"pae_{name}_model_0.npz")
        confidence = json.loads(cif.with_name(f"confidence_{name}_model_0.json").read_text())
        predicted = MMCIFParser(QUIET=True).get_structure(name, cif)
        target, binder = predicted[0]["A"], predicted[0][binder_id]
        if "".join(seq1(r.resname) for r in binder) != row["sequence"] or any(
                "".join(seq1(r.resname) for r in predicted[0][c]) != native_sequence for c in target_ids):
            raise ValueError("prediction chain/sequence mismatch")
        if np.load(pae)["pae"].shape != (target_count+len(binder), target_count+len(binder)):
            raise ValueError("PAE/residue indexing mismatch")
        designed = PDBParser(QUIET=True).get_structure("design", ROOT / row["structures"][0]["path"])[0]["D"]
        align, core_align, best_permutation, best_core, symmetry_trials = align_prediction(
            reference, predicted, designed, target_ids, binder_id, beta_core)
        bca = np.array([r["CA"].coord for r in binder])
        dca = np.array([r["CA"].coord for r in designed])
        fold = Superimposer()
        fold.set_atoms([r["CA"] for r in designed], [r["CA"] for r in binder])
        bxyz = np.array([a.coord for a in binder.get_atoms() if a.element != "H"])
        neighbors = np.array([a.coord for c in reference[0] if c.id != "A" for a in c.get_atoms()])
        anchors = np.array([reference[0][c][n]["ND2"].coord for c in "ABC" for n in (133, 229)])
        native_all = np.array([a.coord for a in reference.get_atoms() if a.element != "H"])
        predicted_target = np.array([a.coord for c in target_ids for a in predicted[0][c].get_atoms() if a.element != "H"])
        sr = ShrakeRupley(n_points=100)
        sr.compute(predicted, level="A")
        complex_area = sum(a.sasa for a in binder.get_atoms())
        isolated = binder.copy()
        sr.compute(isolated, level="A")
        buried_area = sum(a.sasa for a in isolated.get_atoms())-complex_area
        plddt_path = cif.with_name(f"plddt_{name}_model_0.npz")
        plddt = np.load(plddt_path)["plddt"]
        if plddt.shape != (target_count+len(binder),):
            raise ValueError("pLDDT/token count mismatch")
        pair_conf = confidence["pair_chains_iptm"]
        bi = str(len(target_ids))
        binder_iptm = max(float(pair_conf[a][b]) for i in range(len(target_ids))
                         for a, b in ((str(i), bi), (bi, str(i))))
        m = {"boltz.iptm": binder_iptm, "boltz.binder_plddt": float(plddt[target_count:].mean()),
             "boltz.binder_CA_rmsd": float(np.sqrt(np.mean(np.sum((bca-dca)**2, axis=1)))),
             "boltz.binder_fold_CA_rmsd": float(fold.rms), "boltz.target_CA_rmsd": float(align.rms),
             "boltz.target_beta_CA_rmsd": float(core_align.rms),
             "boltz.other_chain_clash_pairs2": int((np.linalg.norm(bxyz[:, None]-neighbors, axis=-1) < 2).sum()),
             "boltz.glycan_ND2_min_distance": float(np.linalg.norm(bxyz[:, None]-anchors, axis=-1).min()),
             "boltz.target_best_beta_CA_rmsd": float(best_core),
             "boltz.native_all_chain_clash_pairs2": int((np.linalg.norm(bxyz[:, None]-native_all, axis=-1) < 2).sum()),
             "boltz.predicted_clash_pairs2": int((np.linalg.norm(bxyz[:, None]-predicted_target, axis=-1) < 2).sum()),
             "boltz.binder_buried_sasa": float(buried_area)}
        iprows = {}
        for cutoff in (10, 15):
            score, raw = ipsae(pae, cif, cutoff, binder_id, target_ids)
            m[f"boltz.ipsae_{cutoff}"] = score
            iprows[str(cutoff)] = raw
        row["metrics"].update({prefix+key.removeprefix("boltz."): value for key, value in m.items()})
        dest = out / "predictions"
        if trimer or templated or deep_msa:
            dest = dest / context_key.removeprefix("boltz_")
        dest.mkdir(parents=True, exist_ok=True)
        for source in (cif, pae, plddt_path):
            shutil.copyfile(source, dest / source.name)
        structure_record = {"path": (dest / cif.name).relative_to(ROOT).as_posix(),
                            "role": context_key+"_predicted_complex", "chain_id": binder_id}
        row["structures"] = [s for s in row["structures"] if s["path"] != structure_record["path"]]
        row["structures"].append(structure_record)
        row["metadata"][context_key] = {"status": "predicted_complex_unvalidated", "confidence": confidence,
            "run": str(predictions.relative_to(ROOT)), "seed": meta["command"][meta["command"].index("--seed")+1],
            "official_equivalence": False, "pae_sha256": hashlib.sha256(pae.read_bytes()).hexdigest(),
            "ipsae_pairs": iprows, "target_template": templated, "target_chain_permutation": list(best_permutation),
            "native_context_geometry_caveat": "alignment unreliable if target core differs substantially from native structure"}
        row["metadata"][context_key]["geometry_alignment"] = "2RE9 SHEET core; symmetry-aware derived geometry v3; pose selection only among fits within 0.5 A of best target-only core"
        row["metadata"][context_key]["symmetry_alignment_trials"] = symmetry_trials
        processed.append(name)
    for control_id, candidate_id in meta.get("binder_controls", {}).items():
        matches = list(predictions.glob(f"boltz_results_*/predictions/{control_id}/{control_id}_model_0.cif"))
        if len(matches) != 1:
            missing.append(control_id)
            continue
        row = next(r for r in dataset["candidates"] if r["id"] == candidate_id)
        cif = matches[0]
        folded = MMCIFParser(QUIET=True).get_structure(control_id, cif)
        if len(list(folded[0])) != 1 or "".join(seq1(r.resname) for r in folded[0]["A"]) != row["sequence"]:
            raise ValueError("binder-alone sequence/chain mismatch")
        design = PDBParser(QUIET=True).get_structure("design", ROOT / row["structures"][0]["path"])
        fit = Superimposer()
        fit.set_atoms([r["CA"] for r in design[0]["D"]], [r["CA"] for r in folded[0]["A"]])
        plddt_path = cif.with_name(f"plddt_{control_id}_model_0.npz")
        plddt = np.load(plddt_path)["plddt"]
        if plddt.shape != (len(row["sequence"]),):
            raise ValueError("binder-alone pLDDT shape mismatch")
        row["metrics"][prefix+"binder_alone_CA_rmsd"] = float(fit.rms)
        row["metrics"][prefix+"binder_alone_plddt"] = float(plddt.mean())
        folder = out / "predictions" / context_key
        folder.mkdir(parents=True, exist_ok=True)
        for source in (cif, plddt_path):
            shutil.copyfile(source, folder / source.name)
        structure_record = dict(path=(folder / cif.name).relative_to(ROOT).as_posix(),
                                chain_id="A", role=context_key+"_predicted_binder_alone")
        row["structures"] = [s for s in row["structures"] if s["path"] != structure_record["path"]]+[structure_record]
        row["metadata"].setdefault(context_key, {})["binder_alone_control"] = dict(
            id=control_id, confidence=json.loads(cif.with_name(f"confidence_{control_id}_model_0.json").read_text()))
    save_candidates(dataset, args.candidates)
    save_csv(dataset, out / "candidates.csv")
    report_name = context_key+"_evaluation.json"
    controls = []
    for control_id in meta.get("controls", []):
        files = list(predictions.glob(f"boltz_results_*/predictions/{control_id}/{control_id}_model_0.cif"))
        if not files:
            continue
        cif = files[0]
        control = MMCIFParser(QUIET=True).get_structure(control_id, cif)
        align = Superimposer()
        align.set_atoms([r["CA"] for r in reference[0]["A"]], [control[0]["A"][n-71]["CA"] for n in range(88, 252)])
        core_align = Superimposer()
        core_align.set_atoms([reference[0]["A"][n]["CA"] for n in beta_core],
                             [control[0]["A"][n-71]["CA"] for n in beta_core])
        confidence = json.loads(cif.with_name(f"confidence_{control_id}_model_0.json").read_text())
        control_folder = out / "predictions" / (context_key if deep_msa or templated or trimer else "")
        control_folder.mkdir(parents=True, exist_ok=True)
        dest = control_folder / cif.name
        shutil.copyfile(cif, dest)
        controls.append({"id": control_id, "target_CA_rmsd": float(align.rms),
                         "target_beta_CA_rmsd": float(core_align.rms), "confidence": confidence,
                         "structure": dest.relative_to(ROOT).as_posix()})
    report_path = out / report_name
    previous = json.loads(report_path.read_text()) if report_path.exists() else {"runs": {}}
    if "runs" not in previous:
        old_command = previous["run"]["command"]
        old_key = Path(old_command[old_command.index("--out_dir")+1]).relative_to(ROOT).as_posix()
        previous = {"runs": {old_key: previous}}
    previous["runs"][predictions.relative_to(ROOT).as_posix()] = {
        "run": meta, "processed": processed, "missing": missing, "controls": controls}
    report_path.write_text(json.dumps(previous, indent=2)+"\n")
    print(json.dumps({"processed": processed, "missing": missing}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidates")
    p.add_argument("predictions")
    main(p.parse_args())
