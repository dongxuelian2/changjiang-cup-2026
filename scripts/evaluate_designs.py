"""Import real RFdiffusion/ProteinMPNN runs and evaluate explicit, limited proxies.

Design coordinates are BACKBONE poses, not sequence-refolded complex predictions.
ipSAE/KD remain missing until produced by the appropriate prediction/experiment.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import sys

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Superimposer
from Bio.SeqUtils import seq1, seq3
from Bio.SeqUtils.ProtParam import ProteinAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from changjiang_select.data import save_candidates, save_csv


def sequence_metrics(sequence):
    p = ProteinAnalysis(sequence)
    return {"sequence.length": len(sequence), "sequence.gravy": p.gravy(),
            "sequence.pI": p.isoelectric_point(), "sequence.instability_index": p.instability_index(),
            "sequence.charged_fraction": sum(x in "DEKR" for x in sequence)/len(sequence),
            "sequence.hydrophobic_fraction": sum(x in "AFILMVWY" for x in sequence)/len(sequence),
            "sequence.entropy": -sum((sequence.count(x)/len(sequence))*math.log2(sequence.count(x)/len(sequence)) for x in set(sequence)),
            "sequence.hydrophobic8_max": max(sum(x in "AFILMVWY" for x in sequence[i:i+8])/8 for i in range(len(sequence)-7))}


def geometry(pdb, reference, binder_chain="B"):
    pose = PDBParser(QUIET=True).get_structure("pose", pdb)
    binder, target = pose[0][binder_chain], pose[0]["A"]
    target_res = list(target)
    native_res = list(reference[0]["A"])
    if "".join(seq1(r.resname) for r in target_res) != "".join(seq1(r.resname) for r in native_res):
        raise ValueError(f"target changed or chain convention incorrect: {pdb}")
    superimposer = Superimposer()
    superimposer.set_atoms([r["CA"] for r in native_res], [r["CA"] for r in target_res])
    superimposer.apply(list(pose.get_atoms()))
    bxyz = np.array([a.coord for a in binder.get_atoms()])
    bca = np.array([r["CA"].coord for r in binder])
    separations = np.arange(len(bca))[:, None]-np.arange(len(bca))[None, :]
    internal_contacts = (np.linalg.norm(bca[:, None]-bca, axis=-1) < 8) & (separations >= 8)
    neighbors = np.array([a.coord for c in reference[0] if c.id != "A" for a in c.get_atoms()])
    glycan = np.array([reference[0][c][n]["ND2"].coord for c in "ABC" for n in (133, 229)])
    epitope = []
    ca_contacts = 0
    for c in reference[0]:
        for r in c:
            xyz = np.array([a.coord for a in r])
            if np.linalg.norm(xyz[:, None]-bxyz, axis=-1).min() < 5:
                epitope.append(f"{c.id}:{r.id[1]}")
            ca_contacts += int((np.linalg.norm(r["CA"].coord-bca, axis=-1) < 8).sum())
    partner_distance = np.linalg.norm(bxyz[:, None]-neighbors, axis=-1)
    # Reconstitute the full observed trimer around the aligned designed binder.
    complex_pose = copy.deepcopy(reference)
    binder_copy = copy.deepcopy(binder)
    binder_copy.detach_parent()
    binder_copy.id = "D"
    complex_pose[0].add(binder_copy)
    metrics = {"geometry.target_CA_rmsd": float(superimposer.rms),
               "geometry.binder_Rg": float(np.sqrt(np.mean(np.sum((bca-bca.mean(0))**2, axis=1)))),
               "geometry.binder_nonlocal_CA_pairs8": int(internal_contacts.sum()),
               "geometry.CA_contact_pairs8": ca_contacts,
               "geometry.epitope_residues5": len(epitope),
               "geometry.other_chain_min_BB_distance": float(partner_distance.min()),
               "geometry.other_chain_BB_clash_pairs2": int((partner_distance < 2).sum()),
               "geometry.glycan_ND2_min_BB_distance": float(np.linalg.norm(bxyz[:, None]-glycan, axis=-1).min()),
               "geometry.missing_tail_anchor_distance": float(np.linalg.norm(bxyz-native_res[0]["N"].coord, axis=-1).min())}
    return complex_pose, metrics, epitope


def main(args):
    out = ROOT / args.out
    (out / "poses").mkdir(parents=True, exist_ok=True)
    reference = PDBParser(QUIET=True).get_structure("TL1A", ROOT / "targets/TL1A/observed.pdb")
    rows, runs = [], []
    definitions = {}
    def define(key, model, version, meaning, units, direction):
        definitions[key] = dict(model=model, version=version, meaning=meaning, units=units, direction=direction)
    for key, meaning, units, direction in [
        ("length", "mature designed chain length", "aa", "unspecified"),
        ("gravy", "Kyte-Doolittle mean hydropathy; not solubility", "hydropathy", "unspecified"),
        ("pI", "sequence-only estimated isoelectric point", "pH", "unspecified"),
        ("charged_fraction", "D/E/K/R fraction; extreme charge may explain a score artifact", "fraction", "unspecified"),
        ("hydrophobic_fraction", "A/F/I/L/M/V/W/Y fraction; not all exposed hydrophobicity", "fraction", "unspecified"),
        ("entropy", "single-residue Shannon entropy; not a novelty test", "bits", "unspecified"),
        ("instability_index", "Guruprasad sequence heuristic; not measured stability", "index", "lower"),
        ("hydrophobic8_max", "maximum hydrophobic fraction in eight-residue window; aggregation-risk proxy", "fraction", "lower")]:
        define("sequence."+key, "BioPython ProtParam / explicit counts", "1.85 / local-v1", meaning, units, direction)
    for key, meaning, units, direction in [
        ("target_CA_rmsd", "generated target CA versus reference A88-251 after alignment", "angstrom", "lower"),
        ("binder_Rg", "binder CA radius of gyration; rod/compactness diagnostic, not an affinity score", "angstrom", "unspecified"),
        ("binder_nonlocal_CA_pairs8", "binder CA pairs within 8 A separated by at least 8 sequence positions", "count", "higher"),
        ("CA_contact_pairs8", "full observed trimer / binder CA pairs within 8 A", "count", "higher"),
        ("epitope_residues5", "target residues with any heavy atom within 5 A of binder BACKBONE", "count", "higher"),
        ("other_chain_min_BB_distance", "binder backbone to unconditioned trimer B/C heavy atoms", "angstrom", "higher"),
        ("other_chain_BB_clash_pairs2", "binder backbone / trimer B/C atom pairs below 2 A", "count", "lower"),
        ("glycan_ND2_min_BB_distance", "minimum binder backbone distance to six N133/N229 ND2 anchors; not a glycan simulation", "angstrom", "higher"),
        ("missing_tail_anchor_distance", "binder backbone distance to resolved A88 N; proximity flag only", "angstrom", "higher")]:
        define("geometry."+key, "observed 2RE9 + generated backbone", "local-v1", meaning, units, direction)
    define("rfd.final_binder_plddt", "RFdiffusion Complex_base", "86507b6", "last denoising-step mean binder confidence; shared by all sequences on one backbone, NOT refold confidence", "fraction 0-1", "higher")
    define("mpnn.score", "ProteinMPNN v_48_020", "8907e66", "mean designed-residue negative log probability, T=0.1, C omitted", "nats/residue", "lower")
    define("official.ipsae", "official ESMFold2 + ipSAE", "unpublished", "official admission score; no substitute imputed", "score", "higher")
    for path in args.runs:
        directory = ROOT / path
        run = json.loads((directory / "run.json").read_text())
        if run["status"] != "generated_not_independently_validated":
            raise ValueError(f"run not finished: {directory}")
        run_id = directory.name
        run["backbone_outputs"] = []
        runs.append(run)
        for pdb in sorted(directory.glob("design_*.pdb")):
            # Only our locally produced trusted pickle is read.
            with pdb.with_suffix(".trb").open("rb") as f:
                trb = pickle.load(f)
            run.setdefault("resolved_rfd_config", trb["config"])
            run["backbone_outputs"].append({"file": pdb.name,
                "generation_seconds": trb["time"], "sha256": hashlib.sha256(pdb.read_bytes()).hexdigest(),
                "seed": int(pdb.stem.split('_')[-1])})
            run.setdefault("mpnn_checkpoint_sha256", hashlib.sha256((ROOT / "build/models/ProteinMPNN/vanilla_model_weights/v_48_020.pt").read_bytes()).hexdigest())
            pose, metrics, epitope = geometry(pdb, reference, run.get("binder_chain", "B"))
            n_binder = len(pose[0]["D"])
            metrics["rfd.final_binder_plddt"] = float(np.asarray(trb["plddt"])[-1, :n_binder].mean())
            fasta = directory / "mpnn/seqs" / (pdb.stem + ".fa")
            lines = fasta.read_text().splitlines()
            sample = 0
            for header, sequence in zip(lines[0::2], lines[1::2]):
                if "sample=" not in header:
                    continue
                sample += 1
                if len(sequence) != n_binder or not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
                    raise ValueError("MPNN chain/sequence mismatch")
                context_tag = "trimer_" if run.get("context") == "trimer" else ""
                name = f"tl1a_{context_tag}{run['site']}_{pdb.stem.split('_')[-1]}_s{sample}"
                candidate_pose = copy.deepcopy(pose)
                for residue, aa in zip(candidate_pose[0]["D"], sequence):
                    residue.resname = seq3(aa).upper()
                dest = out / "poses" / (name + "_design.pdb")
                writer = PDBIO()
                writer.set_structure(candidate_pose)
                writer.save(str(dest))
                score = float(re.search(r"(?:^|, )score=([0-9.eE+-]+)", header).group(1))
                candidate_metrics = {**metrics, **sequence_metrics(sequence), "mpnn.score": score, "official.ipsae": None}
                rows.append({"id": name, "sequence": sequence, "target_id": "TL1A", "design_type": "minibinder",
                    "source_model": "RFdiffusion+ProteinMPNN", "source_version": "86507b6+8907e66", "run_id": run_id,
                    "metrics": candidate_metrics, "risk_group": f"{run_id}:{pdb.stem}", "structural_group": f"{run_id}:{pdb.stem}",
                    "epitope_residues": epitope,
                    "structures": [{"path": dest.relative_to(ROOT).as_posix(), "role": "designed_backbone_complex", "chain_id": "D"}],
                    "metadata": {"synthetic": False, "status": "unvalidated_design", "mpnn_header": header,
                        "mpnn_seed": 2026, "backbone_seed": int(pdb.stem.split('_')[-1]), "hotspots_uniprot": run["hotspots_uniprot"],
                        "binder_sidechains_present": False, "known_positive_binder_used": False,
                        "backbone_pdb_sha256": hashlib.sha256(pdb.read_bytes()).hexdigest()}})
    dataset = {"schema_version": 1, "synthetic": False,
               "description": "Real de novo model output. Backbone poses and limited proxies only; no official score or wet-lab KD.",
               "metric_definitions": definitions, "candidates": rows}
    save_candidates(dataset, out / "candidates.json")
    save_csv(dataset, out / "candidates.csv")
    (out / "candidates.fasta").write_text("".join(f">{r['id']} unvalidated_design\n{r['sequence']}\n" for r in rows))
    (out / "generation.json").write_text(json.dumps(runs, indent=2) + "\n")
    # Native Boltz YAML: FULL official target construct and an unaligned binder.
    pending = ROOT / "build/boltz_inputs"
    pending.mkdir(parents=True, exist_ok=True)
    target_seq = "".join((ROOT / "targets/TL1A/construct.fasta").read_text().splitlines()[1:])
    for row in rows:
        (pending / (row["id"]+".yaml")).write_text(
            f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {target_seq}\n      msa: empty\n"
            f"  - protein:\n      id: B\n      sequence: {row['sequence']}\n      msa: empty\n")
    print(json.dumps({"real_candidates": len(rows), "distinct_sequences": len({r['sequence'] for r in rows}),
                      "backbones": len({r['structural_group'] for r in rows}), "out": str(out)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="results/tl1a_20261101")
    main(p.parse_args())
