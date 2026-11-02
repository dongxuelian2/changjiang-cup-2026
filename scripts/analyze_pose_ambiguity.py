"""Compare real interface footprints after native, symmetry-aware alignment.

Distances to native atoms deliberately differ from predicted-complex contacts.
Rigid motion and internal fold changes are reported separately. No binding
energies or experimental probabilities are inferred.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, Superimposer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.import_boltz import align_prediction


def footprint(binder, native, cutoff=4.5):
    xyz = np.array([a.coord for a in binder.get_atoms() if a.element != "H"])
    result = {}
    for chain in native[0]:
        for residue in chain:
            target = np.array([a.coord for a in residue if a.element != "H"])
            distance = float(np.linalg.norm(xyz[:, None] - target, axis=-1).min())
            if distance < cutoff:
                result[f"{chain.id}:{residue.id[1]}"] = round(distance, 3)
    return result


def compare(first, second):
    x, y = [np.array([r["CA"].coord for r in chain]) for chain in (first, second)]
    fit = Superimposer()
    fit.set_atoms([r["CA"] for r in first], [r["CA"] for r in second])
    rotation = fit.rotran[0]
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    pose_error = np.linalg.norm(x-y,axis=-1)
    fold_error = np.linalg.norm(x-(y@rotation+fit.rotran[1]),axis=-1)
    return dict(pose_CA_rmsd_A=float(np.sqrt(np.mean(np.sum((x-y)**2, axis=-1)))),
                fold_CA_rmsd_A=float(fit.rms),
                centroid_shift_A=float(np.linalg.norm(x.mean(0)-y.mean(0))),
                fitted_rigid_rotation_deg=float(angle),
                per_residue_CA_pose_error_A=pose_error.tolist(),
                per_residue_CA_fold_error_A=fold_error.tolist())


def native_overlap(binder, native):
    ba = [a for a in binder.get_atoms() if a.element != 'H']
    ta = [a for a in native.get_atoms() if a.element != 'H']
    pairs = np.argwhere(np.linalg.norm(np.array([a.coord for a in ba])[:, None]
                                     - np.array([a.coord for a in ta]), axis=-1) < 2)
    bb = {'N', 'CA', 'C', 'O'}
    counts = {'binderBB_targetBB': 0, 'binderBB_targetSC': 0,
              'binderSC_targetBB': 0, 'binderSC_targetSC': 0}
    for i, j in pairs:
        key = ('binderBB' if ba[i].name in bb else 'binderSC')+'_'+('targetBB' if ta[j].name in bb else 'targetSC')
        counts[key] += 1
    return counts


def internal_geometry(binder):
    residues = list(binder)
    backbone = [(i, a) for i,r in enumerate(residues) for a in r if a.name in ('N','CA','C','O')]
    xyz = np.array([a.coord for _,a in backbone])
    index = np.array([i for i,_ in backbone])
    nonbonded = np.abs(index[:,None]-index)>1
    distances = np.linalg.norm(xyz[:,None]-xyz,axis=-1)
    cysteines = [(i+1,r['SG']) for i,r in enumerate(residues) if r.resname=='CYS' and 'SG' in r]
    pairs = [(i,j,float(np.linalg.norm(a.coord-b.coord))) for (i,a),(j,b) in combinations(cysteines,2)]
    close = [p for p in pairs if p[2]<2.2]
    return dict(nonadjacent_BB_clash_pairs2=int((nonbonded & (distances<2)).sum()//2),
        cysteine_SG_pairs22=[dict(positions=[i,j],distance_A=d) for i,j,d in close],
        cysteine_SG_max_degree22=max((sum(i in (a,b) for a,b,_ in close) for i,_ in cysteines),default=0),
        interpretation='SG proximity is not a verified covalent bond; degree >1 cannot describe an ordinary disulfide matching')


def main(args):
    path = (ROOT / args.candidates).resolve()
    data = json.loads(path.read_text())
    if data["synthetic"]:
        raise ValueError("requires real predictions")
    native = PDBParser(QUIET=True).get_structure("native", ROOT / "targets/TL1A/observed.pdb")
    manifest = json.loads((ROOT / "targets/manifest.json").read_text())
    core = next(t["beta_core_uniprot"] for t in manifest["targets"] if t["target_id"] == "TL1A")
    report = dict(synthetic=False, contact_cutoff_A=4.5, candidates={})
    for row in data["candidates"]:
        if args.ids and row["id"] not in args.ids:
            continue
        design = PDBParser(QUIET=True).get_structure("design", ROOT / row["structures"][0]["path"])
        poses = {"design_backbone": design[0]["D"]}
        target_deviations = {}
        for record in row["structures"]:
            role = record["role"]
            if not (role.startswith('boltz_trimer_deepmsa') and role.endswith('_predicted_complex') or
                    role.startswith('af2') and role.endswith('_predicted_complex')):
                continue
            filename = ROOT / record["path"]
            parser = MMCIFParser(QUIET=True) if filename.suffix == ".cif" else PDBParser(QUIET=True)
            prediction = parser.get_structure("prediction", filename)
            # All stored independent complexes are ABC target followed by D.
            for chain in prediction[0]:
                residues = list(chain)
                for residue in residues:
                    chain.detach_child(residue.id)
                for i, residue in enumerate(residues, 1):
                    residue.id = (" ", i, " ")
                    chain.add(residue)
            poses[role] = prediction[0]["D"]
            _, _, permutation, _, _ = align_prediction(native, prediction, poses["design_backbone"], "ABC", "D", core)
            contacts = footprint(poses[role], native)
            mapping = dict(zip('ABC', permutation))
            target_deviations[role] = {key: float(np.linalg.norm(
                prediction[0][mapping[key.split(':')[0]]][int(key.split(':')[1])-71]['CA'].coord
                -native[0][key.split(':')[0]][int(key.split(':')[1])]['CA'].coord)) for key in contacts}
        item = dict(footprints={k: footprint(v, native) for k, v in poses.items()},
                    native_overlaps={k: native_overlap(v, native) for k,v in poses.items()},
                    binder_internal_geometry={k: internal_geometry(v) for k,v in poses.items()},
                    target_contact_CA_deviation_A=target_deviations, comparisons={},
                    atom_sets='design has binder backbone only; independent predictions have all heavy atoms')
        for a, b in combinations(poses, 2):
            # Re-align the second prediction to the FIRST binder, allowing the
            # three equivalent proper trimer rotations; never fit the binder.
            first, second = poses[a], poses[b].copy()
            parent = poses[b].get_parent().get_parent().copy() if b != "design_backbone" else None
            if parent is not None:
                # Already native-aligned, so restore target residue convention
                # expected by align_prediction (full official target 1..180).
                align_prediction(native, parent, first, "ABC", "D", core)
                second = parent[0]["D"]
            values = compare(first, second)
            f, g = set(footprint(first, native)), set(footprint(second, native))
            values["native_residue_contact_jaccard"] = len(f & g) / max(1, len(f | g))
            values["shared_native_contacts"] = sorted(f & g)
            item["comparisons"][a + "__" + b] = values
        report["candidates"][row["id"]] = item
    output = (ROOT / args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
