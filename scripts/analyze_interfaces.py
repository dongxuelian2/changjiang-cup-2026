"""Diagnose real generated interfaces and target prediction disagreement.

Only atom14 generation has actual sidechains; inverse-folded placeholder
sidechains are never used. Prediction disagreement is not measured flexibility.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, ShrakeRupley, Superimposer
from Bio.PDB.Polypeptide import PPBuilder
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from changjiang_select.data import load_candidates
from scripts.import_boltz import align_prediction


def main(args):
    path = (ROOT / args.candidates).resolve()
    data = load_candidates(path)
    if data["synthetic"]:
        raise ValueError("requires actual model outputs")
    pdb, cif = PDBParser(QUIET=True), MMCIFParser(QUIET=True)
    native = pdb.get_structure("native", ROOT / "targets/TL1A/observed.pdb")
    manifest = json.loads((ROOT / "targets/manifest.json").read_text())
    core = next(t["beta_core_uniprot"] for t in manifest["targets"] if t["target_id"] == "TL1A")
    expected="".join(seq1(r.resname) for r in native[0]['A'])
    native_heavy=np.array([a.coord for a in native.get_atoms() if a.element!='H'])
    glycans=np.array([native[0][c][n]['ND2'].coord for c in 'ABC' for n in (133,229)])
    sr, interfaces, deviations = ShrakeRupley(n_points=100), [], []
    for row in data["candidates"]:
        source = row["metadata"].get("generation", {})
        if source.get("stage") == "atom14":
            raw = cif.get_structure(row["id"], path.parent / "generation" / Path(source["file"]).name)
            binder = next(c for c in raw[0] if "".join(seq1(r.resname) for r in c) == row["sequence"])
            targets=[c for c in raw[0] if "".join(seq1(r.resname) for r in c)==expected]
            if len(targets)!=3:raise ValueError('raw atom14 output must retain the exact trimer')
            fit=Superimposer()
            fit.set_atoms([r['CA'] for r in native[0]['A']],
                          [r['CA'] for r in targets[0]])
            tx = np.array([a.coord for c in raw[0] if c.id != binder.id for a in c.get_atoms() if a.element != "H"])
            sr.compute(raw, level="R")
            bound = {r.id: r.sasa for r in binder}
            alone = binder.copy()
            sr.compute(alone, level="R")
            burial = {r.id: max(0., r.sasa-bound[r.id]) for r in alone}
            contacting = [r for r in binder if np.linalg.norm(
                np.array([a.coord for a in r])[:, None]-tx, axis=-1).min() < 4.5]
            area = sum(burial.values())
            angles=[np.degrees(pair) for peptide in PPBuilder().build_peptides(binder)
                    for pair in peptide.get_phi_psi_list() if all(x is not None for x in pair)]
            alpha=sum(-100<=phi<=-30 and -80<=psi<=-5 for phi,psi in angles)/max(1,len(angles))
            bca=np.array([r['CA'].coord for r in binder])
            internal=np.linalg.norm(bca[:,None]-bca,axis=-1)
            nonlocal_mask=np.abs(np.arange(len(bca))[:,None]-np.arange(len(bca)))>=3
            sulfurs=[r['SG'].coord for r in binder if r.resname=='CYS' and 'SG' in r]
            sulfur_min=None; sulfur_pairs=0; sulfur_degree=0
            if len(sulfurs)>1:
                sd=np.linalg.norm(np.array(sulfurs)[:,None]-np.array(sulfurs),axis=-1)
                np.fill_diagonal(sd,np.inf)
                sulfur_min=float(sd.min())
                sulfur_pairs=int((sd<2.2).sum()//2)
                sulfur_degree=int((sd<2.2).sum(-1).max())
            # Preserve the original frame for the finite-point SASA calculation;
            # transform only afterwards for original-pose native steric checks.
            fit.apply(list(raw.get_atoms()))
            bb=np.array([a.coord for a in binder.get_atoms() if a.name in ('N','CA','C','O')])
            interfaces.append(dict(id=row["id"], binding_sites=",".join(source["binding_sites_uniprot"]),
                buried_binder_SASA_A2=area, contacting_binder_residues45=len(contacting),
                hydrophobic_burial_fraction=sum(burial[r.id] for r in binder if seq1(r.resname) in "AFILMVWY")/max(area, 1e-8),
                aromatic_burial_fraction=sum(burial[r.id] for r in binder if seq1(r.resname) in "FWY")/max(area, 1e-8),
                alpha_torsion_fraction=alpha,
                min_nonlocal_CA_distance_A=float(internal[nonlocal_mask].min()),
                cysteine_min_SG_distance_A=sulfur_min,
                cysteine_SG_pairs22=sulfur_pairs,cysteine_SG_max_degree22=sulfur_degree,
                AELK_fraction=sum(x in "AELK" for x in row["sequence"])/len(row["sequence"]),
                native_BB_clashes=int((np.linalg.norm(bb[:,None]-native_heavy,axis=-1)<2).sum()),
                glycan_BB_distance_A=float(np.linalg.norm(bb[:,None]-glycans,axis=-1).min())))
        design = pdb.get_structure("design", ROOT / row["structures"][0]["path"])[0]["D"]
        for record in row["structures"]:
            if record["role"] not in ("af2_predicted_complex", "boltz_trimer_deepmsa_predicted_complex"):
                continue
            filename = ROOT / record["path"]
            pred = (cif if filename.suffix == ".cif" else pdb).get_structure("pred", filename)
            _, _, perm, _, _ = align_prediction(native, pred, design, "ABC", "D", core)
            delta = [[np.linalg.norm(pred[0][p][n-71]["CA"].coord-native[0][c][n]["CA"].coord)
                      for n in range(88,252)] for c,p in zip("ABC",perm)]
            deviations.extend(delta)
    out = (ROOT / args.out).resolve() if args.out else path.parent
    out.mkdir(parents=True, exist_ok=True)
    if interfaces:
        with (out / "interface_audit.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(interfaces[0]))
            writer.writeheader()
            writer.writerows(interfaces)
    if deviations:
        values = np.array(deviations)
        with (out / "target_disagreement.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["uniprot_residue", "beta_core", "chain_predictions", "CA_deviation_p50_A", "CA_deviation_p90_A"])
            for i,n in enumerate(range(88,252)):
                writer.writerow([n,n in core,len(values),*np.quantile(values[:,i],[.5,.9])])
        for n in (110,123,156,158,165,166,167,168,175,188,209,235,238):
            print(n,"beta" if n in core else "loop",np.quantile(values[:,n-88],[.5,.9]).round(2).tolist())
    print("Real raw backbones / independent complex predictions:",len(interfaces),len(deviations)//3)
    if interfaces:
        print("Buried binder SASA median:",float(np.median([r["buried_binder_SASA_A2"] for r in interfaces])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidates")
    p.add_argument("--out")
    main(p.parse_args())
