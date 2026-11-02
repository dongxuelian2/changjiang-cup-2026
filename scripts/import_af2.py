"""Preserve local AF2-Multimer predictions and compare interfaces across models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, Superimposer, ShrakeRupley
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from changjiang_select.data import load_candidates, save_candidates, save_csv
from scripts.import_boltz import align_prediction, ipsae


def main(args):
    namespace=getattr(args,'namespace','af2')
    if not namespace.startswith('af2') or not all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in namespace):
        raise ValueError('AF2 namespace must be a safe lowercase af2-prefixed identifier')
    def metric_name(key):
        return namespace+key.removeprefix('af2')
    dataset = load_candidates(args.candidates)
    predictions = (ROOT / args.predictions).resolve()
    out = Path(args.candidates).resolve().parent
    meta = json.loads((predictions / "run.json").read_text())
    log_path=predictions/"prediction.log"
    if log_path.exists() and "falling back to XLA" in log_path.read_text(encoding="utf-8"):
        meta["effective_kernel_backend"]="XLA fallback (requested CUDA legacy kernel unavailable for this GPU)"
    target_sequence = "".join((ROOT / "targets/TL1A/construct.fasta").read_text().splitlines()[1:])
    parser = PDBParser(QUIET=True)
    native = parser.get_structure("native", ROOT / "targets/TL1A/observed.pdb")
    manifest = json.loads((ROOT / "targets/manifest.json").read_text())
    core = next(t["beta_core_uniprot"] for t in manifest["targets"] if t["target_id"]=="TL1A")
    for key, meaning, units, direction in [
        ("ipsae_10", "pinned ipSAE PAE/distance 10 A, maximum binder-target pair only", "score", "higher"),
        ("ipsae_15", "pinned ipSAE PAE/distance 15 A, maximum binder-target pair only", "score", "higher"),
        ("binder_plddt", "mean binder pLDDT, original AF2 0-100 divided by 100", "fraction", "higher"),
        ("binder_CA_rmsd", "generated-pose agreement after symmetry-aware experimental beta-core alignment", "angstrom", "lower"),
        ("binder_fold_CA_rmsd", "binder-only aligned fold RMSD to generated backbone", "angstrom", "lower"),
        ("target_beta_CA_rmsd", "experimental beta-core RMSD for chosen target-chain permutation", "angstrom", "lower"),
        ("native_all_chain_clash_pairs2", "predicted binder / ALL native target heavy-atom pairs below 2 A", "count", "lower"),
        ("predicted_clash_pairs2", "binder-target heavy-atom pairs below 2 A in predicted complex", "count", "lower"),
        ("glycan_ND2_min_distance", "binder heavy-atom minimum distance to six native N133/N229 ND2 anchors", "angstrom", "higher"),
        ("binder_buried_sasa", "binder SASA lost upon binding, Shrake-Rupley 100 points, 1.4 A probe", "angstrom^2", "higher"),
        ("boltz_pose_CA_rmsd", "AF2 vs Boltz-2 interface pose RMSD, allowing homotrimer relabelling", "angstrom", "lower")]:
        dataset["metric_definitions"][namespace+"."+key] = dict(model="AF2-Multimer-v3 / ipSAE or geometry",
            version=f"ColabFold {meta['package_version']}; model {meta['model_order'][0]} multimer v3; ipSAE 6174cf9",
            meaning=meaning+"; NOT official ESMFold2 score", units=units, direction=direction)
    processed,missing = [],[]
    for row in dataset["candidates"]:
        if row["id"] not in meta["candidate_ids"]:
            continue
        matches = list(predictions.glob(row["id"]+"_unrelaxed_rank_001_*.pdb"))
        if not matches:
            missing.append(row["id"])
            continue
        if len(matches)!=1:
            raise ValueError("ambiguous actual AF2 prediction")
        source_pdb = matches[0]
        score_file = source_pdb.with_name(source_pdb.name.replace("_unrelaxed_","_scores_")).with_suffix(".json")
        confidence = json.loads(score_file.read_text())
        predicted = parser.get_structure(row["id"], source_pdb)
        chains = list(predicted[0])
        sequences = {c.id:"".join(seq1(r.resname) for r in c) for c in chains}
        target_ids = [c.id for c in chains if sequences[c.id]==target_sequence]
        binder_ids = [c.id for c in chains if sequences[c.id]==row["sequence"]]
        if len(target_ids)!=3 or len(binder_ids)!=1 or len(chains)!=4:
            raise ValueError("AF2 did not produce the exact full official trimer and binder")
        binder_id = binder_ids[0]
        expected = 540+len(row["sequence"])
        if np.array(confidence["pae"]).shape!=(expected,expected):
            raise ValueError("AF2 PAE indexing mismatch")
        # Run public ipSAE on unchanged source PDB/JSON in an isolated directory.
        workspace = predictions / "ipsae" / row["id"]
        workspace.mkdir(parents=True,exist_ok=True)
        for source in (source_pdb,score_file):
            shutil.copyfile(source, workspace/source.name)
        values, pair_scores = {}, {}
        for cutoff in (10,15):
            values[f"af2.ipsae_{cutoff}"],pair_scores[str(cutoff)] = ipsae(
                workspace/score_file.name, workspace/source_pdb.name, cutoff, binder_id,"".join(target_ids))
        # Normalize chain/residue labels in memory for shared alignment code;
        # original output files and predictor scores remain untouched.
        target_chains = [predicted[0][c] for c in target_ids]
        binder = predicted[0][binder_id]
        for c in chains:
            predicted[0].detach_child(c.id)
        for c,name in zip(target_chains,"ABC"):
            residues=list(c)
            for r in residues:
                c.detach_child(r.id)
            c.id=name
            for i,r in enumerate(residues,1):
                r.id=(" ",i," ")
                c.add(r)
            predicted[0].add(c)
        binder.id="D"
        predicted[0].add(binder)
        designed=parser.get_structure("design",ROOT/row["structures"][0]["path"])[0]["D"]
        _,fit,permutation,best,trials=align_prediction(native,predicted,designed,"ABC","D",core)
        bca=np.array([r["CA"].coord for r in binder])
        dca=np.array([r["CA"].coord for r in designed])
        fold=Superimposer()
        fold.set_atoms([r["CA"] for r in designed],[r["CA"] for r in binder])
        bx=np.array([a.coord for a in binder.get_atoms() if a.element!="H"])
        tx=np.array([a.coord for c in "ABC" for a in predicted[0][c].get_atoms() if a.element!="H"])
        nx=np.array([a.coord for a in native.get_atoms() if a.element!="H"])
        gx=np.array([native[0][c][n]["ND2"].coord for c in "ABC" for n in (133,229)])
        sr=ShrakeRupley(n_points=100)
        sr.compute(predicted,level="A")
        bound_area=sum(a.sasa for a in binder.get_atoms())
        alone=binder.copy()
        sr.compute(alone,level="A")
        values.update({"af2.binder_plddt":float(np.mean([a.bfactor for r in binder for a in r if a.name=="CA"]))/100,
            "af2.binder_CA_rmsd":float(np.sqrt(np.mean(np.sum((bca-dca)**2,axis=1)))),
            "af2.binder_fold_CA_rmsd":float(fold.rms),"af2.target_beta_CA_rmsd":float(fit.rms),
            "af2.native_all_chain_clash_pairs2":int((np.linalg.norm(bx[:,None]-nx,axis=-1)<2).sum()),
            "af2.predicted_clash_pairs2":int((np.linalg.norm(bx[:,None]-tx,axis=-1)<2).sum()),
            "af2.glycan_ND2_min_distance":float(np.linalg.norm(bx[:,None]-gx,axis=-1).min()),
            "af2.binder_buried_sasa":float(sum(a.sasa for a in alone.get_atoms())-bound_area)})
        boltz_structures=[s for s in row["structures"] if s["role"]=="boltz_trimer_deepmsa_predicted_complex"]
        if boltz_structures:
            boltz=MMCIFParser(QUIET=True).get_structure("boltz",ROOT/boltz_structures[0]["path"])
            align_prediction(native,boltz,designed,"ABC","D",core)
            cross=predicted.copy()
            align_prediction(native,cross,boltz[0]["D"],"ABC","D",core)
            values["af2.boltz_pose_CA_rmsd"]=float(np.sqrt(np.mean(np.sum((np.array([r["CA"].coord for r in cross[0]["D"]])-np.array([r["CA"].coord for r in boltz[0]["D"]]))**2,axis=1))))
        row["metrics"].update({metric_name(k):v for k,v in values.items()})
        destination=out/"predictions"/namespace/row["id"]
        destination.mkdir(parents=True,exist_ok=True)
        for source in (source_pdb,score_file):
            shutil.copyfile(source,destination/source.name)
        record=dict(path=(destination/source_pdb.name).relative_to(ROOT).as_posix(),role=namespace+"_predicted_complex",chain_id=binder_id)
        row["structures"]=[s for s in row["structures"] if s["path"]!=record["path"]]+[record]
        row["metadata"][namespace]=dict(run=predictions.relative_to(ROOT).as_posix(),ipsae_pairs=pair_scores,
            symmetry_alignment_trials=trials,target_chain_permutation=list(permutation),
            confidence_summary={k:v for k,v in confidence.items() if k not in ("pae","plddt")},
            original_score_sha256=hashlib.sha256(score_file.read_bytes()).hexdigest(),official_equivalence=False)
        processed.append(row["id"])
    save_candidates(dataset,args.candidates)
    save_csv(dataset,out/"candidates.csv")
    report_path=out/(namespace+"_evaluation.json")
    history=json.loads(report_path.read_text()) if report_path.exists() else {"runs":{}}
    if "runs" not in history:
        history={"runs":{history["run"].get("candidate_ids",["previous"])[0]:history}}
    history["runs"][predictions.relative_to(ROOT).as_posix()]=dict(run=meta,processed=processed,missing=missing)
    report_path.write_text(json.dumps(history,indent=2)+"\n")
    print(json.dumps(dict(processed=processed,missing=missing),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidates")
    p.add_argument("predictions")
    p.add_argument('--namespace',default='af2',help='preserve additional checkpoints/seeds separately, e.g. af2_model3')
    main(p.parse_args())
