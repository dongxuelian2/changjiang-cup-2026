"""Import actual BoltzGen atom14 and inverse-folding outputs, never placeholders.

Preserve original CIF/NPZ; derive canonical ABC target / D binder backbone poses
for the existing independent prediction and portfolio entry points.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import sys

import numpy as np
import Bio
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Superimposer
from Bio.SeqUtils import seq1, seq3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from changjiang_select.data import load_candidates, save_candidates, save_csv
from scripts.evaluate_designs import geometry, sequence_metrics


def retained_filename(cif, raw, stage):
    # A single IF sequence is emitted with EXACTLY the atom14 parent's name.
    # Keeping only basename would replace its genuine sidechains with IF
    # placeholders and invalidate the raw provenance hash/sequence.
    return 'inverse_folding_'+cif.name if stage == 'inverse_folding' and cif.name == raw.name else cif.name


def fixed_binder_residues(specification, length, spec_directory=None):
    """Audit a fixed-length mixed design/AA specification, not inferred output AAs."""
    proteins=[e['protein'] for e in specification['entities'] if 'protein' in e
              and any(c.isdigit() for c in str(e['protein']['sequence']))]
    if not proteins and spec_directory is not None:
        files=[e['file'] for e in specification['entities'] if 'file' in e and e['file'].get('design')]
        if len(files)!=1:raise ValueError('require one explicitly designed scaffold file')
        entry=files[0]
        if entry.get('design_insertions') or entry.get('not_design'):
            raise ValueError('scaffold audit currently requires fixed length and explicit design positions')
        source=(Path(spec_directory)/entry['path']).resolve()
        parser=MMCIFParser(QUIET=True) if source.suffix=='.cif' else PDBParser(QUIET=True)
        chains=list(parser.get_structure('scaffold',source)[0])
        if len(chains)!=1 or len(chains[0])!=length:
            raise ValueError('scaffold audit requires a prepared single-chain input of exact binder length')
        chain=chains[0];variable=set()
        for item in entry['design']:
            definition=item['chain']
            if definition['id']!=chain.id:raise ValueError('scaffold design-chain mismatch')
            if 'res_index' not in definition:variable.update(range(1,length+1))
            else:
                for token in str(definition['res_index']).split(','):
                    ends=token.split('..')
                    variable.update(range(int(ends[0]),int(ends[-1])+1))
        if not variable or min(variable)<1 or max(variable)>length:raise ValueError('invalid scaffold design positions')
        return {i+1:seq1(r.resname) for i,r in enumerate(chain) if i+1 not in variable}
    if len(proteins)!=1:
        raise ValueError('require one designed protein entity')
    sequence=str(proteins[0]['sequence']).upper()
    if not any(c.isalpha() for c in sequence):
        return {}
    if '..' in sequence:
        raise ValueError('mixed fixed-AA audit requires fixed design lengths')
    fixed={}; position=0
    for token in re.findall(r'\d+|[A-Z]',sequence):
        if token.isdigit(): position+=int(token)
        else:
            position+=1;fixed[position]=token
    if position!=length:
        raise ValueError('mixed design specification length does not match actual binder')
    return fixed


def main(args):
    run = (ROOT / args.run).resolve()
    meta = json.loads((run / "run.json").read_text())
    import yaml
    specification=yaml.safe_load((ROOT/meta['spec']).read_text())
    if meta["synthetic"] or meta["model"] != "BoltzGen":
        raise ValueError("requires genuine BoltzGen provenance")
    out = (ROOT / args.out).resolve()
    existing = load_candidates(out / "candidates.json") if args.append else None
    if (out / "candidates.json").exists() and not args.append:
        raise ValueError("import into a fresh dataset; never overwrite independent evaluations")
    for folder in ("poses", "generation"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    previous = json.loads((ROOT / "results/tl1a_20261101/candidates.json").read_text())
    defs = {k: v for k, v in previous["metric_definitions"].items()
            if k.startswith(("sequence.", "geometry.")) or k == "official.ipsae"}
    seq_version=Bio.__version__+' / local-v1'
    if existing and any(v['version']!=seq_version for k,v in existing['metric_definitions'].items()
                        if k.startswith('sequence.')):
        raise ValueError('append using the original sequence-metric BioPython environment, or import into a separate dataset')
    for key,spec in defs.items():
        if key.startswith('sequence.'):spec['version']=seq_version
    meta['import_environment']=dict(biopython=Bio.__version__,numpy=np.__version__)
    defs["geometry.native_all_chain_BB_clash_pairs2"] = dict(
        model="observed 2RE9 + BoltzGen backbone", version="local-v1",
        meaning="designed binder BACKBONE vs ALL observed native trimer heavy atoms, distance <2 A",
        units="count", direction="lower")
    if existing:
        defs.update(existing["metric_definitions"])
    reference = PDBParser(QUIET=True).get_structure("native", ROOT / "targets/TL1A/observed.pdb")
    expected = "".join((ROOT / "targets/TL1A/construct.fasta").read_text().splitlines()[1:])[16:]
    raw_files = sorted((run / "intermediate_designs").glob("*.cif"))
    raw_files = [f for f in raw_files if not f.name.endswith("_native.cif")]
    files = [(f, f, "atom14") for f in raw_files]
    if not args.raw_only:
        if meta["status"] != "generated_not_independently_validated":
            raise ValueError("wait for the actual inverse-folding stage to finish")
        for f in sorted((run / "intermediate_designs_inverse_folded").glob("*.cif")):
            if f.name.endswith("_native.cif"):
                continue
            parents = [r for r in raw_files if f.stem.startswith(r.stem+"_") or f.stem == r.stem]
            if not parents:
                raise ValueError(f"unknown source backbone: {f.name}")
            files.append((f, max(parents, key=lambda r: len(r.stem)), "inverse_folding"))
    rows = existing["candidates"] if existing else []
    seen = {r["sequence"]:r for r in rows}
    parser = MMCIFParser(QUIET=True)
    for cif, raw, stage in files:
        pose = parser.get_structure(cif.stem, cif)
        chains = list(pose[0])
        sequences = {c.id: "".join(seq1(r.resname) for r in c) for c in chains}
        targets = [c for c in chains if sequences[c.id] == expected]
        binders = [c for c in chains if sequences[c.id] != expected]
        if len(targets) != 3 or len(binders) != 1:
            raise ValueError(f"require intact, sequence-correct trimer and one binder: {cif}")
        binder = binders[0]
        sequence = sequences[binder.id]
        if not 40 <= len(sequence) <= 100 or not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
            if stage == 'atom14' and args.skip_invalid_raw:
                meta.setdefault('rejected_raw_outputs',[]).append(dict(file=cif.name,reason='noncanonical or invalid sequence; not a candidate'))
                for artifact in (cif,cif.with_suffix('.npz')):
                    shutil.copyfile(artifact,out/'generation'/('rejected_'+artifact.name))
                continue
            raise ValueError(f"invalid generated sequence: {cif}")
        masks = np.load(cif.with_suffix(".npz"))
        fixed=fixed_binder_residues(specification,len(sequence),(ROOT/meta['spec']).parent)
        if any(sequence[p-1]!=aa for p,aa in fixed.items()):
            raise ValueError('emitted sequence changed a specified fixed binder residue')
        if int(masks["design_mask"].sum()) != len(sequence)-len(fixed):
            raise ValueError("source design mask disagrees with designed chain")
        original_binder, original_targets = binder.id, [c.id for c in targets]
        source = dict(file=cif.relative_to(ROOT).as_posix(),
                      sha256=hashlib.sha256(cif.read_bytes()).hexdigest(), stage=stage,
                      backbone=raw.stem, original_binder_chain=original_binder,
                      original_target_chains=original_targets)
        if fixed:
            source['fixed_binder_AAs']=[dict(position=p,aa=aa) for p,aa in fixed.items()]
            source['designable_binder_residues']=len(sequence)-len(fixed)
        retained = retained_filename(cif, raw, stage)
        if retained != cif.name:
            source['retained_file'] = (out/'generation'/retained).relative_to(ROOT).as_posix()
        # Audit the ACTUAL emitted conditioning mask, including chain order and
        # the unresolved N-terminal offset, rather than trusting YAML labels.
        token_residues = [(c.id,i) for c in chains for i,_ in enumerate(c)]
        if len(token_residues) != len(masks["binding_type"]):
            raise ValueError("conditioning-mask residue order mismatch")
        if not np.array_equal(masks["design_mask"].astype(bool),
                              np.array([c == original_binder and i+1 not in fixed for c,i in token_residues])):
            raise ValueError("emitted design mask does not match the verified binder chain")
        source["binding_sites_uniprot"] = [
            f"{'ABC'[original_targets.index(c)]}:{i+88}"
            for (c,i),kind in zip(token_residues,masks["binding_type"])
            if kind == 1 and c in original_targets]
        if sequence in seen:
            metadata=seen[sequence]["metadata"]
            if source != metadata["generation"] and source not in metadata.get("duplicate_outputs", []):
                metadata.setdefault("duplicate_outputs", []).append(source)
            continue
        # Upstream writer renames chains A/B/C/D by order. Restore canonical
        # target ABC and designed D by verified sequence, not presumed labels.
        for c in chains:
            pose[0].detach_child(c.id)
        for c, name in zip(targets, "ABC"):
            c.id = name
            residues = list(c)
            for r in residues:
                c.detach_child(r.id)
            for r, n in zip(residues, range(88, 252)):
                r.id = (" ", n, " ")
                c.add(r)
            pose[0].add(c)
        binder.id = "D"
        for r in binder:
            for atom in list(r):
                if atom.name not in ("N", "CA", "C", "O"):
                    r.detach_child(atom.id)
        pose[0].add(binder)
        fit = Superimposer()
        fit.set_atoms([r["CA"] for c in "ABC" for r in reference[0][c]],
                      [r["CA"] for c in "ABC" for r in pose[0][c]])
        fit.apply(list(pose.get_atoms()))
        backbone_index = raw.stem.rsplit("_", 1)[-1]
        suffix = "gen" if stage == "atom14" else "if"+cif.stem[len(raw.stem):].strip("_")
        name = f"{args.label}_{backbone_index}_{suffix}"
        temporary = out / "poses" / (name+"_source.pdb")
        writer = PDBIO()
        writer.set_structure(pose)
        writer.save(str(temporary))
        canonical, metrics, epitope = geometry(temporary, reference, "D")
        temporary.unlink()
        inherited=None
        group=f"{args.label}_backbone_{backbone_index}"
        if args.inherit_refined_backbones and stage=='inverse_folding':
            siblings=[r for r in rows if r['structural_group']==group]
            parents=[r for r in siblings if 'refinement' in r['metadata']]
            if parents:
                if len(parents)!=1 or any(k.startswith(('af2.','boltz')) and v is not None
                        for r in siblings for k,v in r['metrics'].items()):
                    raise ValueError('inherit local repair only within a new, independently unscreened family')
                parent=parents[0];parser_pdb=PDBParser(QUIET=True)
                original=next(s for s in parent['structures'] if s['role']=='unrefined_generated_backbone_complex')
                before=parser_pdb.get_structure('original',ROOT/original['path'])
                bx=np.array([r['CA'].coord for r in canonical[0]['D']])
                px=np.array([r['CA'].coord for r in before[0]['D']])
                if bx.shape!=px.shape:raise ValueError('IF backbone length changed')
                rms=float(np.sqrt(np.mean(np.sum((bx-px)**2,axis=-1))))
                if rms>.02:raise ValueError('IF backbone differs from its atom14 source; cannot inherit repair')
                canonical=parser_pdb.get_structure('refined',ROOT/parent['structures'][0]['path'])
                for residue,aa in zip(canonical[0]['D'],sequence):
                    if any(a.name not in ('N','CA','C','O') for a in residue):
                        raise ValueError('thread only backbone coordinates, never another sequence sidechains')
                    residue.resname=seq3(aa).upper()
                writer.set_structure(canonical);writer.save(str(temporary))
                canonical,metrics,epitope=geometry(temporary,reference,'D');temporary.unlink()
                inherited=dict(parent_id=parent['id'],original_backbone_CA_rmsd=rms,
                               refinement=parent['metadata']['refinement'])
        bxyz = np.array([a.coord for a in canonical[0]["D"].get_atoms()])
        txyz = np.array([a.coord for c in "ABC" for a in reference[0][c].get_atoms()])
        metrics["geometry.native_all_chain_BB_clash_pairs2"] = int((np.linalg.norm(bxyz[:, None]-txyz, axis=-1)<2).sum())
        metrics.update(sequence_metrics(sequence))
        metrics["official.ipsae"] = None
        destination = out / "poses" / (name+"_design.pdb")
        writer.set_structure(canonical)
        writer.save(str(destination))
        for f in (cif, cif.with_suffix(".npz")):
            destination_file = out/'generation'/Path(retained).with_suffix(f.suffix)
            if destination_file.exists() and destination_file.read_bytes() != f.read_bytes():
                raise ValueError(f'different generation artifacts share {destination_file}; preserve both in separate runs')
            shutil.copyfile(f, destination_file)
        row = dict(id=name, sequence=sequence, target_id="TL1A", design_type="minibinder",
                   source_model="BoltzGen" if stage == "atom14" else "BoltzGen + BoltzGen-IF",
                   source_version="0.3.2", run_id=run.name,
                   structural_group=group,
                   epitope_residues=epitope, metrics=metrics,
                   structures=[dict(path=destination.relative_to(ROOT).as_posix(), chain_id="D",
                                    role="generated_backbone_complex_not_refolded")],
                   metadata=dict(generation=source, status="generated_not_independently_validated",
                                 target_trimer_CA_rmsd=float(fit.rms),
                                 coordinates="native target + generated binder BACKBONE only; no inverse-folded sidechains invented"))
        if inherited:
            row['metadata']['backbone_refinement']=inherited
            row['metadata']['coordinates']='native ABC + locally repaired shared BACKBONE, threaded with actual IF sequence; NOT a refold; original CIF retained'
        rows.append(row)
        seen[sequence] = row
    data = dict(schema_version=1, synthetic=False, metric_definitions=defs, candidates=rows)
    save_candidates(data, out / "candidates.json")
    save_csv(data, out / "candidates.csv")
    (out / "candidates.fasta").write_text("".join(f">{r['id']} {r['source_model']} unvalidated\n{r['sequence']}\n" for r in rows))
    generation_path = out / "generation.json"
    if existing:
        history = json.loads(generation_path.read_text())
        if "runs" not in history:
            history = {"runs": {existing["candidates"][0]["run_id"]: history}}
        history["runs"][run.name] = meta
    else:
        history = meta
    generation_path.write_text(json.dumps(history, indent=2)+"\n")
    print(json.dumps(dict(actual_output_files=len(files), distinct_sequences=len(rows),
                         backbones=len(raw_files), ids=[r["id"] for r in rows]), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--label", default="tl1a_bg_a")
    p.add_argument("--out", default="results/tl1a_boltzgen_20261101")
    p.add_argument("--raw-only", action="store_true", help="inspect real generation outputs before inverse folding finishes")
    p.add_argument("--append", action="store_true", help="preserve existing sequences and independent evaluations while adding a new real run")
    p.add_argument('--skip-invalid-raw',action='store_true',help='retain invalid atom14 outputs as rejected provenance; still import independently emitted valid IF sequences')
    p.add_argument('--inherit-refined-backbones',action='store_true',
                   help='thread new IF outputs onto an existing small local repair only if their original backbone is unchanged and the family has no independent evaluations')
    main(p.parse_args())
