"""Place a generic source scaffold as a folding control, not a binding design."""
import argparse
import copy
import json
from pathlib import Path
import sys

import Bio
from Bio.PDB import PDBParser,PDBIO,Superimposer
from Bio.SeqUtils import seq1

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from changjiang_select.data import save_candidates,save_csv
from scripts.evaluate_designs import geometry,sequence_metrics


def main(args):
    data=json.loads((ROOT/args.candidates).read_text())
    parent=next(r for r in data['candidates'] if r['id']==args.id)
    if any(k.startswith(('af2.','boltz')) and v is not None for r in data['candidates'] if r['structural_group']==parent['structural_group'] for k,v in r['metrics'].items()):
        raise ValueError('prepare controls before independent screening')
    out=ROOT/args.out
    if out.exists():raise ValueError('fresh control directory required')
    (out/'poses').mkdir(parents=True)
    p=PDBParser(QUIET=True)
    source=p.get_structure('scaffold',ROOT/args.scaffold)
    binder=source[0]['D'];pose=p.get_structure('parent',ROOT/parent['structures'][0]['path'])
    fit=Superimposer();fit.set_atoms([r['CA'] for r in pose[0]['D']],[r['CA'] for r in binder]);fit.apply(list(binder.get_atoms()))
    for r in binder:
        for a in list(r):
            if a.name not in ('N','CA','C','O'):r.detach_child(a.id)
    pose[0].detach_child('D');pose[0].add(binder.copy())
    file=out/'poses/scaffold_control_design.pdb';writer=PDBIO();writer.set_structure(pose);writer.save(str(file))
    native=p.get_structure('native',ROOT/'targets/TL1A/observed.pdb')
    _,metrics,epitope=geometry(file,native,'D')
    # This source pose is only for fold-reference alignment, not admission.
    metrics['geometry.native_all_chain_BB_clash_pairs2']=None
    sequence=''.join(seq1(r.resname) for r in binder);metrics.update(sequence_metrics(sequence));metrics['official.ipsae']=None
    row=copy.deepcopy(parent);row.update(id='tl1a_a3d_scaffold_control',sequence=sequence,metrics=metrics,epitope_residues=epitope,
        source_model='generic experimental scaffold folding control; not a TL1A binder design',run_id='a3d_source_control',
        structures=[dict(path=file.relative_to(ROOT).as_posix(),chain_id='D',role='generic_scaffold_fold_reference_in_target_frame')],
        metadata=dict(status='folding_control_not_binding_candidate',source_scaffold=args.scaffold,source_url='https://www.rcsb.org/structure/2A3D',
            backbone_fit_to_parent_A=float(fit.rms),purpose='separate scaffold predictability from interface sequence-design damage; no known TL1A binder used'))
    control=dict(schema_version=1,synthetic=False,metric_definitions={k:v for k,v in data['metric_definitions'].items() if k in metrics},candidates=[row])
    save_candidates(control,out/'candidates.json');save_csv(control,out/'candidates.csv')
    print(json.dumps(dict(control=row['id'],parent=args.id,source_sequence=sequence,backbone_fit_A=float(fit.rms))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidates');p.add_argument('--id',required=True)
    p.add_argument('--scaffold',default='targets/scaffolds/a3d73.pdb');p.add_argument('--out',required=True)
    main(p.parse_args())
