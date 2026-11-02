"""One targeted terminal truncation of a newly cross-supported design core.

Uses PARENT predictions to propose a rigid-body corrected design pose. Scores
are NEVER inherited. The new sequence requires fresh, unconditioned prediction.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.SVDSuperimposer import SVDSuperimposer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from changjiang_select.data import load_candidates,save_candidates,save_csv
from scripts.evaluate_designs import geometry,sequence_metrics
from scripts.import_boltz import align_prediction


def main(args):
    path=(ROOT/args.candidates).resolve(); data=load_candidates(path)
    if data['synthetic']: raise ValueError('requires real parent predictions')
    parent=next(r for r in data['candidates'] if r['id']==args.id)
    if any(r['metadata'].get('generation',{}).get('consensus_parent')==parent['id'] for r in data['candidates']):
        raise ValueError('only one terminal-core follow-up per parent, not repeated family expansion')
    if min(parent['metrics'].get(k,0) for k in ('af2.ipsae_10','boltz_trimer_deepmsa.ipsae_10'))<.4:
        raise ValueError('requires interface support from BOTH architectures, not generation confidence')
    start,end=args.start,args.end
    if not 1<=start<=end<=len(parent['sequence']) or not 40<=end-start+1<=100:
        raise ValueError('invalid mature core range')
    parser=PDBParser(QUIET=True)
    native=parser.get_structure('native',ROOT/'targets/TL1A/observed.pdb')
    design=parser.get_structure('design',ROOT/parent['structures'][0]['path'])
    core=next(t['beta_core_uniprot'] for t in json.loads((ROOT/'targets/manifest.json').read_text())['targets'] if t['target_id']=='TL1A')
    poses={}
    for role in ('boltz_trimer_deepmsa_predicted_complex','af2_predicted_complex'):
        record=next(s for s in parent['structures'] if s['role']==role)
        file=ROOT/record['path']
        p=MMCIFParser(QUIET=True) if file.suffix=='.cif' else parser
        pose=p.get_structure(role,file)
        for chain in pose[0]:
            residues=list(chain)
            for r in residues: chain.detach_child(r.id)
            for i,r in enumerate(residues,1):
                r.id=(' ',i,' '); chain.add(r)
        reference=design[0]['D'] if not poses else next(iter(poses.values()))[0]['D']
        align_prediction(native,pose,reference,'ABC','D',core)
        poses[role]=pose
    first,second=[np.array([r['CA'].coord for r in p[0]['D']])[start-1:end] for p in poses.values()]
    core_pose=float(np.sqrt(np.mean(np.sum((first-second)**2,-1))))
    fit=SVDSuperimposer(); fit.set(first,second); fit.run()
    core_fold=float(fit.get_rms())
    if core_pose>2 or core_fold>2:
        raise ValueError('the retained core itself is not cross-supported within 2 A')
    source=ROOT/parent['metadata']['refinement']['retained_allatom_complex']
    allatom=parser.get_structure('allatom',source)[0]['D']
    retained=list(allatom)[start-1:end]
    fit.set((first+second)/2,np.array([r['CA'].coord for r in retained]));fit.run()
    rotation,translation=fit.get_rotran()
    binder=allatom.copy()
    for r in list(binder): binder.detach_child(r.id)
    for i,old in enumerate(retained,1):
        r=old.copy();r.id=(' ',i,' ')
        for a in list(r):
            if a.element=='H' or a.name=='OXT': r.detach_child(a.id)
            else: a.transform(rotation,translation)
        binder.add(r)
    sequence=parent['sequence'][start-1:end]
    name=parent['id'].removesuffix('_gen')+f'_trim{len(sequence)}'
    pose=copy.deepcopy(native);pose[0].add(binder)
    allatom_file=path.parent/'generation'/f'{name}_consensus.pdb'
    writer=PDBIO();writer.set_structure(pose);writer.save(str(allatom_file))
    for r in binder:
        for a in list(r):
            if a.name not in ('N','CA','C','O'):r.detach_child(a.id)
    file=path.parent/'poses'/f'{name}_design.pdb'
    writer.set_structure(pose);writer.save(str(file))
    canonical,metrics,epitope=geometry(file,native,'D')
    bx=np.array([a.coord for a in canonical[0]['D'].get_atoms()])
    nx=np.array([a.coord for a in native.get_atoms()])
    metrics['geometry.native_all_chain_BB_clash_pairs2']=int((np.linalg.norm(bx[:,None]-nx,axis=-1)<2).sum())
    metrics.update(sequence_metrics(sequence));metrics['official.ipsae']=None
    metadata=dict(status='generated_not_independently_validated',
        generation=dict(model='cross-supported core truncation and rigid-body consensus docking',
            stage='consensus_truncation',consensus_parent=parent['id'],parent_interval_inclusive=[start,end],
            source_allatom_complex=allatom_file.relative_to(ROOT).as_posix(),
            parent_core_cross_pose_A=core_pose,parent_core_cross_fold_A=core_fold,
            consensus_fit_CA_rmsd_A=float(fit.get_rms()),
            rotation=rotation.tolist(),translation=translation.tolist(),
            conditioning='parent AF2/Boltz core consensus only; no NEW sequence prediction yet',
            rationale='terminal residues dominate parent cross-model error; keep all core cysteines and interface residues',
            scores_inherited=False),coordinates='native ABC + parent-derived, rigidly moved and truncated backbone; NOT a new sequence refold')
    data['candidates'].append(dict(id=name,sequence=sequence,target_id='TL1A',design_type='minibinder',
        source_model='BoltzGen + parent-prediction-guided core truncation',source_version='local-v1',
        run_id='tl1a_consensus_trim_20261102',structural_group=parent['structural_group'],
        epitope_residues=epitope,metrics=metrics,structures=[dict(path=file.relative_to(ROOT).as_posix(),
        chain_id='D',role='generated_backbone_complex_not_refolded')],metadata=metadata))
    save_candidates(data,path);save_csv(data,path.with_suffix('.csv'))
    (path.parent/'candidates.fasta').write_text(''.join(f">{r['id']} generated_unvalidated\n{r['sequence']}\n" for r in data['candidates']))
    print(json.dumps(dict(id=name,sequence=sequence,generation=metadata['generation'],geometry=metrics),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates');p.add_argument('--id',required=True)
    p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True)
    main(p.parse_args())
