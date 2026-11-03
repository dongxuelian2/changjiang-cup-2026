"""Score one frozen sequence on its design pose and real-prediction pose decoys.

All inputs use the SAME reference fold and native ABC target. Only rigid binder
pose changes. Conditional sequence NLL is NOT P(pose|sequence) or binding energy.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def prepare(args):
    import numpy as np
    from Bio.PDB import PDBParser,MMCIFParser,PDBIO,Superimposer
    from scripts.import_boltz import align_prediction
    path=ROOT/args.candidates;data=json.loads(path.read_text())
    if data['synthetic']:raise ValueError('requires real independently predicted poses')
    row=next(r for r in data['candidates'] if r['id']==args.id)
    out=ROOT/args.out
    if out.exists():raise ValueError('choose a fresh audit directory')
    out.mkdir(parents=True)
    pdb=PDBParser(QUIET=True);cif=MMCIFParser(QUIET=True)
    reference=pdb.get_structure('design',ROOT/row['structures'][0]['path'])
    native=pdb.get_structure('native',ROOT/'targets/TL1A/observed.pdb')
    manifest=json.loads((ROOT/'targets/manifest.json').read_text())
    core=next(t['beta_core_uniprot'] for t in manifest['targets'] if t['target_id']=='TL1A')
    states=[('design',None)]
    for role in ('af2_predicted_complex','boltz_trimer_deepmsa_predicted_complex'):
        record=next(r for r in row['structures'] if r['role']==role)
        file=ROOT/record['path'];pred=(cif if file.suffix=='.cif' else pdb).get_structure('prediction',file)
        align_prediction(native,pred,reference[0]['D'],'ABC','D',core)
        states.append((role,pred[0]['D']))
    meta=dict(synthetic=False,candidate_id=row['id'],sequence=row['sequence'],parent_candidates=args.candidates,
        reference_backbone=row['structures'][0]['path'],
        design_positions=row['metadata']['generation']['design_positions'],
        interpretation='counterfactual rigid-pose audit of a frozen sequence; conditional NLL is not pose probability, affinity, or independent validation',
        target='exact observed native ABC 88-251',seed=args.seed,states=[])
    tx=np.array([a.coord for a in native.get_atoms() if a.element!='H'])
    gx=np.array([native[0][c][n]['ND2'].coord for c in 'ABC' for n in (133,229)])
    for name,predicted in states:
        pose=copy.deepcopy(reference);fold_fit=0.
        if predicted is not None:
            fit=Superimposer();fit.set_atoms([r['CA'] for r in predicted],[r['CA'] for r in pose[0]['D']])
            fit.apply(list(pose[0]['D'].get_atoms()));fold_fit=float(fit.rms)
        file=out/f'{name}.pdb';writer=PDBIO();writer.set_structure(pose);writer.save(str(file))
        bb=np.array([a.coord for a in pose[0]['D'].get_atoms()])
        meta['states'].append(dict(name=name,input=file.relative_to(ROOT).as_posix(),
            counterfactual_rigid_pose=predicted is not None,
            predictor_fold_fit_residual_A=fold_fit,
            native_BB_clash_pairs2=int((np.linalg.norm(bb[:,None]-tx,axis=-1)<2).sum()),
            glycan_ND2_BB_distance_A=float(np.linalg.norm(bb[:,None]-gx,axis=-1).min())))
    (out/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta['states'],indent=2))


def score(args):
    import numpy as np
    import torch
    if not torch.cuda.is_available():raise RuntimeError('use the tested CUDA path')
    out=ROOT/args.out;meta=json.loads((out/'inputs.json').read_text())
    source=ROOT/'build/models/LigandMPNN-main';orders=[]
    meta['model']='LigandMPNN v_32_010_25; native fixed sidechain context enabled'
    meta['autoregressive_score']=True;meta['device']=torch.cuda.get_device_name(0)
    for state in meta['states']:
        result_dir=out/(state['name']+'_score')
        command=[sys.executable,'-u',str(source/'score.py'),'--model_type','ligand_mpnn',
            '--checkpoint_ligand_mpnn',str(source/'model_params/ligandmpnn_v_32_010_25.pt'),
            '--pdb_path',str(ROOT/state['input']),'--chains_to_design','D',
            '--redesigned_residues',' '.join(f'D{i}' for i in meta['design_positions']),
            '--ligand_mpnn_use_side_chain_context','1','--use_sequence','1',
            '--autoregressive_score','1','--single_aa_score','0',
            '--batch_size','1','--number_of_batches','1','--seed',str(meta['seed']),
            '--verbose','0','--out_folder',str(result_dir)]
        state['command']=command;started=time.monotonic()
        with (out/(state['name']+'.log')).open('w') as log:
            subprocess.run(command,cwd=source,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4'},
                stdout=log,stderr=subprocess.STDOUT,check=True)
        stats=torch.load(next(result_dir.glob('*.pt')),map_location='cpu',weights_only=False)
        mask=stats['mask']*stats['chain_mask'];seq=stats['native_sequence']
        if int(mask.sum())!=len(meta['design_positions']):raise ValueError('score mask changed')
        nll=-np.take_along_axis(stats['log_probs'][0],seq[:,None],axis=-1).ravel()
        state.update(conditional_sequence_NLL=float((nll*mask).sum()/mask.sum()),
            per_position_NLL={stats['residue_names'][i]:float(nll[i]) for i in np.flatnonzero(mask)},
            elapsed_seconds=time.monotonic()-started,
            retained_stats=next(result_dir.glob('*.pt')).relative_to(ROOT).as_posix())
        orders.append(stats['decoding_order'])
    meta['identical_decoding_order_across_poses']=all(np.array_equal(orders[0],o) for o in orders)
    if not meta['identical_decoding_order_across_poses']:raise ValueError('not a matched decoder-order comparison')
    (out/'pose_likelihood.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps([(s['name'],s['conditional_sequence_NLL']) for s in meta['states']],indent=2))


def curate(args):
    out=ROOT/args.out;destination=ROOT/args.destination
    destination.mkdir(parents=True,exist_ok=True)
    meta=json.loads((out/'pose_likelihood.json').read_text())
    for state in meta['states']:
        source=ROOT/state['input'];file=destination/source.name;shutil.copyfile(source,file)
        state['retained_input']=file.relative_to(ROOT).as_posix()
        source=ROOT/state['retained_stats'];file=destination/source.name;shutil.copyfile(source,file)
        state['original_stats']=state['retained_stats'];state['retained_stats']=file.relative_to(ROOT).as_posix()
        shutil.copyfile(out/(state['name']+'.log'),destination/(state['name']+'.log'))
    (destination.parent.parent/'pose_likelihood.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','score','curate']);p.add_argument('--candidates');p.add_argument('--id')
    p.add_argument('--destination')
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=20261108)
    args=p.parse_args()
    if args.stage=='prepare':prepare(args)
    elif args.stage=='score':score(args)
    else:curate(args)
