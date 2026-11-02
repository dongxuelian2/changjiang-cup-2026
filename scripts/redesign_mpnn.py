"""Finish a NEW, unscreened backbone's initial sequence batch with soluble MPNN.

Never extend an independently screened family. By default interface residues
stay fixed; optional full-binder sequence design also precedes any evaluation.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import Bio
from Bio import SeqIO
from Bio.PDB import PDBParser,PDBIO
from Bio.SeqUtils import seq1,seq3

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from changjiang_select.data import load_candidates,save_candidates,save_csv
from scripts.evaluate_designs import sequence_metrics


def main(args):
    import torch
    if not args.cpu and not torch.cuda.is_available():raise RuntimeError('CUDA required unless --cpu is explicitly selected')
    path=(ROOT/args.candidates).resolve();data=load_candidates(path)
    if any(v['version']!=Bio.__version__+' / local-v1'
           for k,v in data['metric_definitions'].items() if k.startswith('sequence.')):
        raise ValueError('use the original sequence-metric BioPython environment for this dataset')
    parent=next(r for r in data['candidates'] if r['id']==args.id)
    family=[r for r in data['candidates'] if r['structural_group']==parent['structural_group']]
    if any(k.startswith(('af2.','boltz')) and v is not None
           for r in family for k,v in r['metrics'].items()):
        raise ValueError('do not extend an independently screened family; freeze initial sequence batch first')
    if parent['metrics']['geometry.native_all_chain_BB_clash_pairs2']:
        raise ValueError('repair local backbone sterics before sequence design')
    out=(ROOT/args.out).resolve()
    if out.exists():raise ValueError('choose fresh generation output')
    out.mkdir(parents=True)
    full=ROOT/parent['metadata']['refinement']['retained_allatom_complex']
    structure=PDBParser(QUIET=True).get_structure('interface',full)
    native=PDBParser(QUIET=True).get_structure('native',ROOT/'targets/TL1A/observed.pdb')
    tx=np.array([a.coord for a in native.get_atoms() if a.element!='H'])
    fixed=[i+1 for i,r in enumerate(structure[0]['D']) if np.linalg.norm(
        np.array([a.coord for a in r if a.element!='H'])[:,None]-tx,axis=-1).min()<4.5]
    if args.redesign_interface:fixed=[]
    if len(fixed)>=len(parent['sequence']):raise ValueError('no core positions left to redesign')
    input_pose=ROOT/parent['structures'][0]['path']
    fixed_file=out/'fixed_positions.json'
    fixed_file.write_text(json.dumps({input_pose.stem:{'A':[],'B':[],'C':[],'D':fixed}})+'\n')
    source=ROOT/'build/models/ProteinMPNN';checkpoint=source/'soluble_model_weights/v_48_020.pt'
    command=[sys.executable,str(source/'protein_mpnn_run.py'),
        '--pdb_path',input_pose.as_posix(),'--pdb_path_chains','D','--out_folder',out.as_posix(),
        '--fixed_positions_jsonl',fixed_file.as_posix(),'--num_seq_per_target',str(args.sequences),
        '--batch_size','1','--sampling_temp','0.1','--seed',str(args.seed),
        '--model_name','v_48_020','--path_to_model_weights',(source/'soluble_model_weights').as_posix(),
        '--use_soluble_model','--save_score','1','--omit_AAs','C']
    meta=dict(synthetic=False,model='ProteinMPNN soluble full-binder design' if args.redesign_interface else 'ProteinMPNN soluble core design',parent_id=parent['id'],
        source=json.loads((source/'UPSTREAM.json').read_text()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        fixed_interface_positions=fixed,redesigned_interface=args.redesign_interface,
        interface_distance_A=4.5,seed=args.seed,temperature=.1,
        requested_sequences=args.sequences,torch=torch.__version__,biopython=Bio.__version__,device='CPU' if args.cpu else 'CUDA',command=command,
        status='running',official_equivalence=False)
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    started=time.monotonic()
    env={**os.environ,'OMP_NUM_THREADS':'4','PYTHONIOENCODING':'utf-8'}
    if args.cpu:env['CUDA_VISIBLE_DEVICES']='-1'  # Empty mask is ambiguous on Windows torch 2.0.
    with (out/'generation.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    meta.update(exit_code=result.returncode,elapsed_seconds=time.monotonic()-started)
    if result.returncode:
        meta['status']='failed';(out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
        raise RuntimeError('MPNN generation failed; inspect generation.log')
    fasta=out/'seqs'/f'{input_pose.stem}.fa'
    records=list(SeqIO.parse(fasta,'fasta'))[1:]
    scores=np.load(out/'scores'/f'{input_pose.stem}.npz')['score'].reshape(-1)
    if len(records)!=args.sequences or len(scores)!=len(records):raise ValueError('MPNN output count mismatch')
    destination=path.parent/'generation'/f'mpnn_{parent["id"]}';destination.mkdir(exist_ok=True)
    for f in (fasta,out/'scores'/f'{input_pose.stem}.npz',fixed_file):shutil.copyfile(f,destination/f.name)
    data['metric_definitions']['mpnn.score']=dict(model='ProteinMPNN soluble v_48_020',
        version=meta['source']['revision'],meaning='sample NLL over designable binder positions under the recorded fixed-position constraints; not affinity',units='nats/residue',direction='lower')
    seen={r['sequence'] for r in data['candidates']};added=[]
    for i,(record,score) in enumerate(zip(records,scores),1):
        sequence=str(record.seq)
        if len(sequence)!=len(parent['sequence']) or not set(sequence)<=set('ACDEFGHIKLMNPQRSTVWY'):
            raise ValueError('MPNN did not emit the exact mature binder length/alphabet')
        if any(sequence[p-1]!=parent['sequence'][p-1] for p in fixed):raise ValueError('fixed interface mutated')
        if sequence in seen:continue
        row=copy.deepcopy(parent);row['id']=parent['id'].removesuffix('_gen')+f'_mpnn{i}'
        row['sequence']=sequence;row['source_model']=parent['source_model']+' backbone + soluble ProteinMPNN'
        row['source_version']=meta['source']['revision'];row['run_id']=out.name
        row['metrics']={k:v for k,v in row['metrics'].items() if not k.startswith('afdesign.')}
        row['metrics'].update(sequence_metrics(sequence));row['metrics']['mpnn.score']=float(score)
        backbone=PDBParser(QUIET=True).get_structure('pose',input_pose)
        for r,aa in zip(backbone[0]['D'],sequence):r.resname=seq3(aa).upper()
        output_pose=path.parent/'poses'/f'{row["id"]}_design.pdb'
        writer=PDBIO();writer.set_structure(backbone);writer.save(str(output_pose))
        row['structures']=[dict(path=output_pose.relative_to(ROOT).as_posix(),chain_id='D',role='generated_backbone_complex_not_refolded')]
        row['metadata']=dict(status='generated_not_independently_validated',
            backbone_generation=copy.deepcopy(parent['metadata']['generation']),
            backbone_refinement=copy.deepcopy(parent['metadata']['refinement']),
            generation=dict(model='ProteinMPNN soluble',source_fasta=(destination/fasta.name).relative_to(ROOT).as_posix(),
                sample=i,mpnn_score=float(score),parent_backbone_id=parent['id'],fixed_interface_positions=fixed),
            coordinates='shared refined BACKBONE only, relabelled with actual MPNN sequence; NOT a sequence refold')
        data['candidates'].append(row);seen.add(sequence);added.append(row['id'])
    meta.update(status='generated_not_independently_validated',actual_sequences=len(records),distinct_added=added)
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n');shutil.copyfile(out/'run.json',destination/'run.json')
    save_candidates(data,path);save_csv(data,path.with_suffix('.csv'))
    (path.parent/'candidates.fasta').write_text(''.join(f">{r['id']} generated_unvalidated\n{r['sequence']}\n" for r in data['candidates']))
    print(json.dumps(dict(actual_sequences=len(records),distinct_added=added,fixed_interface_positions=fixed),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates');p.add_argument('--id',required=True);p.add_argument('--out',required=True)
    p.add_argument('--sequences',type=int,default=2);p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--redesign-interface',action='store_true',help='design the full binder sequence in native target context, instead of holding the generated interface fixed')
    p.add_argument('--cpu',action='store_true',help='small MPNN batch on CPU while the GPU runs independent predictions')
    main(p.parse_args())
