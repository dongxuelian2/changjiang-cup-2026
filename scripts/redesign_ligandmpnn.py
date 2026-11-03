"""Initial-batch LigandMPNN ablation: native fixed sidechains as atom context.

Generate in the existing Linux Torch environment, then import using the
dataset's original BioPython environment. Neither NLL nor atom context is an
independent binding evaluation. Never extend a screened backbone family.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def generate(args):
    import importlib.metadata
    import torch
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('requested CUDA device unavailable')
    data=json.loads((ROOT/args.candidates).read_text())
    if data['synthetic']:raise ValueError('real unscreened scaffolds only')
    parent=next(r for r in data['candidates'] if r['id']==args.id)
    if any(k.startswith(('af2.','boltz')) and v is not None
           for r in data['candidates'] if r['structural_group']==parent['structural_group']
           for k,v in r['metrics'].items()):raise ValueError('never extend an independently screened family')
    if parent['metrics']['geometry.native_all_chain_BB_clash_pairs2']:
        raise ValueError('repair backbone sterics first')
    fixed=parent['metadata']['generation']['fixed_binder_AAs']
    if any(parent['sequence'][r['position']-1]!=r['aa'] for r in fixed):raise ValueError('fixed scaffold identity changed')
    positions=[i for i in range(1,len(parent['sequence'])+1) if i not in {r['position'] for r in fixed}]
    out=ROOT/args.out
    if out.exists():raise ValueError('choose a fresh output directory')
    out.mkdir(parents=True)
    input_pose=parent['structures'][0]['path']
    if args.context_state is not None:
        from Bio.PDB import PDBParser,PDBIO
        parser=PDBParser(QUIET=True)
        context=parser.get_structure('ensemble',ROOT/f'targets/TL1A/ensemble/state_{args.context_state}.pdb')
        original=parser.get_structure('binder',ROOT/input_pose)
        for c in 'ABC':
            if [(r.id,r.resname) for r in context[0][c]]!=[(r.id,r.resname) for r in original[0][c]]:
                raise ValueError('ensemble changed native target sequence/order')
        context[0].add(original[0]['D'].copy())
        writer=PDBIO();writer.set_structure(context);writer.save(str(out/'atomic_context.pdb'))
        input_pose=(out/'atomic_context.pdb').relative_to(ROOT).as_posix()
    source=ROOT/'build/models/LigandMPNN-main'
    # Upstream's bundled OpenFold uses NumPy's former builtin-int alias.
    # Replacing only the exact alias preserves its original dtype semantics.
    constants=source/'openfold/np/residue_constants.py'
    legacy=constants.read_text()
    constants.write_text(re.sub(r'\bnp\.int\b','int',legacy))
    checkpoint=source/'model_params/ligandmpnn_v_32_010_25.pt'
    revision=ZipFile(ROOT/'build/sources/ligandmpnn-main.zip').comment.decode()
    meta=dict(synthetic=False,model='LigandMPNN',source_url='https://github.com/dauparas/LigandMPNN',
        revision=revision,checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        checkpoint_url='https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt',
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        parent_candidates=args.candidates,parent_id=args.id,seed=args.seed,temperature=.1,
        fixed_input_identities=fixed,design_positions=positions,device=args.device,
        gpu_name=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        input_pose=input_pose,canonical_input_pose=parent['structures'][0]['path'],context_state=args.context_state,
        context='native-sidechain state plus modelled Man3GlcNAc2 on all six sites; binder backbone only' if args.context_state is not None else 'observed native ABC sidechain atoms; designed binder has backbone only',
        atom_context_ablation='same checkpoint/seed/backbone/fixed identities; toggle fixed-sidechain context only',
        compatibility_changes=['bundled openfold/np/residue_constants.py: exact np.int alias replaced by builtin int for NumPy >=1.24; weights/arithmetic unchanged'],
        environment={p:importlib.metadata.version(p) for p in ('torch','numpy','prody','ml-collections','dm-tree')},
        official_equivalence=False,status='running',trials=[])
    def save():(out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    save()
    for flag in args.contexts:
        folder=out/f'context_{flag}'
        command=[sys.executable,'-u',str(source/'run.py'),'--model_type','ligand_mpnn',
            '--checkpoint_ligand_mpnn',str(checkpoint),'--pdb_path',str(ROOT/meta['input_pose']),
            '--chains_to_design','D','--redesigned_residues',' '.join(f'D{i}' for i in positions),
            '--seed',str(args.seed),'--batch_size','1','--number_of_batches','1','--temperature','.1',
            '--omit_AA','C','--fasta_seq_separation',':','--save_stats','1',
            '--ligand_mpnn_use_side_chain_context',str(flag),'--out_folder',str(folder)]
        trial=dict(native_sidechain_context=flag,command=command,status='running');meta['trials'].append(trial);save()
        started=time.monotonic()
        with (out/f'context_{flag}.log').open('w') as log:
            result=subprocess.run(command,cwd=source,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1' if args.device=='cpu' else '0',
                'OMP_NUM_THREADS':'4','PYTHONIOENCODING':'utf-8'},stdout=log,stderr=subprocess.STDOUT)
        trial.update(exit_code=result.returncode,elapsed_seconds=time.monotonic()-started)
        if result.returncode:
            meta['status']='failed';trial['status']='failed';save();raise RuntimeError('inspect generation log')
        stats=torch.load(next((folder/'stats').glob('*.pt')),map_location='cpu',weights_only=False)
        mask=stats['chain_mask']*stats['mask']
        if int(mask.sum())!=len(positions):raise ValueError('actual designed-position count changed')
        scores=-(stats['log_probs'].gather(-1,stats['generated_sequences'][...,None]).squeeze(-1)*mask).sum(-1)/mask.sum()
        trial.update(status='generated_not_independently_validated',score=float(scores[0]),actual_design_positions=int(mask.sum()))
        save()
    meta['status']='generated_not_independently_validated';save()
    print(json.dumps(dict(status=meta['status'],trials=meta['trials']),indent=2))


def import_results(args):
    import Bio
    from Bio import SeqIO
    from Bio.PDB import PDBIO,PDBParser
    from Bio.SeqUtils import seq3
    from changjiang_select.data import load_candidates,save_candidates,save_csv
    from scripts.evaluate_designs import sequence_metrics
    out=ROOT/args.out;meta=json.loads((out/'run.json').read_text())
    if meta['status']!='generated_not_independently_validated':raise ValueError('requires a complete real paired ablation')
    path=ROOT/meta['parent_candidates'];data=load_candidates(path)
    if any(v['version']!=Bio.__version__+' / local-v1' for k,v in data['metric_definitions'].items() if k.startswith('sequence.')):
        raise ValueError('import using the dataset original BioPython version')
    parent=next(r for r in data['candidates'] if r['id']==meta['parent_id'])
    if any(k.startswith(('af2.','boltz')) and v is not None for r in data['candidates']
           if r['structural_group']==parent['structural_group'] for k,v in r['metrics'].items()):
        raise ValueError('freeze the entire initial batch before independent evaluation')
    expected=''.join((ROOT/'targets/TL1A/construct.fasta').read_text().splitlines()[1:])[16:]
    destination=path.parent/'generation'/out.name;destination.mkdir(exist_ok=True)
    if (out/'atomic_context.pdb').exists():shutil.copyfile(out/'atomic_context.pdb',destination/'atomic_context.pdb')
    seen={r['sequence'] for r in data['candidates']};added=[]
    for trial in meta['trials']:
        flag=trial['native_sidechain_context'];folder=out/f'context_{flag}'
        fasta=next((folder/'seqs').glob('*.fa'));records=list(SeqIO.parse(fasta,'fasta'))
        if len(records)!=2:raise ValueError('expected native reference and exactly one generated sequence')
        reference=str(records[0].seq).split(':');parts=str(records[1].seq).split(':')
        if reference!=[expected]*3+[parent['sequence']] or parts[:3]!=[expected]*3 or len(parts)!=4:
            raise ValueError('actual native ABC target sequences/order changed')
        sequence=parts[3]
        if len(sequence)!=len(parent['sequence']) or not set(sequence)<=set('ACDEFGHIKLMNPQRSTVWY'):
            raise ValueError('invalid binder sequence')
        if any(sequence[r['position']-1]!=r['aa'] for r in meta['fixed_input_identities']):raise ValueError('fixed core mutated')
        retained=destination/f'context_{flag}';retained.mkdir(exist_ok=True)
        for file in (fasta,next((folder/'stats').glob('*.pt')),out/f'context_{flag}.log'):shutil.copyfile(file,retained/file.name)
        if sequence in seen:continue
        row=copy.deepcopy(parent);row['id']=parent['id']+f'_lmpnn_ctx{flag}'
        row.update(sequence=sequence,source_model='native-sidechain-conditioned LigandMPNN' if flag else 'LigandMPNN without atom context',
            source_version=meta['revision'],run_id=out.name)
        row['metrics'].update(sequence_metrics(sequence));row['metrics']['ligandmpnn.score']=trial['score']
        pose=PDBParser(QUIET=True).get_structure('backbone',ROOT/meta.get('canonical_input_pose',meta['input_pose']))
        for residue,aa in zip(pose[0]['D'],sequence):residue.resname=seq3(aa).upper()
        file=path.parent/'poses'/f"{row['id']}_design.pdb"
        writer=PDBIO();writer.set_structure(pose);writer.save(str(file))
        row['structures']=[dict(path=file.relative_to(ROOT).as_posix(),chain_id='D',role='generated_backbone_complex_not_refolded')]
        row['metadata']=dict(status='generated_not_independently_validated',
            generation=dict(model='LigandMPNN',parent_backbone_id=parent['id'],native_sidechain_context=flag,
                glycan_ensemble_state=meta.get('context_state'),atomic_context=meta['context'],
                fixed_input_identities=meta['fixed_input_identities'],design_positions=meta['design_positions'],
                source_fasta=(retained/fasta.name).relative_to(ROOT).as_posix()),
            backbone_generation=copy.deepcopy(parent['metadata']['generation']),
            backbone_refinement=copy.deepcopy(parent['metadata'].get('backbone_refinement')),
            coordinates='shared parent backbone threaded with actual sequence; NOT a sequence refold')
        data['candidates'].append(row);seen.add(sequence);added.append(row['id'])
    data['metric_definitions']['ligandmpnn.score']=dict(model='LigandMPNN v_32_010_25',version=meta['revision'],
        meaning=f"sample NLL over {len(meta['design_positions'])} designable positions, under recorded context; NOT affinity or independent validation",units='nats/residue',direction='lower')
    shutil.copyfile(out/'run.json',destination/'run.json')
    save_candidates(data,path);save_csv(data,path.with_suffix('.csv'))
    path.with_suffix('.fasta').write_text(''.join(f">{r['id']} generated_unvalidated\n{r['sequence']}\n" for r in data['candidates']))
    print(json.dumps(dict(added=added),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['generate','import']);p.add_argument('--out',required=True)
    p.add_argument('--candidates');p.add_argument('--id');p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--contexts',type=int,nargs='+',choices=[0,1],default=[0,1])
    p.add_argument('--context-state',type=int,choices=[0,1,2],help='modelled fully glycosylated native-sidechain snapshot as atomic generation context')
    args=p.parse_args()
    if args.stage=='generate':generate(args)
    else:import_results(args)
