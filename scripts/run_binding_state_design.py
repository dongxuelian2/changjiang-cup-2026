"""NEW-family binding-state AF2 generation with an explicit atomic ensemble.

Generation uses a scaffold intrachain template and pose initial guess, then
withdraws BOTH before the final optimisation/discrete prediction. Independent
full-trimer AF2/Boltz predictions must still be run with no binder conditioning.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(args):
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.80')
    os.environ.setdefault('OMP_NUM_THREADS','4')
    import importlib.metadata
    import jax
    import jax.numpy as jnp
    import numpy as np
    from Bio.PDB import PDBParser,PDBIO,Select,Atom
    from colabdesign import mk_afdesign_model
    from scripts.design_context import select_pose_anchors

    if not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU required')
    path=ROOT/args.candidates;data=json.loads(path.read_text())
    if data['synthetic']:raise ValueError('real new family required')
    parent=next(r for r in data['candidates'] if r['id']==args.id)
    if any(k.startswith(('af2.','boltz')) and v is not None for r in data['candidates']
           if r['structural_group']==parent['structural_group'] for k,v in r['metrics'].items()):
        raise ValueError('never extend an independently screened family')
    fixed=parent['metadata']['generation']['fixed_binder_AAs']
    if any(parent['sequence'][r['position']-1]!=r['aa'] for r in fixed):raise ValueError('fixed scaffold identity changed')
    initial=parent
    if args.initial_sequence_id:
        initial=next(r for r in data['candidates'] if r['id']==args.initial_sequence_id)
        if initial['structural_group']!=parent['structural_group']:raise ValueError('initial sequence must share the unscreened backbone')
        if any(initial['sequence'][r['position']-1]!=r['aa'] for r in fixed):raise ValueError('initial sequence changed fixed identities')
    out=ROOT/args.out
    if out.exists():raise ValueError('choose fresh output directory')
    out.mkdir(parents=True)
    parser=PDBParser(QUIET=True)
    pose=parser.get_structure('pose',ROOT/parent['structures'][0]['path'])
    native=parser.get_structure('native',ROOT/'targets/TL1A/observed.pdb')
    b0=np.array([r['CA'].coord for r in pose[0]['D']]);B=len(b0)
    primary=min('ABC',key=lambda c:np.linalg.norm(b0[:,None]-np.array([native[0][c][n]['CA'].coord for n in (156,235,238)]),axis=-1).min())
    order=[primary]+[c for c in 'ABC' if c!=primary]
    selected={primary:{r.id[1] for r in native[0][primary]}}
    for c in order[1:]:
        nearest=[(float(np.linalg.norm(np.array([a.coord for a in r])[:,None]-b0,axis=-1).min()),r.id[1]) for r in native[0][c]]
        selected[c]={n for _,n in sorted(nearest)[:args.context_neighbors]}
    mapping=[dict(chain=c,uniprot=r.id[1]) for c in order for r in native[0][c] if r.id[1] in selected[c]]
    t0=np.array([native[0][r['chain']][r['uniprot']]['CA'].coord for r in mapping]);T=len(t0)
    anchors,allocation=select_pose_anchors(t0,b0,list(range(T)),6)
    reference=jnp.array(np.linalg.norm(b0[:,None]-t0[anchors],axis=-1))
    pose_mask=reference<allocation['pair_cutoff_A']
    internal=jnp.array(np.linalg.norm(b0[:,None]-b0,axis=-1))
    internal_mask=(jnp.abs(jnp.arange(B)[:,None]-jnp.arange(B))>3)&(internal<14)
    reference_contacts=jnp.array(np.linalg.norm(b0[:,None]-t0,axis=-1)<8.)
    # Positive contacts and off-patch decoys contribute separately: abundant
    # negative pairs must not overwhelm the small intended interface.
    far_contacts=jnp.array(np.linalg.norm(b0[:,None]-t0,axis=-1)>16.)
    ensemble=np.load(ROOT/'targets/TL1A/ensemble/states.npz')
    protein=np.unique(np.round(np.concatenate([ensemble['protein'],ensemble['unoccupied_protein']]).reshape(-1,3),3),axis=0)
    sugars=np.unique(np.round(ensemble['glycan'].reshape(-1,3),3),axis=0)
    px,gx=jnp.array(protein),jnp.array(sugars)
    nd2=jnp.array([native[0][c][n]['ND2'].coord for c in 'ABC' for n in (133,229)])

    # Backbone-only canonical poses lack CB, required by AF2 template distograms.
    # Add ideal pseudo-CB for non-Gly residues as generation features only.
    for residue in pose[0]['D']:
        if residue.resname=='GLY' or 'CB' in residue:continue
        n,ca,c=(residue[a].coord for a in ('N','CA','C'))
        b,cvec=ca-n,c-ca;a=np.cross(b,cvec)
        cb=ca-.58273431*a+.56802827*b-.54067466*cvec
        residue.add(Atom.Atom('CB',cb,0.,1.,' ',' CB ',0,element='C'))
    class Context(Select):
        def accept_residue(self,r):return r.parent.id=='D' or r.id[1] in selected[r.parent.id]
    context=out/'design_context.pdb';writer=PDBIO();writer.set_structure(pose);writer.save(str(context),Context())

    def nearest(points,other):
        # Near-contact subtraction is sensitive to reduced-precision GPU GEMM.
        # Centering plus highest precision was checked against CPU cdist.
        origin=jnp.array(t0.mean(0));points=points-origin;other=other-origin
        d2=jnp.sum(points**2,-1)[:,None]+jnp.sum(other**2,-1)[None,:]-2*jnp.matmul(points,other.T,precision='highest')
        return jnp.sqrt(jnp.maximum(d2.min(-1),1e-6))

    def loss_callback(inputs,outputs,aux):
        xyz=aux['atom_positions'];bca=xyz[T:,1]
        distances=jnp.sqrt(jnp.sum((bca[:,None]-jnp.array(t0[anchors]))**2,-1)+1e-6)
        error=jnp.abs(distances-reference)
        huber=jnp.where(error<2,.5*error**2,2*(error-1))
        # Select all present standard-AA heavy atoms (<=14/residue); retain
        # masks for padding. Unlike a CA sphere this sees protruding sidechains.
        mask=aux['atom_mask'][T:]
        idx=jnp.argsort(-mask,axis=-1,stable=True)[:,:14]
        heavy=jnp.take_along_axis(xyz[T:],idx[...,None],axis=1).reshape(-1,3)
        valid=jnp.take_along_axis(mask,idx,axis=1).reshape(-1)
        pd,gd=nearest(heavy,px),nearest(heavy,gx)
        nd=nearest(heavy,nd2)
        fold=jnp.sqrt(jnp.sum((bca[:,None]-bca)**2,-1)+1e-6)
        logits=outputs['distogram']['logits'][T:,:T]
        edges=outputs['distogram']['bin_edges']
        bins=jnp.concatenate([edges,jnp.array([1e8])])
        prob=(jax.nn.softmax(logits,-1)*(bins<8.)).sum(-1)
        positive=-(jnp.log(prob+1e-6)*reference_contacts).sum()/jnp.maximum(reference_contacts.sum(),1)
        negative=-(jnp.log(1-prob+1e-6)*far_contacts).sum()/jnp.maximum(far_contacts.sum(),1)
        return dict(association=1-aux['i_ptm'],interface_contacts=positive+.1*negative,
            pose_distance=(huber*pose_mask).sum()/pose_mask.sum(),
            scaffold_fold=(((fold-internal)**2)*internal_mask).sum()/internal_mask.sum(),
            native_target=jnp.mean(jnp.sum((xyz[:T,1]-jnp.array(t0))**2,-1)),
            native_ensemble_steric=jnp.max(valid*jax.nn.relu(2.-pd)**2),
            glycan_ensemble_steric=jnp.max(valid*jax.nn.relu(2.3-gd)**2),
            glycan_ND2=jnp.max(valid*jax.nn.relu(12.-nd)**2)/25.)

    weights=dict(plddt=.2,pae=.2,con=.1,i_con=.2,i_pae=.5,dgram_cce=0.,rmsd=0.,fape=0.,
        association=2.,interface_contacts=1.,pose_distance=.1,scaffold_fold=.1,native_target=.1,
        native_ensemble_steric=5.,glycan_ensemble_steric=5.,glycan_ND2=1.)
    revision=subprocess.check_output(['git','-C',str(ROOT/'build/models/ColabDesign'),'rev-parse','HEAD'],text=True).strip()
    trial=dict(seed=args.seed,status='running',sequence_count=0)
    meta=dict(synthetic=False,model='ColabDesign AF2 binding-state plus native-sidechain/glycan ensemble generation',
        upstream_revision=revision,package_version=importlib.metadata.version('colabdesign'),jax_version=jax.__version__,
        design_model='model_3_multimer_v3',checkpoint=json.loads((ROOT/'build/models/colabfold/params/params_model_3_multimer_v3.source.json').read_text()),
        parent_id=parent['id'],parent_candidates=args.candidates,structural_group=parent['structural_group'],
        initial_binder_sequence=initial['sequence'],initial_sequence_id=initial['id'],binder_template=True,initial_guess=True,
        generation_conditioning='binder intrachain template with interchain template geometry masked; pose initial guess; both withdrawn in release phase',
        final_prediction_conditioning=dict(binder_template=False,initial_guess=False),
        sidechain_conditioning='external atomic loss; AF2 template features do not expose full native terminal sidechains',
        target_residue_map=mapping,predictor_stoichiometry='complete primary native protomer + nearest native B/C fragments concatenated as target; binder',
        ensemble_source='targets/TL1A/ensemble/ensemble.json',ensemble_protein_points=len(protein),ensemble_glycan_points=len(sugars),
        atomic_distance_precision='centered float32 squared distances; explicit JAX matmul precision highest; CPU cdist verification',
        ensemble_constraints='worst single-pair steric penalty over all covered native/protein and modelled glycan marginal states, not population free energy',
        fixed_input_identities=fixed,site='a3d_bound_ensemble',loss_weights=weights,
        stages=dict(bound_soft=args.soft_iters,bound_temp=args.temp_iters,bound_hard=args.hard_iters,release_hard=args.release_iters),
        design_recycles=args.recycles,final_hard_recycles=3,pose_anchor_residues=[mapping[i] for i in anchors],
        positive_contact_pairs=int(reference_contacts.sum()),official_equivalence=False,status='initializing',trajectories=[trial])
    meta['contact_proxy']='reference CA pairs <8 A; predicted AF2 CB-distogram probability <8 A; coarse interface proxy'
    def save():(out/'run.json').write_text(json.dumps(meta,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')
    def snapshot(model,folder):
        folder.mkdir();model.save_pdb(str(folder/'generated.pdb'),get_best=False)
        a=model.aux;seq=model.get_seqs(get_best=False)[0]
        (folder/'scores.json').write_text(json.dumps(dict(plddt=(np.asarray(a['plddt'])*100).tolist(),pae=np.asarray(a['pae']).tolist(),iptm=float(a['i_ptm']),ptm=float(a['ptm'])))+'\n')
        (folder/'sequence.fasta').write_text(f'>binding_state_design generated_unvalidated\n{seq}\n')
        return seq
    save();started=time.monotonic()
    try:
        model=mk_afdesign_model(protocol='binder',use_multimer=True,model_names=['model_3_multimer_v3'],
            num_models=1,sample_models=False,data_dir=str(ROOT/'build/models/colabfold'),num_recycles=args.recycles,
            use_remat=True,use_bfloat16=True,use_initial_guess=True,loss_callback=loss_callback,traj_iter=10,traj_max=20)
        model.prep_inputs(pdb_filename=str(context),target_chain=','.join(order),binder_chain='D',
            use_binder_template=True,rm_binder_seq=True,rm_binder_sc=True,rm_template_ic=True,
            rm_target_seq=False,rm_target_sc=False,hotspot=','.join(f'{primary}{n}' for n in (156,235,238)))
        if model._target_len!=T or not np.allclose(model._inputs['batch']['all_atom_positions'][:T,1],t0):raise ValueError('target frame/order changed')
        if np.any(model._inputs['rm_template'][T:]) or not model._args['use_initial_guess']:raise ValueError('bound conditioning missing')
        model.restart(seed=args.seed,seq=initial['sequence'],rm_aa='C')
        model.opt['fix_pos']=np.array([r['position']-1 for r in fixed],dtype=np.int32)
        model.opt['weights'].update(weights);model.set_opt(con=dict(num=2,cutoff=8.,seqsep=9),i_con=dict(num=3,cutoff=8.))
        meta['status']='designing';save()
        def progress(m):
            if (m._k+1)%5==0:
                trial['progress']=dict(iteration=m._k+1,phase=meta['status'],loss=float(m.aux['loss']),
                    iptm=float(m.aux['i_ptm']),losses={k:float(v) for k,v in m.aux['losses'].items()});save()
        model.design_3stage(soft_iters=args.soft_iters,temp_iters=args.temp_iters,hard_iters=args.hard_iters,
            ramp_recycles=False,verbose=5,callback=progress)
        seq=model.get_seqs(get_best=False)[0]
        model.predict(seq=seq,hard=True,soft=False,dropout=False,num_recycles=3,verbose=True)
        meta['bound_sequence']=snapshot(model,out/'bound')
        meta['bound_discrete_log']=model.aux['log']
        # use_initial_guess is read outside JIT in _recycle. clear_prev ensures
        # no bound coordinates persist. Template mask is a dynamic model input.
        model._args['use_initial_guess']=False;model._args['clear_prev']=True
        model._inputs['rm_template'][T:]=True
        model._inputs.pop('prev',None)
        meta['status']='release_design';save()
        model.design_hard(args.release_iters,temp=.01,dropout=False,verbose=5,callback=progress)
        seq=model.get_seqs(get_best=False)[0]
        if any(seq[r['position']-1]!=r['aa'] for r in fixed):raise ValueError('final fixed identities changed')
        model.predict(seq=seq,hard=True,soft=False,dropout=False,num_recycles=3,verbose=True)
        if not np.isfinite(model.aux['atom_positions']).all():raise FloatingPointError('nonfinite output')
        folder=out/f'seed_{args.seed}';seq=snapshot(model,folder)
        (folder/'trajectory.json').write_text(json.dumps(model._tmp['log'],default=lambda x:np.asarray(x).tolist())+'\n')
        trial.update(status='generated_not_independently_validated',sequence_count=1,sequence=seq,final_generator_log=model.aux['log'])
        meta['status']='generated_not_independently_validated'
    except BaseException as exc:
        trial['status']='failed';meta.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        meta['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates');p.add_argument('--id',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20261111);p.add_argument('--context-neighbors',type=int,default=20)
    p.add_argument('--initial-sequence-id',help='optional atomic-context sequence from the same new, unscreened backbone')
    p.add_argument('--soft-iters',type=int,default=30);p.add_argument('--temp-iters',type=int,default=10)
    p.add_argument('--hard-iters',type=int,default=5);p.add_argument('--release-iters',type=int,default=10)
    p.add_argument('--recycles',type=int,default=1)
    main(p.parse_args())
