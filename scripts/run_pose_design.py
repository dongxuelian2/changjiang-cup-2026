"""AF2 sequence/backbone optimization with a NEW scaffold's pose-distance losses.

The scaffold supplies supervised geometry, NEVER binder template features or
an initial guess. Source families must be independently unscreened. Retain the
initial IF sequence as a baseline before independent full-trimer prediction.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(args):
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.80')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from Bio.PDB import PDBParser, PDBIO, Select
    from colabdesign import mk_afdesign_model
    from scripts.design_context import select_pose_anchors

    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('GPU required')
    path = (ROOT/args.candidates).resolve()
    data = json.loads(path.read_text())
    if data['synthetic']:
        raise ValueError('requires a real new scaffold')
    parent = next(r for r in data['candidates'] if r['id'] == args.id)
    family = [r for r in data['candidates'] if r['structural_group'] == parent['structural_group']]
    if any(k.startswith(('af2.', 'boltz')) and v is not None for r in family for k, v in r['metrics'].items()):
        raise ValueError('never extend an independently screened family')
    fixed_input=parent['metadata']['generation'].get('fixed_binder_AAs',[]) if args.preserve_input_identities else []
    if args.preserve_input_identities and not fixed_input:
        raise ValueError('no audited fixed input identities on parent')
    if any(parent['sequence'][p['position']-1]!=p['aa'] for p in fixed_input):
        raise ValueError('parent changed a fixed input residue')
    out = (ROOT/args.out).resolve()
    if out.exists():
        raise ValueError('choose a fresh output directory')
    out.mkdir(parents=True)
    parser = PDBParser(QUIET=True)
    pose = parser.get_structure('scaffold', ROOT/parent['structures'][0]['path'])
    native = parser.get_structure('native', ROOT/'targets/TL1A/observed.pdb')
    binder_ca = np.array([r['CA'].coord for r in pose[0]['D']])
    # Include the closest stable-patch protomer in full; preserve native labels.
    primary = min('ABC', key=lambda c: min(np.linalg.norm(
        binder_ca[:, None]-np.array([native[0][c][n]['CA'].coord for n in (235,238)]), axis=-1).ravel()))
    order = [primary]+[c for c in 'ABC' if c != primary]
    selected = {primary: {r.id[1] for r in native[0][primary]}}
    for c in order[1:]:
        nearest = [(float(np.linalg.norm(np.array([a.coord for a in r])[:, None]-binder_ca, axis=-1).min()), r.id[1])
                   for r in native[0][c]]
        selected[c] = {n for _, n in sorted(nearest)[:args.context_neighbors]}
    # Preserve original native chain labels in the residue map. prep_inputs
    # concatenates these chains into one target entity, as in run_afdesign.py.
    target_map = [dict(chain=c, uniprot=r.id[1]) for c in order for r in native[0][c] if r.id[1] in selected[c]]
    target_ca = np.array([native[0][r['chain']][r['uniprot']]['CA'].coord for r in target_map])
    T, B = len(target_map), len(binder_ca)
    surface = {int(r['uniprot_residue']): r for r in csv.DictReader((ROOT/'targets/TL1A/surface.csv').open())}
    disagreement = {int(r['uniprot_residue']): r for r in csv.DictReader(
        (ROOT/args.target_disagreement).open())}
    eligible = [i for i, r in enumerate(target_map)
                if float(surface[r['uniprot']]['sasa_trimer_A2']) >= 20
                and float(disagreement[r['uniprot']]['CA_deviation_p90_A']) <= 2
                and float(surface[r['uniprot']]['nearest_glycan_ND2_A']) >= 12]
    anchors, allocation = select_pose_anchors(target_ca, binder_ca, eligible, args.anchors)
    reference_distances = np.linalg.norm(binder_ca[:, None]-target_ca[anchors], axis=-1)
    pair_mask = jnp.array(reference_distances < allocation['pair_cutoff_A'])
    reference_distances = jnp.array(reference_distances)
    bb0 = jnp.array(binder_ca)
    internal_distance = jnp.linalg.norm(bb0[:, None]-bb0, axis=-1)
    internal_mask = (jnp.abs(jnp.arange(B)[:, None]-jnp.arange(B)) > 3) & (internal_distance < 14)
    nx = jnp.array([a.coord for a in native.get_atoms() if a.element != 'H'])
    gx = jnp.array([native[0][c][n]['ND2'].coord for c in 'ABC' for n in (133,229)])
    target_ca_j = jnp.array(target_ca)

    def nearest_distance(points, other):
        # Avoid the much larger (atoms, atoms, 3) intermediate on an 8 GB GPU.
        origin=jnp.array(target_ca.mean(0));points=points-origin;other=other-origin
        d2 = (jnp.sum(points**2,-1)[:, None]+jnp.sum(other**2,-1)[None, :]
              -2*jnp.matmul(points,other.T,precision='highest'))
        return jnp.sqrt(jnp.maximum(d2.min(-1), 1e-6))

    class Context(Select):
        def accept_residue(self, r):
            return r.parent.id == 'D' or r.id[1] in selected[r.parent.id]

    context = out/'design_context.pdb'
    writer = PDBIO(); writer.set_structure(pose); writer.save(str(context), Context())

    def loss_callback(inputs, outputs, aux):
        xyz = aux['atom_positions']
        bca = xyz[T:, 1]
        anchor_xyz = target_ca_j[jnp.array(anchors)] if args.anchor_frame == 'native' else xyz[jnp.array(anchors), 1]
        distances = jnp.sqrt(jnp.sum((bca[:, None]-anchor_xyz)**2, -1)+1e-6)
        error = distances-reference_distances
        # Smooth L1, with a 1 A transition; normalized over a fixed pair set.
        huber = jnp.where(jnp.abs(error) < 1, .5*error**2, jnp.abs(error)-.5)
        fold_d = jnp.sqrt(jnp.sum((bca[:, None]-bca)**2, -1)+1e-6)
        bb = xyz[T:, jnp.array([0,1,2,4])].reshape(-1,3)
        d = nearest_distance(bb, nx)
        gd = jnp.sqrt(jnp.sum((bb[:, None]-gx)**2,-1)+1e-6).min(-1)
        heavy = xyz[T:].reshape(-1,3)
        valid = aux['atom_mask'][T:].reshape(-1)
        native_d = nearest_distance(heavy, nx)
        predicted_target = xyz[:T].reshape(-1,3)[pred_target_atom_indices]
        predicted_d = nearest_distance(heavy, predicted_target)
        heavy_gd = jnp.sqrt(jnp.sum((heavy[:, None]-gx)**2,-1)+1e-6).min(-1)
        return dict(pose_distance=(huber*pair_mask).sum()/pair_mask.sum(),
            scaffold_fold=(((fold_d-internal_distance)**2)*internal_mask).sum()/internal_mask.sum(),
            native_target=jnp.mean(jnp.sum((xyz[:T,1]-target_ca_j)**2, -1)),
            context_steric=jnp.sort(jax.nn.relu(2.5-d)**2)[-10:].mean(),
            native_heavy_steric=jnp.sort(valid*jax.nn.relu(2.-native_d)**2)[-10:].mean(),
            predicted_heavy_steric=jnp.sort(valid*jax.nn.relu(2.-predicted_d)**2)[-10:].mean(),
            glycan=jnp.max(jax.nn.relu(15.-gd)**2)/25.,
            heavy_glycan=jnp.max(valid*jax.nn.relu(12.-heavy_gd)**2)/25.)

    revision = subprocess.check_output(['git','-C',str(ROOT/'build/models/ColabDesign'),'rev-parse','HEAD'], text=True).strip()
    checkpoint = ROOT/'build/models/colabfold/params/params_model_3_multimer_v3.source.json'
    weights = dict(plddt=.2, pae=.4, con=.5, i_con=.5, i_pae=.5,
                   dgram_cce=0., rmsd=args.pose_rmsd_weight, fape=0., pose_distance=1., scaffold_fold=args.fold_weight,
                   native_target=.1, context_steric=3., glycan=1.,
                   native_heavy_steric=3., predicted_heavy_steric=3., heavy_glycan=1.)
    seed = args.seed
    trial = dict(seed=seed, status='initializing', sequence_count=0)
    meta = dict(synthetic=False, model='ColabDesign AF2-Multimer supervised pose-distance design',
        design_method='new BoltzGen alpha/beta scaffold + geometry-supervised AF2 sequence/backbone optimization',
        upstream_revision=revision, package_version=importlib.metadata.version('colabdesign'),
        jax_version=jax.__version__, design_model='model_3_multimer_v3',
        checkpoint=json.loads(checkpoint.read_text()), parent_id=parent['id'],
        parent_candidates=path.relative_to(ROOT).as_posix(), structural_group=parent['structural_group'],
        initial_binder_sequence=parent['sequence'], binder_template=False, initial_guess=False,
        template=True, target='O95150:72-251', observed_target='88-251', target_residue_map=target_map,
        predictor_stoichiometry='one complete native protomer plus mapped nearest-neighbor fragments, concatenated target entity + binder',
        geometric_context='all native ABC heavy atoms and six N133/N229 ND2 anchors',
        site='terminal_pose', binder_length=B, context_neighbor_residues_per_protomer=args.context_neighbors,
        native_primary_chain=primary, target_chain_order=order,
        conditioning='new scaffold only as supervised loss; no binder template, initial guess, or known binder',
        fixed_input_identities=fixed_input,
        pose_anchor_residues=[target_map[i] for i in anchors], pose_anchor_allocation=allocation,
        pose_anchor_frame=args.anchor_frame,
        target_disagreement_source=args.target_disagreement,
        loss_weights=weights, stages=[args.soft_iters,args.temp_iters,args.hard_iters],
        schedule=args.schedule,
        design_recycles=args.recycles, final_hard_recycles=1,
        source_backbone=parent['structures'][0]['path'],
        source_backbone_sha256=hashlib.sha256((ROOT/parent['structures'][0]['path']).read_bytes()).hexdigest(),
        native_BB_distance_penalty_A=2.5, native_heavy_distance_penalty_A=2.,
        predicted_heavy_distance_penalty_A=2., glycan_BB_penalty_A=15., glycan_heavy_penalty_A=12.,
        official_equivalence=False, status='initializing', trajectories=[trial])
    def save():
        (out/'run.json').write_text(json.dumps(meta, indent=2, default=lambda x: np.asarray(x).tolist())+'\n')
    save(); started = time.monotonic()
    try:
        model = mk_afdesign_model(protocol='binder',use_multimer=True,
            model_names=['model_3_multimer_v3'],num_models=1,sample_models=False,
            data_dir=str(ROOT/'build/models/colabfold'),num_recycles=args.recycles,
            use_remat=True,use_bfloat16=True,loss_callback=loss_callback,traj_iter=10,traj_max=20)
        model.prep_inputs(pdb_filename=str(context),target_chain=','.join(order),binder_chain='D',
            use_binder_template=False,rm_target_seq=False,rm_target_sc=False,
            hotspot=','.join(f'{primary}{n}' for n in (235,238)))
        if model._target_len != T or not np.allclose(model._inputs['batch']['all_atom_positions'][:T,1],target_ca):
            raise ValueError('target frame or residue order changed')
        if not np.all(model._inputs['rm_template'][T:]) or model._args['use_initial_guess']:
            raise ValueError('scaffold leaked into binder inputs')
        pred_target_atom_indices = jnp.array(np.flatnonzero(
            np.asarray(model._inputs['batch']['all_atom_mask'][:T]).reshape(-1)))
        model.restart(seed=seed,seq=parent['sequence'],rm_aa='C')
        if fixed_input:
            model.opt['fix_pos']=np.array([p['position']-1 for p in fixed_input],dtype=np.int32)
        model.opt['weights'].update(weights)
        model.set_opt(con=dict(num=2,cutoff=8.,seqsep=9),i_con=dict(num=3,cutoff=8.))
        meta['status']='designing'; trial['status']='running'; save()
        def progress(m):
            weighted = sum(float(v)*m.opt['weights'].get(k,1.) for k,v in m.aux['losses'].items())
            if not np.isclose(float(m.aux['loss']),weighted,rtol=1e-4,atol=1e-4):
                raise ValueError('reported loss differs from declared weighted losses')
            if (m._k+1)%10 == 0:
                trial['progress']=dict(iteration=m._k+1,loss=float(m.aux['loss']),
                    binder_plddt=float(np.asarray(m.aux['plddt'])[T:].mean()),
                    losses={k:float(v) for k,v in m.aux['losses'].items()})
                save()
        if args.schedule == 'strict':
            model.design_3stage(soft_iters=args.soft_iters,temp_iters=args.temp_iters,
                hard_iters=args.hard_iters,ramp_recycles=False,verbose=10,callback=progress)
        else:
            def phase(name, overrides):
                model.opt['weights'].update({**weights, **overrides})
                trial.setdefault('phases',[]).append(dict(name=name,iteration=model._k,
                    weights=dict(model.opt['weights'])))
                save()
            phase('fold_and_pose',dict(context_steric=.2,native_heavy_steric=.2,
                predicted_heavy_steric=.2,glycan=.1,heavy_glycan=.1,i_pae=.1))
            model.design_logits(args.soft_iters,e_soft=1,verbose=10,callback=progress)
            phase('steric_transition',dict(context_steric=1.,native_heavy_steric=1.,
                predicted_heavy_steric=1.,glycan=.5,heavy_glycan=.5))
            transition=args.temp_iters//3
            model.design_soft(transition,e_temp=.3,verbose=10,callback=progress)
            phase('strict_interface',{})
            model.design_soft(args.temp_iters-transition,temp=.3,e_temp=.01,verbose=10,callback=progress)
            model.design_hard(args.hard_iters,temp=.01,dropout=False,verbose=10,callback=progress)
        sequence=model.get_seqs(get_best=False)[0]
        if any(sequence[p['position']-1]!=p['aa'] for p in fixed_input):
            raise ValueError('final discrete sequence changed fixed scaffold identities')
        model.predict(seq=sequence,hard=True,soft=False,dropout=False,num_recycles=1,verbose=True)
        folder=out/f'seed_{seed}'; folder.mkdir()
        model.save_pdb(str(folder/'generated.pdb'),get_best=False)
        a=model.aux
        if not np.isfinite(a['atom_positions']).all() or not np.isfinite(a['pae']).all():
            raise FloatingPointError('nonfinite output')
        (folder/'scores.json').write_text(json.dumps(dict(plddt=(np.asarray(a['plddt'])*100).tolist(),
            pae=np.asarray(a['pae']).tolist(),iptm=float(a['i_ptm']),ptm=float(a['ptm'])))+'\n')
        (folder/'sequence.fasta').write_text(f'>pose_design_{seed} generated_unvalidated\n{sequence}\n')
        (folder/'trajectory.json').write_text(json.dumps(model._tmp['log'],default=lambda x:np.asarray(x).tolist())+'\n')
        trial.update(status='generated_not_independently_validated',sequence_count=1,sequence=sequence,
            final_generator_log=a['log'])
        meta['status']='generated_not_independently_validated'
    except BaseException as exc:
        meta.update(status='failed',error=str(exc)); raise
    finally:
        meta['elapsed_seconds']=time.monotonic()-started; save()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates'); p.add_argument('--id',required=True); p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20261121)
    p.add_argument('--soft-iters',type=int,default=50); p.add_argument('--temp-iters',type=int,default=25)
    p.add_argument('--hard-iters',type=int,default=10); p.add_argument('--recycles',type=int,default=1)
    p.add_argument('--anchors',type=int,default=6); p.add_argument('--context-neighbors',type=int,default=20)
    p.add_argument('--schedule',choices=['strict','curriculum'],default='strict')
    p.add_argument('--pose-rmsd-weight',type=float,default=0.)
    p.add_argument('--fold-weight',type=float,default=.1)
    p.add_argument('--anchor-frame',choices=['predicted','native'],default='predicted',
                   help='native anchors prevent reducing distance loss by deforming predicted target landmarks')
    p.add_argument('--target-disagreement',default='results/tl1a_boltzgen_terminal_20261102/target_disagreement.csv')
    p.add_argument('--preserve-input-identities',action='store_true',help='hold audited input scaffold identities one-hot throughout optimization')
    main(p.parse_args())
