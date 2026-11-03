"""Repair a few local clashes, with native target atoms immobile.

This is restrained coordinate minimization, not MD or an affinity prediction.
Reject large backbone movements and never apply this to deeply occluded designs.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from Bio.PDB import PDBParser,MMCIFParser,PDBIO,Superimposer
from Bio.SeqUtils import seq1
import openmm as mm
from openmm import app,unit
from pdbfixer import PDBFixer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from changjiang_select.data import load_candidates,save_candidates,save_csv
from scripts.evaluate_designs import geometry


def main(args):
    path=(ROOT/args.candidates).resolve();data=load_candidates(path)
    row=next(r for r in data['candidates'] if r['id']==args.id)
    screened_family=any(k.startswith(('af2.','boltz')) and v is not None
           for r in data['candidates'] if r['structural_group']==row['structural_group']
           for k,v in r['metrics'].items())
    generation=row['metadata']['generation']
    supported_followup=False
    if generation.get('stage')=='consensus_truncation':
        parent=next(r for r in data['candidates'] if r['id']==generation['consensus_parent'])
        supported_followup=(min(parent['metrics'].get(k,0) for k in ('af2.ipsae_10','boltz_trimer_deepmsa.ipsae_10'))>=.4
            and generation['parent_core_cross_pose_A']<=2 and generation['parent_core_cross_fold_A']<=2
            and not any(k.startswith(('af2.','boltz')) and v is not None for k,v in row['metrics'].items()))
    if screened_family and not supported_followup:
        raise ValueError('local repair is only for a new, independently unscreened family')
    many_overlaps=row['metrics']['geometry.native_all_chain_BB_clash_pairs2']>10
    if many_overlaps and not args.localized_sidechain_repair:
        raise ValueError('not a local repair: more than ten backbone/native atom overlaps')
    out=(ROOT/args.out).resolve()
    if out.exists():raise ValueError('choose a fresh refinement directory')
    out.mkdir(parents=True)
    parser=PDBParser(QUIET=True)
    native=parser.get_structure('native',ROOT/'targets/TL1A/observed.pdb')
    manifest=json.loads((ROOT/'targets/manifest.json').read_text())
    core=next(t['beta_core_uniprot'] for t in manifest['targets'] if t['target_id']=='TL1A')
    fit=Superimposer()
    if generation.get('source_allatom_complex'):
        source=ROOT/generation['source_allatom_complex']
        raw=parser.get_structure('consensus',source)
        binder=raw[0]['D']
        if ''.join(seq1(r.resname) for r in binder)!=row['sequence']:
            raise ValueError('consensus sequence and all-atom coordinates differ')
        fit.set_atoms([native[0][c][n]['CA'] for c in 'ABC' for n in core],
                      [raw[0][c][n]['CA'] for c in 'ABC' for n in core])
    elif generation.get('stage')=='atom14':
        source=path.parent/'generation'/Path(generation['file']).name
        raw=MMCIFParser(QUIET=True).get_structure('generated',source)
        expected=''.join(seq1(r.resname) for r in native[0]['A'])
        targets=[c for c in raw[0] if ''.join(seq1(r.resname) for r in c)==expected]
        binders=[c for c in raw[0] if ''.join(seq1(r.resname) for r in c)==row['sequence']]
        if len(targets)!=3 or len(binders)!=1 or len(list(raw[0]))!=4:
            raise ValueError('raw BoltzGen atom14 must contain the exact trimer and binder')
        fit.set_atoms([native[0][c][n]['CA'] for c in 'ABC' for n in core],
                      [list(c)[n-88]['CA'] for c in targets for n in core])
        binder=binders[0]
    elif row['source_model'].startswith('ColabDesign AF2-Multimer'):
        history=json.loads((path.parent/'generation.json').read_text())['runs'][row['run_id']]
        mapping=history.get('target_residue_map',[dict(chain='A',uniprot=n) for n in range(88,252)])
        source=ROOT/generation['path']
        raw=parser.get_structure('generated',source)
        positions=list(raw[0]['A']);indices=[i for i,r in enumerate(mapping) if r['uniprot'] in core]
        fit.set_atoms([native[0][mapping[i]['chain']][mapping[i]['uniprot']]['CA'] for i in indices],
                      [positions[i]['CA'] for i in indices])
        binder=raw[0]['B']
    else:
        raise ValueError('requires genuine atom14/all-atom output; no placeholder IF sidechains')
    fit.apply(list(raw.get_atoms()))
    pose=copy.deepcopy(native);binder=binder.copy();binder.id='D';pose[0].add(binder)
    localized_admission=None
    if many_overlaps:
        bb_names={'N','CA','C','O'}
        ba=[a for a in binder.get_atoms() if a.name in bb_names]
        ta=[a for a in native.get_atoms() if a.element!='H']
        pairs=np.argwhere(np.linalg.norm(np.array([a.coord for a in ba])[:,None]-np.array([a.coord for a in ta]),axis=-1)<2)
        binder_res=sorted({ba[i].parent.id[1] for i,j in pairs})
        target_res=sorted({(ta[j].parent.parent.id,ta[j].parent.id[1]) for i,j in pairs})
        if (not len(pairs) or any(ta[j].name in bb_names for i,j in pairs)
            or len(target_res)!=1 or len(binder_res)>3 or max(binder_res)-min(binder_res)>2):
            raise ValueError('large overlap count is not confined to one target sidechain and <=3 adjacent binder residues')
        localized_admission=dict(overlap_pairs=int(len(pairs)),binder_residues=binder_res,target_residue=list(target_res[0]),
            all_target_atoms_sidechain=True,
            reason='pair count can multiply one sidechain occlusion; original backbone-movement and zero-residual-overlap limits still apply')
    writer=PDBIO();writer.set_structure(pose);writer.save(str(out/'input.pdb'))
    original={(c.id,str(r.id[1]),a.name):np.array(a.coord)/10 for c in pose[0] for r in c for a in r if a.element!='H'}
    platform=mm.Platform.getPlatformByName('CPU')
    fixer=PDBFixer(filename=str(out/'input.pdb'),platform=platform)
    fixer.missingResidues={}  # Do not invent target 72-87 or any missing peptide.
    fixer.findMissingAtoms();fixer.addMissingAtoms(seed=20261107);fixer.addMissingHydrogens(7.)
    xyz=np.array(fixer.positions.value_in_unit(unit.nanometer))
    atoms=list(fixer.topology.atoms());new=[]
    for a in atoms:
        key=(a.residue.chain.id,a.residue.id,a.name)
        if key in original:xyz[a.index]=original[key]
        elif a.element!=app.element.hydrogen:new.append(':'.join(key))
    ff=app.ForceField('amber14-all.xml','implicit/obc2.xml')
    system=ff.createSystem(fixer.topology,nonbondedMethod=app.NoCutoff,constraints=None)
    restraint=mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k',5000.)
    for p in ('x0','y0','z0'):restraint.addPerParticleParameter(p)
    target_indices=[];binder_ca=[];binder_heavy=[]
    for a in atoms:
        if a.residue.chain.id!='D':
            system.setParticleMass(a.index,0.);target_indices.append(a.index)
        else:
            if a.element!=app.element.hydrogen:binder_heavy.append(a.index)
            if a.name=='CA':binder_ca.append(a.index)
            if a.name in ('N','CA','C','O'):restraint.addParticle(a.index,xyz[a.index].tolist())
    system.addForce(restraint)
    # Same heuristic glycan envelope, with a 1 A buffer; not explicit sugars.
    anchors=[native[0][c][n]['ND2'].coord/10 for c in 'ABC' for n in (133,229)]
    for x,y,z in anchors:
        force=mm.CustomExternalForce('0.5*kg*step(r0-d)*(r0-d)^2; d=sqrt((x-gx)^2+(y-gy)^2+(z-gz)^2+1e-12)')
        for name,value in [('kg',5000.),('r0',1.3)]:
            force.addGlobalParameter(name,value)
        for name in ('gx','gy','gz'):force.addPerParticleParameter(name)
        for i in binder_heavy:force.addParticle(i,[float(x),float(y),float(z)])
        system.addForce(force)
    integrator=mm.VerletIntegrator(.001*unit.picoseconds)
    context=mm.Context(system,integrator,platform,{'Threads':'4'})
    context.setPositions(xyz*unit.nanometer)
    before=context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    started=time.monotonic()
    mm.LocalEnergyMinimizer.minimize(context,10.,args.iterations)
    state=context.getState(getPositions=True,getEnergy=True)
    after=np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    target_move=float(np.max(np.linalg.norm(after[target_indices]-xyz[target_indices],axis=-1))*10)
    displacement=np.linalg.norm(after[binder_ca]-xyz[binder_ca],axis=-1)*10
    rms=float(np.sqrt(np.mean(displacement**2)))
    report=dict(synthetic=False,purpose='local steric repair; NOT binding energy or KD',
        source_allatom_structure=source.relative_to(ROOT).as_posix(),
        source_target_beta_CA_rmsd=float(fit.rms),
        supported_new_sequence_followup=supported_followup,
        openmm=mm.version.full_version,pdbfixer_revision=subprocess.check_output(['git','-C',str(ROOT/'build/models/PDBFixer'),'rev-parse','HEAD'],text=True).strip(),
        forcefield=['amber14-all.xml','implicit/obc2.xml'],platform='CPU',threads=4,
        iterations_limit=args.iterations,tolerance_kJ_mol_nm=10.,binder_BB_restraint_kJ_mol_nm2=5000.,
        target_all_particles_zero_mass=True,unresolved_residues_added=False,
        added_heavy_atoms=new,glycan_heavy_penalty_radius_A=13.,
        source_backbone_overlaps=row['metrics']['geometry.native_all_chain_BB_clash_pairs2'],
        localized_sidechain_overlap_admission=localized_admission,
        target_max_movement_A=target_move,binder_CA_rms_movement_A=rms,
        binder_CA_max_movement_A=float(displacement.max()),elapsed_seconds=time.monotonic()-started,
        timing_scope='LocalEnergyMinimizer call only; topology and hydrogen preparation excluded',
        forcefield_energy_kJ_mol_before=before,forcefield_energy_kJ_mol_after=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    (out/'refinement.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'refined.pdb').open('w') as f:app.PDBFile.writeFile(fixer.topology,after*unit.nanometer,f,keepIds=True)
    if target_move>1e-5 or rms>1 or displacement.max()>2:
        raise ValueError('not an accepted local repair: target moved or binder backbone changed too much')
    refined=parser.get_structure('refined',out/'refined.pdb')
    if ''.join(seq1(r.resname) for r in refined[0]['D'])!=row['sequence']:
        raise ValueError('refinement changed candidate sequence')
    ligand=refined[0]['D'].copy()
    for r in ligand:
        for a in list(r):
            if a.name not in ('N','CA','C','O'):r.detach_child(a.id)
    canonical=copy.deepcopy(native);canonical[0].add(ligand)
    destination=path.parent/'poses'/f"{row['id']}_refined_design.pdb"
    writer.set_structure(canonical);writer.save(str(destination))
    canonical,metrics,epitope=geometry(destination,native,'D')
    bx=np.array([a.coord for a in canonical[0]['D'].get_atoms()]);nx=np.array([a.coord for a in native.get_atoms()])
    metrics['geometry.native_all_chain_BB_clash_pairs2']=int((np.linalg.norm(bx[:,None]-nx,axis=-1)<2).sum())
    if metrics['geometry.native_all_chain_BB_clash_pairs2'] or metrics['geometry.glycan_ND2_min_BB_distance']<12:
        raise ValueError('local minimization did not pass unchanged native steric/glycan gates')
    folder=path.parent/'generation'/f"refined_{row['id']}";folder.mkdir(exist_ok=True)
    import shutil
    for f in ('input.pdb','refined.pdb','refinement.json'):shutil.copyfile(out/f,folder/f)
    row['structures'][0]['role']='unrefined_generated_backbone_complex'
    row['structures'].insert(0,dict(path=destination.relative_to(ROOT).as_posix(),chain_id='D',role='generated_backbone_complex_not_refolded'))
    row['metrics'].update(metrics);row['epitope_residues']=epitope
    row['metadata']['refinement']={**report,'retained_allatom_complex':(folder/'refined.pdb').relative_to(ROOT).as_posix()}
    row['metadata']['coordinates']='native ABC + restrained-minimized binder BACKBONE; original model output retained'
    save_candidates(data,path);save_csv(data,path.with_suffix('.csv'))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates');p.add_argument('--id',required=True);p.add_argument('--out',required=True)
    p.add_argument('--iterations',type=int,default=500)
    p.add_argument('--localized-sidechain-repair',action='store_true',help='allow >10 pairs only if confined to one native sidechain and <=3 adjacent binder residues; retain movement limits')
    main(p.parse_args())
