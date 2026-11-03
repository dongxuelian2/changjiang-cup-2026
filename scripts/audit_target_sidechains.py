"""Separate actual target backbone disagreement from sidechain-state changes.

Local N/CA/C fitting removes backbone rigid motion. Chemically equivalent ARG,
LEU and TYR labels are compared with their coupled symmetry swaps. Dispersion
across related predictions is not measured flexibility or calibrated uncertainty.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from Bio.PDB import MMCIFParser,PDBParser,Superimposer
from Bio.SeqUtils import seq1

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.import_boltz import align_prediction


def measure(native,predicted):
    names=sorted(set(native.child_dict)&set(predicted.child_dict)-{'N','CA','C','O','OXT'})
    names=[n for n in names if native[n].element!='H']
    if not names:return None
    fit=Superimposer();fit.set_atoms([native[n] for n in ('N','CA','C')],[predicted[n] for n in ('N','CA','C')])
    rot,shift=fit.rotran
    swaps={'ARG':{'NH1':'NH2','NH2':'NH1'},'LEU':{'CD1':'CD2','CD2':'CD1'},
           'TYR':{'CD1':'CD2','CD2':'CD1','CE1':'CE2','CE2':'CE1'}}
    mappings=[{}]+([swaps[native.resname]] if native.resname in swaps else [])
    original=np.array([native[n].coord for n in names])
    best=None
    for mapping in mappings:
        if not all(mapping.get(n,n) in predicted for n in names):continue
        current=np.array([predicted[mapping.get(n,n)].coord for n in names])@rot+shift
        delta=np.linalg.norm(original-current,axis=-1)
        if best is None or np.mean(delta**2)<best[0]:best=(float(np.mean(delta**2)),mapping,delta)
    return dict(CA_deviation_global_A=float(np.linalg.norm(native['CA'].coord-predicted['CA'].coord)),
        local_N_CA_C_fit_RMSD_A=float(fit.rms),sidechain_RMSD_after_local_fit_A=float(np.sqrt(best[0])),
        equivalent_label_swap=best[1],atom_displacements_after_local_fit_A={n:float(d) for n,d in zip(names,best[2])})


def main(args):
    path=ROOT/args.candidates;data=json.loads(path.read_text())
    if data['synthetic']:raise ValueError('requires real structures')
    native=PDBParser(QUIET=True).get_structure('native',ROOT/'targets/TL1A/observed.pdb')
    expected=''.join(seq1(r.resname) for r in native[0]['A'])
    manifest=json.loads((ROOT/'targets/manifest.json').read_text())
    core=next(t['beta_core_uniprot'] for t in manifest['targets'] if t['target_id']=='TL1A')
    parsers={'.pdb':PDBParser(QUIET=True),'.cif':MMCIFParser(QUIET=True)}
    observations=[]
    for row in data['candidates']:
        if args.ids and row['id'] not in args.ids:continue
        source=row['metadata']['generation'];states=[]
        if source.get('stage')=='atom14':
            file=ROOT/source['retained_file'] if 'retained_file' in source else path.parent/'generation'/Path(source['file']).name
            state=parsers[file.suffix].get_structure('raw',file)
            targets=[state[0][c] for c in source['original_target_chains']]
            if any(''.join(seq1(r.resname) for r in c)!=expected for c in targets):raise ValueError('raw target mismatch')
            fit=Superimposer();fit.set_atoms([native[0][c][n]['CA'] for c in 'ABC' for n in core],
                [list(c)[n-88]['CA'] for c in targets for n in core]);fit.apply(list(state.get_atoms()))
            states.append(('original_atom14',file,{c:{u:list(t)[u-88] for u in args.residues} for c,t in zip('ABC',targets)}))
        design=parsers['.pdb'].get_structure('design',ROOT/row['structures'][0]['path'])[0]['D']
        for record in row['structures']:
            role=record['role']
            if not role.endswith('_predicted_complex') or not role.startswith(('af2','boltz_trimer_deepmsa')):continue
            file=ROOT/record['path'];state=parsers[file.suffix].get_structure('prediction',file)
            _,_,perm,_,_=align_prediction(native,state,design,'ABC','D',core)
            states.append((role,file,{c:{u:state[0][p][u-71] for u in args.residues} for c,p in zip('ABC',perm)}))
        for role,file,targets in states:
            for c in 'ABC':
                for u in args.residues:
                    value=measure(native[0][c][u],targets[c][u])
                    if value is not None:observations.append(dict(id=row['id'],role=role,path=file.relative_to(ROOT).as_posix(),
                        native_chain=c,uniprot=u,resname=native[0][c][u].resname,**value))
    summary={}
    for u in args.residues:
        records=[r for r in observations if r['uniprot']==u]
        if records:
            summary[str(u)]=dict(chain_predictions=len(records),
                CA_deviation_global_p50_p90_A=np.quantile([r['CA_deviation_global_A'] for r in records],[.5,.9]).tolist(),
                local_sidechain_RMSD_p50_p90_A=np.quantile([r['sidechain_RMSD_after_local_fit_A'] for r in records],[.5,.9]).tolist(),
                stable_CA_but_sidechain_RMSD_above_2_A=sum(r['CA_deviation_global_A']<=1 and r['sidechain_RMSD_after_local_fit_A']>2 for r in records))
    report=dict(synthetic=False,interpretation='coordinate disagreement only; related models/three protomers are not IID replicates or measured flexibility',
        summary=summary,observations=observations)
    (ROOT/args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates');p.add_argument('--ids',nargs='+');p.add_argument('--residues',type=int,nargs='+',default=[133,156,158,229,235,238])
    p.add_argument('--out',required=True)
    main(p.parse_args())
