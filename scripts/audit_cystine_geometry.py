"""Measure specified Cys-pair geometry in real all-atom outputs.

SG proximity is a geometric observation, not a verified covalent bond or a
folding/association prediction. IF and MPNN placeholder sidechains are skipped.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.vectors import calc_dihedral
from Bio.SeqUtils import seq1

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from changjiang_select.data import load_candidates


def measure(binder,pairs):
    residues=list(binder)
    cysteines=[i+1 for i,r in enumerate(residues) if r.resname=='CYS']
    sulfur={i:residues[i-1]['SG'] for i in cysteines if 'SG' in residues[i-1]}
    links=[]
    for a,b in pairs:
        record=dict(positions=[a,b],both_Cys_SG_present=a in sulfur and b in sulfur)
        if record['both_Cys_SG_present']:
            record['SG_distance_A']=float(np.linalg.norm(sulfur[a].coord-sulfur[b].coord))
            if all('CB' in residues[i-1] for i in (a,b)):
                atoms=[residues[a-1]['CB'],sulfur[a],sulfur[b],residues[b-1]['CB']]
                record['CB_SG_SG_CB_dihedral_deg']=float(np.degrees(calc_dihedral(*(x.get_vector() for x in atoms))))
        links.append(record)
    close=[dict(positions=[a,b],distance_A=float(np.linalg.norm(sulfur[a].coord-sulfur[b].coord)))
           for a in sulfur for b in sulfur if a<b and np.linalg.norm(sulfur[a].coord-sulfur[b].coord)<2.3]
    return dict(sequence_Cys_positions=cysteines,specified_pairs=links,SG_pairs_below_2_3_A=close,
        maximum_SG_proximity_degree=max((sum(i in p['positions'] for p in close) for i in sulfur),default=0))


def main(args):
    path=(ROOT/args.candidates).resolve();data=load_candidates(path)
    if data['synthetic']:raise ValueError('requires real all-atom structures')
    pairs=[tuple(map(int,p.split(':'))) for p in args.pairs]
    endpoints=[i for pair in pairs for i in pair]
    if any(len(p)!=2 or p[0]==p[1] for p in pairs) or len(endpoints)!=len(set(endpoints)):
        raise ValueError('specified pairs must form a disulfide matching')
    report=dict(synthetic=False,specified_pairs=[list(p) for p in pairs],
        interpretation='measured coordinates only; proximity does not verify a covalent bond, cystine knot, or binding',structures=[])
    parsers={'.cif':MMCIFParser(QUIET=True),'.pdb':PDBParser(QUIET=True)}
    for row in data['candidates']:
        if args.ids and row['id'] not in args.ids:continue
        if min(endpoints)<1 or max(endpoints)>len(row['sequence']):raise ValueError('pair outside binder')
        source=row['metadata']['generation'];records=[]
        if source.get('stage')=='atom14':
            file=ROOT/source['retained_file'] if 'retained_file' in source else path.parent/'generation'/Path(source['file']).name
            records.append((file,'original_atom14',source['original_binder_chain']))
        for r in row['structures']:
            if ('predicted_complex' in r['role'] or 'predicted_binder' in r['role']):
                records.append((ROOT/r['path'],r['role'],r['chain_id']))
        for file,role,chain in records:
            structure=parsers[file.suffix].get_structure('actual',file)
            binder=structure[0][chain]
            if ''.join(seq1(r.resname) for r in binder)!=row['sequence']:raise ValueError('actual binder sequence mismatch')
            report['structures'].append(dict(id=row['id'],role=role,path=file.relative_to(ROOT).as_posix(),**measure(binder,pairs)))
    (ROOT/args.out).write_text(json.dumps(report,indent=2)+'\n')
    for r in report['structures']:
        print(r['id'],r['role'],'Cys',len(r['sequence_Cys_positions']),
            'specified SG distances',[round(p['SG_distance_A'],3) if p['both_Cys_SG_present'] else None for p in r['specified_pairs']])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates');p.add_argument('--pairs',nargs='+',required=True,help='one-based intended Cys pairs, e.g. 6:35 14:44 23:51')
    p.add_argument('--ids',nargs='+');p.add_argument('--out',required=True)
    main(p.parse_args())
