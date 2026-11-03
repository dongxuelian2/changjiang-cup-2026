"""Stress real generated/predicted poses against native SC/modelled sugars.

Explicit glycan distances are an extra geometric diagnostic, not an affinity
score, glycan-population average, or replacement for the existing ND2 gate.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from Bio.PDB import PDBParser,MMCIFParser
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.import_boltz import align_prediction


def measure(chain,ensemble):
    xyz=np.array([a.coord for a in chain.get_atoms() if a.element!='H'])
    result=[]
    for i,(protein,glycans) in enumerate(zip(ensemble['protein'],ensemble['glycan'])):
        p=cdist(xyz,protein);g=cdist(xyz,glycans.reshape(-1,3))
        roots={f'{c}:N{n}':dict(min_A=float(cdist(xyz,block).min()),pairs23=int((cdist(xyz,block)<2.3).sum()))
               for (c,n),block in zip(((c,n) for c in 'ABC' for n in (133,229)),glycans)}
        result.append(dict(state=i,native_protein_min_A=float(p.min()),native_protein_pairs2=int((p<2).sum()),
            modelled_glycan_min_A=float(g.min()),modelled_glycan_pairs23=int((g<2.3).sum()),glycan_roots=roots))
    return dict(atom_count=len(xyz),states=result,
        worst_native_min_A=min(r['native_protein_min_A'] for r in result),
        worst_glycan_min_A=min(r['modelled_glycan_min_A'] for r in result),
        maximum_native_pairs2=max(r['native_protein_pairs2'] for r in result),
        maximum_glycan_pairs23=max(r['modelled_glycan_pairs23'] for r in result))


def main(args):
    data=json.loads((ROOT/args.candidates).read_text())
    if data['synthetic']:raise ValueError('requires real structures')
    ensemble=np.load(ROOT/'targets/TL1A/ensemble/states.npz')
    native=PDBParser(QUIET=True).get_structure('native',ROOT/'targets/TL1A/observed.pdb')
    manifest=json.loads((ROOT/'targets/manifest.json').read_text())
    core=next(t['beta_core_uniprot'] for t in manifest['targets'] if t['target_id']=='TL1A')
    report=dict(synthetic=False,ensemble='targets/TL1A/ensemble/ensemble.json',
        interpretation='finite geometric stress states, no glycoform occupancy or probability; design poses have backbone only',candidates={})
    for row in data['candidates']:
        poses={};design=PDBParser(QUIET=True).get_structure('design',ROOT/row['structures'][0]['path'])
        poses['design_backbone']=measure(design[0]['D'],ensemble)
        for record in row['structures'][1:]:
            role=record['role']
            if not role.endswith('_predicted_complex'):continue
            file=ROOT/record['path'];parser=MMCIFParser(QUIET=True) if file.suffix=='.cif' else PDBParser(QUIET=True)
            prediction=parser.get_structure('prediction',file)
            for chain in prediction[0]:
                residues=list(chain)
                for r in residues:chain.detach_child(r.id)
                for i,r in enumerate(residues,1):r.id=(' ',i,' ');chain.add(r)
            align_prediction(native,prediction,design[0]['D'],'ABC','D',core)
            poses[role]=measure(prediction[0]['D'],ensemble)
        report['candidates'][row['id']]=poses
    out=ROOT/args.out;out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{p:{q:r[q] for q in ('worst_native_min_A','worst_glycan_min_A','maximum_native_pairs2','maximum_glycan_pairs23')} for p,r in v.items()} for k,v in report['candidates'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidates');p.add_argument('--out',required=True)
    main(p.parse_args())
