"""Inspect actual Boltz interface PAE before interpreting cutoff-dependent ipSAE.

The smooth contact summary has a fixed Angstrom scale and is NOT an ipSAE
replacement, official score, or binding probability. No new predictions occur.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from changjiang_select.data import load_candidates


def summarize(pae,contact,scale=10.):
    target_count, binder_count=contact.shape
    if pae.shape!=(target_count+binder_count,)*2 or not np.isfinite(pae).all():
        raise ValueError('PAE shape or finite-value mismatch')
    forward=pae[:target_count,target_count:]
    reverse=pae[target_count:,:target_count].T
    maximum=np.maximum(forward,reverse)
    if not contact.any():raise ValueError('no predicted CA contacts')
    return dict(contact_pairs=int(contact.sum()),
        all_pair_min_PAE_A=dict(target_to_binder=float(forward.min()),binder_to_target=float(reverse.min())),
        contact_PAE_quantiles_A={name:np.quantile(matrix[contact],[0,.1,.5,.9]).tolist()
            for name,matrix in [('target_to_binder',forward),('binder_to_target',reverse),('reciprocal_max',maximum)]},
        contact_reciprocal_fraction_below={str(cutoff):float((maximum[contact]<cutoff).mean()) for cutoff in (8,10,12,15,20)},
        all_pair_directional_counts_below_10=dict(target_to_binder=int((forward<10).sum()),binder_to_target=int((reverse<10).sum())),
        fixed_scale_smooth_contact_support=float((1/(1+(maximum[contact]/scale)**2)).mean()))


def main(args):
    data=load_candidates(ROOT/args.candidates)
    if data['synthetic']:raise ValueError('requires real model PAE')
    target=''.join((ROOT/'targets/TL1A/construct.fasta').read_text().splitlines()[1:])
    report=dict(synthetic=False,purpose='diagnose actual PAE distributions; NOT official scoring or measured binding',
        contact_definition='predicted CA distance <10 A, all three target protomers',
        quantiles=[0,.1,.5,.9],smooth_scale_A=10.,predictions=[])
    parser=MMCIFParser(QUIET=True)
    for row in data['candidates']:
        if args.ids and row['id'] not in args.ids:continue
        for record in row['structures']:
            role=record['role']
            if not role.startswith('boltz_trimer_deepmsa') or not role.endswith('_predicted_complex'):continue
            file=ROOT/record['path'];prefix=role.removesuffix('_predicted_complex')
            structure=parser.get_structure('prediction',file)
            if list(c.id for c in structure[0])!=list('ABCD'):
                raise ValueError('unexpected actual trimer chain order')
            if any(''.join(seq1(r.resname) for r in structure[0][c])!=target for c in 'ABC'):
                raise ValueError('not the exact full official target')
            binder=structure[0]['D']
            if ''.join(seq1(r.resname) for r in binder)!=row['sequence']:raise ValueError('binder mismatch')
            tc=np.array([r['CA'].coord for c in 'ABC' for r in structure[0][c]])
            bc=np.array([r['CA'].coord for r in binder])
            contact=np.linalg.norm(tc[:,None]-bc,axis=-1)<10
            pae_file=file.with_name('pae_'+file.stem+'.npz')
            pae=np.load(pae_file)['pae']
            report['predictions'].append(dict(id=row['id'],namespace=prefix,
                pae=pae_file.relative_to(ROOT).as_posix(),
                ipsae_10=row['metrics'][prefix+'.ipsae_10'],
                ipsae_15=row['metrics'][prefix+'.ipsae_15'],**summarize(pae,contact)))
    (ROOT/args.out).write_text(json.dumps(report,indent=2)+'\n')
    for p in report['predictions']:
        print(p['id'],p['namespace'],'ipSAE10',round(p['ipsae_10'],3),
            'contact PAE median',round(p['contact_PAE_quantiles_A']['reciprocal_max'][2],2),
            'smooth',round(p['fixed_scale_smooth_contact_support'],3),
            'min directional',p['all_pair_min_PAE_A'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates');p.add_argument('--ids',nargs='+');p.add_argument('--out',required=True)
    main(p.parse_args())
