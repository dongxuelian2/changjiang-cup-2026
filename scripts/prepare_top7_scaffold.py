"""Prepare a tag-free observed Top7 core and one exposed redesignable face.

This generic de novo scaffold is not a known TL1A binder. Only original observed
residues 3--94 are retained; the unresolved N-terminal MG and expression tail
are excluded. The crystallographic MSE is explicitly normalized to MET.
"""
import copy
import json
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser,PDBIO,ShrakeRupley,Structure,Model
from Bio.SeqUtils import seq1
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'targets/scaffolds/1QYS.cif'
    chain=copy.deepcopy(MMCIFParser(QUIET=True).get_structure('Top7',source)[0]['A'])
    for r in list(chain):
        if not 3<=r.id[1]<=94 or r.resname=='HOH':chain.detach_child(r.id)
    if len(chain)!=92:raise ValueError('expected exactly 92 observed scaffold residues')
    normalized=[]
    for r in chain:
        if r.resname=='MSE':
            atom=r['SE'];r.detach_child('SE');atom.name='SD';atom.id='SD';atom.fullname=' SD ';atom.element='S';r.add(atom)
            r.resname='MET';normalized.append(r.id[1])
    residues=list(chain)
    for r in residues:chain.detach_child(r.id)
    chain.id='D'
    for i,r in enumerate(residues,1):r.id=(' ',i,' ');chain.add(r)
    sequence=''.join(seq1(r.resname) for r in chain)
    if not set(sequence)<=set('ACDEFGHIKLMNPQRSTVWY'):raise ValueError('nonstandard scaffold residue remains')
    ShrakeRupley(n_points=100).compute(chain,level='R')
    ca=np.array([r['CA'].coord for r in chain])
    eligible=[i for i,r in enumerate(chain) if r.sasa>=40 and 2<=i<=89]
    anchor=37
    selected=sorted(i+1 for i in sorted(eligible,key=lambda i:float(np.linalg.norm(ca[i]-ca[anchor-1])))[:22])
    folder=ROOT/'targets/scaffolds';folder.mkdir(exist_ok=True)
    pose=Structure.Structure('Top7_core');model=Model.Model(0);pose.add(model);model.add(chain)
    writer=PDBIO();writer.set_structure(pose);writer.save(str(folder/'top7_core92.pdb'))
    record=dict(synthetic=False,source_url='https://files.rcsb.org/download/1QYS.cif',
        primary_reference='https://doi.org/10.1126/science.1089427',
        role='generic de novo scaffold; NOT a known TL1A positive binder or validation template',
        source_chain='A',source_author_residues=[3,94],prepared_chain='D',prepared_residues=[1,92],
        modifications=dict(MSE_to_MET_author_positions=normalized,SE_renamed_SD=True,selenium_element_replaced_by_sulfur=True,
                           unresolved_residues_added=False,expression_tail_retained=False),
        sequence=sequence,designable_surface_positions=selected,
        selection='22 nearest CA positions to scaffold residue 37 among nonterminal residues with observed SASA >=40 A^2; no binding score',
        surface_SASA_A2={str(i+1):float(r.sasa) for i,r in enumerate(chain)},
        scaffold_family='Top7 observed core; different generated poses are correlated members of this scaffold family')
    (folder/'top7_core92.json').write_text(json.dumps(record,indent=2)+'\n')
    base=yaml.safe_load((ROOT/'gpu/tl1a_boltzgen_cys6_terminal.yaml').read_text())
    base['entities'][0]={'file':dict(path='../targets/scaffolds/top7_core92.pdb',include=[{'chain':{'id':'D'}}],
        structure_groups=[{'group':{'id':'D','visibility':2}}],
        design=[{'chain':{'id':'D','res_index':','.join(map(str,selected))}}])}
    base.pop('constraints',None)
    # File residue constraints use a separate per-chain schema. IF defaults are
    # retained here; actual Cys content and geometry are measured after generation.
    (ROOT/'gpu/tl1a_boltzgen_top7_face.yaml').write_text(
        '# Generic Top7 core, 22 exposed sequence/backbone positions redesigned.\n'
        '# Native TL1A ABC context; no TL1A-positive binder input.\n'+yaml.safe_dump(base,sort_keys=False))
    print(json.dumps({k:record[k] for k in ('sequence','designable_surface_positions','modifications')},indent=2))


if __name__=='__main__':main()
