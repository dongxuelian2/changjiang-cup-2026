"""New alpha3D scaffold family: protected packing/caps, redesigned surface.

2A3D is a generic de novo fold, not a known TL1A binder. No sequence or
coordinates from any previously evaluated binder family are reused.
"""
import copy
import json
from pathlib import Path

from Bio.PDB import MMCIFParser,PDBIO,ShrakeRupley,Structure,Model
from Bio.SeqUtils import seq1
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'targets/scaffolds'
    source=folder/'2A3D.cif'
    chain=copy.deepcopy(MMCIFParser(QUIET=True).get_structure('a3d',source)[0]['A'])
    for r in chain:
        for a in list(r):
            if a.element=='H':r.detach_child(a.id)
    if len(chain)!=73:raise ValueError('require the complete 73-residue alpha3D fold')
    chain.id='D'
    sequence=''.join(seq1(r.resname) for r in chain)
    ShrakeRupley(n_points=100).compute(chain,level='R')
    packing={i+1 for i,r in enumerate(chain) if r.sasa<35}
    caps={1,2,22,46,48,50,72,73}
    fixed=packing|caps;variable=[i for i in range(1,74) if i not in fixed]
    pose=Structure.Structure('a3d');model=Model.Model(0);pose.add(model);model.add(chain)
    writer=PDBIO();writer.set_structure(pose);writer.save(str(folder/'a3d73.pdb'))
    record=dict(synthetic=False,source_url='https://files.rcsb.org/download/2A3D.cif',
        role='new generic de novo scaffold family; not a TL1A positive binder',
        sequence=sequence,source_chain='A',prepared_chain='D',packing_positions=sorted(packing),
        protected_caps=sorted(caps),fixed_positions=sorted(fixed),design_positions=variable,
        SASA_A2={str(i+1):float(r.sasa) for i,r in enumerate(chain)},
        selection='keep observed SASA<35 A^2 packing positions and selected terminal/turn caps; redesign the remaining surface',
        scaffold_family='alpha3D73; generated poses share a generic fold and are correlated')
    (folder/'a3d73.json').write_text(json.dumps(record,indent=2)+'\n')
    spec=yaml.safe_load((ROOT/'gpu/tl1a_boltzgen_top7_face.yaml').read_text())
    spec['entities'][0]={'file':dict(path='../targets/scaffolds/a3d73.pdb',
        include=[{'chain':{'id':'D'}}],structure_groups=[{'group':{'id':'D','visibility':2}}],
        design=[{'chain':{'id':'D','res_index':','.join(map(str,variable))}}])}
    spec['entities'][1]['file']['binding_types'][0]['chain']['binding']='86,165,168'
    (ROOT/'gpu/tl1a_boltzgen_a3d_ensemble.yaml').write_text(
        '# New alpha3D73 scaffold family; R156/L235/Y238 target patch.\n'
        '# Native ABC context. Glycan/sidechain ensembles are separate physical design constraints.\n'+yaml.safe_dump(spec,sort_keys=False))
    print(json.dumps(dict(sequence=sequence,fixed=len(fixed),designed=len(variable)),indent=2))


if __name__=='__main__':main()
