"""Finite native-sidechain / modelled N-glycan stress ensemble for TL1A.

R156/M158/L235/Y238 sidechains come from the three observed 2RE9 protomers.
Man3GlcNAc2 cores and linked Asn geometry come from 4BYH, then are grafted and
rotated. These are modelled TL1A glycan states, NOT measured TL1A glycoforms or
Boltzmann samples. No antibody/receptor sequence is supplied to binder design.
"""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from Bio.PDB import MMCIFParser,PDBParser,PDBIO,Superimposer,Chain,Residue,Atom
from Bio.SeqUtils import seq1

ROOT=Path(__file__).resolve().parents[1]
SC=('CB','CG','OD1','ND2')
BB={'N','CA','C','O','OXT'}
SITES=[(c,n) for c in 'ABC' for n in (133,229)]


def rotate(x,origin,axis,degrees):
    unit=axis/np.linalg.norm(axis)
    return (x-origin)@Rotation.from_rotvec(unit*np.radians(degrees)).as_matrix().T+origin


def sidechain_states(native):
    states=[]
    for offset in range(3):
        state=copy.deepcopy(native)
        for i,c in enumerate('ABC'):
            source='ABC'[(i+offset)%3]
            for n in (156,158,235,238):
                a,b=native[0][c][n],native[0][source][n]
                fit=Superimposer();fit.set_atoms([a[k] for k in ('N','CA','C')],[b[k] for k in ('N','CA','C')])
                r,t=fit.rotran
                for atom in state[0][c][n]:
                    if atom.name not in BB:atom.coord=b[atom.name].coord@r+t
        states.append(state)
    return states


def templates():
    source=ROOT/'build/sources/4BYH.cif'
    structure=MMCIFParser(QUIET=True).get_structure('4byh',source) if source.exists() else None
    result=[];folder=ROOT/'targets/glycans';folder.mkdir(exist_ok=True)
    for protein,sugar,branch in (('A','C',8),('B','D',7)):
        if structure is None:
            extracted=PDBParser(QUIET=True).get_structure('core',folder/f'core_4byh_{sugar}.pdb')
            asn=extracted[0]['Z'][297];residues=list(extracted[0]['G'])
        else:
            asn=structure[0][protein][297]
            residues=[next(r for r in structure[0][sugar] if r.id[1]==i) for i in (1,2,3,4,branch)]
        labels=[dict(residue=i+1,ccd=r.resname,atom=a.name,element=a.element)
                for i,r in enumerate(residues) for a in sorted(r,key=lambda a:a.name) if a.element!='H']
        coords=np.array([a.coord for r in residues for a in sorted(r,key=lambda a:a.name) if a.element!='H'])
        c1=next(i for i,a in enumerate(labels) if a['residue']==1 and a['atom']=='C1')
        distance=float(np.linalg.norm(coords[c1]-asn['ND2'].coord))
        if not 1.3<distance<1.6:raise ValueError('source glycosidic link is not chemically plausible')
        # Author residue order does not encode alpha1-3/alpha1-6 identity:
        # 4BYH C and D number these two Man arms in opposite orders.
        links=[dict(child=i,parent=j,parent_atom=a,distance_A=float(np.linalg.norm(residues[i-1]['C1'].coord-residues[j-1][a].coord)))
               for i,j,a in ((2,1,'O4'),(3,2,'O4'))]
        for i in (4,5):
            a=min(('O3','O6'),key=lambda a:np.linalg.norm(residues[i-1]['C1'].coord-residues[2][a].coord))
            links.append(dict(child=i,parent=3,parent_atom=a,distance_A=float(np.linalg.norm(residues[i-1]['C1'].coord-residues[2][a].coord))))
        if not all(1.3<r['distance_A']<1.6 for r in links) or {r['parent_atom'] for r in links[2:]}!={'O3','O6'}:
            raise ValueError('source common-core internal link topology invalid')
        result.append(dict(protein=protein,sugar=sugar,asn=asn,labels=labels,coords=coords,c1=c1,source_link_A=distance,internal_links=links))
        # Retain just the generic glycan and one attachment Asn, not Fc sequence.
        output=copy.deepcopy(structure if structure is not None else extracted)
        for c in list(output[0]):output[0].detach_child(c.id)
        ac=Chain.Chain('Z');ac.add(asn.copy());output[0].add(ac)
        gc=Chain.Chain('G')
        for i,res in enumerate(residues,1):
            res=res.copy();res.id=('H_'+res.resname,i,' ');gc.add(res)
        output[0].add(gc);writer=PDBIO();writer.set_structure(output);writer.save(str(folder/f'core_4byh_{sugar}.pdb'))
    if result[0]['labels']!=result[1]['labels']:raise ValueError('glycan template atom order differs')
    return result


def main():
    folder=ROOT/'targets/TL1A/ensemble';folder.mkdir(exist_ok=True)
    native=PDBParser(QUIET=True).get_structure('native',ROOT/'targets/TL1A/observed.pdb')
    states=sidechain_states(native);source=templates()
    keys=[(c.id,r.id[1],a.name) for c in native[0] for r in c for a in r if a.element!='H']
    base=np.array([[state[0][c][n][a].coord for c,n,a in keys] for state in states])
    removed={(c,n,a) for c,n in SITES for a in SC}
    keep=np.array([k not in removed for k in keys]);trees=[cKDTree(x[keep]) for x in base]
    root_options=[];root_meta=[]
    for site_index,(c,n) in enumerate(SITES):
        target=native[0][c][n];options=[]
        sc_keep=np.array([k not in removed and k[:2]!=(c,n) for k in keys])
        sc_trees=[cKDTree(x[sc_keep]) for x in base]
        for template in source:
            fit=Superimposer();fit.set_atoms([target[a] for a in ('N','CA','C')],[template['asn'][a] for a in ('N','CA','C')])
            r,t=fit.rotran
            sc0=np.array([template['asn'][a].coord for a in SC])@r+t
            gly0=template['coords']@r+t
            for chi1,chi2,phi in itertools.product(range(-180,180,30),range(-180,180,30),(0,-30,30,-60,60,-90,90)):
                sc=rotate(sc0,target['CA'].coord,sc0[0]-target['CA'].coord,chi1)
                gly=rotate(gly0,target['CA'].coord,sc0[0]-target['CA'].coord,chi1)
                gly=rotate(gly,sc[0],sc[1]-sc[0],chi2);sc=rotate(sc,sc[0],sc[1]-sc[0],chi2)
                gly=rotate(gly,sc[3],gly[template['c1']]-sc[3],phi)
                if not 1.4<np.linalg.norm(sc[0]-target['CA'].coord)<1.65:continue
                # Source rigid geometry preserves the ND2--C1 link and rings.
                near=[min(float(tree.query(gly)[0].min()),float(stree.query(sc)[0].min())) for tree,stree in zip(trees,sc_trees)]
                permitted=[v>=1.8 for v in near]
                if not any(permitted):continue
                options.append(dict(sc=sc,gly=gly,allowed=permitted,clearance=near,
                    template=template['sugar'],chi1=chi1,chi2=chi2,phi=phi,
                    source_link_A=template['source_link_A']))
        print(f'{c}:{n}: {len(options)} viable sampled states',flush=True)
        if not options:raise ValueError(f'no sterically viable modelled glycan state at {c}:{n}')
        # A low-rotation seed and farthest-point footprint representatives.
        initial=min(range(len(options)),key=lambda i:abs(options[i]['chi1'])+abs(options[i]['chi2'])+abs(options[i]['phi']))
        chosen=[initial]
        for _ in range(min(3,len(options))-1):
            chosen.append(max((i for i in range(len(options)) if i not in chosen),key=lambda i:
                min(float(np.mean((options[i]['gly']-options[j]['gly'])**2)) for j in chosen)))
        root_options.append([options[i] for i in chosen])
        root_meta.append(dict(chain=c,uniprot=n,valid_sampled_states=len(options),
            representatives=[{k:v for k,v in options[i].items() if k not in ('sc','gly')} for i in chosen]))
    # Pairwise glycan/Asn compatibility is computed before selecting snapshots.
    distances={}
    for i in range(6):
        for j in range(i):
            for a,b in itertools.product(range(len(root_options[i])),range(len(root_options[j]))):
                p=np.concatenate([root_options[i][a]['sc'],root_options[i][a]['gly']])
                q=np.concatenate([root_options[j][b]['sc'],root_options[j][b]['gly']])
                distances[i,j,a,b]=float(cKDTree(p).query(q)[0].min())
    legal=[]
    for rs in range(3):
        for assignment in itertools.product(*(range(len(x)) for x in root_options)):
            if not all(root_options[i][a]['allowed'][rs] for i,a in enumerate(assignment)):continue
            if not all(distances[i,j,assignment[i],assignment[j]]>=1.8 for i in range(6) for j in range(i)):continue
            legal.append((rs,)+assignment)
    if not legal:raise ValueError('no compatible fully occupied glycan/sidechain snapshots')
    pool=set(legal);selected=None
    for first in legal if all(len(x)==3 for x in root_options) else []:
        if first[0]!=0:continue
        for second_g in itertools.product(*[[a for a in range(3) if a!=x] for x in first[1:]]):
            second=(1,)+second_g
            third=(2,)+tuple(3-a-b for a,b in zip(first[1:],second_g))
            if second in pool and third in pool:selected=[first,second,third];break
        if selected:break
    universe={(i,v) for s in legal for i,v in enumerate(s)}
    if selected is None:
        selected=[];covered=set()
        while covered!=universe:
            best=max(legal,key=lambda s:len(set(enumerate(s))-covered))
            selected.append(best);covered.update(enumerate(best))
    coverage=set().union(*(set(enumerate(s)) for s in selected))
    if coverage!=universe:raise ValueError('selected snapshots missed feasible marginal states')
    proteins=[];glycans=[]
    for index,snapshot in enumerate(selected):
        state=copy.deepcopy(states[snapshot[0]])
        gly=[]
        for i,(c,n) in enumerate(SITES):
            option=root_options[i][snapshot[i+1]]
            for a,x in zip(SC,option['sc']):state[0][c][n][a].coord=x
            gc=Chain.Chain('EFGHIJ'[i]);gly.append(option['gly'])
            for label,x in zip(source[0]['labels'],option['gly']):
                rid=('H_'+label['ccd'],label['residue'],' ')
                if rid not in gc:gc.add(Residue.Residue(rid,label['ccd'],''))
                gc[rid].add(Atom.Atom(label['atom'],x,0.,1.,' ',label['atom'].rjust(4),0,element=label['element']))
            state[0].add(gc)
        expected=''.join(seq1(r.resname) for r in native[0]['A'])
        if any(''.join(seq1(r.resname) for r in state[0][c])!=expected for c in 'ABC'):raise ValueError('target sequence changed')
        proteins.append(np.array([state[0][c][n][a].coord for c,n,a in keys]));glycans.append(gly)
        writer=PDBIO();writer.set_structure(state);writer.save(str(folder/f'state_{index}.pdb'))
    np.savez_compressed(folder/'states.npz',protein=np.array(proteins),glycan=np.array(glycans),
        unoccupied_protein=base,protein_keys=np.array(keys,dtype=str))
    record=dict(synthetic=False,modelled=True,
        purpose='finite geometric stress ensemble; not TL1A experimental glycans, populations, or measured binding',
        glycan_source='https://www.rcsb.org/structure/4BYH',
        glycan_model='Man3GlcNAc2 common core only; outer branches/glycoform and occupancy are not known',
        source_glycans=['C:1,2,3,4,8','D:1,2,3,4,7'],
        source_internal_links={t['sugar']:t['internal_links'] for t in source},
        atom_order='Man arms numbered in source residue order, not canonical linkage order; diversity heuristic is label-dependent; clash point sets are label-invariant',
        native_sidechain_source='observed 2RE9 ABC, locally transferred R156/M158/L235/Y238 states',
        rotations_degrees=dict(Asn_chi1=list(range(-180,180,30)),Asn_chi2=list(range(-180,180,30)),glycosidic_phi=[0,-30,30,-60,60,-90,90]),
        geometric_clash_filter_A=1.8,sites=root_meta,compatible_joint_snapshots=len(legal),
        full_cartesian_assignments=3*int(np.prod([len(x) for x in root_options])),selected_assignments=[list(s) for s in selected],
        feasible_marginal_states=len(universe),covered_marginal_states=len(coverage),
        selection='exact three-snapshot marginal cover when feasible; otherwise greedy feasible-marginal set cover',
        proved_minimum_snapshots=len(selected) if len(selected)==max(len({s[i] for s in legal}) for i in range(7)) else None,
        maximum_single_pair_clash_preserved=True,
        limits='marginal coverage preserves worst single-pair sterics, not sums of energies or statistical joint populations')
    (folder/'ensemble.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('compatible_joint_snapshots','selected_assignments','feasible_marginal_states','covered_marginal_states','proved_minimum_snapshots')},indent=2))


if __name__=='__main__':main()
