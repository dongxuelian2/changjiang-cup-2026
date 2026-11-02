"""Allocate native neighbor fragments using actual false-pose clash probes.

Reuses the existing coverage objective and solvers. Coverage concerns the
observed clash points only; it is neither full context equivalence nor KD utility.
"""
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser

from changjiang_select.data import load_candidates
from changjiang_select.objective import ScenarioObjective
from changjiang_select.optimize import portfolio,singleton_top_k,exhaustive

ROOT=Path(__file__).resolve().parents[1]


def select_occlusion_context(native,probes,residue_budget):
    data=load_candidates(ROOT/probes)
    if data['synthetic']:
        raise ValueError('requires real failed-model poses')
    points=[];sources=[]
    for row in data['candidates']:
        if not row['metrics'].get('geometry.native_all_chain_BB_clash_pairs2',0):
            continue
        source=ROOT/row['structures'][0]['path']
        pose=PDBParser(QUIET=True).get_structure('probe',source)
        points.extend(a.coord for a in pose[0]['D'].get_atoms() if a.name in ('N','CA','C','O'))
        sources.append(source.relative_to(ROOT).as_posix())
    if not points:
        raise ValueError('no actual failed-pose backbone points')
    points=np.array(points)
    selected={};report=dict(purpose='feature-context coverage of actual clash probes, NOT binding utility',
        source_poses=sources,motif_length=5,residue_budget_per_neighbor=residue_budget,
        probe_clash_cutoff_A=2.5,feature_coverage_cutoff_A=4.,protomers={})
    for c in 'BC':
        residues=list(native[0][c])
        atoms=np.array([a.coord for r in residues for a in r])
        d=np.linalg.norm(points[:,None]-atoms,axis=-1).min(-1)
        bad=points[d<2.5]
        if not len(bad):
            raise ValueError(f'no observed clash probes for protomer {c}; cannot invent its coverage objective')
        motifs=[residues[i:i+5] for i in range(0,len(residues),5)]
        u=np.array([np.linalg.norm(bad[:,None]-np.array([a.coord for r in m for a in r]),axis=-1).min(-1)<4
                    for m in motifs],dtype=float)
        ids=tuple(f'{c}:{m[0].id[1]}-{m[-1].id[1]}' for m in motifs)
        obj=ScenarioObjective(ids,u,np.ones(len(bad))/len(bad),report['purpose'],False)
        k=min(residue_budget//5,len(motifs))
        if not k:
            raise ValueError('at least one five-residue motif per neighbor is needed')
        greedy_portfolio=portfolio(obj,k);top=singleton_top_k(obj,k);exact=exhaustive(obj,k)
        chosen=exact  # Small enough to solve exactly; use the better computed set.
        selected[c]={r.id[1] for i in chosen for r in motifs[i]}
        report['protomers'][c]=dict(clashing_backbone_points=len(bad),motifs=len(motifs),
            selected_motifs=[ids[i] for i in chosen],selected_residues=sorted(selected[c]),
            selection_method='exact',
            objective=dict(portfolio=obj.value(greedy_portfolio),singleton_topk=obj.value(top),exact=obj.value(exact)),
            exact_motifs=[ids[i] for i in exact],
            caveat='fixed selected false poses; no guarantee for new poses or neural prediction fidelity')
    return selected,report
