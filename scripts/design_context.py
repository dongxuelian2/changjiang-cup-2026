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


def select_pose_anchors(target_ca, binder_ca, eligible, k=6, cutoff=20.):
    """Greedy log-det information for LOCAL rigid-pose distance constraints.

    Translation is in Angstroms; rotation is scaled by binder Rg so both
    Jacobian blocks are dimensionless. This is a geometric linearization,
    not predictor uncertainty or a calibrated binding objective.
    """
    target_ca, binder_ca = np.asarray(target_ca), np.asarray(binder_ca)
    center = binder_ca.mean(0)
    rg = np.sqrt(np.mean(np.sum((binder_ca-center)**2, axis=-1)))
    if rg <= 0 or k <= 0:
        raise ValueError('nondegenerate binder and positive anchor budget required')
    matrices, indices = [], []
    for index in eligible:
        delta = binder_ca-target_ca[index]
        distance = np.linalg.norm(delta, axis=-1)
        use = (distance < cutoff) & (distance > 1e-6)
        if use.sum() < 4:
            continue
        normal = delta[use]/distance[use, None]
        jacobian = np.concatenate([normal, np.cross((binder_ca[use]-center)/rg, normal)], axis=-1)
        matrices.append(jacobian.T@jacobian/len(jacobian))
        indices.append(int(index))
    if len(indices) < k:
        raise ValueError('not enough eligible native anchors around this binder')
    matrices = np.array(matrices)
    information = np.eye(6)
    chosen, gains = [], []
    for _ in range(k):
        baseline = np.linalg.slogdet(information)[1]
        values = [np.linalg.slogdet(information+m)[1]-baseline if i not in chosen else -np.inf
                  for i, m in enumerate(matrices)]
        best = int(np.argmax(values))
        chosen.append(best)
        gains.append(float(values[best]))
        information += matrices[best]
    selected = [indices[i] for i in chosen]
    return selected, dict(method='greedy logdet(I + sum local rigid-distance information)',
        selected_target_indices=selected, eligible_target_indices=indices,
        binder_Rg_A=float(rg), pair_cutoff_A=cutoff, ridge=1., marginal_gains=gains,
        logdet=float(np.linalg.slogdet(information)[1]),
        unregularized_information_eigenvalues=np.linalg.eigvalsh(information-np.eye(6)).tolist(),
        candidate_information_matrices=matrices.tolist(),
        interpretation='local geometric pose observability, NOT binding probability or neural prediction fidelity')


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
