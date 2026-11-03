"""CPU audit of actual discrete AF2 generation losses and source-fold damage."""
import argparse
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser,Superimposer
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    run=ROOT/args.run;meta=json.loads((run/'run.json').read_text())
    if meta['status']!='generated_not_independently_validated':raise ValueError('requires completed real generation')
    e=np.load(ROOT/'targets/TL1A/ensemble/states.npz')
    protein=np.concatenate([e['protein'],e['unoccupied_protein']]).reshape(-1,3)
    glycan=e['glycan'].reshape(-1,3)
    p=PDBParser(QUIET=True);scaffold=p.get_structure('source',ROOT/'targets/scaffolds/a3d73.pdb')[0]['D']
    report=dict(synthetic=False,run=args.run,reference='CPU scipy cdist on saved discrete all-heavy-atom PDB; no target realignment after saving',phases={})
    for phase,folder,log in [('bound','bound',meta['bound_discrete_log']),
                             ('released',f"seed_{meta['trajectories'][0]['seed']}",meta['trajectories'][0]['final_generator_log'])]:
        pose=p.get_structure('generated',run/folder/'generated.pdb');binder=pose[0]['B']
        xyz=np.array([a.coord for a in binder.get_atoms() if a.element!='H'])
        dp=float(cdist(xyz,protein).min());dg=float(cdist(xyz,glycan).min())
        losses=dict(native_ensemble_steric=max(2-dp,0)**2,glycan_ensemble_steric=max(2.3-dg,0)**2)
        errors={k:abs(v-float(log[k])) for k,v in losses.items()}
        fit=Superimposer();fit.set_atoms([r['CA'] for r in scaffold],[r['CA'] for r in binder])
        report['phases'][phase]=dict(native_protein_min_A=dp,modelled_glycan_min_A=dg,cpu_losses=losses,
            logged_gpu_losses={k:log[k] for k in losses},absolute_loss_errors=errors,
            fold_to_original_scaffold_CA_rmsd_A=float(fit.rms),binder_plddt=log['plddt'],iptm=log['i_ptm'])
        if max(errors.values())>.02:raise ValueError(f'CPU and GPU atomic loss disagree in {phase}')
    (ROOT/args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--out',required=True)
    main(p.parse_args())
