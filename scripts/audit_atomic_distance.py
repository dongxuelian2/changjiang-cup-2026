"""Measure GPU nearest-distance error on an actual candidate vs CPU cdist."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from Bio.PDB import PDBParser
    from scipy.spatial.distance import cdist
    if not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU audit required')
    data=json.loads((ROOT/args.candidates).read_text())
    row=next(r for r in data['candidates'] if r['id']==args.id)
    pose=PDBParser(QUIET=True).get_structure('pose',ROOT/row['structures'][0]['path'])
    x=np.array([a.coord for a in pose[0]['D'].get_atoms() if a.element!='H'])
    z=np.load(ROOT/'targets/TL1A/ensemble/states.npz')
    report=dict(synthetic=False,candidate=args.id,reference='scipy.spatial.distance.cdist float64',jax_version=jax.__version__,results={})
    for name,y in [('protein',z['protein'].reshape(-1,3)),('glycan',z['glycan'].reshape(-1,3))]:
        ref=cdist(x,y).min(-1);results={}
        for precision,center in [('default',False),('highest',False),('highest',True)]:
            @jax.jit
            def f(x,y):
                if center:
                    origin=y.mean(0);x=x-origin;y=y-origin
                d=jnp.sum(x*x,-1)[:,None]+jnp.sum(y*y,-1)[None,:]-2*jnp.matmul(x,y.T,precision=precision)
                return jnp.sqrt(jnp.maximum(d.min(-1),1e-6))
            pred=np.array(f(jnp.array(x),jnp.array(y)))
            results[precision+('_centered' if center else '')]=dict(reference_min_A=float(ref.min()),
                calculated_min_A=float(pred.min()),maximum_nearest_distance_error_A=float(np.max(abs(pred-ref))),
                false_atom_clashes2=int(((pred<2)&(ref>=2)).sum()),missed_atom_clashes2=int(((pred>=2)&(ref<2)).sum()))
        report['results'][name]=results
    (ROOT/args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidates');p.add_argument('--id',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
