"""Inspect the actual pinned BoltzGen feature tensors, without loading weights."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from boltzgen.model.modules.masker import BoltzMasker
    cfg=OmegaConf.load(ROOT/args.config)
    module=instantiate(cfg.data,extra_features=['structure','tokenized'])
    features=module.predict_set[0]
    tokens=features['tokenized'].tokens
    structure=features['structure']
    def array(name):return features[name].detach().cpu().numpy()
    chains={int(c['asym_id']):str(c['name']) for c in structure.chains}
    binder_id=next(i for i,c in chains.items() if c==args.binder)
    binder=np.flatnonzero(tokens['asym_id']==binder_id)
    cys=[int(i) for i in binder if tokens[i]['res_name']=='CYS']
    bonds=array('token_bonds').squeeze(-1)
    pairs=[(int(a),int(b)) for a,b in np.argwhere(np.triu(bonds>0,1)) if a in binder and b in binder]
    target=np.array([i for i in range(len(tokens)) if i not in set(binder)])
    visibility=array('token_distance_mask')
    batch=module.collate([features])
    masked=BoltzMasker(**OmegaConf.to_container(cfg.override.masker_args))(batch)
    model_binary=masked['token_bonds'][0].detach().cpu().numpy().squeeze(-1)
    model_typed=masked['type_bonds'][0].detach().cpu().numpy()
    record=dict(synthetic=False,source='actual upstream data module / featurizer; no model execution',
        config=args.config,chains={name:int((tokens['asym_id']==i).sum()) for i,name in chains.items()},
        binder=args.binder,binder_tokens=len(binder),sequence_design_tokens=int(array('design_mask')[binder].sum()),
        coordinate_design_tokens=int(array('chain_design_mask')[binder].sum()),
        sequence_design_positions=[int(tokens[i]['res_idx'])+1 for i in binder if array('design_mask')[i]],
        fixed_Cys_positions=[int(tokens[i]['res_idx'])+1 for i in cys],
        binder_bond_features=[[int(tokens[a]['res_idx'])+1,int(tokens[b]['res_idx'])+1] for a,b in pairs],
        binder_cyclic_feature_values=np.unique(array('cyclic')[binder]).tolist(),
        binder_structure_groups=np.unique(tokens['structure_group'][binder]).tolist(),
        binder_structure_group_counts={str(int(g)):int((tokens['structure_group'][binder]==g).sum())
            for g in np.unique(tokens['structure_group'][binder])},
        distance_conditioning=dict(binder_internal_fraction=float(visibility[np.ix_(binder,binder)].mean()),
            binder_target_fraction=float(visibility[np.ix_(binder,target)].mean()),
            interpretation='nonzero binder-target visibility reveals an input relative pose'),
        after_actual_input_masker=dict(binary_bond_values_at_specified_pairs=[float(model_binary[a,b]) for a,b in pairs],
            typed_bond_values_at_specified_pairs=[int(model_typed[a,b]) for a,b in pairs],
            interpretation='upstream masker zeros binary bonds but retains typed bonds; typed use depends on checkpoint bond_type_feature'))
    if args.inspect_checkpoint:
        import torch
        checkpoint=torch.load(cfg.checkpoint,map_location='cpu',weights_only=False,mmap=True)
        record['checkpoint_bond_type_feature']=checkpoint['hyper_parameters']['bond_type_feature']
    if args.require_free_pose and record['distance_conditioning']['binder_target_fraction']:
        raise ValueError('input leaks a binder-target relative pose; use distinct positive structure groups')
    (ROOT/args.out).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config');p.add_argument('--binder',default='D');p.add_argument('--out',required=True)
    p.add_argument('--require-free-pose',action='store_true')
    p.add_argument('--inspect-checkpoint',action='store_true',help='memory-map genuine checkpoint hyperparameters on CPU; no prediction')
    main(p.parse_args())
