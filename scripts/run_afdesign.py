"""De novo AF2 gradient design with native-trimer geometric context.

ColabDesign uses one complete protomer and optional mapped neighbor fragments
plus a new binder to fit the 8 GB GPU. All observed native protomers and six glycan anchors enter additional geometric
losses. Generation uses a target template; independent screening must NOT.
This is a ColabDesign experiment, not the complete BindCraft pipeline.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
    os.environ.setdefault("TF_FORCE_UNIFIED_MEMORY", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    out = (ROOT / args.out).resolve()
    if out.exists():
        raise ValueError("choose a fresh output directory")
    out.mkdir(parents=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.SeqUtils import seq1
    from scipy.spatial import cKDTree
    from colabdesign import mk_afdesign_model
    from colabdesign.af.loss import get_con_loss

    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("GPU required; no accidental CPU design")
    target_file = ROOT / "targets/TL1A/monomer_A.pdb"
    native = PDBParser(QUIET=True).get_structure("native", ROOT / "targets/TL1A/observed.pdb")
    target = list(native[0]["A"])
    expected = "".join((ROOT / "targets/TL1A/construct.fasta").read_text().splitlines()[1:])[16:]
    if "".join(seq1(r.resname) for r in target) != expected:
        raise ValueError("native target does not match official sequence 88-251")
    sites = {"terminal": [235,238], "beta": [156,158], "outer": [110,123]}
    hotspot = sites[args.site]
    ca = np.array([r["CA"].coord for r in target])
    hs = np.array([native[0]["A"][n]["CA"].coord for n in hotspot])
    sasa = {int(r["uniprot_residue"]): float(r["sasa_trimer_A2"])
            for r in csv.DictReader((ROOT / "targets/TL1A/surface.csv").open())}
    patch = [i for i,r in enumerate(target) if sasa[r.id[1]] >= 20 and
             np.linalg.norm(ca[i]-hs, axis=-1).min() <= 12]
    selected={c:{r.id[1] for r in native[0][c]} if c=='A' else set() for c in 'ABC'}
    context_allocation=None
    if args.context_neighbors:
        if args.clash_probes:
            from scripts.design_context import select_occlusion_context
            neighbor_selection,context_allocation=select_occlusion_context(native,args.clash_probes,args.context_neighbors)
            selected.update(neighbor_selection)
        else:
            for c in 'BC':
                distances=[(np.linalg.norm(np.array([a.coord for a in r])[:,None]-hs,axis=-1).min(),r.id[1])
                           for r in native[0][c]]
                selected[c]={n for _,n in sorted(distances)[:args.context_neighbors]}
        class Context(Select):
            def accept_residue(self,residue):
                return residue.id[1] in selected[residue.parent.id]
        target_file=out/'design_context.pdb'
        writer=PDBIO();writer.set_structure(native);writer.save(str(target_file),Context())
    target_map=[dict(chain=c,uniprot=r.id[1]) for c in 'ABC' for r in native[0][c] if r.id[1] in selected[c]]
    context_ca=np.array([native[0][r['chain']][r['uniprot']]['CA'].coord for r in target_map])
    T, L = len(target_map), len(target_map)+args.length
    nx = jnp.array([a.coord for a in native.get_atoms() if a.element != "H"])
    gx = jnp.array([native[0][c][n]["ND2"].coord for c in "ABC" for n in (133,229)])
    target_ca = jnp.array(context_ca)
    binder_mask = jnp.arange(L) >= T
    patch_mask = jnp.zeros(L).at[np.array(patch)].set(1)
    weights = dict(plddt=.1, pae=.4, con=.5, i_con=1., i_pae=.5,
                   interface_cover=1., native_target=.1, context_steric=3., glycan=1.)
    contact = dict(num=2, cutoff=8., binary=True, num_pos=12)
    # A solvent-side center is a target-derived region, not a binder template.
    # Search directions geometrically, then retain only centers clear of the
    # native trimer and the glycan envelope. Detailed pointwise checks remain.
    h=np.mean([a.coord for n in hotspot for a in native[0]['A'][n]
               if a.name not in ('N','CA','C','O')],axis=0)
    k=np.arange(512);z=1-2*(k+.5)/512;phi=k*np.pi*(3-np.sqrt(5))
    directions=np.array([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z]).T
    centers=h+14*directions
    clearance=cKDTree(np.asarray(nx)).query(centers)[0]
    glycan_clearance=np.linalg.norm(centers[:,None]-np.asarray(gx),axis=-1).min(-1)
    normal=h-ca.mean(0);normal/=np.linalg.norm(normal)
    feasible=(clearance>=6)&(glycan_clearance>=22)
    if not feasible.any():
        raise ValueError('no solvent-side region meets native/glycan center clearance')
    chosen=np.argmax(np.where(feasible,directions@normal+.02*clearance,-np.inf))
    placement_center=jnp.array(centers[chosen])
    if args.schedule=='curriculum':
        weights.update(placement=.5,rg=.3)

    def context_loss(inputs, outputs, aux):
        # Upstream binder loss has already aligned aux positions to the target
        # template. Verify this coordinate convention once below, before JIT.
        xyz = aux["atom_positions"]
        bb = xyz[T:, jnp.array([0,1,2,4]), :].reshape(-1,3)  # N, CA, C, O in atom37
        d = jnp.sqrt(jnp.sum((bb[:,None]-nx)**2, axis=-1)+1e-6).min(-1)
        violation = jax.nn.relu(2.5-d)**2
        gd = jnp.sqrt(jnp.sum((bb[:,None]-gx)**2, axis=-1)+1e-6).min(-1)
        losses=dict(interface_cover=get_con_loss(inputs, outputs, inputs['opt']['coverage'],
                        mask_1d=binder_mask, mask_1b=patch_mask),
                    native_target=jnp.mean(jnp.sum((xyz[:T,1]-target_ca)**2,-1)),
                    context_steric=jnp.sort(violation)[-10:].mean(),
                    glycan=jnp.max(jax.nn.relu(15.-gd)**2)/25.)
        if args.schedule=='curriculum':
            bca=xyz[T:,1];center=bca.mean(0)
            rg=jnp.sqrt(jnp.mean(jnp.sum((bca-center)**2,-1))+1e-6)
            losses.update(placement=jnp.sum((center-placement_center)**2)/100.,
                          rg=jax.nn.relu(rg-2.38*args.length**.365)**2/25.)
        return losses

    upstream = ROOT / "build/models/ColabDesign"
    revision = subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
    checkpoint = ROOT / "build/models/colabfold/params/params_model_3_multimer_v3.npz"
    checkpoint_source = json.loads(checkpoint.with_suffix(".source.json").read_text())
    meta = dict(synthetic=False, model="ColabDesign AF2-Multimer gradient hallucination",
        upstream_revision=revision, package_version=importlib.metadata.version("colabdesign"),
        jax_version=importlib.metadata.version("jax"), python_version=__import__("sys").version,
        checkpoint=checkpoint_source, design_model="model_3_multimer_v3",
        target="O95150:72-251", observed_target="88-251", template=True,
        predictor_stoichiometry="one complete native protomer plus explicitly mapped neighbor fragments + one de novo binder",
        context_neighbor_residues_per_protomer=args.context_neighbors,
        target_residue_map=target_map,
        context_allocation=context_allocation,
        geometric_context="all observed native ABC heavy atoms; six N133/N229 ND2 anchors",
        binder_template=False, initial_binder_sequence=None,
        conditioning="native target only; random sequence logits; no prior binder/scaffold",
        target_sha256=hashlib.sha256(target_file.read_bytes()).hexdigest(),
        target_context_sha256=hashlib.sha256((ROOT/"targets/TL1A/observed.pdb").read_bytes()).hexdigest(),
        site=args.site, hotspots_uniprot=hotspot, patch_uniprot=[88+i for i in patch],
        binder_length=args.length, stages=[args.soft_iters,args.temp_iters,args.hard_iters],
        design_recycles=args.recycles, final_hard_recycles=1,
        loss_weights=weights, interface_cover=contact,
        schedule=args.schedule,
        target_derived_binder_center_A=np.asarray(placement_center).tolist(),
        center_native_clearance_A=float(clearance[chosen]),
        center_glycan_clearance_A=float(glycan_clearance[chosen]),
        native_BB_distance_penalty_A=2.5, glycan_BB_penalty_A=15.,
        omitted_AAs="C", official_equivalence=False,
        status="initializing", trajectories=[])
    def save():
        (out/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
    save()
    started = time.monotonic()
    try:
        model = mk_afdesign_model(protocol="binder", use_multimer=True,
            model_names=["model_3_multimer_v3"], num_models=1, sample_models=False,
            data_dir=str(ROOT/"build/models/colabfold"), num_recycles=args.recycles,
            use_remat=True, use_bfloat16=True, loss_callback=context_loss,
            traj_iter=10, traj_max=20)
        model.prep_inputs(pdb_filename=str(target_file), target_chain="A,B,C" if args.context_neighbors else "A",
            binder_len=args.length, hotspot=",".join(f"A{n}" for n in hotspot),
            rm_target_seq=False, rm_target_sc=False)
        if model._target_len != T or not np.allclose(
                model._inputs["batch"]["all_atom_positions"][:T,1],context_ca,atol=1e-4):
            raise ValueError("target coordinate frame/length changed; context losses would be invalid")
        if list(model._pdb['idx']['residue'][model.opt['hotspot']]) != hotspot:
            raise ValueError('resolved hotspot indices do not match canonical target residues')
        for seed in args.seeds:
            record = dict(seed=seed,status="running",sequence_count=0)
            meta["trajectories"].append(record)
            meta["status"]="designing"
            save()
            model.restart(seed=seed,rm_aa="C")
            # Upstream set_weights updates existing keys only; callbacks add
            # new loss names, which must first be registered explicitly.
            model.opt["weights"].update(weights)
            if any(model.opt["weights"].get(k) != v for k,v in weights.items()):
                raise ValueError("custom loss weights were not applied")
            record["applied_loss_weights"] = dict(model.opt["weights"])
            model.opt['coverage']=dict(contact)
            model.set_opt(con=dict(num=2,cutoff=8.,seqsep=9),
                          i_con=dict(num=3,cutoff=8.),num_recycles=args.recycles)
            seed_started=time.monotonic()
            def progress(m):
                loss=float(m.aux["loss"])
                if not np.isfinite(loss):
                    raise FloatingPointError("nonfinite design loss")
                weighted=sum(float(v)*m.opt["weights"].get(k,1.) for k,v in m.aux["losses"].items())
                if not np.isclose(loss,weighted,rtol=1e-4,atol=1e-4):
                    raise ValueError("reported loss differs from the declared weighted losses")
                if (m._k+1)%10 == 0:
                    record["progress"] = dict(iteration=m._k+1,loss=loss,
                        binder_plddt=float(np.asarray(m.aux["plddt"])[T:].mean()),
                        iptm=float(m.aux["i_ptm"]),
                        losses={k:float(v) for k,v in m.aux["losses"].items()})
                    save()
            if args.schedule=='strict':
                model.design_3stage(soft_iters=args.soft_iters,temp_iters=args.temp_iters,
                    hard_iters=args.hard_iters,ramp_recycles=False,verbose=10,callback=progress)
            else:
                def phase(name,w,intra,inter,cover):
                    model.opt['weights'].update(w)
                    model.opt['coverage'].update(cutoff=cover)
                    model.set_opt(con=dict(cutoff=intra),i_con=dict(cutoff=inter))
                    record.setdefault('phases',[]).append(dict(name=name,iteration=model._k,
                        weights=dict(model.opt['weights']),intra_cutoff=intra,
                        inter_cutoff=inter,cover_cutoff=cover))
                    save()
                warm={**weights,'con':1.,'i_pae':.1,'interface_cover':.25,
                      'context_steric':.1,'glycan':.05,'placement':.2}
                phase('fold_and_approach',warm,14.,20.,14.)
                model.design_logits(args.soft_iters,e_soft=1,verbose=10,callback=progress)
                phase('geometric_transition',{**weights,'context_steric':1.,'glycan':.3},10.,14.,12.)
                transition=args.temp_iters//3
                model.design_soft(transition,e_temp=.3,verbose=10,callback=progress)
                phase('strict_interface',weights,8.,8.,8.)
                model.design_soft(args.temp_iters-transition,temp=.3,e_temp=.01,verbose=10,callback=progress)
                model.design_hard(args.hard_iters,temp=.01,dropout=False,verbose=10,callback=progress)
            sequence=model.get_seqs(get_best=False)[0]
            # Produce final coordinates/confidence from this exact DISCRETE
            # sequence, without dropout; never export a soft-sequence best pose.
            model.predict(seq=sequence,hard=True,soft=False,dropout=False,
                          num_recycles=1,verbose=True)
            if model.get_seqs(get_best=False)[0] != sequence:
                raise ValueError("final discrete sequence changed during prediction")
            folder=out/f"seed_{seed}"
            folder.mkdir()
            model.save_pdb(str(folder/"generated.pdb"),get_best=False)
            a=model.aux
            if not np.isfinite(a["atom_positions"]).all() or not np.isfinite(a["pae"]).all():
                raise FloatingPointError("nonfinite final structure/confidence")
            score=dict(plddt=(np.asarray(a["plddt"])*100).tolist(),
                       pae=np.asarray(a["pae"]).tolist(),iptm=float(a["i_ptm"]),ptm=float(a["ptm"]))
            (folder/"scores.json").write_text(json.dumps(score)+"\n")
            (folder/"sequence.fasta").write_text(f">afdesign_{seed} generated_unvalidated\n{sequence}\n")
            (folder/"trajectory.json").write_text(json.dumps(model._tmp["log"],default=lambda x:np.asarray(x).tolist())+"\n")
            record.update(status="generated_not_independently_validated",sequence=sequence,
                sequence_count=1,elapsed_seconds=time.monotonic()-seed_started,
                final_generator_log=a["log"])
            save()
            print(json.dumps(record),flush=True)
        meta["status"]="generated_not_independently_validated"
    except BaseException as exc:
        meta.update(status="failed",error=str(exc))
        raise
    finally:
        meta["elapsed_seconds"]=time.monotonic()-started
        save()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--site",choices=["terminal","beta","outer"],default="terminal")
    p.add_argument("--length",type=int,default=80)
    p.add_argument("--seeds",type=int,nargs="+",default=[20261105])
    p.add_argument("--soft-iters",type=int,default=75)
    p.add_argument("--temp-iters",type=int,default=45)
    p.add_argument("--hard-iters",type=int,default=5)
    p.add_argument("--recycles",type=int,default=0)
    p.add_argument("--schedule",choices=['strict','curriculum'],default='strict')
    p.add_argument("--context-neighbors",type=int,default=0,
                   help="include this many nearest native residues from EACH neighboring protomer as template features; full ABC remains in geometric losses")
    p.add_argument('--clash-probes',help='use existing coverage solvers to allocate five-residue native motifs from actual failed poses instead of nearest-hotspot residues')
    args=p.parse_args()
    if not 40<=args.length<=100 or min(args.soft_iters,args.temp_iters,args.hard_iters)<0:
        p.error("require mature minibinder length 40-100 and nonnegative iterations")
    main(args)
