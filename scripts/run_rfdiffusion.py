"""Actual TL1A backbone generation, then ProteinMPNN sequence design.

No known binder scaffold/sequence is supplied. Outputs are unvalidated designs.
Run with build/env-rfd/Scripts/python.exe (or the equivalent Linux environment).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def patch_windows(rfd):
    # NumPy int is int32 on Windows. RFdiffusion's one_hot and destination
    # tensors require torch.int64. No numerical/model parameters are changed.
    if sys.platform != "win32":
        return []
    path = rfd / "rfdiffusion/inference/model_runners.py"
    old = 'seq_orig = self.target_feats["seq"]'
    new = old + '.long()'
    text = path.read_text(encoding="utf-8")
    if new not in text:
        if text.count(old) != 1:
            raise RuntimeError("upstream changed; inspect Windows int64 compatibility patch")
        path.write_text(text.replace(old, new), encoding="utf-8")
    return ["model_runners.py: seq_orig int32 -> int64 for Windows NumPy"]


def run(args):
    os.environ.setdefault("DGLBACKEND", "pytorch")
    import torch
    import dgl
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use a CUDA-enabled environment; refusing accidental slow CPU generation")
    # Confirm that DGL is a GPU build, rather than accepting a CPU wheel.
    dgl.graph(([0], [1])).to("cuda")
    rfd = ROOT / "build/models/RFdiffusion"
    mpnn = ROOT / "build/models/ProteinMPNN"
    patches = patch_windows(rfd)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    site = {"receptor": [187, 188], "outer": [110, 123]}[args.site]
    target = ROOT / ("targets/TL1A/observed.pdb" if args.context == "trimer" else "targets/TL1A/monomer_A.pdb")
    contig = "A88-251/0 B88-251/0 C88-251/0" if args.context == "trimer" else "A88-251/0"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(rfd), str(rfd / "env/SE3Transformer"), str(ROOT), env.get("PYTHONPATH", "")])
    env["DGLBACKEND"] = "pytorch"
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = "4"
    command = [sys.executable, str(rfd / "scripts/run_inference.py"),
               f"inference.input_pdb={target.as_posix()}",
               f"inference.output_prefix={(out / 'design').as_posix()}",
               f"contigmap.contigs=[{contig} {args.length}-{args.length}]",
               "ppi.hotspot_res=[" + ",".join(f"A{x}" for x in site) + "]",
               f"inference.num_designs={args.backbones}", f"inference.design_startnum={args.start}",
               "inference.deterministic=True", "inference.write_trajectory=False",
               "denoiser.noise_scale_ca=0", "denoiser.noise_scale_frame=0",
               f"hydra.run.dir={(out / 'hydra').as_posix()}"]
    if args.compact:
        command += ['potentials.guiding_potentials=["type:binder_ROG,weight:1,min_dist:8"]',
                    "potentials.guide_scale=2", "potentials.guide_decay=quadratic"]
    record = {"schema_version": 1, "synthetic": False, "status": "running",
              "target_id": "TL1A", "site": args.site, "context": args.context, "hotspots_uniprot": site,
              "conditioning": f"2RE9 observed {args.context}; no binder sequence or scaffold",
              "omitted_target_context": "72-87 unresolved; glycans not modelled; all poses rechecked against observed trimer",
              "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "torch": torch.__version__, "dgl": dgl.__version__, "gpu": torch.cuda.get_device_name(),
              "compatibility_patches": patches,
              "compactness_guidance": args.compact,
              "upstream": {n: json.loads((ROOT / 'build/models' / n / 'UPSTREAM.json').read_text()) for n in ("RFdiffusion", "ProteinMPNN")},
              "rfd_checkpoint": json.loads((rfd / 'models/Complex_base_ckpt.json').read_text()),
              "command": command, "mpnn_commands": []}
    manifest = out / "run.json"
    def save():
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    save()
    started = time.monotonic()
    try:
        with (out / "rfd.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        for pdb in sorted(out.glob("design_*.pdb")):
            # Input target chain IDs are retained; find the generated poly-G chain.
            from Bio.PDB import PDBParser
            pose = PDBParser(QUIET=True).get_structure("design", pdb)
            binders = [c.id for c in pose[0] if len(c) == args.length and all(r.resname == "GLY" for r in c)]
            if len(binders) != 1 or len(pose[0]["A"]) != 164:
                raise ValueError("unexpected RFdiffusion chain mapping")
            record["binder_chain"] = binders[0]
            command2 = [sys.executable, str(mpnn / "protein_mpnn_run.py"),
                "--pdb_path", pdb.as_posix(), "--pdb_path_chains", binders[0], "--out_folder", (out / "mpnn").as_posix(),
                "--num_seq_per_target", str(args.sequences), "--batch_size", "1", "--sampling_temp", "0.1",
                "--seed", str(args.mpnn_seed), "--model_name", "v_48_020", "--save_score", "1",
                "--path_to_model_weights", (mpnn / "vanilla_model_weights").as_posix(),
                "--omit_AAs", "C"]
            record["mpnn_commands"].append(command2)
            save()
            with (out / (pdb.stem + "_mpnn.log")).open("w", encoding="utf-8") as log:
                subprocess.run(command2, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        record["backbone_outputs"] = []
        for pdb in sorted(out.glob("design_*.pdb")):
            with pdb.with_suffix(".trb").open("rb") as f:
                trb = pickle.load(f)  # trusted local generation output only
            record.setdefault("resolved_rfd_config", trb["config"])
            record["backbone_outputs"].append({"file": pdb.name,
                "sha256": hashlib.sha256(pdb.read_bytes()).hexdigest(),
                "generation_seconds": trb["time"], "seed": int(pdb.stem.split("_")[-1])})
        record["mpnn_checkpoint_sha256"] = hashlib.sha256((mpnn / "vanilla_model_weights/v_48_020.pt").read_bytes()).hexdigest()
        record["status"] = "generated_not_independently_validated"
    except subprocess.CalledProcessError as error:
        record["status"] = "failed"
        record["exit_code"] = error.returncode
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic()-started
        save()
    print(json.dumps({"out": str(out), "status": record["status"], "seconds": record["elapsed_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="build/tl1a_receptor")
    parser.add_argument("--site", choices=["receptor", "outer"], default="receptor")
    parser.add_argument("--context", choices=["monomer", "trimer"], default="monomer")
    parser.add_argument("--compact", action="store_true", help="recorded binder radius-of-gyration guidance; experimental setting")
    parser.add_argument("--length", type=int, default=60)
    parser.add_argument("--backbones", type=int, default=4)
    parser.add_argument("--sequences", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--mpnn-seed", type=int, default=2026)
    args = parser.parse_args()
    if not 40 <= args.length <= 100 or args.backbones < 1 or args.sequences < 1:
        parser.error("require 40-100 aa and positive generation counts")
    run(args)
