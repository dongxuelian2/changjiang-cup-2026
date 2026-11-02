"""Run pinned upstream BoltzGen tasks locally, with kernels disabled on Windows.

Uses upstream CLI configuration and Predict tasks unchanged. Each GPU stage has
its own seeded process. Boltz-2 sequence-only evaluation is a separate command.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", default="gpu/tl1a_boltzgen.yaml")
    p.add_argument("--out", default="build/tl1a_boltzgen")
    p.add_argument("--num-designs", type=int, default=8)
    p.add_argument("--sequences-per-backbone", type=int, default=1)
    p.add_argument("--checkpoint", choices=["adherence", "diverse"], default="adherence")
    p.add_argument("--sampling-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--use-kernels", action="store_true", help="upstream cuEquivariance acceleration; requires compatible Linux CUDA packages")
    p.add_argument("--log-sampling", action="store_true", help="print upstream diffusion progress to the ignored run log")
    p.add_argument("--finish-generated", action="store_true", help="after intentionally interrupting sampling, inverse-fold all completed real backbones; retain requested vs actual counts")
    p.add_argument("--stage", choices=["design", "inverse_folding"], help=argparse.SUPPRESS)
    p.add_argument("--config-file", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.stage:
        import torch
        from pytorch_lightning import seed_everything
        from boltzgen.resources.main import main as upstream_main
        torch.set_num_threads(4)
        seed_everything(args.seed, workers=True)
        upstream_main(args.config_file, [])
        return
    if importlib.metadata.version("boltzgen") != "0.3.2":
        raise ValueError("this entry point is tested against boltzgen 0.3.2")
    out = (ROOT / args.out).resolve()
    if out.exists() and not args.finish_generated:
        raise ValueError("choose a new output directory; preserve previous run provenance")
    if args.finish_generated and not (out / "run.json").exists():
        raise ValueError("no recorded generation to finish")
    out.mkdir(parents=True, exist_ok=args.finish_generated)
    spec = (ROOT / args.spec).resolve()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4",
               HF_HOME=str(ROOT / "build/huggingface"),
               XDG_CACHE_HOME=str(ROOT / "build/cache"),
               NUMBA_CACHE_DIR=str(ROOT / "build/numba"),
               HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1", WANDB_MODE="disabled")
    if os.environ.get("CUEQ_WSL_METADATA") == "1":
        env.update(CUEQ_DEFAULT_CONFIG="1", CUEQ_DISABLE_AOT_TUNING="1")
    executable = Path(sys.executable).with_name("boltzgen.exe" if os.name == "nt" else "boltzgen")
    command = [str(executable), "configure", str(spec), "--output", out.as_posix(),
               "--protocol", "protein-anything", "--num_designs", str(args.num_designs),
               "--diffusion_batch_size", "1", "--devices", "1", "--num_workers", "0",
               "--use_kernels", str(args.use_kernels).lower(), "--cache", (ROOT / "build/models/boltzgen").as_posix(),
               "--design_checkpoints", f"huggingface:boltzgen/boltzgen-1:boltzgen1_{args.checkpoint}.ckpt",
               "--inverse_fold_num_sequences", str(args.sequences_per_backbone),
               # Upstream builds configs before filtering steps. These existing
               # paths avoid downloading unused affinity/refolding weights.
               # Neither checkpoint is executed here.
               "--folding_checkpoint", (ROOT / "build/models/boltz2/boltz2_conf.ckpt").as_posix(),
               "--affinity_checkpoint", (ROOT / "build/models/boltz2/boltz2_aff.ckpt").as_posix(),
               "--steps", "design", "inverse_folding",
               "--config", "design", f"sampling_steps={args.sampling_steps}"]
    if args.log_sampling:
        command.append("override.inference_logging=true")
    meta = dict(synthetic=False, model="BoltzGen", package_version="0.3.2",
                target="O95150:72-251", observed_target="88-251", target_stoichiometry="trimer",
                target_input_sha256=digest(ROOT / "build/sources/2RE9.cif"),
                spec_sha256=digest(spec), spec=spec.relative_to(ROOT).as_posix(),
                requested_backbones=args.num_designs, sequences_per_backbone=args.sequences_per_backbone,
                seed=args.seed, checkpoint=args.checkpoint, use_kernels=args.use_kernels,
                started_unix_seconds=time.time(),
                source_code_changes=[], official_equivalence=False, configure_command=command,
                seed_scope="stage-local global RNG; inverse folding uses seed+1; CUDA reproducibility not guaranteed",
                environment={name: importlib.metadata.version(name) for name in
                             ("torch", "numpy", "numba", "pytorch-lightning", "biotite", "rdkit")},
                status="configuring", stages=[])
    if os.environ.get("CUEQ_WSL_METADATA") == "1":
        meta["compatibility_adapters"] = json.loads((ROOT / "build/linux_compatibility.json").read_text())
    previous_elapsed = 0
    if args.finish_generated:
        meta = json.loads((out / "run.json").read_text())
        if meta["status"] == "generated_not_independently_validated":
            raise ValueError("inverse folding already completed; preserve its outputs")
        completed = list((out / "intermediate_designs").glob("*.cif"))
        completed = [f for f in completed if not f.name.endswith("_native.cif")]
        if not completed:
            raise ValueError("no real completed backbones")
        meta["pilot_stopping_reason"] = "sampling intentionally stopped to validate completed backbones before expanding this site"
        meta["actual_backbones"] = len(completed)
        for old in meta["stages"]:
            if old["stage"] == "design" and old["status"] == "running":
                old["status"] = "interrupted_after_completed_outputs"
        previous_elapsed = meta.get("elapsed_seconds") or 0
    def save():
        (out / "run.json").write_text(json.dumps(meta, indent=2)+"\n")
    save()
    started = time.monotonic()
    try:
        if not args.finish_generated:
            with (out / "configure.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        from Bio.PDB import MMCIFParser
        from Bio.SeqUtils import seq1
        check_pose = MMCIFParser(QUIET=True).get_structure("preflight", out / (spec.stem+".cif"))
        expected = "".join((ROOT / "targets/TL1A/construct.fasta").read_text().splitlines()[1:])[16:]
        target_sequences = ["".join(seq1(r.resname) for r in c) for c in check_pose[0]
                            if len(c) > 100]
        if target_sequences != [expected]*3:
            raise ValueError("preflight failed: require three EXACT full observed TL1A 88-251 sequences")
        meta["preflight_target_lengths"] = [len(s) for s in target_sequences]
        from omegaconf import OmegaConf
        stages = ("inverse_folding",) if args.finish_generated else ("design", "inverse_folding")
        for stage in stages:
            i = 0 if stage == "design" else 1
            cfg = out / "config" / f"{stage}.yaml"
            resolved = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
            checkpoint = Path(resolved["checkpoint"])
            record = dict(stage=stage, seed=meta["seed"]+i, config=resolved,
                          checkpoint_sha256=digest(checkpoint), status="running")
            meta["stages"].append(record)
            meta["status"] = stage
            save()
            command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
                       "--config-file", str(cfg), "--seed", str(meta["seed"]+i)]
            record["command"] = command
            stage_start = time.monotonic()
            with (out / f"{stage}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record.update(exit_code=result.returncode, elapsed_seconds=time.monotonic()-stage_start,
                          status="finished" if result.returncode == 0 else "failed")
            save()
            if result.returncode:
                raise RuntimeError(f"{stage} failed; inspect {out / (stage+'.log')}")
        files = list((out / "intermediate_designs_inverse_folded").glob("*.cif"))
        meta["sequence_structure_count"] = len([f for f in files if not f.name.endswith("_native.cif")])
        meta["actual_backbones"] = len([f for f in (out / "intermediate_designs").glob("*.cif") if not f.name.endswith("_native.cif")])
        meta["status"] = "generated_not_independently_validated" if meta["sequence_structure_count"] else "no_designs"
    except BaseException as exc:
        meta["status"] = "failed"
        meta["error"] = str(exc)
        raise
    finally:
        meta["last_invocation_elapsed_seconds"] = time.monotonic()-started
        meta["elapsed_seconds"] = time.time()-meta["started_unix_seconds"] if "started_unix_seconds" in meta else None
        if meta["elapsed_seconds"] is None:
            meta["timing_note"] = "total walltime unavailable after external sampler interruption; last invocation time is separate"
        save()
    print(json.dumps({k: meta[k] for k in ("status", "sequence_structure_count", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
