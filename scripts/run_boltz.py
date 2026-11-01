"""Independent Boltz-2 sequence-to-complex prediction, explicitly NOT official screening.

Run only after generation has released the GPU. No templates or pocket constraints.
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidates")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--out", default="build/boltz_eval")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--sampling-steps", type=int, default=200)
    p.add_argument("--recycles", type=int, default=3)
    p.add_argument("--msa-depth", type=int, default=64)
    p.add_argument("--trimer", action="store_true", help="three full target protomers A/B/C and binder D")
    p.add_argument("--target-template", action="store_true", help="diagnostic native-target template; NEVER official-equivalent")
    p.add_argument("--with-target-control", action="store_true", help="also fold target alone to diagnose target-state failure")
    p.add_argument("--target-msa", default="build/target_msa/target.a3m")
    args = p.parse_args()
    data = json.loads(Path(args.candidates).read_text())
    if data["synthetic"]:
        raise ValueError("only real candidates allowed")
    rows = [r for r in data["candidates"] if not args.ids or r["id"] in args.ids]
    if not rows or (args.ids and set(args.ids) != {r["id"] for r in rows}):
        raise ValueError("candidate IDs missing")
    out = (ROOT / args.out).resolve()
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    msa = (ROOT / args.target_msa).resolve()
    if not msa.is_file():
        raise ValueError("target MSA missing: run target_msa.py; do not silently switch to no-MSA")
    # CSV explicitly marks all target homologs unpaired; binder has no MSA.
    target_csv = out / "target_msa.csv"
    records, sequence = [], ""
    for line in msa.read_text().splitlines():
        if line.startswith(">"):
            if sequence:
                records.append(sequence)
            sequence = ""
        elif line and not line.startswith("#"):
            sequence += line
    if sequence:
        records.append(sequence)
    target = "".join((ROOT / "targets/TL1A/construct.fasta").read_text().splitlines()[1:])
    if not records or records[0] != target:
        raise ValueError("MSA query does not match full official construct")
    with target_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "sequence"])
        writer.writerows((-1, seq) for seq in records)
    if args.target_template:
        # Original RCSB mmCIF retains entity/polymer sequence metadata, absent
        # from coordinate-only exports. It also avoids Windows tempfile locking
        # in the upstream PDB converter. Its author numbering is +71 to UniProt.
        native_template = ROOT / "build/sources/2RE9.cif"
        if not native_template.is_file():
            raise ValueError("original RCSB mmCIF missing; run fetch_research_inputs.py")
    for row in rows:
        target_ids, binder_id = ("[A, B, C]", "D") if args.trimer else ("A", "B")
        (inputs / (row["id"]+".yaml")).write_text(
            f"version: 1\nsequences:\n  - protein:\n      id: {target_ids}\n      sequence: {target}\n      msa: {target_csv.as_posix()}\n"
            f"  - protein:\n      id: {binder_id}\n      sequence: {row['sequence']}\n      msa: empty\n")
        if args.target_template:
            ids, template_ids = ("[A, B, C]", "[A, B, C]") if args.trimer else ("A", "A")
            with (inputs / (row["id"]+".yaml")).open("a") as f:
                f.write(f"templates:\n  - cif: {native_template.as_posix()}\n"
                        f"    chain_id: {ids}\n    template_id: {template_ids}\n    force: true\n    threshold: 1.0\n")
    if args.with_target_control:
        (inputs / "tl1a_target_only.yaml").write_text(
            f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {target}\n      msa: {target_csv.as_posix()}\n")
    executable = str(Path(sys.executable).with_name("boltz.exe" if os.name == "nt" else "boltz"))
    command = [executable, "predict", str(inputs), "--out_dir", str(out), "--cache", str(ROOT / "build/models/boltz2"),
               "--model", "boltz2", "--accelerator", "gpu", "--devices", "1", "--num_workers", "0",
               "--preprocessing-threads", "1", "--recycling_steps", str(args.recycles),
               "--sampling_steps", str(args.sampling_steps), "--diffusion_samples", "1", "--max_parallel_samples", "1",
               "--seed", str(args.seed), "--write_full_pae", "--no_kernels"]
    record = {"synthetic": False, "model": "Boltz-2", "package_version": importlib.metadata.version("boltz"),
              "torch_version": importlib.metadata.version("torch"), "command": command,
              "candidate_ids": [r["id"] for r in rows], "target": "O95150:72-251",
              "target_stoichiometry": "trimer" if args.trimer else "monomer approximation; inspect poses against observed trimer",
              "target_msa_sha256": hashlib.sha256(msa.read_bytes()).hexdigest(),
              "target_msa_records": len(records), "msa_depth_cap": args.msa_depth, "binder_msa": False, "template": args.target_template,
              "controls": ["tl1a_target_only"] if args.with_target_control else [],
              "official_equivalence": False, "precision": "upstream bf16-mixed", "status": "running"}
    record["seed_scope"] = "one global seed per CLI invocation; batch file order affects RNG draws; not a verified official per-sequence seed implementation"
    command += ["--max_msa_seqs", str(args.msa_depth), "--num_subsampled_msa", str(args.msa_depth)]
    if args.target_template:
        command += ["--use_potentials"]
    checkpoint = ROOT / "build/models/boltz2/boltz2_conf.ckpt"
    if checkpoint.exists():
        digest = hashlib.sha256()
        with checkpoint.open("rb") as f:
            for block in iter(lambda: f.read(2**20), b""):
                digest.update(block)
        record["checkpoint_sha256"] = digest.hexdigest()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4", WANDB_MODE="disabled")
    started = time.monotonic()
    (out / "run.json").write_text(json.dumps(record, indent=2)+"\n")
    try:
        with (out / "prediction.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record["exit_code"] = result.returncode
        record["prediction_count"] = len(list(out.glob("boltz_results_*/predictions/*/*_model_0.cif")))
        record["expected_predictions"] = len(rows)+len(record["controls"])
        record["status"] = "finished" if result.returncode == 0 and record["prediction_count"] == record["expected_predictions"] else "partial_or_failed_prediction"
    finally:
        record["elapsed_seconds"] = time.monotonic()-started
        (out / "run.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))
