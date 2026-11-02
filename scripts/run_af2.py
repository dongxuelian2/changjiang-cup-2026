"""Local AF2-Multimer cross-check through pinned ColabFold's public run API.

Uses only the requested genuine checkpoint, precomputed target-only MSA, no
template or initial guess. Does not access a hosted Colab session.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    out = (ROOT / args.out).resolve()
    if out.exists():
        raise ValueError("choose a fresh output directory")
    (out / "inputs").mkdir(parents=True)
    dataset = json.loads((ROOT / args.candidates).read_text())
    if dataset["synthetic"]:
        raise ValueError("real model outputs only")
    rows = [r for r in dataset["candidates"] if r["id"] in args.ids]
    if {r["id"] for r in rows} != set(args.ids):
        raise ValueError("missing candidate IDs")
    target = "".join((ROOT / "targets/TL1A/construct.fasta").read_text().splitlines()[1:])
    msa = ROOT / "build/target_msa/target.a3m"
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
    if not records or records[0] != target:
        raise ValueError("target MSA query mismatch")
    for row in rows:
        binder = row["sequence"]
        lines = [f"#{len(target)},{len(binder)}\t3,1", ">101\t102", target+binder,
                 ">101", target+"-"*len(binder), ">102", "-"*len(target)+binder]
        for i, sequence in enumerate(records[1:]):
            lines += [f">target_{i}", sequence+"-"*len(binder)]
        (out / "inputs" / (row["id"]+".a3m")).write_text("\n".join(lines)+"\n")
    metadata = dict(synthetic=False, model="AF2-Multimer-v3", candidate_ids=args.ids,
                    target_stoichiometry="trimer", target="O95150:72-251",
                    package_version=importlib.metadata.version("colabfold"),
                    jax_version=importlib.metadata.version("jax"),
                    seed=args.seed, num_seeds=1, model_order=[args.model], num_models=1,
                    num_recycles=3, msa_depth=[128,256], target_msa_records=len(records),
                    target_msa_sha256=hashlib.sha256(msa.read_bytes()).hexdigest(),
                    binder_msa="query only; no evolutionary homologs", template=False,
                    initial_guess=False, official_equivalence=False, status="prepared")
    metadata["kernel_backend"] = args.kernel_backend
    compatibility = ROOT / "build/linux_compatibility.json"
    if compatibility.exists():
        metadata["compatibility_adapters"] = json.loads(compatibility.read_text())
    def save():
        (out / "run.json").write_text(json.dumps(metadata,indent=2)+"\n")
    save()
    if args.prepare_only:
        return
    checkpoint = ROOT / f"build/models/colabfold/params/params_model_{args.model}_multimer_v3.npz"
    provenance = checkpoint.with_suffix(".source.json")
    if not provenance.exists():
        raise ValueError("run fetch_af2_model.py first; checkpoint provenance missing")
    metadata["checkpoint"] = json.loads(provenance.read_text())
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = "0"
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "build/cache-linux"))
    import jax
    from colabfold.batch import get_queries, run, unserialize_msa
    from colabfold.utils import setup_logging
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("GPU unavailable; refusing accidental slow CPU prediction")
    for row in rows:
        text = (out / "inputs" / (row["id"]+".a3m")).read_text()
        unpaired, paired, seqs, counts, _ = unserialize_msa([text], target+row["sequence"])
        if seqs != [target,row["sequence"]] or counts != [3,1]:
            raise ValueError("ColabFold parser changed target/binder stoichiometry")
        metadata.setdefault("parsed_msa", {})[row["id"]] = dict(cardinality=counts,
            binder_unpaired_records=unpaired[1].count(">"), target_unpaired_records=unpaired[0].count(">"))
        if unpaired[1].count(">") != 1:
            raise ValueError("binder must not have evolutionary MSA")
    metadata["status"] = "running"
    save()
    setup_logging(out / "prediction.log")
    queries, is_complex = get_queries(out / "inputs")
    if not is_complex:
        raise ValueError("parser failed to recognize multichain stoichiometry")
    order = {name:i for i,name in enumerate(args.ids)}
    queries.sort(key=lambda q:order[q[0]])
    metadata["execution_order"] = [q[0] for q in queries]
    started = time.monotonic()
    try:
        run(queries=queries, result_dir=out, num_models=1, is_complex=is_complex,
            num_recycles=3, recycle_early_stop_tolerance=0.0, model_order=[args.model],
            model_type="alphafold2_multimer_v3", use_templates=False, num_relax=0,
            data_dir=ROOT / "build/models/colabfold", random_seed=args.seed, num_seeds=1,
            max_seq=128, max_extra_seq=256, recompile_padding=10, pair_mode="unpaired",
            use_fast_kernels=args.kernel_backend!="xla", kernel_backend="auto" if args.kernel_backend=="xla" else args.kernel_backend,
            use_bfloat16=True, keep_existing_results=False)
        metadata["prediction_count"] = len(list(out.glob("*_unrelaxed_rank_001_*.pdb")))
        metadata["status"] = "finished" if metadata["prediction_count"] == len(rows) else "partial_or_failed"
    except BaseException as exc:
        metadata.update(status="failed",error=str(exc))
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic()-started
        log_path = out / "prediction.log"
        if log_path.exists() and "falling back to XLA" in log_path.read_text():
            metadata["effective_kernel_backend"] = "XLA fallback (requested CUDA legacy kernel unavailable for this GPU)"
        else:
            metadata["effective_kernel_backend"] = args.kernel_backend
        save()
    print(json.dumps({k:metadata[k] for k in ("status","prediction_count","elapsed_seconds")},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidates")
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--out", default="build/af2_bg_crosscheck")
    p.add_argument("--model", type=int, choices=range(1,6), default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--kernel-backend", choices=["cuda_legacy","pallas","xla"], default="cuda_legacy")
    main(p.parse_args())
