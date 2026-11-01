"""Portable candidates with explicit metric semantics and preserved metadata."""

import copy
import csv
import json
import math
from pathlib import Path

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
JSON_COLUMNS = ("structures", "epitope_residues", "metadata")


def validate_dataset(dataset):
    if not isinstance(dataset, dict):
        raise ValueError("candidate dataset must be a JSON object")
    d = copy.deepcopy(dataset)
    if d.get("schema_version") != 1 or type(d.get("synthetic")) is not bool:
        raise ValueError("candidate dataset needs schema_version=1 and boolean synthetic")
    definitions = d.get("metric_definitions", {})
    if not isinstance(definitions, dict):
        raise ValueError("metric_definitions must be an object")
    for key, spec in definitions.items():
        if not isinstance(spec, dict) or any(not isinstance(spec.get(s), str) or not spec[s].strip()
                for s in ("model", "version", "meaning", "units")):
            raise ValueError(f"metric {key}: require model, version, meaning, units (use unknown explicitly)")
        if spec.get("direction") not in ("higher", "lower", "unspecified"):
            raise ValueError(f"metric {key}: direction must be higher, lower, or unspecified")
    rows = d.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidates must be a nonempty list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("candidate must be an object")
        for key in ("id", "sequence", "target_id", "design_type", "source_model", "source_version", "run_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"candidate needs nonempty string {key}")
        if row["id"] in seen:
            raise ValueError(f"duplicate candidate ID: {row['id']}")
        seen.add(row["id"])
        original = row["sequence"]
        row["sequence"] = "".join(original.split()).upper()
        if not row["sequence"] or not set(row["sequence"]) <= AA:
            raise ValueError(f"{row['id']}: sequence must contain only the 20 canonical amino acids")
        if original != row["sequence"]:
            row.setdefault("input_sequence", original)
        metrics = row.setdefault("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("metrics must be an object")
        for key, value in metrics.items():
            if key not in definitions:
                raise ValueError(f"{row['id']}: missing definition for metric {key}")
            if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
                raise ValueError(f"{row['id']}: metric {key} must be finite numeric or null")
        epitope = row.setdefault("epitope_residues", [])
        if not isinstance(epitope, list) or not all(isinstance(x, str) and x for x in epitope):
            raise ValueError("epitope_residues must be residue-ID strings in the same target numbering")
        structures = row.setdefault("structures", [])
        if not isinstance(structures, list) or any(not isinstance(s, dict) or not s.get("path") for s in structures):
            raise ValueError("structures must be a list of objects containing path")
        if not isinstance(row.setdefault("metadata", {}), dict):
            raise ValueError("metadata must be an object")
        for key in ("risk_group", "structural_group"):
            if key in row and not isinstance(row[key], str):
                raise ValueError(f"{key} must be a string")
    return d


def load_candidates(path, metadata_path=None):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return validate_dataset(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix.lower() != ".csv":
        raise ValueError("candidates must be CSV or JSON")
    meta = Path(metadata_path) if metadata_path else path.with_suffix(".metrics.json")
    d = json.loads(meta.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError("CSV metadata must be a JSON object")
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV must have unique named columns")
        for raw in reader:
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError("CSV row does not match header width")
            row, metrics = {}, {}
            for key, value in raw.items():
                if key.startswith("metric:"):
                    metrics[key[7:]] = None if value == "" else float(value)
                elif key in JSON_COLUMNS:
                    row[key] = json.loads(value) if value else ({} if key == "metadata" else [])
                else:
                    row[key] = value
            row["metrics"] = metrics
            rows.append(row)
    d["candidates"] = rows
    return validate_dataset(d)


def save_candidates(dataset, path):
    d = validate_dataset(dataset)
    Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(dataset, path):
    """CSV interchange for known columns; reject lossy export of extension fields."""
    d = validate_dataset(dataset)
    keys = ["id", "sequence", "target_id", "design_type", "source_model", "source_version",
            "run_id", "risk_group", "structural_group", *JSON_COLUMNS]
    if any(set(row) - set(keys) - {"metrics"} for row in d["candidates"]):
        raise ValueError("use JSON to preserve candidate extension fields")
    metric_keys = list(d["metric_definitions"])
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + ["metric:" + key for key in metric_keys])
        writer.writeheader()
        for row in d["candidates"]:
            out = {key: row.get(key, "") for key in keys}
            for key in JSON_COLUMNS:
                out[key] = json.dumps(row[key], ensure_ascii=False)
            out.update({"metric:" + key: row["metrics"].get(key) for key in metric_keys})
            writer.writerow(out)
    meta = {k: v for k, v in d.items() if k != "candidates"}
    path.with_suffix(".metrics.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def duplicate_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["target_id"], row["sequence"]), []).append(row["id"])
    return [ids for ids in groups.values() if len(ids) > 1]


def deduplicate(rows):
    """Explicit first-row policy; preserve alternate measurements in the input."""
    seen, kept = set(), []
    for row in rows:
        key = (row["target_id"], row["sequence"])
        if key not in seen:
            kept.append(row)
            seen.add(key)
    return kept


def metric_vector(dataset, key):
    if key not in dataset["metric_definitions"]:
        raise ValueError(f"unknown metric: {key}")
    spec = dataset["metric_definitions"][key]
    if spec["direction"] == "unspecified":
        raise ValueError("raw Top-K requires a metric with specified direction")
    values = [r["metrics"].get(key) for r in dataset["candidates"]]
    if any(v is None for v in values):
        raise ValueError(f"metric {key} has missing values; filter or supply values explicitly")
    return values, spec["direction"] == "higher"
