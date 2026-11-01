"""Request an MSA for the PUBLIC target only; no Colab session or binder sent."""
import hashlib
import json
from pathlib import Path

from boltz.data.msa.mmseqs2 import run_mmseqs2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/target_msa"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    seq = "".join((ROOT / "targets/TL1A/construct.fasta").read_text().splitlines()[1:])
    result = run_mmseqs2([seq], prefix=(OUT / "tl1a").as_posix(), use_env=True, use_filter=True)
    # The pinned Boltz implementation returns a list of A3M strings.
    if isinstance(result, tuple):
        result = result[0]
    msa = result[0]
    (OUT / "target.a3m").write_text(msa)
    metadata = {"target": "O95150:72-251", "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                "server": "https://api.colabfold.com", "retrieved_on": "2026-11-01", "boltz_version": "2.2.1",
                "use_env": True, "use_filter": True, "pairing": False, "binder_sent": False,
                "a3m_sha256": hashlib.sha256((OUT / "target.a3m").read_bytes()).hexdigest(), "sequence_records": msa.count(">"),
                "caveat": "server database/date not pinned; NOT verified official MSA"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata))
