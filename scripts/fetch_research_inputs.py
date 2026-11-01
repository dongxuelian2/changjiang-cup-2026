"""Download public primary data; large/upstream material stays in ignored build/."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "sources"
TARGETS = [
    ("EGFR", "P00533", 1, 645, "6ARU", "A", "VHH"),
    ("TNF", "P01375", 77, 233, "1TNF", "ABC", "VHH"),
    ("GLP1R", "P43220", 21, 139, "6X18", "R", "VHH"),
    ("TfR1", "P02786", 89, 760, "11QC", "AB", "VHH"),
    ("TL1A", "O95150", 72, 251, "2RE9", "ABC", "minibinder"),
    ("TNFR1", "P19438", 1, 211, "9S3I", "A", "VHH"),
    ("IL7RA", "P16871", 1, 236, "3DI3", "B", "minibinder"),
]


def fetch(url, name):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "changjiang-research/0.1"})
        with urllib.request.urlopen(req, timeout=90) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    return {"url": url, "path": str(path.relative_to(ROOT)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("https://aiprotein.sjtu.edu.cn/", "official.html")]
    for _, accession, *_ in TARGETS:
        jobs.append((f"https://rest.uniprot.org/uniprotkb/{accession}.json", accession + ".json"))
    for pdb in ("2RE9", "3DI3", "1TNF", "3K51"):
        jobs += [(f"https://files.rcsb.org/download/{pdb}.{ext}", f"{pdb}.{ext}") for ext in ("pdb", "cif")]
        jobs.append((f"https://data.rcsb.org/rest/v1/core/entry/{pdb}", f"{pdb}.entry.json"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, *j): j for j in jobs}
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                item = future.result()
                results.append(item)
                print(item["path"], item["bytes"], flush=True)
            except Exception as error:
                print("FAILED", futures[future], str(error), flush=True)
    html = (OUT / "official.html").read_text(encoding="utf-8")
    for asset in re.findall(r'src="(/assets/[^\"]+\.js)"', html):
        results.append(fetch("https://aiprotein.sjtu.edu.cn" + asset, "official.js"))
    (OUT / "downloads.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
