"""Fetch pinned open-source model code and RFdiffusion weights into build/."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "models"
RFD = "86507b6538f51fce57b5a72477165f03999ed7ae"


def read(url):
    req = urllib.request.Request(url, headers={"User-Agent": "changjiang-research/0.1"})
    with urllib.request.urlopen(req, timeout=180) as f:
        return f.read()


def source(repo, name, revision=None):
    path = OUT / name
    if (path / "UPSTREAM.json").exists():
        return json.loads((path / "UPSTREAM.json").read_text())
    if revision is None:
        revision = json.loads(read(f"https://api.github.com/repos/{repo}/commits/main"))["sha"]
    url = f"https://codeload.github.com/{repo}/tar.gz/{revision}"
    data = read(url)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        prefix = archive.getmembers()[0].name.split("/")[0] + "/"
        for member in archive.getmembers():
            relative = member.name.removeprefix(prefix)
            target = path / relative
            if member.isfile() and relative and target.resolve().is_relative_to(path.resolve()):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    record = {"repository": repo, "revision": revision, "archive_sha256": hashlib.sha256(data).hexdigest()}
    (path / "UPSTREAM.json").write_text(json.dumps(record, indent=2) + "\n")
    print(name, record, flush=True)
    return record


def weight():
    path = OUT / "RFdiffusion" / "models" / "Complex_base_ckpt.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt"
    if not path.exists():
        path.write_bytes(read(url))
    record = {"url": url, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (path.parent / "Complex_base_ckpt.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Complex checkpoint", record, flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(source, "RosettaCommons/RFdiffusion", "RFdiffusion", RFD),
                pool.submit(source, "dauparas/ProteinMPNN", "ProteinMPNN", "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"),
                pool.submit(source, "DunbrackLab/IPSAE", "IPSAE", "6174cf9e71cb1bd660cc805856a18c4871a6dec3")]
        for job in jobs:
            job.result()
    weight()
