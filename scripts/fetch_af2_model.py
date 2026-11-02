"""Fetch one genuine AF2-Multimer-v3 NPZ from the official public TAR by ranges.

Avoids redownloading a multi-GB corpus after an interrupted streamed download.
No upstream whole-corpus completion marker is fabricated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
import zipfile

URL = "https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-12-06.tar"
ROOT = Path(__file__).resolve().parents[1]


def request(start, end):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}", "Cache-Control": "no-cache"})
    response = urllib.request.urlopen(req, timeout=90)
    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
        response.close()
        raise RuntimeError("server did not honor byte range; refusing an ambiguous partial checkpoint")
    return response


def main(args):
    name = f"params_model_{args.model}_multimer_v3.npz"
    out = ROOT / "build/models/colabfold/params"
    out.mkdir(parents=True, exist_ok=True)
    offset = 0
    while True:
        with request(offset, offset+511) as stream:
            header = stream.read(512)
        if not header.strip(b"\0"):
            raise ValueError("requested checkpoint not found")
        member = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        start = offset+512
        print(f"archive member {member.name}: {member.size}", flush=True)
        if Path(member.name).name == name:
            break
        offset = start+((member.size+511)//512)*512
    destination = out / name
    partial = destination.with_suffix(".npz.part")
    if destination.exists() and destination.stat().st_size == member.size:
        partial = destination
    else:
        for attempt in range(8):
            have = partial.stat().st_size if partial.exists() else 0
            try:
                while have < member.size:
                    stop = min(have+64*2**20, member.size)
                    with request(start+have, start+stop-1) as source, partial.open("ab") as target:
                        shutil.copyfileobj(source, target, length=2**20)
                    previous = have
                    have = partial.stat().st_size
                    if have <= previous or have > stop:
                        raise IOError("unexpected range payload length")
                    print(f"{name}: {have}/{member.size}", flush=True)
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(2)
        if partial.stat().st_size != member.size:
            raise ValueError("checkpoint size mismatch")
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError("NPZ CRC failed")
    if partial != destination:
        partial.rename(destination)
    h = hashlib.sha256()
    with destination.open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            h.update(block)
    destination.with_suffix(".source.json").write_text(json.dumps(dict(url=URL,
        archive_member=member.name, byte_offset=start, size=member.size,
        sha256=h.hexdigest(), validation="complete NPZ ZIP CRC; original parameter bytes"), indent=2)+"\n")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=int, choices=range(1,6), default=3)
    main(parser.parse_args())
