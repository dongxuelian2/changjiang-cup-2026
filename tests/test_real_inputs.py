"""Biologically consequential artifact consistency, with no model/GPU dependency."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AA3 = dict(zip("ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(), "ACDEFGHIKLMNPQRSTVWY"))


class RealInputTests(unittest.TestCase):
    def test_official_constructs_hash_range_and_native_tl1a_coordinates(self):
        manifest = json.loads((ROOT / "targets/manifest.json").read_text())
        self.assertEqual(len(manifest["targets"]), 7)
        for target in manifest["targets"]:
            seq = "".join((ROOT / "targets" / target["target_id"] / "construct.fasta").read_text().splitlines()[1:])
            self.assertEqual(len(seq), target["range_inclusive"][1]-target["range_inclusive"][0]+1)
            self.assertEqual(hashlib.sha256(seq.encode()).hexdigest(), target["sequence_sha256"])
        tl1a = next(t for t in manifest["targets"] if t["target_id"] == "TL1A")
        self.assertEqual(tl1a["author_to_uniprot_offset"], 71)
        for chain in tl1a["chains"].values():
            self.assertEqual(chain["sequence_mismatches"], [])
            self.assertEqual(chain["observed_range"], [88, 251])

    def test_real_binder_sequence_matches_chain_D_and_design_provenance(self):
        dataset = json.loads((ROOT / "results/tl1a_20261101/candidates.json").read_text())
        self.assertFalse(dataset["synthetic"])
        sequences = set()
        for row in dataset["candidates"]:
            self.assertTrue(40 <= len(row["sequence"]) <= 100)
            self.assertFalse(row["metadata"]["known_positive_binder_used"])
            self.assertNotIn(row["sequence"], sequences)
            sequences.add(row["sequence"])
            structure = next(s for s in row["structures"] if s["role"] == "designed_backbone_complex")
            residues = {}
            for line in (ROOT / structure["path"]).read_text().splitlines():
                if line.startswith("ATOM  ") and line[21] == "D":
                    residues.setdefault(int(line[22:26]), AA3[line[17:20]])
            self.assertEqual("".join(residues.values()), row["sequence"])


if __name__ == "__main__":
    unittest.main()
