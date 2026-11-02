"""Scientific geometry regression; requires the optional research dependencies."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("Bio"), "requires BioPython research environment")
class TrimerAlignmentTests(unittest.TestCase):
    def poses(self, displaced=False):
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        reference = parser.get_structure("native", ROOT / "targets/TL1A/observed.pdb")
        data = json.loads((ROOT / "results/tl1a_20261101/candidates.json").read_text())
        designed = parser.get_structure("design", ROOT / data["candidates"][0]["structures"][0]["path"])[0]["D"]
        predicted = reference.copy()
        # Rotate labels without moving the physical protomers, then apply a
        # global rigid transform. The binding pose must remain equivalent.
        for old, new in zip("ABC", "XYZ"):
            predicted[0][old].id = new
        for old, new in zip("XYZ", "BCA"):
            predicted[0][old].id = new
        for chain in predicted[0]:
            for residue in chain:
                residue.id = (" ", residue.id[1]-71, " ")
        predicted[0].add(designed.copy())
        if displaced:
            for atom in predicted[0]["D"].get_atoms():
                atom.coord += np.array([0., 0., 100.])
        theta = .63
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        for atom in predicted.get_atoms():
            atom.transform(rot, np.array([15., -20., 8.]))
        manifest = json.loads((ROOT / "targets/manifest.json").read_text())
        core = next(t["beta_core_uniprot"] for t in manifest["targets"] if t["target_id"] == "TL1A")
        return reference, predicted, designed, core

    def test_chain_relabelling_and_rigid_motion_preserve_the_pose(self):
        from scripts.import_boltz import align_prediction
        reference, predicted, designed, core = self.poses()
        _, _, perm, best, _ = align_prediction(reference, predicted, designed, "ABC", "D", core)
        self.assertLess(best, 1e-4)
        self.assertEqual(perm, ("B", "C", "A"))
        error = np.linalg.norm(np.array([r["CA"].coord for r in predicted[0]["D"]])-
                               np.array([r["CA"].coord for r in designed]), axis=1)
        self.assertLess(error.max(), 1e-4)

    def test_a_dissociated_binder_cannot_be_rescued_by_chain_permutation(self):
        from scripts.import_boltz import align_prediction
        reference, predicted, designed, core = self.poses(displaced=True)
        _, _, _, best, alternatives = align_prediction(reference, predicted, designed, "ABC", "D", core)
        self.assertLess(best, 1e-4)
        self.assertGreater(min(t["binder_pose_CA_rmsd"] for t in alternatives), 50)
