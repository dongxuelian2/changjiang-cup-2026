"""Scientific geometry regression; requires the optional research dependencies."""
import importlib.util
import hashlib
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


@unittest.skipUnless(importlib.util.find_spec("Bio"), "requires BioPython research environment")
class PoseAnchorTests(unittest.TestCase):
    def test_retained_new_atom14_and_if_files_match_actual_sequences(self):
        from Bio.PDB import MMCIFParser, PDBParser
        from Bio.SeqUtils import seq1
        folder = ROOT/'results/tl1a_pose_design_20261102'
        data = json.loads((folder/'candidates.json').read_text())
        for row in data['candidates']:
            design = PDBParser(QUIET=True).get_structure('design', ROOT/row['structures'][0]['path'])
            self.assertEqual(row['sequence'], ''.join(seq1(r.resname) for r in design[0]['D']))
            generation = row['metadata']['generation']
            if generation.get('stage') not in ('atom14', 'inverse_folding'):
                continue
            file = ROOT/generation['retained_file'] if 'retained_file' in generation else folder/'generation'/Path(generation['file']).name
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), generation['sha256'])
            pose = MMCIFParser(QUIET=True).get_structure('real', file)
            self.assertIn(row['sequence'], [''.join(seq1(r.resname) for r in c) for c in pose[0]])

    def test_single_if_output_cannot_overwrite_original_atom14_file(self):
        from scripts.import_boltzgen import retained_filename
        raw = Path('intermediate_designs/design_0.cif')
        inverse = Path('intermediate_designs_inverse_folded/design_0.cif')
        self.assertNotEqual(retained_filename(raw, raw, 'atom14'),
                            retained_filename(inverse, raw, 'inverse_folding'))
        many = Path('intermediate_designs_inverse_folded/design_0_1.cif')
        self.assertEqual(retained_filename(many, raw, 'inverse_folding'), many.name)

    def test_two_anchors_leave_a_rigid_rotation_unobserved(self):
        from scripts.design_context import select_pose_anchors
        rng = np.random.default_rng(42)
        binder = rng.normal(size=(40, 3))*3
        target = np.array([[8., 0., 0.], [0., 8., 0.], [0., 0., 8.], [-5., -6., -7.]])
        _, two = select_pose_anchors(target, binder, range(4), k=2)
        _, four = select_pose_anchors(target, binder, range(4), k=4)
        self.assertLess(abs(two['unregularized_information_eigenvalues'][0]), 1e-12)
        self.assertGreater(four['unregularized_information_eigenvalues'][0], .01)
        matrices = np.array(four['candidate_information_matrices'])
        self.assertGreaterEqual(np.linalg.eigvalsh(matrices).min(), -1e-12)
        gains = four['marginal_gains']
        self.assertTrue(all(a >= b-1e-12 for a,b in zip(gains, gains[1:])))

    def test_anchor_objective_is_invariant_to_global_rigid_motion(self):
        from scripts.design_context import select_pose_anchors
        rng = np.random.default_rng(8)
        binder = rng.normal(size=(35, 3))*3
        target = rng.normal(size=(12, 3))*6
        q, _ = np.linalg.qr(rng.normal(size=(3,3)))
        a, first = select_pose_anchors(target, binder, range(12), k=6)
        b, second = select_pose_anchors(target@q+11, binder@q+11, range(12), k=6)
        self.assertEqual(a, b)
        self.assertAlmostEqual(first['logdet'], second['logdet'], places=11)
        np.testing.assert_allclose(first['unregularized_information_eigenvalues'],
                                   second['unregularized_information_eigenvalues'], atol=1e-12)
