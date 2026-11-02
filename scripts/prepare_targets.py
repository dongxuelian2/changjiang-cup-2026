"""Audit official constructs and prepare native-numbered, observed target coordinates.

Requires numpy and biopython. Never fills unresolved residues with invented atoms.
"""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select, ShrakeRupley
from Bio.SeqUtils import seq1

from fetch_research_inputs import TARGETS

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "build" / "sources"
OUT = ROOT / "targets"
OFFSETS = {"2RE9": 71, "1TNF": 76, "3DI3": 20}


def dump(data, path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


class ProteinOnly(Select):
    def __init__(self, chains):
        self.chains = chains

    def accept_chain(self, chain):
        return chain.id in self.chains

    def accept_residue(self, residue):
        return residue.id[0] == " "

    def accept_atom(self, atom):
        return atom.element != "H" and (not atom.is_disordered() or atom.get_altloc() in ("A", " "))


def prepare():
    OUT.mkdir(exist_ok=True)
    parser = PDBParser(QUIET=True)
    targets, mapping = [], []
    for name, accession, first, last, pdb, chains, design_type in TARGETS:
        data = json.loads((SOURCES / (accession + ".json")).read_text(encoding="utf-8"))
        native = data["sequence"]["value"]
        seq = native[first-1:last]
        assert len(seq) == last-first+1
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        (directory / "construct.fasta").write_text(f">{name}|{accession}|{first}-{last}|official_target\n{seq}\n", encoding="utf-8")
        record = {"target_id": name, "uniprot": accession, "isoform": "canonical",
                  "range_inclusive": [first, last], "length": len(seq), "end_residues": seq[0]+seq[-1],
                  "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                  "uniprot_sequence_version": data["entryAudit"]["sequenceVersion"],
                  "uniprot_last_update": data["entryAudit"]["lastAnnotationUpdateDate"],
                  "reference_pdb": pdb, "reference_chains": list(chains), "design_type": design_type,
                  "features": [{"type": f["type"], "description": f.get("description", ""),
                                "location": f["location"]} for f in data.get("features", [])
                               if f["type"] in ("Signal", "Transmembrane", "Glycosylation", "Disulfide bond")]}
        if pdb not in OFFSETS:
            record["structure_prepared"] = False
            targets.append(record)
            continue
        offset = OFFSETS[pdb]
        structure = parser.get_structure(pdb, SOURCES / (pdb + ".pdb"))
        entry = json.loads((SOURCES / (pdb + ".entry.json")).read_text())
        record.update({"structure_prepared": True, "author_to_uniprot_offset": offset,
                       "resolution_angstrom": entry["rcsb_entry_info"]["resolution_combined"],
                       "method": [x["method"] for x in entry["exptl"]], "chains": {}})
        for chain_id in chains:
            chain = structure[0][chain_id]
            residues = [r for r in chain if r.id[0] == " "]
            observed, mismatches = [], []
            for r in residues:
                n = r.id[1] + offset
                observed.append(n)
                aa = seq1(r.resname)
                if not 1 <= n <= len(native) or aa != native[n-1]:
                    mismatches.append({"uniprot_residue": n, "pdb_aa": aa,
                                       "native_aa": native[n-1] if 1 <= n <= len(native) else None})
                mapping.append([name, pdb, chain_id, r.id[1], n, aa,
                                all(a in r for a in ("N", "CA", "C", "O")), float(r["CA"].bfactor)])
            record["chains"][chain_id] = {"observed_count": len(observed),
                "observed_range": [min(observed), max(observed)],
                "missing_in_official_construct": sorted(set(range(first, last+1))-set(observed)),
                "sequence_mismatches": mismatches,
                "incomplete_backbones": [r.id[1]+offset for r in residues if not all(a in r for a in ("N", "CA", "C", "O"))]}
        # Keep original immutable; copy and change IDs without in-place collisions.
        prepared = copy.deepcopy(structure)
        for chain in list(prepared[0]):
            if chain.id not in chains:
                prepared[0].detach_child(chain.id)
                continue
            residues = list(chain)
            for r in residues:
                chain.detach_child(r.id)
            for r in residues:
                if r.id[0] == " ":
                    r.id = (" ", r.id[1]+offset, r.id[2])
                    chain.add(r)
        writer = PDBIO()
        writer.set_structure(prepared)
        writer.save(str(directory / "observed.pdb"), ProteinOnly(chains))
        if name == "TL1A":
            record["mmcif_label_to_uniprot_offset"] = 70
            record["mmcif_index_note"] = "2RE9 label_seq_id = author residue number +1; observed labels18-181 map to UniProt88-251"
            beta_core = set()
            for line in (SOURCES / "2RE9.pdb").read_text().splitlines():
                if line.startswith("SHEET ") and line[21] == "A":
                    beta_core.update(range(int(line[22:26])+offset, int(line[33:37])+offset+1))
            record["beta_core_uniprot"] = sorted(beta_core)
            record["beta_core_basis"] = "union of chain A SHEET records in original 2RE9; not fitted to predictions"
            if any(record["chains"][c]["sequence_mismatches"] for c in chains):
                raise ValueError("TL1A requires native sequence; inspect structure mismatch")
            writer.save(str(directory / "monomer_A.pdb"), ProteinOnly("A"))
            surface_analysis(prepared, directory)
            record["oligomer"] = "native homotrimer, A/B/C already present in asymmetric unit"
            record["glycan_warning"] = "N133 and N229; unglycosylated reference is not evidence of glycan absence in assay"
        elif name == "TNF":
            record["oligomer"] = "native homotrimer, A/B/C"
        else:
            record["oligomer"] = "3DI3 contains IL7(A):IL7RA(B) heterodimer, not receptor homodimer"
        targets.append(record)
    sources = json.loads((SOURCES / "downloads.json").read_text())
    dump({"retrieved_on": "2026-11-01", "sources": sources,
          "construct_basis": "official UniProt accession and inclusive range",
          "targets": targets}, OUT / "manifest.json")
    with (OUT / "residue_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "pdb", "chain", "author_residue", "uniprot_residue", "aa", "complete_backbone", "CA_Bfactor"])
        writer.writerows(mapping)
    print(json.dumps([{k: t.get(k) for k in ("target_id", "length", "resolution_angstrom", "chains")} for t in targets], indent=2))


def surface_analysis(structure, directory):
    """Observed-trimer SASA, glycan-anchor distances, and receptor-context contacts."""
    ShrakeRupley(n_points=100).compute(structure[0], level="R")
    chain = structure[0]["A"]
    monomer = copy.deepcopy(chain)
    ShrakeRupley(n_points=100).compute(monomer, level="R")
    other = np.array([a.coord for c in structure[0] if c.id != "A" for a in c.get_atoms() if a.element != "H"])
    glycan = np.array([structure[0][c][n]["ND2"].coord for c in "ABC" for n in (133, 229)])
    context = PDBParser(QUIET=True).get_structure("3K51", SOURCES / "3K51.pdb")
    a, b = context[0]["A"], context[0]["B"]
    receptor = np.array([atom.coord for r in b if r.id[0] == " " for atom in r if atom.element != "H"])
    contacts = {r.id[1]+67 for r in a if r.id[0] == " " and
                min(np.linalg.norm(np.array([x.coord for x in r])[:, None]-receptor, axis=-1).ravel()) < 5}
    rows = []
    for r in chain:
        heavy = np.array([a.coord for a in r if a.element != "H"])
        rows.append({"uniprot_residue": r.id[1], "aa": seq1(r.resname), "sasa_trimer_A2": float(r.sasa),
                     "sasa_monomer_A2": float(monomer[r.id].sasa),
                     "buried_by_trimer_A2": max(0., float(monomer[r.id].sasa-r.sasa)),
                     "nearest_other_chain_A": float(np.linalg.norm(heavy[:, None]-other, axis=-1).min()),
                     "nearest_glycan_ND2_A": float(np.linalg.norm(heavy[:, None]-glycan, axis=-1).min()),
                     "DcR3_contact_3K51": r.id[1] in contacts})
    with (directory / "surface.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    exposed = [r for r in rows if r["sasa_trimer_A2"] >= 40 and r["nearest_glycan_ND2_A"] >= 14]
    print("TL1A exposed glycan-distant candidates:", [(r["uniprot_residue"], r["aa"], round(r["sasa_trimer_A2"]), r["DcR3_contact_3K51"]) for r in exposed])


if __name__ == "__main__":
    prepare()
