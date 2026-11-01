"""Fixed joint scenarios, with nonnegative higher-is-better utilities."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScenarioObjective:
    ids: tuple[str, ...]
    utilities: np.ndarray  # candidate x joint scenario
    weights: np.ndarray  # sum exactly one up to floating-point error
    description: str = ""
    synthetic: bool = False

    def __post_init__(self):
        object.__setattr__(self, "ids", tuple(self.ids))
        u = np.array(self.utilities, dtype=float, copy=True)
        w = np.array(self.weights, dtype=float, copy=True)
        if len(set(self.ids)) != len(self.ids) or not all(isinstance(i, str) and i for i in self.ids):
            raise ValueError("scenario candidate IDs must be unique nonempty strings")
        if u.ndim != 2 or u.shape[0] != len(self.ids) or u.shape[1] == 0:
            raise ValueError("utilities must have shape (number of IDs, positive scenarios)")
        if w.shape != (u.shape[1],):
            raise ValueError("weights and scenario dimensions differ")
        if not np.isfinite(u).all() or not np.isfinite(w).all() or (u < 0).any() or (w < 0).any():
            raise ValueError("utilities and weights must be finite and nonnegative")
        if not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-10):
            raise ValueError("scenario weights must sum to one; no silent normalization")
        if type(self.synthetic) is not bool or not isinstance(self.description, str):
            raise ValueError("invalid scenario provenance")
        u.flags.writeable = w.flags.writeable = False
        object.__setattr__(self, "utilities", u)
        object.__setattr__(self, "weights", w)

    def value(self, indices) -> float:
        indices = list(indices)
        if not indices:
            return 0.0
        if any(type(i) is not int or i < 0 or i >= len(self.ids) for i in indices):
            raise ValueError("invalid candidate index")
        return float(self.utilities[indices].max(axis=0) @ self.weights)

    def marginal_gains(self, indices) -> np.ndarray:
        indices = list(indices)
        self.value(indices)  # validate indices
        best = self.utilities[indices].max(axis=0) if indices else np.zeros(len(self.weights))
        return np.maximum(self.utilities - best, 0) @ self.weights

    def align(self, ids) -> "ScenarioObjective":
        """Align/subset by ID, never by input row position."""
        ids = tuple(ids)
        lookup = {name: i for i, name in enumerate(self.ids)}
        if any(i not in lookup for i in ids):
            raise ValueError("scenario file is missing candidate IDs")
        return ScenarioObjective(tuple(ids), self.utilities[[lookup[i] for i in ids]],
                                 self.weights, self.description, self.synthetic)


def load_scenarios(path) -> ScenarioObjective:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict) or d.get("schema_version") != 1 or not d.get("description") or "synthetic" not in d:
        raise ValueError("scenario file needs schema_version=1, description, and synthetic")
    if not isinstance(d.get("candidate_ids"), list):
        raise ValueError("scenario candidate_ids must be a list")
    return ScenarioObjective(tuple(d["candidate_ids"]), d["utilities"], d["weights"],
                             d["description"], d["synthetic"])


def save_scenarios(obj, path):
    Path(path).write_text(json.dumps({"schema_version": 1, "synthetic": obj.synthetic,
        "description": obj.description, "candidate_ids": obj.ids,
        "utilities": obj.utilities.tolist(), "weights": obj.weights.tolist()}, indent=2) + "\n",
        encoding="utf-8")
