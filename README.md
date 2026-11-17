# Changjiang Cup 2026: TL1A minibinder research

Computational TL1A minibinder design and mathematical methods for allocating
candidate-evaluation budgets. The project combines RFdiffusion/ProteinMPNN,
BoltzGen, local AF2 design, independent structure predictions, and portfolio
selection.

## Results and evidence

The evaluated candidates currently have zero complete-screen survivors.
Generation and prediction records describe computational designs; experimental
binding affinity and the official competition score remain open measurements.
The [experiment record](docs/experiments.md) gives sequence counts, predictor
settings, observed failures, and the retained evidence for each scaffold family.

[Target definitions and rules](RULES.md) describe the retrieved competition
requirements. `targets/` contains official construct sequences and structural
inputs. The native TL1A trimer and glycan context are part of the design model.

The [mathematical research body](math/PROOF.md) develops candidate-allocation
and geometric-context results. The [selection guide](docs/candidate-selection.md)
connects those guarantees to actual metric tables, states their assumptions,
and gives the existing benchmark commands.

## Run and reproduce

Install the candidate-selection package and run the standard-library tests:

```sh
python -m pip install -e .
python -m changjiang_select benchmark
python -m unittest discover -s tests -v
```

The [local reproduction guide](docs/reproduction.md) gives the generation,
import, prediction, and triage commands and the environment versions used.
Run those commands from the repository root. Model weights and GPU environments
are kept under the ignored `build/` directory.

## Repository map

- `math/`: mathematical definitions, proofs, and checks.
- `targets/`: construct sequences and structural context.
- `scripts/` and `gpu/`: processing tools and local model execution.
- `results/`: curated experimental inputs, predictions, and metrics.
- `tests/`: implementation checks.
- `docs/`: experiment interpretation, selection, and reproduction guides.
