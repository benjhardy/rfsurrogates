# rfsurrogates

Surrogate models of passive microwave devices, trained on full-wave simulations, for fast analysis and inverse
design. There is one model per device family. A model maps a geometry vector to complex S-parameters in
milliseconds instead of minutes, and an optimiser on the model finds a geometry for a specification.

Training needs a few hundred full-wave solves per family. The plan is to share that compute: contributors simulate
designs of a registered family with their own solver on their own machines, check each result with one script, and
submit the records. The project trains the models and publishes the weights. The model error and the mesh error of
the training data are published next to each other.

Status v0.1 (2026-10-26): one family, data from one contributor, contributions open with v0.2 (see
[Roadmap](#roadmap)).

| family | data | model | held-out passband \|S21\| error | page |
|---|---|---|---|---|
| [`edge_coupled_bpf`](families/edge_coupled_bpf/README.md): edge-coupled microstrip band-pass filter, RO4350B, 3-7 GHz | 261 Palace FEM solves + 10 finer-mesh holdouts | analytic prior + DeepONet correction, 5 seeds | 0.52 dB (prior alone 2.86 dB) | [browser demo](https://benjhardy.github.io/filter-surrogate/) |

Simulated against simulated. Nothing here was measured. Details and limits: the
[model card](families/edge_coupled_bpf/MODEL_CARD.md).

Coverage: [docs/COVERAGE.md](docs/COVERAGE.md) lists 31 passive device types and the 20 building blocks they are made
of, with the status of each (surrogate, planned, experiment, open). `web/coverage.html` shows the same map as an
interactive graph.

## Quick start

```
git clone https://github.com/benjhardy/rfsurrogates && cd rfsurrogates
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python tools/fetch.py                                 # weights and records of edge_coupled_bpf from Hugging Face
python core/inverse.py --f0 5.2 --bw 0.5 --rl 10      # geometry for a spec, about 30 s on a CPU
python core/inverse.py --list-box                     # the design box and the range the data covers
python core/predict.py && python core/eval.py         # rebuild the held-out metrics from the weights
python -m pytest tests -q
```

`core/inverse.py` writes the geometry, the predicted response and the exact `build_case.py` command to check the
result with Palace.

## How it works

For each family:

1. **Generator**: p -> geometry, mesh and solver input for the reference solver (Palace), with a geometry gate that
   stops a layout that is not the intended device before any solve.
2. **Analytic prior**: p, f -> S from textbook formulas. The network learns a correction to it.
3. **Box and validity rule**: the parameter ranges and a cheap test that a design is the intended device.
4. **Data**: one checked record per solved design (`docs/DATA_CONTRACT.md`), indexed in
   `data_index/<family_id>/manifest.jsonl`.
5. **Model**: S = S_prior + DeepONet(p, f); stage A pretrains on prior samples, stage B fine-tunes on solver data;
   a DeepONet without the prior is trained on the same data as a baseline.
6. **Fidelity**: every family keeps designs solved at a finer mesh that are never trained on, and reports
   model-vs-fine next to coarse-vs-fine (`docs/FIDELITY.md`).

The trained models are per family. Whether one model can share structure across families (for example a common
trunk) is a research question for when there are two or three families; the result will be reported either way.

## Layout

| path | content |
|---|---|
| `families.json`, `registry.py` | the device-family registry |
| `coverage.json` | the coverage map: device types, building blocks, status (`tools/make_coverage.py` writes `docs/COVERAGE.md` and `web/coverage.json`) |
| `families/<family_id>/` | generator, prior, box (`samples.json`), validity rule (`family.py`), results, figures, model card |
| `core/` | data loader, network, trainer, evaluation, inverse design, prediction from checkpoints, web export |
| `contrib/check_sample.py` | the checker every record goes through |
| `data_index/<family_id>/manifest.jsonl` | one row per record: provenance, checks, hashes |
| `tools/` | records from solver output, model card, Hugging Face staging and fetch |
| `web/` | the browser page: a dependency-free JavaScript port of the prior and the network, and `rfscene.js`, a reusable 3D view of layered devices (`web/README.md`) |
| `docs/` | data contract, fidelity policy, project design, coverage, agent roles (draft), experiment reports |

## Data and weights

- Weights: Hugging Face model repo per family, `rfsurrogates/edge-coupled-bpf` (checkpoints, `weights.json` for
  the browser page, held-out predictions, model card).
- Records: Hugging Face dataset `rfsurrogates/samples`, one folder per family. The manifest rows in this repo carry
  the sha256 of every file; `tools/fetch.py` checks them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Data contributions open with v0.2; code issues and pull requests are
welcome now.

## Roadmap

- **v0.1** (this): repo, registry, checker, `edge_coupled_bpf` data and weights, model card, browser page.
- **v0.2**: checker in CI for data pull requests (hashes and verdicts rerun on the submitted files); data
  contributions open; second family, hairpin band-pass filter (same coupled-line prior, folded geometry).
- **v0.3**: shared-trunk experiment across the two filter families; a second substrate (stage C, transfer with about
  30 samples).

## Licence

Code and weights: Apache-2.0 (`LICENSE`). Records and data index: CC-BY-4.0 (`DATA_LICENSE`); each record names
its contributor. Cite with `CITATION.cff`.
