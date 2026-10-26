# Contributing

> **Status (v0.1, 2026-10-26): data contributions open with v0.2**, when the checker runs in CI on every data pull
> request. Until then, the steps below are the plan and they work locally; please open an issue first if you want
> to prepare data. Code issues and pull requests are welcome now.

This repo trains surrogate models of passive microwave devices from full-wave solves. The data comes from
contributors who run a solver on their own machines. This page is the path from "I have a solver and some CPU time"
to a merged pull request. The record you produce is defined in `docs/DATA_CONTRACT.md`, the fidelity rules are in
`docs/FIDELITY.md`, and the project design is in `docs/OPEN_PROJECT.md`.

## What it costs

Measured for `edge_coupled_bpf` on one 8-thread desktop (Palace 0.16 in Apptainer under WSL2, 25 GB):

| fidelity | setup | per sample | peak RAM | notes |
|---|---|---|---|---|
| coarse (production) | mesh scale 1.8, order 2, 4 ranks, two jobs at a time | 11 min mean (7-24 min) | 2.0 GB mean, 2.5 GB max | about 258k unknowns; 9.7 min serial on 8 ranks |
| medium (holdout) | scale 1.4, order 2, 8 ranks | about 15-20 min | about 3.9 GB | 470k-500k unknowns |
| fine (reference) | scale 1.0, order 2, 8 ranks | 27.5 min | 8.7 GB | 1.12 M unknowns; solved once, for the nominal |

So a batch of 50 coarse samples is about 5 hours on a desktop with two jobs at a time. Other solvers cost what they
cost; the record carries wall time and RAM.

## The path

### (a) Pick or propose a device family

The registered families are in `families.json`. Today there is one: `edge_coupled_bpf`, the edge-coupled microstrip
band-pass filter on RO4350B, p = (w1, w2, g1, g2, l1, l2, eps_r, h) in mm; box and design list in
`families/edge_coupled_bpf/samples.json`.

To propose a family, open an issue with: the device, the parameter vector and its box, the analytic prior you would
use (a textbook reference is enough), the validity rule (which geometries are not the device), the frequency grid,
the port definition, and the metrics that define "good" for that device. Next on the roadmap: the hairpin
band-pass filter.

### (b) Draw designs from the family's box

Designs must come from the documented box and pass the validity rule, so that the data stays a sample of one
well-defined set. Use the family's own sampler:

```
cd families/edge_coupled_bpf
python design.py --extend 300 10        # annotates every design with clearance/valid and draws new valid ones
                                        # (Latin hypercube in the box, rejection-sampled on the clearance rule)
```

Or check one design of your own against the rule before you solve it (from the repo root):

```
python -c "import registry; f = registry.load_family('edge_coupled_bpf'); p = [0.9, 1.1, 0.18, 0.6, 8.5, 8.5, 3.66, 0.508]; print(f['clearance'](p), f['is_valid'](p))"
```

The rule for this family is `clearance(p) > 0.05 mm` with
`clearance = min(g1 - max(0, (w2 - w1)/2), g2 - max(0, (w1 - w2)/2), g1 - max(0, (W50 - w1)/2))`. A design that
fails it is a shorted device (fused copper), not a filter. Do not solve it; the checker rejects it anyway.

To reserve ids so that two contributors do not solve the same designs, open an issue
"claim: edge_coupled_bpf sNNNN-sMMMM", or state the seed you gave `design.py --extend`.

### (c) Run the solver on your machine

**Palace (the reference solver).** The generator writes the mesh and the config; the geometry gate runs before any
solve:

```
cd families/edge_coupled_bpf
python build_case.py --id s0601 --scale 1.8 --tol 1e-2 --sweep narrow          # by id from samples.json
python build_case.py --p 0.9 1.1 0.18 0.6 8.5 8.5 3.66 0.508 --name my_run --scale 1.8 --tol 1e-2 --sweep narrow
#   -> runs/<name>/<name>.msh, config.json, build_summary.json (tets, unknowns, conductor counts)
#   exit code 3 and no config.json when the gate rejects the layout
```

`--scale 1.8 --tol 1e-2 --sweep narrow` is the production (coarse) setting (`prod_build_args.txt`); `--scale 1.4`
is the medium holdout setting (`holdout_build_args.txt`). Then solve with Palace 0.16 (`palace -np 4 config.json`
or your own wrapper) and collect:

```
cd runs/my_run && /usr/bin/time -v palace -np 4 config.json > palace.log 2> time.log && cd ../..
python collect.py --run my_run --ranks 4        # -> data/aux/my_run.npz (sNNNN ids go to data/palace)
```

`run_queue.sh` runs a list of ids unattended (one job per line, `par: a 4, b 4` for two at a time; stops when a
`STOP` file exists) and calls `collect.py` after each run. `collect.py` writes the npz on the 3-7 GHz / 0.02 GHz
grid with the solver statistics read from the logs.

**Other solvers (openEMS, Elmer, MEEP, HFSS, CST, FEKO, ...).** Build the same geometry: `build_case.layout(p)`
returns every copper rectangle and the feed positions; the stack is substrate (eps_r, h, tan d) on a PEC ground,
copper as a zero-thickness PEC sheet, 50 ohm feeds of width W50 = 1.112 mm and length 3 mm, ports de-embedded to the
start of the coupled section. Export a two-port Touchstone file (`.s2p`, any unit, RI / MA / DB) on any grid that
covers 3-7 GHz; the checker resamples it to the family grid. Write down the solver version, the mesh size (cells or
elements, unknowns), the element order if FEM, and the port type.

Commercial solvers: you are responsible for what your solver licence allows you to share. The project does not
publish accuracy comparisons per solver or between solvers. Records from any solver other than the reference setup
are grouped as fidelity `other` and measured only against the family's Palace holdouts (`docs/FIDELITY.md`).

### (d) Run the checker; it also writes the contract record

```
python contrib/check_sample.py families/edge_coupled_bpf/data/palace/s0601.npz --family edge_coupled_bpf \
    --contributor <github handle> --licence CC-BY-4.0 --out rows/s0601.json --write-npz samples/s0601.npz

python contrib/check_sample.py my_run.s2p --family edge_coupled_bpf \
    --p 0.9 1.1 0.18 0.6 8.5 8.5 3.66 0.508 \
    --solver openems --solver-version 0.0.36 --fidelity other \
    --mesh-elements 2400000 --mesh-unknowns 2400000 --order 0 \
    --port "lumped 50 ohm port at the feed end, 3 mm feed de-embedded" \
    --contributor <github handle> --licence CC-BY-4.0 \
    --out rows/my_run.json --write-npz samples/my_run.npz
```

Then run the checker once more on the written contract npz (`samples/<id>.npz`): that row, with the sha256 of the
file you submit, is the one that goes into the pull request.

Verdicts: `PASS` (exit 0), `NEEDS REVIEW` (exit 2: physically fine but far from the prior, asymmetric, out of box
with `--allow-out-of-box`, or missing mesh statistics / licence), `FAIL` (exit 1: schema, box, validity rule, NaN,
passivity, reciprocity or missing provenance). `contrib/README.md` lists every check and threshold. A `FAIL` is not
negotiable; a `NEEDS REVIEW` needs one sentence from you on why the sample looks the way it does.

### (e) Submit

1. **Files**: open a pull request on the Hugging Face dataset `rfsurrogates/samples` that adds your contract npz
   files under `<family_id>/palace/` (or `holdout/` for finer-mesh records, see `docs/FIDELITY.md`).
2. **Rows**: open a pull request here that appends your rows to `data_index/<family_id>/manifest.jsonl` (one JSON
   object per line, as the checker wrote them, plus `"split"` and `"path"`), with a link to the dataset pull request.
   Title: `data: <family_id>, N samples, <solver> <fidelity>`.
3. Add one paragraph on your setup: machine, solver build, anything that differs from the generator's layout.

If you cannot use Hugging Face, open an issue with the rows attached and a link to the files (a release asset or a
Zenodo record you own).

### (f) What reviewers check

- The checker verdict of every row, rerun on the submitted files; `file_sha256` and `content_sha256` must match.
- Mesh statistics against the fidelity tag; port definition against the family's; frequency coverage.
- Duplicates: the same `content_sha256`, or the same p as an existing row.
- For a new solver or a new contributor: one of the designs is solved again with Palace at the production setting
  and compared (passband |S21|, f0, BW); the result is noted on the pull request. A large difference does not reject
  the batch; it is recorded and the batch becomes a separate fidelity group.
- The licence field (CC-BY-4.0 for the dataset).

### (g) Credit

Every row carries `contributor`. The model card of each family lists the contributors with their record counts and
fidelity groups, and the data index is the record of who computed what. Releases list the data index version they
were trained on.

## Retraining

Merged data does not retrain anything by itself. A maintainer retrains a family locally and cuts a release (weights,
regenerated model card, data index version) when the family has about 25 % more records or a new fidelity group.

## Code contributions

Code (a new family, a prior, a checker improvement) follows the usual path: an issue first for anything beyond a
fix, tests next to the code (`python -m pytest tests -q`), no change to training code in a data pull request. Sign
off your commits with a Developer Certificate of Origin line (`git commit -s`).
