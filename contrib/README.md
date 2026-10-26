# contrib: the contribution checker

`check_sample.py` is the one tool a data contributor needs. It takes one simulated sample, checks it against the
family's contract, and writes the manifest row that goes into the pull request. It uses only `registry.py`,
the family's `family.py`, `prior.py` and `design.py`, numpy and scipy; it does not need gmsh, torch or a solver.

## Usage

```
# an npz written by the family's collect.py (provenance keys missing: filled from the defaults of the original dataset)
python contrib/check_sample.py families/edge_coupled_bpf/data/palace/s0001.npz --family edge_coupled_bpf

# a Touchstone file from any solver: give p, the solver and the fidelity tag
python contrib/check_sample.py my_run.s2p --family edge_coupled_bpf \
    --p 0.9 1.1 0.18 0.6 8.5 8.5 3.66 0.508 \
    --solver openems --solver-version 0.0.36 --fidelity other \
    --mesh-elements 2400000 --mesh-unknowns 2400000 --order 0 \
    --port "lumped port 50 ohm at the feed end, 3 mm feed de-embedded" \
    --contributor <github handle> --licence CC-BY-4.0 \
    --out rows/my_run.json --write-npz samples/my_run.npz
```

Output: one verdict line, notes, and the manifest row (JSON). `--json` prints the row instead.
`--write-npz` also writes the sample as a contract npz on the family grid (this is the file that goes with the row).

| exit code | verdict | meaning |
|---|---|---|
| 0 | PASS | schema, box, validity rule and physics checks pass; the prior agrees within the review thresholds |
| 2 | NEEDS REVIEW | everything checkable passes, but a reviewer must look (flags below) |
| 1 | FAIL | the sample cannot enter the dataset as it is |

## What is checked

1. Schema and units: `p`, `f`, `S11`, `S21` present (`S12`, `S22` optional); `param_names` match the family; lengths
   in mm within 0.001-100, `eps_r` within 1-20; `f` strictly increasing and read as GHz (Hz and MHz are converted).
2. Box and validity: every parameter inside the family box (the family's `samples.json`); the family validity
   rule `is_valid(p)` from `family.py` (clearance > 0.05 mm for edge_coupled_bpf). `--allow-out-of-box` turns the box failure into the `out_of_box` review flag.
3. Grid: the family grid is 3.0-7.0 GHz at 0.02 GHz (201 points). A sample on another grid must cover it and is
   resampled by cubic spline on re/im (`collect.py` does the same with Palace output).
4. Physics: finite values; `|S11| <= 1`, `|S21| <= 1`; `|S11|^2 + |S21|^2 <= 1.001`; `max |S12 - S21| <= 0.02`
   when `S12` is present (fail); `max |S22 - S11| <= 0.05` when `S22` is present (review flag `asymmetric`).
5. Prior: the family prior (`family.prior(p, f)`) for the same design. Review flags: `prior_passband_mismatch` when the mean
   passband `|S21|` difference exceeds 8 dB, `prior_f0_mismatch` when f0 differs by more than 10 %, `no_passband`
   when no 3 dB band exists inside the grid. These never reject: the prior is what the model corrects.
6. Provenance: solver, version, fidelity tag and contributor are required; mesh statistics and a licence are flagged
   when missing (`no_mesh_stats`, `no_licence`).
7. Hashes: `file_sha256` of the input file and `content_sha256` of the canonical record (p, f, S11, S21 on the family
   grid, rounded to 1e-9 / 1e-6 GHz / 1e-9), so the same record hashes the same as npz or as Touchstone.

On the 261 Palace training records of edge_coupled_bpf (2026-10-26): 253 PASS, 8 NEEDS REVIEW (`prior_f0_mismatch`, all
split-passband designs where the -3 dB band centre sits on one of the two peaks), 0 FAIL.

## Tests

```
python -m pytest tests/test_check_sample.py -q
```

`tests/test_check_sample.py` runs the checker on `families/edge_coupled_bpf/data/palace/nominal.npz` (PASS), on corrupted copies (NaN,
non-passive, out of box, missing key: FAIL), on an npz-to-Touchstone round trip (same content hash), and on
synthetic samples built from the prior (resampling, Hz conversion, missing provenance, fused geometry,
non-reciprocal S, out-of-box review, a 15 % frequency shift that must be flagged and not rejected, hash
determinism). The tests that need `nominal.npz` are skipped when the records are not on disk (`python tools/fetch.py --data-only`).

The record itself is described in `docs/DATA_CONTRACT.md`; the contributor path in `CONTRIBUTING.md`.
