# Data contract: one simulated sample

Schema id: `rfsurrogates.sample/1`. One record per solved design. The record is an
npz file plus one manifest row (JSON) written by `contrib/check_sample.py --write-npz`. This page lists every field, says which
ones the trainer reads, and says how the first 281 records were made.

## Where it comes from

What the family's `collect.py` writes for a Palace run (`families/edge_coupled_bpf/data/palace/<id>.npz`), before
the checker adds the provenance:

| key | dtype / shape | content |
|---|---|---|
| `p` | float64 (8) | (w1, w2, g1, g2, l1, l2, eps_r, h), lengths in mm |
| `f` | float64 (201) | frequency in GHz, 3.0-7.0 at 0.02 (the production sweep, resampled) |
| `S11`, `S21` | complex128 (201) | S-parameters on `f`, 50 ohm reference, port 1 excited |
| `f_raw`, `S11_raw`, `S21_raw` | float64 / complex128 (161) | the solver's own sample points before resampling |
| `wall_s` | float64 scalar | wall time of the solve (s) |
| `ranks` | int64 scalar | MPI ranks |
| `substrate` | str | `RO4350B` |
| `run` | str | run folder name |
| `param_names` | str (8) | names of the entries of `p` |
| `scale` | float64 scalar | mesh scale factor (1.8 coarse, 1.4 medium, 1.0 fine) |
| `unknowns`, `tets` | int64 scalars | FEM degrees of freedom, tetrahedra |
| `peak_ram_gb` | float64 scalar | peak resident memory of the job (GB) |
| `metrics` | str (JSON) | f0, BW, IL, RL, passivity, reflection zeros, \|S11\| peaks |
| `kind`, `clearance_mm`, `note` | str / float64 | present on `nominal.npz` only |

And the matching entry in `samples.json`: `{"id", "p", "kind", "clearance_mm", "valid"}`, with the box, the
centre, the validity rule and the frequency grid at the top level of that file.

## The record (version 1)

Arrays and scalars in the npz. "Trainer" says whether `core/data.py` reads the field (T) or whether it is metadata
(M). Required fields are marked R; the others are optional but the checker flags some when they are missing.

| field | type | T/M | R | meaning |
|---|---|---|---|---|
| `schema` | str | M | R | `rfsurrogates.sample/1` |
| `family_id` | str | M | R | registry id, e.g. `edge_coupled_bpf` |
| `p` | float64 (P) | T | R | design vector in the family's units and order |
| `param_names` | str (P) | T | R | must equal the family's `param_names` |
| `units` | in the manifest row | M | R | per parameter: `mm` or `1`; taken from the family |
| `substrate` | str | T | R | substrate name; eps_r, h, tan d are in the family definition (and in p for this family) |
| `f` | float64 (M) | T | R | GHz, strictly increasing, equal to the family grid (3.0-7.0 at 0.02, 201 points) |
| `S11`, `S21` | complex128 (M) | T | R | on `f`, 50 ohm reference, complex |
| `S12`, `S22` | complex128 (M) | M | - | optional; when present the checker tests reciprocity (S12 vs S21, fail) and symmetry (S22 vs S11, review flag). Not stored in the family record: this family is reciprocal and symmetric, `S12 := S21`, `S22 := S11` (`net.to_2x2`) |
| `solver` | str | M | R | lower-case solver name: `palace`, `openems`, `hfss`, `cst`, `feko`, ... |
| `solver_version` | str | M | R | e.g. `0.16` |
| `fidelity` | str | M | R | family tag: `coarse`, `medium`, `fine`, or `other` (any setup not matching a family-defined tag) |
| `unknowns` | int | M | flag | degrees of freedom (FEM) or cells (FDTD); 0 when unknown |
| `tets` | int | M | flag | mesh elements (tets, cells, triangles) |
| `order` | int | M | flag | element order (FEM); 0 for FDTD / MoM |
| `scale` | float | M | - | Palace mesh scale (reference solver only); NaN otherwise |
| `adaptive_tol` | float | M | - | adaptive frequency sweep tolerance, if any |
| `port` | str | M | R | port definition in words: type, reference impedance, de-embedding, reference plane |
| `wall_s` | float | T (timing figure) | - | wall time of the solve (s) |
| `peak_ram_gb` | float | M | - | peak memory (GB, all ranks) |
| `ranks` | int | M | - | MPI ranks / threads |
| `contributor` | str | M | R | GitHub handle or ORCID |
| `licence` | str | M | flag | licence of this record, e.g. `CC-BY-4.0` |
| `content_sha256` | str | M | R | hash of the canonical record (below) |
| `run` | str | M | - | contributor's run name (for audit requests) |
| `metrics` | str (JSON) | M | - | f0, BW, IL, S21_max, RL as computed by the checker on the family grid |
| `f_raw`, `S11_raw`, `S21_raw` | arrays | M | - | the solver's own grid, optional; kept when the sample was resampled |

Conventions:

- Frequencies are GHz. The checker converts Hz and MHz (heuristic: max > 1e5 is Hz, max > 100 is MHz) and notes it.
- Lengths are mm. `eps_r` is dimensionless. Nothing in the record is in inches or mils.
- S is linear complex (not dB), reference impedance 50 ohm, passive sign convention (|S| <= 1).
- The frequency grid in the file is the family grid. A sample solved on another grid is accepted by the checker if
  it covers the family grid; the checker resamples (cubic spline on real and imaginary parts, as `collect.py`) and
  records `resampled_from` in the manifest row. A grid that does not cover 3-7 GHz is rejected.
- Reciprocity is assumed and S12 is not stored. Symmetry (S22 = S11) is assumed for this family because the layout is
  mirror-symmetric; a family that is not symmetric declares `stored_s: ["S11", "S21", "S22"]` in its registry entry.
  N-port families declare the full list of stored elements.

## The manifest row

One JSON object per sample, written by the checker, appended to `data_index/<family_id>/manifest.jsonl` in the pull
request. It contains every metadata field above, plus: `id`, `source_file`, `file_sha256`, `in_box`,
`clearance_mm`, `valid_geometry`, `f_grid_GHz` as `[lo, hi, step]`, `n_f`, `resampled_from` (when applicable),
`physics` (max abs S11, max abs S21, passivity max, reciprocity and symmetry residuals when testable), `metrics`,
`prior_delta` (prior f0 and BW, mean passband abs S21 difference in dB, relative f0 difference), `verdict`, `flags`,
`errors`, `notes`, `checker_version`, `checked_on`; the pull request adds `split` (`train`, `holdout`, `twin`) and
`path` (`<family_id>/<folder>/<file>.npz` in the dataset repo). Example: run
`python contrib/check_sample.py families/edge_coupled_bpf/data/palace/nominal.npz --json`.

## A contributed batch

`tools/contribute.py` stages one batch for a data pull request as `incoming/<family_id>/<batch_id>/`:

| file | content |
|---|---|
| `batch.json` | schema `rfsurrogates.batch/1`: batch id, family, contributor, licence, sampler seed, solver (name, version, fidelity, order, port), claim issue, setup notes, a sentence per NEEDS REVIEW record, and every design with its state and verdict |
| `rows.jsonl` | one manifest row per submitted record, plus `batch_id`; `path` is `<family_id>/contrib/<batch_id>/<id>.npz` |
| `records/<id>.npz` | the contract npz of each PASS and NEEDS REVIEW record |

CI (`tools/verify_batch.py`) checks the batch. After the merge, `tools/ingest_batch.py` uploads the records to the
dataset repo, appends the rows to the manifest, keeps `batch.json` as `data_index/<family_id>/batches/<batch_id>.json`
and removes the `incoming/` folder.

## Content hash

`content_sha256` = SHA-256 over, in this order, as little-endian float64 bytes: `p` rounded to 1e-9; `f` in GHz
rounded to 1e-6; `Re S11`, `Im S11`, `Re S21`, `Im S21` on the family grid, each rounded to 1e-9. Negative zero is
normalised to zero. The rounding makes the hash identical for one record stored as npz or as a full-precision
Touchstone file, and the hash is the duplicate key of the index: two rows with the same `content_sha256` are the same
solve. `file_sha256` is the plain hash of the submitted file and changes with the container.

## Checker thresholds (version 0.1.0)

| check | limit | result when violated |
|---|---|---|
| finite S, f strictly increasing | - | FAIL |
| p inside the family box | relative tolerance 1e-6 of the box width | FAIL (`--allow-out-of-box`: flag `out_of_box`) |
| validity rule | `design.clearance(p) > 0.05 mm` | FAIL |
| magnitude | max \|S11\|, max \|S21\| <= 1 + 1e-3 | FAIL |
| passivity | max(\|S11\|^2 + \|S21\|^2) <= 1 + 1e-3 | FAIL |
| reciprocity (S12 present) | max \|S12 - S21\| <= 0.02 | FAIL |
| symmetry (S22 present) | max \|S22 - S11\| <= 0.05 | flag `asymmetric` |
| prior, passband | mean abs(\|S21\| dB - \|S21_prior\| dB) over the union of the two 3 dB passbands <= 8 dB | flag `prior_passband_mismatch` |
| prior, f0 | abs(f0 - f0_prior) / f0_prior <= 10 % | flag `prior_f0_mismatch` |
| passband exists | a 3 dB band inside the grid | flag `no_passband` |
| provenance | solver, version, fidelity, contributor present | FAIL; mesh statistics or licence missing: flags `no_mesh_stats`, `no_licence` |

On the 261 Palace samples: 253 PASS, 8 flagged `prior_f0_mismatch` (split passbands), 0 FAIL. The prior's own
error against Palace is 2.86 dB mean passband |S21| and 1.2 % in f0 on the held-out set, so the review thresholds
are about three and eight times the typical prior error.

## Fidelity tags for `edge_coupled_bpf`

| tag | reference solver setup | typical size | measured bias vs fine (nominal) |
|---|---|---|---|
| `coarse` | Palace 0.16, mesh scale 1.8 (0.144 mm at copper edges), order 2, AdaptiveTol 1e-2 | 35k tets, 220k-305k unknowns | f0 -1.44 %, BW +4 %, passband \|S21\| up to 4.3 dB at the band edges |
| `medium` | scale 1.4, order 2 | 67k tets, 470k-500k unknowns | f0 -0.28 % |
| `fine` | scale 1.0, order 2 | 163k tets, 1.12 M unknowns | reference |
| `other` | any other solver or setup; the mesh statistics and the port definition must be given | - | measured per contributor against the medium holdouts |

The production data is `coarse`; the holdout set is `medium`. The checker fills the tag from `scale` for
`collect.py` files.

## How the first records were made

The 281 records of `edge_coupled_bpf` (261 training, 10 medium-mesh holdouts, 10 coarse twins) were made from the
`collect.py` files of the original experiment with

    python tools/make_records.py --src <folder with palace/, holdout/, holdout_coarse/> --contributor benjhardy

A `collect.py` file has every trainer field (`p`, `f`, `S11`, `S21`, `param_names`, `substrate`, `wall_s`) and the
mesh statistics, but no provenance keys. The checker recognises it by `scale` + `run` without `solver` and fills
the reference-solver defaults: `solver = palace`, `solver_version = 0.16`, `fidelity` from `scale` (1.8 coarse,
1.4 medium, 1.0 fine), `order = 2`, the Palace wave-port definition. The contributor and the licence are never
filled by default; they must be given. `make_records.py` writes the contract npz, checks the written file again (so
the manifest row carries the sha256 of the published file), and stops if the content hash changes or any file
fails. Result: 253 PASS and 8 NEEDS REVIEW (`prior_f0_mismatch`) on the training records, 9 PASS + 1 and 8 PASS + 2
on the holdouts and twins, 0 FAIL.

## What a reviewer looks at

The manifest row: verdict, flags, prior delta, mesh statistics against the fidelity tag, port definition, and the
contributor's note on how the solver was set up. For a new solver or a new contributor, one sample is re-solved with
the reference solver (`build_case.py --p ...` then Palace) and the two records are compared the same way the
checker compares with the prior.
