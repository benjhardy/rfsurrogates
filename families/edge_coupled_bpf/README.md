# edge_coupled_bpf: edge-coupled microstrip band-pass filter

Family definition, generator, prior and reference-solver setup. The registry entry is in `families.json`; the
model card (accuracy, fidelity, limits) is `MODEL_CARD.md`; the history of how the data and the model were made is
`docs/reports/filter-fm.md`. Simulation only; nothing here was measured.

## The device

Edge-coupled (parallel-coupled) microstrip band-pass filter, Chebyshev N = 3 (0.5 dB ripple), designed for
f0 = 5 GHz and 10 % bandwidth with the Pozar coupled-line method (ch. 8.7). Four coupled-line sections; sections
1 and 4 share dimensions (w1, g1, l1), sections 2 and 3 share dimensions (w2, g2, l2). Substrate: Rogers RO4350B,
eps_r 3.66, h 0.508 mm, tan d 0.0037; copper as a zero-thickness PEC sheet; 50 ohm feeds (W50 = 1.112 mm).

Parameter vector p = (w1, w2, g1, g2, l1, l2, eps_r, h), lengths in mm:

| symbol | meaning | Pozar nominal | box |
|---|---|---|---|
| w1 | strip width of sections 1, 4 (input/output strips, outer parts of resonators 1 and 3) | 0.8834 | +-30 % |
| w2 | strip width of sections 2, 3 (resonator 2, inner parts of resonators 1 and 3) | 1.0771 | +-30 % |
| g1 | gap (edge to edge) of sections 1, 4 | 0.1225 | +-30 % |
| g2 | gap of sections 2, 3 | 0.5319 | +-30 % |
| l1 | coupled length of sections 1, 4 | 8.7411 | +-15 % |
| l2 | coupled length of sections 2, 3 | 8.6560 | +-15 % |
| eps_r, h | substrate | 3.66, 0.508 | fixed |

The box centre is the Palace-tuned nominal (space mapping, `tune.py`, `tune_log.json`: gaps 43 % / 11 % wider
than the prior's and sections about 2 % shorter). `samples.json` carries the centre, the box, every design point
with its clearance and validity, and the frequency grid.

Geometry: l1, l2 are the coupled (overlap) lengths = the stagger between neighbouring strips. Resonators 1 and 3
are l1 + l2 long with a w1 -> w2 width step centred on the strip axis; resonator 2 is 2 l2 long. No open-end
compensation in the layout. `figs/nominal_layout_params.png` draws and labels it; `build_case.layout(p)` returns
every copper rectangle and the feed positions.

Output per sample: S11 and S21 (complex) on 3-7 GHz at 0.02 GHz (201 points), 50 ohm, referenced to the start of
the coupled section. The layout is mirror-symmetric and reciprocal, so S22 = S11 and S12 = S21 are not stored.

## Validity rule

In the centred-step layout a wider element can reach past a coupling gap and fuse with its neighbour. Such designs
have 3 instead of 5 copper conductors: shorted devices, not filters. Rule:

    clearance = min(g1 - max(0, (w2 - w1)/2), g2 - max(0, (w1 - w2)/2), g1 - max(0, (W50 - w1)/2)) > 0.05 mm

0 < clearance < 0.05 mm is a corner-to-corner near-touch that the production mesh (0.144 mm at copper edges) cannot
resolve, so it is excluded too. 260 of the 600 box points of the original sample pass. `build_case.py` also counts
the conductors after the CAD fuse and in the copper mesh and stops (exit code 3) before Palace when the count is not
5 (`tests/test_build_gate.py`). The tuned nominal has clearance 0.061 mm: valid, in the marginal band.
`figs/fused_vs_valid_layout.png` shows a fused and a valid layout with their responses.

## Prior

`prior.py`: prototype values by the closed-form recurrence (Matthaei, Young & Jones 1980, sec. 4.05; Pozar eq.
8.72), inverters by Pozar eq. 8.121, Kirschning-Jansen coupled microstrip (static) on Hammerstad-Jensen single-line
formulas, exact coupled-line section two-ports (Pozar 8.102-8.104, separate even/odd phase velocities), Hammerstad
open-end extension, dielectric loss via the filling factor, Kobayashi dispersion. `s_prior(p, f_ghz) -> (S11, S21)`.
`tests/test_prior.py` (16 checks): single-line Z0/eps_eff vs scikit-rf `MLine` to 1e-4; prototype tables and
ripple; Pozar Example 8.8 impedances (to 0.02 ohm) and response; every section built independently in scikit-rf
from even/odd TEM lines matches the ABCD path to 1e-8; the synthesised lossless filter is equiripple.

## Reference solver setup (Palace 0.16)

- Domain: substrate slab + 8 mm of air; 6 mm side margin beyond the copper; first-order absorbing walls and top;
  PEC ground; copper = PEC sheet in the z = h plane.
- Ports: wave ports on the two x faces (whole face), mode 1, `Offset` = 3 mm feed length, so the S-parameters are
  referenced to the start of the coupled section (the W50 -> w1 step is part of the device).
- Sweep (production): 3-3.4 GHz at 0.1, 3.5-6.5 at 0.02, 6.6-7 at 0.1 (161 points), adaptive PROM (`AdaptiveTol`
  1e-2), resampled by cubic spline to the 0.02 GHz grid; port 1 excited.
- Elements: order 2; MUMPS direct + GMRES; `EstimatorMG` on, `EstimatorTol` 1e-4.
- Mesh levels (`--scale` multiplies every size: 0.08 x scale at copper edges, 0.8 x scale near the copper, 1.2 x
  scale in the substrate, 2.5 x scale in air):

| tag | scale | tets / unknowns | nominal f0 vs fine | cost on an 8-thread desktop |
|---|---|---|---|---|
| coarse (production) | 1.8 | 35k / 218k-305k | -1.44 % | 11.0 min mean per 4-rank job, two jobs at a time; 2.0 GB |
| medium (holdout) | 1.4 | 67k / 470k-500k | -0.28 % | about 15-20 min on 8 ranks; 3.9 GB |
| fine (reference) | 1.0 | 163k / 1.12 M | reference | 27.5 min on 8 ranks; 8.7 GB |

Order 1 on the coarse mesh was rejected (f0 -5.7 %, BW +33 %). The coarse level carries an f0 bias of about -1.2
to -1.4 % against a converged answer; the 10 medium-mesh holdouts measure what that costs on the trained model
(`MODEL_CARD.md`, `docs/FIDELITY.md`).

## Files

| file | what it does |
|---|---|
| `family.py` | what `registry.load_family()` exposes: box, validity rule, prior, passband metrics |
| `prior.py` | analytic prior (above) and `passband_metrics`, `passband_features` |
| `design.py` | Pozar synthesis -> p, prior-level equiripple tune, sampling box, LHS sampler -> `design.json`, `samples.json`; `--extend` adds valid designs; `--stage-a` writes `data/prior_stageA.npz` (20,000 prior samples, 3 substrates) |
| `build_case.py` | p -> Gmsh mesh (OCC, size fields at the copper edges) + Palace config + geometry gate; `--id` from `samples.json` or `--p` for any design |
| `collect.py` | Palace `port-S.csv` + logs -> `data/<split>/<id>.npz` with p, f, S11, S21, raw points, wall time, peak RAM, mesh statistics; `--db` adds an rflab run record |
| `run_queue.sh` | unattended Palace queue (one job per line, `par:` for concurrent jobs, `STOP` file) |
| `tune.py` | space-mapping tune of the nominal in Palace (prior = coarse model, Palace = fine model) |
| `results/`, `figs/` | metrics, learning curve, fidelity holdouts, figures (`figs/INDEX.md`) |

## Commands

```
cd families/edge_coupled_bpf
python design.py --extend 300 10                                 # more valid designs in samples.json
python build_case.py --id s0601 --scale 1.8 --tol 1e-2 --sweep narrow
RUN_PALACE=run_palace bash run_queue.sh queue.txt                 # or run Palace yourself in runs/<id>/
python collect.py --run s0601 --ranks 4
python ../../contrib/check_sample.py data/palace/s0601.npz --contributor <handle>
```

`build_case.py` and `collect.py` need gmsh and `palace-agent-tools` (`pip install -r requirements.txt`; set
`PALACE_AGENT_TOOLS` to use a local checkout instead). `run_queue.sh` runs under Linux or WSL; set `RFS_PYTHON` to
the Python that has those packages and `RUN_PALACE` to a command that takes `config.json <ranks>`.
