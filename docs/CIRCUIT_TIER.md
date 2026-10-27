# Circuit tier (draft, 2026-10-26)

Status: draft. First working version, for `edge_coupled_bpf` only. Simulation only. Every number below comes from
`families/edge_coupled_bpf/results/circuit_tier.json` or `families/edge_coupled_bpf/results/circuit_data_physical_n500.json`
(written by `families/edge_coupled_bpf/circuit_eval.py` and `tools/make_circuit_data.py`).

## Three tiers

A family can have three sources of S-parameters. They differ in cost and in the physics they contain.

| tier | what solves the device | cost per design (measured) | physics |
|---|---|---|---|
| prior | closed-form model in `prior.py` (Python) | 1.0 ms | coupled-line sections in cascade, Kirschning-Jansen even/odd modes, Kobayashi dispersion, Hammerstad open-end extension; no steps |
| circuit | qucsator through circuit-agent-tools (`circuit.py`) | 0.62-0.70 s per solve in a batch with Palace on all 8 cores (S-block circuit 0.84 s), down to 0.23 s with no EM process running; one solve alone 0.83 s | the same layout element by element: coupled lines, open ends, width steps, each a closed-form model |
| FEM | Palace 0.16, coarse mesh (production) | 663 s mean wall per record (4 ranks) | full-wave field solution of the whole layout |

With Palace running, one circuit solve in a batch costs about 600 prior evaluations and about 1/1000 of a coarse
Palace record. The circuit wall time depends on the host load: the runs go at nice 10, so they wait for CPU while
Palace uses all cores. The files
record the load at each measurement (`timing` in `circuit_tier.json`, `batches` in
`circuit_data_physical_n500.json`).

## What the circuit contains

`circuit.py` maps p = (w1, w2, g1, g2, l1, l2, eps_r, h) to a circuit-agent-tools spec of the layout that
`build_case.py` meshes. It uses `build_case.layout(p)`, so the two cannot drift apart.

- Four coupled-line sections (`mcline`): sections 1 and 4 with (w1, g1, l1), sections 2 and 3 with (w2, g2, l2).
- Eight open strip ends (`mopen`).
- Four width steps (`mstep`): W50 -> w1 at both feeds, and the w1 <-> w2 steps of resonators 1 and 3.
- Ports: 50 ohm, on the W50 side of the feed steps (x = 0 and x = 2 l1 + 2 l2). Palace de-embeds its wave ports by the
  3 mm feed, so its reference planes are at the same places and the feed step belongs to the device in both.
  Palace refers its S-parameters to the W50 feed mode. That line is 49.98 ohm at 3 GHz and 50.01 ohm at 7 GHz
  (scikit-rf MLine, `w50_feed`), so a 50 ohm port is a good match to it.
- Copper is a perfect conductor (sigma 1e30 S/m, zero thickness), as in Palace. The loss tangent is 0.0037.

qucsator models (where qucsator offers a choice, the Kirschning-Jansen one): MCOUPLED with Model and DispModel
Kirschning; MOPEN with Model Kirschning (Kirschning-Jansen-Koster end capacitance); MSTEP has one model (Gupta-Garg
shunt C and series L). The MSTEP fits are for wide/narrow width ratios of 1.5 to 3.5. The
522 steps of the 261 training designs have ratios from 1.00 to 1.52, so 97 % of them are below
the fit range, and 85 % are below 1.28, where the fitted capacitance is negative. Lint rule TLN006 reports this.
The steps are small at these ratios.

Two circuit models are measured:

- **circuit (physical)**: MCOUPLED + MOPEN + MSTEP.
- **circuit (S-block)**: the same circuit with each MCOUPLED replaced by a 4-port Touchstone block computed from the
  prior's coupled-line physics (`circuit.section_s4`). This proves the S-block-in-circuit path. Later the blocks can
  come from Palace solves of single sections.

Checks of the S-block path (`crosscheck_sblock_path`, nominal and 3 test designs):

- The prior built as a circuit of S-blocks (`prior_blocks` mode: blocks with the open-end extension, ideal open
  ends, no steps) reproduces `prior.s_prior` to max |dS| 3.5e-11 in qucsator, 7.1e-11 in circuitRF and 3.5e-11 in
  scikit-rf.
- S-blocks plus microstrip lines for the open ends (`sblock_mlin` mode, the only mode that circuitRF can run, because
  circuitRF 1.0.9 has no coupled-line, open-end or step model): circuitRF differs from qucsator by max |dS|
  1.2e-03 to 1.9e-03, scikit-rf by 3.0e-03 to 4.5e-03. The difference is the single-line MLIN model of each tool.

## Accuracy against Palace

Metrics are the ones of `core/eval.py`: mean |S21| error in dB inside the Palace 3 dB band, f0 = |S21|^2 centroid
over 3-7 GHz, -3 dB bandwidth error over single-passband designs, dead filters excluded. f0 signed = circuit minus
Palace.

| group | method | passband \|S21\| err dB, mean / max | f0 err MHz, mean abs / signed | BW err MHz | \|S11\| lin err |
|---|---|---|---|---|---|
| test (26, coarse) | prior | 2.86 / 14.0 | 61.0 / +61.0 | 113.8 | 0.050 |
| test (26, coarse) | circuit (physical) | 3.00 / 13.4 | 79.7 / +79.7 | 94.8 | 0.056 |
| test (26, coarse) | circuit (S-block) | 2.63 / 13.3 | 62.6 / +62.6 | 105.9 | 0.049 |
| test (26, coarse) | prior + DeepONet (stage B) | 0.52 / 4.0 | 7.9 / -0.7 | 10.8 | 0.011 |
| all training (261, coarse) | prior | 2.62 / 14.9 | 62.4 / +62.4 | 136.2 | 0.055 |
| all training (261, coarse) | circuit (physical) | 2.75 / 14.5 | 81.0 / +81.0 | 133.9 | 0.061 |
| all training (261, coarse) | circuit (S-block) | 2.40 / 13.8 | 63.2 / +63.2 | 132.6 | 0.054 |
| holdouts (10, medium mesh) | prior | 1.25 / 5.0 | 17.9 / +17.9 | 8.9 | 0.032 |
| holdouts (10, medium mesh) | circuit (physical) | 1.53 / 6.0 | 38.3 / +38.3 | 30.8 | 0.039 |
| holdouts (10, medium mesh) | circuit (S-block) | 1.11 / 3.9 | 19.5 / +19.5 | 4.2 | 0.030 |
| coarse twins of the holdouts (10) | prior | 2.86 / 10.1 | 52.5 / +52.5 | 30.9 | 0.051 |
| coarse twins of the holdouts (10) | circuit (physical) | 3.06 / 9.9 | 72.9 / +72.9 | 8.8 | 0.057 |
| coarse twins of the holdouts (10) | circuit (S-block) | 2.60 / 9.3 | 54.1 / +54.1 | 19.9 | 0.050 |

Findings:

1. The physical circuit is not more accurate than the prior. Its passband error is larger in every group, and its f0
   is further from Palace.
2. The S-block circuit (prior sections, qucsator open ends and steps) is a little better than the prior in passband
   error in every group. Its f0 error is about the same as the prior's.
3. Every method puts f0 too high on every group: the mean signed f0 error equals the mean absolute error. The error is
   a bias, not scatter.
4. Part of that bias is the coarse mesh. On the 10 medium-mesh holdouts the prior f0 error is 17.9 MHz; on
   their coarse twins it is 52.5 MHz. On the first (Pozar) nominal of the mesh study, Palace gives f0 4.909,
   4.967 and 4.981 GHz on the coarse, medium and fine meshes (quoted from `docs/reports/filter-fm.md`); the prior
   gives 5.008 GHz, the physical circuit 5.022 GHz and the S-block circuit 5.007 GHz (-3 dB band centre).
5. The stage-B network (prior + DeepONet, trained on coarse Palace data) is far better than both circuits on the
   test designs: 0.52 dB and 7.9 MHz.
6. At the nominal (`families/edge_coupled_bpf/figs/circuit_tier_nominal.png`) the |S11| ripple peaks of the prior
   and of both circuits are higher than those of Palace, and their lower stop-band skirt is steeper. The closed-form
   models see less coupling than Palace at the same gaps. This agrees with the nominal tuning, where Palace wanted
   the gaps wider than the prior did (`docs/reports/filter-fm.md`, sec. 2).

Why the physical circuit has a higher f0 than the prior (`mode_diagnostic_nominal`, nominal design, lossless): the
even mode agrees to 0.006 in effective permittivity, but the odd mode does not. For section 1 at 3 / 5 / 7 GHz,
qucsator's Kirschning-Jansen coupled-line dispersion gives eeff_odd 2.489 / 2.489 / 2.490 (almost no dispersion);
the prior's Kobayashi single-line dispersion, applied to each mode, gives 2.504 / 2.521 / 2.540. A faster odd mode
moves the passband up. Palace is lower than both.

## Does the circuit capture the physics the network learns?

For each design the residual is r = S_model - S_Palace (S11 and S21, complex, every frequency). The table gives the
cosine similarity of the residual vectors, pooled over the group.

| group | cos(r_circuit, r_prior), physical / S-block | share of the prior residual removed, physical / S-block | cos(circuit correction, Palace - prior), physical / S-block | Pearson r of the f0 errors, physical / S-block |
|---|---|---|---|---|
| test (26, coarse) | 0.974 / 0.998 | -24 % / 4 % | -0.32 / 0.33 | 0.98 / 0.99 |
| all training (261, coarse) | 0.974 / 0.997 | -25 % / 4 % | -0.33 / 0.31 | 0.97 / 0.99 |
| holdouts (10, medium mesh) | 0.932 / 0.992 | -69 % / 14 % | -0.41 / 0.54 | 0.83 / 0.87 |
| coarse twins of the holdouts (10) | 0.971 / 0.997 | -25 % / 6 % | -0.31 / 0.40 | 0.97 / 0.98 |

On the test designs the network's correction (S_network - S_prior) points along the Palace correction with cosine
0.988. The circuit corrections (S_circuit - S_prior) point along the network's correction with cosine
-0.33 (physical) and 0.34 (S-block).

Reading: the circuit residual is almost the prior residual (cosine 0.93 to 0.998). The physical circuit moves the
response away from Palace (negative alignment). The S-block circuit moves it a little towards Palace and removes
4 to 14 % of the prior's residual energy. The closed-form circuit models carry the same physics as the prior. The
physics that the network learns from Palace (and the coarse-mesh bias it also learns) is not in them.

## How the circuit tier would be used

**(i) Pretraining on circuit data.** Stage A pretrains on prior data today. Circuit data could replace the prior
data, or follow it as a second stage before the Palace fine-tuning. `tools/make_circuit_data.py` makes such a set
in the box of `samples.json`, with the family's validity rule
(`data/circuit/circuit_physical_n500.npz`). Measured: 500 designs, 0 failed, in 289 s
(0.58 s per design, 6220 designs per hour; 0.63-0.70 s per design in the batches with Palace running, 0.23 to
0.60 s in the batches with no Palace process). For this family the gain
is likely small, because the circuit residual is almost the prior residual. The test is a stage-A run on the
circuit set with the stage-B split fixed. It is not done.

**(ii) Input space mapping.** Learn p_c = P(p_f) from a few Palace solves so that the circuit with input p_c lines up
with Palace at p_f. The f0 error is a bias of the same sign on every design, so a simple map (for example one scale
factor on l1, l2 and a shift on g1, g2) can remove most of it. `tune.py` already did this once, for the nominal only,
with the prior as the coarse model. A circuit solve costs less than a second, so an aligned circuit can be called inside a
design loop. Not done.

**(iii) EM S-blocks from Palace (composition).** Solve single sections (a coupled pair with its open ends, or a
resonator end with its step) in Palace, store them as 4-port Touchstone files, and connect them in the circuit
(`sblock` mode with Palace blocks instead of prior blocks). The circuit then carries the EM physics of each part
and the parts can be reused over many designs. Coupling between non-adjacent strips and between section ends is
still missing; the remaining error measures it. Not done; it needs Palace runs through the launcher.

## Not done yet

- No model is trained on circuit data.
- No space-mapping fit (ii) and no Palace S-blocks (iii).
- Only qucsator runs the physical circuit. circuitRF and ngspice have no coupled-line model, so there is no second
  simulator for that mode.
- qucsator's other dispersion model (Getsinger) and the Hammerstad open-end model are not compared.
- The steps are outside the fit range of the MSTEP model (above). The coupling of a step to its neighbouring strip is
  not modelled.
- `families.json` has no `circuit` fidelity tag yet; the circuit set is tagged `fidelity = circuit` inside its npz.
- `core/data.find_prior_dataset()` takes any npz directly in `data/` as the prior dataset when
  `prior_stageA.npz` is missing (a circuit file there made `tests/test_model.py` compare it with the prior). The
  circuit set is therefore in `data/circuit/`. A loader for circuit data is not written yet.
