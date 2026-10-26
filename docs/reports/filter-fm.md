# Report: how the edge_coupled_bpf data and model were made (filter-fm, 2026-10-24 to 10-26)

The first family of this project came from a weekend experiment called filter-fm: can a small physics-informed
surrogate, pretrained on an analytic prior and fine-tuned on a few hundred FEM solves, predict an edge-coupled
microstrip band-pass filter well enough to design with it? This report lists what was done, what went wrong, and
the numbers. Every number comes from a file in `families/edge_coupled_bpf/results/` or a run of the original
experiment. Figures: `families/edge_coupled_bpf/figs/` (`INDEX.md` describes each one). Simulation only.

## 1. Fidelity choice (2026-10-24)

A three-level mesh study on the first nominal design (Palace 0.16, order 2, 2-8 GHz, AdaptiveTol 1e-3):

| level | tets | unknowns | f0 (GHz) | wall (8 ranks) | peak RAM |
|---|---|---|---|---|---|
| coarse, scale 1.8 | 35,203 | 254,466 | 4.909 | 13.3 min | 2.4 GB |
| medium, scale 1.4 | 67,208 | 474,608 | 4.967 | 18.6 min | 3.9 GB |
| fine, scale 1.0 | 163,375 | 1,123,486 | 4.981 | 27.5 min | 8.7 GB |
| coarse mesh, order 1 | 35,245 | 50,633 | 4.685 | 0.6 min | 1.1 GB |

Coarse is 1.44 % low in f0 against fine, medium 0.28 % low. Order 1 was rejected (f0 -5.7 %, BW +33 %). Decision:
production at the coarse level with AdaptiveTol 1e-2 on 3-7 GHz (9.7 min per sample on 8 ranks, about 1.4 times
the throughput as two concurrent 4-rank jobs), plus 10 medium-mesh holdouts to measure the bias on the trained
model. Figure: `nominal_convergence.png`.

## 2. Nominal tuning

The Pozar design solved in Palace sits at f0 4.909 GHz (prior: 5.000). A space-mapping loop (prior = coarse model,
Palace = fine model, 4 Palace iterations, `families/edge_coupled_bpf/tune_log.json`) moved the nominal to Palace f0 5.023 GHz, BW 11.2 %, IL 1.00 dB,
|S11| peaks -10.5 / -10.6 dB. The FEM wants the gaps 43 % (g1) and 11 % (g2) wider than the prior and the sections
about 2 % shorter. The box (+-30 % widths and gaps, +-15 % lengths) is centred on this tuned nominal. Figure:
`nominal_equiripple_tuning.png`.

## 3. The validity problem (2026-10-25)

The first training round on 151 samples was worse than the prior alone (prior 3.95 dB, prior + DeepONet 5.32 dB
passband error). The cause was in the data, not the model: in the centred-step layout a wider element can reach
past a coupling gap and fuse with its neighbour. About half of the first 157 finished samples had 3 instead of 5
copper conductors: shorted devices with a narrow double peak and a deep notch, not filters.

Fix: the clearance rule (`clearance(p) > 0.05 mm`, `families/edge_coupled_bpf/README.md`), 92 finished invalid
solves set aside (not part of the published data), the queue rewritten to the 260 valid box points, and a geometry
gate in `build_case.py` that counts conductors after the CAD fuse and in the copper mesh and stops before Palace.
The first valid-only round (64 samples, 7 held out) gave prior 2.29 dB, prior + DeepONet 0.71 dB. Figure:
`fused_vs_valid_layout.png`.

## 4. The sweep as run

260 valid box points + the tuned nominal, 0 failures, finished 2026-10-26 02:23. Mean 11.0 min per 4-rank job (two
at a time; 7.3-24.1 min), 258k unknowns on average (218k-305k), 2.0 GB mean peak RAM, 47.9 h of Palace job time.
Then 10 medium-mesh holdouts and 10 coarse twins of the same designs.

## 5. Final model (209 train / 26 validation / 26 test)

| method | passband \|S21\| error, dB mean / max | \|f0 error\| MHz mean | \|S11\| linear |
|---|---|---|---|
| prior only | 2.86 / 14.0 | 61 | 0.050 |
| prior + DeepONet (stage B, 5 seeds) | 0.52 / 4.0 | 7.9 | 0.011 |
| DeepONet from scratch (5 seeds) | 0.32 / 1.9 | 4.2 | 0.012 |
| plain MLP | 2.49 / 11.5 | 29 | 0.037 |

20 of the 26 held-out designs have a split passband: most of the box is detuned filters, so f0 is the
|S21|^2-weighted centroid. Figures: `method_comparison.png`, `parity_f0.png`, `examples_heldout.png`.

**Was early stopping the reason stage B lost to scratch?** Two of five stage-B seeds stopped early (patience 300 on
validation MSE) while the scratch runs went to the 6000-epoch cap. A matched-schedule run (3 seeds, 6000 epochs, no
early stop) gave 0.50 dB against 0.52 dB with the default schedule (`results/ablation.md`). So no: the prior-started
model converges to about 0.5 dB and the scratch model to about 0.3 dB at this data size.

## 6. Learning curve

`results/learning_curve.md`, `figs/learning_curve.png`. Passband error with the prior: 1.83 dB at N = 10, 1.16 at
25, 0.82 at 50, 0.62 at 100, 0.54 at 209. From scratch: 9.49, 2.62, 1.31, 0.49, 0.34. The crossover is between 50
and 100 samples. The prior keeps the lower seed spread at every N and the lower |S11| error up to about 150.

## 7. Fidelity holdouts

`results/holdout_fidelity.md`, `figs/holdout_fidelity.png`. On 10 designs solved at both meshes: model vs medium
0.91 dB / 38 MHz; coarse vs medium 1.03 dB / 35 MHz; model vs coarse 0.52 dB / 7 MHz; prior vs medium 1.25 dB /
18 MHz. The model reproduces its training mesh to 0.5 dB, and the training mesh is 1.0 dB from the finer one: the
mesh is the larger error term. The prior is closer to the medium mesh in f0 (18 MHz) than the coarse data is, which
is consistent with the coarse mesh's -1.2 to -1.4 % f0 bias.

## 8. Inverse design checked in Palace

`core/inverse.py` (multi-start Latin hypercube + L-BFGS-B on the stage-B ensemble mean, clearance as a hard
constraint, about 27 s on a CPU). Two specs, each solved once in Palace at the production fidelity:

| case | spec | Palace f0 / BW | model f0 / BW | passband \|S21\| error model / prior |
|---|---|---|---|---|
| good | f0 5.2 GHz, BW 0.5 GHz, RL 10 dB | 5.198 / 0.482 GHz | 5.200 / 0.500 GHz | 0.33 / 1.52 dB |
| bad (outside what the box can do) | f0 4.4 GHz, BW 0.12 GHz, RL 18 dB | 4.435 / 0.231 GHz | 4.451 / 0.240 GHz | 0.74 / 2.72 dB |

The good spec was met (return loss 12.0 dB in Palace over the inner 80 % of the 3 dB band against 10.2 dB
predicted). For the bad spec the optimiser went to the box edge and stopped with a large residual cost (7.8 against
0 for the good spec): the narrow bandwidth is outside the data (range in the data: f0 4.28-6.00 GHz, BW
0.05-0.90 GHz, mostly on split passbands). The model still predicted that design to 0.74 dB, so the failure is
the box, not the surrogate. Figures: `inverse_good_vs_palace.png`, `inverse_bad_vs_palace.png`.

## 9. What was not done

- Stage C (transfer to a second substrate) was not run: there is no second substrate in the Palace data.
- The ensemble spread underestimates the error (spread 0.22 dB against error 0.52 dB, r = 0.55).
- No measurement. No comparison with a commercial solver.
