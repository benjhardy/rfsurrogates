# Fidelity policy

A surrogate can only be as close to the truth as its training data. The data here comes from meshes coarse enough
to solve hundreds of designs, so every family measures how far its production mesh is from a finer one and
publishes that number next to the model error.

## Rules

1. **Tags.** Every record carries a fidelity tag from its family's list (`families.json`, `fidelity_tags`):
   `coarse`, `medium`, `fine` for the family's reference-solver setups, `other` for anything else. Every record also
   carries the solver name and version and its mesh statistics (elements, unknowns or cells, element order), so a
   record tagged `other` can still be audited.
2. **Production and holdout fidelity.** Each family names a production fidelity (the training data) and a finer
   holdout fidelity. A set of designs is solved at the holdout fidelity and never trained on. The same designs are
   also solved at the production fidelity ("twins"), so the mesh bias is measured on exactly those designs.
3. **Published side by side.** The model card always shows model-vs-holdout next to production-vs-holdout. For
   `edge_coupled_bpf` (10 medium-mesh holdouts): model vs medium 0.91 dB mean passband |S21| and 38 MHz in f0;
   coarse vs medium 1.03 dB and 35 MHz; model vs coarse 0.52 dB and 7 MHz. At this data size the mesh bias is the
   larger error term.
4. **Mixed data.** Training data may mix solvers and fidelities. The (solver, fidelity) pair is a column of the
   manifest, so a training run can select, weight or hold out any group. The model card lists the groups it was
   trained on.
5. **Finer records go to the holdout first.** A contributed record at a finer fidelity than production goes to the
   family's holdout set until that set has at least 10 % of the training count.
6. **Pairs.** Two records of the same p at different fidelities measure the bias directly; the index keeps both.

## Records from other solvers

Records from solvers or setups other than the family's reference setup are tagged `other` and form their own
group. Each new group is measured against the family's holdout designs (a contributor solves some of those designs,
or a reviewer re-solves some of the contributor's designs with the reference setup).

The project publishes these measurements per fidelity group, not per solver: it does not publish accuracy tables
for a named commercial solver, and it does not publish comparisons between solvers. Contributors are responsible
for what their own solver licence allows them to share.

## Reference setups of edge_coupled_bpf

| tag | setup | nominal f0 vs fine | role |
|---|---|---|---|
| coarse | Palace 0.16, mesh scale 1.8, order 2, AdaptiveTol 1e-2 | -1.44 % | production (training) |
| medium | Palace 0.16, mesh scale 1.4, order 2 | -0.28 % | holdout (10 designs + 10 coarse twins) |
| fine | Palace 0.16, mesh scale 1.0, order 2 | reference | solved once, for the nominal |

Order 1 on the coarse mesh was rejected (f0 -5.7 %, BW +33 %). Mesh details: `families/edge_coupled_bpf/README.md`.
