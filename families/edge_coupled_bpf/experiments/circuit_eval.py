"""Measure the circuit tier against Palace: prior alone, qucsator circuit, S-block circuit (writes results/circuit_tier.json).

    python circuit_eval.py [--reuse]        # --reuse: take the circuit S-parameters from runs/circuit/eval_cache.npz

Groups: the 26 held-out test designs of the stage-B split (core/data.split_indices, seed 0), all 261 coarse-mesh
training records (data/palace, the test designs included), the 10 medium-mesh fidelity holdouts (data/holdout) and
their coarse twins (data/holdout_coarse). Metrics are core/eval.per_sample_metrics + summarize (passband |S21| error
in dB inside the Palace 3 dB band, f0 = |S21|^2 centroid, dead filters excluded), so the numbers compare directly
with results/metrics.json and MODEL_CARD.md.

Residual correlation: for each method X, the residual r_X = S_X - S_Palace (S11 and S21, complex, every frequency,
pooled over the group). cos(r_circuit, r_prior) = Re<r_c, r_p> / (|r_c| |r_p|) says how much of the circuit's error
is the prior's error. The correction alignment cos(S_c - S_prior, S_Palace - S_prior) says whether the circuit moves
the prior towards Palace; "explained" = 1 - |r_c|^2 / |r_p|^2. On the test designs the same is done against the
stage-B network's correction (results/preds_stageB.npz), the physics the network learned. Pearson r of the
per-sample f0 errors is given too.

Circuit runs: qucsator through circuit-agent-tools, one netlist at a time, nice 10, in one WSL call per batch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]     # the family folder (this script is in experiments/)
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "core"))
sys.path.insert(0, str(HERE))
import circuit  # noqa: E402
import data as D  # noqa: E402
import eval as E  # noqa: E402
import prior  # noqa: E402

RESULTS = HERE / "results"
CACHE = circuit.RUN_ROOT / "eval_cache.npz"
METHODS = ("prior", "circuit_physical", "circuit_sblock")


def _cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.real(np.vdot(b, a)) / den) if den > 0 else float("nan")


def _pack(s11, s21):
    return D.pack_np(s11, s21)


def _load_folder(sub):
    ds = D.load_palace_samples(HERE / "data" / sub)
    return ds


def correlations(f, s_pal, s_pri, s_cir, per_pri, per_cir, live, s_net=None):
    """Residual and correction similarity for one group (arrays in the real (N,4,M) layout)."""
    def cplx(s):
        return np.stack([s[:, 0] + 1j * s[:, 1], s[:, 2] + 1j * s[:, 3]], axis=1)[live]
    pal, pri, cir = cplx(s_pal), cplx(s_pri), cplx(s_cir)
    r_p, r_c = pri - pal, cir - pal
    out = {"cos_residual_circuit_vs_prior": _cos(r_c, r_p),
           "cos_residual_circuit_vs_prior_S21": _cos(r_c[:, 1], r_p[:, 1]),
           "cos_correction_circuit_vs_palace": _cos(cir - pri, pal - pri),
           "explained_fraction_of_prior_residual": float(1 - np.sum(np.abs(r_c) ** 2) / np.sum(np.abs(r_p) ** 2)),
           "pearson_f0_err_circuit_vs_prior": float(np.corrcoef(per_cir["f0_err_mhz"][live],
                                                                per_pri["f0_err_mhz"][live])[0, 1]),
           "per_sample_cos_residual_median": float(np.median([_cos(r_c[i], r_p[i]) for i in range(len(r_c))]))}
    if s_net is not None:
        net = cplx(s_net)
        out["cos_correction_circuit_vs_network"] = _cos(cir - pri, net - pri)
        out["cos_correction_network_vs_palace"] = _cos(net - pri, pal - pri)
    return out


def solve_all(P, names, reuse):
    """S (N,4,M real layout) for the two circuit modes, from the cache or by running qucsator in batches."""
    if reuse and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["names"]) == list(names):
            return {m: z[m] for m in METHODS[1:]}, json.loads(str(z["timing"]))
    from circuit_tools import run as runmod

    out, timing = {}, {}
    for mode in ("physical", "sblock"):
        st = runmod.host_status()
        r = circuit.solve_batch(P, mode, names=names, case_dir=circuit.RUN_ROOT / f"eval_{mode}")
        if r["failed"]:
            raise RuntimeError(f"{mode}: {len(r['failed'])} failed, first {r['failed'][0]}")
        out[f"circuit_{mode}"] = _pack(r["S11"], r["S21"])
        timing[f"batch_{mode}"] = {"n": r["n"], "wall_s": r["wall_s"], "s_per_solve": round(r["wall_s"] / r["n"], 4),
                                   "host": {k: st.get(k) for k in ("cpus", "load_1m", "heavy_em_processes")}}
        print(f"[{mode}] {r['n']} solves in {r['wall_s']:.1f} s ({r['wall_s'] / r['n']:.3f} s each)")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, names=np.array(names), timing=json.dumps(timing), **out)
    return out, timing


def single_solve_timing(P, n=5):
    """End-to-end time of one circuit-agent-tools solve (spec, netlist, one WSL start, qucsator, parse)."""
    from circuit_tools import run as runmod

    st = runmod.host_status()
    out = {"host": {k: st.get(k) for k in ("cpus", "load_1m", "heavy_em_processes", "ram_available_gb")}}
    for mode in ("physical", "sblock"):
        t = [circuit.solve(p, mode, name=f"timing{i}", check_host=False)["wall_s"] for i, p in enumerate(P[:n])]
        out[mode] = {"n": n, "wall_s_mean": float(np.mean(t)), "wall_s_min": float(np.min(t)),
                     "wall_s_max": float(np.max(t))}
    t0 = time.time()
    for p in P[:50]:
        prior.s_prior(p, circuit.F_GHZ)
    out["prior_python"] = {"n": 50, "wall_s_mean": (time.time() - t0) / 50}
    return out


def crosscheck(P, names):
    """sblock_mlin in qucsator, circuitRF and scikit-rf; prior_blocks vs prior.s_prior."""
    rows = []
    for p, nm in zip(P, names):
        q = circuit.solve(p, "sblock_mlin", "qucsator", name=f"xc_{nm}", check_host=False)
        row = {"id": nm}
        for sim in ("circuitrf", "scikit-rf"):
            r = circuit.solve(p, "sblock_mlin", sim, name=f"xc_{nm}", check_host=False)
            row[f"{sim}_vs_qucsator_max_abs_dS"] = float(max(np.abs(r["S11"] - q["S11"]).max(),
                                                             np.abs(r["S21"] - q["S21"]).max()))
            row[f"{sim}_wall_s"] = r["wall_s"]
        s11, s21 = prior.s_prior(p, circuit.F_GHZ)
        for sim in ("qucsator", "circuitrf", "scikit-rf"):
            r = circuit.solve(p, "prior_blocks", sim, name=f"xc_{nm}", check_host=False)
            row[f"prior_blocks_{sim}_vs_s_prior_max_abs_dS"] = float(max(np.abs(r["S11"] - s11).max(),
                                                                         np.abs(r["S21"] - s21).max()))
        rows.append(row)
        print("[crosscheck]", row)
    return rows


def _abcd(s, z0=50.0):
    s11, s12, s21, s22 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
    a = ((1 + s11) * (1 - s22) + s12 * s21) / (2 * s21)
    b = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)
    c = ((1 - s11) * (1 - s22) - s12 * s21) / (2 * z0 * s21)
    return a, b, c


def mode_diagnostic(p, f_pts=(3.0, 5.0, 7.0)):
    """Even/odd Z and eeff of sections 1 and 2 of p: qucsator MCOUPLED vs the prior's 4-port (section_s4).

    Both 4-ports are split into even (S_AA + S_AB) and odd (S_AA - S_AB) two-ports; cos(theta) = A of the ABCD matrix,
    Z = sqrt(B / C), eeff = (Re theta / (k0 l))^2."""
    from circuit_tools import engine

    w1, w2, g1, g2, l1, l2, eps_r, h = prior.unpack(p)
    f = np.asarray(f_pts, float)
    secs = {"section_1": (w1, g1, l1), "section_2": (w2, g2, l2)}
    sub = {circuit.SUB: {"er": eps_r, "h": h * 1e-3, "t": 0.0, "tand": 0.0, "sigma": circuit.SIGMA_PEC}}
    S = {}
    for nm, (w, g, l) in secs.items():
        nd = [f"n{j + 1}" for j in range(4)]
        comps = [{"type": "port", "name": f"P{j + 1}", "nodes": [nd[j], "0"], "num": j + 1} for j in range(4)]
        comps.append({"type": "mcline", "name": nm, "nodes": nd, "w": w * 1e-3, "s": g * 1e-3, "l": l * 1e-3,
                      "substrate": circuit.SUB})
        sp = {"name": nm, "components": comps, "substrates": sub,
              "analyses": [{"type": "sparam", "start": f[0] * 1e9, "stop": f[-1] * 1e9, "points": len(f)}]}
        r = engine.simulate(sp, "qucsator", circuit.RUN_ROOT / "mode_diagnostic" / nm, do_record=False,
                            check_host=False)
        if not r.get("ok"):
            raise RuntimeError(f"mode diagnostic {nm}: {r.get('reason')}")
        S[nm] = r["result"]["s"]
    k0 = 2 * np.pi * f * 1e9 / prior.C0
    out = {"f_GHz": list(f), "note": "lossless (tan d = 0); qucsator = MCOUPLED Kirschning-Jansen static and "
                                     "dispersion; prior = Kirschning-Jansen static + Kobayashi dispersion per mode"}
    for i, (nm, (w, g, l)) in enumerate(secs.items()):
        src = {"qucsator": S[nm],
               "prior": circuit.section_s4(f, w, g, l, eps_r, h, 0.0)}
        row = {}
        for who, s4 in src.items():
            for mode, sgn in (("even", 1), ("odd", -1)):
                a, b, c = _abcd(s4[:, :2, :2] + sgn * s4[:, :2, 2:])
                th = np.arccos(a)
                row[f"{who}_Z{mode}_ohm"] = [float(x) for x in np.real(np.sqrt(b / c))]
                row[f"{who}_eeff_{mode}"] = [float(x) for x in (np.real(th) / (k0 * l * 1e-3)) ** 2]
        out[nm] = row
    return out


def pozar_nominal(f_ghz=circuit.F_GHZ):
    """The first (Pozar) nominal of the mesh study in docs/reports/filter-fm.md sec. 1: f0 (-3 dB band centre,
    prior.passband_metrics) of each method, next to the Palace f0 of the three mesh levels quoted from that report."""
    d = json.loads((HERE / "design.json").read_text())
    p = d["nominal"]["p"]
    out = {"p": p, "f0_definition": "-3 dB band centre (prior.passband_metrics), as in the mesh study",
           "palace_f0_GHz_quoted": {"coarse_scale_1.8": 4.909, "medium_scale_1.4": 4.967, "fine_scale_1.0": 4.981,
                                    "source": "docs/reports/filter-fm.md sec. 1 (AdaptiveTol 1e-3, 2-8 GHz)"}}
    out["prior_f0_GHz"] = prior.passband_metrics(f_ghz, prior.s_prior(p, f_ghz)[1])[0]
    for mode in ("physical", "sblock"):
        r = circuit.solve(p, mode, name="pozar_nominal", check_host=False)
        out[f"circuit_{mode}_f0_GHz"] = prior.passband_metrics(r["f"], r["S21"])[0]
    return out


def fig_nominal(f, s_pal, s_meth, fn):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    curves = [("Palace coarse", s_pal)] + list(s_meth.items())
    for lab, s in curves:
        axes[0].plot(f, E.db(s[2] + 1j * s[3]), label=lab)
        axes[1].plot(f, E.db(s[0] + 1j * s[1]), label=lab)
    axes[0].set_ylim(-60, 2); axes[1].set_ylim(-40, 2)
    for ax, t in zip(axes, ("|S21| (dB)", "|S11| (dB)")):
        ax.set_xlabel("f (GHz)"); ax.set_ylabel(t); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("edge_coupled_bpf nominal: Palace, prior and the two circuit models")
    fig.tight_layout(); fig.savefig(fn, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reuse", action="store_true", help="reuse runs/circuit/eval_cache.npz if it matches")
    a = ap.parse_args()
    train = D.load_palace_samples()
    _, _, te = D.split_indices(len(train), seed=0)
    hold, twin = _load_folder("holdout"), _load_folder("holdout_coarse")
    f = train.f.astype(float)
    assert np.allclose(hold.f, f) and np.allclose(twin.f, f) and np.allclose(f, circuit.F_GHZ)
    assert np.allclose(hold.p, twin.p), "holdouts and coarse twins are not the same designs in the same order"
    P = np.vstack([train.p, hold.p]).astype(float)
    names = list(train.ids) + list(hold.ids)
    sims, timing = solve_all(P, names, a.reuse)
    s_prior = _pack(*zip(*[prior.s_prior(p, f) for p in P]))
    s_all = {"prior": s_prior, **sims}
    n_tr = len(train)
    net = np.load(RESULTS / "preds_stageB.npz", allow_pickle=True)
    test_ids = [train.ids[i] for i in te]
    assert list(net["ids"]) == test_ids, "test split differs from results/preds_stageB.npz"
    groups = {"test_26": (np.asarray(te), train.s), "train_all_261": (np.arange(n_tr), train.s),
              "holdout_medium_10": (n_tr + np.arange(len(hold)), hold.s),
              "holdout_coarse_twins_10": (n_tr + np.arange(len(hold)), twin.s)}
    res = {"schema": "rfsurrogates.circuit_tier/1", "family_id": "edge_coupled_bpf",
           "note": "circuit tier vs Palace; metrics as core/eval.py (passband = within 3 dB of the Palace |S21| "
                   "peak; f0 = |S21|^2 centroid over 3-7 GHz; dead filters excluded); simulation only",
           "circuit": {"simulator": "qucsator_rf 1.0.7 via circuit-agent-tools",
                       "physical": "MCOUPLED Kirschning-Jansen (Model and DispModel), MOPEN Kirschning-Jansen-Koster "
                                   "(MSModel Hammerstad, MSDispModel Kirschning), MSTEP wide side first",
                       "sblock": "4-port Touchstone per coupled section from the prior's physics (length l), "
                                 "same MOPEN and MSTEP",
                       "copper": "PEC (sigma 1e30 S/m, t = 0)", "ports": "50 ohm at x = 0 and x = 2 l1 + 2 l2"},
           "groups": {}, "ids": {"test_26": test_ids}}
    for g, (idx, s_true) in groups.items():
        sl = idx                                   # rows of s_all (training records first, then holdouts)
        st = s_true[idx] if g.startswith(("test", "train")) else s_true
        per = {m: E.per_sample_metrics(f, st, s_all[m][sl]) for m in METHODS}
        live = per["prior"]["live"]
        gr = {"n": int(len(idx)), "methods": {m: E.summarize(per[m]) for m in METHODS}, "correlation": {}}
        for m in METHODS:
            gr["methods"][m]["f0_err_MHz_mean_signed"] = float(np.mean(per[m]["f0_err_mhz"][live]))
        for m in METHODS[1:]:
            gr["correlation"][m] = correlations(f, st, s_all["prior"][sl], s_all[m][sl], per["prior"], per[m], live,
                                                net["s_pred"] if g == "test_26" else None)
        if g == "test_26":
            pn = E.per_sample_metrics(f, st, net["s_pred"])
            gr["methods"]["network_stageB"] = E.summarize(pn)
            gr["methods"]["network_stageB"]["f0_err_MHz_mean_signed"] = float(np.mean(pn["f0_err_mhz"][pn["live"]]))
        res["groups"][g] = gr
        print(g, {m: (round(v["passband_S21_dB_mean"], 3), round(v["f0_err_MHz_mean_abs"], 1))
                  for m, v in gr["methods"].items()})
    # circuit vs prior directly (how far the circuit moves the response), on all 261
    per = E.per_sample_metrics(f, s_all["circuit_physical"][:n_tr], s_all["prior"][:n_tr])
    res["prior_vs_circuit_physical_all_261"] = E.summarize(per)
    res["timing"] = {**timing, "single_solve": single_solve_timing(P),
                     "palace_coarse_wall_s_mean": float(np.nanmean(train.wall_s)) if train.wall_s is not None else None}
    xc = crosscheck(P[[names.index("nominal")] + list(te[:3])], ["nominal"] + test_ids[:3])
    res["crosscheck_sblock_path"] = xc
    from circuit_tools.microstrip import line_props
    w50 = circuit.layout(P[0])[1]["w50"]
    lp = line_props(f * 1e9, w50 * 1e-3, {"er": 3.66, "h": 0.508e-3, "t": 0.0, "tand": 0.0037, "sigma": 1e30})
    res["w50_feed"] = {"w_mm": w50, "z0_ohm_3GHz": float(np.real(lp["z0"][0])),
                       "z0_ohm_7GHz": float(np.real(lp["z0"][-1])),
                       "source": "scikit-rf MLine (Hammerstad-Jensen + Kirschning-Jansen dispersion), as qucsator MLIN"}
    from circuit_tools.lint import lint
    lr = lint(circuit.spec(P[names.index("nominal")], "physical", name="lint_nominal"), {"skip_reference": True})
    res["lint_nominal_physical"] = {"counts": lr["counts"], "findings": [f"{x['severity']} {x['id']}: {x['message']}"
                                                                         for x in lr["findings"]]}
    ratios = np.concatenate([np.maximum(w50 / P[:n_tr, 0], P[:n_tr, 0] / w50),
                             np.maximum(P[:n_tr, 1] / P[:n_tr, 0], P[:n_tr, 0] / P[:n_tr, 1])])
    r0 = (12.6 * np.log10(3.66) + 3.17) / (10.1 * np.log10(3.66) + 2.33)
    res["step_width_ratios_all_261"] = {"n_steps": int(ratios.size), "min": float(ratios.min()),
                                        "max": float(ratios.max()), "frac_below_1.5": float(np.mean(ratios < 1.5)),
                                        "frac_with_negative_fitted_C": float(np.mean(ratios < r0)),
                                        "ratio_where_fitted_C_is_zero": float(r0)}
    i_nom = names.index("nominal")
    res["mode_diagnostic_nominal"] = mode_diagnostic(P[i_nom])
    res["pozar_nominal_mesh_study"] = pozar_nominal()
    fig_nominal(f, train.s[i_nom], {"prior": s_all["prior"][i_nom], "circuit (qucsator, physical)":
                                    s_all["circuit_physical"][i_nom], "circuit (prior S-blocks)":
                                    s_all["circuit_sblock"][i_nom]}, HERE / "figs" / "circuit_tier_nominal.png")
    (RESULTS / "circuit_tier.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("wrote", RESULTS / "circuit_tier.json")


if __name__ == "__main__":
    main()
