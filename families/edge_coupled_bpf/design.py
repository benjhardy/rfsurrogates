"""Nominal design (Pozar coupled-line synthesis), sampling box and Latin-hypercube sampler.

    python design.py [--centre w1 w2 g1 g2 l1 l2]   -> design.json, samples.json (stage B, 600 samples by default)
    python design.py --stage-a                      -> data/prior_stageA.npz (20,000 prior samples, 3 substrates)

Synthesis (Pozar ch. 8.7, Chebyshev N = 3, 0.5 dB ripple, f0 = 5 GHz, fractional bandwidth 10 %):
  prototype g from the Matthaei-Young-Jones recurrence (prior.chebyshev_g), inverters and Z0e/Z0o from Pozar 8.121
  (prior.coupled_line_impedances). Each section has its own strip width, so (w1, g1) solve the outer sections'
  Z0e/Z0o exactly and (w2, g2) the inner sections' (Kirschning-Jansen, 2 unknowns, 2 equations each).
  l1, l2 start at a quarter wave at f0 with the mean of the even/odd effective permittivities of their section minus
  the open-end extension, then both are scaled by one common factor so the prior's centre frequency is exactly f0.
Lengths are the coupled (overlap) length of each section: resonators 1 and 3 are l1 + l2 long (w1 part + w2 part,
centred on one axis), resonator 2 is 2 l2 long; the stagger between neighbouring strips is the section length.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.stats import qmc

import prior
from prior import PARAM_NAMES

HERE = Path(__file__).resolve().parent
F0, FBW, RIPPLE_DB, N_ORDER, Z0 = 5.0, 0.10, 0.5, 3, 50.0
F_GRID = np.round(np.arange(2.0, 8.0 + 1e-9, 0.02), 4)          # GHz, 301 points
# stage B box: +-30 % on widths and gaps; +-15 % on lengths (agreed with Ben); substrate fixed
BOX_FRAC = {"w1": 0.30, "w2": 0.30, "g1": 0.30, "g2": 0.30, "l1": 0.15, "l2": 0.15}
# stage A (prior only): wider box, three substrates
BOX_FRAC_A = {"w1": 0.50, "w2": 0.50, "g1": 0.50, "g2": 0.50, "l1": 0.25, "l2": 0.25}


def solve_section(z0e, z0o, h, eps_r):
    """(w, s) in mm whose Kirschning-Jansen even/odd impedances equal (z0e, z0o)."""
    def res(x):
        w, s = np.exp(x)
        a, b, _, _ = prior.coupled_microstrip(w, s, h, eps_r)
        return [a - z0e, b - z0o]
    x, info, ok, msg = fsolve(res, np.log([h * (1.0 if eps_r < 5 else 0.6), 0.3 * h]), xtol=1e-12, full_output=True)
    if ok != 1:
        raise RuntimeError(f"section synthesis did not converge: {msg}")
    return float(np.exp(x[0])), float(np.exp(x[1]))


def synthesize(eps_r, h, f0=F0, fbw=FBW, z0=Z0, n=N_ORDER, ripple_db=RIPPLE_DB):
    """Return the nominal p (mm) for a substrate, plus the synthesis record."""
    zz, g, jz = prior.coupled_line_impedances(n, ripple_db, fbw, z0)
    (ze1, zo1), (ze2, zo2) = zz[0], zz[1]
    w1, g1 = solve_section(ze1, zo1, h, eps_r)
    w2, g2 = solve_section(ze2, zo2, h, eps_r)
    lens = []
    for w, gg in ((w1, g1), (w2, g2)):
        _, _, ee, eo = prior.coupled_microstrip(w, gg, h, eps_r)
        v = 0.5 * (np.sqrt(ee) + np.sqrt(eo))                 # mean phase-velocity factor
        lens.append(prior.C0 / (4 * f0 * 1e9 * v) * 1e3 - prior.open_end_extension(w, h, eps_r))
    l1, l2 = lens

    def f0_err(k):
        s11, s21 = prior.s_prior((w1, w2, g1, g2, k * l1, k * l2, eps_r, h), F_GRID)
        return prior.passband_metrics(F_GRID, s21)[0] - f0

    k = brentq(f0_err, 0.85, 1.15, xtol=1e-7)
    p_syn = [float(v) for v in (w1, w2, g1, g2, k * l1, k * l2, eps_r, h)]
    p, tune_info = tune_prior(p_syn, f0, fbw, ripple_db, n)
    f_fine = np.round(np.arange(2.0, 8.0001, 0.005), 4)
    s11, s21 = prior.s_prior(p, f_fine)
    feat = prior.passband_features(f_fine, s11, s21)
    s11l, s21l = prior.s_prior(p, f_fine, tand=0.0)
    feat_lossless = prior.passband_features(f_fine, s11l, s21l)
    rec = {"p": p, "params": dict(zip(PARAM_NAMES, p)), "p_synthesis": p_syn, "prior_tune": tune_info,
           "prototype_g": g, "JZ0": jz,
           "Z0e_Z0o_target": [[float(a), float(b)] for a, b in zz],
           "Z0e_Z0o_microstrip": [list(prior.coupled_microstrip(w, gg, h, eps_r)[:2]) for w, gg in ((w1, g1), (w2, g2), (w2, g2), (w1, g1))],
           "eps_e_eps_o": [list(prior.coupled_microstrip(w, gg, h, eps_r)[2:]) for w, gg in ((w1, g1), (w2, g2))],
           "open_end_extension_mm": [prior.open_end_extension(w1, h, eps_r), prior.open_end_extension(w2, h, eps_r)],
           "length_scale_applied": float(k), "prior_features": feat, "prior_features_lossless": feat_lossless,
           "prior_f0_GHz": feat["f0_GHz"], "prior_BW_GHz": feat["BW_GHz"], "prior_IL_dB": feat["IL_dB"],
           "W50_mm": float(brentq(lambda ww: prior.microstrip(ww, h, eps_r)[0] - z0, 0.05 * h, 20 * h))}
    return p, rec


def chebyshev_bandpass_target(f_ghz, f0=F0, fbw=FBW, ripple_db=RIPPLE_DB, n=N_ORDER):
    """|S21|, |S11| of the ideal lossless Chebyshev band-pass (prototype + standard band-pass transformation
    omega' = (f/f0 - f0/f) / Delta): the equiripple target the prior is tuned to."""
    f = np.asarray(f_ghz, float)
    eps2 = 10 ** (ripple_db / 10) - 1
    x = (f / f0 - f0 / f) / fbw
    t = np.where(np.abs(x) <= 1, np.cos(n * np.arccos(np.clip(x, -1, 1))),
                 np.sign(x) ** n * np.cosh(n * np.arccosh(np.maximum(np.abs(x), 1))))
    s21 = 1 / np.sqrt(1 + eps2 * t ** 2)
    return s21, np.sqrt(1 - s21 ** 2)


def tune_prior(p0, f0=F0, fbw=FBW, ripple_db=RIPPLE_DB, n=N_ORDER, span=0.35):
    """Least-squares fit of the six geometry parameters so the LOSSLESS prior reproduces the ideal Chebyshev
    response over f0 +- 1.5 BW (|S21| in dB, clipped at -30 dB, and |S11| linear). The Pozar synthesis only
    approximates the inverters (quarter-wave sections, equal even/odd velocities); this step removes that error
    inside the prior so its own response is equiripple. Returns (p_tuned, info)."""
    from scipy.optimize import least_squares
    p0 = list(p0)
    eps_r, h = p0[6], p0[7]
    f = np.round(np.arange(f0 * (1 - 1.5 * fbw), f0 * (1 + 1.5 * fbw) + 1e-9, 0.004), 4)
    t21, t11 = chebyshev_bandpass_target(f, f0, fbw, ripple_db, n)
    t21_db = np.maximum(20 * np.log10(t21), -30)

    def resid(x):     # x = log(g1, g2, l1, l2); the strip widths stay at their Pozar values (well-posed 4-parameter fit)
        s11, s21 = prior.s_prior(p0[:2] + list(np.exp(x)) + [eps_r, h], f, tand=0.0)
        r21 = np.maximum(20 * np.log10(np.abs(s21) + 1e-12), -30) - t21_db
        r11 = 25 * (np.abs(s11) - t11)
        return np.concatenate([r21, r11])

    x0 = np.log(p0[2:6])
    res = least_squares(resid, x0, bounds=(x0 + np.log(1 - span), x0 + np.log(1 + span)), xtol=1e-10, ftol=1e-10)
    p = p0[:2] + [float(v) for v in np.exp(res.x)] + [eps_r, h]
    info = {"cost": float(res.cost), "nfev": int(res.nfev), "tuned": ["g1", "g2", "l1", "l2"],
            "shift_pct": [float(100 * (b / a - 1)) for a, b in zip(p0[:6], p[:6])],
            "fit_band_GHz": [float(f[0]), float(f[-1])]}
    return p, info


def make_box(p, frac):
    box = {}
    for name, v in zip(PARAM_NAMES, p):
        box[name] = [v * (1 - frac[name]), v * (1 + frac[name])] if name in frac else [v, v]
    return box


def sample_box(box, n, seed, edge_frac=0.10):
    """Latin hypercube over the free parameters; `edge_frac` of the samples are snapped to box corners/edges.

    Returns an (n, 8) array ordered so that a prefix is still useful: edge samples first, then the LHS points in a
    space-filling order (greedy farthest-point ordering of the normalised coordinates).
    """
    free = [k for k in PARAM_NAMES if box[k][0] != box[k][1]]
    lo = np.array([box[k][0] for k in free]); hi = np.array([box[k][1] for k in free])
    rng = np.random.default_rng(seed)
    n_edge = int(round(edge_frac * n))
    n_lhs = n - n_edge
    u = qmc.LatinHypercube(d=len(free), seed=seed).random(n_lhs)
    order = [int(np.argmin(np.sum((u - 0.5) ** 2, axis=1)))]
    dmin = np.sum((u - u[order[0]]) ** 2, axis=1)
    for _ in range(n_lhs - 1):
        j = int(np.argmax(dmin)); order.append(j)
        dmin = np.minimum(dmin, np.sum((u - u[j]) ** 2, axis=1))
    u = u[order]
    d = len(free)
    corners = np.array([[(i >> k) & 1 for k in range(d)] for i in range(2 ** d)], float)
    rng.shuffle(corners)
    e = corners[:n_edge]
    if n_edge > len(corners):
        faces = qmc.LatinHypercube(d=d, seed=seed + 1).random(n_edge - len(corners))
        axis = rng.integers(0, d, size=len(faces))
        faces[np.arange(len(faces)), axis] = rng.integers(0, 2, size=len(faces))
        e = np.vstack([e, faces])
    ue = np.vstack([e, u]) if n_edge else u
    x = lo + ue * (hi - lo)
    out = np.tile(np.array([box[k][0] for k in PARAM_NAMES]), (n, 1))
    for j, k in enumerate(free):
        out[:, PARAM_NAMES.index(k)] = x[:, j]
    kinds = ["edge"] * n_edge + ["lhs"] * n_lhs
    return out, kinds


def write_stage_b(n, seed, centre=None, n_holdout=10, out=HERE / "samples.json"):
    """samples.json around `centre` (the Pozar nominal unless a tuned centre is given) + medium-mesh holdout."""
    sub = prior.SUBSTRATES["RO4350B"]
    p_syn, rec = synthesize(sub["eps_r"], sub["h"])
    p = list(centre) + [sub["eps_r"], sub["h"]] if centre is not None else p_syn
    box = make_box(p, BOX_FRAC)
    P, kinds = sample_box(box, n, seed)
    samples = [{"id": "nominal", "p": p, "kind": "nominal"}]
    samples += [{"id": f"s{i:04d}", "p": [float(v) for v in row], "kind": kd} for i, (row, kd) in enumerate(zip(P, kinds))]
    H, _ = sample_box(box, n_holdout, seed + 777, edge_frac=0.0)
    samples += [{"id": f"h{i:04d}", "p": [float(v) for v in row], "kind": "fidelity-holdout"} for i, row in enumerate(H)]
    design = {"family": "edge-coupled microstrip band-pass, N=3 Chebyshev 0.5 dB, f0 5 GHz, 10 % BW",
              "substrate": {"name": "RO4350B", **sub, "copper": "PEC sheet (zero thickness)"},
              "param_names": list(PARAM_NAMES), "units": {"lengths": "mm", "f": "GHz"},
              "nominal": rec, "centre": p, "centre_is_tuned": centre is not None,
              "box": box, "box_frac": BOX_FRAC, "f_grid_GHz": [2.0, 8.0, 0.02]}
    (HERE / "design.json").write_text(json.dumps(design, indent=1))
    Path(out).write_text(json.dumps({"schema": "rfsurrogates.samples/2", "family_id": "edge_coupled_bpf", "seed": seed,
                                     "param_names": list(PARAM_NAMES), "box": box, "centre": p,
                                     "centre_is_tuned": centre is not None, "edge_fraction": 0.10,
                                     "holdout": {"ids": f"h0000..h{n_holdout - 1:04d}", "mesh_scale": 1.4,
                                                 "tags": ["fidelity-holdout", "medium"], "note": "not for training"},
                                     "f_grid_GHz": [2.0, 8.0, 0.02], "samples": samples}, indent=1))
    return design


# ----------------------------------------------------------------------------- geometry validity (added 2026-10-25)
C_MIN = 0.05       # mm: minimum clearance between conductors that are not meant to touch (see clearance())


def w50_for(eps_r, h, z0=Z0):
    return float(brentq(lambda ww: prior.microstrip(ww, h, eps_r)[0] - z0, 0.05 * h, 20 * h))


def clearance(p, w50=None):
    """Smallest clearance (mm) between conductors that must not touch in the centred-step layout of build_case.py:
      (i)  the w2 part of resonators 1/3 starts at x = l1 where the input/output strip ends: g1 - max(0, (w2 - w1)/2)
      (ii) the w1 part of resonators 1/3 ends at x = l1 where resonator 2 starts:          g2 - max(0, (w1 - w2)/2)
      (iii) the 50 ohm feed (width W50) ends at x = 0 where resonator 1 starts:            g1 - max(0, (W50 - w1)/2)
    A clearance <= 0 means the rectangles share an edge segment and OCC fuses them: the device is shorted and is not
    a coupled-line filter (verified on 157 built meshes by counting connected copper components)."""
    w1, w2, g1, g2, l1, l2, eps_r, h = prior.unpack(p)
    w50 = w50_for(eps_r, h) if w50 is None else w50
    return float(min(g1 - max(0.0, (w2 - w1) / 2), g2 - max(0.0, (w1 - w2) / 2), g1 - max(0.0, (w50 - w1) / 2)))


def is_valid(p, c_min=C_MIN, w50=None):
    return clearance(p, w50) > c_min


VALIDITY_NOTE = {
    "rule": f"clearance(p) > {C_MIN} mm",
    "clearance": "min(g1 - max(0, (w2 - w1)/2), g2 - max(0, (w1 - w2)/2), g1 - max(0, (W50 - w1)/2)) [mm]",
    "why": "In the centred-step layout the w2 part of resonators 1/3 begins at x = l1 where the input/output strip ends, "
           "and the fixed-width 50 ohm feed ends at x = 0 where resonator 1 begins; when the wider element reaches past "
           "the gap the rectangles share an edge and are fused into one conductor (3 instead of 5 copper components): "
           "a shorted device, not a coupled-line filter. 0 < clearance < 0.05 mm is a corner-to-corner near-touch that "
           "the production mesh (0.144 mm edge size, ~1 element across g1) cannot resolve, so those are excluded too.",
    "nominal": "the tuned nominal has clearance 0.061 mm (feed corner vs resonator 1): valid, in the marginal band; "
               "its coarse-vs-fine mesh check (f0 -1.44 %) is what covers that band",
    "invalid_samples": "kept on disk under data/invalid (npz) and tagged invalid-geometry in the run DB; never used for training",
}


def extend_samples(n_valid, n_hold_valid, seed, c_min=C_MIN, path=HERE / "samples.json"):
    """Annotate every sample with clearance/valid, keep ids, add new valid samples (sNNNN continuing after the last id,
    hNNNN likewise) by rejection sampling in the same box until n_valid valid production ids and n_hold_valid valid
    holdout ids exist. New production ids are ordered farthest-point from the existing valid set."""
    S = json.loads(Path(path).read_text())
    box = S["box"]; eps_r, h = S["centre"][6], S["centre"][7]
    w50 = w50_for(eps_r, h)
    for s in S["samples"]:
        s["clearance_mm"] = round(clearance(s["p"], w50), 5)
        s["valid"] = bool(is_valid(s["p"], c_min, w50))
    prod = [s for s in S["samples"] if s["id"].startswith("s")]
    hold = [s for s in S["samples"] if s["id"].startswith("h")]
    n_prod_valid = sum(s["valid"] for s in prod); n_hold_ok = sum(s["valid"] for s in hold)
    rng = np.random.default_rng(seed)
    free = [k for k in PARAM_NAMES if box[k][0] != box[k][1]]
    lo = np.array([box[k][0] for k in free]); hi = np.array([box[k][1] for k in free])

    def draw(n_needed, seed_k):
        got = []
        k = 0
        while len(got) < n_needed:
            u = qmc.LatinHypercube(d=len(free), seed=seed_k + k).random(max(4 * n_needed, 64))
            x = lo + u * (hi - lo)
            for row in x:
                p = list(S["centre"]);
                for j, name in enumerate(free):
                    p[PARAM_NAMES.index(name)] = float(row[j])
                if is_valid(p, c_min, w50):
                    got.append(p)
                if len(got) >= n_needed:
                    break
            k += 1
        return got

    new_prod = draw(max(0, n_valid - n_prod_valid), seed)
    if new_prod:   # farthest-point order relative to the existing valid production points (normalised coordinates)
        def norm(p): return np.array([(p[PARAM_NAMES.index(k)] - box[k][0]) / (box[k][1] - box[k][0]) for k in free])
        base = [norm(s["p"]) for s in prod if s["valid"]]
        cand = [norm(p) for p in new_prod]
        order = []
        dmin = np.array([min(np.sum((c - b) ** 2) for b in base) if base else 1.0 for c in cand])
        for _ in range(len(cand)):
            j = int(np.argmax(dmin)); order.append(j); dmin[j] = -1
            dmin = np.where(dmin >= 0, np.minimum(dmin, [np.sum((c - cand[j]) ** 2) for c in cand]), dmin)
        new_prod = [new_prod[j] for j in order]
    next_s = max(int(s["id"][1:]) for s in prod) + 1
    for i, p in enumerate(new_prod):
        S["samples"].append({"id": f"s{next_s + i:04d}", "p": p, "kind": "lhs-valid", "clearance_mm": round(clearance(p, w50), 5), "valid": True})
    new_hold = draw(max(0, n_hold_valid - n_hold_ok), seed + 999)
    next_h = max(int(s["id"][1:]) for s in hold) + 1
    for i, p in enumerate(new_hold):
        S["samples"].append({"id": f"h{next_h + i:04d}", "p": p, "kind": "fidelity-holdout", "clearance_mm": round(clearance(p, w50), 5), "valid": True})
    S["validity"] = {**VALIDITY_NOTE, "c_min_mm": c_min, "W50_mm": round(w50, 4), "n_valid_production": n_valid,
                     "n_valid_holdout": n_hold_valid, "extended_on": "2026-10-25", "extend_seed": seed}
    S["holdout"]["ids"] = [s["id"] for s in S["samples"] if s["id"].startswith("h") and s["valid"]]
    Path(path).write_text(json.dumps(S, indent=1))
    return S, len(new_prod), len(new_hold)


def write_stage_a(n_total, seed, out=HERE / "data" / "prior_stageA.npz"):
    names = list(prior.SUBSTRATES)
    per = [n_total // len(names)] * len(names)
    per[-1] += n_total - sum(per)
    P, S11, S21, subs, boxes = [], [], [], [], {}
    for k, (name, n) in enumerate(zip(names, per)):
        s = prior.SUBSTRATES[name]
        p, _ = synthesize(s["eps_r"], s["h"])
        box = make_box(p, BOX_FRAC_A)
        boxes[name] = {"nominal": p, "box": box}
        X, _ = sample_box(box, n, seed + 10 * k, edge_frac=0.0)
        for row in X:
            a, b = prior.s_prior(row, F_GRID, tand=s["tand"])
            P.append(row); S11.append(a); S21.append(b); subs.append(name)
    P = np.asarray(P, np.float64)
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, p=P, f=F_GRID, S11=np.asarray(S11, np.complex64), S21=np.asarray(S21, np.complex64),
                        substrate=np.asarray(subs), param_names=np.asarray(PARAM_NAMES),
                        boxes=json.dumps(boxes), note="prior.s_prior (Kirschning-Jansen + exact coupled-line sections); "
                        "p = (w1, w2, g1, g2, l1, l2, eps_r, h); box +-50 % on widths/gaps and +-25 % on lengths "
                        "around each substrate's Pozar nominal")
    return out, boxes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="stage B sample count written to samples.json")
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--centre", nargs=6, type=float, help="tuned box centre w1 w2 g1 g2 l1 l2 (mm)")
    ap.add_argument("--stage-a", action="store_true")
    ap.add_argument("--n-a", type=int, default=20000)
    ap.add_argument("--extend", nargs=2, type=int, metavar=("N_VALID", "N_HOLD_VALID"),
                    help="annotate samples.json with the geometry rule and add valid samples up to these counts")
    a = ap.parse_args()
    if a.extend:
        S, np_, nh = extend_samples(a.extend[0], a.extend[1], a.seed + 1)
        prod = [s for s in S["samples"] if s["id"].startswith("s")]
        print(f"samples.json: {len(prod)} production ids, {sum(s['valid'] for s in prod)} valid (+{np_} new); "
              f"holdouts valid {len(S['holdout']['ids'])} (+{nh} new); rule {S['validity']['rule']}")
    elif a.stage_a:
        out, boxes = write_stage_a(a.n_a, a.seed)
        print("wrote", out)
        for k, v in boxes.items():
            print(k, "nominal", [round(x, 4) for x in v["nominal"]])
    else:
        d = write_stage_b(a.n, a.seed, a.centre)
        print(json.dumps({k: v for k, v in d["nominal"].items() if k != "prior_features_lossless"}, indent=1))
        print("centre", d["centre"], "tuned" if d["centre_is_tuned"] else "(Pozar nominal)")
        print("box", json.dumps(d["box"]))
