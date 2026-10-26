"""Inverse design on the stage-B ensemble: find the geometry p that meets a band-pass spec.

    python core/inverse.py --f0 5.2 --bw 0.5 [--rl 12] [--ripple 0.5] [--reject 6.5:30] [--n-starts 64] [--out runs_inverse/<name>]
    python core/inverse.py --list-box

Spec
  --f0      centre frequency (GHz).  By default the centre of the -3 dB band (eval.band_edges); --f0-def centroid
            uses the |S21|^2-weighted centroid (eval.centroid_freq).  Both are reported.
  --bw      -3 dB bandwidth (GHz), eval.band_edges on the ensemble-mean |S21|.
  --rl      minimum return loss (dB, default 10) inside the spec band f0 +- (bw/2) * inband_frac.
  --ripple  optional maximum passband ripple (dB) = max - min of |S21| dB inside the spec band.
  --reject  optional "f_GHz:dB" (repeatable): |S21| <= -dB at f_GHz.
  The spec band for RL / ripple is f0 +- inband_frac * bw / 2 (default inband_frac 0.8): at the -3 dB edges the
  ripple would be 3 dB by definition, so RL and ripple are judged on the inner 80 % of the band.

Objective (evaluated on the ENSEMBLE MEAN of the 5 stage-B seeds, S = S_prior + DeepONet correction)
  J = ((f0_pred - f0) / 50 MHz)^2 + ((bw_pred - bw) / 50 MHz)^2
      + w_rl * relu(rl - RL_pred)^2 + w_ripple * relu(ripple_pred - ripple)^2
      + sum_i w_reject * relu(|S21|dB(f_i) + dB_i)^2
      + 1e5 * relu(C_MIN + 0.005 - clearance(p))^2           (soft part of the hard geometry constraint)
  Hard constraints: the six geometry parameters inside the samples.json box (L-BFGS-B bounds), substrate fixed at
  RO4350B (eps_r 3.66, h 0.508: stage B has only that substrate), clearance(p) > C_MIN = 0.05 mm (design.py rule;
  starts that violate it are rejected, refined candidates that violate it are discarded).

Search
  1. Latin hypercube (or Sobol) over the box, rejection-sampled to valid clearance, --n-starts points.
  2. One batched forward pass of prior + 5 seeds over all starts; rank by J.
  3. Top k (--k, default 8) refined with scipy L-BFGS-B in normalised box coordinates.  The prior is numpy, so the
     gradient is a batched central finite difference (13 designs in one forward pass per gradient); the DeepONet part
     is differentiable but the finite difference covers it at no extra cost.  A short bounded Nelder-Mead polish
     follows (the -3 dB edges and the hinges make J piecewise smooth).
  4. Candidates are ranked by J; ties (within 1e-3, common because 6 parameters over-determine 2 targets) go to the
     candidate with the smallest ensemble spread, then the lowest insertion loss.
     The best valid candidate is reported together with the top-5 table, the prior-only prediction for the same p
     (what the learned correction changed) and the ensemble spread (std over seeds of passband |S21| dB, the trust
     indicator; the held-out calibration is in families/<id>/results/metrics.json).

Outputs in --out (default runs_inverse/f<f0>_bw<bw>)
  result.json, sparams_pred.npz (f, S11/S21 of the ensemble mean, each seed and the prior), inverse_<name>.png,
  verify_<name>.json (a samples.json-style entry + the exact build_case.py command for a Palace verification run;
  this script never runs Palace).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _p in (str(HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from paths import FAMILY_DIR, RESULTS, RUNS_INVERSE  # noqa: E402
import data as D                                        # noqa: E402
import prior_wrap as PW                                 # noqa: E402
from eval import band_edges, centroid_freq, db, passband_is_split   # noqa: E402
from train import DEFAULTS, CKPT, load_ckpt             # noqa: E402
import design                                           # noqa: E402  (clearance rule; from the family folder)

SUBSTRATE = {"name": "RO4350B", "eps_r": 3.66, "h": 0.508}
GEOM = ("w1", "w2", "g1", "g2", "l1", "l2")
C_MIN = design.C_MIN
CLEAR_MARGIN = 0.005          # mm of slack above C_MIN asked of the optimiser (returned points are checked against C_MIN)
W_CLEAR = 1e5
F0_TOL = 0.05                 # GHz: f0 error that costs 1 in J
BW_TOL = 0.05                 # GHz: BW error that costs 1 in J
# achievable range seen in the data (fallback for --list-box when data/palace is not on disk)
DATA_RANGE_NOTE = "f0 4.28-6.00 GHz, BW 0.05-0.90 GHz over the 259 valid production samples (eval.py definitions)"


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------
@dataclass
class Spec:
    f0: float
    bw: float
    rl: float = 10.0
    ripple: Optional[float] = None
    reject: List[Tuple[float, float]] = field(default_factory=list)    # (f_GHz, dB)
    f0_def: str = "band"                                                # "band" | "centroid"
    inband_frac: float = 0.8
    w_rl: float = 1.0
    w_ripple: float = 1.0
    w_reject: float = 1.0

    def band_mask(self, f: np.ndarray) -> np.ndarray:
        half = 0.5 * self.bw * self.inband_frac
        m = (f >= self.f0 - half) & (f <= self.f0 + half)
        if m.sum() < 2:                      # very narrow spec band vs the 20 MHz grid: take the two nearest points
            idx = np.argsort(np.abs(f - self.f0))[:2]
            m = np.zeros_like(m); m[idx] = True
        return m

    def to_dict(self):
        d = {"f0_GHz": self.f0, "bw_GHz": self.bw, "rl_min_dB": self.rl, "ripple_max_dB": self.ripple,
             "reject": [{"f_GHz": a, "dB": b} for a, b in self.reject], "f0_def": self.f0_def,
             "inband_frac": self.inband_frac, "weights": {"rl": self.w_rl, "ripple": self.w_ripple, "reject": self.w_reject},
             "f0_tol_GHz": F0_TOL, "bw_tol_GHz": BW_TOL, "substrate": SUBSTRATE, "clearance_min_mm": C_MIN}
        return d


def parse_reject(items: Optional[Sequence[str]]) -> List[Tuple[float, float]]:
    out = []
    for it in items or ():
        try:
            a, b = it.split(":")
            out.append((float(a), float(b)))
        except ValueError:
            raise SystemExit(f"--reject expects f_GHz:dB, got {it!r}")
    return out


# ---------------------------------------------------------------------------
# surrogate: stage-B ensemble + prior on the Palace frequency grid
# ---------------------------------------------------------------------------
def palace_grid() -> np.ndarray:
    """The frequency grid the stage-B models were trained on (3-7 GHz, 201 points)."""
    z = RESULTS / "preds_stageB.npz"
    if z.exists():
        return np.asarray(np.load(z, allow_pickle=True)["f"], dtype=np.float32)
    files = sorted(D.PALACE_DIR.glob("*.npz")) if D.PALACE_DIR.exists() else []
    if files:
        return np.asarray(D.to_ghz(D._get(np.load(files[0], allow_pickle=True), "f")), dtype=np.float32)
    js = D._read_samples_json() or {}
    lo, hi, step = js.get("f_grid_GHz", [2.0, 8.0, 0.02])
    return np.round(np.arange(lo, hi + 1e-9, step), 4).astype(np.float32)


class Surrogate:
    """S(p) = S_prior(p) + correction_k(p), k over the stage-B seeds; all evaluated in one batch."""

    def __init__(self, ckpt_dir: Path = CKPT, f: Optional[np.ndarray] = None, seeds: Optional[Sequence[int]] = None):
        args = argparse.Namespace(**DEFAULTS)
        paths = sorted(ckpt_dir.glob("stageB_seed*.pt"))
        if seeds is not None:
            paths = [ckpt_dir / f"stageB_seed{k}.pt" for k in seeds]
        if not paths:
            raise FileNotFoundError(f"no stageB_seed*.pt under {ckpt_dir}: run core/train.py B first")
        self.models = []
        for pth in paths:
            m = load_ckpt(pth, args)
            if not m.use_prior:
                raise ValueError(f"{pth.name} has use_prior=False; not a stage-B checkpoint")
            m.eval()
            self.models.append(m)
        self.param_names = tuple(torch.load(paths[0], map_location="cpu").get("param_names", D.FALLBACK_PARAM_NAMES))
        self.f = palace_grid() if f is None else np.asarray(f, dtype=np.float32)
        self.f_t = torch.as_tensor(self.f)
        self.n_eval = 0

    @property
    def n_seeds(self):
        return len(self.models)

    @torch.no_grad()
    def predict(self, P: np.ndarray):
        """P (N,8) -> (S_mean (N,4,M), S_seeds (S,N,4,M), S_prior (N,4,M)) as float32 numpy."""
        P = np.atleast_2d(np.asarray(P, dtype=np.float32))
        s_prior = PW.s_prior_batch(P, self.f)
        p_t = torch.as_tensor(P); sp_t = torch.as_tensor(s_prior)
        seeds = torch.stack([m(p_t, self.f_t, sp_t) for m in self.models])
        self.n_eval += P.shape[0]
        return seeds.mean(0).numpy(), seeds.numpy(), s_prior


# ---------------------------------------------------------------------------
# response metrics and objective
# ---------------------------------------------------------------------------
def unpack_np(s):
    return s[..., 0, :] + 1j * s[..., 1, :], s[..., 2, :] + 1j * s[..., 3, :]


def response_metrics(f: np.ndarray, s: np.ndarray, spec: Spec) -> dict:
    """Metrics of one response s (4,M) against the spec (eval.py definitions for f0 / BW)."""
    s11, s21 = unpack_np(s)
    s21_db, s11_db = db(s21), db(s11)
    f_lo, f_hi, f_pk = band_edges(f, s21_db)
    m = spec.band_mask(f)
    out = {
        "f0_band_GHz": float(0.5 * (f_lo + f_hi)), "f0_centroid_GHz": float(centroid_freq(f, s21)),
        "bw_GHz": float(f_hi - f_lo), "f_lo_GHz": float(f_lo), "f_hi_GHz": float(f_hi), "f_peak_GHz": float(f_pk),
        "il_dB": float(-s21_db.max()),                              # insertion loss at the |S21| peak
        "il_specband_max_dB": float(-s21_db[m].min()),              # worst insertion loss inside the spec band
        "rl_min_dB": float(-s11_db[m].max()),                       # worst return loss inside the spec band
        "ripple_dB": float(s21_db[m].max() - s21_db[m].min()),
        "reject_dB": [float(np.interp(fr, f, s21_db)) for fr, _ in spec.reject],
        "split_passband": bool(passband_is_split(s21_db)),
    }
    out["f0_GHz"] = out["f0_band_GHz"] if spec.f0_def == "band" else out["f0_centroid_GHz"]
    return out


def objective_terms(met: dict, clearance: float, spec: Spec) -> dict:
    t = {"f0": ((met["f0_GHz"] - spec.f0) / F0_TOL) ** 2,
         "bw": ((met["bw_GHz"] - spec.bw) / BW_TOL) ** 2,
         "rl": spec.w_rl * max(0.0, spec.rl - met["rl_min_dB"]) ** 2,
         "ripple": 0.0 if spec.ripple is None else spec.w_ripple * max(0.0, met["ripple_dB"] - spec.ripple) ** 2,
         "reject": float(sum(spec.w_reject * max(0.0, v + lvl) ** 2 for v, (_, lvl) in zip(met["reject_dB"], spec.reject))),
         "clearance": W_CLEAR * max(0.0, C_MIN + CLEAR_MARGIN - clearance) ** 2}
    t["total"] = float(sum(t.values()))
    return t


class Problem:
    """Box, clearance rule and the batched objective J(p) on the ensemble mean."""

    def __init__(self, spec: Spec, sur: Surrogate, box: Optional[dict] = None):
        self.spec, self.sur = spec, sur
        box = box or D.load_box()
        if box is None:
            raise FileNotFoundError("samples.json box not found")
        self.names = tuple(sur.param_names)
        self.box = box
        self.lo = np.array([box[k][0] for k in GEOM]); self.hi = np.array([box[k][1] for k in GEOM])
        self.eps_r = float(box.get("eps_r", [SUBSTRATE["eps_r"]])[0]); self.h = float(box.get("h", [SUBSTRATE["h"]])[0])
        self.w50 = design.w50_for(self.eps_r, self.h)

    # --- coordinates ---------------------------------------------------
    def to_p(self, U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(U)
        X = self.lo + U * (self.hi - self.lo)
        return np.concatenate([X, np.tile([self.eps_r, self.h], (U.shape[0], 1))], axis=1)

    def to_u(self, p: np.ndarray) -> np.ndarray:
        return (np.asarray(p, float)[:6] - self.lo) / (self.hi - self.lo)

    def clearance(self, p) -> float:
        return design.clearance(p, self.w50)

    def in_box(self, p, tol=1e-9) -> bool:
        x = np.asarray(p, float)[:6]
        return bool(np.all(x >= self.lo - tol) and np.all(x <= self.hi + tol))

    def valid(self, p) -> bool:
        return self.in_box(p) and self.clearance(p) > C_MIN

    # --- sampling --------------------------------------------------------
    def starts(self, n: int, seed: int = 0, sampler: str = "lhs") -> np.ndarray:
        """n valid designs (N,8): space-filling draw over the box, rejection-sampled on clearance > C_MIN + margin."""
        from scipy.stats import qmc
        got, k = [], 0
        while len(got) < n:
            m = max(2 * n, 64)
            if sampler == "sobol":
                m = 1 << int(np.ceil(np.log2(m)))
                U = qmc.Sobol(d=6, scramble=True, seed=seed + k).random(m)
            else:
                U = qmc.LatinHypercube(d=6, seed=seed + k).random(m)
            for p in self.to_p(U):
                if self.clearance(p) > C_MIN + CLEAR_MARGIN:
                    got.append(p)
                if len(got) >= n:
                    break
            k += 1
            if k > 50:
                raise RuntimeError("could not draw enough valid starts")
        return np.asarray(got)

    # --- objective -------------------------------------------------------
    def evaluate(self, P: np.ndarray):
        """Batched: P (N,8) -> (J (N,), metrics list, terms list, S_mean, S_seeds, S_prior)."""
        P = np.atleast_2d(P)
        s_mean, s_seeds, s_prior = self.sur.predict(P)
        mets, terms, J = [], [], np.empty(P.shape[0])
        for i in range(P.shape[0]):
            met = response_metrics(self.sur.f, s_mean[i], self.spec)
            t = objective_terms(met, self.clearance(P[i]), self.spec)
            mets.append(met); terms.append(t); J[i] = t["total"]
        return J, mets, terms, s_mean, s_seeds, s_prior

    def fun_u(self, u: np.ndarray) -> float:
        return float(self.evaluate(self.to_p(u))[0][0])

    def fun_grad_u(self, u: np.ndarray, h: float = 1e-3):
        """J and its central-difference gradient in normalised coordinates: 13 designs in one batched pass."""
        U = np.vstack([u[None], u[None] + h * np.eye(6), u[None] - h * np.eye(6)])
        J = self.evaluate(self.to_p(U))[0]
        g = (J[1:7] - J[7:13]) / (2 * h)
        return float(J[0]), g

    def spread(self, s_seeds_i: np.ndarray, s_mean_i: np.ndarray) -> dict:
        """Ensemble spread for one design: std over seeds of passband |S21| dB (mean over the -3 dB band of the
        ensemble mean, as in eval.fig_calibration), plus the std over seeds of f0 and BW."""
        _, s21m = unpack_np(s_mean_i)
        dbm = db(s21m)
        mask = dbm >= dbm.max() - 3.0
        db_seeds = np.stack([db(unpack_np(s_seeds_i[k])[1]) for k in range(s_seeds_i.shape[0])])
        sd = db_seeds.std(0)
        per_seed = [response_metrics(self.sur.f, s_seeds_i[k], self.spec) for k in range(s_seeds_i.shape[0])]
        return {"passband_S21_std_dB_mean": float(sd[mask].mean()), "passband_S21_std_dB_max": float(sd[mask].max()),
                "f0_std_MHz": float(1e3 * np.std([m["f0_GHz"] for m in per_seed])),
                "bw_std_MHz": float(1e3 * np.std([m["bw_GHz"] for m in per_seed])),
                "rl_min_std_dB": float(np.std([m["rl_min_dB"] for m in per_seed])),
                "per_seed": [{k: m[k] for k in ("f0_GHz", "bw_GHz", "rl_min_dB", "il_dB", "ripple_dB")} for m in per_seed]}


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
def solve(spec: Spec, sur: Surrogate, n_starts: int = 64, k: int = 8, seed: int = 0, sampler: str = "lhs",
          maxiter: int = 150, polish: bool = True, verbose: bool = True, box: Optional[dict] = None) -> dict:
    from scipy.optimize import minimize
    log = print if verbose else (lambda *a, **kw: None)
    prob = Problem(spec, sur, box)
    t0 = time.perf_counter()
    P0 = prob.starts(n_starts, seed, sampler)
    J0, met0, _, _, _, _ = prob.evaluate(P0)
    t_starts = time.perf_counter() - t0
    order = np.argsort(J0)
    log(f"[inverse] {n_starts} valid starts evaluated in {t_starts:.2f} s; best start J = {J0[order[0]]:.3f} "
        f"(f0 {met0[order[0]]['f0_GHz']:.3f} GHz, BW {met0[order[0]]['bw_GHz']:.3f} GHz), median J = {np.median(J0):.1f}")

    t1 = time.perf_counter()
    cands = []
    for rank, i in enumerate(order[:k]):
        u0 = np.clip(prob.to_u(P0[i]), 0, 1)
        res = minimize(prob.fun_grad_u, u0, jac=True, method="L-BFGS-B", bounds=[(0, 1)] * 6,
                       options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6})
        u, J, nfev = res.x, float(res.fun), int(res.nfev)
        if polish and J > 1e-8:               # nothing left to polish when every term is already zero
            r2 = minimize(prob.fun_u, u, method="Nelder-Mead", bounds=[(0, 1)] * 6,
                          options={"maxfev": 300, "xatol": 1e-5, "fatol": 1e-8, "initial_simplex": _simplex(u, 0.01)})
            if r2.fun < J:
                u, J = r2.x, float(r2.fun)
            nfev += int(r2.nfev)
        p = prob.to_p(np.clip(u, 0, 1))[0]
        ok = prob.valid(p)
        cands.append({"start_rank": rank, "p": p, "J": J, "J_start": float(J0[i]), "nfev": nfev, "valid": ok,
                      "lbfgs_message": str(res.message)})
        log(f"[inverse] refine {rank + 1}/{k}: J {J0[i]:.3f} -> {J:.4f}  ({nfev} evals, clearance {prob.clearance(p):.4f} mm"
            f"{'' if ok else ', INVALID -> dropped'})")
    t_refine = time.perf_counter() - t1
    cands = [c for c in cands if c["valid"]]
    if not cands:
        raise RuntimeError("every refined candidate violates the clearance rule; try more starts")
    P = np.array([c["p"] for c in cands])
    J, mets, terms, s_mean, s_seeds, s_prior = prob.evaluate(P)
    for i, (c, met, t, j) in enumerate(zip(cands, mets, terms, J)):
        c.update(metrics=met, terms=t, J=float(j), clearance_mm=prob.clearance(c["p"]), idx=i,
                 spread=prob.spread(s_seeds[:, i], s_mean[i]))
    # the spec usually under-determines p (6 parameters, 2 targets + hinges): several candidates reach J ~ 0.
    # Ties (J within 1e-3) go to the candidate the ensemble agrees on most (smallest passband spread), then lowest IL.
    cands.sort(key=lambda c: (round(c["J"], 3), c["spread"]["passband_S21_std_dB_mean"], c["metrics"]["il_dB"]))
    best = cands[0]; b = best["idx"]
    best["prior_only"] = response_metrics(sur.f, s_prior[b], spec)
    best["prior_only_terms"] = objective_terms(best["prior_only"], best["clearance_mm"], spec)
    total = time.perf_counter() - t0
    return {"spec": spec, "problem": prob, "candidates": cands, "best": best,
            "S": {"f": sur.f, "mean": s_mean[b], "seeds": s_seeds[:, b], "prior": s_prior[b]},
            "starts": {"n": int(n_starts), "J": J0, "P": P0, "sampler": sampler, "seed": seed},
            "timing_s": {"starts": t_starts, "refine": t_refine, "total": total, "surrogate_evals": sur.n_eval}}


def _simplex(u, step):
    S = np.tile(u, (7, 1))
    for i in range(6):
        S[i + 1, i] = np.clip(u[i] + (step if u[i] + step <= 1 else -step), 0, 1)
    return S


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def fmt_p(p, nd=4):
    return "  ".join(f"{n}={v:.{nd}f}" for n, v in zip(GEOM, p[:6]))


def _met_line(m, spec: Spec):
    s = (f"f0 {m['f0_band_GHz']:.4f} GHz (band centre) / {m['f0_centroid_GHz']:.4f} (centroid)   "
         f"BW {1e3 * m['bw_GHz']:.1f} MHz [{m['f_lo_GHz']:.3f}, {m['f_hi_GHz']:.3f}]   "
         f"RL(min, spec band) {m['rl_min_dB']:.2f} dB   IL(peak) {m['il_dB']:.2f} dB   ripple {m['ripple_dB']:.2f} dB")
    if spec.reject:
        s += "   " + ", ".join(f"|S21|({fr:g} GHz) {v:.1f} dB" for (fr, _), v in zip(spec.reject, m["reject_dB"]))
    if m["split_passband"]:
        s += "   [split passband]"
    return s


def report(res: dict, name: str, out: Path, cmd_python: str) -> dict:
    spec: Spec = res["spec"]; prob: Problem = res["problem"]; best = res["best"]; p = best["p"]
    m, pr, sp, T = best["metrics"], best["prior_only"], best["spread"], res["timing_s"]
    lines = []
    A = lines.append
    A("=" * 100)
    A(f"inverse design  [{name}]   substrate {SUBSTRATE['name']} eps_r {prob.eps_r} h {prob.h} mm   "
      f"{res['starts']['n']} starts ({res['starts']['sampler']}) -> {len(res['candidates'])} refined candidates")
    rej = ", ".join(f"|S21| <= -{b:g} dB @ {a:g} GHz" for a, b in spec.reject) or "none"
    A(f"spec: f0 {spec.f0:.4f} GHz ({spec.f0_def})   BW {spec.bw:.4f} GHz   RL >= {spec.rl:g} dB   "
      f"ripple <= {spec.ripple if spec.ripple is not None else '-'} dB   rejection: {rej}   "
      f"(RL/ripple judged on f0 +- {spec.inband_frac:g} x BW/2)")
    A("-" * 100)
    A(f"best p (mm):  {fmt_p(p)}   clearance {best['clearance_mm']:.4f} mm (rule > {C_MIN} mm)")
    A(f"objective J = {best['J']:.4f}   terms: " + ", ".join(f"{k} {v:.3f}" for k, v in best["terms"].items() if k != "total"))
    A("ensemble mean :  " + _met_line(m, spec))
    A("prior only    :  " + _met_line(pr, spec) + f"   (J_prior = {best['prior_only_terms']['total']:.2f})")
    A(f"correction moved f0 by {1e3 * (m['f0_GHz'] - pr['f0_GHz']):+.1f} MHz, BW by {1e3 * (m['bw_GHz'] - pr['bw_GHz']):+.1f} MHz, "
      f"RL(min) by {m['rl_min_dB'] - pr['rl_min_dB']:+.2f} dB vs the prior")
    A(f"ensemble spread (trust): passband |S21| std over {prob.sur.n_seeds} seeds {sp['passband_S21_std_dB_mean']:.3f} dB mean / "
      f"{sp['passband_S21_std_dB_max']:.3f} dB max;  f0 std {sp['f0_std_MHz']:.1f} MHz, BW std {sp['bw_std_MHz']:.1f} MHz, "
      f"RL std {sp['rl_min_std_dB']:.2f} dB   (held-out: spread 0.22 dB mean vs error 0.52 dB, r = 0.55)")
    A(f"wall time: starts {T['starts']:.2f} s, refinement {T['refine']:.2f} s, total {T['total']:.2f} s "
      f"({T['surrogate_evals']} surrogate evaluations)")
    A("-" * 100)
    A(f"{'#':>2} {'J':>9} {'f0 GHz':>8} {'BW MHz':>7} {'RL dB':>6} {'IL dB':>6} {'rip dB':>6} {'sprd dB':>7} {'clr mm':>7}  p (w1 w2 g1 g2 l1 l2)")
    for i, c in enumerate(res["candidates"][:5]):
        cm = c["metrics"]
        A(f"{i + 1:>2} {c['J']:>9.4f} {cm['f0_GHz']:>8.4f} {1e3 * cm['bw_GHz']:>7.1f} {cm['rl_min_dB']:>6.2f} {cm['il_dB']:>6.2f} "
          f"{cm['ripple_dB']:>6.2f} {c['spread']['passband_S21_std_dB_mean']:>7.3f} {c['clearance_mm']:>7.4f}  "
          + " ".join(f"{v:.4f}" for v in c["p"][:6]))
    A("   (ties in J within 1e-3 are ranked by ensemble spread, then IL)")
    A("-" * 100)
    p_str = " ".join(f"{v:.6f}" for v in p[:6]) + f" {prob.eps_r} {prob.h}"
    cmd = f"{cmd_python} build_case.py --p {p_str} --name inv_{name}"
    A("Palace verification (not run here):")
    A("  " + cmd)
    A(f"  then: python collect.py (routes the run) and compare runs/inv_{name} against {out / 'sparams_pred.npz'}")
    A("=" * 100)
    print("\n".join(lines))

    out.mkdir(parents=True, exist_ok=True)
    result = {
        "name": name, "spec": spec.to_dict(), "param_names": list(prob.names),
        "p": [float(v) for v in p], "p_mm": {n: float(v) for n, v in zip(GEOM, p[:6])}, "clearance_mm": best["clearance_mm"],
        "objective": best["terms"], "predicted": m, "prior_only": pr, "prior_only_objective": best["prior_only_terms"],
        "ensemble_spread": sp, "n_seeds": prob.sur.n_seeds,
        "candidates_top5": [{"J": c["J"], "p": [float(v) for v in c["p"]], "clearance_mm": c["clearance_mm"],
                             "metrics": c["metrics"], "terms": c["terms"], "start_rank": c["start_rank"], "nfev": c["nfev"],
                             "spread_passband_S21_std_dB": c["spread"]["passband_S21_std_dB_mean"]}
                            for c in res["candidates"][:5]],
        "starts": {"n": res["starts"]["n"], "sampler": res["starts"]["sampler"], "seed": res["starts"]["seed"],
                   "J_min": float(res["starts"]["J"].min()), "J_median": float(np.median(res["starts"]["J"]))},
        "timing_s": T, "box": prob.box, "build_case_cmd": cmd,
        "f_grid_GHz": [float(prob.sur.f.min()), float(prob.sur.f.max()), int(len(prob.sur.f))],
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=float))
    S = res["S"]
    s11m, s21m = unpack_np(S["mean"]); s11s, s21s = unpack_np(S["seeds"]); s11p, s21p = unpack_np(S["prior"])
    np.savez(out / "sparams_pred.npz", f=S["f"], S11=s11m, S21=s21m, S11_seeds=s11s, S21_seeds=s21s,
             S11_prior=s11p, S21_prior=s21p, p=np.asarray(p, float), param_names=np.array(prob.names),
             note="ensemble mean of the stage-B seeds (S11, S21), per seed (S11_seeds, S21_seeds) and the prior; f in GHz")
    verify = {"id": f"inv_{name}", "p": [float(v) for v in p], "kind": "inverse", "clearance_mm": round(best["clearance_mm"], 5),
              "valid": True, "spec": spec.to_dict(), "predicted": {k: m[k] for k in ("f0_band_GHz", "f0_centroid_GHz", "bw_GHz", "rl_min_dB", "il_dB", "ripple_dB")},
              "build_case_cmd": cmd, "note": "inverse.py candidate; add to samples.json['samples'] or run build_case.py --p directly"}
    (out / f"verify_{name}.json").write_text(json.dumps(verify, indent=1))
    make_figure(res, name, out)
    print(f"[inverse] wrote {out / 'result.json'}, sparams_pred.npz, verify_{name}.json, inverse_{name}.png")
    return result


def make_figure(res: dict, name: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec: Spec = res["spec"]; S = res["S"]; f = S["f"]; m = res["best"]["metrics"]
    s11m, s21m = unpack_np(S["mean"]); s11s, s21s = unpack_np(S["seeds"]); s11p, s21p = unpack_np(S["prior"])
    d21s, d11s = db(s21s), db(s11s)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    for ax, dm, dsd, dp, lab, ylo in ((axes[0], db(s21m), d21s, db(s21p), "|S21|", -60), (axes[1], db(s11m), d11s, db(s11p), "|S11|", -40)):
        ax.fill_between(f, dsd.min(0), dsd.max(0), alpha=0.25, label=f"seed band (min-max of {dsd.shape[0]} seeds)")
        ax.plot(f, dm, lw=2, label="ensemble mean (prior + correction)")
        ax.plot(f, dp, "--", lw=1.2, label="prior only")
        ax.axvline(spec.f0, color="k", lw=1, ls=":", label="target f0")
        ax.axvspan(spec.f0 - spec.bw / 2, spec.f0 + spec.bw / 2, color="k", alpha=0.06, label="target -3 dB band")
        ax.set_ylabel(f"{lab} (dB)"); ax.set_ylim(ylo, 2); ax.grid(alpha=0.3)
    half = 0.5 * spec.bw * spec.inband_frac
    axes[0].axhline(-3 + 20 * np.log10(max(abs(s21m).max(), 1e-12)), color="k", lw=0.8, ls="-.", label="-3 dB from peak")
    axes[0].plot([m["f_lo_GHz"], m["f_hi_GHz"]], [-3 + 20 * np.log10(max(abs(s21m).max(), 1e-12))] * 2, "k|", ms=10)
    for fr, lvl in spec.reject:
        axes[0].plot(fr, -lvl, "rv", ms=8, label="rejection: |S21| <= this" if (fr, lvl) == spec.reject[0] else None)
    axes[1].plot([spec.f0 - half, spec.f0 + half], [-spec.rl, -spec.rl], "r-", lw=2, label=f"RL >= {spec.rl:g} dB over spec band")
    axes[0].legend(fontsize=7, loc="lower left"); axes[1].legend(fontsize=7, loc="lower left")
    axes[1].set_xlabel("f (GHz)")
    rej = ", ".join(f"{b:g} dB @ {a:g} GHz" for a, b in spec.reject)
    fig.suptitle(f"inverse design {name}: spec f0 {spec.f0:g} GHz, BW {spec.bw:g} GHz, RL >= {spec.rl:g} dB"
                 + (f", reject {rej}" if rej else "") + f"\npredicted f0 {m['f0_GHz']:.3f} GHz, BW {1e3 * m['bw_GHz']:.0f} MHz, "
                 f"RL {m['rl_min_dB']:.1f} dB, IL {m['il_dB']:.2f} dB; ensemble spread {res['best']['spread']['passband_S21_std_dB_mean']:.2f} dB"
                 f"\np (mm): {fmt_p(res['best']['p'], 4)}", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / f"inverse_{name}.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# --list-box
# ---------------------------------------------------------------------------
def list_box(sur: Optional[Surrogate] = None):
    box = D.load_box()
    js = D._read_samples_json() or {}
    centre = js.get("centre")
    print("parameter box (samples.json):")
    print(f"  {'param':>6} {'lo':>10} {'hi':>10} {'centre':>10}   {'span':>8}")
    for n in D.FALLBACK_PARAM_NAMES:
        lo, hi = box[n]
        c = centre[list(js["param_names"]).index(n)] if centre else 0.5 * (lo + hi)
        span = f"+-{50 * (hi - lo) / c:.0f} %" if hi > lo else "fixed"
        print(f"  {n:>6} {lo:>10.4f} {hi:>10.4f} {c:>10.4f}   {span:>8}")
    v = js.get("validity", {})
    print(f"  geometry rule: {v.get('rule', f'clearance(p) > {C_MIN} mm')}; W50 = {v.get('W50_mm', '?')} mm; "
          f"centre clearance {design.clearance(centre) if centre else float('nan'):.4f} mm")
    files = sorted(D.PALACE_DIR.glob("*.npz")) if D.PALACE_DIR.exists() else []
    if not files:
        print(f"achievable range (data/palace not on disk; recorded): {DATA_RANGE_NOTE}")
        return
    valid = {s["id"]: s.get("valid", True) for s in js.get("samples", [])}
    rows = []
    for fp in files:
        try:
            z = np.load(fp, allow_pickle=True)
            f = D.to_ghz(D._get(z, "f")); s21 = D._complex(D._get(z, "S21")).reshape(-1); s11 = D._complex(D._get(z, "S11")).reshape(-1)
        except Exception as e:
            print(f"  skip {fp.name}: {e}"); continue
        if not valid.get(fp.stem, True):
            continue
        d21 = db(s21)
        lo, hi, _ = band_edges(f, d21)
        rows.append((fp.stem, centroid_freq(f, s21), 0.5 * (lo + hi), hi - lo, -d21.max(), passband_is_split(d21),
                     -db(s11)[(f >= 0.5 * (lo + hi) - 0.4 * (hi - lo)) & (f <= 0.5 * (lo + hi) + 0.4 * (hi - lo))].max() if hi > lo else np.nan))
    prod = [r for r in rows if r[0].startswith("s")]
    nom = [r for r in rows if r[0] == "nominal"]
    a = np.array([[r[1], r[2], r[3], r[4], r[6]] for r in prod], float)
    split = np.array([r[5] for r in prod])
    print(f"achievable range in the Palace data ({len(prod)} valid production samples under data/palace; eval.py definitions):")
    print(f"  {'metric':<28} {'min':>8} {'p5':>8} {'median':>8} {'p95':>8} {'max':>8}")
    for lab, col in (("f0 centroid (GHz)", 0), ("f0 -3 dB band centre (GHz)", 1), ("-3 dB BW (GHz)", 2), ("IL at peak (dB)", 3), ("RL min, inner 80 % band (dB)", 4)):
        x = a[:, col]; x = x[np.isfinite(x)]
        print(f"  {lab:<28} {x.min():>8.3f} {np.percentile(x, 5):>8.3f} {np.median(x):>8.3f} {np.percentile(x, 95):>8.3f} {x.max():>8.3f}")
    x = a[~split, 2]
    print(f"  -3 dB BW, single-passband only ({(~split).sum()} samples): {x.min():.3f} - {x.max():.3f} GHz; "
          f"{split.sum()} samples have a split passband (dip > 3 dB between peaks)")
    if nom:
        r = nom[0]
        print(f"  nominal (Palace): f0 centroid {r[1]:.4f} / band centre {r[2]:.4f} GHz, BW {1e3 * r[3]:.1f} MHz, IL {r[4]:.2f} dB, RL {r[6]:.2f} dB")
    mp = RESULTS / "metrics.json"
    if mp.exists():
        s = json.loads(mp.read_text())["methods"].get("stageB")
        if s:
            print(f"  stage-B ensemble held-out accuracy (n={s['n']}): passband |S21| {s['passband_S21_dB_mean']:.2f} dB, "
                  f"f0 {s['f0_err_MHz_mean_abs']:.1f} MHz mean abs (max {s['f0_err_MHz_max_abs']:.0f}), BW {s['bw_err_MHz_mean_abs']:.1f} MHz mean abs")
    print("  (specs outside these ranges extrapolate the surrogate; watch the ensemble spread)")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f0", type=float, help="centre frequency (GHz)")
    ap.add_argument("--bw", type=float, help="-3 dB bandwidth (GHz)")
    ap.add_argument("--rl", type=float, default=10.0, help="minimum in-band return loss (dB, default 10)")
    ap.add_argument("--ripple", type=float, default=None, help="maximum passband ripple (dB) inside the spec band")
    ap.add_argument("--reject", action="append", default=None, metavar="f_GHz:dB", help="|S21| <= -dB at f_GHz (repeatable)")
    ap.add_argument("--f0-def", choices=["band", "centroid"], default="band", help="f0 definition used in the objective")
    ap.add_argument("--inband-frac", type=float, default=0.8, help="RL/ripple are judged on f0 +- this x BW/2")
    ap.add_argument("--w-rl", type=float, default=1.0); ap.add_argument("--w-ripple", type=float, default=1.0); ap.add_argument("--w-reject", type=float, default=1.0)
    ap.add_argument("--n-starts", type=int, default=64)
    ap.add_argument("--k", type=int, default=8, help="starts refined by L-BFGS-B")
    ap.add_argument("--sampler", choices=["lhs", "sobol"], default="lhs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--no-polish", action="store_true", help="skip the Nelder-Mead polish after L-BFGS-B")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default runs_inverse/<name>)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--list-box", action="store_true", help="print the parameter box and the achievable range in the data")
    args = ap.parse_args(argv)
    if args.list_box:
        list_box()
        if args.f0 is None:
            return
    if args.f0 is None or args.bw is None:
        ap.error("--f0 and --bw are required (or --list-box)")
    spec = Spec(f0=args.f0, bw=args.bw, rl=args.rl, ripple=args.ripple, reject=parse_reject(args.reject),
                f0_def=args.f0_def, inband_frac=args.inband_frac, w_rl=args.w_rl, w_ripple=args.w_ripple, w_reject=args.w_reject)
    name = args.name or f"f{args.f0:.2f}_bw{args.bw:.2f}"
    out = args.out or (RUNS_INVERSE / name)
    t0 = time.perf_counter()
    sur = Surrogate()
    print(f"[inverse] loaded {sur.n_seeds} stage-B seeds from {CKPT} ({time.perf_counter() - t0:.2f} s); "
          f"f grid {sur.f.min():.2f}-{sur.f.max():.2f} GHz, {len(sur.f)} points")
    res = solve(spec, sur, n_starts=args.n_starts, k=args.k, seed=args.seed, sampler=args.sampler,
                maxiter=args.maxiter, polish=not args.no_polish)
    py = ".venv/Scripts/python" if sys.platform.startswith("win") else ".venv/bin/python"
    report(res, name, out, py)


if __name__ == "__main__":
    main()
