"""Space-mapping tune of the nominal filter in Palace: make the FEM response equiripple at 5 GHz.

Coarse model: the analytic prior (prior.s_prior). Fine model: Palace (coarse mesh, order 2, AdaptiveTol 1e-2,
3-7 GHz sweep). Aggressive space mapping on x = (g1, g2, l1, l2), widths fixed at the Pozar values:
  1. run Palace at p_f(k);
  2. parameter extraction: x_c(k) = argmin || prior(x) - Palace(p_f(k)) || over f0 +- 1.5 BW
     (|S21| in dB clipped at -30 dB, |S11| linear x 25), i.e. the prior parameters that reproduce the FEM response;
  3. p_f(k+1) = p_f(k) - (x_c(k) - x*), where x* is the prior-optimal design (design.json nominal p);
  4. stop when the Palace features meet the targets or after max_iter runs.
Targets (on the Palace response): f0 within 0.5 % of the prior's f0, 3 dB BW within +-1 % (absolute fractional
bandwidth) of the prior's, |S11| in-band peaks within 1 dB of each other. Everything is logged to tune_log.json.

    python tune.py [--max-iter 5] [--ranks 8]
    python tune.py --design design_rl20.json --log tune_log_rl20.json --prefix tune20_ --rl 20
        (another nominal; --rl also requires every in-band |S11| peak to be at least RL - 1 dB down)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_case  # noqa: E402
import collect  # noqa: E402
import design  # noqa: E402
import prior  # noqa: E402

def _queue_cmd() -> list:
    """bash run_queue.sh queue_tune.txt; on Windows set RFS_WSL_DISTRO (e.g. Ubuntu-24.04) to run it inside WSL."""
    sh, q = HERE / "run_queue.sh", HERE / "queue_tune.txt"
    distro = os.environ.get("RFS_WSL_DISTRO")
    if not distro:
        return ["bash", str(sh), str(q)]

    def wsl(p: Path) -> str:                  # C:/x -> /mnt/c/x
        return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]
    return ["wsl", "-d", distro, "--", "bash", wsl(sh), wsl(q)]


WSL = _queue_cmd()
SETTINGS = dict(scale=1.8, tol=1e-2, samples=build_case.SAMPLES_NARROW, order=2)
LOG = HERE / "tune_log.json"
DAMP = 0.8                      # space-mapping step damping
EXTRACT_SPAN = 0.5              # +-50 % bounds for the parameter extraction


def run_palace(name, p, ranks):
    d = HERE / "runs" / name
    if not (d / "output" / "port-S.csv").exists():
        build_case.build(name, p, **SETTINGS)
        (HERE / "queue_tune.txt").write_bytes(f"{name} {ranks}\n".encode())      # LF: bash reads it
        t0 = time.time()
        subprocess.run(WSL, check=False)
        wall = time.time() - t0
    else:
        wall = None
    res = collect.collect(name, ranks)
    z = np.load(res["npz"])
    return res, z["f_raw"], z["S11_raw"], z["S21_raw"], wall


def extract(p_f, f, s11, s21, f0, bw, span=0.35):
    """Prior parameters x_c = (g1, g2, l1, l2) that reproduce the Palace response (widths of p_f kept)."""
    band = (f >= f0 - 1.5 * bw) & (f <= f0 + 1.5 * bw)
    fb, s11b, s21b = f[band], s11[band], s21[band]
    t21 = np.maximum(20 * np.log10(np.abs(s21b) + 1e-12), -30)
    t11 = np.abs(s11b)

    def resid(x):
        a, b = prior.s_prior(list(p_f[:2]) + list(np.exp(x)) + list(p_f[6:]), fb)
        return np.concatenate([np.maximum(20 * np.log10(np.abs(b) + 1e-12), -30) - t21, 25 * (np.abs(a) - t11)])

    x0 = np.log(np.asarray(p_f[2:6], float))
    r = least_squares(resid, x0, bounds=(x0 + np.log(1 - span), x0 + np.log(1 + span)), xtol=1e-10, ftol=1e-10)
    return [float(v) for v in np.exp(r.x)], float(r.cost)


def features(f, s11, s21):
    return prior.passband_features(f, s11, s21)


def meets(feat, target, rl=None):
    ok_f0 = abs(feat["f0_GHz"] - target["f0_GHz"]) / target["f0_GHz"] <= 0.005
    ok_bw = abs(feat["BW_GHz"] / feat["f0_GHz"] - target["BW_GHz"] / target["f0_GHz"]) <= 0.01
    ok_rip = len(feat["peaks_dB"]) == 2 and feat["peak_spread_dB"] <= 1.0
    parts = {"f0": ok_f0, "BW": ok_bw, "ripple": ok_rip}
    if rl is not None:
        parts["RL"] = len(feat["peaks_dB"]) > 0 and max(feat["peaks_dB"]) <= -(rl - 1.0)
    return all(parts.values()), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--design", default="design.json", help="file with the prior-optimal nominal")
    ap.add_argument("--log", default="tune_log.json")
    ap.add_argument("--prefix", default="tune_", help="run name prefix")
    ap.add_argument("--rl", type=float, help="return-loss target (dB) for the stop rule")
    a = ap.parse_args()
    global LOG
    LOG = HERE / a.log
    d = json.loads((HERE / a.design).read_text())
    p_star = list(d["nominal"]["p"])                         # prior-optimal (equiripple in the prior)
    target = d["nominal"]["prior_features"]                   # lossy prior features at p_star
    log = {"settings": {k: (v if not isinstance(v, list) else v) for k, v in SETTINGS.items()}, "p_star": p_star,
           "target": target, "iterations": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t_start = time.time()
    p_f = list(d["nominal"].get("start_p") or p_star)        # optional first guess, e.g. mapped from an earlier tune
    for k in range(a.max_iter):
        name = f"{a.prefix}{k}"
        res, f, s11, s21, wall = run_palace(name, p_f, a.ranks)
        feat = features(f, s11, s21)
        x_c, cost = extract(p_f, f, s11, s21, feat["f0_GHz"], feat["BW_GHz"], span=EXTRACT_SPAN)
        ok, parts = meets(feat, target, a.rl)
        # aggressive space-mapping step, damped (0.8) and clamped to +-20 % per iteration per parameter
        step = [DAMP * (xs - xc) for xs, xc in zip(p_star[2:6], x_c)]
        step = [float(np.clip(st, -0.2 * pf, 0.2 * pf)) for st, pf in zip(step, p_f[2:6])]
        p_next = list(p_f[:2]) + [pf + st for pf, st in zip(p_f[2:6], step)] + list(p_f[6:])
        it = {"k": k, "run": name, "p_f": p_f, "features": feat, "x_c": x_c, "extraction_cost": cost,
              "meets": parts, "converged": ok, "p_next": p_next, "palace_wall_s": res.get("wall_s"),
              "iteration_wall_s": wall, "solves": res.get("solves")}
        log["iterations"].append(it)
        LOG.write_text(json.dumps(log, indent=1))
        print(f"iter {k}: f0 {feat['f0_GHz']:.4f} BW {feat['BW_GHz']:.4f} IL {feat['IL_dB']:.3f} zeros {np.round(feat['zeros_GHz'], 4).tolist()} "
              f"peaks {np.round(feat['peaks_dB'], 2).tolist()} spread {feat['peak_spread_dB']:.2f} dB  meets {parts}  palace {res.get('wall_s')} s")
        if ok:
            break
        p_f = p_next
    log["final_run"] = log["iterations"][-1]["run"]
    log["final_p"] = log["iterations"][-1]["p_f"]
    log["converged"] = log["iterations"][-1]["converged"]
    log["total_wall_s"] = time.time() - t_start
    log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG.write_text(json.dumps(log, indent=1))
    print("final p", log["final_p"], "converged", log["converged"], f"total {log['total_wall_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
