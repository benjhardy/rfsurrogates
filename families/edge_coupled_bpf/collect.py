"""Palace run folder -> npz (f, S11, S21 on the common 2-8 GHz / 0.02 GHz grid, plus the raw sample points),
passband metrics (f0, 3 dB BW, IL, RL), and, with --db, one rflab run record (project "rfsurrogates").

    python collect.py --run s0001 [--ranks 8]          -> data/palace/s0001.npz  (production samples sNNNN)
    python collect.py --run conv_c                       -> data/aux/conv_c.npz     (mesh / timing studies)
    python collect.py --all                              -> every finished run folder

The npz keys follow what core/data.py reads: p (7), f (M, GHz), S11, S21 (complex, M), wall_s, substrate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("PALACE_AGENT_TOOLS"):              # a local checkout instead of `pip install palace-agent-tools`
    sys.path.insert(0, os.environ["PALACE_AGENT_TOOLS"])
import prior  # noqa: E402
from palace_tools.logparse import parse_log, parse_palace_json, parse_time_v  # noqa: E402

F_GRID = np.round(np.arange(2.0, 8.0 + 1e-9, 0.02), 4)
PROD = re.compile(r"^s\d{4}$")          # production samples -> data/palace (training)
HOLD = re.compile(r"^h\d{4}$")          # medium-mesh fidelity holdouts -> data/holdout (never for training)
HOLDC = re.compile(r"^hc\d{4}$")        # coarse-mesh twins of the holdouts -> data/holdout_coarse (never for training)
INV = re.compile(r"^inv_")              # inverse-design verification solves -> data/inverse_check (never for training)


def read_port_s(path: Path):
    """Palace port-S.csv -> f (GHz), S (M, 2, 2) complex (ports 1,2). Missing columns (unexcited port) are NaN."""
    with open(path) as fh:
        header = [h.strip() for h in fh.readline().split(",")]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    f = data[:, 0]
    s = np.full((f.size, 2, 2), np.nan + 0j)
    for k, h in enumerate(header):
        m = re.match(r"\|S\[(\d)\]\[(\d)\]\| \(dB\)", h)
        if m:
            i, j = int(m.group(1)) - 1, int(m.group(2)) - 1
            mag = 10 ** (data[:, k] / 20)
            ph = np.deg2rad(data[:, k + 1])
            s[:, i, j] = mag * np.exp(1j * ph)
    order = np.argsort(f)
    return f[order], s[order]


def resample(f_raw, s_raw, f_new):
    """Cubic spline on real and imaginary parts (the raw grid is 0.02 GHz across the band, 0.1 GHz outside)."""
    re_, im_ = CubicSpline(f_raw, s_raw.real), CubicSpline(f_raw, s_raw.imag)
    return re_(f_new) + 1j * im_(f_new)


def metrics(f, s11, s21):
    f0, bw, il = prior.passband_metrics(f, s21)
    out = {"f0_GHz": f0, "BW_GHz": bw, "BW_frac": bw / f0 if np.isfinite(f0) else float("nan"), "IL_dB": il}
    if np.isfinite(f0):
        band = (f >= 0.95 * f0) & (f <= 1.05 * f0)        # design band: 10 % centred on the measured f0
        out["RL_dB"] = float(-20 * np.log10(np.abs(s11[band]).max()))          # worst return loss in the design band
        out["RL_best_dB"] = float(-20 * np.log10(np.abs(s11[band]).min()))
    else:
        out["RL_dB"] = float("nan")
    out["S21_max_dB"] = float(20 * np.log10(np.abs(s21).max()))
    out["passivity_max"] = float((np.abs(s11) ** 2 + np.abs(s21) ** 2).max())
    return out


def collect(run: str, ranks: int | None = None, record: bool = False, run_dir: Path | None = None,
            out_npz: Path | None = None):
    """run_dir / out_npz: a case folder and an output file anywhere (tools/contribute.py); default runs/<run> and
    data/<split>/<run>.npz by the run-name pattern."""
    d = Path(run_dir) if run_dir else HERE / "runs" / run
    summ = json.loads((d / "build_summary.json").read_text())
    csv = d / "output" / "port-S.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    f_raw, s_raw = read_port_s(csv)
    s11_raw, s21_raw = s_raw[:, 0, 0], s_raw[:, 1, 0]
    if f_raw.size >= 4:
        f = F_GRID[(F_GRID >= f_raw.min() - 1e-9) & (F_GRID <= f_raw.max() + 1e-9)]
        s11, s21 = resample(f_raw, s11_raw, f), resample(f_raw, s21_raw, f)
    else:                                   # fields-only run: keep the points
        f, s11, s21 = f_raw, s11_raw, s21_raw
    met = metrics(f_raw, s11_raw, s21_raw) if f_raw.size >= 4 else {}
    log = parse_log((d / "palace.log").read_text(errors="replace")) if (d / "palace.log").exists() else {}
    tv = parse_time_v((d / "time.log").read_text(errors="replace")) if (d / "time.log").exists() else {}
    pj = sorted(d.rglob("palace.json"))
    pjs = parse_palace_json(pj[-1]) if pj else {}
    ranks = ranks or log.get("ranks") or pjs.get("ranks")
    wall = tv.get("wall_s") or log.get("elapsed_s")
    peak = log.get("peak_mem_gb") or pjs.get("peak_mem_gb")
    n_solves = (pjs.get("linear_solves") or log.get("krylov", {}).get("solves"))
    p = summ["p"]
    eps_r, h_sub = p[-2], p[-1]
    sub = "RO4350B" if abs(eps_r - 3.66) < 1e-6 else f"eps{eps_r}_h{h_sub}"
    if met:                                 # equiripple diagnostics, flattened to scalars for the run DB
        feat = prior.passband_features(f_raw, s11_raw, s21_raw)
        for i, z in enumerate(feat["zeros_GHz"]):
            met[f"zero{i + 1}_GHz"] = z
        for i, pk in enumerate(feat["peaks_dB"]):
            met[f"peak{i + 1}_dB"] = pk
        met["peak_spread_dB"] = feat["peak_spread_dB"]
        met["rl_band_dB"] = feat["rl_band_dB"]
    kind = ("production" if PROD.match(run) else "holdout" if HOLD.match(run) else
            "holdout_coarse" if HOLDC.match(run) else "inverse" if INV.match(run) else "study")
    out_dir = HERE / "data" / {"production": "palace", "holdout": "holdout", "holdout_coarse": "holdout_coarse",
                               "inverse": "inverse_check", "study": "aux"}[kind]
    if out_npz:
        out_dir = Path(out_npz).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = Path(out_npz) if out_npz else out_dir / f"{run}.npz"
    np.savez_compressed(npz, p=np.asarray(p, float), f=f.astype(np.float64), S11=s11.astype(np.complex128),
                        S21=s21.astype(np.complex128), f_raw=f_raw, S11_raw=s11_raw, S21_raw=s21_raw,
                        wall_s=float(wall or np.nan), ranks=int(ranks or 0), substrate=sub, run=run,
                        param_names=np.asarray(summ["param_names"]), scale=float(summ["scale"]),
                        unknowns=int(log.get("unknowns") or summ["unknowns_order2"]), tets=int(summ["tets"]),
                        peak_ram_gb=float(peak or np.nan), metrics=json.dumps(met))
    res = {"run": run, "npz": str(npz), "wall_s": wall, "peak_ram_gb": peak, "ranks": ranks,
           "unknowns": log.get("unknowns") or summ["unknowns_order2"], "tets": summ["tets"], "solves": n_solves,
           "adaptive": log.get("adaptive"), "timing_s": log.get("timing_s"), "n_errors": log.get("n_errors"), **met}
    (d / "result.json").write_text(json.dumps(res, indent=1, default=str))
    if record:
        import rflab
        checks = []
        if met:
            checks.append({"name": "passive", "passed": met["passivity_max"] <= 1.001, "value": met["passivity_max"],
                           "limit": 1.001, "note": "|S11|^2 + |S21|^2 at the worst sample"})
        checks.append({"name": "log_errors", "passed": not log.get("n_errors"), "value": log.get("n_errors"), "limit": 0, "note": ""})
        if "conductors_mesh" in summ:      # geometry gate (build_case.py): separate copper conductors in the mesh
            checks.append({"name": "conductors", "passed": summ["conductors_mesh"] == summ.get("conductors_expected", 5),
                           "value": summ["conductors_mesh"], "limit": summ.get("conductors_expected", 5),
                           "note": f"CAD fuse {summ.get('conductors_cad')}, mesh components {summ['conductors_mesh']}"})
        rec = rflab.record(
            "rfsurrogates", run, "palace@0.16", "em",
            params={**dict(zip(summ["param_names"], p)), "mesh_scale": summ["scale"], "tand": summ["tand"],
                    "samples_GHz": summ["samples_GHz"], "excite": summ["excite"], "order": 2},
            files=[d / f"{run}.msh", d / "config.json"],
            metrics={k: (v if np.isfinite(v) else "nan") for k, v in met.items()},
            outputs=[npz, csv], checks=checks,
            mesh={"elements": summ["tets"], "unknowns": res["unknowns"], "order": 2},
            compute={"host": "wsl:Ubuntu-24.04", "cores": 8, "ranks": ranks, "wall_s": wall, "peak_ram_gb": peak, "gpu": None},
            tags=["edge-coupled-bpf", kind, f"scale{summ['scale']}"] + (["fidelity-holdout", "medium"] if kind == "holdout" else
                                                                        ["fidelity-holdout", "coarse-twin"] if kind == "holdout_coarse" else
                                                                        ["inverse-check"] if kind == "inverse" else []),
            git=str(HERE), notes=f"adaptive sweep: {log.get('adaptive')}; solves {n_solves}",
            extra={"timing_s": log.get("timing_s"), "max_rss_one_rank_gb": tv.get("max_rss_one_process_gb")})
        res["run_id"] = rec["run_id"]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--ranks", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--db", action="store_true", help="also write an rflab run record (needs the rflab package)")
    ap.add_argument("--no-db", action="store_true", help="default; kept for old command lines")
    a = ap.parse_args()
    runs = [a.run] if a.run else sorted(p.parent.parent.name for p in (HERE / "runs").glob("*/output/port-S.csv"))
    for r in runs:
        try:
            res = collect(r, a.ranks, record=a.db and not a.no_db)
            print(f"{r}: f0 {res.get('f0_GHz', float('nan')):.4f} GHz, BW {res.get('BW_GHz', float('nan')):.4f} GHz, "
                  f"IL {res.get('IL_dB', float('nan')):.3f} dB, RL {res.get('RL_dB', float('nan')):.2f} dB, wall {res['wall_s']} s, "
                  f"peak {res['peak_ram_gb']} GB, {res['unknowns']} unknowns, {res['solves']} solves -> {res['npz']}")
        except Exception as e:  # noqa: BLE001
            print(f"{r}: collect failed: {e!r}")
