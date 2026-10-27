"""Circuit-tier dataset: designs sampled in the family box, solved in a circuit simulator, written as one npz.

    python tools/make_circuit_data.py --family edge_coupled_bpf --n 500 [--mode physical] [--seed 20261026]

The family must have a circuit.py with solve_batch(P, mode, names, case_dir) (edge_coupled_bpf: qucsator through
circuit-agent-tools). Designs: Latin hypercube over the free parameters of the box in samples.json, rejection-sampled
with the family's validity rule (the same rule the Palace data obey), so the circuit data cover the same design
space. Output families/<id>/data/circuit/circuit_<mode>_n<N>.npz (a subfolder, so that
core/data.find_prior_dataset() does not take it for the prior dataset) with the keys of the prior dataset (p, f in GHz, S11, S21,
param_names, substrate) plus ids, wall_s per design, solver, mode, fidelity "circuit", seed and the host load.
The file is rewritten after every batch, so a stopped run keeps what it finished. The run summary (rate, failures,
host load) goes to families/<id>/results/circuit_data_<mode>_n<N>.json. Nothing here starts an EM solver.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import registry  # noqa: E402


def sample_valid(fam: dict, n: int, seed: int) -> np.ndarray:
    """n valid designs: LHS over the free box parameters, rejection by fam['is_valid'], in draw order."""
    names, box = fam["param_names"], fam["box"]
    free = [k for k in names if box[k][1] > box[k][0]]
    lo = np.array([box[k][0] for k in free]); hi = np.array([box[k][1] for k in free])
    base = np.array([box[k][0] for k in names], float)
    got, k = [], 0
    while len(got) < n:
        u = qmc.LatinHypercube(d=len(free), seed=seed + k).random(max(4 * n, 64))
        for row in lo + u * (hi - lo):
            p = base.copy()
            for j, name in enumerate(free):
                p[names.index(name)] = row[j]
            if fam["is_valid"](p):
                got.append(p)
                if len(got) == n:
                    break
        k += 1
    return np.array(got)


def load_circuit(family_id: str):
    d = registry.family_dir(family_id)
    if not (d / "circuit.py").exists():
        raise SystemExit(f"family {family_id} has no circuit.py")
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(f"rfs_circuit_{family_id}", d / "circuit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", default=registry.DEFAULT_FAMILY, choices=registry.family_ids())
    ap.add_argument("--n", type=int, required=True, help="number of valid designs")
    ap.add_argument("--mode", default="physical", help="circuit mode of the family's circuit.py (default physical)")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--batch", type=int, default=50, help="designs per simulator-host call")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    fam = registry.load_family(a.family)
    cir = load_circuit(a.family)
    from circuit_tools import run as runmod

    # a subfolder: core/data.find_prior_dataset() takes any npz directly in data/ as the prior dataset
    out = a.out or registry.family_dir(a.family) / "data" / "circuit" / f"circuit_{a.mode}_n{a.n}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    P = sample_valid(fam, a.n, a.seed)
    ids = [f"c{i:05d}" for i in range(a.n)]
    sub = fam["substrate"]
    substrate = [sub["name"] if np.isclose(p[-2], sub["eps_r"]) and np.isclose(p[-1], sub["h_mm"])
                 else f"eps{p[-2]:g}_h{p[-1]:g}" for p in P]
    host = runmod.host_status()
    print(f"[circuit-data] {a.family} {a.mode}: {a.n} designs, batches of {a.batch}; host load "
          f"{host.get('load_1m')} on {host.get('cpus')} CPUs, {host.get('heavy_em_processes')} EM processes")
    S11 = np.full((a.n, len(cir.F_GHZ)), np.nan + 0j)
    S21 = S11.copy()
    wall = np.full(a.n, np.nan)
    failed, batches, t_start = [], [], time.time()
    for b0 in range(0, a.n, a.batch):
        sl = slice(b0, min(a.n, b0 + a.batch))
        st = runmod.host_status()
        r = cir.solve_batch(P[sl], a.mode, names=ids[sl], case_dir=cir.RUN_ROOT / f"data_{a.mode}")
        S11[sl], S21[sl] = r["S11"], r["S21"]
        wall[sl] = r["wall_s"] / r["n"]
        failed += r["failed"]
        batches.append({"first": b0, "n": r["n"], "s_per_design": round(r["wall_s"] / r["n"], 4),
                        "load_1m": st.get("load_1m"), "heavy_em_processes": st.get("heavy_em_processes")})
        done = np.isfinite(wall)
        ok = done & np.all(np.isfinite(S21), axis=1)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, p=P[ok], f=np.asarray(cir.F_GHZ), S11=S11[ok], S21=S21[ok],
                            param_names=np.array(fam["param_names"]), substrate=np.array(substrate)[ok],
                            ids=np.array(ids)[ok], wall_s=wall[ok], family_id=a.family,
                            schema="rfsurrogates.circuit_samples/1", fidelity="circuit", mode=a.mode,
                            solver="qucsator_rf 1.0.7 via circuit-agent-tools", seed=a.seed,
                            host=json.dumps(host), failed=json.dumps(failed))
        tmp.replace(out)
        el = time.time() - t_start
        print(f"[circuit-data] {int(done.sum())}/{a.n} solved, {len(failed)} failed, {el:.0f} s "
              f"({el / done.sum():.3f} s per design)")
    el = time.time() - t_start
    summary = {"family": a.family, "mode": a.mode, "n_requested": a.n, "n_ok": int(np.sum(np.isfinite(S21).all(1))),
               "n_failed": len(failed), "wall_s_total": round(el, 1), "s_per_design": round(el / a.n, 4),
               "designs_per_hour": round(3600 * a.n / el), "batch": a.batch, "out": out.relative_to(REPO).as_posix(), "host_at_start": host,
               "batches": batches}
    rec = registry.family_dir(a.family) / "results" / f"circuit_data_{a.mode}_n{a.n}.json"
    rec.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
