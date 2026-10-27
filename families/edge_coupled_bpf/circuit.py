"""Circuit-solver tier for edge_coupled_bpf: p -> circuit-agent-tools spec of the build_case.py layout -> qucsator.

    python circuit.py --id nominal [--mode physical|sblock|sblock_mlin|prior_blocks] [--sim qucsator]
    python circuit.py --p w1 w2 g1 g2 l1 l2 eps_r h --mode sblock

The circuit is the layout that build_case.py meshes, element by element (x along the filter, rows 0-4 along y):

    P1 -- step W50->w1 -- section 1 (rows 0-1, w1 g1 l1) -- step w1->w2 (row 1) -- section 2 (rows 1-2, w2 g2 l2)
       -- section 3 (rows 2-3, w2 g2 l2) -- step w2->w1 (row 3) -- section 4 (rows 3-4, w1 g1 l1) -- step w1->W50 -- P2

Each section is a symmetric coupled pair with line 1 on the lower row and line 2 on the upper row, both running in
+x. Eight strip ends are open (row 0 at l1; row 1 at 0 and l1 + l2; row 2 at l1 and l1 + 2 l2; row 3 at l1 + l2 and
2 l1 + 2 l2; row 4 at l1 + 2 l2). Reference planes: the ports sit on the W50 side of the feed steps, at x = 0 and
x = 2 l1 + 2 l2. That is where Palace puts them: its wave ports are de-embedded by the 3 mm feed (Offset), so the
W50 -> w1 step belongs to the device (README, "Ports"). Copper is a zero-thickness perfect conductor (sigma 1e30,
t = 0) as in Palace; the substrate loss tangent is prior.tand_for(eps_r). Ports are 50 ohm; Palace refers its
S-parameters to the W50 feed mode, which is 50 ohm by the static Hammerstad-Jensen formula.

Modes (what stands for each coupled section, and what else is in the circuit):
  physical      qucsator MCOUPLED (Kirschning-Jansen) + 8 MOPEN (Kirschning-Jansen-Koster) + 4 MSTEP
  sblock        the same circuit with each MCOUPLED replaced by a 4-port Touchstone block computed from the prior's
                coupled-line physics (prior.coupled_microstrip, Kobayashi dispersion, filling-factor loss),
                physical length l. Later the blocks can come from Palace.
  sblock_mlin   prior blocks (length l) + a microstrip line of the Hammerstad open-end extension at each open end,
                no steps: S-blocks and MLIN only, so qucsator, circuitRF and scikit-rf can all run it (cross-check)
  prior_blocks  prior blocks of length l + open-end extension, ideal open ends, no steps: the prior itself, built
                as a circuit (it must reproduce prior.s_prior; a check of the S-block path)

The simulators come from circuit-agent-tools (a sibling checkout, or CIRCUIT_AGENT_TOOLS = its folder). Runs go to
runs/circuit/ (not in git). Nothing here starts an EM solver.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
_CAT = Path(os.environ.get("CIRCUIT_AGENT_TOOLS") or REPO.parent / "circuit-agent-tools")
if _CAT.exists() and str(_CAT) not in sys.path:
    sys.path.insert(0, str(_CAT))
import prior  # noqa: E402
from build_case import L_FEED, layout  # noqa: E402

MODES = ("physical", "sblock", "sblock_mlin", "prior_blocks")
F_GHZ = np.round(np.arange(3.0, 7.0 + 1e-9, 0.02), 4)        # the family grid (families.json f_grid_GHz)
SIGMA_PEC = 1e30                                              # S/m: copper as a perfect conductor, as in Palace
RUN_ROOT = HERE / "runs" / "circuit"
SUB = "rfs"                                                   # substrate name inside the spec


# ----------------------------------------------------------------------------- prior coupled section as a 4-port

def _line_s(zc, theta, z0):
    """(M,2,2) S-matrix of a line with impedance zc and complex electrical length theta = (beta - j alpha) l."""
    a, b, c = np.cos(theta), 1j * zc * np.sin(theta), 1j * np.sin(theta) / zc
    den = 2 * a + b / z0 + c * z0
    s = np.empty((np.size(theta), 2, 2), complex)
    s[:, 0, 0] = s[:, 1, 1] = (b / z0 - c * z0) / den
    s[:, 0, 1] = s[:, 1, 0] = 2 / den
    return s


def section_s4(f_ghz, w, s, l_mm, eps_r, h, tand, dispersion=True, z0=50.0):
    """(M,4,4) S of a symmetric coupled microstrip pair with the prior's physics, in the spec node order of
    circuit-agent-tools' mcline: 1 start and 2 end of line 1, 3 start and 4 end of line 2 (3 beside 1).

    Even and odd modes are independent lines (Kirschning-Jansen static values, Kobayashi dispersion with the air-line
    impedance held fixed, dielectric loss by the filling factor: prior.s_prior does the same per section). With equal
    port impedances the 4-port is S_AA = S_BB = (S_even + S_odd) / 2 and S_AB = S_BA = (S_even - S_odd) / 2.
    """
    f = np.atleast_1d(np.asarray(f_ghz, float)) * 1e9
    z0e, z0o, ee, eo = prior.coupled_microstrip(w, s, h, eps_r)
    if dispersion:
        ee_f, eo_f = prior.kobayashi_eeff(f, ee, eps_r, w, h), prior.kobayashi_eeff(f, eo, eps_r, w, h)
        z0e_f, z0o_f = z0e * np.sqrt(ee / ee_f), z0o * np.sqrt(eo / eo_f)
    else:
        ee_f, eo_f, z0e_f, z0o_f = ee, eo, z0e, z0o
    se = _line_s(z0e_f, prior._mode_theta(f, l_mm, ee_f, eps_r, tand), z0)
    so = _line_s(z0o_f, prior._mode_theta(f, l_mm, eo_f, eps_r, tand), z0)
    out = np.empty((f.size, 4, 4), complex)
    out[:, :2, :2] = out[:, 2:, 2:] = (se + so) / 2
    out[:, :2, 2:] = out[:, 2:, :2] = (se - so) / 2
    return out


def write_block(path, f_ghz, S):
    from circuit_tools.engine import touchstone_text

    Path(path).write_text(touchstone_text(np.asarray(f_ghz) * 1e9, S, 50.0, "edge_coupled_bpf prior section"))
    return Path(path)


# ----------------------------------------------------------------------------- p -> spec

def spec(p, mode="physical", name="ecbpf", f_ghz=F_GHZ, block_dir=None, tand=None):
    """circuit-agent-tools spec of the layout of p. Modes with S-blocks write their Touchstone files to block_dir."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}; known {MODES}")
    w1, w2, g1, g2, l1, l2, eps_r, h = prior.unpack(p)
    tand = prior.tand_for(eps_r) if tand is None else tand
    _, lay = layout(p, L_FEED)
    w50 = lay["w50"]
    mm = 1e-3
    comps = [{"type": "port", "name": "P1", "nodes": ["p1", "0"], "num": 1},
             {"type": "port", "name": "P2", "nodes": ["p2", "0"], "num": 2}]
    # sections in layout order: (w, gap, length, nodes [line 1 start, line 1 end, line 2 start, line 2 end])
    secs = [(w1, g1, l1, ["a0", "o0R", "o1L", "r1a"]),
            (w2, g2, l2, ["r1b", "o1R", "o2L", "r2m"]),
            (w2, g2, l2, ["r2m", "o2R", "o3L", "r3a"]),
            (w1, g1, l1, ["r3b", "o3R", "o4L", "a4"])]
    opens = {"o0R": w1, "o1L": w1, "o1R": w2, "o2L": w2, "o2R": w2, "o3L": w2, "o3R": w1, "o4L": w1}
    steps = [("ST0", "p1", "a0", w50, w1), ("ST1", "r1a", "r1b", w1, w2), ("ST2", "r3a", "r3b", w2, w1),
             ("ST3", "a4", "p2", w1, w50)]
    if mode in ("physical", "sblock"):
        comps += [{"type": "mstep", "name": n, "nodes": [a, b], "w1": wa * mm, "w2": wb * mm, "substrate": SUB}
                  for n, a, b, wa, wb in steps]
    else:      # no step models: the step nodes are joined (same node name)
        alias = {"a0": "p1", "r1b": "r1a", "r3b": "r3a", "a4": "p2"}
        secs = [(w, g, l, [alias.get(n, n) for n in nd]) for w, g, l, nd in secs]
    if mode != "physical":
        block_dir = Path(block_dir or RUN_ROOT / name / "blocks")
        block_dir.mkdir(parents=True, exist_ok=True)
    for k, (w, g, l, nd) in enumerate(secs, 1):
        if mode == "physical":
            comps.append({"type": "mcline", "name": f"CL{k}", "nodes": nd, "w": w * mm, "s": g * mm, "l": l * mm,
                          "substrate": SUB})
            continue
        l_blk = l + (prior.open_end_extension(w, h, eps_r) if mode == "prior_blocks" else 0.0)
        fn = write_block(block_dir / f"{name}_sec{k}.s4p", f_ghz, section_s4(f_ghz, w, g, l_blk, eps_r, h, tand))
        comps.append({"type": "sblock", "name": f"SB{k}", "nodes": nd, "file": str(fn)})
    if mode in ("physical", "sblock"):
        comps += [{"type": "mopen", "name": f"OE_{n}", "nodes": [n], "w": w * mm, "substrate": SUB}
                  for n, w in opens.items()]
    elif mode == "sblock_mlin":
        comps += [{"type": "mline", "name": f"EXT_{n}", "nodes": [n, n + "_end"], "w": w * mm,
                   "l": prior.open_end_extension(w, h, eps_r) * mm, "substrate": SUB} for n, w in opens.items()]
    return {"name": name, "z0": 50,
            "substrates": {SUB: {"er": eps_r, "h": h * mm, "t": 0.0, "tand": tand, "sigma": SIGMA_PEC}},
            "components": comps,
            "analyses": [{"type": "sparam", "start": float(f_ghz[0]) * 1e9, "stop": float(f_ghz[-1]) * 1e9,
                          "points": int(len(f_ghz))}]}


# ----------------------------------------------------------------------------- solve

def _s11_s21(res, f_ghz):
    f = np.asarray(res["f"]) / 1e9
    if len(f) != len(f_ghz) or not np.allclose(f, f_ghz, atol=1e-9):
        raise RuntimeError("simulator frequencies differ from the requested grid")
    return res["s"][:, 0, 0], res["s"][:, 1, 0]


def solve(p, mode="physical", sim="qucsator", name="ecbpf", case_dir=None, f_ghz=F_GHZ, check_host=True):
    """One circuit solve through circuit-agent-tools. Returns {f, S11, S21, wall_s (end to end), sim_wall_s}."""
    from circuit_tools import engine

    case_dir = Path(case_dir or RUN_ROOT / name / mode)
    sp = spec(p, mode, name, f_ghz, block_dir=case_dir / "blocks")
    t0 = time.time()
    r = engine.simulate(sp, sim, case_dir, do_record=False, check_host=check_host)
    if not r.get("ok"):
        raise RuntimeError(f"{sim} {mode} {name}: {r.get('reason')}")
    s11, s21 = _s11_s21(r["result"], f_ghz)
    return {"f": np.asarray(f_ghz), "S11": s11, "S21": s21, "wall_s": round(time.time() - t0, 4),
            "sim_wall_s": (r.get("run") or {}).get("wall_s"), "case_dir": str(r["case_dir"])}


def solve_batch(P, mode="physical", names=None, case_dir=None, f_ghz=F_GHZ, timeout_s=60):
    """Many qucsator solves in one simulator-host call (one WSL start for the whole batch, one netlist at a time,
    nice 10). Returns S11, S21 (N, M) complex, wall_s of the batch, and the per-design failures."""
    from circuit_tools import run as runmod
    from circuit_tools.spec import normalize
    from circuit_tools.translate import Job
    from circuit_tools.translate.qucs import QUCSATOR, read_dataset

    names = list(names or [f"c{i:05d}" for i in range(len(P))])
    case_dir = Path(case_dir or RUN_ROOT / f"batch_{mode}")
    case_dir.mkdir(parents=True, exist_ok=True)
    job = Job("qucsator")
    for nm, p in zip(names, P):
        s = normalize(spec(p, mode, nm, f_ghz, block_dir=case_dir))
        one = QUCSATOR.build(s, s["analyses"][0])
        job.files[f"{nm}.net"] = one.files["case.net"]
        job.copies.update(one.copies)
        job.commands.append(f"{{exe}} -i {nm}.net -o {nm}.dat > {nm}.log 2>&1")
    t0 = time.time()
    r = runmod.run_job(job, case_dir, timeout_s=timeout_s)
    wall = time.time() - t0
    S11 = np.full((len(names), len(f_ghz)), np.nan + 0j)
    S21 = S11.copy()
    failed = []
    for i, nm in enumerate(names):
        try:
            ds = read_dataset(case_dir / f"{nm}.dat")
            S11[i], S21[i] = ds["dep"]["S[1,1]"], ds["dep"]["S[2,1]"]
            if not np.all(np.isfinite(S21[i])):
                raise ValueError("non-finite S")
        except Exception as e:  # noqa: BLE001  a failed design is reported, not fatal
            failed.append({"name": nm, "error": f"{type(e).__name__}: {e}"})
    return {"f": np.asarray(f_ghz), "S11": S11, "S21": S21, "wall_s": round(wall, 3), "n": len(names),
            "failed": failed, "exit_code": r["exit_code"], "case_dir": str(case_dir)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--id", help="sample id from samples.json (e.g. nominal, s0001)")
    ap.add_argument("--p", nargs=8, type=float, help="w1 w2 g1 g2 l1 l2 eps_r h (mm)")
    ap.add_argument("--mode", default="physical", choices=MODES)
    ap.add_argument("--sim", default="qucsator", help="qucsator (all modes); circuitrf, scikit-rf (sblock_mlin, "
                                                      "prior_blocks)")
    a = ap.parse_args()
    if a.id:
        p = {x["id"]: x for x in json.loads((HERE / "samples.json").read_text())["samples"]}[a.id]["p"]
    elif a.p:
        p = a.p
    else:
        ap.error("give --id or --p")
    r = solve(p, a.mode, a.sim, name=a.id or "case")
    f0, bw, il = prior.passband_metrics(r["f"], r["S21"])
    print(json.dumps({"mode": a.mode, "sim": a.sim, "f0_GHz": f0, "BW_GHz": bw, "IL_dB": il, "wall_s": r["wall_s"],
                      "case_dir": r["case_dir"]}, indent=1))


if __name__ == "__main__":
    main()
