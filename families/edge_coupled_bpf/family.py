"""edge_coupled_bpf: the functions that registry.load_family() exposes for this family.

Box and validity rule from samples.json, clearance rule from design.py, prior and passband metrics from prior.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import design  # noqa: E402
import prior as _prior  # noqa: E402

_S = json.loads((HERE / "samples.json").read_text())
_W50 = design.w50_for(_S["centre"][6], _S["centre"][7])


def box() -> dict:
    return {k: (float(v[0]), float(v[1])) for k, v in _S["box"].items()}


def validity() -> dict:
    return {"rule": _S["validity"]["rule"], "clearance": _S["validity"]["clearance"]}


def clearance(p) -> float:
    return design.clearance(p, _W50)


def is_valid(p) -> bool:
    return design.is_valid(p, design.C_MIN, _W50)


def prior(p, f):
    return _prior.s_prior(p, f)


def passband_metrics(f, s21):
    return _prior.passband_metrics(f, s21)


# ---- contributor hooks (tools/contribute.py). build_case and collect import gmsh and palace-agent-tools, so they are
# imported only when a hook that needs them runs; the checker does not need them.

PORT_PALACE = "wave port on the x face, mode 1, de-embedded 3 mm to the start of the coupled section"
BUILD_ARGS = {"coarse": "prod_build_args.txt", "medium": "holdout_build_args.txt"}


def sample_designs(n: int, seed: int) -> list[list[float]]:
    """n valid designs from the box: Latin hypercube (design.sample_box), invalid ones dropped. No box-corner
    samples: corners are the same for every seed, so they would repeat designs already in the data index."""
    b = {k: [lo, hi] for k, (lo, hi) in box().items()}
    out, m = [], n
    while len(out) < n and m <= 50 * n:
        m *= 2
        arr, _ = design.sample_box(b, m, seed, edge_frac=0.0)
        out = [list(map(float, p)) for p in arr if is_valid(p)]
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} valid designs in {m} draws")
    return out[:n]


def build_args(fidelity: str = "coarse") -> dict:
    """The agreed Palace build settings of a fidelity tag (prod_build_args.txt, holdout_build_args.txt)."""
    if fidelity not in BUILD_ARGS:
        raise ValueError(f"Palace builds are defined for {sorted(BUILD_ARGS)}, not '{fidelity}'")
    words = (HERE / BUILD_ARGS[fidelity]).read_text().split()
    kv = dict(zip(words[::2], words[1::2]))
    return {"scale": float(kv["--scale"]), "tol": float(kv["--tol"]), "sweep": kv.get("--sweep", "full")}


def build(name: str, p, out_root, fidelity: str = "coarse") -> dict:
    """Mesh + Palace config in out_root/name. Raises registry.GateReject when the geometry gate rejects the layout."""
    import build_case
    import registry
    a = build_args(fidelity)
    smp = build_case.SAMPLES_NARROW if a["sweep"] == "narrow" else build_case.SAMPLES_FULL
    try:
        return build_case.build(name, list(p), scale=a["scale"], tol=a["tol"], samples=smp, out_root=Path(out_root))
    except build_case.GeometryError as e:
        raise registry.GateReject(str(e)) from e


def collect(case_dir, out_npz, ranks: int | None = None) -> dict:
    """Palace output in case_dir -> npz with p, f, S11, S21 and the solver statistics."""
    import collect as _collect
    case_dir = Path(case_dir)
    return _collect.collect(case_dir.name, ranks=ranks, run_dir=case_dir, out_npz=Path(out_npz))


def export_geometry(p, out_dir, name: str) -> list[Path]:
    """Geometry for another solver (HFSS, CST, openEMS, ...): <name>.json (stack, copper, ports) and <name>.dxf."""
    import build_case
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rects, lay = build_case.layout(list(p))
    w1, w2, g1, g2, l1, l2, eps_r, h = map(float, p)
    feeds = lay["feeds"]
    spec = {
        "schema": "rfsurrogates.geometry/1", "family_id": "edge_coupled_bpf", "name": name, "units": "mm",
        "p": dict(zip(_prior.PARAM_NAMES, map(float, p))),
        "stack": {"ground": "PEC plane at z = 0", "substrate": {"eps_r": eps_r, "tand": build_case.TAND_DEFAULT,
                                                                 "z": [0.0, h]},
                  "copper": f"zero-thickness PEC sheet at z = {h} mm (all rectangles below, as one layer)"},
        "copper_rectangles": [{"x0": r[0], "y0": r[1], "dx": r[2], "dy": r[3]} for r in rects],
        "ports": [{"n": 1, "feed": {"x0": feeds[0][0], "y0": feeds[0][1], "dx": feeds[0][2], "dy": feeds[0][3]},
                   "excitation_face_x": feeds[0][0], "reference_plane_x": 0.0},
                  {"n": 2, "feed": {"x0": feeds[1][0], "y0": feeds[1][1], "dx": feeds[1][2], "dy": feeds[1][3]},
                   "excitation_face_x": feeds[1][0] + feeds[1][2], "reference_plane_x": lay["lx"]}],
        "port_rule": (f"50 ohm ports at the outer feed ends (feed width W50 = {lay['w50']:.4f} mm, length "
                      f"{lay['l_feed']} mm); de-embed both ports by the feed length to the reference planes "
                      "x = 0 and x = lx (the start and end of the coupled section)"),
        "frequency_GHz": {"start": 3.0, "stop": 7.0, "step_max": 0.02},
        "deliver": f"a two-port Touchstone file {name}.s2p (any format: RI, MA or DB; any grid that covers 3-7 GHz)",
    }
    js = out_dir / f"{name}.json"
    js.write_text(json.dumps(spec, indent=1))
    dxf = out_dir / f"{name}.dxf"
    dxf.write_text(_dxf_r12(rects))
    return [js, dxf]


def _dxf_r12(rects) -> str:
    """Minimal DXF R12 (closed POLYLINE per rectangle, layer COPPER, units mm): imports in HFSS, CST and KLayout."""
    out = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
           "0", "SECTION", "2", "ENTITIES"]
    for x0, y0, dx, dy in rects:
        out += ["0", "POLYLINE", "8", "COPPER", "66", "1", "70", "1"]
        for x, y in ((x0, y0), (x0 + dx, y0), (x0 + dx, y0 + dy), (x0, y0 + dy)):
            out += ["0", "VERTEX", "8", "COPPER", "10", f"{x:.6f}", "20", f"{y:.6f}", "30", "0.0"]
        out += ["0", "SEQEND", "8", "COPPER"]
    out += ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(out) + "\n"
