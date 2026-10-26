"""One edge-coupled filter design p -> Gmsh mesh + Palace 0.16 config (two wave ports, adaptive driven sweep).

Stack (same as pixel-filter-palace / em-benchmark): copper as a zero-thickness PEC sheet at z = h, substrate block
with loss tangent, PEC ground at z = 0, first-order absorbing side walls and top, wave ports on the two x faces each
de-embedded by the feed length (Offset) to the start of the coupled section.

    python build_case.py --id nominal [--scale 1.8] [--tol 1e-2] [--sweep narrow] [--order 2] [--fields 5.0]
    python build_case.py --p w1 w2 g1 g2 l1 l2 eps_r h --name test

Layout (x along the filter, strips in rows along y; sections 1, 4 use width w1, sections 2, 3 use width w2):
    row 0  input strip  x in [0, l1], width w1            + 50 ohm feed (W50) x in [-l_feed, 0]
    row 1  resonator 1  x in [0, l1] width w1, then x in [l1, l1 + l2] width w2 (step centred on the row axis)
    row 2  resonator 2  x in [l1, l1 + 2 l2], width w2
    row 3  resonator 3  x in [l1 + l2, l1 + 2 l2] width w2, then x in [l1 + 2 l2, 2 l1 + 2 l2] width w1
    row 4  output strip x in [l1 + 2 l2, 2 l1 + 2 l2], width w1   + 50 ohm feed to Lx + l_feed
    row axes: y1 = y0 + w1 + g1, y2 = y1 + w2 + g2, y3 = y2 + w2 + g2, y4 = y3 + w1 + g1 (edge-to-edge gaps g1, g2).
Sections: 1 = rows 0-1 (l1, g1, w1), 2 = rows 1-2 (l2, g2, w2), 3 = rows 2-3 (l2, g2, w2), 4 = rows 3-4 (l1, g1, w1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gmsh
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("PALACE_AGENT_TOOLS"):              # a local checkout instead of `pip install palace-agent-tools`
    sys.path.insert(0, os.environ["PALACE_AGENT_TOOLS"])
import prior  # noqa: E402
from prior import PARAM_NAMES  # noqa: E402

TAND_DEFAULT = 0.0037          # RO4350B
L_FEED, MARGIN, H_AIR = 3.0, 6.0, 8.0
# frequency samples (GHz): fine over the band the +-15 % lengths can move the passband to, coarse elsewhere
SAMPLES_FULL = [(2.0, 3.4, 0.1), (3.5, 6.5, 0.02), (6.6, 8.0, 0.1)]
SAMPLES_NARROW = [(3.0, 3.4, 0.1), (3.5, 6.5, 0.02), (6.6, 7.0, 0.1)]
SAMPLES = SAMPLES_FULL


def _w50(eps_r, h, z0=50.0):
    from scipy.optimize import brentq
    return brentq(lambda ww: prior.microstrip(ww, h, eps_r)[0] - z0, 0.05 * h, 20 * h)


def layout(p, l_feed=L_FEED):
    """Rectangles (x0, y0, dx, dy) of the copper (strips first, feeds last), plus the layout record."""
    w1, w2, g1, g2, l1, l2, eps_r, h = prior.unpack(p)
    w50 = float(_w50(eps_r, h))
    y = [0.0]
    for dy in (w1 + g1, w2 + g2, w2 + g2, w1 + g1):
        y.append(y[-1] + dy)                      # strip axes
    lx = 2 * l1 + 2 * l2
    strips = [(0.0, y[0] - w1 / 2, l1, w1),                                   # row 0: input strip
              (0.0, y[1] - w1 / 2, l1, w1), (l1, y[1] - w2 / 2, l2, w2),      # row 1: resonator 1 (w1 part, w2 part)
              (l1, y[2] - w2 / 2, 2 * l2, w2),                                # row 2: resonator 2
              (l1 + l2, y[3] - w2 / 2, l2, w2), (l1 + 2 * l2, y[3] - w1 / 2, l1, w1),   # row 3: resonator 3
              (l1 + 2 * l2, y[4] - w1 / 2, l1, w1)]                            # row 4: output strip
    feeds = [(-l_feed, y[0] - w50 / 2, l_feed, w50), (lx, y[4] - w50 / 2, l_feed, w50)]
    sections = [{"n": 1, "rows": [0, 1], "x": [0.0, l1], "w": w1, "gap": g1, "length": l1,
                 "box": [0.0, y[0] - w1 / 2, l1, y[1] + w1 / 2]},
                {"n": 2, "rows": [1, 2], "x": [l1, l1 + l2], "w": w2, "gap": g2, "length": l2,
                 "box": [l1, y[1] - w2 / 2, l1 + l2, y[2] + w2 / 2]},
                {"n": 3, "rows": [2, 3], "x": [l1 + l2, l1 + 2 * l2], "w": w2, "gap": g2, "length": l2,
                 "box": [l1 + l2, y[2] - w2 / 2, l1 + 2 * l2, y[3] + w2 / 2]},
                {"n": 4, "rows": [3, 4], "x": [l1 + 2 * l2, lx], "w": w1, "gap": g1, "length": l1,
                 "box": [l1 + 2 * l2, y[3] - w1 / 2, lx, y[4] + w1 / 2]}]
    rec = {"w50": w50, "row_y": y, "lx": lx, "ly": y[4] + w1 / 2 - (y[0] - w1 / 2), "strips": strips, "feeds": feeds,
           "sections": sections, "l_feed": l_feed,
           "steps": [{"row": 1, "x": l1, "from": w1, "to": w2}, {"row": 3, "x": l1 + 2 * l2, "from": w2, "to": w1}]}
    return strips + feeds, rec


class GeometryError(RuntimeError):
    """The layout does not have the expected number of separate conductors (fused or missing elements)."""


def copper_components(msh_path, phys=10):
    """Number of connected components of the copper-sheet surface mesh (physical group `phys`) in an MSH 2.2 file."""
    tris = []
    with open(msh_path) as fh:
        lines = fh.read().split("\n")
    j = lines.index("$Elements"); m = int(lines[j + 1])
    for ln in lines[j + 2:j + 2 + m]:
        a = ln.split()
        if int(a[1]) == 2 and int(a[3]) == phys:
            ntag = int(a[2]); tris.append([int(x) for x in a[3 + ntag:3 + ntag + 3]])
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for t in tris:
        a = find(t[0])
        for n in t[1:]:
            b = find(n)
            if a != b:
                parent[b] = a
    return len({find(n) for t in tris for n in t})


def _vrect(occ, x, y0, y1, z0, z1):
    pts = [occ.addPoint(x, yy, zz) for yy, zz in ((y0, z0), (y1, z0), (y1, z1), (y0, z1))]
    return occ.addPlaneSurface([occ.addCurveLoop([occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)])])


def build(name, p, scale=1.8, tand=TAND_DEFAULT, margin=MARGIN, h_air=H_AIR, l_feed=L_FEED, samples=SAMPLES,
          tol=1e-3, excite=(1,), fields_at=None, out_root=HERE / "runs", threads=4, gap_div=None, dist_max=1.5,
          order=2, expect_conductors=5):
    w1, w2, g1, g2, l1, l2, eps_r, h = prior.unpack(p)
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    rects, lay = layout(p, l_feed)
    x0, x1 = -l_feed, lay["lx"] + l_feed
    y0, y1 = -w1 / 2 - margin, lay["row_y"][4] + w1 / 2 + margin
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)
    occ = gmsh.model.occ
    faces = [(2, occ.addRectangle(rx, ry, h, dx, dy)) for rx, ry, dx, dy in rects]
    cu, _ = occ.fuse(faces[:1], faces[1:])
    # geometry gate 1 (CAD): OCC fuse merges touching rectangles into one face, so the number of fused faces is the
    # number of separate conductors. An edge-coupled filter has exactly 5 (input strip + feed, 3 resonators,
    # output strip + feed); fewer means two elements overlap and the device is shorted (see design.clearance).
    n_cad = len(cu)
    if expect_conductors and n_cad != expect_conductors:
        gmsh.finalize()
        rec = {"name": name, "p": [w1, w2, g1, g2, l1, l2, eps_r, h], "param_names": list(PARAM_NAMES),
               "status": "rejected", "reason": f"{n_cad} conductors after CAD fuse, expected {expect_conductors}",
               "conductors_cad": n_cad, "conductors_expected": expect_conductors}
        (out / "build_summary.json").write_text(json.dumps(rec, indent=1))
        raise GeometryError(rec["reason"])
    sub = occ.addBox(x0, y0, 0, x1 - x0, y1 - y0, h)
    air = occ.addBox(x0, y0, h, x1 - x0, y1 - y0, h_air)
    p1, p2 = _vrect(occ, x0, y0, y1, 0, h + h_air), _vrect(occ, x1, y0, y1, 0, h + h_air)
    _, omap = occ.fragment([(3, sub), (3, air)], cu + [(2, p1), (2, p2)])
    occ.synchronize()
    nc = len(cu)
    sub_v = sorted({t for d, t in omap[0] if d == 3}); air_v = sorted({t for d, t in omap[1] if d == 3})
    cu_s = sorted({t for k in range(2, 2 + nc) for d, t in omap[k] if d == 2})
    port = {1: sorted({t for d, t in omap[2 + nc] if d == 2}), 2: sorted({t for d, t in omap[3 + nc] if d == 2})}
    taken = set(cu_s) | set(port[1]) | set(port[2])
    gnd, absorb = [], []
    for v in sub_v + air_v:
        for _, s in gmsh.model.getBoundary([(3, v)], oriented=False):
            s = abs(s)
            if s in taken or len(gmsh.model.getAdjacencies(2, s)[0]) != 1:
                continue
            bb = gmsh.model.getBoundingBox(2, s)
            (gnd if abs(bb[2]) < 1e-6 and abs(bb[5]) < 1e-6 else absorb).append(s)
    gmsh.model.addPhysicalGroup(3, sub_v, 1, name="substrate")
    gmsh.model.addPhysicalGroup(3, air_v, 2, name="air")
    gmsh.model.addPhysicalGroup(2, cu_s, 10, name="copper_sheet")
    gmsh.model.addPhysicalGroup(2, sorted(set(gnd)), 11, name="ground")
    gmsh.model.addPhysicalGroup(2, sorted(set(absorb)), 12, name="absorbing")
    gmsh.model.addPhysicalGroup(2, port[1], 21, name="port1")
    gmsh.model.addPhysicalGroup(2, port[2], 22, name="port2")
    # mesh size: fine at every copper edge, growing with distance; substrate never coarser than s_sub; air up to
    # s_air. `scale` multiplies every length. gap_div forces >= gap_div elements across the tightest gap.
    edges = sorted({abs(c) for s in cu_s for _, c in gmsh.model.getBoundary([(2, s)], oriented=False)})
    fld = gmsh.model.mesh.field
    dist = fld.add("Distance"); fld.setNumbers(dist, "CurvesList", edges); fld.setNumber(dist, "Sampling", 400)
    thr = fld.add("Threshold"); fld.setNumber(thr, "InField", dist)
    s_min = 0.08 * scale
    if gap_div:
        s_min = min(s_min, min(g1, g2) / gap_div)
    s_near, s_air, s_sub = 0.8 * scale, 2.5 * scale, 1.2 * scale
    for k, v in (("SizeMin", s_min), ("SizeMax", s_near), ("DistMin", 0.05), ("DistMax", dist_max)):
        fld.setNumber(thr, k, v)
    box = fld.add("Box"); fld.setNumber(box, "VIn", s_sub); fld.setNumber(box, "VOut", s_air)
    for k, v in zip(("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax"), (x0, x1, y0, y1, 0, h)):
        fld.setNumber(box, k, v)
    mn = fld.add("Min"); fld.setNumbers(mn, "FieldsList", [thr, box]); fld.setAsBackgroundMesh(mn)
    for k, v in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0),
                 ("Mesh.MeshSizeMax", s_air), ("Mesh.Algorithm", 6), ("Mesh.Algorithm3D", 10),
                 ("General.NumThreads", threads), ("Mesh.MshFileVersion", 2.2), ("Mesh.SaveAll", 0)):
        gmsh.option.setNumber(k, v)
    gmsh.model.mesh.generate(3)
    tets = int(len(gmsh.model.mesh.getElementsByType(4)[0]))
    gmsh.model.mesh.createEdges(); gmsh.model.mesh.createFaces()
    n_edges = len(gmsh.model.mesh.getAllEdges()[0]); n_faces = len(gmsh.model.mesh.getAllFaces(3)[0])
    gmsh.write(str(out / f"{name}.msh"))
    gmsh.finalize()
    # geometry gate 2 (mesh): connected components of the copper-sheet surface mesh must match the CAD count
    n_mesh = copper_components(out / f"{name}.msh", 10)
    if expect_conductors and n_mesh != expect_conductors:
        rec = {"name": name, "p": [w1, w2, g1, g2, l1, l2, eps_r, h], "param_names": list(PARAM_NAMES),
               "status": "rejected", "reason": f"{n_mesh} copper components in the mesh, expected {expect_conductors}",
               "conductors_cad": n_cad, "conductors_mesh": n_mesh, "conductors_expected": expect_conductors}
        (out / "build_summary.json").write_text(json.dumps(rec, indent=1))
        (out / f"{name}.msh").unlink(missing_ok=True)
        raise GeometryError(rec["reason"])
    try:
        from palace_tools.estimate import estimate, nedelec_unknowns
        unknowns = int(nedelec_unknowns(n_edges, n_faces, tets, order))
        est = estimate(unknowns=unknowns, problem="Driven", order=order, solver_mode="direct", ranks=8)
    except Exception as e:  # noqa: BLE001
        unknowns, est = int(6.3 * tets), {"error": str(e)}
    smp = [{"Type": "Linear", "MinFreq": a, "MaxFreq": b, "FreqStep": c, "SaveStep": 0} for a, b, c in samples]
    if fields_at:          # fields-only run: one frequency, ParaView output, no sweep
        smp = [{"Type": "Point", "Freq": [float(fields_at)], "SaveStep": 1}]
    wave = [{"Index": k, "Attributes": [20 + k], "Mode": 1, "Offset": l_feed, **({"Excitation": k} if k in excite else {})}
            for k in (1, 2)]
    cfg = {
        "Problem": {"Type": "Driven", "Verbose": 2, "Output": "output"},
        "Model": {"Mesh": f"{name}.msh", "L0": 1e-3},
        "Domains": {"Materials": [{"Attributes": [1], "Permittivity": eps_r, "LossTan": tand}, {"Attributes": [2], "Permittivity": 1.0}]},
        "Boundaries": {"PEC": {"Attributes": [10, 11]}, "Absorbing": {"Attributes": [12], "Order": 1}, "WavePort": wave},
        "Solver": {"Order": order, "Device": "CPU", "Driven": {"Samples": smp, "AdaptiveTol": tol},
                   "Linear": {"Type": "MUMPS", "KSPType": "GMRES", "Tol": 1e-8, "MaxIts": 200,
                              "EstimatorMG": True, "EstimatorTol": 1e-4}},   # palace-tools SOL003: same S, less time
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    summary = {"name": name, "p": [w1, w2, g1, g2, l1, l2, eps_r, h], "param_names": list(PARAM_NAMES), "status": "ok",
               "conductors_cad": n_cad, "conductors_mesh": n_mesh, "conductors_expected": expect_conductors,
               "tand": tand, "scale": scale, "gap_div": gap_div, "dist_max": dist_max, "order": order, "adaptive_tol": tol,
               "mesh_sizes_mm": {"min": s_min, "near_copper_max": s_near, "substrate": s_sub, "air": s_air},
               "tets": tets, "edges": n_edges, "faces": n_faces, "unknowns_order2": unknowns, "estimate_8ranks": est,
               "domain_mm": {"x": [x0, x1], "y": [y0, y1], "z": [0, h + h_air]}, "margin_mm": margin, "h_air_mm": h_air,
               "layout": lay, "samples_GHz": samples, "fields_at_GHz": fields_at, "excite": list(excite),
               "msh_MB": round((out / f"{name}.msh").stat().st_size / 1e6, 2)}
    (out / "build_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="sample id from samples.json (e.g. nominal, s0001, h0003)")
    ap.add_argument("--p", nargs=8, type=float, help="w1 w2 g1 g2 l1 l2 eps_r h (mm)")
    ap.add_argument("--name", help="run folder name (default: the id)")
    ap.add_argument("--scale", type=float, default=1.8)
    ap.add_argument("--fields", type=float, default=None, help="fields-only run: solve and save fields at this one frequency (GHz)")
    ap.add_argument("--tand", type=float, default=TAND_DEFAULT)
    ap.add_argument("--gap-div", type=float, default=None, help="force >= this many elements across the tightest gap")
    ap.add_argument("--dist-max", type=float, default=1.5)
    ap.add_argument("--tol", type=float, default=1e-3, help="adaptive sweep tolerance")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--sweep", default="full", choices=["full", "narrow"], help="narrow = 3-7 GHz only")
    ap.add_argument("--expect-conductors", type=int, default=5, help="geometry gate: separate copper conductors expected (0 = off)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.id:
        s = {x["id"]: x for x in json.loads((HERE / "samples.json").read_text())["samples"]}[a.id]
        p = s["p"]
    else:
        p = a.p
    name = a.name or (a.id if a.id else "case")
    smp = SAMPLES_FULL if a.sweep == "full" else SAMPLES_NARROW
    try:
        summ = build(name, p, scale=a.scale, tand=a.tand, fields_at=a.fields, gap_div=a.gap_div, dist_max=a.dist_max,
                     tol=a.tol, order=a.order, samples=smp, expect_conductors=a.expect_conductors)
    except GeometryError as e:
        print(f"{name}: REJECTED by the geometry gate ({e}); no config.json written, Palace will not run")
        sys.exit(3)
    if a.quiet:
        print(f"{name}: {summ['tets']} tets, {summ['unknowns_order2']} unknowns, {summ['conductors_mesh']} conductors")
    else:
        print(json.dumps({k: v for k, v in summ.items() if k != "layout"}, indent=1))


if __name__ == "__main__":
    main()
