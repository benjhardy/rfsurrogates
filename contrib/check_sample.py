"""Contribution checker: one simulated sample -> verdict + manifest row.

    python contrib/check_sample.py <file.npz|file.s2p> --family edge_coupled_bpf [--p w1 w2 g1 g2 l1 l2 eps_r h]
                                   [--solver palace --solver-version 0.16 --fidelity coarse --contributor <id>]
                                   [--licence CC-BY-4.0] [--mesh-elements N --mesh-unknowns N --order 2]
                                   [--port "wave port, de-embedded 3 mm"] [--out row.json] [--write-npz out.npz]
                                   [--allow-out-of-box] [--json]

What it does, in order (docs/DATA_CONTRACT.md has the record this produces):
  1. reads the sample (contract npz, a collect.py npz, or a Touchstone .s2p) and validates schema and units;
  2. checks that p lies inside the family box and passes the family validity rule (families.json + family.py);
  3. resamples S to the family frequency grid when the grids differ (cubic spline on re/im, as collect.py does);
  4. physics sanity: finite values, |S11| <= 1, |S21| <= 1, |S11|^2 + |S21|^2 <= 1 + tol, reciprocity when S12 is
     present, symmetry S22 ~ S11 when S22 is present (review flag only: this family is symmetric by design);
  5. compares with the family's analytic prior: passband |S21| difference and f0 difference; large
     differences flag the sample "needs review", they do not reject it (the prior is the thing the model corrects);
  6. prints a one-line verdict and writes a manifest row (JSON) for the pull request.

Exit codes: 0 PASS, 2 NEEDS REVIEW, 1 FAIL (also 1 on a read error).
Repo modules used: registry.py and the family's family.py, prior.py, design.py (numpy + scipy); nothing here
imports gmsh, torch or Palace.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import registry  # noqa: E402

CHECKER_VERSION = "0.2.0"
SCHEMA = "rfsurrogates.sample/1"

# thresholds (documented in docs/DATA_CONTRACT.md; change them there and here together)
PASSIVITY_TOL = 1e-3        # |S11|^2 + |S21|^2 <= 1 + tol  (collect.py uses the same 1.001 for the run DB check)
MAG_TOL = 1e-3              # |Sij| <= 1 + tol
RECIP_TOL = 0.02            # max |S12 - S21| (linear); the device is reciprocal, a larger gap is a solver error
SYMM_REVIEW = 0.05          # max |S22 - S11| (linear) above which the sample is flagged for review
PRIOR_DB_REVIEW = 8.0       # mean passband | |S21| dB sample - |S21| dB prior | above which -> needs review
PRIOR_F0_REVIEW = 0.10      # relative f0 difference to the prior above which -> needs review
P_TOL_REL = 1e-6            # --p vs the p stored in an npz
BOX_TOL_REL = 1e-6          # tolerance on the box edges, relative to the box width

FIDELITY_TAGS = ("coarse", "medium", "fine", "other")
LEGACY_SCALE_TO_FIDELITY = {1.8: "coarse", 1.4: "medium", 1.0: "fine"}


# ----------------------------------------------------------------------------- family registry
def load_family(family_id: str) -> dict:
    """families.json entry + box, validity rule, prior and metrics from the family's family.py (registry.py)."""
    return registry.load_family(family_id)


# ----------------------------------------------------------------------------- readers
def to_ghz(f: np.ndarray) -> tuple[np.ndarray, str]:
    """Return (f in GHz, unit found). Heuristic as model/data.py: > 1e5 -> Hz, > 1e2 -> MHz, else GHz."""
    f = np.asarray(f, float)
    if f.max() > 1e5:
        return f / 1e9, "Hz"
    if f.max() > 1e2:
        return f / 1e3, "MHz"
    return f, "GHz"


def _scalar(z, key, default=None):
    if key not in z.files:
        return default
    v = np.asarray(z[key]).reshape(-1)
    if v.size == 0:
        return default
    v = v[0]
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode()
    if isinstance(v, np.generic):
        v = v.item()
    return v


def read_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)     # records are untrusted input (CI): no pickled objects
    keys = {k.lower(): k for k in z.files}
    missing = [k for k in ("p", "f", "s11", "s21") if k not in keys]
    if missing:
        raise ValueError(f"npz lacks required key(s) {missing}; has {sorted(z.files)}")
    rec = {"source": "npz", "p": np.asarray(z[keys["p"]], float).reshape(-1), "f": np.asarray(z[keys["f"]], float).reshape(-1),
           "S": {"S11": np.asarray(z[keys["s11"]]).reshape(-1).astype(complex), "S21": np.asarray(z[keys["s21"]]).reshape(-1).astype(complex)}}
    for opt in ("S12", "S22"):
        if opt.lower() in keys:
            rec["S"][opt] = np.asarray(z[keys[opt.lower()]]).reshape(-1).astype(complex)
    meta = {}
    for k in ("param_names", "substrate", "run", "scale", "unknowns", "tets", "ranks", "wall_s", "peak_ram_gb", "kind",
              "clearance_mm", "metrics", "note", "family_id", "solver", "solver_version", "fidelity", "order",
              "contributor", "licence", "port", "adaptive_tol"):
        if k in z.files:
            v = z[k]
            meta[k] = [str(x) for x in np.asarray(v).reshape(-1)] if k == "param_names" else _scalar(z, k)
    rec["meta"] = meta
    legacy = "scale" in meta and "run" in meta and "solver" not in meta
    rec["legacy_collect"] = legacy
    return rec


_TS_OPT = re.compile(r"^#\s*(?P<unit>[kMG]?Hz)?\s*(?P<type>[SYZGH])?\s*(?P<fmt>RI|MA|DB)?\s*(?:R\s*(?P<z0>[0-9.eE+-]+))?", re.I)


def read_touchstone(path: Path) -> dict:
    """Touchstone v1 (and the common subset of v2) two-port file -> f (GHz), S11, S21, S12, S22."""
    unit, fmt, z0, order = "GHz", "MA", 50.0, "21_12"
    nums: list[float] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            m = _TS_OPT.match(line)
            if m:
                unit = (m.group("unit") or unit)
                fmt = (m.group("fmt") or fmt).upper()
                if m.group("type") and m.group("type").upper() != "S":
                    raise ValueError(f"Touchstone type {m.group('type')} not supported; export S-parameters")
                if m.group("z0"):
                    z0 = float(m.group("z0"))
            continue
        if line.startswith("["):
            if line.lower().startswith("[two-port data order]"):
                order = line.split("]", 1)[1].strip().replace(" ", "")
            elif line.lower().startswith("[number of ports]") and int(line.split("]", 1)[1]) != 2:
                raise ValueError("only two-port Touchstone files are supported")
            continue
        nums.extend(float(x) for x in line.replace(",", " ").split())
    if not nums or len(nums) % 9:
        raise ValueError(f"Touchstone data has {len(nums)} numbers; a two-port file needs 9 per frequency")
    a = np.asarray(nums).reshape(-1, 9)
    scale = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0}[unit.lower()]
    f = a[:, 0] * scale

    def cx(x, y):
        if fmt == "RI":
            return x + 1j * y
        mag = 10 ** (x / 20) if fmt == "DB" else x
        return mag * np.exp(1j * np.deg2rad(y))

    pairs = [cx(a[:, 1 + 2 * k], a[:, 2 + 2 * k]) for k in range(4)]
    names = ("S11", "S21", "S12", "S22") if order == "21_12" else ("S11", "S12", "S21", "S22")
    S = dict(zip(names, pairs))
    idx = np.argsort(f)
    return {"source": "touchstone", "p": None, "f": f[idx], "S": {k: v[idx] for k, v in S.items()},
            "meta": {"touchstone_format": fmt, "touchstone_unit": unit, "z0_ohm": z0}, "legacy_collect": False}


def write_touchstone(path: Path, f_ghz, S11, S21, S12=None, S22=None, comment: str = ""):
    """Write a two-port Touchstone v1 file (GHz, RI, 50 ohm, full double precision). Used by the tests and offered to
    contributors who want a text container."""
    S12 = S21 if S12 is None else S12
    S22 = S11 if S22 is None else S22
    lines = [f"! {comment}" if comment else "! two-port S-parameters", "# GHz S RI R 50"]
    for k in range(len(f_ghz)):
        vals = [f_ghz[k]] + [x for s in (S11[k], S21[k], S12[k], S22[k]) for x in (s.real, s.imag)]
        lines.append(" ".join(f"{v:.17g}" for v in vals))
    Path(path).write_text("\n".join(lines) + "\n")


def read_sample(path: Path) -> dict:
    suf = path.suffix.lower()
    if suf == ".npz":
        return read_npz(path)
    if re.fullmatch(r"\.s\d+p", suf) or suf in (".ts", ".snp"):
        return read_touchstone(path)
    raise ValueError(f"unsupported input '{path.name}': give an .npz or a Touchstone .s2p")


# ----------------------------------------------------------------------------- helpers
def resample(f_raw, s_raw, f_new):
    re_, im_ = CubicSpline(f_raw, s_raw.real), CubicSpline(f_raw, s_raw.imag)
    return re_(f_new) + 1j * im_(f_new)


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-30)


def passband_mask(f, s21, drop_db=3.0):
    m = db(s21)
    return m >= m.max() - drop_db


HASH_DECIMALS_P, HASH_DECIMALS_F, HASH_DECIMALS_S = 9, 6, 9


def content_hash(p, f, S11, S21) -> str:
    """sha256 of the canonical record: p (rounded to 1e-9), f in GHz (1e-6), S11 and S21 re/im (1e-9) on the family
    grid, as little-endian float64 bytes. The rounding makes the hash the same for one record stored as npz or as a
    full-precision Touchstone file."""
    h = hashlib.sha256()
    parts = [np.round(np.asarray(p, float), HASH_DECIMALS_P), np.round(np.asarray(f, float), HASH_DECIMALS_F)]
    for s in (S11, S21):
        s = np.asarray(s, complex)
        parts += [np.round(s.real, HASH_DECIMALS_S), np.round(s.imag, HASH_DECIMALS_S)]
    for a in parts:
        h.update((a + 0.0).astype("<f8").tobytes())        # + 0.0 turns -0.0 into 0.0
    return h.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fin(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


# ----------------------------------------------------------------------------- the check
def check(path, family_id="edge_coupled_bpf", p=None, solver=None, solver_version=None, fidelity=None,
          contributor=None, licence=None, mesh_elements=None, mesh_unknowns=None, order=None, port=None,
          allow_out_of_box=False, sample_id=None) -> dict:
    """Run every check on one file. Returns the manifest row (dict) with row['verdict'] in PASS / NEEDS REVIEW / FAIL."""
    path = Path(path)
    fam = load_family(family_id)
    names = fam["param_names"]
    errors: list[str] = []
    flags: list[str] = []
    notes: list[str] = []
    row = {"schema": SCHEMA, "checker_version": CHECKER_VERSION, "checked_on": _dt.date.today().isoformat(),
           "id": sample_id or path.stem, "family_id": fam["family_id"], "source_file": path.name}
    try:
        rec = read_sample(path)
    except Exception as e:  # noqa: BLE001
        row.update({"verdict": "FAIL", "errors": [f"read: {e}"], "flags": [], "notes": []})
        return row
    meta = rec["meta"]

    # ---- p: from the argument, the file, or both (must agree)
    p_file = rec["p"]
    if p is not None:
        p = np.asarray(p, float).reshape(-1)
    if p is None and p_file is None:
        errors.append("no design vector: a Touchstone file needs --p w1 w2 g1 g2 l1 l2 eps_r h")
    elif p is None:
        p = p_file
    elif p_file is not None:
        if p_file.size != p.size or np.any(np.abs(p_file - p) > P_TOL_REL * np.maximum(1.0, np.abs(p))):
            errors.append(f"--p {p.tolist()} differs from the p stored in the file {p_file.tolist()}")
    if p is not None and p.size != len(names):
        errors.append(f"p has {p.size} entries, family {fam['family_id']} has {len(names)}: {list(names)}")
        p = None
    if "param_names" in meta and tuple(meta["param_names"]) != tuple(names):
        errors.append(f"param_names in file {meta['param_names']} != family {list(names)}")

    # ---- units and plausibility of p
    if p is not None:
        row["p"] = [float(x) for x in p]
        row["param_names"] = list(names)
        row["units"] = fam["units"]
        for k, v in zip(names, p):
            u = fam["units"][k]
            if u == "mm" and not (1e-3 < v < 100):
                errors.append(f"{k} = {v} is not a plausible length in mm (expected 0.001-100)")
            if k == "eps_r" and not (1.0 <= v <= 20):
                errors.append(f"eps_r = {v} out of range 1-20")
        if not np.all(np.isfinite(p)):
            errors.append("p has non-finite entries")

    # ---- box and validity rule
    if p is not None and np.all(np.isfinite(p)):
        out_of_box = []
        for k, v in zip(names, p):
            lo, hi = fam["box"][k]
            tol = BOX_TOL_REL * max(hi - lo, abs(lo), 1e-12)
            if v < lo - tol or v > hi + tol:
                out_of_box.append(f"{k}={v:.6g} not in [{lo:.6g}, {hi:.6g}]")
        row["in_box"] = not out_of_box
        if out_of_box:
            if allow_out_of_box:
                flags.append("out_of_box")
                notes.append("outside the family box: " + "; ".join(out_of_box))
            else:
                errors.append("outside the family box: " + "; ".join(out_of_box) + " (use --allow-out-of-box to submit for review)")
        try:
            c = float(fam["clearance"](p))
            row["clearance_mm"] = round(c, 5)
            row["valid_geometry"] = bool(fam["is_valid"](p))
            if not row["valid_geometry"]:
                errors.append(f"fails the validity rule {fam['validity']['rule']}: clearance {c:.4f} mm")
        except Exception as e:  # noqa: BLE001
            errors.append(f"validity rule could not be evaluated: {e}")

    # ---- frequency grid and S arrays
    f_raw, unit = to_ghz(rec["f"])
    if unit != "GHz":
        notes.append(f"frequency read as {unit} and converted to GHz")
    S = rec["S"]
    if f_raw.size < 4:
        errors.append(f"only {f_raw.size} frequency points")
    if np.any(np.diff(f_raw) <= 0):
        errors.append("frequency grid is not strictly increasing")
    if not (0.05 <= f_raw.min() and f_raw.max() <= 300):
        errors.append(f"frequency range {f_raw.min():.4g}-{f_raw.max():.4g} GHz is not plausible")
    for k, v in S.items():
        if v.size != f_raw.size:
            errors.append(f"{k} has {v.size} points, f has {f_raw.size}")
        if not np.all(np.isfinite(v)):
            errors.append(f"{k} contains NaN or inf")
    if errors:
        row.update({"verdict": "FAIL", "errors": errors, "flags": flags, "notes": notes})
        return row

    fg = fam["f_grid_GHz"]
    row["f_grid_GHz"] = [float(fg[0]), float(fg[-1]), float(np.round(fg[1] - fg[0], 6))]
    row["n_f"] = int(fg.size)
    same = f_raw.size == fg.size and np.allclose(f_raw, fg, atol=1e-6)
    if not same:
        if f_raw.min() > fg[0] + 1e-6 or f_raw.max() < fg[-1] - 1e-6:
            errors.append(f"grid {f_raw.min():.3f}-{f_raw.max():.3f} GHz does not cover the family grid "
                          f"{fg[0]:.3f}-{fg[-1]:.3f} GHz")
            row.update({"verdict": "FAIL", "errors": errors, "flags": flags, "notes": notes})
            return row
        S = {k: resample(f_raw, v, fg) for k, v in S.items()}
        row["resampled_from"] = {"n_f": int(f_raw.size), "f_min_GHz": float(f_raw.min()), "f_max_GHz": float(f_raw.max()),
                                 "step_median_GHz": float(np.median(np.diff(f_raw)))}
        notes.append(f"resampled from {f_raw.size} points to the family grid ({fg.size} points)")
    s11, s21 = S["S11"], S["S21"]

    # ---- physics sanity
    phys = {}
    phys["max_abs_S11"] = float(np.abs(s11).max())
    phys["max_abs_S21"] = float(np.abs(s21).max())
    phys["passivity_max"] = float((np.abs(s11) ** 2 + np.abs(s21) ** 2).max())
    if phys["max_abs_S11"] > 1 + MAG_TOL:
        errors.append(f"|S11| reaches {phys['max_abs_S11']:.4f} > 1")
    if phys["max_abs_S21"] > 1 + MAG_TOL:
        errors.append(f"|S21| reaches {phys['max_abs_S21']:.4f} > 1")
    if phys["passivity_max"] > 1 + PASSIVITY_TOL:
        errors.append(f"not passive: max(|S11|^2 + |S21|^2) = {phys['passivity_max']:.4f} > 1 + {PASSIVITY_TOL}")
    if "S12" in S:
        phys["reciprocity_max_abs"] = float(np.abs(S["S12"] - s21).max())
        if phys["reciprocity_max_abs"] > RECIP_TOL:
            errors.append(f"not reciprocal: max |S12 - S21| = {phys['reciprocity_max_abs']:.4f} > {RECIP_TOL}")
    if "S22" in S:
        phys["symmetry_max_abs"] = float(np.abs(S["S22"] - s11).max())
        if phys["symmetry_max_abs"] > SYMM_REVIEW:
            flags.append("asymmetric")
            notes.append(f"max |S22 - S11| = {phys['symmetry_max_abs']:.4f} > {SYMM_REVIEW}: the family is symmetric by design")
    row["physics"] = phys
    row["s_stored"] = list(fam["stored_s"])
    row["s_present_in_source"] = sorted(S)
    row["reciprocal_assumed"] = "S12" not in S
    row["symmetric_assumed"] = "S22" not in S

    # ---- metrics of the sample
    f0, bw, il = fam["passband_metrics"](fg, s21)
    met = {"f0_GHz": _fin(f0), "BW_GHz": _fin(bw), "IL_dB": _fin(il), "S21_max_dB": float(db(s21).max())}
    if np.isfinite(f0):
        band = (fg >= 0.95 * f0) & (fg <= 1.05 * f0)
        met["RL_dB"] = float(-db(s11[band]).max())
    row["metrics"] = met

    # ---- prior comparison (review flags only)
    try:
        p11, p21 = fam["prior"](p, fg)
        pf0, pbw, pil = fam["passband_metrics"](fg, p21)
        mask = passband_mask(fg, p21) | passband_mask(fg, s21)
        d_db = float(np.mean(np.abs(db(s21[mask]) - db(p21[mask]))))
        pr = {"prior_f0_GHz": _fin(pf0), "prior_BW_GHz": _fin(pbw), "passband_dB_mean_abs_diff": d_db,
              "n_passband_points": int(mask.sum())}
        if np.isfinite(f0) and np.isfinite(pf0):
            pr["f0_rel_diff"] = float((f0 - pf0) / pf0)
            if abs(pr["f0_rel_diff"]) > PRIOR_F0_REVIEW:
                flags.append("prior_f0_mismatch")
                notes.append(f"f0 {f0:.3f} GHz vs prior {pf0:.3f} GHz ({100 * pr['f0_rel_diff']:+.1f} %) exceeds {100 * PRIOR_F0_REVIEW:.0f} %")
        elif not np.isfinite(f0):
            flags.append("no_passband")
            notes.append("no 3 dB passband found inside the grid (prior f0 "
                         + (f"{pf0:.3f} GHz)" if np.isfinite(pf0) else "also undefined)"))
        if d_db > PRIOR_DB_REVIEW:
            flags.append("prior_passband_mismatch")
            notes.append(f"passband |S21| differs from the prior by {d_db:.1f} dB mean (> {PRIOR_DB_REVIEW} dB)")
        row["prior_delta"] = pr
    except Exception as e:  # noqa: BLE001
        flags.append("prior_failed")
        notes.append(f"prior could not be evaluated: {e}")

    # ---- provenance: solver, fidelity, mesh, compute, contributor, licence
    legacy = rec["legacy_collect"]
    scale = meta.get("scale")
    solver = solver or meta.get("solver") or ("palace" if legacy else None)
    solver_version = solver_version or meta.get("solver_version") or ("0.16" if legacy else None)
    fidelity = fidelity or meta.get("fidelity") or (LEGACY_SCALE_TO_FIDELITY.get(round(float(scale), 3)) if legacy and scale else None)
    order = order if order is not None else meta.get("order", 2 if legacy else None)
    mesh_elements = mesh_elements if mesh_elements is not None else meta.get("tets")
    mesh_unknowns = mesh_unknowns if mesh_unknowns is not None else meta.get("unknowns")
    port = port or meta.get("port") or ("wave port on the x face, mode 1, de-embedded 3 mm to the start of the coupled section" if legacy else None)
    contributor = contributor or meta.get("contributor")
    licence = licence or meta.get("licence")
    if legacy:
        notes.append("collect.py npz without provenance keys: solver, version, fidelity, order and port filled from the "
                     "reference-solver defaults of edge_coupled_bpf")
    for name, val in (("solver", solver), ("solver_version", solver_version), ("fidelity", fidelity), ("contributor", contributor)):
        if not val:
            errors.append(f"missing {name} (give --{name.replace('_', '-')})")
    if fidelity and fidelity not in FIDELITY_TAGS:
        errors.append(f"fidelity '{fidelity}' not in {FIDELITY_TAGS}")
    if mesh_elements is None or mesh_unknowns is None:
        flags.append("no_mesh_stats")
        notes.append("mesh statistics missing (--mesh-elements, --mesh-unknowns): fidelity cannot be audited")
    if not licence:
        flags.append("no_licence")
        notes.append("no licence stated (--licence); the row cannot be merged without one")
    row["solver"] = {"name": solver, "version": solver_version, "fidelity": fidelity,
                     "mesh": {"elements": int(mesh_elements) if mesh_elements is not None else None,
                              "unknowns": int(mesh_unknowns) if mesh_unknowns is not None else None,
                              "order": int(order) if order is not None else None,
                              "scale": float(scale) if scale is not None else None},
                     "adaptive_tol": meta.get("adaptive_tol"), "ports": port}
    row["substrate"] = fam["substrate"]
    row["compute"] = {"wall_s": _fin(meta.get("wall_s")), "peak_ram_gb": _fin(meta.get("peak_ram_gb")), "ranks": meta.get("ranks")}
    row["contributor"] = contributor
    row["licence"] = licence
    row["run"] = meta.get("run")
    row["file_sha256"] = file_hash(path)
    row["content_sha256"] = content_hash(p, fg, s11, s21)

    row["errors"], row["flags"], row["notes"] = errors, sorted(set(flags)), notes
    row["verdict"] = "FAIL" if errors else ("NEEDS REVIEW" if row["flags"] else "PASS")
    row["_arrays"] = {"f": fg, "S11": s11, "S21": s21}      # stripped before writing; used by --write-npz
    return row


def write_contract_npz(row: dict, out: Path):
    """Write the checked sample as a contract npz (family grid, legacy keys kept, new provenance keys added)."""
    a = row["_arrays"]
    sol = row["solver"]
    np.savez_compressed(
        out, p=np.asarray(row["p"], float), f=a["f"].astype(np.float64), S11=a["S11"].astype(np.complex128),
        S21=a["S21"].astype(np.complex128), param_names=np.asarray(row["param_names"]),
        substrate=row["substrate"]["name"], family_id=row["family_id"], schema=SCHEMA,
        solver=sol["name"], solver_version=sol["version"], fidelity=sol["fidelity"],
        unknowns=int(sol["mesh"]["unknowns"] or 0), tets=int(sol["mesh"]["elements"] or 0), order=int(sol["mesh"]["order"] or 0),
        scale=float(sol["mesh"]["scale"] if sol["mesh"]["scale"] is not None else np.nan), port=str(sol["ports"] or ""),
        wall_s=float(row["compute"]["wall_s"] if row["compute"]["wall_s"] is not None else np.nan),
        peak_ram_gb=float(row["compute"]["peak_ram_gb"] if row["compute"]["peak_ram_gb"] is not None else np.nan),
        ranks=int(row["compute"]["ranks"] or 0), contributor=str(row["contributor"]), licence=str(row["licence"] or ""),
        content_sha256=row["content_sha256"], metrics=json.dumps(row["metrics"]), run=str(row.get("run") or ""))


def one_line(row: dict) -> str:
    m = row.get("metrics", {})
    f0 = m.get("f0_GHz"); bw = m.get("BW_GHz")
    parts = [f"{row['verdict']}: {row['id']} [{row['family_id']}]"]
    if f0 is not None:
        parts.append(f"f0 {f0:.3f} GHz" + (f", BW {bw * 1e3:.0f} MHz" if bw is not None else ""))
    if "prior_delta" in row:
        parts.append(f"prior delta {row['prior_delta']['passband_dB_mean_abs_diff']:.2f} dB")
    sol = row.get("solver")
    if sol and sol.get("name"):
        parts.append(f"{sol['name']} {sol['version']} {sol['fidelity']}")
    if row.get("flags"):
        parts.append("flags: " + ",".join(row["flags"]))
    if row.get("errors"):
        parts.append("errors: " + " | ".join(row["errors"]))
    return "; ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path, help="sample: .npz (contract or collect.py keys) or Touchstone .s2p")
    ap.add_argument("--family", default=registry.DEFAULT_FAMILY, choices=registry.family_ids())
    ap.add_argument("--p", nargs="+", type=float, help="design vector in the family's units (required for .s2p)")
    ap.add_argument("--id", help="sample id for the manifest (default: file stem)")
    ap.add_argument("--solver", help="palace | openems | hfss | cst | ... (lower case)")
    ap.add_argument("--solver-version")
    ap.add_argument("--fidelity", choices=FIDELITY_TAGS, help="family fidelity tag; see docs/DATA_CONTRACT.md")
    ap.add_argument("--mesh-elements", type=int, help="tets / cells / elements of the solved mesh")
    ap.add_argument("--mesh-unknowns", type=int, help="degrees of freedom (FEM) or cells (FDTD)")
    ap.add_argument("--order", type=int, help="element order (FEM); 0 for FDTD/MoM")
    ap.add_argument("--port", help="port definition in words (type, de-embedding, reference plane)")
    ap.add_argument("--contributor", help="GitHub handle or ORCID")
    ap.add_argument("--licence", help="licence of the sample, e.g. CC-BY-4.0")
    ap.add_argument("--allow-out-of-box", action="store_true", help="flag instead of fail when p is outside the box")
    ap.add_argument("--out", type=Path, help="manifest row path (default: <file>.manifest.json next to the input)")
    ap.add_argument("--write-npz", type=Path, help="also write the sample as a contract npz on the family grid")
    ap.add_argument("--json", action="store_true", help="print the manifest row instead of the verdict lines")
    a = ap.parse_args(argv)
    row = check(a.file, a.family, p=a.p, solver=a.solver, solver_version=a.solver_version, fidelity=a.fidelity,
                contributor=a.contributor, licence=a.licence, mesh_elements=a.mesh_elements, mesh_unknowns=a.mesh_unknowns,
                order=a.order, port=a.port, allow_out_of_box=a.allow_out_of_box, sample_id=a.id)
    arrays = row.pop("_arrays", None)
    if a.write_npz and arrays is not None and row["verdict"] != "FAIL":
        row["_arrays"] = arrays
        write_contract_npz(row, a.write_npz)
        row.pop("_arrays")
        row["contract_npz"] = a.write_npz.name
    out = a.out or a.file.with_suffix(".manifest.json")
    out.write_text(json.dumps(row, indent=1))
    if a.json:
        print(json.dumps(row, indent=1))
    else:
        print(one_line(row))
        for n in row.get("notes", []):
            print("  note:", n)
        print(f"  manifest row -> {out}")
    return {"PASS": 0, "NEEDS REVIEW": 2, "FAIL": 1}[row["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
