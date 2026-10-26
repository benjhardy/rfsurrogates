"""Export the trained ensembles and the Palace references for the in-browser page (web/).

    python core/export_web.py            # web/weights.json, web/palace_refs.json, web/check_ref.json
    python core/export_web.py --check 24 # only the reference values for web/check_port.mjs (N random valid designs)
    python core/export_web.py --plain    # 7-digit text tensors instead of base64 Float32 (larger; misses the 1e-4 dB check)

weights.json: stage-B ensemble (prior + DeepONet correction, 5 seeds) and the scratch DeepONet ensemble (5 seeds)
as base64 Float32 per tensor (default: exact float32, passes web/check_port.mjs), or with --plain as
plain nested float arrays rounded to 7 significant digits (or base64 Float32 per tensor when the plain file
would exceed MAX_PLAIN_MB), every normaliser constant (p mean/std, f_lo/f_hi, Fourier k), the Palace frequency
grid, the sampling box, the centre, the parameter names and metadata (commit, date, split sizes).

palace_refs.json: Palace |S11|/|S21| in dB (3 decimals) on the grid for the tuned nominal and five held-out
designs of the stage-B test split (best, median and worst by passband error, plus two single-passband designs),
with p and the measured per-design error of each method.

check_ref.json: N random valid designs in the box with prior.py S11/S21 (re/im) and the torch ensemble-mean
|S21|, |S11| in dB for both ensembles, for web/check_port.mjs.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import CKPT, DATA, FAMILY, FAMILY_DIR, REPO, RESULTS, WEB  # noqa: E402
ROOT = REPO

import data as D                       # noqa: E402
import prior_wrap as PW                # noqa: E402
from net import DeepONetCorrection     # noqa: E402
import design                          # noqa: E402

MAX_PLAIN_MB = 6.0
BUFFERS = {"p_std.mean", "p_std.std", "p_std.fitted", "f_norm.f_lo", "f_norm.f_hi", "fourier_k"}
ENSEMBLES = {
    "stageB": ("prior + DeepONet (5-seed ensemble)", "stageB_seed{k}.pt"),
    "scratch": ("DeepONet from scratch (5-seed ensemble)", "scratch_seed{k}.pt"),
}


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def round_sig(x: np.ndarray, sig: int = 7):
    return [float(f"{v:.{sig}g}") for v in np.asarray(x, dtype=np.float64).reshape(-1)]


def nest(flat, shape):
    arr = np.asarray(flat).reshape(shape)
    return arr.tolist()


def load_model(path: Path) -> tuple[DeepONetCorrection, dict]:
    st = torch.load(path, map_location="cpu")
    sd = st["state_dict"]
    m = DeepONetCorrection(n_params=st["n_params"], n_basis=sd["branch_head.weight"].shape[0] // 4,
                           branch_hidden=sd["branch_body.0.weight"].shape[0], trunk_hidden=sd["trunk.0.weight"].shape[0],
                           n_fourier=int(sd["fourier_k"].numel()), warp="warp_head.weight" in sd, use_prior=st["use_prior"])
    m.load_state_dict(sd)
    m.use_prior = st["use_prior"]
    m.eval()
    return m, st


def export_model(m: DeepONetCorrection, st: dict, b64: bool) -> dict:
    sd = m.state_dict()
    tensors = {}
    for k, v in sd.items():
        if k in BUFFERS:
            continue
        v32 = v.detach().to(torch.float32).contiguous()
        if b64:
            data = base64.b64encode(v32.numpy().tobytes()).decode("ascii")
        else:
            data = nest(round_sig(v32.numpy()), tuple(v32.shape))
        tensors[k] = {"shape": list(v32.shape), "data": data}
    return {
        "seed": int(st.get("seed", -1)), "stage": st.get("stage"), "use_prior": bool(m.use_prior),
        "f_lo": float(sd["f_norm.f_lo"]), "f_hi": float(sd["f_norm.f_hi"]),
        "p_mean": round_sig(sd["p_std.mean"], 9), "p_std": round_sig(sd["p_std.std"], 9),
        "fourier_k": round_sig(sd["fourier_k"], 9),
        "n_parameters": int(m.n_parameters()),
        "tensors": tensors,
    }


def palace_grid() -> np.ndarray:
    z = np.load(DATA / "palace" / "nominal.npz", allow_pickle=True)
    return D.to_ghz(z["f"]).astype(np.float64)


def export_weights(b64: bool) -> Path:
    S = json.loads((FAMILY_DIR / "samples.json").read_text())
    names = S["param_names"]
    f = palace_grid()
    stage_b = json.loads((RESULTS / "stageB.json").read_text())
    n_palace = len(list((DATA / "palace").glob("*.npz")))
    meta = {
        "project": "rfsurrogates", "family_id": FAMILY, "commit": _git("rev-parse", "--short", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "exported": dt.datetime.now().isoformat(timespec="seconds"),
        "n_palace_samples": n_palace, "n_train": stage_b["n_train"], "n_val": stage_b["n_val"], "n_test": stage_b["n_test"],
        "n_basis": None, "n_fourier": None, "format": "b64f32" if b64 else "plain7g",
        "note": "S = S_prior(f;p) + sum_k b_k(p) t_k(f) (stage B) or S = sum_k b_k(p) t_k(f) (scratch); p = (w1, w2, g1, g2, l1, l2, eps_r, h), lengths mm, f GHz",
    }
    out = {"meta": meta, "param_names": names, "box": S["box"], "centre": S["centre"],
           "substrate": {"name": "RO4350B", "eps_r": S["centre"][6], "h_mm": S["centre"][7], "tand": 0.0037, "copper": "PEC sheet"},
           "validity": {"rule": S["validity"]["rule"], "c_min_mm": S["validity"]["c_min_mm"], "W50_mm": S["validity"]["W50_mm"]},
           "f_grid_GHz": [float(f"{v:.6g}") for v in f], "ensembles": {}}
    for key, (label, pat) in ENSEMBLES.items():
        models = []
        for k in range(5):
            p = CKPT / pat.format(k=k)
            if not p.exists():
                print(f"[export] missing {p.name}, skipped")
                continue
            m, st = load_model(p)
            meta["n_basis"], meta["n_fourier"] = m.n_basis, m.n_fourier
            models.append(export_model(m, st, b64))
        out["ensembles"][key] = {"label": label, "use_prior": bool(models[0]["use_prior"]) if models else None, "models": models}
    WEB.mkdir(exist_ok=True)
    path = WEB / "weights.json"
    txt = json.dumps(out, separators=(",", ":"))
    if not b64 and len(txt) > MAX_PLAIN_MB * 1e6:
        print(f"[export] plain export {len(txt) / 1e6:.2f} MB > {MAX_PLAIN_MB} MB, switching to base64 Float32")
        return export_weights(True)
    path.write_text(txt)
    print(f"[export] {path} {len(txt) / 1e6:.2f} MB, format {meta['format']}, "
          + ", ".join(f"{k}: {len(v['models'])} models" for k, v in out["ensembles"].items()))
    return path


def export_refs() -> Path:
    import eval as E
    f = palace_grid()
    rows = list(csv.DictReader((RESULTS / "per_sample_stageB.csv").open()))
    rows.sort(key=lambda r: float(r["pb_db_mean"]))
    best, median, worst = rows[0], rows[len(rows) // 2], rows[-1]
    single = [r for r in rows if r["split_passband"] == "0" and r["id"] not in (best["id"], median["id"], worst["id"])]
    pick = [("best", best), ("median", median), ("worst", worst)] + [("single-passband", r) for r in single[:2]]
    # per-sample error of every method on the test split (same code as eval.py)
    per = {}
    for key in ("prior_only", "stageB", "scratch"):
        z = E.load_preds(key)
        if z is None:
            continue
        m = E.per_sample_metrics(z["f"], z["s_true"], z["s_pred"])
        per[key] = {str(i): {"pb_db_mean": float(m["pb_db_mean"][j]), "pb_db_max": float(m["pb_db_max"][j]),
                             "f0_err_MHz": float(m["f0_err_mhz"][j]), "bw_err_MHz": float(m["bw_err_mhz"][j])}
                    for j, i in enumerate(z["ids"])}
    refs = []

    def one(sid, role, csvrow=None):
        z = np.load(DATA / "palace" / f"{sid}.npz", allow_pickle=True)
        fz = D.to_ghz(z["f"])
        assert fz.shape == f.shape and np.allclose(fz, f), f"{sid}: grid differs"
        s11, s21 = np.asarray(z["S11"], complex), np.asarray(z["S21"], complex)
        d11, d21 = E.db(s11), E.db(s21)
        lo, hi, _ = E.band_edges(f, d21)
        rec = {"id": sid, "role": role, "p": [float(v) for v in np.asarray(z["p"]).reshape(-1)],
               "clearance_mm": float(design.clearance(np.asarray(z["p"]).reshape(-1), design.w50_for(float(z["p"][6]), float(z["p"][7])))),
               "palace": {"f0_centroid_GHz": E.centroid_freq(f, s21), "bw3dB_GHz": float(hi - lo), "peak_S21_dB": float(d21.max()),
                          "split_passband": bool(E.passband_is_split(d21)), "wall_s": float(np.asarray(z["wall_s"]).reshape(-1)[0]) if "wall_s" in z.files else None},
               "s11_dB": [round(float(v), 3) for v in d11], "s21_dB": [round(float(v), 3) for v in d21],
               "error": {k: v[sid] for k, v in per.items() if sid in v}}
        if csvrow is not None:
            rec["dist_to_edge"] = float(csvrow["dist_to_edge"])
        return rec

    refs.append(one("nominal", "tuned nominal (box centre)"))
    for role, r in pick:
        refs.append(one(r["id"], role, r))
    out = {"note": "Palace FEM (coarse production mesh, scale 1.8, order 2) |S11|, |S21| in dB on the 3-7 GHz grid; "
                   "error = per-design held-out error of each surrogate (passband |S21| dB mean/max, f0 and BW error)",
           "f_grid_GHz": [float(f"{v:.6g}") for v in f], "refs": refs}
    path = WEB / "palace_refs.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"[export] {path} {path.stat().st_size / 1e3:.0f} kB: " + ", ".join(f"{r['id']} ({r['role']})" for r in refs))
    return path


def export_check(n: int, seed: int = 20261026) -> Path:
    S = json.loads((FAMILY_DIR / "samples.json").read_text())
    names, box, centre = S["param_names"], S["box"], S["centre"]
    f = palace_grid()
    w50 = design.w50_for(centre[6], centre[7])
    rng = np.random.default_rng(seed)
    P = []
    while len(P) < n:
        p = list(centre)
        for i, nm in enumerate(names):
            lo, hi = box[nm]
            if hi > lo:
                p[i] = float(rng.uniform(lo, hi))
        if design.is_valid(p, w50=w50):
            P.append(p)
    P = np.asarray(P, dtype=np.float64)
    s_prior = PW.s_prior_batch(P, f).astype(np.float64)       # float32 layout used by the model
    out = {"seed": seed, "f_grid_GHz": [float(f"{v:.6g}") for v in f], "p": P.tolist(), "prior": [], "ensembles": {}}
    s_prior64 = np.empty((len(P), 4, f.size), dtype=np.float64)
    for i, p in enumerate(P):
        s11, s21 = PW.s_prior(p, f)                             # full float64 prior for the exact prior check
        s_prior64[i, 0], s_prior64[i, 1], s_prior64[i, 2], s_prior64[i, 3] = s11.real, s11.imag, s21.real, s21.imag
        out["prior"].append({"s11re": s11.real.tolist(), "s11im": s11.imag.tolist(), "s21re": s21.real.tolist(), "s21im": s21.imag.tolist()})
    ft = torch.as_tensor(f, dtype=torch.float32)
    pt = torch.as_tensor(P, dtype=torch.float32)
    spt = torch.as_tensor(s_prior, dtype=torch.float32)
    # Two references per ensemble: the network run in float64 (same float32 weights cast up; this is what a port in
    # double precision must reproduce) and in float32 as trained (its own rounding noise is ~1e-6 in S, which in dB
    # is amplified wherever |S| is small, so it is reported for information only).
    ft64, pt64 = torch.as_tensor(f, dtype=torch.float64), torch.as_tensor(P, dtype=torch.float64)
    spt64 = torch.as_tensor(s_prior64, dtype=torch.float64)
    for key, (_, pat) in ENSEMBLES.items():
        preds32, preds64 = [], []
        for k in range(5):
            path = CKPT / pat.format(k=k)
            if not path.exists():
                continue
            m, _ = load_model(path)
            with torch.no_grad():
                preds32.append(m(pt, ft, spt if m.use_prior else None).double())
                m64 = m.double()
                preds64.append(m64(pt64, ft64, spt64 if m.use_prior else None))
        rec = {"n_models": len(preds64)}
        for tag, preds in (("", preds64), ("_f32", preds32)):
            mean = torch.stack(preds).mean(0).numpy().astype(np.float64)   # (N,4,M)
            s11 = mean[:, 0] + 1j * mean[:, 1]; s21 = mean[:, 2] + 1j * mean[:, 3]
            rec["s21_dB" + tag] = (20 * np.log10(np.maximum(np.abs(s21), 1e-12))).tolist()
            rec["s11_dB" + tag] = (20 * np.log10(np.maximum(np.abs(s11), 1e-12))).tolist()
            if tag == "":
                rec["s21_re"], rec["s21_im"] = s21.real.tolist(), s21.imag.tolist()
        out["ensembles"][key] = rec
    WEB.mkdir(exist_ok=True)
    path = WEB / "check_ref.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"[export] {path} {path.stat().st_size / 1e6:.2f} MB, {n} designs")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, nargs="?", const=24, default=None, help="write only check_ref.json with N designs")
    ap.add_argument("--plain", action="store_true", help="7-digit text tensors instead of base64 Float32")
    ap.add_argument("--b64", action="store_true", help="default; kept for old command lines")
    ap.add_argument("--n-check", type=int, default=24)
    a = ap.parse_args(argv)
    if a.check is not None:
        export_check(a.check)
        return
    export_weights(not a.plain)
    export_refs()
    export_check(a.n_check)


if __name__ == "__main__":
    main()
