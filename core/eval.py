"""Held-out metrics and figures from families/<id>/results/preds_*.npz.

    python core/eval.py            # writes figs/*.png, figs/INDEX.md, families/<id>/results/metrics.json

Metrics per method (prior only, scratch DeepONet, plain MLP, stage-B ensemble):
  passband |S21| error in dB (mean, max; passband = within 3 dB of the Palace peak)
  f0 error (MHz and %), f0 = |S21|^2-weighted centroid frequency (continuous when the passband splits)
  BW error over single-passband samples only (a split passband has no -3 dB band)
  BW error (MHz and %), BW = -3 dB bandwidth
  |S11| error: mean abs linear over the band; mean dB error where Palace |S11| > -20 dB
Figures use matplotlib's default colour cycle and style on purpose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

from paths import FIGS, RESULTS  # noqa: E402
FIGS.mkdir(exist_ok=True)
DEAD_DB = -10.0  # Palace peak |S21| below this = no passband (box-corner designs); excluded from the metric table
SPLIT_DB = 3.0   # dip deeper than this between two passband peaks = split passband (no single -3 dB band)

METHODS = [("prior_only", "prior only"), ("scratch", "DeepONet from scratch"), ("mlp", "plain MLP"),
           ("stageB", "prior + DeepONet (stage B ensemble)")]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def unpack_np(s):
    return s[:, 0] + 1j * s[:, 1], s[:, 2] + 1j * s[:, 3]


def db(z):
    return 20 * np.log10(np.maximum(np.abs(z), 1e-12))


def band_edges(f, s21_db, level=3.0):
    """(f_lo, f_hi, f_peak) of the band within `level` dB of the peak, linear-interpolated."""
    i = int(np.argmax(s21_db))
    thr = s21_db[i] - level
    lo = i
    while lo > 0 and s21_db[lo] >= thr:
        lo -= 1
    hi = i
    while hi < len(f) - 1 and s21_db[hi] >= thr:
        hi += 1
    def cross(a, b):
        if s21_db[a] == s21_db[b]:
            return f[a]
        t = (thr - s21_db[a]) / (s21_db[b] - s21_db[a])
        return f[a] + t * (f[b] - f[a])
    f_lo = cross(lo, lo + 1) if s21_db[lo] < thr else f[lo]
    f_hi = cross(hi, hi - 1) if s21_db[hi] < thr else f[hi]
    return f_lo, f_hi, f[i]


def centroid_freq(f, s21):
    """|S21|^2-weighted centre frequency over the whole window (continuous even when the passband splits)."""
    w = np.abs(s21) ** 2
    return float((f * w).sum() / max(w.sum(), 1e-30))


def passband_is_split(s21_db, within_db=6.0, dip_db=SPLIT_DB):
    """True when two or more peaks within `within_db` of the maximum are separated by a dip deeper than `dip_db`."""
    from scipy.signal import find_peaks
    pk, _ = find_peaks(s21_db, prominence=1.0)
    pk = [p for p in pk if s21_db[p] > s21_db.max() - within_db]
    if len(pk) < 2:
        return False
    return bool(s21_db[min(pk):max(pk) + 1].min() < s21_db.max() - dip_db)


def per_sample_metrics(f, s_true, s_pred):
    """Return dict of per-sample arrays."""
    s11_t, s21_t = unpack_np(s_true)
    s11_p, s21_p = unpack_np(s_pred)
    n = s_true.shape[0]
    out = {k: np.zeros(n) for k in ("pb_db_mean", "pb_db_max", "f0_true", "f0_pred", "bw_true", "bw_pred",
                                    "s11_lin", "s11_db", "pb_s11_db")}
    # a sample whose Palace peak |S21| is below DEAD_DB has no passband: f0/BW/passband metrics are meaningless there
    out["live"] = np.zeros(n, dtype=bool)
    # a detuned design whose Palace passband splits into two peaks with a dip deeper than SPLIT_DB has no single
    # -3 dB band: its -3 dB BW is meaningless (reported over single-passband samples only); f0 is the |S21|^2 centroid
    # for every sample so that it stays continuous across the split
    out["split"] = np.zeros(n, dtype=bool)
    for i in range(n):
        dt, dp = db(s21_t[i]), db(s21_p[i])
        out["live"][i] = dt.max() >= DEAD_DB
        out["split"][i] = passband_is_split(dt)
        lo, hi, _ = band_edges(f, dt)
        mask = dt >= dt.max() - 3.0
        err = np.abs(dp - dt)[mask]
        out["pb_db_mean"][i] = err.mean(); out["pb_db_max"][i] = err.max()
        out["f0_true"][i] = centroid_freq(f, s21_t[i]); out["bw_true"][i] = hi - lo
        plo, phi, _ = band_edges(f, dp)
        out["f0_pred"][i] = centroid_freq(f, s21_p[i]); out["bw_pred"][i] = phi - plo
        out["s11_lin"][i] = np.abs(np.abs(s11_p[i]) - np.abs(s11_t[i])).mean()
        m11 = db(s11_t[i]) > -20
        out["s11_db"][i] = np.abs(db(s11_p[i]) - db(s11_t[i]))[m11].mean() if m11.any() else np.nan
        out["pb_s11_db"][i] = np.abs(db(s11_p[i]) - db(s11_t[i]))[mask].mean()
    out["f0_err_mhz"] = 1e3 * (out["f0_pred"] - out["f0_true"])
    out["f0_err_pct"] = 100 * (out["f0_pred"] - out["f0_true"]) / out["f0_true"]
    out["bw_err_mhz"] = 1e3 * (out["bw_pred"] - out["bw_true"])
    out["bw_err_pct"] = 100 * (out["bw_pred"] - out["bw_true"]) / np.maximum(out["bw_true"], 1e-9)
    return out


def summarize(m_all):
    """Aggregate over live samples only (dead filters, Palace peak |S21| < DEAD_DB, are counted but excluded)."""
    live = m_all["live"]
    m = {k: v[live] for k, v in m_all.items() if k != "live"}
    single = ~m["split"]
    bw_mhz = np.abs(m["bw_err_mhz"][single]); bw_pct = np.abs(m["bw_err_pct"][single])
    return {
        "n_dead_excluded": int((~live).sum()),
        "n_split_passband": int(m["split"].sum()),
        "passband_S21_dB_mean": float(np.mean(m["pb_db_mean"])),
        "passband_S21_dB_max": float(np.max(m["pb_db_max"])),
        "passband_S21_dB_p90_of_sample_means": float(np.percentile(m["pb_db_mean"], 90)),
        "f0_err_MHz_mean_abs": float(np.mean(np.abs(m["f0_err_mhz"]))),
        "f0_err_MHz_max_abs": float(np.max(np.abs(m["f0_err_mhz"]))),
        "f0_err_pct_mean_abs": float(np.mean(np.abs(m["f0_err_pct"]))),
        "bw_err_MHz_mean_abs": float(np.mean(bw_mhz)) if single.any() else float("nan"),
        "bw_err_pct_mean_abs": float(np.mean(bw_pct)) if single.any() else float("nan"),
        "S11_lin_mean_abs": float(np.mean(m["s11_lin"])),
        "S11_dB_mean_abs_where_above_m20dB": float(np.nanmean(m["s11_db"])),
        "n": int(len(m["pb_db_mean"])),
    }


def load_preds(name):
    p = RESULTS / f"preds_{name}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _names(z):
    """param_names stored in a preds npz (falls back to generic names of the right length)."""
    if "param_names" in z:
        return tuple(str(x) for x in np.asarray(z["param_names"]).reshape(-1))
    return tuple(f"p{i}" for i in range(z["p"].shape[1]))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_method_bars(summ, per, index):
    names = [lab for key, lab in METHODS if key in summ]
    keys = [key for key, _ in METHODS if key in summ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (metric, title) in zip(axes, [("pb_db_mean", "passband |S21| error (dB)"),
                                          ("f0_err_mhz", "|f0 error| (MHz)"), ("bw_err_mhz", "|BW error| (MHz)")]):
        vals = [np.abs(per[k][metric]) for k in keys]
        ax.boxplot(vals, tick_labels=[n.replace(" (", "\n(") for n in names], showmeans=True)
        ax.set_title(title); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
    n = summ[keys[0]]["n"]
    fig.suptitle(f"Held-out Palace samples (N={n}): per-sample error distribution")
    fig.tight_layout()
    fn = "method_comparison.png"
    fig.savefig(FIGS / fn, dpi=150); plt.close(fig)
    index.append((fn, f"Per-sample passband |S21| dB error, f0 error and BW error on the N={n} held-out Palace samples, for each method. Box = quartiles, triangle = mean."))


def fig_parity(per, index):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for key, lab in METHODS:
        if key in per and key in ("prior_only", "stageB"):
            ax.scatter(per[key]["f0_true"], per[key]["f0_pred"], s=18, alpha=0.8, label=lab)
    k0 = "stageB" if "stageB" in per else next(iter(per))
    lo = min(per[k0]["f0_true"].min(), per[k0]["f0_pred"].min()); hi = max(per[k0]["f0_true"].max(), per[k0]["f0_pred"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("f0 Palace (GHz)"); ax.set_ylabel("f0 predicted (GHz)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Centre frequency parity (held-out)")
    fig.tight_layout(); fn = "parity_f0.png"; fig.savefig(FIGS / fn, dpi=150); plt.close(fig)
    index.append((fn, "Centre frequency (|S21|^2-weighted centroid over 3-7 GHz) predicted vs Palace, held-out samples: prior only and stage-B ensemble."))


def fig_error_vs_edge(per, preds, box, index):
    if "stageB" not in per:
        return
    p = preds["stageB"]["p"]
    d = D.distance_to_box_edge(p, box, _names(preds["stageB"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("prior_only", "stageB"):
        if key in per:
            ax.scatter(d, per[key]["pb_db_mean"], s=18, alpha=0.8, label=dict(METHODS)[key])
    ax.set_xlabel("normalized distance to nearest box edge (0 = edge, 0.5 = centre)")
    ax.set_ylabel("passband |S21| error (dB)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Error vs position in the design box")
    fig.tight_layout(); fn = "error_vs_box_edge.png"; fig.savefig(FIGS / fn, dpi=150); plt.close(fig)
    index.append((fn, "Per-sample passband |S21| error against the normalized distance of the design point to the nearest edge of the sampling box."))
    return d


def fig_calibration(preds, per, index):
    z = preds.get("stageB")
    if z is None or z["s_pred_seeds"].shape[0] < 2:
        return None
    s_true, seeds = z["s_true"], z["s_pred_seeds"]            # (N,4,M), (S,N,4,M)
    _, s21_t = unpack_np(s_true)
    dbt = db(s21_t)
    mask = dbt >= dbt.max(axis=1, keepdims=True) - 3.0
    db_seeds = np.stack([db(unpack_np(seeds[k])[1]) for k in range(seeds.shape[0])])   # (S,N,M)
    spread = db_seeds.std(0)                                 # (N,M)
    err = np.abs(db_seeds.mean(0) - dbt)
    sp = np.array([spread[i][mask[i]].mean() for i in range(len(s_true))])
    er = np.array([err[i][mask[i]].mean() for i in range(len(s_true))])
    r = float(np.corrcoef(sp, er)[0, 1]) if len(sp) > 2 else float("nan")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(sp, er, s=18)
    m = max(sp.max(), er.max())
    ax.plot([0, m], [0, m], "k--", lw=1, label="spread = error")
    ax.set_xlabel("ensemble spread, passband |S21| std over seeds (dB)")
    ax.set_ylabel("ensemble error, passband |S21| (dB)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"Calibration: spread vs error (Pearson r = {r:.2f})")
    fig.tight_layout(); fn = "calibration_spread_vs_error.png"; fig.savefig(FIGS / fn, dpi=150); plt.close(fig)
    index.append((fn, f"Per-sample 5-seed spread of passband |S21| (dB) against the actual ensemble error; Pearson r = {r:.2f}."))
    return {"pearson_r": r, "spread_mean_db": float(sp.mean()), "error_mean_db": float(er.mean()),
            "frac_error_below_2sigma": float(np.mean(er <= 2 * sp + 1e-9))}


def fig_examples(preds, per, index, n_show=4):
    z = preds.get("stageB")
    if z is None:
        return
    f = z["f"]
    order = np.argsort(per["stageB"]["pb_db_mean"])
    pick = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]][:min(n_show, len(order))]
    fig, axes = plt.subplots(2, len(pick), figsize=(3.6 * len(pick), 6), squeeze=False)
    for j, i in enumerate(pick):
        s11_t, s21_t = unpack_np(z["s_true"][i:i + 1]); s11_p, s21_p = unpack_np(z["s_prior"][i:i + 1]); s11_m, s21_m = unpack_np(z["s_pred"][i:i + 1])
        ax = axes[0, j]
        ax.plot(f, db(s21_t[0]), label="Palace"); ax.plot(f, db(s21_p[0]), "--", label="prior"); ax.plot(f, db(s21_m[0]), ":", lw=2, label="prior + NN")
        ax.set_title(f"{z['ids'][i]}  err {per['stageB']['pb_db_mean'][i]:.2f} dB", fontsize=9); ax.set_ylabel("|S21| (dB)"); ax.set_ylim(-60, 2); ax.grid(alpha=0.3)
        if j == 0: ax.legend(fontsize=8)
        ax = axes[1, j]
        ax.plot(f, db(s11_t[0])); ax.plot(f, db(s11_p[0]), "--"); ax.plot(f, db(s11_m[0]), ":", lw=2)
        ax.set_ylabel("|S11| (dB)"); ax.set_xlabel("f (GHz)"); ax.set_ylim(-40, 2); ax.grid(alpha=0.3)
    fig.suptitle("Held-out examples: best, two middle, worst (by passband |S21| error)")
    fig.tight_layout(); fn = "examples_heldout.png"; fig.savefig(FIGS / fn, dpi=150); plt.close(fig)
    index.append((fn, "Four held-out designs (best, two middle, worst by passband error): Palace, prior, and prior + DeepONet correction, |S21| and |S11| in dB."))


def fig_stage_c(index):
    p = RESULTS / "stageC.json"
    if not p.exists():
        return None
    js = json.loads(p.read_text())
    if js.get("status") != "done":
        return js
    ns = [int(n) for n in js["n_samples"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, lab in (("adapt", "frozen trunk, branch adapted from stage B"), ("scratch", "from scratch (prior ON)")):
        mean = [js[key][str(n)]["ensemble"]["pb_db_mean"] for n in ns]
        seeds = np.array([[s["pb_db_mean"] for s in js[key][str(n)]["seeds"]] for n in ns])
        ax.errorbar(ns, mean, yerr=seeds.std(1), marker="o", capsize=3, label=lab)
    ax.axhline(js["prior_only"]["pb_db_mean"], color="k", ls="--", lw=1, label="prior only")
    ax.set_xlabel("number of second-substrate Palace samples used"); ax.set_ylabel("passband |S21| error (dB), held-out")
    ax.set_xticks(ns); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    # with a single substrate in the data, group "all" means the pool and the test set are the SAME substrate:
    # this is a same-substrate few-shot test, not a new-substrate adaptation
    same = js.get("group") == "all"
    js["label"] = ("same-substrate few-shot (only one substrate in the Palace data; not a new-substrate test)"
                   if same else f"adaptation to substrate {js['group']}")
    ax.set_title(("Stage C (same-substrate few-shot, one substrate only)" if same else
                  f"Stage C: adaptation to substrate '{js['group']}'") + f" (test N={js['n_test']})")
    fig.tight_layout(); fn = "stageC_adaptation.png"; fig.savefig(FIGS / fn, dpi=150); plt.close(fig)
    index.append((fn, ("Stage C as run so far: SAME-substrate few-shot (the Palace data has one substrate, so pool and test "
                       "share it; this is not a new-substrate test). " if same else "Stage C: ")
                  + "Held-out passband |S21| error vs number of samples, frozen-trunk adaptation vs training from scratch; error bars = std over seeds."))
    return js


def fig_timing(index):
    rows = []
    b = RESULTS / "stageB.json"
    if b.exists():
        js = json.loads(b.read_text())
        t = [js["seeds"][k]["timing"] for k in js["seeds"]]
        rows.append(("prior + DeepONet, batch 1", np.mean([x["ms_per_query_batch1"] for x in t])))
        rows.append(("prior + DeepONet, batched", np.mean([x["ms_per_query_batched"] for x in t])))
        wall = js.get("palace_wall_s_mean")
        if wall:
            rows.append(("Palace FEM per sample", 1e3 * wall))
    bl = RESULTS / "baselines.json"
    if bl.exists():
        js = json.loads(bl.read_text())
        for key, lab in (("scratch_deeponet", "scratch DeepONet, batch 1"), ("plain_mlp", "plain MLP, batch 1")):
            t = [v["timing"]["ms_per_query_batch1"] for k, v in js[key].items() if k.startswith("seed")]
            rows.append((lab, np.mean(t)))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh([r[0] for r in rows], [r[1] for r in rows])
    ax.set_xscale("log"); ax.set_xlabel("ms per query (log)"); ax.grid(alpha=0.3, axis="x")
    for i, r in enumerate(rows):
        ax.text(r[1], i, f" {r[1]:.3g} ms", va="center", fontsize=8)
    ax.set_title("Time per design query (network forward only; prior eval excluded)")
    fig.tight_layout(); fn = "timing.png"; fig.savefig(FIGS / fn, dpi=150); plt.close(fig)
    index.append((fn, "Milliseconds per query (CPU) for the networks vs the mean Palace wall time per sample recorded in the dataset."))
    return {r[0]: float(r[1]) for r in rows}


def fig_training_curves(index):
    b = RESULTS / "stageB.json"
    if not b.exists():
        return
    js = json.loads(b.read_text())
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in js["seeds"].items():
        h = v["history"]
        ax.semilogy(h["epoch"], h["val_mse"], label=f"{k} val")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation MSE (re/im)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Stage B fine-tuning, validation loss per seed")
    fig.tight_layout(); fn = "stageB_training.png"; fig.savefig(FIGS / fn, dpi=150); plt.close(fig)
    index.append((fn, "Stage B validation MSE vs epoch for the five seeds (early stopping restores the best epoch)."))


# ---------------------------------------------------------------------------
def main(argv=None):
    preds = {k: load_preds(k) for k, _ in METHODS}
    preds = {k: v for k, v in preds.items() if v is not None}
    if not preds:
        print("[eval] no predictions found in", RESULTS)
        return
    per, summ = {}, {}
    for k, z in preds.items():
        per[k] = per_sample_metrics(z["f"], z["s_true"], z["s_pred"])
        summ[k] = summarize(per[k])
    z0 = next(iter(preds.values()))
    box = D.load_box() or D.box_from_data(z0["p"], _names(z0))
    index = []
    fig_method_bars(summ, per, index)
    fig_parity(per, index)
    fig_error_vs_edge(per, preds, box, index)
    calib = fig_calibration(preds, per, index)
    fig_examples(preds, per, index)
    fig_training_curves(index)
    stage_c = fig_stage_c(index)
    timing = fig_timing(index)

    metrics = {"methods": summ, "calibration": calib, "timing_ms": timing,
               "stageC": None if stage_c is None else {k: v for k, v in stage_c.items() if k != "test_ids"},
               "box": box}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    # per-sample table for the stage-B ensemble
    if "stageB" in per:
        z = preds["stageB"]
        with open(RESULTS / "per_sample_stageB.csv", "w") as fh:
            fh.write("id,pb_db_mean,pb_db_max,f0_true_GHz,f0_pred_GHz,f0_err_MHz,bw_err_MHz,s11_lin,dist_to_edge,split_passband\n")
            d = D.distance_to_box_edge(z["p"], box, _names(z))
            for i in range(len(z["ids"])):
                m = per["stageB"]
                fh.write(f"{z['ids'][i]},{m['pb_db_mean'][i]:.4f},{m['pb_db_max'][i]:.4f},{m['f0_true'][i]:.5f},{m['f0_pred'][i]:.5f},"
                         f"{m['f0_err_mhz'][i]:.2f},{m['bw_err_mhz'][i]:.2f},{m['s11_lin'][i]:.4f},{d[i]:.3f},{int(m['split'][i])}\n")

    # figures index + metrics table
    s0 = summ[next(iter(summ))]
    lines = ["# Figures index", "",
             f"Held-out Palace samples: N = {s0['n']} live filters in the table below "
             f"({s0['n_dead_excluded']} dead filters with Palace peak |S21| < {DEAD_DB:g} dB excluded from the metrics; "
             f"{s0['n_split_passband']} have a split passband (dip > {SPLIT_DB:g} dB between peaks) and are left out of the BW column only; "
             f"f0 = |S21|^2 centroid; "
             f"the figures still show every held-out sample)", "",
             "| method | passband \\|S21\\| err dB (mean / max) | f0 err MHz (mean abs / max) | f0 err % | BW err MHz | \\|S11\\| lin err |",
             "|---|---|---|---|---|---|"]
    for k, lab in METHODS:
        if k in summ:
            s = summ[k]
            lines.append(f"| {lab} | {s['passband_S21_dB_mean']:.3f} / {s['passband_S21_dB_max']:.2f} | "
                         f"{s['f0_err_MHz_mean_abs']:.1f} / {s['f0_err_MHz_max_abs']:.1f} | {s['f0_err_pct_mean_abs']:.2f} | "
                         f"{s['bw_err_MHz_mean_abs']:.1f} | {s['S11_lin_mean_abs']:.4f} |")
    if calib:
        lines += ["", f"Calibration: Pearson r(spread, error) = {calib['pearson_r']:.2f}; "
                      f"{100 * calib['frac_error_below_2sigma']:.0f} % of samples have error <= 2 x spread."]
    if timing:
        lines += ["", "Timing (ms per query): " + "; ".join(f"{k} {v:.3g}" for k, v in timing.items())]
    if stage_c is not None and stage_c.get("status") != "done":
        lines += ["", f"Stage C: pending ({stage_c.get('reason', '')})"]
    lines += ["", "| file | caption |", "|---|---|"] + [f"| {fn} | {cap} |" for fn, cap in index]
    (FIGS / "INDEX.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
