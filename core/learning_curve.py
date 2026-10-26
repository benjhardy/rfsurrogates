"""Learning curve on the valid Palace data: held-out error vs number of training samples.

    python core/learning_curve.py [--ns 10 25 50 100 150 all] [--seeds 5 5 5 3 3 3] [--quiet]

Fixed held-out test set and validation set: the stage-B split (D.split_indices, split_seed 0) of every file in
data/palace. For each N and seed k, the training subset is the first N of a seed-k permutation of the train pool.
Methods: prior + DeepONet (stage-A checkpoint of seed k, switch_prior_on), DeepONet from scratch (no prior),
plain MLP (only at N = 50 and all). Metrics on LIVE test samples (eval.per_sample_metrics): passband |S21| dB mean,
|f0 error| MHz mean, |S11| linear error mean; prior-only as the flat reference.
Outputs: families/<id>/results/learning_curve.json, figs/learning_curve.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_num_threads(int(os.environ.get("RFS_THREADS", 3)))

import data as D                                                   # noqa: E402
import eval as E                                                   # noqa: E402
import train as T                                                  # noqa: E402
from net import PlainMLP                                           # noqa: E402
from trainer import TensorSet, fit, predict                        # noqa: E402

from paths import FIGS, RESULTS  # noqa: E402


def live_metrics(f, s_true, s_pred):
    per = E.per_sample_metrics(f, s_true, s_pred)
    live = per["live"]
    return {"pb_db_mean": float(np.mean(per["pb_db_mean"][live])), "pb_db_max": float(np.max(per["pb_db_max"][live])),
            "f0_err_mhz": float(np.mean(np.abs(per["f0_err_mhz"][live]))), "s11_lin": float(np.mean(per["s11_lin"][live])),
            "n_live": int(live.sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", nargs="+", default=["10", "25", "50", "100", "150", "all"])
    ap.add_argument("--seeds-per-n", nargs="+", type=int, default=[5, 5, 5, 3, 3, 3])
    ap.add_argument("--mlp-ns", nargs="+", default=["50", "all"])
    ap.add_argument("--quiet", action="store_true")
    for k, v in T.DEFAULTS.items():
        if not isinstance(v, tuple) and k != "seeds":
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=(6000 if k == "b_epochs" else v))
    args = ap.parse_args(argv)
    args.seeds = max(args.seeds_per_n)
    args.prior_npz = None; args.palace_dir = None; args.all_substrates = True
    ds, s_prior, gname = T.palace_sets(args, "primary")
    full = TensorSet.from_numpy(ds.p, ds.f, ds.s, s_prior)
    tr, va, te = D.split_indices(len(ds), seed=args.split_seed)
    f = ds.f; s_true_te = ds.s[te]
    prior_only = live_metrics(f, s_true_te, s_prior[te])
    out = {"group": gname, "n_total": int(len(ds)), "n_pool": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "test_ids": [ds.ids[i] for i in te], "split_seed": args.split_seed, "prior_only": prior_only,
           "epoch_cap": int(args.b_epochs), "patience": int(args.b_patience),
           "ns": [], "methods": {"prior_nn": {}, "scratch": {}, "mlp": {}}, "seconds": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"[LC] pool {len(tr)}, val {len(va)}, test {len(te)} (live {prior_only['n_live']}); prior-only {prior_only['pb_db_mean']:.3f} dB")
    preds_store = {"f": f, "s_true": s_true_te, "s_prior": s_prior[te], "ids": np.array([ds.ids[i] for i in te]),
                   "p": ds.p[te], "param_names": np.array(ds.param_names)}
    t_all = time.time()
    for n_str, n_seeds in zip(args.ns, args.seeds_per_n):
        n = len(tr) if n_str == "all" else int(n_str)
        n = min(n, len(tr))
        out["ns"].append(n)
        for method in ("prior_nn", "scratch", "mlp"):
            if method == "mlp" and n_str not in args.mlp_ns:
                continue
            runs = []
            t0 = time.time()
            for k in range(n_seeds):
                sub = np.random.default_rng(k).permutation(tr)[:n]
                torch.manual_seed(k)
                if method == "prior_nn":
                    m = T.load_ckpt(T.CKPT / f"stageA_seed{k}.pt", args, ds.param_names)
                    m.switch_prior_on()
                    tset = full
                elif method == "scratch":
                    m = T.new_model(args, use_prior=False, n_params=ds.n_params)
                    m.p_std.fit(full.p); m.f_norm.fit(full.f)
                    tset = TensorSet(full.p, full.f, full.s, None)
                else:
                    m = PlainMLP(n_freq=len(ds.f), n_params=ds.n_params)
                    m.p_std.fit(full.p)
                    tset = TensorSet(full.p, full.f, full.s, None)
                res = fit(m, tset.take(sub), tset.take(va), epochs=args.b_epochs, lr=args.b_lr, weight_decay=args.b_wd,
                          patience=args.b_patience, seed=k, lam_passive=args.lam_passive, log_every=10 ** 9, verbose=False)
                pred = predict(m, tset.take(te)).numpy()
                preds_store[f"{method}_N{n}_seed{k}"] = pred.astype(np.float32)   # kept so metrics can be recomputed
                met = live_metrics(f, s_true_te, pred)
                met.update(seed=k, epochs=res.epochs_run, best_epoch=res.best_epoch, seconds=res.seconds)
                runs.append(met)
                print(f"[LC] {method:9s} N={n:3d} seed {k}: pb {met['pb_db_mean']:.3f} dB  f0 {met['f0_err_mhz']:.1f} MHz  "
                      f"|S11| {met['s11_lin']:.4f}  ({res.epochs_run} ep, {res.seconds:.0f}s)", flush=True)
            agg = {key: {"mean": float(np.mean([r[key] for r in runs])), "std": float(np.std([r[key] for r in runs]))}
                   for key in ("pb_db_mean", "f0_err_mhz", "s11_lin")}
            out["methods"][method][str(n)] = {"n_seeds": n_seeds, "seeds": runs, **agg}
            out["seconds"][f"{method}_{n}"] = time.time() - t0
            (RESULTS / "learning_curve.json").write_text(json.dumps(out, indent=1))
            np.savez_compressed(RESULTS / "learning_curve_preds.npz", **preds_store)
    out["total_seconds"] = time.time() - t_all
    out["metric_definitions"] = "eval.per_sample_metrics: f0 = |S21|^2-weighted centroid (eval 5b042ad), passband = within 3 dB of the Palace peak, |S11| linear error mean"
    (RESULTS / "learning_curve.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(RESULTS / "learning_curve_preds.npz", **preds_store)
    make_figure(out)
    print_table(out)


def print_table(out):
    lines = ["| N train | prior only | prior + DeepONet | DeepONet from scratch | plain MLP |", "|---|---|---|---|---|"]
    po = out["prior_only"]
    for n in out["ns"]:
        cells = []
        for method in ("prior_nn", "scratch", "mlp"):
            r = out["methods"][method].get(str(n))
            cells.append("-" if r is None else f"{r['pb_db_mean']['mean']:.2f} +- {r['pb_db_mean']['std']:.2f} dB / {r['f0_err_mhz']['mean']:.0f} MHz / {r['s11_lin']['mean']:.3f} ({r['n_seeds']} seeds)")
        lines.append(f"| {n} | {po['pb_db_mean']:.2f} dB / {po['f0_err_mhz']:.0f} MHz / {po['s11_lin']:.3f} | " + " | ".join(cells) + " |")
    txt = "\n".join(lines)
    print(txt)
    (RESULTS / "learning_curve.md").write_text(f"# Learning curve (held-out N_test = {out['n_test']}, live {out['prior_only']['n_live']}; "
                                               f"cells: passband |S21| dB mean +- std over seeds / |f0 err| MHz / |S11| lin)\n\n{txt}\n")


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = out["ns"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, lab in zip(axs, ("pb_db_mean", "f0_err_mhz", "s11_lin"),
                            ("passband |S21| error (dB)", "|f0 error| (MHz)", "|S11| linear error")):
        po = out["prior_only"][key if key != "f0_err_mhz" else "f0_err_mhz"]
        ax.axhline(po, ls="--", lw=1.2, label="prior only")
        for method, mlab, mk in (("prior_nn", "prior + DeepONet (stage-A init)", "o"), ("scratch", "DeepONet from scratch", "s"), ("mlp", "plain MLP", "^")):
            xs = [n for n in ns if str(n) in out["methods"][method]]
            if not xs:
                continue
            ys = [out["methods"][method][str(n)][key]["mean"] for n in xs]
            es = [out["methods"][method][str(n)][key]["std"] for n in xs]
            ax.errorbar(xs, ys, yerr=es, marker=mk, capsize=3, label=mlab)
        ticks = [n for i, n in enumerate(ns) if i == len(ns) - 1 or ns[-1] / n > 1.15]   # drop a tick that would collide with the last one
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([str(n) for n in ticks], fontsize=9)
        ax.tick_params(axis="x", which="minor", labelbottom=False)
        ax.set_xlabel("training samples N"); ax.set_ylabel(lab); ax.grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=8)
    capped = sorted({f"{m} N={n}" for m, d in out["methods"].items() for n, r in d.items()
                     if any(s["epochs"] >= out.get("epoch_cap", 10 ** 9) for s in r["seeds"])})
    cap_note = f"; runs at the {out.get('epoch_cap')}-epoch cap: {', '.join(capped)}" if capped else ""
    fig.suptitle(f"Learning curve on valid Palace samples (held-out N = {out['n_test']}, error bars = std over seeds"
                 f"{'; ' + str(out['epoch_cap']) + '-epoch cap, early stopping on val MSE' if out.get('epoch_cap') else ''})"
                 + (f"\n{cap_note[2:]}" if capped else ""), fontsize=10)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "learning_curve.png", dpi=150); plt.close(fig)
    print("[LC] wrote", FIGS / "learning_curve.png")


if __name__ == "__main__":
    main()
