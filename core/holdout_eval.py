"""Fidelity check on the medium-mesh holdouts: model (trained on coarse-mesh data) vs coarse Palace vs medium Palace
on the SAME designs.

    python core/holdout_eval.py        -> families/<id>/results/holdout_fidelity.json, .md, figs/holdout_fidelity.png

Inputs: data/holdout/hNNNN.npz (medium mesh, scale 1.4), data/holdout_coarse/hNNNN.npz (coarse twins, scale 1.8,
same settings as the training data; built by `python build_case.py --id hNNNN --name hcNNNN --scale 1.8 ...` and
collected with collect.py), families/<id>/ckpt/stageB_seed*.pt (the final stage-B ensemble).
Metrics (eval.per_sample_metrics, live samples): passband |S21| dB, |f0 err| MHz (centroid), |S11| lin, for
  model -> medium (the number that matters), coarse -> medium (the mesh bias alone), model -> coarse (the fit).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D                     # noqa: E402
import eval as E                     # noqa: E402
import prior_wrap as PW              # noqa: E402
import train as T                    # noqa: E402
from trainer import TensorSet, predict   # noqa: E402

from paths import DATA, FIGS, RESULTS  # noqa: E402


def load_set(folder):
    return D.load_palace_samples(DATA / folder)


def summarize(f, s_ref, s_cmp):
    per = E.per_sample_metrics(f, s_ref, s_cmp)
    live = per["live"]
    return {"n": int(live.sum()), "pb_db_mean": float(np.mean(per["pb_db_mean"][live])),
            "pb_db_max": float(np.max(per["pb_db_max"][live])), "f0_err_mhz": float(np.mean(np.abs(per["f0_err_mhz"][live]))),
            "f0_err_mhz_max": float(np.max(np.abs(per["f0_err_mhz"][live]))), "s11_lin": float(np.mean(per["s11_lin"][live])),
            "per_sample_pb_db": [float(x) for x in per["pb_db_mean"]]}


def main():
    med = load_set("holdout")
    coarse = load_set("holdout_coarse") if (DATA / "holdout_coarse").exists() else None
    class A: pass
    args = A(); args.n_basis, args.branch_hidden, args.trunk_hidden = 64, 96, 64
    for k, v in T.DEFAULTS.items():
        setattr(args, k, v)
    s_prior = PW.s_prior_batch(med.p, med.f)
    ts = TensorSet.from_numpy(med.p, med.f, med.s, s_prior)
    preds = []
    for k in range(T.DEFAULTS["seeds"]):
        ck = T.CKPT / f"stageB_seed{k}.pt"
        if not ck.exists():
            break
        m = T.load_ckpt(ck, args, med.param_names)
        preds.append(predict(m, ts).numpy())
    model = np.mean(preds, axis=0)
    out = {"ids": list(med.ids), "n_seeds": len(preds), "medium_mesh": "scale 1.4", "coarse_mesh": "scale 1.8",
           "model_vs_medium": summarize(med.f, med.s, model), "prior_vs_medium": summarize(med.f, med.s, s_prior)}
    if coarse is not None:
        twin = lambda i: "hc" + i[1:]                     # h0000 -> hc0000
        order = [list(coarse.ids).index(twin(i)) for i in med.ids if twin(i) in coarse.ids]
        keep = [j for j, i in enumerate(med.ids) if twin(i) in coarse.ids]
        sc = coarse.s[order]
        if len(coarse.f) != len(med.f) or not np.allclose(coarse.f, med.f):
            raise SystemExit("coarse and medium holdouts on different grids")
        out["coarse_vs_medium"] = summarize(med.f, med.s[keep], sc)
        out["model_vs_coarse"] = summarize(med.f, sc, model[keep])
        out["prior_vs_coarse"] = summarize(med.f, sc, s_prior[keep])
    (RESULTS / "holdout_fidelity.json").write_text(json.dumps(out, indent=1))
    rows = ["| comparison | N live | passband |S21| dB mean / max | |f0 err| MHz mean / max | |S11| lin |", "|---|---|---|---|---|"]
    for key in ("model_vs_medium", "coarse_vs_medium", "model_vs_coarse", "prior_vs_medium", "prior_vs_coarse"):
        if key in out:
            r = out[key]
            rows.append(f"| {key.replace('_', ' ')} | {r['n']} | {r['pb_db_mean']:.2f} / {r['pb_db_max']:.1f} | {r['f0_err_mhz']:.0f} / {r['f0_err_mhz_max']:.0f} | {r['s11_lin']:.3f} |")
    md = ("# Fidelity holdouts: medium-mesh Palace (scale 1.4) vs the coarse-mesh-trained model\n\n"
          f"{len(med.ids)} holdout designs ({', '.join(med.ids)}), stage-B ensemble of {len(preds)} seeds, centroid f0.\n\n" + "\n".join(rows) + "\n")
    (RESULTS / "holdout_fidelity.md").write_text(md)
    print(md)
    # figure: per-sample passband error, model->medium vs coarse->medium
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(med.ids))
    ax.bar(x - 0.2, out["model_vs_medium"]["per_sample_pb_db"], 0.4, label="model (coarse-trained) vs medium Palace")
    if "coarse_vs_medium" in out:
        ax.bar(x[keep] + 0.2, out["coarse_vs_medium"]["per_sample_pb_db"], 0.4, label="coarse Palace vs medium Palace")
    ax.set_xticks(x); ax.set_xticklabels(med.ids, rotation=45, fontsize=8); ax.set_ylabel("passband |S21| error (dB)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    ax.set_title("Fidelity holdouts: what the mesh level costs vs what the model misses")
    fig.tight_layout(); fig.savefig(FIGS / "holdout_fidelity.png", dpi=150); plt.close(fig)
    print("wrote", FIGS / "holdout_fidelity.png")


if __name__ == "__main__":
    main()
