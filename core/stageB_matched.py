"""Ablation: stage B with the stopping rule matched to the scratch runs (full 6000-epoch schedule, no early stop;
best-validation-MSE weights restored at the end), written to separate files so metrics.json stays untouched.

    python core/stageB_matched.py [--seeds 3] [--b-epochs 6000]

Outputs: families/<id>/results/stageB_matched.json, families/<id>/results/preds_stageB_matched.npz, families/<id>/ckpt/stageB_matched_seed*.pt.
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
import data as D                                        # noqa: E402
import train as T                                       # noqa: E402
from trainer import TensorSet, fit, predict             # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    for k, v in T.DEFAULTS.items():
        if not isinstance(v, tuple) and k != "seeds":
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=(6000 if k == "b_epochs" else v))
    args = ap.parse_args(argv)
    args.prior_npz = None; args.palace_dir = None; args.all_substrates = True; args.quiet = True
    ds, s_prior, gname = T.palace_sets(args, "primary")
    full = TensorSet.from_numpy(ds.p, ds.f, ds.s, s_prior)
    tr, va, te = D.split_indices(len(ds), seed=args.split_seed)
    out = {"variant": "stage B, matched schedule: epochs = scratch cap, patience = epochs (no early stop), best val-MSE weights restored",
           "epochs": args.b_epochs, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "test_ids": [ds.ids[i] for i in te], "seeds": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    preds = []
    for k in range(args.seeds):
        m = T.load_ckpt(T.CKPT / f"stageA_seed{k}.pt", args, ds.param_names)
        m.switch_prior_on()
        res = fit(m, full.take(tr), full.take(va), epochs=args.b_epochs, lr=args.b_lr, weight_decay=args.b_wd,
                  patience=args.b_epochs, seed=k, lam_passive=args.lam_passive, log_every=10 ** 9, verbose=False)
        pr = predict(m, full.take(te))
        ev = T._metrics(pr, full.take(te))
        T.save_ckpt(m, T.CKPT / f"stageB_matched_seed{k}.pt", ds.param_names, stage="B-matched", seed=k, test=ev)
        out["seeds"][f"seed{k}"] = {"test": ev, "epochs": res.epochs_run, "best_epoch": res.best_epoch, "seconds": res.seconds}
        preds.append(pr)
        print(f"[B-matched] seed {k}: passband {ev['pb_db_mean']:.3f} dB (max {ev['pb_db_max']:.2f}), best epoch {res.best_epoch}, {res.seconds:.0f}s", flush=True)
    preds = torch.stack(preds)
    out["ensemble_test"] = T._metrics(preds.mean(0), full.take(te))
    out["prior_only_test"] = T._metrics(full.s_prior[te], full.take(te))
    T._dump_preds("stageB_matched", ds.subset(te), full.take(te), preds, gname)
    (T.RESULTS / "stageB_matched.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[B-matched] ensemble passband {out['ensemble_test']['pb_db_mean']:.3f} dB | prior {out['prior_only_test']['pb_db_mean']:.3f} dB")


if __name__ == "__main__":
    main()
