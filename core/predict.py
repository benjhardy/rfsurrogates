"""Held-out predictions from saved checkpoints, without training (for weights downloaded from Hugging Face).

    python core/predict.py            # families/<id>/results/preds_{stageB,scratch,mlp,prior_only}.npz
    python core/eval.py               # metrics.json and figures from those files

Same data loader, substrate group and split (split_seed) as core/train.py, so the test set is the one the published
metrics were computed on. Methods whose checkpoints are missing are skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D                                   # noqa: E402
import train as T                                  # noqa: E402
from net import PlainMLP                           # noqa: E402
from trainer import TensorSet, predict             # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--palace-dir", type=Path, default=None)
    ap.add_argument("--all-substrates", action="store_true")
    a = ap.parse_args(argv)
    args = argparse.Namespace(**T.DEFAULTS, palace_dir=a.palace_dir, all_substrates=a.all_substrates)
    ds, s_prior, gname = T.palace_sets(args, "primary")
    full = TensorSet.from_numpy(ds.p, ds.f, ds.s, s_prior)
    _, _, te = D.split_indices(len(ds), seed=args.split_seed)
    test = full.take(te)
    no_prior = TensorSet(full.p, full.f, full.s, None).take(te)

    T._dump_preds("prior_only", ds.subset(te), test, full.s_prior[te][None], gname)
    print(f"[predict] prior_only: passband |S21| err {T._metrics(full.s_prior[te], test)['pb_db_mean']:.3f} dB")
    for name, ts in (("stageB", test), ("scratch", no_prior), ("mlp", no_prior)):
        paths = sorted(T.CKPT.glob(f"{name}_seed*.pt"))
        if not paths:
            print(f"[predict] {name}: no checkpoints under {T.CKPT}, skipped")
            continue
        preds = []
        for pth in paths:
            if name == "mlp":
                st = torch.load(pth, map_location="cpu")
                m = PlainMLP(n_freq=st["n_freq"], n_params=ds.n_params)
                m.load_state_dict(st["state_dict"])
            else:
                m = T.load_ckpt(pth, args, ds.param_names)
            preds.append(predict(m, ts))
        preds = torch.stack(preds)
        T._dump_preds(name, ds.subset(te), test, preds, gname)
        print(f"[predict] {name}: {len(paths)} seeds, ensemble passband |S21| err "
              f"{T._metrics(preds.mean(0), test)['pb_db_mean']:.3f} dB on {len(te)} held-out samples")


if __name__ == "__main__":
    main()
