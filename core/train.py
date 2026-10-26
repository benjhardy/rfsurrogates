"""Staged training for the filter foundation-model experiment.

    python core/train.py A          # stage A: pretrain on the prior dataset (prior term OFF)
    python core/train.py B          # stage B: fine-tune on Palace samples (prior term ON), 5 seeds
    python core/train.py baselines  # (i) prior only, (ii) scratch DeepONet, (iii) plain MLP
    python core/train.py C          # stage C: second-substrate adaptation (frozen trunk vs scratch)
    python core/train.py all        # A, B, baselines, C (if a second substrate exists), then eval

Stage A -> B switch (explicit):
    In stage A the net has use_prior=False and is trained to output S_prior(f;p)
    itself, i.e. it learns a 64-function basis (trunk) and a coefficient map
    (branch) that span the response manifold of the prior.  There is no
    "correction" to learn in stage A because prior-minus-prior is zero.
    `DeepONetCorrection.switch_prior_on()` then zeroes the last branch layer and
    the output bias and sets use_prior=True.  At that moment the model outputs
    exactly S_prior; stage B training moves it only where Palace disagrees with
    the prior, expressed in the basis learned in stage A.

Outputs: families/<id>/ckpt/*.pt, families/<id>/results/*.json, families/<id>/results/preds_*.npz
(the eval script reads the npz files and makes the figures).
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

import data as D                               # noqa: E402
import prior_wrap as PW                        # noqa: E402
from net import DeepONetCorrection, PlainMLP   # noqa: E402
from trainer import TensorSet, ensemble_predict, evaluate, fit, predict  # noqa: E402

from paths import CKPT, RESULTS  # noqa: E402
CKPT.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
# leave cores for the Palace queue that runs on the same machine; override with RFS_THREADS
torch.set_num_threads(int(os.environ.get("RFS_THREADS", 3)))

DEFAULTS = dict(
    n_basis=64, branch_hidden=96, trunk_hidden=64,
    # stage A
    a_epochs=60, a_lr=5e-3, a_batch=256, a_patience=10,      # 20k prior samples -> ~4.7k steps, ~14 min/seed at 3 threads
    # stage B / scratch / mlp
    b_epochs=3000, b_lr=1e-3, b_patience=300, b_wd=1e-5,
    # stage C
    c_epochs=600, c_lr=1e-3, c_n=(5, 10, 20), c_min_group=30,
    seeds=5, split_seed=0, lam_passive=1.0,
)


def _save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"[train] wrote {path}")


def new_model(args, use_prior: bool, n_params: int) -> DeepONetCorrection:
    """n_params comes from the dataset (len(param_names)); never hard-coded."""
    return DeepONetCorrection(n_params=n_params, n_basis=args.n_basis, branch_hidden=args.branch_hidden,
                              trunk_hidden=args.trunk_hidden, use_prior=use_prior)


def load_ckpt(path: Path, args, expect_names=None) -> DeepONetCorrection:
    st = torch.load(path, map_location="cpu")
    names = tuple(st.get("param_names", ()))
    if expect_names is not None and names and tuple(expect_names) != names:
        raise ValueError(f"{path.name} was trained with param_names {names}, data has {tuple(expect_names)}; rerun stage A")
    m = new_model(args, use_prior=st.get("use_prior", False), n_params=st.get("n_params", len(names) or 8))
    m.load_state_dict(st["state_dict"])
    m.use_prior = st.get("use_prior", False)
    return m


def save_ckpt(model, path: Path, param_names, **meta):
    torch.save({"state_dict": model.state_dict(), "use_prior": model.use_prior, "n_params": model.n_params,
                "param_names": list(param_names), **meta}, path)


# ---------------------------------------------------------------------------
# stage A
# ---------------------------------------------------------------------------
def stage_a(args):
    ds = D.load_prior_dataset(args.prior_npz)
    js_names = D.samples_json_names()
    if js_names and tuple(js_names) != tuple(ds.param_names):
        raise SystemExit(f"[A] prior dataset param_names {ds.param_names} differ from samples.json {js_names}: "
                         "the prior npz is stale, regenerate it with design.py --stage-a")
    print(f"[A] prior dataset: N={len(ds)} M={len(ds.f)} f=[{ds.f.min():.3f},{ds.f.max():.3f}] GHz "
          f"params={ds.param_names} (n_params={ds.n_params})")
    tr, va, _ = D.split_indices(len(ds), seed=args.split_seed, frac=(0.9, 0.1, 0.0))
    full = TensorSet.from_numpy(ds.p, ds.f, ds.s)
    out = {"param_names": list(ds.param_names), "n_params": ds.n_params}
    for k in range(args.seeds):
        torch.manual_seed(k)
        m = new_model(args, use_prior=False, n_params=ds.n_params)   # prior term OFF: learn the manifold itself
        m.p_std.fit(full.p); m.f_norm.fit(full.f)
        print(f"[A] seed {k}: {m.n_parameters()} parameters")
        res = fit(m, full.take(tr), full.take(va), epochs=args.a_epochs, lr=args.a_lr, batch_size=args.a_batch,
                  patience=args.a_patience, seed=k, lam_passive=args.lam_passive, log_every=20,
                  verbose=not args.quiet)
        ev = evaluate(m, full.take(va))
        save_ckpt(m, CKPT / f"stageA_seed{k}.pt", ds.param_names, stage="A", seed=k, val=ev, n_train=len(tr))
        out[f"seed{k}"] = {"val": ev, "epochs": res.epochs_run, "best_epoch": res.best_epoch, "seconds": res.seconds,
                           "n_parameters": m.n_parameters(), "history": res.history}
        print(f"[A] seed {k} done: val mse {ev['mse']:.3e}, passband |S21| err {ev['pb_db_mean']:.3f} dB, {res.seconds:.0f}s")
    out["n_train"], out["n_val"], out["n_freq"] = int(len(tr)), int(len(va)), int(len(ds.f))
    _save_json(out, RESULTS / "stageA.json")


# ---------------------------------------------------------------------------
# Palace data helpers
# ---------------------------------------------------------------------------
def palace_sets(args, group: str = "primary"):
    """Load Palace samples, pick a substrate group, attach prior predictions."""
    ds = D.load_palace_samples(args.palace_dir)
    groups = D.substrate_groups(ds)
    order = sorted(groups, key=lambda k: -len(groups[k]))
    print(f"[data] {len(ds)} Palace samples, substrate groups: " + ", ".join(f"{k}:{len(groups[k])}" for k in order))
    if args.all_substrates or len(order) == 1:
        idx = np.arange(len(ds)); gname = "all"
    elif group == "primary":
        idx = groups[order[0]]; gname = order[0]
    elif group == "secondary":
        second = [k for k in order[1:] if len(groups[k]) >= args.c_min_group]
        if not second:
            return None, None, None
        idx = groups[second[0]]; gname = second[0]
    else:
        idx = groups[group]; gname = group
    sub = ds.subset(idx)
    t0 = time.time()
    s_prior = PW.s_prior_batch(sub.p, sub.f)
    prior_ms = 1e3 * (time.time() - t0) / max(1, len(sub))
    print(f"[data] group '{gname}': {len(sub)} samples; prior eval {prior_ms:.2f} ms/sample")
    return sub, s_prior, gname


def _metrics(pred, ts: TensorSet):
    from net import passband_db_error, passband_mask
    mse = torch.mean((pred - ts.s) ** 2).item()
    pbm, pbx = passband_db_error(pred, ts.s, passband_mask(ts.s))
    return {"mse": mse, "pb_db_mean": pbm.item(), "pb_db_max": pbx.item()}


def _dump_preds(name, ds_test, ts_test, preds_all, gname, extra=None):
    mean = preds_all.mean(0); std = preds_all.std(0, unbiased=False) if preds_all.shape[0] > 1 else torch.zeros_like(mean)
    np.savez(RESULTS / f"preds_{name}.npz", p=ds_test.p, f=ds_test.f, s_true=ds_test.s,
             s_prior=ts_test.s_prior.numpy() if ts_test.s_prior is not None else np.zeros_like(ds_test.s),
             s_pred=mean.numpy(), s_std=std.numpy(), s_pred_seeds=preds_all.numpy(),
             ids=np.array(ds_test.ids), wall_s=ds_test.wall_s if ds_test.wall_s is not None else np.full(len(ds_test), np.nan),
             group=gname, param_names=np.array(ds_test.param_names), **(extra or {}))


def _query_timing(model, ts: TensorSet, reps: int = 20):
    """ms per query for batch 1 and for the whole test batch."""
    model.eval()
    with torch.no_grad():
        one = ts.take([0])
        for _ in range(3):
            predict(model, one)
        t0 = time.perf_counter()
        for _ in range(reps):
            predict(model, one)
        ms_one = 1e3 * (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            predict(model, ts)
        ms_batch = 1e3 * (time.perf_counter() - t0) / reps / len(ts)
    return {"ms_per_query_batch1": ms_one, "ms_per_query_batched": ms_batch, "batch": len(ts)}


# ---------------------------------------------------------------------------
# stage B
# ---------------------------------------------------------------------------
def stage_b(args):
    ds, s_prior, gname = palace_sets(args, "primary")
    full = TensorSet.from_numpy(ds.p, ds.f, ds.s, s_prior)
    tr, va, te = D.split_indices(len(ds), seed=args.split_seed)
    print(f"[B] split train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    out = {"group": gname, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "test_ids": [ds.ids[i] for i in te], "seeds": {}}
    models, preds = [], []
    for k in range(args.seeds):
        ck = CKPT / f"stageA_seed{k}.pt"
        if not ck.exists():
            raise FileNotFoundError(f"{ck} missing: run stage A first")
        m = load_ckpt(ck, args, ds.param_names)
        m.switch_prior_on()                 # <-- the documented stage A -> B switch
        # sanity: the model now returns exactly the prior
        with torch.no_grad():
            d0 = (predict(m, full.take(tr)) - full.s_prior[tr]).abs().max().item()
        assert d0 < 1e-6, f"after switch, model != prior (max dev {d0})"
        res = fit(m, full.take(tr), full.take(va), epochs=args.b_epochs, lr=args.b_lr, weight_decay=args.b_wd,
                  patience=args.b_patience, seed=k, lam_passive=args.lam_passive, log_every=100,
                  verbose=not args.quiet)
        ev = _metrics(predict(m, full.take(te)), full.take(te))
        timing = _query_timing(m, full.take(te))
        save_ckpt(m, CKPT / f"stageB_seed{k}.pt", ds.param_names, stage="B", seed=k, test=ev, group=gname)
        out["seeds"][f"seed{k}"] = {"test": ev, "epochs": res.epochs_run, "best_epoch": res.best_epoch,
                                    "seconds": res.seconds, "timing": timing, "history": res.history}
        models.append(m); preds.append(predict(m, full.take(te)))
        print(f"[B] seed {k}: test mse {ev['mse']:.3e}  passband |S21| err {ev['pb_db_mean']:.3f} dB (max {ev['pb_db_max']:.2f})")
    preds = torch.stack(preds)
    ens = _metrics(preds.mean(0), full.take(te))
    prior_only = _metrics(full.s_prior[te], full.take(te))
    out["ensemble_test"] = ens
    out["prior_only_test"] = prior_only
    out["palace_wall_s_mean"] = float(np.nanmean(ds.wall_s)) if ds.wall_s is not None else None
    _dump_preds("stageB", ds.subset(te), full.take(te), preds, gname)
    _save_json(out, RESULTS / "stageB.json")
    print(f"[B] ensemble test: mse {ens['mse']:.3e}, passband |S21| err {ens['pb_db_mean']:.3f} dB | prior only: {prior_only['pb_db_mean']:.3f} dB")


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def baselines(args):
    ds, s_prior, gname = palace_sets(args, "primary")
    full = TensorSet.from_numpy(ds.p, ds.f, ds.s, s_prior)
    tr, va, te = D.split_indices(len(ds), seed=args.split_seed)
    out = {"group": gname, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}

    # (i) prior only
    out["prior_only"] = {"test": _metrics(full.s_prior[te], full.take(te))}
    _dump_preds("prior_only", ds.subset(te), full.take(te), full.s_prior[te][None], gname)

    # (ii) from-scratch DeepONet, no prior, no stage A
    scratch_tr = TensorSet(full.p, full.f, full.s, None)
    preds, info = [], {}
    for k in range(args.seeds):
        torch.manual_seed(k)
        m = new_model(args, use_prior=False, n_params=ds.n_params)
        m.p_std.fit(full.p); m.f_norm.fit(full.f)
        res = fit(m, scratch_tr.take(tr), scratch_tr.take(va), epochs=args.b_epochs, lr=args.b_lr, weight_decay=args.b_wd,
                  patience=args.b_patience, seed=k, lam_passive=args.lam_passive, log_every=100, verbose=not args.quiet)
        pr = predict(m, scratch_tr.take(te)); preds.append(pr)
        ev = _metrics(pr, full.take(te))
        info[f"seed{k}"] = {"test": ev, "epochs": res.epochs_run, "seconds": res.seconds, "timing": _query_timing(m, scratch_tr.take(te))}
        save_ckpt(m, CKPT / f"scratch_seed{k}.pt", ds.param_names, stage="scratch", seed=k)
        print(f"[scratch] seed {k}: passband |S21| err {ev['pb_db_mean']:.3f} dB")
    preds = torch.stack(preds)
    info["ensemble_test"] = _metrics(preds.mean(0), full.take(te))
    out["scratch_deeponet"] = info
    _dump_preds("scratch", ds.subset(te), full.take(te), preds, gname)

    # (iii) plain MLP on the frequency grid
    preds, info = [], {}
    for k in range(args.seeds):
        torch.manual_seed(k)
        m = PlainMLP(n_freq=len(ds.f), n_params=ds.n_params)
        m.p_std.fit(full.p)
        res = fit(m, scratch_tr.take(tr), scratch_tr.take(va), epochs=args.b_epochs, lr=args.b_lr, weight_decay=args.b_wd,
                  patience=args.b_patience, seed=k, lam_passive=args.lam_passive, log_every=100, verbose=not args.quiet)
        pr = predict(m, scratch_tr.take(te)); preds.append(pr)
        ev = _metrics(pr, full.take(te))
        info[f"seed{k}"] = {"test": ev, "epochs": res.epochs_run, "seconds": res.seconds, "n_parameters": m.n_parameters(),
                            "timing": _query_timing(m, scratch_tr.take(te))}
        torch.save({"state_dict": m.state_dict(), "n_freq": len(ds.f)}, CKPT / f"mlp_seed{k}.pt")
        print(f"[mlp] seed {k}: passband |S21| err {ev['pb_db_mean']:.3f} dB")
    preds = torch.stack(preds)
    info["ensemble_test"] = _metrics(preds.mean(0), full.take(te))
    out["plain_mlp"] = info
    _dump_preds("mlp", ds.subset(te), full.take(te), preds, gname)
    _save_json(out, RESULTS / "baselines.json")


# ---------------------------------------------------------------------------
# stage C: adaptation to a second substrate
# ---------------------------------------------------------------------------
def stage_c(args):
    ds, s_prior, gname = palace_sets(args, "secondary")
    if ds is None:
        msg = {"status": "pending", "reason": f"no second substrate group with >= {args.c_min_group} Palace samples"}
        _save_json(msg, RESULTS / "stageC.json")
        print("[C] pending:", msg["reason"])
        return
    full = TensorSet.from_numpy(ds.p, ds.f, ds.s, s_prior)
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(len(ds))
    n_pool = max(args.c_n)
    pool, te = perm[:n_pool], perm[n_pool:]
    print(f"[C] group '{gname}': training pool {len(pool)}, test {len(te)}")
    out = {"group": gname, "n_test": int(len(te)), "n_samples": list(args.c_n), "test_ids": [ds.ids[i] for i in te],
           "prior_only": _metrics(full.s_prior[te], full.take(te)), "adapt": {}, "scratch": {}}
    for n in args.c_n:
        tr = pool[:n]
        pa, ps = [], []
        for k in range(args.seeds):
            # (i) frozen trunk, branch adapted from the stage-B model
            m = load_ckpt(CKPT / f"stageB_seed{k}.pt", args, ds.param_names)
            m.freeze_trunk(True)
            fit(m, full.take(tr), None, epochs=args.c_epochs, lr=args.c_lr, weight_decay=1e-4, seed=k,
                lam_passive=args.lam_passive, log_every=10 ** 9, verbose=False)
            pa.append(predict(m, full.take(te)))
            # (ii) from scratch, prior ON, nothing frozen, same n samples (standardizer copied from stage A)
            m2 = new_model(args, use_prior=True, n_params=ds.n_params)
            a = torch.load(CKPT / f"stageA_seed{k}.pt", map_location="cpu")["state_dict"]
            m2.p_std.load_state_dict({kk.split(".", 1)[1]: v for kk, v in a.items() if kk.startswith("p_std.")})
            m2.f_norm.load_state_dict({kk.split(".", 1)[1]: v for kk, v in a.items() if kk.startswith("f_norm.")})
            torch.manual_seed(k)
            fit(m2, full.take(tr), None, epochs=args.c_epochs, lr=args.c_lr, weight_decay=1e-4, seed=k,
                lam_passive=args.lam_passive, log_every=10 ** 9, verbose=False)
            ps.append(predict(m2, full.take(te)))
        pa, ps = torch.stack(pa), torch.stack(ps)
        out["adapt"][str(n)] = {"ensemble": _metrics(pa.mean(0), full.take(te)),
                                "seeds": [_metrics(pa[k], full.take(te)) for k in range(args.seeds)]}
        out["scratch"][str(n)] = {"ensemble": _metrics(ps.mean(0), full.take(te)),
                                  "seeds": [_metrics(ps[k], full.take(te)) for k in range(args.seeds)]}
        _dump_preds(f"stageC_adapt_n{n}", ds.subset(te), full.take(te), pa, gname)
        _dump_preds(f"stageC_scratch_n{n}", ds.subset(te), full.take(te), ps, gname)
        print(f"[C] n={n}: adapt {out['adapt'][str(n)]['ensemble']['pb_db_mean']:.3f} dB | scratch "
              f"{out['scratch'][str(n)]['ensemble']['pb_db_mean']:.3f} dB | prior {out['prior_only']['pb_db_mean']:.3f} dB")
    out["status"] = "done"
    _save_json(out, RESULTS / "stageC.json")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["A", "B", "baselines", "C", "all"])
    ap.add_argument("--prior-npz", type=Path, default=None)
    ap.add_argument("--palace-dir", type=Path, default=None)
    ap.add_argument("--all-substrates", action="store_true", help="use every Palace sample in stage B (no substrate grouping)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--quick", action="store_true", help="short runs for smoke tests")
    for k, v in DEFAULTS.items():
        if isinstance(v, tuple):
            ap.add_argument(f"--{k.replace('_', '-')}", type=int, nargs="+", default=list(v))
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args(argv)
    if args.quick:
        args.a_epochs, args.b_epochs, args.c_epochs, args.seeds = 5, 20, 10, 2
    stages = ["A", "B", "baselines", "C"] if args.stage == "all" else [args.stage]
    for s in stages:
        {"A": stage_a, "B": stage_b, "baselines": baselines, "C": stage_c}[s](args)
    if args.stage == "all":
        import eval as E
        E.main([])


if __name__ == "__main__":
    main()
