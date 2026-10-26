"""Generic training loop used by every stage and baseline (CPU PyTorch).

fit(model, train, val, ...) minimises MSE(re/im) + lam * passivity penalty with
Adam + cosine LR, full-batch or minibatch, early stopping on validation MSE.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from net import loss_fn, passband_db_error, passband_mask


@dataclass
class TensorSet:
    """Tensors for one split: p (N,7), f (M,), s (N,4,M) target, s_prior (N,4,M) or None."""
    p: torch.Tensor
    f: torch.Tensor
    s: torch.Tensor
    s_prior: Optional[torch.Tensor] = None

    @staticmethod
    def from_numpy(p, f, s, s_prior=None) -> "TensorSet":
        t = lambda a: None if a is None else torch.as_tensor(np.asarray(a), dtype=torch.float32)
        return TensorSet(t(p), t(f), t(s), t(s_prior))

    def __len__(self):
        return self.p.shape[0]

    def take(self, idx) -> "TensorSet":
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        return TensorSet(self.p[idx], self.f, self.s[idx],
                         None if self.s_prior is None else self.s_prior[idx])


@dataclass
class FitResult:
    history: Dict[str, List[float]] = field(default_factory=lambda: {"epoch": [], "train_mse": [], "val_mse": [], "val_pb_db": [], "lr": []})
    best_val: float = math.inf
    best_epoch: int = -1
    epochs_run: int = 0
    seconds: float = 0.0


def _forward(model, ts: TensorSet, idx=None) -> torch.Tensor:
    if idx is None:
        return model(ts.p, ts.f, ts.s_prior)
    sp = None if ts.s_prior is None else ts.s_prior[idx]
    return model(ts.p[idx], ts.f, sp)


@torch.no_grad()
def evaluate(model, ts: TensorSet):
    model.eval()
    pred = _forward(model, ts)
    mse = torch.mean((pred - ts.s) ** 2).item()
    pb_mean, pb_max = passband_db_error(pred, ts.s, passband_mask(ts.s))
    return {"mse": mse, "pb_db_mean": pb_mean.item(), "pb_db_max": pb_max.item()}


def fit(model, train: TensorSet, val: Optional[TensorSet] = None, *, epochs: int = 2000, lr: float = 2e-3,
        weight_decay: float = 0.0, batch_size: int = 0, lam_passive: float = 1.0, patience: int = 200,
        min_epochs: int = 50, seed: int = 0, log_every: int = 100, verbose: bool = True,
        lr_min_ratio: float = 0.02) -> FitResult:
    """Train in place; restores the best-validation weights (or the final ones without val)."""
    torch.manual_seed(seed)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * lr_min_ratio)
    res = FitResult()
    best_state = copy.deepcopy(model.state_dict())
    n = len(train)
    bs = n if batch_size <= 0 or batch_size >= n else batch_size
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    stale = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g) if bs < n else torch.arange(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = _forward(model, train, idx)
            loss, mse, pen = loss_fn(pred, train.s[idx], lam_passive)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += mse.item() * len(idx); cnt += len(idx)
        sched.step()
        train_mse = tot / cnt
        if val is not None and len(val) > 0:
            ev = evaluate(model, val)
            val_mse, val_pb = ev["mse"], ev["pb_db_mean"]
            if val_mse < res.best_val - 1e-12:
                res.best_val, res.best_epoch, stale = val_mse, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        else:
            val_mse, val_pb = float("nan"), float("nan")
            res.best_val, res.best_epoch = train_mse, ep
        if ep % log_every == 0 or ep == 1:
            res.history["epoch"].append(ep)
            res.history["train_mse"].append(train_mse)
            res.history["val_mse"].append(val_mse)
            res.history["val_pb_db"].append(val_pb)
            res.history["lr"].append(sched.get_last_lr()[0])
            if verbose:
                print(f"  ep {ep:5d}  train_mse {train_mse:.3e}  val_mse {val_mse:.3e}  val_pb_dB {val_pb:.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        res.epochs_run = ep
        if val is not None and len(val) > 0 and ep >= min_epochs and stale >= patience:
            if verbose:
                print(f"  early stop at ep {ep} (best val_mse {res.best_val:.3e} @ ep {res.best_epoch})")
            break
    if val is not None and len(val) > 0:
        model.load_state_dict(best_state)
    res.seconds = time.time() - t0
    return res


@torch.no_grad()
def predict(model, ts: TensorSet) -> torch.Tensor:
    model.eval()
    return _forward(model, ts)


@torch.no_grad()
def ensemble_predict(models, ts: TensorSet):
    """Mean and std (over seeds) of the real (N,4,M) prediction."""
    preds = torch.stack([predict(m, ts) for m in models])          # (S,N,4,M)
    return preds.mean(0), preds.std(0, unbiased=False) if len(models) > 1 else torch.zeros_like(preds[0]), preds
