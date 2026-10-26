"""Thin wrapper around the family's prior.py (families/<family_id>/prior.py, the analytic prior).

The wrapper never reimplements the prior.  It imports `prior.py` from the
family folder and exposes one calling convention:

    s11, s21 = s_prior(p, f_ghz)          # p: (P,)  -> complex (M,), (M,)
    S        = s_prior_batch(P, f_ghz)    # P: (N,P) -> real (N,4,M) in net.pack layout

p has whatever length and order the data files declare in `param_names`
(currently w1, w2, g1, g2, l1, l2, eps_r, h), lengths in the unit prior.py uses
(the npz files carry p in the same units, so no conversion is done here) and f
in GHz.  Pass `param_names` only if prior.py takes keyword arguments per parameter.

prior.py's exact function name / signature is discovered at import time from a
list of candidates (`CANDIDATE_FUNCS`).  If it is not found, set the module
variable `PRIOR_FN` by hand or pass `fn=` to `s_prior`.  The unit test
`tests/test_prior_wrap.py` checks the wrapper against prior.py directly.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FAMILY_DIR  # noqa: E402

ROOT = FAMILY_DIR                                      # families/<family_id>
PRIOR_PATH = Path(os.environ.get("RFS_PRIOR", ROOT / "prior.py"))

CANDIDATE_FUNCS = (
    "s_prior", "S_prior", "sparams", "s_params", "sparam", "s_parameters",
    "prior", "predict", "evaluate", "response", "filter_response", "simulate",
    "compute", "model", "run",
)

PRIOR_FN: Optional[Callable] = None     # resolved lazily


@lru_cache(maxsize=1)
def load_prior_module():
    if not PRIOR_PATH.exists():
        raise ImportError(f"prior.py not found at {PRIOR_PATH}")
    spec = importlib.util.spec_from_file_location("rfs_prior", PRIOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    # prior.py may import siblings (design.py etc.) from the family folder
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    return mod


def resolve_fn() -> Callable:
    global PRIOR_FN
    if PRIOR_FN is not None:
        return PRIOR_FN
    mod = load_prior_module()
    for name in CANDIDATE_FUNCS:
        fn = getattr(mod, name, None)
        if callable(fn):
            PRIOR_FN = fn
            return fn
    pub = [n for n, v in vars(mod).items() if callable(v) and not n.startswith("_")]
    raise AttributeError(f"no known prior function in {PRIOR_PATH}; callables: {pub}")


def _as_complex_pair(out) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize whatever prior.py returns to (S11, S21) complex arrays (M,)."""
    if isinstance(out, dict):
        keys = {k.lower(): k for k in out}
        s11 = out[keys.get("s11")]
        s21 = out[keys.get("s21")]
        return np.asarray(s11, dtype=complex), np.asarray(s21, dtype=complex)
    if isinstance(out, (tuple, list)):
        if len(out) == 2:
            return np.asarray(out[0], dtype=complex), np.asarray(out[1], dtype=complex)
        if len(out) >= 3:                     # maybe (f, S11, S21)
            return np.asarray(out[-2], dtype=complex), np.asarray(out[-1], dtype=complex)
    arr = np.asarray(out)
    if arr.ndim == 3 and arr.shape[-2:] == (2, 2):          # (M,2,2)
        return arr[:, 0, 0].astype(complex), arr[:, 1, 0].astype(complex)
    if arr.ndim == 3 and arr.shape[:2] == (2, 2):           # (2,2,M)
        return arr[0, 0].astype(complex), arr[1, 0].astype(complex)
    if arr.ndim == 2 and arr.shape[0] == 2:                 # (2,M) = [S11, S21]
        return arr[0].astype(complex), arr[1].astype(complex)
    if arr.ndim == 2 and arr.shape[1] == 2:                 # (M,2)
        return arr[:, 0].astype(complex), arr[:, 1].astype(complex)
    raise TypeError(f"cannot interpret prior output of type {type(out)} shape {getattr(arr, 'shape', None)}")


def _call(fn: Callable, p: np.ndarray, f_ghz: np.ndarray, param_names=None):
    """Try the plausible signatures of prior.py's function (any design-vector length)."""
    p = np.asarray(p, dtype=float).reshape(-1)
    f_ghz = np.asarray(f_ghz, dtype=float).reshape(-1)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    names = [q.name for q in params]
    # (a) prior(p, f)   (b) prior(f, p)   (c) prior(w, g1, ..., h, f)   (d) prior(**dict, f=)
    try:
        if len(params) >= 2 and names[0] in ("p", "params", "x", "design", "theta"):
            return fn(p, f_ghz)
        if len(params) >= 2 and names[0] in ("f", "freq", "f_ghz", "frequency", "freqs"):
            return fn(f_ghz, p)
        if len(params) >= p.size + 1:              # one positional argument per design parameter, then f
            return fn(*p, f_ghz)
        if len(params) >= 2:
            return fn(p, f_ghz)
    except TypeError:
        pass
    kw = {}
    for k, v in zip(param_names or (), p):
        if k in names:
            kw[k] = v
    for fname in ("f", "freq", "f_ghz", "frequency", "freqs"):
        if fname in names:
            kw[fname] = f_ghz
            break
    return fn(**kw)


def s_prior(p, f_ghz, fn: Optional[Callable] = None, param_names=None) -> Tuple[np.ndarray, np.ndarray]:
    """S11, S21 (complex, shape (M,)) of the prior for one design p (any length) at frequencies f_ghz."""
    fn = fn or resolve_fn()
    return _as_complex_pair(_call(fn, p, f_ghz, param_names))


def s_prior_batch(P, f_ghz, fn: Optional[Callable] = None, param_names=None) -> np.ndarray:
    """Real (N,4,M) array in the net.pack layout: [S11 re, S11 im, S21 re, S21 im]."""
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P[None]
    f_ghz = np.asarray(f_ghz, dtype=float).reshape(-1)
    out = np.empty((P.shape[0], 4, f_ghz.size), dtype=np.float32)
    for i, p in enumerate(P):
        s11, s21 = s_prior(p, f_ghz, fn, param_names)
        out[i, 0], out[i, 1] = s11.real, s11.imag
        out[i, 2], out[i, 3] = s21.real, s21.imag
    return out


def pack_np(s11: np.ndarray, s21: np.ndarray) -> np.ndarray:
    """Complex (N,M),(N,M) -> real (N,4,M)."""
    s11 = np.asarray(s11, dtype=complex)
    s21 = np.asarray(s21, dtype=complex)
    if s11.ndim == 1:
        s11, s21 = s11[None], s21[None]
    return np.stack([s11.real, s11.imag, s21.real, s21.imag], axis=1).astype(np.float32)


if __name__ == "__main__":
    fn = resolve_fn()
    print("prior.py:", PRIOR_PATH)
    print("function:", fn.__name__, inspect.signature(fn))
