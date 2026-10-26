"""Loaders for the family data files (read-only; formats in docs/DATA_CONTRACT.md).

* prior dataset : one npz with p (N,P), f (M), S11, S21 complex (N,M), param_names (P)
* Palace samples: data/palace/<id>.npz with p (P), f (M), S11, S21 (M), wall_s, substrate;
                  samples.json carries param_names and the box

The design-vector length P and the parameter names are read from the files
(`param_names` key in the npz or samples.json); nothing here hard-codes them.
Current data side: (w1, w2, g1, g2, l1, l2, eps_r, h).  Frequencies are
returned in GHz.  S is returned in the real (N,4,M) layout of net.pack.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, FAMILY_DIR  # noqa: E402

ROOT = FAMILY_DIR
PRIOR_DIR = DATA
PALACE_DIR = DATA / "palace"
SAMPLES_JSON = FAMILY_DIR / "samples.json"
FALLBACK_PARAM_NAMES = ("w1", "w2", "g1", "g2", "l1", "l2", "eps_r", "h")

ALIASES = {
    "p": ("p", "params", "x", "design", "theta"),
    "f": ("f", "freq", "f_ghz", "f_hz", "frequency", "freqs"),
    "S11": ("s11", "S11"),
    "S21": ("s21", "S21"),
    "wall": ("wall_s", "wall", "wall_time", "wall_time_s", "time_s", "elapsed_s", "elapsed", "runtime_s", "t_wall"),
    "substrate": ("substrate", "substrate_id", "subst", "stack"),
    "param_names": ("param_names", "params_names", "names", "p_names"),
}


def _get(npz, key: str, required: bool = True):
    keys = {k.lower(): k for k in npz.files}
    for a in ALIASES[key]:
        if a.lower() in keys:
            return npz[keys[a.lower()]]
    if required:
        raise KeyError(f"npz has no key for '{key}' (have {list(npz.files)})")
    return None


def to_ghz(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    if f.max() > 1e5:          # Hz
        return f / 1e9
    if f.max() > 1e2:          # MHz
        return f / 1e3
    return f                   # already GHz


def pack_np(s11, s21) -> np.ndarray:
    s11 = np.asarray(s11, dtype=complex)
    s21 = np.asarray(s21, dtype=complex)
    if s11.ndim == 1:
        s11, s21 = s11[None], s21[None]
    return np.stack([s11.real, s11.imag, s21.real, s21.imag], axis=1).astype(np.float32)


def _complex(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if np.iscomplexobj(a):
        return a
    if a.ndim >= 1 and a.shape[-1] == 2:       # (..., re/im)
        return a[..., 0] + 1j * a[..., 1]
    return a.astype(complex)


def _names_from_npz(z, n: int) -> Tuple[str, ...]:
    names = _get(z, "param_names", required=False)
    if names is not None and len(names) == n:
        return tuple(str(x) for x in np.asarray(names).reshape(-1))
    js = samples_json_names()
    if js is not None and len(js) == n:
        return js
    if n == len(FALLBACK_PARAM_NAMES):
        return FALLBACK_PARAM_NAMES
    return tuple(f"p{i}" for i in range(n))


def samples_json_names(path: Optional[Path] = None) -> Optional[Tuple[str, ...]]:
    js = _read_samples_json(path)
    if js and isinstance(js.get("param_names"), list):
        return tuple(str(x) for x in js["param_names"])
    return None


# ---------------------------------------------------------------------------
@dataclass
class SDataset:
    p: np.ndarray                 # (N,P)
    f: np.ndarray                 # (M,) GHz, common grid
    s: np.ndarray                 # (N,4,M) real layout
    param_names: Tuple[str, ...] = FALLBACK_PARAM_NAMES
    ids: List[str] = field(default_factory=list)
    wall_s: Optional[np.ndarray] = None   # (N,) Palace wall time per sample, if known
    substrate: Optional[np.ndarray] = None

    def __len__(self):
        return self.p.shape[0]

    @property
    def n_params(self) -> int:
        return self.p.shape[1]

    def col(self, name: str) -> np.ndarray:
        return self.p[:, self.param_names.index(name)]

    def subset(self, idx) -> "SDataset":
        idx = np.asarray(idx)
        return SDataset(
            p=self.p[idx], f=self.f, s=self.s[idx], param_names=self.param_names,
            ids=[self.ids[i] for i in idx] if self.ids else [],
            wall_s=None if self.wall_s is None else self.wall_s[idx],
            substrate=None if self.substrate is None else self.substrate[idx],
        )


def find_prior_dataset(path: Optional[Path] = None) -> Optional[Path]:
    if path is not None:
        return Path(path)
    cands = sorted(PRIOR_DIR.glob("*.npz")) if PRIOR_DIR.exists() else []
    cands = [c for c in cands if "palace" not in c.name.lower()]
    pref = [c for c in cands if "prior" in c.name.lower() or "stage" in c.name.lower()]
    return (pref or cands or [None])[0]


def load_prior_dataset(path: Optional[Path] = None) -> SDataset:
    path = find_prior_dataset(path)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"prior dataset npz not found under {PRIOR_DIR}")
    z = np.load(path, allow_pickle=True)
    p = np.asarray(_get(z, "p"), dtype=np.float32)
    f = to_ghz(_get(z, "f"))
    s11 = _complex(_get(z, "S11"))
    s21 = _complex(_get(z, "S21"))
    sub = _get(z, "substrate", required=False)
    return SDataset(p=p, f=f.astype(np.float32), s=pack_np(s11, s21), param_names=_names_from_npz(z, p.shape[1]),
                    ids=[f"prior{i}" for i in range(p.shape[0])],
                    substrate=None if sub is None else np.asarray(sub).astype(str))


def load_palace_samples(folder: Optional[Path] = None, f_grid: Optional[np.ndarray] = None) -> SDataset:
    """Load every data/palace/*.npz; interpolate onto a common grid if the grids differ."""
    folder = Path(folder) if folder else PALACE_DIR
    files = sorted(folder.glob("*.npz")) if folder.exists() else []
    if not files:
        raise FileNotFoundError(f"no Palace npz files under {folder}")
    P, S11, S21, F, ids, walls, subs = [], [], [], [], [], [], []
    names = None
    for fp in files:
        try:
            z = np.load(fp, allow_pickle=True)
            p = np.asarray(_get(z, "p"), dtype=float).reshape(-1)
            f = to_ghz(_get(z, "f"))
            s11 = _complex(_get(z, "S11")).reshape(-1)
            s21 = _complex(_get(z, "S21")).reshape(-1)
        except Exception as e:                    # skip half-written files
            print(f"[data] skip {fp.name}: {e}")
            continue
        if names is None:
            names = _names_from_npz(z, p.size)
        if p.size != len(names) or s11.size != f.size or s21.size != f.size:
            print(f"[data] skip {fp.name}: shape mismatch p{p.shape} f{f.shape} s11{s11.shape}")
            continue
        if not (np.all(np.isfinite(s11)) and np.all(np.isfinite(s21))):
            print(f"[data] skip {fp.name}: non-finite S")
            continue
        w = _get(z, "wall", required=False)
        sub = _get(z, "substrate", required=False)
        P.append(p); S11.append(s11); S21.append(s21); F.append(f); ids.append(fp.stem)
        walls.append(float(np.asarray(w).reshape(-1)[0]) if w is not None else np.nan)
        subs.append(str(np.asarray(sub).reshape(-1)[0]) if sub is not None else "")
    if not P:
        raise FileNotFoundError(f"no readable Palace npz files under {folder}")
    if f_grid is None:
        f_grid = F[0]
        same = all(len(fi) == len(f_grid) and np.allclose(fi, f_grid) for fi in F)
        if not same:
            lo = max(fi.min() for fi in F); hi = min(fi.max() for fi in F)
            n = int(np.median([len(fi) for fi in F]))
            f_grid = np.linspace(lo, hi, n)
    s = np.empty((len(P), 4, len(f_grid)), dtype=np.float32)
    for i, (fi, a, b) in enumerate(zip(F, S11, S21)):
        if len(fi) == len(f_grid) and np.allclose(fi, f_grid):
            a_i, b_i = a, b
        else:
            a_i = np.interp(f_grid, fi, a.real) + 1j * np.interp(f_grid, fi, a.imag)
            b_i = np.interp(f_grid, fi, b.real) + 1j * np.interp(f_grid, fi, b.imag)
        s[i] = pack_np(a_i, b_i)[0]
    walls = np.asarray(walls, dtype=float)
    subs = np.asarray(subs)
    return SDataset(p=np.asarray(P, dtype=np.float32), f=np.asarray(f_grid, dtype=np.float32), s=s,
                    param_names=names, ids=ids,
                    wall_s=None if np.all(np.isnan(walls)) else walls,
                    substrate=None if np.all(subs == "") else subs)


# ---------------------------------------------------------------------------
def _read_samples_json(path: Optional[Path] = None) -> Optional[dict]:
    path = Path(path) if path else SAMPLES_JSON
    if not path.exists():
        alt = DATA / "samples.json"
        if alt.exists():
            path = alt
        else:
            return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_box(path: Optional[Path] = None) -> Optional[Dict[str, Tuple[float, float]]]:
    """Return {param: (lo, hi)} from samples.json if it can be found there."""
    js = _read_samples_json(path)
    if not isinstance(js, dict):
        return None
    names = tuple(js.get("param_names") or FALLBACK_PARAM_NAMES)
    box = None
    for k in ("box", "bounds", "ranges", "space", "design_box"):
        if k in js:
            box = js[k]
            break
    if box is None and all(n in js for n in names):
        box = {n: js[n] for n in names}
    if box is None:
        return None
    out = {}
    for n in names:
        v = box.get(n)
        if v is None:
            continue
        if isinstance(v, dict):
            lo = v.get("lo", v.get("min", v.get("low")))
            hi = v.get("hi", v.get("max", v.get("high")))
        else:
            lo, hi = float(v[0]), float(v[1])
        out[n] = (float(lo), float(hi))
    return out or None


def box_from_data(p: np.ndarray, names: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    return {n: (float(p[:, i].min()), float(p[:, i].max())) for i, n in enumerate(names)}


def distance_to_box_edge(p: np.ndarray, box: Dict[str, Tuple[float, float]], names: Sequence[str]) -> np.ndarray:
    """Normalized distance (0 = on an edge, 0.5 = box centre) over the *varying* dims."""
    d = np.full(p.shape[0], 0.5)
    for i, n in enumerate(names):
        if n not in box:
            continue
        lo, hi = box[n]
        if hi - lo <= 0:
            continue
        di = np.minimum(p[:, i] - lo, hi - p[:, i]) / (hi - lo)
        d = np.minimum(d, di)
    return d


def substrate_groups(ds: SDataset, decimals: int = 3) -> Dict[str, np.ndarray]:
    """Group sample indices by the explicit substrate key, else by (eps_r, h)."""
    if ds.substrate is not None:
        keys = ds.substrate
    else:
        e = ds.col("eps_r") if "eps_r" in ds.param_names else np.zeros(len(ds))
        h = ds.col("h") if "h" in ds.param_names else np.zeros(len(ds))
        keys = np.array([f"eps{round(float(a), decimals)}_h{round(float(b), decimals + 2)}" for a, b in zip(e, h)])
    groups = {}
    for k in np.unique(keys):
        groups[str(k)] = np.where(keys == k)[0]
    return groups


def split_indices(n: int, seed: int = 0, frac=(0.8, 0.1, 0.1)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 80/10/10 split by sample index."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    n_va = max(1, n_va) if n >= 3 else n_va
    n_te = n - n_tr - n_va
    if n_te < 1 and n >= 3:
        n_tr -= 1
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
