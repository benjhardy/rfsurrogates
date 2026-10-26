"""Device-family registry: families.json + each family's family.py.

    import registry
    registry.family_ids()                  # ['edge_coupled_bpf']
    fam = registry.load_family("edge_coupled_bpf")
    fam["box"], fam["clearance"](p), fam["is_valid"](p), fam["prior"](p, f), fam["passband_metrics"](f, s21)

The family used by the trainer and the tools is RFS_FAMILY (environment variable), default edge_coupled_bpf.

Limit (v0.1): family modules import their siblings by plain name (`import design`, `import prior`), so one process
works with one family. A second family needs these imports made package-relative first.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
FAMILIES_JSON = REPO / "families.json"
DEFAULT_FAMILY = "edge_coupled_bpf"


@lru_cache(maxsize=1)
def _registry() -> dict:
    return json.loads(FAMILIES_JSON.read_text())


def family_ids() -> list[str]:
    return [e["family_id"] for e in _registry()["families"]]


def entry(family_id: str) -> dict:
    for e in _registry()["families"]:
        if e["family_id"] == family_id:
            return e
    raise SystemExit(f"unknown family '{family_id}'; known: {', '.join(family_ids())}")


def current_family() -> str:
    return os.environ.get("RFS_FAMILY", DEFAULT_FAMILY)


def family_dir(family_id: str | None = None) -> Path:
    return REPO / entry(family_id or current_family())["dir"]


def f_grid(family_id: str | None = None) -> np.ndarray:
    lo, hi, step = entry(family_id or current_family())["f_grid_GHz"]
    return np.round(np.arange(lo, hi + 1e-9, step), 4)


@lru_cache(maxsize=None)
def _module(family_id: str):
    e = entry(family_id)
    d = REPO / e["dir"]
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(f"rfs_family_{family_id}", d / e["module"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_family(family_id: str | None = None) -> dict:
    """The registry entry plus the family's box (from samples.json) and functions (from family.py)."""
    family_id = family_id or current_family()
    e = dict(entry(family_id))
    mod = _module(family_id)
    e["f_grid_GHz"] = f_grid(family_id)
    e["param_names"] = tuple(e["param_names"])
    e["box"] = mod.box()
    e["validity"] = mod.validity()
    for fn in ("clearance", "is_valid", "prior", "passband_metrics"):
        e[fn] = getattr(mod, fn)
    return e
