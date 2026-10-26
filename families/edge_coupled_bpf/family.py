"""edge_coupled_bpf: the functions that registry.load_family() exposes for this family.

Box and validity rule from samples.json, clearance rule from design.py, prior and passband metrics from prior.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import design  # noqa: E402
import prior as _prior  # noqa: E402

_S = json.loads((HERE / "samples.json").read_text())
_W50 = design.w50_for(_S["centre"][6], _S["centre"][7])


def box() -> dict:
    return {k: (float(v[0]), float(v[1])) for k, v in _S["box"].items()}


def validity() -> dict:
    return {"rule": _S["validity"]["rule"], "clearance": _S["validity"]["clearance"]}


def clearance(p) -> float:
    return design.clearance(p, _W50)


def is_valid(p) -> bool:
    return design.is_valid(p, design.C_MIN, _W50)


def prior(p, f):
    return _prior.s_prior(p, f)


def passband_metrics(f, s21):
    return _prior.passband_metrics(f, s21)
