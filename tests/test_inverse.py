"""Tests for model/inverse.py (need the stage-B checkpoints under model/ckpt; skipped otherwise).

Run:  python -m pytest tests/test_inverse.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import inverse as INV   # noqa: E402

CKPTS = sorted(INV.CKPT.glob("stageB_seed*.pt"))
pytestmark = pytest.mark.skipif(len(CKPTS) < 2, reason="stage-B checkpoints not present (run core/train.py B, or fetch the weights from Hugging Face)")


@pytest.fixture(scope="module")
def sur():
    return INV.Surrogate()


@pytest.fixture(scope="module")
def nominal():
    js = json.loads((INV.FAMILY_DIR / "samples.json").read_text())
    return np.asarray(next(s for s in js["samples"] if s["id"] == "nominal")["p"], float)


def test_recovers_nominal_spec(sur, nominal):
    """A spec equal to the tuned nominal's own predicted f0 / BW is met by a design inside the box with J ~ 0.

    The map p -> (f0, BW) is many-to-one, so the recovered p need not equal the nominal; what must hold is that the
    returned design is in the box, valid, and that its predicted f0 / BW match the spec closely.
    """
    s_mean, _, _ = sur.predict(nominal[None])
    spec0 = INV.Spec(f0=5.0, bw=0.5)                       # only used for the metric definitions here
    m0 = INV.response_metrics(sur.f, s_mean[0], spec0)
    spec = INV.Spec(f0=m0["f0_band_GHz"], bw=m0["bw_GHz"], rl=min(10.0, m0["rl_min_dB"] - 0.5))
    res = INV.solve(spec, sur, n_starts=48, k=4, seed=1, maxiter=100, polish=True, verbose=False)
    best = res["best"]; prob = res["problem"]; m = best["metrics"]
    assert prob.in_box(best["p"])
    assert best["clearance_mm"] > INV.C_MIN
    assert best["J"] < 0.05, best["terms"]
    assert abs(m["f0_GHz"] - spec.f0) < 0.005          # 5 MHz
    assert abs(m["bw_GHz"] - spec.bw) < 0.010          # 10 MHz
    assert m["rl_min_dB"] >= spec.rl - 0.2
    # the nominal itself is a feasible point of this spec: evaluating it must give a small objective too
    J_nom = prob.evaluate(nominal[None])[0][0]
    assert J_nom < 1.0, J_nom
    # ensemble spread is reported and finite
    assert np.isfinite(best["spread"]["passband_S21_std_dB_mean"])


def test_candidates_respect_clearance_and_box(sur):
    """Every returned candidate (not just the best) satisfies clearance > C_MIN and lies inside the box, also with a
    rejection constraint and a spec that pushes toward the narrow-gap edge of the box."""
    spec = INV.Spec(f0=5.5, bw=0.3, rl=10.0, reject=[(6.5, 30.0)])
    res = INV.solve(spec, sur, n_starts=32, k=4, seed=3, maxiter=60, polish=False, verbose=False)
    prob = res["problem"]
    assert len(res["candidates"]) >= 1
    for c in res["candidates"]:
        p = np.asarray(c["p"])
        assert prob.in_box(p)
        assert c["clearance_mm"] > INV.C_MIN
        assert INV.design.is_valid(p)                  # the data agent's own rule
        assert p[6] == pytest.approx(3.66) and p[7] == pytest.approx(0.508)
        assert len(c["metrics"]["reject_dB"]) == 1
    # the starts were all valid as well
    for p in res["starts"]["P"]:
        assert INV.design.is_valid(p) and prob.in_box(p)


def test_parse_reject_and_metrics_shapes(sur):
    assert INV.parse_reject(["6.5:30", "4.0:20"]) == [(6.5, 30.0), (4.0, 20.0)]
    with pytest.raises(SystemExit):
        INV.parse_reject(["6.5"])
    P = INV.Problem(INV.Spec(5.2, 0.5), sur).starts(5, seed=0)
    s_mean, s_seeds, s_prior = sur.predict(P)
    assert s_mean.shape == (5, 4, len(sur.f)) and s_seeds.shape[1:] == (5, 4, len(sur.f)) and s_prior.shape == s_mean.shape
    assert s_seeds.shape[0] == sur.n_seeds
