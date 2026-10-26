"""Tests for contrib/check_sample.py.

    python -m pytest tests/test_check_sample.py -q

The tests that need data/palace/nominal.npz are skipped when the file is not on disk (data/ is not in git); the
synthetic tests build their sample from prior.py and run everywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contrib"))
sys.path.insert(0, str(ROOT))
import check_sample as C  # noqa: E402
import prior  # noqa: E402

NOMINAL = ROOT / "families" / "edge_coupled_bpf" / "data" / "palace" / "nominal.npz"
needs_nominal = pytest.mark.skipif(not NOMINAL.exists(), reason="data/palace/nominal.npz not on disk")

P_NOMINAL = [0.8833594228971242, 1.0771281780311608, 0.17565304471745427, 0.5895673097186458,
             8.55957705554429, 8.522884937635837, 3.66, 0.508]
FAM = C.load_family("edge_coupled_bpf")
PROV = dict(solver="openems", solver_version="0.0.36", fidelity="other", contributor="tester", licence="CC-BY-4.0",
            mesh_elements=1000, mesh_unknowns=1000, order=0)


def synthetic_npz(path: Path, p=P_NOMINAL, f=None, **extra):
    """A sample made from the prior itself: passes every physics check, prior delta ~ 0."""
    f = FAM["f_grid_GHz"] if f is None else np.asarray(f, float)
    s11, s21 = prior.s_prior(p, f)
    np.savez(path, p=np.asarray(p, float), f=f, S11=s11, S21=s21, param_names=np.asarray(FAM["param_names"]), **extra)
    return path


# ----------------------------------------------------------------------------- real data
@needs_nominal
def test_nominal_passes(tmp_path):
    rc = C.main([str(NOMINAL), "--family", "edge_coupled_bpf", "--out", str(tmp_path / "row.json")])
    assert rc == 0
    row = json.loads((tmp_path / "row.json").read_text())
    assert row["verdict"] == "PASS" and row["errors"] == [] and row["flags"] == []
    assert row["family_id"] == "edge_coupled_bpf" and row["in_box"] and row["valid_geometry"]
    assert row["solver"] == {**row["solver"], "name": "palace", "version": "0.16", "fidelity": "coarse"}
    assert row["solver"]["mesh"]["order"] == 2 and row["solver"]["mesh"]["unknowns"] > 1e5
    assert row["n_f"] == 201 and row["f_grid_GHz"] == [3.0, 7.0, 0.02]
    assert abs(row["metrics"]["f0_GHz"] - 5.023) < 0.01
    assert row["prior_delta"]["passband_dB_mean_abs_diff"] < C.PRIOR_DB_REVIEW
    assert len(row["content_sha256"]) == 64 and len(row["file_sha256"]) == 64
    assert row["licence"] == "CC-BY-4.0" and row["contributor"]


@needs_nominal
def test_corrupted_copies_fail(tmp_path):
    z = dict(np.load(NOMINAL, allow_pickle=True))
    # (a) NaN in S21
    bad = dict(z); s21 = bad["S21"].copy(); s21[50] = np.nan; bad["S21"] = s21
    np.savez(tmp_path / "nan.npz", **bad)
    r = C.check(tmp_path / "nan.npz")
    assert r["verdict"] == "FAIL" and any("NaN" in e for e in r["errors"])
    # (b) non-passive: |S21| scaled up
    bad = dict(z); bad["S21"] = bad["S21"] * 1.5
    np.savez(tmp_path / "gain.npz", **bad)
    r = C.check(tmp_path / "gain.npz")
    assert r["verdict"] == "FAIL" and any("passive" in e or "|S21|" in e for e in r["errors"])
    # (c) p outside the box
    bad = dict(z); p = bad["p"].copy(); p[0] = 5.0; bad["p"] = p
    np.savez(tmp_path / "box.npz", **bad)
    r = C.check(tmp_path / "box.npz")
    assert r["verdict"] == "FAIL" and any("outside the family box" in e for e in r["errors"])
    # (d) missing key
    bad = {k: v for k, v in z.items() if k != "S11"}
    np.savez(tmp_path / "nokey.npz", **bad)
    r = C.check(tmp_path / "nokey.npz")
    assert r["verdict"] == "FAIL" and any("required key" in e for e in r["errors"])
    # exit code 1 through the CLI
    assert C.main([str(tmp_path / "nan.npz"), "--out", str(tmp_path / "row.json")]) == 1


@needs_nominal
def test_p_argument_must_match_file(tmp_path):
    p = list(np.load(NOMINAL)["p"]); p[2] *= 1.01
    r = C.check(NOMINAL, p=p)
    assert r["verdict"] == "FAIL" and any("differs from the p stored" in e for e in r["errors"])
    assert C.check(NOMINAL, p=list(np.load(NOMINAL)["p"]))["verdict"] == "PASS"


@needs_nominal
def test_touchstone_roundtrip_matches_npz(tmp_path):
    z = np.load(NOMINAL)
    C.write_touchstone(tmp_path / "nominal.s2p", z["f"], z["S11"], z["S21"])
    r_s2p = C.check(tmp_path / "nominal.s2p", p=list(z["p"]), solver="palace", solver_version="0.16",
                    fidelity="coarse", contributor="tester", licence="CC-BY-4.0", mesh_elements=35526, mesh_unknowns=256270, order=2)
    r_npz = C.check(NOMINAL)
    assert r_s2p["verdict"] == "PASS"
    assert r_s2p["content_sha256"] == r_npz["content_sha256"]          # same record, two containers
    assert not r_s2p["reciprocal_assumed"] and not r_s2p["symmetric_assumed"]   # S12 and S22 were in the file
    assert r_s2p["physics"]["reciprocity_max_abs"] < 1e-9 and r_s2p["physics"]["symmetry_max_abs"] < 1e-9


# ----------------------------------------------------------------------------- synthetic (no data needed)
def test_synthetic_prior_sample_passes_and_writes_contract(tmp_path):
    synthetic_npz(tmp_path / "syn.npz")
    out = tmp_path / "syn.contract.npz"
    rc = C.main([str(tmp_path / "syn.npz"), "--solver", "openems", "--solver-version", "0.0.36", "--fidelity", "other",
                 "--contributor", "tester", "--licence", "CC-BY-4.0", "--mesh-elements", "1000", "--mesh-unknowns", "1000",
                 "--order", "0", "--out", str(tmp_path / "row.json"), "--write-npz", str(out)])
    assert rc == 0
    row = json.loads((tmp_path / "row.json").read_text())
    assert row["verdict"] == "PASS" and row["prior_delta"]["passband_dB_mean_abs_diff"] < 1e-6
    z = np.load(out)
    for k in ("p", "f", "S11", "S21", "param_names", "family_id", "solver", "solver_version", "fidelity", "contributor",
              "licence", "content_sha256", "unknowns", "tets", "order"):
        assert k in z.files, k
    assert str(z["family_id"]) == "edge_coupled_bpf" and len(z["f"]) == 201
    assert C.check(out)["content_sha256"] == row["content_sha256"]   # the contract npz re-checks to the same hash


def test_resampling_from_a_finer_wider_grid(tmp_path):
    f = np.round(np.arange(2.5, 7.5 + 1e-9, 0.005), 4)
    synthetic_npz(tmp_path / "fine.npz", f=f)
    r = C.check(tmp_path / "fine.npz", **PROV)
    assert r["verdict"] == "PASS" and r["resampled_from"]["n_f"] == len(f) and r["n_f"] == 201
    assert r["prior_delta"]["passband_dB_mean_abs_diff"] < 0.01


def test_grid_not_covering_family_grid_fails(tmp_path):
    synthetic_npz(tmp_path / "short.npz", f=np.round(np.arange(4.0, 6.0 + 1e-9, 0.02), 4))
    r = C.check(tmp_path / "short.npz", **PROV)
    assert r["verdict"] == "FAIL" and any("does not cover" in e for e in r["errors"])


def test_frequency_in_hz_is_converted(tmp_path):
    f = FAM["f_grid_GHz"]; s11, s21 = prior.s_prior(P_NOMINAL, f)
    np.savez(tmp_path / "hz.npz", p=np.asarray(P_NOMINAL), f=f * 1e9, S11=s11, S21=s21)
    r = C.check(tmp_path / "hz.npz", **PROV)
    assert r["verdict"] == "PASS" and any("converted to GHz" in n for n in r["notes"])


def test_missing_provenance_fails_for_non_legacy(tmp_path):
    synthetic_npz(tmp_path / "noprov.npz")
    r = C.check(tmp_path / "noprov.npz")
    assert r["verdict"] == "FAIL"
    assert any("missing solver" in e for e in r["errors"]) and any("missing contributor" in e for e in r["errors"])


def test_collect_npz_gets_solver_defaults_but_never_a_contributor(tmp_path):
    """A collect.py npz (scale + run, no solver key): solver, version and fidelity come from the reference setup,
    the contributor and the licence must be given."""
    synthetic_npz(tmp_path / "s9999.npz", scale=1.8, run="s9999", tets=35000, unknowns=256000)
    r = C.check(tmp_path / "s9999.npz")
    assert r["verdict"] == "FAIL" and any("missing contributor" in e for e in r["errors"])
    assert not any("missing solver" in e for e in r["errors"])
    r = C.check(tmp_path / "s9999.npz", contributor="tester", licence="CC-BY-4.0")
    assert r["verdict"] == "PASS", r["errors"] + r["flags"]
    assert r["solver"]["name"] == "palace" and r["solver"]["fidelity"] == "coarse" and r["contributor"] == "tester"


def test_touchstone_without_p_fails(tmp_path):
    f = FAM["f_grid_GHz"]; s11, s21 = prior.s_prior(P_NOMINAL, f)
    C.write_touchstone(tmp_path / "x.s2p", f, s11, s21)
    r = C.check(tmp_path / "x.s2p", **PROV)
    assert r["verdict"] == "FAIL" and any("needs --p" in e for e in r["errors"])


def test_invalid_geometry_fails():
    fused = [0.808, 1.170, 0.165, 0.540, 8.699, 8.419, 3.66, 0.508]      # tests/test_build_gate.py FUSED, clearance < 0
    f = FAM["f_grid_GHz"]; s11, s21 = prior.s_prior(fused, f)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        np.savez(Path(d) / "fused.npz", p=np.asarray(fused), f=f, S11=s11, S21=s21)
        r = C.check(Path(d) / "fused.npz", **PROV)
    assert r["verdict"] == "FAIL" and any("validity rule" in e for e in r["errors"]) and r["clearance_mm"] < 0


def test_prior_mismatch_is_review_not_reject(tmp_path):
    """A response shifted by 15 % in frequency is physically fine (passive, reciprocal) but far from the prior."""
    f = FAM["f_grid_GHz"]
    s11, s21 = prior.s_prior(P_NOMINAL, f / 1.15)        # response of the prior at f/1.15: passband moves up 15 %
    np.savez(tmp_path / "shift.npz", p=np.asarray(P_NOMINAL), f=f, S11=s11, S21=s21)
    r = C.check(tmp_path / "shift.npz", **PROV)
    assert r["verdict"] == "NEEDS REVIEW" and "prior_f0_mismatch" in r["flags"] and r["errors"] == []
    assert C.main([str(tmp_path / "shift.npz"), "--solver", "openems", "--solver-version", "x", "--fidelity", "other",
                   "--contributor", "t", "--licence", "CC-BY-4.0", "--out", str(tmp_path / "row.json")]) == 2


def test_non_reciprocal_fails(tmp_path):
    f = FAM["f_grid_GHz"]; s11, s21 = prior.s_prior(P_NOMINAL, f)
    np.savez(tmp_path / "nr.npz", p=np.asarray(P_NOMINAL), f=f, S11=s11, S21=s21, S12=s21 * 0.8, S22=s11)
    r = C.check(tmp_path / "nr.npz", **PROV)
    assert r["verdict"] == "FAIL" and any("reciprocal" in e for e in r["errors"])


def test_out_of_box_can_be_submitted_for_review(tmp_path):
    p = list(P_NOMINAL); p[0] = FAM["box"]["w1"][1] * 1.05        # 5 % above the box edge, still a valid geometry
    synthetic_npz(tmp_path / "oob.npz", p=p)
    assert C.check(tmp_path / "oob.npz", **PROV)["verdict"] == "FAIL"
    r = C.check(tmp_path / "oob.npz", allow_out_of_box=True, **PROV)
    assert r["verdict"] == "NEEDS REVIEW" and "out_of_box" in r["flags"] and r["in_box"] is False


def test_content_hash_is_deterministic_and_sensitive(tmp_path):
    synthetic_npz(tmp_path / "a.npz")
    h1 = C.check(tmp_path / "a.npz", **PROV)["content_sha256"]
    h2 = C.check(tmp_path / "a.npz", **PROV)["content_sha256"]
    assert h1 == h2
    z = dict(np.load(tmp_path / "a.npz")); z["S21"] = z["S21"] * (1 - 1e-6)
    np.savez(tmp_path / "b.npz", **z)
    assert C.check(tmp_path / "b.npz", **PROV)["content_sha256"] != h1
