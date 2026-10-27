"""Circuit tier of edge_coupled_bpf (families/edge_coupled_bpf/circuit.py).

The section and topology checks are pure Python. The qucsator check skips when circuit-agent-tools or the
simulator is not available (set CIRCUIT_TOOLS_SKIP_SIM=1 to skip it on purpose)."""
import json
import os

import numpy as np
import pytest

import prior

circuit = pytest.importorskip("circuit", reason="families/edge_coupled_bpf/circuit.py")
P_NOM = json.loads((circuit.HERE / "samples.json").read_text())["centre"]
F = np.array([3.0, 4.5, 5.0, 5.5, 7.0])


def _open_far_ends(s4):
    """Two-port (line 1 start, line 2 end) of a 4-port with ports 2 and 3 open (reflection +1)."""
    keep, opn = [0, 3], [1, 2]
    a, b = s4[:, keep][:, :, keep], s4[:, keep][:, :, opn]
    c, d = s4[:, opn][:, :, keep], s4[:, opn][:, :, opn]
    return a + b @ np.linalg.solve(np.eye(2) - d, c)


def test_section_s4_matches_prior_section():
    w1, w2, g1, g2, l1, l2, eps_r, h = P_NOM
    tand = prior.tand_for(eps_r)
    for w, g, l in ((w1, g1, l1), (w2, g2, l2)):
        s4 = circuit.section_s4(F, w, g, l, eps_r, h, tand)
        assert np.allclose(s4, np.transpose(s4, (0, 2, 1)))
        z0e, z0o, ee, eo = prior.coupled_microstrip(w, g, h, eps_r)
        f = F * 1e9
        ee_f, eo_f = prior.kobayashi_eeff(f, ee, eps_r, w, h), prior.kobayashi_eeff(f, eo, eps_r, w, h)
        abcd = prior.section_abcd(f, z0e * np.sqrt(ee / ee_f), z0o * np.sqrt(eo / eo_f),
                                  prior._mode_theta(f, l, ee_f, eps_r, tand), prior._mode_theta(f, l, eo_f, eps_r, tand))
        s11, s21, _, _ = prior.abcd_to_s(abcd)
        two = _open_far_ends(s4)
        assert np.allclose(two[:, 0, 0], s11, atol=1e-12) and np.allclose(two[:, 1, 0], s21, atol=1e-12)


def test_spec_topology(tmp_path):
    counts = {"physical": {"mcline": 4, "mopen": 8, "mstep": 4}, "sblock": {"sblock": 4, "mopen": 8, "mstep": 4},
              "sblock_mlin": {"sblock": 4, "mline": 8}, "prior_blocks": {"sblock": 4}}
    for mode, want in counts.items():
        sp = circuit.spec(P_NOM, mode, f_ghz=F, block_dir=tmp_path / mode)
        got = {}
        for c in sp["components"]:
            got[c["type"]] = got.get(c["type"], 0) + 1
        assert got.pop("port") == 2 and got == want, mode
    sp = circuit.spec(P_NOM, "physical", f_ghz=F)
    steps = {c["name"]: c for c in sp["components"] if c["type"] == "mstep"}
    w50 = circuit.layout(P_NOM)[1]["w50"]
    assert steps["ST0"]["nodes"] == ["p1", "a0"] and steps["ST0"]["w1"] == pytest.approx(w50 * 1e-3)
    assert steps["ST1"]["w2"] == pytest.approx(P_NOM[1] * 1e-3) and steps["ST3"]["nodes"][1] == "p2"


def test_spec_lints_clean(tmp_path):
    lint = pytest.importorskip("circuit_tools.lint")
    for mode in ("physical", "sblock"):
        r = lint.lint(circuit.spec(P_NOM, mode, f_ghz=circuit.F_GHZ, block_dir=tmp_path), {"skip_reference": True})
        assert r["counts"]["error"] == 0, [f for f in r["findings"] if f["severity"] == "error"]


def test_prior_blocks_reproduce_the_prior_in_qucsator(tmp_path):
    if os.environ.get("CIRCUIT_TOOLS_SKIP_SIM") == "1":
        pytest.skip("CIRCUIT_TOOLS_SKIP_SIM=1")
    run = pytest.importorskip("circuit_tools.run")
    try:
        v = run.versions()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no simulator host: {e}")
    if not (v.get("qucsator") or {}).get("available"):
        pytest.skip("qucsator not installed")
    r = circuit.solve(P_NOM, "prior_blocks", name="test", case_dir=tmp_path, check_host=False)
    s11, s21 = prior.s_prior(P_NOM, circuit.F_GHZ)
    assert np.abs(r["S11"] - s11).max() < 1e-9 and np.abs(r["S21"] - s21).max() < 1e-9
