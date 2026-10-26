"""families.json + family.py: every registered family loads and exposes what the checker and the trainer call.

    python -m pytest tests/test_registry.py -q
"""
import numpy as np

import registry

P_NOMINAL = [0.8833594228971242, 1.0771281780311608, 0.17565304471745427, 0.5895673097186458,
             8.55957705554429, 8.522884937635837, 3.66, 0.508]
P_FUSED = [0.808, 1.170, 0.165, 0.540, 8.699, 8.419, 3.66, 0.508]      # s0060: clearance -0.015 mm


def test_every_family_loads():
    for fid in registry.family_ids():
        fam = registry.load_family(fid)
        assert set(fam["box"]) == set(fam["param_names"]) == set(fam["units"])
        assert (registry.REPO / fam["dir"] / "samples.json").exists()
        assert fam["production_fidelity"] in fam["fidelity_tags"] and fam["holdout_fidelity"] in fam["fidelity_tags"]
        lo, hi, step = registry.entry(fid)["f_grid_GHz"]
        assert fam["f_grid_GHz"][0] == lo and abs(fam["f_grid_GHz"][-1] - hi) < 1e-9
        assert np.allclose(np.diff(fam["f_grid_GHz"]), step)


def test_edge_coupled_bpf_rule_and_prior():
    fam = registry.load_family("edge_coupled_bpf")
    assert fam["f_grid_GHz"].size == 201
    assert abs(fam["clearance"](P_NOMINAL) - 0.061) < 1e-3 and fam["is_valid"](P_NOMINAL)
    assert fam["clearance"](P_FUSED) < 0 and not fam["is_valid"](P_FUSED)
    s11, s21 = fam["prior"](P_NOMINAL, fam["f_grid_GHz"])
    assert s11.shape == s21.shape == (201,)
    f0, bw, il = fam["passband_metrics"](fam["f_grid_GHz"], s21)
    assert 4.8 < f0 < 5.4 and 0.3 < bw < 0.8


def test_unknown_family_exits():
    try:
        registry.entry("no_such_family")
    except SystemExit as e:
        assert "edge_coupled_bpf" in str(e)
    else:
        raise AssertionError("expected SystemExit")
