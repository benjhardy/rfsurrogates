"""The geometry gate in build_case.build: a fused layout (clearance <= 0) must be rejected before any Palace config
is written, a valid one must pass with 5 conductors counted both after the CAD fuse and in the copper mesh.

    python -m pytest tests/test_build_gate.py      (builds two small meshes, ~30 s; do not run while Palace is busy)
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("gmsh")
pytest.importorskip("palace_tools")
import build_case  # noqa: E402
import design  # noqa: E402

VALID = [1.148, 1.400, 0.228, 0.766, 9.844, 9.801, 3.66, 0.508]      # s0005: clearance +0.102 mm
FUSED = [0.808, 1.170, 0.165, 0.540, 8.699, 8.419, 3.66, 0.508]      # s0060: clearance -0.015 mm (w2 part hits the strip)
FEED_FUSED = [0.618, 0.754, 0.228, 0.413, 7.276, 9.801, 3.66, 0.508]  # s0002: feed (W50) hits resonator 1


def test_clearance_rule_matches_examples():
    assert design.clearance(VALID) > 0.05
    assert design.clearance(FUSED) <= 0
    assert design.clearance(FEED_FUSED) <= 0


def test_gate_rejects_fused_before_config(tmp_path):
    for p, what in ((FUSED, "step"), (FEED_FUSED, "feed")):
        with pytest.raises(build_case.GeometryError):
            build_case.build(f"gate_{what}", p, scale=2.4, out_root=tmp_path)
        d = tmp_path / f"gate_{what}"
        assert not (d / "config.json").exists()
        rec = json.loads((d / "build_summary.json").read_text())
        assert rec["status"] == "rejected" and rec["conductors_cad"] == 3


def test_gate_passes_valid_and_records_counts(tmp_path):
    summ = build_case.build("gate_ok", VALID, scale=2.4, out_root=tmp_path)
    assert summ["status"] == "ok" and summ["conductors_cad"] == 5 and summ["conductors_mesh"] == 5
    assert (tmp_path / "gate_ok" / "config.json").exists()
    assert build_case.copper_components(tmp_path / "gate_ok" / "gate_ok.msh") == 5
