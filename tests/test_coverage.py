"""coverage.json: consistent with families.json, and the generated docs/COVERAGE.md and web/coverage.json are current.

    python -m pytest tests/test_coverage.py -q
"""
import copy
import sys

import registry

sys.path.insert(0, str(registry.REPO / "tools"))
import make_coverage as mc  # noqa: E402


def test_coverage_is_consistent():
    assert mc.problems(mc.load()) == []


def test_outputs_are_current():
    cov = mc.load()
    assert mc.OUT_MD.read_text(encoding="utf-8") == mc.markdown(cov)
    assert mc.OUT_WEB.read_text(encoding="utf-8") == mc.web_json(cov)


def test_checks_catch_errors():
    cov = mc.load()
    bad = copy.deepcopy(cov)
    bad["devices"][0]["blocks"].append("no_such_block")
    bad["devices"].append(dict(bad["devices"][1], id=bad["devices"][1]["id"]))
    msgs = "\n".join(mc.problems(bad))
    assert "unknown block 'no_such_block'" in msgs and "duplicate device id" in msgs
    gone = copy.deepcopy(cov)
    gone["devices"] = [d for d in gone["devices"] if d.get("family_id") != "edge_coupled_bpf"]
    assert any("missing from coverage.json" in m for m in mc.problems(gone))
    fake = copy.deepcopy(cov)
    fake["devices"][1].update(status="surrogate", family_id="hairpin_bpf")
    assert any("not in families.json" in m for m in mc.problems(fake))
