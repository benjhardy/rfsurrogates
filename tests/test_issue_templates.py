"""The issue forms in .github/ISSUE_TEMPLATE use only labels from .github/labels.json, and the role options match
the role labels."""
import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
GH = Path(__file__).resolve().parents[1] / ".github"


def _labels():
    return {x["name"] for x in json.loads((GH / "labels.json").read_text())["labels"]}


def _forms():
    return {p.name: yaml.safe_load(p.read_text()) for p in (GH / "ISSUE_TEMPLATE").glob("*.yml") if p.name != "config.yml"}


def test_forms_use_known_labels():
    known = _labels()
    for name, form in _forms().items():
        assert {"name", "description", "body"} <= set(form), name
        assert set(form.get("labels", [])) <= known, (name, set(form["labels"]) - known)
        ids = [b["id"] for b in form["body"] if "id" in b]
        assert len(ids) == len(set(ids)), name


def test_role_options_match_role_labels():
    roles = {n.split(":", 1)[1] for n in _labels() if n.startswith("role:")}
    task = _forms()["task.yml"]
    opts = next(b for b in task["body"] if b.get("id") == "role")["attributes"]["options"]
    assert set(opts) == roles
