"""tools/contribute.py and tools/verify_batch.py, end to end with a stand-in solver (tests/fake_palace.py).

The stand-in writes the prior's response in Palace's port-S.csv format, so build -> solve -> collect -> check ->
stage -> verify runs without Palace. Needs gmsh and palace-agent-tools (requirements.txt); skipped without them.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "contrib"))
sys.path.insert(0, str(REPO))
import contribute as C  # noqa: E402
import registry  # noqa: E402
import verify_batch as V  # noqa: E402

FAKE = f"{sys.executable} {REPO / 'tests' / 'fake_palace.py'}"
needs_build = pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ("gmsh", "palace_tools")),
    reason="gmsh and palace-agent-tools are needed to build and collect cases")


@pytest.fixture(autouse=True)
def lock_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RFS_LOCK_DIR", str(tmp_path / "lock"))
    monkeypatch.delenv("FAKE_PALACE_MODE", raising=False)


def run(*args) -> int:
    return C.main([str(a) for a in args])


def manifest_p():
    f = REPO / "data_index" / "edge_coupled_bpf" / "manifest.jsonl"
    return np.array([json.loads(x)["p"] for x in f.read_text().splitlines() if x.strip()])


# ---------------------------------------------------------------------------------------------------- hooks

def test_sample_designs_valid_and_new():
    fam = registry.load_family("edge_coupled_bpf")
    ds = np.array(fam["sample_designs"](20, 123456))
    assert ds.shape == (20, 8)
    assert all(fam["is_valid"](p) for p in ds)
    known = manifest_p()
    assert not any(np.any(np.all(np.isclose(known, p, rtol=1e-9, atol=0), axis=1)) for p in ds)


@needs_build
def test_export_geometry(tmp_path):
    fam = registry.load_family("edge_coupled_bpf")
    p = fam["sample_designs"](1, 7)[0]
    js, dxf = fam["export_geometry"](p, tmp_path, "d1")
    spec = json.loads(js.read_text())
    assert spec["units"] == "mm" and len(spec["copper_rectangles"]) == 9 and len(spec["ports"]) == 2
    text = dxf.read_text()
    assert text.count("POLYLINE") == 9 and text.rstrip().endswith("EOF")


# ---------------------------------------------------------------------------------------------------- launcher

def test_lock_is_exclusive_and_stale_lock_is_removed(tmp_path):
    with C.Lock("a"):
        with pytest.raises(C.LauncherBusy):
            with C.Lock("b"):
                pass
    lk = C.lock_dir() / "launcher.lock"
    lk.parent.mkdir(parents=True, exist_ok=True)
    lk.write_text(json.dumps({"pid": 999999, "host": __import__("socket").gethostname()}))
    with C.Lock("c"):                       # the process 999999 does not exist: the lock is stale
        assert json.loads(lk.read_text())["batch"] == "c"
    assert not lk.exists()


@needs_build
def test_end_to_end_stage_and_verify(tmp_path):
    w, inc = tmp_path / "work", tmp_path / "incoming"
    assert run("new", "--n", 3, "--contributor", "tester", "--seed", 31337, "--work", w) == 0
    b = C.load_batch(w / "tester-31337")
    assert b["estimate"]["machine_hours"] and len(b["designs"]) == 3
    assert run("go", "--batch", "tester-31337", "--work", w, "--incoming", inc) == 3     # claim not accepted yet
    assert run("go", "--batch", "tester-31337", "--work", w, "--incoming", inc, "--claim-accepted",
               "--budget-hours", 1, "--runner", FAKE) == 0
    b = C.load_batch(w / "tester-31337")
    assert all(x["state"] == "checked" and x["verdict"] in ("PASS", "NEEDS REVIEW") for x in b["designs"])
    out = inc / "edge_coupled_bpf" / "tester-31337"
    assert sorted(p.name for p in out.iterdir()) == ["batch.json", "records", "rows.jsonl"]
    res = V.verify_batch(out)
    assert res["errors"] == [], res["errors"]
    assert len(res["rows"]) == 3 and all(r[3] for r in res["rows"])

    # tampering is caught
    bad = inc / "edge_coupled_bpf" / "tester-31338"
    shutil.copytree(out, bad)
    bj = json.loads((bad / "batch.json").read_text())
    bj["batch_id"] = "tester-31338"
    (bad / "batch.json").write_text(json.dumps(bj))
    rows = [json.loads(x) for x in (bad / "rows.jsonl").read_text().splitlines()]
    for r in rows:
        r["batch_id"] = "tester-31338"
        r["path"] = r["path"].replace("tester-31337", "tester-31338")
    rows[1]["verdict"] = "PASS" if rows[1]["verdict"] != "PASS" else "NEEDS REVIEW"
    (bad / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    f0 = bad / "records" / f"{rows[0]['id']}.npz"
    f0.write_bytes(f0.read_bytes() + b"\0")
    (bad / "records" / "stray.npz").write_bytes(b"x")
    errs = "\n".join(V.verify_batch(bad)["errors"])
    assert "sha256 differs" in errs
    assert "checker rerun gives" in errs
    assert "record files without a row" in errs
    assert "same record twice" not in errs
    assert V.check_changed(["incoming/edge_coupled_bpf/x/rows.jsonl", "core/train.py"])
    assert not V.check_changed(["incoming/edge_coupled_bpf/x/rows.jsonl"])


@needs_build
def test_timeout_and_failure_are_recorded(tmp_path, monkeypatch):
    w = tmp_path / "work"
    run("new", "--n", 1, "--contributor", "tester", "--seed", 5, "--local", "--work", w)
    monkeypatch.setenv("FAKE_PALACE_MODE", "slow")
    t0 = time.time()
    run("run", "--batch", "tester-5", "--work", w, "--budget-hours", 1, "--timeout-min", 0.05, "--runner", FAKE)
    assert time.time() - t0 < 120
    x = C.load_batch(w / "tester-5")["designs"][0]
    assert x["state"] == "failed" and "timeout" in x["note"]
    assert not (C.lock_dir() / "launcher.lock").exists()
    monkeypatch.setenv("FAKE_PALACE_MODE", "fail")
    run("run", "--batch", "tester-5", "--work", w, "--budget-hours", 1, "--retry-failed", "--runner", FAKE)
    assert "exit 1" in C.load_batch(w / "tester-5")["designs"][0]["note"]
    monkeypatch.setenv("FAKE_PALACE_MODE", "ok")
    run("run", "--batch", "tester-5", "--work", w, "--budget-hours", 1, "--retry-failed", "--runner", FAKE)
    assert C.load_batch(w / "tester-5")["designs"][0]["state"] == "collected"


def test_stop_file_and_budget_start_nothing(tmp_path):
    w = tmp_path / "work"
    run("new", "--n", 2, "--contributor", "tester", "--seed", 6, "--local", "--work", w)
    (w / "tester-6" / "STOP").write_text("")
    run("run", "--batch", "tester-6", "--work", w, "--budget-hours", 1, "--runner", FAKE)
    assert all(x["state"] == "drawn" for x in C.load_batch(w / "tester-6")["designs"])
    (w / "tester-6" / "STOP").unlink()
    run("run", "--batch", "tester-6", "--work", w, "--budget-hours", 0.01, "--runner", FAKE)   # < one 11-min solve
    assert all(x["state"] == "drawn" for x in C.load_batch(w / "tester-6")["designs"])


def test_local_batch_cannot_be_submitted(tmp_path):
    w = tmp_path / "work"
    run("new", "--n", 1, "--contributor", "tester", "--seed", 8, "--local", "--work", w)
    with pytest.raises(SystemExit, match="local"):
        C.stage(w / "tester-8", tmp_path / "incoming")


# ---------------------------------------------------------------------------------------------------- other solvers

@needs_build
def test_external_solver_round_trip(tmp_path):
    import check_sample as cs
    w = tmp_path / "work"
    run("new", "--n", 2, "--contributor", "tester", "--seed", 9, "--local", "--work", w, "--solver", "hfss",
        "--solver-version", "2025.1", "--port", "lumped 50 ohm port at the feed end, 3 mm feed de-embedded")
    d = w / "tester-9"
    assert run("go", "--batch", "tester-9", "--work", w) == 4                  # exported, nothing solved yet
    assert (d / "EXTERNAL_SOLVER.md").exists() and len(list((d / "geometry").glob("*.dxf"))) == 2
    fam = registry.load_family("edge_coupled_bpf")
    f = np.round(np.arange(2.9, 7.1 + 1e-9, 0.01), 4)
    for x in C.load_batch(d)["designs"]:
        s11, s21 = fam["prior"](x["p"], f)
        cs.write_touchstone(d / "touchstone" / f"{x['id']}.s2p", f, s11, s21)
    assert run("go", "--batch", "tester-9", "--work", w, "--mesh-elements", 100000, "--mesh-unknowns", 600000) == 0
    b = C.load_batch(d)
    assert all(x["verdict"] in ("PASS", "NEEDS REVIEW") for x in b["designs"])
    row = json.loads((d / "rows" / f"{b['designs'][0]['id']}.json").read_text())
    assert row["solver"]["name"] == "hfss" and row["solver"]["fidelity"] == "other"
    assert row["solver"]["mesh"]["unknowns"] == 600000


@needs_build
def test_ingest_appends_rows_and_removes_the_batch(tmp_path):
    import ingest_batch as IB
    w, inc, idx = tmp_path / "work", tmp_path / "incoming", tmp_path / "data_index"
    shutil.copytree(REPO / "data_index" / "edge_coupled_bpf", idx / "edge_coupled_bpf")
    n0 = len((idx / "edge_coupled_bpf" / "manifest.jsonl").read_text().splitlines())
    run("new", "--n", 2, "--contributor", "tester", "--seed", 11, "--work", w)
    run("go", "--batch", "tester-11", "--work", w, "--incoming", inc, "--claim-accepted", "--budget-hours", 1,
        "--runner", FAKE)
    bdir = inc / "edge_coupled_bpf" / "tester-11"
    assert IB.main([str(bdir), "--index-root", str(idx)]) == 0 and bdir.exists()          # dry run changes nothing
    assert IB.main([str(bdir), "--index-root", str(idx), "--apply", "--no-upload"]) == 0
    lines = (idx / "edge_coupled_bpf" / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == n0 + 2 and json.loads(lines[-1])["batch_id"] == "tester-11"
    assert (idx / "edge_coupled_bpf" / "batches" / "tester-11.json").exists() and not bdir.exists()
    # the same batch again is now a duplicate of the index
    run("submit", "--batch", "tester-11", "--work", w, "--incoming", inc)
    errs = "\n".join(V.verify_batch(bdir, idx)["errors"])
    assert "already in the data index" in errs
