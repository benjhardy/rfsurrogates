"""The verifier: checks every batch a data pull request adds under incoming/<family_id>/<batch_id>/.

    python tools/verify_batch.py                                   # every batch under incoming/
    python tools/verify_batch.py incoming/edge_coupled_bpf/alice-123456
    python tools/verify_batch.py --changed-files changed.txt --summary "$GITHUB_STEP_SUMMARY"

CI runs this file and the checker from the base branch on the files of the pull request, so a pull request cannot
change the rules it is checked against. Nothing here edits a record.

Errors (exit 1): batch layout and fields, a record file that does not match its row (sha256), a checker rerun that
gives another verdict or content hash, a FAIL, an id / record / design that is already in the data index or twice in
the batch, a data pull request that also changes files outside incoming/.
Warnings (exit 0, shown to the reviewer): NEEDS REVIEW rows without the contributor's sentence, no claim issue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "contrib"))
import check_sample as cs  # noqa: E402
import registry  # noqa: E402

BATCH_SCHEMA = "rfsurrogates.batch/1"
LICENCES = {"CC-BY-4.0"}
MAX_RECORD_BYTES = 200_000
MAX_RECORDS = 500


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(family_id: str, index_root: Path) -> list[dict]:
    f = index_root / family_id / "manifest.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def verify_batch(bdir: Path, index_root: Path = REPO / "data_index") -> dict:
    """Return {'batch', 'errors', 'warnings', 'rows': [(id, submitted, rerun, ok)]} for one batch folder."""
    errors: list[str] = []
    warns: list[str] = []
    table: list[tuple] = []
    res = {"batch": bdir.name, "family": bdir.parent.name, "errors": errors, "warnings": warns, "rows": table}
    bj, rj, rec = bdir / "batch.json", bdir / "rows.jsonl", bdir / "records"
    extra = sorted(p.name for p in bdir.iterdir() if p.name not in ("batch.json", "rows.jsonl", "records"))
    if extra:
        errors.append(f"unexpected files in the batch folder: {extra}")
    if not bj.exists() or not rj.exists() or not rec.is_dir():
        errors.append("a batch folder holds batch.json, rows.jsonl and records/")
        return res
    try:
        b = json.loads(bj.read_text())
        rows = [json.loads(x) for x in rj.read_text().splitlines() if x.strip()]
    except ValueError as e:
        errors.append(f"cannot read batch.json or rows.jsonl: {e}")
        return res
    fam_id = bdir.parent.name
    if b.get("schema") != BATCH_SCHEMA:
        errors.append(f"batch.json schema is {b.get('schema')!r}, expected {BATCH_SCHEMA}")
    if b.get("batch_id") != bdir.name:
        errors.append(f"batch_id {b.get('batch_id')!r} differs from the folder name {bdir.name!r}")
    if fam_id not in registry.family_ids() or b.get("family_id") != fam_id:
        errors.append(f"family {b.get('family_id')!r} / folder {fam_id!r}: not a registered family or not the same")
        return res
    if not b.get("contributor"):
        errors.append("batch.json has no contributor")
    if b.get("licence") not in LICENCES:
        errors.append(f"licence {b.get('licence')!r} is not one of {sorted(LICENCES)}")
    if not (b.get("claim") or {}).get("issue"):
        warns.append("no claim issue in batch.json: the maintainer cannot see that the ids were reserved")
    if not rows:
        errors.append("rows.jsonl is empty")
    if len(rows) > MAX_RECORDS:
        errors.append(f"{len(rows)} records in one batch; the limit is {MAX_RECORDS}")

    man = load_manifest(fam_id, index_root)
    man_ids = {r["id"] for r in man}
    man_hash = {r.get("content_sha256") for r in man}
    man_p = np.array([r["p"] for r in man], float) if man else np.zeros((0, 1))
    files = {p.name for p in rec.iterdir()}
    seen_ids, seen_hash, seen_p = set(), set(), []
    review = b.get("review_notes") or {}
    for r in rows:
        rid = r.get("id", "?")
        problems = []
        if rid in seen_ids:
            problems.append("id twice in the batch")
        seen_ids.add(rid)
        if rid in man_ids:
            problems.append("id already in the data index")
        for key, want in (("family_id", fam_id), ("batch_id", bdir.name), ("contributor", b.get("contributor")),
                          ("path", f"{fam_id}/contrib/{bdir.name}/{rid}.npz")):
            if r.get(key) != want:
                problems.append(f"{key} is {r.get(key)!r}, expected {want!r}")
        f = rec / f"{rid}.npz"
        rerun_v = "-"
        if not f.exists():
            problems.append("record file missing")
        else:
            files.discard(f.name)
            if f.stat().st_size > MAX_RECORD_BYTES:
                problems.append(f"record is {f.stat().st_size} bytes (limit {MAX_RECORD_BYTES})")
            elif sha256(f) != r.get("file_sha256"):
                problems.append("file sha256 differs from the row")
            else:
                again = cs.check(f, fam_id, sample_id=rid)
                rerun_v = again["verdict"]
                if again["verdict"] != r.get("verdict"):
                    problems.append(f"checker rerun gives {again['verdict']}, the row says {r.get('verdict')}")
                if again.get("content_sha256") != r.get("content_sha256"):
                    problems.append("content hash differs on rerun")
                if again["verdict"] == "FAIL":
                    problems.append("FAIL: " + "; ".join(again["errors"]))
                if again.get("contributor") != b.get("contributor"):
                    problems.append("the record names another contributor than batch.json")
        h = r.get("content_sha256")
        if h in man_hash:
            problems.append("the same record is already in the data index")
        if h in seen_hash:
            problems.append("the same record twice in the batch")
        seen_hash.add(h)
        p = np.asarray(r.get("p", []), float)
        if man_p.size and p.size == man_p.shape[1] and np.any(np.all(np.isclose(man_p, p, rtol=1e-9, atol=0), axis=1)):
            problems.append("the same design p is already in the data index")
        if any(p.size == q.size and np.allclose(p, q, rtol=1e-9, atol=0) for q in seen_p):
            problems.append("the same design p twice in the batch")
        seen_p.append(p)
        if r.get("verdict") == "NEEDS REVIEW" and not str(review.get(rid, "")).strip():
            warns.append(f"{rid}: NEEDS REVIEW ({', '.join(r.get('flags', []))}) without the contributor's sentence")
        for msg in problems:
            errors.append(f"{rid}: {msg}")
        table.append((rid, r.get("verdict", "?"), rerun_v, not problems))
    if files:
        errors.append(f"record files without a row: {sorted(files)}")
    return res


def check_changed(changed: list[str]) -> list[str]:
    inc = [c for c in changed if c.startswith("incoming/")]
    other = [c for c in changed if not c.startswith("incoming/")]
    if inc and other:
        return [f"a data pull request changes only incoming/; it also changes {other[:10]}"]
    return []


def markdown(results: list[dict], global_errors: list[str]) -> str:
    out = ["## Data verifier", ""]
    for e in global_errors:
        out.append(f"- **error:** {e}")
    for r in results:
        n_ok = sum(1 for x in r["rows"] if x[3])
        head = "passed" if not r["errors"] else f"{len(r['errors'])} error(s)"
        out += ["", f"### {r['family']} / {r['batch']}: {head}, {n_ok} of {len(r['rows'])} records accepted", ""]
        for e in r["errors"]:
            out.append(f"- **error:** {e}")
        for w in r["warnings"]:
            out.append(f"- warning: {w}")
        if r["rows"]:
            out += ["", "| id | verdict in the row | verdict on rerun | accepted |", "|---|---|---|---|"]
            out += [f"| {i} | {v} | {rv} | {'yes' if ok else 'no'} |" for i, v, rv, ok in r["rows"]]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify the batches of a data pull request.")
    ap.add_argument("batches", nargs="*", type=Path, help="batch folders (default: every incoming/<family>/<batch>/)")
    ap.add_argument("--incoming", type=Path, default=REPO / "incoming")
    ap.add_argument("--index-root", type=Path, default=REPO / "data_index", help="data_index/ of the base branch")
    ap.add_argument("--changed-files", type=Path, help="paths changed by the pull request, one per line")
    ap.add_argument("--summary", type=Path, help="append a markdown report (e.g. $GITHUB_STEP_SUMMARY)")
    a = ap.parse_args(argv)
    batches = a.batches or sorted(p for p in a.incoming.glob("*/*") if p.is_dir())
    global_errors = check_changed(a.changed_files.read_text().split()) if a.changed_files else []
    results = [verify_batch(p.resolve(), a.index_root) for p in batches]
    md = markdown(results, global_errors)
    print(md)
    if a.summary:
        with open(a.summary, "a", encoding="utf-8") as fh:
            fh.write(md)
    if not batches:
        print("no batch found under", a.incoming)
    return 1 if global_errors or any(r["errors"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
