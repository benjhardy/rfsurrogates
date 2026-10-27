"""Maintainer step after a data pull request is merged: records to Hugging Face, rows to the data index.

    python tools/ingest_batch.py incoming/edge_coupled_bpf/alice-123456             # dry run: shows the plan
    python tools/ingest_batch.py incoming/edge_coupled_bpf/alice-123456 --apply     # upload, index, remove incoming
    python tools/ingest_batch.py incoming/edge_coupled_bpf/alice-123456 --apply --no-upload   # index only

--apply runs the verifier again and stops on any error. Then it:
1. uploads records/*.npz to the Hugging Face dataset (rfsurrogates/samples) under <family_id>/contrib/<batch_id>/
   (needs `pip install huggingface_hub` and a token with write access: `huggingface-cli login`);
2. appends the rows to data_index/<family_id>/manifest.jsonl and keeps batch.json as
   data_index/<family_id>/batches/<batch_id>.json;
3. removes incoming/<family_id>/<batch_id>/ with `git rm`.
It does not commit or push: review `git diff` and commit yourself. Retraining is a separate maintainer step.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import verify_batch as vb  # noqa: E402

DATASET = "rfsurrogates/samples"


def plan(bdir: Path, index_root: Path = REPO / "data_index") -> dict:
    b = json.loads((bdir / "batch.json").read_text())
    rows = [json.loads(x) for x in (bdir / "rows.jsonl").read_text().splitlines() if x.strip()]
    fam = bdir.parent.name
    return {"family": fam, "batch": bdir.name, "contributor": b["contributor"], "rows": rows, "batch_json": b,
            "hf_path": f"{fam}/contrib/{bdir.name}", "manifest": index_root / fam / "manifest.jsonl",
            "batch_copy": index_root / fam / "batches" / f"{bdir.name}.json"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batch", type=Path, help="incoming/<family_id>/<batch_id>")
    ap.add_argument("--apply", action="store_true", help="do it (default: show the plan only)")
    ap.add_argument("--no-upload", action="store_true", help="skip the Hugging Face upload")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--index-root", type=Path, default=REPO / "data_index")
    a = ap.parse_args(argv)
    bdir = a.batch.resolve()
    res = vb.verify_batch(bdir, a.index_root)
    print(vb.markdown([res], []))
    if res["errors"]:
        print("verifier errors: nothing is ingested")
        return 1
    p = plan(bdir, a.index_root)
    n = len(p["rows"])
    print(f"plan: {n} records from {p['contributor']}\n"
          f"  upload  {bdir / 'records'} -> {a.dataset}:{p['hf_path']}/" + (" (skipped)" if a.no_upload else "") + "\n"
          f"  append  {n} rows -> {p['manifest']}\n"
          f"  keep    batch.json -> {p['batch_copy']}\n"
          f"  remove  {bdir.relative_to(REPO) if bdir.is_relative_to(REPO) else bdir} (git rm)")
    if not a.apply:
        print("dry run: add --apply to do it")
        return 0
    if not a.no_upload:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=str(bdir / "records"), path_in_repo=p["hf_path"], repo_id=a.dataset,
                              repo_type="dataset",
                              commit_message=f"{p['family']}: batch {p['batch']} from {p['contributor']} ({n} records)")
        print(f"uploaded {n} records to {a.dataset}:{p['hf_path']}/")
    p["manifest"].parent.mkdir(parents=True, exist_ok=True)
    with open(p["manifest"], "a", encoding="utf-8", newline="\n") as fh:
        for r in p["rows"]:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    p["batch_copy"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bdir / "batch.json", p["batch_copy"])
    if bdir.is_relative_to(REPO):
        subprocess.run(["git", "-C", str(REPO), "rm", "-r", "-q", str(bdir.relative_to(REPO))], check=True)
    else:
        shutil.rmtree(bdir)
    print(f"indexed {n} rows. Review `git diff` and commit; retrain when the family has about 25 % more records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
