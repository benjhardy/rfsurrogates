"""Get the published weights and records of a family into this checkout.

    python tools/fetch.py [--family edge_coupled_bpf] [--weights-only | --data-only]
    python tools/fetch.py --source dist/hf          # from a folder laid out by tools/stage_hf.py instead of the Hub

Weights (Hugging Face model repo of the family, families.json "hf_model") -> families/<id>/ckpt/, results/preds_*.npz,
web/weights.json. Records (dataset repo rfsurrogates/samples, folder <family_id>/) -> families/<id>/data/. Every
record is checked against the file_sha256 of its row in data_index/<id>/manifest.jsonl, which is in git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import registry  # noqa: E402

HF_DATASET = "rfsurrogates/samples"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def model_folder(family_id: str, source: Path | None) -> Path:
    if source:
        return source / "model" / family_id
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(registry.entry(family_id)["hf_model"], repo_type="model"))


def dataset_folder(family_id: str, source: Path | None) -> Path:
    if source:
        return source / "samples"
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(HF_DATASET, repo_type="dataset", allow_patterns=[f"{family_id}/*"]))


def fetch_weights(family_id: str, src: Path):
    d = registry.family_dir(family_id)
    (d / "ckpt").mkdir(exist_ok=True)
    (d / "results").mkdir(exist_ok=True)
    ck = sorted((src / "ckpt").glob("*.pt"))
    for p in ck:
        shutil.copy2(p, d / "ckpt" / p.name)
    for p in sorted((src / "results").glob("preds_*.npz")):
        shutil.copy2(p, d / "results" / p.name)
    shutil.copy2(src / "weights.json", REPO / "web" / "weights.json")
    print(f"weights: {len(ck)} checkpoints -> {(d / 'ckpt').relative_to(REPO)}, web/weights.json")


def fetch_data(family_id: str, src: Path):
    idx = REPO / "data_index" / family_id / "manifest.jsonl"
    rows = [json.loads(line) for line in idx.read_text(encoding="utf-8").splitlines() if line.strip()]
    dst = registry.family_dir(family_id) / "data"
    bad = []
    for r in rows:
        rel = Path(r["path"])
        s = src / rel
        if not s.exists() or _sha(s) != r["file_sha256"]:
            bad.append(r["path"])
            continue
        out = dst / rel.relative_to(family_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, out)
    print(f"data: {len(rows) - len(bad)} of {len(rows)} records -> {dst.relative_to(REPO)}")
    if bad:
        raise SystemExit(f"{len(bad)} records missing or with a sha256 different from the manifest, e.g. {bad[:3]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", default=registry.DEFAULT_FAMILY, choices=registry.family_ids())
    ap.add_argument("--source", type=Path, help="local folder laid out by tools/stage_hf.py (default: the Hub)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--weights-only", action="store_true")
    g.add_argument("--data-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.data_only:
        fetch_weights(a.family, model_folder(a.family, a.source))
    if not a.weights_only:
        fetch_data(a.family, dataset_folder(a.family, a.source))


if __name__ == "__main__":
    main()
