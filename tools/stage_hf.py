"""Lay out the Hugging Face model repo and dataset repo in dist/hf/ (and, only with --upload, push them).

    python tools/stage_hf.py [--family edge_coupled_bpf]
        dist/hf/model/<family_id>/   README.md (MODEL_CARD.md), LICENSE, ckpt/*.pt, weights.json, results/preds_*.npz
        dist/hf/samples/             README.md (DATASET_CARD.md), <family_id>/{palace,holdout,holdout_coarse}/*.npz,
                                     <family_id>/manifest.jsonl
    python tools/stage_hf.py --upload    # needs HF_TOKEN with write access to the organisation; pushes both folders

Run tools/make_model_card.py and core/export_web.py first. Every record file is checked against the file_sha256 of
its manifest row before it is staged.
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

DIST = REPO / "dist" / "hf"
HF_DATASET = "rfsurrogates/samples"
CKPT_PATTERNS = ("stageA_seed*.pt", "stageB_seed*.pt", "scratch_seed*.pt", "mlp_seed*.pt")
PREDS = ("preds_stageB.npz", "preds_scratch.npz", "preds_mlp.npz", "preds_prior_only.npz")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stage_model(family_id: str) -> Path:
    d = registry.family_dir(family_id)
    out = DIST / "model" / family_id
    if out.exists():
        shutil.rmtree(out)
    (out / "ckpt").mkdir(parents=True)
    (out / "results").mkdir()
    shutil.copy2(d / "MODEL_CARD.md", out / "README.md")
    shutil.copy2(REPO / "LICENSE", out / "LICENSE")
    n = 0
    for pat in CKPT_PATTERNS:
        for p in sorted((d / "ckpt").glob(pat)):
            shutil.copy2(p, out / "ckpt" / p.name)
            n += 1
    for name in PREDS:
        shutil.copy2(d / "results" / name, out / "results" / name)
    w = json.loads((REPO / "web" / "weights.json").read_text())
    if w.get("meta", {}).get("format") != "b64f32":
        raise SystemExit("web/weights.json is not base64 float32: run core/export_web.py")
    shutil.copy2(REPO / "web" / "weights.json", out / "weights.json")
    print(f"model   {out.relative_to(REPO)}: {n} checkpoints, {len(PREDS)} prediction files, weights.json")
    return out


def stage_dataset() -> Path:
    out = DIST / "samples"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(REPO / "data_index" / "DATASET_CARD.md", out / "README.md")
    for fid in registry.family_ids():
        idx = REPO / "data_index" / fid / "manifest.jsonl"
        if not idx.exists():
            continue
        data = registry.family_dir(fid) / "data"
        rows = [json.loads(line) for line in idx.read_text(encoding="utf-8").splitlines() if line.strip()]
        for r in rows:
            rel = Path(r["path"])                      # <family_id>/<subfolder>/<file>.npz
            src = data / rel.relative_to(fid)
            if _sha(src) != r["file_sha256"]:
                raise SystemExit(f"{src}: sha256 differs from its manifest row")
            (out / rel.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / rel)
        shutil.copy2(idx, out / fid / "manifest.jsonl")
        print(f"dataset {out.relative_to(REPO)}/{fid}: {len(rows)} records, hashes checked")
    return out


def upload(model_dir: Path, family_id: str, dataset_dir: Path):
    from huggingface_hub import HfApi
    api = HfApi()
    repo = registry.entry(family_id)["hf_model"]
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=model_dir)
    api.create_repo(HF_DATASET, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=HF_DATASET, repo_type="dataset", folder_path=dataset_dir)
    print(f"uploaded {repo} and {HF_DATASET}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", default=registry.DEFAULT_FAMILY, choices=registry.family_ids())
    ap.add_argument("--upload", action="store_true", help="push both folders to Hugging Face (needs HF_TOKEN)")
    a = ap.parse_args(argv)
    m = stage_model(a.family)
    d = stage_dataset()
    if a.upload:
        upload(m, a.family, d)


if __name__ == "__main__":
    main()
