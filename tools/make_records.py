"""collect.py npz files -> contract records (data/<split>/) + manifest rows (data_index/<family_id>/manifest.jsonl).

    python tools/make_records.py --src <old data folder> --contributor <github handle> [--family edge_coupled_bpf]

<old data folder> holds palace/, holdout/ and holdout_coarse/ as collect.py writes them. Each file goes through
contrib/check_sample.py twice: once to fill the provenance and write the contract npz, once on the written npz so
that the manifest row describes the published file (its file_sha256). The content hash must be the same both times.
Rows are sorted by split and id; the run stops on any FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "contrib"))
sys.path.insert(0, str(REPO))
import check_sample as C  # noqa: E402
import registry  # noqa: E402

# source subfolder -> (split name in the manifest, destination subfolder read by core/)
SPLITS = {"palace": ("train", "palace"), "holdout": ("holdout", "holdout"), "holdout_coarse": ("twin", "holdout_coarse")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="folder with palace/, holdout/, holdout_coarse/")
    ap.add_argument("--family", default=registry.DEFAULT_FAMILY, choices=registry.family_ids())
    ap.add_argument("--contributor", required=True, help="GitHub handle of whoever ran the solves")
    ap.add_argument("--licence", default="CC-BY-4.0")
    a = ap.parse_args(argv)
    dst_root = registry.family_dir(a.family) / "data"
    rows, verdicts = [], Counter()
    for sub, (split, dst_sub) in SPLITS.items():
        files = sorted((a.src / sub).glob("*.npz"))
        if not files:
            print(f"{sub}: no npz files under {a.src / sub}, skipped")
            continue
        (dst_root / dst_sub).mkdir(parents=True, exist_ok=True)
        for fp in files:
            first = C.check(fp, a.family, contributor=a.contributor, licence=a.licence)
            if first["verdict"] == "FAIL":
                print(C.one_line(first))
                return 1
            out = dst_root / dst_sub / fp.name
            C.write_contract_npz(first, out)
            row = C.check(out, a.family, sample_id=fp.stem)
            row.pop("_arrays", None)
            if row["content_sha256"] != first["content_sha256"]:
                print(f"{fp.name}: content hash changed between the source and the contract npz")
                return 1
            row["split"] = split
            row["path"] = f"{a.family}/{dst_sub}/{fp.name}"
            rows.append(row)
            verdicts[(split, row["verdict"])] += 1
    idx = REPO / "data_index" / a.family / "manifest.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    order = {s: k for k, (s, _) in enumerate(SPLITS.values())}
    rows.sort(key=lambda r: (order[r["split"]], r["id"]))
    idx.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    for (split, verdict), n in sorted(verdicts.items()):
        print(f"{split:8s} {verdict:13s} {n}")
    print(f"{len(rows)} rows -> {idx.relative_to(REPO)}; records under {dst_root.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
