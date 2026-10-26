"""Coverage map: device types, the building blocks they are made of, and what has a surrogate.

    python tools/make_coverage.py            check coverage.json, write docs/COVERAGE.md and web/coverage.json
    python tools/make_coverage.py --check    check only; exit 1 if the outputs are not up to date

coverage.json is the only file to edit. A device with a family_id must be registered in families.json, and every
registered family must appear in coverage.json. Do not edit the outputs by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import registry  # noqa: E402

SRC = REPO / "coverage.json"
OUT_MD = REPO / "docs" / "COVERAGE.md"
OUT_WEB = REPO / "web" / "coverage.json"
PAGE = "https://benjhardy.github.io/rfsurrogates-coverage/"
SCHEMA = "rfsurrogates.coverage/1"
TRAINED = {"trained"}                         # families.json status values that count as a published surrogate


def load(path: Path = SRC) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def problems(cov: dict, families: list[dict] | None = None) -> list[str]:
    """Every inconsistency in coverage.json, as text; empty when the file is good."""
    if families is None:
        families = [registry.entry(fid) for fid in registry.family_ids()]
    out = []
    if cov.get("schema") != SCHEMA:
        out.append(f"schema is {cov.get('schema')!r}, expected {SCHEMA!r}")
    groups = {g["id"] for g in cov["groups"]}
    kinds = {k["id"] for k in cov["block_kinds"]}
    blocks = {b["id"]: b for b in cov["blocks"]}
    devices = {d["id"]: d for d in cov["devices"]}
    for name, items in (("group", cov["groups"]), ("block kind", cov["block_kinds"]),
                        ("block", cov["blocks"]), ("device", cov["devices"])):
        dup = [k for k, n in Counter(i["id"] for i in items).items() if n > 1]
        if dup:
            out.append(f"duplicate {name} id: {', '.join(dup)}")
    if set(blocks) & set(devices):
        out.append(f"id used for a block and a device: {', '.join(sorted(set(blocks) & set(devices)))}")
    for b in cov["blocks"]:
        if b["kind"] not in kinds:
            out.append(f"block {b['id']}: unknown kind {b['kind']!r}")
        if b["status"] not in cov["block_status"]:
            out.append(f"block {b['id']}: unknown status {b['status']!r}")
    for d in cov["devices"]:
        i = d["id"]
        if d["group"] not in groups:
            out.append(f"device {i}: unknown group {d['group']!r}")
        if d["status"] not in cov["device_status"]:
            out.append(f"device {i}: unknown status {d['status']!r}")
        for r in d.get("routes", []):
            if r not in cov["routes"]:
                out.append(f"device {i}: unknown route {r!r}")
        for b in d["blocks"]:
            if b not in blocks:
                out.append(f"device {i}: unknown block {b!r}")
        for r in d.get("related", []):
            if r["id"] not in devices or r["id"] == i:
                out.append(f"device {i}: bad related id {r['id']!r}")
        if (d["status"] == "surrogate") != bool(d.get("family_id")):
            out.append(f"device {i}: status 'surrogate' and a family_id go together")
    reg = {f["family_id"]: f for f in families}
    listed = {d["family_id"]: d for d in cov["devices"] if d.get("family_id")}
    for fid in listed:
        if fid not in reg:
            out.append(f"family_id {fid!r} is not in families.json")
        elif reg[fid].get("status") not in TRAINED:
            out.append(f"family {fid!r} has status {reg[fid].get('status')!r} in families.json, not trained")
    for fid in reg:
        if fid not in listed:
            out.append(f"registered family {fid!r} is missing from coverage.json")
    unused = sorted(set(blocks) - {b for d in cov["devices"] for b in d["blocks"]})
    if unused:
        out.append(f"blocks that no device uses: {', '.join(unused)}")
    return out


def web_json(cov: dict) -> str:
    """The page's copy: the source plus, per block, the devices that use it."""
    used = {b["id"]: [] for b in cov["blocks"]}
    for d in cov["devices"]:
        for b in d["blocks"]:
            used[b].append(d["id"])
    data = dict(cov, blocks=[dict(b, used_by=used[b["id"]]) for b in cov["blocks"]])
    data["generated_by"] = "tools/make_coverage.py from coverage.json"
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def _mermaid(cov: dict) -> list[str]:
    """Devices with work started (not 'open') and their blocks."""
    blocks = {b["id"]: b for b in cov["blocks"]}
    active = [d for d in cov["devices"] if d["status"] != "open"]
    used = sorted({b for d in active for b in d["blocks"]}, key=lambda b: [x["id"] for x in cov["blocks"]].index(b))
    lines = ["```mermaid", "flowchart LR"]
    for d in active:
        lines.append(f'  {d["id"]}["{d["label"]}<br/><i>{d["status"]}</i>"]:::{d["status"]}')
    for b in used:
        lines.append(f'  {b}(["{blocks[b]["label"]}<br/><i>{blocks[b]["status"]}</i>"]):::blk_{blocks[b]["status"]}')
    for d in active:
        for b in d["blocks"]:
            lines.append(f"  {d['id']} --> {b}")
    lines += ["  classDef surrogate fill:#2A7F50,color:#fff,stroke:#2A7F50",
              "  classDef planned fill:#2160A8,color:#fff,stroke:#2160A8",
              "  classDef experiment fill:#B85A0B,color:#fff,stroke:#B85A0B",
              "  classDef blk_learned fill:#2A7F50,color:#fff,stroke:#2A7F50",
              "  classDef blk_experiment fill:#B85A0B,color:#fff,stroke:#B85A0B",
              "  classDef blk_textbook fill:#fff,color:#142230,stroke:#2160A8",
              "  classDef blk_none fill:#fff,color:#142230,stroke:#586678,stroke-dasharray:4 3", "```"]
    return lines


def markdown(cov: dict) -> str:
    groups = {g["id"]: g["label"] for g in cov["groups"]}
    kinds = {k["id"]: k["label"] for k in cov["block_kinds"]}
    blocks = {b["id"]: b for b in cov["blocks"]}
    devs = cov["devices"]
    n = Counter(d["status"] for d in devs)
    used = Counter(b for d in devs for b in d["blocks"])
    L = ["# Coverage", "",
         "<!-- generated by tools/make_coverage.py from coverage.json; do not edit -->", "",
         f"Which passive devices have a surrogate, which are planned, and which building blocks they share. "
         f"Interactive map: `web/coverage.html`. To propose a device, see [CONTRIBUTING.md](../CONTRIBUTING.md).", "",
         f"{len(devs)} device types: " + ", ".join(f"{n[s]} {s}" for s in cov["device_status"]) + ".", "",
         "| status | meaning |", "|---|---|"]
    L += [f"| {s} | {t} |" for s, t in cov["device_status"].items()]
    L += ["", "## Work started", ""] + _mermaid(cov)
    L += ["", "## Devices", "", "| group | device | status | routes | blocks | outputs |", "|---|---|---|---|---|---|"]
    order = {g["id"]: k for k, g in enumerate(cov["groups"])}
    for d in sorted(devs, key=lambda d: order[d["group"]]):
        name = f"[{d['label']}](../families/{d['family_id']}/README.md)" if d.get("family_id") else d["label"]
        L.append(f"| {groups[d['group']]} | {name} | {d['status']} | {', '.join(d.get('routes', []))} | "
                 f"{', '.join(blocks[b]['label'] for b in d['blocks']) or '-'} | {d['outputs']} |")
    L += ["", "## Building blocks", "",
          "A block model fitted once serves every device that uses it (route `compose`). Sorted by use.", "",
          "| block | kind | status | used by |", "|---|---|---|---|"]
    for b in sorted(cov["blocks"], key=lambda b: -used[b["id"]]):
        L.append(f"| {b['label']} | {kinds[b['kind']]} | {b['status']} | {used[b['id']]} |")
    L += ["", "| block status | meaning |", "|---|---|"] + [f"| {s} | {t} |" for s, t in cov["block_status"].items()]
    L += ["", "## Routes", "", "| route | meaning |", "|---|---|"] + [f"| {r} | {t} |" for r, t in cov["routes"].items()]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="check only; exit 1 if the outputs are stale")
    a = ap.parse_args()
    cov = load()
    bad = problems(cov)
    if bad:
        print("coverage.json:\n  " + "\n  ".join(bad))
        return 1
    want = {OUT_MD: markdown(cov), OUT_WEB: web_json(cov)}
    if a.check:
        stale = [str(p.relative_to(REPO)) for p, t in want.items() if not p.exists() or p.read_text(encoding="utf-8") != t]
        if stale:
            print("out of date, run python tools/make_coverage.py: " + ", ".join(stale))
            return 1
        print("coverage ok")
        return 0
    for p, t in want.items():
        p.write_text(t, encoding="utf-8", newline="\n")
        print("wrote", p.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
