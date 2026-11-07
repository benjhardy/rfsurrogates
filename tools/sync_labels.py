"""Create or update the GitHub labels of the task queue from .github/labels.json (maintainer, once per change).

    python tools/sync_labels.py            # dry run: prints the gh commands
    python tools/sync_labels.py --apply    # runs them (needs gh with write access to the repo)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default="benjhardy/rfsurrogates")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    labels = json.loads((REPO / ".github" / "labels.json").read_text())["labels"]
    for lab in labels:
        cmd = ["gh", "label", "create", lab["name"], "--repo", a.repo, "--color", lab["color"],
               "--description", lab["description"], "--force"]
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        if a.apply:
            subprocess.run(cmd, check=True)
    if not a.apply:
        print(f"dry run: {len(labels)} labels; add --apply to create them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
