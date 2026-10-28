"""A stand-in for Palace in tests: called as `fake_palace.py config.json <ranks>` in a case folder.

It reads p from build_summary.json, evaluates the family prior and writes output/port-S.csv in Palace's format, so
tools/contribute.py runs build -> solve -> collect -> check without a solver. FAKE_PALACE_MODE: ok (default),
fail (exit 1, no output), slow (sleep 600 s, for the timeout test).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import registry  # noqa: E402

mode = os.environ.get("FAKE_PALACE_MODE", "ok")
if mode == "slow":
    time.sleep(600)
if mode == "fail":
    print("fake palace: failing on purpose")
    sys.exit(1)
summ = json.loads(Path("build_summary.json").read_text())
fam = registry.load_family("edge_coupled_bpf")
f = np.round(np.arange(3.0, 7.0 + 1e-9, 0.02), 4)
s = fam["prior"](summ["p"], f)
s11, s21 = (s["S11"], s["S21"]) if isinstance(s, dict) else (s[0], s[1])
Path("output").mkdir(exist_ok=True)
cols = ["f (GHz)", "|S[1][1]| (dB)", "arg(S[1][1]) (deg.)", "|S[2][1]| (dB)", "arg(S[2][1]) (deg.)"]
rows = np.column_stack([f, 20 * np.log10(np.abs(s11)), np.degrees(np.angle(s11)),
                        20 * np.log10(np.abs(s21)), np.degrees(np.angle(s21))])
with open("output/port-S.csv", "w") as fh:
    fh.write(",".join(f"{c:>24}" for c in cols) + "\n")
    for r in rows:
        fh.write(",".join(f"{v:24.12e}" for v in r) + "\n")
print(f"fake palace: {len(f)} frequencies, ranks {sys.argv[2] if len(sys.argv) > 2 else '?'}")
