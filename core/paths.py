"""Where the trainer and the tools read and write, for the family named by RFS_FAMILY (default edge_coupled_bpf).

    families/<family_id>/data/      Palace (or contributed) records: palace/, holdout/, holdout_coarse/  (not in git)
    families/<family_id>/ckpt/      checkpoints (not in git; published on Hugging Face)
    families/<family_id>/results/   metrics and tables (json/md/csv in git; preds_*.npz not in git)
    families/<family_id>/figs/      figures
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import registry  # noqa: E402

FAMILY = registry.current_family()
FAMILY_DIR = registry.family_dir(FAMILY)
if str(FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(FAMILY_DIR))          # the family's design.py / prior.py import by plain name

DATA = FAMILY_DIR / "data"
CKPT = FAMILY_DIR / "ckpt"
RESULTS = FAMILY_DIR / "results"
FIGS = FAMILY_DIR / "figs"
RUNS_INVERSE = FAMILY_DIR / "runs_inverse"
WEB = REPO / "web"
