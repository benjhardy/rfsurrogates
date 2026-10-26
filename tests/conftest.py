"""Import paths for the tests: repo root (registry.py), core/, contrib/, tests/, and the family folder of RFS_FAMILY
(default edge_coupled_bpf, so `import design`, `import prior`, `import build_case` resolve to that family)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "core", REPO / "contrib", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import registry  # noqa: E402

FAMILY_DIR = registry.family_dir()
if str(FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(FAMILY_DIR))
