#!/usr/bin/env bash
# Default Palace runner for tools/contribute.py, called as: run_palace.sh config.json <ranks>
# Uses, in this order: $PALACE_SIF (an Apptainer image of Palace), a `palace` executable on PATH.
# Any other setup: pass --runner "<your command>" to contribute.py; it is called the same way.
set -e
CONFIG=${1:?config.json}
NP=${2:-4}
if [ -n "$PALACE_SIF" ]; then
  exec apptainer run "$PALACE_SIF" -np "$NP" "$CONFIG"
elif command -v palace >/dev/null 2>&1; then
  exec palace -np "$NP" "$CONFIG"
else
  echo "run_palace.sh: no Palace found. Set PALACE_SIF to a Palace Apptainer image, put palace on PATH," >&2
  echo "or give contribute.py --runner \"<command>\" (called as <command> config.json <ranks>)." >&2
  exit 127
fi
