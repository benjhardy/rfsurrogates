#!/usr/bin/env bash
# Palace queue for one family folder. Run in Linux (or in ONE foreground WSL call per batch on Windows):
#     bash families/edge_coupled_bpf/experiments/run_queue.sh [queue file]
# Queue file (default queue.txt in the family folder): one job per line, "<run name> [ranks]";
# "par: a 4, b 4" runs the jobs concurrently. The queue file is re-read before every job, so lines can be appended
# while it runs. Stops at the next job boundary when a STOP file exists in the family folder. Finished runs
# (output/port-S.csv + Elapsed in time.log) are skipped. After each run, collect.py writes data/palace/<id>.npz.
# Environment: RFS_PYTHON (python with gmsh, numpy, scipy, palace-agent-tools; default python3),
#              RUN_PALACE (command that takes "config.json <ranks>"; default run_palace, e.g. a wrapper around
#              `mpirun -np <ranks> palace config.json` or the Palace Apptainer image).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # the family folder
PY=${RFS_PYTHON:-python3}
RUN_PALACE=${RUN_PALACE:-run_palace}
QUEUE=${1:-$ROOT/queue.txt}
LOG=$ROOT/logs/queue_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] queue start: $QUEUE, $(nproc) cpus, $(free -g | awk '/Mem/{print $7}') GB free"

done_run() { [ -s "$ROOT/runs/$1/output/port-S.csv" ] && grep -q Elapsed "$ROOT/runs/$1/time.log" 2>/dev/null; }
# a run that finished (Elapsed in time.log) without port-S.csv failed: it is not retried; remove its time.log to retry
tried_run() { grep -q Elapsed "$ROOT/runs/$1/time.log" 2>/dev/null; }

run_one() {   # name ranks
  local name=$1 np=${2:-8} d=$ROOT/runs/$1
  if [ ! -f "$d/config.json" ] && [[ $name =~ ^[sh][0-9]{4}$ ]]; then
    # sample without a case folder yet: build it now with the agreed settings (prod_build_args.txt for sNNNN,
    # holdout_build_args.txt for the medium-mesh fidelity holdouts hNNNN)
    local args=$ROOT/prod_build_args.txt; [[ $name == h* ]] && args=$ROOT/holdout_build_args.txt
    [ -f "$args" ] && (cd "$ROOT" && "$PY" build_case.py --id "$name" $(cat "$args") --quiet 2>&1 | tail -1)
  fi
  if [ ! -f "$d/config.json" ]; then echo "[$(date '+%T')] $name: no config.json, skipped"; return; fi
  if tried_run "$name"; then echo "[$(date '+%T')] $name: already run, skipped"; return; fi
  echo "[$(date '+%T')] $name: start, $np ranks"
  ( cd "$d" && rm -rf output && /usr/bin/time -v $RUN_PALACE config.json "$np" > palace.log 2> time.log )
  local wall rss unk
  wall=$(grep -E 'Elapsed' "$d/time.log" | awk '{print $NF}')
  rss=$(grep 'Maximum resident' "$d/time.log" | awk '{printf "%.2f GB", $NF/1048576}')
  unk=$(grep -oE 'global unknowns = [0-9]+' "$d/palace.log" | tail -1)
  done_run "$name" || { echo "[$(date '+%T')] $name: FAILED (no port-S.csv), wall $wall; see $d/palace.log"; return; }
  echo "[$(date '+%T')] $name: done, wall $wall, max RSS (one rank) $rss, $unk, $(grep -E 'peak per-node' "$d/palace.log" | tail -1 | sed 's/Estimated //')"
  (cd "$ROOT" && "$PY" collect.py --run "$name" --ranks "$np" 2>&1 | tail -3)
}

while true; do
  [ -f "$ROOT/STOP" ] && { echo "[$(date '+%T')] STOP file found, queue stops"; break; }
  next=""
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}; line=$(echo "$line" | xargs)
    [ -z "$line" ] && continue
    if [[ $line == par:* ]]; then
      all_done=1
      IFS=',' read -ra jobs <<< "${line#par:}"
      for j in "${jobs[@]}"; do set -- $j; tried_run "$1" || all_done=0; done
      [ $all_done = 1 ] && continue
    else
      set -- $line; tried_run "$1" && continue
      [ -f "$ROOT/runs/$1/config.json" ] || [[ $1 =~ ^[sh][0-9]{4}$ ]] || { echo "[$(date '+%T')] $1: no config.json, skipped"; continue; }
    fi
    next=$line; break
  done < "$QUEUE"
  [ -z "$next" ] && { echo "[$(date '+%T')] queue empty, done"; break; }
  if [[ $next == par:* ]]; then
    IFS=',' read -ra jobs <<< "${next#par:}"
    echo "[$(date '+%T')] concurrent: ${jobs[*]}"
    # wait on the job pids only: a bare `wait` also waits for the tee process substitution (bash >= 5.1) and hangs
    pids=()
    for j in "${jobs[@]}"; do set -- $j; run_one "$1" "$2" & pids+=($!); done
    wait "${pids[@]}"
  else
    set -- $next; run_one "$1" "$2"
  fi
done
echo "[$(date '+%F %T')] queue exit"
