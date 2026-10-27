"""Overnight batch for a 20 dB return-loss nominal (N = 3 Chebyshev, 0.0436 dB ripple, f0 5 GHz, 5 % bandwidth).

    set RFS_WSL_DISTRO=Ubuntu-24.04
    python rl20_batch.py [--n 60] [--seed 20261027] [--max-iter 5]

1. Space-mapping tune of the nominal in Palace (tune.py --design design_rl20.json, runs tune20_<k>), unless
   tune_log_rl20.json already holds a converged result.
2. Box around the tuned centre (same fractions as the production box: +-30 % widths and gaps, +-15 % lengths),
   Latin hypercube with the family validity rule -> samples_rl20.json, ids r0000...
3. Build the cases with the production settings of tune.py and run them through run_queue.sh, two jobs at a time
   on 4 ranks. Cases are built in chunks while the queue runs; the queue re-reads queue_rl20.txt before every job.
   A STOP file next to run_queue.sh stops the queue at the next job boundary; this script then stops too.
4. Collect every finished run on Windows (collect.py -> data/aux/r*.npz, runs/r*/result.json).

This is one launcher: it starts Palace only through run_queue.sh, one queue at a time. Start it only when no other
Palace queue runs on the machine.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_case  # noqa: E402
import collect  # noqa: E402
import design  # noqa: E402
import tune  # noqa: E402

QUEUE = HERE / "queue_rl20.txt"
STOP = HERE / "STOP"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tuned_centre(max_iter):
    path = HERE / "tune_log_rl20.json"
    if not (path.exists() and json.loads(path.read_text()).get("converged")):
        cmd = [sys.executable, str(HERE / "tune.py"), "--design", "design_rl20.json", "--log", path.name,
               "--prefix", "tune20_", "--rl", "20", "--max-iter", str(max_iter)]
        log("tune: " + " ".join(cmd[1:]))
        subprocess.run(cmd, check=True, cwd=HERE)
    t = json.loads(path.read_text())
    log(f"tune: converged {t['converged']}, final p {np.round(t['final_p'], 4).tolist()}")
    return list(t["final_p"]), bool(t["converged"])


def draw(centre, n, seed):
    box = design.make_box(centre, design.BOX_FRAC)
    w50 = design.w50_for(centre[6], centre[7])
    out, k = [], 0
    while len(out) < n:
        P, kinds = design.sample_box(box, 4 * n, seed + k, edge_frac=0.0)
        out += [[float(v) for v in row] for row in P if design.is_valid(row, w50=w50)]
        k += 1
    samples = [{"id": f"r{i:04d}", "p": p, "kind": "rl20", "clearance_mm": round(design.clearance(p, w50), 5)}
               for i, p in enumerate(out[:n])]
    S = {"schema": "rfsurrogates.samples/2", "family_id": "edge_coupled_bpf", "subset": "rl20", "seed": seed,
         "param_names": list(design.PARAM_NAMES), "box": box, "centre": centre, "box_frac": design.BOX_FRAC,
         "spec": "N=3 Chebyshev 0.0436 dB ripple (RL 20 dB), f0 5 GHz, 5 % ripple bandwidth",
         "validity": f"clearance > {design.C_MIN} mm", "samples": samples}
    (HERE / "samples_rl20.json").write_text(json.dumps(S, indent=1))
    return samples


def queue_cmd():
    q = tune.WSL[:-1] + [tune.WSL[-1].replace("queue_tune.txt", QUEUE.name)]
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=6, help="cases built before the queue starts / per refill")
    ap.add_argument("--tune-only", action="store_true", help="step 1 only")
    a = ap.parse_args()
    if STOP.exists():
        log("STOP file present; nothing to do"); return
    centre, ok = tuned_centre(a.max_iter)
    if a.tune_only:
        return
    if not ok:
        log("tune did not converge; the batch uses the last tuned point as the centre")
    samples = draw(centre, a.n, a.seed)
    log(f"{len(samples)} valid samples in the rl20 box -> samples_rl20.json")
    names, built, queue = [s["id"] for s in samples], [], None
    head = "# rl20 batch: two jobs at a time, 4 ranks each (production setting)\n"
    QUEUE.write_bytes(head.encode())                       # LF: bash reads it

    def build_next(k):
        for s in samples[len(built):len(built) + k]:
            try:
                build_case.build(s["id"], s["p"], **tune.SETTINGS)
                built.append(s["id"])
            except build_case.GeometryError as e:
                log(f"{s['id']}: rejected by the gate ({e})"); built.append(None)
        ok_ids = [i for i in built if i]
        lines = [f"par: {ok_ids[j]} 4, {ok_ids[j + 1]} 4" if j + 1 < len(ok_ids) else f"{ok_ids[j]} 4"
                 for j in range(0, len(ok_ids), 2)]
        QUEUE.write_bytes((head + "\n".join(lines) + "\n").encode())

    build_next(a.chunk)
    queue = subprocess.Popen(queue_cmd(), cwd=HERE)
    log(f"queue started ({len([b for b in built if b])} cases built)")
    while not STOP.exists():
        if len(built) < len(names):
            build_next(2)
        if queue.poll() is not None:            # the queue emptied: restart it for cases built since, or finish
            todo = [i for i in built if i and not (HERE / "runs" / i / "time.log").exists()]
            if not todo and len(built) == len(names):
                break
            if todo:
                log(f"queue exited with {len(todo)} built cases left; restarting it")
                queue = subprocess.Popen(queue_cmd(), cwd=HERE)
        time.sleep(30)
    queue.wait()
    n_ok = 0
    for i in [b for b in built if b]:
        if (HERE / "runs" / i / "output" / "port-S.csv").exists():
            try:
                collect.collect(i, 4); n_ok += 1
            except Exception as e:  # noqa: BLE001
                log(f"{i}: collect failed: {e!r}")
    log(f"done: {n_ok} of {len(names)} rl20 samples solved and collected")


if __name__ == "__main__":
    main()
