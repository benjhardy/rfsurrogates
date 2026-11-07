"""One command for data contributors: draw designs, claim them, solve them, check them, open one pull request.

    python tools/contribute.py go --family edge_coupled_bpf --n 20 --contributor <github handle>
        first run:  draws 20 valid designs from the family box, writes work/<batch>/ and the claim issue text
                    (--open-claim opens it with gh). Then it stops: solve only after the maintainer accepts the claim.
    python tools/contribute.py go --batch <batch> --budget-hours 4
        later runs: checks the claim, builds and solves every design on this machine (Palace), collects and checks
                    each record, and stages the pull request (--open-pr opens it with gh). Safe to run again: it
                    continues where it stopped.

Other solvers (HFSS, CST, openEMS, ...): add `--solver hfss --solver-version 2025.1` to the first run. `go` then
exports the geometry of every design to work/<batch>/geometry/ (JSON + DXF), you solve them, put <id>.s2p files in
work/<batch>/touchstone/, and run `go --batch <batch>` again.

Single steps: new, status, run, export, import, check, submit, unlock. `--local` makes a batch for practice or for
your own use; it never needs a claim and cannot be submitted.

Launcher rules (the guard rails for unattended or agent-driven runs):
- one launcher per machine: a lock file in ~/.rfsurrogates (RFS_LOCK_DIR overrides);
- a STOP file in the batch folder or in ~/.rfsurrogates stops the run before the next case;
- the contributor sets the budget (machine hours); the launcher starts no case that would go past it;
- every case has a timeout (default 4 x the family estimate) and runs only inside work/<batch>/runs/.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "contrib"))
import registry  # noqa: E402

SCHEMA = "rfsurrogates.batch/1"
REPO_SLUG = "benjhardy/rfsurrogates"
LICENCE = "CC-BY-4.0"
WORK = REPO / "work"
INCOMING = REPO / "incoming"
TERMINAL = {"rejected", "failed", "checked"}          # a design in one of these states needs no more work


# ---------------------------------------------------------------------------------------------------- batch file

def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def batch_dir(work: Path, batch_id: str) -> Path:
    return work / batch_id


def load_batch(d: Path) -> dict:
    f = d / "batch.json"
    if not f.exists():
        raise SystemExit(f"no batch at {d} (batch.json missing)")
    return json.loads(f.read_text())


def save_batch(d: Path, b: dict) -> None:
    tmp = d / "batch.json.tmp"
    tmp.write_text(json.dumps(b, indent=1))
    tmp.replace(d / "batch.json")


def log(d: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    (d / "logs").mkdir(exist_ok=True)
    with open(d / "logs" / "contribute.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def estimate(fam: dict, solver: dict, n: int, jobs: int) -> dict:
    """Machine hours for n solves from the family cost table (families.json 'cost'); None for other solvers."""
    cost = (fam.get("cost") or {}).get(solver["fidelity"]) if solver["name"] == "palace" else None
    if not cost:
        return {"minutes_per_solve": None, "machine_hours": None}
    return {"minutes_per_solve": cost["minutes_per_solve"], "ranks": cost["ranks"],
            "peak_ram_gb_per_job": cost["peak_ram_gb"],
            "machine_hours": round(n * cost["minutes_per_solve"] / 60 / max(1, jobs), 2),
            "source": cost.get("source", "families.json")}


def new_batch(a) -> Path:
    fam = registry.load_family(a.family)
    if "sample_designs" not in fam:
        raise SystemExit(f"family {a.family} has no contributor hooks yet (family.py: sample_designs, build, ...)")
    if not a.contributor:
        raise SystemExit("give --contributor <your GitHub handle> (it is written into every record)")
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(100000, 999999)
    batch_id = a.batch or f"{a.contributor}-{seed}"
    d = batch_dir(a.work, batch_id)
    if (d / "batch.json").exists():
        raise SystemExit(f"batch {batch_id} exists already: {d}")
    known = [json.loads(x)["p"] for x in (REPO / "data_index" / a.family / "manifest.jsonl").read_text().splitlines()
             if x.strip()] if (REPO / "data_index" / a.family / "manifest.jsonl").exists() else []
    designs = [p for p in fam["sample_designs"](a.n + len(known), seed)
               if not any(all(abs(u - v) <= 1e-9 * max(1.0, abs(v)) for u, v in zip(p, q)) for q in known)][:a.n]
    palace = a.solver == "palace"
    solver = {"name": a.solver, "version": a.solver_version or ("0.16" if palace else None),
              "fidelity": a.fidelity or (fam["production_fidelity"] if palace else "other"),
              "order": a.order if a.order is not None else (2 if palace else None),
              "port": a.port or (fam["solver_defaults"]["palace"]["port"] if palace else None),
              "mesh_elements": a.mesh_elements, "mesh_unknowns": a.mesh_unknowns}
    if not palace and not solver["version"]:
        raise SystemExit("give --solver-version for a solver other than Palace")
    b = {"schema": SCHEMA, "batch_id": batch_id, "family_id": a.family, "contributor": a.contributor,
         "licence": LICENCE, "seed": seed, "created": now(), "local": bool(a.local), "solver": solver,
         "claim": {"issue": None, "url": None, "accepted": False},
         "estimate": estimate(fam, solver, a.n, a.jobs),
         "notes": a.notes or "", "review_notes": {},
         "designs": [{"id": f"{batch_id}-{k + 1:03d}", "p": p, "state": "drawn"} for k, p in enumerate(designs)]}
    d.mkdir(parents=True)
    save_batch(d, b)
    (d / "claim.md").write_text(claim_text(b))
    log(d, f"batch {batch_id}: {a.n} valid designs of {a.family} drawn (seed {seed}) -> {d}")
    est = b["estimate"]
    if est["machine_hours"] is not None:
        log(d, f"estimate: about {est['minutes_per_solve']} min per solve, about {est['machine_hours']} machine hours "
               f"with {a.jobs} job(s) at a time, about {est['peak_ram_gb_per_job']} GB RAM per job")
    return d


# ---------------------------------------------------------------------------------------------------- claim

def claim_text(b: dict) -> str:
    ids = [x["id"] for x in b["designs"]]
    est = b["estimate"]["machine_hours"]
    return "\n".join([
        f"### Family id\n\n{b['family_id']}\n",
        f"### Designs\n\n{len(ids)} designs, ids {ids[0]} to {ids[-1]}, drawn with `tools/contribute.py` (seed "
        f"{b['seed']}). The design list is in the batch file and goes with the pull request.\n",
        f"### Fidelity\n\n{b['solver']['fidelity']}\n",
        f"### Solver and version\n\n{b['solver']['name']} {b['solver']['version']}\n",
        f"### Estimated solver-hours on your machine\n\n{est if est is not None else 'not estimated (own solver)'}\n",
        "### Who runs the solves\n\na person, with tools/contribute.py\n"])


def gh(*args, check=True) -> subprocess.CompletedProcess:
    if not shutil.which("gh"):
        raise SystemExit("GitHub CLI `gh` not found; install it (https://cli.github.com) or do this step in the browser")
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check)


def open_claim(d: Path, b: dict, repo: str) -> None:
    ids = [x["id"] for x in b["designs"]]
    title = f"claim: {b['family_id']} {ids[0]}-{ids[-1]}"
    r = gh("issue", "create", "--repo", repo, "--title", title, "--body-file", str(d / "claim.md"),
           "--label", "kind:claim,role:sampler,status:needs-maintainer", check=False)
    if r.returncode != 0:
        raise SystemExit(f"gh issue create failed: {r.stderr.strip()}")
    url = r.stdout.strip().splitlines()[-1]
    b["claim"].update({"url": url, "issue": int(url.rstrip("/").split("/")[-1])})
    save_batch(d, b)
    log(d, f"claim issue opened: {url}")


def claim_accepted(d: Path, b: dict, repo: str) -> bool:
    if b["local"] or b["claim"]["accepted"]:
        return True
    if not b["claim"]["issue"]:
        return False
    if not shutil.which("gh"):
        log(d, "gh not found: cannot read the claim labels; install gh, or wait and pass --claim-accepted")
        return False
    r = gh("issue", "view", str(b["claim"]["issue"]), "--repo", repo, "--json", "labels,state", check=False)
    if r.returncode != 0:
        log(d, f"gh issue view failed: {r.stderr.strip()}")
        return False
    labels = {x["name"] for x in json.loads(r.stdout)["labels"]}
    if "status:accepted" in labels:
        b["claim"]["accepted"] = True
        save_batch(d, b)
        return True
    return False


# ---------------------------------------------------------------------------------------------------- launcher

class LauncherBusy(RuntimeError):
    pass


def lock_dir() -> Path:
    return Path(os.environ.get("RFS_LOCK_DIR", Path.home() / ".rfsurrogates"))


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Lock:
    """One launcher per machine. A lock left by a dead process on this host is removed; any other lock refuses."""

    def __init__(self, batch_id: str):
        self.path = lock_dir() / "launcher.lock"
        self.batch_id = batch_id

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    held = json.loads(self.path.read_text())
                except (OSError, ValueError):
                    held = {}
                if held.get("host") == socket.gethostname() and not _pid_alive(int(held.get("pid", -1))):
                    self.path.unlink(missing_ok=True)
                    continue
                raise LauncherBusy(f"another launcher holds {self.path}: {held}. Wait for it, or run "
                                   f"`contribute.py unlock` if you are sure it is gone.")
            with os.fdopen(fd, "w") as fh:
                json.dump({"pid": os.getpid(), "host": socket.gethostname(), "batch": self.batch_id,
                           "started": now()}, fh)
            return self
        raise LauncherBusy(f"cannot take {self.path}")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def stop_requested(d: Path) -> bool:
    return (d / "STOP").exists() or (lock_dir() / "STOP").exists()


def wsl_path(p: Path) -> str:
    p = Path(p).resolve()
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


def runner_argv(runner: str | None) -> list[str]:
    """The Palace command, called as `<runner> config.json <ranks>` (the run_queue.sh convention)."""
    r = runner or os.environ.get("RFS_RUNNER") or os.environ.get("RUN_PALACE")
    if r:
        return [w.strip('"') for w in shlex.split(r, posix=os.name != "nt")]
    return ["bash", str(REPO / "tools" / "run_palace.sh")]


def solve_case(case: Path, ranks: int, runner: str | None, wsl: str | None, timeout_s: float) -> dict:
    """Run the solver in case/; stdout -> palace.log, stderr (and /usr/bin/time -v) -> time.log."""
    out = case / "output"
    if out.exists():
        shutil.rmtree(out)
    t0 = time.time()
    if wsl:
        r = runner or os.environ.get("RFS_RUNNER") or os.environ.get("RUN_PALACE") or "run_palace"
        script = (f"cd '{wsl_path(case)}' && echo $$ > launcher.pid && "
                  f"if [ -x /usr/bin/time ]; then exec /usr/bin/time -v {r} config.json {ranks} > palace.log 2> time.log; "
                  f"else exec {r} config.json {ranks} > palace.log 2> time.log; fi")
        proc = subprocess.Popen(["wsl", "-d", wsl, "--", "setsid", "--wait", "bash", "-c", script])
    else:
        argv = runner_argv(runner) + ["config.json", str(ranks)]
        if os.name != "nt" and Path("/usr/bin/time").exists():
            argv = ["/usr/bin/time", "-v"] + argv
        kw = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        with open(case / "palace.log", "w") as so, open(case / "time.log", "w") as se:
            proc = subprocess.Popen(argv, cwd=case, stdout=so, stderr=se, **kw)
    try:
        rc = proc.wait(timeout=timeout_s)
        status = "ok" if rc == 0 else f"exit {rc}"
    except subprocess.TimeoutExpired:
        kill_case(proc, case, wsl)
        status = "timeout"
    wall = time.time() - t0
    tl = case / "time.log"
    if "Elapsed (wall clock)" not in (tl.read_text(errors="replace") if tl.exists() else ""):
        with open(tl, "a") as fh:                      # no /usr/bin/time: record the launcher's own wall time
            fh.write("\tElapsed (wall clock) time (h:mm:ss or m:ss): " + f"{int(wall // 60)}:{wall % 60:05.2f}" + "\n")
    if status == "ok" and not (out / "port-S.csv").exists():
        status = "no port-S.csv"
    return {"status": status, "wall_s": round(wall, 1)}


def kill_case(proc, case: Path, wsl: str | None) -> None:
    if wsl:
        subprocess.run(["wsl", "-d", wsl, "--", "bash", "-c",
                        f"kill -TERM -- -$(cat '{wsl_path(case)}/launcher.pid') 2>/dev/null; sleep 5; "
                        f"kill -KILL -- -$(cat '{wsl_path(case)}/launcher.pid') 2>/dev/null; true"], check=False)
        proc.kill()
    elif os.name != "nt":
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            time.sleep(5 if sig == signal.SIGTERM else 0)
    else:
        proc.kill()
    proc.wait()


def run_batch(d: Path, a) -> None:
    """Build, solve and collect every open design, under the launcher rules."""
    b = load_batch(d)
    fam = registry.load_family(b["family_id"])
    if b["solver"]["name"] != "palace":
        raise SystemExit(f"this batch uses {b['solver']['name']}: use `export` and `import`, or `go`")
    est_min = b["estimate"].get("minutes_per_solve") or 15
    budget_h = a.budget_hours
    if getattr(a, "retry_failed", False):
        for x in b["designs"]:
            if x["state"] == "failed":
                x["state"] = "drawn"
    todo = [x for x in b["designs"] if x["state"] not in TERMINAL and x["state"] != "collected"]
    if not todo:
        log(d, "nothing to solve")
        return
    need_h = round(len(todo) * est_min / 60 / max(1, a.jobs), 2)
    if budget_h is None:
        if not sys.stdin.isatty():
            raise SystemExit(f"{len(todo)} solves, about {need_h} machine hours: give --budget-hours (you set the budget)")
        ans = input(f"{len(todo)} solves, about {need_h} machine hours on this machine. Budget in hours [{need_h}]: ")
        budget_h = float(ans) if ans.strip() else need_h
    timeout_s = (a.timeout_min if a.timeout_min else 4 * est_min) * 60
    ranks = a.ranks or b["estimate"].get("ranks") or 4
    runs = d / "runs"
    t_start = time.time()
    with Lock(b["batch_id"]):
        log(d, f"launcher: {len(todo)} designs, {a.jobs} job(s) x {ranks} ranks, budget {budget_h} h, "
               f"timeout {timeout_s / 60:.0f} min per case" + (f", WSL {a.wsl}" if a.wsl else ""))

        def prepare(x) -> bool:
            """Build the case (main thread: gmsh is not thread-safe). False when the geometry gate rejects it."""
            case = runs / x["id"]
            if x["state"] in ("drawn", "failed") or not (case / "config.json").exists():
                try:
                    s = fam["build"](x["id"], x["p"], runs, b["solver"]["fidelity"])
                    x.update(state="built", mesh_elements=s.get("tets"), mesh_unknowns=s.get("unknowns_order2"))
                except registry.GateReject as e:
                    x.update(state="rejected", note=f"geometry gate: {e}")
                    log(d, f"{x['id']}: rejected by the geometry gate ({e})")
                    return False
            return True

        def finish(x, r) -> None:
            """Collect a finished solve (main thread)."""
            case = runs / x["id"]
            x["wall_s"] = r["wall_s"]
            if r["status"] != "ok":
                x.update(state="failed", note=f"solver: {r['status']}; see {case / 'palace.log'}")
                log(d, f"{x['id']}: FAILED ({r['status']}) after {r['wall_s'] / 60:.1f} min")
                return
            fam["collect"](case, d / "records" / f"{x['id']}.npz", ranks)
            x.update(state="collected", note="")
            log(d, f"{x['id']}: solved and collected in {r['wall_s'] / 60:.1f} min")

        queue = list(todo)
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
            running = {}
            while queue or running:
                while queue and len(running) < max(1, a.jobs):
                    spent_h = (time.time() - t_start) / 3600
                    if stop_requested(d):
                        log(d, "STOP file found: no new cases")
                        queue.clear()
                        break
                    if spent_h + est_min / 60 / max(1, a.jobs) > budget_h:
                        log(d, f"budget: {spent_h:.2f} of {budget_h} h used; the next case would go past it")
                        queue.clear()
                        break
                    x = queue.pop(0)
                    try:
                        ok = prepare(x)
                    except Exception as e:  # noqa: BLE001
                        x.update(state="failed", note=f"build: {type(e).__name__}: {e}")
                        log(d, f"{x['id']}: FAILED in the build ({type(e).__name__}: {e})")
                        ok = False
                    save_batch(d, b)
                    if ok:
                        running[pool.submit(solve_case, runs / x["id"], ranks, a.runner, a.wsl, timeout_s)] = x
                for fut in [f for f in running if f.done()]:
                    x = running.pop(fut)
                    try:
                        finish(x, fut.result())
                    except Exception as e:  # noqa: BLE001
                        x.update(state="failed", note=f"{type(e).__name__}: {e}")
                        log(d, f"{x['id']}: FAILED ({type(e).__name__}: {e})")
                    save_batch(d, b)
                time.sleep(0.2)
        save_batch(d, b)


# ---------------------------------------------------------------------------------------------------- other solvers

def export_batch(d: Path) -> None:
    b = load_batch(d)
    fam = registry.load_family(b["family_id"])
    g = d / "geometry"
    for x in b["designs"]:
        if x["state"] == "drawn":
            fam["export_geometry"](x["p"], g, x["id"])
            x["state"] = "exported"
    save_batch(d, b)
    (d / "touchstone").mkdir(exist_ok=True)
    (d / "EXTERNAL_SOLVER.md").write_text(external_instructions(b))
    log(d, f"geometry of {len(b['designs'])} designs in {g}; read {d / 'EXTERNAL_SOLVER.md'}, then put <id>.s2p "
           f"files in {d / 'touchstone'}")


def external_instructions(b: dict) -> str:
    s = b["solver"]
    return "\n".join([
        f"# Solve batch {b['batch_id']} with {s['name']} {s['version']}", "",
        "1. For each design, `geometry/<id>.json` gives the stack, the copper rectangles and the ports. "
        "`geometry/<id>.dxf` has the same copper as closed polylines on layer COPPER, in mm.",
        "2. Unite all copper rectangles of one design into one zero-thickness PEC sheet on top of the substrate. "
        "The ground plane is PEC at z = 0.",
        "3. Put a 50 ohm port at the outer end of each feed. De-embed both ports by the feed length, to the start and "
        "the end of the coupled section (`reference_plane_x` in the JSON).",
        "4. Solve 3-7 GHz, with steps of 0.02 GHz or finer.",
        "5. Export a two-port Touchstone file as `touchstone/<id>.s2p` (RI, MA or DB; any unit).",
        "6. Write the mesh size into `touchstone/<id>.json`, for example "
        '`{"mesh_elements": 2400000, "mesh_unknowns": 2400000}`, or give it once for the batch with '
        "`import --mesh-elements N --mesh-unknowns N`.",
        "7. Run `python tools/contribute.py go --batch " + b["batch_id"] + "` again.", "",
        "Records from this solver are grouped as fidelity `other`. The project publishes no accuracy comparison "
        "per solver or between solvers. You are responsible for what your solver licence allows you to share.", ""])


def import_batch(d: Path, a) -> None:
    b = load_batch(d)
    ts = d / "touchstone"
    found = 0
    for x in b["designs"]:
        if x["state"] in TERMINAL or x["state"] == "collected":
            continue
        f = next((p for p in sorted(ts.glob(f"{x['id']}.*")) if p.suffix.lower() in (".s2p", ".ts")), None)
        if f is None:
            continue
        side = ts / f"{x['id']}.json"
        meta = json.loads(side.read_text()) if side.exists() else {}
        x["source"] = f.name
        x["mesh_elements"] = meta.get("mesh_elements", a.mesh_elements or b["solver"]["mesh_elements"])
        x["mesh_unknowns"] = meta.get("mesh_unknowns", a.mesh_unknowns or b["solver"]["mesh_unknowns"])
        x["state"] = "collected"
        found += 1
    save_batch(d, b)
    left = [x["id"] for x in b["designs"] if x["state"] == "exported"]
    log(d, f"imported {found} Touchstone file(s); {len(left)} design(s) still without a result")


# ---------------------------------------------------------------------------------------------------- check

def record_path(b: dict, x: dict) -> str:
    return f"{b['family_id']}/contrib/{b['batch_id']}/{x['id']}.npz"


def check_batch(d: Path) -> None:
    import check_sample as cs
    b = load_batch(d)
    s = b["solver"]
    (d / "contract").mkdir(exist_ok=True)
    (d / "rows").mkdir(exist_ok=True)
    for x in b["designs"]:
        if x["state"] != "collected":
            continue
        if s["name"] == "palace":
            src = d / "records" / f"{x['id']}.npz"
            p = None
        else:
            src = d / "touchstone" / x["source"]
            p = x["p"]
        row = cs.check(src, b["family_id"], p=p, solver=s["name"], solver_version=s["version"], fidelity=s["fidelity"],
                       contributor=b["contributor"], licence=b["licence"], mesh_elements=x.get("mesh_elements"),
                       mesh_unknowns=x.get("mesh_unknowns"), order=s["order"], port=s["port"], sample_id=x["id"])
        if row["verdict"] != "FAIL":
            out = d / "contract" / f"{x['id']}.npz"
            cs.write_contract_npz(row, out)
            row = cs.check(out, b["family_id"], sample_id=x["id"])          # the row of the file that is submitted
        row.pop("_arrays", None)
        row["path"], row["split"], row["batch_id"] = record_path(b, x), "train", b["batch_id"]
        (d / "rows" / f"{x['id']}.json").write_text(json.dumps(row, indent=1, sort_keys=True))
        x.update(state="checked", verdict=row["verdict"], flags=row["flags"], errors=row["errors"])
        log(d, f"{x['id']}: {row['verdict']}" + (f" ({', '.join(row['flags'] + row['errors'])})"
                                                   if row["flags"] or row["errors"] else ""))
    save_batch(d, b)


# ---------------------------------------------------------------------------------------------------- submit

def stage(d: Path, incoming: Path) -> Path:
    """incoming/<family>/<batch>/: batch.json, rows.jsonl and records/ (PASS and NEEDS REVIEW only)."""
    b = load_batch(d)
    if b["local"]:
        raise SystemExit("a --local batch cannot be submitted (it has no claim)")
    open_ = [x["id"] for x in b["designs"] if x["state"] not in TERMINAL]
    if open_:
        raise SystemExit(f"{len(open_)} design(s) are not finished: {', '.join(open_[:5])}")
    keep = [x for x in b["designs"] if x.get("verdict") in ("PASS", "NEEDS REVIEW")]
    if not keep:
        raise SystemExit("no record passed the checker; nothing to submit")
    out = incoming / b["family_id"] / b["batch_id"]
    if out.exists():
        shutil.rmtree(out)
    (out / "records").mkdir(parents=True)
    rows = []
    for x in keep:
        shutil.copy2(d / "contract" / f"{x['id']}.npz", out / "records" / f"{x['id']}.npz")
        rows.append(json.loads((d / "rows" / f"{x['id']}.json").read_text()))
    (out / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    review = {x["id"]: b["review_notes"].get(x["id"], "") for x in keep if x["verdict"] == "NEEDS REVIEW"}
    pub = {k: b[k] for k in ("schema", "batch_id", "family_id", "contributor", "licence", "seed", "created", "solver",
                             "claim", "notes")}
    pub["review_notes"] = review
    pub["designs"] = [{k: x.get(k) for k in ("id", "p", "state", "verdict", "flags", "wall_s", "note") if k in x}
                      for x in b["designs"]]
    pub["counts"] = {v: sum(1 for x in b["designs"] if (x.get("verdict") or x["state"]) == v)
                     for v in ("PASS", "NEEDS REVIEW", "FAIL", "rejected", "failed")}
    (out / "batch.json").write_text(json.dumps(pub, indent=1))
    log(d, f"staged {len(rows)} record(s) in {out}")
    missing = [i for i, t in review.items() if not t.strip()]
    if missing:
        log(d, f"{len(missing)} NEEDS REVIEW record(s) need one sentence each: add them to review_notes in "
               f"{d / 'batch.json'} and run submit again ({', '.join(missing[:5])})")
    return out


def open_pr(out: Path, b: dict, repo: str, remote: str) -> None:
    branch = f"data/{b['batch_id']}"
    n = len((out / "rows.jsonl").read_text().splitlines())
    title = f"data: {b['family_id']}, {n} samples, {b['solver']['name']} {b['solver']['fidelity']}"
    rel = out.relative_to(REPO).as_posix()
    body = (f"Batch `{b['batch_id']}` from `tools/contribute.py`.\n\nClaim: {b['claim']['url'] or 'none'}\n\n"
            f"Files: `{rel}/` (records, rows.jsonl, batch.json). CI reruns the checker on every record.\n")
    git = ["git", "-C", str(REPO)]
    subprocess.run(git + ["switch", "-c", branch], check=True)
    subprocess.run(git + ["add", rel], check=True)
    subprocess.run(git + ["commit", "-s", "-m", title], check=True)
    subprocess.run(git + ["push", "-u", remote, branch], check=True)
    user = gh("api", "user", "--jq", ".login").stdout.strip()
    r = gh("pr", "create", "--repo", repo, "--head", f"{user}:{branch}", "--title", title, "--body", body, check=False)
    if r.returncode != 0:
        raise SystemExit(f"gh pr create failed: {r.stderr.strip()}")
    print(r.stdout.strip())


def print_pr_commands(out: Path, b: dict) -> None:
    rel = out.relative_to(REPO).as_posix() if out.is_relative_to(REPO) else str(out)
    print("\nTo open the pull request yourself (from your fork's clone):\n"
          f"  git switch -c data/{b['batch_id']}\n  git add {rel}\n"
          f"  git commit -s -m \"data: {b['family_id']}, {b['solver']['name']} {b['solver']['fidelity']}\"\n"
          f"  git push -u origin data/{b['batch_id']}\n  gh pr create --repo {REPO_SLUG} --fill\n"
          "Or run submit with --open-pr to do these steps.")


# ---------------------------------------------------------------------------------------------------- status

def status(d: Path) -> dict:
    b = load_batch(d)
    counts: dict[str, int] = {}
    for x in b["designs"]:
        k = x.get("verdict") or x["state"]
        counts[k] = counts.get(k, 0) + 1
    print(f"batch {b['batch_id']} ({b['family_id']}, {b['solver']['name']} {b['solver']['fidelity']}, "
          f"{len(b['designs'])} designs): " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    c = b["claim"]
    print("claim: " + ("not needed (--local)" if b["local"] else
                       f"{c['url'] or 'not opened'}, {'accepted' if c['accepted'] else 'not accepted yet'}"))
    return counts


# ---------------------------------------------------------------------------------------------------- go

def go(a) -> int:
    if not a.batch:
        d = new_batch(a)
        b = load_batch(d)
        if b["local"]:
            print(f"\nLocal batch {b['batch_id']}. Next: python tools/contribute.py go --batch {b['batch_id']} "
                  "--budget-hours <hours>")
            return 0
        if a.open_claim:
            open_claim(d, b, a.repo)
        else:
            print(f"\nClaim text: {d / 'claim.md'}\nOpen the claim: python tools/contribute.py go --batch "
                  f"{b['batch_id']} --open-claim, or open an issue with the 'Claim designs to solve' form.")
        print(f"After the maintainer accepts the claim, run: python tools/contribute.py go --batch {b['batch_id']} "
              "--budget-hours <hours>")
        return 0
    d = batch_dir(a.work, a.batch)
    b = load_batch(d)
    if a.open_claim and not b["claim"]["issue"] and not b["local"]:
        open_claim(d, b, a.repo)
    if a.claim_accepted:
        b["claim"]["accepted"] = True
        save_batch(d, b)
    if not claim_accepted(d, b, a.repo):
        status(d)
        print("Wait for the maintainer to accept the claim (label status:accepted), then run go again.")
        return 3
    if b["solver"]["name"] == "palace":
        run_batch(d, a)
    else:
        if any(x["state"] == "drawn" for x in b["designs"]):
            export_batch(d)
        import_batch(d, a)
    check_batch(d)
    counts = status(d)
    b = load_batch(d)
    if any(x["state"] not in TERMINAL for x in b["designs"]):
        print("Not finished yet. Run go again to continue.")
        return 4
    if b["local"]:
        print("Local batch finished. Records: " + str(d / "contract"))
        return 0
    out = stage(d, a.incoming)
    if a.open_pr:
        open_pr(out, load_batch(d), a.repo, a.remote)
    else:
        print_pr_commands(out, b)
    return 0 if counts else 1


# ---------------------------------------------------------------------------------------------------- CLI

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work", type=Path, default=WORK, help="batch folders (default: work/ in the repo)")
    common.add_argument("--incoming", type=Path, default=INCOMING, help="where submit stages files (default: incoming/)")
    common.add_argument("--repo", default=REPO_SLUG, help="GitHub repo for claims and pull requests")
    newp = argparse.ArgumentParser(add_help=False)
    newp.add_argument("--family", default=registry.DEFAULT_FAMILY, choices=registry.family_ids())
    newp.add_argument("--n", type=int, default=20, help="designs in the batch")
    newp.add_argument("--contributor", help="your GitHub handle")
    newp.add_argument("--seed", type=int, help="sampler seed (default: random)")
    newp.add_argument("--solver", default="palace", help="palace (runs here) or hfss, cst, openems, ... (you run it)")
    newp.add_argument("--solver-version")
    newp.add_argument("--fidelity", help="default: the family's production fidelity for Palace, 'other' otherwise")
    newp.add_argument("--order", type=int, help="element order (FEM); 0 for FDTD/MoM")
    newp.add_argument("--port", help="port definition in words (other solvers)")
    newp.add_argument("--notes", help="one paragraph on your setup: machine, solver build, anything different")
    newp.add_argument("--local", action="store_true", help="practice or own use: no claim, cannot be submitted")
    runp = argparse.ArgumentParser(add_help=False)
    runp.add_argument("--budget-hours", type=float, help="machine hours you allow for this run")
    runp.add_argument("--timeout-min", type=float, help="per case (default 4 x the family estimate)")
    runp.add_argument("--ranks", type=int, help="MPI ranks per job (default from the family cost table)")
    runp.add_argument("--jobs", type=int, default=1, help="jobs at a time")
    runp.add_argument("--runner", help="Palace command called as `<runner> config.json <ranks>` "
                                       "(default: $RUN_PALACE, else tools/run_palace.sh)")
    runp.add_argument("--retry-failed", action="store_true", help="solve failed designs again")
    runp.add_argument("--wsl", help="Windows: run Palace in this WSL distribution (e.g. Ubuntu-24.04)")
    meshp = argparse.ArgumentParser(add_help=False)
    meshp.add_argument("--mesh-elements", type=int, help="other solvers: elements or cells of the solved mesh")
    meshp.add_argument("--mesh-unknowns", type=int, help="other solvers: unknowns (FEM) or cells (FDTD)")

    g = sub.add_parser("go", parents=[common, newp, runp, meshp], help="do every step that can be done now")
    g.add_argument("--batch", help="continue this batch (omit to start a new one)")
    g.add_argument("--open-claim", action="store_true", help="open the claim issue with gh")
    g.add_argument("--claim-accepted", action="store_true", help="the claim is accepted (when gh cannot check)")
    g.add_argument("--open-pr", action="store_true", help="commit, push and open the pull request with git and gh")
    g.add_argument("--remote", default="origin", help="git remote of your fork")
    n = sub.add_parser("new", parents=[common, newp, runp, meshp], help="draw designs and write the batch")
    n.add_argument("--batch", help="batch id (default <contributor>-<seed>)")
    for name, extra, hlp in (("status", [], "show the state of a batch"),
                             ("run", [runp], "build, solve and collect (Palace)"),
                             ("export", [], "write the geometry for another solver"),
                             ("import", [meshp], "read Touchstone results of another solver"),
                             ("check", [], "check every collected record"),
                             ("submit", [], "stage the pull request files")):
        p = sub.add_parser(name, parents=[common, *extra], help=hlp)
        p.add_argument("--batch", required=True)
        if name == "submit":
            p.add_argument("--open-pr", action="store_true")
            p.add_argument("--remote", default="origin")
    sub.add_parser("unlock", help="remove a launcher lock that a crash left behind")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.cmd == "unlock":
        p = lock_dir() / "launcher.lock"
        print(f"removed {p}" if p.exists() else "no lock")
        p.unlink(missing_ok=True)
        return 0
    if a.cmd == "go":
        return go(a)
    if a.cmd == "new":
        new_batch(a)
        return 0
    d = batch_dir(a.work, a.batch)
    if a.cmd == "status":
        status(d)
    elif a.cmd == "run":
        run_batch(d, a)
    elif a.cmd == "export":
        export_batch(d)
    elif a.cmd == "import":
        import_batch(d, a)
    elif a.cmd == "check":
        check_batch(d)
    elif a.cmd == "submit":
        out = stage(d, a.incoming)
        if a.open_pr:
            open_pr(out, load_batch(d), a.repo, a.remote)
        else:
            print_pr_commands(out, load_batch(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
