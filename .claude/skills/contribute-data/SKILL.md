---
name: contribute-data
description: Walk a contributor through adding simulation data to rfsurrogates with tools/contribute.py - draw designs, claim them, solve them on their own machine (Palace, or HFSS/CST/openEMS by export and import), check every record and open one pull request. Use when someone says "contribute data", "solve a batch", "run designs for rfsurrogates", "submit records", or asks how to help with compute.
---

# Contribute data to rfsurrogates

You help a contributor run `tools/contribute.py` on their own machine. The contributor owns the compute and the
decisions. You run commands, explain results and ask at every gate. Read `AGENTS.md` first.

## Gates: always ask, never assume

| step | ask the contributor | never |
|---|---|---|
| batch size and budget | how many designs, and how many machine hours they allow | choose the budget yourself |
| claim issue | show `work/<batch>/claim.md`, then ask before `--open-claim` | open it without a yes |
| NEEDS REVIEW | one sentence per flagged record, in their words | write the sentence yourself |
| pull request | show the staged files, then ask before `--open-pr` | push or open it without a yes |
| stale lock | whether another run is still going | run `unlock` without a yes |

## Steps

1. **Check the setup.** A clone of the contributor's fork, a virtual environment with `pip install -r requirements.txt`.
   For Palace: `palace` on PATH, or `PALACE_SIF` set to an Apptainer image, or their own command for `--runner`
   (called as `<command> config.json <ranks>`). On Windows, Palace runs in WSL: add `--wsl <distribution>`.
2. **First run.** Ask for their GitHub handle, the number of designs (start with 5 to 20) and the solver.
   ```
   python tools/contribute.py go --n 20 --contributor <handle>
   python tools/contribute.py go --n 20 --contributor <handle> --solver hfss --solver-version <version>
   ```
   It prints the cost estimate. Show it. Then show `claim.md` and ask before you open the claim:
   `python tools/contribute.py go --batch <batch> --open-claim`.
3. **Wait for the claim.** The maintainer adds `status:accepted`. `go --batch <batch>` checks it. Do not check more
   often than once an hour.
4. **Solve.** Palace:
   ```
   python tools/contribute.py go --batch <batch> --budget-hours <their number>
   ```
   Run it in the background; it can take hours. It continues where it stopped, so a second `go` is safe.
   To stop it, create the file `work/<batch>/STOP`. Other solvers: `go` exports `work/<batch>/geometry/` and
   `EXTERNAL_SOLVER.md`. The contributor solves each design and saves `work/<batch>/touchstone/<id>.s2p`. Then run
   `go --batch <batch> --mesh-elements N --mesh-unknowns N` with their mesh size.
5. **Read the verdicts.** `python tools/contribute.py status --batch <batch>`.
   - PASS: nothing to do.
   - NEEDS REVIEW: show the flags from `work/<batch>/rows/<id>.json`. Ask the contributor for one sentence on why the
     result looks like that. Put it in `review_notes` in `work/<batch>/batch.json`.
   - FAIL, rejected or failed: report the note. A FAIL is not submitted. Do not edit a record, a row or the checker,
     and do not rerun the checker with other arguments to change a verdict.
   - Solver failures can be retried with `--retry-failed` after the cause is fixed.
6. **Submit.** `go` stages `incoming/<family>/<batch>/` when every design is finished. Show the counts and the
   files. Ask, then `python tools/contribute.py submit --batch <batch> --open-pr`. It commits with a DCO sign-off,
   pushes the branch `data/<batch>` to their fork and opens the pull request.
7. **After the pull request.** CI reruns the checker on every record (`tools/verify_batch.py`). If CI reports an
   error, read the job summary, fix the cause in `work/<batch>/`, and stage again. The maintainer merges and uploads.

## Rules

- Change nothing outside `work/` and `incoming/`. A data pull request changes only `incoming/`.
- Never edit files in `incoming/` by hand; `submit` writes them.
- One launcher per machine. Do not start a second run while one is going.
- Numbers you report come from the batch files or the tool output, never from memory.
- Commercial solvers: the contributor decides what their licence allows them to share. Never write an accuracy
  comparison between solvers.
