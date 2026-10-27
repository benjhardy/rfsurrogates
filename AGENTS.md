# Rules for agents in this repo

These rules apply to any coding agent (Claude Code, Codex, Cursor, others) working in rfsurrogates. A person can do
any of this work too, and passes the same checks. `docs/AGENT_ROLES.md` has the full role design.

## People hold the gates

- **The contributor** sets the solver budget on their own machine, approves the claim issue and approves the pull
  request.
- **The maintainer** accepts claims, merges, uploads to Hugging Face, releases and publishes.
- Ask before any push, pull request, issue, upload or other public action. A yes for one action is not a yes for the
  next.

## Separation of duties

- The step that makes data does not verify it: the checker and CI verify every record.
- Do not edit a record, a manifest row, a split or the holdout list. Do not change the checker or a family rule to
  make a record pass.
- Training does not choose or edit data. Evaluation is not done by the step that trained.

## Solver runs

- Start solvers only through `tools/contribute.py` (the launcher): one launcher per machine, a STOP file, a budget the
  contributor set, a timeout per case, case folders only under `work/`.
- To contribute data, use the `contribute-data` skill (`.claude/skills/contribute-data/SKILL.md`).

## Text and numbers

- A number in a document comes from a result file you can cite.
- Public text (README, docs, pages) is short and plain: one idea per sentence, active voice, no slogans.
- Never publish an accuracy comparison per solver or between solvers.

## Code

- Tests next to the code: `python -m pytest tests -q`.
- Contributor commits carry a DCO sign-off (`git commit -s`).
