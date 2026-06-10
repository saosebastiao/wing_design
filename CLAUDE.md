# wing_design — project conventions

Free-rotating CFRP wingsail designed as shell-following form beams + filament-wound
load-bearing skin, sized FEA-in-the-loop. Scales 1 m drone wing → 100 m turbine blade.

## Documents (read before structural/sizing work)

- `docs/plan.md` — the forward plan ONLY (phases V/M/P/G/H). No findings here.
- `docs/findings.md` — append-only archive of analyses, findings, decisions log.
  All new findings and decisions go HERE, in the established house format.
- `docs/improvement_backlog.md` — classified backlog (validity / manufacturability /
  performance / toolbox) + the intended manufacturing concept. Cross-ref as V#/M#/P#.

## Non-negotiable conventions

- **Measure, never estimate, wall-clock.** Every FEA/sizing run is timed (`time` or
  Bash timing) and the measured duration recorded in the finding. Estimates have run
  ~10× off.
- **A sizing result counts only if converged AND feasible.** Check both before
  reporting any mass. Comparisons within ~2–3% are inside the ftol/basin noise floor —
  don't claim wins there.
- **Negative results are findings too.** Record them with the same rigor (see the tip
  gusset / diagonal lattice entries in findings.md for the pattern).
- **Milestones export CAD + analyses.** Any important optimization-progress milestone
  (new headline mass, new lever validated, re-baseline) must produce exported — or at
  minimum exportable via an example — CAD geometry (STL/STEP through the sized-export
  path, e.g. `examples/37_sized_export.py`) and the supporting analysis artifacts
  (FEA fields to `exports/*.vtu`, screenshots via `just shot`), referenced from the
  finding. exports/ is gitignored: regenerability is the requirement, the finding
  records how.
- **Run python through `just`** (`just py`, `just example NN_name`) so
  PYTHONPYCACHEPREFIX keeps bytecode out of the tree. Tooling interpreter:
  `.venv/bin/python`.
- **Legacy sizers are frozen.** `beams/sizing.py` and `beams/shell_sizing.py` are
  historical baselines; new features go to the laminate path
  (`laminate_sizing.py` + `sensitivity.py`) only.
- **Every new analytic gradient gets an FD-validation test** (central difference,
  rel-err ≤ ~1e-4, tiny mesh — follow `tests/beams/test_sensitivity.py` patterns).
- **Warm-start sweeps.** Sweep/refinement points start from the nearest prior optimum
  (`x0=`), fine meshes from resampled coarse solutions.

## Project skills (in .claude/skills/)

- `nlp-sizing` — SLSQP/IPOPT conventions, multipliers, KS, convergence pitfalls.
- `sizing-run` — how to launch/monitor/report long sizing runs.
- `record-finding` — the findings.md + decisions-log recording workflow.
- `justfile-tasks` — project task runner recipes and how to extend them.
- `ortools-cp` — CP-SAT modeling (stock catalog / discrete selection).

## Layout notes

- `examples/` numbered scripts are the measurement record; not run in CI.
- `tests/` is the fast unit suite (`uv run pytest`).
- `docs/superpowers/` holds per-feature design specs and implementation plans.
