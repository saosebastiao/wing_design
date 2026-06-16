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
- **Memory-budget parallel runs.** Sum of worker peak footprints ≤ ~half of machine
  RAM, and never stack independent heavy runs concurrently. Record `ru_maxrss`
  (self + children) in run meta. **Measured:** an IPOPT medium-16×8 worker peaks
  **~18 GiB** (not the ~9 GiB first estimated) → on this 32 GiB machine the safe
  cap is **n_workers=1**; 2 is over the edge (two coincident peaks ≈ 36 GiB).
  8-way multistart oversubscribed RAM and caused two watchdog kernel panics
  (2026-06-11/12) — see findings.md.
- **A sizing result counts only if converged AND feasible.** Check both before
  reporting any mass. Comparisons within ~2–3% are inside the ftol/basin noise floor —
  don't claim wins there.
- **Verify buckling at a CONVERGED mesh (n_levels ≥ 24), not the 16×8 sizing mesh.**
  The n=8 linear-buckling eigen is unconservatively overstiff (it can't resolve the
  short-wavelength mode): it overshoots low at n≈12 then settles to a converged
  cluster floor by n≥24. Judge feasibility on that floor over the lowest 2–3 modes
  (eigen-only, ~15–40 s/mesh). The closed-form foundation model is well-calibrated
  to the converged eigen for the OPERATIONAL envelope but NON-conservative for slam
  (2026-06-16) — so slam-sized designs especially must be eigen-verified at n≥24.
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

## Pending upgrade

- **Python 3.14 when build123d > 0.10.0 releases.** We're on 3.13; 3.14 is blocked
  solely by build123d 0.10.0 (`requires-python <3.14`, OCP pin `<7.9`). cp314 OCP
  wheels already exist and build123d's dev branch already supports `<3.15` + OCP 7.9.
  When a new build123d release appears: bump `requires-python` to `>=3.14,<3.15` and
  `.python-version` to 3.14, `uv lock && uv sync`, run the full suite, commit. 3.14
  carries the real interpreter gains (tail-call interpreter, incremental GC).

## Layout notes

- `examples/` numbered scripts are the measurement record; not run in CI.
- `tests/` is the fast unit suite (`uv run pytest`).
- `docs/superpowers/` holds per-feature design specs and implementation plans.
