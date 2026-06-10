# P.0 Sizer extensibility refactor (DesignVector + Constraint): design

**Date:** 2026-06-10
**Status:** Draft (plan.md P.0; Toolbox #1 — the gate for P.1 hollow members)

## Why now

`size_beam_shell_laminate` has grown to ~700 lines through the 2026-06-10 validity
sprint (shadow prices, strip widths, body loads + dF, panel pressures + qw2 all
threaded by hand). Adding one constraint or DV block touches ~6 places (evaluate
loop, constraint closure, scipy dict + idx bookkeeping, sensitivity gradient,
Jacobian registration, result/feasibility). P.1 adds a member type with new DV
blocks (`r_tube(z)`, `t_wall(z)`) AND new constraints (wall buckling/crimping) —
the wrong shape to do by hand again. User direction (2026-06-10): performance
harvest is the priority; slam re-introduction waits on it.

## Scope (behavior-preserving)

1. **`DesignVector`** — named blocks with explicit slices:
   `dv = DesignVector(("r_group", G), ("t_band", B), ("f0", L), ("f45", L))`;
   `dv.unpack(x) -> dict[str, np.ndarray]`, `dv.slice("t_band") -> slice`,
   `dv.nx`. One pack/unpack; no hand-indexed `x[G:G+B]` anywhere. P.1 appends
   `("r_tube", K), ("t_wall", K)` without touching existing blocks.
2. **`Constraint`** — small object: `name`, `fun(xd) -> np.ndarray` (normalized,
   feasible ≥ 0), optional `jac(xd) -> (m, nx)`, `feasible(result) -> bool` entry,
   `shadow_key: str | None` (+ the unit conversion for V.1 prices). One driver
   loop builds the scipy dicts, registers Jacobians, runs the feasibility check,
   and attributes multipliers — the row-layout bookkeeping in `_shadow_prices`
   becomes structural instead of positional.
3. **Evaluate context** — one per-design-point cache object holding the decoded
   design, laminates, effective loads (V.4 combos), solves/factorizations,
   SensCache, dFs, qw2 — replacing the closure-captured locals. Constraints
   consume it.
4. **No formulation change.** Identical x layout for existing configs, identical
   normalized constraints in the same order (shadow-price layout preserved),
   identical gradients (the closures move, the math doesn't).

## Acceptance (per plan: behavior-preserving)

- Full test suite passes unchanged (179 tests incl. the sizing-marked SLSQP runs).
- A medium strip-mode headline re-run reproduces the V.6 number exactly (the
  determinism the sprint demonstrated repeatedly: same config → same 0.1 kg).
- `examples/42` shadow prices reproduce.

## Out of scope

IPOPT switch (trigger: SLSQP stalls or DVs > ~150 — watch during P.1); KS
aggregation; any new constraint physics.
