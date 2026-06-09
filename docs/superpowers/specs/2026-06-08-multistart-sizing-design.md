# Multi-start sizing: design

**Date:** 2026-06-08
**Status:** Approved (brainstorming → ready for implementation plan)

## Goal

Add a multi-start wrapper around `size_beam_shell_laminate` that runs the sizer from N
initial guesses and returns the **best feasible** result, to escape SLSQP local optima.
Built and verified cheaply; **no actual multi-start run is performed by the implementer**
(user constraint) — the orchestration is tested with a stubbed sizer, and the validation
example is created compile-checked-only for the user to run if they choose.

## Motivation

SLSQP is a local optimizer; the datum+buckling design reached 30.15 kg while the
per-triangle-local variant hit 26.06 kg, hinting at multiple basins. Random-restart
multi-start is the simplest way to probe for a better basin. The sizer is expensive
(FEA solve per `evaluate` × FD-Jacobian × maxiter), so multi-start is N× that — this
increment delivers the **machinery** and cheap correctness verification, deferring any
full-scale headline run.

## Decisions (from brainstorming + constraints)

- **Serial, parallel-ready:** run the N starts in a single serial map, structured so it
  can later be swapped for `multiprocessing.Pool` without an interface change.
- **Deliverable = machinery + cheap verification.** Defer full-scale headline.
- **Do NOT run the process:** the implementer does not execute any multi-start sizing
  (no example run, no headline). Orchestration is tested via a stubbed sizer; the
  example is compile-checked only.

## Design

### 1. Expose the start to the sizer — `beams/laminate_sizing.py`

- Add `x0: np.ndarray | None = None` to `size_beam_shell_laminate`. `None` reproduces
  today's internal default guess (`radii=r_max`, `t_band=t_max`, `f*_grp=1/3`) exactly —
  **backward-compatible**. When provided, `x0` is clipped to the design bounds and its
  layup groups are fraction-simplex-projected (per group: if `f0_g+f45_g>1`, rescale),
  then used as the SLSQP start. Length must equal `nx = n + B + 2L`.
- Extract the bounds construction into a reusable helper:
  ```
  laminate_design_bounds(model, config) -> tuple[np.ndarray, np.ndarray]   # (lo, hi), each (nx,)
  ```
  ordered `[radii(n), t_band(B), f0_grp(L), f45_grp(L)]` (`L = B if per_band_layup else 1`).
  Both the sizer and the multi-start wrapper use it (one source of truth for the layout).

### 2. Feasibility check — `laminate_result_is_feasible`

```
laminate_result_is_feasible(result: LaminateSizingResult, config: LaminateSizingConfig, *, tol: float = 1.0e-3) -> bool
```
True iff all active constraints hold within relative `tol`:
- `max_beam_vm_Pa <= sigma_allow_Pa * (1 + tol)`;
- skin: if `config.skin_failure == "tsai_wu"`, `min_skin_strength_ratio >=
  config.tsai_wu_safety_factor * (1 - tol)`; else `max_skin_vm_Pa <= sigma_allow_Pa *
  (1 + tol)`;
- `tip_defl_m <= tip_defl_max_m * (1 + tol)`;
- `tip_twist_deg <= tip_twist_max_deg * (1 + tol)`;
- if `buckling_safety_factor is not None`: `max_beam_buckling_util <= 1 + tol` AND
  `max_panel_buckling_util <= 1 + tol`.
This is exactly what separates a real optimum from a maxed-out non-converged run (e.g.
the coarse per-band run that ended at beam-buckling util 1.70).

### 3. Multi-start wrapper — `size_beam_shell_laminate_multistart`

```
size_beam_shell_laminate_multistart(model, loads, config, *, ply, rho,
    n_starts: int = 8, seed: int = 0, maxiter: int = 80, ftol: float = 1.0e-4,
) -> MultiStartResult
```
- **Start 0 = the default guess** (`x0=None`), so the returned best is never worse than
  today's single-start result.
- Starts `1..n_starts-1`: sampled uniform within `laminate_design_bounds`, then
  per-group fraction-simplex-projected, from a seeded `np.random.default_rng(seed)`
  (deterministic for a given seed).
- Run each start: `size_beam_shell_laminate(model, loads, config, ply=ply, rho=rho,
  maxiter=maxiter, ftol=ftol, x0=start)`. Implement the start loop as ONE serial map
  (list comprehension / for-loop) so a later `multiprocessing.Pool.map` swap is local.
- **Selection:** among results where `laminate_result_is_feasible(...)`, choose min
  `mass_kg`. If none feasible, choose min `mass_kg` overall and set `n_feasible = 0`.
- Return `MultiStartResult` (frozen dataclass):
  - `best: LaminateSizingResult`
  - `best_start_index: int`
  - `n_starts: int`
  - `n_feasible: int`
  - `start_masses: tuple[float, ...]`
  - `start_feasible: tuple[bool, ...]`

`n_starts >= 1` required (`n_starts == 1` ⇒ just the default start = single-start sizing).

### 4. Exports — `beams/__init__.py`

Export `size_beam_shell_laminate_multistart`, `MultiStartResult`,
`laminate_result_is_feasible`, `laminate_design_bounds`.

### 5. Example — `examples/36_multistart.py` (compile-checked only, NOT run)

A cheap validation harness (for the user to run): medium scenario but a small mesh
(n_levels=5, n_beams=8), `n_starts=4`, low `maxiter`, so it would run in minutes. Prints
each start's mass + feasibility and the chosen best, demonstrating multi-start ≥
single-start. Docstring states it validates the machinery, not a converged design, and
that it is intentionally small. The implementer does NOT execute it.

### 6. Testing — `tests/beams/test_multistart.py`

Orchestration is tested WITHOUT running real sizings, by monkeypatching the sizer:

- `test_design_bounds_shape`: `laminate_design_bounds` returns `(lo, hi)` of length
  `n + B + 2L`, `lo < hi`, fraction entries bounded `[0, 1]` (no real sizing).
- `test_feasible_check`: build `LaminateSizingResult` instances by hand — one satisfying
  all limits (feasible True) and one with `max_beam_buckling_util = 1.7` (feasible False)
  — and assert `laminate_result_is_feasible` accordingly (no real sizing).
- `test_multistart_selects_best_feasible` (monkeypatched): patch
  `wing_design.beams.laminate_sizing.size_beam_shell_laminate` with a stub that returns
  canned `LaminateSizingResult`s keyed by call index (varying mass + feasibility, one
  infeasible-but-lighter to prove it's rejected). Assert: `best` is the lightest
  *feasible* one; `best_start_index` matches; `start_masses`/`start_feasible` lengths ==
  `n_starts`; `n_feasible` correct. Run twice with the same seed → identical selection
  (determinism). No real FEA.
- `test_multistart_never_worse_than_default` (monkeypatched): stub returns the default
  start (index 0) as feasible; assert `best.mass_kg <= start_masses[0] + 1e-9`.
- `test_x0_roundtrip_backward_compat` (ONE real small sizing — same cost class as the
  existing suite): `size_beam_shell_laminate(..., x0=None)` equals a call with `x0` set
  to the explicit default vector (same mass/radii, allclose) on a small problem
  (n_levels=5, n_beams=8, maxiter≤15).

The existing laminate/banded/tsai-wu/per-band tests cover the `x0=None` default path and
must stay green.

## Components / file map

| File | Responsibility | Change |
|---|---|---|
| `beams/laminate_sizing.py` | `x0` param; `laminate_design_bounds`; `laminate_result_is_feasible`; `size_beam_shell_laminate_multistart` + `MultiStartResult` | modify |
| `beams/__init__.py` | export the four new symbols | modify |
| `examples/36_multistart.py` | cheap validation harness (compile-checked, not run) | **new** |
| `tests/beams/test_multistart.py` | stubbed-orchestration + feasibility + x0 tests | **new** |
| `docs/plan.md` | finding + Decisions-log row (at close) | modify |

## Success criteria

- `x0=None` keeps `size_beam_shell_laminate` byte-identical to today (full suite green).
- Multi-start returns the best **feasible** start, is deterministic under a fixed seed,
  and never worse than the default single start — all proven with a stubbed sizer (no
  real multi-start executed).
- `laminate_design_bounds` / `laminate_result_is_feasible` are reusable and unit-tested.
- The validation example exists and compiles (not run).

## Out of scope (deferred)

- Parallel execution (`multiprocessing`) — structured for it, not implemented.
- Any full-scale converged multi-start headline run.
- Global optimizers (basin-hopping, CMA-ES) — random restarts only.
- Auto-tuning `n_starts`/`seed`/perturbation distribution.
