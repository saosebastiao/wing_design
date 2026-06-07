# E.3 — MILP Stock Catalog (CP-SAT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Discretize the continuous beam radii to a manufacturable stock catalog using a CP-SAT MILP that minimizes mass under a hard cap on the number of distinct stock sizes (co-linear grouping), then verify the discrete design with one FEA solve. Show the mass-vs-part-count trade.

**Scope note (approach chosen):** Rounding a beam radius *up* from the continuous optimum is feasibility-preserving — axial stress `N/A` and bending stress `4M/(πr³)` both fall, Euler buckling utilization `N·SF/Pcr` falls (`Pcr∝r⁴`), and stiffer beams only reduce deflection/twist. So each element merely needs a stock size ≥ its continuous radius; no SLP / FEA-sensitivities are needed (the full-SLP path is deferred). The MILP is therefore a clean **min-mass assignment with a ≤K-distinct-sizes cardinality cap**. A final FEA solve verifies (and would surface the rare load-redistribution edge case). The skin is left at its continuous value; only beam radii are catalogued.

**Architecture:** `beams/stock_catalog.py::select_stock_sizes(req_radii, catalog_radii, lengths, *, rho, max_distinct_sizes)` builds a CP-SAT model: boolean `x[e,j]` (element e uses catalog size j), exactly-one per element over the *feasible* sizes (catalog ≥ req), `used[j]` channeling with `Σ used ≤ K`, objective = integer-scaled total mass. Returns a `StockSelection` (assigned radii, distinct sizes, mass). The example sizes continuously (with buckling), runs the selector at several K, and re-solves the discrete design to verify constraints.

**Tech Stack:** OR-Tools CP-SAT (9.15, installed; see `.claude/skills/ortools-cp`), numpy, the E.2 sizing + beam-shell solver + buckling helpers.

---

## Background facts (verified)

- OR-Tools CP-SAT available: `from ortools.sat.python import cp_model`.
- `beams/shell_sizing.py::size_beam_shell(model, load_arrays, config, *, rho, maxiter, ftol) -> BeamShellSizingResult` with `.radii` (per longitudinal element, `n = n_beams*(n_levels-1)`), `.mass_kg`, `.max_*`. `BeamShellSizingConfig(..., buckling_safety_factor=...)`.
- `beams/shell_sizing.py::beam_lengths(model) -> (n,)`, `skin_areas(model)`.
- `structural/beam_shell.py::solve_beam_shell(model.nodes, model.beam_elements, sections, model.shell_tris, *, E_beam, G_beam, E_skin, nu_skin, t_skin, fixed_nodes, loads) -> FrameResult`; `BeamSection.circular(r)`; `von_mises_per_element(res, sections)`.
- `structural/shell.py::recover_membrane_stress`, `membrane_von_mises`; `structural/buckling.py::beam_euler_utilization`, `panel_buckling_utilization`.
- `beams/shell_model.py::build_beam_shell_model`; `BeamShellModel` fields incl. `tip_nodes`, `E_beam/G_beam/E_skin/nu_skin`, `shell_tris`.
- Aero/scenario wiring like `examples/25_resize_with_skin.py`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/beams/stock_catalog.py` (create) | `StockSelection`, `select_stock_sizes` (CP-SAT min-mass, ≤K distinct). |
| `src/wing_design/beams/__init__.py` (modify) | Export both. |
| `examples/29_stock_catalog.py` (create) | Continuous size (w/ buckling) → MILP snap at several K → FEA-verify → mass-vs-part-count. |
| `tests/beams/test_stock_catalog.py` (create) | CP-SAT selector: feasibility (≥req), cardinality (≤K), min-mass optimum, catalog-too-small error. |
| `docs/plan.md` (modify) | Mark E.3 done. |

---

## Task 1: CP-SAT stock-size selector

**Files:**
- Create: `src/wing_design/beams/stock_catalog.py`
- Modify: `src/wing_design/beams/__init__.py`
- Test: `tests/beams/test_stock_catalog.py`

- [ ] **Step 1: Write the failing tests** (`tests/beams/test_stock_catalog.py`)

```python
import numpy as np
import pytest

from wing_design.beams.stock_catalog import select_stock_sizes


CAT = [0.010, 0.015, 0.020, 0.025]


def test_k1_uses_one_size_covering_max():
    req = [0.010, 0.012, 0.020]
    sel = select_stock_sizes(req, CAT, [1.0, 1.0, 1.0], rho=1550.0, max_distinct_sizes=1)
    assert len(sel.distinct_sizes) == 1
    assert np.allclose(sel.assigned_radii, 0.020)          # smallest single size >= max req
    assert np.all(sel.assigned_radii >= np.array(req) - 1e-12)


def test_k2_minimizes_mass():
    req = [0.010, 0.012, 0.020]
    L = [1.0, 1.0, 1.0]
    sel2 = select_stock_sizes(req, CAT, L, rho=1550.0, max_distinct_sizes=2)
    sel1 = select_stock_sizes(req, CAT, L, rho=1550.0, max_distinct_sizes=1)
    assert len(sel2.distinct_sizes) <= 2
    assert np.all(sel2.assigned_radii >= np.array(req) - 1e-12)
    assert sel2.mass_kg <= sel1.mass_kg                     # more sizes >= as light
    assert np.allclose(sorted(sel2.assigned_radii), [0.015, 0.015, 0.020])  # {0.015,0.020} optimal


def test_catalog_too_small_raises():
    with pytest.raises(ValueError):
        select_stock_sizes([0.030], [0.010, 0.020], [1.0], rho=1550.0, max_distinct_sizes=1)
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/beams/test_stock_catalog.py -v` → ImportError.

- [ ] **Step 3: Implement** `src/wing_design/beams/stock_catalog.py`

```python
"""Discretize continuous beam radii onto a manufacturable stock catalog (CP-SAT).

Rounding a radius UP from the continuous (feasible) optimum preserves feasibility —
stress and Euler-buckling utilization fall, deflection/twist improve — so each
element just needs a stock size >= its continuous radius. `select_stock_sizes` then
picks <= K distinct catalog sizes and assigns one (>= req) to each element to minimize
total beam mass (the co-linear-grouping / part-count trade). Verify the chosen
discrete design with a real FEA solve (see examples/29).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model


@dataclass(frozen=True)
class StockSelection:
    assigned_radii: np.ndarray   # (n,) chosen stock radius per element [m]
    distinct_sizes: list[float]  # the (<=K) stock radii actually used, ascending
    mass_kg: float               # total beam mass at the assigned radii


def select_stock_sizes(
    req_radii,
    catalog_radii,
    lengths,
    *,
    rho: float,
    max_distinct_sizes: int,
    mass_scale: float = 1.0e6,
) -> StockSelection:
    """Min-mass assignment of catalog stock radii to elements, <= K distinct sizes.

    Each element e gets a stock radius >= req_radii[e]; at most `max_distinct_sizes`
    distinct catalog radii are used across all elements; total mass
    `rho * sum(pi r^2 L)` is minimized. Raises ValueError if the catalog's largest
    radius cannot cover the largest required radius.
    """
    req = np.asarray(req_radii, dtype=float)
    L = np.asarray(lengths, dtype=float)
    cat = np.sort(np.asarray(catalog_radii, dtype=float))
    n, m = req.shape[0], cat.shape[0]
    if cat[-1] < req.max() - 1e-12:
        raise ValueError(
            f"catalog max radius {cat[-1]:.4g} < required {req.max():.4g}; extend the catalog"
        )

    model = cp_model.CpModel()
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for e in range(n):
        feasible = [j for j in range(m) if cat[j] >= req[e] - 1e-12]
        for j in feasible:
            x[e, j] = model.NewBoolVar(f"x_{e}_{j}")
        model.AddExactlyOne(x[e, j] for j in feasible)

    used = [model.NewBoolVar(f"used_{j}") for j in range(m)]
    for (e, j), var in x.items():
        model.AddImplication(var, used[j])
    model.Add(sum(used) <= max_distinct_sizes)

    model.Minimize(
        sum(
            int(round(rho * np.pi * cat[j] ** 2 * L[e] * mass_scale)) * var
            for (e, j), var in x.items()
        )
    )

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no solution: {solver.StatusName(status)}")

    assigned = np.empty(n)
    for e in range(n):
        for j in range(m):
            if (e, j) in x and solver.Value(x[e, j]) == 1:
                assigned[e] = cat[j]
    distinct = sorted({float(cat[j]) for j in range(m) if solver.Value(used[j]) == 1})
    mass = float(rho * np.sum(np.pi * assigned**2 * L))
    return StockSelection(assigned_radii=assigned, distinct_sizes=distinct, mass_kg=mass)
```

- [ ] **Step 4:** Export `StockSelection`, `select_stock_sizes` from `beams/__init__.py`. Run `uv run pytest tests/beams/test_stock_catalog.py -v` (3 PASS) and `uv run pytest` (green).

- [ ] **Step 5: Commit** — `feat(beams): CP-SAT stock-catalog selection (min-mass, <=K distinct sizes)` + trailer.

---

## Task 2: Demonstrator + plan update

**Files:**
- Create: `examples/29_stock_catalog.py`
- Modify: `docs/plan.md`

- [ ] **Step 1: Create `examples/29_stock_catalog.py`** — the full demonstrator:
  1. Build model + frame + LL envelope load_arrays (as in examples/25).
  2. Continuous size WITH buckling: `cfg = BeamShellSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02*spec.span, tip_twist_max_deg=5.0, buckling_safety_factor=1.5)`; `cont = size_beam_shell(model, load_arrays, cfg, rho=P.rho_kgm3, maxiter=150)`. Print continuous mass + radius range.
  3. Catalog: `catalog = np.arange(4, 21, 2) * 1e-3` (4–20 mm in 2 mm steps). Ensure `catalog.max() >= cont.radii.max()` — if not, extend (print a note).
  4. `Lb = beam_lengths(model)`.
  5. For `K in (2, 4, 8)`: `sel = select_stock_sizes(cont.radii, catalog, Lb, rho=P.rho_kgm3, max_distinct_sizes=K)`. Build per-element `sections = [BeamSection.circular(r) for r in sel.assigned_radii]`; for each load case `solve_beam_shell(model.nodes, model.beam_elements, sections, model.shell_tris, E_beam=model.E_beam, G_beam=model.G_beam, E_skin=model.E_skin, nu_skin=model.nu_skin, t_skin=cont.t_skin, fixed_nodes=model.fixed_nodes, loads=loads)`; recover worst beam vm (`von_mises_per_element`), skin vm (`membrane_von_mises(recover_membrane_stress(...))`), tip defl/twist (model.tip_nodes), beam buckling util (`beam_euler_utilization(res.axial_force, sel.assigned_radii, Lb, E=model.E_beam, K=1.0, safety_factor=1.5)`), panel buckling util (`panel_buckling_utilization(skin_s, skin_areas(model), D11=model.E_skin*cont.t_skin**3/(12*(1-model.nu_skin**2)), t=cont.t_skin, kc=4.0, safety_factor=1.5)`). Print: K, #distinct sizes, the distinct sizes (mm), beam mass, %vs continuous, and worst stress/defl/twist/buckling utilization (to confirm feasibility, all ≤ limits / util ≤ ~1.0).
  6. Summary: the mass-vs-part-count Pareto (fewer distinct sizes → heavier, all feasible).

  (FOREGROUND run, ~2–4 min: one continuous sizing + a few fast MILP+verify solves.)

- [ ] **Step 2: Run** `uv run python examples/29_stock_catalog.py`. Paste full output. Report: continuous mass + radius range; for each K the #sizes, the stock sizes chosen, discrete mass (% heavier than continuous), and that the discrete design is FEA-feasible (stress under allow, defl/twist under limits, buckling util ≤ ~1.0 — confirming the monotonic round-up preserved feasibility). If any discrete design is infeasible (e.g. a load-redistribution edge case), report it honestly.

- [ ] **Step 3: Update `docs/plan.md`** — Phase-E status: E.3 done; record the catalog, the mass-vs-K Pareto numbers, and that round-up + CP-SAT yields manufacturable discrete stock sizing that stays FEA-feasible. Decisions-log row:

```markdown
| Phase-E.3 stock catalog | Beam radii discretized to a stock catalog via CP-SAT (`beams.select_stock_sizes`): min-mass assignment with a <=K-distinct-sizes cap (co-linear grouping). Monotonic round-up from the continuous optimum preserves feasibility (no SLP/sensitivities); verified by FEA re-solve. Full SLP+FEA-sensitivity MILP deferred. |
```

- [ ] **Step 4:** `uv run pytest` → green.

- [ ] **Step 5: Commit** — `feat(beams): E.3 stock-catalog example; mark done` + trailer.

---

## Self-Review

- **Spec coverage:** MILP discrete catalog → `select_stock_sizes` (CP-SAT, Task 1); co-linear grouping → the ≤K-distinct-sizes cardinality cap; FEA verification → the example re-solve (Task 2). Full SLP+sensitivities explicitly deferred (monotonicity replaces it).
- **Placeholder scan:** none.
- **Type consistency:** `select_stock_sizes(req_radii, catalog_radii, lengths, *, rho, max_distinct_sizes)` → `StockSelection(assigned_radii, distinct_sizes, mass_kg)`; `assigned_radii` feeds `BeamSection.circular` per element; the verify block reuses `solve_beam_shell` + `von_mises_per_element` + buckling helpers with the discrete radii and the continuous `t_skin`.
- **Correctness of the MILP:** exactly-one assignment per element over feasible sizes (catalog ≥ req); `used[j]` channeled from any `x[e,j]`; `Σ used ≤ K`; integer-scaled mass objective. K=1 is always feasible when the catalog covers `max(req)` (pick the smallest size ≥ max req). The tests pin the K=1 (single covering size) and K=2 ({0.015,0.020}) optima plus the catalog-too-small error.
- **Known limitations (intentional):** only beam radii are catalogued (skin thickness stays continuous); monotonic-round-up feasibility assumes load redistribution doesn't make a stiffened member infeasible (true for stress/buckling; the final FEA solve verifies and would surface the rare exception); the catalog is a fixed arithmetic series.
