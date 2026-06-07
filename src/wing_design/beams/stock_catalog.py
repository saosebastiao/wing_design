"""Discretize continuous beam radii onto a manufacturable stock catalog (CP-SAT).

`select_stock_sizes` picks <= K distinct catalog sizes and assigns one (>= the
continuous radius) to each element, minimizing total beam mass (the co-linear-grouping
/ part-count trade). Rounding each element's radius UP improves it *in isolation* for
stress, deflection and twist (axial/bending stress and the deflection/twist response
all fall with radius). IMPORTANT: this per-element monotonicity does NOT hold for beam
buckling once load redistribution is considered — non-uniform stiffening shifts
compression onto beams pinned at the minimum radius, raising their Euler utilization.
So the chosen discrete design MUST be FEA-verified, and buckling violations repaired by
bumping the over-utilized beams up the catalog and re-verifying (see examples/29).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model


@dataclass(frozen=True)
class StockSelection:
    assigned_radii: np.ndarray
    distinct_sizes: list[float]
    mass_kg: float


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
    distinct catalog radii are used; total mass `rho * sum(pi r^2 L)` is minimized.
    Raises ValueError if the catalog's largest radius cannot cover the largest required.
    `mass_scale` converts the float mass objective to CP-SAT integers (raise it if
    element lengths/radii are so small the scaled costs lose resolution).
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
    # Report the sizes actually assigned (robust regardless of the one-directional
    # used[] channeling, which only enforces the <=K cardinality constraint).
    distinct = sorted({float(r) for r in assigned})
    mass = float(rho * np.sum(np.pi * assigned**2 * L))
    return StockSelection(assigned_radii=assigned, distinct_sizes=distinct, mass_kg=mass)
