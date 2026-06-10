"""Named constraint specs for the sizing NLP (P.0, Toolbox #1).

Each constraint declares its name, row count, scipy closures, and how its KKT
multipliers convert to a physical shadow price — so adding a constraint is ONE
append, and the multiplier attribution is structural instead of positional
(`laminate_sizing._shadow_prices` previously rebuilt the row layout by hand and
returned None on any mismatch).

Shadow kinds (matching the V.1 conversions, FD-validated 2026-06-10):
- ("limit", key, L):  g = 1 − v/L      → dm*/dL  = −m_ref·Σλ̃/L   (≤ 0)
- ("sf", key, SF):    g = 1 − SF·v/cap or R/SF − 1
                                        → dm*/dSF = +m_ref·Σλ̃/SF  (≥ 0)
- None: no config parameter (fraction simplex, monotonic taper).
Multiple specs may share a key (beam vM + skin vM share sigma_allow_Pa); their
contributions sum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    fun: Callable                      # x -> (n_rows,) or scalar, feasible >= 0
    n_rows: int
    jac: Callable | None = None        # x -> (n_rows, nx); None = FD
    shadow: tuple[str, str, float] | None = None   # (kind, key, value)

    def scipy_dict(self) -> dict:
        d = {"type": "ineq", "fun": self.fun}
        if self.jac is not None:
            d["jac"] = self.jac
        return d


def shadow_prices_from_specs(
    multipliers,
    specs: list[ConstraintSpec],
    *,
    m_ref: float,
) -> dict[str, float] | None:
    """Physical dm*/dparam from SLSQP's normalized KKT multipliers.

    ``multipliers`` is scipy's [eq | ineq] vector, ineq rows in constraint-list
    order (bounds excluded). Returns None on total-row mismatch rather than
    mis-attributing.
    """
    if multipliers is None:
        return None
    lam = np.asarray(multipliers, dtype=float).ravel()
    total = sum(s.n_rows for s in specs)
    if lam.size != total:
        return None
    out: dict[str, float] = {}
    pos = 0
    for s in specs:
        rows = lam[pos:pos + s.n_rows]
        pos += s.n_rows
        if s.shadow is None:
            continue
        kind, key, value = s.shadow
        lam_sum = float(rows.sum())
        if kind == "limit":
            out[key] = out.get(key, 0.0) - m_ref * lam_sum / value
        elif kind == "sf":
            out[key] = out.get(key, 0.0) + m_ref * lam_sum / value
        else:
            raise ValueError(f"unknown shadow kind {kind!r} on {s.name!r}")
    return out
