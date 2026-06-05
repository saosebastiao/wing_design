"""FEA-in-the-loop cross-section sizing for the form-beam frame (Phase C).

Design variables: the radius of every longitudinal beam element (ring connectors
stay at a fixed minimum radius). Objective: minimize total beam mass. Constraints:
per-longitudinal-element von Mises <= allowable, and max-over-load-cases tip
deflection / tip twist <= their limits. Solved with SLSQP, re-solving the
Phase-B frame FEA inside the constraint evaluation (memoized per design point).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..structural.frame import BeamSection, solve_frame, von_mises_per_element
from .fea_model import BeamFrame


@dataclass(frozen=True)
class SizingConfig:
    sigma_allow_Pa: float
    tip_defl_max_m: float
    tip_twist_max_deg: float
    r_min: float = 0.004        # manufacturing floor
    r_max: float = 0.04         # geometric ceiling (inset room in the section)
    ring_radius: float = 0.008  # fixed ring-connector radius


@dataclass(frozen=True)
class SizingResult:
    radii: np.ndarray           # (n_longitudinal,) sized longitudinal radii [m]
    mass_kg: float
    converged: bool
    n_iter: int
    max_vm_stress_Pa: float     # worst longitudinal element, over all cases
    max_ring_vm_stress_Pa: float  # worst ring element (fixed radius; NOT a design var)
    tip_defl_m: float           # worst case
    tip_twist_deg: float        # worst case


def n_longitudinal(frame: BeamFrame) -> int:
    """Number of longitudinal elements (the first block of `frame.elements`)."""
    return frame.n_beams * (frame.n_levels - 1)


def element_lengths(frame: BeamFrame) -> np.ndarray:
    """(n_elem,) Euclidean length of every frame element."""
    i = frame.elements[:, 0]
    j = frame.elements[:, 1]
    return np.linalg.norm(frame.nodes[j] - frame.nodes[i], axis=1)


def build_sections(frame: BeamFrame, long_radii: np.ndarray, ring_radius: float) -> list[BeamSection]:
    """Per-element sections: longitudinal use `long_radii`, rings use `ring_radius`."""
    nl = n_longitudinal(frame)
    if len(long_radii) != nl:
        raise ValueError(f"expected {nl} longitudinal radii, got {len(long_radii)}")
    n_ring = frame.elements.shape[0] - nl
    return [BeamSection.circular(float(r)) for r in long_radii] + [
        BeamSection.circular(ring_radius)
    ] * n_ring


def frame_mass(frame: BeamFrame, long_radii: np.ndarray, *, ring_radius: float, rho: float) -> float:
    """Total beam mass [kg] = rho * sum(area * length) over longitudinal + ring elements."""
    nl = n_longitudinal(frame)
    L = element_lengths(frame)
    long_area = np.pi * np.asarray(long_radii) ** 2
    ring_area = np.pi * ring_radius**2
    return float(rho * (np.sum(long_area * L[:nl]) + ring_area * np.sum(L[nl:])))


def size_beams(
    frame: BeamFrame,
    load_arrays: list[np.ndarray],
    config: SizingConfig,
    *,
    rho: float,
    maxiter: int = 50,
    ftol: float = 1.0e-4,
    x0_radius: float | None = None,
) -> SizingResult:
    """Minimize beam mass s.t. stress, tip-deflection, and tip-twist constraints.

    `load_arrays` is one (n_nodes, 6) nodal-load array per load case (already
    safety-factored). The frame geometry is fixed; only longitudinal radii vary.

    Ring connectors keep a fixed radius and are NOT design variables, so their
    stress is not constrained — only reported (`SizingResult.max_ring_vm_stress_Pa`).
    Check it against the allowable: if rings are over-stressed, raise
    `config.ring_radius` (a converged result can still be ring-infeasible).
    """
    if not load_arrays:
        raise ValueError("load_arrays is empty: no loads to size against")
    nl = n_longitudinal(frame)
    L = element_lengths(frame)
    x0 = np.full(nl, config.r_max if x0_radius is None else x0_radius)

    cache: dict[bytes, tuple[np.ndarray, float, float, float]] = {}

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        key = np.asarray(x, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        sections = build_sections(frame, x, config.ring_radius)
        worst_vm = np.zeros(nl)
        worst_ring_vm = 0.0
        max_defl = 0.0
        max_twist = 0.0
        for loads in load_arrays:
            res = solve_frame(
                frame.nodes, frame.elements, sections,
                E=frame.E, G=frame.G, fixed_nodes=frame.fixed_nodes, loads=loads,
            )
            vm = von_mises_per_element(res, sections)
            worst_vm = np.maximum(worst_vm, vm[:nl])
            worst_ring_vm = max(worst_ring_vm, float(vm[nl:].max()))
            tip = frame.tip_nodes
            d = float(np.linalg.norm(res.displacements[tip, :3], axis=1).max())
            t = float(np.degrees(np.abs(res.displacements[tip, 5]).max()))
            max_defl = max(max_defl, d)
            max_twist = max(max_twist, t)
        out = (worst_vm, max_defl, max_twist, worst_ring_vm)
        cache[key] = out
        return out

    # Normalize the objective and all constraints to O(1). The raw scales differ
    # by ~1e9 (stress in Pa vs deflection in m vs twist in deg), which makes
    # SLSQP's QP subproblem ill-conditioned — the small-magnitude deflection
    # constraint becomes numerically invisible and the solver quits early at an
    # infeasible point. Dividing each constraint by its own limit (and the mass
    # by a reference mass) puts them all on the same footing.
    m_ref = frame_mass(frame, np.full(nl, config.r_max), ring_radius=config.ring_radius, rho=rho)

    def mass(x: np.ndarray) -> float:
        return frame_mass(frame, x, ring_radius=config.ring_radius, rho=rho) / m_ref

    def mass_grad(x: np.ndarray) -> np.ndarray:
        return rho * 2.0 * np.pi * np.asarray(x) * L[:nl] / m_ref

    def stress_con(x: np.ndarray) -> np.ndarray:
        worst_vm, _, _, _ = evaluate(x)
        return 1.0 - worst_vm / config.sigma_allow_Pa            # >= 0

    def defl_con(x: np.ndarray) -> np.ndarray:
        _, d, _, _ = evaluate(x)
        return np.array([1.0 - d / config.tip_defl_max_m])       # >= 0

    def twist_con(x: np.ndarray) -> np.ndarray:
        _, _, t, _ = evaluate(x)
        return np.array([1.0 - t / config.tip_twist_max_deg])    # >= 0

    res = minimize(
        mass, x0, jac=mass_grad, method="SLSQP",
        bounds=[(config.r_min, config.r_max)] * nl,
        constraints=[
            {"type": "ineq", "fun": stress_con},
            {"type": "ineq", "fun": defl_con},
            {"type": "ineq", "fun": twist_con},
        ],
        options={"maxiter": maxiter, "ftol": ftol},
    )

    # SLSQP can return points fractionally outside the bounds; clip before reporting.
    x_final = np.clip(np.asarray(res.x), config.r_min, config.r_max)
    worst_vm, d, t, ring_vm = evaluate(x_final)
    return SizingResult(
        radii=x_final,
        mass_kg=frame_mass(frame, x_final, ring_radius=config.ring_radius, rho=rho),
        converged=bool(res.success),
        n_iter=int(res.nit),
        max_vm_stress_Pa=float(worst_vm.max()),
        max_ring_vm_stress_Pa=float(ring_vm),
        tip_defl_m=float(d),
        tip_twist_deg=float(t),
    )
