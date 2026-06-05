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
