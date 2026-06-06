"""Co-size form-beam radii and skin thickness against the combined beam+shell FEA.

Design variables: per-element beam radii + a single uniform skin thickness.
Objective: total (beam + skin) mass. Constraints: per-beam-element von Mises,
per-skin-triangle von Mises, tip deflection, and tip twist — re-solving
`structural.solve_beam_shell` each iteration. SLSQP with O(1)-normalized
objective/constraints (as in Phase C).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..structural.beam_shell import solve_beam_shell
from ..structural.frame import BeamSection, von_mises_per_element
from ..structural.shell import _triangle_local_frame, membrane_von_mises, recover_membrane_stress
from .shell_model import BeamShellModel


@dataclass(frozen=True)
class BeamShellSizingConfig:
    sigma_allow_Pa: float
    tip_defl_max_m: float
    tip_twist_max_deg: float
    r_min: float = 0.004
    r_max: float = 0.04
    t_min: float = 0.0005
    t_max: float = 0.02


@dataclass(frozen=True)
class BeamShellSizingResult:
    radii: np.ndarray
    t_skin: float
    mass_kg: float
    beam_mass_kg: float
    skin_mass_kg: float
    converged: bool
    n_iter: int
    max_beam_vm_Pa: float
    max_skin_vm_Pa: float
    tip_defl_m: float
    tip_twist_deg: float


def beam_lengths(model: BeamShellModel) -> np.ndarray:
    """(n_beam_elem,) Euclidean length of each longitudinal beam element."""
    i = model.beam_elements[:, 0]
    j = model.beam_elements[:, 1]
    return np.linalg.norm(model.nodes[j] - model.nodes[i], axis=1)


def skin_areas(model: BeamShellModel) -> np.ndarray:
    """(n_tris,) area of each skin triangle."""
    out = np.empty(model.shell_tris.shape[0])
    for e in range(model.shell_tris.shape[0]):
        a, b, c = (int(v) for v in model.shell_tris[e])
        _, _, area = _triangle_local_frame(model.nodes[a], model.nodes[b], model.nodes[c])
        out[e] = area
    return out


def beam_mass(model: BeamShellModel, radii: np.ndarray, *, rho: float) -> float:
    """Total longitudinal-beam mass [kg]."""
    return float(rho * np.sum(np.pi * np.asarray(radii) ** 2 * beam_lengths(model)))


def skin_mass(model: BeamShellModel, t_skin: float, *, rho: float) -> float:
    """Total skin mass [kg] = rho * t * sum(triangle areas)."""
    return float(rho * t_skin * skin_areas(model).sum())
