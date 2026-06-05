"""Assemble + solve a Phase-B beam-frame FEA from the Phase-A form-beam grid.

The form-beam splines become longitudinal beam members; transverse ring members
connect adjacent beams at every z-level, producing a cantilevered space-frame
clamped at the keel-step. Aero panel forces are projected onto the nearest beam
node (aero→geom frame) and the frame is solved per load case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.wing import WingSpec
from ..materials.unidir import T700_EPOXY, UDPly
from ..structural.frame import BeamSection
from .splines import default_z_levels, form_beam_grid


@dataclass(frozen=True)
class BeamFrame:
    nodes: np.ndarray          # (n_beams*n_levels, 3) geom frame
    elements: np.ndarray       # (n_elem, 2) int
    n_beams: int
    n_levels: int
    fixed_nodes: np.ndarray    # (n_beams,) keel-step ring
    section: BeamSection
    E: float
    G: float


def build_beam_frame(
    spec: WingSpec,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    beam_radius: float = 0.02,
    material: UDPly = T700_EPOXY,
    knockdown: float = 0.5,
    nu: float = 0.32,
) -> BeamFrame:
    z = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z, n_beams)          # (n_beams, n_levels, 3)
    nodes = grid.reshape(-1, 3)

    elems: list[tuple[int, int]] = []
    # longitudinal members: consecutive z-levels of each beam
    for b in range(n_beams):
        for k in range(n_levels - 1):
            elems.append((b * n_levels + k, b * n_levels + (k + 1)))
    # ring members: adjacent beams at each level, wrapping around
    for k in range(n_levels):
        for b in range(n_beams):
            bn = (b + 1) % n_beams
            elems.append((b * n_levels + k, bn * n_levels + k))
    elements = np.asarray(elems, dtype=int)

    keel_k = n_levels - 1  # default_z_levels descends tip -> keel-step
    fixed_nodes = np.array([b * n_levels + keel_k for b in range(n_beams)], dtype=int)

    E = material.isotropic_equivalent_modulus(knockdown=knockdown)
    G = E / (2.0 * (1.0 + nu))
    return BeamFrame(
        nodes=nodes,
        elements=elements,
        n_beams=n_beams,
        n_levels=n_levels,
        fixed_nodes=fixed_nodes,
        section=BeamSection.circular(beam_radius),
        E=E,
        G=G,
    )
