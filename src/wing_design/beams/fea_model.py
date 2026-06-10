"""Assemble + solve a Phase-B beam-frame FEA from the Phase-A form-beam grid.

The form-beam splines become longitudinal beam members; transverse ring members
connect adjacent beams at every z-level, producing a cantilevered space-frame
clamped at the keel-step. Aero panel forces are projected onto the nearest beam
node (aero→geom frame) and the frame is solved per load case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..aero.loads import PanelLoads
from ..geometry.wing import WingSpec
from ..materials.unidir import T700_EPOXY, UDPly
from ..structural.frame import BeamSection, FrameResult, solve_frame
from ..structural.projection import R_GEOM_FROM_AERO
from .splines import default_z_levels, form_beam_grid


@dataclass(frozen=True)
class BeamFrame:
    nodes: np.ndarray          # (n_beams*n_levels, 3) geom frame
    elements: np.ndarray       # (n_elem, 2) int
    n_beams: int
    n_levels: int
    fixed_nodes: np.ndarray    # (n_beams,) keel-step ring (clamped)
    tip_nodes: np.ndarray      # (n_beams,) tip ring (where deflection/twist are read)
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
    nu: float = 0.32,  # isotropic-equivalent; matches MaterialParameters.nu_isotropic in scenario.py
) -> BeamFrame:
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2 to form longitudinal members, got {n_levels}")
    z = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z, n_beams)          # (n_beams, n_levels, 3)
    nodes = grid.reshape(-1, 3)

    def nid(b: int, k: int) -> int:
        return b * n_levels + k

    elems: list[tuple[int, int]] = []
    # longitudinal members: consecutive z-levels of each beam
    for b in range(n_beams):
        for k in range(n_levels - 1):
            elems.append((nid(b, k), nid(b, k + 1)))
    # ring members: adjacent beams at each level, wrapping around
    for k in range(n_levels):
        for b in range(n_beams):
            bn = (b + 1) % n_beams
            elems.append((nid(b, k), nid(bn, k)))
    elements = np.asarray(elems, dtype=int)

    keel_k = int(np.argmin(z))  # keel-step = bottom of the default_z_levels range
    tip_k = int(np.argmax(z))   # tip = top of the range
    fixed_nodes = np.array([nid(b, keel_k) for b in range(n_beams)], dtype=int)
    tip_nodes = np.array([nid(b, tip_k) for b in range(n_beams)], dtype=int)

    E = material.isotropic_equivalent_modulus(knockdown=knockdown)
    G = E / (2.0 * (1.0 + nu))
    return BeamFrame(
        nodes=nodes,
        elements=elements,
        n_beams=n_beams,
        n_levels=n_levels,
        fixed_nodes=fixed_nodes,
        tip_nodes=tip_nodes,
        section=BeamSection.circular(beam_radius),
        E=E,
        G=G,
    )


def project_panels_to_beam_nodes(
    frame: BeamFrame,
    panels: PanelLoads,
    *,
    safety_factor: float = 1.0,
) -> np.ndarray:
    """(n_nodes, 6) nodal loads: each panel force (aero→geom) lumped at the nearest node.

    Panel centers and forces are rotated from the aero frame (span +Y) to the
    geom frame (span +Z) with `R_GEOM_FROM_AERO`, then each panel's full force
    vector is assigned to the single nearest beam node. Moments are left zero.

    Nearest-node search spans all nodes; this is safe because aero panels live
    at geom z >= 0 while the clamped keel-step ring sits at z < 0, so a panel
    load never snaps onto a support (where it would be reacted out and lost).
    """
    loads = np.zeros((frame.nodes.shape[0], 6))
    centers_geom = panels.centers_xyz @ R_GEOM_FROM_AERO.T
    forces_geom = (panels.forces_xyz * safety_factor) @ R_GEOM_FROM_AERO.T
    for k in range(panels.n_panels):
        d = np.linalg.norm(frame.nodes - centers_geom[k], axis=1)
        loads[int(d.argmin()), :3] += forces_geom[k]
    return loads


def project_panels_to_skin(
    model,
    panels: PanelLoads,
    *,
    safety_factor: float = 1.0,
) -> np.ndarray:
    """(n_nodes, 6) nodal loads: each panel force spread over its nearest skin triangle.

    V.5 (backlog V#4): instead of snapping each aero panel's force to one node
    (`project_panels_to_beam_nodes`), assign it a third each to the 3 nodes of the
    skin triangle whose centroid is nearest — the consistent CST lumping of a
    uniform pressure over that triangle. Loads remain design-independent.
    """
    loads = np.zeros((model.nodes.shape[0], 6))
    centers_geom = panels.centers_xyz @ R_GEOM_FROM_AERO.T
    forces_geom = (panels.forces_xyz * safety_factor) @ R_GEOM_FROM_AERO.T
    centroids = model.nodes[model.shell_tris].mean(axis=1)
    for k in range(panels.n_panels):
        d = np.linalg.norm(centroids - centers_geom[k], axis=1)
        tri = model.shell_tris[int(d.argmin())]
        for n in tri:
            loads[int(n), :3] += forces_geom[k] / 3.0
    return loads


def panel_pressure_per_tri(
    model,
    panels: PanelLoads,
    *,
    safety_factor: float = 1.0,
) -> np.ndarray:
    """(n_tris,) lateral pressure magnitude [Pa] per skin triangle.

    Each aero panel's force magnitude is assigned to its nearest skin triangle
    (centroid distance) and divided by that triangle's area; triangles receiving
    no panel carry zero. Feeds the V.5 strip-bending term in the skin failure
    check (sigma_b = 0.75 q w^2 / t^2 — the sub-mesh physics this mesh cannot
    resolve with DKT moments).
    """
    centers_geom = panels.centers_xyz @ R_GEOM_FROM_AERO.T
    forces_geom = (panels.forces_xyz * safety_factor) @ R_GEOM_FROM_AERO.T
    tris = model.shell_tris
    centroids = model.nodes[tris].mean(axis=1)
    v1 = model.nodes[tris[:, 1]] - model.nodes[tris[:, 0]]
    v2 = model.nodes[tris[:, 2]] - model.nodes[tris[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)
    force_sum = np.zeros(tris.shape[0])
    for k in range(panels.n_panels):
        d = np.linalg.norm(centroids - centers_geom[k], axis=1)
        force_sum[int(d.argmin())] += float(np.linalg.norm(forces_geom[k]))
    return force_sum / np.maximum(areas, 1e-30)


def solve_beam_frame(frame: BeamFrame, loads: np.ndarray) -> FrameResult:
    sections = [frame.section] * frame.elements.shape[0]
    return solve_frame(
        frame.nodes,
        frame.elements,
        sections,
        E=frame.E,
        G=frame.G,
        fixed_nodes=frame.fixed_nodes,
        loads=loads,
    )


@dataclass(frozen=True)
class FrameMetrics:
    case_name: str
    applied_force_N: float
    tip_translation_m: float
    tip_twist_deg: float
    max_axial_stress_Pa: float
    max_bending_stress_Pa: float
    max_vm_stress_Pa: float


def summarize_frame(
    frame: BeamFrame,
    result: FrameResult,
    case_name: str,
    applied_force_N: float,
) -> FrameMetrics:
    """Reduce a FrameResult to the headline tip/stress metrics for a load case."""
    sec = frame.section
    if sec.Iy != sec.Iz:
        raise ValueError(
            "summarize_frame's bending-stress proxy assumes a circular section (Iy == Iz)"
        )
    sigma_axial = np.abs(result.axial_force) / sec.A
    sigma_bend = result.bending_moment * sec.r / sec.Iz
    tau = np.abs(result.torsion) * sec.r / sec.J
    vm = np.sqrt((sigma_axial + sigma_bend) ** 2 + 3.0 * tau**2)

    tip = frame.tip_nodes
    tip_tr = float(np.linalg.norm(result.displacements[tip, :3], axis=1).max())
    tip_twist = float(np.abs(result.displacements[tip, 5]).max())  # rz about span axis

    return FrameMetrics(
        case_name=case_name,
        applied_force_N=applied_force_N,
        tip_translation_m=tip_tr,
        tip_twist_deg=float(np.degrees(tip_twist)),
        max_axial_stress_Pa=float(sigma_axial.max()),
        max_bending_stress_Pa=float(sigma_bend.max()),
        max_vm_stress_Pa=float(vm.max()),
    )
