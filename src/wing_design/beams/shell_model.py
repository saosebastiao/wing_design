"""Beam-shell model of the form-beam structure: longitudinal beams + load-bearing skin.

Builds the Phase-A/B beam grid, keeps the longitudinal beam members, and replaces the
ring connectors with a triangulated skin (the load-bearing shell). Solves with the
combined `structural.solve_beam_shell`. The skin is isotropic-equivalent here; CLT
anisotropy and skin-stress-driven re-sizing are later Phase-E increments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.wing import WingSpec
from ..materials.unidir import T700_EPOXY, UDPly
from ..structural.beam_shell import solve_beam_shell
from ..structural.frame import BeamSection, FrameResult
from ..structural.shell import _triangle_local_frame
from .splines import default_z_levels, form_beam_grid


def skin_triangles(n_beams: int, n_levels: int) -> np.ndarray:
    """(2*n_beams*(n_levels-1), 3) skin triangles tiling the beam grid.

    Each quad between beams b,(b+1)%n_beams and levels k,k+1 is split into two
    triangles. Node id = b*n_levels + k (beam-major/level-minor).
    """
    def nid(b: int, k: int) -> int:
        return b * n_levels + k

    tris: list[tuple[int, int, int]] = []
    for k in range(n_levels - 1):
        for b in range(n_beams):
            bn = (b + 1) % n_beams
            tris.append((nid(b, k), nid(bn, k), nid(bn, k + 1)))
            tris.append((nid(b, k), nid(bn, k + 1), nid(b, k + 1)))
    return np.asarray(tris, dtype=int)


@dataclass(frozen=True)
class BeamShellModel:
    nodes: np.ndarray
    beam_elements: np.ndarray
    shell_tris: np.ndarray
    n_beams: int
    n_levels: int
    fixed_nodes: np.ndarray
    tip_nodes: np.ndarray
    section: BeamSection
    E_beam: float
    G_beam: float
    E_skin: float
    nu_skin: float
    t_skin: float


def build_beam_shell_model(
    spec: WingSpec,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    beam_radius: float = 0.02,
    material: UDPly = T700_EPOXY,
    knockdown: float = 0.5,
    nu: float = 0.32,
    skin_thickness: float = 0.003,
) -> BeamShellModel:
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2, got {n_levels}")
    z = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z, n_beams)
    nodes = grid.reshape(-1, 3)

    def nid(b: int, k: int) -> int:
        return b * n_levels + k

    beam_elems = [(nid(b, k), nid(b, k + 1)) for b in range(n_beams) for k in range(n_levels - 1)]
    beam_elements = np.asarray(beam_elems, dtype=int)

    keel_k = int(np.argmin(z))
    tip_k = int(np.argmax(z))
    fixed_nodes = np.array([nid(b, keel_k) for b in range(n_beams)], dtype=int)
    tip_nodes = np.array([nid(b, tip_k) for b in range(n_beams)], dtype=int)

    E = material.isotropic_equivalent_modulus(knockdown=knockdown)
    G = E / (2.0 * (1.0 + nu))
    return BeamShellModel(
        nodes=nodes,
        beam_elements=beam_elements,
        shell_tris=skin_triangles(n_beams, n_levels),
        n_beams=n_beams,
        n_levels=n_levels,
        fixed_nodes=fixed_nodes,
        tip_nodes=tip_nodes,
        section=BeamSection.circular(beam_radius),
        E_beam=E,
        G_beam=G,
        E_skin=E,
        nu_skin=nu,
        t_skin=skin_thickness,
    )


def skin_datum_angles(model: BeamShellModel, datum_dir=(0.0, 0.0, 1.0)) -> np.ndarray:
    """(n_tris,) angle [rad] from each skin triangle's local x-axis to ``datum_dir``,
    measured in the triangle's plane.

    `datum_dir` is a global direction (default span = +Z). For each triangle it is
    projected onto the triangle plane and expressed in local (e1,e2); the returned
    angle delta = atan2(d.e2, d.e1). A laminate defined against this datum is built
    per-triangle via `materials.laminate_stiffness_offset(..., offset_deg=degrees(delta))`.
    If the datum is normal to a triangle (no in-plane component) delta defaults to 0.
    """
    d = np.asarray(datum_dir, dtype=float)
    d = d / np.linalg.norm(d)
    out = np.zeros(model.shell_tris.shape[0])
    for e in range(model.shell_tris.shape[0]):
        a, b, c = (int(v) for v in model.shell_tris[e])
        R, _, _ = _triangle_local_frame(model.nodes[a], model.nodes[b], model.nodes[c])
        e1, e2 = R[:, 0], R[:, 1]
        dx, dy = float(d @ e1), float(d @ e2)
        out[e] = 0.0 if (dx == 0.0 and dy == 0.0) else float(np.arctan2(dy, dx))
    return out


def solve_beam_shell_model(model: BeamShellModel, loads: np.ndarray) -> FrameResult:
    """Solve the beam-shell model under nodal ``loads`` (n_nodes, 6) → FrameResult.

    Beam internal forces in the result are the longitudinal members' forces; the
    skin's contribution shows up as the (stiffer) displacement field.
    """
    sections = [model.section] * model.beam_elements.shape[0]
    return solve_beam_shell(
        model.nodes, model.beam_elements, sections, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam,
        E_skin=model.E_skin, nu_skin=model.nu_skin, t_skin=model.t_skin,
        fixed_nodes=model.fixed_nodes, loads=loads,
    )
