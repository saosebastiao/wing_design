"""Beam-shell model of the form-beam structure: longitudinal beams + load-bearing skin.

Builds the Phase-A/B beam grid, keeps the longitudinal beam members, and replaces the
ring connectors with a triangulated skin (the load-bearing shell). Solves with the
combined `structural.solve_beam_shell`. The skin is isotropic-equivalent here; CLT
anisotropy and skin-stress-driven re-sizing are later Phase-E increments.
"""
from __future__ import annotations

import dataclasses
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
    tip_gusset_elements: np.ndarray | None = None
    tip_gusset_radius: float | None = None
    # P.1 core tube: rows of `beam_elements` that are tube segments (annular
    # sections sized by the r_tube/t_wall DV blocks), the per-segment outer-radius
    # fit bounds (inscribed-circle bound at the segment's wider end), and the
    # stiff massless tube->beam bond pairs (assembled via the gusset path:
    # excluded from force recovery and mass).
    tube_elements: np.ndarray | None = None        # (S,) indices into beam_elements
    tube_r_bounds: np.ndarray | None = None        # (S,)
    tube_bond_elements: np.ndarray | None = None   # (M, 2) node pairs
    # P#1b hollow form beams: form-element rows eligible for annular (wound-
    # hollow) sections — both endpoint levels in-wing AND the beam path measured
    # straight (the windability requirement; asserted at build).
    hollow_elements: np.ndarray | None = None      # (Hn,) indices into beam_elements
    # task #22 lateral bracing: rows of `beam_elements` that are transverse ring
    # members (sized by the brace_radius DV), and the spanwise stations carrying
    # a ring (level indices into the z-grid).
    brace_elements: np.ndarray | None = None      # (n_rings*n_beams,) indices into beam_elements
    brace_stations: np.ndarray | None = None      # (n_rings,) level indices

    @property
    def n_form_elements(self) -> int:
        """Count of form-beam elements (tube rows, if any, come after)."""
        return self.n_beams * (self.n_levels - 1)


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
    arc_fractions: np.ndarray | None = None,
    tip_gusset_radius: float | None = None,
    core_tube: bool = False,
    tube_bonds_per_level: int = 4,
    hollow_beams: bool = False,
    straightness_tol_m: float = 1.0e-3,
    lateral_bracing: bool = False,
    brace_stations: str | tuple[int, ...] = "interior",
) -> BeamShellModel:
    """Build a beam-shell model; ``arc_fractions`` (default None = even) gives non-uniform beam spacing.

    ``core_tube=True`` (P.1) appends one tube node per z-level on the pivot axis
    (X=0, Y=0), tube beam elements between consecutive levels (straight by
    construction — the windability requirement), stiff massless bond pairs to the
    ``tube_bonds_per_level`` nearest form-beam nodes per level, and per-segment
    outer-radius fit bounds (0.95 x the min node distance to the axis). The tube
    keel node joins ``fixed_nodes`` (the root-socket bond).
    """
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2, got {n_levels}")
    z = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z, n_beams, arc_fractions=arc_fractions)
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
    from .tip_coupling import tip_clique_elements  # local import to avoid circular dependency
    tip_gusset_elements = tip_clique_elements(tip_nodes) if tip_gusset_radius is not None else None

    hollow_elements = None
    if hollow_beams:
        # in-wing levels: z >= 0 (the below-root spar transition stays solid RTM)
        in_wing = z >= -1e-9
        rows = []
        for b in range(n_beams):
            ks = sorted((k for k in range(n_levels) if in_wing[k]), key=lambda k: z[k])
            if len(ks) >= 2:
                pts = np.array([nodes[nid(b, k)] for k in ks])
                d = pts[-1] - pts[0]
                u = d / np.linalg.norm(d)
                rel = pts - pts[0]
                dev = float(np.linalg.norm(rel - np.outer(rel @ u, u), axis=1).max())
                if dev > straightness_tol_m:
                    raise ValueError(
                        f"beam {b} in-wing path deviates {dev*1e3:.1f} mm from straight "
                        f"(> {straightness_tol_m*1e3:.1f} mm) — not windable hollow")
            for k in range(n_levels - 1):
                if in_wing[k] and in_wing[k + 1]:
                    rows.append(b * (n_levels - 1) + k)
        hollow_elements = np.asarray(sorted(rows), dtype=int)

    tube_elements = tube_r_bounds = tube_bond_elements = None
    if core_tube:
        n_grid = nodes.shape[0]
        # level order follows z index k (default_z_levels runs tip->keel)
        tube_nodes = np.array([[0.0, 0.0, float(z[k])] for k in range(n_levels)])
        nodes = np.vstack([nodes, tube_nodes])
        tube_node = lambda k: n_grid + k
        n_form = beam_elements.shape[0]
        tube_rows = [(tube_node(k), tube_node(k + 1)) for k in range(n_levels - 1)]
        beam_elements = np.vstack([beam_elements, np.asarray(tube_rows, dtype=int)])
        tube_elements = np.arange(n_form, n_form + n_levels - 1, dtype=int)
        # fit bound per segment: min over the segment's two levels of the
        # inscribed-circle bound sampled from this level's form-beam nodes.
        level_bound = np.empty(n_levels)
        for k in range(n_levels):
            ring = np.array([nodes[b * n_levels + k] for b in range(n_beams)])
            level_bound[k] = 0.95 * float(np.linalg.norm(ring[:, :2], axis=1).min())
        tube_r_bounds = np.minimum(level_bound[:-1], level_bound[1:])
        bonds = []
        for k in range(n_levels):
            ring = np.array([nodes[b * n_levels + k] for b in range(n_beams)])
            d = np.linalg.norm(ring[:, :2], axis=1)
            for b in np.argsort(d)[:tube_bonds_per_level]:
                bonds.append((tube_node(k), int(b) * n_levels + k))
        tube_bond_elements = np.asarray(bonds, dtype=int)
        fixed_nodes = np.append(fixed_nodes, tube_node(keel_k))

    brace_elements = brace_stations_arr = None
    if lateral_bracing:
        if brace_stations == "interior":
            stations = list(range(1, n_levels - 1))
        elif brace_stations == "all":
            # "all" includes k=0 (root) and k=n_levels-1 (tip), whose nodes
            # coincide with fixed_nodes/tip_nodes — harmless but redundant
            # constraints; "interior" is the intended default for that reason.
            stations = list(range(n_levels))
        else:
            stations = sorted(int(k) for k in brace_stations)
        if not stations:
            raise ValueError("lateral_bracing=True but no brace stations selected")
        n0 = beam_elements.shape[0]
        ring_rows = [(nid(b, k), nid((b + 1) % n_beams, k))
                     for k in stations for b in range(n_beams)]
        beam_elements = np.vstack([beam_elements, np.asarray(ring_rows, dtype=int)])
        brace_elements = np.arange(n0, n0 + len(ring_rows), dtype=int)
        brace_stations_arr = np.asarray(stations, dtype=int)

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
        tip_gusset_elements=tip_gusset_elements,
        tip_gusset_radius=tip_gusset_radius,
        tube_elements=tube_elements,
        tube_r_bounds=tube_r_bounds,
        tube_bond_elements=tube_bond_elements,
        hollow_elements=hollow_elements,
        brace_elements=brace_elements,
        brace_stations=brace_stations_arr,
    )


def model_with_tip_gusset(model: BeamShellModel, radius: float) -> BeamShellModel:
    """Return a copy of model with a rigid tip-gusset clique (radius = connector-beam radius)."""
    from .tip_coupling import tip_clique_elements  # local import to avoid circular dependency
    return dataclasses.replace(model, tip_gusset_radius=float(radius),
                               tip_gusset_elements=tip_clique_elements(model.tip_nodes))


def skin_panel_widths(model: BeamShellModel) -> np.ndarray:
    """(n_tris,) physical panel-strip width per skin triangle (V.3b, backlog V#1).

    Each triangle lies in the quad strip between adjacent beams b and b+1 (the
    `skin_triangles` tiling); the physical compression panel is the *long strip*
    between those beam lines, whose plate-buckling stress is set by the strip
    WIDTH — the chordwise distance between the two bounding beams at that station
    (mean of the quad's two level widths) — not by `sqrt(triangle area)`, which
    mis-scales with beam count (∝n vs ∝n² physically) and over-sizes b ~2× on the
    medium mesh. Beam membership is recovered from the node ids (b*n_levels + k).
    """
    nl = model.n_levels
    tris = model.shell_tris
    out = np.empty(tris.shape[0])
    for e in range(tris.shape[0]):
        beams = tris[e] // nl
        levels = tris[e] % nl
        uniq = np.unique(beams)
        if uniq.size != 2:   # degenerate (shouldn't happen on this tiling)
            n0, n1, n2 = (int(v) for v in tris[e])
            _R, _loc, area = _triangle_local_frame(model.nodes[n0], model.nodes[n1],
                                                   model.nodes[n2])
            out[e] = np.sqrt(area)
            continue
        b0, b1 = int(uniq[0]), int(uniq[1])
        ks = np.unique(levels)
        w = 0.0
        for k in ks:
            w += float(np.linalg.norm(model.nodes[b0 * nl + int(k)]
                                      - model.nodes[b1 * nl + int(k)]))
        out[e] = w / ks.size
    return out


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


def beam_adjacent_widths(model: BeamShellModel) -> "np.ndarray":
    """(n_form_elements, 2) chordwise widths of the two skin strips adjacent to
    each form-beam element (to beams b-1 and b+1 at the element's segment,
    mean of the two level widths) — the V.3c foundation-stiffness geometry."""
    nl = model.n_levels
    nb = model.n_beams
    out = np.empty((model.n_form_elements, 2))

    def width(b0, b1, k):
        w = 0.0
        for kk in (k, k + 1):
            w += float(np.linalg.norm(model.nodes[b0 * nl + kk]
                                      - model.nodes[b1 * nl + kk]))
        return 0.5 * w

    for b in range(nb):
        for k in range(nl - 1):
            e = b * (nl - 1) + k
            out[e, 0] = width(b, (b - 1) % nb, k)
            out[e, 1] = width(b, (b + 1) % nb, k)
    return out


def braced_segment_mask(model: BeamShellModel) -> np.ndarray:
    """(n_form_elements,) bool: True where a form-beam SEGMENT has a ring at BOTH
    of its end stations (so the ring shortens that segment's lateral buckling
    length). Returns all-False when no bracing is present."""
    nb, nl = model.n_beams, model.n_levels
    mask = np.zeros(model.n_form_elements, dtype=bool)
    if model.brace_stations is None:
        return mask
    braced = set(int(k) for k in model.brace_stations)
    for b in range(nb):
        for k in range(nl - 1):
            if k in braced and (k + 1) in braced:
                mask[b * (nl - 1) + k] = True
    return mask
