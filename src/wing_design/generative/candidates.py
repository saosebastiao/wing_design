"""Candidate-menu generator for the constraint-based truss stack.

`build_beam_library` builds a small, deliberately simple beam library whose every
beam satisfies the generator contracts by construction (see `validate_menu`):
monotonic-z; every keel-rooted run goes keel->deck vertically first; non-spar
beams are hosted at the deck node; landmark nodes sit exactly on beam control
points; mirror pairs are reciprocal. `build_candidate_menu` adds the real
background-FEA-driven coverage targets and the conflict table.

Curved stress-line-conformant beams are a richer drop-in library for a later
milestone; this milestone proves the end-to-end pipeline. See the design spec.
"""
from __future__ import annotations

import math

from ..scenario import DesignParameters
from .menu import (
    CandidateBeam,
    CandidateMenu,
    CandidateNode,
    CrossSectionOption,
    CrossSectionShape,
    NodeKind,
)


def _cross_section_catalog(params) -> tuple[CrossSectionOption, ...]:
    """Evenly-spaced circular area buckets up to the manufacturability max."""
    n = params.generative.n_area_buckets
    a_max = params.generative.cross_section_area_max_m2
    return tuple(
        CrossSectionOption(
            bucket=i,
            shape=CrossSectionShape.CIRCLE,
            area_m2=a_max * (i + 1) / n,
        )
        for i in range(n)
    )


def build_beam_library(params: DesignParameters):
    """Return (nodes, beams, cross_sections): the contract-correct simple library.

    Geometry: the pivot axis is at x=0, y=0. The central spar runs
    keel-step -> deck-step -> tip on the axis. Chordwise branches start ON the
    spar at the deck node, bow out to a chord offset at mid-span, and return to
    the tip. One out-of-plane (+/-y) mirror pair exercises the symmetry tie.
    """
    spec = params.geometry
    z_keel = spec.z_keel_step
    z_deck = spec.z_deck_step
    z_tip = spec.z_wing_tip
    z_mid = 0.5 * (z_deck + z_tip)

    keel = (0.0, 0.0, z_keel)
    deck = (0.0, 0.0, z_deck)
    tip = (0.0, 0.0, z_tip)

    nodes = [
        CandidateNode(id=0, xyz=keel, kind=NodeKind.KEEL_STEP, z_layer=0),
        CandidateNode(id=1, xyz=deck, kind=NodeKind.DECK_STEP, z_layer=1),
        CandidateNode(id=2, xyz=tip, kind=NodeKind.TIP, z_layer=2),
    ]

    beams: list[CandidateBeam] = []

    # Central spar: keel -> deck -> tip on the axis.
    spar = CandidateBeam(
        id=0,
        control_points=(keel, deck, tip),
        start_kind=NodeKind.KEEL_STEP,
        end_kind=NodeKind.TIP,
        start_node=0,
        end_node=2,
        length_m=(z_tip - z_keel),
        min_radius_m=math.inf,
        on_chord_plane=True,
        mirror_id=None,
        host_id=None,
        covers=(),
    )
    beams.append(spar)

    # Chordwise in-plane branches: deck -> (x_off, 0, z_mid) -> tip. The chord
    # half-extent at z_mid bounds the bow so beams stay inside the OML.
    chord_mid = spec.chord_at_z(z_mid)
    x_offsets = [0.25 * chord_mid, -0.25 * chord_mid]
    next_id = 1
    for x_off in x_offsets:
        beams.append(
            CandidateBeam(
                id=next_id,
                control_points=(deck, (x_off, 0.0, z_mid), tip),
                start_kind=NodeKind.ON_BEAM,
                end_kind=NodeKind.TIP,
                start_node=1,
                end_node=2,
                length_m=2.0 * math.hypot(x_off, 0.5 * (z_tip - z_deck)),
                min_radius_m=1.0,
                on_chord_plane=True,
                mirror_id=None,
                host_id=spar.id,
                covers=(),
            )
        )
        next_id += 1

    # One out-of-plane mirror pair: deck -> (0, +/-y_off, z_mid) -> tip.
    thickness_half = 0.5 * spec.thickness * chord_mid  # airfoil half-thickness at z_mid
    y_off = 0.4 * thickness_half
    a_id, b_id = next_id, next_id + 1
    beams.append(
        CandidateBeam(
            id=a_id, control_points=(deck, (0.0, y_off, z_mid), tip),
            start_kind=NodeKind.ON_BEAM, end_kind=NodeKind.TIP,
            start_node=1, end_node=2,
            length_m=2.0 * math.hypot(y_off, 0.5 * (z_tip - z_deck)),
            min_radius_m=1.0, on_chord_plane=False, mirror_id=b_id, host_id=spar.id,
            covers=(),
        )
    )
    beams.append(
        CandidateBeam(
            id=b_id, control_points=(deck, (0.0, -y_off, z_mid), tip),
            start_kind=NodeKind.ON_BEAM, end_kind=NodeKind.TIP,
            start_node=1, end_node=2,
            length_m=2.0 * math.hypot(y_off, 0.5 * (z_tip - z_deck)),
            min_radius_m=1.0, on_chord_plane=False, mirror_id=a_id, host_id=spar.id,
            covers=(),
        )
    )

    return nodes, beams, _cross_section_catalog(params)


import numpy as np

from ..aero.cases import DESIGN_CASES
from ..aero.loads import run_case_lifting_line
from ..aero.model import build_airplane
from ..structural.mesh import tet_mesh_wing
from ..structural.shell import shell_mesh_from_tet_mesh, solve_shell_elastic
from .menu import ConflictTable, CoverageTarget, validate_menu


def _background_stress(params):
    """Run the shell FEA for the first design case; return (centroids, sigma_vm).

    centroids: (M,3) triangle centroids in the geometry frame; sigma_vm: (M,)
    membrane von Mises per triangle.
    """
    spec = params.geometry
    tet = tet_mesh_wing(spec, target_element_size=params.mesh.target_element_size_m)
    shell = shell_mesh_from_tet_mesh(tet)
    airplane = build_airplane(spec)
    case = DESIGN_CASES[0]
    aero = run_case_lifting_line(airplane, case,
                                 spanwise_resolution=params.aero.spanwise_resolution)
    # Shell loads: project panel forces onto the shell's loaded triangles.
    tri_forces = _shell_tri_forces(shell, aero, spec.span, case.safety_factor)
    res = solve_shell_elastic(
        shell, E=params.E_iso_Pa, nu=params.nu_iso,
        thickness_m=params.skin_sizing.t_baseline_m, tri_force_vectors=tri_forces,
    )
    tri = shell.triangles
    centroids = shell.nodes[tri].mean(axis=1)
    return centroids, res.membrane_von_mises()


def _shell_tri_forces(shell, aero, span_m, safety_factor):
    """Distribute the case's spanwise normal force onto loaded shell triangles."""
    tri = shell.triangles
    centroids = shell.nodes[tri].mean(axis=1)
    areas = _tri_areas(shell.nodes, tri)
    loaded = shell.loaded_tris
    forces = np.zeros((len(tri), 3))
    z = centroids[:, 2]
    # normal force density (N/m) sampled along span at each loaded centroid
    dens = aero.distributed_normal_force(np.clip(z, 0.0, span_m))
    w = areas * loaded
    contrib = dens * w
    total = contrib.sum()
    target = aero.factored_normal_force_N
    scale = (target / total) if total > 0 else 0.0
    forces[:, 1] = contrib * scale  # apply in +Y (airfoil normal)
    return forces


def _tri_areas(nodes, tri):
    p = nodes[tri]
    return 0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
    )


def _segment_point_distance(seg_a, seg_b, pts):
    """Min distance between two polylines, sampled at their points (coarse)."""
    a = np.asarray(seg_a, dtype=float)
    b = np.asarray(seg_b, dtype=float)
    dmin = math.inf
    for pa in a:
        d = np.min(np.linalg.norm(b - pa, axis=1))
        dmin = min(dmin, float(d))
    return dmin


def _coverage_targets(beams, centroids, sigma_vm, params):
    """Stress-informed coverage targets: cluster the highest-sigma triangles,
    require a section sized to the local stress, and attach every beam passing
    within a tolerance of the target centroid.
    """
    sf = params.generative.coverage_safety_factor
    sigma_allow = params.sigma_allow_Pa
    a_max = params.generative.cross_section_area_max_m2
    order = np.argsort(sigma_vm)[::-1]
    n_targets = min(5, len(order))
    tol = 0.5  # m: a beam covers a target if any control point is within tol
    targets = []
    for k in range(n_targets):
        idx = int(order[k])
        c = centroids[idx]
        # required area scales with local stress vs allowable, capped at a_max
        frac = min(1.0, float(sigma_vm[idx]) * sf / sigma_allow)
        req = max(a_max / params.generative.n_area_buckets, frac * a_max)
        covering = []
        for b in beams:
            pts = np.asarray(b.control_points, dtype=float)
            if np.min(np.linalg.norm(pts - c, axis=1)) <= tol:
                covering.append(b.id)
        if not covering:
            covering = [beams[0].id]  # spar always available as a fallback
        targets.append(
            CoverageTarget(id=k, centroid=tuple(float(v) for v in c),
                           required_min_area_m2=min(req, a_max),
                           candidate_beams=tuple(covering))
        )
    return targets


def _conflict_table(beams, cross_sections):
    """Forbid (beam_i, bucket_a, beam_j, bucket_b) where centerlines (excluding
    shared endpoints) pass closer than the sum of the two bucket radii.
    """
    forbidden = []
    for ia in range(len(beams)):
        for ib in range(ia + 1, len(beams)):
            ba, bb = beams[ia], beams[ib]
            # skip pairs that legitimately share an endpoint node
            shared = set(ba.control_points[:1] + ba.control_points[-1:]) & set(
                bb.control_points[:1] + bb.control_points[-1:]
            )
            # interior min distance: sample interior points only
            ia_pts = ba.control_points[1:-1] or ba.control_points
            ib_pts = bb.control_points[1:-1] or bb.control_points
            dist = _segment_point_distance(ia_pts, ib_pts, None)
            for ca in cross_sections:
                for cb in cross_sections:
                    if dist < (ca.radius_m + cb.radius_m) and not shared:
                        forbidden.append((ba.id, ca.bucket, bb.id, cb.bucket))
    return ConflictTable(forbidden=tuple(forbidden))


def build_candidate_menu(params: DesignParameters) -> CandidateMenu:
    """Build the full CandidateMenu: simple beam library + FEA-driven coverage
    targets + conflict table. Runs the background shell FEA (slow).
    """
    nodes, beams, cross_sections = build_beam_library(params)
    centroids, sigma_vm = _background_stress(params)
    targets = _coverage_targets(beams, centroids, sigma_vm, params)
    conflicts = _conflict_table(beams, cross_sections)
    # attach covers back onto beams (which targets each beam serves)
    covers_by_beam = {}
    for t in targets:
        for bid in t.candidate_beams:
            covers_by_beam.setdefault(bid, []).append(t.id)
    beams = [
        CandidateBeam(
            id=b.id, control_points=b.control_points, start_kind=b.start_kind,
            end_kind=b.end_kind, start_node=b.start_node, end_node=b.end_node,
            length_m=b.length_m, min_radius_m=b.min_radius_m,
            on_chord_plane=b.on_chord_plane, mirror_id=b.mirror_id, host_id=b.host_id,
            covers=tuple(covers_by_beam.get(b.id, ())),
        )
        for b in beams
    ]
    menu = CandidateMenu(
        nodes=tuple(nodes), beams=tuple(beams), cross_sections=tuple(cross_sections),
        conflicts=conflicts, coverage_targets=tuple(targets),
        rho_kgm3=params.material.rho_kgm3,
    )
    validate_menu(menu)
    return menu
