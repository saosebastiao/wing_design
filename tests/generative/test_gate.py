import math

import numpy as np
import pytest

from wing_design.generative.gate import (
    FrameModel,
    assemble_global_K,
    beam_transform,
    bearing_couple_fixed_dofs,
    build_frame,
    element_global_stiffness,
    local_beam_stiffness,
    recover_max_stress_ratio,
    section_properties,
    solve_displacements,
    solve_frame,
    tip_deflection,
    tip_node_indices,
)
from wing_design.generative.menu import (
    CandidateBeam,
    CandidateMenu,
    CandidateNode,
    ConflictTable,
    CrossSectionOption,
    CrossSectionShape,
    GateResult,
    NodeKind,
    WingCandidate,
)
from wing_design.scenario import default_scenario


def test_section_properties_circle():
    area = 4.0e-3
    A, I, J = section_properties(area)
    assert math.isclose(A, area, rel_tol=1e-12)
    # circle: I = A^2 / (4 pi), J = 2 I
    assert math.isclose(I, area**2 / (4.0 * math.pi), rel_tol=1e-12)
    assert math.isclose(J, 2.0 * I, rel_tol=1e-12)


def test_local_beam_stiffness_symmetric_and_axial():
    E, G, A, I, J, L = 200e9, 80e9, 1e-3, 1e-6, 2e-6, 2.0
    k = local_beam_stiffness(E, G, A, I, J, L)
    assert k.shape == (12, 12)
    # symmetric
    assert np.allclose(k, k.T)
    # axial sub-terms: k[0,0] = EA/L, k[0,6] = -EA/L
    assert math.isclose(k[0, 0], E * A / L, rel_tol=1e-12)
    assert math.isclose(k[0, 6], -E * A / L, rel_tol=1e-12)
    # torsion: k[3,3] = GJ/L
    assert math.isclose(k[3, 3], G * J / L, rel_tol=1e-12)
    # bending: k[1,1] = 12 EI / L^3
    assert math.isclose(k[1, 1], 12.0 * E * I / L**3, rel_tol=1e-12)


def test_beam_transform_length_and_orthonormal():
    T, L = beam_transform((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    assert math.isclose(L, 3.0, rel_tol=1e-12)
    assert T.shape == (12, 12)
    R = T[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(R[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_beam_transform_handles_vertical_beam():
    T, L = beam_transform((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
    assert math.isclose(L, 5.0, rel_tol=1e-12)
    R = T[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(R[0], [0.0, 0.0, 1.0], atol=1e-12)


def test_element_global_stiffness_symmetric_and_axisaligned():
    E, G, A, I, J = 200e9, 80e9, 1e-3, 1e-6, 2e-6
    ke = element_global_stiffness((0, 0, 0), (2.0, 0, 0), E, G, A, I, J)
    assert ke.shape == (12, 12)
    assert np.allclose(ke, ke.T)
    k_local = local_beam_stiffness(E, G, A, I, J, 2.0)
    assert np.allclose(ke, k_local, atol=1e-6)


def test_beam_transform_rejects_zero_length_element():
    with pytest.raises(ValueError, match="degenerate"):
        beam_transform((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))


# Shared beam properties for the analytic single-element checks.
_E, _G, _A, _I, _J, _L = 200e9, 80e9, 1e-3, 1e-6, 2e-6, 2.0
_CLAMP = [0, 1, 2, 3, 4, 5]  # all 6 DOF at node i


def test_axial_extension_matches_PL_over_EA():
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, _A, _I, _J)
    P = 1000.0
    load = np.zeros(12)
    load[6 + 0] = P  # axial (Fx) at node j
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 0], P * _L / (_E * _A), rtol=1e-6)


def test_horizontal_cantilever_matches_PL3_over_3EI():
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, _A, _I, _J)
    P = 1000.0
    load = np.zeros(12)
    load[6 + 1] = P  # transverse (Fy) at node j
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 1], P * _L**3 / (3.0 * _E * _I), rtol=1e-6)


def test_vertical_cantilever_matches_PL3_over_3EI():
    ke = element_global_stiffness((0, 0, 0), (0, 0, _L), _E, _G, _A, _I, _J)
    P = 1000.0
    load = np.zeros(12)
    load[6 + 0] = P  # transverse (Fx) at the top node
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 0], P * _L**3 / (3.0 * _E * _I), rtol=1e-6)


def test_torsion_matches_TL_over_GJ():
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, _A, _I, _J)
    Tq = 1000.0
    load = np.zeros(12)
    load[6 + 3] = Tq  # torque (Mx) about the beam axis at node j
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 3], Tq * _L / (_G * _J), rtol=1e-6)


def _single_beam_menu():
    """A keel->tip beam through the deck-step, with the three landmark nodes."""
    nodes = (
        CandidateNode(id=0, xyz=(0.0, 0.0, -0.95), kind=NodeKind.KEEL_STEP, z_layer=0),
        CandidateNode(id=1, xyz=(0.0, 0.0, -0.20), kind=NodeKind.DECK_STEP, z_layer=1),
        CandidateNode(id=2, xyz=(0.0, 0.0, 5.0), kind=NodeKind.TIP, z_layer=9),
    )
    beam = CandidateBeam(
        id=0,
        control_points=((0.0, 0.0, -0.95), (0.0, 0.0, -0.20), (0.0, 0.0, 5.0)),
        start_kind=NodeKind.KEEL_STEP,
        end_kind=NodeKind.TIP,
        start_node=0,
        end_node=2,
        length_m=5.95,
        min_radius_m=100.0,
        on_chord_plane=True,
        mirror_id=None,
        host_id=None,
        covers=(),
    )
    cs = (CrossSectionOption(bucket=0, shape=CrossSectionShape.CIRCLE, area_m2=4.0e-3),)
    menu = CandidateMenu(
        nodes=nodes, beams=(beam,), cross_sections=cs,
        conflicts=ConflictTable(forbidden=()), coverage_targets=(), rho_kgm3=1550.0,
    )
    return menu


def test_build_frame_nodes_elements_and_kinds():
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    assert isinstance(frame, FrameModel)
    assert frame.coords.shape == (3, 3)
    assert len(frame.elements) == 2
    assert frame.node_kinds[0] == NodeKind.KEEL_STEP
    assert frame.node_kinds[1] == NodeKind.DECK_STEP
    assert frame.node_kinds[2] == NodeKind.TIP
    for (_i, _j, area) in frame.elements:
        assert math.isclose(area, 4.0e-3, rel_tol=1e-12)
    assert math.isclose(frame.mass_kg, 12.5, rel_tol=1e-12)


def test_assemble_global_K_shape_and_symmetry():
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    K = assemble_global_K(frame, E=135e9, G=4.5e9)
    n = 6 * frame.coords.shape[0]
    assert K.shape == (n, n)
    assert np.allclose(K, K.T, atol=1e-3)


def test_bearing_couple_fixed_dofs():
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    fixed = bearing_couple_fixed_dofs(frame)
    assert set(fixed) == {0, 1, 2, 5, 6, 7}


def test_tip_node_indices():
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    assert tip_node_indices(frame) == [2]


def test_recover_axial_stress_ratio():
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, _A, _I, _J)
    P = 1000.0
    load = np.zeros(12)
    load[6 + 0] = P
    u = solve_displacements(ke, load, _CLAMP)
    frame = FrameModel(
        coords=np.array([[0, 0, 0], [_L, 0, 0]], dtype=float),
        elements=[(0, 1, _A)],
        node_kinds=[None, None],
        mass_kg=0.0,
    )
    sigma_allow = 1.0e9
    ratio = recover_max_stress_ratio(frame, u, _E, _G, sigma_allow)
    assert np.isclose(ratio, (P / _A) / sigma_allow, rtol=1e-4)


def test_recover_bending_stress_ratio():
    # Single X-aligned cantilever, transverse tip load P. Max bending moment is
    # P*L at the clamped end, so peak bending stress = P*L*r/I. Build the element
    # stiffness from the SAME area-derived I that recovery uses, so the solve and
    # the recovery are consistent.
    A, I, J = section_properties(_A)
    radius = math.sqrt(_A / math.pi)
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, A, I, J)
    P = 500.0
    load = np.zeros(12)
    load[6 + 1] = P  # transverse Fy at the tip
    u = solve_displacements(ke, load, _CLAMP)
    frame = FrameModel(
        coords=np.array([[0, 0, 0], [_L, 0, 0]], dtype=float),
        elements=[(0, 1, _A)],
        node_kinds=[None, None],
        mass_kg=0.0,
    )
    sigma_allow = 1.0e9
    expected = (P * _L * radius / I) / sigma_allow
    ratio = recover_max_stress_ratio(frame, u, _E, _G, sigma_allow)
    assert np.isclose(ratio, expected, rtol=1e-3)


def test_tip_deflection_lateral():
    coords = np.array([[0, 0, -0.95], [0, 0, 5.0]], dtype=float)
    frame = FrameModel(coords=coords, elements=[(0, 1, _A)],
                       node_kinds=[NodeKind.KEEL_STEP, NodeKind.TIP], mass_kg=0.0)
    u = np.zeros(12)
    u[6 + 0] = 0.03
    u[6 + 1] = 0.04
    assert np.isclose(tip_deflection(frame, u), 0.05, rtol=1e-9)


def test_solve_frame_feasible_for_stout_beam_small_load():
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    params = default_scenario()
    tip = tip_node_indices(frame)[0]
    loads = {tip: (200.0, 0.0, 0.0)}
    result = solve_frame(frame, params, loads, governing_case="nominal")
    assert isinstance(result, GateResult)
    assert result.governing_case == "nominal"
    assert math.isclose(result.mass_kg, 12.5, rel_tol=1e-12)
    assert result.tip_deflection_m > 0.0
    assert result.max_stress_ratio > 0.0
    assert result.feasible is True


def test_solve_frame_requires_keel_and_deck_landmarks():
    # If build_frame failed to locate the keel/deck landmarks (e.g. a menu-vs-
    # endpoint coordinate mismatch), the structure is under-constrained. Reject
    # it loudly rather than returning a silent rigid-body-drift verdict.
    frame = FrameModel(
        coords=np.array([[0, 0, 0], [0, 0, 5.0]], dtype=float),
        elements=[(0, 1, _A)],
        node_kinds=[None, NodeKind.TIP],  # no KEEL_STEP / DECK_STEP
        mass_kg=1.0,
    )
    params = default_scenario()
    with pytest.raises(ValueError, match="KEEL_STEP"):
        solve_frame(frame, params, {1: (100.0, 0.0, 0.0)})


def test_solve_frame_infeasible_under_huge_load():
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    params = default_scenario()
    tip = tip_node_indices(frame)[0]
    loads = {tip: (5.0e5, 0.0, 0.0)}
    result = solve_frame(frame, params, loads)
    assert result.feasible is False


from wing_design.generative.loads import lump_spanwise_force_to_nodes


def test_build_frame_subdivides_long_segments():
    menu = _single_beam_menu()  # spar keel(-0.95)->deck(-0.20)->tip(5.0)
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=1.0)
    # No subdivision: 3 nodes / 2 elements (the M1 behavior).
    coarse = build_frame(candidate, menu, max_element_length_m=None)
    assert coarse.coords.shape == (3, 3)
    assert len(coarse.elements) == 2
    # Subdivided: each ~0.3 m -> many interior nodes; landmarks keep their kinds.
    fine = build_frame(candidate, menu, max_element_length_m=0.3)
    assert fine.coords.shape[0] > 3
    assert len(fine.elements) > 2
    assert NodeKind.KEEL_STEP in fine.node_kinds
    assert NodeKind.DECK_STEP in fine.node_kinds
    assert NodeKind.TIP in fine.node_kinds
    # interior nodes are unclassified
    assert fine.node_kinds.count(None) >= 1


def test_subdivision_improves_distributed_load_capture():
    # Non-uniform line load density(z)=z^2 over the wing span [0, span]; analytic
    # integral = span^3 / 3. A coarse spar frame (only the tip node is in-span)
    # badly under/over-integrates; a finely subdivided one converges to it.
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=1.0)
    span = 5.0
    analytic = span**3 / 3.0

    def density(z):
        return z * z

    coarse = build_frame(candidate, menu, max_element_length_m=None)
    fine = build_frame(candidate, menu, max_element_length_m=0.1)

    def total(frame):
        loads = lump_spanwise_force_to_nodes(frame, density, z_min=0.0, z_max=span,
                                             direction=(0.0, 1.0, 0.0))
        return sum(fy for (_fx, fy, _fz) in loads.values())

    coarse_err = abs(total(coarse) - analytic) / analytic
    fine_err = abs(total(fine) - analytic) / analytic
    assert fine_err < 0.02          # fine frame integrates the load accurately
    assert fine_err < coarse_err    # and is strictly better than the coarse frame
