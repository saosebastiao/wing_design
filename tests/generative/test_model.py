import dataclasses
import math

from ortools.sat.python import cp_model

from wing_design.generative.menu import NodeKind, WingCandidate
from wing_design.generative.model import build_cp_model, solve_designs
from wing_design.scenario import GenerativeParameters

from _menu_factory import beam, cs_catalog, menu, target


def _params(**overrides):
    return dataclasses.replace(GenerativeParameters(), **overrides)


def _solve(model):
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    return solver, status


def test_count_forces_exact_number_selected():
    m = menu([beam(0), beam(1), beam(2)])
    params = _params(n_beams_min=2, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    n_selected = sum(solver.value(select[b.id]) for b in m.beams)
    assert n_selected == 2


def test_one_hot_section_matches_selection():
    m = menu([beam(0), beam(1), beam(2)])
    params = _params(n_beams_min=2, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    for b in m.beams:
        n_sect = sum(solver.value(sect[(b.id, cs.bucket)]) for cs in m.cross_sections)
        assert n_sect == solver.value(select[b.id])


def test_symmetry_ties_mirror_pair():
    # Beams 0 and 1 are a mirror pair; selecting one must select the other.
    b0 = beam(0, on_chord_plane=False, mirror_id=1)
    b1 = beam(1, on_chord_plane=False, mirror_id=0)
    m = menu([b0, b1])
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    model.add(select[0] == 1)  # assume one half is selected
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.value(select[1]) == 1


def test_symmetry_ties_mirror_pair_sections():
    b0 = beam(0, on_chord_plane=False, mirror_id=1)
    b1 = beam(1, on_chord_plane=False, mirror_id=0)
    m = menu([b0, b1], cross_sections=cs_catalog([1.0e-3, 2.0e-3]))
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    model.add(sect[(0, 1)] == 1)  # beam 0 uses bucket 1
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.value(sect[(1, 1)]) == 1  # mirror uses the same bucket


def test_host_implication_requires_host_selected():
    # Beam 1 starts on beam 0; selecting beam 1 forces beam 0.
    host = beam(0, start_kind=NodeKind.KEEL_STEP, end_kind=NodeKind.TIP)
    dependent = beam(1, start_kind=NodeKind.ON_BEAM, end_kind=NodeKind.TIP, host_id=0)
    m = menu([host, dependent])
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    model.add(select[1] == 1)  # select the dependent beam
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.value(select[0]) == 1  # host pulled in


def test_host_can_exist_without_dependent():
    host = beam(0, start_kind=NodeKind.KEEL_STEP, end_kind=NodeKind.TIP)
    dependent = beam(1, start_kind=NodeKind.ON_BEAM, end_kind=NodeKind.TIP, host_id=0)
    m = menu([host, dependent])
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    model.add(select[0] == 1)
    model.add(select[1] == 0)  # host selected, dependent not — must be feasible
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_reach_tip_forces_a_tip_beam():
    # Only beam 0 reaches the tip; beam 1 ends on beam 0. With min=0, the model
    # must still select at least one tip-reaching beam (beam 0).
    tip_beam = beam(0, start_kind=NodeKind.KEEL_STEP, end_kind=NodeKind.TIP)
    inner = beam(1, start_kind=NodeKind.KEEL_STEP, end_kind=NodeKind.ON_BEAM, host_id=0)
    m = menu([tip_beam, inner])
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.value(select[0]) == 1


def test_reach_tip_infeasible_when_no_tip_beam_in_menu():
    # A menu whose only beam ends at the deck-step (no tip-reaching beam) must
    # be infeasible: sum([]) >= 1 cannot be satisfied.
    root = beam(0, start_kind=NodeKind.KEEL_STEP, end_kind=NodeKind.DECK_STEP)
    m = menu([root])
    params = _params(n_beams_min=0, n_beams_max=1)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status == cp_model.INFEASIBLE


def test_conflict_makes_both_at_forbidden_buckets_infeasible():
    # Single-bucket catalog; beams 0 and 1 conflict at (bucket 0, bucket 0).
    # Forcing both selected (min=max=2) must be infeasible.
    m = menu(
        [beam(0), beam(1)],
        cross_sections=cs_catalog([1.0e-3]),
        conflicts=[(0, 0, 1, 0)],
    )
    params = _params(n_beams_min=2, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status == cp_model.INFEASIBLE


def test_conflict_allows_different_buckets():
    # Two buckets; the conflict is only at (0,0,1,0). Both beams can be selected
    # if at least one uses bucket 1.
    m = menu(
        [beam(0), beam(1)],
        cross_sections=cs_catalog([1.0e-3, 2.0e-3]),
        conflicts=[(0, 0, 1, 0)],
    )
    params = _params(n_beams_min=2, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_coverage_forces_adequate_section():
    # Target needs area >= 2e-3 and only beam 0 can cover it. The model must
    # select beam 0 at a bucket whose area >= 2e-3 (bucket 1 in the catalog).
    m = menu(
        [beam(0, covers=(5,)), beam(1)],
        cross_sections=cs_catalog([1.0e-3, 2.0e-3]),
        coverage=[target(5, required_min_area_m2=2.0e-3, candidate_beams=[0])],
    )
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.value(sect[(0, 1)]) == 1  # adequate bucket chosen
    assert solver.value(select[0]) == 1


def test_coverage_infeasible_when_no_bucket_is_big_enough():
    # Target needs 5e-3 but the catalog tops out at 2e-3 -> infeasible.
    m = menu(
        [beam(0, covers=(5,))],
        cross_sections=cs_catalog([1.0e-3, 2.0e-3]),
        coverage=[target(5, required_min_area_m2=5.0e-3, candidate_beams=[0])],
    )
    params = _params(n_beams_min=0, n_beams_max=1)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status == cp_model.INFEASIBLE


def test_solve_picks_minimum_mass_design():
    # Both beams can cover the target at bucket 0 (area 1e-3). Beam 0 is short
    # (length 1), beam 1 is long (length 10). Minimum mass picks beam 0.
    m = menu(
        [beam(0, length=1.0, covers=(5,)), beam(1, length=10.0, covers=(5,))],
        cross_sections=cs_catalog([1.0e-3, 2.0e-3]),
        coverage=[target(5, required_min_area_m2=1.0e-3, candidate_beams=[0, 1])],
        rho=1550.0,
    )
    params = _params(n_beams_min=0, n_beams_max=2)
    designs = solve_designs(m, params, top_n=1)
    assert len(designs) == 1
    best = designs[0]
    assert isinstance(best, WingCandidate)
    assert best.beam_sections == ((0, 0),)
    # mass = length * area * rho = 1.0 * 1e-3 * 1550 = 1.55 kg
    assert math.isclose(best.mass_kg, 1.55, rel_tol=1e-6)


def test_top_n_returns_distinct_non_decreasing_designs():
    # Three independent tip beams, each can cover its own target at bucket 0.
    # Different lengths -> three distinct single-beam-ish designs by mass.
    m = menu(
        [
            beam(0, length=1.0, covers=(10,)),
            beam(1, length=2.0, covers=(11,)),
            beam(2, length=3.0, covers=(12,)),
        ],
        cross_sections=cs_catalog([1.0e-3]),
        coverage=[
            target(10, 1.0e-3, [0]),
            target(11, 1.0e-3, [1]),
            target(12, 1.0e-3, [2]),
        ],
        rho=1550.0,
    )
    params = _params(n_beams_min=3, n_beams_max=3)
    designs = solve_designs(m, params, top_n=3)
    # With all three targets, the only feasible selection is all three beams;
    # so only one distinct design exists and top_n must not fabricate more.
    assert len(designs) == 1
    assert set(designs[0].beam_ids) == {0, 1, 2}


def test_top_n_enumerates_multiple_when_choices_exist():
    # One target coverable by either beam 0 or beam 1 (different masses).
    # min=1 lets the solver pick exactly one; top_n=2 should surface both,
    # lightest first.
    m = menu(
        [beam(0, length=1.0, covers=(7,)), beam(1, length=2.0, covers=(7,))],
        cross_sections=cs_catalog([1.0e-3]),
        coverage=[target(7, 1.0e-3, [0, 1])],
        rho=1550.0,
    )
    params = _params(n_beams_min=1, n_beams_max=1)
    designs = solve_designs(m, params, top_n=2)
    assert len(designs) == 2
    masses = [d.mass_kg for d in designs]
    assert masses == sorted(masses)  # non-decreasing
    assert designs[0].beam_ids == (0,)  # lightest first
    assert designs[1].beam_ids == (1,)


def test_enforce_coverage_flag_toggles_constraint():
    # A target needs area >= 5e-3 but the catalog tops out at 2e-3, so coverage
    # is unsatisfiable. With enforce_coverage=True the model is INFEASIBLE; with
    # enforce_coverage=False the constraint is dropped and the model is feasible.
    m = menu(
        [beam(0, covers=(5,))],
        cross_sections=cs_catalog([1.0e-3, 2.0e-3]),
        coverage=[target(5, required_min_area_m2=5.0e-3, candidate_beams=[0])],
    )
    params = _params(n_beams_min=0, n_beams_max=1)

    model_on, _s, _x = build_cp_model(m, params, enforce_coverage=True)
    solver_on, status_on = _solve(model_on)
    assert status_on == cp_model.INFEASIBLE

    model_off, _s2, _x2 = build_cp_model(m, params, enforce_coverage=False)
    solver_off, status_off = _solve(model_off)
    assert status_off in (cp_model.OPTIMAL, cp_model.FEASIBLE)
