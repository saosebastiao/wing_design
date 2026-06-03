import dataclasses

from ortools.sat.python import cp_model

from wing_design.generative.model import build_cp_model
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


from wing_design.generative.menu import NodeKind


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
