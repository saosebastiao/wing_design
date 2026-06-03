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
