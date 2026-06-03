"""CP-SAT model that selects a wing-truss design from a CandidateMenu.

Conventions follow the ortools-cp skill: snake_case API, booleans + reification
over big-M, integer-scaled objective. The model exposes its variable
dictionaries so tests (and the outer loop) can add assumptions/cuts.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from ..scenario import GenerativeParameters
from .menu import CandidateMenu, WingCandidate

# kg -> milligrams: keeps the mass objective in integer coefficients.
MASS_SCALE = 1_000_000


def _add_variables(model, menu):
    """select[beam_id] and sect[(beam_id, bucket)] booleans, one-hot-tied."""
    select = {b.id: model.new_bool_var(f"select_{b.id}") for b in menu.beams}
    sect = {}
    for b in menu.beams:
        for cs in menu.cross_sections:
            sect[(b.id, cs.bucket)] = model.new_bool_var(f"sect_{b.id}_{cs.bucket}")
    # Exactly one section iff the beam is selected.
    for b in menu.beams:
        model.add(
            sum(sect[(b.id, cs.bucket)] for cs in menu.cross_sections) == select[b.id]
        )
    return select, sect


def _add_count(model, menu, params, select):
    """n_beams_min <= number of selected beams <= n_beams_max."""
    total = sum(select.values())
    model.add(total >= params.n_beams_min)
    model.add(total <= params.n_beams_max)


def _add_symmetry(model, menu, select, sect):
    """Tie each mirror pair: same selection and same cross-section bucket."""
    seen = set()
    for b in menu.beams:
        if b.mirror_id is None:
            continue
        key = frozenset((b.id, b.mirror_id))
        if key in seen:
            continue
        seen.add(key)
        model.add(select[b.id] == select[b.mirror_id])
        for cs in menu.cross_sections:
            model.add(sect[(b.id, cs.bucket)] == sect[(b.mirror_id, cs.bucket)])


def build_cp_model(menu: CandidateMenu, params: GenerativeParameters):
    """Build the CP-SAT model. Returns (model, select, sect)."""
    model = cp_model.CpModel()
    select, sect = _add_variables(model, menu)
    _add_count(model, menu, params, select)
    _add_symmetry(model, menu, select, sect)
    return model, select, sect
