"""CP-SAT model that selects a wing-truss design from a CandidateMenu.

Conventions follow the ortools-cp skill: snake_case API, booleans + reification
over big-M, integer-scaled objective. The model exposes its variable
dictionaries so tests (and the outer loop) can add assumptions/cuts.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from ..scenario import GenerativeParameters
from .menu import CandidateMenu, NodeKind, WingCandidate

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


def _add_topology(model, menu, select):
    """A beam that starts/ends on a host requires that host: select_b <= select_host.

    Hosts always start at strictly lower z, so these implications form a DAG that
    transitively grounds every selected beam back to the keel-step.
    """
    for b in menu.beams:
        if b.host_id is not None:
            model.add(select[b.id] <= select[b.host_id])


def _add_reach_tip(model, menu, select):
    """At least one selected beam must terminate at the wing tip.

    The monotonic-increasing-z invariant means the tip is always a beam's
    high-z *end*, never its start. If the menu has no tip-reaching beam this
    becomes sum([]) == 0 >= 1, i.e. genuinely infeasible — a malformed menu,
    not a silent pass (mirrors `_add_coverage`).
    """
    tip_beams = [select[b.id] for b in menu.beams if b.end_kind == NodeKind.TIP]
    model.add(sum(tip_beams) >= 1)


def _add_no_intersection(model, menu, sect):
    """For each forbidden (beam_i, bucket_a, beam_j, bucket_b): not both."""
    for (bi, a, bj, b) in menu.conflicts.forbidden:
        model.add(sect[(bi, a)] + sect[(bj, b)] <= 1)


def _add_coverage(model, menu, sect):
    """Each high-stress target must be served by a selected beam whose assigned
    section is large enough. Because sect implies select (one-hot tie), coverage
    also forces selection of a covering beam.
    """
    for tgt in menu.coverage_targets:
        adequate = [
            sect[(bid, cs.bucket)]
            for bid in tgt.candidate_beams
            for cs in menu.cross_sections
            if cs.area_m2 >= tgt.required_min_area_m2
        ]
        # If no (beam, bucket) can satisfy the target, this is an empty sum == 0
        # >= 1, i.e. genuinely infeasible — which is the correct verdict.
        model.add(sum(adequate) >= 1)


def build_cp_model(menu: CandidateMenu, params: GenerativeParameters):
    """Build the CP-SAT model. Returns (model, select, sect)."""
    model = cp_model.CpModel()
    select, sect = _add_variables(model, menu)
    _add_count(model, menu, params, select)
    _add_symmetry(model, menu, select, sect)
    _add_topology(model, menu, select)
    _add_reach_tip(model, menu, select)
    _add_no_intersection(model, menu, sect)
    _add_coverage(model, menu, sect)
    return model, select, sect


def _beam_bucket_mass_kg(menu, beam_id, bucket):
    """Mass of a single beam at a given cross-section bucket, in kg."""
    b = menu.beam_by_id(beam_id)
    cs = next(c for c in menu.cross_sections if c.bucket == bucket)
    return b.length_m * cs.area_m2 * menu.rho_kgm3


def _set_mass_objective(model, menu, sect):
    """Minimize total mass, scaled to integer milligrams."""
    terms = []
    for (beam_id, bucket), var in sect.items():
        mass_mg = round(_beam_bucket_mass_kg(menu, beam_id, bucket) * MASS_SCALE)
        terms.append(mass_mg * var)
    model.minimize(sum(terms))


def _extract_design(solver, menu, sect):
    """Reconstruct a WingCandidate from a solved model's variable values.

    Relies on the one-hot tie: an active `sect` variable implies its beam is
    selected, so iterating `sect` alone fully describes the chosen design.
    """
    chosen = []
    mass_kg = 0.0
    for (beam_id, bucket), var in sect.items():
        if solver.value(var) == 1:
            chosen.append((beam_id, bucket))
            mass_kg += _beam_bucket_mass_kg(menu, beam_id, bucket)
    return WingCandidate(beam_sections=tuple(chosen), mass_kg=mass_kg)


def solve_designs(menu, params, top_n=1):
    """Solve the menu and return up to `top_n` distinct designs by ascending mass.

    Each round minimizes mass, records the design, then adds a no-good cut
    forbidding that exact (beam, bucket) set before re-solving — so successive
    designs are distinct.

    Rounds accept OPTIMAL *or* FEASIBLE results; under a `solver_max_time_s`
    time-out a round may return a non-optimal FEASIBLE design lighter than a
    prior round's optimum, so the harvested list is sorted by mass before
    returning to guarantee the lightest-first ordering regardless of status.
    """
    model, select, sect = build_cp_model(menu, params)
    _set_mass_objective(model, menu, sect)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params.solver_max_time_s

    designs = []
    for _ in range(top_n):
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        designs.append(_extract_design(solver, menu, sect))
        chosen_vars = [
            var for key, var in sect.items() if solver.value(var) == 1
        ]
        # Forbid this exact selection on the next round.
        model.add(sum(chosen_vars) <= len(chosen_vars) - 1)
    designs.sort(key=lambda d: d.mass_kg)
    return designs
