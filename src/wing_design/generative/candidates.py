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
