import math

import pytest

from wing_design.generative.menu import (
    CandidateBeam,
    CandidateMenu,
    CandidateNode,
    ConflictTable,
    CrossSectionOption,
    CrossSectionShape,
    NodeKind,
    validate_menu,
)


def _beam(beam_id, pts, start_kind, end_kind, mirror_id=None, host_id=None, on_chord=True):
    return CandidateBeam(
        id=beam_id,
        control_points=tuple(pts),
        start_kind=start_kind,
        end_kind=end_kind,
        start_node=0,
        end_node=1,
        length_m=10.0,
        min_radius_m=100.0,
        on_chord_plane=on_chord,
        mirror_id=mirror_id,
        host_id=host_id,
        covers=(),
    )


def _menu(nodes, beams):
    return CandidateMenu(
        nodes=tuple(nodes),
        beams=tuple(beams),
        cross_sections=(CrossSectionOption(0, CrossSectionShape.CIRCLE, 1e-3),),
        conflicts=ConflictTable(forbidden=()),
        coverage_targets=(),
        rho_kgm3=1550.0,
    )


def _good_menu():
    nodes = (
        CandidateNode(id=0, xyz=(0.0, 0.0, -0.95), kind=NodeKind.KEEL_STEP, z_layer=0),
        CandidateNode(id=1, xyz=(0.0, 0.0, -0.20), kind=NodeKind.DECK_STEP, z_layer=1),
        CandidateNode(id=2, xyz=(0.0, 0.0, 5.0), kind=NodeKind.TIP, z_layer=9),
    )
    spar = _beam(0, [(0, 0, -0.95), (0, 0, -0.20), (0, 0, 5.0)],
                 NodeKind.KEEL_STEP, NodeKind.TIP)
    branch = _beam(1, [(0, 0, -0.20), (0.3, 0, 2.0), (0, 0, 5.0)],
                   NodeKind.ON_BEAM, NodeKind.TIP, host_id=0)
    return _menu(nodes, [spar, branch])


def test_validate_accepts_good_menu():
    validate_menu(_good_menu())  # must not raise


def test_validate_rejects_non_monotonic_z():
    nodes = _good_menu().nodes
    bad = _beam(0, [(0, 0, -0.95), (0, 0, 2.0), (0, 0, 1.0)],
                NodeKind.KEEL_STEP, NodeKind.TIP)
    with pytest.raises(ValueError, match="monotonic"):
        validate_menu(_menu(nodes, [bad]))


def test_validate_rejects_missing_landmark_node():
    # Beam endpoints have no matching keel landmark node.
    nodes = (
        CandidateNode(id=2, xyz=(0.0, 0.0, 5.0), kind=NodeKind.TIP, z_layer=9),
    )
    spar = _beam(0, [(0, 0, -0.95), (0, 0, 5.0)], NodeKind.KEEL_STEP, NodeKind.TIP)
    with pytest.raises(ValueError, match="landmark"):
        validate_menu(_menu(nodes, [spar]))


def test_validate_rejects_host_cycle():
    nodes = _good_menu().nodes
    b0 = _beam(0, [(0, 0, -0.20), (0, 0, 5.0)], NodeKind.ON_BEAM, NodeKind.TIP, host_id=1)
    b1 = _beam(1, [(0, 0, -0.20), (0, 0, 5.0)], NodeKind.ON_BEAM, NodeKind.TIP, host_id=0)
    with pytest.raises(ValueError, match="cycle"):
        validate_menu(_menu(nodes, [b0, b1]))


def test_validate_rejects_nonreciprocal_mirror():
    nodes = _good_menu().nodes
    b0 = _beam(0, [(0, 0, -0.95), (0, 0, 5.0)], NodeKind.KEEL_STEP, NodeKind.TIP,
               mirror_id=1, on_chord=False)
    b1 = _beam(1, [(0, 0, -0.95), (0, 0, 5.0)], NodeKind.KEEL_STEP, NodeKind.TIP,
               mirror_id=None, on_chord=False)
    with pytest.raises(ValueError, match="mirror"):
        validate_menu(_menu(nodes, [b0, b1]))
