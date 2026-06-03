import numpy as np

from wing_design.generative.gate import FrameModel
from wing_design.generative.loads import lump_spanwise_force_to_nodes
from wing_design.generative.menu import NodeKind


def _frame():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.5],
        [0.0, 0.0, 5.0],
    ], dtype=float)
    return FrameModel(coords=coords, elements=[(0, 1, 1e-3), (1, 2, 1e-3)],
                      node_kinds=[NodeKind.KEEL_STEP, None, NodeKind.TIP], mass_kg=1.0)


def test_lump_conserves_total_force():
    frame = _frame()
    # uniform 100 N/m over a 5 m span -> 500 N total, applied in +Y
    loads = lump_spanwise_force_to_nodes(frame, lambda z: 100.0, z_min=0.0, z_max=5.0,
                                         direction=(0.0, 1.0, 0.0))
    total_fy = sum(fy for (_fx, fy, _fz) in loads.values())
    assert np.isclose(total_fy, 500.0, rtol=1e-6)


def test_lump_only_nodes_in_span_range_get_load():
    frame = _frame()
    loads = lump_spanwise_force_to_nodes(frame, lambda z: 100.0, z_min=0.0, z_max=5.0,
                                         direction=(0.0, 1.0, 0.0))
    # the keel node at z=0 is on the boundary; the loaded nodes are within range
    for node_idx, (_fx, fy, _fz) in loads.items():
        assert fy >= 0.0
        assert 0.0 <= frame.coords[node_idx][2] <= 5.0
