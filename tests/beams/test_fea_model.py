import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.fea_model import build_beam_frame


def test_frame_topology():
    spec = WingSpec()
    nb, nl = 16, 10
    frame = build_beam_frame(spec, n_beams=nb, n_levels=nl)
    assert frame.nodes.shape == (nb * nl, 3)
    # longitudinal: nb*(nl-1); rings: nl*nb
    assert frame.elements.shape[0] == nb * (nl - 1) + nl * nb
    assert frame.fixed_nodes.shape[0] == nb
    # the clamped ring is the most-negative-z (keel-step) level
    z_fixed = frame.nodes[frame.fixed_nodes, 2]
    assert np.allclose(z_fixed, frame.nodes[:, 2].min())
