import numpy as np

from wing_design.geometry import WingSpec
from wing_design.aero.loads import PanelLoads
from wing_design.beams.fea_model import build_beam_frame, project_panels_to_beam_nodes
from wing_design.structural.projection import R_GEOM_FROM_AERO


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
    assert frame.E > 0.0
    assert frame.G > 0.0


def test_projection_conserves_force():
    spec = WingSpec()
    frame = build_beam_frame(spec, n_beams=16, n_levels=10)
    # one fake panel in the aero frame (span along +Y), out on the wing
    panels = PanelLoads(
        centers_xyz=np.array([[0.0, 2.0, 0.1]]),
        areas=np.array([0.5]),
        forces_xyz=np.array([[10.0, 0.0, 500.0]]),
        normals_xyz=np.array([[0.0, 0.0, 1.0]]),
        chords=np.array([0.8]),
        spanwise_widths=np.array([0.3]),
    )
    loads = project_panels_to_beam_nodes(frame, panels, safety_factor=2.0)
    expected = R_GEOM_FROM_AERO @ (panels.forces_xyz[0] * 2.0)
    assert np.allclose(loads[:, :3].sum(axis=0), expected)
    assert np.allclose(loads[:, 3:], 0.0)  # forces only, no applied moments
