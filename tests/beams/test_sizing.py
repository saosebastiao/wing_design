import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.fea_model import build_beam_frame
from wing_design.beams.sizing import element_lengths, n_longitudinal, frame_mass


def test_n_longitudinal():
    spec = WingSpec()
    frame = build_beam_frame(spec, n_beams=8, n_levels=5)
    assert n_longitudinal(frame) == 8 * (5 - 1)


def test_element_lengths_match_geometry():
    spec = WingSpec()
    frame = build_beam_frame(spec, n_beams=8, n_levels=5)
    L = element_lengths(frame)
    assert L.shape[0] == frame.elements.shape[0]
    i, j = frame.elements[0]
    expected = np.linalg.norm(frame.nodes[j] - frame.nodes[i])
    assert abs(L[0] - expected) < 1e-12


def test_frame_mass_uniform_radius():
    spec = WingSpec()
    frame = build_beam_frame(spec, n_beams=8, n_levels=5)
    nl = n_longitudinal(frame)
    L = element_lengths(frame)
    r_long, r_ring, rho = 0.02, 0.01, 1550.0
    radii = np.full(nl, r_long)
    m = frame_mass(frame, radii, ring_radius=r_ring, rho=rho)
    expected = rho * (
        np.pi * r_long**2 * L[:nl].sum() + np.pi * r_ring**2 * L[nl:].sum()
    )
    assert abs(m - expected) / expected < 1e-12
