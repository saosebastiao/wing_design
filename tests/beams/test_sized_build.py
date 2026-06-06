import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.build import build_sized_lens_beams, build_sized_circular_beams


def _radii(n_beams, n_levels):
    # one radius per longitudinal segment, beam-major/level-minor
    return np.full(n_beams * (n_levels - 1), 0.02)


def test_sized_circular_beams_build():
    spec = WingSpec()
    beams = build_sized_circular_beams(spec, _radii(8, 5), n_beams=8, n_levels=5)
    assert len(beams) == 8
    for b in beams:
        assert b.volume > 0.0


def test_sized_lens_beams_build():
    spec = WingSpec()
    beams = build_sized_lens_beams(spec, _radii(8, 5), n_beams=8, n_levels=5)
    assert len(beams) == 8
    for b in beams:
        assert b.volume > 0.0
