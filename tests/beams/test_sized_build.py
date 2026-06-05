import numpy as np

from wing_design.geometry import WingSpec, build_wing_solid
from wing_design.beams.build import build_sized_lens_beams, build_sized_circular_beams, apply_wing_fillets


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


def test_apply_wing_fillets_returns_valid_solid():
    spec = WingSpec()
    base = build_wing_solid(spec)
    out = apply_wing_fillets(base, spec)
    # Whatever fillets succeed, the result must be a valid, non-degenerate solid.
    assert out.volume > 0.0
    # Filleting only rounds edges, so volume changes modestly (never collapses/explodes).
    assert 0.5 * base.volume < out.volume < 1.5 * base.volume
