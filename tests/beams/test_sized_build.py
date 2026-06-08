import numpy as np

from wing_design.geometry import small_wingsail
from wing_design.beams.build import build_sized_lens_beams, build_sized_circular_beams, resample_segment_radii


def _radii(n_beams, n_levels):
    # one radius per longitudinal segment, beam-major/level-minor
    return np.full(n_beams * (n_levels - 1), 0.02)


def test_sized_circular_beams_build():
    spec = small_wingsail
    beams = build_sized_circular_beams(spec, _radii(8, 5), n_beams=8, n_levels=5)
    assert len(beams) == 8
    for b in beams:
        assert b.volume > 0.0


def test_sized_lens_beams_build():
    spec = small_wingsail
    beams = build_sized_lens_beams(spec, _radii(8, 5), n_beams=8, n_levels=5)
    assert len(beams) == 8
    for b in beams:
        assert b.volume > 0.0


def test_resample_uniform_radii_stays_uniform():
    out = resample_segment_radii(np.full(4 * 3, 0.02), n_beams=4, n_from=4, n_to=9)
    assert out.shape == (4 * 8,)          # n_beams * (n_to - 1)
    assert np.allclose(out, 0.02)


def test_resample_preserves_per_beam_trend():
    # beam0 segments [1,3], beam1 segments [5,7]; resampled to 4 segments each stays
    # monotone and within the original range.
    n_beams, n_from = 2, 3
    long_radii = np.array([1.0, 3.0, 5.0, 7.0])
    out = resample_segment_radii(long_radii, n_beams=n_beams, n_from=n_from, n_to=5)
    seg_to = 4
    b0 = out[:seg_to]
    b1 = out[seg_to:]
    assert np.all(np.diff(b0) >= 0) and b0.min() >= 1.0 - 1e-9 and b0.max() <= 3.0 + 1e-9
    assert np.all(np.diff(b1) >= 0) and b1.min() >= 5.0 - 1e-9 and b1.max() <= 7.0 + 1e-9
