import numpy as np
import pytest

from wing_design.geometry import small_wingsail, medium_wingsail, large_wingsail
from wing_design.geometry.wing import WingSpec


def _ar(w):
    return w.span / ((w.root_chord + w.tip_chord) / 2.0)


def test_spans():
    assert small_wingsail.span == 5.0
    assert medium_wingsail.span == 22.0
    assert large_wingsail.span == 45.0


def test_similar_aspect_ratio():
    ars = [_ar(small_wingsail), _ar(medium_wingsail), _ar(large_wingsail)]
    assert max(ars) - min(ars) <= 0.05 * max(ars)  # within 5%


def test_dimensionless_fields_shared():
    for w in (medium_wingsail, large_wingsail):
        assert w.thickness == small_wingsail.thickness
        assert w.pivot_frac == small_wingsail.pivot_frac
        assert w.n_sections == small_wingsail.n_sections
        assert w.n_airfoil_points == small_wingsail.n_airfoil_points
        assert w.taper_profile == small_wingsail.taper_profile


def test_small_matches_legacy_default():
    assert (small_wingsail.span, small_wingsail.root_chord, small_wingsail.tip_chord) == (5.0, 1.0, 0.6)
    assert small_wingsail.spar_length == 0.75
    assert small_wingsail.transition_length == 0.20


def test_are_wingspec_instances():
    for w in (small_wingsail, medium_wingsail, large_wingsail):
        assert isinstance(w, WingSpec)


def test_wingspec_has_no_defaults():
    with pytest.raises(TypeError):
        WingSpec()
