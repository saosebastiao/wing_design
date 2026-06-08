"""Named, fully-instantiated wingsail geometries (a size family).

`WingSpec` is a pure schema; concrete designs live here as module-level instances,
mirroring how `materials.unidir` defines `UDPly` plus named plies (`T700_EPOXY`). The
three sizes share the same airfoil/taper shape (identical dimensionless fields) and a
similar aspect ratio (~6.1-6.25 = span / mean-chord); lengths scale with span.

- `small_wingsail`  - 5 m demo (the historical default); used by the test suite.
- `medium_wingsail` - 22 m wingsail; used by the examples.
- `large_wingsail`  - 45 m wingsail.
"""
from __future__ import annotations

from .wing import WingSpec

small_wingsail = WingSpec(
    span=5.0, root_chord=1.0, tip_chord=0.6, thickness=0.18, pivot_frac=0.25,
    spar_length=0.75, transition_length=0.20,
    n_transition_sections=4, n_sections=5, n_airfoil_points=160, taper_profile=None,
)

medium_wingsail = WingSpec(
    span=22.0, root_chord=4.5, tip_chord=2.7, thickness=0.18, pivot_frac=0.25,
    spar_length=3.3, transition_length=0.9,
    n_transition_sections=4, n_sections=5, n_airfoil_points=160, taper_profile=None,
)

large_wingsail = WingSpec(
    span=45.0, root_chord=9.0, tip_chord=5.4, thickness=0.18, pivot_frac=0.25,
    spar_length=6.75, transition_length=1.8,
    n_transition_sections=4, n_sections=5, n_airfoil_points=160, taper_profile=None,
)
