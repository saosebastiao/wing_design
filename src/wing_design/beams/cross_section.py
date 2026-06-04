"""Sample the OML cross-section into evenly arc-spaced form-beam anchor points."""
from __future__ import annotations

import numpy as np

from ..geometry.wing import WingSpec, oml_section_polyline


def resample_closed_polyline(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline into ``n`` points equally spaced by arc length.

    ``poly`` is (M, 2). If its first and last points coincide they are treated
    as the single loop-closure vertex. Output point 0 sits at ``poly[0]``; the
    remaining points march at equal arc-length steps around the loop (the loop's
    closing point is excluded, since it would duplicate point 0).
    """
    pts = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    loop = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    targets = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(targets, s, loop[:, 0])
    y = np.interp(targets, s, loop[:, 1])
    return np.column_stack([x, y])


def beam_section_points(
    spec: WingSpec, z: float, n_beams: int, n_pts: int | None = None
) -> np.ndarray:
    """``n_beams`` (x, y) points equally arc-spaced around the OML at height ``z``.

    Point 0 lies at the trailing edge; for a symmetric section and even
    ``n_beams``, point ``n_beams // 2`` lies at the leading edge.
    """
    poly = oml_section_polyline(spec, z, n_pts=n_pts)
    return resample_closed_polyline(poly, n_beams)
