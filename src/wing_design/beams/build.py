"""build123d geometry for the form beams, the skin wrap, and their assembly.

`build_form_beams`/`build_assembly` are the Phase-A crude builders (fixed-radius
circular sections inset along the radial-from-pivot direction). Phase D adds the
sized builders driven by the FEA-optimized radii — `build_sized_circular_beams`
and `build_sized_lens_beams` (area-exact inward-arc sections placed with true OML
normals). The eased transition (Task 1) removes the crease at spar/root junctions,
making post-build fillets unnecessary.
"""
from __future__ import annotations

import numpy as np
from build123d import (
    BuildLine,
    BuildPart,
    BuildSketch,
    Circle,
    Compound,
    Locations,
    Plane,
    Polyline,
    Solid,
    loft,
    make_face,
)

from ..geometry.wing import WingSpec, build_wing_solid
from .cross_section import beam_section_points
from .sections import lens_section_polyline, oml_outward_normals
from .splines import default_z_levels, form_beam_grid


def build_form_beams(
    spec: WingSpec,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    beam_radius: float = 0.02,
) -> list:
    """One lofted solid per form beam (crude fixed-radius circular section)."""
    z_levels = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z_levels, n_beams)  # (n_beams, n_levels, 3)
    beams = []
    for b in range(n_beams):
        with BuildPart() as bp:
            for k in range(n_levels):
                px, py, pz = grid[b, k]
                radial = np.array([px, py])
                norm = float(np.linalg.norm(radial))
                n_out = radial / norm if norm > 1e-9 else np.array([1.0, 0.0])
                cx, cy = radial - beam_radius * n_out  # inset inward from shell
                with BuildSketch(Plane.XY.offset(float(pz))):
                    with Locations((float(cx), float(cy))):
                        Circle(beam_radius)
            loft(ruled=True)
        beams.append(bp.part)
    return beams


def build_skin_wrap(spec: WingSpec, *, wall: float = 0.003):
    """Phase-A skin: the solid OML.

    ``wall`` is accepted for forward-compatibility but not applied here. A true
    wall-thickness, load-bearing skin is built in Phase D from the beam-arc
    endpoints (the doc's intended winding-formed wrap), not by hollow-offsetting
    this lofted solid — 3D solid ``offset`` is unreliable on the morphing
    spar/transition geometry, and the offset approach is superseded anyway.
    """
    del wall  # not applied at spike fidelity; see docstring
    return build_wing_solid(spec)


def build_assembly(
    spec: WingSpec,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    beam_radius: float = 0.02,
    wall: float = 0.003,
) -> Compound:
    """Compound of the skin wrap + all form beams."""
    beams = build_form_beams(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=beam_radius
    )
    skin = build_skin_wrap(spec, wall=wall)
    for i, b in enumerate(beams):
        b.label = f"beam_{i:02d}"
    skin.label = "skin"
    return Compound(label="wingsail", children=[skin, *beams])


def _check_long_radii(long_radii: np.ndarray, n_beams: int, n_levels: int) -> None:
    """Fail clearly if the sized-radius array isn't beam-major/level-minor sized."""
    expected = n_beams * (n_levels - 1)
    if len(long_radii) != expected:
        raise ValueError(
            f"long_radii must have n_beams*(n_levels-1) = {expected} entries, "
            f"got {len(long_radii)}"
        )


def _segment_station_radii(long_radii: np.ndarray, b: int, n_levels: int) -> np.ndarray:
    """Per-station radii for beam ``b`` from its (n_levels-1) segment radii.

    Station k uses the radius of segment min(k, n_seg-1), so the loft is
    piecewise-linear (each interior segment tapers from r_k to r_{k+1}); only the
    final, clamped segment is prismatic.
    """
    seg = n_levels - 1
    radii_b = np.asarray(long_radii)[b * seg:(b + 1) * seg]
    return np.array([radii_b[min(k, seg - 1)] for k in range(n_levels)])


def build_sized_circular_beams(
    spec: WingSpec,
    long_radii: np.ndarray,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
) -> list:
    """Loft circular beams whose per-station radius comes from the sized segment radii.

    Circular sections inset inward along the true OML normal — the FEA-faithful
    fallback when lens lofting is not desired/available.
    """
    _check_long_radii(long_radii, n_beams, n_levels)
    z_levels = default_z_levels(spec, n_levels)
    beams = []
    for b in range(n_beams):
        st_r = _segment_station_radii(long_radii, b, n_levels)
        with BuildPart() as bp:
            for k in range(n_levels):
                z = float(z_levels[k])
                P = beam_section_points(spec, z, n_beams)[b]
                n_out = oml_outward_normals(spec, z, n_beams)[b]
                c = P - st_r[k] * n_out          # inset inward by the radius
                with BuildSketch(Plane.XY.offset(z)):
                    with Locations((float(c[0]), float(c[1]))):
                        Circle(float(st_r[k]))
            loft(ruled=True)
        beams.append(bp.part)
    return beams


def build_sized_lens_beams(
    spec: WingSpec,
    long_radii: np.ndarray,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    n_arc: int = 24,
) -> list:
    """Loft inward-arc 'lens' beams sized to the Phase-C areas (π r²) per segment.

    Each station's lens has area = π·(station radius)², its flat edge on the OML and
    its arc bulging inward. Raises if build123d cannot loft the custom faces — the
    caller may fall back to ``build_sized_circular_beams``.
    """
    _check_long_radii(long_radii, n_beams, n_levels)
    z_levels = default_z_levels(spec, n_levels)
    beams = []
    for b in range(n_beams):
        st_r = _segment_station_radii(long_radii, b, n_levels)
        with BuildPart() as bp:
            for k in range(n_levels):
                z = float(z_levels[k])
                P = beam_section_points(spec, z, n_beams)[b]
                n_out = oml_outward_normals(spec, z, n_beams)[b]
                area = float(np.pi * st_r[k] ** 2)
                poly = lens_section_polyline(P, n_out, area, n_arc=n_arc)
                with BuildSketch(Plane.XY.offset(z)):
                    with BuildLine():
                        Polyline([(float(x), float(y)) for x, y in poly], close=True)
                    make_face()
            loft(ruled=True)
        beams.append(bp.part)
    return beams


def build_sized_ring_braces(model, brace_radius: float) -> list:
    """Build one circular-rod solid per transverse ring-brace element.

    Each brace element is a straight circular rod of radius ``brace_radius``
    connecting the two beam nodes given by ``model.brace_elements[i]``.
    Degenerate (zero-length) segments are silently skipped.

    Parameters
    ----------
    model:
        A ``BeamShellModel`` whose ``brace_elements`` attribute is a 1-D array
        of row-indices into ``model.beam_elements``.
    brace_radius:
        Circular cross-section radius (metres).

    Returns
    -------
    list of build123d Solid objects, one per brace element (skipping degenerate ones).
    """
    if model.brace_elements is None or len(model.brace_elements) == 0:
        return []
    solids = []
    for e in model.brace_elements:
        pair = model.beam_elements[int(e)]
        p0 = model.nodes[pair[0]]
        p1 = model.nodes[pair[1]]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length < 1e-12:
            continue
        z_dir = vec / length
        plane = Plane(origin=(float(p0[0]), float(p0[1]), float(p0[2])),
                      z_dir=(float(z_dir[0]), float(z_dir[1]), float(z_dir[2])))
        cyl = Solid.make_cylinder(float(brace_radius), length, plane)
        solids.append(cyl)
    return solids


def resample_segment_radii(
    long_radii: np.ndarray,
    *,
    n_beams: int,
    n_from: int,
    n_to: int,
) -> np.ndarray:
    """Resample per-segment longitudinal radii from ``n_from`` to ``n_to`` levels.

    Beam sizing runs at a coarse level count (SLSQP speed); the geometry is built
    at a finer one. For each beam, its ``n_from-1`` segment radii are interpolated
    (by normalized position along the beam) onto ``n_to-1`` segments. Returns a
    ``(n_beams*(n_to-1),)`` array in the same beam-major/level-minor layout.

    Interpolation is at segment centres, so target segments beyond the source
    centre range (the outermost ~half-segment at each end) are flat-clamped to the
    first/last source radius (``np.interp`` boundary behaviour). Negligible for
    densification (e.g. 8→60) but pronounced if ``n_from`` is tiny.
    """
    _check_long_radii(long_radii, n_beams, n_from)
    seg_from = n_from - 1
    seg_to = n_to - 1
    x_from = (np.arange(seg_from) + 0.5) / seg_from
    x_to = (np.arange(seg_to) + 0.5) / seg_to
    src = np.asarray(long_radii, dtype=float)
    out = np.empty(n_beams * seg_to)
    for b in range(n_beams):
        rb = src[b * seg_from:(b + 1) * seg_from]
        out[b * seg_to:(b + 1) * seg_to] = np.interp(x_to, x_from, rb)
    return out


