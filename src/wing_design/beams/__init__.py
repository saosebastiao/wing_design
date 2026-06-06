"""Shell-following form beams: section sampling, splines, and build123d geometry."""
from __future__ import annotations

from .cross_section import beam_section_points, resample_closed_polyline
from .splines import default_z_levels, fit_beam_splines, form_beam_grid, sample_spline
from .build import (
    build_assembly,
    build_form_beams,
    build_sized_circular_beams,
    build_sized_lens_beams,
    build_skin_wrap,
    resample_segment_radii,
)
from .fea_model import (
    BeamFrame,
    FrameMetrics,
    build_beam_frame,
    project_panels_to_beam_nodes,
    solve_beam_frame,
    summarize_frame,
)
from .fidelity import spline_surface_error
from .sections import lens_section_polyline, oml_outward_normals
from .sizing import (
    SizingConfig,
    SizingResult,
    frame_mass,
    n_longitudinal,
    size_beams,
)

__all__ = [
    "beam_section_points",
    "resample_closed_polyline",
    "default_z_levels",
    "form_beam_grid",
    "fit_beam_splines",
    "sample_spline",
    "build_assembly",
    "build_form_beams",
    "build_sized_circular_beams",
    "build_sized_lens_beams",
    "build_skin_wrap",
    "resample_segment_radii",
    "BeamFrame",
    "FrameMetrics",
    "build_beam_frame",
    "project_panels_to_beam_nodes",
    "solve_beam_frame",
    "summarize_frame",
    "spline_surface_error",
    "lens_section_polyline",
    "oml_outward_normals",
    "SizingConfig",
    "SizingResult",
    "frame_mass",
    "n_longitudinal",
    "size_beams",
]
