"""Shell-following form beams: section sampling, splines, and build123d geometry."""
from __future__ import annotations

from .cross_section import beam_section_points, resample_closed_polyline
from .layout import chord_symmetrize_weights, cross_section_stress_weights, stress_weighted_targets
from .splines import default_z_levels, fit_beam_splines, form_beam_grid, sample_spline
from .build import (
    build_assembly,
    build_form_beams,
    build_sized_circular_beams,
    build_sized_lens_beams,
    build_sized_ring_braces,
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
from .shell_model import BeamShellModel, build_beam_shell_model, model_with_tip_gusset, skin_datum_angles, skin_triangles, solve_beam_shell_model
from .tip_coupling import tip_clique_elements, solve_beam_shell_tip_coupled
from .shell_sizing import (
    BeamShellSizingConfig,
    BeamShellSizingResult,
    beam_lengths,
    beam_mass,
    beam_radius_groups,
    size_beam_shell,
    skin_areas,
    skin_band_areas,
    skin_band_map,
    skin_mass,
)
from .laminate_sizing import (
    LaminateSizingConfig, LaminateSizingResult, size_beam_shell_laminate,
    laminate_design_bounds, laminate_result_is_feasible,
    MultiStartResult, size_beam_shell_laminate_multistart,
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
from .stock_catalog import StockSelection, select_stock_sizes

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
    "build_sized_ring_braces",
    "build_skin_wrap",
    "resample_segment_radii",
    "BeamFrame",
    "FrameMetrics",
    "build_beam_frame",
    "project_panels_to_beam_nodes",
    "solve_beam_frame",
    "summarize_frame",
    "BeamShellModel",
    "build_beam_shell_model",
    "model_with_tip_gusset",
    "skin_datum_angles",
    "skin_triangles",
    "solve_beam_shell_model",
    "BeamShellSizingConfig",
    "BeamShellSizingResult",
    "beam_lengths",
    "beam_mass",
    "beam_radius_groups",
    "size_beam_shell",
    "skin_areas",
    "skin_band_map",
    "skin_band_areas",
    "skin_mass",
    "LaminateSizingConfig",
    "LaminateSizingResult",
    "size_beam_shell_laminate",
    "laminate_design_bounds",
    "laminate_result_is_feasible",
    "MultiStartResult",
    "size_beam_shell_laminate_multistart",
    "spline_surface_error",
    "lens_section_polyline",
    "oml_outward_normals",
    "SizingConfig",
    "SizingResult",
    "frame_mass",
    "n_longitudinal",
    "size_beams",
    "StockSelection",
    "select_stock_sizes",
    "chord_symmetrize_weights",
    "stress_weighted_targets",
    "cross_section_stress_weights",
    "tip_clique_elements",
    "solve_beam_shell_tip_coupled",
]
