"""Stress-aligned principal-direction extraction on the wing σ field.

Phase 5a — volumetric streamlines through the σ tensor's principal-direction
field on a tet mesh (`streamline`, `extract`). Produces the σ_1 / σ_2 / σ_3
trajectories that are the classical Michell visualization.

Phase 5b — sign-aligned frame field + Poisson parametrization φ(x)
(`frame_field`, `parametrization`). BFS-aligns per-tet eigenvector signs so
the principal axes are continuous across the mesh; solves a Poisson system
for a smooth scalar φ that downstream isocurve / lattice extraction would
read.

Phase 5e — surface streamlines on the OML shell triangulation
(`surface_streamlines`). The shell-track version of 5a.
"""
from .extract import (
    StreamlineFamily,
    extract_stress_lines,
    mirror_family_across_chord_plane,
    seed_oml_surface,
    seed_volumetric_by_stress,
)
from .frame_field import (
    PrincipalFrame,
    align_signs_bfs,
    principal_frame_from_voigt,
)
from .parametrization import fit_parametrization, gradient_fit_residual
from .streamline import StreamlineIntegrator, trace_streamline
from .surface_streamlines import (
    SurfaceStreamline,
    seed_high_sigma_triangles,
    trace_surface_streamline,
    trace_surface_streamlines,
)

__all__ = [
    "PrincipalFrame",
    "StreamlineFamily",
    "StreamlineIntegrator",
    "SurfaceStreamline",
    "align_signs_bfs",
    "extract_stress_lines",
    "mirror_family_across_chord_plane",
    "principal_frame_from_voigt",
    "seed_oml_surface",
    "seed_volumetric_by_stress",
    "fit_parametrization",
    "gradient_fit_residual",
    "seed_high_sigma_triangles",
    "trace_streamline",
    "trace_surface_streamline",
    "trace_surface_streamlines",
]
