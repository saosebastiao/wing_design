"""Coarse structural sizing for the wingsail (Phase 3) + structural FEA.

Several FEA backends are wired in:

  * `solve_linear_elastic` (in `fea`) — volumetric linear-elastic tet FEA
    on the wing solid (Phase 4a). Reference implementation; useful for
    comparison and for the volumetric stress-line track (Phase 5a).
  * `solve_shell_elastic` (in `shell`) — DKT+CST flat-shell FEA on the
    OML surface (Phase 4b). Architecturally what the wingsail actually is
    (a stressed-skin structure); produces 2-D principal directions on each
    triangle that Phase 5e traces.
  * `solve_frame` (in `frame`) — 3D Euler–Bernoulli frame FEA for the
    form-beam space frame (Phase B).
  * `solve_beam_shell` (in `beam_shell`) — combined beam + DKT+CST shell
    FEA: form beams plus a load-bearing skin, assembled into one system
    (Phase E.1).
"""
from .beam import TubeSparSizing, size_tube_spar
from .buckling import beam_euler_utilization, panel_buckling_utilization
from .beam_shell import solve_beam_shell, solve_beam_shell_laminate
from .fea import FEAResult, solve_linear_elastic
from .frame import BeamSection, FrameResult, solve_frame, von_mises_per_element
from .mesh import TetMesh, tet_mesh_wing
from .projection import project_panels_to_oml_tris
from .shell import (
    ShellFEAResult,
    ShellMesh,
    membrane_von_mises,
    recover_membrane_stress,
    recover_membrane_stress_C,
    shell_mesh_from_tet_mesh,
    solve_shell_elastic,
    tri_element_stiffness,
    tri_element_stiffness_laminate,
)

__all__ = [
    "BeamSection",
    "beam_euler_utilization",
    "panel_buckling_utilization",
    "FEAResult",
    "FrameResult",
    "ShellFEAResult",
    "ShellMesh",
    "TetMesh",
    "TubeSparSizing",
    "membrane_von_mises",
    "project_panels_to_oml_tris",
    "recover_membrane_stress",
    "recover_membrane_stress_C",
    "shell_mesh_from_tet_mesh",
    "size_tube_spar",
    "solve_beam_shell",
    "solve_beam_shell_laminate",
    "solve_frame",
    "solve_linear_elastic",
    "solve_shell_elastic",
    "tet_mesh_wing",
    "tri_element_stiffness",
    "tri_element_stiffness_laminate",
    "von_mises_per_element",
]
