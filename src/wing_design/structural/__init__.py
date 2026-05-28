"""Coarse structural sizing for the wingsail (Phase 3) + structural FEA.

Two FEA backends are wired in:

  * `solve_linear_elastic` (in `fea`) — volumetric linear-elastic tet FEA
    on the wing solid (Phase 4a). Reference implementation; useful for
    comparison and for the volumetric stress-line track (Phase 5a).
  * `solve_shell_elastic` (in `shell`) — DKT+CST flat-shell FEA on the
    OML surface (Phase 4b). Architecturally what the wingsail actually is
    (a stressed-skin structure); produces 2-D principal directions on each
    triangle that Phase 5e traces.
"""
from .beam import TubeSparSizing, size_tube_spar
from .fea import FEAResult, solve_linear_elastic
from .mesh import TetMesh, tet_mesh_wing
from .projection import project_panels_to_oml_tris
from .shell import (
    ShellFEAResult,
    ShellMesh,
    shell_mesh_from_tet_mesh,
    solve_shell_elastic,
)

__all__ = [
    "FEAResult",
    "ShellFEAResult",
    "ShellMesh",
    "TetMesh",
    "TubeSparSizing",
    "project_panels_to_oml_tris",
    "shell_mesh_from_tet_mesh",
    "size_tube_spar",
    "solve_linear_elastic",
    "solve_shell_elastic",
    "tet_mesh_wing",
]
