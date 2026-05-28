"""Tet-mesh the wingsail OML for volumetric FEA (Phase 4).

The pipeline: `build_wing_solid` → STEP export → `gmsh.merge` → 3D tet mesh.
We work via STEP rather than re-implementing the wing geometry in gmsh's OCC
kernel so the geometry has exactly one source of truth (`geometry.wing`).

The returned `TetMesh` carries:

  * `nodes`: (N, 3) coordinates in metres, in the wing geometry frame
    (chord +X, span +Z, airfoil normal +Y — same frame as `build_wing_solid`).
  * `tets`: (M, 4) integer node indices for linear tets.
  * `surface_tris`: (K, 3) all OML boundary triangles.
  * `oml_tris`: subset of `surface_tris` lying on the aero-loaded wing skin
    (z ≥ aero_z_min). These are what panel tractions get projected onto.
  * `dirichlet_nodes`: indices of nodes on the spar-base disk, clamped in FEA.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ..geometry.wing import WingSpec, build_wing_solid


@dataclass(frozen=True)
class TetMesh:
    nodes: np.ndarray            # (N, 3) float
    tets: np.ndarray             # (M, 4) int
    surface_tris: np.ndarray     # (K, 3) int, all boundary facets
    oml_tris: np.ndarray         # (K_oml, 3) int, aero-loaded skin only
    dirichlet_nodes: np.ndarray  # (P,) int, indices to clamp

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_tets(self) -> int:
        return int(self.tets.shape[0])

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.nodes.min(axis=0), self.nodes.max(axis=0)


def tet_mesh_wing(
    spec: WingSpec = WingSpec(),
    *,
    target_element_size: float | None = None,
    min_element_size: float | None = None,
    aero_z_min: float = 0.0,
    spar_base_tol: float = 1.0e-3,
    verbose: bool = False,
) -> TetMesh:
    """Tet-mesh the wing OML produced by `build_wing_solid(spec)`.

    `target_element_size` defaults to ~1/4 of the spar diameter, which is the
    coarsest length scale we need to resolve. `aero_z_min` is the spanwise
    cutoff below which surface facets are not considered aero-loaded (the
    fairing + spar are below this in the geometry frame).
    """
    if target_element_size is None:
        target_element_size = 0.25 * spec.spar_diameter
    if min_element_size is None:
        min_element_size = 0.2 * target_element_size

    part = build_wing_solid(spec)
    spar_base_z = -(spec.transition_length + spec.spar_length)

    with TemporaryDirectory() as tmpdir:
        from build123d import export_step
        step_path = Path(tmpdir) / "wing.step"
        export_step(part, str(step_path))
        return _tet_mesh_from_step(
            step_path,
            target_element_size=target_element_size,
            min_element_size=min_element_size,
            aero_z_min=aero_z_min,
            spar_base_z=spar_base_z,
            spar_base_tol=spar_base_tol,
            verbose=verbose,
        )


def _tet_mesh_from_step(
    step_path: Path,
    *,
    target_element_size: float,
    min_element_size: float,
    aero_z_min: float,
    spar_base_z: float,
    spar_base_tol: float,
    verbose: bool,
) -> TetMesh:
    import gmsh

    gmsh.initialize()
    try:
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("wingsail")
        gmsh.merge(str(step_path))

        gmsh.option.setNumber("Mesh.MeshSizeMax", target_element_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_element_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay

        gmsh.model.mesh.generate(3)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        # gmsh tags are 1-based and may not be contiguous; build a remap.
        tag_to_idx = np.full(int(np.max(node_tags)) + 1, -1, dtype=np.int64)
        tag_to_idx[np.asarray(node_tags, dtype=np.int64)] = np.arange(len(node_tags))

        tets = _extract_elements(gmsh, dim=3, element_type=4, n_nodes=4, tag_to_idx=tag_to_idx)
        surface_tris = _extract_elements(gmsh, dim=2, element_type=2, n_nodes=3, tag_to_idx=tag_to_idx)

        # Aero-loaded OML: facets whose centroid z >= aero_z_min
        tri_centroids = coords[surface_tris].mean(axis=1)
        oml_mask = tri_centroids[:, 2] >= aero_z_min
        oml_tris = surface_tris[oml_mask]

        # Dirichlet region: nodes on the bottom spar-base face
        dirichlet_nodes = np.where(np.abs(coords[:, 2] - spar_base_z) < spar_base_tol)[0]

        return TetMesh(
            nodes=coords,
            tets=tets,
            surface_tris=surface_tris,
            oml_tris=oml_tris,
            dirichlet_nodes=dirichlet_nodes,
        )
    finally:
        gmsh.finalize()


def _extract_elements(gmsh, *, dim: int, element_type: int, n_nodes: int, tag_to_idx: np.ndarray) -> np.ndarray:
    """Pull every element of the given gmsh type/dim, remapped to 0-based contiguous node indices."""
    types, _, node_tag_lists = gmsh.model.mesh.getElements(dim=dim)
    for t, ntags in zip(types, node_tag_lists):
        if int(t) == element_type:
            raw = np.asarray(ntags, dtype=np.int64).reshape(-1, n_nodes)
            return tag_to_idx[raw]
    return np.empty((0, n_nodes), dtype=np.int64)
