"""Phase 5e: trace stress lines on the OML shell.

Runs Phase 4b (shell FEA) → seeds high-σ_VM triangles → traces both
families' principal-direction streamlines forward and backward → exports
the resulting polylines as VTU.

These polylines are the candidate **spar-cap centerlines** for the
internal frame. On a symmetric wing under positive lift:

  * Family 0 (max tension σ_1) lights up on the **bottom skin**, with
    curves running roughly along the span — the natural placement for
    the bottom-chord spar cap.
  * Family 1 (max compression σ_2) lights up on the **top skin**,
    similarly span-aligned — the top-chord spar cap.

The two families together give us the matching upper / lower cap layout
that classical wing structure uses.

Phase 6e (queued) will combine these with chordwise ribs + shear webs
from `truss.interior_candidates` and feed everything to a coupled
shell + beam ALP.
"""
from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np

from wing_design import default_scenario
from wing_design.aero import build_airplane, run_case_lifting_line
from wing_design.structural import (
    shell_mesh_from_tet_mesh,
    solve_shell_elastic,
    tet_mesh_wing,
)
from wing_design.structural.projection import R_GEOM_FROM_AERO
from wing_design.truss import trace_surface_streamlines


def project_panel_forces_to_shell_tris(panels, shell_mesh, *, safety_factor):
    centers = panels.centers_xyz @ R_GEOM_FROM_AERO.T
    forces = panels.forces_xyz * safety_factor @ R_GEOM_FROM_AERO.T
    tri_centroids = shell_mesh.nodes[shell_mesh.triangles].mean(axis=1)
    p = shell_mesh.nodes[shell_mesh.triangles]
    tri_areas = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    loaded_idx = np.where(shell_mesh.loaded_tris)[0]
    out = np.zeros((shell_mesh.n_tris, 3))
    for k in range(panels.n_panels):
        z_lo = float(panels.centers_xyz[k, 1] - 0.5 * panels.spanwise_widths[k])
        z_hi = float(panels.centers_xyz[k, 1] + 0.5 * panels.spanwise_widths[k])
        band = (tri_centroids[loaded_idx, 2] >= z_lo) & (tri_centroids[loaded_idx, 2] < z_hi)
        band_area = float(tri_areas[loaded_idx][band].sum())
        if band_area <= 0.0:
            continue
        share = tri_areas[loaded_idx][band] / band_area
        out[loaded_idx[band]] += share[:, None] * forces[k]
    return out


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "exports"
    out_dir.mkdir(exist_ok=True)
    P = default_scenario()
    spec = P.geometry
    SKIN_THICKNESS_M = P.skin_sizing.t_baseline_m

    print("Meshing wing + extracting OML shell...")
    tet = tet_mesh_wing(spec, target_element_size=P.mesh.target_element_size_m)
    shell = shell_mesh_from_tet_mesh(tet)
    print(f"  shell: {shell.n_nodes} nodes, {shell.n_tris} tris")

    print("\nRunning Phase-2 aero (survival case) + Phase 4b shell FEA...")
    airplane = build_airplane(spec)
    case = next(c for c in P.load_cases if c.name == "survival")
    case_result = run_case_lifting_line(airplane, case, spanwise_resolution=P.aero.spanwise_resolution)
    tri_forces = project_panel_forces_to_shell_tris(
        case_result.panels, shell, safety_factor=case_result.case.safety_factor,
    )
    E = P.E_iso_Pa
    fea = solve_shell_elastic(shell, E=E, nu=P.nu_iso, thickness_m=SKIN_THICKNESS_M, tri_force_vectors=tri_forces)
    sigma_vm = fea.membrane_von_mises()
    print(f"  membrane σ_VM p99 = {np.percentile(sigma_vm, 99)/1e6:.2f} MPa, max = {sigma_vm.max()/1e6:.2f} MPa")

    print("\nTracing surface streamlines...")
    streamlines = trace_surface_streamlines(
        shell, fea,
        families=(0, 1),
        max_seeds=40,
        min_spacing_m=0.30,
        sigma_floor_fraction=0.10,
    )
    n_family = {0: 0, 1: 0}
    total_len = {0: 0.0, 1: 0.0}
    for sl in streamlines:
        n_family[sl.family] += 1
        total_len[sl.family] += sl.length_m()
    print(f"  family 0 (σ_1, tension lines on bottom skin): {n_family[0]} curves, {total_len[0]:.2f} m total")
    print(f"  family 1 (σ_2, compression lines on top skin): {n_family[1]} curves, {total_len[1]:.2f} m total")

    if not streamlines:
        print("\nNo streamlines traced — lower sigma_floor_fraction or refine the mesh.")
        return

    # Assemble VTU: each streamline becomes a polyline (cell type "line", per-segment)
    point_blocks = []
    segs: list[np.ndarray] = []
    seg_family: list[int] = []
    seg_sigma_MPa: list[float] = []
    seg_streamline_id: list[int] = []
    cursor = 0
    for sl_idx, sl in enumerate(streamlines):
        point_blocks.append(sl.points)
        n_pts = sl.points.shape[0]
        for i in range(n_pts - 1):
            segs.append([cursor + i, cursor + i + 1])
            seg_family.append(int(sl.family))
            seg_sigma_MPa.append(float(sl.sigma_along[i]) / 1.0e6)
            seg_streamline_id.append(int(sl_idx))
        cursor += n_pts

    out_path = out_dir / "shell_stress_lines.vtu"
    meshio.write_points_cells(
        out_path,
        points=np.vstack(point_blocks),
        cells=[("line", np.asarray(segs, dtype=np.int64))],
        cell_data={
            "family": [np.asarray(seg_family, dtype=np.int32)],
            "sigma_along_MPa": [np.asarray(seg_sigma_MPa)],
            "streamline_id": [np.asarray(seg_streamline_id, dtype=np.int32)],
        },
    )
    print(f"\nWrote {out_path}")
    print("In ParaView: open this alongside `shell_fea.vtu`. Color the streamlines")
    print("by `sigma_along_MPa` to see the load along each candidate spar cap;")
    print("color by `family` (0 = bottom-skin tension, 1 = top-skin compression).")


if __name__ == "__main__":
    main()
