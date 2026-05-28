"""Phase-4 architectural refactor: wing as a thin shell (no internal frame yet).

Replaces the volumetric tet FEA with a DKT+CST shell on the OML surface.
The σ field is now 2-D **on the wing skin** instead of 3-D inside a solid
block — exactly the physical model for the stressed-skin space-frame
wingsail. The output includes:

  * shell membrane stress (σ_xx, σ_yy, σ_xy) per triangle in its local
    frame, plus von-Mises and 2-D principal stresses;
  * 3-D unit vectors for the two principal stress directions at every
    triangle — Phase 5 traces stress lines along these by streamline
    integration on the surface;
  * shell bending moments (M_xx, M_yy, M_xy) for visualizing skin
    pressure-bending separately from wing-bending membrane stress.

ParaView visualization tips:
  * Color by `sigma_VM_membrane_MPa` to see the load path on the skin.
  * Use the `Glyph` filter with `principal_dir_max` (and a small scale)
    to see the max-tension direction field — these tell you where to
    run the **top/bottom chord caps** of the internal frame.
  * Use `Stream Tracer` seeded on the surface and following
    `principal_dir_max` to render the actual stress lines.
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


def project_panel_forces_to_shell_tris(
    panels, shell_mesh, *, safety_factor: float,
) -> np.ndarray:
    """Spanwise-band, area-weighted projection of LL panel forces to shell triangles.

    Same idea as `structural.projection.project_panels_to_oml_tris` for the
    tet mesh, just adapted to the shell mesh's flagged `loaded_tris`.
    Returns a (n_tris, 3) array in the global geometry frame.
    """
    centers_geom = panels.centers_xyz @ R_GEOM_FROM_AERO.T
    forces_geom = panels.forces_xyz * safety_factor @ R_GEOM_FROM_AERO.T

    tri_centroids = shell_mesh.nodes[shell_mesh.triangles].mean(axis=1)
    tri_areas = _tri_areas(shell_mesh)
    loaded_idx = np.where(shell_mesh.loaded_tris)[0]
    loaded_z = tri_centroids[loaded_idx, 2]
    loaded_areas = tri_areas[loaded_idx]

    panel_z = centers_geom[:, 2]
    panel_w = panels.spanwise_widths
    out = np.zeros((shell_mesh.n_tris, 3))
    for k in range(panels.n_panels):
        z_lo = float(panel_z[k] - 0.5 * panel_w[k])
        z_hi = float(panel_z[k] + 0.5 * panel_w[k])
        band = (loaded_z >= z_lo) & (loaded_z < z_hi)
        band_area = float(loaded_areas[band].sum())
        if band_area <= 0.0:
            continue
        share = loaded_areas[band] / band_area
        out[loaded_idx[band]] += share[:, None] * forces_geom[k]
    return out


def _tri_areas(shell):
    p = shell.nodes[shell.triangles]
    return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "exports"
    out_dir.mkdir(exist_ok=True)
    P = default_scenario()
    spec = P.geometry
    SKIN_THICKNESS_M = P.skin_sizing.t_baseline_m

    print("Meshing wing volume + extracting OML shell...")
    tet = tet_mesh_wing(spec, target_element_size=P.mesh.target_element_size_m)
    shell = shell_mesh_from_tet_mesh(tet)
    print(f"  shell: {shell.n_nodes} nodes, {shell.n_tris} tris, "
          f"{shell.dirichlet_nodes.shape[0]} clamped, {int(shell.loaded_tris.sum())} loaded")

    print("\nRunning Phase-2 aero (survival case)...")
    airplane = build_airplane(spec)
    case = next(c for c in P.load_cases if c.name == "survival")
    case_result = run_case_lifting_line(airplane, case, spanwise_resolution=P.aero.spanwise_resolution)
    tri_forces = project_panel_forces_to_shell_tris(
        case_result.panels, shell, safety_factor=case_result.case.safety_factor,
    )
    F_total = tri_forces.sum(axis=0)
    print(f"  total applied force on shell: ({F_total[0]:+.1f}, {F_total[1]:+.1f}, {F_total[2]:+.1f}) N")

    E = P.E_iso_Pa
    print(f"\nShell FEA (DKT+CST, t={SKIN_THICKNESS_M*1000:.0f} mm, E={E/1e9:.1f} GPa)...")
    res = solve_shell_elastic(
        shell, E=E, nu=P.nu_iso, thickness_m=SKIN_THICKNESS_M, tri_force_vectors=tri_forces,
    )
    u_mag = np.linalg.norm(res.displacements[:, :3], axis=1)
    tip_mask = shell.nodes[:, 2] > 4.5
    sigma_vm = res.membrane_von_mises()
    print(f"  tip displacement (z > 4.5 m): max = {u_mag[tip_mask].max() * 1000:.2f} mm")
    print(f"  membrane σ_VM: p50 = {np.percentile(sigma_vm, 50) / 1e6:.2f}, "
          f"p99 = {np.percentile(sigma_vm, 99) / 1e6:.2f}, max = {sigma_vm.max() / 1e6:.2f} MPa")

    # Principal stress + direction field
    sigma_p, _ = res.membrane_principal_2d()       # (M, 2): σ_1, σ_2
    dirs = res.membrane_principal_dirs_3d()        # (M, 2, 3): unit vectors

    # Bending Von-Mises (per unit thickness — for visualizing local skin bending)
    M = res.bending_moment
    M_vm = np.sqrt(M[:, 0] ** 2 - M[:, 0] * M[:, 1] + M[:, 1] ** 2 + 3.0 * M[:, 2] ** 2)
    # Equivalent fiber stress from bending: σ = 6 M / t²
    fiber_stress_from_bending = 6.0 * M_vm / (SKIN_THICKNESS_M ** 2)

    # Top-vs-bottom skin σ_zz sanity check
    centroids = shell.nodes[shell.triangles].mean(axis=1)
    # σ_zz in global frame: project membrane σ tensor through the element frame
    # σ_global = R @ (σ_local_3x3) @ R^T, take [2,2].
    sxx = res.membrane_stress[:, 0]
    syy = res.membrane_stress[:, 1]
    sxy = res.membrane_stress[:, 2]
    sigma_local_3 = np.zeros((shell.n_tris, 3, 3))
    sigma_local_3[:, 0, 0] = sxx
    sigma_local_3[:, 1, 1] = syy
    sigma_local_3[:, 0, 1] = sxy
    sigma_local_3[:, 1, 0] = sxy
    R = res.element_local_frames
    sigma_global = np.einsum("eij,ejk,elk->eil", R, sigma_local_3, R)    # R σ R^T
    sigma_zz = sigma_global[:, 2, 2]

    top_mask = (centroids[:, 1] > 0.05) & (centroids[:, 2] > 0.5) & shell.loaded_tris
    bot_mask = (centroids[:, 1] < -0.05) & (centroids[:, 2] > 0.5) & shell.loaded_tris
    print(f"\nMichell stress check (skin tris z > 0.5 m):")
    print(f"  TOP    σ_zz mean = {sigma_zz[top_mask].mean() / 1e6:+.2f} MPa")
    print(f"  BOTTOM σ_zz mean = {sigma_zz[bot_mask].mean() / 1e6:+.2f} MPa")

    # Export
    out_path = out_dir / "shell_fea.vtu"
    meshio.write_points_cells(
        out_path,
        points=shell.nodes,
        cells=[("triangle", shell.triangles)],
        point_data={
            "displacement_mm": res.displacements[:, :3] * 1000.0,
        },
        cell_data={
            "sigma_VM_membrane_MPa": [sigma_vm / 1e6],
            "sigma_zz_membrane_MPa": [sigma_zz / 1e6],
            "sigma_1_MPa": [sigma_p[:, 0] / 1e6],
            "sigma_2_MPa": [sigma_p[:, 1] / 1e6],
            "principal_dir_max": [dirs[:, 0, :]],         # 3-vec per tri
            "principal_dir_min": [dirs[:, 1, :]],
            "skin_bending_fiber_MPa": [fiber_stress_from_bending / 1e6],
            "loaded_tri": [shell.loaded_tris.astype(np.int32)],
        },
    )
    print(f"\nWrote {out_path}")
    print("In ParaView: color by `sigma_VM_membrane_MPa`, Glyph by `principal_dir_max`,")
    print("then Stream Tracer along `principal_dir_max` to see stress lines on the OML.")


if __name__ == "__main__":
    main()
