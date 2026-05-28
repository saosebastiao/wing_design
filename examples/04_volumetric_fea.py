"""Phase-4 spike: volumetric FEA of the wingsail under the LL load envelope.

Runs the full pipeline: build wing OML → tet-mesh in gmsh → LL aero envelope →
project per-panel loads to OML facets → linear elastic FEA per case → export
displacements and σ(x) as VTU.

The mesh is intentionally coarse (~3 cm elements) for spike speed. Replace with
a refined size once the result looks sensible.
"""
from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np

from wing_design import default_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.structural import (
    project_panels_to_oml_tris,
    solve_linear_elastic,
    tet_mesh_wing,
)


def main() -> None:
    P = default_scenario()
    spec = P.geometry
    out_dir = Path(__file__).resolve().parent.parent / "exports"
    out_dir.mkdir(exist_ok=True)

    print("Meshing wing OML...")
    mesh = tet_mesh_wing(spec, target_element_size=P.mesh.target_element_size_m)
    print(f"  nodes={mesh.n_nodes}, tets={mesh.n_tets}")
    print(f"  surface tris={mesh.surface_tris.shape[0]}, OML tris={mesh.oml_tris.shape[0]}")
    print(f"  Dirichlet (spar-base) nodes={mesh.dirichlet_nodes.shape[0]}")
    bb_lo, bb_hi = mesh.bounding_box()
    print(f"  bbox: x[{bb_lo[0]:.3f}, {bb_hi[0]:.3f}]  y[{bb_lo[1]:.3f}, {bb_hi[1]:.3f}]  z[{bb_lo[2]:.3f}, {bb_hi[2]:.3f}]")

    print("\nRunning aero envelope (LiftingLine)...")
    airplane = build_airplane(spec)
    envelope = sweep_envelope(
        airplane, P.load_cases,
        method="lifting_line", spanwise_resolution=P.aero.spanwise_resolution,
    )

    E = P.E_iso_Pa
    nu = P.nu_iso
    sigma_allow = P.sigma_allow_Pa
    print(f"\nMaterial (isotropic-equivalent): E = {E/1e9:.1f} GPa, ν = {nu}")
    print(f"Allowable σ = {sigma_allow/1e6:.0f} MPa (Xt / SF={P.material_iso.sigma_allow_safety_factor})")

    print("\nPer-case linear FEA:")
    print(f"  {'case':<14} {'|F|_geom':>10} {'σ_max':>10} {'σ_p99':>10} {'u_max':>10} {'u_tip':>10}")
    print(f"  {'':14} {'N':>10} {'MPa':>10} {'MPa':>10} {'µm':>10} {'µm':>10}")

    cell_data = {"von_mises": [], "case": []}
    point_data: dict[str, list[np.ndarray]] = {"displacement": [], "nodal_vm": []}

    for case_result in envelope:
        case = case_result.case
        if case_result.panels is None or abs(case_result.factored_normal_force_N) < 1.0:
            continue

        tri_forces = project_panels_to_oml_tris(
            mesh, case_result.panels, span_m=case_result.span_m, safety_factor=case.safety_factor,
        )
        F_applied = float(np.linalg.norm(tri_forces.sum(axis=0)))

        result = solve_linear_elastic(mesh, E=E, nu=nu, tri_force_vectors=tri_forces)
        vm_tet = result.von_mises_per_tet
        sigma_max = float(vm_tet.max())
        sigma_p99 = float(np.percentile(vm_tet, 99))
        u_norms = np.linalg.norm(result.displacements, axis=1)
        u_max = float(u_norms.max())
        tip_mask = mesh.nodes[:, 2] >= spec.span - 0.05
        u_tip = float(u_norms[tip_mask].max()) if tip_mask.any() else 0.0

        print(
            f"  {case.name:<14} {F_applied:>10.1f} "
            f"{sigma_max/1e6:>10.3f} {sigma_p99/1e6:>10.3f} "
            f"{u_max*1e6:>10.1f} {u_tip*1e6:>10.1f}"
        )

        # Write a per-case VTU for inspection
        out_path = out_dir / f"fea_{case.name}.vtu"
        meshio.write_points_cells(
            out_path,
            points=mesh.nodes,
            cells=[("tetra", mesh.tets)],
            point_data={
                "displacement": result.displacements,
                "nodal_von_mises": result.nodal_von_mises,
            },
            cell_data={"von_mises": [vm_tet]},
        )

    print(f"\nWrote per-case VTU files to {out_dir}/fea_<case>.vtu")


if __name__ == "__main__":
    main()
