"""Phase-5b spike: SO(3) sign-aligned frame field + Poisson parametrization.

Compares against the Phase-5a output side-by-side on the survival case:
  1. Build mesh + LL + FEA → nodal σ(x) and raw principal frame.
  2. BFS sign-align the eigenvector field so adjacent nodes agree on direction.
  3. Solve the gradient-fitting Poisson L φ = b for φ : Ω → ℝ³ with ∇φ ≈ R.
  4. Re-trace streamlines on the smoothed frame → cleaner curves than 5a.
  5. Export:
       * exports/frame_phi.vtu       — the wing volume tagged with φ_k as
         nodal scalars and the smoothed eigenvectors as nodal vectors.
       * exports/frame_streamlines.vtu — curves traced through the smoothed
         field (compare against exports/stress_lines.vtu from 5a).

The integer-isocurve extraction step that follows the parametrization in the
Arora pipeline is queued for Phase 5c; this spike validates the smoothed
field + parametrization come out consistent on real wing geometry.
"""
from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np

from wing_design import default_scenario
from wing_design.aero import build_airplane, run_case_lifting_line
from wing_design.aero.loads import AeroResult
from wing_design.structural import (
    FEAResult,
    project_panels_to_oml_tris,
    solve_linear_elastic,
    tet_mesh_wing,
)
from wing_design.truss import (
    StreamlineIntegrator,
    align_signs_bfs,
    extract_stress_lines,
    fit_parametrization,
    gradient_fit_residual,
    principal_frame_from_voigt,
    seed_volumetric_by_stress,
)


GOVERNING_CASE = "survival"


def _solve(mesh, case_result: AeroResult, E: float, nu: float) -> FEAResult:
    tri_forces = project_panels_to_oml_tris(
        mesh, case_result.panels, span_m=case_result.span_m,
        safety_factor=case_result.case.safety_factor,
    )
    return solve_linear_elastic(mesh, E=E, nu=nu, tri_force_vectors=tri_forces)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "exports"
    out_dir.mkdir(exist_ok=True)
    P = default_scenario()
    spec = P.geometry

    print("Meshing wing OML...")
    mesh = tet_mesh_wing(spec, target_element_size=P.mesh.target_element_size_m)
    print(f"  nodes={mesh.n_nodes}, tets={mesh.n_tets}")

    print(f"\nFEA on {GOVERNING_CASE}...")
    airplane = build_airplane(spec)
    case = next(c for c in P.load_cases if c.name == GOVERNING_CASE)
    case_result = run_case_lifting_line(airplane, case, spanwise_resolution=P.aero.spanwise_resolution)
    E = P.E_iso_Pa
    nu = P.nu_iso
    fea = _solve(mesh, case_result, E, nu)
    print(f"  σ_VM p99 = {np.percentile(fea.nodal_von_mises, 99)/1e6:.3f} MPa")

    raw_frame = principal_frame_from_voigt(fea.nodal_stress)
    print("\nSign-aligning the eigenvector field (BFS from highest-σ node)...")
    smooth_frame = align_signs_bfs(raw_frame, mesh.tets)
    # Quality check: count nodes where signs differ between raw and smoothed
    flipped = 0
    for k in range(3):
        d = np.einsum("ni,ni->n", raw_frame.eigenvectors[:, :, k], smooth_frame.eigenvectors[:, :, k])
        flipped += int((d < 0).sum())
    print(f"  flipped signs across families: {flipped} of {3 * mesh.n_nodes} (raw was inconsistent on {100*flipped/(3*mesh.n_nodes):.1f}% of slots)")

    print("\nFitting Poisson parametrization L φ = b ...")
    phi = fit_parametrization(mesh, smooth_frame)
    residual = gradient_fit_residual(mesh, smooth_frame, phi)
    print(f"  per-tet ‖∇φ_k − v_k‖ mean: {residual.mean(axis=0)} (1.0 = totally misaligned, 0 = perfect)")

    # Export the volume with φ and smoothed eigenvectors attached
    phi_path = out_dir / "frame_phi.vtu"
    meshio.write_points_cells(
        phi_path,
        points=mesh.nodes,
        cells=[("tetra", mesh.tets)],
        point_data={
            "phi_1": phi[:, 0],
            "phi_2": phi[:, 1],
            "phi_3": phi[:, 2],
            "e1_smooth": smooth_frame.eigenvector(0),
            "e2_smooth": smooth_frame.eigenvector(1),
            "e3_smooth": smooth_frame.eigenvector(2),
            "sigma_1_MPa": smooth_frame.eigenvalue(0) / 1e6,
            "sigma_3_MPa": smooth_frame.eigenvalue(2) / 1e6,
        },
    )
    print(f"Wrote {phi_path}")

    print("\nRetracing streamlines through the smoothed frame field...")
    integrator = StreamlineIntegrator(mesh, smooth_frame)
    sigma_p99 = float(np.percentile(fea.nodal_von_mises, 99))
    min_eig = 0.05 * sigma_p99

    all_points: list[np.ndarray] = []
    all_segs: list[np.ndarray] = []
    seg_family: list[int] = []
    seg_sigma: list[float] = []

    for k in (0, 1, 2):
        seeds = seed_volumetric_by_stress(
            mesh, eigenvalue_magnitude=np.abs(smooth_frame.eigenvalue(k)), max_seeds=200,
        )
        fam = extract_stress_lines(
            mesh, smooth_frame, family=k, seeds=seeds, min_abs_eigenvalue=min_eig,
        )
        print(f"  family {k}: {fam.n_lines:3d} smooth curves, total {fam.total_length_m:.2f} m")
        for poly in fam.polylines:
            mids = 0.5 * (poly[:-1] + poly[1:])
            base = sum(p.shape[0] for p in all_points)
            all_points.append(poly)
            segs = np.column_stack([
                np.arange(base, base + poly.shape[0] - 1),
                np.arange(base + 1, base + poly.shape[0]),
            ])
            all_segs.append(segs)
            seg_family.extend([k] * segs.shape[0])
            for m in mids:
                _, lam = integrator.eigenvector_at(m, k)
                seg_sigma.append(float(lam))

    if all_segs:
        stream_path = out_dir / "frame_streamlines.vtu"
        meshio.write_points_cells(
            stream_path,
            points=np.vstack(all_points),
            cells=[("line", np.vstack(all_segs))],
            cell_data={
                "family": [np.asarray(seg_family, dtype=np.int32)],
                "sigma_along_MPa": [np.asarray(seg_sigma) / 1e6],
            },
        )
        print(f"Wrote {stream_path}")


if __name__ == "__main__":
    main()
