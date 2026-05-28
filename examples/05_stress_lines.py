"""Phase-5a spike: stress-line tracing for the full load envelope.

For each loaded design case we:
  1. Run LiftingLine + project loads to OML facets.
  2. Solve linear-elastic FEA → nodal σ(x).
  3. Fit per-node principal frame.
  4. Volumetric stress-weighted Poisson-disk seeding per family, so seeds land
     where each family actually carries load (not just on the OML surface).
  5. Trace 3 streamline families through the volume.
  6. Mirror every polyline across y=0 to cover the wingsail's other tack
     (symmetric airfoil → (AWS, −α) is the chord-plane reflection of (AWS, +α)).

The union is exported to exports/stress_lines.vtu with one scalar per
segment: the principal eigenvalue at the segment midpoint. Color the lines by
`sigma_along` in ParaView to see where each candidate beam is carrying load
vs. just noise.
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
    extract_stress_lines,
    mirror_family_across_chord_plane,
    principal_frame_from_voigt,
    seed_volumetric_by_stress,
)


FAMILY_NAMES = {0: "tension", 1: "shear", 2: "compression"}


def _solve_case(mesh, case_result: AeroResult, E: float, nu: float) -> FEAResult:
    tri_forces = project_panels_to_oml_tris(
        mesh, case_result.panels, span_m=case_result.span_m,
        safety_factor=case_result.case.safety_factor,
    )
    return solve_linear_elastic(mesh, E=E, nu=nu, tri_force_vectors=tri_forces)


def _segment_sigma_along(polyline: np.ndarray, integrator: StreamlineIntegrator, family: int) -> np.ndarray:
    """Signed eigenvalue at every segment midpoint of `polyline` (n_pts → n_pts-1 values)."""
    mids = 0.5 * (polyline[:-1] + polyline[1:])
    out = np.empty(mids.shape[0], dtype=np.float64)
    for i, p in enumerate(mids):
        _, lam = integrator.eigenvector_at(p, family)
        out[i] = lam
    return out


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "exports"
    out_dir.mkdir(exist_ok=True)
    P = default_scenario()
    spec = P.geometry

    print("Meshing wing OML...")
    mesh = tet_mesh_wing(spec, target_element_size=P.mesh.target_element_size_m)
    print(f"  nodes={mesh.n_nodes}, tets={mesh.n_tets}, OML tris={mesh.oml_tris.shape[0]}")

    airplane = build_airplane(spec)
    E = P.E_iso_Pa
    nu = P.nu_iso

    all_points: list[np.ndarray] = []
    all_segments: list[np.ndarray] = []
    seg_case_idx: list[int] = []
    seg_family: list[int] = []
    seg_mirror: list[int] = []
    seg_sigma_along: list[float] = []
    case_names: list[str] = []

    def add_family(case_name: str, fam, integrator: StreamlineIntegrator, mirror: bool) -> None:
        cidx = case_names.index(case_name)
        for poly in fam.polylines:
            sigmas = _segment_sigma_along(poly, integrator, fam.family)
            base = sum(p.shape[0] for p in all_points)
            all_points.append(poly)
            segs = np.column_stack([
                np.arange(base, base + poly.shape[0] - 1),
                np.arange(base + 1, base + poly.shape[0]),
            ])
            all_segments.append(segs)
            seg_case_idx.extend([cidx] * segs.shape[0])
            seg_family.extend([fam.family] * segs.shape[0])
            seg_mirror.extend([int(mirror)] * segs.shape[0])
            seg_sigma_along.extend(sigmas.tolist())

    print(f"\n{'case':<14} {'σ_p99':>10} {'tension':>10} {'shear':>10} {'compr.':>10} {'+ mirror':>12}")
    print(f"{'':14} {'MPa':>10} {'lines':>10} {'lines':>10} {'lines':>10} {'lines':>12}")
    print("-" * 80)

    for case in P.load_cases:
        case_result = run_case_lifting_line(airplane, case, spanwise_resolution=16)
        if abs(case_result.factored_normal_force_N) < 1.0:
            print(f"{case.name:<14} {'skip — zero-force':>50}")
            continue

        fea = _solve_case(mesh, case_result, E, nu)
        sigma_p99 = float(np.percentile(fea.nodal_von_mises, 99))
        frame = principal_frame_from_voigt(fea.nodal_stress)
        integrator = StreamlineIntegrator(mesh, frame)
        case_names.append(case.name)
        min_eig = 0.05 * sigma_p99

        counts = []
        for k in (0, 1, 2):
            # Volumetric seeds biased toward this family's high-stress regions
            seeds = seed_volumetric_by_stress(
                mesh,
                eigenvalue_magnitude=np.abs(frame.eigenvalue(k)),
                max_seeds=200,
            )
            fam = extract_stress_lines(
                mesh, frame, family=k, seeds=seeds, min_abs_eigenvalue=min_eig,
            )
            add_family(case.name, fam, integrator, mirror=False)
            add_family(case.name, mirror_family_across_chord_plane(fam), integrator, mirror=True)
            counts.append(fam.n_lines)
        mirrored = sum(counts)
        print(
            f"{case.name:<14} {sigma_p99/1e6:>10.3f} {counts[0]:>10d} {counts[1]:>10d} {counts[2]:>10d} {mirrored:>12d}"
        )

    if not all_segments:
        print("\nNo stress lines traced.")
        return

    out_path = out_dir / "stress_lines.vtu"
    meshio.write_points_cells(
        out_path,
        points=np.vstack(all_points),
        cells=[("line", np.vstack(all_segments))],
        cell_data={
            "case": [np.asarray(seg_case_idx, dtype=np.int32)],
            "family": [np.asarray(seg_family, dtype=np.int32)],
            "mirror": [np.asarray(seg_mirror, dtype=np.int32)],
            "sigma_along_MPa": [np.asarray(seg_sigma_along, dtype=np.float64) / 1.0e6],
        },
    )
    n_lines = sum(len(p) - 1 for p in all_points)
    print(f"\n{n_lines} line segments across {len(case_names)} cases × 2 tacks × 3 families")
    print(f"Cases (index → name): {dict(enumerate(case_names))}")
    print(f"Color by `sigma_along_MPa` in ParaView to see structural relevance.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
