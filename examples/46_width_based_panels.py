"""V.3b — Width-based panel buckling: calibration against eigen + the re-sized mass.

Three measurements on the medium problem (16x8, span-datum + buckling, analytic
Jacobian):
  1. CALIBRATE: evaluate the strip-width check on the sqrt-area optimum (no
     re-size). Its worst utilization implies a capacity multiplier to compare
     against V.3's eigen lower bound (lambda_cr = 2.443 at this mesh; ~5.7 at
     16x16 of the same design).
  2. HARVEST: re-size with panel_width_mode="strip" (cold start) — the mass the
     V#1 conservatism was costing.
  3. VERIFY: eigen lambda_cr of the NEW optimum must stay >= 1.5 for every load
     case (everything this mesh family can see: beam, global, coupled modes; the
     across-width panel half-wave is exactly what the strip formula now checks).
Caveats carried forward: kc=4 assumes simply-supported strip edges (flexible beams
may be softer — eigen watches the coupled modes); D11 direction mismatch (V#2) open.
"""
from __future__ import annotations

import time

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell_laminate,
)
from wing_design.beams.laminate_sizing import laminate_result_is_feasible
from wing_design.beams.shell_model import skin_datum_angles, skin_panel_widths
from wing_design.beams.shell_sizing import skin_areas, skin_band_map
from wing_design.materials.unidir import T700_EPOXY, laminate_stiffness_offset
from wing_design.structural.beam_shell import (
    solve_beam_shell_laminate, solve_beam_shell_laminate_factored,
)
from wing_design.structural.buckling import panel_buckling_utilization
from wing_design.structural.eigen_buckling import linear_buckling
from wing_design.structural.frame import BeamSection
from wing_design.structural.geometric_stiffness import assemble_geometric_stiffness
from wing_design.structural.shell import recover_membrane_stress_C

SF = 1.5
DATUM = (0.0, 0.0, 1.0)


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8

    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    model = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    def cfg(mode):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
            tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=SF,
            use_analytic_jacobian=True, panel_width_mode=mode)

    widths = skin_panel_widths(model)
    Atri = skin_areas(model)
    print(f"panel geometry: width median {np.median(widths):.3f} m vs sqrt(area) median "
          f"{np.median(np.sqrt(Atri)):.3f} m -> level credit ~{np.median(Atri/widths**2):.1f}x")

    def evaluate_design(r, b2, tag):
        """Worst panel-buckling util of design r with characteristic b^2 array."""
        band_of_tri = skin_band_map(model, 1)
        t_tri = r.t_bands[band_of_tri]
        th = skin_datum_angles(model, DATUM)
        mats = [laminate_stiffness_offset(T700_EPOXY, f0=float(r.f0_bands[0]),
                                          f45=float(r.f45_bands[0]), f90=float(r.f90_bands[0]),
                                          thickness=float(t_tri[e]), offset_deg=float(np.degrees(th[e])))
                for e in range(t_tri.shape[0])]
        A_arg = np.stack([m[0] for m in mats]); D_arg = np.stack([m[1] for m in mats])
        C_arg = np.stack([m[2] for m in mats])
        D11 = D_arg[:, 0, 0]
        sections = [BeamSection.circular(float(x)) for x in r.radii]
        worst = 0.0
        for lc in loads:
            res = solve_beam_shell_laminate(model.nodes, model.beam_elements, sections,
                model.shell_tris, E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg,
                D_skin=D_arg, fixed_nodes=model.fixed_nodes, loads=lc)
            s = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=C_arg)
            u = panel_buckling_utilization(np.asarray(s), b2, D11=D11, t=t_tri, kc=4.0,
                                           safety_factor=SF)
            worst = max(worst, float(u.max()))
        print(f"  [{tag}] worst panel util = {worst:.3f}")
        return worst

    def eigen_worst(r):
        band_of_tri = skin_band_map(model, 1)
        t_tri = r.t_bands[band_of_tri]
        th = skin_datum_angles(model, DATUM)
        mats = [laminate_stiffness_offset(T700_EPOXY, f0=float(r.f0_bands[0]),
                                          f45=float(r.f45_bands[0]), f90=float(r.f90_bands[0]),
                                          thickness=float(t_tri[e]), offset_deg=float(np.degrees(th[e])))
                for e in range(t_tri.shape[0])]
        A_arg = np.stack([m[0] for m in mats]); D_arg = np.stack([m[1] for m in mats])
        C_arg = np.stack([m[2] for m in mats])
        sections = [BeamSection.circular(float(x)) for x in r.radii]
        fac = solve_beam_shell_laminate_factored(model.nodes, model.beam_elements, sections,
            model.shell_tris, E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg,
            D_skin=D_arg, fixed_nodes=model.fixed_nodes, loads=loads[0])
        lmin = np.inf
        for lc in loads:
            res = solve_beam_shell_laminate(model.nodes, model.beam_elements, sections,
                model.shell_tris, E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg,
                D_skin=D_arg, fixed_nodes=model.fixed_nodes, loads=lc)
            s = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=C_arg)
            Kg = assemble_geometric_stiffness(model.nodes, beam_elements=model.beam_elements,
                axial_forces=res.axial_force, shell_tris=model.shell_tris, t_tri=t_tri,
                tri_stress_local=np.asarray(s))
            lams, _ = linear_buckling(fac.K_ff, Kg[np.ix_(fac.free, fac.free)], k=2)
            if lams.size:
                lmin = min(lmin, float(lams[0]))
        return lmin

    print("\n1) sqrt-area optimum (baseline)...")
    t0 = time.perf_counter()
    base = size_beam_shell_laminate(model, loads, cfg("sqrt_area"), ply=T700_EPOXY,
                                    rho=P.rho_kgm3, maxiter=400)
    print(f"  mass={base.mass_kg:.1f} kg conv={base.converged} "
          f"feas={laminate_result_is_feasible(base, cfg('sqrt_area'))} "
          f"wall={time.perf_counter()-t0:.0f}s")
    evaluate_design(base, Atri, "sqrt_area check @ sqrt_area optimum")
    u_strip = evaluate_design(base, widths**2, "strip check  @ sqrt_area optimum")
    print(f"  implied capacity multiplier of strip check: {SF/max(u_strip,1e-9):.2f} "
          f"(V.3 eigen: 2.44 lower bound @16x8, ~5.7 @16x16)")

    print("\n2) re-size with strip widths...")
    t0 = time.perf_counter()
    strip = size_beam_shell_laminate(model, loads, cfg("strip"), ply=T700_EPOXY,
                                     rho=P.rho_kgm3, maxiter=400)
    feas = laminate_result_is_feasible(strip, cfg("strip"))
    print(f"  mass={strip.mass_kg:.1f} kg conv={strip.converged} feas={feas} "
          f"iters={strip.n_iter} wall={time.perf_counter()-t0:.0f}s")
    print(f"  twist {strip.tip_twist_deg:.2f}/5.0 deg | defl {strip.tip_defl_m*1e3:.0f} mm | "
          f"buck beam {strip.max_beam_buckling_util:.2f} / panel {strip.max_panel_buckling_util:.2f}")
    d = 100.0 * (strip.mass_kg - base.mass_kg) / base.mass_kg
    print(f"  HARVEST: {base.mass_kg:.1f} -> {strip.mass_kg:.1f} kg ({d:+.1f}%)")

    print("\n3) eigen verification of the strip optimum...")
    lam = eigen_worst(strip)
    print(f"  worst lambda_cr = {lam:.3f} vs required {SF} -> "
          f"{'OK' if lam >= SF else 'VIOLATED — strip check overreached what eigen can see'}")


if __name__ == "__main__":
    main()
