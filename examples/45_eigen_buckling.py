"""V.3 — Eigenvalue buckling check (K + Kσ) of the medium optimum.

Sizes the medium wingsail (example-32 config), then per load case assembles the
geometric stiffness from the solved stress state (beam axial forces + per-triangle
local membrane stresses) and solves K φ = −λ Kσ φ. Loads are already factored
(case safety factors applied at projection), so the closed-form checks claim
capacity ≥ 1.5× demand: **λ_cr ≈ 1.5 validates the binding closed-form level,
λ_cr > 1.5 means it was conservative (mass on the table), λ_cr < 1.5 means
unconservative (honest baseline moves up).** Worst modes are exported to
exports/eigen_mode_*.vtu (ParaView / `just shot`); regenerate with
`just example 45_eigen_buckling`.

Then the eigen-grade version of the example-44 pretension probe: hoop prestress
added to the panel stresses entering Kσ of the worst case (one-sided, like ex-44:
no equilibrating member compression).
"""
from __future__ import annotations

import time
from pathlib import Path

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
from wing_design.beams.shell_model import skin_datum_angles
from wing_design.beams.shell_sizing import skin_band_map
from wing_design.materials.unidir import T700_EPOXY, laminate_stiffness_offset
from wing_design.structural.beam_shell import (
    solve_beam_shell_laminate, solve_beam_shell_laminate_factored,
)
from wing_design.structural.eigen_buckling import linear_buckling
from wing_design.structural.frame import BeamSection
from wing_design.structural.geometric_stiffness import assemble_geometric_stiffness
from wing_design.structural.shell import recover_membrane_stress_C
from wing_design.viz.vtu import write_vtu_triangles

PRETENSION_MPA = [0.0, 5.0, 10.0, 20.0]
SF_REQUIRED = 1.5
DATUM = (0.0, 0.0, 1.0)
N_MODES = 6


def participation(mode, n_nodes, free):
    """(spread, peak_node): spread = participation ratio of nodal translational
    amplitudes in [1/N (one node) .. 1 (uniform)]; classifies local vs global."""
    full = np.zeros(6 * n_nodes)
    full[free] = mode
    a = np.linalg.norm(full.reshape(n_nodes, 6)[:, :3], axis=1)
    a2 = a**2
    s2 = a2.sum()
    spread = (s2**2 / (n_nodes * (a2**2).sum())) if s2 > 0 else 0.0
    return float(spread), int(np.argmax(a))


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

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        ply_angle_datum=DATUM, buckling_safety_factor=SF_REQUIRED, use_analytic_jacobian=True,
    )
    print("sizing the medium optimum (analytic Jacobian)...")
    t0 = time.perf_counter()
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=400)
    print(f"  mass={r.mass_kg:.1f} kg conv={r.converged} "
          f"feas={laminate_result_is_feasible(r, cfg)} wall={time.perf_counter()-t0:.0f}s "
          f"closed-form util: beam={r.max_beam_buckling_util:.3f} panel={r.max_panel_buckling_util:.3f}")

    # Optimum laminate, exactly as the sizer builds it.
    B = cfg.n_skin_bands
    band_of_tri = skin_band_map(model, B)
    t_tri = r.t_bands[band_of_tri]
    theta_tri = skin_datum_angles(model, DATUM)
    mats = [laminate_stiffness_offset(
                T700_EPOXY, f0=float(r.f0_bands[band_of_tri[e]]),
                f45=float(r.f45_bands[band_of_tri[e]]), f90=float(r.f90_bands[band_of_tri[e]]),
                thickness=float(t_tri[e]), offset_deg=float(np.degrees(theta_tri[e])))
            for e in range(t_tri.shape[0])]
    A_arg = np.stack([m[0] for m in mats])
    D_arg = np.stack([m[1] for m in mats])
    C_arg = np.stack([m[2] for m in mats])
    sections = [BeamSection.circular(float(x)) for x in r.radii]
    n_nodes = model.nodes.shape[0]

    fac = solve_beam_shell_laminate_factored(
        model.nodes, model.beam_elements, sections, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
        fixed_nodes=model.fixed_nodes, loads=loads[0])
    free = fac.free
    K_ff = fac.K_ff
    print(f"  eigen system: {free.size} free DOFs (dense path)")

    out = Path("exports"); out.mkdir(exist_ok=True)
    worst = (np.inf, -1, None, None)   # (lambda, lc index, stress_local, axial)
    print(f"\n  per-load-case buckling multipliers (loads already factored; "
          f"closed-form claims capacity at {SF_REQUIRED}x):")
    for li, lc in enumerate(loads):
        res = solve_beam_shell_laminate(
            model.nodes, model.beam_elements, sections, model.shell_tris,
            E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
            fixed_nodes=model.fixed_nodes, loads=lc)
        s_loc = recover_membrane_stress_C(model.nodes, model.shell_tris,
                                          res.displacements, C=C_arg)
        Kg = assemble_geometric_stiffness(
            model.nodes, beam_elements=model.beam_elements, axial_forces=res.axial_force,
            shell_tris=model.shell_tris, t_tri=t_tri, tri_stress_local=np.asarray(s_loc))
        t1 = time.perf_counter()
        lams, modes = linear_buckling(K_ff, Kg[np.ix_(free, free)], k=N_MODES)
        dt = time.perf_counter() - t1
        if lams.size == 0:
            print(f"   lc{li}: no buckling below cap ({dt:.1f}s)")
            continue
        spread, peak = participation(modes[:, 0], n_nodes, free)
        kind = "global" if spread > 0.15 else "local"
        zpk = model.nodes[peak, 2]
        print(f"   lc{li}: lambda_cr={lams[0]:7.3f}  (next: "
              + ", ".join(f"{x:.2f}" for x in lams[1:4])
              + f")  mode={kind} spread={spread:.3f} peak z={zpk:.1f} m  ({dt:.1f}s)")
        if lams[0] < worst[0]:
            worst = (float(lams[0]), li, np.asarray(s_loc), res.axial_force)
        # export the first mode
        full = np.zeros(6 * n_nodes); full[free] = modes[:, 0]
        write_vtu_triangles(
            out / f"eigen_mode_lc{li}.vtu", model.nodes, model.shell_tris,
            point_data={"mode_translation": full.reshape(n_nodes, 6)[:, :3]})

    lam_w, li_w, s_loc_w, axial_w = worst
    verdict = ("VALIDATED (~exact)" if 0.9 * SF_REQUIRED <= lam_w <= 1.1 * SF_REQUIRED
               else "closed-form CONSERVATIVE (mass on the table)" if lam_w > SF_REQUIRED
               else "closed-form UNCONSERVATIVE (honest baseline moves up)")
    print(f"\n  WORST lambda_cr = {lam_w:.3f} (lc{li_w}) vs required {SF_REQUIRED}: {verdict}")

    # Eigen-grade pretension probe on the worst case (one-sided, as in ex-44).
    print("\n  hoop-pretension parametric on the worst case (eigen-grade, one-sided):")
    for sp_mpa in PRETENSION_MPA:
        s_dat = np.empty_like(s_loc_w)
        c, q = np.cos(-theta_tri), np.sin(-theta_tri)
        sxx, syy, sxy = s_loc_w[:, 0], s_loc_w[:, 1], s_loc_w[:, 2]
        s_dat[:, 0] = c*c*sxx + q*q*syy + 2*c*q*sxy
        s_dat[:, 1] = q*q*sxx + c*c*syy - 2*c*q*sxy
        s_dat[:, 2] = -c*q*sxx + c*q*syy + (c*c - q*q)*sxy
        s_dat[:, 1] += sp_mpa * 1.0e6
        s_net = np.empty_like(s_dat)
        c, q = np.cos(theta_tri), np.sin(theta_tri)
        dxx, dyy, dxy = s_dat[:, 0], s_dat[:, 1], s_dat[:, 2]
        s_net[:, 0] = c*c*dxx + q*q*dyy + 2*c*q*dxy
        s_net[:, 1] = q*q*dxx + c*c*dyy - 2*c*q*dxy
        s_net[:, 2] = -c*q*dxx + c*q*dyy + (c*c - q*q)*dxy
        Kg = assemble_geometric_stiffness(
            model.nodes, beam_elements=model.beam_elements, axial_forces=axial_w,
            shell_tris=model.shell_tris, t_tri=t_tri, tri_stress_local=s_net)
        lams, modes = linear_buckling(K_ff, Kg[np.ix_(free, free)], k=2)
        if lams.size:
            spread, _peak = participation(modes[:, 0], n_nodes, free)
            kind = "global" if spread > 0.15 else "local"
            print(f"   sigma_p={sp_mpa:5.1f} MPa: lambda_cr={lams[0]:7.3f}  mode={kind}")
        else:
            print(f"   sigma_p={sp_mpa:5.1f} MPa: no buckling below cap")


if __name__ == "__main__":
    main()
