"""P.1 — Core tube vs the V.6 baseline: the first hollow-member lever.

Medium 16x8 at the V.6 standard config (strip widths, upright/30deg-heel gravity,
skin-distributed pressure loads, SF 1.5, 1-band, analytic Jacobian), with and
without the filament-wound core tube (annular r(z)/t_wall(z) DVs on the pivot
axis, wall-crimping check at the 0.65 knockdown, bonds to the 4 nearest beams per
level). The tube optimum is eigen-verified (lambda_cr >= 1.5 over all combos; the
eigen solve sees tube elements automatically). Baseline: 2248.0 kg.
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
from wing_design.beams.body_loads import body_load_vector
from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
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

G0 = 9.81
TH = np.radians(30.0)
ACCELS = ((0.0, 0.0, -G0), (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)))
SF = 1.5
DATUM = (0.0, 0.0, 1.0)


def eigen_worst(model, loads_aero, r, rho):
    band_of_tri = skin_band_map(model, 1)
    t_tri = r.t_bands[band_of_tri]
    th = skin_datum_angles(model, DATUM)
    mats = [laminate_stiffness_offset(T700_EPOXY, f0=float(r.f0_bands[0]),
                                      f45=float(r.f45_bands[0]), f90=float(r.f90_bands[0]),
                                      thickness=float(t_tri[e]),
                                      offset_deg=float(np.degrees(th[e])))
            for e in range(t_tri.shape[0])]
    A_arg = np.stack([m[0] for m in mats]); D_arg = np.stack([m[1] for m in mats])
    C_arg = np.stack([m[2] for m in mats])
    n_form = model.n_form_elements
    sections = [BeamSection.circular(float(x)) for x in r.radii]
    gusset = gusset_sec = None
    if r.r_tube is not None:
        sections += [BeamSection.annular(float(r.r_tube[s]), float(r.t_wall[s]))
                     for s in range(len(r.r_tube))]
        gusset, gusset_sec = model.tube_bond_elements, BeamSection.circular(0.05)
    areas_e = np.array([s.A for s in sections])
    fac = solve_beam_shell_laminate_factored(
        model.nodes, model.beam_elements, sections, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
        fixed_nodes=model.fixed_nodes, loads=loads_aero[0],
        gusset_elements=gusset, gusset_section=gusset_sec)
    lmin = np.inf
    for lc in loads_aero:
        for acc in ACCELS:
            f = lc + body_load_vector(model, r.radii, t_tri, rho=rho, accel=acc,
                                      areas=areas_e)
            res = solve_beam_shell_laminate(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
                fixed_nodes=model.fixed_nodes, loads=f,
                gusset_elements=gusset, gusset_section=gusset_sec)
            s = recover_membrane_stress_C(model.nodes, model.shell_tris,
                                          res.displacements, C=C_arg)
            Kg = assemble_geometric_stiffness(
                model.nodes, beam_elements=model.beam_elements,
                axial_forces=res.axial_force, shell_tris=model.shell_tris,
                t_tri=t_tri, tri_stress_local=np.asarray(s))
            lams, _ = linear_buckling(fac.K_ff, Kg[np.ix_(fac.free, fac.free)], k=1)
            if lams.size:
                lmin = min(lmin, float(lams[0]))
    return lmin


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    frame = build_beam_frame(spec, n_beams=16, n_levels=8, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
        tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=SF,
        use_analytic_jacobian=True, panel_width_mode="strip", accel_vectors=ACCELS)

    results = {}
    for tag, with_tube in (("baseline", False), ("core tube", True)):
        model = build_beam_shell_model(spec, n_beams=16, n_levels=8, beam_radius=0.02,
            material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
            skin_thickness=P.skin_sizing.t_baseline_m, core_tube=with_tube)
        loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
                 for ar in active]
        press = [panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor)
                 for ar in active]
        if with_tube:
            print(f"tube fit bounds: {np.round(model.tube_r_bounds, 3)} m", flush=True)
        t0 = time.perf_counter()
        r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                     maxiter=500, panel_pressures=press)
        wall = time.perf_counter() - t0
        feas = laminate_result_is_feasible(r, cfg)
        lam = eigen_worst(model, loads, r, P.rho_kgm3)
        line = (f"[{tag}] mass={r.mass_kg:8.1f} kg (beams {r.beam_mass_kg:.0f} + skin "
                f"{r.skin_mass_kg:.0f} + tube {r.tube_mass_kg:.0f}) conv={r.converged} "
                f"feas={feas} iters={r.n_iter} wall={wall:.0f}s eigen_lam={lam:.2f}")
        if with_tube:
            line += (f"\n    r_tube {np.round(r.r_tube*1e3, 1)} mm | t_wall "
                     f"{np.round(r.t_wall*1e3, 2)} mm | wall util {r.max_tube_wall_util:.2f}")
        if r.shadow_prices:
            sp = r.shadow_prices
            line += (f"\n    shadow: twist {sp['tip_twist_max_deg']:+.1f} kg/deg | "
                     f"buckSF beam {sp.get('buckling_sf_beam', 0):+.0f} / panel "
                     f"{sp.get('buckling_sf_panel', 0):+.0f} / tube-wall "
                     f"{sp.get('buckling_sf_tube_wall', 0):+.0f} kg/SF")
        print(line, flush=True)
        results[tag] = r

    b, t = results["baseline"].mass_kg, results["core tube"].mass_kg
    print(f"\nLEVER: {b:.1f} -> {t:.1f} kg ({100*(t-b)/b:+.1f}%)")


if __name__ == "__main__":
    main()
