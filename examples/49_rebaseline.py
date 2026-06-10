"""V.6 — Re-baseline the small + medium headlines with everything honest at once.

Configuration (the post-validity-sprint standard): strip-width panel buckling
(V.3b, eigen-calibrated), self-weight/heel/slam load combos with the lam^T dF/dx
adjoint term (V.4), skin-distributed aero projection + strip-bending pressure term
in the skin failure check (V.5), span-datum CLT + closed-form beam Euler, SF 1.5,
analytic Jacobian. 1-band and 4-band per scenario (4-band warm-started from
1-band). Every optimum is eigen-verified (worst lambda_cr over all load combos
must clear 1.5). The medium 4-band result is THE new headline; STL geometry +
worst-eigen-mode VTU are exported (milestone convention).

All Phase-P levers measure against these numbers.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from wing_design import medium_scenario, small_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    build_sized_circular_beams,
    build_sized_lens_beams,
    project_panels_to_beam_nodes,
    resample_segment_radii,
    size_beam_shell_laminate,
)
from wing_design.beams.body_loads import body_load_vector
from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
from wing_design.beams.laminate_sizing import laminate_result_is_feasible
from wing_design.beams.shell_model import skin_datum_angles
from wing_design.beams.shell_sizing import beam_radius_groups, skin_band_map
from wing_design.materials.unidir import T700_EPOXY, laminate_stiffness_offset
from wing_design.structural.beam_shell import (
    solve_beam_shell_laminate, solve_beam_shell_laminate_factored,
)
from wing_design.structural.eigen_buckling import linear_buckling
from wing_design.structural.frame import BeamSection
from wing_design.structural.geometric_stiffness import assemble_geometric_stiffness
from wing_design.structural.shell import recover_membrane_stress_C
from wing_design.viz.vtu import write_vtu_triangles

G0 = 9.81
SF = 1.5
DATUM = (0.0, 0.0, 1.0)
TH = np.radians(30.0)
# Upright + 30 deg heel gravity. The 1 g lateral slam static-equivalent is
# EXCLUDED pending the V#12 envelope decision: ex-47 measured it driving mass
# toward +110% and SLSQP infeasible at maxiter 400 — a requirements question,
# not a sizing default.
ACCELS = (
    (0.0, 0.0, -G0),                                   # upright gravity
    (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)),          # 30 deg heel
)


def x0_from(model, r, n_bands):
    goe, G = beam_radius_groups(model)
    rg = np.zeros(G)
    seen = set()
    for e, g in enumerate(goe):
        if g not in seen:
            rg[g] = r.radii[e]
            seen.add(g)
    t = np.repeat(np.asarray(r.t_bands, float), n_bands // len(r.t_bands))
    return np.concatenate([rg, t, [float(r.f0_bands[0])], [float(r.f45_bands[0])]])


def eigen_worst(model, loads_aero, r, n_bands, rho):
    """Worst lambda_cr over all (aero, accel) combos at the optimum design."""
    band_of_tri = skin_band_map(model, n_bands)
    t_tri = r.t_bands[band_of_tri]
    th = skin_datum_angles(model, DATUM)
    mats = [laminate_stiffness_offset(
                T700_EPOXY, f0=float(r.f0_bands[band_of_tri[e]]),
                f45=float(r.f45_bands[band_of_tri[e]]),
                f90=float(r.f90_bands[band_of_tri[e]]),
                thickness=float(t_tri[e]), offset_deg=float(np.degrees(th[e])))
            for e in range(t_tri.shape[0])]
    A_arg = np.stack([m[0] for m in mats]); D_arg = np.stack([m[1] for m in mats])
    C_arg = np.stack([m[2] for m in mats])
    sections = [BeamSection.circular(float(x)) for x in r.radii]
    fac = solve_beam_shell_laminate_factored(
        model.nodes, model.beam_elements, sections, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
        fixed_nodes=model.fixed_nodes, loads=loads_aero[0])
    lmin, mode_min = np.inf, None
    for lc in loads_aero:
        for acc in ACCELS:
            f = lc + body_load_vector(model, r.radii, t_tri, rho=rho, accel=acc)
            res = solve_beam_shell_laminate(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
                fixed_nodes=model.fixed_nodes, loads=f)
            s = recover_membrane_stress_C(model.nodes, model.shell_tris,
                                          res.displacements, C=C_arg)
            Kg = assemble_geometric_stiffness(
                model.nodes, beam_elements=model.beam_elements,
                axial_forces=res.axial_force, shell_tris=model.shell_tris,
                t_tri=t_tri, tri_stress_local=np.asarray(s))
            lams, modes = linear_buckling(fac.K_ff, Kg[np.ix_(fac.free, fac.free)], k=1)
            if lams.size and lams[0] < lmin:
                lmin, mode_min = float(lams[0]), (modes[:, 0], fac.free)
    return lmin, mode_min


def run_scenario(name, P):
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
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]
    pressures = [panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor)
                 for ar in active]

    results = {}
    prev = None
    for n_bands in (1, 4):
        cfg = LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
            tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=SF,
            n_skin_bands=n_bands, use_analytic_jacobian=True,
            panel_width_mode="strip", accel_vectors=ACCELS)
        x0 = x0_from(model, prev, n_bands) if prev is not None else None
        t0 = time.perf_counter()
        r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                     maxiter=500, x0=x0, panel_pressures=pressures)
        wall = time.perf_counter() - t0
        feas = laminate_result_is_feasible(r, cfg)
        lam, mode = eigen_worst(model, loads, r, n_bands, P.rho_kgm3)
        govern = []
        if r.tip_twist_deg > 4.9: govern.append("twist")
        if r.tip_defl_m > 0.98 * 0.02 * spec.span: govern.append("defl")
        if r.max_beam_buckling_util > 0.98: govern.append("beam-buck")
        if r.max_panel_buckling_util > 0.98: govern.append("panel-buck")
        if r.max_skin_vm_Pa > 0.95 * P.sigma_allow_Pa: govern.append("skin-vm")
        print(f"[{name} {n_bands}-band] mass={r.mass_kg:8.2f} kg conv={r.converged} "
              f"feas={feas} iters={r.n_iter} wall={wall:5.0f}s eigen_lam={lam:.2f} "
              f"governing={'+'.join(govern) or 'none'}", flush=True)
        if r.shadow_prices:
            sp = r.shadow_prices
            print(f"    shadow: twist {sp['tip_twist_max_deg']:+.2f} kg/deg | "
                  f"defl {sp['tip_defl_max_m']:+.1f} kg/m | "
                  f"buckSF beam {sp.get('buckling_sf_beam', 0):+.1f} / "
                  f"panel {sp.get('buckling_sf_panel', 0):+.1f} kg/SF", flush=True)
        results[n_bands] = (r, lam, mode)
        prev = r
    return model, results


def main() -> None:
    out = Path("exports"); out.mkdir(exist_ok=True)
    print("=== V.6 re-baseline (strip + upright/heel gravity + pressure bending; "
          "slam deferred per 2026-06-10 decision) ===")
    _small_model, _small = run_scenario("small", small_scenario())

    model, med = run_scenario("medium", medium_scenario())
    r4, lam4, mode4 = med[4]
    if mode4 is not None:
        vec, free = mode4
        full = np.zeros(6 * model.nodes.shape[0]); full[free] = vec
        write_vtu_triangles(out / "rebaseline_medium_worst_mode.vtu", model.nodes,
                            model.shell_tris,
                            point_data={"mode_translation":
                                        full.reshape(-1, 6)[:, :3]})
        print(f"  worst eigen mode -> {out}/rebaseline_medium_worst_mode.vtu")

    # CAD export of the new medium headline (milestone convention)
    spec = medium_scenario().geometry
    n_geom = 40
    dense = resample_segment_radii(r4.radii, n_beams=16, n_from=8, n_to=n_geom)
    try:
        beams = build_sized_lens_beams(spec, dense, n_beams=16, n_levels=n_geom)
        kind = "lens"
    except Exception as exc:
        print(f"  lens loft failed ({exc}); circular fallback")
        beams = build_sized_circular_beams(spec, dense, n_beams=16, n_levels=n_geom)
        kind = "circular"
    from build123d import Compound, export_stl
    comp = Compound(label="beams_rebaseline", children=list(beams))
    export_stl(comp, str(out / "rebaseline_medium_beams.stl"))
    print(f"  sized beams ({kind}) -> {out}/rebaseline_medium_beams.stl "
          f"(t_bands {[round(t*1e3, 2) for t in r4.t_bands]} mm)")


if __name__ == "__main__":
    main()
