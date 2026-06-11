"""P.4 — n_beams sweep on the tube+hollow running-best config (user-requested).

Sweeps n_beams in {12, 16, 20, 24, 28} (even counts preserve mirror grouping) at
n_levels = 8, medium scenario, V.6 standard config + core tube + hollow form
beams. Physically meaningful only post-V.3b: the strip widths scale ~1/n so panel
capacity scales ~n^2 like reality (the legacy b=sqrt(area) credited only ~n and
would have understated the win — backlog V#1/P#4 gate).

Per-point protocol (the P#1b lesson — cold tube+hollow starts get eigen-rejected):
(1) tube-only sizing cold, (2) tube+hollow warm-started from it with
solid-equivalent walls, (3) eigen verification of the final optimum. Points run
in parallel processes (V.0.4 discipline: independent, single-threaded each).
Reference: 16 beams = 1924.6 kg (eigen lambda 2.54).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_shell_model,
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

G0 = 9.81
TH = np.radians(30.0)
ACCELS = ((0.0, 0.0, -G0), (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)))
N_BEAMS = (12, 16, 20, 24, 28)
N_LEVELS = 8


def _sections_for(model, r):
    goe, _G = beam_radius_groups(model)
    secs = []
    if r.t_hollow is not None:
        hgroups = sorted({int(goe[e]) for e in model.hollow_elements})
        tmap = {g: i for i, g in enumerate(hgroups)}
        hset = set(int(e) for e in model.hollow_elements)
    for e in range(model.n_form_elements):
        rr = float(r.radii[e])
        if r.t_hollow is not None and e in hset:
            secs.append(BeamSection.annular(rr, min(float(r.t_hollow[tmap[int(goe[e])]]), rr)))
        else:
            secs.append(BeamSection.circular(rr))
    if r.r_tube is not None:
        secs += [BeamSection.annular(float(rt), min(float(tw), float(rt)))
                 for rt, tw in zip(r.r_tube, r.t_wall)]
    return secs


def _eigen_worst(model, loads, r, rho):
    secs = _sections_for(model, r)
    areas_e = np.array([s.A for s in secs])
    t_tri = np.full(model.shell_tris.shape[0], float(r.t_bands[0]))
    th = skin_datum_angles(model, (0.0, 0.0, 1.0))
    mats = [laminate_stiffness_offset(T700_EPOXY, f0=float(r.f0_bands[0]),
                                      f45=float(r.f45_bands[0]), f90=float(r.f90_bands[0]),
                                      thickness=float(t_tri[e]),
                                      offset_deg=float(np.degrees(th[e])))
            for e in range(t_tri.shape[0])]
    A_arg = np.stack([m[0] for m in mats]); D_arg = np.stack([m[1] for m in mats])
    C_arg = np.stack([m[2] for m in mats])
    fac = solve_beam_shell_laminate_factored(
        model.nodes, model.beam_elements, secs, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
        fixed_nodes=model.fixed_nodes, loads=loads[0],
        gusset_elements=model.tube_bond_elements, gusset_section=BeamSection.circular(0.05))
    lmin = np.inf
    for lc in loads:
        for acc in ACCELS:
            f = lc + body_load_vector(model, r.radii, t_tri, rho=rho, accel=acc, areas=areas_e)
            res = solve_beam_shell_laminate(
                model.nodes, model.beam_elements, secs, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
                fixed_nodes=model.fixed_nodes, loads=f,
                gusset_elements=model.tube_bond_elements,
                gusset_section=BeamSection.circular(0.05))
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


def run_point(n_beams: int) -> str:
    P = medium_scenario()
    spec = P.geometry
    airplane = build_airplane(spec)
    env = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in env
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    def build(hollow):
        m = build_beam_shell_model(spec, n_beams=n_beams, n_levels=N_LEVELS,
            beam_radius=0.02, material=P.material,
            knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
            skin_thickness=P.skin_sizing.t_baseline_m,
            core_tube=True, hollow_beams=hollow)
        loads = [project_panels_to_skin(m, ar.panels, safety_factor=ar.case.safety_factor)
                 for ar in active]
        press = [panel_pressure_per_tri(m, ar.panels, safety_factor=ar.case.safety_factor)
                 for ar in active]
        return m, loads, press

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
        tip_twist_max_deg=5.0, ply_angle_datum=(0.0, 0.0, 1.0),
        buckling_safety_factor=1.5, use_analytic_jacobian=True,
        panel_width_mode="strip", accel_vectors=ACCELS)

    t0 = time.perf_counter()
    m1, loads1, press1 = build(False)
    r1 = size_beam_shell_laminate(m1, loads1, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                  maxiter=500, panel_pressures=press1)
    ok1 = r1.converged and laminate_result_is_feasible(r1, cfg)

    m2, loads2, press2 = build(True)
    goe, G = beam_radius_groups(m2)
    rg = np.zeros(G); seen = set()
    for e, g in enumerate(goe):
        if g not in seen:
            rg[g] = r1.radii[e]; seen.add(g)
    hgroups = sorted({int(goe[e]) for e in m2.hollow_elements})
    x0 = np.concatenate([rg, np.asarray(r1.t_bands, float),
                         [float(r1.f0_bands[0])], [float(r1.f45_bands[0])],
                         r1.r_tube, r1.t_wall,
                         np.array([rg[g] for g in hgroups])])
    r2 = size_beam_shell_laminate(m2, loads2, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                  maxiter=800, x0=x0, panel_pressures=press2)
    ok2 = r2.converged and laminate_result_is_feasible(r2, cfg)
    lam = _eigen_worst(m2, loads2, r2, P.rho_kgm3) if ok2 else float("nan")
    wall = time.perf_counter() - t0
    return (f"[n_beams={n_beams:2d}] tube-only={r1.mass_kg:7.1f} (ok={ok1}) | "
            f"tube+hollow={r2.mass_kg:7.1f} kg ok={ok2} iters={r2.n_iter} "
            f"eigen_lam={lam:.2f} (beams {r2.beam_mass_kg:.0f} + skin "
            f"{r2.skin_mass_kg:.0f} + tube {r2.tube_mass_kg:.0f}) wall={wall:.0f}s")


def main() -> None:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(N_BEAMS)) as pool:
        for line in pool.map(run_point, N_BEAMS):
            print(line, flush=True)
    print(f"sweep total wall={time.perf_counter()-t0:.0f}s "
          f"(reference: 16 beams = 1924.6 kg, eigen 2.54)")


if __name__ == "__main__":
    main()
