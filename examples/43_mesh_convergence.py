"""V.2 — Mesh-convergence study of the optimized medium design.

Sweeps n_levels at the medium headline configuration (16 beams, span-datum CLT +
closed-form buckling, analytic Jacobian), warm-starting each point from the previous
level's optimum (radii resampled via `resample_segment_radii`, bands/fractions carried
over), plus an n_beams spot-check at the base level count. Quantifies the known
unconservative mesh dependences (backlog V#3, V#9): refinement shortens the beam Euler
length L_element AND shrinks panel b = sqrt(area), both raising apparent capacity and
letting the optimizer legally remove mass — so the headline is converged only if mass
flattens. Every point reports converged + feasible + wall-clock; non-feasible points
are diagnostics, not results.
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
from wing_design.beams.build import resample_segment_radii
from wing_design.beams.laminate_sizing import laminate_result_is_feasible
from wing_design.beams.shell_sizing import beam_radius_groups
from wing_design.materials.unidir import T700_EPOXY

N_LEVELS_SWEEP = [6, 8, 10, 12]
N_BEAMS_SPOT = 20          # spot-check at n_levels = 8
BASE_LEVELS = 8


def x0_from(model, res):
    """Sizer design vector from a result on the SAME model (global layup, B bands)."""
    goe, G = beam_radius_groups(model)
    rg = np.zeros(G)
    seen = set()
    for e, g in enumerate(goe):
        if g not in seen:
            rg[g] = res.radii[e]
            seen.add(g)
    return np.concatenate([rg, np.asarray(res.t_bands, float),
                           [float(res.f0_bands[0])], [float(res.f45_bands[0])]])


def x0_resampled(model_to, res_from, *, n_beams, n_from, n_to):
    """Warm start on a finer mesh: per-element radii resampled, bands/fracs carried."""
    radii_to = resample_segment_radii(res_from.radii, n_beams=n_beams,
                                      n_from=n_from, n_to=n_to)
    goe, G = beam_radius_groups(model_to)
    rg = np.zeros(G)
    seen = set()
    for e, g in enumerate(goe):
        if g not in seen:
            rg[g] = radii_to[e]
            seen.add(g)
    return np.concatenate([rg, np.asarray(res_from.t_bands, float),
                           [float(res_from.f0_bands[0])], [float(res_from.f45_bands[0])]])


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
        tip_twist_max_deg=5.0, ply_angle_datum=(0.0, 0.0, 1.0),
        buckling_safety_factor=1.5, use_analytic_jacobian=True,
    )

    def build_point(n_beams, n_levels):
        frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
            material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
        model = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
            material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
            skin_thickness=P.skin_sizing.t_baseline_m)
        loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
                 for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
        return model, loads

    def run_point(tag, model, loads, x0=None, maxiter=400):
        t0 = time.perf_counter()
        r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                     maxiter=maxiter, x0=x0)
        wall = time.perf_counter() - t0
        feas = laminate_result_is_feasible(r, cfg)
        govern = []
        if r.tip_twist_deg > 0.98 * cfg.tip_twist_max_deg: govern.append("twist")
        if r.tip_defl_m > 0.98 * cfg.tip_defl_max_m: govern.append("defl")
        if r.max_beam_buckling_util > 0.98: govern.append("beam-buck")
        if r.max_panel_buckling_util > 0.98: govern.append("panel-buck")
        print(f"[{tag}] mass={r.mass_kg:8.1f} kg (beams {r.beam_mass_kg:6.1f} + skin {r.skin_mass_kg:7.1f})"
              f"  conv={r.converged} feas={feas} iters={r.n_iter:3d} wall={wall:6.1f}s"
              f"  governing={'+'.join(govern) or 'none'}"
              f"  twist={r.tip_twist_deg:.2f} defl={r.tip_defl_m*1e3:.0f}mm"
              f"  buck={r.max_beam_buckling_util:.2f}/{r.max_panel_buckling_util:.2f}",
              flush=True)
        return r

    print(f"n_levels sweep at n_beams=16: {N_LEVELS_SWEEP}, warm-started coarse->fine")
    prev = None
    prev_levels = None
    results = {}
    for nl in N_LEVELS_SWEEP:
        model, loads = build_point(16, nl)
        x0 = (x0_resampled(model, prev, n_beams=16, n_from=prev_levels, n_to=nl)
              if prev is not None else None)
        prev = run_point(f"16x{nl}", model, loads, x0=x0)
        prev_levels = nl
        results[nl] = prev

    print(f"\nn_beams spot-check at n_levels={BASE_LEVELS}: {N_BEAMS_SPOT} beams (cold start)")
    model, loads = build_point(N_BEAMS_SPOT, BASE_LEVELS)
    run_point(f"{N_BEAMS_SPOT}x{BASE_LEVELS}", model, loads)

    base = results[BASE_LEVELS]
    print("\nconvergence vs the 16x8 headline mesh:")
    for nl, r in results.items():
        d = 100.0 * (r.mass_kg - base.mass_kg) / base.mass_kg
        print(f"  n_levels={nl:2d}: {r.mass_kg:8.1f} kg ({d:+5.1f}%)")
    print("\nNote: noise floor ~2-3%; differences beyond it with monotonic trend = real"
          "\nmesh dependence (expected unconservative: mass drops as the mesh refines).")


if __name__ == "__main__":
    main()
