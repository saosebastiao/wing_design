"""Pinned-ring 3 g slam survival sizing — get the real survival mass.

The eigen sweep proved 3 g slam IS survivable with ring bracing (~14 mm rings →
λ≥1.5, 20 mm → λ 3.16), but the closed-form sizer is non-conservative under slam
and drives brace_radius to its floor (it can't "see" that the rings are needed).
Fix: PIN the ring floor at a value we know clears the gate (brace_r_min = 20 mm,
2× eigen margin) and let the optimizer size everything ELSE (skin/beams/core) for
deflection + panel + stress under 3 g. The optimizer will sit at the 20 mm floor
(it wants to minimize ring mass), so rings = 20 mm and the post-hoc eigen confirms
buckling survival.

IPOPT does not fully converge slam problems within 3000 iters (true on every slam
run), so the deliverable is the best HARD-FEASIBLE incumbent (incumbent guard),
eigen-verified λ ≥ 1.5. Reported as such (feasible + eigen-verified; may not be
mass-optimal). Single worker. Usage: python runs/slam_pinned.py [G] [brace_min_mm]
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

RUNS = Path(__file__).parent
sys.path.insert(0, str(RUNS))

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import build_beam_shell_model
from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig, design_vector_from_result, laminate_design_bounds,
    laminate_result_is_feasible, size_beam_shell_laminate,
)
from wing_design.materials.unidir import PVC_H80, T700_EPOXY
from chain_rebuild import DATUM
from slam_braced import eigen_worst_braced, G0, TH, GATE


def main(slam_g: float, brace_min_mm: float) -> None:
    tag = f"slam_pinned_{slam_g:g}g_{brace_min_mm:g}mm"
    accels = ((0.0, 0.0, -G0), (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)),
              (0.0, slam_g * G0, -G0), (0.0, -slam_g * G0, -G0))
    P = medium_scenario(); spec = P.geometry
    envelope = sweep_envelope(build_airplane(spec), P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    model = build_beam_shell_model(spec, n_beams=16, n_levels=8, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m, core_tube=True, hollow_beams=True,
        lateral_bracing=True)
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]
    press = [panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]

    brace_min = brace_min_mm / 1e3
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
        tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=1.5,
        use_analytic_jacobian=True, panel_width_mode="strip", accel_vectors=accels,
        core=PVC_H80, ks_rho=50.0, optimizer="ipopt",
        beam_buckling_model="foundation", panel_d_mode="datum_ortho",
        brace_r_min=brace_min, brace_r_max=0.06)

    head = np.asarray(np.load(RUNS / "multistart_v2_best.npz")["x"], dtype=float)
    lo, hi = laminate_design_bounds(model, cfg)
    x0 = np.concatenate([head, np.array([brace_min])])   # rings pinned-engaged at the floor
    assert x0.shape == lo.shape, f"x0 {x0.shape} != bounds {lo.shape}"
    print(f"[{tag}] pinned rings >= {brace_min_mm:g}mm; warm from headline + rings at floor "
          f"(x0={x0.shape[0]})", flush=True)

    t0 = time.perf_counter()
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                 maxiter=3000, panel_pressures=press, x0=x0)
    wall = time.perf_counter() - t0
    feas = laminate_result_is_feasible(r, cfg)
    lam = eigen_worst_braced(model, loads, r, P.rho_kgm3, accels)
    br_mm = float(r.brace_radius) * 1e3 if r.brace_radius is not None else float("nan")
    # SURVIVAL = feasible AND eigen-verified (IPOPT convergence not required; incumbent guard)
    survives = bool(feas) and lam >= GATE
    lim_defl = 0.02 * spec.span
    utils = dict(tip_twist=float(r.tip_twist_deg / cfg.tip_twist_max_deg),
                 tip_defl=float(getattr(r, "tip_defl_m", float("nan")) / lim_defl),
                 beam_buckling=float(r.max_beam_buckling_util),
                 panel_buckling=float(r.max_panel_buckling_util))
    meta = dict(slam_g=float(slam_g), brace_min_mm=float(brace_min_mm), gate=float(GATE),
                mass_kg=float(r.mass_kg), beam_mass_kg=float(r.beam_mass_kg),
                skin_mass_kg=float(r.skin_mass_kg), core_mass_kg=float(r.core_mass_kg),
                tube_mass_kg=float(r.tube_mass_kg), brace_mass_kg=float(r.brace_mass_kg),
                converged=bool(r.converged), feasible=bool(feas), eigen_lam=float(lam),
                survives=survives, brace_radius_mm=br_mm, n_iter=int(r.n_iter),
                used_incumbent=bool(r.used_incumbent), wall_s=float(wall), utils=utils,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
    out = {k: getattr(r, k) for k in ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands")}
    for k in ("r_tube", "t_wall", "t_hollow", "t_core", "brace_radius"):
        v = getattr(r, k, None)
        if v is not None:
            out[k] = np.atleast_1d(v)
    np.savez(RUNS / f"{tag}.npz", x=design_vector_from_result(model, cfg, r),
             meta=json.dumps(meta), **out)
    print(f"SLAM_PINNED DONE [{tag}] SURVIVES={survives} mass={r.mass_kg:.1f}kg "
          f"(beams {r.beam_mass_kg:.0f}+skin {r.skin_mass_kg:.0f}+core {r.core_mass_kg:.0f}"
          f"+tube {r.tube_mass_kg:.0f}+brace {r.brace_mass_kg:.0f}) conv={r.converged} "
          f"feas={feas} eigen={lam:.2f} brace_r={br_mm:.1f}mm utils={utils} "
          f"wall={wall:.0f}s", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    g = float(args[0]) if args else 3.0
    bmm = float(args[1]) if len(args) > 1 else 20.0
    main(g, bmm)
