"""g-continuation braced slam sizing — V#12 lateral bracing, robust solve.

The direct 3 g braced run (runs/slam_braced.py 3) stalled: seeded from the
FAILED unbraced 3 g diagnostic, IPOPT dropped into deep feasibility-restoration
(multiple violations) where beam buckling — the rings' only lever — was not
marginal, so it drove brace_radius to its floor instead of engaging the rings.
Diagnosis confirmed the rings ARE physically capable (k_ring gives foundation
Pcr ×1.84 at r=20 mm, ×9.7 at 50 mm); the failure was the solve, not the rings.

This script keeps the design NEAR-FEASIBLE throughout by ramping the lateral
slam g from a gentle 1 g up to 3 g, warm-chaining each step from the previous
converged design, and STARTING from the FEASIBLE 1021.6 kg headline
(multistart_v2_best.npz) with the rings pre-engaged (brace_radius = 20 mm). At
each g it sizes at SF 1.5, eigen-verifies (braced-aware), and — only if a step is
feasible+converged but the true eigen < 1.5 (closed-form non-conservative) —
bumps SF once to 2.0 and re-sizes that step. It STOPS at the first g that cannot
be made to pass (the survival boundary), reporting the full mass-vs-g curve.

Single worker (~18 GiB peak, memory-budget rule). Saves
runs/slam_braced_continuation.npz with the per-g history + the last passing
design.
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

# Ramp from a gentle slam up to the 3 g survival target. Warm-chained.
G_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0]
SF_BASE = 1.5          # standard buckling safety factor
SF_BUMP = 2.0          # single bump if a step is feas+conv but eigen < GATE
BRACE_START_M = 0.020  # pre-engage rings at 20 mm (foundation Pcr ~1.84x)


def accels_for(slam_g: float):
    return (
        (0.0, 0.0, -G0),
        (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)),
        (0.0, slam_g * G0, -G0),
        (0.0, -slam_g * G0, -G0),
    )


def main(smoke: bool = False) -> None:
    P = medium_scenario()
    spec = P.geometry
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
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

    def cfg_for(slam_g: float, sf: float) -> LaminateSizingConfig:
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
            tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=sf,
            use_analytic_jacobian=True, panel_width_mode="strip",
            accel_vectors=accels_for(slam_g), core=PVC_H80, ks_rho=50.0,
            optimizer="ipopt", beam_buckling_model="foundation",
            panel_d_mode="datum_ortho")

    # First step warm-starts from the FEASIBLE headline (no-slam) + engaged rings.
    head = np.asarray(np.load(RUNS / "multistart_v2_best.npz")["x"], dtype=float)
    lo, hi = laminate_design_bounds(model, cfg_for(G_LEVELS[0], SF_BASE))
    x0 = np.concatenate([head, np.array([BRACE_START_M])])
    assert x0.shape == lo.shape, f"x0 {x0.shape} != bounds {lo.shape} (headline block mismatch)"
    print(f"continuation: warm from headline ({head.shape[0]} vars) + brace_radius "
          f"{BRACE_START_M*1e3:.0f}mm → x0={x0.shape[0]} (bounds {lo.shape[0]})", flush=True)

    g_levels = G_LEVELS[:2] if smoke else G_LEVELS
    maxiter = 3 if smoke else 3000

    history: list[dict] = []
    x_warm = x0
    last_pass_r = None
    last_pass_cfg = None
    t_total = time.perf_counter()

    for slam_g in g_levels:
        passed_step = False
        for sf in (SF_BASE, SF_BUMP):
            cfg = cfg_for(slam_g, sf)
            print(f"\n[cont g={slam_g:g} SF={sf}] sizing ...", flush=True)
            t0 = time.perf_counter()
            r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY,
                                         rho=P.rho_kgm3, maxiter=maxiter,
                                         panel_pressures=press, x0=x_warm)
            wall = time.perf_counter() - t0
            feas = laminate_result_is_feasible(r, cfg)
            lam = eigen_worst_braced(model, loads, r, P.rho_kgm3, accels_for(slam_g))
            br_mm = (float(r.brace_radius) * 1e3
                     if getattr(r, "brace_radius", None) is not None else float("nan"))
            ok = bool(r.converged) and bool(feas) and lam >= GATE
            print(f"[cont g={slam_g:g} SF={sf}] mass={r.mass_kg:.1f} conv={r.converged} "
                  f"feas={feas} eigen={lam:.3f} brace_r={br_mm:.1f}mm ok={ok} "
                  f"wall={wall:.0f}s", flush=True)
            history.append(dict(slam_g=slam_g, sf=sf, mass_kg=float(r.mass_kg),
                                converged=bool(r.converged), feasible=bool(feas),
                                eigen_lam=float(lam), brace_r_mm=br_mm,
                                passed=ok, wall_s=float(wall), n_iter=int(r.n_iter)))
            if ok:
                passed_step = True
                last_pass_r, last_pass_cfg = r, cfg
                x_warm = design_vector_from_result(model, cfg, r)
                break
            # Only bump SF when feasible+converged but eigen short; otherwise bumping
            # SF (strictly harder) cannot help — abandon this g.
            if not (bool(r.converged) and bool(feas) and lam < GATE):
                print(f"[cont g={slam_g:g}] not feas/converged at SF={sf} — SF bump "
                      f"won't help; stopping ramp.", flush=True)
                break
            print(f"[cont g={slam_g:g}] feas+conv but eigen {lam:.3f}<{GATE}; "
                  f"bumping SF→{SF_BUMP}.", flush=True)
        if not passed_step:
            print(f"\n[cont] SURVIVAL BOUNDARY: g={slam_g:g} could not pass; "
                  f"highest passing g recorded below.", flush=True)
            break

    wall_total = time.perf_counter() - t_total
    passing = [h for h in history if h["passed"]]
    max_g = max((h["slam_g"] for h in passing), default=0.0)
    meta = dict(g_levels=g_levels, sf_base=SF_BASE, sf_bump=SF_BUMP, gate=GATE,
                brace_start_mm=BRACE_START_M * 1e3, history=history,
                max_passing_g=max_g, wall_s=float(wall_total),
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
    out: dict = {}
    if last_pass_r is not None:
        out = {k: getattr(last_pass_r, k) for k in
               ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands")}
        for k in ("r_tube", "t_wall", "t_hollow", "t_core", "brace_radius"):
            v = getattr(last_pass_r, k, None)
            if v is not None:
                out[k] = np.atleast_1d(v)
        meta["best_mass_kg"] = float(last_pass_r.mass_kg)
        meta["best_brace_r_mm"] = (float(last_pass_r.brace_radius) * 1e3
                                   if last_pass_r.brace_radius is not None else float("nan"))
        x_saved = design_vector_from_result(model, last_pass_cfg, last_pass_r)
        np.savez(RUNS / "slam_braced_continuation.npz", x=x_saved,
                 meta=json.dumps(meta), **out)
    else:
        np.savez(RUNS / "slam_braced_continuation.npz", meta=json.dumps(meta))
    print(f"\nCONTINUATION DONE max_passing_g={max_g:g} "
          f"curve={[(h['slam_g'], round(h['mass_kg'],1), round(h['eigen_lam'],2), h['passed']) for h in history]} "
          f"wall={wall_total:.0f}s", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
