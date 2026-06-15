"""3 g slam SURVIVAL sizing — failure-only (deflection/twist are NOT survival criteria).

Correction (user, 2026-06-15): the earlier slam runs sized the slam case against
the 2 %-span tip-deflection budget. That is wrong — deflection and twist are
SERVICEABILITY criteria (trim, clearance, aero) for NORMAL operation. A once-in-a-
lifetime 3 g wave slam is a SURVIVAL/ultimate case: all that matters is that the
structure does not FAIL — fibre/matrix fracture (CFRP is brittle, no yield, so
"permanent damage" = exceeding strength) or buckling COLLAPSE. Forcing operational
stiffness onto the slam case inflated the mass (the deflection constraint takes the
max over all load combos incl. slam — laminate_sizing.py:756).

This run keeps only the FAILURE constraints for the slam envelope:
  - strength: von-Mises stress <= allowable (fracture/damage)
  - stability: beam buckling (rings, foundation) + panel buckling + eigen >= 1.5
               (collapse), plus core wrinkle/crimp and tube-wall
and RELAXES deflection + twist to generous values. Deflection is not removed
entirely but capped at a GEOMETRIC-VALIDITY guardrail (5 % span) so the LINEAR FEA
(small-deflection assumption behind the stress/eigen solve) stays defensible; the
actual deflection is reported. Twist relaxed to 20 deg.

Rings pinned at brace_r_min = 20 mm (the eigen sweep showed ~14 mm clears the
gate; 20 mm gives lambda 3.16 = 2x margin). The result is the SLAM-FAILURE design;
the final wing is the envelope (max) of this and the operational 1021.6 kg headline
(which is sized for operational deflection/twist). IPOPT may not fully converge —
deliverable is the best hard-feasible incumbent, eigen-verified. Single worker.

Usage: python runs/slam_survival.py [G] [brace_min_mm] [defl_frac] [twist_deg]
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


def main(slam_g: float, brace_min_mm: float, defl_frac: float, twist_deg: float) -> None:
    tag = f"slam_survival_{slam_g:g}g_{brace_min_mm:g}mm"
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
    defl_guard = defl_frac * spec.span   # geometric-validity guardrail, NOT serviceability
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_guard,
        tip_twist_max_deg=twist_deg, ply_angle_datum=DATUM, buckling_safety_factor=1.5,
        use_analytic_jacobian=True, panel_width_mode="strip", accel_vectors=accels,
        core=PVC_H80, ks_rho=50.0, optimizer="ipopt",
        beam_buckling_model="foundation", panel_d_mode="datum_ortho",
        brace_r_min=brace_min, brace_r_max=0.06)
    print(f"[{tag}] FAILURE-ONLY: defl guardrail={defl_guard:.2f}m ({defl_frac*100:.0f}% span, "
          f"vs 2% serviceability), twist<={twist_deg:g}deg, rings>={brace_min_mm:g}mm, "
          f"buckling SF=1.5, eigen gate={GATE}", flush=True)

    head = np.asarray(np.load(RUNS / "multistart_v2_best.npz")["x"], dtype=float)
    lo, hi = laminate_design_bounds(model, cfg)
    x0 = np.concatenate([head, np.array([brace_min])])
    assert x0.shape == lo.shape, f"x0 {x0.shape} != bounds {lo.shape}"
    print(f"[{tag}] warm from headline + rings at floor (x0={x0.shape[0]})", flush=True)

    t0 = time.perf_counter()
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                 maxiter=3000, panel_pressures=press, x0=x0)
    wall = time.perf_counter() - t0
    feas = laminate_result_is_feasible(r, cfg)
    lam = eigen_worst_braced(model, loads, r, P.rho_kgm3, accels)
    br_mm = float(r.brace_radius) * 1e3 if r.brace_radius is not None else float("nan")
    survives = bool(feas) and lam >= GATE
    defl_m = float(getattr(r, "tip_defl_m", float("nan")))
    utils = dict(tip_twist=float(r.tip_twist_deg / cfg.tip_twist_max_deg),
                 tip_defl_pct_span=float(defl_m / spec.span * 100.0),
                 beam_buckling=float(r.max_beam_buckling_util),
                 panel_buckling=float(r.max_panel_buckling_util))
    meta = dict(slam_g=float(slam_g), brace_min_mm=float(brace_min_mm),
                defl_frac=float(defl_frac), twist_deg=float(twist_deg), gate=float(GATE),
                mass_kg=float(r.mass_kg), beam_mass_kg=float(r.beam_mass_kg),
                skin_mass_kg=float(r.skin_mass_kg), core_mass_kg=float(r.core_mass_kg),
                tube_mass_kg=float(r.tube_mass_kg), brace_mass_kg=float(r.brace_mass_kg),
                converged=bool(r.converged), feasible=bool(feas), eigen_lam=float(lam),
                survives=survives, brace_radius_mm=br_mm, tip_defl_m=defl_m,
                n_iter=int(r.n_iter), used_incumbent=bool(r.used_incumbent),
                wall_s=float(wall), utils=utils,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
    out = {k: getattr(r, k) for k in ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands")}
    for k in ("r_tube", "t_wall", "t_hollow", "t_core", "brace_radius"):
        v = getattr(r, k, None)
        if v is not None:
            out[k] = np.atleast_1d(v)
    np.savez(RUNS / f"{tag}.npz", x=design_vector_from_result(model, cfg, r),
             meta=json.dumps(meta), **out)
    print(f"SLAM_SURVIVAL DONE [{tag}] SURVIVES={survives} mass={r.mass_kg:.1f}kg "
          f"(beams {r.beam_mass_kg:.0f}+skin {r.skin_mass_kg:.0f}+core {r.core_mass_kg:.0f}"
          f"+tube {r.tube_mass_kg:.0f}+brace {r.brace_mass_kg:.0f}) conv={r.converged} "
          f"feas={feas} eigen={lam:.2f} brace_r={br_mm:.1f}mm tip_defl={defl_m:.2f}m "
          f"({utils['tip_defl_pct_span']:.1f}% span) utils={utils} wall={wall:.0f}s", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    g = float(args[0]) if args else 3.0
    bmm = float(args[1]) if len(args) > 1 else 20.0
    dfrac = float(args[2]) if len(args) > 2 else 0.05
    tdeg = float(args[3]) if len(args) > 3 else 20.0
    main(g, bmm, dfrac, tdeg)
