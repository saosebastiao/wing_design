"""P.6 — re-extract shadow prices on the current optimum (the next-lever read).

The old "beam-buckling SF is the dominant lever (+531 kg/SF)" ranking (V.6/beam-
autopsy) is SPENT — the foundation model harvested it. This re-pulls the KKT
shadow prices on the CURRENT headline so we know the genuinely-next lever before
more P-phase work. Shadow prices are SLSQP-only (IPOPT multipliers unwired), so
this runs SLSQP WARM from the saved IPOPT optimum (it's already optimal, so it
should converge fast) and reports `result.shadow_prices` (kg per unit limit / per
SF unit, FD-validated by V.1), the binding utilizations, and the mass breakdown.

Usage: python runs/shadow_prices.py [npz_name]   (default multistart_v2_best.npz)
"""
from __future__ import annotations

import json
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
    LaminateSizingConfig, laminate_result_is_feasible, size_beam_shell_laminate,
)
from wing_design.materials.unidir import PVC_H80, T700_EPOXY
from chain_rebuild import DATUM


def main(npz_name: str) -> None:
    d = np.load(RUNS / npz_name, allow_pickle=False)
    x0 = np.asarray(d["x"], dtype=float)
    meta = json.loads(str(d["meta"]))
    braced = "brace_radius" in d.files or "brace_radius_mm" in meta
    print(f"loaded {npz_name}: mass={meta.get('mass_kg', float('nan')):.1f} kg "
          f"braced={braced}", flush=True)

    P = medium_scenario(); spec = P.geometry
    envelope = sweep_envelope(build_airplane(spec), P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    model = build_beam_shell_model(spec, n_beams=16, n_levels=8, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m, core_tube=True, hollow_beams=True,
        lateral_bracing=braced)
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]
    press = [panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]

    # SLSQP (for KKT multipliers → shadow prices), warm from the IPOPT optimum.
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
        tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=1.5,
        use_analytic_jacobian=True, panel_width_mode="strip",
        accel_vectors=((0.0, 0.0, -9.81),
                       (0.0, 9.81 * np.sin(np.radians(30.0)),
                        -9.81 * np.cos(np.radians(30.0)))),
        core=PVC_H80, ks_rho=50.0, optimizer="slsqp",
        beam_buckling_model="foundation", panel_d_mode="datum_ortho")

    t0 = time.perf_counter()
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                 maxiter=300, panel_pressures=press, x0=x0)
    wall = time.perf_counter() - t0
    feas = laminate_result_is_feasible(r, cfg)
    lim_defl = 0.02 * spec.span
    print(f"\nSLSQP warm settle: mass={r.mass_kg:.1f} kg conv={r.converged} feas={feas} "
          f"iters={r.n_iter} wall={wall:.0f}s", flush=True)
    print("mass breakdown: beams %.0f + skin %.0f + core %.0f + tube %.0f + brace %.0f" % (
        r.beam_mass_kg, r.skin_mass_kg, r.core_mass_kg, r.tube_mass_kg,
        getattr(r, "brace_mass_kg", 0.0)), flush=True)
    print("\n--- binding utilizations (1.0 = at the limit) ---", flush=True)
    print(f"  tip twist      : {r.tip_twist_deg / cfg.tip_twist_max_deg:.3f}")
    if hasattr(r, "tip_defl_m"):
        print(f"  tip deflection : {r.tip_defl_m / lim_defl:.3f}")
    print(f"  beam buckling  : {r.max_beam_buckling_util:.3f}")
    print(f"  panel buckling : {r.max_panel_buckling_util:.3f}")
    print("\n--- SHADOW PRICES (kg per unit limit / per SF unit; nonzero ⇒ binding) ---",
          flush=True)
    sp = getattr(r, "shadow_prices", None)
    if sp:
        for k, v in sorted(sp.items(), key=lambda kv: -abs(kv[1])):
            mark = "  <== binding" if abs(v) > 1e-3 else ""
            print(f"  {k:26s}: {v:+11.3f}{mark}")
    else:
        print("  (no shadow_prices — SLSQP did not attach multipliers; "
              "did it converge?)")
    print("SHADOW DONE", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args[0] if args else "multistart_v2_best.npz")
