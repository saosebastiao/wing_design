"""Re-evaluate the saved headline design and report which constraints bind.

Loads runs/multistart_v2_best.npz, rebuilds the final-config model/loads, and
warm-seeds the sizer at the saved optimum for a couple of iterations so the
returned result carries the optimum's constraint utilizations + KKT shadow
prices (nonzero multiplier => binding constraint). Read-only diagnostic.
"""
from __future__ import annotations

import sys
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
from chain_rebuild import ACCELS, DATUM, SF


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    model = build_beam_shell_model(spec, n_beams=16, n_levels=8, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m, core_tube=True, hollow_beams=True)
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]
    press = [panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
        tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=SF,
        use_analytic_jacobian=True, panel_width_mode="strip", accel_vectors=ACCELS,
        core=PVC_H80, ks_rho=50.0, optimizer="ipopt",
        beam_buckling_model="foundation", panel_d_mode="datum_ortho")

    x0 = np.asarray(np.load(RUNS / "multistart_v2_best.npz")["x"], dtype=float)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                 maxiter=0, panel_pressures=press, x0=x0)

    print(f"mass={r.mass_kg:.1f} kg  feasible={laminate_result_is_feasible(r, cfg)}")
    print("\n--- constraint utilizations (1.0 = at the limit) ---")
    lim_defl = 0.02 * spec.span
    print(f"  tip twist      : {r.tip_twist_deg:.3f} / {cfg.tip_twist_max_deg:.1f} deg "
          f"= {r.tip_twist_deg / cfg.tip_twist_max_deg:.3f}")
    if hasattr(r, "tip_defl_m"):
        print(f"  tip deflection : {r.tip_defl_m*1e3:.1f} / {lim_defl*1e3:.0f} mm "
              f"= {r.tip_defl_m / lim_defl:.3f}")
    print(f"  beam buckling  : util {r.max_beam_buckling_util:.3f} (SF {SF})")
    print(f"  panel buckling : util {r.max_panel_buckling_util:.3f} (SF {SF})")
    for extra in ("max_wrinkle_util", "max_crimp_util", "min_strength_ratio",
                  "max_stress_util", "tip_defl_max_m"):
        if hasattr(r, extra):
            print(f"  {extra:14s}: {getattr(r, extra)}")

    print("\n--- KKT shadow prices (nonzero => BINDING) ---")
    sp = getattr(r, "shadow_prices", None)
    if sp:
        for k, v in sorted(sp.items(), key=lambda kv: -abs(kv[1])):
            mark = "  <== binding" if abs(v) > 1e-3 else ""
            print(f"  {k:22s}: {v:+10.3f}{mark}")
    else:
        print("  (no shadow_prices on result)")


if __name__ == "__main__":
    main()
