"""Measure the V.9 (in-loop eigen) prize: mass vs TRUE eigenvalue on the headline.

P.6 found beam buckling is the sole binding lever (+1030.8 kg/SF) and the only
remaining harvest is the closed-form CONSERVATISM (design meets SF 1.5 while the
true eigen is 2.76 = 1.84x). V.9 (sizing to the true eigenvalue in-loop) would
recover that. This estimates the prize WITHOUT building V.9: re-size the headline
at decreasing closed-form buckling SF (warm-chained, SLSQP, fast from the optimum),
and at each point measure the POST-HOC true eigenvalue. The mass at the SF whose
true eigen ≈ 1.5 is roughly what in-loop-eigen sizing would deliver — the V.9
prize = headline 1021.6 kg minus that mass.

(Approximation: lowering the closed-form SF uniformly is not identical to sizing
the true mode shape, but it brackets the harvest. Single worker, ~minutes/point.)
"""
from __future__ import annotations

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
    LaminateSizingConfig, design_vector_from_result, laminate_result_is_feasible,
    size_beam_shell_laminate,
)
from wing_design.materials.unidir import PVC_H80, T700_EPOXY
from chain_rebuild import DATUM, eigen_worst   # no-slam 2-case ACCELS (chain_rebuild default)

SF_SWEEP = [1.5, 1.25, 1.0, 0.85, 0.7]


def main() -> None:
    P = medium_scenario(); spec = P.geometry
    envelope = sweep_envelope(build_airplane(spec), P.load_cases, method="lifting_line",
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

    def cfg_for(sf):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
            tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=sf,
            use_analytic_jacobian=True, panel_width_mode="strip",
            accel_vectors=((0.0, 0.0, -9.81),
                           (0.0, 9.81 * np.sin(np.radians(30.0)),
                            -9.81 * np.cos(np.radians(30.0)))),
            core=PVC_H80, ks_rho=50.0, optimizer="slsqp",
            beam_buckling_model="foundation", panel_d_mode="datum_ortho")

    x = np.asarray(np.load(RUNS / "multistart_v2_best.npz")["x"], dtype=float)
    curve = []
    for sf in SF_SWEEP:
        cfg = cfg_for(sf)
        t0 = time.perf_counter()
        r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                     maxiter=300, panel_pressures=press, x0=x)
        wall = time.perf_counter() - t0
        feas = laminate_result_is_feasible(r, cfg)
        lam = eigen_worst(model, loads, r, P.rho_kgm3)   # TRUE eigenvalue, no-slam
        curve.append((sf, float(r.mass_kg), bool(r.converged), bool(feas), float(lam)))
        print(f"[harvest] SF={sf:.2f} mass={r.mass_kg:.1f}kg conv={r.converged} "
              f"feas={feas} TRUE_eigen={lam:.2f} wall={wall:.0f}s", flush=True)
        x = design_vector_from_result(model, cfg, r)     # warm-chain next (lower) SF

    # interpolate the mass where true eigen == 1.5
    print("\n[harvest] curve (SF, mass, true_eigen):", flush=True)
    for sf, m, c, f, lam in curve:
        print(f"   SF={sf:.2f}  mass={m:7.1f}  eigen={lam:.2f}  {'conv+feas' if c and f else 'NOT-conv/feas'}")
    pts = [(lam, m) for sf, m, c, f, lam in curve if c and f]
    target = None
    for i in range(len(pts) - 1):
        (l0, m0), (l1, m1) = pts[i], pts[i + 1]
        if (l0 - 1.5) * (l1 - 1.5) <= 0 and l0 != l1:
            t = (1.5 - l0) / (l1 - l0)
            target = m0 + t * (m1 - m0)
            break
    if target is not None:
        print(f"\n[harvest] est. mass at TRUE eigen=1.5 ≈ {target:.0f} kg "
              f"→ V.9 in-loop-eigen prize ≈ {1021.6 - target:.0f} kg "
              f"({(1021.6 - target) / 1021.6 * 100:.0f}% of the headline)", flush=True)
    else:
        print("\n[harvest] true eigen did not cross 1.5 within the SF sweep "
              "(extend the range).", flush=True)
    print("HARVEST DONE", flush=True)


if __name__ == "__main__":
    main()
