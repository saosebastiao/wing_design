"""Re-introduce the slam load case at 3 g lateral (survival envelope) — V#12.

User decision (2026-06-14): after the Phase-P harvest (medium headline 1021.6 kg,
beam-buckling-governed at util 1.000), re-introduce the deferred slam case at
**3 g lateral (survival)**. The design must now survive the full envelope:

    upright gravity            (0, 0, -g)
    30 deg heel gravity        (0, g sin30, -g cos30)
    +3 g lateral slam + grav   (0, +3g, -g)
    -3 g lateral slam + grav   (0, -3g, -g)

(lateral = chord-normal y; span/up = z. Slam combines with 1 g vertical; both
signs because aero side-force breaks y-symmetry.) Static-equivalent slam, applied
in combination with each aero case via the existing body-load machinery, same as
the gravity cases — conservative (slam co-occurs with full design aero).

Warm-started from the headline (runs/multistart_v2_best.npz): the seed geometry
is reasonable but will be HARD infeasible under 3 g (the 1 g diagnostic already
went +110% / infeasible), so the incumbent guard has no feasible seed to fall
back on — this is a genuine feasibility probe. IPOPT+KS, single worker (~18 GiB
peak, safe on 32 GiB per the memory-budget rule). Saves runs/slam_3g.npz,
eigen-verified, with per-constraint utilizations so we can see what binds (or
what blows up) under slam.
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
    LaminateSizingConfig, design_vector_from_result, laminate_result_is_feasible,
    size_beam_shell_laminate,
)
from wing_design.materials.unidir import PVC_H80, T700_EPOXY
import chain_rebuild
from chain_rebuild import DATUM, SF, eigen_worst

G0 = 9.81
TH = np.radians(30.0)
SLAM_ACCELS = (
    (0.0, 0.0, -G0),
    (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)),
    (0.0, 3.0 * G0, -G0),
    (0.0, -3.0 * G0, -G0),
)
# eigen_worst() reads chain_rebuild.ACCELS as a module global; point it at the
# slam envelope so the eigen sweep matches what the sizer optimized against.
chain_rebuild.ACCELS = SLAM_ACCELS


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
        use_analytic_jacobian=True, panel_width_mode="strip", accel_vectors=SLAM_ACCELS,
        core=PVC_H80, ks_rho=50.0, optimizer="ipopt",
        beam_buckling_model="foundation", panel_d_mode="datum_ortho")

    x0 = np.asarray(np.load(RUNS / "multistart_v2_best.npz")["x"], dtype=float)
    print(f"warm-starting from headline (1021.6 kg), {len(SLAM_ACCELS)} accel cases "
          f"incl +/-3g lateral slam", flush=True)

    t0 = time.perf_counter()
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                 maxiter=3000, panel_pressures=press, x0=x0)
    wall = time.perf_counter() - t0
    feas = laminate_result_is_feasible(r, cfg)
    lam = eigen_worst(model, loads, r, P.rho_kgm3)

    lim_defl = 0.02 * spec.span
    utils = dict(
        tip_twist=float(r.tip_twist_deg / cfg.tip_twist_max_deg),
        tip_defl=float(getattr(r, "tip_defl_m", float("nan")) / lim_defl),
        beam_buckling=float(r.max_beam_buckling_util),
        panel_buckling=float(r.max_panel_buckling_util),
    )
    meta = dict(mass_kg=r.mass_kg, beam_mass_kg=r.beam_mass_kg,
                skin_mass_kg=r.skin_mass_kg, tube_mass_kg=r.tube_mass_kg,
                core_mass_kg=r.core_mass_kg, converged=bool(r.converged),
                used_incumbent=bool(r.used_incumbent), n_iter=int(r.n_iter),
                wall_s=float(wall), feasible=bool(feas), eigen_lam=float(lam),
                slam_g=3.0, n_accels=len(SLAM_ACCELS), utils=utils,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
    out = {k: getattr(r, k) for k in
           ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands")}
    for k in ("r_tube", "t_wall", "t_hollow", "t_core"):
        v = getattr(r, k)
        if v is not None:
            out[k] = v
    x_saved = design_vector_from_result(model, cfg, r)
    np.savez(RUNS / "slam_3g.npz", x=x_saved, meta=json.dumps(meta), **out)
    print(f"SLAM_3G DONE mass={r.mass_kg:.1f} kg conv={r.converged} feas={feas} "
          f"inc={r.used_incumbent} iters={r.n_iter} wall={wall:.0f}s eigen={lam:.2f} "
          f"utils={utils} layup f0/f45/f90="
          f"{r.f0_bands[0]:.2f}/{r.f45_bands[0]:.2f}/{r.f90_bands[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
