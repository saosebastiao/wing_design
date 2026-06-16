"""V.10 — mesh-converged re-baseline of the operational headline at n_levels=12.

The mesh-convergence diagnostic showed the 16x8 eigen VERIFICATION is
unconservatively biased: the 1021.6 kg headline's worst lambda falls 2.76 (N=8) ->
~1.3 (N>=12), BELOW the SF 1.5 it was sized to. The closed-form foundation
constraint is mesh-robust (so re-meshing alone keeps the mass ~the same and the
n=12 eigen ~1.3), therefore the fix is to SIZE AT n=12 with the buckling SF
laddered UP until the n=12 post-hoc eigen clears 1.5 — i.e. calibrate the
closed-form SF against the converged-mesh true eigenvalue. The mass at the passing
rung is the corrected, mesh-converged operational headline.

Warm-started from the 16x8 headline resampled to n=12 (radii via V.0.5
resample_segment_radii; tube/hollow via the diagnostic's spanwise interp; skin
bands are single-band -> mesh-independent). Single worker. IPOPT+KS, foundation +
sandwich + datum_ortho. Saves runs/mesh_rebaseline_n12.npz.
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
from wing_design.beams import build_beam_shell_model, resample_segment_radii
from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig, design_vector_from_result, laminate_design_bounds,
    laminate_result_is_feasible, size_beam_shell_laminate,
)
from wing_design.materials.unidir import PVC_H80, T700_EPOXY
from chain_rebuild import DATUM, eigen_worst
from mesh_converge_diag import Carrier, map_hollow_vector, resample_tube_segments

N_BEAMS = 16
N_FROM = 8
N_TO = 12
GATE = 1.5
SF_STEPS = [1.5, 1.65, 1.8, 2.0, 2.3]


def main() -> None:
    P = medium_scenario(); spec = P.geometry
    envelope = sweep_envelope(build_airplane(spec), P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    base_mk = dict(n_beams=N_BEAMS, beam_radius=0.02, material=P.material,
                   knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
                   skin_thickness=P.skin_sizing.t_baseline_m)
    model_from = build_beam_shell_model(spec, core_tube=True, hollow_beams=True,
                                        n_levels=N_FROM, **base_mk)
    model = build_beam_shell_model(spec, core_tube=True, hollow_beams=True,
                                   n_levels=N_TO, **base_mk)
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]
    press = [panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]

    # --- resample the 16x8 headline onto n=12 -> build x0 ---
    d = np.load(RUNS / "multistart_v2_best.npz")
    carrier = Carrier(
        radii=resample_segment_radii(d["radii"], n_beams=N_BEAMS, n_from=N_FROM, n_to=N_TO),
        t_bands=d["t_bands"], f0_bands=d["f0_bands"], f45_bands=d["f45_bands"],
        f90_bands=d["f90_bands"],
        r_tube=resample_tube_segments(d["r_tube"], N_FROM, N_TO),
        t_wall=resample_tube_segments(d["t_wall"], N_FROM, N_TO),
        t_hollow=map_hollow_vector(d["t_hollow"], model_from, model, N_FROM, N_TO),
        t_core=d["t_core"], brace_radius=None)

    def cfg_for(sf):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
            tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=sf,
            use_analytic_jacobian=True, panel_width_mode="strip",
            accel_vectors=((0.0, 0.0, -9.81),
                           (0.0, 9.81 * np.sin(np.radians(30.0)),
                            -9.81 * np.cos(np.radians(30.0)))),
            core=PVC_H80, ks_rho=50.0, optimizer="ipopt",
            beam_buckling_model="foundation", panel_d_mode="datum_ortho")

    x0 = design_vector_from_result(model, cfg_for(SF_STEPS[0]), carrier)
    lo, hi = laminate_design_bounds(model, cfg_for(SF_STEPS[0]))
    x0 = np.clip(x0, lo, hi)
    assert x0.shape == lo.shape, f"x0 {x0.shape} != bounds {lo.shape}"
    print(f"V.10 re-baseline at n_levels={N_TO}: warm from resampled 16x8 headline "
          f"(x0={x0.shape[0]}), SF ladder gated on n={N_TO} eigen >= {GATE}", flush=True)

    history = []
    x_warm = x0
    final_r = None
    final_lam = float("nan")
    t_total = time.perf_counter()
    for sf in SF_STEPS:
        cfg = cfg_for(sf)
        t0 = time.perf_counter()
        r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                     maxiter=2000, panel_pressures=press, x0=x_warm)
        wall = time.perf_counter() - t0
        feas = laminate_result_is_feasible(r, cfg)
        lam = eigen_worst(model, loads, r, P.rho_kgm3)   # TRUE eigen at n=12
        ok = bool(r.converged) and bool(feas) and lam >= GATE
        history.append(dict(sf=sf, mass_kg=float(r.mass_kg), converged=bool(r.converged),
                            feasible=bool(feas), n12_eigen=float(lam), passed=ok,
                            wall_s=float(wall), n_iter=int(r.n_iter)))
        print(f"[n12 SF={sf}] mass={r.mass_kg:.1f}kg conv={r.converged} feas={feas} "
              f"n12_eigen={lam:.3f} ok={ok} wall={wall:.0f}s", flush=True)
        final_r, final_lam = r, lam
        x_warm = design_vector_from_result(model, cfg, r)
        if ok:
            print(f"[n12] GATE PASSED at SF={sf}: n12_eigen={lam:.3f} >= {GATE}", flush=True)
            break
    wall_total = time.perf_counter() - t_total

    feas_final = laminate_result_is_feasible(final_r, cfg_for(history[-1]["sf"]))
    passed = bool(final_r.converged) and bool(feas_final) and final_lam >= GATE
    meta = dict(n_levels=N_TO, gate=GATE, sf_steps=SF_STEPS, history=history,
                mass_kg=float(final_r.mass_kg), n12_eigen=float(final_lam),
                converged=bool(final_r.converged), feasible=bool(feas_final),
                passed=passed, wall_s=float(wall_total),
                headline_16x8_kg=1021.6,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
    out = {k: getattr(final_r, k) for k in
           ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands")}
    for k in ("r_tube", "t_wall", "t_hollow", "t_core"):
        v = getattr(final_r, k, None)
        if v is not None:
            out[k] = v
    np.savez(RUNS / "mesh_rebaseline_n12.npz",
             x=design_vector_from_result(model, cfg_for(history[-1]["sf"]), final_r),
             meta=json.dumps(meta), **out)
    delta = (final_r.mass_kg - 1021.6) / 1021.6 * 100.0
    print(f"\nMESH_REBASELINE DONE n=12 PASSED={passed} mass={final_r.mass_kg:.1f}kg "
          f"(vs 16x8 headline 1021.6 -> {delta:+.1f}%) n12_eigen={final_lam:.2f} "
          f"SF={history[-1]['sf']} wall={wall_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
