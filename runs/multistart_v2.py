"""Parallel cold multi-start exploration of the FINAL config (V#2-corrected).

Re-run of the reboot-lost big-budget push, now under panel_d_mode="datum_ortho":
8 independent starts (start 0 = cold default, 1-7 seeded-random in bounds) over
a process pool, IPOPT+KS, foundation model. Complements runs/chain_rebuild.py
(single warm trajectory): the chain regenerates/attributes, this hunts basins
the warm path can't reach. Best design saved to runs/multistart_v2_best.npz;
eigen-verified before any number is reported.

MEMORY BUDGET (hard rule): each IPOPT worker peaks ~18 GiB on the medium 16x8
model (MEASURED 2026-06-12: ru_maxrss 17.8 GiB, ~2x the first page-count
estimate); 8 workers oversubscribed the 32 GiB machine and caused the
2026-06-11/12 watchdog kernel panics (vm-compressor-space-shortage -> configd
jetsam-killed -> watchdogd panic). On 32 GiB the safe cap is n_workers=1 (two
coincident ~18 GiB peaks ~= 36 GiB > RAM); the 2-worker run survived only because
peaks did not coincide. This script must NOT run concurrently with
chain_rebuild.py or any other sizing run. Peak RSS recorded in the result meta.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

RUNS = Path(__file__).parent


def main() -> None:
    from wing_design import medium_scenario
    from wing_design.aero import build_airplane, sweep_envelope
    from wing_design.beams import build_beam_shell_model
    from wing_design.beams.fea_model import (
        panel_pressure_per_tri, project_panels_to_skin,
    )
    from wing_design.beams.laminate_sizing import (
        LaminateSizingConfig, design_vector_from_result,
        size_beam_shell_laminate_multistart,
    )
    from wing_design.materials.unidir import PVC_H80, T700_EPOXY
    from chain_rebuild import ACCELS, DATUM, SF, eigen_worst

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

    # Seed start 0 with the chain's stage-D optimum (same model + config -> the
    # saved x is directly usable); incumbent guard => never worse than 1094.2 kg.
    d = np.load(RUNS / "chain_D_datum_ortho.npz", allow_pickle=False)
    x0 = np.asarray(d["x"], dtype=float)
    print(f"seeding start 0 from chain_D ({json.loads(str(d['meta']))['mass_kg']:.1f} kg)",
          flush=True)

    t0 = time.perf_counter()
    ms = size_beam_shell_laminate_multistart(
        model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
        n_starts=8, n_workers=1, maxiter=3000, panel_pressures=press, x0=x0)
    wall = time.perf_counter() - t0
    r = ms.best
    lam = eigen_worst(model, loads, r, P.rho_kgm3)
    x = design_vector_from_result(model, cfg, r)
    out = {k: getattr(r, k) for k in
           ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands")}
    for k in ("r_tube", "t_wall", "t_hollow", "t_core"):
        v = getattr(r, k)
        if v is not None:
            out[k] = v
    import resource
    meta = dict(mass_kg=r.mass_kg, converged=bool(r.converged),
                used_incumbent=bool(r.used_incumbent), n_iter=int(r.n_iter),
                best_start=int(ms.best_start_index), n_feasible=int(ms.n_feasible),
                start_masses=list(ms.start_masses),
                start_feasible=list(ms.start_feasible),
                wall_s=float(wall), eigen_lam=float(lam),
                # macOS ru_maxrss is bytes; CHILDREN = max over pool workers
                peak_rss_self_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
                peak_rss_worker_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30)
    np.savez(RUNS / "multistart_v2_best.npz", x=x, meta=json.dumps(meta), **out)
    print(f"MULTISTART DONE {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
