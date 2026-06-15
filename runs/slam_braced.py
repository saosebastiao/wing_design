"""Eigen-gated, SF-bump braced slam run — V#12 lateral bracing variant.

Usage:  python runs/slam_braced.py [G]       (G = lateral slam g-level, default 3.0)
        python runs/slam_braced.py [G] --smoke   (quick smoke test: 1 rung, maxiter=3)

Sizes the medium wingsail against the full lateral-slam envelope WITH transverse
ring bracing (lateral_bracing=True) using:
  - closed-form elastic-foundation beam buckling model in-loop
  - post-hoc eigenvalue solve as an acceptance GATE (GATE = 1.5)
  - automated SF-bump ladder: stops at the first rung that is feasible + converged
    AND passes the eigen gate

Design envelope (identical to slam.py):
    upright gravity            (0,  0,       -g)
    30 deg heel gravity        (0,  g sin30, -g cos30)
    +G g lateral slam + grav   (0, +G*g,     -g)
    -G g lateral slam + grav   (0, -G*g,     -g)

Warm-started from runs/slam_<G>g.npz (unbraced 3g seed) with one brace_radius
appended; falls back to multistart_v2_best.npz if the per-g file is absent.

Saves runs/slam_braced_<G>g.npz with the x-vector, per-rung history, and meta.
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
from wing_design.beams.body_loads import body_load_vector
from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig, design_vector_from_result, laminate_design_bounds,
    laminate_result_is_feasible, size_beam_shell_laminate,
)
from wing_design.beams.shell_model import skin_datum_angles
from wing_design.beams.shell_sizing import beam_radius_groups, skin_band_map
from wing_design.materials.unidir import (
    PVC_H80, T700_EPOXY, laminate_stiffness_offset, sandwich_D_factor,
)
from wing_design.structural.beam_shell import (
    solve_beam_shell_laminate, solve_beam_shell_laminate_factored,
)
from wing_design.structural.eigen_buckling import linear_buckling
from wing_design.structural.frame import BeamSection
from wing_design.structural.geometric_stiffness import assemble_geometric_stiffness
from wing_design.structural.shell import recover_membrane_stress_C

# import chain_rebuild to reuse constants DATUM and SF
import chain_rebuild
from chain_rebuild import DATUM, SF, build_sections_from_result

G0 = 9.81
TH = np.radians(30.0)

GATE = 1.5
SF_STEPS = [1.5, 1.75, 2.0, 2.3]


def eigen_worst_braced(model, loads_aero, r, rho, accels):
    """Worst linear-buckling eigenvalue over (aero cases x accels) at the
    optimum, for a BRACED model. Mirrors chain_rebuild.eigen_worst but appends
    BeamSection.circular(r.brace_radius) for each brace element so the sections
    list length matches model.beam_elements.shape[0].

    Section order must match beam_elements rows: form → tube → brace.
    build_sections_from_result returns form + tube; we append brace here.
    """
    band_of_tri = skin_band_map(model, 1)
    t_tri = r.t_bands[band_of_tri]
    th = skin_datum_angles(model, DATUM)
    mats = [laminate_stiffness_offset(T700_EPOXY, f0=float(r.f0_bands[0]),
                                      f45=float(r.f45_bands[0]), f90=float(r.f90_bands[0]),
                                      thickness=float(t_tri[e]),
                                      offset_deg=float(np.degrees(th[e])))
            for e in range(t_tri.shape[0])]
    A_arg = np.stack([m[0] for m in mats]); D_arg = np.stack([m[1] for m in mats])
    C_arg = np.stack([m[2] for m in mats])
    extra = None
    if r.t_core is not None:
        c_tri = np.asarray(r.t_core, dtype=float)[band_of_tri]
        D_arg = D_arg * sandwich_D_factor(t_tri, c_tri)[:, None, None]
        extra = PVC_H80.rho * c_tri

    # Build sections: form + tube from helper, then append brace sections.
    sections, gusset, gusset_sec = build_sections_from_result(model, r)

    if (getattr(model, "brace_elements", None) is not None
            and getattr(r, "brace_radius", None) is not None):
        n_brace = len(model.brace_elements)
        brace_r = float(r.brace_radius)
        sections += [BeamSection.circular(brace_r)] * n_brace
        print(f"  [eigen] appended {n_brace} brace sections r={brace_r*1e3:.2f} mm "
              f"(total sections={len(sections)}, "
              f"beam_elements={model.beam_elements.shape[0]})", flush=True)

    assert len(sections) == model.beam_elements.shape[0], (
        f"Section count mismatch: {len(sections)} != {model.beam_elements.shape[0]}"
    )

    areas_e = np.array([s.A for s in sections])
    fac = solve_beam_shell_laminate_factored(
        model.nodes, model.beam_elements, sections, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
        fixed_nodes=model.fixed_nodes, loads=loads_aero[0],
        gusset_elements=gusset, gusset_section=gusset_sec)
    lmin = np.inf
    for lc in loads_aero:
        for acc in accels:
            f = lc + body_load_vector(model, r.radii, t_tri, rho=rho, accel=acc,
                                      areas=areas_e, extra_skin_density=extra)
            res = solve_beam_shell_laminate(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
                fixed_nodes=model.fixed_nodes, loads=f,
                gusset_elements=gusset, gusset_section=gusset_sec)
            s = recover_membrane_stress_C(model.nodes, model.shell_tris,
                                          res.displacements, C=C_arg)
            Kg = assemble_geometric_stiffness(
                model.nodes, beam_elements=model.beam_elements,
                axial_forces=res.axial_force, shell_tris=model.shell_tris,
                t_tri=t_tri, tri_stress_local=np.asarray(s))
            lams, _ = linear_buckling(fac.K_ff, Kg[np.ix_(fac.free, fac.free)], k=1)
            if lams.size:
                lmin = min(lmin, float(lams[0]))
    return lmin


def main(slam_g: float, smoke: bool = False) -> None:
    tag = f"slam_braced_{slam_g:g}g"
    accels = (
        (0.0, 0.0, -G0),
        (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)),
        (0.0, slam_g * G0, -G0),
        (0.0, -slam_g * G0, -G0),
    )

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

    def cfg_for(sf: float) -> LaminateSizingConfig:
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
            tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=sf,
            use_analytic_jacobian=True, panel_width_mode="strip", accel_vectors=accels,
            core=PVC_H80, ks_rho=50.0, optimizer="ipopt",
            beam_buckling_model="foundation", panel_d_mode="datum_ortho")

    # --- Warm start for the first rung ---
    seedp = RUNS / f"slam_{slam_g:g}g.npz"
    if not seedp.exists():
        seedp = RUNS / "multistart_v2_best.npz"
        print(f"[{tag}] per-g seed not found; falling back to multistart_v2_best.npz",
              flush=True)
    x_seed = np.asarray(np.load(seedp)["x"], dtype=float)
    cfg0 = cfg_for(SF_STEPS[0])
    lo, hi = laminate_design_bounds(model, cfg0)
    # Unbraced seed has no brace_radius entry; append midpoint of brace bounds.
    x0_candidate = np.concatenate(
        [x_seed, np.array([0.5 * (cfg0.brace_r_min + cfg0.brace_r_max)])])
    if x0_candidate.shape == lo.shape:
        x0 = x0_candidate
        print(f"[{tag}] warm-start from {seedp.name}: x_seed={x_seed.shape[0]} + "
              f"1 brace_radius → x0={x0.shape[0]} (matches bounds {lo.shape[0]})",
              flush=True)
    else:
        x0 = None
        print(f"[{tag}] WARNING: x0 shape {x0_candidate.shape} != bounds {lo.shape} "
              f"(seed block mismatch); falling back to cold braced start.", flush=True)

    sf_steps = SF_STEPS[:1] if smoke else SF_STEPS
    maxiter = 3 if smoke else 3000

    history: list[dict] = []
    x_rung: np.ndarray | None = x0
    final_r = None
    final_lam = float("nan")
    t_total = time.perf_counter()

    for i, sf in enumerate(sf_steps):
        cfg = cfg_for(sf)
        print(f"\n[{tag}] rung {i+1}/{len(sf_steps)} SF={sf} "
              f"{'(SMOKE)' if smoke else ''}", flush=True)

        t0 = time.perf_counter()
        r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY,
                                     rho=P.rho_kgm3, maxiter=maxiter,
                                     panel_pressures=press, x0=x_rung)
        wall_rung = time.perf_counter() - t0
        feas = laminate_result_is_feasible(r, cfg)

        print(f"[{tag}] SF={sf} solved in {wall_rung:.0f}s: mass={r.mass_kg:.1f} kg "
              f"conv={r.converged} feas={feas} iters={r.n_iter}", flush=True)

        # Post-hoc eigen gate (braced-aware)
        print(f"[{tag}] running eigen_worst_braced ...", flush=True)
        lam = eigen_worst_braced(model, loads, r, P.rho_kgm3, accels)
        print(f"[{tag}] SF={sf} eigen={lam:.3f} (gate={GATE})", flush=True)

        entry = dict(sf=sf, mass_kg=r.mass_kg, converged=bool(r.converged),
                     feasible=bool(feas), eigen_lam=float(lam),
                     n_iter=int(r.n_iter), wall_rung_s=float(wall_rung))
        history.append(entry)

        final_r = r
        final_lam = lam

        # Warm-restart next rung from this result
        x_rung = design_vector_from_result(model, cfg, r)

        # Stop if this rung satisfies all criteria
        if bool(r.converged) and bool(feas) and lam >= GATE:
            print(f"[{tag}] GATE PASSED at SF={sf}: eigen={lam:.3f} >= {GATE}", flush=True)
            break
        else:
            reasons = []
            if not r.converged:
                reasons.append("not converged")
            if not feas:
                reasons.append("infeasible")
            if lam < GATE:
                reasons.append(f"eigen {lam:.3f} < {GATE}")
            print(f"[{tag}] rung did not pass ({', '.join(reasons)}); "
                  f"{'bumping SF' if i+1 < len(sf_steps) else 'ladder exhausted'}",
                  flush=True)

    wall_total = time.perf_counter() - t_total

    # --- Save results ---
    # Use feasibility from the actual last cfg used
    last_cfg = cfg_for(history[-1]["sf"])
    feas_final = laminate_result_is_feasible(final_r, last_cfg)
    passed = bool(final_r.converged) and bool(feas_final) and final_lam >= GATE

    brace_r_mm = (float(final_r.brace_radius) * 1e3
                  if getattr(final_r, "brace_radius", None) is not None else float("nan"))

    lim_defl = 0.02 * spec.span
    utils = dict(
        tip_twist=float(final_r.tip_twist_deg / last_cfg.tip_twist_max_deg),
        tip_defl=float(getattr(final_r, "tip_defl_m", float("nan")) / lim_defl),
        beam_buckling=float(final_r.max_beam_buckling_util),
        panel_buckling=float(final_r.max_panel_buckling_util),
    )
    meta = dict(
        slam_g=float(slam_g), gate=float(GATE), sf_steps=SF_STEPS,
        history=history,
        final=dict(sf=history[-1]["sf"], mass_kg=final_r.mass_kg,
                   converged=bool(final_r.converged), feasible=bool(feas_final),
                   eigen_lam=float(final_lam)),
        brace_radius_mm=brace_r_mm,
        wall_s=float(wall_total),
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
        n_accels=len(accels), utils=utils,
    )

    out = {k: getattr(final_r, k) for k in
           ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands")}
    for k in ("r_tube", "t_wall", "t_hollow", "t_core"):
        v = getattr(final_r, k)
        if v is not None:
            out[k] = v
    if getattr(final_r, "brace_radius", None) is not None:
        out["brace_radius"] = final_r.brace_radius
    x_saved = design_vector_from_result(model, last_cfg, final_r)
    np.savez(RUNS / f"{tag}.npz", x=x_saved, meta=json.dumps(meta), **out)
    print(f"\nSLAM_BRACED DONE [{slam_g:g}g] PASSED={passed} "
          f"final={json.dumps(meta['final'])} "
          f"brace_r={brace_r_mm:.2f}mm wall={wall_total:.0f}s", flush=True)


if __name__ == "__main__":
    _smoke = "--smoke" in sys.argv
    _args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(float(_args[0]) if _args else 3.0, smoke=_smoke)
