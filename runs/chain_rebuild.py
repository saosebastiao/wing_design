"""Regenerate the medium-config warm-start chain under the V#2-corrected model.

The 2026-06-11 reboot wiped /tmp: both in-flight runs and every saved design
array (including the V.3c running best) were lost. This chain rebuilds them —
and since everything re-runs anyway, it re-baselines with panel_d_mode=
"datum_ortho" (V#2 close) as an attributed final stage:

  A0 tube_only    SLSQP cold      (ex-50 reference: 2060.7 kg)
  A tube_hollow   SLSQP warm A0   (ex-51 reference: 1784.5 kg corrected;
                                   cold start is eigen-REJECTED at λ 0.91 —
                                   P#1b record, reproduced 2026-06-11)
  B sandwich      IPOPT+KS warm A (P.3 reference:   ~1679 kg, arrays lost)
  C foundation    IPOPT+KS warm B (V.3c reference:  1095.3 kg corrected)
  D datum_ortho   IPOPT+KS warm C (V#2 measurement: new)

Each stage saves runs/chain_<tag>.npz immediately (reboot-proof, resumable:
existing npz = stage skipped and reloaded) and is eigen-verified. Wall times
measured per house rules.
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import build_beam_shell_model, size_beam_shell_laminate
from wing_design.beams.body_loads import body_load_vector
from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig, design_vector_from_result, laminate_result_is_feasible,
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

G0 = 9.81
TH = np.radians(30.0)
ACCELS = ((0.0, 0.0, -G0), (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)))
SF = 1.5
DATUM = (0.0, 0.0, 1.0)
RUNS = Path(__file__).parent


def build_sections_from_result(model, r):
    if r.t_hollow is not None:
        goe, _G = beam_radius_groups(model)
        hgroups = sorted({int(goe[e]) for e in model.hollow_elements})
        tmap = {g: i for i, g in enumerate(hgroups)}
        hset = set(int(e) for e in model.hollow_elements)
        sections = []
        for e, rr in enumerate(r.radii):
            if e in hset:
                t_e = min(float(r.t_hollow[tmap[int(goe[e])]]), float(rr))
                sections.append(BeamSection.annular(float(rr), t_e))
            else:
                sections.append(BeamSection.circular(float(rr)))
    else:
        sections = [BeamSection.circular(float(x)) for x in r.radii]
    gusset = gusset_sec = None
    if r.r_tube is not None:
        sections += [BeamSection.annular(float(r.r_tube[s]), float(r.t_wall[s]))
                     for s in range(len(r.r_tube))]
        gusset, gusset_sec = model.tube_bond_elements, BeamSection.circular(0.05)
    return sections, gusset, gusset_sec


def eigen_worst(model, loads_aero, r, rho):
    """Worst linear-buckling eigenvalue over (aero cases x accels) at the
    optimum, sandwich-aware (D x phi, core density in body loads)."""
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
    sections, gusset, gusset_sec = build_sections_from_result(model, r)
    areas_e = np.array([s.A for s in sections])
    fac = solve_beam_shell_laminate_factored(
        model.nodes, model.beam_elements, sections, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
        fixed_nodes=model.fixed_nodes, loads=loads_aero[0],
        gusset_elements=gusset, gusset_section=gusset_sec)
    lmin = np.inf
    for lc in loads_aero:
        for acc in ACCELS:
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


def save_stage(tag, r, x, wall_s, feas, lam):
    out = {k: getattr(r, k) for k in
           ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands")}
    for k in ("r_tube", "t_wall", "t_hollow", "t_core"):
        v = getattr(r, k)
        if v is not None:
            out[k] = v
    meta = dict(mass_kg=r.mass_kg, beam_mass_kg=r.beam_mass_kg,
                skin_mass_kg=r.skin_mass_kg, tube_mass_kg=r.tube_mass_kg,
                core_mass_kg=r.core_mass_kg, converged=bool(r.converged),
                used_incumbent=bool(r.used_incumbent), n_iter=int(r.n_iter),
                wall_s=float(wall_s), feasible=bool(feas), eigen_lam=float(lam),
                # macOS ru_maxrss is bytes; memory budget rule: see multistart_v2
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
    np.savez(RUNS / f"chain_{tag}.npz", x=x, meta=json.dumps(meta), **out)
    print(f"[{tag}] SAVED {meta}", flush=True)


def main(smoke: bool = False) -> None:
    P = medium_scenario()
    spec = P.geometry
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    mk = dict(n_beams=16, n_levels=8, beam_radius=0.02, material=P.material,
              knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
              skin_thickness=P.skin_sizing.t_baseline_m)
    model_t = build_beam_shell_model(spec, core_tube=True, hollow_beams=False, **mk)
    model = build_beam_shell_model(spec, core_tube=True, hollow_beams=True, **mk)
    # identical meshes (hollow is a section property) — loads shared
    assert model_t.nodes.shape == model.nodes.shape
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]
    press = [panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]

    base = dict(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
                tip_twist_max_deg=5.0, ply_angle_datum=DATUM,
                buckling_safety_factor=SF, use_analytic_jacobian=True,
                panel_width_mode="strip", accel_vectors=ACCELS)
    stages = [
        ("A0_tube_only", model_t, LaminateSizingConfig(**base), 500),
        ("A_tube_hollow", model, LaminateSizingConfig(**base), 500),
        ("B_sandwich", model, LaminateSizingConfig(**base, core=PVC_H80, ks_rho=50.0,
                                                   optimizer="ipopt"), 2000),
        ("C_foundation", model, LaminateSizingConfig(**base, core=PVC_H80, ks_rho=50.0,
                                                     optimizer="ipopt",
                                                     beam_buckling_model="foundation"),
         2000),
        ("D_datum_ortho", model, LaminateSizingConfig(**base, core=PVC_H80, ks_rho=50.0,
                                                      optimizer="ipopt",
                                                      beam_buckling_model="foundation",
                                                      panel_d_mode="datum_ortho"),
         2000),
    ]

    prev = None
    for tag, m_st, cfg, maxiter in stages:
        if smoke:
            tag, maxiter = "smoke_" + tag, 3
        f = RUNS / f"chain_{tag}.npz"
        if f.exists():
            d = np.load(f, allow_pickle=False)
            print(f"[{tag}] exists — reusing ({json.loads(str(d['meta']))['mass_kg']:.1f} kg)",
                  flush=True)

            class _R:                                # minimal warm-start carrier
                pass
            prev = _R()
            for k in ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands"):
                setattr(prev, k, d[k])
            for k in ("r_tube", "t_wall", "t_hollow", "t_core"):
                setattr(prev, k, d[k] if k in d.files else None)
            continue
        x0 = design_vector_from_result(m_st, cfg, prev) if prev is not None else None
        t0 = time.perf_counter()
        r = size_beam_shell_laminate(m_st, loads, cfg, ply=T700_EPOXY,
                                     rho=P.rho_kgm3, maxiter=maxiter,
                                     panel_pressures=press, x0=x0)
        wall = time.perf_counter() - t0
        feas = laminate_result_is_feasible(r, cfg)
        lam = eigen_worst(m_st, loads, r, P.rho_kgm3)
        x_saved = design_vector_from_result(m_st, cfg, r)
        save_stage(tag, r, x_saved, wall, feas, lam)
        print(f"[{tag}] mass={r.mass_kg:8.1f} kg (beams {r.beam_mass_kg:.0f} + skin "
              f"{r.skin_mass_kg:.0f} + core {r.core_mass_kg:.0f} + tube "
              f"{r.tube_mass_kg:.0f}) conv={r.converged} feas={feas} "
              f"inc={r.used_incumbent} iters={r.n_iter} wall={wall:.0f}s "
              f"eigen={lam:.2f} layup f0/f45/f90 = "
              f"{r.f0_bands[0]:.2f}/{r.f45_bands[0]:.2f}/{r.f90_bands[0]:.2f}",
              flush=True)
        prev = r

    print("CHAIN DONE", flush=True)


if __name__ == "__main__":
    import sys
    main(smoke="--smoke" in sys.argv)
