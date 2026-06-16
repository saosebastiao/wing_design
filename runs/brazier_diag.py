"""Brazier-crush validity diagnostic for the 3g slam survival design.

Analysis only. Rebuilds the braced model + survival sections exactly as
slam_braced.eigen_worst_braced does, solves the beam-shell FEA under the full
4-accel slam envelope x all active aero cases, extracts the per-tube-segment
worst internal bending moment, and compares to the (isotropic-approximate)
Brazier ovalization critical moment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RUNS = Path(__file__).parent
sys.path.insert(0, str(RUNS))

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import build_beam_shell_model
from wing_design.beams.body_loads import body_load_vector
from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
from wing_design.beams.shell_model import skin_datum_angles
from wing_design.beams.shell_sizing import skin_band_map
from wing_design.materials.unidir import (
    PVC_H80, T700_EPOXY, laminate_stiffness_offset, sandwich_D_factor,
)
from wing_design.structural.beam_shell import solve_beam_shell_laminate
from wing_design.structural.frame import BeamSection

from chain_rebuild import DATUM, build_sections_from_result

G0 = 9.81
TH = np.radians(30.0)

NPZ = RUNS / "slam_survival_3g_20mm.npz"


def load_result(npz):
    d = np.load(npz, allow_pickle=True)
    def get(k):
        return np.asarray(d[k], dtype=float) if k in d.files else None
    r = SimpleNamespace(
        radii=get("radii"), t_bands=get("t_bands"),
        f0_bands=get("f0_bands"), f45_bands=get("f45_bands"), f90_bands=get("f90_bands"),
        r_tube=get("r_tube"), t_wall=get("t_wall"),
        t_hollow=get("t_hollow"), t_core=get("t_core"),
        brace_radius=(float(np.ravel(d["brace_radius"])[0]) if "brace_radius" in d.files else None),
    )
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return r, meta


def main():
    P = medium_scenario()
    spec = P.geometry
    r, meta = load_result(NPZ)
    slam_g = float(meta.get("slam_g", 3.0))

    accels = (
        (0.0, 0.0, -G0),
        (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)),
        (0.0, slam_g * G0, -G0),
        (0.0, -slam_g * G0, -G0),
    )
    accel_names = ["upright(0,0,-g)", "heel30", f"+{slam_g:g}g lat", f"-{slam_g:g}g lat"]

    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    model = build_beam_shell_model(
        spec, n_beams=16, n_levels=8, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m, core_tube=True, hollow_beams=True,
        lateral_bracing=True)

    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]

    # --- Skin laminate args (mirror eigen_worst_braced) ---
    band_of_tri = skin_band_map(model, 1)
    t_tri = r.t_bands[band_of_tri]
    th = skin_datum_angles(model, DATUM)
    mats = [laminate_stiffness_offset(T700_EPOXY, f0=float(r.f0_bands[0]),
                                      f45=float(r.f45_bands[0]), f90=float(r.f90_bands[0]),
                                      thickness=float(t_tri[e]),
                                      offset_deg=float(np.degrees(th[e])))
            for e in range(t_tri.shape[0])]
    A_arg = np.stack([m[0] for m in mats]); D_arg = np.stack([m[1] for m in mats])
    extra = None
    if r.t_core is not None:
        c_tri = np.asarray(r.t_core, dtype=float)[band_of_tri]
        D_arg = D_arg * sandwich_D_factor(t_tri, c_tri)[:, None, None]
        extra = PVC_H80.rho * c_tri

    sections, gusset, gusset_sec = build_sections_from_result(model, r)
    if (getattr(model, "brace_elements", None) is not None and r.brace_radius is not None):
        n_brace = len(model.brace_elements)
        sections += [BeamSection.circular(float(r.brace_radius))] * n_brace
    assert len(sections) == model.beam_elements.shape[0], (
        f"{len(sections)} != {model.beam_elements.shape[0]}")
    areas_e = np.array([s.A for s in sections])

    tube_el = np.asarray(model.tube_elements, dtype=int)
    S = len(tube_el)
    rho = P.rho_kgm3

    # worst bending moment per tube segment, and the driving (aero,accel) combo
    M_worst = np.zeros(S)
    M_drv = [("", "")] * S
    # also track tip vs root sanity (full bending vector for one representative combo)
    sanity_bending = None

    for ai, lc in enumerate(loads):
        for ki, acc in enumerate(accels):
            f = lc + body_load_vector(model, r.radii, t_tri, rho=rho, accel=acc,
                                      areas=areas_e, extra_skin_density=extra)
            res = solve_beam_shell_laminate(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
                fixed_nodes=model.fixed_nodes, loads=f,
                gusset_elements=gusset, gusset_section=gusset_sec)
            Mtube = np.asarray(res.bending_moment)[tube_el]
            if sanity_bending is None:
                sanity_bending = (Mtube.copy(), accel_names[ki])
            for s in range(S):
                if Mtube[s] > M_worst[s]:
                    M_worst[s] = Mtube[s]
                    M_drv[s] = (f"aero[{ai}]", accel_names[ki])
            # track the global worst-lateral combo for sanity print
            if accel_names[ki].endswith("lat") and Mtube.max() > 0:
                pass

    # --- Brazier critical moment per segment ---
    E = T700_EPOXY.isotropic_equivalent_modulus(knockdown=P.material_iso.skin_E_knockdown)
    nu = P.nu_iso
    r_t = np.asarray(r.r_tube, dtype=float)
    t_w = np.asarray(r.t_wall, dtype=float)
    M_cr = (2.0 * np.sqrt(2.0) / 9.0) * np.pi * E * r_t * t_w**2 / np.sqrt(1.0 - nu**2)
    rt_ratio = r_t / t_w
    margin = M_cr / np.maximum(M_worst, 1e-30)

    print("\n=== Brazier-crush diagnostic: 3g slam survival core tube ===")
    print(f"npz={NPZ.name}  slam_g={slam_g:g}  mass={meta.get('mass_kg'):.1f} kg")
    print(f"E_iso_equiv = {E/1e9:.2f} GPa (knockdown={P.material_iso.skin_E_knockdown}), nu={nu}")
    print(f"n_active_aero={len(loads)}  n_accels={len(accels)}  tube_segments={S}")
    print(f"tube_elements (rows into beam_elements): {tube_el.tolist()}")

    print("\n--- sanity: bending magnitude across tube segments, single combo "
          f"({sanity_bending[1]}) ---")
    print("  seg:  " + "  ".join(f"{m/1e3:8.1f}" for m in sanity_bending[0]) + "   (kN.m)")
    print(f"  root/tip ratio = {sanity_bending[0].max()/max(sanity_bending[0].min(),1e-9):.1f}x"
          f"  (max seg {int(np.argmax(sanity_bending[0]))}, min seg {int(np.argmin(sanity_bending[0]))})")

    print("\nseg |  r(mm) | t(mm) |  r/t  | M_dem(kN.m) | M_cr(kN.m) | margin | driver | RISK")
    print("-" * 92)
    worst_margin = np.inf
    worst_seg = -1
    for s in range(S):
        risk = "  <-- BRAZIER RISK" if margin[s] < 2.0 else ""
        if margin[s] < worst_margin:
            worst_margin = margin[s]; worst_seg = s
        print(f"{s:3d} | {r_t[s]*1e3:6.1f} | {t_w[s]*1e3:5.2f} | {rt_ratio[s]:5.1f} | "
              f"{M_worst[s]/1e3:11.2f} | {M_cr[s]/1e3:10.2f} | {margin[s]:6.2f} | "
              f"{M_drv[s][1]:>9s}{risk}")

    print("-" * 92)
    print(f"\nHEADLINE: worst-segment margin = {worst_margin:.2f} at segment {worst_seg} "
          f"(r/t={rt_ratio[worst_seg]:.1f}, r={r_t[worst_seg]*1e3:.0f}mm, t={t_w[worst_seg]*1e3:.1f}mm)")
    print(f"  driven by accel: {M_drv[worst_seg][1]} / {M_drv[worst_seg][0]}")
    verdict = "GOVERNS / THREATENS" if worst_margin < 2.0 else "does NOT govern"
    print(f"  VERDICT: Brazier crush {verdict} (threshold margin < 2.0)")


if __name__ == "__main__":
    main()
