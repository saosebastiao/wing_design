"""P.2 probe — does hoop winding pretension buy panel-buckling headroom? (no re-sizing)

Sizes the medium wingsail once (example-32 config), then superposes a parametric
chordwise (hoop) pretension on the recovered per-panel membrane stresses and
re-evaluates panel-buckling utilization — bounding the prestress prize before any
sizer work (backlog Performance #2 "cheap first experiment").

Direction caveat (audited 2026-06-09): the hoop wrap tensions the CHORD direction,
transverse to the mostly-spanwise bending compression, so two metrics are reported:
  (a) the implemented scalar check (most-compressive principal stress vs a single
      sigma_cr) on the net stress — expected to show ~no credit, demonstrating why a
      scalar offset cannot price prestress;
  (b) a direction-resolved linear biaxial interaction in the span/chord (datum)
      frame: util = SF*comp_span/sigma_cr_span - SF*tens_chord/sigma_cr_chord,
      with per-direction sigma_cr = kc*pi^2*D_ii*t^2... (plate D from the optimized
      band layup built directly in the datum frame), clamped at >= 0. Linear
      interaction is a first-order, roughly conservative plate approximation — the
      real answer is V.3's eigenvalue solve with a prestress load case.
Panel geometry stays the implemented b = sqrt(area) (known ~3x conservative in level,
backlog V#1) — the probe is COMPARATIVE in the pretension, not absolute.
"""
from __future__ import annotations

import time

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell_laminate,
)
from wing_design.beams.laminate_sizing import laminate_result_is_feasible
from wing_design.beams.shell_model import skin_datum_angles
from wing_design.beams.shell_sizing import skin_areas, skin_band_map
from wing_design.materials.unidir import T700_EPOXY, laminate_stiffness, laminate_stiffness_offset
from wing_design.structural.beam_shell import solve_beam_shell_laminate
from wing_design.structural.buckling import panel_buckling_utilization
from wing_design.structural.frame import BeamSection
from wing_design.structural.shell import recover_membrane_stress_C

PRETENSION_MPA = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]
SF = 1.5
KC = 4.0
DATUM = (0.0, 0.0, 1.0)


def rotate_stress_2d(s, theta):
    """Rotate plane-stress [sxx, syy, sxy] by +theta (rad): components in the frame
    whose x-axis is at +theta from the current frame's x-axis."""
    c, q = np.cos(theta), np.sin(theta)
    sxx, syy, sxy = s[..., 0], s[..., 1], s[..., 2]
    out = np.empty_like(s)
    out[..., 0] = c * c * sxx + q * q * syy + 2 * c * q * sxy
    out[..., 1] = q * q * sxx + c * c * syy - 2 * c * q * sxy
    out[..., 2] = -c * q * sxx + c * q * syy + (c * c - q * q) * sxy
    return out


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8

    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    model = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        ply_angle_datum=DATUM, buckling_safety_factor=SF, use_analytic_jacobian=True,
    )
    print("sizing the medium optimum (analytic Jacobian)...")
    t0 = time.perf_counter()
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=400)
    print(f"  mass={r.mass_kg:.1f} kg conv={r.converged} "
          f"feas={laminate_result_is_feasible(r, cfg)} wall={time.perf_counter()-t0:.0f}s "
          f"panel util={r.max_panel_buckling_util:.3f}")

    # Rebuild the optimum's per-triangle laminate exactly as the sizer does.
    B = cfg.n_skin_bands
    band_of_tri = skin_band_map(model, B)
    t_tri = r.t_bands[band_of_tri]
    f0_tri = r.f0_bands[band_of_tri]
    f45_tri = r.f45_bands[band_of_tri]
    f90_tri = r.f90_bands[band_of_tri]
    theta_tri = skin_datum_angles(model, DATUM)        # local-x -> span angle (rad)
    offs_deg = np.degrees(theta_tri)
    mats = [laminate_stiffness_offset(T700_EPOXY, f0=float(f0_tri[e]), f45=float(f45_tri[e]),
                                      f90=float(f90_tri[e]), thickness=float(t_tri[e]),
                                      offset_deg=float(o))
            for e, o in enumerate(offs_deg)]
    A_arg = np.stack([m[0] for m in mats])
    D_arg = np.stack([m[1] for m in mats])
    C_arg = np.stack([m[2] for m in mats])
    D11_local = D_arg[:, 0, 0]
    sections = [BeamSection.circular(float(x)) for x in r.radii]
    Atri = skin_areas(model)

    # Datum-frame plate D per triangle (layup built directly against span datum):
    # D11_datum = span-direction bending stiffness, D22_datum = chord.
    Dd = np.stack([laminate_stiffness(T700_EPOXY, f0=float(f0_tri[e]), f45=float(f45_tri[e]),
                                      f90=float(f90_tri[e]), thickness=float(t_tri[e]))[1]
                   for e in range(t_tri.shape[0])])
    sigcr_span = KC * np.pi**2 * Dd[:, 0, 0] / (Atri * t_tri)     # b^2 = area
    sigcr_chord = KC * np.pi**2 * Dd[:, 1, 1] / (Atri * t_tri)

    # Worst-case net stresses per panel over load cases, in the datum frame.
    s_datum_worst_span = np.full(t_tri.shape[0], np.inf)          # most-compressive span stress
    s_local_all = []
    for lc in loads:
        res = solve_beam_shell_laminate(
            model.nodes, model.beam_elements, sections, model.shell_tris,
            E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
            fixed_nodes=model.fixed_nodes, loads=lc)
        s_loc = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=C_arg)
        s_local_all.append(s_loc)

    print(f"\n  pretension sweep (chordwise/hoop tension superposed; {len(loads)} load cases):")
    print("   sigma_p   max util (a) scalar-principal   max util (b) biaxial-interaction")
    for sp_mpa in PRETENSION_MPA:
        sp_pa = sp_mpa * 1.0e6
        util_a = 0.0
        util_b = 0.0
        for s_loc in s_local_all:
            s_dat = rotate_stress_2d(np.asarray(s_loc), -theta_tri)  # local -> datum frame
            s_dat_net = s_dat.copy()
            s_dat_net[:, 1] += sp_pa                                  # hoop tension on chord axis
            # (a) implemented scalar check on the net stress, rotated back to local
            s_loc_net = rotate_stress_2d(s_dat_net, theta_tri)
            ua = panel_buckling_utilization(s_loc_net, Atri, D11=D11_local, t=t_tri,
                                            kc=KC, safety_factor=SF)
            util_a = max(util_a, float(ua.max()))
            # (b) biaxial linear interaction in the datum frame
            comp_span = np.maximum(0.0, -s_dat_net[:, 0])
            tens_chord = np.maximum(0.0, s_dat_net[:, 1])
            comp_chord = np.maximum(0.0, -s_dat_net[:, 1])
            ub = (SF * comp_span / sigcr_span
                  - SF * tens_chord / sigcr_chord
                  + SF * comp_chord / sigcr_chord)
            util_b = max(util_b, float(np.maximum(ub, 0.0).max()))
        print(f"   {sp_mpa:6.1f} MPa        {util_a:8.3f}                    {util_b:8.3f}")

    print("\n  Reading: if (b) drops materially with sigma_p while (a) doesn't, the hoop"
          "\n  pretension prize is real but invisible to the scalar check — sizing it needs"
          "\n  the biaxial check (P.2) or the eigenvalue solve (V.3). Panel b=sqrt(area)"
          "\n  caveat applies to absolute levels (V#1); compare columns DOWN, not across.")


if __name__ == "__main__":
    main()
