"""Print the project's current design scenario.

`wing_design.scenario.DesignParameters` is the single source of truth
for every parameter the Phase 4-5 pipeline consumes — geometry, material
model, mesh resolution, aero solver, frame-field parametrization, and
skin thickness. `default_scenario()` returns the working scenario
(the 5 m demo wingsail today); every other example does:

    from wing_design import default_scenario
    P = default_scenario()
    ...
    spec = P.geometry
    mesh = tet_mesh_wing(spec, target_element_size=P.mesh.target_element_size_m)

To change a parameter across the whole project, edit
`src/wing_design/scenario.py`. To run a one-off variant without
modifying the working scenario, `dataclasses.replace(P, …)` gives an
immutable copy with overrides.
"""
from __future__ import annotations

import dataclasses

from wing_design import default_scenario


def main() -> None:
    P = default_scenario()
    print("=" * 70)
    print("  wing_design — current scenario")
    print("=" * 70)

    print("\n# Geometry (WingSpec)")
    print(f"  span                     = {P.geometry.span:>8.3f} m")
    print(f"  root_chord               = {P.geometry.root_chord:>8.3f} m")
    print(f"  tip_chord                = {P.geometry.tip_chord:>8.3f} m")
    print(f"  thickness (NACA 00xx)    = {P.geometry.thickness:>8.3f} (t/c)")
    print(f"  pivot_frac               = {P.geometry.pivot_frac:>8.3f}")
    print(f"  spar_length              = {P.geometry.spar_length:>8.3f} m")
    print(f"  spar_diameter            = {P.geometry.spar_diameter:>8.3f} m")
    if P.geometry.taper_profile is not None:
        print(f"  taper_profile            = {P.geometry.taper_profile}")
    else:
        print(f"  taper_profile            = (linear: {P.geometry.root_chord} → {P.geometry.tip_chord})")

    print("\n# Material (UDPly)")
    print(f"  name                     = {P.material.name}")
    print(f"  E1, E2, G12              = {P.material.E1_Pa/1e9:.1f} / {P.material.E2_Pa/1e9:.1f} / {P.material.G12_Pa/1e9:.1f} GPa")
    print(f"  Xt, Xc                   = {P.material.Xt_Pa/1e6:.0f} / {P.material.Xc_Pa/1e6:.0f} MPa")
    print(f"  ρ                        = {P.material.rho_kgm3:.0f} kg/m³")

    print("\n# Material (isotropic-equivalent, used by Phase-4 FEA)")
    print(f"  skin_E_knockdown         = {P.material_iso.skin_E_knockdown:.2f}")
    print(f"  ν                        = {P.material_iso.nu_isotropic:.2f}")
    print(f"  safety_factor            = {P.material_iso.sigma_allow_safety_factor:.1f}")
    print(f"  → E_iso                  = {P.E_iso_Pa/1e9:.1f} GPa")
    print(f"  → G_iso                  = {P.G_iso_Pa/1e9:.1f} GPa")
    print(f"  → σ_allow                = {P.sigma_allow_Pa/1e6:.0f} MPa")

    print(f"\n# Load cases ({len(P.load_cases)} total)")
    for c in P.load_cases:
        print(f"  {c.name:<14} AWS = {c.airspeed_mps:>5.1f} m/s, α = {c.alpha_deg:>5.1f}°, SF = {c.safety_factor:.1f}")

    print(f"\n# Mesh")
    print(f"  target_element_size      = {P.mesh.target_element_size_m*1000:.0f} mm")

    print(f"\n# Aero solver")
    print(f"  spanwise_resolution      = {P.aero.spanwise_resolution} panels")

    print(f"\n# Phase-5 frame-field parametrization (volumetric track)")
    print(f"  n_levels                 = {P.frame_field.n_levels}")
    print(f"  sigma_floor_fraction     = {P.frame_field.sigma_floor_fraction}")
    print(f"  sigma_augment_fraction   = {P.frame_field.sigma_augment_fraction}")

    print(f"\n# Skin thickness (used by Phase 4b shell FEA)")
    print(f"  t_baseline               = {P.skin_sizing.t_baseline_m*1000:.1f} mm")

    print()
    print("To override a parameter for one example without changing the project")
    print("default, use `dataclasses.replace`:")
    print()
    print("    from wing_design import default_scenario")
    print("    import dataclasses")
    print("    P = dataclasses.replace(default_scenario(),")
    print("                            mesh=dataclasses.replace(default_scenario().mesh,")
    print("                                                     target_element_size_m=0.03))")


if __name__ == "__main__":
    main()
