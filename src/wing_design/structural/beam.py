"""Phase 3: tapered tube-spar sizing under the full aero load envelope.

The wingsail's primary load path in this baseline is a single tapered tube spar
running the length of the wing. The spar is treated as a cantilever beam fixed
at the root (z = 0) and free at the tip (z = span). All aero load cases are
added to one AeroSandbox `Opti` problem; the structural sizing (root + tip
diameter and root + tip wall thickness, linearly interpolated along the span)
is the only set of design variables. Mass is minimized subject to:

  * axial stress <= material allowable (UD tensile / compressive), per case
  * Euler column buckling margin >= 1, per case
  * tip deflection <= tip_defl_max_m, per case
  * spar fits inside the airfoil envelope (D <= thickness * chord)
  * gauge constraints on wall thickness

Notes / simplifications:

  - Material is treated as isotropic-equivalent: E = ply.isotropic_equivalent_modulus(knockdown),
    σ_allow = ply.allowable_tensile_stress(safety_factor). Real anisotropy enters in Phase 8.
  - Loads are the elliptic spanwise distribution scaled to AeroBuildup total lift.
  - The 0.75 m below-root spar stub is included only as a tare mass; its sizing
    (bearing reactions in the hull) is a later phase.
"""
from __future__ import annotations

from dataclasses import dataclass

import aerosandbox as asb
import aerosandbox.numpy as asbnp
from aerosandbox.structures.tube_spar_bending import TubeSparBendingStructure

from ..aero.loads import AeroResult
from ..geometry.wing import WingSpec
from ..materials.unidir import T700_EPOXY, UDPly


@dataclass(frozen=True)
class TubeSparSizing:
    material: UDPly
    diameter_root_m: float
    diameter_tip_m: float
    wall_root_m: float
    wall_tip_m: float
    mass_spar_kg: float
    mass_stub_kg: float
    mass_total_kg: float
    max_stress_Pa: float
    max_tip_deflection_m: float
    governing_case: str


def size_tube_spar(
    spec: WingSpec,
    aero_envelope: list[AeroResult],
    material: UDPly = T700_EPOXY,
    *,
    isotropic_knockdown: float = 0.50,
    safety_factor: float = 2.0,
    tip_defl_max_m: float | None = None,
    min_wall_m: float = 1.0e-3,
    max_wall_m: float = 2.0e-2,
    diameter_margin: float = 0.85,  # fraction of local airfoil thickness available to spar
    points_per_case: int = 80,
) -> TubeSparSizing:
    if not aero_envelope:
        raise ValueError("aero_envelope must contain at least one case")

    # Cases with negligible normal force break ASB's automatic variable scaling
    # (which divides by total load) and add nothing to the structural envelope.
    loaded_envelope = [r for r in aero_envelope if abs(r.factored_normal_force_N) > 1.0]
    if not loaded_envelope:
        raise ValueError("aero_envelope contains no cases with meaningful normal force")

    if tip_defl_max_m is None:
        tip_defl_max_m = 0.02 * spec.span  # 2% of span

    span = spec.span
    E = material.isotropic_equivalent_modulus(knockdown=isotropic_knockdown)
    sigma_allow = material.allowable_tensile_stress(safety_factor=safety_factor)

    # Maximum spar diameter that fits inside the NACA0018 envelope at each station
    def diameter_envelope_at(y):
        chord = spec.root_chord + (y / span) * (spec.tip_chord - spec.root_chord)
        return diameter_margin * spec.thickness * chord

    opti = asb.Opti()

    # Linearly-tapered design: 4 free variables (D_root, D_tip, t_root, t_tip)
    D_root = opti.variable(init_guess=0.08, lower_bound=0.005, upper_bound=diameter_envelope_at(0.0))
    D_tip = opti.variable(init_guess=0.04, lower_bound=0.005, upper_bound=diameter_envelope_at(span))
    t_root = opti.variable(init_guess=2.0e-3, lower_bound=min_wall_m, upper_bound=max_wall_m)
    t_tip = opti.variable(init_guess=2.0e-3, lower_bound=min_wall_m, upper_bound=max_wall_m)

    def diameter_fn(y):
        return D_root + (y / span) * (D_tip - D_root)

    def wall_fn(y):
        return t_root + (y / span) * (t_tip - t_root)

    # One TubeSparBendingStructure per load case, all sharing the same design vars
    beams: list = []
    for case_result in loaded_envelope:
        beam = TubeSparBendingStructure(
            opti=opti,
            length=span,
            diameter_function=diameter_fn,
            wall_thickness_function=wall_fn,
            elastic_modulus_function=E,
            bending_distributed_force_function=lambda y, r=case_result: r.distributed_normal_force(y),
            points_per_point_load=points_per_case,
            assume_thin_tube=True,
        )
        opti.subject_to(
            [
                beam.stress_axial <= sigma_allow,
                -beam.stress_axial <= material.allowable_compressive_stress(safety_factor),
                beam.u[-1] <= tip_defl_max_m,
                beam.u[-1] >= -tip_defl_max_m,
            ]
        )
        beams.append(beam)

    # Geometric constraint: spar fits inside the airfoil envelope at root and tip
    opti.subject_to(
        [
            D_root <= diameter_envelope_at(0.0),
            D_tip <= diameter_envelope_at(span),
            t_root <= 0.4 * D_root,
            t_tip <= 0.4 * D_tip,
        ]
    )

    spar_volume = beams[0].volume()  # same shape across cases
    mass_spar = spar_volume * material.rho_kgm3
    opti.minimize(mass_spar)

    sol = opti.solve(verbose=False)

    D_r, D_t = float(sol(D_root)), float(sol(D_tip))
    w_r, w_t = float(sol(t_root)), float(sol(t_tip))
    mass = float(sol(mass_spar))

    # Below-root stub mass: same root section, constant for spar_length
    stub_volume = asbnp.pi * D_r * w_r * spec.spar_length  # thin-tube approximation
    stub_mass = stub_volume * material.rho_kgm3

    # Walk the cases to find the governing one
    governing = loaded_envelope[0].case.name
    worst_stress = 0.0
    worst_defl = 0.0
    for case_result, beam in zip(loaded_envelope, beams):
        s_max = float(sol(asbnp.max(asbnp.abs(beam.stress_axial))))
        d_max = float(sol(asbnp.max(asbnp.abs(beam.u))))
        if s_max > worst_stress:
            worst_stress = s_max
            governing = case_result.case.name
        worst_defl = max(worst_defl, d_max)

    return TubeSparSizing(
        material=material,
        diameter_root_m=D_r,
        diameter_tip_m=D_t,
        wall_root_m=w_r,
        wall_tip_m=w_t,
        mass_spar_kg=mass,
        mass_stub_kg=stub_mass,
        mass_total_kg=mass + stub_mass,
        max_stress_Pa=worst_stress,
        max_tip_deflection_m=worst_defl,
        governing_case=governing,
    )
