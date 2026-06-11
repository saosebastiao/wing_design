"""Unidirectional CFRP ply properties.

Values are typical for pultruded / RTM-produced UD CFRP at Vf ≈ 0.60. Replace
with data from the actual supplier when one is selected.

References: Hexcel HexPly data sheets; Toray T700/T800 prepreg data; Soden et al.
"Lamina properties, lay-up configurations and loading conditions for a range of
fibre-reinforced composite laminates" (1998).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UDPly:
    name: str
    E1_Pa: float              # longitudinal (fiber-direction) modulus
    E2_Pa: float              # transverse modulus
    G12_Pa: float             # in-plane shear modulus
    nu12: float               # major Poisson ratio
    Xt_Pa: float              # longitudinal tensile strength
    Xc_Pa: float              # longitudinal compressive strength
    Yt_Pa: float              # transverse tensile strength
    Yc_Pa: float              # transverse compressive strength
    S12_Pa: float             # in-plane shear strength
    rho_kgm3: float
    Vf: float = 0.60          # fiber volume fraction

    def isotropic_equivalent_modulus(self, knockdown: float = 0.5) -> float:
        """Lump E for an isotropic-equivalent beam: knockdown * E1.

        A 0.5 knockdown approximates a [0/±45/90] quasi-isotropic layup driven by
        the 0° plies along the spar axis. Use ~0.85 for a pure ±5° pultrusion.
        """
        return knockdown * self.E1_Pa

    def allowable_tensile_stress(self, safety_factor: float = 2.0) -> float:
        return self.Xt_Pa / safety_factor

    def allowable_compressive_stress(self, safety_factor: float = 2.0) -> float:
        return self.Xc_Pa / safety_factor


T700_EPOXY = UDPly(
    name="T700/Epoxy UD (pultruded, typical)",
    E1_Pa=135e9, E2_Pa=8.5e9, G12_Pa=4.5e9, nu12=0.32,
    Xt_Pa=2200e6, Xc_Pa=1200e6, Yt_Pa=60e6, Yc_Pa=200e6, S12_Pa=80e6,
    rho_kgm3=1550.0, Vf=0.60,
)


T800_EPOXY = UDPly(
    name="T800/Epoxy UD (prepreg, typical)",
    E1_Pa=165e9, E2_Pa=8.5e9, G12_Pa=5.0e9, nu12=0.32,
    Xt_Pa=2900e6, Xc_Pa=1500e6, Yt_Pa=60e6, Yc_Pa=210e6, S12_Pa=90e6,
    rho_kgm3=1580.0, Vf=0.60,
)


def reduced_stiffness_Q(ply: UDPly) -> np.ndarray:
    """Plane-stress reduced stiffness Q (3,3) in the ply material axes [Pa]."""
    nu21 = ply.nu12 * ply.E2_Pa / ply.E1_Pa
    denom = 1.0 - ply.nu12 * nu21
    Q11 = ply.E1_Pa / denom
    Q22 = ply.E2_Pa / denom
    Q12 = ply.nu12 * ply.E2_Pa / denom
    Q66 = ply.G12_Pa
    return np.array([[Q11, Q12, 0.0], [Q12, Q22, 0.0], [0.0, 0.0, Q66]])


def transformed_Qbar(Q: np.ndarray, angle_deg: float) -> np.ndarray:
    """Q rotated to laminate axes by ply `angle_deg` (3,3) [Pa]."""
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    c2, s2 = c * c, s * s
    c4, s4 = c2 * c2, s2 * s2
    s2c2 = s2 * c2
    Q11b = Q11 * c4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * s4
    Q22b = Q11 * s4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * c4
    Q12b = (Q11 + Q22 - 4.0 * Q66) * s2c2 + Q12 * (s4 + c4)
    Q66b = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * s2c2 + Q66 * (s4 + c4)
    Q16b = (Q11 - Q12 - 2.0 * Q66) * s * c * c2 + (Q12 - Q22 + 2.0 * Q66) * s * s2 * c
    Q26b = (Q11 - Q12 - 2.0 * Q66) * s * s2 * c + (Q12 - Q22 + 2.0 * Q66) * s * c * c2
    return np.array([[Q11b, Q12b, Q16b], [Q12b, Q22b, Q26b], [Q16b, Q26b, Q66b]])


def laminate_stiffness(
    ply: UDPly,
    *,
    f0: float,
    f45: float,
    f90: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric-balanced laminate (A, D, Qeff) from 0/±45/90 area fractions.

    `f45` is the total ±45 fraction (split equally +45/−45, so balanced → A16=A26=0).
    Qeff = f0·Qbar(0) + ½f45·(Qbar(45)+Qbar(−45)) + f90·Qbar(90)  (effective stiffness).
    A = thickness·Qeff (N/m). Smeared bending D = (thickness³/12)·Qeff (N·m): a
    fractions-only layup does not fix stacking order, so D uses the smeared model.
    """
    Q = reduced_stiffness_Q(ply)
    Qeff = (
        f0 * transformed_Qbar(Q, 0.0)
        + 0.5 * f45 * (transformed_Qbar(Q, 45.0) + transformed_Qbar(Q, -45.0))
        + f90 * transformed_Qbar(Q, 90.0)
    )
    A = thickness * Qeff
    D = (thickness**3 / 12.0) * Qeff
    return A, D, Qeff


def laminate_stiffness_offset(
    ply: UDPly, *, f0: float, f45: float, f90: float, thickness: float, offset_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like `laminate_stiffness` but with every ply angle shifted by `offset_deg`.

    Used to express a laminate defined against a global datum in an element's local
    frame whose x-axis sits `offset_deg` from that datum. With offset_deg=0 this is
    identical to `laminate_stiffness`. (Off-datum, a balanced laminate legitimately
    gains A16/A26 != 0 -- the shell element handles the full 3x3.)
    """
    Q = reduced_stiffness_Q(ply)
    o = offset_deg
    Qeff = (
        f0 * transformed_Qbar(Q, o)
        + 0.5 * f45 * (transformed_Qbar(Q, 45.0 + o) + transformed_Qbar(Q, -45.0 + o))
        + f90 * transformed_Qbar(Q, 90.0 + o)
    )
    A = thickness * Qeff
    D = (thickness**3 / 12.0) * Qeff
    return A, D, Qeff


# --- P.3 sandwich (cored) skin helpers --------------------------------------


@dataclass(frozen=True)
class CoreMaterial:
    """Structural core for the sandwich skin (P.3). Defaults ~PVC foam H80."""

    E_c: float = 80.0e6     # through-thickness/compressive modulus [Pa]
    G_c: float = 27.0e6     # shear modulus [Pa]
    rho: float = 80.0       # [kg/m^3]


PVC_H80 = CoreMaterial()

# smooth activation ramp width for the core failure checks: the checks vanish
# continuously as the core disappears (1 mm scale)
CORE_RAMP_M = 1.0e-3


def sandwich_D_factor(t: float | "np.ndarray", c: float | "np.ndarray"):
    """phi(t, c): multiply the monolithic laminate D by this for a symmetric
    sandwich with total face thickness t (split t/2 each side) and core c.

    D_sand = Qeff*(t^3/48 + (t/4)(c + t/2)^2) = D_mono * phi,
    phi = 1/4 + 3*(c/t + 1/2)^2;  phi(t, 0) = 1 exactly (monolithic recovered).
    """
    return 0.25 + 3.0 * (c / t + 0.5) ** 2


def core_ramp(c):
    """s(c) = c/(c + 1 mm): smooth 0-at-zero activation for core failure checks."""
    return c / (c + CORE_RAMP_M)


def wrinkling_stress(E_face: float | "np.ndarray", core: CoreMaterial):
    """Hoff face-wrinkling allowable: 0.5*(E_f*E_c*G_c)^(1/3) [Pa]."""
    return 0.5 * (E_face * core.E_c * core.G_c) ** (1.0 / 3.0)


def crimping_stress(t, c, core: CoreMaterial):
    """Shear-crimping allowable on the face membrane stress: G_c*(c + t)/t [Pa]."""
    return core.G_c * (c + t) / t
