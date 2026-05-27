"""Unidirectional CFRP ply properties.

Values are typical for pultruded / RTM-produced UD CFRP at Vf ≈ 0.60. Replace
with data from the actual supplier when one is selected.

References: Hexcel HexPly data sheets; Toray T700/T800 prepreg data; Soden et al.
"Lamina properties, lay-up configurations and loading conditions for a range of
fibre-reinforced composite laminates" (1998).
"""
from __future__ import annotations

from dataclasses import dataclass


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
