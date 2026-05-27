"""Load-case envelope for the wingsail.

A `LoadCase` is a single aerodynamic operating point: apparent wind speed (AWS),
trim angle of attack relative to the apparent wind, altitude (atmosphere), and
the safety/gust factor to scale the resulting loads by before they're handed to
the structural sizing problem.

The default `DESIGN_CASES` list is sized for a 5 m demonstration wingsail on a
small sailing dinghy. Tune for your platform.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoadCase:
    name: str
    airspeed_mps: float        # apparent wind speed
    alpha_deg: float           # wingsail trim AoA
    altitude_m: float = 0.0
    safety_factor: float = 1.0
    description: str = ""


DESIGN_CASES: tuple[LoadCase, ...] = (
    LoadCase("nominal_trim",   airspeed_mps=10.0, alpha_deg=8.0,  description="10 m/s AWS, trimmed near best L/D"),
    LoadCase("max_aoa",        airspeed_mps=15.0, alpha_deg=12.0, description="Approaching stall at design AWS"),
    LoadCase("design_gust",    airspeed_mps=15.0, alpha_deg=8.0,  safety_factor=1.5, description="Trimmed + 1.5x gust factor"),
    LoadCase("survival",       airspeed_mps=25.0, alpha_deg=6.0,  description="Survival AWS, dumped AoA"),
    LoadCase("feathered",      airspeed_mps=25.0, alpha_deg=0.0,  description="Passively feathered in survival AWS"),
)
