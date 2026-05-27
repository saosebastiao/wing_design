"""Run AeroSandbox load cases and project the totals onto a spanwise distribution.

For the Phase 2 / 3 spike we use `AeroBuildup` (fast, differentiable, full
envelope sweep in one call) for the totals, and an analytic elliptic spanwise
distribution scaled to the total normal force for the load fed to the spar
sizer. This is a known simplification (the true distribution for a tapered
NACA0018 wing is close to but not exactly elliptic). Phase 4 will replace the
spanwise approximation with per-panel data from `LiftingLine`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import aerosandbox as asb
import numpy as np

from .cases import LoadCase


@dataclass(frozen=True)
class AeroResult:
    case: LoadCase
    span_m: float
    # All forces / moments are about the airplane xyz_ref (pivot at root), in wind axes
    # except where noted. Sign convention: positive lift = pushes wingsail to leeward.
    lift_N: float            # heel force on the boat
    drag_N: float
    side_N: float
    pitch_moment_Nm: float   # in body axes about pivot; "feather" moment for the wingsail
    roll_moment_Nm: float    # heel moment at the root (= base reaction)
    yaw_moment_Nm: float
    CL: float
    CD: float
    Cm: float                # pitching coeff about pivot
    dynamic_pressure_Pa: float

    @property
    def factored_normal_force_N(self) -> float:
        """Total spanwise-normal force after the case safety factor."""
        return self.lift_N * self.case.safety_factor

    def distributed_normal_force(self, y: np.ndarray) -> np.ndarray:
        """Elliptic spanwise distribution of normal force [N/m] at stations y in [0, span]."""
        L = self.factored_normal_force_N
        b = self.span_m
        eta = np.clip(y / b, 0.0, 1.0)
        return (4.0 * L / (np.pi * b)) * np.sqrt(np.maximum(1.0 - eta**2, 0.0))


def _scalar(x) -> float:
    """AeroBuildup wraps even scalar outputs in 1-D arrays; flatten safely."""
    return float(np.asarray(x).reshape(()))


def run_case(airplane: asb.Airplane, case: LoadCase) -> AeroResult:
    atm = asb.Atmosphere(altitude=case.altitude_m)
    op = asb.OperatingPoint(
        atmosphere=atm,
        velocity=case.airspeed_mps,
        alpha=case.alpha_deg,
        beta=0.0,
    )
    aero = asb.AeroBuildup(airplane=airplane, op_point=op).run()
    return AeroResult(
        case=case,
        span_m=float(airplane.b_ref),
        lift_N=_scalar(aero["L"]),
        drag_N=_scalar(aero["D"]),
        side_N=_scalar(aero["Y"]),
        pitch_moment_Nm=_scalar(aero["m_b"]),
        roll_moment_Nm=_scalar(aero["l_b"]),
        yaw_moment_Nm=_scalar(aero["n_b"]),
        CL=_scalar(aero["CL"]),
        CD=_scalar(aero["CD"]),
        Cm=_scalar(aero["Cm"]),
        dynamic_pressure_Pa=_scalar(0.5 * atm.density() * case.airspeed_mps**2),
    )


def sweep_envelope(airplane: asb.Airplane, cases: Iterable[LoadCase]) -> list[AeroResult]:
    return [run_case(airplane, c) for c in cases]
