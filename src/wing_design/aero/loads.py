"""Run AeroSandbox load cases and expose totals + per-panel loads.

Two solvers are wired in:

  * `run_case` uses `AeroBuildup` — fast, full-envelope sweeps, totals only.
  * `run_case_lifting_line` uses `LiftingLine` — slower, but exposes the
    per-panel force vector that we need for (a) span-resolved beam loading and
    (b) surface-traction boundary conditions in Phase 4's volumetric FEA.

The spanwise normal-force distribution `AeroResult.distributed_normal_force(y)`
prefers the LL panel data if present and falls back to an analytic elliptic
distribution scaled to total lift otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import aerosandbox as asb
import numpy as np

from .cases import LoadCase


@dataclass(frozen=True)
class PanelLoads:
    """Per-panel Kutta-Joukowski force vectors from a `LiftingLine` solve.

    All quantities are in the airplane (geometry) frame: chord along +X, span
    along +Y, airfoil normal along +Z. The `forces_xyz` vectors are the total
    force on each panel in Newtons (lift component is +Z; induced drag is +X).
    """

    centers_xyz: np.ndarray         # (N, 3) panel centroids
    areas: np.ndarray               # (N,)
    forces_xyz: np.ndarray          # (N, 3) total force per panel [N]
    normals_xyz: np.ndarray         # (N, 3) panel outward normals (unit)
    chords: np.ndarray              # (N,) local chord at panel
    spanwise_widths: np.ndarray     # (N,) panel extent along +Y

    @property
    def n_panels(self) -> int:
        return int(self.centers_xyz.shape[0])

    @property
    def total_force(self) -> np.ndarray:
        return self.forces_xyz.sum(axis=0)


def _compute_panel_loads(ll: asb.LiftingLine) -> PanelLoads:
    """Kutta-Joukowski per panel: F = ρ Γ (V_∞ × dl_bound), applied at panel centroid."""
    rho = float(ll.op_point.atmosphere.density())
    V_inf = np.asarray(ll.freestream_velocities)        # (N, 3)
    bound = np.asarray(ll.vortex_bound_leg)             # (N, 3)
    Gamma = np.asarray(ll.vortex_strengths).ravel()     # (N,)
    forces = rho * Gamma[:, None] * np.cross(V_inf, bound)
    centers = 0.25 * (
        np.asarray(ll.front_left_vertices)
        + np.asarray(ll.front_right_vertices)
        + np.asarray(ll.back_left_vertices)
        + np.asarray(ll.back_right_vertices)
    )
    return PanelLoads(
        centers_xyz=centers,
        areas=np.asarray(ll.areas).astype(float),
        forces_xyz=forces.astype(float),
        normals_xyz=np.asarray(ll.normal_directions).astype(float),
        chords=np.asarray(ll.chords).astype(float),
        spanwise_widths=np.abs(bound[:, 1]).astype(float),
    )


@dataclass(frozen=True)
class AeroResult:
    case: LoadCase
    span_m: float
    lift_N: float                       # +Z body force (perpendicular to chord, away from low-pressure side)
    drag_N: float
    side_N: float
    pitch_moment_Nm: float              # about pivot, body axes
    roll_moment_Nm: float                # base reaction at the root
    yaw_moment_Nm: float
    CL: float
    CD: float
    Cm: float
    dynamic_pressure_Pa: float
    panels: PanelLoads | None = None    # populated by LiftingLine, None for AeroBuildup

    @property
    def factored_normal_force_N(self) -> float:
        """Total airfoil-normal force (lift) after the case safety factor."""
        return self.lift_N * self.case.safety_factor

    def distributed_normal_force(self, y: np.ndarray) -> np.ndarray:
        """Spanwise normal-force density [N/m] at stations y ∈ [0, span].

        Uses LiftingLine per-panel data when available (piecewise-linear over
        panel centers, zero at the tips); otherwise an analytic elliptic
        distribution scaled to total lift. Scales by the case safety factor.
        """
        y = np.asarray(y, dtype=float)
        sf = self.case.safety_factor
        if self.panels is not None:
            p = self.panels
            q_panel = sf * p.forces_xyz[:, 2] / np.maximum(p.spanwise_widths, 1e-12)
            order = np.argsort(p.centers_xyz[:, 1])
            y_sorted = p.centers_xyz[order, 1]
            q_sorted = q_panel[order]
            # Linear interp between panel centers, force-to-zero outside (tips)
            return np.interp(y, y_sorted, q_sorted, left=0.0, right=0.0)
        L = self.factored_normal_force_N
        b = self.span_m
        eta = np.clip(y / b, 0.0, 1.0)
        return (4.0 * L / (np.pi * b)) * np.sqrt(np.maximum(1.0 - eta**2, 0.0))


def _scalar(x) -> float:
    """ASB sometimes wraps scalars in 1-D arrays; flatten safely."""
    return float(np.asarray(x).reshape(()))


def _operating_point(case: LoadCase) -> tuple[asb.Atmosphere, asb.OperatingPoint]:
    atm = asb.Atmosphere(altitude=case.altitude_m)
    op = asb.OperatingPoint(
        atmosphere=atm,
        velocity=case.airspeed_mps,
        alpha=case.alpha_deg,
        beta=0.0,
    )
    return atm, op


def run_case(airplane: asb.Airplane, case: LoadCase) -> AeroResult:
    """AeroBuildup-driven totals only (no per-panel data). Fast envelope sweep."""
    atm, op = _operating_point(case)
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


def run_case_lifting_line(
    airplane: asb.Airplane,
    case: LoadCase,
    *,
    spanwise_resolution: int = 12,
) -> AeroResult:
    """LiftingLine-driven solve with per-panel forces attached as `panels`."""
    atm, op = _operating_point(case)
    ll = asb.LiftingLine(
        airplane=airplane,
        op_point=op,
        spanwise_resolution=spanwise_resolution,
    )
    aero = ll.run()
    panels = _compute_panel_loads(ll)
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
        panels=panels,
    )


def sweep_envelope(
    airplane: asb.Airplane,
    cases: Iterable[LoadCase],
    *,
    method: Literal["buildup", "lifting_line"] = "buildup",
    spanwise_resolution: int = 12,
) -> list[AeroResult]:
    if method == "buildup":
        return [run_case(airplane, c) for c in cases]
    if method == "lifting_line":
        return [
            run_case_lifting_line(airplane, c, spanwise_resolution=spanwise_resolution)
            for c in cases
        ]
    raise ValueError(f"unknown method: {method!r}")
