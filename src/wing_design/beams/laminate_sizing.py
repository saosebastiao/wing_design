"""CLT co-sizing: beam radii + uniform skin thickness + laminate layup fractions.

Design vector x = [beam_radii (n), t_skin, f0, f45]; f90 = 1 - f0 - f45. Each
evaluate builds the laminate (A, D, Qeff) from the fractions + thickness, solves the
combined beam+shell FEA with the anisotropic skin, and constrains beam von Mises,
skin (laminate-average membrane) von Mises, tip deflection, and tip twist. SLSQP with
O(1)-normalized objective/constraints; fractions enter only through the constraints
(mass is fraction-independent at uniform density). Symmetric-balanced laminate,
smeared bending D — see materials.laminate_stiffness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..materials.unidir import UDPly, laminate_stiffness
from ..structural.beam_shell import solve_beam_shell_laminate
from ..structural.frame import BeamSection, von_mises_per_element
from ..structural.shell import membrane_von_mises, recover_membrane_stress_C
from .shell_model import BeamShellModel
from .shell_sizing import beam_lengths, beam_mass, skin_areas, skin_mass


@dataclass(frozen=True)
class LaminateSizingConfig:
    sigma_allow_Pa: float
    tip_defl_max_m: float
    tip_twist_max_deg: float
    r_min: float = 0.004
    r_max: float = 0.04
    t_min: float = 0.0005
    t_max: float = 0.02


@dataclass(frozen=True)
class LaminateSizingResult:
    radii: np.ndarray
    t_skin: float
    f0: float
    f45: float
    f90: float
    mass_kg: float
    beam_mass_kg: float
    skin_mass_kg: float
    converged: bool
    n_iter: int
    max_beam_vm_Pa: float
    max_skin_vm_Pa: float
    tip_defl_m: float
    tip_twist_deg: float


def size_beam_shell_laminate(
    model: BeamShellModel,
    load_arrays: list[np.ndarray],
    config: LaminateSizingConfig,
    *,
    ply: UDPly,
    rho: float,
    maxiter: int = 80,
    ftol: float = 1.0e-4,
) -> LaminateSizingResult:
    if not load_arrays:
        raise ValueError("load_arrays is empty")
    n = model.beam_elements.shape[0]
    Lb = beam_lengths(model)
    A_skin_area = float(skin_areas(model).sum())
    x0 = np.concatenate([np.full(n, config.r_max), [config.t_max, 1.0 / 3.0, 1.0 / 3.0]])
    cache: dict = {}

    def evaluate(x):
        key = np.asarray(x, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        radii = x[:n]
        t = float(x[n])
        f0 = float(x[n + 1])
        f45 = float(x[n + 2])
        f90 = max(0.0, 1.0 - f0 - f45)
        A, D, Qeff = laminate_stiffness(ply, f0=f0, f45=f45, f90=f90, thickness=t)
        sections = [BeamSection.circular(float(r)) for r in radii]
        wb = np.zeros(n)
        ws = 0.0
        md = 0.0
        mt = 0.0
        for loads in load_arrays:
            res = solve_beam_shell_laminate(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A, D_skin=D,
                fixed_nodes=model.fixed_nodes, loads=loads,
            )
            wb = np.maximum(wb, von_mises_per_element(res, sections))
            skin_s = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=Qeff)
            ws = max(ws, float(membrane_von_mises(skin_s).max()))
            tip = model.tip_nodes
            md = max(md, float(np.linalg.norm(res.displacements[tip, :3], axis=1).max()))
            mt = max(mt, float(np.degrees(np.abs(res.displacements[tip, 5]).max())))
        out = (wb, ws, md, mt)
        cache[key] = out
        return out

    m_ref = beam_mass(model, np.full(n, config.r_max), rho=rho) + skin_mass(model, config.t_max, rho=rho)

    def mass(x):
        return (rho * np.sum(np.pi * x[:n] ** 2 * Lb) + rho * x[n] * A_skin_area) / m_ref

    def mass_grad(x):
        g = np.zeros(n + 3)
        g[:n] = rho * 2.0 * np.pi * x[:n] * Lb / m_ref
        g[n] = rho * A_skin_area / m_ref
        return g

    def beam_con(x):
        wb, _, _, _ = evaluate(x)
        return 1.0 - wb / config.sigma_allow_Pa

    def skin_con(x):
        _, ws, _, _ = evaluate(x)
        return np.array([1.0 - ws / config.sigma_allow_Pa])

    def defl_con(x):
        _, _, d, _ = evaluate(x)
        return np.array([1.0 - d / config.tip_defl_max_m])

    def twist_con(x):
        _, _, _, t = evaluate(x)
        return np.array([1.0 - t / config.tip_twist_max_deg])

    def frac_con(x):
        return np.array([1.0 - (x[n + 1] + x[n + 2])])

    bounds = (
        [(config.r_min, config.r_max)] * n
        + [(config.t_min, config.t_max), (0.0, 1.0), (0.0, 1.0)]
    )
    res = minimize(
        mass, x0, jac=mass_grad, method="SLSQP", bounds=bounds,
        constraints=[
            {"type": "ineq", "fun": beam_con},
            {"type": "ineq", "fun": skin_con},
            {"type": "ineq", "fun": defl_con},
            {"type": "ineq", "fun": twist_con},
            {"type": "ineq", "fun": frac_con},
        ],
        options={"maxiter": maxiter, "ftol": ftol},
    )

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = np.clip(res.x, lo, hi)
    wb, ws, d, t = evaluate(x)
    radii = x[:n]
    t_skin = float(x[n])
    f0 = float(x[n + 1])
    f45 = float(x[n + 2])
    f90 = max(0.0, 1.0 - f0 - f45)
    bm = beam_mass(model, radii, rho=rho)
    sm = skin_mass(model, t_skin, rho=rho)
    return LaminateSizingResult(
        radii=radii, t_skin=t_skin, f0=f0, f45=f45, f90=f90,
        mass_kg=bm + sm, beam_mass_kg=bm, skin_mass_kg=sm,
        converged=bool(res.success), n_iter=int(res.nit),
        max_beam_vm_Pa=float(wb.max()), max_skin_vm_Pa=float(ws),
        tip_defl_m=float(d), tip_twist_deg=float(t),
    )
