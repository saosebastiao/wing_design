"""CLT co-sizing: beam radii + uniform skin thickness + laminate layup fractions.

Design vector x = [beam_radii (n), t_skin, f0, f45]; f90 = 1 - f0 - f45. Each
evaluate builds the laminate stiffness from the fractions + thickness, solves the
combined beam+shell FEA with the anisotropic skin, and constrains beam von Mises,
skin (laminate-average membrane) von Mises, tip deflection, and tip twist. SLSQP with
O(1)-normalized objective/constraints; fractions enter only through the constraints
(mass is fraction-independent at uniform density). Symmetric-balanced laminate,
smeared bending D — see materials.laminate_stiffness.

PLY-ANGLE DATUM (E.4b): by default (`config.ply_angle_datum is None`) ply angles are
measured relative to each skin triangle's local frame, which is NOT globally
consistent (the tiling's frames are ~50/50 spanwise/chordwise) — so the resulting
`(f0, f45, f90)` is self-consistent but not a manufacturable global layup. Set
`config.ply_angle_datum` (e.g. `(0,0,1)` = span) to measure angles against that global
datum instead: each triangle's laminate is built with its ply angles offset by the
triangle's local-frame angle to the datum (`skin_datum_angles` + `laminate_stiffness_offset`),
making the optimized layup a coherent, manufacturable prescription (0° = the datum).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..materials.failure import laminate_min_strength_ratio_batch
from ..materials.unidir import UDPly, laminate_stiffness, laminate_stiffness_offset
from ..structural.beam_shell import solve_beam_shell_laminate
from ..structural.buckling import beam_euler_utilization, panel_buckling_utilization
from ..structural.frame import BeamSection, von_mises_per_element
from ..structural.shell import membrane_von_mises, recover_membrane_strain, recover_membrane_stress_C
from .shell_model import BeamShellModel, skin_datum_angles
from .shell_sizing import beam_lengths, beam_mass, skin_areas, skin_band_areas, skin_band_map, skin_mass


@dataclass(frozen=True)
class LaminateSizingConfig:
    sigma_allow_Pa: float
    tip_defl_max_m: float
    tip_twist_max_deg: float
    r_min: float = 0.004
    r_max: float = 0.04
    t_min: float = 0.0005
    t_max: float = 0.02
    buckling_safety_factor: float | None = None  # if set, enforce beam Euler + panel buckling
    euler_K: float = 1.0
    panel_kc: float = 4.0
    ply_angle_datum: tuple[float, float, float] | None = None
    n_skin_bands: int = 1
    skin_failure: str = "von_mises"          # "von_mises" (default) or "tsai_wu"
    tsai_wu_safety_factor: float = 2.0


@dataclass(frozen=True)
class LaminateSizingResult:
    radii: np.ndarray
    t_skin: float
    t_bands: np.ndarray
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
    max_beam_buckling_util: float
    max_panel_buckling_util: float
    min_skin_strength_ratio: float | None


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
    B = config.n_skin_bands
    Lb = beam_lengths(model)
    Atri = skin_areas(model)
    A_skin_area = float(Atri.sum())
    band_of_tri = skin_band_map(model, B)
    band_area = skin_band_areas(model, band_of_tri, B)
    nx = n + B + 2
    fi0 = n + B
    fi45 = n + B + 1
    x0 = np.concatenate([np.full(n, config.r_max), np.full(B, config.t_max), [1.0 / 3.0, 1.0 / 3.0]])
    datum_offsets_deg = (
        np.degrees(skin_datum_angles(model, config.ply_angle_datum))
        if config.ply_angle_datum is not None else None
    )
    cache: dict = {}

    def evaluate(x):
        key = np.asarray(x, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        radii = x[:n]
        t_band_vec = x[n:n + B]
        t_tri = t_band_vec[band_of_tri]
        f0 = float(x[fi0])
        f45 = float(x[fi45])
        f90 = max(0.0, 1.0 - f0 - f45)
        if datum_offsets_deg is None:
            band_mats = [laminate_stiffness(ply, f0=f0, f45=f45, f90=f90, thickness=float(tb))
                         for tb in t_band_vec]
            A_band = np.stack([m[0] for m in band_mats])
            D_band = np.stack([m[1] for m in band_mats])
            C_band = np.stack([m[2] for m in band_mats])
            A_arg = A_band[band_of_tri]
            D_arg = D_band[band_of_tri]
            C_arg = C_band[band_of_tri]
        else:
            mats = [laminate_stiffness_offset(ply, f0=f0, f45=f45, f90=f90,
                                              thickness=float(tt), offset_deg=float(o))
                    for tt, o in zip(t_tri, datum_offsets_deg)]
            A_arg = np.stack([m[0] for m in mats])
            D_arg = np.stack([m[1] for m in mats])
            C_arg = np.stack([m[2] for m in mats])
        D11 = D_arg[:, 0, 0]
        sections = [BeamSection.circular(float(r)) for r in radii]
        wb = np.zeros(n)
        ws = 0.0
        md = 0.0
        mt = 0.0
        worst_beam_buck = 0.0
        worst_panel_buck = 0.0
        worst_R = np.inf
        for loads in load_arrays:
            res = solve_beam_shell_laminate(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
                fixed_nodes=model.fixed_nodes, loads=loads,
            )
            wb = np.maximum(wb, von_mises_per_element(res, sections))
            skin_s = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=C_arg)
            ws = max(ws, float(membrane_von_mises(skin_s).max()))
            if config.skin_failure == "tsai_wu":
                eps = recover_membrane_strain(model.nodes, model.shell_tris, res.displacements)
                offs = datum_offsets_deg if datum_offsets_deg is not None else np.zeros(eps.shape[0])
                R = laminate_min_strength_ratio_batch(
                    ply, eps, f0=f0, f45=f45, f90=f90, offset_deg=offs)
                worst_R = min(worst_R, float(R.min()))
            tip = model.tip_nodes
            md = max(md, float(np.linalg.norm(res.displacements[tip, :3], axis=1).max()))
            mt = max(mt, float(np.degrees(np.abs(res.displacements[tip, 5]).max())))
            if config.buckling_safety_factor is not None:
                bu = beam_euler_utilization(res.axial_force, radii, Lb, E=model.E_beam,
                                            K=config.euler_K, safety_factor=config.buckling_safety_factor)
                pu = panel_buckling_utilization(skin_s, Atri, D11=D11, t=t_tri,
                                                kc=config.panel_kc, safety_factor=config.buckling_safety_factor)
                worst_beam_buck = max(worst_beam_buck, float(bu.max()))
                worst_panel_buck = max(worst_panel_buck, float(pu.max()))
        out = (wb, ws, md, mt, worst_beam_buck, worst_panel_buck, worst_R)
        cache[key] = out
        return out

    m_ref = beam_mass(model, np.full(n, config.r_max), rho=rho) + skin_mass(model, config.t_max, rho=rho)

    def mass(x):
        m = rho * np.sum(np.pi * x[:n] ** 2 * Lb) + rho * np.sum(x[n:n + B] * band_area)
        return m / m_ref

    def mass_grad(x):
        g = np.zeros(nx)
        g[:n] = rho * 2.0 * np.pi * x[:n] * Lb / m_ref
        g[n:n + B] = rho * band_area / m_ref
        return g

    def beam_con(x):
        return 1.0 - evaluate(x)[0] / config.sigma_allow_Pa

    def skin_con(x):
        out = evaluate(x)
        if config.skin_failure == "tsai_wu":
            return np.array([out[6] / config.tsai_wu_safety_factor - 1.0])
        return np.array([1.0 - out[1] / config.sigma_allow_Pa])

    def defl_con(x):
        return np.array([1.0 - evaluate(x)[2] / config.tip_defl_max_m])

    def twist_con(x):
        return np.array([1.0 - evaluate(x)[3] / config.tip_twist_max_deg])

    def frac_con(x):
        return np.array([1.0 - (x[fi0] + x[fi45])])

    bounds = (
        [(config.r_min, config.r_max)] * n
        + [(config.t_min, config.t_max)] * B
        + [(0.0, 1.0), (0.0, 1.0)]
    )
    constraints = [
        {"type": "ineq", "fun": beam_con},
        {"type": "ineq", "fun": skin_con},
        {"type": "ineq", "fun": defl_con},
        {"type": "ineq", "fun": twist_con},
        {"type": "ineq", "fun": frac_con},
    ]
    if config.buckling_safety_factor is not None:
        def beam_buck_con(x):
            return np.array([1.0 - evaluate(x)[4]])
        def panel_buck_con(x):
            return np.array([1.0 - evaluate(x)[5]])
        constraints += [{"type": "ineq", "fun": beam_buck_con}, {"type": "ineq", "fun": panel_buck_con}]
    res = minimize(
        mass, x0, jac=mass_grad, method="SLSQP", bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol},
    )

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = np.clip(res.x, lo, hi)
    s = x[fi0] + x[fi45]
    if s > 1.0:
        x[fi0] /= s
        x[fi45] /= s
    wb, ws, d, t, bbu, pbu, worst_R = evaluate(x)
    radii = x[:n]
    t_bands = x[n:n + B].copy()
    t_tri = t_bands[band_of_tri]
    f0 = float(x[fi0])
    f45 = float(x[fi45])
    f90 = max(0.0, 1.0 - f0 - f45)
    bm = beam_mass(model, radii, rho=rho)
    sm = float(rho * np.sum(t_tri * Atri))
    t_skin_mean = float(np.sum(t_tri * Atri) / A_skin_area)
    return LaminateSizingResult(
        radii=radii, t_skin=t_skin_mean, t_bands=t_bands, f0=f0, f45=f45, f90=f90,
        mass_kg=bm + sm, beam_mass_kg=bm, skin_mass_kg=sm,
        converged=bool(res.success), n_iter=int(res.nit),
        max_beam_vm_Pa=float(wb.max()), max_skin_vm_Pa=float(ws),
        tip_defl_m=float(d), tip_twist_deg=float(t),
        max_beam_buckling_util=float(bbu),
        max_panel_buckling_util=float(pbu),
        min_skin_strength_ratio=(float(worst_R) if config.skin_failure == "tsai_wu" else None),
    )
