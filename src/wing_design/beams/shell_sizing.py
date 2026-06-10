"""Co-size form-beam radii and skin thickness against the combined beam+shell FEA.

Design variables: per-element beam radii + a single uniform skin thickness.
Objective: total (beam + skin) mass. Constraints: per-beam-element von Mises,
per-skin-triangle von Mises, tip deflection, and tip twist — re-solving
`structural.solve_beam_shell` each iteration. SLSQP with O(1)-normalized
objective/constraints (as in Phase C).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..structural.beam_shell import solve_beam_shell
from ..structural.buckling import beam_euler_utilization, panel_buckling_utilization
from ..structural.frame import BeamSection, von_mises_per_element
from ..structural.shell import _triangle_local_frame, membrane_von_mises, recover_membrane_stress
from .shell_model import BeamShellModel


@dataclass(frozen=True)
class BeamShellSizingConfig:
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


@dataclass(frozen=True)
class BeamShellSizingResult:
    radii: np.ndarray
    t_skin: float
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


def beam_lengths(model: BeamShellModel) -> np.ndarray:
    """(n_beam_elem,) Euclidean length of each longitudinal beam element."""
    i = model.beam_elements[:, 0]
    j = model.beam_elements[:, 1]
    return np.linalg.norm(model.nodes[j] - model.nodes[i], axis=1)


def skin_areas(model: BeamShellModel) -> np.ndarray:
    """(n_tris,) area of each skin triangle."""
    out = np.empty(model.shell_tris.shape[0])
    for e in range(model.shell_tris.shape[0]):
        a, b, c = (int(v) for v in model.shell_tris[e])
        _, _, area = _triangle_local_frame(model.nodes[a], model.nodes[b], model.nodes[c])
        out[e] = area
    return out


def skin_band_map(model: BeamShellModel, n_bands: int) -> np.ndarray:
    """(n_tris,) spanwise band index in [0, n_bands) for each skin triangle.

    The ``n_levels - 1`` spanwise levels are split into ``n_bands`` contiguous,
    near-equal groups; triangle ``e`` (at level ``(e//2)//n_beams``) inherits its
    level's band. ``n_bands == 1`` returns all zeros. Band index increases with level
    index (level 0 = tip end, since ``default_z_levels`` runs tip->keel).
    """
    n_seg_levels = model.n_levels - 1
    if not (1 <= n_bands <= n_seg_levels):
        raise ValueError(f"n_bands must be in [1, n_levels-1={n_seg_levels}], got {n_bands}")
    level_band = np.empty(n_seg_levels, dtype=int)
    for bi, group in enumerate(np.array_split(np.arange(n_seg_levels), n_bands)):
        level_band[group] = bi
    e = np.arange(model.shell_tris.shape[0])
    level = (e // 2) // model.n_beams
    return level_band[level]


def skin_band_areas(model: BeamShellModel, band_of_tri: np.ndarray, n_bands: int) -> np.ndarray:
    """(n_bands,) total skin-triangle area in each band."""
    areas = skin_areas(model)
    out = np.zeros(n_bands)
    np.add.at(out, np.asarray(band_of_tri, dtype=int), areas)
    return out


def beam_radius_groups(model: BeamShellModel) -> tuple[np.ndarray, int]:
    """Map each beam-element to a radius design-variable group, exploiting chord symmetry.

    Beams are ordered TE(0) -> LE(n_beams//2) around a chord-symmetric section, so beam
    ``b`` mirrors ``(n_beams - b) % n_beams``. When the beam node positions are actually
    mirror-symmetric across the chord plane (y -> -y) -- the default even arc-spacing --
    mirror-paired beams SHARE radii: element ``(b,k)`` maps to group
    ``rank(rep(b))*(n_levels-1) + k`` where ``rep(b) = min(b, mirror(b))``. Returns
    ``(group_of_element (n,), n_groups)``. If placement is NOT mirror-symmetric (e.g.
    non-uniform arc_fractions) it falls back to one group per element
    (``arange(n), n``), so callers can stay symmetry-on by default safely.
    """
    nb = model.n_beams
    nl = model.n_levels
    seg = nl - 1
    # form-beam elements only: tube rows (P.1), if any, come after and carry
    # their own DV blocks (the tube sits on the symmetry axis).
    n = getattr(model, "n_form_elements", model.beam_elements.shape[0])
    nodes = model.nodes
    scale = float(np.abs(nodes).max()) if nodes.size else 1.0
    tol = 1.0e-6 * max(1.0, scale)

    symmetric = True
    for b in range(nb):
        mb = (nb - b) % nb
        for k in range(nl):
            pb = nodes[b * nl + k]
            pm = nodes[mb * nl + k]
            if (abs(pb[0] - pm[0]) > tol or abs(pb[2] - pm[2]) > tol
                    or abs(pb[1] + pm[1]) > tol):
                symmetric = False
                break
        if not symmetric:
            break

    if not symmetric:
        return np.arange(n, dtype=int), n

    reps = sorted({min(b, (nb - b) % nb) for b in range(nb)})
    rank = {r: i for i, r in enumerate(reps)}
    group_of_element = np.empty(n, dtype=int)
    for b in range(nb):
        ui = rank[min(b, (nb - b) % nb)]
        for k in range(seg):
            group_of_element[b * seg + k] = ui * seg + k
    return group_of_element, len(reps) * seg


def beam_mass(model: BeamShellModel, radii: np.ndarray, *, rho: float) -> float:
    """Total longitudinal-beam mass [kg]."""
    return float(rho * np.sum(np.pi * np.asarray(radii) ** 2 * beam_lengths(model)))


def skin_mass(model: BeamShellModel, t_skin: float, *, rho: float) -> float:
    """Total skin mass [kg] = rho * t * sum(triangle areas)."""
    return float(rho * t_skin * skin_areas(model).sum())


def size_beam_shell(
    model: BeamShellModel,
    load_arrays: list[np.ndarray],
    config: BeamShellSizingConfig,
    *,
    rho: float,
    maxiter: int = 60,
    ftol: float = 1.0e-4,
) -> BeamShellSizingResult:
    """Co-size beam radii + uniform skin thickness to minimize beam+skin mass.

    Design vector x = [radii (n_beam_elem), t_skin]. Constraints (all >= 0 after
    normalization): beam vm, skin vm, tip deflection, tip twist. `load_arrays` is
    one (n_nodes, 6) array per (already safety-factored) load case.
    """
    if not load_arrays:
        raise ValueError("load_arrays is empty: no loads to size against")
    n = model.beam_elements.shape[0]
    Lb = beam_lengths(model)
    Atri = skin_areas(model)
    A_skin = float(Atri.sum())

    x0 = np.concatenate([np.full(n, config.r_max), [config.t_max]])
    cache: dict[bytes, tuple] = {}

    def evaluate(x):
        key = np.asarray(x, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        radii = x[:n]
        t_skin = float(x[n])
        sections = [BeamSection.circular(float(r)) for r in radii]
        worst_beam = np.zeros(n)
        worst_skin = 0.0
        max_defl = 0.0
        max_twist = 0.0
        worst_beam_buck = 0.0
        worst_panel_buck = 0.0
        for loads in load_arrays:
            res = solve_beam_shell(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam,
                E_skin=model.E_skin, nu_skin=model.nu_skin, t_skin=t_skin,
                fixed_nodes=model.fixed_nodes, loads=loads,
            )
            worst_beam = np.maximum(worst_beam, von_mises_per_element(res, sections))
            skin_s = recover_membrane_stress(model.nodes, model.shell_tris, res.displacements,
                                             E=model.E_skin, nu=model.nu_skin)
            worst_skin = max(worst_skin, float(membrane_von_mises(skin_s).max()))
            tip = model.tip_nodes
            max_defl = max(max_defl, float(np.linalg.norm(res.displacements[tip, :3], axis=1).max()))
            max_twist = max(max_twist, float(np.degrees(np.abs(res.displacements[tip, 5]).max())))
            if config.buckling_safety_factor is not None:
                bu = beam_euler_utilization(res.axial_force, radii, Lb, E=model.E_beam,
                                            K=config.euler_K, safety_factor=config.buckling_safety_factor)
                D11 = model.E_skin * t_skin ** 3 / (12.0 * (1.0 - model.nu_skin ** 2))
                pu = panel_buckling_utilization(skin_s, Atri, D11=D11, t=t_skin,
                                                kc=config.panel_kc, safety_factor=config.buckling_safety_factor)
                worst_beam_buck = max(worst_beam_buck, float(bu.max()))
                worst_panel_buck = max(worst_panel_buck, float(pu.max()))
        out = (worst_beam, worst_skin, max_defl, max_twist, worst_beam_buck, worst_panel_buck)
        cache[key] = out
        return out

    m_ref = (beam_mass(model, np.full(n, config.r_max), rho=rho)
             + skin_mass(model, config.t_max, rho=rho))

    def mass(x):
        return (rho * np.sum(np.pi * x[:n] ** 2 * Lb) + rho * x[n] * A_skin) / m_ref

    def mass_grad(x):
        g = np.empty(n + 1)
        g[:n] = rho * 2.0 * np.pi * x[:n] * Lb / m_ref
        g[n] = rho * A_skin / m_ref
        return g

    def beam_con(x):
        wb, _, _, _, _, _ = evaluate(x)
        return 1.0 - wb / config.sigma_allow_Pa

    def skin_con(x):
        _, ws, _, _, _, _ = evaluate(x)
        return np.array([1.0 - ws / config.sigma_allow_Pa])

    def defl_con(x):
        _, _, d, _, _, _ = evaluate(x)
        return np.array([1.0 - d / config.tip_defl_max_m])

    def twist_con(x):
        _, _, _, tw, _, _ = evaluate(x)
        return np.array([1.0 - tw / config.tip_twist_max_deg])

    bounds = [(config.r_min, config.r_max)] * n + [(config.t_min, config.t_max)]
    constraints = [
        {"type": "ineq", "fun": beam_con},
        {"type": "ineq", "fun": skin_con},
        {"type": "ineq", "fun": defl_con},
        {"type": "ineq", "fun": twist_con},
    ]
    if config.buckling_safety_factor is not None:
        def beam_buck_con(x):
            *_, bbu, _ = evaluate(x)
            return np.array([1.0 - bbu])
        def panel_buck_con(x):
            *_, pbu = evaluate(x)
            return np.array([1.0 - pbu])
        constraints += [{"type": "ineq", "fun": beam_buck_con}, {"type": "ineq", "fun": panel_buck_con}]
    res = minimize(
        mass, x0, jac=mass_grad, method="SLSQP", bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol},
    )

    x = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
    wb, ws, d, tw, bbu, pbu = evaluate(x)
    radii = x[:n]
    t_skin = float(x[n])
    bm = beam_mass(model, radii, rho=rho)
    sm = skin_mass(model, t_skin, rho=rho)
    return BeamShellSizingResult(
        radii=radii, t_skin=t_skin, mass_kg=bm + sm, beam_mass_kg=bm, skin_mass_kg=sm,
        converged=bool(res.success), n_iter=int(res.nit),
        max_beam_vm_Pa=float(wb.max()), max_skin_vm_Pa=float(ws),
        tip_defl_m=float(d), tip_twist_deg=float(tw),
        max_beam_buckling_util=float(bbu),
        max_panel_buckling_util=float(pbu),
    )
