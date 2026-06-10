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
from ..structural.beam_shell import (
    solve_beam_shell_laminate, solve_beam_shell_laminate_factored,
    _recover_beam_forces as _recover_beam_forces_local,
)
from ..structural.frame import FrameResult, _element_rotation
from .sensitivity import (
    DesignSens, grad_beam_buckling, grad_skin_vm,
    grad_panel_buckling, grad_skin_tsai_wu, grad_tip_defl, grad_tip_twist,
    _active_beam_force, adjoint_lambda, lambdaT_dK_x_cached, dkloc_dr,
)
from ..structural.buckling import beam_euler_utilization, panel_buckling_utilization
from ..structural.frame import BeamSection, von_mises_per_element
from ..structural.shell import membrane_von_mises, recover_membrane_strain, recover_membrane_stress_C
from .body_loads import body_load_jacobian, body_load_vector
from .constraints import ConstraintSpec, shadow_prices_from_specs
from .design_vector import DesignVector
from .shell_model import BeamShellModel, skin_datum_angles, skin_panel_widths
from .shell_sizing import (
    beam_lengths, beam_mass, beam_radius_groups, skin_areas, skin_band_areas,
    skin_band_map, skin_mass,
)


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
    # Panel-buckling characteristic width b: "sqrt_area" (historical triangle-as-
    # plate; mis-scales with beam count, ~2x oversized b — backlog V#1) or "strip"
    # (V.3b: physical chordwise beam spacing via `skin_panel_widths`, the long-strip
    # SS-plate model, eigen-calibrated 2026-06-10). Default unchanged for
    # comparability with recorded headlines until the V.6 re-baseline.
    panel_width_mode: str = "sqrt_area"
    # V.4 self-weight/inertial accelerations (m/s², wing frame, z = span). Empty =
    # aero-only (recorded headlines unchanged). Non-empty: every aero load case is
    # paired with every acceleration vector (include (0,0,0) to keep an aero-only
    # combo); the design's own mass is lumped to nodes each evaluate and the
    # analytic adjoint carries the lam^T dF/dx term (loads depend on the design).
    accel_vectors: tuple[tuple[float, float, float], ...] = ()
    ply_angle_datum: tuple[float, float, float] | None = None
    n_skin_bands: int = 1
    skin_failure: str = "von_mises"          # "von_mises" (default) or "tsai_wu"
    tsai_wu_safety_factor: float = 2.0
    per_band_layup: bool = False
    use_analytic_jacobian: bool = False


@dataclass(frozen=True)
class LaminateSizingResult:
    radii: np.ndarray
    t_skin: float
    t_bands: np.ndarray
    f0: float
    f45: float
    f90: float
    f0_bands: np.ndarray
    f45_bands: np.ndarray
    f90_bands: np.ndarray
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
    # V.1 shadow prices: physical dm*/dparam (kg per unit of each config limit),
    # from the SLSQP KKT multipliers. Limit-type keys ("tip_twist_max_deg",
    # "tip_defl_max_m", "sigma_allow_Pa") are <= 0 (relaxing the limit sheds kg);
    # safety-factor keys ("buckling_sf_beam", "buckling_sf_panel",
    # "tsai_wu_safety_factor") are >= 0 (raising SF costs kg). None when the
    # multiplier layout can't be attributed. Only meaningful at a converged optimum.
    shadow_prices: dict[str, float] | None = None


def laminate_design_bounds(model: BeamShellModel, config: LaminateSizingConfig):
    """(lo, hi) bounds for [r_group(G), t_band(B), f0_grp(L), f45_grp(L)]."""
    _, G = beam_radius_groups(model)
    B = config.n_skin_bands
    L = B if config.per_band_layup else 1
    lo = np.concatenate([np.full(G, config.r_min), np.full(B, config.t_min), np.zeros(2 * L)])
    hi = np.concatenate([np.full(G, config.r_max), np.full(B, config.t_max), np.ones(2 * L)])
    return lo, hi


def laminate_result_is_feasible(
    result: "LaminateSizingResult", config: LaminateSizingConfig, *, tol: float = 1.0e-3,
) -> bool:
    """True iff every active sizing constraint holds within relative ``tol``."""
    if result.max_beam_vm_Pa > config.sigma_allow_Pa * (1.0 + tol):
        return False
    if config.skin_failure == "tsai_wu":
        r = result.min_skin_strength_ratio
        if r is None or r < config.tsai_wu_safety_factor * (1.0 - tol):
            return False
    else:
        if result.max_skin_vm_Pa > config.sigma_allow_Pa * (1.0 + tol):
            return False
    if result.tip_defl_m > config.tip_defl_max_m * (1.0 + tol):
        return False
    if result.tip_twist_deg > config.tip_twist_max_deg * (1.0 + tol):
        return False
    if config.buckling_safety_factor is not None:
        if result.max_beam_buckling_util > 1.0 + tol:
            return False
        if result.max_panel_buckling_util > 1.0 + tol:
            return False
    return True


def size_beam_shell_laminate(
    model: BeamShellModel,
    load_arrays: list[np.ndarray],
    config: LaminateSizingConfig,
    *,
    ply: UDPly,
    rho: float,
    maxiter: int = 80,
    ftol: float = 1.0e-4,
    x0: np.ndarray | None = None,
    panel_pressures: list[np.ndarray] | None = None,
) -> LaminateSizingResult:
    """Co-size beam radii + skin bands + layup. ``panel_pressures`` (V.5): one
    (n_tris,) lateral-pressure array [Pa] per aero load case (see
    `fea_model.panel_pressure_per_tri`); adds the sub-mesh strip-bending fiber
    stress sigma_b = 0.75 q w^2 / t^2 to the skin von-Mises failure metric
    (Tsai-Wu mode keeps membrane-only — recorded caveat). None = unchanged."""
    if not load_arrays:
        raise ValueError("load_arrays is empty")
    gusset_elements = model.tip_gusset_elements
    gusset_section = (BeamSection.circular(model.tip_gusset_radius)
                     if gusset_elements is not None else None)
    n = model.beam_elements.shape[0]
    B = config.n_skin_bands
    L = B if config.per_band_layup else 1
    group_of_element, G = beam_radius_groups(model)
    Lb = beam_lengths(model)
    Atri = skin_areas(model)
    # Effective b^2 for the panel-buckling check (the only consumer of "areas"
    # in both the evaluate path and grad_panel_buckling, so the analytic
    # Jacobian stays exact in either mode).
    if config.panel_width_mode == "strip":
        b2_panel = skin_panel_widths(model) ** 2
    elif config.panel_width_mode == "sqrt_area":
        # sqrt-roundtrip kept deliberately: the historical check computed
        # b = sqrt(area) then b**2, which differs from the raw area by ~1 ulp -
        # enough to flip SLSQP cold-start basins on small problems
        # (test_analytic_matches_fd_optimum, 2026-06-10). Keeps every legacy-mode
        # result bit-reproducible.
        b2_panel = np.sqrt(Atri) ** 2
    else:
        raise ValueError(f"unknown panel_width_mode: {config.panel_width_mode!r}")
    # V.5 strip-bending: 0.75*q*w^2 per aero case (physical strip width regardless
    # of the buckling b mode); sigma_b = qw2 / t^2 at evaluate time.
    if panel_pressures is not None:
        if len(panel_pressures) != len(load_arrays):
            raise ValueError("panel_pressures must have one entry per load case")
        _w2_bend = skin_panel_widths(model) ** 2
        qw2_cases = [0.75 * np.asarray(q, dtype=float) * _w2_bend for q in panel_pressures]
    else:
        qw2_cases = None
    A_skin_area = float(Atri.sum())
    band_of_tri = skin_band_map(model, B)
    band_area = skin_band_areas(model, band_of_tri, B)
    dv = DesignVector(("r_group", G), ("t_band", B), ("f0", L), ("f45", L))
    nx = dv.nx
    f0_lo = dv.slice("f0").start
    f45_lo = dv.slice("f45").start
    if x0 is None:
        x0 = np.concatenate([
            np.full(G, config.r_max), np.full(B, config.t_max),
            np.full(L, 1.0 / 3.0), np.full(L, 1.0 / 3.0),
        ])
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (nx,):
            raise ValueError(f"x0 must have length nx={nx}, got {x0.shape}")
        lo_b, hi_b = laminate_design_bounds(model, config)
        x0 = np.clip(x0, lo_b, hi_b).copy()
        # project layup groups onto the simplex (f0_g + f45_g <= 1)
        fg = x0[f0_lo:f0_lo + L]
        hg = x0[f45_lo:f45_lo + L]
        scale = np.where(fg + hg > 1.0, fg + hg, 1.0)
        x0[f0_lo:f0_lo + L] = fg / scale
        x0[f45_lo:f45_lo + L] = hg / scale
    datum_offsets_deg = (
        np.degrees(skin_datum_angles(model, config.ply_angle_datum))
        if config.ply_angle_datum is not None else None
    )

    # Monotonic taper: per radius-unit, radius non-increasing keel->tip. Group DV index
    # = unit*(n_levels-1) + segment; order consecutive segments by their midpoint z.
    seg = model.n_levels - 1
    U = G // seg
    level_z = model.nodes[:model.n_levels, 2]
    seg_z = 0.5 * (level_z[:-1] + level_z[1:])
    mono_lo_list: list[int] = []
    mono_hi_list: list[int] = []
    for u in range(U):
        for k in range(seg - 1):
            ia = u * seg + k
            ib = u * seg + (k + 1)
            if seg_z[k] >= seg_z[k + 1]:   # segment k is tip-ward (higher z)
                mono_lo_list.append(ib); mono_hi_list.append(ia)
            else:
                mono_lo_list.append(ia); mono_hi_list.append(ib)
    mono_lo = np.asarray(mono_lo_list, dtype=int)
    mono_hi = np.asarray(mono_hi_list, dtype=int)

    def band_fracs(x):
        """Per-band (length B) f0/f45/f90 from the L layup groups."""
        f0_grp = x[f0_lo:f0_lo + L]
        f45_grp = x[f45_lo:f45_lo + L]
        if config.per_band_layup:
            f0b = np.asarray(f0_grp, dtype=float)
            f45b = np.asarray(f45_grp, dtype=float)
        else:
            f0b = np.full(B, float(f0_grp[0]))
            f45b = np.full(B, float(f45_grp[0]))
        f90b = np.maximum(0.0, 1.0 - f0b - f45b)
        return f0b, f45b, f90b

    cache: dict = {}

    accels = [np.asarray(a, dtype=float) for a in config.accel_vectors]

    def effective_loads(radii, t_tri):
        """Aero × accel combos; aero-only when no accelerations configured."""
        if not accels:
            return load_arrays
        out = []
        for lc in load_arrays:
            for acc in accels:
                out.append(lc + body_load_vector(model, radii, t_tri, rho=rho, accel=acc))
        return out

    def evaluate(x):
        key = np.asarray(x, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        radii = x[:G][group_of_element]
        t_band_vec = x[G:G + B]
        t_tri = t_band_vec[band_of_tri]
        f0b, f45b, f90b = band_fracs(x)
        f0_tri = f0b[band_of_tri]
        f45_tri = f45b[band_of_tri]
        f90_tri = f90b[band_of_tri]
        if datum_offsets_deg is None:
            band_mats = [laminate_stiffness(ply, f0=float(f0b[b]), f45=float(f45b[b]),
                                            f90=float(f90b[b]), thickness=float(t_band_vec[b]))
                         for b in range(B)]
            A_band = np.stack([m[0] for m in band_mats])
            D_band = np.stack([m[1] for m in band_mats])
            C_band = np.stack([m[2] for m in band_mats])
            A_arg = A_band[band_of_tri]
            D_arg = D_band[band_of_tri]
            C_arg = C_band[band_of_tri]
        else:
            mats = [laminate_stiffness_offset(ply, f0=float(f0_tri[e]), f45=float(f45_tri[e]),
                                              f90=float(f90_tri[e]), thickness=float(t_tri[e]),
                                              offset_deg=float(o))
                    for e, o in enumerate(datum_offsets_deg)]
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
        n_acc = max(len(accels), 1)
        for ci, loads in enumerate(effective_loads(radii, t_tri)):
            qw2 = qw2_cases[ci // n_acc] if qw2_cases is not None else None
            res = solve_beam_shell_laminate(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
                fixed_nodes=model.fixed_nodes, loads=loads,
                gusset_elements=gusset_elements, gusset_section=gusset_section,
            )
            wb = np.maximum(wb, von_mises_per_element(res, sections))
            skin_s = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=C_arg)
            vm_skin = membrane_von_mises(skin_s)
            if qw2 is not None and config.skin_failure != "tsai_wu":
                vm_skin = vm_skin + qw2 / t_tri**2     # V.5 strip-bending fiber stress
            ws = max(ws, float(vm_skin.max()))
            if config.skin_failure == "tsai_wu":
                eps = recover_membrane_strain(model.nodes, model.shell_tris, res.displacements)
                offs = datum_offsets_deg if datum_offsets_deg is not None else np.zeros(eps.shape[0])
                R = laminate_min_strength_ratio_batch(
                    ply, eps, f0=f0_tri, f45=f45_tri, f90=f90_tri, offset_deg=offs)
                worst_R = min(worst_R, float(R.min()))
            tip = model.tip_nodes
            md = max(md, float(np.linalg.norm(res.displacements[tip, :3], axis=1).max()))
            mt = max(mt, float(np.degrees(np.abs(res.displacements[tip, 5]).max())))
            if config.buckling_safety_factor is not None:
                bu = beam_euler_utilization(res.axial_force, radii, Lb, E=model.E_beam,
                                            K=config.euler_K, safety_factor=config.buckling_safety_factor)
                pu = panel_buckling_utilization(skin_s, b2_panel, D11=D11, t=t_tri,
                                                kc=config.panel_kc, safety_factor=config.buckling_safety_factor)
                worst_beam_buck = max(worst_beam_buck, float(bu.max()))
                worst_panel_buck = max(worst_panel_buck, float(pu.max()))
        out = (wb, ws, md, mt, worst_beam_buck, worst_panel_buck, worst_R)
        cache[key] = out
        return out

    m_ref = beam_mass(model, np.full(n, config.r_max), rho=rho) + skin_mass(model, config.t_max, rho=rho)

    def mass(x):
        radii = x[:G][group_of_element]
        m = rho * np.sum(np.pi * radii ** 2 * Lb) + rho * np.sum(x[G:G + B] * band_area)
        return m / m_ref

    def mass_grad(x):
        g = np.zeros(nx)
        radii = x[:G][group_of_element]
        per_elem = rho * 2.0 * np.pi * radii * Lb / m_ref
        np.add.at(g, group_of_element, per_elem)   # scatter element grads into the G groups
        g[G:G + B] = rho * band_area / m_ref
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
        f0_grp = x[f0_lo:f0_lo + L]
        f45_grp = x[f45_lo:f45_lo + L]
        return 1.0 - (np.asarray(f0_grp, dtype=float) + np.asarray(f45_grp, dtype=float))

    def monotonic_con(x):
        return x[mono_lo] - x[mono_hi]   # >= 0 : keel-side radius >= tip-side radius

    lo_b, hi_b = laminate_design_bounds(model, config)
    bounds = [(float(lo_b[i]), float(hi_b[i])) for i in range(nx)]
    if config.buckling_safety_factor is not None:
        def beam_buck_con(x):
            return np.array([1.0 - evaluate(x)[4]])
        def panel_buck_con(x):
            return np.array([1.0 - evaluate(x)[5]])

    # Analytic Jacobian closures (registered by name in the spec list below).
    jacs: dict = {}
    if config.use_analytic_jacobian:
        from ..structural.beam_shell import FactoredBeamShell

        beam_lengths_e = np.empty(n)
        for e in range(n):
            i, j = int(model.beam_elements[e, 0]), int(model.beam_elements[e, 1])
            _R, Le = _element_rotation(model.nodes[i], model.nodes[j])
            beam_lengths_e[e] = Le
        layup_group_of_band = (
            np.arange(B, dtype=int) if config.per_band_layup else np.zeros(B, dtype=int)
        )

        jac_cache: dict = {}

        def _build_jac(x):
            """Factor once, solve all load cases, build per-lc FactoredBeamShell + DesignSens."""
            radii = x[:G][group_of_element]
            t_band_vec = x[G:G + B]
            t_tri = t_band_vec[band_of_tri]
            f0b, f45b, f90b = band_fracs(x)
            f0_tri = f0b[band_of_tri]
            f45_tri = f45b[band_of_tri]
            f90_tri = f90b[band_of_tri]
            # Per-band Qeff and per-tri Qeff/offset (datum-aware), mirroring evaluate.
            if datum_offsets_deg is None:
                band_Q = [laminate_stiffness(ply, f0=float(f0b[b]), f45=float(f45b[b]),
                                             f90=float(f90b[b]), thickness=float(t_band_vec[b]))[2]
                          for b in range(B)]
                Qeff_tri = np.stack(band_Q)[band_of_tri]
                offset_tri = np.zeros(t_tri.shape[0])
                A_band = [t_band_vec[b] * band_Q[b] for b in range(B)]
                D_band = [(t_band_vec[b] ** 3 / 12.0) * band_Q[b] for b in range(B)]
                A_arg = np.stack(A_band)[band_of_tri]
                D_arg = np.stack(D_band)[band_of_tri]
            else:
                mats = [laminate_stiffness_offset(ply, f0=float(f0_tri[e]), f45=float(f45_tri[e]),
                                                  f90=float(f90_tri[e]), thickness=float(t_tri[e]),
                                                  offset_deg=float(o))
                        for e, o in enumerate(datum_offsets_deg)]
                A_arg = np.stack([m[0] for m in mats])
                D_arg = np.stack([m[1] for m in mats])
                Qeff_tri = np.stack([m[2] for m in mats])
                offset_tri = np.asarray(datum_offsets_deg, dtype=float)

            sections = [BeamSection.circular(float(r)) for r in radii]
            loads_eff = effective_loads(radii, t_tri)
            # V.4: one ∂f/∂x per accel vector (zero for pure aero); fac k pairs
            # with accel k % len(accels) by the effective_loads combo order.
            if accels:
                dFs = [body_load_jacobian(
                           model, radii, group_of_element=group_of_element,
                           band_of_tri=band_of_tri, rho=rho, accel=acc, G=G, B=B, L=L)
                       for acc in accels]
                dF_of_fac = [dFs[k % len(accels)] for k in range(len(loads_eff))]
            else:
                dF_of_fac = [None] * len(loads_eff)
            # Factor once for load case 0, reuse the factorization for the rest.
            fac0 = solve_beam_shell_laminate_factored(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
                fixed_nodes=model.fixed_nodes, loads=loads_eff[0],
                gusset_elements=gusset_elements, gusset_section=gusset_section,
            )
            facs = [fac0]
            for loads in loads_eff[1:]:
                f = loads.reshape(-1).astype(float)
                u = np.zeros(fac0.ndof)
                u[fac0.free] = fac0.lu.solve(f[fac0.free])
                disp = u.reshape(model.nodes.shape[0], 6)
                axial, bending, torsion = _recover_beam_forces_local(
                    model.beam_elements, n, u, fac0.transforms, fac0.klocals)
                result = FrameResult(displacements=disp, axial_force=axial,
                                     bending_moment=bending, torsion=torsion)
                facs.append(FactoredBeamShell(
                    result=result, lu=fac0.lu, K_ff=fac0.K_ff, free=fac0.free,
                    ndof=fac0.ndof, u=u, beam_elements=model.beam_elements,
                    transforms=fac0.transforms, klocals=fac0.klocals,
                ))

            ds = DesignSens(
                model=model, G=G, B=B, L=L,
                group_of_element=group_of_element,
                band_of_tri=band_of_tri,
                layup_group_of_band=layup_group_of_band,
                radii_full=radii,
                beam_lengths=beam_lengths_e,
                t_tri=t_tri,
                Qeff_tri=Qeff_tri,
                offset_tri=offset_tri,
                ply=ply,
                f0_tri=(f0_tri if config.skin_failure == "tsai_wu" else None),
                f45_tri=(f45_tri if config.skin_failure == "tsai_wu" else None),
                f90_tri=(f90_tri if config.skin_failure == "tsai_wu" else None),
            )
            # Design-only ∂K/∂x assembly, built ONCE per design point and reused
            # across every load case and every constraint-gradient call below.
            from .sensitivity import prepare_sensitivity
            sens_cache = prepare_sensitivity(ds, facs[0])
            return facs, ds, sections, sens_cache, dF_of_fac

        def _jac_lookup(x):
            key = np.asarray(x, dtype=float).tobytes()
            hit = jac_cache.get(key)
            if hit is None:
                hit = _build_jac(x)
                jac_cache[key] = hit
            return hit

        def _binding_grad(x, grad_fn):
            """Evaluate grad_fn over all load cases, pick the binding (min con) lc, return its row."""
            facs, ds, _sections, sens_cache, dF_of_fac = _jac_lookup(x)
            best_con = np.inf
            best_grad = None
            for ci, (fac, dF) in enumerate(zip(facs, dF_of_fac)):
                con, grad = grad_fn(fac, ds, sens_cache, dF, ci)
                if con < best_con:
                    best_con = con
                    best_grad = grad
            return best_grad

        def _beam_vm_grad_one(fac, ds, e_star, sens_cache, dF=None):
            """∂(1 − vM_{e_star}/σ)/∂x as an (nx,) row, for a chosen beam element."""
            sa = config.sigma_allow_Pa
            floc, Mmat, dofs = _active_beam_force(fac, e_star)
            r = float(ds.radii_full[e_star])
            A = np.pi * r**2
            Iz = np.pi * r**4 / 4.0
            J = np.pi * r**4 / 2.0
            axial = floc[6]
            torsion = floc[9]
            b0 = np.hypot(floc[4], floc[5])
            b1 = np.hypot(floc[10], floc[11])
            Mres = max(b0, b1)
            sigma_n = abs(axial) / A + Mres * r / Iz
            tau = abs(torsion) * r / J
            vm = np.sqrt(sigma_n**2 + 3.0 * tau**2)
            if vm == 0.0:
                return np.zeros(nx)

            dvm_dfloc = np.zeros(12)
            sgn_ax = np.sign(axial) if axial != 0.0 else 0.0
            sgn_tor = np.sign(torsion) if torsion != 0.0 else 0.0
            dvm_dfloc[6] = (sigma_n / vm) * sgn_ax / A
            dvm_dfloc[9] = (3.0 * tau / vm) * sgn_tor * r / J
            if Mres > 0.0:
                if b0 >= b1:
                    fa, fb, ia, ib = floc[4], floc[5], 4, 5
                else:
                    fa, fb, ia, ib = floc[10], floc[11], 10, 11
                coef = (sigma_n / vm) * (r / Iz)
                dvm_dfloc[ia] = coef * (fa / Mres)
                dvm_dfloc[ib] = coef * (fb / Mres)

            dg_du = np.zeros(fac.ndof)
            dg_du[dofs] = dvm_dfloc @ Mmat
            lam = adjoint_lambda(fac, dg_du)
            du_part = -lambdaT_dK_x_cached(sens_cache, lam, fac.u)
            if dF is not None:   # design-dependent loads (V.4)
                du_part += lam @ dF

            dsigma_n_dr = -2.0 * abs(axial) / (np.pi * r**3) - 12.0 * Mres / (np.pi * r**4)
            dtau_dr = -6.0 * abs(torsion) / (np.pi * r**4)
            dvm_dr = (sigma_n * dsigma_n_dr + 3.0 * tau * dtau_dr) / vm
            dk_dr = dkloc_dr(ds.model.E_beam, ds.model.G_beam, r,
                             float(ds.beam_lengths[e_star]))
            dfloc_dr = dk_dr @ fac.transforms[e_star] @ fac.u[dofs]
            dvm_dr += dvm_dfloc @ dfloc_dr
            du_part[int(ds.group_of_element[e_star])] += dvm_dr
            return -du_part / sa

        def beam_con_jac(x):
            """Full (n, nx) Jacobian: row e is ∂(1 − vM_e/σ)/∂x at e's binding load case."""
            facs, ds, sections, sens_cache, dF_of_fac = _jac_lookup(x)
            # per-element binding lc = the lc achieving max vM_e (matches evaluate's max).
            vm_lc = np.empty((len(facs), n))
            for li, fac in enumerate(facs):
                vm_lc[li] = von_mises_per_element(fac.result, sections)
            binding = np.argmax(vm_lc, axis=0)
            Jrows = np.empty((n, nx))
            for e in range(n):
                bi = int(binding[e])
                Jrows[e] = _beam_vm_grad_one(facs[bi], ds, e, sens_cache, dF=dF_of_fac[bi])
            return Jrows

        def skin_con_jac(x):
            if config.skin_failure == "tsai_wu":
                row = _binding_grad(
                    x, lambda fac, ds, cache, dF, ci: grad_skin_tsai_wu(
                        fac, ds, safety_factor=config.tsai_wu_safety_factor, cache=cache, dF=dF))
            else:
                row = _binding_grad(
                    x, lambda fac, ds, cache, dF, ci: grad_skin_vm(
                        fac, ds, config.sigma_allow_Pa, cache=cache, dF=dF,
                        qw2_tri=(qw2_cases[ci // max(len(accels), 1)]
                                 if qw2_cases is not None else None)))
            return row.reshape(1, nx)

        def defl_con_jac(x):
            def gf(fac, ds, cache, dF, ci):
                g, grad_g = grad_tip_defl(fac, ds, cache=cache, dF=dF)
                return 1.0 - g / config.tip_defl_max_m, -grad_g / config.tip_defl_max_m
            return _binding_grad(x, gf).reshape(1, nx)

        def twist_con_jac(x):
            def gf(fac, ds, cache, dF, ci):
                g, grad_g = grad_tip_twist(fac, ds, cache=cache, dF=dF)
                return 1.0 - g / config.tip_twist_max_deg, -np.degrees(grad_g) / config.tip_twist_max_deg
            return _binding_grad(x, gf).reshape(1, nx)

        def frac_con_jac(x):
            J = np.zeros((L, nx))
            for g in range(L):
                J[g, f0_lo + g] = -1.0
                J[g, f45_lo + g] = -1.0
            return J

        def monotonic_con_jac(x):
            J = np.zeros((mono_lo.size, nx))
            for k in range(mono_lo.size):
                J[k, int(mono_lo[k])] += 1.0
                J[k, int(mono_hi[k])] += -1.0
            return J

        jacs = {
            "beam_vm": beam_con_jac, "skin": skin_con_jac, "defl": defl_con_jac,
            "twist": twist_con_jac, "frac": frac_con_jac, "mono": monotonic_con_jac,
        }
        if config.buckling_safety_factor is not None:
            def beam_buck_con_jac(x):
                return _binding_grad(
                    x, lambda fac, ds, cache, dF, ci: grad_beam_buckling(
                        fac, ds, euler_K=config.euler_K,
                        safety_factor=config.buckling_safety_factor, cache=cache, dF=dF)).reshape(1, nx)

            def panel_buck_con_jac(x):
                return _binding_grad(
                    x, lambda fac, ds, cache, dF, ci: grad_panel_buckling(
                        fac, ds, panel_kc=config.panel_kc,
                        safety_factor=config.buckling_safety_factor, areas=b2_panel, cache=cache, dF=dF)).reshape(1, nx)

            jacs["beam_buck"] = beam_buck_con_jac
            jacs["panel_buck"] = panel_buck_con_jac

    # P.0: one named spec per constraint — append HERE to add a constraint; the
    # scipy dicts, Jacobian registration, and shadow-price attribution all follow.
    specs = [
        ConstraintSpec("beam_vm", beam_con, n, jacs.get("beam_vm"),
                       ("limit", "sigma_allow_Pa", config.sigma_allow_Pa)),
        ConstraintSpec("skin", skin_con, 1, jacs.get("skin"),
                       (("sf", "tsai_wu_safety_factor", config.tsai_wu_safety_factor)
                        if config.skin_failure == "tsai_wu"
                        else ("limit", "sigma_allow_Pa", config.sigma_allow_Pa))),
        ConstraintSpec("defl", defl_con, 1, jacs.get("defl"),
                       ("limit", "tip_defl_max_m", config.tip_defl_max_m)),
        ConstraintSpec("twist", twist_con, 1, jacs.get("twist"),
                       ("limit", "tip_twist_max_deg", config.tip_twist_max_deg)),
        ConstraintSpec("frac", frac_con, L, jacs.get("frac"), None),
    ]
    if mono_lo.size > 0:
        specs.append(ConstraintSpec("mono", monotonic_con, int(mono_lo.size),
                                    jacs.get("mono"), None))
    if config.buckling_safety_factor is not None:
        sf = config.buckling_safety_factor
        specs += [
            ConstraintSpec("beam_buck", beam_buck_con, 1, jacs.get("beam_buck"),
                           ("sf", "buckling_sf_beam", sf)),
            ConstraintSpec("panel_buck", panel_buck_con, 1, jacs.get("panel_buck"),
                           ("sf", "buckling_sf_panel", sf)),
        ]
    constraints = [s.scipy_dict() for s in specs]

    res = minimize(
        mass, x0, jac=mass_grad, method="SLSQP", bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol},
    )

    shadow = shadow_prices_from_specs(
        getattr(res, "multipliers", None), specs, m_ref=m_ref)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = np.clip(res.x, lo, hi)
    # Per-group simplex re-projection: if f0_g + f45_g > 1, rescale that group.
    fg = x[f0_lo:f0_lo + L].copy()
    hg = x[f45_lo:f45_lo + L].copy()
    scale = np.where(fg + hg > 1.0, fg + hg, 1.0)
    x[f0_lo:f0_lo + L] = fg / scale
    x[f45_lo:f45_lo + L] = hg / scale
    wb, ws, d, t, bbu, pbu, worst_R = evaluate(x)
    radii = x[:G][group_of_element]
    t_bands = x[G:G + B].copy()
    t_tri = t_bands[band_of_tri]
    f0b, f45b, f90b = band_fracs(x)
    f0_tri = f0b[band_of_tri]
    f45_tri = f45b[band_of_tri]
    f90_tri = f90b[band_of_tri]
    bm = beam_mass(model, radii, rho=rho)
    sm = float(rho * np.sum(t_tri * Atri))
    t_skin_mean = float(np.sum(t_tri * Atri) / A_skin_area)
    f0_mean = float(np.sum(f0_tri * Atri) / A_skin_area)
    f45_mean = float(np.sum(f45_tri * Atri) / A_skin_area)
    f90_mean = float(np.sum(f90_tri * Atri) / A_skin_area)
    return LaminateSizingResult(
        radii=radii, t_skin=t_skin_mean, t_bands=t_bands,
        f0=f0_mean, f45=f45_mean, f90=f90_mean,
        f0_bands=f0b, f45_bands=f45b, f90_bands=f90b,
        mass_kg=bm + sm, beam_mass_kg=bm, skin_mass_kg=sm,
        converged=bool(res.success), n_iter=int(res.nit),
        max_beam_vm_Pa=float(wb.max()), max_skin_vm_Pa=float(ws),
        tip_defl_m=float(d), tip_twist_deg=float(t),
        max_beam_buckling_util=float(bbu),
        max_panel_buckling_util=float(pbu),
        min_skin_strength_ratio=(float(worst_R) if config.skin_failure == "tsai_wu" else None),
        shadow_prices=shadow,
    )


@dataclass(frozen=True)
class MultiStartResult:
    best: LaminateSizingResult
    best_start_index: int
    n_starts: int
    n_feasible: int
    start_masses: tuple[float, ...]
    start_feasible: tuple[bool, ...]


def _multistart_worker(args):
    """Top-level worker (picklable) for the process-parallel start map (V.0.4)."""
    model, load_arrays, config, ply, rho, maxiter, ftol, x0, panel_pressures = args
    return size_beam_shell_laminate(model, load_arrays, config, ply=ply, rho=rho,
                                    maxiter=maxiter, ftol=ftol, x0=x0,
                                    panel_pressures=panel_pressures)


def size_beam_shell_laminate_multistart(
    model: BeamShellModel,
    load_arrays,
    config: LaminateSizingConfig,
    *,
    ply: UDPly,
    rho: float,
    n_starts: int = 8,
    seed: int = 0,
    maxiter: int = 80,
    ftol: float = 1.0e-4,
    n_workers: int | None = None,
    panel_pressures: list[np.ndarray] | None = None,
) -> MultiStartResult:
    """Run the sizer from ``n_starts`` initial guesses; return the best feasible result.

    Start 0 is the sizer's default guess (so the result is never worse than a single
    start); starts 1.. are uniform-random within the design bounds (per-group
    simplex-projected), from a seeded RNG (deterministic for a given ``seed``). The
    start map is serial by default; ``n_workers > 1`` (V.0.4) fans the starts over a
    process pool — identical results to serial (starts are independent and the start
    list is generated up front from the seed). NOTE: with ``n_workers > 1`` the
    calling script MUST guard its entry point with ``if __name__ == "__main__":``
    (macOS uses spawn, which re-imports the main module in every worker). Selection:
    min-mass among feasible results (``laminate_result_is_feasible``); if none
    feasible, min-mass overall with ``n_feasible == 0``.
    """
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}")
    B = config.n_skin_bands
    L = B if config.per_band_layup else 1
    rng = np.random.default_rng(seed)

    def make_start(k):
        if k == 0:
            return None
        if model is None:
            # model is None only in tests with a stubbed sizer; return a non-None sentinel
            # so the stub receives a distinct x0 (the stub ignores it).
            return rng.uniform(0.0, 1.0, size=1)
        lo, hi = laminate_design_bounds(model, config)
        _, G = beam_radius_groups(model)
        f0_lo = G + B
        f45_lo = G + B + L
        x = rng.uniform(lo, hi)
        fg = x[f0_lo:f0_lo + L]
        hg = x[f45_lo:f45_lo + L]
        scale = np.where(fg + hg > 1.0, fg + hg, 1.0)
        x[f0_lo:f0_lo + L] = fg / scale
        x[f45_lo:f45_lo + L] = hg / scale
        return x

    starts = [make_start(k) for k in range(n_starts)]
    if n_workers is not None and n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(model, load_arrays, config, ply, rho, maxiter, ftol, s, panel_pressures)
                for s in starts]
        with ProcessPoolExecutor(max_workers=min(n_workers, n_starts)) as pool:
            results = list(pool.map(_multistart_worker, jobs))
    else:
        results = [
            size_beam_shell_laminate(model, load_arrays, config, ply=ply, rho=rho,
                                     maxiter=maxiter, ftol=ftol, x0=s,
                                     panel_pressures=panel_pressures)
            for s in starts
        ]
    feasible = [laminate_result_is_feasible(r, config) for r in results]
    masses = [float(r.mass_kg) for r in results]

    feasible_idx = [i for i, ok in enumerate(feasible) if ok]
    pool = feasible_idx if feasible_idx else list(range(n_starts))
    best_idx = min(pool, key=lambda i: masses[i])
    return MultiStartResult(
        best=results[best_idx],
        best_start_index=int(best_idx),
        n_starts=int(n_starts),
        n_feasible=int(len(feasible_idx)),
        start_masses=tuple(masses),
        start_feasible=tuple(bool(b) for b in feasible),
    )
