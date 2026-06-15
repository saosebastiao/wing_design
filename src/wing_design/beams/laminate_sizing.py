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
from ..materials.unidir import (
    CORE_RAMP_M, CoreMaterial, UDPly, crimping_stress, laminate_stiffness,
    laminate_stiffness_offset, sandwich_D_factor, wrinkling_stress,
)
from ..structural.beam_shell import (
    solve_beam_shell_laminate, solve_beam_shell_laminate_factored,
    _recover_beam_forces as _recover_beam_forces_local,
)
from ..structural.frame import FrameResult, _element_rotation
from .sensitivity import (
    DesignSens, grad_beam_buckling, grad_skin_vm,
    grad_panel_buckling, grad_skin_tsai_wu, grad_tip_defl, grad_tip_twist,
    grad_core_check, grad_tube_wall_one,
    grad_ks_engine, ks_aggregate, ks_values_only,
    provider_beam_buck, provider_core_check, provider_defl, provider_panel,
    provider_skin_vm, provider_tube_wall, provider_twist,
    _active_beam_force, _annulus_props, _beam_vm_explicit_tube,
    adjoint_lambda, lambdaT_dK_x_cached, dkloc_annular, dkloc_dr,
)
from ..structural.buckling import (
    TUBE_WALL_KNOCKDOWN, beam_euler_utilization, beam_euler_utilization_I,
    beam_foundation_utilization, panel_buckling_utilization, tube_wall_utilization,
)
from ..structural.frame import BeamSection, von_mises_per_element
from ..structural.shell import membrane_von_mises, recover_membrane_strain, recover_membrane_stress_C
from .body_loads import body_load_jacobian, body_load_vector
from .constraints import ConstraintSpec, shadow_prices_from_specs
from .design_vector import DesignVector
from .shell_model import (
    BeamShellModel, beam_adjacent_widths, skin_datum_angles, skin_panel_widths,
)
from .shell_sizing import (
    beam_lengths, beam_mass, beam_radius_groups, skin_areas, skin_band_areas,
    skin_band_map, skin_mass,
)

# Test hook: populated by size_beam_shell_laminate just before scipy.minimize so
# _constraint_names_for_test can read constraint names without post-processing.
_last_specs: list = []


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
    # P.1 core tube DV bounds (used when the model carries tube_elements).
    tube_r_min: float = 0.02
    tube_t_min: float = 0.001
    # P.3 sandwich skin: set a CoreMaterial to enable per-band core-thickness DVs
    # (faces = the existing laminate split symmetrically; A unchanged, D scaled by
    # the sandwich factor; wrinkling + crimping checks gate the credit).
    core: CoreMaterial | None = None
    t_core_max: float = 0.06
    # P.2 lateral bracing ring radius DV bounds (used when model.brace_elements present).
    brace_r_min: float = 0.002
    brace_r_max: float = 0.05
    # Optimizer backend: "slsqp" (scipy, default) or "ipopt" (cyipopt; the
    # documented Toolbox #4 fallback — interior-point handles the large design
    # migrations SLSQP's active set stalls on, e.g. the P.3 skin->core shift).
    # Shadow prices are SLSQP-only for now (IPOPT multiplier signs unwired).
    optimizer: str = "slsqp"
    # V.0.3/P#7 KS smoothing: when set (e.g. 50.0), the active max-based
    # constraints (panel/beam-Euler/tube-wall/wrinkle/crimp/skin-vM/defl/twist)
    # each become ONE smooth KS scalar over (items x load combos) — removes the
    # argmax Jacobian kinks that stall optimizers on large migrations.
    # Conservative (KS >= max). Requires use_analytic_jacobian and the
    # von-Mises skin path. Hard maxes still drive reporting/feasibility.
    ks_rho: float | None = None
    # V.3c beam buckling model: "element" (historical element-length Euler) or
    # "foundation" (in-wing form beams as stringers on the bonded-skin elastic
    # foundation, Pcr = 2*sqrt(k*EI), k = 3*D22_chord*(1/w_l^3 + 1/w_r^3) from
    # the two adjacent strips). Requires ks_rho + ply_angle_datum (D22 needs a
    # coherent chord direction). Transition + tube elements keep element Euler.
    beam_buckling_model: str = "element"
    # V#2 panel-buckling D: "local" (historical triangle-local-frame D11 — the
    # local x-axis is a mesh-dependent span/chord patchwork) or "datum_ortho"
    # (datum-frame long-plate combination D* = sqrt(D11·D22)+D12+2·D66, span =
    # datum; exact kc=4 for isotropic laminates, directionally coherent with
    # the V.3c foundation D22). Requires ks_rho + ply_angle_datum (KS-only,
    # V.3c precedent). Demand stays the most-compressive principal stress.
    panel_d_mode: str = "local"
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
    # P.1 core tube (None when the model has no tube)
    r_tube: np.ndarray | None = None
    t_wall: np.ndarray | None = None
    t_hollow: np.ndarray | None = None     # P#1b wall per hollow radius-group
    tube_mass_kg: float = 0.0
    max_tube_wall_util: float = 0.0        # over ALL annular elements (tube + hollow)
    # P.3 sandwich skin
    t_core: np.ndarray | None = None       # (B,) core thickness per band
    core_mass_kg: float = 0.0
    max_wrinkle_util: float = 0.0
    max_crimp_util: float = 0.0
    # P.2 lateral bracing mass + solved radius
    brace_mass_kg: float = 0.0
    brace_radius: float | None = None
    # Incumbent guard: True when the returned design is the best hard-feasible
    # iterate seen during the run rather than the optimizer's final iterate
    # (the endpoint was heavier or infeasible). Feasible by construction;
    # converged stays False in that case — it is a design, not a certified
    # optimum. A feasible x0 therefore can never produce a worse result.
    used_incumbent: bool = False


def laminate_design_bounds(model: BeamShellModel, config: LaminateSizingConfig):
    """(lo, hi) bounds for [r_group(G), t_band(B), f0(L), f45(L) | r_tube(S), t_wall(S)]."""
    _, G = beam_radius_groups(model)
    B = config.n_skin_bands
    L = B if config.per_band_layup else 1
    lo = np.concatenate([np.full(G, config.r_min), np.full(B, config.t_min), np.zeros(2 * L)])
    hi = np.concatenate([np.full(G, config.r_max), np.full(B, config.t_max), np.ones(2 * L)])
    if getattr(model, "tube_elements", None) is not None:
        S = len(model.tube_elements)
        rb = np.asarray(model.tube_r_bounds, dtype=float)
        lo = np.concatenate([lo, np.full(S, config.tube_r_min), np.full(S, config.tube_t_min)])
        hi = np.concatenate([hi, rb, rb])
    if (getattr(model, "hollow_elements", None) is not None
            and len(model.hollow_elements) > 0):
        goe, _G = beam_radius_groups(model)
        Hn = len({int(goe[e]) for e in model.hollow_elements})
        lo = np.concatenate([lo, np.full(Hn, config.tube_t_min)])
        hi = np.concatenate([hi, np.full(Hn, config.r_max)])
    if config.core is not None:
        lo = np.concatenate([lo, np.zeros(B)])
        hi = np.concatenate([hi, np.full(B, config.t_core_max)])
    if getattr(model, "brace_elements", None) is not None:
        lo = np.concatenate([lo, np.array([config.brace_r_min])])
        hi = np.concatenate([hi, np.array([config.brace_r_max])])
    return lo, hi


def design_vector_from_result(
    model: BeamShellModel, config: LaminateSizingConfig, result: "LaminateSizingResult",
) -> np.ndarray:
    """Rebuild the optimizer design vector x from a prior result (warm starts).

    House convention: sweeps/refinements/feature-chain stages seed from the
    nearest prior optimum (``x0=``). Blocks the prior result lacks are padded
    EQUIVALENCE-PRESERVING where the physics allows it, so the seed evaluates
    identically to the prior optimum (hard-feasible -> incumbent-guarded):
    t_hollow seeds at the group radius (clamps to solid) and t_core at 0
    (phi(t,0)=1, monolithic exactly). A 3 mm t_hollow pad instead lost stage A
    of the 2026-06-12 chain (seed infeasible, 500 iters no convergence). Tube
    blocks have no "absent" equivalent; they keep the cold defaults. Blocks the
    new config lacks are dropped. The sizer clips to bounds and re-projects
    layup fractions onto the simplex on entry.
    """
    goe, G = beam_radius_groups(model)
    B = config.n_skin_bands
    L = B if config.per_band_layup else 1
    rg = np.empty(G)
    for g in range(G):
        rg[g] = float(np.asarray(result.radii, dtype=float)[np.argmax(goe == g)])
    tb = np.resize(np.asarray(result.t_bands, dtype=float), B)
    f0 = np.resize(np.asarray(result.f0_bands, dtype=float), L)
    f45 = np.resize(np.asarray(result.f45_bands, dtype=float), L)
    x = np.concatenate([rg, tb, f0, f45])
    if getattr(model, "tube_elements", None) is not None:
        S = len(model.tube_elements)
        if result.r_tube is not None and len(result.r_tube) == S:
            x = np.concatenate([x, np.asarray(result.r_tube, dtype=float),
                                np.asarray(result.t_wall, dtype=float)])
        else:
            x = np.concatenate([x, 0.8 * np.asarray(model.tube_r_bounds, dtype=float),
                                np.full(S, 0.003)])
    if (getattr(model, "hollow_elements", None) is not None
            and len(model.hollow_elements) > 0):
        hgroups = sorted({int(goe[e]) for e in model.hollow_elements})
        if result.t_hollow is not None and len(result.t_hollow) == len(hgroups):
            x = np.concatenate([x, np.asarray(result.t_hollow, dtype=float)])
        else:
            # solid-equivalent pad: t = group radius (decode clamps t <= r)
            x = np.concatenate([x, rg[hgroups]])
    if config.core is not None:
        if result.t_core is not None and len(result.t_core) == B:
            x = np.concatenate([x, np.asarray(result.t_core, dtype=float)])
        else:
            x = np.concatenate([x, np.zeros(B)])   # c=0: exactly monolithic
    if getattr(model, "brace_elements", None) is not None:
        rb = (float(result.brace_radius) if getattr(result, "brace_radius", None) is not None
              else float(config.brace_r_min))
        x = np.concatenate([x, np.array([rb])])
    return x


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
        if result.max_tube_wall_util > 1.0 + tol:
            return False
        if result.max_wrinkle_util > 1.0 + tol:
            return False
        if result.max_crimp_util > 1.0 + tol:
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
    # Extra stiff massless elements assembled into K but excluded from force
    # recovery and mass: the opt-in tip gusset and (P.1) the tube->beam bonds.
    # One shared section: the solver takes a single gusset array, so compose.
    _extra = [e for e in (model.tip_gusset_elements,
                          getattr(model, "tube_bond_elements", None)) if e is not None]
    gusset_elements = np.vstack(_extra) if _extra else None
    if gusset_elements is not None:
        _r_bond = max(filter(None, (model.tip_gusset_radius, 0.05)))
        gusset_section = BeamSection.circular(_r_bond)
    else:
        gusset_section = None
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
    tube = getattr(model, "tube_elements", None) is not None
    S = len(model.tube_elements) if tube else 0
    tube_el = model.tube_elements if tube else None
    hollow = (getattr(model, "hollow_elements", None) is not None
              and len(model.hollow_elements) > 0)
    if hollow:
        hollow_el = np.asarray(model.hollow_elements, dtype=int)
        hollow_groups = sorted({int(group_of_element[e]) for e in hollow_el})
        # all-or-none per group (mirror pairs share the segment, so this holds
        # by construction; guard against future grouping changes)
        hset = set(hollow_el.tolist())
        for g in hollow_groups:
            members = np.nonzero(group_of_element == g)[0]
            if not all(int(e) in hset for e in members):
                raise ValueError(f"radius group {g} mixes hollow and solid elements")
        H = len(hollow_groups)
        t_idx_of_group = {g: i for i, g in enumerate(hollow_groups)}
        hollow_mask = np.zeros(model.n_form_elements, dtype=bool)
        hollow_mask[hollow_el] = True
        # per hollow element: its t_hollow index
        t_idx_of_element = np.array([t_idx_of_group[int(group_of_element[e])]
                                     for e in hollow_el], dtype=int)
    else:
        H = 0
    annular = tube or hollow
    blocks = [("r_group", G), ("t_band", B), ("f0", L), ("f45", L)]
    if tube:
        blocks += [("r_tube", S), ("t_wall", S)]
    if hollow:
        blocks += [("t_hollow", H)]
    sandwich = config.core is not None
    if sandwich:
        blocks += [("t_core", B)]
    braced = getattr(model, "brace_elements", None) is not None
    N_br = len(model.brace_elements) if braced else 0
    brace_mask_all = np.zeros(n, dtype=bool)
    if braced:
        brace_mask_all[np.asarray(model.brace_elements, dtype=int)] = True
        blocks += [("brace_radius", 1)]
    dv = DesignVector(*blocks)
    # Precompute total brace length (constant; outside per-iter closures).
    brace_len_total = 0.0
    if braced:
        be = model.beam_elements[model.brace_elements]
        brace_len_total = float(np.sum(np.linalg.norm(
            model.nodes[be[:, 0]] - model.nodes[be[:, 1]], axis=1)))
    foundation = config.beam_buckling_model == "foundation"
    if config.beam_buckling_model not in ("element", "foundation"):
        raise ValueError(f"unknown beam_buckling_model: {config.beam_buckling_model!r}")
    panel_datum_ortho = config.panel_d_mode == "datum_ortho"
    if config.panel_d_mode not in ("local", "datum_ortho"):
        raise ValueError(f"unknown panel_d_mode: {config.panel_d_mode!r}")
    if panel_datum_ortho and (config.ks_rho is None or config.ply_angle_datum is None):
        raise ValueError("panel_d_mode='datum_ortho' requires ks_rho and ply_angle_datum")

    def panel_qstar_bands(f0b, f45b, f90b):
        """Per-band datum-frame (Qstar, dQstar_df0, dQstar_df45) for V#2.

        Qeff is thickness-independent, so thickness=1 is arbitrary; the t/c
        geometry enters via the same geom(t,c)/t factor as the local path.
        """
        from .sensitivity import dQeff_df, dQstar_df, ortho_plate_Qstar
        qs = np.empty(B); d0 = np.empty(B); d45 = np.empty(B)
        dQ0 = dQeff_df(ply, which="f0", offset_deg=0.0)
        dQ45 = dQeff_df(ply, which="f45", offset_deg=0.0)
        for b in range(B):
            Qd = laminate_stiffness(ply, f0=float(f0b[b]), f45=float(f45b[b]),
                                    f90=float(f90b[b]), thickness=1.0)[2]
            qs[b] = ortho_plate_Qstar(Qd)
            d0[b] = dQstar_df(Qd, dQ0)
            d45[b] = dQstar_df(Qd, dQ45)
        return qs, d0, d45
    if foundation:
        if config.ks_rho is None or config.ply_angle_datum is None:
            raise ValueError("beam_buckling_model='foundation' requires ks_rho "
                             "and ply_angle_datum")
        n_form_f = model.n_form_elements
        adj_w = beam_adjacent_widths(model)
        kgeo = 3.0 * (1.0 / adj_w[:, 0] ** 3 + 1.0 / adj_w[:, 1] ** 3)   # (n_form,)
        # P.2 lateral-bracing: rings credit the in-loop sizing ONLY through the
        # elastic-foundation stiffness (k_ring), not the FEA stiffness — so the
        # analytic gradient w.r.t. brace_radius flows purely through this chain.
        from ..structural.buckling import k_ring_stiffness, dk_ring_dr
        from .shell_model import braced_segment_mask
        braced_seg = braced_segment_mask(model)                  # (n_form,) bool
        L_ring_e = adj_w.mean(axis=1)                            # perimeter (adjacent-beam) spacing ≈ ring member length
        s_span_e = np.array([float(np.linalg.norm(
            model.nodes[model.beam_elements[e, 0]] - model.nodes[model.beam_elements[e, 1]]))
            for e in range(n_form_f)])                           # (n_form,) spanwise seg length
        brace_col = dv.slice("brace_radius").start if braced else None
        brace_alpha = 3.0  # ring end-restraint constant; 3 = cantilever (conservative)
        z_lv = model.nodes[:model.n_levels, 2]
        seg_in_wing = np.array([(z_lv[k] >= -1e-9 and z_lv[k + 1] >= -1e-9)
                                for k in range(model.n_levels - 1)])
        found_mask_all = np.zeros(n, dtype=bool)
        for e in range(n_form_f):
            found_mask_all[e] = seg_in_wing[e % (model.n_levels - 1)]
        # band of each form element = band of its segment level (skin_band_map
        # tiles 2 tris per (level, beam); reuse the level->band mapping)
        n_seg = model.n_levels - 1
        band_of_level = np.array([band_of_tri[2 * k * model.n_beams]
                                  for k in range(n_seg)])
        band_of_elem = np.array([band_of_level[e % n_seg] for e in range(n_form_f)])

        def foundation_k(x):
            """(k_e (n_form,), per-element chain list [(col, dk/dcol), ...])."""
            t_b = np.asarray(x[dv.slice("t_band")], dtype=float)
            f0b, f45b, f90b = band_fracs(x)
            c_b = (np.asarray(x[dv.slice("t_core")], dtype=float)
                   if sandwich else np.zeros(B))
            D22 = np.empty(B)
            dD22_dt = np.empty(B); dD22_df0 = np.empty(B); dD22_df45 = np.empty(B)
            dD22_dc = np.empty(B)
            from .sensitivity import (dAD_dt, dAD_dt_sandwich, dD_dc_sandwich,
                                      dQeff_df, sandwich_geom_factor)
            for b in range(B):
                _A, Dm, Qd = laminate_stiffness(ply, f0=float(f0b[b]), f45=float(f45b[b]),
                                                f90=float(f90b[b]), thickness=float(t_b[b]))
                tt = float(t_b[b]); cc = float(c_b[b])
                geom = sandwich_geom_factor(tt, cc) if sandwich else tt**3 / 12.0
                D22[b] = Qd[1, 1] * geom
                if sandwich:
                    _dA, dDt = dAD_dt_sandwich(Qd, tt, cc)
                    dD22_dt[b] = dDt[1, 1]
                    dD22_dc[b] = dD_dc_sandwich(Qd, tt, cc)[1, 1]
                else:
                    _dA, dDt = dAD_dt(Qd, tt)
                    dD22_dt[b] = dDt[1, 1]
                    dD22_dc[b] = 0.0
                dD22_df0[b] = dQeff_df(ply, which="f0", offset_deg=0.0)[1, 1] * geom
                dD22_df45[b] = dQeff_df(ply, which="f45", offset_deg=0.0)[1, 1] * geom
            k_e = kgeo * D22[band_of_elem]
            if braced:
                rb = float(x[brace_col])
                kr = k_ring_stiffness(rb, model.E_beam, L_ring_e, s_span_e, brace_alpha)
                k_e = k_e + np.where(braced_seg, kr, 0.0)
            chains = []
            lg_of_b = (np.arange(B) if config.per_band_layup else np.zeros(B, dtype=int))
            for e in range(n_form_f):
                b = int(band_of_elem[e])
                lg = int(lg_of_b[b])
                ch = [(G + b, kgeo[e] * dD22_dt[b]),
                      (f0_lo + lg, kgeo[e] * dD22_df0[b]),
                      (f45_lo + lg, kgeo[e] * dD22_df45[b])]
                if sandwich:
                    ch.append((dv.slice("t_core").start + b, kgeo[e] * dD22_dc[b]))
                if braced and braced_seg[e]:
                    dkr = dk_ring_dr(float(x[brace_col]), model.E_beam,
                                     L_ring_e[e], s_span_e[e], brace_alpha)
                    ch.append((brace_col, float(dkr)))
                chains.append(ch)
            return k_e, chains
    nx = dv.nx
    f0_lo = dv.slice("f0").start
    f45_lo = dv.slice("f45").start

    def decode_tube(x):
        """(r_t, t_eff) tube DVs; t clamped to r so infeasible SLSQP trial points
        still evaluate (the r - t >= 0 constraint drives feasibility)."""
        r_t = np.asarray(x[dv.slice("r_tube")], dtype=float)
        t_w = np.asarray(x[dv.slice("t_wall")], dtype=float)
        return r_t, np.minimum(t_w, r_t)

    def decode_hollow(x, radii_form):
        """(t_eff per hollow element,) clamped to the element's outer radius."""
        t_h = np.asarray(x[dv.slice("t_hollow")], dtype=float)[t_idx_of_element]
        return np.minimum(t_h, radii_form[hollow_el])

    def build_sections(x, radii_form):
        if hollow:
            t_eff_h = decode_hollow(x, radii_form)
            secs = []
            hi = {int(e): k for k, e in enumerate(hollow_el)}
            for e, r in enumerate(radii_form):
                if hollow_mask[e]:
                    secs.append(BeamSection.annular(float(r), float(t_eff_h[hi[e]])))
                else:
                    secs.append(BeamSection.circular(float(r)))
        else:
            secs = [BeamSection.circular(float(r)) for r in radii_form]
        if tube:
            r_t, t_eff = decode_tube(x)
            secs += [BeamSection.annular(float(r_t[s]), float(t_eff[s])) for s in range(S)]
        if braced:
            # In-loop FEA uses a FIXED nominal brace radius, NOT the DV: the
            # rings' sizing benefit is modeled via the elastic-foundation credit
            # (Task 5, k_ring), keeping the adjoint clean (∂K/∂brace_radius = 0
            # in-loop). The post-hoc eigen check rebuilds rings at the sized
            # radius. Brace MASS still scales with the DV (see mass()).
            r_br = float(config.brace_r_min)
            secs += [BeamSection.circular(r_br)] * N_br
        return secs

    def annular_arrays(x, radii_form):
        """(elements, r, t, r_cols, t_cols) over the combined annular set
        (hollow form elements first, then tube segments)."""
        els, rs, ts, rcols, tcols = [], [], [], [], []
        if hollow:
            t_eff_h = decode_hollow(x, radii_form)
            for k, e in enumerate(hollow_el):
                els.append(int(e))
                rs.append(float(radii_form[e]))
                ts.append(float(t_eff_h[k]))
                rcols.append(int(group_of_element[e]))
                tcols.append(dv.slice("t_hollow").start + int(t_idx_of_element[k]))
        if tube:
            r_t, t_eff = decode_tube(x)
            r0 = dv.slice("r_tube").start
            t0 = dv.slice("t_wall").start
            for s in range(S):
                els.append(int(tube_el[s]))
                rs.append(float(r_t[s]))
                ts.append(float(t_eff[s]))
                rcols.append(r0 + s)
                tcols.append(t0 + s)
        return (np.asarray(els, dtype=int), np.asarray(rs), np.asarray(ts),
                np.asarray(rcols, dtype=int), np.asarray(tcols, dtype=int))
    if x0 is None:
        x0 = np.concatenate([
            np.full(G, config.r_max), np.full(B, config.t_max),
            np.full(L, 1.0 / 3.0), np.full(L, 1.0 / 3.0),
        ])
        if tube:
            x0 = np.concatenate([x0, 0.8 * np.asarray(model.tube_r_bounds, dtype=float),
                                 np.full(S, 0.003)])
        if hollow:
            x0 = np.concatenate([x0, np.full(H, 0.003)])
        if sandwich:
            x0 = np.concatenate([x0, np.full(B, 0.005)])
        if braced:
            x0 = np.concatenate([x0, np.array([(config.brace_r_min + config.brace_r_max) / 2.0])])
        assert x0.shape[0] == nx, f"cold-start x0 length {x0.shape[0]} != nx {nx}"
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
    incumbent = {"mass": np.inf, "x": None}

    accels = [np.asarray(a, dtype=float) for a in config.accel_vectors]

    def effective_loads(x, radii, t_tri):
        """Aero × accel combos; aero-only when no accelerations configured."""
        if not accels:
            return load_arrays
        # areas only when a tube is present OR the model is braced: the no-tube
        # no-brace path keeps the original radii-based product association inside
        # body_load_vector (bit-compat — regrouping 0.5*rho*pi*r^2*L is a 1-ulp
        # change that shifts cold-start basins; see the P.0 ulp lesson).
        # Braced models MUST pass areas (full-length over all beam_elements incl.
        # braces); indexing form-only `radii` with len(beam_elements) overruns.
        areas_e = (np.array([sec.A for sec in build_sections(x, radii)])
                   if (annular or braced) else None)
        extra = None
        if sandwich:
            extra = config.core.rho * np.asarray(
                x[dv.slice("t_core")], dtype=float)[band_of_tri]
        out = []
        for lc in load_arrays:
            for acc in accels:
                out.append(lc + body_load_vector(model, radii, t_tri, rho=rho,
                                                 accel=acc, areas=areas_e,
                                                 extra_skin_density=extra))
        return out

    def evaluate(x):
        key = np.asarray(x, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        radii = x[:G][group_of_element]          # form-beam radii per form element
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
        if sandwich:
            c_band = np.asarray(x[dv.slice("t_core")], dtype=float)
            c_tri_e = c_band[band_of_tri]
            phi_tri = sandwich_D_factor(t_tri, c_tri_e)
            D_arg = D_arg * phi_tri[:, None, None]
        D11 = D_arg[:, 0, 0]
        if panel_datum_ortho:
            # V#2: σcr from ½·D* in place of the local D11 (same kc semantics).
            qs_b, _d0, _d45 = panel_qstar_bands(f0b, f45b, f90b)
            Dhalf_tri = 0.5 * qs_b[band_of_tri] * t_tri**3 / 12.0
            if sandwich:
                Dhalf_tri = Dhalf_tri * phi_tri
        sections = build_sections(x, radii)
        areas_e = np.array([sec.A for sec in sections])
        I_e = np.array([sec.Iy for sec in sections])
        if annular:
            ann_el, ann_r, ann_t, _arc, _atc = annular_arrays(x, radii)
            tube_util = np.zeros(len(ann_el))
        wb = np.zeros(n)
        ws = 0.0
        md = 0.0
        mt = 0.0
        worst_beam_buck = 0.0
        worst_panel_buck = 0.0
        worst_wrk = 0.0
        worst_crp = 0.0
        worst_R = np.inf
        if sandwich:
            sig_wr = wrinkling_stress(C_arg[:, 0, 0], config.core)
            sig_cp = crimping_stress(t_tri, c_tri_e, config.core)
            ramp = c_tri_e / (c_tri_e + CORE_RAMP_M)
        n_acc = max(len(accels), 1)
        for ci, loads in enumerate(effective_loads(x, radii, t_tri)):
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
                # I-based Euler when a tube is present (annular sections); the
                # radii-based form is kept for the no-tube path bit-compatibly.
                if annular:
                    bu = beam_euler_utilization_I(res.axial_force, I_e, Lb, E=model.E_beam,
                                                  K=config.euler_K,
                                                  safety_factor=config.buckling_safety_factor)
                    tu = tube_wall_utilization(res.axial_force[ann_el], ann_r, ann_t,
                                               areas_e[ann_el], E=model.E_beam,
                                               safety_factor=config.buckling_safety_factor)
                    tube_util = np.maximum(tube_util, tu)
                else:
                    # `radii` covers form elements only; when braced, the brace
                    # (ring) elements append after them with the fixed nominal
                    # radius (build_sections). Use full-length radii so the
                    # Euler check aligns with res.axial_force/Lb (length n). The
                    # unbraced path passes `radii` unchanged (bit-compat).
                    radii_e = (np.array([sec.r for sec in sections]) if braced
                               else radii)
                    bu = beam_euler_utilization(res.axial_force, radii_e, Lb, E=model.E_beam,
                                                K=config.euler_K, safety_factor=config.buckling_safety_factor)
                if foundation:
                    k_e, _ch = foundation_k(x)
                    EI_form = model.E_beam * I_e[:model.n_form_elements]
                    bu_f = beam_foundation_utilization(
                        res.axial_force[:model.n_form_elements], EI_form, k_e,
                        safety_factor=config.buckling_safety_factor)
                    fm = found_mask_all[:model.n_form_elements]
                    bu = bu.copy()
                    bu[:model.n_form_elements][fm] = bu_f[fm]
                if panel_datum_ortho:
                    pu = panel_buckling_utilization(skin_s, b2_panel, D11=Dhalf_tri, t=t_tri,
                                                    kc=config.panel_kc, safety_factor=config.buckling_safety_factor)
                else:
                    pu = panel_buckling_utilization(skin_s, b2_panel, D11=D11, t=t_tri,
                                                    kc=config.panel_kc, safety_factor=config.buckling_safety_factor)
                worst_beam_buck = max(worst_beam_buck, float(bu.max()))
                worst_panel_buck = max(worst_panel_buck, float(pu.max()))
                if sandwich:
                    sarr = np.asarray(skin_s)
                    mean = 0.5 * (sarr[:, 0] + sarr[:, 1])
                    rad = np.sqrt(0.25 * (sarr[:, 0] - sarr[:, 1])**2 + sarr[:, 2]**2)
                    compp = np.maximum(0.0, -(mean - rad))
                    sf = config.buckling_safety_factor
                    worst_wrk = max(worst_wrk, float((compp * sf / sig_wr * ramp).max()))
                    worst_crp = max(worst_crp, float((compp * sf / sig_cp * ramp).max()))
        out = (wb, ws, md, mt, worst_beam_buck, worst_panel_buck, worst_R,
               tube_util if annular else None, worst_wrk, worst_crp)
        cache[key] = out
        # incumbent guard: keep the lightest HARD-feasible design ever visited
        tol = 1.0e-3
        feas = float(wb.max()) <= config.sigma_allow_Pa * (1 + tol)
        if config.skin_failure == "tsai_wu":
            feas &= (worst_R >= config.tsai_wu_safety_factor * (1 - tol))
        else:
            feas &= ws <= config.sigma_allow_Pa * (1 + tol)
        feas &= md <= config.tip_defl_max_m * (1 + tol)
        feas &= mt <= config.tip_twist_max_deg * (1 + tol)
        if config.buckling_safety_factor is not None:
            feas &= worst_beam_buck <= 1 + tol and worst_panel_buck <= 1 + tol
            if annular:
                feas &= float(tube_util.max()) <= 1 + tol
            feas &= worst_wrk <= 1 + tol and worst_crp <= 1 + tol
        if feas:
            m_now = mass(x) * m_ref
            if m_now < incumbent["mass"]:
                incumbent["mass"] = m_now
                incumbent["x"] = np.asarray(x, dtype=float).copy()
        return out

    m_ref = beam_mass(model, np.full(n, config.r_max), rho=rho) + skin_mass(model, config.t_max, rho=rho)

    Lb_form = Lb[:model.n_form_elements]
    Lb_tube = Lb[tube_el] if tube else None

    def mass(x):
        radii = x[:G][group_of_element]
        if hollow:
            area_form = np.pi * radii ** 2
            t_eff_h = decode_hollow(x, radii)
            rh = radii[hollow_el]
            area_form[hollow_el] = np.pi * (rh**2 - (rh - t_eff_h)**2)
            m = rho * np.sum(area_form * Lb_form) + rho * np.sum(x[G:G + B] * band_area)
        else:
            # plain-path expression kept character-identical (ulp discipline)
            m = rho * np.sum(np.pi * radii ** 2 * Lb_form) + rho * np.sum(x[G:G + B] * band_area)
        if tube:
            r_t, t_eff = decode_tube(x)
            ri = r_t - t_eff
            m += rho * np.sum(np.pi * (r_t**2 - ri**2) * Lb_tube)
        if sandwich:
            m += config.core.rho * np.sum(x[dv.slice("t_core")] * band_area)
        if braced:
            r_brace = float(x[dv.slice("brace_radius").start])
            m += rho * np.pi * r_brace ** 2 * brace_len_total
        return m / m_ref

    def mass_grad(x):
        g = np.zeros(nx)
        radii = x[:G][group_of_element]
        if hollow:
            t_eff_h = decode_hollow(x, radii)
            per_elem = rho * 2.0 * np.pi * radii * Lb_form / m_ref
            per_elem[hollow_el] = rho * 2.0 * np.pi * t_eff_h * Lb_form[hollow_el] / m_ref
            np.add.at(g, group_of_element, per_elem)
            dA_dt = 2.0 * np.pi * (radii[hollow_el] - t_eff_h)
            t0 = dv.slice("t_hollow").start
            np.add.at(g, t0 + t_idx_of_element,
                      rho * dA_dt * Lb_form[hollow_el] / m_ref)
        else:
            # plain-path expression kept character-identical (ulp discipline)
            per_elem = rho * 2.0 * np.pi * radii * Lb_form / m_ref
            np.add.at(g, group_of_element, per_elem)   # scatter element grads into the G groups
        g[G:G + B] = rho * band_area / m_ref
        if tube:
            r_t, t_eff = decode_tube(x)
            ri = r_t - t_eff
            g[dv.slice("r_tube")] = rho * 2.0 * np.pi * t_eff * Lb_tube / m_ref
            g[dv.slice("t_wall")] = rho * 2.0 * np.pi * ri * Lb_tube / m_ref
        if sandwich:
            g[dv.slice("t_core")] = config.core.rho * band_area / m_ref
        if braced:
            r_brace = float(x[dv.slice("brace_radius").start])
            g[dv.slice("brace_radius").start] = (
                rho * 2.0 * np.pi * r_brace * brace_len_total / m_ref
            )
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
            if sandwich:
                c_band_j = np.asarray(x[dv.slice("t_core")], dtype=float)
                c_tri_j = c_band_j[band_of_tri]
                D_arg = D_arg * sandwich_D_factor(t_tri, c_tri_j)[:, None, None]
            if panel_datum_ortho:
                pq_j, pd0_j, pd45_j = panel_qstar_bands(f0b, f45b, f90b)

            sections = build_sections(x, radii)
            # radii_full spans ALL beam elements (form + tube + brace); brace
            # (ring) rows carry the fixed nominal radius (build_sections). `radii`
            # is form-only, so derive the full-length array from the sections.
            radii_full_arr = (np.array([sec.r for sec in sections]) if braced
                              else radii)
            loads_eff = effective_loads(x, radii, t_tri)
            tube_kw = {}
            if annular:
                ann_el_j, ann_r_j, ann_t_j, ann_rc_j, ann_tc_j = annular_arrays(x, radii)
                tube_kw = dict(S=len(ann_el_j), tube_elements=ann_el_j,
                               tube_r=ann_r_j, tube_t=ann_t_j,
                               tube_r_cols=ann_rc_j, tube_t_cols=ann_tc_j)
            if annular or sandwich:
                tube_kw["nx_extra"] = nx - (G + B + 2 * L)
            if sandwich:
                tube_kw["core_rho"] = config.core.rho
                tube_kw["core_col_of_tri"] = (dv.slice("t_core").start
                                              + band_of_tri.astype(int))
            # V.4: one ∂f/∂x per accel vector (zero for pure aero); fac k pairs
            # with accel k % len(accels) by the effective_loads combo order.
            if accels:
                dFs = [body_load_jacobian(
                           model, radii, group_of_element=group_of_element,
                           band_of_tri=band_of_tri, rho=rho, accel=acc, G=G, B=B, L=L,
                           brace_elements=(model.brace_elements if braced else None),
                           **tube_kw)
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
                radii_full=radii_full_arr,
                brace_mask=(brace_mask_all if braced else None),
                beam_lengths=beam_lengths_e,
                t_tri=t_tri,
                Qeff_tri=Qeff_tri,
                offset_tri=offset_tri,
                ply=ply,
                f0_tri=(f0_tri if config.skin_failure == "tsai_wu" else None),
                f45_tri=(f45_tri if config.skin_failure == "tsai_wu" else None),
                f90_tri=(f90_tri if config.skin_failure == "tsai_wu" else None),
                S=(len(ann_el_j) if annular else 0),
                tube_elements=(ann_el_j if annular else None),
                tube_r=(ann_r_j if annular else None),
                tube_t=(ann_t_j if annular else None),
                tube_r_cols=(ann_rc_j if annular else None),
                tube_t_cols=(ann_tc_j if annular else None),
                nx_extra=(nx - (G + B + 2 * L)),
                core=(config.core if sandwich else None),
                c_tri=(c_tri_j if sandwich else None),
                core_col_of_tri=((dv.slice("t_core").start
                                  + band_of_tri.astype(int)) if sandwich else None),
                found_k=(foundation_k(x)[0] if foundation else None),
                found_dk=(foundation_k(x)[1] if foundation else None),
                found_mask=(found_mask_all if foundation else None),
                panel_Qstar=(pq_j if panel_datum_ortho else None),
                panel_dQstar_df0=(pd0_j if panel_datum_ortho else None),
                panel_dQstar_df45=(pd45_j if panel_datum_ortho else None),
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
            s_tube = ds.tube_seg_of_element(e_star)
            if s_tube >= 0:
                r = float(ds.tube_r[s_tube])
                A, Iz, J, _dA, _dI, _dJ = _annulus_props(r, float(ds.tube_t[s_tube]))
            else:
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

            if s_tube >= 0:
                dvm_dr_t, dvm_dt_t = _beam_vm_explicit_tube(
                    floc, dvm_dfloc, fac, ds, e_star, s_tube, sigma_n, tau, vm)
                du_part[ds.tube_dv_r(s_tube)] += dvm_dr_t
                du_part[ds.tube_dv_t(s_tube)] += dvm_dt_t
            elif ds.brace_mask is not None and bool(ds.brace_mask[e_star]):
                pass  # brace: fixed nominal radius, no explicit radius-DV column
            else:
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
        if annular and config.buckling_safety_factor is not None:
            def tube_wall_jac(x):
                """(Q, nx): row q at its binding load combo (max wall util)."""
                facs, ds, _sections, sens_cache, dF_of_fac = _jac_lookup(x)
                Q = ds.S
                util = np.empty((len(facs), Q))
                c = TUBE_WALL_KNOCKDOWN * 0.605 * model.E_beam
                for li, fac in enumerate(facs):
                    ax = np.array([fac.result.axial_force[int(e)] for e in ds.tube_elements])
                    A_t = np.pi * (ds.tube_r**2 - (ds.tube_r - ds.tube_t)**2)
                    scr = c * ds.tube_t / ds.tube_r
                    util[li] = (np.maximum(0.0, -ax) / A_t
                                * config.buckling_safety_factor / scr)
                binding = np.argmax(util, axis=0)
                Jrows = np.empty((Q, nx))
                for s_i in range(Q):
                    bi = int(binding[s_i])
                    _con, row = grad_tube_wall_one(
                        facs[bi], ds, s_i, safety_factor=config.buckling_safety_factor,
                        E=model.E_beam, knockdown=TUBE_WALL_KNOCKDOWN,
                        cache=sens_cache, dF=dF_of_fac[bi])
                    Jrows[s_i] = row
                return Jrows
            jacs["tube_wall"] = tube_wall_jac
        if sandwich and config.buckling_safety_factor is not None:
            def core_wrinkle_jac(x):
                return _binding_grad(
                    x, lambda fac, ds, cache, dF, ci: grad_core_check(
                        fac, ds, which="wrinkle",
                        safety_factor=config.buckling_safety_factor,
                        cache=cache, dF=dF)).reshape(1, nx)

            def core_crimp_jac(x):
                return _binding_grad(
                    x, lambda fac, ds, cache, dF, ci: grad_core_check(
                        fac, ds, which="crimp",
                        safety_factor=config.buckling_safety_factor,
                        cache=cache, dF=dF)).reshape(1, nx)
            jacs["core_wrinkle"] = core_wrinkle_jac
            jacs["core_crimp"] = core_crimp_jac
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

    if hollow:
        h_t0 = dv.slice("t_hollow").start

        def hollow_validity_con(x):
            t_h = x[dv.slice("t_hollow")]
            r_g = np.array([x[g] for g in hollow_groups])
            return r_g - t_h                   # >= 0: wall fits inside the radius

        def hollow_validity_jac(x):
            Jm = np.zeros((H, nx))
            for i, g in enumerate(hollow_groups):
                Jm[i, g] = 1.0
                Jm[i, h_t0 + i] = -1.0
            return Jm

    if sandwich:
        def core_wrinkle_con(x):
            return np.array([1.0 - evaluate(x)[8]])

        def core_crimp_con(x):
            return np.array([1.0 - evaluate(x)[9]])

    if annular:
        def tube_wall_con(x):
            return 1.0 - evaluate(x)[7]

    if tube:
        def tube_validity_con(x):
            r_t = x[dv.slice("r_tube")]
            t_w = x[dv.slice("t_wall")]
            return r_t - t_w                       # >= 0: annulus validity

        # monotonic non-increasing r_tube keel -> tip (same z ordering as beams)
        tmono_lo, tmono_hi = [], []
        for k in range(S - 1):
            if seg_z[k] >= seg_z[k + 1]:           # segment k is tip-ward
                tmono_lo.append(k + 1); tmono_hi.append(k)
            else:
                tmono_lo.append(k); tmono_hi.append(k + 1)
        r_tube0 = dv.slice("r_tube").start

        def tube_mono_con(x):
            r_t = x[dv.slice("r_tube")]
            return r_t[tmono_lo] - r_t[tmono_hi]

        def tube_validity_jac(x):
            Jm = np.zeros((S, nx))
            for s_i in range(S):
                Jm[s_i, r_tube0 + s_i] = 1.0
                Jm[s_i, dv.slice("t_wall").start + s_i] = -1.0
            return Jm

        def tube_mono_jac(x):
            Jm = np.zeros((S - 1, nx))
            for k in range(S - 1):
                Jm[k, r_tube0 + tmono_lo[k]] += 1.0
                Jm[k, r_tube0 + tmono_hi[k]] += -1.0
            return Jm

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
    if annular:
        Q_ann = (len(hollow_el) if hollow else 0) + S
        if config.buckling_safety_factor is not None:
            specs.append(ConstraintSpec(
                "tube_wall", tube_wall_con, Q_ann, jacs.get("tube_wall"),
                ("sf", "buckling_sf_tube_wall", config.buckling_safety_factor)))
    if sandwich and config.buckling_safety_factor is not None:
        specs += [
            ConstraintSpec("core_wrinkle", core_wrinkle_con, 1, jacs.get("core_wrinkle"),
                           ("sf", "buckling_sf_core_wrinkle", config.buckling_safety_factor)),
            ConstraintSpec("core_crimp", core_crimp_con, 1, jacs.get("core_crimp"),
                           ("sf", "buckling_sf_core_crimp", config.buckling_safety_factor)),
        ]
    if hollow:
        specs.append(ConstraintSpec("hollow_validity", hollow_validity_con, H,
                                    hollow_validity_jac if config.use_analytic_jacobian else None))
    if tube:
        specs.append(ConstraintSpec("tube_validity", tube_validity_con, S,
                                    tube_validity_jac if config.use_analytic_jacobian else None))
        if S > 1:
            specs.append(ConstraintSpec("tube_mono", tube_mono_con, S - 1,
                                        tube_mono_jac if config.use_analytic_jacobian else None))
    if config.ks_rho is not None:
        if not config.use_analytic_jacobian:
            raise ValueError("ks_rho requires use_analytic_jacobian=True")
        if config.skin_failure == "tsai_wu":
            raise NotImplementedError("ks_rho with tsai_wu skin is not wired")
        rho_ks = float(config.ks_rho)
        sf_b = config.buckling_safety_factor
        n_combos = len(load_arrays) * max(len(accels), 1)
        n_acc_ks = max(len(accels), 1)
        ks_providers = {
            "defl": provider_defl(config.tip_defl_max_m, model.tip_nodes),
            "twist": provider_twist(config.tip_twist_max_deg, model.tip_nodes),
            "skin": [provider_skin_vm(
                         config.sigma_allow_Pa,
                         (lambda ci: qw2_cases[ci // n_acc_ks])
                         if qw2_cases is not None else None)(ci)
                     for ci in range(n_combos)],
        }
        if sf_b is not None:
            ks_providers["panel_buck"] = provider_panel(config.panel_kc, sf_b, b2_panel)
            ks_providers["beam_buck"] = provider_beam_buck(config.euler_K, sf_b)
            if annular:
                ks_providers["tube_wall"] = provider_tube_wall(sf_b, model.E_beam)
            if sandwich:
                ks_providers["core_wrinkle"] = provider_core_check("wrinkle", sf_b)
                ks_providers["core_crimp"] = provider_core_check("crimp", sf_b)

        def ks_pair(provider):
            def fun(x):
                facs, ds, _s, _cache, _dFs = _jac_lookup(x)
                return np.array([ks_values_only(facs, ds, rho_ks, provider)])

            def jac(x):
                facs, ds, _s, cache, dFs = _jac_lookup(x)
                _con, grad = grad_ks_engine(facs, ds, cache, dFs, rho_ks, provider)
                return grad.reshape(1, nx)
            return fun, jac

        new_specs = []
        for sp in specs:
            if sp.name in ks_providers:
                fun, jac = ks_pair(ks_providers[sp.name])
                new_specs.append(ConstraintSpec(sp.name + "_ks", fun, 1, jac, sp.shadow))
            else:
                new_specs.append(sp)
        specs = new_specs

    constraints = [s.scipy_dict() for s in specs]

    # Expose the assembled spec list for test introspection (see _constraint_names_for_test).
    global _last_specs
    _last_specs = list(specs)

    # seed the incumbent guard with the exact x0 (the optimizer perturbs its
    # start, so without this the seed itself might never enter the visited set)
    evaluate(np.asarray(x0, dtype=float))

    if config.optimizer == "ipopt":
        from cyipopt import minimize_ipopt
        res = minimize_ipopt(
            mass, x0, jac=mass_grad, bounds=bounds, constraints=constraints,
            options={"max_iter": int(maxiter), "tol": float(ftol),
                     "hessian_approximation": "limited-memory",
                     "mu_strategy": "adaptive", "print_level": 0,
                     "sb": "yes",
                     # honor x0: without these, the barrier init pushes the
                     # iterate to the interior centre (= cold start)
                     "warm_start_init_point": "yes",
                     "mu_init": 1.0e-6,
                     "warm_start_bound_push": 1.0e-8,
                     "warm_start_mult_bound_push": 1.0e-8,
                     "bound_push": 1.0e-8, "bound_frac": 1.0e-8},
        )
    elif config.optimizer == "slsqp":
        res = minimize(
            mass, x0, jac=mass_grad, method="SLSQP", bounds=bounds,
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": ftol},
        )
    else:
        raise ValueError(f"unknown optimizer: {config.optimizer!r}")

    shadow = shadow_prices_from_specs(
        getattr(res, "multipliers", None), specs, m_ref=m_ref)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = np.clip(res.x, lo, hi)
    used_incumbent = False
    if incumbent["x"] is not None:
        end_mass = mass(x) * m_ref
        if incumbent["mass"] < end_mass - 1e-9 or not bool(res.success):
            if incumbent["mass"] <= end_mass + 1e-9:
                x_inc = np.clip(incumbent["x"], lo, hi)
                used_incumbent = not np.allclose(x_inc, x)
                x = x_inc
    # Per-group simplex re-projection: if f0_g + f45_g > 1, rescale that group.
    fg = x[f0_lo:f0_lo + L].copy()
    hg = x[f45_lo:f45_lo + L].copy()
    scale = np.where(fg + hg > 1.0, fg + hg, 1.0)
    x[f0_lo:f0_lo + L] = fg / scale
    x[f45_lo:f45_lo + L] = hg / scale
    wb, ws, d, t, bbu, pbu, worst_R, tube_util_vec, worst_wrk, worst_crp = evaluate(x)
    radii = x[:G][group_of_element]
    t_bands = x[G:G + B].copy()
    t_tri = t_bands[band_of_tri]
    f0b, f45b, f90b = band_fracs(x)
    f0_tri = f0b[band_of_tri]
    f45_tri = f45b[band_of_tri]
    f90_tri = f90b[band_of_tri]
    if hollow:
        area_res = np.pi * radii**2
        t_eff_res = decode_hollow(x, radii)
        rh = radii[hollow_el]
        area_res[hollow_el] = np.pi * (rh**2 - (rh - t_eff_res)**2)
        bm = float(rho * np.sum(area_res * Lb_form))
    else:
        bm = float(rho * np.sum(np.pi * radii**2 * Lb_form))   # form beams only
    tm = 0.0
    if tube:
        r_t_f, t_w_f = decode_tube(x)
        tm = float(rho * np.sum(np.pi * (r_t_f**2 - (r_t_f - t_w_f)**2) * Lb_tube))
    sm = float(rho * np.sum(t_tri * Atri))
    bracem = 0.0
    if braced:
        r_brace_f = float(x[dv.slice("brace_radius").start])
        bracem = float(rho * np.pi * r_brace_f ** 2 * brace_len_total)
    t_skin_mean = float(np.sum(t_tri * Atri) / A_skin_area)
    f0_mean = float(np.sum(f0_tri * Atri) / A_skin_area)
    f45_mean = float(np.sum(f45_tri * Atri) / A_skin_area)
    f90_mean = float(np.sum(f90_tri * Atri) / A_skin_area)
    return LaminateSizingResult(
        radii=radii, t_skin=t_skin_mean, t_bands=t_bands,
        f0=f0_mean, f45=f45_mean, f90=f90_mean,
        f0_bands=f0b, f45_bands=f45b, f90_bands=f90b,
        mass_kg=(bm + tm + sm + bracem
                 + (float(config.core.rho * np.sum(x[dv.slice("t_core")] * band_area))
                    if sandwich else 0.0)),
        beam_mass_kg=bm, skin_mass_kg=sm,
        converged=bool(res.success), n_iter=int(getattr(res, "nit", -1)),
        max_beam_vm_Pa=float(wb.max()), max_skin_vm_Pa=float(ws),
        tip_defl_m=float(d), tip_twist_deg=float(t),
        max_beam_buckling_util=float(bbu),
        max_panel_buckling_util=float(pbu),
        min_skin_strength_ratio=(float(worst_R) if config.skin_failure == "tsai_wu" else None),
        shadow_prices=shadow,
        r_tube=(decode_tube(x)[0] if tube else None),
        t_wall=(decode_tube(x)[1] if tube else None),
        t_hollow=(np.asarray(x[dv.slice("t_hollow")], dtype=float) if hollow else None),
        tube_mass_kg=tm,
        max_tube_wall_util=(float(tube_util_vec.max()) if annular else 0.0),
        t_core=(np.asarray(x[dv.slice("t_core")], dtype=float) if sandwich else None),
        core_mass_kg=(float(config.core.rho * np.sum(x[dv.slice("t_core")] * band_area))
                      if sandwich else 0.0),
        max_wrinkle_util=float(worst_wrk),
        max_crimp_util=float(worst_crp),
        brace_mass_kg=bracem,
        brace_radius=(float(x[dv.slice("brace_radius").start]) if braced else None),
        used_incumbent=used_incumbent,
    )


def _objective_and_grad_for_test(
    model: BeamShellModel,
    config: LaminateSizingConfig,
    x: np.ndarray,
    *,
    rho: float,
    ply: UDPly | None = None,
) -> tuple[float, np.ndarray]:
    """Return (normalized_objective, gradient) at x WITHOUT running the optimizer.

    Thin test hook exposing the mass / mass_grad closures used by
    size_beam_shell_laminate.  The objective is the same normalized mass
    minimized by SLSQP: mass(x) = total_mass_kg / m_ref.  Only the
    mass/mass_grad machinery is reconstructed here; no FEA is performed, so
    this is cheap and suitable for FD-validation tests.

    Parameters
    ----------
    model, config, x
        Same as the optimizer inputs.
    rho
        Structural density [kg/m³].
    ply
        Unused (kept for forward-compat); can be None.
    """
    n = model.beam_elements.shape[0]
    B = config.n_skin_bands
    L = B if config.per_band_layup else 1
    group_of_element, G = beam_radius_groups(model)
    Lb = beam_lengths(model)
    Lb_form = Lb[:model.n_form_elements]
    band_of_tri = skin_band_map(model, B)
    band_area = skin_band_areas(model, band_of_tri, B)

    tube = getattr(model, "tube_elements", None) is not None
    hollow = (getattr(model, "hollow_elements", None) is not None
              and len(model.hollow_elements) > 0)
    sandwich = config.core is not None
    braced = getattr(model, "brace_elements", None) is not None

    blocks: list = [("r_group", G), ("t_band", B), ("f0", L), ("f45", L)]
    if tube:
        S = len(model.tube_elements)
        blocks += [("r_tube", S), ("t_wall", S)]
    else:
        S = 0
    if hollow:
        hollow_el = np.asarray(model.hollow_elements, dtype=int)
        H = len({int(group_of_element[e]) for e in hollow_el})
        blocks += [("t_hollow", H)]
    if sandwich:
        blocks += [("t_core", B)]
    if braced:
        blocks += [("brace_radius", 1)]
    dv = DesignVector(*blocks)

    brace_len_total = 0.0
    if braced:
        be = model.beam_elements[model.brace_elements]
        brace_len_total = float(np.sum(np.linalg.norm(
            model.nodes[be[:, 0]] - model.nodes[be[:, 1]], axis=1)))

    m_ref = beam_mass(model, np.full(n, config.r_max), rho=rho) + skin_mass(model, config.t_max, rho=rho)

    nx = dv.nx
    x = np.asarray(x, dtype=float)

    def _mass(xv):
        radii = xv[:G][group_of_element]
        m = (rho * np.sum(np.pi * radii ** 2 * Lb_form)
             + rho * np.sum(xv[G:G + B] * band_area))
        if tube:
            r_t = xv[dv.slice("r_tube")]
            t_w = xv[dv.slice("t_wall")]
            tube_elems = np.asarray(model.tube_elements, dtype=int)
            Lb_tube = Lb[tube_elems]
            ri = r_t - t_w
            m += rho * np.sum(np.pi * (r_t ** 2 - ri ** 2) * Lb_tube)
        if sandwich:
            m += config.core.rho * np.sum(xv[dv.slice("t_core")] * band_area)
        if braced:
            r_brace = float(xv[dv.slice("brace_radius").start])
            m += rho * np.pi * r_brace ** 2 * brace_len_total
        return m / m_ref

    def _mass_grad(xv):
        g = np.zeros(nx)
        radii = xv[:G][group_of_element]
        per_elem = rho * 2.0 * np.pi * radii * Lb_form / m_ref
        np.add.at(g, group_of_element, per_elem)
        g[G:G + B] = rho * band_area / m_ref
        if tube:
            r_t = xv[dv.slice("r_tube")]
            t_w = xv[dv.slice("t_wall")]
            tube_elems = np.asarray(model.tube_elements, dtype=int)
            Lb_tube = Lb[tube_elems]
            ri = r_t - t_w
            g[dv.slice("r_tube")] = rho * 2.0 * np.pi * t_w * Lb_tube / m_ref
            g[dv.slice("t_wall")] = rho * 2.0 * np.pi * ri * Lb_tube / m_ref
        if sandwich:
            g[dv.slice("t_core")] = config.core.rho * band_area / m_ref
        if braced:
            r_brace = float(xv[dv.slice("brace_radius").start])
            g[dv.slice("brace_radius").start] = (
                rho * 2.0 * np.pi * r_brace * brace_len_total / m_ref
            )
        return g

    return float(_mass(x)), _mass_grad(x)


def _constraint_names_for_test(
    model,
    config: "LaminateSizingConfig",
) -> list[str]:
    """Return the list of constraint names assembled by size_beam_shell_laminate.

    Test-only hook: runs the sizer with maxiter=1 via a spy on scipy.minimize
    that aborts immediately, then reads the module-level ``_last_specs`` list
    that is populated by the sizer just before the minimize call.
    """
    import wing_design.beams.laminate_sizing as _ls

    from wing_design.materials.unidir import T700_EPOXY
    from wing_design import small_scenario
    from wing_design.beams.fea_model import project_panels_to_skin
    from wing_design.aero import build_airplane, sweep_envelope
    from wing_design.geometry import small_wingsail

    # Build a minimal load set from the small scenario so the sizer has valid
    # FEA inputs (beam_vm depends on FEA forces so we need at least one load).
    P = small_scenario()
    ap = build_airplane(small_wingsail)
    env = sweep_envelope(ap, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_skin(model, ar.panels,
                                    safety_factor=ar.case.safety_factor)
             for ar in env
             if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    orig = _ls.minimize

    def spy(fun, x0, jac=None, method=None, bounds=None, constraints=None,
            options=None, **kw):
        class _R:
            x = x0
            success = False
            nit = 0
        return _R()

    _ls.minimize = spy
    try:
        size_beam_shell_laminate(model, loads, config, ply=T700_EPOXY,
                                 rho=P.rho_kgm3, maxiter=1)
    finally:
        _ls.minimize = orig

    return [s.name for s in _ls._last_specs]


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
    x0: np.ndarray | None = None,
) -> MultiStartResult:
    """Run the sizer from ``n_starts`` initial guesses; return the best feasible result.

    Start 0 is the sizer's default guess, or ``x0`` when given (seed with a known
    feasible design and, via the incumbent guard, the result can only improve on
    it); starts 1.. are uniform-random within the design bounds (per-group
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
            return None if x0 is None else np.asarray(x0, dtype=float)
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
