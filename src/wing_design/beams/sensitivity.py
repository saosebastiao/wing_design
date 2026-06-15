"""Element-stiffness derivative primitives for the analytic adjoint Jacobian.

Each primitive returns a directional/parametric derivative of an element
stiffness routine, exploiting the linearity of those routines in their
section / laminate inputs. All are validated against central finite
differences in `tests/beams/test_sensitivity.py`.

Linearity facts used here:
- `local_beam_stiffness(E, G, sec, L)` is linear in `(A, Iy, Iz, J)`, so
  d(kloc)/dr is the routine evaluated on the section built from dA/dr,
  dIy/dr, dIz/dr, dJ/dr (circular: A=πr², Iy=Iz=πr⁴/4, J=πr⁴/2).
- `tri_element_stiffness_laminate(p1,p2,p3, A, D, ...)` is linear in the
  3×3 matrices A, D, so its directional derivative is the routine evaluated
  on the directional derivatives (dA, dD).
- Laminate `A = t·Qeff`, `D = (t³/12)·Qeff`, with
  `Qeff = f0·Qbar(0) + ½f45·(Qbar(45)+Qbar(−45)) + f90·Qbar(90)`,
  `f90 = 1 − f0 − f45`. A ply-angle datum offset `o` shifts every angle.
"""

from dataclasses import dataclass

import numpy as np

from wing_design.structural.frame import BeamSection, local_beam_stiffness
from wing_design.structural.shell import (
    tri_element_stiffness_laminate, _triangle_local_frame,
)
from wing_design.materials.unidir import reduced_stiffness_Q, transformed_Qbar
from wing_design.materials.failure import (
    reduced_stiffness_Q as _ply_Q,
    _strain_to_ply_axes,
    tsai_wu_strength_ratio,
    tsai_wu_strength_ratio_grad,
)


def central_diff(f, x0, h):
    """Central difference of scalar-or-array-valued f at x0 (float) with step h."""
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


def dkloc_dr(E_beam, G_beam, r, L):
    """∂(local beam stiffness)/∂r for a circular section (12x12).

    Circular: A=πr², Iy=Iz=πr⁴/4, J=πr⁴/2, so
    dA/dr=2πr, dIy/dr=dIz/dr=πr³, dJ/dr=2πr³. local_beam_stiffness is linear
    in (A, Iy, Iz, J), so feed the r-derivatives in as a "section".
    """
    dsec = BeamSection(
        A=2 * np.pi * r,
        Iy=np.pi * r**3,
        Iz=np.pi * r**3,
        J=2 * np.pi * r**3,
        r=r,
    )
    return local_beam_stiffness(E_beam, G_beam, dsec, L)


def dkloc_annular(E_beam, G_beam, r, t, L, *, wrt):
    """∂(local beam stiffness)/∂r_outer or ∂t_wall for an annular section (12x12).

    Annulus: A = π(r²−(r−t)²), I = π/4(r⁴−(r−t)⁴), J = 2I (P.1 core tube).
    wrt="r": dA = 2πt, dI = π(r³−(r−t)³);  wrt="t": dA = 2π(r−t), dI = π(r−t)³.
    `local_beam_stiffness` is linear in (A, Iy, Iz, J).
    """
    ri = r - t
    if wrt == "r":
        dA = 2.0 * np.pi * t
        dI = np.pi * (r**3 - ri**3)
    elif wrt == "t":
        dA = 2.0 * np.pi * ri
        dI = np.pi * ri**3
    else:
        raise ValueError(wrt)
    dsec = BeamSection(A=dA, Iy=dI, Iz=dI, J=2.0 * dI, r=r)
    return local_beam_stiffness(E_beam, G_beam, dsec, L)


def dke_dAD(p1, p2, p3, dA, dD, drilling_factor=1.0e-4):
    """Element shell stiffness directional derivative given dA, dD (18x18).

    Linear in (A, D) -> the derivative is just the routine on (dA, dD).
    """
    return tri_element_stiffness_laminate(
        p1, p2, p3, A=dA, D=dD, drilling_factor=drilling_factor
    )


def dAD_dt(Qeff, t):
    """(∂A/∂t, ∂D/∂t) = (Qeff, (t²/4)·Qeff)."""
    return Qeff, (t**2 / 4.0) * Qeff


def dAD_dt_sandwich(Qeff, t, c):
    """(∂A/∂t, ∂D/∂t) for the P.3 sandwich: A = t·Qeff (faces only),
    D = Qeff·(t³/48 + (t/4)(c+t/2)²)  →  ∂D/∂t = Qeff·(t²/16 + (c+t/2)²/4
    + t(c+t/2)/4). Degenerates to dAD_dt at c = 0."""
    h = c + t / 2.0
    return Qeff, (t**2 / 16.0 + h**2 / 4.0 + t * h / 4.0) * Qeff


def dD_dc_sandwich(Qeff, t, c):
    """∂D/∂c = Qeff·t·(c + t/2)/2 (membrane A is c-independent)."""
    return (t * (c + t / 2.0) / 2.0) * Qeff


def sandwich_geom_factor(t, c):
    """D geometric factor t³/48 + (t/4)(c+t/2)²  (= t³/12 at c = 0)."""
    return t**3 / 48.0 + (t / 4.0) * (c + t / 2.0) ** 2


def dQeff_df(ply, *, which, offset_deg=0.0):
    """∂Qeff/∂f0 (which='f0') or ∂Qeff/∂f45 (which='f45'), with datum offset.

    f90 = 1 − f0 − f45 is eliminated, so ∂Qeff/∂f0 = Qbar(0) − Qbar(90) and
    ∂Qeff/∂f45 = ½(Qbar(45)+Qbar(−45)) − Qbar(90), all angles shifted by `offset_deg`.
    """
    Q = reduced_stiffness_Q(ply)
    qb = lambda a: transformed_Qbar(Q, a + offset_deg)
    if which == "f0":
        return qb(0.0) - qb(90.0)
    if which == "f45":
        return 0.5 * (qb(45.0) + qb(-45.0)) - qb(90.0)
    raise ValueError(which)


def ortho_plate_Qstar(Qd):
    """Long-SS-plate orthotropic combination (V#2, datum frame).

    Qstar = sqrt(Q11·Q22) + Q12 + 2·Q66 over the DATUM-frame effective Q
    (1 = span, 2 = chord), so σcr = ½·kc·π²·Qstar·geom/(b²·t) is the classical
    N_cr = (2π²/b²)(√(D11·D22) + D12 + 2·D66) long-plate result and reduces
    exactly to kc·π²·D/(b²·t) with kc = 4 for an isotropic laminate.
    """
    return np.sqrt(Qd[0, 0] * Qd[1, 1]) + Qd[0, 1] + 2.0 * Qd[2, 2]


def dQstar_df(Qd, dQ):
    """∂(ortho_plate_Qstar)/∂f given Qd and dQ = dQeff_df(..., offset_deg=0)."""
    s = np.sqrt(Qd[0, 0] * Qd[1, 1])
    return ((dQ[0, 0] * Qd[1, 1] + Qd[0, 0] * dQ[1, 1]) / (2.0 * s)
            + dQ[0, 1] + 2.0 * dQ[2, 2])


def dAD_df(dQeff, t):
    """(∂A/∂f, ∂D/∂f) = (t·dQeff, (t³/12)·dQeff) for a given ∂Qeff/∂f."""
    return t * dQeff, (t**3 / 12.0) * dQeff


# --- Adjoint engine + simple constraint gradients --------------------------


@dataclass
class DesignSens:
    """Everything needed to assemble λᵀ(∂K/∂x)u for the general design vector.

    Design vector layout: ``x = [r_group(G), t_band(B), f0_grp(L), f45_grp(L)]``
    with ``nx = G + B + 2L``.
    """

    model: object                     # BeamShellModel
    G: int
    B: int
    L: int
    group_of_element: np.ndarray      # (n,) -> [0, G)
    band_of_tri: np.ndarray           # (M,) -> [0, B)
    layup_group_of_band: np.ndarray   # (B,) -> [0, L)
    radii_full: np.ndarray            # (n,)
    beam_lengths: np.ndarray          # (n,)
    t_tri: np.ndarray                 # (M,)
    Qeff_tri: np.ndarray              # (M, 3, 3)
    offset_tri: np.ndarray            # (M,) degrees
    ply: object
    drilling_factor: float = 1.0e-4
    # per-triangle layup fractions (needed to identify the present orientations for
    # the per-ply Tsai-Wu skin constraint). Optional: only the Tsai-Wu gradient uses
    # them. If None, all four orientations are treated as present.
    f0_tri: np.ndarray = None         # (M,)
    f45_tri: np.ndarray = None        # (M,)
    f90_tri: np.ndarray = None        # (M,)
    # Annular elements (P.1 core tube and/or P#1b hollow form beams): rows of
    # beam_elements carrying annular sections. Despite the historical "tube_"
    # names these describe ANY annular element; r/t DV columns are given
    # explicitly via tube_r_cols/tube_t_cols (absolute columns — a hollow form
    # beam's r col is its existing radius-group column). When the col arrays are
    # None the legacy contiguous layout [.. | r_tube(S) | t_wall(S)] applies.
    S: int = 0
    tube_elements: np.ndarray = None  # (S,) element indices (into beam_elements)
    tube_r: np.ndarray = None         # (S,) outer radii
    tube_t: np.ndarray = None         # (S,) wall thicknesses
    tube_r_cols: np.ndarray = None    # (S,) absolute DV col of each r (None = legacy)
    tube_t_cols: np.ndarray = None    # (S,) absolute DV col of each t (None = legacy)
    nx_extra: int = None              # DV cols beyond G+B+2L (None = legacy 2*S)
    # P.3 sandwich skin (optional): per-triangle core thickness + the absolute DV
    # column of each triangle's t_core band variable. None = monolithic.
    core: object = None               # CoreMaterial
    c_tri: np.ndarray = None          # (M,)
    core_col_of_tri: np.ndarray = None  # (M,) absolute DV columns
    # V.3c beam-on-elastic-foundation (in-wing form beams): per-element k and
    # its design chains. None = element-length Euler.
    found_k: np.ndarray = None        # (n_form,) [N/m^2]
    found_dk: list = None             # per form element: [(col, dk_dcol), ...]
    found_mask: np.ndarray = None     # (n_all,) bool: use foundation for element
    # P.2 lateral-bracing transverse ring members: rows of beam_elements that are
    # brace (ring) elements. Their sections use a FIXED nominal radius (not a DV),
    # so they contribute NO ∂K/∂x and get no Euler radius-gradient column. None =
    # no bracing.
    brace_mask: np.ndarray = None     # (n_all,) bool
    # V#2 datum-frame orthotropic strip panel buckling (opt-in, KS-only):
    # per-band Qstar = sqrt(Q11·Q22)+Q12+2·Q66 in the DATUM frame + f-chains.
    # None = historical triangle-local-frame D11 path (bit-exact).
    panel_Qstar: np.ndarray = None       # (B,)
    panel_dQstar_df0: np.ndarray = None  # (B,)
    panel_dQstar_df45: np.ndarray = None # (B,)

    @property
    def nx(self) -> int:
        extra = self.nx_extra if self.nx_extra is not None else 2 * self.S
        return self.G + self.B + 2 * self.L + extra

    def tube_dv_r(self, s: int) -> int:
        if self.tube_r_cols is not None:
            return int(self.tube_r_cols[s])
        return self.G + self.B + 2 * self.L + s

    def tube_dv_t(self, s: int) -> int:
        if self.tube_t_cols is not None:
            return int(self.tube_t_cols[s])
        return self.G + self.B + 2 * self.L + self.S + s

    def tube_seg_of_element(self, e: int) -> int:
        """Annular-set index of element e, or -1 if e is a solid form beam."""
        if self.tube_elements is None:
            return -1
        hit = np.nonzero(self.tube_elements == e)[0]
        return int(hit[0]) if hit.size else -1


def adjoint_lambda(factored, dg_du):
    """Solve K λ = ∂g/∂u on free DOFs; λ = 0 on fixed DOFs. Returns (ndof,)."""
    lam = np.zeros(factored.ndof)
    lam[factored.free] = factored.lu.solve(dg_du[factored.free])
    return lam


def lambdaT_dK_x(factored, ds, lam):
    """Vector (nx,) of λᵀ (∂K/∂x_i) u for every design variable i."""
    if ds.core is not None:
        raise NotImplementedError("sandwich skin requires the cached path "
                                  "(prepare_sensitivity / lambdaT_dK_x_cached)")
    if getattr(ds, "brace_mask", None) is not None:
        raise NotImplementedError(
            "lambdaT_dK_x (uncached) does not support braced models; "
            "use the cached path (lambdaT_dK_x_cached)")
    nx = ds.nx
    out = np.zeros(nx)
    u = factored.u
    nodes = ds.model.nodes
    be = factored.beam_elements

    # radius groups (form beams) + tube segments (annular, two DVs each)
    for e in range(be.shape[0]):
        i, j = int(be[e, 0]), int(be[e, 1])
        dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        T = factored.transforms[e]
        s = ds.tube_seg_of_element(e)
        if s >= 0:
            for wrt, col in (("r", ds.tube_dv_r(s)), ("t", ds.tube_dv_t(s))):
                dk = dkloc_annular(ds.model.E_beam, ds.model.G_beam,
                                   float(ds.tube_r[s]), float(ds.tube_t[s]),
                                   float(ds.beam_lengths[e]), wrt=wrt)
                out[col] += lam[dofs] @ (T.T @ dk @ T) @ u[dofs]
            continue
        g = int(ds.group_of_element[e])
        dk = dkloc_dr(
            ds.model.E_beam, ds.model.G_beam,
            float(ds.radii_full[e]), float(ds.beam_lengths[e]),
        )
        dkg = T.T @ dk @ T
        out[g] += lam[dofs] @ dkg @ u[dofs]

    # thickness bands + layup groups
    tris = ds.model.shell_tris
    for t in range(tris.shape[0]):
        n0, n1, n2 = int(tris[t, 0]), int(tris[t, 1]), int(tris[t, 2])
        p1, p2, p3 = nodes[n0], nodes[n1], nodes[n2]
        dofs = np.r_[6 * n0:6 * n0 + 6, 6 * n1:6 * n1 + 6, 6 * n2:6 * n2 + 6]
        le = lam[dofs]
        ue = u[dofs]
        b = int(ds.band_of_tri[t])
        Qe = ds.Qeff_tri[t]
        tt = float(ds.t_tri[t])
        off = float(ds.offset_tri[t])

        # thickness
        dA, dD = dAD_dt(Qe, tt)
        dke = dke_dAD(p1, p2, p3, dA, dD, ds.drilling_factor)
        out[ds.G + b] += le @ dke @ ue

        # layup fractions (f0, f45) for this triangle's layup group
        lg = int(ds.layup_group_of_band[b])
        for which, base in (("f0", ds.G + ds.B), ("f45", ds.G + ds.B + ds.L)):
            dQ = dQeff_df(ds.ply, which=which, offset_deg=off)
            dA2, dD2 = dAD_df(dQ, tt)
            dke2 = dke_dAD(p1, p2, p3, dA2, dD2, ds.drilling_factor)
            out[base + lg] += le @ dke2 @ ue

    return out


@dataclass(frozen=True)
class SensCache:
    """Precomputed design-dependent ∂K/∂x element matrices (reused across all
    adjoints / load cases). Independent of `lam` and `u`.
    """

    nx: int
    # beam (radius) contributions
    beam_dofs: list          # list of (12,) int dof-index arrays
    beam_dkg: list           # list of (12,12) global ∂kg/∂r
    beam_dv: np.ndarray      # (n_beam,) radius-group DV index per element
    # triangle contributions
    tri_dofs: list           # list of (18,) int dof-index arrays
    tri_dke_t: list          # (18,18) ∂ke/∂t
    tri_dke_f0: list         # (18,18) ∂ke/∂f0
    tri_dke_f45: list        # (18,18) ∂ke/∂f45
    tri_dv_t: np.ndarray     # (n_tri,) thickness DV index = G + band
    tri_dv_f0: np.ndarray    # (n_tri,) f0 DV index = G+B + layup_group
    tri_dv_f45: np.ndarray   # (n_tri,) f45 DV index = G+B+L + layup_group
    # P.1 tube entries (empty lists when no tube)
    tube_dofs: list = None
    tube_dkg_r: list = None
    tube_dkg_t: list = None
    tube_dv_r: np.ndarray = None
    tube_dv_t: np.ndarray = None
    # P.3 sandwich core entries (None = monolithic)
    tri_dke_c: list = None
    tri_dv_c: np.ndarray = None


def prepare_sensitivity(ds, factored) -> "SensCache":
    """Precompute design-dependent ∂K element matrices once (reused across all adjoints/LCs)."""
    G, B, L = ds.G, ds.B, ds.L
    nx = ds.nx
    be = factored.beam_elements
    beam_dofs, beam_dkg, beam_dv_list = [], [], []
    tube_dofs, tube_dkg_r, tube_dkg_t, tube_dv_r, tube_dv_t = [], [], [], [], []
    for e in range(be.shape[0]):
        i, j = int(be[e, 0]), int(be[e, 1])
        dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        T = factored.transforms[e]
        s = ds.tube_seg_of_element(e)
        if s >= 0:
            dk_r = dkloc_annular(ds.model.E_beam, ds.model.G_beam,
                                 float(ds.tube_r[s]), float(ds.tube_t[s]),
                                 float(ds.beam_lengths[e]), wrt="r")
            dk_t = dkloc_annular(ds.model.E_beam, ds.model.G_beam,
                                 float(ds.tube_r[s]), float(ds.tube_t[s]),
                                 float(ds.beam_lengths[e]), wrt="t")
            tube_dofs.append(dofs)
            tube_dkg_r.append(T.T @ dk_r @ T)
            tube_dkg_t.append(T.T @ dk_t @ T)
            tube_dv_r.append(ds.tube_dv_r(s))
            tube_dv_t.append(ds.tube_dv_t(s))
            continue
        if ds.brace_mask is not None and bool(ds.brace_mask[e]):
            # Brace (ring) elements use a FIXED nominal radius -> ∂K/∂x = 0;
            # their sizing benefit enters only via the foundation k_ring credit.
            continue
        dk = dkloc_dr(
            ds.model.E_beam, ds.model.G_beam,
            float(ds.radii_full[e]), float(ds.beam_lengths[e]),
        )
        beam_dkg.append(T.T @ dk @ T)
        beam_dofs.append(dofs)
        beam_dv_list.append(int(ds.group_of_element[e]))
    beam_dv = np.asarray(beam_dv_list, dtype=int)

    tris = ds.model.shell_tris
    sandwich = ds.core is not None
    tri_dofs, tri_dke_t, tri_dke_f0, tri_dke_f45 = [], [], [], []
    tri_dke_c = [] if sandwich else None
    tri_dv_t = np.empty(tris.shape[0], dtype=int)
    tri_dv_f0 = np.empty(tris.shape[0], dtype=int)
    tri_dv_f45 = np.empty(tris.shape[0], dtype=int)
    for t in range(tris.shape[0]):
        n0, n1, n2 = int(tris[t, 0]), int(tris[t, 1]), int(tris[t, 2])
        p1, p2, p3 = ds.model.nodes[n0], ds.model.nodes[n1], ds.model.nodes[n2]
        tri_dofs.append(np.r_[6 * n0:6 * n0 + 6, 6 * n1:6 * n1 + 6, 6 * n2:6 * n2 + 6])
        tt = float(ds.t_tri[t]); off = float(ds.offset_tri[t]); Qe = ds.Qeff_tri[t]
        if sandwich:
            cc = float(ds.c_tri[t])
            dA, dD = dAD_dt_sandwich(Qe, tt, cc)
            tri_dke_t.append(dke_dAD(p1, p2, p3, dA, dD, ds.drilling_factor))
            dDc = dD_dc_sandwich(Qe, tt, cc)
            tri_dke_c.append(dke_dAD(p1, p2, p3, 0.0 * Qe, dDc, ds.drilling_factor))
            geom = sandwich_geom_factor(tt, cc)
            dA0, dD0 = dAD_df(dQeff_df(ds.ply, which="f0", offset_deg=off), tt)
            tri_dke_f0.append(dke_dAD(p1, p2, p3, dA0, (12.0 * geom / tt**3) * dD0,
                                      ds.drilling_factor))
            dA4, dD4 = dAD_df(dQeff_df(ds.ply, which="f45", offset_deg=off), tt)
            tri_dke_f45.append(dke_dAD(p1, p2, p3, dA4, (12.0 * geom / tt**3) * dD4,
                                       ds.drilling_factor))
        else:
            dA, dD = dAD_dt(Qe, tt)
            tri_dke_t.append(dke_dAD(p1, p2, p3, dA, dD, ds.drilling_factor))
            dA0, dD0 = dAD_df(dQeff_df(ds.ply, which="f0", offset_deg=off), tt)
            tri_dke_f0.append(dke_dAD(p1, p2, p3, dA0, dD0, ds.drilling_factor))
            dA4, dD4 = dAD_df(dQeff_df(ds.ply, which="f45", offset_deg=off), tt)
            tri_dke_f45.append(dke_dAD(p1, p2, p3, dA4, dD4, ds.drilling_factor))
        b = int(ds.band_of_tri[t]); lg = int(ds.layup_group_of_band[b])
        tri_dv_t[t] = G + b; tri_dv_f0[t] = G + B + lg; tri_dv_f45[t] = G + B + L + lg
    return SensCache(nx, beam_dofs, beam_dkg, beam_dv, tri_dofs, tri_dke_t,
                     tri_dke_f0, tri_dke_f45, tri_dv_t, tri_dv_f0, tri_dv_f45,
                     tube_dofs=tube_dofs, tube_dkg_r=tube_dkg_r, tube_dkg_t=tube_dkg_t,
                     tube_dv_r=np.asarray(tube_dv_r, dtype=int),
                     tube_dv_t=np.asarray(tube_dv_t, dtype=int),
                     tri_dke_c=tri_dke_c,
                     tri_dv_c=(ds.core_col_of_tri if sandwich else None))


def lambdaT_dK_x_cached(cache: "SensCache", lam, u) -> np.ndarray:
    """Cheap contraction λᵀ(∂K/∂x)u using cached ∂K matrices (identical to lambdaT_dK_x)."""
    out = np.zeros(cache.nx)
    for e in range(len(cache.beam_dofs)):
        d = cache.beam_dofs[e]
        out[cache.beam_dv[e]] += lam[d] @ cache.beam_dkg[e] @ u[d]
    if cache.tube_dofs:
        for s in range(len(cache.tube_dofs)):
            d = cache.tube_dofs[s]
            le, ue = lam[d], u[d]
            out[cache.tube_dv_r[s]] += le @ cache.tube_dkg_r[s] @ ue
            out[cache.tube_dv_t[s]] += le @ cache.tube_dkg_t[s] @ ue
    for t in range(len(cache.tri_dofs)):
        d = cache.tri_dofs[t]; le = lam[d]; ue = u[d]
        out[cache.tri_dv_t[t]] += le @ cache.tri_dke_t[t] @ ue
        out[cache.tri_dv_f0[t]] += le @ cache.tri_dke_f0[t] @ ue
        out[cache.tri_dv_f45[t]] += le @ cache.tri_dke_f45[t] @ ue
        if cache.tri_dke_c is not None:
            out[cache.tri_dv_c[t]] += le @ cache.tri_dke_c[t] @ ue
    return out


def _annulus_props(r, t):
    """(A, I, J) and their (d/dr, d/dt) for an annular section."""
    ri = r - t
    A = np.pi * (r**2 - ri**2)
    I = 0.25 * np.pi * (r**4 - ri**4)
    J = 2.0 * I
    dA = (2.0 * np.pi * t, 2.0 * np.pi * ri)            # (d/dr, d/dt)
    dI = (np.pi * (r**3 - ri**3), np.pi * ri**3)
    dJ = (2.0 * dI[0], 2.0 * dI[1])
    return A, I, J, dA, dI, dJ


def _beam_vm_explicit_tube(floc, dvm_dfloc, fac, ds, e_star, s, sigma_n, tau, vm):
    """Explicit (∂vM/∂r_tube, ∂vM/∂t_wall) for an annular active element:
    section-property chains + the recovered-force dependence via ∂klocal."""
    r = float(ds.tube_r[s]); t = float(ds.tube_t[s])
    A, I, J, dA, dI, dJ = _annulus_props(r, t)
    axial = floc[6]; torsion = floc[9]
    b0 = np.hypot(floc[4], floc[5]); b1 = np.hypot(floc[10], floc[11])
    Mres = max(b0, b1)
    i, j = int(fac.beam_elements[e_star, 0]), int(fac.beam_elements[e_star, 1])
    dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
    out = []
    for k, wrt in enumerate(("r", "t")):
        dr_outer = 1.0 if wrt == "r" else 0.0   # fiber distance = r_outer
        dsig = (-abs(axial) * dA[k] / A**2
                + Mres * (dr_outer * I - r * dI[k]) / I**2)
        dtau = abs(torsion) * (dr_outer * J - r * dJ[k]) / J**2
        dvm = (sigma_n * dsig + 3.0 * tau * dtau) / vm
        dk = dkloc_annular(ds.model.E_beam, ds.model.G_beam, r, t,
                           float(ds.beam_lengths[e_star]), wrt=wrt)
        dfloc = dk @ fac.transforms[e_star] @ fac.u[dofs]
        dvm += dvm_dfloc @ dfloc
        out.append(dvm)
    return out[0], out[1]


def _active_beam_force(factored, e):
    """Recover (floc 12-vector, M=klocals@T, dofs) for beam element e (match beam_shell.py)."""
    be = factored.beam_elements
    i, j = int(be[e, 0]), int(be[e, 1])
    dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
    M = factored.klocals[e] @ factored.transforms[e]   # 12x12, ∂floc/∂u_e
    ue = factored.u[dofs]
    floc = M @ ue
    return floc, M, dofs


def grad_beam_vm(factored, ds, sigma_allow, cache: "SensCache | None" = None,
                 dF: np.ndarray | None = None):
    """beam von-Mises feasibility constraint gradient.

    con = 1 − max_e(vM_e)/σ_allow. Returns (con_value, grad (nx,)).
    """
    if getattr(ds, "brace_mask", None) is not None:
        raise NotImplementedError(
            "grad_beam_vm does not support braced models; use the foundation+KS path")
    be = factored.beam_elements
    n = be.shape[0]
    # find active element (max vM) using the documented force recovery
    vms = np.empty(n)
    rows = []
    for e in range(n):
        floc, M, dofs = _active_beam_force(factored, e)
        s_e = ds.tube_seg_of_element(e)
        if s_e >= 0:
            r = float(ds.tube_r[s_e])
            A, Iz, J, _dA, _dI, _dJ = _annulus_props(r, float(ds.tube_t[s_e]))
        else:
            r = float(ds.radii_full[e])
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
        vms[e] = vm
        rows.append((floc, M, dofs, r, A, Iz, J, axial, torsion, b0, b1, Mres,
                     sigma_n, tau, vm))

    e_star = int(np.argmax(vms))
    (floc, M, dofs, r, A, Iz, J, axial, torsion, b0, b1, Mres,
     sigma_n, tau, vm) = rows[e_star]

    con_value = 1.0 - vm / sigma_allow

    # ∂vM/∂floc (12-vector)
    dvm_dfloc = np.zeros(12)
    sgn_ax = np.sign(axial) if axial != 0.0 else 0.0
    sgn_tor = np.sign(torsion) if torsion != 0.0 else 0.0
    # axial (index 6)
    dvm_dfloc[6] = (sigma_n / vm) * sgn_ax / A
    # torsion (index 9)
    dvm_dfloc[9] = (3.0 * tau / vm) * sgn_tor * r / J
    # bending: active end only
    if Mres > 0.0:
        if b0 >= b1:
            fa, fb = floc[4], floc[5]
            ia, ib = 4, 5
        else:
            fa, fb = floc[10], floc[11]
            ia, ib = 10, 11
        coef = (sigma_n / vm) * (r / Iz)
        dvm_dfloc[ia] = coef * (fa / Mres)
        dvm_dfloc[ib] = coef * (fb / Mres)

    # implicit term: dg_du = scatter of (∂vM/∂floc) @ M into ndof
    dg_du = np.zeros(factored.ndof)
    dg_du[dofs] = dvm_dfloc @ M
    lam = adjoint_lambda(factored, dg_du)
    _cache = cache if cache is not None else prepare_sensitivity(ds, factored)
    dkx = lambdaT_dK_x_cached(_cache, lam, factored.u)
    du_part = -dkx
    # design-dependent loads (V.4): dg/dx implicit = lam^T (dF - dK u)
    if dF is not None:
        du_part += lam @ dF

    # explicit section terms for e*: solid rod (radius-group DV) or annular tube
    # (two tube DVs). Two pieces each: (1) section props in the stress formula;
    # (2) the recovered force floc = klocal(x)·T·u_e depends on the section.
    s_tube = ds.tube_seg_of_element(e_star)
    if s_tube >= 0:
        dvm_dr_t, dvm_dt_t = _beam_vm_explicit_tube(
            floc, dvm_dfloc, factored, ds, e_star, s_tube, sigma_n, tau, vm)
        du_part[ds.tube_dv_r(s_tube)] += dvm_dr_t
        du_part[ds.tube_dv_t(s_tube)] += dvm_dt_t
    else:
        dsigma_n_dr = -2.0 * abs(axial) / (np.pi * r**3) - 12.0 * Mres / (np.pi * r**4)
        dtau_dr = -6.0 * abs(torsion) / (np.pi * r**4)
        dvm_dr = (sigma_n * dsigma_n_dr + 3.0 * tau * dtau_dr) / vm
        dk_dr = dkloc_dr(ds.model.E_beam, ds.model.G_beam, r,
                         float(ds.beam_lengths[e_star]))
        dfloc_dr = dk_dr @ factored.transforms[e_star] @ factored.u[dofs]
        dvm_dr += dvm_dfloc @ dfloc_dr
        du_part[int(ds.group_of_element[e_star])] += dvm_dr

    grad = -du_part / sigma_allow
    return con_value, grad


def grad_beam_buckling(factored, ds, *, euler_K, safety_factor,
                       cache: "SensCache | None" = None, dF: np.ndarray | None = None):
    """beam Euler-buckling feasibility constraint gradient.

    con = 1 − max_e(util_e), util = comp·SF/Pcr, comp=max(0,−axial),
    Pcr = π²·E·Iy/(K·L)², Iy=πr⁴/4. Returns (con_value, grad (nx,)).
    """
    if getattr(ds, "brace_mask", None) is not None:
        raise NotImplementedError(
            "grad_beam_buckling does not support braced models; use the foundation+KS path")
    be = factored.beam_elements
    n = be.shape[0]
    E = ds.model.E_beam
    utils = np.empty(n)
    rows = []
    for e in range(n):
        floc, M, dofs = _active_beam_force(factored, e)
        s_e = ds.tube_seg_of_element(e)
        if s_e >= 0:
            r = float(ds.tube_r[s_e])
            _A, Iy, _J, _dA, _dI, _dJ = _annulus_props(r, float(ds.tube_t[s_e]))
        else:
            r = float(ds.radii_full[e])
            Iy = np.pi * r**4 / 4.0
        L = float(ds.beam_lengths[e])
        axial = floc[6]
        comp = max(0.0, -axial)
        Pcr = np.pi**2 * E * Iy / (euler_K * L) ** 2
        util = comp * safety_factor / max(Pcr, 1e-30)
        utils[e] = util
        rows.append((floc, M, dofs, r, axial, comp, Pcr, util))

    e_star = int(np.argmax(utils))
    floc, M, dofs, r, axial, comp, Pcr, util = rows[e_star]

    con_value = 1.0 - util

    if comp == 0.0:
        return con_value, np.zeros(ds.nx)

    # ∂util/∂axial = SF/Pcr · ∂comp/∂axial = SF/Pcr · (−1)  (axial<0 here)
    dutil_daxial = -safety_factor / Pcr
    # ∂axial/∂u_e* = M[6,:]
    dg_du = np.zeros(factored.ndof)
    dg_du[dofs] = dutil_daxial * M[6, :]
    lam = adjoint_lambda(factored, dg_du)
    _cache = cache if cache is not None else prepare_sensitivity(ds, factored)
    dkx = lambdaT_dK_x_cached(_cache, lam, factored.u)
    du_part = -dkx
    # design-dependent loads (V.4): dg/dx implicit = lam^T (dF - dK u)
    if dF is not None:
        du_part += lam @ dF

    # explicit section terms for e*: util ∝ 1/I plus the recovered-axial
    # dependence via ∂klocal. Solid rod: one radius-group DV; tube: two DVs.
    L = float(ds.beam_lengths[e_star])
    s_tube = ds.tube_seg_of_element(e_star)
    if s_tube >= 0:
        rt = float(ds.tube_r[s_tube]); tt = float(ds.tube_t[s_tube])
        _A, Iy, _J, _dA, dI, _dJ = _annulus_props(rt, tt)
        for k, wrt in enumerate(("r", "t")):
            dutil = -util * dI[k] / Iy
            dk = dkloc_annular(E, ds.model.G_beam, rt, tt, L, wrt=wrt)
            dutil += dutil_daxial * (dk @ factored.transforms[e_star] @ factored.u[dofs])[6]
            col = ds.tube_dv_r(s_tube) if wrt == "r" else ds.tube_dv_t(s_tube)
            du_part[col] += dutil
    else:
        dutil_dr = -4.0 * util / r
        dk_dr = dkloc_dr(E, ds.model.G_beam, r, L)
        daxial_dr = (dk_dr @ factored.transforms[e_star] @ factored.u[dofs])[6]
        dutil_dr += dutil_daxial * daxial_dr
        du_part[int(ds.group_of_element[e_star])] += dutil_dr

    grad = -du_part
    return con_value, grad


def grad_tube_wall_one(factored, ds, s, *, safety_factor, E,
                       knockdown=0.65, cache: "SensCache | None" = None,
                       dF: np.ndarray | None = None):
    """Row s of the tube wall-crimping constraint gradient (P.1).

    con_s = 1 − util_s, util = SF·max(0,−N)/A / σ_cr, σ_cr = γ·0.605·E·t/r
    ⇒ util = SF·(−N)·r / (A·γ·0.605·E·t) for N < 0. Returns (con_s, grad (nx,)).
    """
    e = int(ds.tube_elements[s])
    floc, M, dofs = _active_beam_force(factored, e)
    axial = floc[6]
    r = float(ds.tube_r[s]); t = float(ds.tube_t[s])
    A, _I, _J, dA, _dI, _dJ = _annulus_props(r, t)
    c = knockdown * 0.605 * E
    sigma_cr = c * t / r
    comp = max(0.0, -axial)
    util = (comp / A) * safety_factor / sigma_cr
    con_value = 1.0 - util
    if comp == 0.0:
        return con_value, np.zeros(ds.nx)

    # implicit: ∂util/∂N = −SF/(A·σcr) for N<0; scatter through M row 6 + adjoint
    dutil_daxial = -safety_factor / (A * sigma_cr)
    dg_du = np.zeros(factored.ndof)
    dg_du[dofs] = dutil_daxial * M[6, :]
    lam = adjoint_lambda(factored, dg_du)
    _cache = cache if cache is not None else prepare_sensitivity(ds, factored)
    du_part = -lambdaT_dK_x_cached(_cache, lam, factored.u)
    if dF is not None:
        du_part += lam @ dF

    # explicit: util = K0·r/(A·t), K0 = SF·comp/c; plus N(klocal(r,t)) chains
    K0 = safety_factor * comp / c
    dutil_dr = K0 * (A * t - r * t * dA[0]) / (A * t) ** 2
    dutil_dt = K0 * (-r) * (dA[1] * t + A) / (A * t) ** 2
    for wrt, dutil in (("r", dutil_dr), ("t", dutil_dt)):
        dk = dkloc_annular(ds.model.E_beam, ds.model.G_beam, r, t,
                           float(ds.beam_lengths[e]), wrt=wrt)
        dutil += dutil_daxial * (dk @ factored.transforms[e] @ factored.u[dofs])[6]
        col = ds.tube_dv_r(s) if wrt == "r" else ds.tube_dv_t(s)
        du_part[col] += dutil

    return con_value, -du_part


# --- Skin membrane stress sensitivities -----------------------------------


def _membrane_strain_map(p1, p2, p3):
    """Return (Gt 3x18, area) mapping element 18 global DOFs -> element-local
    membrane strain eps=[exx,eyy,gxy]. Replicates `recover_membrane_strain`.

    Only the three translational DOFs per node participate: eps = Bm @ u_m,
    with u_m[2k:2k+2] = (R^T u_trans_k)[:2]. So Gt scatters Bm @ S, where S maps
    the 18-DOF element vector to the 6-vector u_m via per-node R^T (first 2 rows).
    """
    R, local, area = _triangle_local_frame(p1, p2, p3)
    x1, y1 = local[0]; x2, y2 = local[1]; x3, y3 = local[2]
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    two_A = 2.0 * area
    Bm = np.zeros((3, 6))
    for k in range(3):
        Bm[0, 2 * k + 0] = b[k] / two_A
        Bm[1, 2 * k + 1] = c[k] / two_A
        Bm[2, 2 * k + 0] = c[k] / two_A
        Bm[2, 2 * k + 1] = b[k] / two_A
    # S (6x18): per node, u_m rows = (R^T)[:2] applied to translational DOFs.
    S = np.zeros((6, 18))
    Rt = R.T
    for nidx in range(3):
        S[2 * nidx:2 * nidx + 2, 6 * nidx:6 * nidx + 3] = Rt[:2, :]
    Gt = Bm @ S
    return Gt, area


def _active_tri_stress(factored, ds, t):
    """Return (s 3-vec, eps 3-vec, Gt 3x18, area, dofs 18, C_t 3x3) for triangle t."""
    tris = ds.model.shell_tris
    nodes = ds.model.nodes
    n0, n1, n2 = int(tris[t, 0]), int(tris[t, 1]), int(tris[t, 2])
    Gt, area = _membrane_strain_map(nodes[n0], nodes[n1], nodes[n2])
    dofs = np.r_[6 * n0:6 * n0 + 6, 6 * n1:6 * n1 + 6, 6 * n2:6 * n2 + 6]
    u_tri = factored.u[dofs]
    eps = Gt @ u_tri
    C_t = ds.Qeff_tri[t]
    s = C_t @ eps
    return s, eps, Gt, area, dofs, C_t


def grad_skin_vm(factored, ds, sigma_allow, cache: "SensCache | None" = None,
                 dF: np.ndarray | None = None, qw2_tri: np.ndarray | None = None):
    """skin von-Mises feasibility constraint gradient.

    con = 1 − max_t(vM_t + σ_b,t)/σ_allow, with σ_b = qw2/t² the V.5 strip-bending
    fiber stress when ``qw2_tri`` (= 0.75·q·w²) is given (else 0). Membrane stress
    is thickness-independent; σ_b is design-explicit only. Returns (con_value, grad).
    """
    tris = ds.model.shell_tris
    M = tris.shape[0]
    sb = (qw2_tri / ds.t_tri**2) if qw2_tri is not None else np.zeros(M)
    vms = np.empty(M)
    rows = []
    for t in range(M):
        s, eps, Gt, area, dofs, C_t = _active_tri_stress(factored, ds, t)
        sxx, syy, sxy = s
        vm = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
        vms[t] = vm + sb[t]
        rows.append((s, eps, Gt, area, dofs, C_t, vm))

    t_star = int(np.argmax(vms))
    s, eps, Gt, area, dofs, C_t, vm = rows[t_star]
    sxx, syy, sxy = s
    con_value = 1.0 - (vm + sb[t_star]) / sigma_allow

    # ∂vM/∂s
    dg_ds = np.array([
        (2.0 * sxx - syy) / (2.0 * vm),
        (2.0 * syy - sxx) / (2.0 * vm),
        3.0 * sxy / vm,
    ])

    # implicit: dg_du = scatter of dg_ds @ (C_t @ Gt) (18-vec) into ndof
    dg_du = np.zeros(factored.ndof)
    dg_du[dofs] = dg_ds @ (C_t @ Gt)
    lam = adjoint_lambda(factored, dg_du)
    _cache = cache if cache is not None else prepare_sensitivity(ds, factored)
    dkx = lambdaT_dK_x_cached(_cache, lam, factored.u)
    du_part = -dkx
    # design-dependent loads (V.4): dg/dx implicit = lam^T (dF - dK u)
    if dF is not None:
        du_part += lam @ dF

    # explicit terms. Membrane stress is thickness-independent, but the V.5
    # strip-bending adder is: ∂(qw2/t²)/∂t = −2·qw2/t³ on the active band.
    b = int(ds.band_of_tri[t_star])
    if qw2_tri is not None:
        du_part[ds.G + b] += -2.0 * float(qw2_tri[t_star]) / float(ds.t_tri[t_star]) ** 3
    lg = int(ds.layup_group_of_band[b])
    off = float(ds.offset_tri[t_star])
    for which, base in (("f0", ds.G + ds.B), ("f45", ds.G + ds.B + ds.L)):
        dQ = dQeff_df(ds.ply, which=which, offset_deg=off)
        ds_df = dQ @ eps                # ∂s/∂f|explicit
        dvm_df = float(dg_ds @ ds_df)
        du_part[base + lg] += dvm_df

    grad = -du_part / sigma_allow
    return con_value, grad


def grad_panel_buckling(factored, ds, *, panel_kc, safety_factor, areas,
                        cache: "SensCache | None" = None, dF: np.ndarray | None = None):
    """skin (panel) buckling feasibility constraint gradient.

    con = 1 − max_t(util_t), util = comp·SF/σcr, comp=max(0,−s_min),
    s_min=mean−R, σcr = kc·π²·Qeff00·t²/(12 b²), b=√area. Returns (con_value, grad).
    """
    tris = ds.model.shell_tris
    M = tris.shape[0]
    SF = safety_factor
    sandwich = ds.core is not None
    utils = np.empty(M)
    rows = []
    for t in range(M):
        s, eps, Gt, area, dofs, C_t = _active_tri_stress(factored, ds, t)
        sxx, syy, sxy = s
        mean = 0.5 * (sxx + syy)
        R = np.sqrt(0.25 * (sxx - syy) ** 2 + sxy**2)
        s_min = mean - R
        comp = max(0.0, -s_min)
        tt = float(ds.t_tri[t])
        Qeff00 = float(C_t[0, 0])
        b2 = float(areas[t])           # b² = area
        if sandwich:
            cc = float(ds.c_tri[t])
            sigma_cr = (panel_kc * np.pi**2 * Qeff00
                        * sandwich_geom_factor(tt, cc) / (b2 * tt))
        else:
            sigma_cr = panel_kc * np.pi**2 * Qeff00 * tt**2 / (12.0 * b2)
        util = comp * SF / max(sigma_cr, 1e-30)
        utils[t] = util
        rows.append((s, eps, Gt, area, dofs, C_t, comp, R, sigma_cr, util, tt, Qeff00))

    t_star = int(np.argmax(utils))
    (s, eps, Gt, area, dofs, C_t, comp, R, sigma_cr, util, tt,
     Qeff00) = rows[t_star]
    sxx, syy, sxy = s
    con_value = 1.0 - util

    if comp == 0.0:
        return con_value, np.zeros(ds.nx)

    # ∂comp/∂s with comp = R − mean (since s_min<0 here)
    # ∂R/∂sxx = (sxx−syy)/(4R), ∂R/∂syy = −(sxx−syy)/(4R), ∂R/∂sxy = sxy/R
    dR = np.array([
        (sxx - syy) / (4.0 * R),
        -(sxx - syy) / (4.0 * R),
        sxy / R,
    ]) if R > 0.0 else np.zeros(3)
    dmean = np.array([0.5, 0.5, 0.0])
    dcomp_ds = dR - dmean
    dutil_ds = (SF / sigma_cr) * dcomp_ds

    # implicit term
    dg_du = np.zeros(factored.ndof)
    dg_du[dofs] = dutil_ds @ (C_t @ Gt)
    lam = adjoint_lambda(factored, dg_du)
    _cache = cache if cache is not None else prepare_sensitivity(ds, factored)
    dkx = lambdaT_dK_x_cached(_cache, lam, factored.u)
    du_part = -dkx
    # design-dependent loads (V.4): dg/dx implicit = lam^T (dF - dK u)
    if dF is not None:
        du_part += lam @ dF

    # explicit thickness/core terms via σcr
    b = int(ds.band_of_tri[t_star])
    if sandwich:
        cc = float(ds.c_tri[t_star])
        h = cc + tt / 2.0
        # σcr ∝ geom/t, d/dt(geom/t) = t/24 + h/4; d/dc(geom/t) = h/2
        gov_t = tt / 24.0 + h / 4.0
        gov_c = h / 2.0
        ratio = (sandwich_geom_factor(tt, cc) / tt)
        du_part[ds.G + b] += -util * gov_t / ratio
        du_part[int(ds.core_col_of_tri[t_star])] += -util * gov_c / ratio
    else:
        # util ∝ 1/σcr ∝ 1/t² ⇒ ∂util/∂t = −2·util/t (kept exact, ulp discipline)
        du_part[ds.G + b] += -2.0 * util / tt

    # explicit f: util = comp(f)·SF/σcr(f); σcr ∝ Qeff00(f); comp via ∂s/∂f.
    lg = int(ds.layup_group_of_band[b])
    off = float(ds.offset_tri[t_star])
    for which, base in (("f0", ds.G + ds.B), ("f45", ds.G + ds.B + ds.L)):
        dQ = dQeff_df(ds.ply, which=which, offset_deg=off)
        ds_df = dQ @ eps
        dcomp_df = float(dcomp_ds @ ds_df)
        dsigcr_df = float(dQ[0, 0]) * (sigma_cr / Qeff00)
        dutil_df = (SF / sigma_cr) * dcomp_df - util * dsigcr_df / sigma_cr
        du_part[base + lg] += dutil_df

    grad = -du_part
    return con_value, grad


def grad_core_check(factored, ds, *, which, safety_factor,
                    cache: "SensCache | None" = None,
                    dF: np.ndarray | None = None):
    """P.3 core failure gradient: which='wrinkle' (Hoff) or 'crimp' (shear).

    util = SF·comp/σ_allow(t,c,f) · s(c), s = c/(c+1mm) (smooth activation).
    con = 1 − max util. Returns (con_value, grad (nx,)).
    """
    from wing_design.materials.unidir import CORE_RAMP_M
    tris = ds.model.shell_tris
    M = tris.shape[0]
    SF = safety_factor
    core = ds.core
    utils = np.empty(M)
    rows = []
    for t in range(M):
        s, eps, Gt, area, dofs, C_t = _active_tri_stress(factored, ds, t)
        sxx, syy, sxy = s
        mean = 0.5 * (sxx + syy)
        R = np.sqrt(0.25 * (sxx - syy) ** 2 + sxy**2)
        comp = max(0.0, -(mean - R))
        tt = float(ds.t_tri[t]); cc = float(ds.c_tri[t])
        Qeff00 = float(C_t[0, 0])
        if which == "wrinkle":
            sig = 0.5 * (Qeff00 * core.E_c * core.G_c) ** (1.0 / 3.0)
        else:
            sig = core.G_c * (cc + tt) / tt
        ramp = cc / (cc + CORE_RAMP_M)
        util = comp * SF / max(sig, 1e-30) * ramp
        utils[t] = util
        rows.append((s, eps, Gt, dofs, C_t, comp, R, sig, ramp, util, tt, cc, Qeff00))

    t_star = int(np.argmax(utils))
    (s, eps, Gt, dofs, C_t, comp, R, sig, ramp, util, tt, cc, Qeff00) = rows[t_star]
    sxx, syy, sxy = s
    con_value = 1.0 - util
    if comp == 0.0 or ramp == 0.0:
        return con_value, np.zeros(ds.nx)

    dR = np.array([(sxx - syy) / (4.0 * R), -(sxx - syy) / (4.0 * R),
                   sxy / R]) if R > 0.0 else np.zeros(3)
    dcomp_ds = dR - np.array([0.5, 0.5, 0.0])
    dutil_ds = (SF * ramp / sig) * dcomp_ds

    dg_du = np.zeros(factored.ndof)
    dg_du[dofs] = dutil_ds @ (C_t @ Gt)
    lam = adjoint_lambda(factored, dg_du)
    _cache = cache if cache is not None else prepare_sensitivity(ds, factored)
    du_part = -lambdaT_dK_x_cached(_cache, lam, factored.u)
    if dF is not None:
        du_part += lam @ dF

    b = int(ds.band_of_tri[t_star])
    lg = int(ds.layup_group_of_band[b])
    off = float(ds.offset_tri[t_star])
    # explicit f: comp via ∂s/∂f, plus σ_wr ∝ Qeff00^(1/3) for wrinkle
    for whichf, base in (("f0", ds.G + ds.B), ("f45", ds.G + ds.B + ds.L)):
        dQ = dQeff_df(ds.ply, which=whichf, offset_deg=off)
        dcomp_df = float(dcomp_ds @ (dQ @ eps))
        dutil_df = (SF * ramp / sig) * dcomp_df
        if which == "wrinkle":
            dsig_df = sig * float(dQ[0, 0]) / (3.0 * Qeff00)
            dutil_df += -util * dsig_df / sig
        du_part[base + lg] += dutil_df
    # explicit t (crimp only: σ = G(c+t)/t → ∂σ/∂t = −G·c/t²)
    if which == "crimp":
        dsig_dt = -core.G_c * cc / tt**2
        du_part[ds.G + b] += -util * dsig_dt / sig
    # explicit c: ramp everywhere; crimp also via σ
    dramp_dc = CORE_RAMP_M / (cc + CORE_RAMP_M) ** 2
    dutil_dc = (comp * SF / sig) * dramp_dc
    if which == "crimp":
        dsig_dc = core.G_c / tt
        dutil_dc += -util * dsig_dc / sig
    du_part[int(ds.core_col_of_tri[t_star])] += dutil_dc

    return con_value, -du_part


def _Teps(theta_deg):
    """Engineering-strain rotation matrix (3x3): eps_ply = Teps @ eps_local."""
    th = np.radians(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [c * c, s * s, s * c],
        [s * s, c * c, -s * c],
        [-2.0 * s * c, 2.0 * s * c, c * c - s * s],
    ])


def grad_skin_tsai_wu(factored, ds, *, safety_factor,
                      cache: "SensCache | None" = None, dF: np.ndarray | None = None):
    """Per-ply Tsai-Wu skin feasibility constraint gradient.

    con = min_R/SF − 1, where min_R is the minimum Tsai-Wu strength ratio over all
    triangles and all *present* ply orientations (gated by the per-triangle layup
    fractions, shifted by the datum offset). Returns (con_value, grad (nx,)).

    The explicit term is zero: at fixed strain, R depends on u and the (discrete)
    active orientation only, not continuously on the thickness/fraction design vars.
    """
    tris = ds.model.shell_tris
    nodes = ds.model.nodes
    M = tris.shape[0]
    SF = safety_factor
    ply = ds.ply
    Q = _ply_Q(ply)
    eps_tol = 1.0e-9

    # present-orientation fractions per triangle (default: all present)
    if ds.f0_tri is None:
        f0a = np.ones(M); f45a = np.ones(M); f90a = np.ones(M)
    else:
        f0a = np.asarray(ds.f0_tri, dtype=float)
        f45a = np.asarray(ds.f45_tri, dtype=float)
        f90a = np.asarray(ds.f90_tri, dtype=float)

    best_R = np.inf
    best = None  # (t, theta_deg, Gt, dofs, eps)
    for t in range(M):
        n0, n1, n2 = int(tris[t, 0]), int(tris[t, 1]), int(tris[t, 2])
        Gt, _area = _membrane_strain_map(nodes[n0], nodes[n1], nodes[n2])
        dofs = np.r_[6 * n0:6 * n0 + 6, 6 * n1:6 * n1 + 6, 6 * n2:6 * n2 + 6]
        eps = Gt @ factored.u[dofs]
        off = float(ds.offset_tri[t])
        orientations = (
            (0.0, f0a[t]), (45.0, f45a[t]), (-45.0, f45a[t]), (90.0, f90a[t]),
        )
        for base, frac in orientations:
            if frac <= eps_tol:
                continue
            theta = base + off
            sigma = Q @ _strain_to_ply_axes(eps, theta)
            R = tsai_wu_strength_ratio(sigma, ply)
            if R < best_R:
                best_R = R
                best = (t, theta, Gt, dofs, eps)

    t_star, theta_star, Gt, dofs, eps = best
    con_value = best_R / SF - 1.0

    # σ123* and ∂R/∂σ123
    sigma_star = Q @ _strain_to_ply_axes(eps, theta_star)
    dR_dsigma = tsai_wu_strength_ratio_grad(sigma_star, ply)  # (3,)

    # ∂σ123/∂u_tri = Q @ Teps(θ*) @ Gt  -> dR/du_tri (18,)
    dR_du = dR_dsigma @ (Q @ _Teps(theta_star) @ Gt)

    dg_du = np.zeros(factored.ndof)
    dg_du[dofs] = dR_du
    lam = adjoint_lambda(factored, dg_du)
    _cache = cache if cache is not None else prepare_sensitivity(ds, factored)
    dkx = lambdaT_dK_x_cached(_cache, lam, factored.u)
    dR_dx = -dkx  # explicit term is zero
    # design-dependent loads (V.4): dg/dx implicit = lam^T (dF - dK u)
    if dF is not None:
        dR_dx += lam @ dF

    grad = dR_dx / SF
    return con_value, grad


def grad_tip_defl(factored, ds, cache: "SensCache | None" = None,
                  dF: np.ndarray | None = None):
    """g = max tip ||u_trans||; returns (g, grad (nx,)). Explicit term is 0."""
    u = factored.u.reshape(-1, 6)
    tip = ds.model.tip_nodes
    mags = np.linalg.norm(u[tip, :3], axis=1)
    a = int(np.argmax(mags))
    node = int(tip[a])
    g = float(mags[a])
    uhat = u[node, :3] / max(g, 1e-30)
    dg_du = np.zeros(factored.ndof)
    dg_du[6 * node:6 * node + 3] = uhat
    lam = adjoint_lambda(factored, dg_du)
    _cache = cache if cache is not None else prepare_sensitivity(ds, factored)
    dkx = lambdaT_dK_x_cached(_cache, lam, factored.u)
    grad = -dkx
    # design-dependent loads (V.4): dg/dx implicit = lam^T (dF - dK u)
    if dF is not None:
        grad += lam @ dF
    return g, grad


def grad_tip_twist(factored, ds, cache: "SensCache | None" = None,
                   dF: np.ndarray | None = None):
    """g = max tip |u_twist (dof 5)|; returns (g, grad (nx,)). Explicit term is 0."""
    u = factored.u.reshape(-1, 6)
    tip = ds.model.tip_nodes
    tw = np.abs(u[tip, 5])
    a = int(np.argmax(tw))
    node = int(tip[a])
    g = float(tw[a])
    s = np.sign(u[node, 5]) or 1.0
    dg_du = np.zeros(factored.ndof)
    dg_du[6 * node + 5] = s
    lam = adjoint_lambda(factored, dg_du)
    _cache = cache if cache is not None else prepare_sensitivity(ds, factored)
    dkx = lambdaT_dK_x_cached(_cache, lam, factored.u)
    grad = -dkx
    # design-dependent loads (V.4): dg/dx implicit = lam^T (dF - dK u)
    if dF is not None:
        grad += lam @ dF
    return g, grad


# --- V.0.3/P#7: KS (smooth-max) constraint aggregation ----------------------
#
# Every max-based constraint becomes one C-infinity scalar over (items x load
# combos): KS = m + ln(sum exp(rho(g_i - m)))/rho >= max(g_i) (conservative).
# This removes the argmax-switching Jacobian kinks that stall both SLSQP and
# IPOPT on large design migrations (P.3 evidence, 2026-06-11). Engine cost:
# one adjoint per load combo per constraint (the per-item dg/du rows combine
# into a single weighted dg_du before the back-substitution).


def ks_aggregate(vals, rho):
    """(KS value, softmax weights w_i with sum w = 1) — numerically stable."""
    v = np.asarray(vals, dtype=float)
    m = float(v.max())
    e = np.exp(rho * (v - m))
    W = float(e.sum())
    return m + np.log(W) / rho, e / W


def grad_ks_engine(facs, ds, cache, dFs, rho, provider):
    """con = 1 − KS(g) over all (items × combos); returns (con, grad (nx,)).

    ``provider(fac, ds)`` yields per-combo lists: (vals (K,), dofs_list,
    dgdu_local_list, explicit_list) where explicit_list[k] = [(col, dval), ...].
    """
    provs = (list(provider) if isinstance(provider, (list, tuple))
             else [provider] * len(facs))
    per_combo = [provs[ci](fac, ds) for ci, fac in enumerate(facs)]
    all_vals = np.concatenate([pc[0] for pc in per_combo])
    ks, w = ks_aggregate(all_vals, rho)
    con = 1.0 - ks

    grad = np.zeros(ds.nx)
    pos = 0
    for ci, (vals, dofs_list, dgdu_list, expl_list) in enumerate(per_combo):
        K = len(vals)
        wc = w[pos:pos + K]
        pos += K
        if wc.sum() < 1e-300:
            continue
        fac = facs[ci]
        dg_du = np.zeros(fac.ndof)
        for k in range(K):
            if wc[k] == 0.0:
                continue
            dg_du[dofs_list[k]] += wc[k] * dgdu_list[k]
            for col, dval in expl_list[k]:
                grad[col] += wc[k] * dval
        lam = adjoint_lambda(fac, dg_du)
        du = -lambdaT_dK_x_cached(cache, lam, fac.u)
        if dFs is not None and dFs[ci] is not None:
            du += lam @ dFs[ci]
        grad += du
    return con, -grad


def ks_values_only(facs, ds, rho, provider):
    """KS constraint value without gradients (for the scipy fun closures)."""
    provs = (list(provider) if isinstance(provider, (list, tuple))
             else [provider] * len(facs))
    all_vals = np.concatenate([provs[ci](fac, ds)[0] for ci, fac in enumerate(facs)])
    ks, _w = ks_aggregate(all_vals, rho)
    return 1.0 - ks


# --- per-family providers (reuse the per-item formulas of the hard grads) ---


def provider_panel(panel_kc, safety_factor, areas):
    def p(fac, ds):
        tris = ds.model.shell_tris
        M = tris.shape[0]
        sandwich = ds.core is not None
        vals = np.empty(M)
        dofs_l, dgdu_l, expl_l = [], [], []
        for t in range(M):
            s, eps, Gt, area, dofs, C_t = _active_tri_stress(fac, ds, t)
            sxx, syy, sxy = s
            mean = 0.5 * (sxx + syy)
            R = np.sqrt(0.25 * (sxx - syy) ** 2 + sxy**2)
            comp = max(0.0, -(mean - R))
            tt = float(ds.t_tri[t]); Qeff00 = float(C_t[0, 0]); b2 = float(areas[t])
            datum_ortho = ds.panel_Qstar is not None
            if datum_ortho:
                # V#2: σcr from the datum-frame long-plate combination, ½·Qstar
                # standing in for the local Qeff00 (kc semantics preserved).
                bq = int(ds.band_of_tri[t])
                Qpb = 0.5 * float(ds.panel_Qstar[bq])
                if sandwich:
                    cc = float(ds.c_tri[t])
                    sigma_cr = panel_kc * np.pi**2 * Qpb * sandwich_geom_factor(tt, cc) / (b2 * tt)
                else:
                    sigma_cr = panel_kc * np.pi**2 * Qpb * tt**2 / (12.0 * b2)
            elif sandwich:
                cc = float(ds.c_tri[t])
                sigma_cr = panel_kc * np.pi**2 * Qeff00 * sandwich_geom_factor(tt, cc) / (b2 * tt)
            else:
                sigma_cr = panel_kc * np.pi**2 * Qeff00 * tt**2 / (12.0 * b2)
            util = comp * safety_factor / max(sigma_cr, 1e-30)
            vals[t] = util
            if comp == 0.0:
                dofs_l.append(dofs); dgdu_l.append(np.zeros(18)); expl_l.append([])
                continue
            dR = (np.array([(sxx - syy) / (4.0 * R), -(sxx - syy) / (4.0 * R), sxy / R])
                  if R > 0.0 else np.zeros(3))
            dcomp_ds = dR - np.array([0.5, 0.5, 0.0])
            dutil_ds = (safety_factor / sigma_cr) * dcomp_ds
            dofs_l.append(dofs)
            dgdu_l.append(dutil_ds @ (C_t @ Gt))
            expl = []
            b = int(ds.band_of_tri[t]); lg = int(ds.layup_group_of_band[b])
            off = float(ds.offset_tri[t])
            if sandwich:
                cc = float(ds.c_tri[t]); h = cc + tt / 2.0
                ratio = sandwich_geom_factor(tt, cc) / tt
                expl.append((ds.G + b, -util * (tt / 24.0 + h / 4.0) / ratio))
                expl.append((int(ds.core_col_of_tri[t]), -util * (h / 2.0) / ratio))
            else:
                expl.append((ds.G + b, -2.0 * util / tt))
            for which, base in (("f0", ds.G + ds.B), ("f45", ds.G + ds.B + ds.L)):
                dQ = dQeff_df(ds.ply, which=which, offset_deg=off)
                dcomp_df = float(dcomp_ds @ (dQ @ eps))
                if datum_ortho:
                    # σcr ∝ ½·Qstar ⇒ dσcr/df = dQstar_df · σcr/Qstar
                    dst = (ds.panel_dQstar_df0 if which == "f0"
                           else ds.panel_dQstar_df45)
                    dsigcr_df = float(dst[b]) * (sigma_cr / float(ds.panel_Qstar[b]))
                else:
                    dsigcr_df = float(dQ[0, 0]) * (sigma_cr / Qeff00)
                expl.append((base + lg, (safety_factor / sigma_cr) * dcomp_df
                             - util * dsigcr_df / sigma_cr))
            expl_l.append(expl)
        return vals, dofs_l, dgdu_l, expl_l
    return p


def provider_beam_buck(euler_K, safety_factor):
    def p(fac, ds):
        be = fac.beam_elements
        n = be.shape[0]
        E = ds.model.E_beam
        vals = np.empty(n)
        dofs_l, dgdu_l, expl_l = [], [], []
        for e in range(n):
            floc, Mx, dofs = _active_beam_force(fac, e)
            s_e = ds.tube_seg_of_element(e)
            foundation = (ds.found_mask is not None and e < len(ds.found_mask)
                          and bool(ds.found_mask[e]))
            if s_e >= 0:
                r = float(ds.tube_r[s_e])
                _A, Iy, _J, _dA, dI, _dJ = _annulus_props(r, float(ds.tube_t[s_e]))
            else:
                r = float(ds.radii_full[e])
                Iy = np.pi * r**4 / 4.0
            L = float(ds.beam_lengths[e])
            axial = floc[6]
            comp = max(0.0, -axial)
            if foundation:
                # V.3c: Pcr = 2*sqrt(k*EI), k from the bonded skin (design-dep.)
                kf = float(ds.found_k[e])
                EI = E * Iy
                Pcr = 2.0 * np.sqrt(max(kf * EI, 1e-30))
            else:
                Pcr = np.pi**2 * E * Iy / (euler_K * L) ** 2
            util = comp * safety_factor / max(Pcr, 1e-30)
            vals[e] = util
            if comp == 0.0:
                dofs_l.append(dofs); dgdu_l.append(np.zeros(12)); expl_l.append([])
                continue
            dutil_daxial = -safety_factor / Pcr
            dofs_l.append(dofs)
            dgdu_l.append(dutil_daxial * Mx[6, :])
            expl = []
            if s_e >= 0:
                rt = float(ds.tube_r[s_e]); tt = float(ds.tube_t[s_e])
                for k, wrt in enumerate(("r", "t")):
                    if foundation:
                        dutil = -util * (E * dI[k]) / (2.0 * E * Iy)
                    else:
                        dutil = -util * dI[k] / Iy
                    dk = dkloc_annular(E, ds.model.G_beam, rt, tt, L, wrt=wrt)
                    dutil += dutil_daxial * (dk @ fac.transforms[e] @ fac.u[dofs])[6]
                    col = ds.tube_dv_r(s_e) if wrt == "r" else ds.tube_dv_t(s_e)
                    expl.append((col, dutil))
            elif ds.brace_mask is not None and bool(ds.brace_mask[e]):
                # Brace (ring) element: FIXED nominal radius (not a DV) -> no
                # radius-explicit gradient column. Its util enters only via the
                # adjoint (dg/du); the ring's sizing benefit is the k_ring
                # foundation credit on the form elements it restrains.
                pass
            else:
                if foundation:
                    # util ∝ 1/sqrt(EI): ∂util/∂r = −util·(dEI/dr)/(2EI)
                    dEI_dr = E * np.pi * r**3
                    dutil_dr = -util * dEI_dr / (2.0 * E * Iy)
                else:
                    dutil_dr = -4.0 * util / r
                dk_dr = dkloc_dr(E, ds.model.G_beam, r, L)
                dutil_dr += dutil_daxial * (dk_dr @ fac.transforms[e] @ fac.u[dofs])[6]
                expl.append((int(ds.group_of_element[e]), dutil_dr))
            if foundation and ds.found_dk is not None:
                # util ∝ 1/sqrt(k): ∂util/∂col = −util·(dk/dcol)/(2k)
                # (applies to solid AND hollow/annular in-wing form elements)
                kf = float(ds.found_k[e])
                for col, dk_dcol in ds.found_dk[e]:
                    expl.append((int(col), -util * dk_dcol / (2.0 * kf)))
            expl_l.append(expl)
        return vals, dofs_l, dgdu_l, expl_l
    return p


def provider_defl(limit, tip_nodes):
    def p(fac, ds):
        u = fac.u.reshape(-1, 6)
        K = len(tip_nodes)
        vals = np.empty(K)
        dofs_l, dgdu_l, expl_l = [], [], []
        for k, node in enumerate(tip_nodes):
            node = int(node)
            g = float(np.linalg.norm(u[node, :3]))
            vals[k] = g / limit
            dofs = np.arange(6 * node, 6 * node + 3)
            uhat = u[node, :3] / max(g, 1e-30)
            dofs_l.append(dofs)
            dgdu_l.append(uhat / limit)
            expl_l.append([])
        return vals, dofs_l, dgdu_l, expl_l
    return p


def provider_twist(limit_deg, tip_nodes):
    def p(fac, ds):
        u = fac.u.reshape(-1, 6)
        K = len(tip_nodes)
        vals = np.empty(K)
        dofs_l, dgdu_l, expl_l = [], [], []
        for k, node in enumerate(tip_nodes):
            node = int(node)
            tw = float(u[node, 5])
            # twist limit compared in DEGREES like the hard path
            vals[k] = abs(np.degrees(tw)) / limit_deg
            sgn = np.sign(tw) or 1.0
            dofs_l.append(np.array([6 * node + 5]))
            dgdu_l.append(np.array([np.degrees(sgn) / limit_deg]))
            expl_l.append([])
        return vals, dofs_l, dgdu_l, expl_l
    return p


def provider_skin_vm(sigma_allow, qw2_of_combo=None):
    """qw2_of_combo: callable ci -> (M,) 0.75*q*w^2 array or None (V.5 adder)."""
    def make(ci):
        def p(fac, ds):
            tris = ds.model.shell_tris
            M = tris.shape[0]
            qw2 = qw2_of_combo(ci) if qw2_of_combo is not None else None
            vals = np.empty(M)
            dofs_l, dgdu_l, expl_l = [], [], []
            for t in range(M):
                s, eps, Gt, area, dofs, C_t = _active_tri_stress(fac, ds, t)
                sxx, syy, sxy = s
                vm = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
                tt = float(ds.t_tri[t])
                sb = float(qw2[t]) / tt**2 if qw2 is not None else 0.0
                vals[t] = (vm + sb) / sigma_allow
                if vm == 0.0:
                    dofs_l.append(dofs); dgdu_l.append(np.zeros(18)); expl_l.append([])
                    continue
                dg_ds = np.array([(2.0 * sxx - syy) / (2.0 * vm),
                                  (2.0 * syy - sxx) / (2.0 * vm),
                                  3.0 * sxy / vm])
                dofs_l.append(dofs)
                dgdu_l.append((dg_ds @ (C_t @ Gt)) / sigma_allow)
                expl = []
                b = int(ds.band_of_tri[t]); lg = int(ds.layup_group_of_band[b])
                off = float(ds.offset_tri[t])
                if qw2 is not None and sb > 0.0:
                    expl.append((ds.G + b, (-2.0 * float(qw2[t]) / tt**3) / sigma_allow))
                for which, base in (("f0", ds.G + ds.B), ("f45", ds.G + ds.B + ds.L)):
                    dQ = dQeff_df(ds.ply, which=which, offset_deg=off)
                    expl.append((base + lg, float(dg_ds @ (dQ @ eps)) / sigma_allow))
                expl_l.append(expl)
            return vals, dofs_l, dgdu_l, expl_l
        return p
    return make


def provider_core_check(which, safety_factor):
    from wing_design.materials.unidir import CORE_RAMP_M

    def p(fac, ds):
        tris = ds.model.shell_tris
        M = tris.shape[0]
        core = ds.core
        vals = np.empty(M)
        dofs_l, dgdu_l, expl_l = [], [], []
        for t in range(M):
            s, eps, Gt, area, dofs, C_t = _active_tri_stress(fac, ds, t)
            sxx, syy, sxy = s
            mean = 0.5 * (sxx + syy)
            R = np.sqrt(0.25 * (sxx - syy) ** 2 + sxy**2)
            comp = max(0.0, -(mean - R))
            tt = float(ds.t_tri[t]); cc = float(ds.c_tri[t])
            Qeff00 = float(C_t[0, 0])
            if which == "wrinkle":
                sig = 0.5 * (Qeff00 * core.E_c * core.G_c) ** (1.0 / 3.0)
            else:
                sig = core.G_c * (cc + tt) / tt
            ramp = cc / (cc + CORE_RAMP_M)
            util = comp * safety_factor / max(sig, 1e-30) * ramp
            vals[t] = util
            if comp == 0.0:
                dofs_l.append(dofs); dgdu_l.append(np.zeros(18)); expl_l.append([])
                continue
            dR = (np.array([(sxx - syy) / (4.0 * R), -(sxx - syy) / (4.0 * R), sxy / R])
                  if R > 0.0 else np.zeros(3))
            dcomp_ds = dR - np.array([0.5, 0.5, 0.0])
            dofs_l.append(dofs)
            dgdu_l.append(((safety_factor * ramp / sig) * dcomp_ds) @ (C_t @ Gt))
            expl = []
            b = int(ds.band_of_tri[t]); lg = int(ds.layup_group_of_band[b])
            off = float(ds.offset_tri[t])
            for whichf, base in (("f0", ds.G + ds.B), ("f45", ds.G + ds.B + ds.L)):
                dQ = dQeff_df(ds.ply, which=whichf, offset_deg=off)
                dutil_df = (safety_factor * ramp / sig) * float(dcomp_ds @ (dQ @ eps))
                if which == "wrinkle":
                    dutil_df += -util * (sig * float(dQ[0, 0]) / (3.0 * Qeff00)) / sig
                expl.append((base + lg, dutil_df))
            if which == "crimp":
                expl.append((ds.G + b, -util * (-core.G_c * cc / tt**2) / sig))
            dramp_dc = CORE_RAMP_M / (cc + CORE_RAMP_M) ** 2
            dutil_dc = (comp * safety_factor / sig) * dramp_dc
            if which == "crimp":
                dutil_dc += -util * (core.G_c / tt) / sig
            expl.append((int(ds.core_col_of_tri[t]), dutil_dc))
            expl_l.append(expl)
        return vals, dofs_l, dgdu_l, expl_l
    return p


def provider_tube_wall(safety_factor, E, knockdown=0.65):
    def p(fac, ds):
        Q = ds.S
        vals = np.empty(Q)
        dofs_l, dgdu_l, expl_l = [], [], []
        c0 = knockdown * 0.605 * E
        for s_i in range(Q):
            e = int(ds.tube_elements[s_i])
            floc, Mx, dofs = _active_beam_force(fac, e)
            axial = floc[6]
            r = float(ds.tube_r[s_i]); t = float(ds.tube_t[s_i])
            A, _I, _J, dA, _dI, _dJ = _annulus_props(r, t)
            sigma_cr = c0 * t / r
            comp = max(0.0, -axial)
            util = (comp / A) * safety_factor / sigma_cr
            vals[s_i] = util
            if comp == 0.0:
                dofs_l.append(dofs); dgdu_l.append(np.zeros(12)); expl_l.append([])
                continue
            dutil_daxial = -safety_factor / (A * sigma_cr)
            dofs_l.append(dofs)
            dgdu_l.append(dutil_daxial * Mx[6, :])
            K0 = safety_factor * comp / c0
            dutil_dr = K0 * (A * t - r * t * dA[0]) / (A * t) ** 2
            dutil_dt = K0 * (-r) * (dA[1] * t + A) / (A * t) ** 2
            expl = []
            for wrt, dutil in (("r", dutil_dr), ("t", dutil_dt)):
                dk = dkloc_annular(ds.model.E_beam, ds.model.G_beam, r, t,
                                   float(ds.beam_lengths[e]), wrt=wrt)
                dutil += dutil_daxial * (dk @ fac.transforms[e] @ fac.u[dofs])[6]
                col = ds.tube_dv_r(s_i) if wrt == "r" else ds.tube_dv_t(s_i)
                expl.append((col, dutil))
            expl_l.append(expl)
        return vals, dofs_l, dgdu_l, expl_l
    return p
