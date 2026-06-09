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
from wing_design.structural.shell import tri_element_stiffness_laminate
from wing_design.materials.unidir import reduced_stiffness_Q, transformed_Qbar


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


def adjoint_lambda(factored, dg_du):
    """Solve K λ = ∂g/∂u on free DOFs; λ = 0 on fixed DOFs. Returns (ndof,)."""
    lam = np.zeros(factored.ndof)
    lam[factored.free] = factored.lu.solve(dg_du[factored.free])
    return lam


def lambdaT_dK_x(factored, ds, lam):
    """Vector (nx,) of λᵀ (∂K/∂x_i) u for every design variable i."""
    nx = ds.G + ds.B + 2 * ds.L
    out = np.zeros(nx)
    u = factored.u
    nodes = ds.model.nodes
    be = factored.beam_elements

    # radius groups
    for e in range(be.shape[0]):
        g = int(ds.group_of_element[e])
        i, j = int(be[e, 0]), int(be[e, 1])
        dk = dkloc_dr(
            ds.model.E_beam, ds.model.G_beam,
            float(ds.radii_full[e]), float(ds.beam_lengths[e]),
        )
        T = factored.transforms[e]
        dkg = T.T @ dk @ T
        dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
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


def _active_beam_force(factored, e):
    """Recover (floc 12-vector, M=klocals@T, dofs) for beam element e (match beam_shell.py)."""
    be = factored.beam_elements
    i, j = int(be[e, 0]), int(be[e, 1])
    dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
    M = factored.klocals[e] @ factored.transforms[e]   # 12x12, ∂floc/∂u_e
    ue = factored.u[dofs]
    floc = M @ ue
    return floc, M, dofs


def grad_beam_vm(factored, ds, sigma_allow):
    """beam von-Mises feasibility constraint gradient.

    con = 1 − max_e(vM_e)/σ_allow. Returns (con_value, grad (nx,)).
    """
    be = factored.beam_elements
    n = be.shape[0]
    # find active element (max vM) using the documented force recovery
    vms = np.empty(n)
    cache = []
    for e in range(n):
        floc, M, dofs = _active_beam_force(factored, e)
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
        cache.append((floc, M, dofs, r, A, Iz, J, axial, torsion, b0, b1, Mres,
                      sigma_n, tau, vm))

    e_star = int(np.argmax(vms))
    (floc, M, dofs, r, A, Iz, J, axial, torsion, b0, b1, Mres,
     sigma_n, tau, vm) = cache[e_star]

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
    du_part = -lambdaT_dK_x(factored, ds, lam)

    # explicit ∂vM/∂r for the group of e*. Two pieces:
    #  (1) section A,Iz,J depend on r (the stress formula);
    #  (2) the recovered internal force floc = klocals(r)·T·u_e also depends on r,
    #      because klocals scales with the section. Both feed vM at fixed u.
    dsigma_n_dr = -2.0 * abs(axial) / (np.pi * r**3) - 12.0 * Mres / (np.pi * r**4)
    dtau_dr = -6.0 * abs(torsion) / (np.pi * r**4)
    dvm_dr = (sigma_n * dsigma_n_dr + 3.0 * tau * dtau_dr) / vm
    # piece (2): dfloc/dr = (∂klocals/∂r)·T·u_e ; chain through ∂vM/∂floc
    dk_dr = dkloc_dr(ds.model.E_beam, ds.model.G_beam, r,
                     float(ds.beam_lengths[e_star]))
    dfloc_dr = dk_dr @ factored.transforms[e_star] @ factored.u[dofs]
    dvm_dr += dvm_dfloc @ dfloc_dr
    du_part[int(ds.group_of_element[e_star])] += dvm_dr

    grad = -du_part / sigma_allow
    return con_value, grad


def grad_beam_buckling(factored, ds, *, euler_K, safety_factor):
    """beam Euler-buckling feasibility constraint gradient.

    con = 1 − max_e(util_e), util = comp·SF/Pcr, comp=max(0,−axial),
    Pcr = π²·E·Iy/(K·L)², Iy=πr⁴/4. Returns (con_value, grad (nx,)).
    """
    be = factored.beam_elements
    n = be.shape[0]
    E = ds.model.E_beam
    utils = np.empty(n)
    cache = []
    for e in range(n):
        floc, M, dofs = _active_beam_force(factored, e)
        r = float(ds.radii_full[e])
        L = float(ds.beam_lengths[e])
        axial = floc[6]
        comp = max(0.0, -axial)
        Iy = np.pi * r**4 / 4.0
        Pcr = np.pi**2 * E * Iy / (euler_K * L) ** 2
        util = comp * safety_factor / max(Pcr, 1e-30)
        utils[e] = util
        cache.append((floc, M, dofs, r, axial, comp, Pcr, util))

    e_star = int(np.argmax(utils))
    floc, M, dofs, r, axial, comp, Pcr, util = cache[e_star]

    con_value = 1.0 - util

    if comp == 0.0:
        return con_value, np.zeros(ds.G + ds.B + 2 * ds.L)

    # ∂util/∂axial = SF/Pcr · ∂comp/∂axial = SF/Pcr · (−1)  (axial<0 here)
    dutil_daxial = -safety_factor / Pcr
    # ∂axial/∂u_e* = M[6,:]
    dg_du = np.zeros(factored.ndof)
    dg_du[dofs] = dutil_daxial * M[6, :]
    lam = adjoint_lambda(factored, dg_du)
    du_part = -lambdaT_dK_x(factored, ds, lam)

    # explicit ∂util/∂r for the group of e*. Two pieces:
    #  (1) Pcr ∝ r⁴ ⇒ util ∝ r^−4 ⇒ ∂util/∂r = −4·util/r;
    #  (2) the recovered axial = floc[6] = (klocals(r)·T·u_e)[6] also depends on r.
    L = float(ds.beam_lengths[e_star])
    dutil_dr = -4.0 * util / r
    dk_dr = dkloc_dr(E, ds.model.G_beam, r, L)
    daxial_dr = (dk_dr @ factored.transforms[e_star] @ factored.u[dofs])[6]
    dutil_dr += dutil_daxial * daxial_dr
    du_part[int(ds.group_of_element[e_star])] += dutil_dr

    grad = -du_part
    return con_value, grad


def grad_tip_defl(factored, ds):
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
    grad = -lambdaT_dK_x(factored, ds, lam)
    return g, grad


def grad_tip_twist(factored, ds):
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
    grad = -lambdaT_dK_x(factored, ds, lam)
    return g, grad
