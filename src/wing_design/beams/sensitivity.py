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
