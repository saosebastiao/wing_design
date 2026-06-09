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
