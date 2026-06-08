"""Hard tip-coupling experiment: a stiff 'gusset' tying all wing-tip beam nodes.

Models a rigid tip joint (clamp/gusset) as a clique of very stiff connector beams
among the tip nodes, appended to the model's longitudinal beams and solved by the
existing combined beam+shell solver (`solve_beam_shell`). Lets load transfer directly
beam-to-beam at the tip in addition to the skin's shear path. `gusset_radius` is the
stiffness knob — larger = closer to rigid.

The original longitudinal beams are the first `n_beam` elements; the clique elements
follow and are NOT design members, so the returned `FrameResult` internal forces are
sliced back to `[:n_beam]` for downstream stress checks.

Caveat (measured): the load-bearing skin already couples the tip nodes nearly rigidly
in the spanwise direction, so the *additional* effect of an explicit gusset is modest —
see `examples/31_tip_coupling.py`.
"""
from __future__ import annotations

import numpy as np

from ..structural.beam_shell import solve_beam_shell
from ..structural.frame import BeamSection, FrameResult
from .shell_model import BeamShellModel


def tip_clique_elements(tip_nodes: np.ndarray) -> np.ndarray:
    """(n*(n-1)/2, 2) all unordered node-index pairs among the tip nodes."""
    t = np.asarray(tip_nodes).astype(int)
    pairs = [(int(t[i]), int(t[j])) for i in range(t.shape[0]) for j in range(i + 1, t.shape[0])]
    return np.asarray(pairs, dtype=int)


def solve_beam_shell_tip_coupled(
    model: BeamShellModel,
    loads: np.ndarray,
    *,
    gusset_radius: float | None,
    beam_sections: list[BeamSection] | None = None,
) -> tuple[FrameResult, int]:
    """Solve the beam-shell model with a stiff tip gusset; return (result, n_beam).

    ``gusset_radius`` None ⇒ no coupling (identical to the plain solve). Otherwise a
    clique of ``BeamSection.circular(gusset_radius)`` beams ties the tip nodes,
    appended to the longitudinal beams and assembled by ``solve_beam_shell``. The
    returned ``FrameResult`` internal forces are sliced to the first ``n_beam``
    (design) beams; the clique elements are excluded.
    """
    n_beam = model.beam_elements.shape[0]
    if beam_sections is None:
        beam_sections = [model.section] * n_beam

    if gusset_radius is None:
        elements = model.beam_elements
        sections = list(beam_sections)
    else:
        clique = tip_clique_elements(model.tip_nodes)
        elements = np.vstack([model.beam_elements, clique])
        sections = list(beam_sections) + [BeamSection.circular(float(gusset_radius))] * clique.shape[0]

    res = solve_beam_shell(
        model.nodes, elements, sections, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam,
        E_skin=model.E_skin, nu_skin=model.nu_skin, t_skin=model.t_skin,
        fixed_nodes=model.fixed_nodes, loads=loads,
    )
    sliced = FrameResult(
        displacements=res.displacements,
        axial_force=res.axial_force[:n_beam],
        bending_moment=res.bending_moment[:n_beam],
        torsion=res.torsion[:n_beam],
    )
    return sliced, n_beam
