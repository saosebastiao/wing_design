# Hard Tip-Coupling Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Model a hard tip joint (a rigid gusset/clamp that all beams seat into) as a clique of very stiff connector beams tying the wing-tip nodes, and measure how it redistributes stress between beams (peak-beam stress, load spread) and stiffens the tip (deflection, twist) versus the baseline where the tip is joined only through the skin. Sweep the gusset stiffness soft→rigid to show the trend and the near-rigid limit.

**Scope note (chosen):** Stiff connector beams (tunable), not a true rigid-body MPC. The coupling is a clique of `BeamSection.circular(gusset_radius)` beams among the `n_beams` tip nodes, appended to the model's longitudinal beams and assembled by the existing `solve_beam_shell` — no new constraint machinery. The stiffness knob (`gusset_radius`) lets us sweep soft→rigid; very large radii are capped to stay well-conditioned ("rigid enough"). This is a measurement experiment, not a permanent design feature; geometry/CAD is out of scope.

**Architecture:** `beams/tip_coupling.py::solve_beam_shell_tip_coupled(model, loads, *, gusset_radius, beam_sections=None)` builds the tip-node clique as extra stiff beam elements, appends them to `model.beam_elements` + sections, calls `solve_beam_shell`, and returns `(FrameResult, n_beam)` so callers slice the *original* beams' internal forces (`[:n_beam]`) for stress, ignoring the coupling elements. An experiment example sweeps `gusset_radius` and reports the redistribution.

**Tech Stack:** numpy, the existing `structural.solve_beam_shell` + beam-shell model + `von_mises_per_element`.

---

## Background facts (verified)

- `beams/shell_model.py::BeamShellModel` fields: `nodes`, `beam_elements` (n_beam,2), `shell_tris`, `n_beams`, `n_levels`, `fixed_nodes`, `tip_nodes` (n_beams,), `section` (uniform `BeamSection`), `E_beam`, `G_beam`, `E_skin`, `nu_skin`, `t_skin`. In the beam-shell model the tip ring is joined ONLY by skin triangles (no ring beams).
- `structural/beam_shell.py::solve_beam_shell(nodes, beam_elements, beam_sections, shell_tris, *, E_beam, G_beam, E_skin, nu_skin, t_skin, fixed_nodes, loads, drilling_factor=1e-4) -> FrameResult`. It validates `len(beam_sections) == beam_elements.shape[0]`, assembles all beam elements + skin, solves, and recovers per-element axial/bending/torsion (tension-positive). Extra beam elements are assembled into the same global K.
- `structural/frame.py::BeamSection.circular(r)` (A=πr², I=πr⁴/4, J=πr⁴/2); `von_mises_per_element(result, sections) -> (n_elem,)`.
- `beams/fea_model.py::build_beam_frame`, `project_panels_to_beam_nodes`; aero/scenario wiring as in examples/24-25.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/beams/tip_coupling.py` (create) | `tip_clique_elements`, `solve_beam_shell_tip_coupled`. |
| `src/wing_design/beams/__init__.py` (modify) | Export both. |
| `examples/31_tip_coupling.py` (create) | Sweep gusset stiffness; measure stress redistribution + tip deflection/twist. |
| `tests/beams/test_tip_coupling.py` (create) | Clique count; gusset rigidifies the tip (relative tip motion shrinks); no-gusset path == plain solve. |
| `docs/plan.md` (modify) | Record the tip-coupling finding (under an "Investigations" note). |

---

## Task 1: Tip-coupling solver

**Files:**
- Create: `src/wing_design/beams/tip_coupling.py`
- Modify: `src/wing_design/beams/__init__.py`
- Test: `tests/beams/test_tip_coupling.py`

- [ ] **Step 1: Write the failing tests** (`tests/beams/test_tip_coupling.py`)

```python
import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.shell_model import build_beam_shell_model, solve_beam_shell_model
from wing_design.beams.tip_coupling import tip_clique_elements, solve_beam_shell_tip_coupled


def test_clique_element_count():
    nb = 8
    elems = tip_clique_elements(np.arange(nb))
    assert elems.shape == (nb * (nb - 1) // 2, 2)      # all unordered pairs
    assert elems.min() >= 0 and elems.max() < nb


def test_tip_coupling_reduces_relative_tip_motion():
    # Asymmetric tip load: a stiff gusset makes the tip nodes move together,
    # so the spread (std) of tip-node displacement collapses vs the skin-only baseline.
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes[0], 2] = 500.0     # push ONE tip node only

    base = solve_beam_shell_model(model, loads)
    coupled, n_beam = solve_beam_shell_tip_coupled(model, loads, gusset_radius=0.08)

    tip = model.tip_nodes
    spread_base = np.std(base.displacements[tip, 2])
    spread_coupled = np.std(coupled.displacements[tip, 2])
    assert spread_coupled < 0.5 * spread_base          # tip moves more rigidly
    assert n_beam == model.beam_elements.shape[0]      # real-beam count returned


def test_no_gusset_matches_plain_solve():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=6, n_levels=4, beam_radius=0.02)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 50.0
    plain = solve_beam_shell_model(model, loads)
    # gusset_radius=None -> no coupling elements -> identical to the plain solve
    none_res, _ = solve_beam_shell_tip_coupled(model, loads, gusset_radius=None)
    assert np.allclose(plain.displacements, none_res.displacements, atol=1e-12)
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/beams/test_tip_coupling.py -v` → ImportError.

- [ ] **Step 3: Implement** `src/wing_design/beams/tip_coupling.py`

```python
"""Hard tip-coupling experiment: a stiff 'gusset' tying all wing-tip beam nodes.

Models a rigid tip joint (clamp/gusset) as a clique of very stiff connector beams
among the tip nodes, appended to the model's longitudinal beams and solved by the
existing combined beam+shell solver. Lets load transfer directly beam-to-beam at the
tip (instead of only through skin shear). `gusset_radius` is the stiffness knob —
larger = closer to rigid. Returns the real-beam count so callers can slice the
original beams' internal forces (the coupling elements are not design members).
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

    `gusset_radius` None ⇒ no coupling (identical to the plain solve). Otherwise a
    clique of `BeamSection.circular(gusset_radius)` beams ties the tip nodes. The
    first `n_beam` elements of the result are the original longitudinal beams; the
    clique elements follow and are ignored for design-stress purposes.
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
    return res, n_beam
```

- [ ] **Step 4:** Export `tip_clique_elements`, `solve_beam_shell_tip_coupled` from `beams/__init__.py`. Run `uv run pytest tests/beams/test_tip_coupling.py -v` (3 PASS) and `uv run pytest` (green).

- [ ] **Step 5: Commit** — `feat(beams): hard tip-coupling solver (stiff gusset clique)` + trailer.

---

## Task 2: Measurement experiment + plan note

**Files:**
- Create: `examples/31_tip_coupling.py`
- Modify: `docs/plan.md`

- [ ] **Step 1: Create `examples/31_tip_coupling.py`** — the measurement:
  1. Build a uniform-radius beam-shell model (n_beams=16, n_levels=8, beam_radius=0.02 — a fixed design isolates the coupling effect) + frame + LL envelope `load_arrays`.
  2. For `gusset_radius in (None, 0.02, 0.04, 0.08, 0.15)` (None = skin-only baseline; increasing = softer→rigid gusset): for each load case `solve_beam_shell_tip_coupled(model, loads, gusset_radius=g)`, slice the real beams `vm = von_mises_per_element(res, [model.section]*n_beam)[:n_beam]`, take the worst over cases; also tip deflection (max translation norm over `model.tip_nodes`) and tip twist (deg, DOF 5). Track per-gusset: **max beam σ**, **beam-σ spread = max/mean** (how evenly load is shared), **tip deflection**, **tip twist**.
  3. Print a table over the sweep and summarize: as the gusset stiffens, does peak beam stress fall (load redistributes), does the σ spread shrink (more uniform sharing), and does the tip deflection/twist drop? Quantify baseline→rigid (e.g. "peak beam stress X→Y MPa, −Z%").

  (FOREGROUND, ~1–2 min: no sizing, just envelope solves × 5 gusset values.)

- [ ] **Step 2: Run** `uv run python examples/31_tip_coupling.py`. Paste full output. Report the measured effect: the baseline (skin-only) vs near-rigid peak beam stress, σ-spread, tip deflection, and twist, and whether the hard tip coupling meaningfully transfers/redistributes stress between beams. **Honest reporting** — if the skin already shares tip load well (small effect), say so; if the gusset markedly drops peak stress, quantify it. Watch for any spsolve conditioning warning at the largest gusset_radius (if so, note the rigid limit was capped).

- [ ] **Step 3: Update `docs/plan.md`** — add an "Investigations" note (a new short subsection, e.g. after the Phase-F.1 note or near the decisions log) recording: a hard tip joint modeled as a stiff connector-beam clique (`beams.solve_beam_shell_tip_coupled`), and the measured effect from the sweep (peak-beam-stress reduction, σ-spread change, tip-deflection/twist change baseline→rigid). Decisions-log row:

```markdown
| Tip-coupling study | Hard tip joint (gusset) modeled as a stiff connector-beam clique tying the tip nodes (`beams.solve_beam_shell_tip_coupled`, tunable gusset_radius), not a rigid MPC. Measured stress redistribution between beams + tip stiffening vs the skin-only baseline: <fill from run>. Investigation only (no CAD / not in the sizing loop). |
```

- [ ] **Step 4:** `uv run pytest` → green.

- [ ] **Step 5: Commit** — `feat(beams): tip-coupling measurement example; record finding` + trailer.

---

## Self-Review

- **Spec coverage:** hard tip coupling via FEA → `solve_beam_shell_tip_coupled` (stiff gusset clique, Task 1); measure stress transfer between beams → the sweep example (Task 2). Rigid-MPC and CAD explicitly out of scope.
- **Placeholder scan:** none.
- **Correctness anchors:** `gusset_radius=None` ⇒ identical to the plain `solve_beam_shell_model` (test); the clique has n(n−1)/2 elements (test); a stiff gusset collapses the spread of tip-node displacement under an asymmetric load (test — the core "it actually couples" check). The returned `n_beam` lets the example slice the *original* beams' stress (coupling elements excluded).
- **Type consistency:** `solve_beam_shell_tip_coupled(model, loads, *, gusset_radius, beam_sections=None) -> (FrameResult, n_beam)`; augmented `elements`/`sections` satisfy `solve_beam_shell`'s length guard; `von_mises_per_element(res, sections)[:n_beam]` isolates the design beams.
- **Known limitations (intentional):** stiff-connector approximation (not exact rigid — the sweep approaches it; very large `gusset_radius` is capped for conditioning); uniform-radius baseline design (isolates the coupling effect; the sized design would show the effect on the optimized structure too); coupling not fed into the sizing loop (measurement only); no CAD/joint geometry.
