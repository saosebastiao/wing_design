# Tip Gusset in the Sizing Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Opt-in rigid massless tip-gusset (a stiff connector-beam clique among the tip-ring nodes) wired into the CLT laminate sizer, so twist-governed designs relax the tip-twist constraint and shed mass.

**Architecture:** The gusset is carried on `BeamShellModel` (off by default). The laminate solve assembles it into `K` but **excludes it from beam-force recovery** (so `n_design` and `transforms`/`klocals` stay design-only). Because it's constant stiffness, the analytic adjoint Jacobian, symmetry, banding, etc. compose with **no sensitivity changes**.

**Domain facts:**
- `beams.tip_coupling.tip_clique_elements(tip_nodes) -> (n*(n-1)/2, 2)` all-pairs node-index pairs among the tip nodes.
- `_assemble_beam_shell_laminate(...)` (in `structural/beam_shell.py`) builds `K` from a beam loop (appends to `rows/cols/vals` AND `transforms`/`klocals`) + a shell loop; returns `(K, free, fixed_dofs, ndof, n_beam, transforms, klocals)`. `_recover_beam_forces` uses `transforms/klocals` (length n_beam).
- `solve_beam_shell_laminate` and `solve_beam_shell_laminate_factored` both call `_assemble_...`. `BeamSection.circular(r)`, `local_beam_stiffness`, `_element_rotation` available.
- `default_scenario` removed → tests use `small_scenario`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**Hard rule:** default off (`tip_gusset_radius=None`) → byte-identical to today; full suite green is the gate.

---

## Task 1: Carry the gusset on the model

**Files:** Modify `src/wing_design/beams/shell_model.py`, `src/wing_design/beams/__init__.py`; Test `tests/beams/test_tip_gusset.py` (new).

- [ ] **Step 1: Failing tests** — create `tests/beams/test_tip_gusset.py`:
```python
import numpy as np
from wing_design.scenario import small_scenario
from wing_design.beams.shell_model import build_beam_shell_model, model_with_tip_gusset
from wing_design.beams.tip_coupling import tip_clique_elements


def _m(**kw):
    return build_beam_shell_model(small_scenario().geometry, n_beams=8, n_levels=5, beam_radius=0.02, **kw)


def test_default_no_gusset():
    m = _m()
    assert m.tip_gusset_elements is None and m.tip_gusset_radius is None


def test_build_with_gusset():
    m = _m(tip_gusset_radius=0.05)
    assert m.tip_gusset_radius == 0.05
    exp = tip_clique_elements(m.tip_nodes)
    assert m.tip_gusset_elements.shape == exp.shape
    assert np.array_equal(m.tip_gusset_elements, exp)


def test_model_with_tip_gusset_roundtrip():
    m = _m()
    g = model_with_tip_gusset(m, 0.06)
    assert g.tip_gusset_radius == 0.06 and g.tip_gusset_elements is not None
    assert m.tip_gusset_elements is None  # original unchanged
```

- [ ] **Step 2:** Run → FAIL (kwarg/field/helper missing).

- [ ] **Step 3: Implement** in `shell_model.py`:
- Add fields to `BeamShellModel` (after the existing optional fields, defaulted):
  ```python
      tip_gusset_elements: np.ndarray | None = None
      tip_gusset_radius: float | None = None
  ```
- `build_beam_shell_model(..., tip_gusset_radius: float | None = None)`: import `from .tip_coupling import tip_clique_elements`; after computing `tip_nodes`, set
  `tip_gusset_elements = tip_clique_elements(tip_nodes) if tip_gusset_radius is not None else None`; pass both into the `BeamShellModel(...)` construction.
- Add `model_with_tip_gusset(model, radius)`:
  ```python
  def model_with_tip_gusset(model: BeamShellModel, radius: float) -> BeamShellModel:
      return dataclasses.replace(model, tip_gusset_radius=float(radius),
                                 tip_gusset_elements=tip_clique_elements(model.tip_nodes))
  ```
  (`import dataclasses` already present from the helix work; if not, add it.)
- Export `model_with_tip_gusset` from `beams/__init__.py` (import + `__all__`).

- [ ] **Step 4:** Run the 3 tests → PASS. **Step 5:** Commit (`Tip gusset: carry optional tip clique on the model`).

---

## Task 2: Assemble gusset into K, exclude from recovery

**Files:** Modify `src/wing_design/structural/beam_shell.py`; Test `tests/beams/test_tip_gusset.py` (append).

- [ ] **Step 1: Failing test** — at a FIXED design + a torsion-inducing tip load, the gusset slashes tip twist:
```python
from wing_design.structural.frame import BeamSection
from wing_design.materials.unidir import T700_EPOXY, laminate_stiffness
from wing_design.structural.beam_shell import solve_beam_shell_laminate
from wing_design.beams.tip_coupling import tip_clique_elements


def _fixed_case(n_beams=8, n_levels=5):
    m = _m(n_beams=n_beams, n_levels=n_levels) if False else build_beam_shell_model(
        small_scenario().geometry, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02)
    secs = [BeamSection.circular(0.01)] * m.beam_elements.shape[0]
    A, D, _ = laminate_stiffness(T700_EPOXY, f0=0.34, f45=0.33, f90=0.33, thickness=0.0015)
    loads = np.zeros((m.nodes.shape[0], 6))
    # asymmetric chordwise tip load -> tip twist
    loads[m.tip_nodes[0], 0] = 500.0
    return m, secs, A, D, loads


def test_gusset_reduces_tip_twist_fixed_design():
    m, secs, A, D, loads = _fixed_case()
    base = solve_beam_shell_laminate(m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D, fixed_nodes=m.fixed_nodes, loads=loads)
    clique = tip_clique_elements(m.tip_nodes)
    gus = solve_beam_shell_laminate(m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D, fixed_nodes=m.fixed_nodes, loads=loads,
        gusset_elements=clique, gusset_section=BeamSection.circular(0.05))
    base_tw = np.abs(base.displacements[m.tip_nodes, 5]).max()
    gus_tw = np.abs(gus.displacements[m.tip_nodes, 5]).max()
    assert gus_tw < 0.5 * base_tw            # large reduction
    assert base.axial_force.shape == gus.axial_force.shape == (m.beam_elements.shape[0],)  # recovery design-only
```

- [ ] **Step 2:** Run → FAIL (`gusset_elements` kwarg missing).

- [ ] **Step 3: Implement** in `beam_shell.py`:
- `_assemble_beam_shell_laminate(...)`: add params `gusset_elements: np.ndarray | None = None`, `gusset_section: BeamSection | None = None`. After the beam loop (before/after the shell loop, either order), add:
  ```python
  if gusset_elements is not None:
      if gusset_section is None:
          raise ValueError("gusset_section required when gusset_elements given")
      for e in range(gusset_elements.shape[0]):
          i, j = int(gusset_elements[e, 0]), int(gusset_elements[e, 1])
          R, L = _element_rotation(nodes[i], nodes[j])
          kloc = local_beam_stiffness(E_beam, G_beam, gusset_section, L)
          T = np.zeros((12, 12))
          for blk in range(4):
              T[3*blk:3*blk+3, 3*blk:3*blk+3] = R
          kg = T.T @ kloc @ T
          dofs = np.r_[6*i:6*i+6, 6*j:6*j+6]
          for a_ in range(12):
              for b_ in range(12):
                  rows.append(int(dofs[a_])); cols.append(int(dofs[b_])); vals.append(kg[a_, b_])
  ```
  NOTE: do NOT append to `transforms`/`klocals` (gusset excluded from recovery). `n_beam` stays the design count.
- `solve_beam_shell_laminate` and `solve_beam_shell_laminate_factored`: add the same two params and forward them to `_assemble_beam_shell_laminate(...)`. No other change (recovery already uses design `n_beam`/`transforms`/`klocals`).

- [ ] **Step 4:** Run the test → PASS. Run `tests/beams/test_sensitivity.py tests/beams/test_laminate_sizing.py -q` (no gusset args → unchanged) → green. **Step 5:** Commit.

---

## Task 3: Sizer wiring (FD + analytic) + composition tests

**Files:** Modify `src/wing_design/beams/laminate_sizing.py`; Test `tests/beams/test_tip_gusset.py` (append).

**Context:** When `model.tip_gusset_elements is not None`, pass the gusset to every laminate solve (the FD `evaluate` path AND the factored/analytic path). `BeamSection` is already imported in laminate_sizing.

- [ ] **Step 1: Failing tests:**
```python
from wing_design.beams import (LaminateSizingConfig, build_beam_frame, project_panels_to_beam_nodes,
                               size_beam_shell_laminate, model_with_tip_gusset)
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams.laminate_sizing import laminate_result_is_feasible


def _sizer_case(gusset=False, n_beams=8, n_levels=5):
    P = small_scenario(); spec = P.geometry
    m = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m,
        **({"tip_gusset_radius": 0.05} if gusset else {}))
    fr = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    env = sweep_envelope(build_airplane(spec), P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(fr, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02*spec.span,
        tip_twist_max_deg=5.0, buckling_safety_factor=1.5, n_skin_bands=2)
    return P, m, loads, cfg


def test_sizer_with_gusset_feasible_not_heavier():
    P, m, loads, cfg = _sizer_case(gusset=True)
    Pn, mn, loadsn, cfgn = _sizer_case(gusset=False)
    rg = size_beam_shell_laminate(m, loads, cfg, ply=P.material, rho=P.rho_kgm3, maxiter=60)
    rn = size_beam_shell_laminate(mn, loadsn, cfgn, ply=Pn.material, rho=Pn.rho_kgm3, maxiter=60)
    assert laminate_result_is_feasible(rg, cfg)
    assert rg.mass_kg <= rn.mass_kg + 0.25     # twist relief shouldn't cost mass


def test_analytic_jacobian_composes_with_gusset():
    P, m, loads, cfg = _sizer_case(gusset=True)
    cfg_an = LaminateSizingConfig(sigma_allow_Pa=cfg.sigma_allow_Pa, tip_defl_max_m=cfg.tip_defl_max_m,
        tip_twist_max_deg=cfg.tip_twist_max_deg, buckling_safety_factor=1.5, n_skin_bands=2,
        use_analytic_jacobian=True)
    r = size_beam_shell_laminate(m, loads, cfg_an, ply=P.material, rho=P.rho_kgm3, maxiter=60)
    assert laminate_result_is_feasible(r, cfg_an)
```
(The second test exercises the analytic path with the gusset in K; if a dedicated FD-gradient check is easy to add via the sensitivity harness, even better, but feasible convergence on the gusset-on analytic path is the minimum gate.)

- [ ] **Step 2:** Run → FAIL (gusset not passed to solve; ply=P.material is a UDPly — fine).

- [ ] **Step 3: Implement** in `laminate_sizing.py`: compute once near the top of `size_beam_shell_laminate`:
```python
    gusset_elements = model.tip_gusset_elements
    gusset_section = (BeamSection.circular(model.tip_gusset_radius)
                     if gusset_elements is not None else None)
```
Then add `gusset_elements=gusset_elements, gusset_section=gusset_section` to EVERY `solve_beam_shell_laminate(...)` call (FD `evaluate`) AND every `solve_beam_shell_laminate_factored(...)` call (the analytic/factored path). Nothing else changes (forces/sensitivity are design-only; the gusset just stiffens K).

- [ ] **Step 4:** Run the 2 tests → PASS. **Step 5:** FULL suite `uv run pytest -q` → green (default off unchanged). `grep -rn "LaminateSizingResult(" src/` → one site. **Step 6:** Commit.

---

## Task 4: Example + finding (orchestrator runs the timed comparison)

**Files:** Create `examples/39_tip_gusset.py`; Modify `docs/plan.md`.

- [ ] **Step 1:** Create `examples/39_tip_gusset.py` (compile-check only): medium wingsail, size gusset-off vs gusset-on (`tip_gusset_radius` ~0.1 m or a value scaled to the wing) under identical constraints (span datum, buckling SF 1.5, n_skin_bands=4, maxiter 300). Print for each: mass + split, tip twist, governing buckling util, beam radii range; then Δmass / Δtwist. Mirror examples/37 structure; use `medium_scenario` + `build_beam_shell_model(..., tip_gusset_radius=...)` for the on-case.
- [ ] **Step 2:** `uv run python -m py_compile examples/39_tip_gusset.py`. Commit.
- [ ] **Step 3 (orchestrator):** run it timed (background); record wall-clock + numbers.
- [ ] **Step 4:** `docs/plan.md` — status note + Decisions-log row: tip gusset in the sizer (opt-in, massless, rigid clique), off-vs-on result [Moff]→[Mon] kg, twist [Toff]→[Ton]°, governing-constraint flip, wall-clock; composes with analytic jac/symmetry/banding. Commit.

---

## Final verification
- [ ] `uv run pytest -q` — all green (default off byte-identical; gusset tests pass).
- [ ] `grep -rn "LaminateSizingResult(" src/` — one site.
- [ ] Orchestrator: timed off-vs-on run recorded; STL export optional.
- [ ] Final review, then superpowers:finishing-a-development-branch.

## Notes / risks
- **Default off byte-identical** — gusset args are None unless the model carries a gusset; the full suite under default is the gate.
- **Excluded from recovery** — do NOT add gusset elements to `transforms`/`klocals`; `n_beam`/recovery stay design-only. The FD test asserts `axial_force.shape == (n_design,)`.
- **Analytic Jacobian composes for free** — the gusset is constant stiffness; `factored.beam_elements`/`transforms` stay design-only, so `beams.sensitivity` needs NO change; the gusset only enters via `K`/`u`/the adjoint solves. The gusset-on analytic feasibility test guards this.
- **Massless** — mass objective unchanged.
