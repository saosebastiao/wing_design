# Buckling Constraints (closed-form) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add closed-form buckling constraints — Euler buckling of compression-loaded beam members and classical flat-plate buckling of skin panels — to the beam-shell sizing loops (E.2 isotropic and E.4 CLT), gated by a config flag, and re-size to determine whether buckling binds and how it changes the ~25.6 kg twist-governed design. This is a **validity check** on the headline sizing results: the sized members are thin (0.7 mm skin, 4–14 mm beams) with large stress margins, so buckling, not stress, may govern.

**Scope note:** Closed-form per-element/per-panel checks (the chosen fidelity), not eigenvalue buckling. Beam Euler uses element length as the buckling length (the skin laterally restrains intermediate nodes). Panel buckling treats each skin triangle as a simply-supported flat plate with characteristic width `b = sqrt(area)` and a fixed coefficient `kc` — explicitly approximate (documented). Eigenvalue/global buckling and refined panel geometry are deferred.

**Architecture:** Pure, analytic buckling utilization helpers in `structural/buckling.py` (`beam_euler_utilization`, `panel_buckling_utilization`) — each returns per-element utilization `= demand·SF / capacity`, so a constraint is simply `1 − util ≥ 0`. Both sizers (`shell_sizing.size_beam_shell`, `laminate_sizing.size_beam_shell_laminate`) gain optional buckling config (`buckling_safety_factor`, `euler_K`, `panel_kc`); when set, `evaluate()` also returns worst-case beam and panel buckling utilization, and two extra constraints are added. Default `None` ⇒ buckling off ⇒ existing tests unaffected.

**Tech Stack:** numpy, scipy SLSQP (existing), the E.2/E.4 sizing stack.

---

## Background facts (verified)

- `shell_sizing.size_beam_shell(model, load_arrays, config, *, rho, maxiter, ftol)`: `evaluate(x)` builds `sections=[BeamSection.circular(r)]`, solves `solve_beam_shell(...)` per case → `res` (FrameResult: `.axial_force`, `.displacements`), computes `vm = von_mises_per_element(res, sections)`, `skin_s = recover_membrane_stress(nodes, tris, disp, E=E_skin, nu=nu_skin)` then `membrane_von_mises`. Returns `(worst_vm (n,), worst_skin_vm, max_defl, max_twist)`. Uses `Lb = beam_lengths(model)`, `Atri = skin_areas(model)`. Constraints: `beam_con` (vector), `skin_con/defl_con/twist_con` (1-arrays). Result `BeamShellSizingResult`.
- `laminate_sizing.size_beam_shell_laminate(model, load_arrays, config, *, ply, rho, ...)`: design vector `[radii (n), t, f0, f45]`; `evaluate` builds `A,D,Qeff = laminate_stiffness(ply,f0,f45,f90,t)`, solves `solve_beam_shell_laminate(..., A_skin=A, D_skin=D)`, skin stress via `recover_membrane_stress_C(..., C=Qeff)`. Returns `(worst_beam (n,), worst_skin, max_defl, max_twist)`. Result `LaminateSizingResult`.
- `BeamSection.circular(r)`: `Iy=Iz=π r⁴/4`. `FrameResult.axial_force` is tension-positive (compression negative).
- Skin membrane stress `(M,3)` = `[σxx, σyy, σxy]`; principal `σ_min = mean − sqrt(0.25(σxx−σyy)² + σxy²)`.
- Isotropic skin plate bending stiffness `D11 = E·t³/(12(1−ν²))`; laminate `D11 = D[0,0]` from `laminate_stiffness`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/structural/buckling.py` (create) | `beam_euler_utilization`, `panel_buckling_utilization` (pure analytic). |
| `src/wing_design/structural/__init__.py` (modify) | Export both. |
| `src/wing_design/beams/shell_sizing.py` (modify) | Buckling config fields + evaluate + constraints + result fields (isotropic sizer). |
| `src/wing_design/beams/laminate_sizing.py` (modify) | Same for the CLT sizer. |
| `examples/27_buckling.py` (create) | Re-size CLT with vs without buckling; report binding constraint + mass change. |
| Tests: `tests/structural/test_buckling.py`, additions to `tests/beams/test_shell_sizing.py` & `test_laminate_sizing.py`. |
| `docs/plan.md` (modify) | Record the buckling finding. |

---

## Task 1: Buckling utilization helpers

**Files:**
- Create: `src/wing_design/structural/buckling.py`
- Modify: `src/wing_design/structural/__init__.py`
- Test: `tests/structural/test_buckling.py`

- [ ] **Step 1: Write the failing tests** (`tests/structural/test_buckling.py`)

```python
import numpy as np

from wing_design.structural.buckling import beam_euler_utilization, panel_buckling_utilization


def test_euler_utilization_against_closed_form():
    E, r, L = 70e9, 0.02, 1.0
    I = np.pi * r**4 / 4.0
    Pcr = np.pi**2 * E * I / L**2          # K=1
    # compression equal to Pcr/2 with SF=2 → utilization exactly 1.0
    axial = np.array([-Pcr / 2.0])         # negative = compression
    u = beam_euler_utilization(axial, np.array([r]), np.array([L]), E=E, K=1.0, safety_factor=2.0)
    assert abs(u[0] - 1.0) < 1e-9
    # tension never buckles
    ut = beam_euler_utilization(np.array([+Pcr]), np.array([r]), np.array([L]), E=E, safety_factor=2.0)
    assert ut[0] == 0.0


def test_panel_utilization_against_closed_form():
    E, nu, t, area, kc = 70e9, 0.3, 0.003, 0.04, 4.0
    D11 = E * t**3 / (12 * (1 - nu**2))
    b = np.sqrt(area)
    sigma_cr = kc * np.pi**2 * D11 / (b**2 * t)
    # uniaxial compression σxx = -sigma_cr (most-compressive principal = sigma_cr), SF=1 → util 1
    stress = np.array([[-sigma_cr, 0.0, 0.0]])
    u = panel_buckling_utilization(stress, np.array([area]), D11=D11, t=t, kc=kc, safety_factor=1.0)
    assert abs(u[0] - 1.0) < 1e-9
    # pure tension → no buckling
    ut = panel_buckling_utilization(np.array([[sigma_cr, 0.0, 0.0]]), np.array([area]), D11=D11, t=t, kc=kc)
    assert ut[0] == 0.0
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/structural/test_buckling.py -v` → ImportError.

- [ ] **Step 3: Implement** `src/wing_design/structural/buckling.py`

```python
"""Closed-form buckling utilization checks for the beam-shell structure.

Each function returns a per-element *utilization* = demand·safety_factor / capacity,
so a feasibility constraint is simply `utilization <= 1`. Approximations (documented):
beam buckling uses the element length as the effective buckling length (the skin
laterally restrains intermediate nodes); panel buckling treats each skin triangle as
a simply-supported flat plate of characteristic width `b = sqrt(area)` with a fixed
coefficient `kc`. Tension never buckles (utilization 0).
"""
from __future__ import annotations

import numpy as np


def beam_euler_utilization(
    axial_force: np.ndarray,
    radii: np.ndarray,
    lengths: np.ndarray,
    *,
    E: float,
    K: float = 1.0,
    safety_factor: float = 1.0,
) -> np.ndarray:
    """Per-beam Euler buckling utilization = (compression·SF) / Pcr.

    `axial_force` tension-positive (compression negative). Pcr = π²·E·I / (K·L)²,
    I = π r⁴/4. Tension members return 0.
    """
    comp = np.maximum(0.0, -np.asarray(axial_force, dtype=float))
    r = np.asarray(radii, dtype=float)
    L = np.asarray(lengths, dtype=float)
    second_moment = np.pi * r**4 / 4.0
    pcr = np.pi**2 * E * second_moment / (K * L) ** 2
    return comp * safety_factor / np.maximum(pcr, 1e-30)


def panel_buckling_utilization(
    membrane_stress: np.ndarray,
    areas: np.ndarray,
    *,
    D11: float,
    t: float,
    kc: float = 4.0,
    safety_factor: float = 1.0,
) -> np.ndarray:
    """Per-panel plate-buckling utilization = (compressive principal stress·SF) / σcr.

    σcr = kc·π²·D11 / (b²·t), b = sqrt(area). `membrane_stress` is (M,3) [σxx,σyy,σxy];
    the most-compressive principal stress drives buckling. Tension panels return 0.
    Approximate (triangle-as-plate, fixed kc) — see module docstring.
    """
    s = np.asarray(membrane_stress, dtype=float)
    mean = 0.5 * (s[:, 0] + s[:, 1])
    radius = np.sqrt(0.25 * (s[:, 0] - s[:, 1]) ** 2 + s[:, 2] ** 2)
    s_min = mean - radius
    comp = np.maximum(0.0, -s_min)
    b = np.sqrt(np.maximum(np.asarray(areas, dtype=float), 1e-30))
    sigma_cr = kc * np.pi**2 * D11 / (b**2 * t)
    return comp * safety_factor / np.maximum(sigma_cr, 1e-30)
```

- [ ] **Step 4:** Export both from `structural/__init__.py`. Run `uv run pytest tests/structural/test_buckling.py -v` → 2 PASS; `uv run pytest` → green.

- [ ] **Step 5: Commit** — `feat(structural): closed-form beam Euler + panel buckling utilization` + trailer.

---

## Task 2: Buckling constraints in the isotropic sizer

**Files:**
- Modify: `src/wing_design/beams/shell_sizing.py`
- Test: `tests/beams/test_shell_sizing.py`

- [ ] **Step 1: Write the failing test** (append to `tests/beams/test_shell_sizing.py`)

```python
def test_buckling_constraint_respected():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=4)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 800.0
    loads[model.tip_nodes, 0] = 400.0
    cfg = BeamShellSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=1.5,
    )
    res = size_beam_shell(model, [loads], cfg, rho=1550.0, maxiter=80)
    # with buckling on, the worst beam/panel buckling utilization must be <= 1 (feasible)
    assert res.max_beam_buckling_util <= 1.05
    assert res.max_panel_buckling_util <= 1.05
```

- [ ] **Step 2: Run to verify it fails** — `AttributeError`/`TypeError` (config field + result fields missing).

- [ ] **Step 3: Implement** — edit `src/wing_design/beams/shell_sizing.py`:

(a) Add to `BeamShellSizingConfig`:
```python
    buckling_safety_factor: float | None = None  # if set, enforce beam Euler + panel buckling
    euler_K: float = 1.0
    panel_kc: float = 4.0
```

(b) Add to `BeamShellSizingResult` (after the stress fields):
```python
    max_beam_buckling_util: float
    max_panel_buckling_util: float
```

(c) Import the helpers at top: `from ..structural.buckling import beam_euler_utilization, panel_buckling_utilization`.

(d) In `size_beam_shell`, extend `evaluate(x)` to also compute buckling utilizations. The current loop already has `res` (per case) and computes `skin_s` for the skin vm — capture `skin_s` and `res.axial_force`. Track:
```python
        worst_beam_buck = 0.0
        worst_panel_buck = 0.0
```
and inside the per-case loop, AFTER computing `skin_s` (the membrane stress array used for skin vm):
```python
            if config.buckling_safety_factor is not None:
                bu = beam_euler_utilization(res.axial_force, radii, Lb, E=model.E_beam,
                                            K=config.euler_K, safety_factor=config.buckling_safety_factor)
                D11 = model.E_skin * t ** 3 / (12.0 * (1.0 - model.nu_skin ** 2))
                pu = panel_buckling_utilization(skin_s, Atri, D11=D11, t=t,
                                                kc=config.panel_kc, safety_factor=config.buckling_safety_factor)
                worst_beam_buck = max(worst_beam_buck, float(bu.max()))
                worst_panel_buck = max(worst_panel_buck, float(pu.max()))
```
(`radii` and `t` are the local design split already present in evaluate; `skin_s` must be the `recover_membrane_stress(...)` result — if the current code inlines it into `membrane_von_mises(...)`, refactor to a named `skin_s` variable first.) Extend the cached `out` tuple to `(worst_vm, worst_skin, max_defl, max_twist, worst_beam_buck, worst_panel_buck)` and update ALL constraint-function unpackings accordingly (use `_` for unused slots).

(e) Add buckling constraints conditionally, before the `minimize` call:
```python
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
    res = minimize(mass, x0, jac=mass_grad, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": maxiter, "ftol": ftol})
```

(f) After the final `evaluate(x)` (post-clip), unpack the two buckling utilizations and populate the new result fields:
```python
    worst_vm, worst_skin, d, t_metric, bbu, pbu = evaluate(x)
    ...
    return BeamShellSizingResult(..., max_beam_buckling_util=float(bbu), max_panel_buckling_util=float(pbu))
```
(Keep the existing field assignments; add the two new ones.)

- [ ] **Step 4:** Run `uv run pytest tests/beams/test_shell_sizing.py -v` (the existing tests — buckling off by default — must still pass; the new one passes) and `uv run pytest`. If buckling-on is infeasible (utilization stays > 1), STOP and report.

- [ ] **Step 5: Commit** — `feat(beams): optional buckling constraints in isotropic beam-shell sizing` + trailer.

---

## Task 3: Buckling constraints in the CLT sizer

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py`
- Test: `tests/beams/test_laminate_sizing.py`

- [ ] **Step 1: Write the failing test** (append to `tests/beams/test_laminate_sizing.py`)

```python
def test_laminate_buckling_constraint_respected():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=4)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 800.0
    loads[model.tip_nodes, 0] = 400.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=1.5,
    )
    res = size_beam_shell_laminate(model, [loads], cfg, ply=T700_EPOXY, rho=1550.0, maxiter=80)
    assert res.max_beam_buckling_util <= 1.05
    assert res.max_panel_buckling_util <= 1.05
    assert abs(res.f0 + res.f45 + res.f90 - 1.0) < 1e-6
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — mirror Task 2 in `laminate_sizing.py`:
  - `LaminateSizingConfig` gains `buckling_safety_factor: float | None = None`, `euler_K: float = 1.0`, `panel_kc: float = 4.0`.
  - `LaminateSizingResult` gains `max_beam_buckling_util: float`, `max_panel_buckling_util: float`.
  - Import the helpers.
  - In `evaluate`, after `skin_s = recover_membrane_stress_C(..., C=Qeff)`, add the same buckling block but with **laminate** plate stiffness `D11 = D[0, 0]` (from the `A, D, Qeff = laminate_stiffness(...)` already computed), using `radii` and `Lb`. Extend the returned tuple with the two buckling utils; update all unpackings.
  - Add the two conditional buckling constraints exactly as in Task 2.
  - Populate the new result fields after the final evaluate.

- [ ] **Step 4:** `uv run pytest tests/beams/test_laminate_sizing.py -v` (existing tests still pass; new passes) and `uv run pytest`. If infeasible, STOP and report.

- [ ] **Step 5: Commit** — `feat(beams): optional buckling constraints in CLT beam-shell sizing` + trailer.

---

## Task 4: Validity-check example + plan update

**Files:**
- Create: `examples/27_buckling.py`
- Modify: `docs/plan.md`

- [ ] **Step 1: Create `examples/27_buckling.py`** — re-size the CLT model at the working scenario (n_beams=16, n_levels=8) twice: WITHOUT buckling (baseline = E.4) and WITH `buckling_safety_factor=1.5`, under the same envelope loads/limits. Print for each: total mass + beam/skin split + skin thickness + layup + max beam/panel buckling utilization + tip defl/twist + max stresses. Then state whether buckling binds and the mass change. Mirror `examples/26_clt_skin.py` structure; use `LaminateSizingConfig(..., buckling_safety_factor=1.5)` for the second run. (Two SLSQP sizings; run FOREGROUND to completion, ~4–7 min.)

- [ ] **Step 2: Run** `uv run python examples/27_buckling.py`. Paste full output. Report: did buckling bind (utilization → 1)? How much heavier is the buckling-feasible design vs the E.4 result? Did the binding constraint change from twist to buckling? **Honest reporting** — if buckling barely matters (low utilization, ~no mass change) that is itself the validity finding (the design was already buckling-safe); if it dominates, the prior mass numbers were optimistic.

- [ ] **Step 3: Update `docs/plan.md`** — add a "Buckling (closed-form)" status note under Phase E with the ACTUAL numbers: whether beam/panel buckling binds, the buckling-feasible mass vs the no-buckling 25.6 kg, and the new governing constraint. Decisions-log row:

```markdown
| Buckling check | Closed-form beam Euler (element-length) + skin panel plate-buckling (triangle-as-plate, b=sqrt(area), fixed kc) added as optional sizing constraints in both sizers (gated by buckling_safety_factor). Eigenvalue/global buckling + refined panel geometry deferred. |
```

- [ ] **Step 4:** `uv run pytest` → green.

- [ ] **Step 5: Commit** — `feat(beams): buckling validity-check example; record finding` + trailer.

---

## Self-Review

- **Spec coverage:** closed-form beam Euler + panel buckling → Task 1 helpers; integrated into both sizers gated by config → Tasks 2–3; re-size validity check → Task 4.
- **Placeholder scan:** none.
- **Type consistency:** helpers take `(axial_force, radii, lengths)` / `(membrane_stress, areas)` already available in `evaluate` (`res.axial_force`, `radii`, `Lb`, `skin_s`, `Atri`); `D11` from isotropic formula (Task 2) or laminate `D[0,0]` (Task 3); utilization → constraint `1−util≥0`; new result fields populated from the final evaluate.
- **Backward compatibility:** `buckling_safety_factor` defaults to `None` ⇒ no buckling constraints ⇒ existing E.2/E.4 tests and results unchanged. The new result fields are always populated (0.0 when buckling off, since the evaluate block is gated — ensure the tuple still returns 0.0 defaults when off so unpacking doesn't break; initialize `worst_beam_buck=worst_panel_buck=0.0` unconditionally and only update them inside the `if`).
- **Known approximations (intentional, documented):** beam buckling length = element length (skin restrains nodes); panel buckling = triangle-as-plate with `b=sqrt(area)` and fixed `kc` (rough); compression from the most-compressive principal membrane stress; no eigenvalue/global modes. These make the check conservative-ish and transparent, suitable for a validity spike.
