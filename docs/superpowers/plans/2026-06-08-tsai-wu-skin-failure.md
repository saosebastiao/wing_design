# Per-ply Tsai-Wu Skin Failure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in per-ply Tsai-Wu skin failure criterion to the CLT co-sizer (strength-ratio R ≥ SF), replacing the laminate-average von-Mises proxy when enabled; default behavior unchanged.

**Architecture:** A new `materials/failure.py` holds the Tsai-Wu math (coefficients, index, strength ratio) plus a per-ply and a laminate-min strength-ratio helper that transform the element-local membrane strain into each present ply's material axes. `structural/shell.py` gains `recover_membrane_strain`. The sizer gains a `skin_failure` flag and `tsai_wu_safety_factor`; when `"tsai_wu"`, the skin constraint becomes `min_R/SF − 1 ≥ 0`.

**Tech Stack:** Python, numpy, scipy SLSQP, pytest, `uv run pytest`. Existing: `materials.unidir` (`UDPly` with `Xt_Pa,Xc_Pa,Yt_Pa,Yc_Pa,S12_Pa`, `reduced_stiffness_Q`), `structural.shell` (`recover_membrane_stress_C`, `_triangle_local_frame`), `beams.laminate_sizing.size_beam_shell_laminate`.

**Domain notes for the implementer (read once):**
- `UDPly` strengths are positive magnitudes: `Xt_Pa, Xc_Pa, Yt_Pa, Yc_Pa, S12_Pa` (e.g. T700: Xt=2200e6, Xc=1200e6, Yt=60e6, Yc=210e6, S12=90e6).
- `reduced_stiffness_Q(ply)` → material-axis plane-stress stiffness `Q` (3,3):
  `[[Q11,Q12,0],[Q12,Q22,0],[0,0,Q66]]`.
- Membrane strain/stress vectors are `[xx, yy, xy]` with **engineering** shear γ.
- The sizer's `evaluate(x)` currently returns a 6-tuple
  `(wb, ws, md, mt, worst_beam_buck, worst_panel_buck)`; this plan extends it to a
  7-tuple by appending `worst_R`. Several constraint closures unpack that tuple — they
  MUST be updated (the buckling closures use negative indexing today and will break
  silently if not fixed).
- Loads are already load-factored; the Tsai-Wu SF is the material side only.
- Full suite is currently **92 passed**. Commit trailer (every commit):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `src/wing_design/materials/failure.py` | Tsai-Wu coefficients/index/strength-ratio + per-ply & laminate-min helpers | **new** |
| `src/wing_design/materials/__init__.py` | export the failure helpers | modify |
| `src/wing_design/structural/shell.py` | `recover_membrane_strain` | modify |
| `src/wing_design/beams/laminate_sizing.py` | `skin_failure` + `tsai_wu_safety_factor` config; Tsai-Wu skin constraint; result `min_skin_strength_ratio` | modify |
| `examples/34_tsai_wu_skin.py` | von-Mises-vs-Tsai-Wu comparison | **new** |
| `tests/materials/test_failure.py` | failure-math + per-ply tests | **new** |
| `tests/beams/test_tsai_wu_sizing.py` | sizer + strain-recovery tests | **new** |
| `docs/plan.md` | finding + Decisions-log row (final task) | modify |

---

## Task 1: Tsai-Wu failure math

**Files:**
- Create: `src/wing_design/materials/failure.py`
- Test: `tests/materials/test_failure.py`

- [ ] **Step 1: Write the failing test**

Create `tests/materials/test_failure.py`:

```python
import numpy as np

from wing_design.materials.unidir import T700_EPOXY
from wing_design.materials.failure import (
    tsai_wu_coefficients, tsai_wu_index, tsai_wu_strength_ratio,
    ply_strength_ratio, laminate_min_strength_ratio,
)

PLY = T700_EPOXY


def test_index_unit_at_uniaxial_tension():
    assert np.isclose(tsai_wu_index([PLY.Xt_Pa, 0.0, 0.0], PLY), 1.0, rtol=1e-9)


def test_strength_ratio_one_at_each_strength_point():
    assert np.isclose(tsai_wu_strength_ratio([PLY.Xt_Pa, 0.0, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([-PLY.Xc_Pa, 0.0, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([0.0, PLY.Yt_Pa, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([0.0, -PLY.Yc_Pa, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([0.0, 0.0, PLY.S12_Pa], PLY), 1.0, rtol=1e-9)


def test_strength_ratio_scales_inverse_with_shear():
    r1 = tsai_wu_strength_ratio([0.0, 0.0, 1.0e7], PLY)
    r2 = tsai_wu_strength_ratio([0.0, 0.0, 5.0e6], PLY)
    assert np.isclose(r2, 2.0 * r1, rtol=1e-9)


def test_zero_stress_is_safe():
    assert tsai_wu_strength_ratio([0.0, 0.0, 0.0], PLY) >= 1.0e8


def test_f12_convention():
    F1, F2, F11, F22, F66, F12 = tsai_wu_coefficients(PLY)
    assert np.isclose(F12, -0.5 * np.sqrt(F11 * F22), rtol=1e-12)


def test_ply_strength_ratio_fibre_vs_transverse():
    # A uniaxial extensional element-local strain is far safer aligned to the fibre
    # (angle 0) than transverse to it (angle 90).
    eps = [1.0e-3, 0.0, 0.0]
    r0 = ply_strength_ratio(PLY, eps, 0.0)
    r90 = ply_strength_ratio(PLY, eps, 90.0)
    assert r0 > r90


def test_laminate_min_skips_absent_orientations():
    eps = [1.0e-3, 0.0, 0.0]
    # all orientations present
    r_all = laminate_min_strength_ratio(PLY, eps, f0=0.34, f45=0.33, f90=0.33, offset_deg=0.0)
    # only 0-degree plies present -> >= the all-present min (fewer/safer orientations checked)
    r_0only = laminate_min_strength_ratio(PLY, eps, f0=1.0, f45=0.0, f90=0.0, offset_deg=0.0)
    assert r_0only >= r_all
    assert np.isclose(r_0only, ply_strength_ratio(PLY, eps, 0.0), rtol=1e-9)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/materials/test_failure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wing_design.materials.failure'`

- [ ] **Step 3: Implement**

Create `src/wing_design/materials/failure.py`:

```python
"""Per-ply Tsai-Wu failure criterion for a unidirectional CFRP ply.

Tsai-Wu in the ply material axes:
    FI = F1 s1 + F2 s2 + F11 s1^2 + F22 s2^2 + F66 t12^2 + 2 F12 s1 s2,  failure when FI >= 1.
The strength ratio R is the factor by which the current stress can scale before FI = 1
(failure when R < 1); it is the interpretable margin used by the sizer's constraint.

`ply_strength_ratio` / `laminate_min_strength_ratio` take the element-local membrane
strain [exx, eyy, gxy] (engineering shear), transform it into each present ply's
material axes, recover the ply stress with the material-axis reduced stiffness Q, and
return the (min) strength ratio.
"""
from __future__ import annotations

import numpy as np

from .unidir import UDPly, reduced_stiffness_Q

_SAFE_R = 1.0e9  # returned when there is effectively no stress (no failure)


def tsai_wu_coefficients(ply: UDPly) -> tuple[float, float, float, float, float, float]:
    """(F1, F2, F11, F22, F66, F12) with F12 = -0.5*sqrt(F11*F22)."""
    F1 = 1.0 / ply.Xt_Pa - 1.0 / ply.Xc_Pa
    F2 = 1.0 / ply.Yt_Pa - 1.0 / ply.Yc_Pa
    F11 = 1.0 / (ply.Xt_Pa * ply.Xc_Pa)
    F22 = 1.0 / (ply.Yt_Pa * ply.Yc_Pa)
    F66 = 1.0 / (ply.S12_Pa ** 2)
    F12 = -0.5 * np.sqrt(F11 * F22)
    return F1, F2, F11, F22, F66, F12


def tsai_wu_index(sigma123, ply: UDPly) -> float:
    """Tsai-Wu failure index FI for material-axis stress [s1, s2, t12]; failure at FI>=1."""
    s1, s2, t12 = (float(v) for v in sigma123)
    F1, F2, F11, F22, F66, F12 = tsai_wu_coefficients(ply)
    return float(F1 * s1 + F2 * s2 + F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t12 ** 2 + 2.0 * F12 * s1 * s2)


def tsai_wu_strength_ratio(sigma123, ply: UDPly) -> float:
    """Strength ratio R (R*sigma reaches the envelope); failure when R<1.

    Solves a*R^2 + b*R - 1 = 0 with a = quadratic terms, b = linear terms. Returns a
    large finite value when there is essentially no stress (a,b ~ 0).
    """
    s1, s2, t12 = (float(v) for v in sigma123)
    F1, F2, F11, F22, F66, F12 = tsai_wu_coefficients(ply)
    a = F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t12 ** 2 + 2.0 * F12 * s1 * s2
    b = F1 * s1 + F2 * s2
    if a <= 1.0e-30:
        if b <= 1.0e-30:
            return _SAFE_R
        return min(_SAFE_R, 1.0 / b)
    R = (-b + np.sqrt(b * b + 4.0 * a)) / (2.0 * a)
    return float(min(_SAFE_R, R))


def _strain_to_ply_axes(eps_local, angle_deg: float) -> np.ndarray:
    """Transform element-local engineering strain [exx,eyy,gxy] to ply axes at angle_deg."""
    ex, ey, gxy = (float(v) for v in eps_local)
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    e1 = ex * c * c + ey * s * s + gxy * s * c
    e2 = ex * s * s + ey * c * c - gxy * s * c
    g12 = -2.0 * ex * s * c + 2.0 * ey * s * c + gxy * (c * c - s * s)
    return np.array([e1, e2, g12])


def ply_strength_ratio(ply: UDPly, eps_local, angle_deg: float) -> float:
    """Tsai-Wu strength ratio of a single ply at `angle_deg` under element-local strain."""
    eps123 = _strain_to_ply_axes(eps_local, angle_deg)
    sigma123 = reduced_stiffness_Q(ply) @ eps123
    return tsai_wu_strength_ratio(sigma123, ply)


def laminate_min_strength_ratio(
    ply: UDPly, eps_local, *, f0: float, f45: float, f90: float, offset_deg: float,
) -> float:
    """Min Tsai-Wu strength ratio over the PRESENT ply orientations.

    Orientations are 0/+45/-45/90 (relative to the datum) shifted into element-local
    axes by ``offset_deg``; an orientation is checked only if its area fraction > 0.
    """
    eps_tol = 1.0e-9
    angles: list[float] = []
    if f0 > eps_tol:
        angles.append(0.0 + offset_deg)
    if f45 > eps_tol:
        angles.append(45.0 + offset_deg)
        angles.append(-45.0 + offset_deg)
    if f90 > eps_tol:
        angles.append(90.0 + offset_deg)
    if not angles:
        return _SAFE_R
    return min(ply_strength_ratio(ply, eps_local, a) for a in angles)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/materials/test_failure.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/materials/failure.py tests/materials/test_failure.py
git commit -m "$(cat <<'EOF'
Tsai-Wu: per-ply failure math (coefficients, index, strength ratio)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Export failure helpers

**Files:**
- Modify: `src/wing_design/materials/__init__.py`
- Test: `tests/materials/test_failure.py` (append)

- [ ] **Step 1: Append the failing test**

```python
def test_failure_helpers_exported():
    import wing_design.materials as m
    for name in ("tsai_wu_index", "tsai_wu_strength_ratio", "tsai_wu_coefficients",
                 "ply_strength_ratio", "laminate_min_strength_ratio"):
        assert hasattr(m, name), name
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/materials/test_failure.py -k exported -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `src/wing_design/materials/__init__.py`, add an import block and extend `__all__`:
```python
from .failure import (
    laminate_min_strength_ratio,
    ply_strength_ratio,
    tsai_wu_coefficients,
    tsai_wu_index,
    tsai_wu_strength_ratio,
)
```
Add to `__all__`:
```python
    "tsai_wu_coefficients",
    "tsai_wu_index",
    "tsai_wu_strength_ratio",
    "ply_strength_ratio",
    "laminate_min_strength_ratio",
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/materials/test_failure.py -k exported -v`  → PASS

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/materials/__init__.py tests/materials/test_failure.py
git commit -m "$(cat <<'EOF'
Tsai-Wu: export failure helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `recover_membrane_strain`

**Files:**
- Modify: `src/wing_design/structural/shell.py`
- Test: `tests/beams/test_tsai_wu_sizing.py` (create)

**Context:** The Tsai-Wu check needs the per-triangle element-local membrane strain. Add
`recover_membrane_strain`, mechanically identical to `recover_membrane_stress_C` with
`C = I3`. Place it right after `recover_membrane_stress_C` in `shell.py`.

- [ ] **Step 1: Create the failing test** `tests/beams/test_tsai_wu_sizing.py`:

```python
import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.shell_model import build_beam_shell_model, solve_beam_shell_model
from wing_design.structural.shell import recover_membrane_strain, recover_membrane_stress_C


def test_recover_membrane_strain_matches_identity_C():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 0] = 200.0
    res = solve_beam_shell_model(model, loads)
    eps = recover_membrane_strain(model.nodes, model.shell_tris, res.displacements)
    eps_via_C = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=np.eye(3))
    assert eps.shape == (model.shell_tris.shape[0], 3)
    assert np.allclose(eps, eps_via_C, atol=1e-12)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/beams/test_tsai_wu_sizing.py -k strain -v`
Expected: FAIL with `ImportError: cannot import name 'recover_membrane_strain'`

- [ ] **Step 3: Implement**

In `src/wing_design/structural/shell.py`, add after `recover_membrane_stress_C`:
```python
def recover_membrane_strain(
    nodes: np.ndarray, triangles: np.ndarray, displacements: np.ndarray,
) -> np.ndarray:
    """Per-triangle element-local membrane strain (M,3) = [exx, eyy, gxy] (engineering).

    eps = Bm @ u_local; identical to recover_membrane_stress_C with C = identity. This
    is the input to the per-ply Tsai-Wu check (materials.failure).
    """
    M = triangles.shape[0]
    out = np.zeros((M, 3))
    for e in range(M):
        i, j, l = (int(v) for v in triangles[e])
        R, local, area = _triangle_local_frame(nodes[i], nodes[j], nodes[l])
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
        u_m = np.zeros(6)
        for n_idx, gn in enumerate((i, j, l)):
            u_local = R.T @ displacements[gn, 0:3]
            u_m[2 * n_idx + 0] = u_local[0]
            u_m[2 * n_idx + 1] = u_local[1]
        out[e] = Bm @ u_m
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/beams/test_tsai_wu_sizing.py -k strain -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/structural/shell.py tests/beams/test_tsai_wu_sizing.py
git commit -m "$(cat <<'EOF'
Tsai-Wu: recover_membrane_strain (element-local membrane strain)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Tsai-Wu skin constraint in the sizer

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py`
- Test: `tests/beams/test_tsai_wu_sizing.py` (append)

**Context:** Core change. Add `skin_failure`/`tsai_wu_safety_factor` to the config and
`min_skin_strength_ratio` to the result. In `evaluate`, compute the worst (min) Tsai-Wu
strength ratio when enabled; swap the skin constraint accordingly. CRITICAL: `evaluate`
becomes a 7-tuple — every closure that unpacks it must be updated (the buckling closures
currently use negative indexing and will silently read the wrong element otherwise).

Read the current `size_beam_shell_laminate` fully before editing.

- [ ] **Step 1: Append the failing tests**

```python
from wing_design import default_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig, build_beam_frame, build_beam_shell_model,
    project_panels_to_beam_nodes, size_beam_shell_laminate,
)
from wing_design.materials.unidir import T700_EPOXY


def _scenario_small(n_beams=8, n_levels=5):
    P = default_scenario()
    spec = P.geometry
    model = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    return P, spec, model, loads


def test_von_mises_default_has_none_ratio():
    P, spec, model, loads = _scenario_small()
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=15)
    assert r.min_skin_strength_ratio is None


def test_tsai_wu_feasible_and_active():
    P, spec, model, loads = _scenario_small()
    sf = 2.0
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        ply_angle_datum=(0.0, 0.0, 1.0), skin_failure="tsai_wu", tsai_wu_safety_factor=sf)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=40)
    assert r.min_skin_strength_ratio is not None
    # feasible: margin at least SF (small tol for SLSQP)
    assert r.min_skin_strength_ratio >= sf - 0.05
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/beams/test_tsai_wu_sizing.py -k "none_ratio or tsai_wu_feasible" -v`
Expected: FAIL with `AttributeError: 'LaminateSizingResult' object has no attribute 'min_skin_strength_ratio'`

- [ ] **Step 3: Implement**

3a. Add the imports near the top of `laminate_sizing.py` (with the other `from ..` imports):
```python
from ..materials.failure import laminate_min_strength_ratio
from ..structural.shell import membrane_von_mises, recover_membrane_strain, recover_membrane_stress_C
```
(Extend the EXISTING `from ..structural.shell import ...` line to add `recover_membrane_strain` rather than duplicating it; keep `membrane_von_mises` and `recover_membrane_stress_C`.)

3b. Add to `LaminateSizingConfig` (after `n_skin_bands`):
```python
    n_skin_bands: int = 1
    skin_failure: str = "von_mises"          # "von_mises" (default) or "tsai_wu"
    tsai_wu_safety_factor: float = 2.0
```

3c. Add to `LaminateSizingResult` (after `max_panel_buckling_util`):
```python
    max_panel_buckling_util: float
    min_skin_strength_ratio: float | None
```

3d. In `evaluate`, initialise a worst-ratio accumulator and compute it per load case.
Find this block:
```python
        wb = np.zeros(n)
        ws = 0.0
        md = 0.0
        mt = 0.0
        worst_beam_buck = 0.0
        worst_panel_buck = 0.0
```
and append one line:
```python
        worst_R = np.inf
```
Then, INSIDE the `for loads in load_arrays:` loop, AFTER the existing
`ws = max(ws, ...)` line, add:
```python
            if config.skin_failure == "tsai_wu":
                eps = recover_membrane_strain(model.nodes, model.shell_tris, res.displacements)
                offs = datum_offsets_deg if datum_offsets_deg is not None else np.zeros(eps.shape[0])
                for e in range(eps.shape[0]):
                    R_e = laminate_min_strength_ratio(
                        ply, eps[e], f0=f0, f45=f45, f90=f90, offset_deg=float(offs[e]))
                    if R_e < worst_R:
                        worst_R = R_e
```
Then change the return tuple from:
```python
        out = (wb, ws, md, mt, worst_beam_buck, worst_panel_buck)
```
to:
```python
        out = (wb, ws, md, mt, worst_beam_buck, worst_panel_buck, worst_R)
```

3e. Update the constraint closures. Replace the existing `skin_con` and the two
buckling closures, and adjust the other closures' unpacking to the 7-tuple. Concretely:

`beam_con`, `defl_con`, `twist_con` currently unpack 6 values with `_` placeholders —
change each to index explicitly to avoid tuple-length coupling:
```python
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
```
And the buckling closures (inside the `if config.buckling_safety_factor is not None:`
block) become explicit-index:
```python
        def beam_buck_con(x):
            return np.array([1.0 - evaluate(x)[4]])
        def panel_buck_con(x):
            return np.array([1.0 - evaluate(x)[5]])
```
Leave `frac_con` unchanged (it does not call `evaluate`).

3f. At the end, the final unpack `wb, ws, d, t, bbu, pbu = evaluate(x)` must take the
7th value:
```python
    wb, ws, d, t, bbu, pbu, worst_R = evaluate(x)
```
and the result construction gains the new field:
```python
        max_beam_buckling_util=float(bbu),
        max_panel_buckling_util=float(pbu),
        min_skin_strength_ratio=(float(worst_R) if config.skin_failure == "tsai_wu" else None),
    )
```

- [ ] **Step 4: Run the new sizer tests**

Run: `uv run pytest tests/beams/test_tsai_wu_sizing.py -v`
Expected: PASS (strain + von_mises + tsai_wu tests).

- [ ] **Step 5: FULL suite (backward-compat gate) + construction-site check**

Run: `uv run pytest -q`  → ALL pass (the prior 92 + new tests). The default
`skin_failure="von_mises"` path is unchanged except the additive
`min_skin_strength_ratio=None` field.
Run: `grep -rn "LaminateSizingResult(" src/ examples/ tests/`  → only the one site in
`laminate_sizing.py`. If any other positional construction exists, STOP and report.

- [ ] **Step 6: Self-review then commit**

Re-read the diff: confirm the 7-tuple is consumed correctly everywhere (no leftover
6-element unpack or negative indexing into `evaluate`), and the von-Mises path is
unchanged. Then:
```bash
git add src/wing_design/beams/laminate_sizing.py tests/beams/test_tsai_wu_sizing.py
git commit -m "$(cat <<'EOF'
Tsai-Wu: opt-in per-ply skin failure constraint in the CLT sizer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: von-Mises-vs-Tsai-Wu comparison example

**Files:**
- Create: `examples/34_tsai_wu_skin.py`

**Context:** Mirror `examples/32_final_design.py`; size with `skin_failure="von_mises"`
and `"tsai_wu"` (SF 2.0) on a uniform skin, span datum + buckling, and compare. Examples
are run manually (not pytest).

- [ ] **Step 1: Create `examples/34_tsai_wu_skin.py`**

```python
"""Per-ply Tsai-Wu vs laminate-average von-Mises skin failure.

Sizes the span-datum CLT design (as in examples/32, uniform skin) twice under identical
constraints: the default von-Mises skin proxy vs a per-ply Tsai-Wu criterion (material
SF 2.0). Prints mass, layup, and the governing skin margin for each. FEA-only.
"""
from __future__ import annotations

from wing_design import default_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell_laminate,
)
from wing_design.materials.unidir import T700_EPOXY


def main() -> None:
    P = default_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8

    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    model = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    load_arrays = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
                   for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    defl_lim, twist_lim, sf_buck = 0.02 * spec.span, 5.0, 1.5
    sf_mat = P.material_iso.sigma_allow_safety_factor

    def cfg(mode):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim,
            ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=sf_buck,
            skin_failure=mode, tsai_wu_safety_factor=sf_mat)

    print("sizing with von-Mises skin proxy...")
    vm = size_beam_shell_laminate(model, load_arrays, cfg("von_mises"), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=200)
    print("sizing with per-ply Tsai-Wu...")
    tw = size_beam_shell_laminate(model, load_arrays, cfg("tsai_wu"), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300)

    def show(tag, r):
        print(f"\n  [{tag}] converged={r.converged} iters={r.n_iter}")
        print(f"    TOTAL MASS = {r.mass_kg:6.2f} kg  (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f})")
        print(f"    skin t {r.t_skin*1e3:.2f} mm | layup {r.f0*100:.0f}/{r.f45*100:.0f}/{r.f90*100:.0f} span/±45/chord")
        msr = "-" if r.min_skin_strength_ratio is None else f"{r.min_skin_strength_ratio:.2f}"
        print(f"    skin: vM sigma {r.max_skin_vm_Pa/1e6:.0f} MPa (allow {P.sigma_allow_Pa/1e6:.0f}) | Tsai-Wu min R {msr} (SF {sf_mat:.1f})")
        print(f"    tip defl {r.tip_defl_m*1e3:.1f}/{defl_lim*1e3:.0f} mm | twist {r.tip_twist_deg:.2f}/{twist_lim:.0f} deg")
        print(f"    buckling util: beam {r.max_beam_buckling_util:.2f} | panel {r.max_panel_buckling_util:.2f}")

    show("von_mises", vm)
    show("tsai_wu", tw)

    dmass = tw.mass_kg - vm.mass_kg
    print(f"\n  Δmass {dmass:+.2f} kg ({100*dmass/vm.mass_kg:+.1f}%)")
    print("  Verdict: per-ply Tsai-Wu " +
          ("lightens" if dmass < -0.05 else "adds" if dmass > 0.05 else "≈ no change to") +
          " the design; compare which skin mode governs (layup shift, R vs vM margin).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example end-to-end**

Run: `uv run python examples/34_tsai_wu_skin.py`
Expected: completes (a few minutes — two full sizings), prints both blocks and the Δmass
+ verdict. Non-convergence is still a completing run — report it, don't treat as a crash.
Note: the Tsai-Wu inner per-triangle loop makes its `evaluate` slower than von-Mises;
the higher `maxiter` is intentional.

- [ ] **Step 3: CAPTURE THE FULL OUTPUT** verbatim into the report (Task 6 needs the numbers).

- [ ] **Step 4: Commit**

```bash
git add examples/34_tsai_wu_skin.py
git commit -m "$(cat <<'EOF'
Tsai-Wu: von-Mises-vs-Tsai-Wu comparison example

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Record the finding in the living plan

**Files:**
- Modify: `docs/plan.md`

**Context:** Add a status note after the "Per-band skin thickness done" note (before
`### Phase F`) and a Decisions-log row after the "Per-band skin thickness" row. Use the
ACTUAL numbers from the Task 5 run — do not invent.

- [ ] **Step 1: Add the status note** (after the per-band-skin paragraph):

```markdown
**Per-ply Tsai-Wu skin failure done (2026-06-08).** Opt-in `skin_failure="tsai_wu"`
(default `"von_mises"`; `materials.failure`) replaces the laminate-average von-Mises
skin proxy with a per-ply Tsai-Wu strength-ratio check (R ≥ SF, F12 = -0.5·√(F11·F22),
material SF 2.0) over the present ply orientations, using `recover_membrane_strain` for
the per-triangle strain. `examples/34_tsai_wu_skin.py` compares the two criteria at
equal constraints (span datum + buckling SF 1.5, uniform skin): von-Mises [Mvm] kg vs
Tsai-Wu [Mtw] kg ([±X]%); layup [vm layup] → [tw layup]; governing skin margin
[vM sigma] / Tsai-Wu min R [R]. [One-sentence verdict: whether the per-ply criterion
changes the design and which skin mode governs.]
```

- [ ] **Step 2: Add the Decisions-log row** (after the "Per-band skin thickness" row):

```markdown
| Per-ply Tsai-Wu skin failure | Opt-in `skin_failure="tsai_wu"` (default von-Mises unchanged; `materials.failure`) — per-ply Tsai-Wu strength ratio R≥SF (F12=-0.5√(F11 F22), material SF 2.0) over present orientations, from per-triangle membrane strain (`recover_membrane_strain`). vs von-Mises proxy at equal constraints: [Mvm]→[Mtw] kg ([±X]%), layup [shift], governing margin [...]. [Verdict]. First-ply only; smeared laminate (no stacking/delamination). |
```

- [ ] **Step 3: Commit**

```bash
git add docs/plan.md
git commit -m "$(cat <<'EOF'
Tsai-Wu: record finding in plan

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification (after all tasks)

- [ ] `uv run pytest -q` — expect prior 92 + new `test_failure.py` and
  `test_tsai_wu_sizing.py`, all green.
- [ ] `grep -rn "LaminateSizingResult(" src/ examples/ tests/` shows only the one site.
- [ ] Spot-confirm `examples/32` still runs (reads named attributes only).
- [ ] Dispatch a final code review over the whole diff, then use
  superpowers:finishing-a-development-branch.

## Notes / risks for the implementer

- **The 7-tuple change is the sharp edge.** The buckling closures use negative indexing
  today (`*_, bbu, _` and `*_, pbu`); after appending `worst_R` they MUST switch to
  explicit positive indices (`evaluate(x)[4]`, `[5]`) or they will read the wrong value.
  The full suite + the tsai_wu feasibility test are the gate.
- **Backward compatibility:** `skin_failure="von_mises"` must reproduce today's design;
  only the additive `min_skin_strength_ratio=None` field differs.
- **Strength-ratio R is the margin** (R≥SF feasible); it is NOT a utilization. Do not
  invert the constraint sense.
- **Present orientations only:** absent plies (fraction 0) must not constrain — already
  handled in `laminate_min_strength_ratio`.
- **Performance:** the per-triangle Python loop in `evaluate` (tsai_wu mode) adds cost
  comparable to the datum laminate construction; acceptable, vectorize later if needed.
