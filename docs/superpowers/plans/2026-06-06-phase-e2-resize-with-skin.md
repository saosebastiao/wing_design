# Phase E.2 — Re-size Beams + Skin Together Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** With the load-bearing skin now in the model (E.1), co-size the per-element beam radii **and** the skin thickness to minimize total (beam + skin) mass, subject to per-beam-element von Mises, per-skin-triangle von Mises, tip-deflection, and tip-twist constraints — re-solving the combined beam+shell FEA each iteration. Quantify the mass versus Phase-C's beams-only (deflection-bound ~44 kg) result: this is where the shell-beam thesis's mass payoff is realized.

**Scope note:** Phase E increment 2 of 4. Co-sizes beam radii + a **single uniform skin thickness** and constrains skin stress (the user-chosen scope). **Deferred:** per-spanwise-band skin thickness, MILP discrete stock catalog (E.3), buckling+twist eigenvalue constraints beyond the current twist limit, CLT anisotropic skin (E.4). Skin remains isotropic-equivalent.

**Architecture:** A reusable skin membrane-stress recovery (`recover_membrane_stress` + `membrane_von_mises`) is factored out of `shell.py`'s solver (same CST Bm / plane-stress Dm convention). A new `beams/shell_sizing.py` runs an SLSQP NLP whose design vector is `[beam_radii (n_long), t_skin]`, objective is total mass (analytic gradient), and constraints (normalized to O(1), as in Phase C) are beam vm ≤ allow, skin vm ≤ allow, tip deflection ≤ limit, tip twist ≤ limit — each evaluated by re-solving `structural.solve_beam_shell` and recovering both beam and skin stress (memoized per design point).

**Tech Stack:** numpy, scipy.optimize SLSQP, `structural.solve_beam_shell` + `structural.frame.von_mises_per_element` + the new `recover_membrane_stress`, `beams.shell_model`.

---

## Background facts (verified)

- `shell.py` membrane stress recovery (lines ~545–575): per triangle, `R,local,area = _triangle_local_frame(p1,p2,p3)`; build CST `Bm (3,6)` from local coords; `Dm = E/(1-nu²)·[[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]`; rotate global translations to local (`R.T @ u_trans`), take `(u,v)` of each node → `u_membrane (6,)`; **membrane stress (Pa) = `Dm @ Bm @ u_membrane`** (thickness-independent — the `*t` then `/t` in the source cancels). `_triangle_local_frame` raises on degenerate (collinear) triangles.
- `ShellFEAResult.membrane_von_mises`: `sqrt(sxx² - sxx·syy + syy² + 3·sxy²)`.
- `structural.solve_beam_shell(nodes, beam_elements, beam_sections, shell_tris, *, E_beam, G_beam, E_skin, nu_skin, t_skin, fixed_nodes, loads, drilling_factor=1e-4) -> FrameResult` (FrameResult.displacements (N,6), axial_force/bending_moment/torsion over the beam elements).
- `structural.frame.von_mises_per_element(result, sections) -> (n_beam,)`; `BeamSection.circular(r)`.
- `beams.shell_model.BeamShellModel` fields: nodes, beam_elements (longitudinal only), shell_tris, n_beams, n_levels, fixed_nodes, tip_nodes, section, E_beam, G_beam, E_skin, nu_skin, t_skin; `build_beam_shell_model(...)`, `solve_beam_shell_model`.
- `beams.fea_model.project_panels_to_beam_nodes(frame, panels, *, safety_factor)`; tip metrics: defl = max translation norm over `tip_nodes`; twist(deg) = degrees(max |disp[:,5]|).
- Phase-C reference: `size_beams` minimized frame mass (beams + fixed rings), deflection-bound ~44 kg at n_levels=8; SLSQP needed objective/constraint normalization to O(1).
- Scenario: `P.sigma_allow_Pa`, `P.rho_kgm3`, `P.E_iso_Pa`, `P.nu_iso`, `P.skin_sizing.t_baseline_m`, etc.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/structural/shell.py` (modify) | Add `recover_membrane_stress(nodes, triangles, displacements, *, E, nu) -> (M,3)` and `membrane_von_mises(stress) -> (M,)`. |
| `src/wing_design/structural/__init__.py` (modify) | Export both. |
| `src/wing_design/beams/shell_sizing.py` (create) | `BeamShellSizingConfig`, `BeamShellSizingResult`, mass/length/area helpers, `size_beam_shell` (SLSQP). |
| `src/wing_design/beams/__init__.py` (modify) | Export the new sizing API. |
| `examples/25_resize_with_skin.py` (create) | Co-size beams+skin under the envelope; compare mass/stiffness to Phase-C beams-only. |
| `tests/structural/test_membrane_stress.py` (create) | Uniaxial-stretch analytic check of `recover_membrane_stress`/`membrane_von_mises`. |
| `tests/beams/test_shell_sizing.py` (create) | Mass helpers; `size_beam_shell` reduces mass + satisfies constraints on a tiny model. |
| `docs/plan.md` (modify) | Record E.2 done + the mass payoff. |

---

## Task 1: Skin membrane-stress recovery

**Files:**
- Modify: `src/wing_design/structural/shell.py`
- Modify: `src/wing_design/structural/__init__.py`
- Test: `tests/structural/test_membrane_stress.py`

- [ ] **Step 1: Write the failing test** (`tests/structural/test_membrane_stress.py`)

```python
import numpy as np

from wing_design.structural.shell import recover_membrane_stress, membrane_von_mises

E, NU = 70e9, 0.3


def test_uniaxial_stretch_recovers_known_stress():
    # One triangle in the XY plane (local frame == global). Impose a uniform
    # x-strain eps via nodal displacements u = eps*x, v = 0. Plane-stress CST then
    # gives sigma_xx = E/(1-nu^2)*eps, sigma_yy = nu*sigma_xx, sigma_xy = 0.
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    tris = np.array([[0, 1, 2]], dtype=int)
    eps = 1e-3
    disp = np.zeros((3, 6))
    disp[:, 0] = eps * nodes[:, 0]      # u = eps*x
    s = recover_membrane_stress(nodes, tris, disp, E=E, nu=NU)
    sxx_expected = E / (1 - NU**2) * eps
    assert s.shape == (1, 3)
    assert abs(s[0, 0] - sxx_expected) / sxx_expected < 1e-9
    assert abs(s[0, 1] - NU * sxx_expected) / (NU * sxx_expected) < 1e-9
    assert abs(s[0, 2]) < 1e-3          # ~0 shear


def test_membrane_von_mises_matches_formula():
    stress = np.array([[100.0, 40.0, 30.0]])
    vm = membrane_von_mises(stress)
    expected = np.sqrt(100**2 - 100 * 40 + 40**2 + 3 * 30**2)
    assert abs(vm[0] - expected) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/structural/test_membrane_stress.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement** — add to `src/wing_design/structural/shell.py` (after `solve_shell_elastic`; reuse `_triangle_local_frame` already in the module):

```python
def recover_membrane_stress(
    nodes: np.ndarray,
    triangles: np.ndarray,
    displacements: np.ndarray,
    *,
    E: float,
    nu: float,
) -> np.ndarray:
    """Per-triangle CST membrane stress (M, 3) = [σxx, σyy, σxy] in element-local frame.

    Stress in Pa, independent of thickness (membrane stress is D·ε). `displacements`
    is the global (N, 6) field from any solver that shares the 6-DOF node layout
    (e.g. `solve_beam_shell`). Mirrors the recovery in `solve_shell_elastic`.
    """
    M = triangles.shape[0]
    out = np.zeros((M, 3))
    Dm = (E / (1.0 - nu * nu)) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
    )
    for e in range(M):
        i, j, l = (int(v) for v in triangles[e])
        R, local, area = _triangle_local_frame(nodes[i], nodes[j], nodes[l])
        x1, y1 = local[0]; x2, y2 = local[1]; x3, y3 = local[2]
        b_grad = np.array([y2 - y3, y3 - y1, y1 - y2])
        c_grad = np.array([x3 - x2, x1 - x3, x2 - x1])
        two_A = 2.0 * area
        Bm = np.zeros((3, 6))
        for k in range(3):
            Bm[0, 2 * k + 0] = b_grad[k] / two_A
            Bm[1, 2 * k + 1] = c_grad[k] / two_A
            Bm[2, 2 * k + 0] = c_grad[k] / two_A
            Bm[2, 2 * k + 1] = b_grad[k] / two_A
        u_membrane = np.zeros(6)
        for n_idx, gn in enumerate((i, j, l)):
            u_local = R.T @ displacements[gn, 0:3]
            u_membrane[2 * n_idx + 0] = u_local[0]
            u_membrane[2 * n_idx + 1] = u_local[1]
        out[e] = Dm @ Bm @ u_membrane
    return out


def membrane_von_mises(stress: np.ndarray) -> np.ndarray:
    """Per-row plane-stress von Mises from (M, 3) [σxx, σyy, σxy] arrays."""
    sxx = stress[:, 0]
    syy = stress[:, 1]
    sxy = stress[:, 2]
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
```

- [ ] **Step 4:** Export both from `structural/__init__.py` (add to the `from .shell import (...)` block and `__all__`).

- [ ] **Step 5: Run** — `uv run pytest tests/structural/test_membrane_stress.py -v` → 2 PASS; `uv run pytest` → green.

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/structural/shell.py src/wing_design/structural/__init__.py tests/structural/test_membrane_stress.py
git commit -m "feat(structural): reusable skin membrane-stress recovery + von Mises"
# trailer
```

---

## Task 2: Mass/length/area helpers for the beam-shell model

**Files:**
- Create: `src/wing_design/beams/shell_sizing.py`
- Test: `tests/beams/test_shell_sizing.py`

- [ ] **Step 1: Write the failing test** (`tests/beams/test_shell_sizing.py`)

```python
import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_lengths, skin_areas, beam_mass, skin_mass


def test_beam_lengths_match_geometry():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    L = beam_lengths(model)
    assert L.shape[0] == model.beam_elements.shape[0]
    i, j = model.beam_elements[0]
    assert abs(L[0] - np.linalg.norm(model.nodes[j] - model.nodes[i])) < 1e-12


def test_beam_mass_uniform():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    L = beam_lengths(model)
    r, rho = 0.02, 1550.0
    m = beam_mass(model, np.full(model.beam_elements.shape[0], r), rho=rho)
    assert abs(m - rho * np.pi * r**2 * L.sum()) / m < 1e-12


def test_skin_mass_scales_with_thickness():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    A = skin_areas(model).sum()
    rho = 1550.0
    assert abs(skin_mass(model, 0.003, rho=rho) - rho * 0.003 * A) / (rho * 0.003 * A) < 1e-12
    assert abs(skin_mass(model, 0.006, rho=rho) - 2 * skin_mass(model, 0.003, rho=rho)) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_shell_sizing.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/wing_design/beams/shell_sizing.py`** (helpers + config/result; the optimizer is added in Task 3):

```python
"""Co-size form-beam radii and skin thickness against the combined beam+shell FEA.

Design variables: per-element beam radii + a single uniform skin thickness.
Objective: total (beam + skin) mass. Constraints: per-beam-element von Mises,
per-skin-triangle von Mises, tip deflection, and tip twist — re-solving
`structural.solve_beam_shell` each iteration. SLSQP with O(1)-normalized
objective/constraints (as in Phase C).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..structural.beam_shell import solve_beam_shell
from ..structural.frame import BeamSection, von_mises_per_element
from ..structural.shell import _triangle_local_frame, membrane_von_mises, recover_membrane_stress
from .shell_model import BeamShellModel


@dataclass(frozen=True)
class BeamShellSizingConfig:
    sigma_allow_Pa: float
    tip_defl_max_m: float
    tip_twist_max_deg: float
    r_min: float = 0.004
    r_max: float = 0.04
    t_min: float = 0.0005
    t_max: float = 0.02


@dataclass(frozen=True)
class BeamShellSizingResult:
    radii: np.ndarray            # (n_beam_elem,)
    t_skin: float
    mass_kg: float               # total beam + skin
    beam_mass_kg: float
    skin_mass_kg: float
    converged: bool
    n_iter: int
    max_beam_vm_Pa: float
    max_skin_vm_Pa: float
    tip_defl_m: float
    tip_twist_deg: float


def beam_lengths(model: BeamShellModel) -> np.ndarray:
    """(n_beam_elem,) Euclidean length of each longitudinal beam element."""
    i = model.beam_elements[:, 0]
    j = model.beam_elements[:, 1]
    return np.linalg.norm(model.nodes[j] - model.nodes[i], axis=1)


def skin_areas(model: BeamShellModel) -> np.ndarray:
    """(n_tris,) area of each skin triangle."""
    out = np.empty(model.shell_tris.shape[0])
    for e in range(model.shell_tris.shape[0]):
        a, b, c = (int(v) for v in model.shell_tris[e])
        _, _, area = _triangle_local_frame(model.nodes[a], model.nodes[b], model.nodes[c])
        out[e] = area
    return out


def beam_mass(model: BeamShellModel, radii: np.ndarray, *, rho: float) -> float:
    """Total longitudinal-beam mass [kg]."""
    return float(rho * np.sum(np.pi * np.asarray(radii) ** 2 * beam_lengths(model)))


def skin_mass(model: BeamShellModel, t_skin: float, *, rho: float) -> float:
    """Total skin mass [kg] = rho * t * sum(triangle areas)."""
    return float(rho * t_skin * skin_areas(model).sum())
```

- [ ] **Step 4: Run** — `uv run pytest tests/beams/test_shell_sizing.py -v` → 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/shell_sizing.py tests/beams/test_shell_sizing.py
git commit -m "feat(beams): beam-shell sizing config + mass/length/area helpers"
# trailer
```

---

## Task 3: `size_beam_shell` co-sizing optimizer

**Files:**
- Modify: `src/wing_design/beams/shell_sizing.py`
- Test: `tests/beams/test_shell_sizing.py`

- [ ] **Step 1: Write the failing test** (append to `tests/beams/test_shell_sizing.py`)

```python
from wing_design.beams.shell_sizing import BeamShellSizingConfig, size_beam_shell


def test_size_beam_shell_reduces_mass_and_feasible():
    # Tiny model, gentle tip load, generous limits → optimum drives radii and skin
    # thickness to their lower bounds; mass must fall below the all-max start.
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    n = model.beam_elements.shape[0]
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 10.0
    cfg = BeamShellSizingConfig(
        sigma_allow_Pa=1.0e8, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.03, t_min=0.0005, t_max=0.01,
    )
    rho = 1550.0
    start = beam_mass(model, np.full(n, cfg.r_max), rho=rho) + skin_mass(model, cfg.t_max, rho=rho)
    res = size_beam_shell(model, [loads], cfg, rho=rho, maxiter=40)
    assert res.radii.shape == (n,)
    assert cfg.t_min - 1e-9 <= res.t_skin <= cfg.t_max + 1e-9
    assert res.mass_kg < start
    assert res.max_beam_vm_Pa <= cfg.sigma_allow_Pa * 1.05
    assert res.max_skin_vm_Pa <= cfg.sigma_allow_Pa * 1.05
    assert np.all(res.radii >= cfg.r_min - 1e-9) and np.all(res.radii <= cfg.r_max + 1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_shell_sizing.py::test_size_beam_shell_reduces_mass_and_feasible -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement** — append to `src/wing_design/beams/shell_sizing.py`:

```python
def size_beam_shell(
    model: BeamShellModel,
    load_arrays: list[np.ndarray],
    config: BeamShellSizingConfig,
    *,
    rho: float,
    maxiter: int = 60,
    ftol: float = 1.0e-4,
) -> BeamShellSizingResult:
    """Co-size beam radii + uniform skin thickness to minimize beam+skin mass.

    Design vector x = [radii (n_beam_elem), t_skin]. Constraints (all >= 0 after
    normalization): beam vm, skin vm, tip deflection, tip twist. `load_arrays` is
    one (n_nodes, 6) array per (already safety-factored) load case.
    """
    if not load_arrays:
        raise ValueError("load_arrays is empty: no loads to size against")
    n = model.beam_elements.shape[0]
    Lb = beam_lengths(model)
    Atri = skin_areas(model)
    A_skin = float(Atri.sum())

    x0 = np.concatenate([np.full(n, config.r_max), [config.t_max]])
    cache: dict[bytes, tuple[np.ndarray, float, float, float]] = {}

    def evaluate(x):
        key = np.asarray(x, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        radii = x[:n]
        t_skin = float(x[n])
        sections = [BeamSection.circular(float(r)) for r in radii]
        worst_beam = np.zeros(n)
        worst_skin = 0.0
        max_defl = 0.0
        max_twist = 0.0
        for loads in load_arrays:
            res = solve_beam_shell(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam,
                E_skin=model.E_skin, nu_skin=model.nu_skin, t_skin=t_skin,
                fixed_nodes=model.fixed_nodes, loads=loads,
            )
            worst_beam = np.maximum(worst_beam, von_mises_per_element(res, sections))
            skin_s = recover_membrane_stress(model.nodes, model.shell_tris, res.displacements,
                                             E=model.E_skin, nu=model.nu_skin)
            worst_skin = max(worst_skin, float(membrane_von_mises(skin_s).max()))
            tip = model.tip_nodes
            max_defl = max(max_defl, float(np.linalg.norm(res.displacements[tip, :3], axis=1).max()))
            max_twist = max(max_twist, float(np.degrees(np.abs(res.displacements[tip, 5]).max())))
        out = (worst_beam, worst_skin, max_defl, max_twist)
        cache[key] = out
        return out

    m_ref = (beam_mass(model, np.full(n, config.r_max), rho=rho)
             + skin_mass(model, config.t_max, rho=rho))

    def mass(x):
        return (rho * np.sum(np.pi * x[:n] ** 2 * Lb) + rho * x[n] * A_skin) / m_ref

    def mass_grad(x):
        g = np.empty(n + 1)
        g[:n] = rho * 2.0 * np.pi * x[:n] * Lb / m_ref
        g[n] = rho * A_skin / m_ref
        return g

    def beam_con(x):
        wb, _, _, _ = evaluate(x)
        return 1.0 - wb / config.sigma_allow_Pa

    def skin_con(x):
        _, ws, _, _ = evaluate(x)
        return np.array([1.0 - ws / config.sigma_allow_Pa])

    def defl_con(x):
        _, _, d, _ = evaluate(x)
        return np.array([1.0 - d / config.tip_defl_max_m])

    def twist_con(x):
        _, _, _, t = evaluate(x)
        return np.array([1.0 - t / config.tip_twist_max_deg])

    bounds = [(config.r_min, config.r_max)] * n + [(config.t_min, config.t_max)]
    res = minimize(
        mass, x0, jac=mass_grad, method="SLSQP", bounds=bounds,
        constraints=[
            {"type": "ineq", "fun": beam_con},
            {"type": "ineq", "fun": skin_con},
            {"type": "ineq", "fun": defl_con},
            {"type": "ineq", "fun": twist_con},
        ],
        options={"maxiter": maxiter, "ftol": ftol},
    )

    x = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
    wb, ws, d, t = evaluate(x)
    radii = x[:n]
    t_skin = float(x[n])
    bm = beam_mass(model, radii, rho=rho)
    sm = skin_mass(model, t_skin, rho=rho)
    return BeamShellSizingResult(
        radii=radii, t_skin=t_skin, mass_kg=bm + sm, beam_mass_kg=bm, skin_mass_kg=sm,
        converged=bool(res.success), n_iter=int(res.nit),
        max_beam_vm_Pa=float(wb.max()), max_skin_vm_Pa=float(ws),
        tip_defl_m=float(d), tip_twist_deg=float(t),
    )
```

- [ ] **Step 4: Run** — `uv run pytest tests/beams/test_shell_sizing.py -v` → all PASS (tiny problem, a few seconds). Then `uv run pytest` (full suite). **If the optimizer returns infeasible (max beam/skin vm ≫ allow, or deflection/twist over limit) or fails to reduce mass, STOP and report** with the SLSQP result — do not weaken the test.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/shell_sizing.py tests/beams/test_shell_sizing.py
git commit -m "feat(beams): SLSQP co-sizing of beam radii + skin thickness"
# trailer
```

---

## Task 4: Export + deliverable example + plan update

**Files:**
- Modify: `src/wing_design/beams/__init__.py`
- Create: `examples/25_resize_with_skin.py`
- Modify: `docs/plan.md`

- [ ] **Step 1: Export** — in `src/wing_design/beams/__init__.py` add `from .shell_sizing import BeamShellSizingConfig, BeamShellSizingResult, beam_mass, size_beam_shell, skin_mass` and add those names to `__all__`.

- [ ] **Step 2: Create `examples/25_resize_with_skin.py`**

```python
"""Phase-E.2 spike: co-size beam radii + skin thickness with the skin load-bearing.

Re-sizes the beam-shell structure (skin replaces rings) minimizing total beam+skin
mass under beam-stress, skin-stress, tip-deflection and tip-twist constraints, then
compares against Phase-C's beams-only sizing (deflection-bound) to show where the
mass payoff of the load-bearing skin lands.
"""
from __future__ import annotations

import numpy as np

from wing_design import default_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    BeamShellSizingConfig,
    SizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell,
    size_beams,
)


def main() -> None:
    P = default_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8

    frame = build_beam_frame(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
    )
    model = build_beam_shell_model(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m,
    )

    airplane = build_airplane(spec)
    envelope = sweep_envelope(
        airplane, P.load_cases, method="lifting_line",
        spanwise_resolution=P.aero.spanwise_resolution,
    )
    load_arrays = [
        project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
        for ar in envelope
        if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0
    ]

    defl_lim = 0.02 * spec.span
    print("Phase-C reference: beams-only sizing (skin absent, rings present)...")
    c_cfg = SizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=5.0)
    c = size_beams(frame, load_arrays, c_cfg, rho=P.rho_kgm3, maxiter=60)
    print(f"  beams-only mass = {c.mass_kg:6.2f} kg (tip defl {c.tip_defl_m*1e3:.0f} mm, "
          f"max σ {c.max_vm_stress_Pa/1e6:.0f} MPa)")

    print("\nPhase-E.2: co-size beams + skin (skin load-bearing)...")
    e_cfg = BeamShellSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=5.0)
    e = size_beam_shell(model, load_arrays, e_cfg, rho=P.rho_kgm3, maxiter=60)
    print(f"  converged={e.converged}  iters={e.n_iter}")
    print(f"  total mass      = {e.mass_kg:6.2f} kg   (beams {e.beam_mass_kg:.2f} + skin {e.skin_mass_kg:.2f})")
    print(f"  skin thickness  = {e.t_skin*1e3:6.2f} mm   (bounds {e_cfg.t_min*1e3:.1f}-{e_cfg.t_max*1e3:.0f})")
    print(f"  beam radii      = {e.radii.min()*1e3:.1f}-{e.radii.max()*1e3:.1f} mm")
    print(f"  max beam σ      = {e.max_beam_vm_Pa/1e6:6.0f} MPa   max skin σ = {e.max_skin_vm_Pa/1e6:.0f} MPa "
          f"(allow {e_cfg.sigma_allow_Pa/1e6:.0f})")
    print(f"  tip deflection  = {e.tip_defl_m*1e3:6.1f} mm   tip twist = {e.tip_twist_deg:.2f} deg "
          f"(limits {defl_lim*1e3:.0f} mm / 5 deg)")

    if e.mass_kg < c.mass_kg:
        print(f"\n  → load-bearing skin cuts structural mass {c.mass_kg:.1f} → {e.mass_kg:.1f} kg "
              f"({100*(1-e.mass_kg/c.mass_kg):.0f}% lighter).")
    else:
        print(f"\n  → co-sized mass {e.mass_kg:.1f} kg vs beams-only {c.mass_kg:.1f} kg.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the example** — `uv run python examples/25_resize_with_skin.py` (two SLSQP sizings; ~2–4 min total). **Paste the full output.** Confirm: both sizings converge/feasible; report the E.2 total mass, beam vs skin split, chosen skin thickness, beam+skin stresses (≤ allow), deflection/twist (≤ limits), and the mass comparison to Phase-C beams-only. A feasible mass reduction is the expected payoff; if E.2 is *not* lighter, report it honestly (do not fake it).

- [ ] **Step 4: Update `docs/plan.md`** — under the Phase-E status, append an E.2 note with the ACTUAL numbers from Step 3: co-sized total mass + beam/skin split + chosen skin thickness vs Phase-C beams-only mass; constraints satisfied; the finding (skin lets the deflection-bound beams shrink → structural mass drops). Add a Decisions-log row:

```markdown
| Phase-E.2 co-sizing | SLSQP co-sizes per-element beam radii + a uniform skin thickness minimizing total beam+skin mass under beam-vm, skin-vm, tip-deflection and tip-twist constraints (skin membrane stress recovered each solve). Per-band skin / MILP / buckling / CLT deferred. |
```

- [ ] **Step 5: Full suite** — `uv run pytest` → green.

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/__init__.py examples/25_resize_with_skin.py docs/plan.md
git commit -m "feat(beams): Phase-E.2 co-size beams+skin example; quantify mass payoff"
# trailer
```

---

## Self-Review

- **Spec coverage:** co-size skin thickness → `size_beam_shell` design vector `[radii, t_skin]` + skin-mass objective term (Task 3); constrain skin stress → `recover_membrane_stress`/`membrane_von_mises` (Task 1) + `skin_con` (Task 3); mass payoff vs Phase C → example (Task 4).
- **Placeholder scan:** none.
- **Type consistency:** `recover_membrane_stress(nodes, triangles, displacements, *, E, nu)` consumed in `size_beam_shell` with `model.E_skin/nu_skin`; `solve_beam_shell` called with the trial `t_skin`; `von_mises_per_element(res, sections)` over the beam elements; `BeamShellModel` fields used as defined; `beam_mass`/`skin_mass` shared between helpers, optimizer, and example.
- **Risk:** the membrane recovery must match `solve_shell_elastic`'s convention exactly — Task 1's uniaxial analytic test guards it. SLSQP conditioning is handled by the same O(1) normalization that fixed Phase C. Cost scales with (n_long+1) FD evaluations × cases × beam+shell solves; the example uses n_levels=8 to stay tractable (~minutes). Degenerate skin triangles raise from `_triangle_local_frame` — surfaces rather than silently corrupts.
- **Known limitations (intentional):** single uniform skin thickness (per-band deferred); isotropic-equivalent skin (CLT in E.4); skin uses the same density as the beams; bending stress in the skin (DKT) is not added to the skin-vm constraint (membrane only) — reasonable for a thin stressed skin, noted for later.
