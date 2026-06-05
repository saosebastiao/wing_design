# Phase C — FEA-in-the-Loop Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Size every longitudinal form-beam segment's circular cross-section by minimizing total beam mass subject to per-element stress, tip-deflection, and tip-twist constraints, with the Phase-B frame FEA re-solved inside the optimizer, and compare the sized mass against the Phase-3 tube-spar baseline.

**Architecture:** A continuous nonlinear program solved with `scipy.optimize.minimize(method="SLSQP")`. Design variables are the radii of the longitudinal beam elements (rings stay at a fixed minimum radius). The objective (total beam mass) is analytic with an exact gradient; the constraints (von Mises ≤ allowable per longitudinal element, max-over-cases tip deflection ≤ limit, max-over-cases tip twist ≤ limit) require a frame re-solve, memoized per design point. Aero nodal loads are projected once up front (load is independent of cross-section). A new generic `von_mises_per_element` lives in `structural/frame.py`; the sizing loop and config live in `beams/sizing.py`.

**Tech Stack:** numpy, scipy.optimize (SLSQP), the Phase-B `structural.frame` solver + `beams.fea_model`, `structural.beam.size_tube_spar` (baseline), scenario-driven example.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/structural/frame.py` (modify) | Add `von_mises_per_element(result, sections)` — per-element combined-stress proxy from a `FrameResult` + per-element sections. Generic structural primitive. |
| `src/wing_design/structural/__init__.py` (modify) | Export `von_mises_per_element`. |
| `src/wing_design/beams/sizing.py` (create) | `SizingConfig`, `SizingResult`, `element_lengths`, `n_longitudinal`, `build_sections`, `frame_mass`, `size_beams` (the SLSQP loop). |
| `src/wing_design/beams/__init__.py` (modify) | Export `SizingConfig`, `SizingResult`, `size_beams`, `frame_mass`. |
| `examples/22_sized_beams.py` (create) | Build frame → project envelope loads → `size_beams` → print sized metrics + per-beam radius profile → compare to `size_tube_spar` baseline. |
| `tests/structural/test_frame.py` (modify) | Add analytic `von_mises_per_element` tests (pure axial, pure bending). |
| `tests/beams/test_sizing.py` (create) | `frame_mass` correctness; `size_beams` reduces mass + satisfies stress constraint on a tiny problem. |
| `docs/plan.md` (modify) | Mark Phase C done; record the sizing-method decision + the baseline-comparability caveat. |

---

## Background facts the implementer must rely on (already in the codebase)

- `structural.frame.solve_frame(nodes, elements, sections, *, E, G, fixed_nodes, loads) -> FrameResult`. `sections` is a **list of `BeamSection`, one per element** — this is how per-element sizing is realized. `FrameResult` has `.displacements` (n_nodes,6), `.axial_force` (n_elem,), `.bending_moment` (n_elem,), `.torsion` (n_elem,).
- `structural.frame.BeamSection.circular(r)` → fields `A=πr²`, `Iy=Iz=πr⁴/4`, `J=πr⁴/2`, `r`.
- `beams.fea_model.BeamFrame` fields: `nodes`, `elements` (n_elem,2), `n_beams`, `n_levels`, `fixed_nodes`, `tip_nodes`, `section`, `E`, `G`. **Element ordering from `build_beam_frame`: the first `n_beams*(n_levels-1)` elements are the longitudinal members; the remaining `n_levels*n_beams` are ring members.**
- `beams.fea_model.build_beam_frame(spec, *, n_beams, n_levels, beam_radius, material, knockdown, nu) -> BeamFrame`.
- `beams.fea_model.project_panels_to_beam_nodes(frame, panels, *, safety_factor) -> (n_nodes,6)` nodal loads (load is independent of cross-section → project once).
- DOF order per node `[ux,uy,uz,rx,ry,rz]`; span is geom +Z so tip twist = displacement column 5 at `frame.tip_nodes`.
- `structural.beam.size_tube_spar(spec, aero_envelope, material=...) -> TubeSparSizing` with `.mass_total_kg`, `.max_stress_Pa`, `.max_tip_deflection_m`, `.governing_case`.
- Scenario: `default_scenario()` → `P` with `P.geometry`, `P.material`, `P.material_iso.skin_E_knockdown`, `P.nu_iso`, `P.sigma_allow_Pa`, `P.rho_kgm3`, `P.load_cases`, `P.aero.spanwise_resolution`. Aero: `from wing_design.aero import build_airplane, sweep_envelope`.

**Comparability caveat (must be surfaced, not hidden):** the Phase-3 tube spar is a *single central bending member*; the form-beam frame is the *entire load-bearing skin structure* (16 full-span beams + rings). A raw total-mass comparison is not apples-to-apples — the tube spar would still need a separate skin. The example must print both masses with this caveat; tests must NOT assert that the frame beats the spar.

---

## Task 1: `von_mises_per_element` structural primitive

**Files:**
- Modify: `src/wing_design/structural/frame.py`
- Test: `tests/structural/test_frame.py`

- [ ] **Step 1: Add the failing tests** (append to `tests/structural/test_frame.py`)

```python
from wing_design.structural.frame import von_mises_per_element


def test_von_mises_pure_axial():
    sec = BeamSection.circular(0.02)
    nodes, elements, sections = _two_node([0, 0, 0], [1, 0, 0], sec)
    loads = np.zeros((2, 6))
    P = 1000.0
    loads[1, 0] = P
    res = solve_frame(nodes, elements, sections, E=E, G=G,
                      fixed_nodes=np.array([0]), loads=loads)
    vm = von_mises_per_element(res, sections)
    assert vm.shape == (1,)
    assert abs(vm[0] - P / sec.A) / (P / sec.A) < 1e-9  # pure axial: vm == |N|/A


def test_von_mises_pure_bending():
    sec = BeamSection.circular(0.02)
    nodes, elements, sections = _two_node([0, 0, 0], [1, 0, 0], sec)
    loads = np.zeros((2, 6))
    Pz = 100.0
    loads[1, 2] = Pz  # transverse tip load → bending moment M = Pz*L at the root
    res = solve_frame(nodes, elements, sections, E=E, G=G,
                      fixed_nodes=np.array([0]), loads=loads)
    vm = von_mises_per_element(res, sections)
    expected = (Pz * 1.0) * sec.r / sec.Iz  # M*r/I at the clamped end
    assert abs(vm[0] - expected) / expected < 1e-6
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/structural/test_frame.py -k von_mises -v`
Expected: FAIL — `ImportError: cannot import name 'von_mises_per_element'`.

- [ ] **Step 3: Implement** (append to `src/wing_design/structural/frame.py`, after `solve_frame`)

```python
def von_mises_per_element(result: FrameResult, sections: list[BeamSection]) -> np.ndarray:
    """Per-element combined-stress proxy (Pa) from a solved frame.

    sigma_n = |N|/A + M_resultant * r / Iz   (axial + bending at the outer fibre)
    tau     = |T| * r / J                     (St-Venant torsional shear)
    sigma_vm = sqrt(sigma_n**2 + 3*tau**2)

    Assumes circular sections (Iy == Iz); `M_resultant` is the resultant bending
    moment already stored on `FrameResult.bending_moment`.
    """
    A = np.array([s.A for s in sections])
    r = np.array([s.r for s in sections])
    Iz = np.array([s.Iz for s in sections])
    J = np.array([s.J for s in sections])
    sigma_n = np.abs(result.axial_force) / A + result.bending_moment * r / Iz
    tau = np.abs(result.torsion) * r / J
    return np.sqrt(sigma_n**2 + 3.0 * tau**2)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/structural/test_frame.py -v`
Expected: all PASS (the 4 prior frame tests + the 2 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/structural/frame.py tests/structural/test_frame.py
git commit -m "feat(structural): von_mises_per_element for per-element sizing"
# end the message with the Co-Authored-By trailer
```

---

## Task 2: Export `von_mises_per_element`

**Files:**
- Modify: `src/wing_design/structural/__init__.py`

- [ ] **Step 1:** Add `von_mises_per_element` to the existing `from .frame import ...` line and add `"von_mises_per_element"` to `__all__` (keep ordering style).

- [ ] **Step 2: Verify**

Run: `uv run python -c "from wing_design.structural import von_mises_per_element; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/wing_design/structural/__init__.py
git commit -m "feat(structural): export von_mises_per_element"
# trailer
```

---

## Task 3: Sizing config, mass, and helpers

**Files:**
- Create: `src/wing_design/beams/sizing.py`
- Test: `tests/beams/test_sizing.py`

- [ ] **Step 1: Write the failing test** (`tests/beams/test_sizing.py`)

```python
import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.fea_model import build_beam_frame
from wing_design.beams.sizing import element_lengths, n_longitudinal, frame_mass


def test_n_longitudinal():
    spec = WingSpec()
    frame = build_beam_frame(spec, n_beams=8, n_levels=5)
    assert n_longitudinal(frame) == 8 * (5 - 1)


def test_element_lengths_match_geometry():
    spec = WingSpec()
    frame = build_beam_frame(spec, n_beams=8, n_levels=5)
    L = element_lengths(frame)
    assert L.shape[0] == frame.elements.shape[0]
    # spot-check one element length directly from node coordinates
    i, j = frame.elements[0]
    expected = np.linalg.norm(frame.nodes[j] - frame.nodes[i])
    assert abs(L[0] - expected) < 1e-12


def test_frame_mass_uniform_radius():
    spec = WingSpec()
    frame = build_beam_frame(spec, n_beams=8, n_levels=5)
    nl = n_longitudinal(frame)
    L = element_lengths(frame)
    r_long, r_ring, rho = 0.02, 0.01, 1550.0
    radii = np.full(nl, r_long)
    m = frame_mass(frame, radii, ring_radius=r_ring, rho=rho)
    expected = rho * (
        np.pi * r_long**2 * L[:nl].sum() + np.pi * r_ring**2 * L[nl:].sum()
    )
    assert abs(m - expected) / expected < 1e-12
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_sizing.py -v`
Expected: FAIL — `ModuleNotFoundError: wing_design.beams.sizing`.

- [ ] **Step 3: Create `src/wing_design/beams/sizing.py`** (config + helpers; the optimizer is added in Task 4)

```python
"""FEA-in-the-loop cross-section sizing for the form-beam frame (Phase C).

Design variables: the radius of every longitudinal beam element (ring connectors
stay at a fixed minimum radius). Objective: minimize total beam mass. Constraints:
per-longitudinal-element von Mises <= allowable, and max-over-load-cases tip
deflection / tip twist <= their limits. Solved with SLSQP, re-solving the
Phase-B frame FEA inside the constraint evaluation (memoized per design point).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..structural.frame import BeamSection, solve_frame, von_mises_per_element
from .fea_model import BeamFrame


@dataclass(frozen=True)
class SizingConfig:
    sigma_allow_Pa: float
    tip_defl_max_m: float
    tip_twist_max_deg: float
    r_min: float = 0.004        # manufacturing floor
    r_max: float = 0.04         # geometric ceiling (inset room in the section)
    ring_radius: float = 0.008  # fixed ring-connector radius


@dataclass(frozen=True)
class SizingResult:
    radii: np.ndarray           # (n_longitudinal,) sized longitudinal radii [m]
    mass_kg: float
    converged: bool
    n_iter: int
    max_vm_stress_Pa: float     # worst longitudinal element, over all cases
    tip_defl_m: float           # worst case
    tip_twist_deg: float        # worst case


def n_longitudinal(frame: BeamFrame) -> int:
    """Number of longitudinal elements (the first block of `frame.elements`)."""
    return frame.n_beams * (frame.n_levels - 1)


def element_lengths(frame: BeamFrame) -> np.ndarray:
    """(n_elem,) Euclidean length of every frame element."""
    i = frame.elements[:, 0]
    j = frame.elements[:, 1]
    return np.linalg.norm(frame.nodes[j] - frame.nodes[i], axis=1)


def build_sections(frame: BeamFrame, long_radii: np.ndarray, ring_radius: float) -> list[BeamSection]:
    """Per-element sections: longitudinal use `long_radii`, rings use `ring_radius`."""
    nl = n_longitudinal(frame)
    if len(long_radii) != nl:
        raise ValueError(f"expected {nl} longitudinal radii, got {len(long_radii)}")
    n_ring = frame.elements.shape[0] - nl
    return [BeamSection.circular(float(r)) for r in long_radii] + [
        BeamSection.circular(ring_radius)
    ] * n_ring


def frame_mass(frame: BeamFrame, long_radii: np.ndarray, *, ring_radius: float, rho: float) -> float:
    """Total beam mass [kg] = rho * sum(area * length) over longitudinal + ring elements."""
    nl = n_longitudinal(frame)
    L = element_lengths(frame)
    long_area = np.pi * np.asarray(long_radii) ** 2
    ring_area = np.pi * ring_radius**2
    return float(rho * (np.sum(long_area * L[:nl]) + ring_area * np.sum(L[nl:])))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/beams/test_sizing.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/sizing.py tests/beams/test_sizing.py
git commit -m "feat(beams): sizing config + mass/length/section helpers"
# trailer
```

---

## Task 4: `size_beams` SLSQP optimizer

**Files:**
- Modify: `src/wing_design/beams/sizing.py`
- Test: `tests/beams/test_sizing.py`

- [ ] **Step 1: Write the failing test** (append to `tests/beams/test_sizing.py`)

```python
from wing_design.beams.sizing import SizingConfig, size_beams


def test_size_beams_reduces_mass_and_respects_stress():
    # Tiny frame + a small synthetic tip load. With generous deflection/twist
    # limits and a high stress allowable, the optimum drives every longitudinal
    # radius to r_min, so sized mass must be well below the r_max starting mass.
    spec = WingSpec()
    frame = build_beam_frame(spec, n_beams=4, n_levels=3)
    nl = n_longitudinal(frame)

    loads = np.zeros((frame.nodes.shape[0], 6))
    loads[frame.tip_nodes, 2] = 10.0  # gentle push on the tip ring

    cfg = SizingConfig(
        sigma_allow_Pa=1.0e8,
        tip_defl_max_m=1.0,         # generous → not binding
        tip_twist_max_deg=90.0,     # generous → not binding
        r_min=0.004,
        r_max=0.03,
        ring_radius=0.008,
    )
    rho = 1550.0
    start_mass = frame_mass(frame, np.full(nl, cfg.r_max), ring_radius=cfg.ring_radius, rho=rho)

    result = size_beams(frame, [loads], cfg, rho=rho, maxiter=30)

    assert result.radii.shape == (nl,)
    assert result.mass_kg < start_mass                       # mass was reduced
    assert result.max_vm_stress_Pa <= cfg.sigma_allow_Pa * 1.05  # stress feasible
    assert np.all(result.radii >= cfg.r_min - 1e-9)          # bounds respected
    assert np.all(result.radii <= cfg.r_max + 1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_sizing.py::test_size_beams_reduces_mass_and_respects_stress -v`
Expected: FAIL — `ImportError: cannot import name 'size_beams'`.

- [ ] **Step 3: Implement** (append to `src/wing_design/beams/sizing.py`)

```python
def size_beams(
    frame: BeamFrame,
    load_arrays: list[np.ndarray],
    config: SizingConfig,
    *,
    rho: float,
    maxiter: int = 50,
    ftol: float = 1.0e-4,
    x0_radius: float | None = None,
) -> SizingResult:
    """Minimize beam mass s.t. stress, tip-deflection, and tip-twist constraints.

    `load_arrays` is one (n_nodes, 6) nodal-load array per load case (already
    safety-factored). The frame geometry is fixed; only longitudinal radii vary.
    """
    nl = n_longitudinal(frame)
    L = element_lengths(frame)
    x0 = np.full(nl, config.r_max if x0_radius is None else x0_radius)

    cache: dict[bytes, tuple[np.ndarray, float, float]] = {}

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, float, float]:
        key = np.asarray(x, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        sections = build_sections(frame, x, config.ring_radius)
        worst_vm = np.zeros(nl)
        max_defl = 0.0
        max_twist = 0.0
        for loads in load_arrays:
            res = solve_frame(
                frame.nodes, frame.elements, sections,
                E=frame.E, G=frame.G, fixed_nodes=frame.fixed_nodes, loads=loads,
            )
            vm = von_mises_per_element(res, sections)
            worst_vm = np.maximum(worst_vm, vm[:nl])
            tip = frame.tip_nodes
            d = float(np.linalg.norm(res.displacements[tip, :3], axis=1).max())
            t = float(np.degrees(np.abs(res.displacements[tip, 5]).max()))
            max_defl = max(max_defl, d)
            max_twist = max(max_twist, t)
        out = (worst_vm, max_defl, max_twist)
        cache[key] = out
        return out

    def mass(x: np.ndarray) -> float:
        return frame_mass(frame, x, ring_radius=config.ring_radius, rho=rho)

    def mass_grad(x: np.ndarray) -> np.ndarray:
        return rho * 2.0 * np.pi * np.asarray(x) * L[:nl]

    def stress_con(x: np.ndarray) -> np.ndarray:
        worst_vm, _, _ = evaluate(x)
        return config.sigma_allow_Pa - worst_vm          # >= 0

    def defl_con(x: np.ndarray) -> np.ndarray:
        _, d, _ = evaluate(x)
        return np.array([config.tip_defl_max_m - d])     # >= 0

    def twist_con(x: np.ndarray) -> np.ndarray:
        _, _, t = evaluate(x)
        return np.array([config.tip_twist_max_deg - t])  # >= 0

    res = minimize(
        mass, x0, jac=mass_grad, method="SLSQP",
        bounds=[(config.r_min, config.r_max)] * nl,
        constraints=[
            {"type": "ineq", "fun": stress_con},
            {"type": "ineq", "fun": defl_con},
            {"type": "ineq", "fun": twist_con},
        ],
        options={"maxiter": maxiter, "ftol": ftol},
    )

    worst_vm, d, t = evaluate(res.x)
    return SizingResult(
        radii=np.asarray(res.x),
        mass_kg=mass(res.x),
        converged=bool(res.success),
        n_iter=int(res.nit),
        max_vm_stress_Pa=float(worst_vm.max()),
        tip_defl_m=float(d),
        tip_twist_deg=float(t),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/beams/test_sizing.py -v`
Expected: all PASS. (The tiny problem solves in a second or two.)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/sizing.py tests/beams/test_sizing.py
git commit -m "feat(beams): SLSQP FEA-in-the-loop beam sizing"
# trailer
```

---

## Task 5: Export + deliverable example + plan update

**Files:**
- Modify: `src/wing_design/beams/__init__.py`
- Create: `examples/22_sized_beams.py`
- Modify: `docs/plan.md`

- [ ] **Step 1: Export the sizing API** — in `src/wing_design/beams/__init__.py` add:

```python
from .sizing import SizingConfig, SizingResult, frame_mass, n_longitudinal, size_beams
```

and add `"SizingConfig"`, `"SizingResult"`, `"frame_mass"`, `"n_longitudinal"`, `"size_beams"` to `__all__` (match ordering style). (`n_longitudinal` is exported because the example uses it.)

- [ ] **Step 2: Create `examples/22_sized_beams.py`**

```python
"""Phase-C spike: FEA-in-the-loop sizing of the form-beam frame.

Builds the Phase-B beam frame, projects the LiftingLine load envelope onto its
nodes, then sizes every longitudinal element's radius by minimizing beam mass
subject to stress + tip-deflection + tip-twist constraints (SLSQP, re-solving the
frame FEA each iteration). Prints the sized metrics and a per-beam root→tip radius
profile, then reports the Phase-3 tube-spar baseline mass for context.

NOTE: the tube spar is a single central bending member; this frame is the entire
load-bearing skin structure (16 beams + rings). The two masses are reported side
by side but are NOT directly comparable — the spar would still need a skin.
"""
from __future__ import annotations

import numpy as np

from wing_design import default_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    SizingConfig,
    build_beam_frame,
    n_longitudinal,
    project_panels_to_beam_nodes,
    size_beams,
)
from wing_design.structural import size_tube_spar


def main() -> None:
    P = default_scenario()
    spec = P.geometry

    frame = build_beam_frame(
        spec, n_beams=16, n_levels=8, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
    )
    print(f"Frame: {frame.nodes.shape[0]} nodes, {frame.elements.shape[0]} elements, "
          f"{n_longitudinal(frame)} sizing variables")

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
    print(f"Sizing against {len(load_arrays)} non-feathered load cases.")

    cfg = SizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span,
        tip_twist_max_deg=5.0,
    )
    print(f"\nConstraints: σ_allow={cfg.sigma_allow_Pa/1e6:.0f} MPa, "
          f"tip_defl≤{cfg.tip_defl_max_m*1e3:.0f} mm, tip_twist≤{cfg.tip_twist_max_deg:.1f}°")
    print(f"Radius bounds: [{cfg.r_min*1e3:.0f}, {cfg.r_max*1e3:.0f}] mm; "
          f"ring radius {cfg.ring_radius*1e3:.0f} mm\n")

    print("Sizing (SLSQP, FEA re-solve per step) — this takes a minute...")
    result = size_beams(frame, load_arrays, cfg, rho=P.rho_kgm3, maxiter=60)

    print(f"\n  converged={result.converged}  iters={result.n_iter}")
    print(f"  beam mass        = {result.mass_kg:8.3f} kg")
    print(f"  max σ_vm         = {result.max_vm_stress_Pa/1e6:8.1f} MPa "
          f"(allow {cfg.sigma_allow_Pa/1e6:.0f})")
    print(f"  tip deflection   = {result.tip_defl_m*1e3:8.1f} mm "
          f"(limit {cfg.tip_defl_max_m*1e3:.0f})")
    print(f"  tip twist        = {result.tip_twist_deg:8.3f} °  "
          f"(limit {cfg.tip_twist_max_deg:.1f})")

    # Per-beam root→tip radius profile (longitudinal radii are ordered beam-major,
    # level-minor: beam b occupies indices [b*(n_levels-1), (b+1)*(n_levels-1))).
    seg = frame.n_levels - 1
    radii_mm = result.radii.reshape(frame.n_beams, seg) * 1e3
    print("\n  per-beam radii [mm], tip→keel segment order:")
    for b in range(frame.n_beams):
        row = " ".join(f"{v:4.1f}" for v in radii_mm[b])
        print(f"    beam {b:2d}: {row}")

    baseline = size_tube_spar(spec, envelope, material=P.material)
    print(f"\nPhase-3 tube-spar baseline: mass={baseline.mass_total_kg:.3f} kg "
          f"(σ_max={baseline.max_stress_Pa/1e6:.0f} MPa, governing={baseline.governing_case})")
    print("  NOTE: not directly comparable — the spar is a single bending member;")
    print("  the form-beam frame is the whole load-bearing skin structure.")


if __name__ == "__main__":
    main()
```

(Note: this example imports `n_longitudinal` from `wing_design.beams` — ensure it is exported. If Task 3 did not export it, add `n_longitudinal` to the `from .sizing import ...` line and `__all__` in this step alongside the other sizing names.)

- [ ] **Step 3: Run the example end-to-end**

Run: `uv run python examples/22_sized_beams.py`
Expected: frame summary; a converged sizing with `max σ_vm` at/under the allowable, tip deflection/twist under their limits; a per-beam radius profile that is generally thicker toward the keel and thinner toward the tip; and the baseline line with the caveat.
**If SLSQP reports `converged=False` or returns an infeasible design (max σ_vm ≫ allowable, or deflection/twist over limit): report DONE_WITH_CONCERNS** with the full output — loosen `maxiter`/`ftol` or the constraint limits may need revisiting; do not silently present an infeasible result as success.

- [ ] **Step 4: Update `docs/plan.md`** — in **Phase C — FEA-in-the-loop sizing spike**, append a status note:

```markdown
**Status: done (2026-06-04).** Per-element longitudinal radii sized by SLSQP
(`beams.sizing.size_beams`) minimizing beam mass s.t. per-element von Mises +
tip-deflection + tip-twist constraints, re-solving the Phase-B frame FEA each
step (rings held at a fixed minimum radius). Deliverable: `examples/22_sized_beams.py`.
Caveat logged: the tube-spar baseline is a single bending member, so its mass is
reported for context but is not directly comparable to the full form-beam frame.
```

and add to the **Decisions log** table:

```markdown
| Phase-C sizing | Continuous SLSQP NLP over per-element longitudinal radii; stress + tip-deflection + tip-twist constraints; analytic mass gradient, FD constraint Jacobian over FEA re-solves. (MILP stock catalog deferred to Phase E.) |
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: all green (Phase A + B + new frame/sizing tests).

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/__init__.py examples/22_sized_beams.py docs/plan.md
git commit -m "feat(beams): Phase-C sizing example + export API; mark phase done"
# trailer
```

---

## Self-Review

- **Spec coverage:** per-element longitudinal sizing → `n_longitudinal` block + `build_sections` (Task 3); gradient optimizer → SLSQP `size_beams` (Task 4); stress+deflection+twist constraints → the three `*_con` callbacks (Task 4); mass-vs-baseline → example (Task 5). Deliverable `examples/22_sized_beams.py` present.
- **Placeholder scan:** none — all code concrete.
- **Type consistency:** `BeamSection`/`solve_frame`/`von_mises_per_element` (frame.py) consumed unchanged in sizing.py; `BeamFrame.tip_nodes`/`fixed_nodes`/`elements` used per their Phase-B definitions; `n_longitudinal(frame)` reused across helpers, optimizer, and example; `SizingConfig`/`SizingResult` fields match their construction and consumption sites.
- **Known spike limitations (intentional):** (1) rings are not sized (fixed `ring_radius`) — accepted per the granularity decision; (2) `von_mises_per_element` is exact only for circular sections (Iy==Iz), which `build_sections` always produces; (3) SLSQP constraint Jacobian is finite-differenced, so cost scales with the number of longitudinal variables — the example uses `n_levels=8` to stay tractable; (4) the baseline mass comparison is informational, not a pass/fail gate — tests assert mass reduction + stress feasibility only.
