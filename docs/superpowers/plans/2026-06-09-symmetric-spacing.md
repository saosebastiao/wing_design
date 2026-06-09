# Mirror-Symmetric Non-Uniform Spacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A `chord_symmetrize_weights` helper that turns per-segment stress weights into mirror-symmetric ones (max-of-mirror), so stress-weighted beam spacing (`stress_weighted_targets` + the existing `arc_fractions` path) yields a chord-symmetric placement that keeps `beam_radius_groups` grouping. Measure even-vs-symmetric-spaced.

**Domain facts:**
- `beams/layout.py`: `stress_weighted_targets(weights, n)` → ascending arc fractions (cumulative-weight placement; uniform weights → `arange(n)/n`). `cross_section_stress_weights(model, load_arrays)` → `(n_beams,)` per-perimeter-segment peak membrane vM (segment `b` between beams `b`,`b+1`).
- `build_beam_shell_model(..., arc_fractions=...)` places beams at those fractional arc positions at every level.
- `beam_radius_groups(model)` (in `beams/shell_sizing.py`) → `(group_of_element, n_groups)`; detects mirror symmetry from node positions, falls back to `(arange(n), n)` if asymmetric. Mirror: beam `b` ↔ `(n_beams−b)%n_beams`; segment `b` ↔ `(n_beams−1−b)%n_beams`.
- `default_scenario` removed → tests use `small_scenario`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1: `chord_symmetrize_weights` + composition test

**Files:** Modify `src/wing_design/beams/layout.py`, `src/wing_design/beams/__init__.py`; Test `tests/beams/test_symmetric_spacing.py` (new).

- [ ] **Step 1: Failing tests** — create `tests/beams/test_symmetric_spacing.py`:
```python
import numpy as np

from wing_design.scenario import small_scenario
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_radius_groups
from wing_design.beams.layout import chord_symmetrize_weights, stress_weighted_targets


def test_symmetrize_is_mirror_symmetric():
    rng = np.random.default_rng(0)
    n = 8
    w = rng.uniform(1.0, 5.0, size=n)
    ws = chord_symmetrize_weights(w)
    assert ws.shape == (n,)
    for b in range(n):
        assert np.isclose(ws[b], ws[(n - 1 - b) % n])           # mirror-symmetric
        assert np.isclose(ws[b], max(w[b], w[(n - 1 - b) % n]))  # per-pair max
    assert np.allclose(chord_symmetrize_weights(ws), ws)         # idempotent


def test_symmetric_spacing_keeps_grouping():
    rng = np.random.default_rng(1)
    n_beams, n_levels = 8, 5
    w = rng.uniform(1.0, 5.0, size=n_beams)
    af = stress_weighted_targets(chord_symmetrize_weights(w), n_beams)
    m = build_beam_shell_model(small_scenario().geometry, n_beams=n_beams, n_levels=n_levels,
                               beam_radius=0.02, arc_fractions=af)
    _, ng = beam_radius_groups(m)
    assert ng == (n_beams // 2 + 1) * (n_levels - 1)   # symmetry detected (reduced), not the n fallback


def test_asymmetric_spacing_falls_back():
    n_beams, n_levels = 8, 5
    af = np.linspace(0.0, 1.0, n_beams, endpoint=False) ** 1.5   # deliberately asymmetric
    m = build_beam_shell_model(small_scenario().geometry, n_beams=n_beams, n_levels=n_levels,
                               beam_radius=0.02, arc_fractions=af)
    _, ng = beam_radius_groups(m)
    assert ng == n_beams * (n_levels - 1)              # fallback to independent radii


def test_even_spacing_is_symmetric():
    n_beams, n_levels = 8, 5
    af = np.arange(n_beams) / n_beams
    m = build_beam_shell_model(small_scenario().geometry, n_beams=n_beams, n_levels=n_levels,
                               beam_radius=0.02, arc_fractions=af)
    _, ng = beam_radius_groups(m)
    assert ng == (n_beams // 2 + 1) * (n_levels - 1)
```

- [ ] **Step 2:** Run → FAIL (`chord_symmetrize_weights` missing).

- [ ] **Step 3: Implement** in `beams/layout.py`:
```python
def chord_symmetrize_weights(weights: np.ndarray) -> np.ndarray:
    """Make per-perimeter-segment weights mirror-symmetric across the chord (max-of-mirror).

    Segment ``b`` (between beams b and b+1) mirrors segment ``(n-1-b) % n`` across the
    chord plane (the section is traversed TE->upper->LE->lower->TE). Returns
    ``w_sym[b] = max(weights[b], weights[(n-1-b) % n])`` so stress-weighted placement
    (``stress_weighted_targets``) yields a chord-symmetric beam layout that keeps the
    radius-symmetry grouping (``beam_radius_groups``) active. Both-tack-conservative.
    """
    w = np.asarray(weights, dtype=float)
    n = w.shape[0]
    mirror = w[(n - 1 - np.arange(n)) % n]
    return np.maximum(w, mirror)
```
Export `chord_symmetrize_weights` from `beams/__init__.py` (it already exports `stress_weighted_targets`, `cross_section_stress_weights` — add alongside, + `__all__`).

- [ ] **Step 4:** Run the 4 tests → PASS. (If `test_symmetric_spacing_keeps_grouping` fails because the symmetric placement isn't detected, the mirror-segment convention is off — verify `stress_weighted_targets` of symmetric weights gives `af` with `af[b]` ↔ `1−af[n−b]`; do NOT loosen the grouping tolerance.)

- [ ] **Step 5:** Commit:
```bash
git add src/wing_design/beams/layout.py src/wing_design/beams/__init__.py tests/beams/test_symmetric_spacing.py
git commit -m "$(cat <<'EOF'
Symmetric spacing: chord_symmetrize_weights (max-of-mirror); keeps radii grouping

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Even-vs-symmetric-spaced example

**Files:** Create `examples/40_symmetric_spacing.py`.

- [ ] **Step 1: Create** `examples/40_symmetric_spacing.py`:
```python
"""Mirror-symmetric stress-weighted beam spacing vs even spacing (medium wingsail).

Builds an even-spaced model, solves the load envelope to get per-perimeter-segment skin
stress, symmetrizes it across the chord (max-of-mirror) so the placement stays groupable,
re-places the beams at the resulting stress-weighted arc positions, and sizes both layouts
under identical constraints (span datum + buckling + 4 bands, analytic Jacobian). Prints
the stress-concentration ratio and the mass comparison. FEA-only.
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig, beam_radius_groups, build_beam_frame, build_beam_shell_model,
    chord_symmetrize_weights, cross_section_stress_weights, project_panels_to_beam_nodes,
    size_beam_shell_laminate, stress_weighted_targets,
)
from wing_design.materials.unidir import T700_EPOXY


def main() -> None:
    P = medium_scenario(); spec = P.geometry
    n_beams, n_levels = 16, 8
    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)

    def model(af=None):
        return build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
            material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
            skin_thickness=P.skin_sizing.t_baseline_m, arc_fractions=af)

    airplane = build_airplane(spec)
    env = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    even = model()
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    w = cross_section_stress_weights(even, loads)
    w_sym = chord_symmetrize_weights(w)
    af = stress_weighted_targets(w_sym, n_beams)
    sym = model(af)
    print(f"stress concentration (max/mean segment): {w.max()/w.mean():.2f}; "
          f"symmetric-spaced groups={beam_radius_groups(sym)[1]} (even={beam_radius_groups(even)[1]})")

    defl_lim, twist_lim = 0.02 * spec.span, 5.0
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim,
        tip_twist_max_deg=twist_lim, ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=1.5,
        n_skin_bands=4, use_analytic_jacobian=True)

    print("sizing even spacing..."); re = size_beam_shell_laminate(even, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300)
    print("sizing symmetric stress-weighted spacing..."); rs = size_beam_shell_laminate(sym, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300)

    for tag, r in (("even", re), ("symmetric-weighted", rs)):
        print(f"  [{tag}] mass={r.mass_kg:.2f} kg (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f}) "
              f"conv={r.converged} twist={r.tip_twist_deg:.2f} buck={r.max_beam_buckling_util:.2f}/{r.max_panel_buckling_util:.2f}")
    dm = rs.mass_kg - re.mass_kg
    print(f"\n  Δmass {dm:+.2f} kg ({100*dm/re.mass_kg:+.1f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** `uv run python -m py_compile examples/40_symmetric_spacing.py`. Commit.
- [ ] **Step 3 (orchestrator):** run it timed (background); record numbers + wall-clock.

---

## Task 3: Record finding

**Files:** Modify `docs/plan.md`.

- [ ] **Step 1:** Status note (after the tip-gusset note, before `### Phase F`) + Decisions-log row, using the Task-2 run numbers: mirror-symmetric stress-weighted spacing via `chord_symmetrize_weights`; keeps `beam_radius_groups` grouping; even [Me] kg vs symmetric-weighted [Ms] kg ([±X]%); stress concentration [ratio]; verdict (likely small, per F.1's ~1% on this skin-dominated design). Note even spacing stays the default.
- [ ] **Step 2:** Commit.

---

## Final verification
- [ ] `uv run pytest -q` — all green (the 4 new spacing tests + suite).
- [ ] `uv run python -m py_compile examples/40_symmetric_spacing.py`.
- [ ] Orchestrator: timed even-vs-symmetric run recorded.
- [ ] Final review, then superpowers:finishing-a-development-branch.

## Notes / risks
- The grouping-preservation test is the crux — it proves symmetric stress-weighted placement keeps the radii mirror-grouped (the whole point). If it fails, fix the mirror-segment convention, not the test tolerance.
- Even spacing stays the default; this is opt-in via `arc_fractions`. No default change.
- Composition is automatic (arc placement only moves nodes; sizer/sensitivity/taper/banding/datum are position-driven) — no sizer/sensitivity changes.
