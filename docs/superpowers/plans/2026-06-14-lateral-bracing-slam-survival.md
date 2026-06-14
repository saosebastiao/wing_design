# Lateral ring-frame bracing for slam survival — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add transverse ring-frame bracing (slender beam members tying adjacent form beams at spanwise stations) so the wingsail can be sized to survive a lateral slam load (λ_cr ≥ 1.5), with the rings made visible to the in-loop foundation-buckling model via an FD-validated stiffness-augmentation term.

**Architecture:** Rings are extra beam elements on the existing form-beam grid (core-tube pattern), sized by one shared `brace_radius` DV. They enter every FEA solve automatically. They are credited in the sizing loop by augmenting the elastic-foundation stiffness `k_eff = k_skin + k_ring(r_brace)` (one new analytic gradient, FD-validated and gated first), and verified by the existing post-hoc eigenvalue solve used as an acceptance gate with an automated safety-factor-bump re-size loop.

**Tech Stack:** Python 3.13, NumPy, SciPy (SLSQP / IPOPT via cyipopt), the project's beam-shell FEA + laminate sizer. Run via `just`/`.venv` with `PYTHONPYCACHEPREFIX=$PWD/.pycache`. Tests: `uv run pytest` (fast unit suite).

**Conventions (CLAUDE.md):** new analytic gradients get a central-difference FD-validation test (rel-err ≤ ~1e-4, tiny mesh); legacy sizers are frozen (work only in the laminate path); measure wall-clock; a sizing result counts only if converged AND feasible AND (here) eigen λ_cr ≥ 1.5.

**Test command prefix (use everywhere):**
`PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest`

---

## File structure

- `src/wing_design/beams/shell_model.py` — add `brace_elements` field + ring generator in `build_beam_shell_model` (+ `lateral_bracing`, `brace_stations` args) + a `braced_segment_mask` helper.
- `src/wing_design/beams/laminate_sizing.py` — add `brace_radius` DV block + bounds; config fields `brace_r_min/brace_r_max`; brace mass term; `k_ring` augmentation inside `foundation_k`; `brace_vm` ConstraintSpec; `design_vector_from_result` block.
- `src/wing_design/structural/buckling.py` — `k_ring_stiffness(r_brace, E, L_ring, s, alpha)` + `dk_ring_dr` (pure functions, easy to FD-test).
- `src/wing_design/beams/build.py` — `build_sized_ring_braces(...)` loft (milestone export).
- `runs/slam_braced.py` — eigen-gated, SF-bump slam re-probe (new run script, tracked).
- Tests: `tests/structural/test_buckling.py` (k_ring + gradient), `tests/beams/test_shell_model.py` (ring generator), `tests/beams/test_ks_smoothing.py` (k_ring through KS), `tests/beams/test_laminate_sizing.py` (roundtrip + braced smoke).

---

## Task 1: Ring-frame geometry generator + model field

**Files:**
- Modify: `src/wing_design/beams/shell_model.py:41-75` (add field), `:77-194` (generate rings), append `braced_segment_mask` helper near `beam_adjacent_widths:275`.
- Test: `tests/beams/test_shell_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/beams/test_shell_model.py  (append)
import numpy as np
from wing_design.geometry import medium_wingsail
from wing_design.beams.shell_model import build_beam_shell_model, braced_segment_mask


def test_lateral_bracing_adds_expected_ring_elements():
    nb, nl = 8, 6
    base = build_beam_shell_model(medium_wingsail, n_beams=nb, n_levels=nl)
    braced = build_beam_shell_model(medium_wingsail, n_beams=nb, n_levels=nl,
                                    lateral_bracing=True)  # default stations="interior"
    # interior stations = 1..nl-2  -> (nl-2) rings, each nb closed-loop elements
    n_rings = nl - 2
    assert braced.brace_elements is not None
    assert braced.brace_elements.shape[0] == n_rings * nb
    # rings are extra beam elements appended after the form (+ no tube here)
    assert braced.beam_elements.shape[0] == base.beam_elements.shape[0] + n_rings * nb
    # each ring element connects two nodes at the SAME z-level (a transverse ring)
    for e in braced.brace_elements:
        a, b = braced.beam_elements[e]
        assert np.isclose(braced.nodes[a, 2], braced.nodes[b, 2])
    # default off is unchanged
    assert base.brace_elements is None


def test_braced_segment_mask_interior():
    nb, nl = 8, 6
    m = build_beam_shell_model(medium_wingsail, n_beams=nb, n_levels=nl,
                               lateral_bracing=True)
    mask = braced_segment_mask(m)            # (n_form_elements,) bool
    assert mask.shape == (m.n_form_elements,)
    # a form segment is "braced" iff BOTH its stations carry a ring (interior 1..nl-2);
    # per beam: segments between stations (0,1)..(nl-2,nl-1); only fully-interior
    # segments (1,2)..(nl-3,nl-2) have both ends braced.
    per_beam = mask.reshape(nb, nl - 1)
    assert not per_beam[:, 0].any()          # segment (0,1): station 0 unbraced
    assert not per_beam[:, -1].any()         # segment (nl-2,nl-1): station nl-1 unbraced
    assert per_beam[:, 1:-1].all()           # interior segments fully braced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_shell_model.py -k "lateral_bracing or braced_segment" -q`
Expected: FAIL — `build_beam_shell_model() got an unexpected keyword argument 'lateral_bracing'` / `cannot import name 'braced_segment_mask'`.

- [ ] **Step 3: Add the field, args, generator, and helper**

In the `BeamShellModel` dataclass (after `hollow_elements`, `shell_model.py:69`):

```python
    # task #22 lateral bracing: rows of `beam_elements` that are transverse ring
    # members (sized by the brace_radius DV), and the spanwise stations carrying
    # a ring (level indices into the z-grid).
    brace_elements: np.ndarray | None = None      # (n_rings*n_beams,) indices into beam_elements
    brace_stations: np.ndarray | None = None      # (n_rings,) level indices
```

Add args to `build_beam_shell_model` signature (after `straightness_tol_m`, `shell_model.py:92`):

```python
    lateral_bracing: bool = False,
    brace_stations: "str | tuple[int, ...]" = "interior",
```

Insert ring generation AFTER the `core_tube` block (after `shell_model.py:172`, before the `return`):

```python
    brace_elements = brace_stations_arr = None
    if lateral_bracing:
        if brace_stations == "interior":
            stations = list(range(1, n_levels - 1))
        elif brace_stations == "all":
            stations = list(range(n_levels))
        else:
            stations = sorted(int(k) for k in brace_stations)
        if not stations:
            raise ValueError("lateral_bracing=True but no brace stations selected")
        n0 = beam_elements.shape[0]
        ring_rows = [(nid(b, k), nid((b + 1) % n_beams, k))
                     for k in stations for b in range(n_beams)]
        beam_elements = np.vstack([beam_elements, np.asarray(ring_rows, dtype=int)])
        brace_elements = np.arange(n0, n0 + len(ring_rows), dtype=int)
        brace_stations_arr = np.asarray(stations, dtype=int)
```

Add to the `BeamShellModel(...)` return (after `hollow_elements=hollow_elements,`, `shell_model.py:193`):

```python
        brace_elements=brace_elements,
        brace_stations=brace_stations_arr,
```

Append the helper near `beam_adjacent_widths` (end of `shell_model.py`):

```python
def braced_segment_mask(model: BeamShellModel) -> np.ndarray:
    """(n_form_elements,) bool: True where a form-beam SEGMENT has a ring at BOTH
    of its end stations (so the ring shortens that segment's lateral buckling
    length). Returns all-False when no bracing is present."""
    nb, nl = model.n_beams, model.n_levels
    mask = np.zeros(model.n_form_elements, dtype=bool)
    if model.brace_stations is None:
        return mask
    braced = set(int(k) for k in model.brace_stations)
    for b in range(nb):
        for k in range(nl - 1):
            if k in braced and (k + 1) in braced:
                mask[b * (nl - 1) + k] = True
    return mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_shell_model.py -k "lateral_bracing or braced_segment" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/shell_model.py tests/beams/test_shell_model.py
git commit -m "feat(bracing): transverse ring-frame generator + braced_segment_mask"
```

---

## Task 2: `k_ring` stiffness + analytic gradient (the GATED gradient)

**Files:**
- Modify: `src/wing_design/structural/buckling.py` (append two pure functions)
- Test: `tests/structural/test_buckling.py`

This is the one genuinely new analytic gradient. It is a pure scalar/array function, so it is FD-validated in isolation here BEFORE any sizing code uses it (gradient gated first).

- [ ] **Step 1: Write the failing test**

```python
# tests/structural/test_buckling.py  (append)
import numpy as np
from wing_design.structural.buckling import k_ring_stiffness, dk_ring_dr


def _central_diff(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


def test_k_ring_quartic_and_gradient_fd():
    E = 7.0e10
    L_ring = np.array([0.45, 0.50, 0.40])   # perimeter spacing per element
    s = np.array([0.9, 0.9, 1.1])           # spanwise tributary length
    alpha = 3.0
    r = 0.012
    k = k_ring_stiffness(r, E, L_ring, s, alpha)
    # closed form: k = alpha * E * (pi r^4 / 4) / (L^3 * s); strictly positive
    expect = alpha * E * (np.pi * r**4 / 4.0) / (L_ring**3 * s)
    assert np.allclose(k, expect, rtol=1e-12)
    # analytic gradient vs central difference
    ana = dk_ring_dr(r, E, L_ring, s, alpha)
    fd = _central_diff(lambda rr: k_ring_stiffness(rr, E, L_ring, s, alpha), r, 1e-7)
    assert np.allclose(ana, fd, rtol=1e-6, atol=1e-6 * np.abs(fd).max())
    # quartic scaling: dk/dr == 4 k / r
    assert np.allclose(ana, 4.0 * k / r, rtol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/structural/test_buckling.py -k k_ring -q`
Expected: FAIL — `cannot import name 'k_ring_stiffness'`.

- [ ] **Step 3: Implement the functions**

Append to `src/wing_design/structural/buckling.py`:

```python
def k_ring_stiffness(r_brace, E, L_ring, s, alpha=3.0):
    """Distributed elastic-foundation stiffness (force / length / length) that a
    transverse ring member of circular radius ``r_brace`` contributes to a braced
    form-beam segment, smeared over the spanwise tributary length ``s``.

    Model: the ring acts as a transverse spring of stiffness alpha*E*I_ring/L_ring^3
    (I_ring = pi r^4 / 4, alpha = end-restraint constant, 3 = cantilever, conservative),
    smeared over the segment's spanwise length s -> k = spring / s. ``L_ring`` and ``s``
    may be arrays (per element); ``r_brace`` is the single shared DV (scalar)."""
    I = np.pi * r_brace**4 / 4.0
    return alpha * E * I / (np.asarray(L_ring, float) ** 3 * np.asarray(s, float))


def dk_ring_dr(r_brace, E, L_ring, s, alpha=3.0):
    """d k_ring / d r_brace = 4 * k_ring / r_brace (quartic in r)."""
    return 4.0 * k_ring_stiffness(r_brace, E, L_ring, s, alpha) / r_brace
```

(Add `import numpy as np` at the top of `buckling.py` if not already present — check `buckling.py:1-10`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/structural/test_buckling.py -k k_ring -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/structural/buckling.py tests/structural/test_buckling.py
git commit -m "feat(bracing): k_ring foundation-stiffness term + FD-validated gradient"
```

---

## Task 3: `brace_radius` DV block, bounds, and config fields

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py` — config dataclass (near other geom fields), bounds in `laminate_design_bounds` (`:184-199`), block list (`:371-378`).
- Test: `tests/beams/test_laminate_sizing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/beams/test_laminate_sizing.py  (append)
import numpy as np
from wing_design.geometry import medium_wingsail
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.laminate_sizing import LaminateSizingConfig, laminate_design_bounds


def test_brace_radius_block_present_and_bounded():
    m = build_beam_shell_model(medium_wingsail, n_beams=8, n_levels=6,
                               lateral_bracing=True)
    cfg = LaminateSizingConfig(brace_r_min=0.002, brace_r_max=0.05)
    lo, hi = laminate_design_bounds(m, cfg)
    # exactly one brace_radius variable, bounded [brace_r_min, brace_r_max]
    assert lo[-1] == 0.002 and hi[-1] == 0.05
    # unbraced model has no extra var
    m0 = build_beam_shell_model(medium_wingsail, n_beams=8, n_levels=6)
    lo0, _ = laminate_design_bounds(m0, cfg)
    assert lo.size == lo0.size + 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_laminate_sizing.py -k brace_radius_block -q`
Expected: FAIL — `LaminateSizingConfig` has no field `brace_r_min`, or last bound is not the brace radius.

- [ ] **Step 3: Add config fields, bounds, and the block**

Add to `LaminateSizingConfig` (next to `r_min`/`r_max`; grep `r_min:` in `laminate_sizing.py`):

```python
    brace_r_min: float = 0.002
    brace_r_max: float = 0.05
```

Define a single source of truth for "is this model braced" near the top of the sizer body (where `tube`/`hollow` booleans are set, around `laminate_sizing.py:355`):

```python
    braced = getattr(model, "brace_elements", None) is not None
```

Add the block to the `blocks` list (`laminate_sizing.py:371-378`, append AFTER the sandwich `t_core` block so existing layouts are unchanged when unbraced):

```python
    if braced:
        blocks += [("brace_radius", 1)]
```

Add bounds in `laminate_design_bounds` (`laminate_sizing.py:184-199`, mirror the `r_tube` block; append after the `t_core` bounds, and add the same `braced` check there):

```python
    if getattr(model, "brace_elements", None) is not None:
        lo = np.concatenate([lo, np.array([config.brace_r_min])])
        hi = np.concatenate([hi, np.array([config.brace_r_max])])
```

> NOTE: `laminate_design_bounds` and the sizer body must agree on block ORDER. Keep `brace_radius` LAST in both. Verify by reading both sites.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_laminate_sizing.py -k brace_radius_block -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/laminate_sizing.py tests/beams/test_laminate_sizing.py
git commit -m "feat(bracing): brace_radius DV block + bounds + config fields"
```

---

## Task 4: Brace mass + gradient

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py` — mass accumulation (`:1406`) and the objective gradient; result field `brace_mass_kg` (dataclass near `tube_mass_kg:164`).
- Test: `tests/beams/test_laminate_sizing.py`

- [ ] **Step 1: Write the failing test** — a tiny braced sizer evaluation reports a positive `brace_mass_kg` consistent with `Σ len·πr²·ρ`, and the objective gradient w.r.t. `brace_radius` matches central difference.

```python
# tests/beams/test_laminate_sizing.py  (append)
def test_brace_mass_and_objective_gradient_fd():
    import numpy as np
    from wing_design.geometry import medium_wingsail
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.laminate_sizing import (
        LaminateSizingConfig, laminate_design_bounds, _objective_and_grad_for_test)
    m = build_beam_shell_model(medium_wingsail, n_beams=8, n_levels=5,
                               lateral_bracing=True)
    cfg = LaminateSizingConfig(brace_r_min=0.002, brace_r_max=0.05)
    lo, hi = laminate_design_bounds(m, cfg)
    x = 0.5 * (lo + hi)
    f, g = _objective_and_grad_for_test(m, cfg, x, rho=1600.0)  # see Step 3
    fd = (_objective_and_grad_for_test(m, cfg, x + 1e-7 * np.eye(x.size)[-1], rho=1600.0)[0]
          - _objective_and_grad_for_test(m, cfg, x - 1e-7 * np.eye(x.size)[-1], rho=1600.0)[0]) / 2e-7
    assert abs(g[-1] - fd) <= 1e-5 * max(1.0, abs(fd))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_laminate_sizing.py -k brace_mass_and_objective -q`
Expected: FAIL — `cannot import name '_objective_and_grad_for_test'`.

- [ ] **Step 3: Implement brace mass + gradient + a tiny test hook**

In the sizer, compute brace length once (near where `beam_lengths_e` is built, `laminate_sizing.py:845`):

```python
    brace_len_total = 0.0
    if braced:
        be = model.beam_elements[model.brace_elements]
        brace_len_total = float(np.sum(np.linalg.norm(
            model.nodes[be[:, 0]] - model.nodes[be[:, 1]], axis=1)))
```

Add the mass term where total mass is assembled (`laminate_sizing.py:1406`, define `bracem`):

```python
    bracem = (rho * np.pi * float(x[dv.slice("brace_radius")][0])**2 * brace_len_total
              if braced else 0.0)
```

Include `bracem` in `mass_kg=(bm + tm + sm + ... )` and add `brace_mass_kg=bracem` to the returned `LaminateSizingResult` (add the field `brace_mass_kg: float = 0.0` next to `tube_mass_kg:164`).

In the normalized objective `mass(x)` closure (the one returning `end_mass = mass(x) * m_ref`, `laminate_sizing.py:1365`), add the brace contribution to BOTH value and gradient. The gradient w.r.t. the `brace_radius` column is `2·π·r·brace_len_total·rho / m_ref`. Locate the objective-gradient assembly and scatter this into the `brace_radius` slice via `dv.scatter("brace_radius", [d])` (mirror how `r_tube` mass-gradient is scattered).

Add the test hook at module scope (small, deterministic — evaluates objective value+grad at x without optimizing):

```python
def _objective_and_grad_for_test(model, config, x, *, rho):
    """Test-only: build the sizer's objective + analytic gradient and evaluate at x."""
    obj = _build_objective(model, config, rho)   # factor the existing closure out, or
    return obj.value(x), obj.grad(x)              # expose val/grad used by SLSQP/IPOPT
```

> If factoring `_build_objective` out is invasive, instead expose the existing internal `mass`/`mass_grad` closures through a thin module-level shim used only by tests. Keep production call sites unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_laminate_sizing.py -k brace_mass_and_objective -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/laminate_sizing.py tests/beams/test_laminate_sizing.py
git commit -m "feat(bracing): brace mass + objective gradient"
```

---

## Task 5: Augment `foundation_k` with `k_ring` (+ chain entry)

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py:406-465` (`foundation_k` closure).
- Test: `tests/beams/test_ks_smoothing.py`

This wires the Task-2 term into the in-loop buckling constraint so the optimizer SIZES the rings. The acceptance gate is the existing FD-through-the-live-KS-closure pattern (`test_ks_smoothing.py`), which proves the `brace_radius` column of the `beam_buck` KS-constraint gradient is correct.

- [ ] **Step 1: Write the failing test** (spy-on-`ls.minimize` / direct KS-gradient FD, following the existing `test_ks_panel_datum_ortho_gradients_match_fd` pattern in `tests/beams/test_ks_smoothing.py`):

```python
# tests/beams/test_ks_smoothing.py  (append; mirror the existing KS-FD test's harness)
def test_ks_beam_buck_brace_radius_gradient_matches_fd():
    """The beam_buck KS constraint gradient must have a correct, NONZERO
    brace_radius column once k_ring is wired into foundation_k."""
    import numpy as np
    from wing_design.geometry import medium_wingsail
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.laminate_sizing import LaminateSizingConfig
    from tests.beams._ks_fd_harness import beam_buck_ks_value_and_grad  # see note
    m = build_beam_shell_model(medium_wingsail, n_beams=8, n_levels=5,
                               core_tube=False, hollow_beams=False, lateral_bracing=True)
    cfg = LaminateSizingConfig(ply_angle_datum=(0, 0, 1), ks_rho=50.0,
                               buckling_safety_factor=1.5, use_analytic_jacobian=True,
                               beam_buckling_model="foundation", panel_width_mode="strip",
                               brace_r_min=0.002, brace_r_max=0.05)
    val, grad, jcol = beam_buck_ks_value_and_grad(m, cfg)   # jcol = brace_radius column idx
    # finite-difference the brace_radius column of the beam_buck KS constraint
    fd = beam_buck_ks_fd_column(m, cfg, jcol, h=1e-7)
    assert abs(grad[jcol]) > 0.0                 # rings are now visible to beam_buck
    assert abs(grad[jcol] - fd) <= 1e-4 * max(1.0, abs(fd))
```

> NOTE: reuse the harness the existing KS-FD test already uses (it constructs the sizer's constraint closures and FD-checks one column). If that harness is inline in `test_ks_smoothing.py`, factor the minimal piece into `tests/beams/_ks_fd_harness.py` and import from both. Do not duplicate logic.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_ks_smoothing.py -k brace_radius_gradient -q`
Expected: FAIL — `grad[jcol] == 0` (rings not yet credited) or harness import error.

- [ ] **Step 3: Augment `foundation_k`**

At the top of the `if foundation:` block (`laminate_sizing.py:406`), precompute the braced-segment geometry once (outside the closure):

```python
        from ..structural.buckling import k_ring_stiffness, dk_ring_dr
        braced_seg = braced_segment_mask(model)              # (n_form,) bool
        L_ring_e = adj_w.mean(axis=1)                        # perimeter spacing per elem
        s_span_e = np.array([float(np.linalg.norm(
            model.nodes[model.beam_elements[e, 0]] - model.nodes[model.beam_elements[e, 1]]))
            for e in range(n_form_f)])                       # spanwise segment length
        brace_col = dv.slice("brace_radius").start if braced else None
        brace_alpha = 3.0
```

Inside `foundation_k(x)`, after `k_e = kgeo * D22[band_of_elem]` (`laminate_sizing.py:453`):

```python
            if braced:
                rb = float(x[brace_col])
                kr = k_ring_stiffness(rb, model.E_beam, L_ring_e, s_span_e, brace_alpha)
                k_e = k_e + np.where(braced_seg, kr, 0.0)
```

And inside the per-element chain loop (`laminate_sizing.py:456-464`), append the brace column to braced elements' chains:

```python
                if braced and braced_seg[e]:
                    dkr = dk_ring_dr(float(x[brace_col]), model.E_beam,
                                     L_ring_e[e], s_span_e[e], brace_alpha)
                    ch.append((brace_col, float(dkr)))
```

(`braced_segment_mask` must be imported at the top of `laminate_sizing.py` from `.shell_model`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_ks_smoothing.py -k brace_radius_gradient -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/laminate_sizing.py tests/beams/
git commit -m "feat(bracing): credit rings in-loop via k_ring foundation augmentation (FD-through-KS verified)"
```

---

## Task 6: `brace_vm` stress constraint

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py` (constraint closures + the specs list `:1233-1278`; KS provider map `:1288-1304`).
- Test: `tests/beams/test_laminate_sizing.py`

- [ ] **Step 1: Write the failing test** — a braced sizer build registers a `brace_vm` constraint (and `brace_vm_ks` when `ks_rho` is set), with one row, feasible (≥0) at a lightly-loaded design.

```python
# tests/beams/test_laminate_sizing.py  (append)
def test_brace_vm_constraint_registered():
    from wing_design.geometry import medium_wingsail
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.laminate_sizing import LaminateSizingConfig, _constraint_names_for_test
    m = build_beam_shell_model(medium_wingsail, n_beams=8, n_levels=5,
                               lateral_bracing=True)
    cfg = LaminateSizingConfig(ply_angle_datum=(0, 0, 1), ks_rho=50.0,
                               buckling_safety_factor=1.5, use_analytic_jacobian=True,
                               beam_buckling_model="foundation", panel_width_mode="strip")
    names = _constraint_names_for_test(m, cfg)     # returns [s.name for s in specs]
    assert "brace_vm_ks" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_laminate_sizing.py -k brace_vm_constraint -q`
Expected: FAIL — `cannot import name '_constraint_names_for_test'` or `"brace_vm_ks" not in names`.

- [ ] **Step 3: Implement the constraint + KS provider + the test hook**

Define a `brace_con(x)` closure (mirror `beam_con` at the von-Mises pipeline) that recovers axial+bending stress on the brace elements and returns `1 - util` (≥0 feasible), with `brace_jac`. Append the spec (in the `if braced:` region of the specs list, after the tube specs ~`:1278`):

```python
    if braced:
        specs.append(ConstraintSpec("brace_vm", brace_con, 1, jacs.get("brace_vm"),
                                    ("limit", "sigma_allow_Pa", config.sigma_allow_Pa)))
```

Add a KS provider so it aggregates like the others (`laminate_sizing.py:1288-1304`):

```python
        ks_providers["brace_vm"] = provider_brace_vm(config.sigma_allow_Pa)
```

(Implement `provider_brace_vm` next to `provider_beam_buck`; it reuses the beam-stress provider over `model.brace_elements`.)

Expose the test hook at module scope:

```python
def _constraint_names_for_test(model, config, *, loads=None):
    """Test-only: build the spec list for (model, config) and return spec names."""
    # minimal build path that stops after `specs` is assembled; return [s.name for s in specs]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_laminate_sizing.py -k brace_vm_constraint -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/laminate_sizing.py tests/beams/test_laminate_sizing.py
git commit -m "feat(bracing): brace_vm stress constraint (+ KS provider)"
```

---

## Task 7: `design_vector_from_result` round-trips the brace block

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py:203-...` (`design_vector_from_result`); add `brace_radius` to `LaminateSizingResult` reload.
- Test: `tests/beams/test_laminate_sizing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/beams/test_laminate_sizing.py  (append)
def test_design_vector_from_result_brace_roundtrip():
    import numpy as np
    from wing_design.geometry import medium_wingsail
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.laminate_sizing import (
        LaminateSizingConfig, laminate_design_bounds, design_vector_from_result,
        _result_from_design_vector_for_test)
    m = build_beam_shell_model(medium_wingsail, n_beams=8, n_levels=5,
                               lateral_bracing=True)
    cfg = LaminateSizingConfig(brace_r_min=0.002, brace_r_max=0.05)
    lo, hi = laminate_design_bounds(m, cfg)
    x = 0.5 * (lo + hi)
    r = _result_from_design_vector_for_test(m, cfg, x)     # decode x -> result-like carrier
    x2 = design_vector_from_result(m, cfg, r)
    assert np.allclose(x2[-1], x[-1])                       # brace_radius survives the round-trip
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_laminate_sizing.py -k brace_roundtrip -q`
Expected: FAIL — brace value dropped or helper missing.

- [ ] **Step 3: Implement**

Add a `brace_radius` attribute to `LaminateSizingResult` (default `None`) and set it from `x[dv.slice("brace_radius")]` where the result is built. In `design_vector_from_result`, append the brace block last (mirroring the existing tube/hollow/core pads):

```python
    if getattr(model, "brace_elements", None) is not None:
        rb = (np.atleast_1d(result.brace_radius) if getattr(result, "brace_radius", None)
              is not None else np.array([config.brace_r_min]))
        x = np.concatenate([x, np.asarray(rb, dtype=float)])
```

(Provide `_result_from_design_vector_for_test` as a thin decode helper if one does not already exist; otherwise reuse the existing decode path.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_laminate_sizing.py -k brace_roundtrip -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/laminate_sizing.py tests/beams/test_laminate_sizing.py
git commit -m "feat(bracing): brace_radius round-trips through design_vector_from_result"
```

---

## Task 8: Full fast-suite regression + unbraced byte-identity

**Files:** none (verification task)

- [ ] **Step 1: Run the whole fast suite**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest -q -W ignore::DeprecationWarning`
Expected: all green. In particular the pre-existing `test_design_vector_from_result_roundtrip`, KS-gradient, and foundation tests must still pass (unbraced path unchanged — `braced` is False so no block, no `k_ring`, no `brace_vm`).

- [ ] **Step 2: Spot-check unbraced equivalence**

Run a tiny unbraced foundation sizer build and confirm the spec list and DV length match `main` (no `brace_*`). If anything differs when `lateral_bracing=False`, fix the `braced` guard that leaked.

- [ ] **Step 3: Commit** (only if fixes were needed)

```bash
git commit -am "test(bracing): keep unbraced path byte-identical"
```

---

## Task 9: Eigen-gated, SF-bump slam re-probe run script

**Files:**
- Create: `runs/slam_braced.py` (tracked; artifacts gitignored)
- Test: none (it is a run script; correctness is the measured result)

- [ ] **Step 1: Write the run script**

```python
# runs/slam_braced.py
"""Slam re-probe on the ring-braced topology (task #22), eigen-gated with SF bump.

Adds transverse ring frames (lateral_bracing=True) + the brace_radius DV to the
final config, then sizes against the slam envelope and uses the post-hoc eigen
solve as the acceptance gate: if lambda_cr < 1.5, raise the in-loop buckling SF
and warm-restart, up to a small cap. Usage: python runs/slam_braced.py [G]."""
from __future__ import annotations
import json, resource, sys, time
from pathlib import Path
import numpy as np

RUNS = Path(__file__).parent
sys.path.insert(0, str(RUNS))
from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import build_beam_shell_model
from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig, design_vector_from_result, laminate_result_is_feasible,
    size_beam_shell_laminate)
from wing_design.materials.unidir import PVC_H80, T700_EPOXY
import chain_rebuild
from chain_rebuild import DATUM, eigen_worst

G0 = 9.81
TH = np.radians(30.0)
GATE = 1.5            # required eigen lambda_cr
SF_STEPS = [1.5, 1.75, 2.0, 2.3]   # in-loop buckling SF ladder


def main(slam_g: float) -> None:
    accels = ((0.0, 0.0, -G0), (0.0, G0*np.sin(TH), -G0*np.cos(TH)),
              (0.0, slam_g*G0, -G0), (0.0, -slam_g*G0, -G0))
    chain_rebuild.ACCELS = accels
    P = medium_scenario(); spec = P.geometry
    envelope = sweep_envelope(build_airplane(spec), P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in envelope if ar.panels is not None
              and abs(ar.factored_normal_force_N) >= 1.0]
    model = build_beam_shell_model(spec, n_beams=16, n_levels=8, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m, core_tube=True, hollow_beams=True,
        lateral_bracing=True)
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]
    press = [panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]

    def cfg_for(sf):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02*spec.span,
            tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=sf,
            use_analytic_jacobian=True, panel_width_mode="strip", accel_vectors=accels,
            core=PVC_H80, ks_rho=50.0, optimizer="ipopt",
            beam_buckling_model="foundation", panel_d_mode="datum_ortho")

    # warm seed: the unbraced 3g geometry padded with a starting brace radius
    seed = np.load(RUNS / f"slam_{slam_g:g}g.npz") if (RUNS/f"slam_{slam_g:g}g.npz").exists() \
        else np.load(RUNS / "multistart_v2_best.npz")
    x0 = None  # let the sizer's default-start pad brace_radius; or build from seed if shapes align
    t0 = time.perf_counter(); history = []
    r = None
    for sf in SF_STEPS:
        cfg = cfg_for(sf)
        r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                     maxiter=3000, panel_pressures=press, x0=x0)
        lam = eigen_worst(model, loads, r, P.rho_kgm3)
        feas = laminate_result_is_feasible(r, cfg)
        history.append(dict(sf=sf, mass=float(r.mass_kg), conv=bool(r.converged),
                            feas=bool(feas), eigen=float(lam)))
        print(f"[braced {slam_g:g}g] sf={sf} mass={r.mass_kg:.1f} conv={r.converged} "
              f"feas={feas} eigen={lam:.2f}", flush=True)
        x0 = design_vector_from_result(model, cfg, r)   # warm-restart next rung
        if feas and r.converged and lam >= GATE:
            break
    wall = time.perf_counter() - t0
    lam = history[-1]["eigen"]
    meta = dict(slam_g=float(slam_g), gate=GATE, history=history, final=history[-1],
                wall_s=float(wall),
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30,
                brace_radius=float(np.atleast_1d(getattr(r, "brace_radius", [np.nan]))[0]))
    out = {k: getattr(r, k) for k in ("radii","t_bands","f0_bands","f45_bands","f90_bands")}
    for k in ("r_tube","t_wall","t_hollow","t_core","brace_radius"):
        v = getattr(r, k, None)
        if v is not None:
            out[k] = np.atleast_1d(v)
    np.savez(RUNS / f"slam_braced_{slam_g:g}g.npz",
             x=design_vector_from_result(model, cfg_for(history[-1]["sf"]), r),
             meta=json.dumps(meta), **out)
    passed = history[-1]["feas"] and history[-1]["conv"] and lam >= GATE
    print(f"SLAM_BRACED DONE [{slam_g:g}g] PASSED={passed} final={history[-1]} "
          f"brace_r={meta['brace_radius']*1e3:.1f}mm wall={wall:.0f}s", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 3.0)
```

- [ ] **Step 2: Smoke-run with a tiny maxiter to confirm it executes** (not a result)

Run: edit `maxiter=3000`→`maxiter=3` temporarily, `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python runs/slam_braced.py 3`; confirm it prints a `[braced 3g] sf=...` line and saves an npz without error. Revert `maxiter`.

- [ ] **Step 3: Commit**

```bash
git add runs/slam_braced.py
git commit -m "feat(bracing): eigen-gated SF-bump slam re-probe run script"
```

---

## Task 10: Production run — 3 g then 2 g (measured)

**Files:** none (measurement); records go to `docs/findings.md` via the record-finding skill.

- [ ] **Step 1: Launch 3 g (single worker, background, monitored)**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache nohup .venv/bin/python runs/slam_braced.py 3 > runs/slam_braced_3g.log 2>&1 &`
Watch for `SLAM_BRACED DONE [3g] PASSED=...`. Memory: single worker (~18 GiB peak) is safe; do not stack runs.

- [ ] **Step 2: If 3 g PASSED, launch 2 g**; else record the negative (the SF ladder exhausted without λ≥1.5 ⇒ rings via the smeared model are insufficient → escalate to the deferred analytic in-loop eigen, or per-station/larger braces).

- [ ] **Step 3: Record the finding** (record-finding skill): masses, converged+feasible+eigen for every quoted number, the SF rung that passed, `brace_radius`, measured wall-clock + peak RSS, and whether the headline now carries a slam-survival variant. Update `docs/findings.md` + decisions row; mark task #22 done.

- [ ] **Step 4: Milestone export (only if a survivor exists)** — implement `build_sized_ring_braces` in `build.py` (loft circular rings at the braced stations from `brace_radius`), export STL + worst-mode VTU, reference the regen command in the finding.

---

## Self-review

**Spec coverage:** model/rings (T1) ✓; brace DV+bounds (T3) ✓; brace mass (T4) ✓; the k_found augmentation crux + its FD-validated gradient (T2 gated, T5 wired) ✓; brace_vm constraint (T6) ✓; eigen gate + SF-bump (T9) ✓; run plan 3g→2g (T10) ✓; milestone CAD/VTU (T10.4) ✓; roundtrip (T7) ✓; non-goals respected (single solid shared radius; no analytic in-loop eigen; no webs/lattice; brace self-buckling deferred) ✓.

**Placeholder scan:** the two test-only hooks (`_objective_and_grad_for_test`, `_constraint_names_for_test`, `_result_from_design_vector_for_test`) and the KS-FD harness import are explicitly described with their contract; the implementer wires them to existing closures rather than inventing logic. `k_ring` constant `alpha=3.0` is a documented modeling choice, backstopped by the eigen gate.

**Type/name consistency:** `brace_elements`/`brace_stations` (model), `brace_radius` (DV block + result attr), `braced` (sizer bool), `braced_segment_mask`, `k_ring_stiffness`/`dk_ring_dr` — names used identically across tasks. The DV block is appended LAST in both `laminate_design_bounds` and the sizer body (called out in T3).

**Risk flagged:** if T5's smeared-foundation credit is too weak for the true global slam mode, T10.2's SF ladder will exhaust without λ≥1.5 — that is the explicit trigger to escalate to the deferred analytic in-loop eigen (spec §fallback), not a silent failure.
