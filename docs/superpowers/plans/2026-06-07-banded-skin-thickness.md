# Per-Spanwise-Band Skin Thickness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single uniform skin-thickness design variable in the CLT co-sizer with B contiguous spanwise thickness bands (opt-in via `n_skin_bands`, default 1 = current behavior), so the optimizer can taper the skin and shed tip mass.

**Architecture:** A `skin_band_map(model, n_bands)` assigns each skin triangle to a contiguous spanwise band (triangle `e` is at level `(e//2)//n_beams`). The laminate sizer's design vector replaces the lone `t` with `B` band thicknesses; per-triangle thickness `t_tri = t_band[band_of_tri]` drives per-triangle CLT stiffness, panel buckling, and skin mass. `n_skin_bands=1` reproduces today's design numerically.

**Tech Stack:** Python, numpy, scipy SLSQP (`scipy.optimize.minimize`), pytest, `uv run pytest`. Existing: `wing_design.beams.laminate_sizing.size_beam_shell_laminate`, `wing_design.beams.shell_sizing` (`skin_areas`, `beam_lengths`, `beam_mass`), `wing_design.structural.buckling.panel_buckling_utilization` (already broadcasts a per-element `t`), `wing_design.materials.unidir` (`laminate_stiffness`, `laminate_stiffness_offset`).

**Domain notes for the implementer (read once):**
- Skin triangle ordering is level-major: triangle `e` sits at spanwise level
  `(e // 2) // n_beams`, with `n_levels - 1` levels. `default_z_levels` runs tip→keel,
  so level 0 is the tip end. Band index increases with level index (band 0 = tip end).
  The sizer does not assume a physical orientation; only the example labels ends by z.
- `solve_beam_shell_laminate(nodes, elements, sections, tris, *, E_beam, G_beam,
  A_skin, D_skin, fixed_nodes, loads)` accepts `A_skin`/`D_skin` as `(3,3)` OR
  `(M,3,3)` per-triangle.
- `panel_buckling_utilization(membrane_stress, areas, *, D11, t, kc, safety_factor)`:
  `σcr = kc·π²·D11/(b²·t)`. `D11` and `t` may be `(M,)` arrays (numpy broadcasts).
- `laminate_stiffness(ply, *, f0, f45, f90, thickness) -> (A, D, C)` each `(3,3)`;
  `laminate_stiffness_offset(..., offset_deg=) -> (A, D, C)`.
- Full suite is currently **84 passed**. Run with `uv run pytest`.
- Commit trailer (every commit): `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `src/wing_design/beams/shell_sizing.py` | `skin_band_map`, `skin_band_areas` | modify |
| `src/wing_design/beams/laminate_sizing.py` | `n_skin_bands` config; banded design vector / laminate / mass / buckling; result `t_bands` | modify |
| `src/wing_design/beams/__init__.py` | export the two helpers | modify |
| `examples/33_banded_skin.py` | uniform-vs-banded comparison | **new** |
| `tests/beams/test_banded_skin.py` | tests | **new** |
| `docs/plan.md` | finding + Decisions-log row (final task) | modify |

---

## Task 1: Band-mapping helpers

**Files:**
- Modify: `src/wing_design/beams/shell_sizing.py`
- Test: `tests/beams/test_banded_skin.py`

- [ ] **Step 1: Write the failing test**

Create `tests/beams/test_banded_skin.py`:

```python
import numpy as np
import pytest

from wing_design.geometry import WingSpec
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import skin_areas, skin_band_map, skin_band_areas


def _model(n_beams=8, n_levels=6):
    return build_beam_shell_model(WingSpec(), n_beams=n_beams, n_levels=n_levels, beam_radius=0.02)


def test_band_map_single_band_all_zero():
    m = _model()
    bm = skin_band_map(m, 1)
    assert bm.shape == (m.shell_tris.shape[0],)
    assert np.all(bm == 0)


def test_band_map_contiguous_and_complete():
    m = _model(n_beams=8, n_levels=6)  # 5 levels -> bands
    n_bands = 3
    bm = skin_band_map(m, n_bands)
    assert bm.min() == 0 and bm.max() == n_bands - 1
    assert set(np.unique(bm)) == set(range(n_bands))
    # triangles at the same level share a band; bands are contiguous in level
    n_beams = m.n_beams
    e = np.arange(m.shell_tris.shape[0])
    level = (e // 2) // n_beams
    for lv in range(m.n_levels - 1):
        bands_at_level = np.unique(bm[level == lv])
        assert bands_at_level.shape == (1,)
    # band index is non-decreasing with level
    level_band = np.array([bm[level == lv][0] for lv in range(m.n_levels - 1)])
    assert np.all(np.diff(level_band) >= 0)


def test_band_map_invalid_raises():
    m = _model(n_beams=8, n_levels=6)  # 5 segment-levels
    with pytest.raises(ValueError):
        skin_band_map(m, 0)
    with pytest.raises(ValueError):
        skin_band_map(m, 6)  # > n_levels-1


def test_band_areas_sum_to_total():
    m = _model(n_beams=8, n_levels=6)
    bm = skin_band_map(m, 3)
    ba = skin_band_areas(m, bm, 3)
    assert ba.shape == (3,)
    assert np.isclose(ba.sum(), skin_areas(m).sum())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/beams/test_banded_skin.py -v`
Expected: FAIL with `ImportError: cannot import name 'skin_band_map'`

- [ ] **Step 3: Implement**

In `src/wing_design/beams/shell_sizing.py`, add after `skin_areas`:

```python
def skin_band_map(model: BeamShellModel, n_bands: int) -> np.ndarray:
    """(n_tris,) spanwise band index in [0, n_bands) for each skin triangle.

    The ``n_levels - 1`` spanwise levels are split into ``n_bands`` contiguous,
    near-equal groups; triangle ``e`` (at level ``(e//2)//n_beams``) inherits its
    level's band. ``n_bands == 1`` returns all zeros. Band index increases with level
    index (level 0 = tip end, since ``default_z_levels`` runs tip->keel).
    """
    n_seg_levels = model.n_levels - 1
    if not (1 <= n_bands <= n_seg_levels):
        raise ValueError(f"n_bands must be in [1, n_levels-1={n_seg_levels}], got {n_bands}")
    level_band = np.empty(n_seg_levels, dtype=int)
    for bi, group in enumerate(np.array_split(np.arange(n_seg_levels), n_bands)):
        level_band[group] = bi
    e = np.arange(model.shell_tris.shape[0])
    level = (e // 2) // model.n_beams
    return level_band[level]


def skin_band_areas(model: BeamShellModel, band_of_tri: np.ndarray, n_bands: int) -> np.ndarray:
    """(n_bands,) total skin-triangle area in each band."""
    areas = skin_areas(model)
    out = np.zeros(n_bands)
    np.add.at(out, np.asarray(band_of_tri, dtype=int), areas)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/beams/test_banded_skin.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/shell_sizing.py tests/beams/test_banded_skin.py
git commit -m "$(cat <<'EOF'
Banded skin: skin_band_map + skin_band_areas helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Panel-buckling array-`t` unit test

**Files:**
- Test: `tests/beams/test_banded_skin.py` (append)

**Context:** The banded sizer passes a per-triangle thickness array to
`panel_buckling_utilization`. The function already broadcasts, but we pin that contract
with a direct test (no production code change in this task).

- [ ] **Step 1: Append the test**

```python
from wing_design.structural.buckling import panel_buckling_utilization


def test_panel_buckling_accepts_array_t():
    # two compressive panels
    ms = np.array([[-1.0e7, 0.0, 0.0], [-1.0e7, 0.0, 0.0]])
    areas = np.array([0.01, 0.01])
    D11 = 50.0
    u_scalar = panel_buckling_utilization(ms, areas, D11=D11, t=0.002, kc=4.0, safety_factor=1.0)
    u_array = panel_buckling_utilization(ms, areas, D11=D11, t=np.array([0.002, 0.002]),
                                         kc=4.0, safety_factor=1.0)
    assert np.allclose(u_scalar, u_array)
    # With D11 held fixed, sigma_cr = kc*pi^2*D11/(b^2*t) so utilization scales as t
    # element-wise: halving t halves utilization. (Confirms array-t broadcasting.)
    u_thin = panel_buckling_utilization(ms, areas, D11=D11, t=np.array([0.002, 0.001]),
                                        kc=4.0, safety_factor=1.0)
    assert np.isclose(u_thin[1], 0.5 * u_thin[0])
```

- [ ] **Step 2: Run to verify pass**

Run: `uv run pytest tests/beams/test_banded_skin.py -k panel_buckling -v`
Expected: PASS (the function already supports array `t`). If it does NOT pass, STOP and report — the spec assumed broadcasting; do not silently change `buckling.py` without flagging.

- [ ] **Step 3: Commit**

```bash
git add tests/beams/test_banded_skin.py
git commit -m "$(cat <<'EOF'
Banded skin: pin panel_buckling array-t contract

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Banded skin thickness in the CLT sizer

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py`
- Test: `tests/beams/test_banded_skin.py` (append)

**Context:** Core change. Add `n_skin_bands: int = 1` to the config and `t_bands:
np.ndarray` to the result. Replace the lone thickness DV with B band thicknesses;
per-triangle thickness drives per-triangle CLT stiffness, panel buckling, and mass.
Read the current `size_beam_shell_laminate` fully before editing — you are replacing its
body while preserving the layup/datum/buckling logic.

- [ ] **Step 1: Append the failing test**

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


def test_uniform_band_one_is_consistent():
    P, spec, model, loads = _scenario_small()
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0)  # n_skin_bands defaults to 1
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=15)
    assert r.t_bands.shape == (1,)
    assert np.isclose(r.t_skin, r.t_bands[0])
    # skin mass identity: rho * t_skin (mean) * total area
    from wing_design.beams.shell_sizing import skin_areas
    assert np.isclose(r.skin_mass_kg, P.rho_kgm3 * r.t_skin * skin_areas(model).sum(), rtol=1e-6)


def test_banded_feasible_and_not_heavier():
    P, spec, model, loads = _scenario_small()
    base_cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0, buckling_safety_factor=1.5)
    band_cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0, buckling_safety_factor=1.5,
        n_skin_bands=4)
    uni = size_beam_shell_laminate(model, loads, base_cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=40)
    ban = size_beam_shell_laminate(model, loads, band_cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=40)
    assert ban.t_bands.shape == (4,)
    assert np.all(ban.t_bands >= band_cfg.t_min - 1e-9)
    assert np.all(ban.t_bands <= band_cfg.t_max + 1e-9)
    # banded has >= the freedom of uniform; allow a small SLSQP slack
    assert ban.mass_kg <= uni.mass_kg + 0.25
    # skin mass identity with per-triangle thickness
    from wing_design.beams.shell_sizing import skin_areas, skin_band_map
    bm = skin_band_map(model, 4)
    t_tri = ban.t_bands[bm]
    assert np.isclose(ban.skin_mass_kg, P.rho_kgm3 * np.sum(t_tri * skin_areas(model)), rtol=1e-6)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/beams/test_banded_skin.py -k "band_one or banded_feasible" -v`
Expected: FAIL with `AttributeError: 'LaminateSizingResult' object has no attribute 't_bands'`

- [ ] **Step 3: Implement**

3a. Extend the shell_sizing import to include the band helpers:
```python
from .shell_sizing import beam_lengths, beam_mass, skin_areas, skin_band_areas, skin_band_map, skin_mass
```

3b. Add `n_skin_bands` to `LaminateSizingConfig` (after `ply_angle_datum`):
```python
    ply_angle_datum: tuple[float, float, float] | None = None
    n_skin_bands: int = 1
```

3c. Add `t_bands` to `LaminateSizingResult` (immediately after `t_skin`):
```python
    radii: np.ndarray
    t_skin: float
    t_bands: np.ndarray
    f0: float
    f45: float
    f90: float
    mass_kg: float
    beam_mass_kg: float
    skin_mass_kg: float
    converged: bool
    n_iter: int
    max_beam_vm_Pa: float
    max_skin_vm_Pa: float
    tip_defl_m: float
    tip_twist_deg: float
    max_beam_buckling_util: float
    max_panel_buckling_util: float
```

3d. Replace the ENTIRE body of `size_beam_shell_laminate` with:

```python
def size_beam_shell_laminate(
    model: BeamShellModel,
    load_arrays: list[np.ndarray],
    config: LaminateSizingConfig,
    *,
    ply: UDPly,
    rho: float,
    maxiter: int = 80,
    ftol: float = 1.0e-4,
) -> LaminateSizingResult:
    if not load_arrays:
        raise ValueError("load_arrays is empty")
    n = model.beam_elements.shape[0]
    B = config.n_skin_bands
    Lb = beam_lengths(model)
    Atri = skin_areas(model)
    A_skin_area = float(Atri.sum())
    band_of_tri = skin_band_map(model, B)            # (M,)
    band_area = skin_band_areas(model, band_of_tri, B)  # (B,)
    # design vector: [long radii (n), t_band (B), f0, f45]
    nx = n + B + 2
    fi0 = n + B          # index of f0
    fi45 = n + B + 1     # index of f45
    x0 = np.concatenate([np.full(n, config.r_max), np.full(B, config.t_max), [1.0 / 3.0, 1.0 / 3.0]])
    datum_offsets_deg = (
        np.degrees(skin_datum_angles(model, config.ply_angle_datum))
        if config.ply_angle_datum is not None else None
    )
    cache: dict = {}

    def evaluate(x):
        key = np.asarray(x, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        radii = x[:n]
        t_band_vec = x[n:n + B]
        t_tri = t_band_vec[band_of_tri]              # (M,)
        f0 = float(x[fi0])
        f45 = float(x[fi45])
        f90 = max(0.0, 1.0 - f0 - f45)
        if datum_offsets_deg is None:
            band_mats = [laminate_stiffness(ply, f0=f0, f45=f45, f90=f90, thickness=float(tb))
                         for tb in t_band_vec]
            A_band = np.stack([m[0] for m in band_mats])
            D_band = np.stack([m[1] for m in band_mats])
            C_band = np.stack([m[2] for m in band_mats])
            A_arg = A_band[band_of_tri]
            D_arg = D_band[band_of_tri]
            C_arg = C_band[band_of_tri]
        else:
            mats = [laminate_stiffness_offset(ply, f0=f0, f45=f45, f90=f90,
                                              thickness=float(tt), offset_deg=float(o))
                    for tt, o in zip(t_tri, datum_offsets_deg)]
            A_arg = np.stack([m[0] for m in mats])
            D_arg = np.stack([m[1] for m in mats])
            C_arg = np.stack([m[2] for m in mats])
        D11 = D_arg[:, 0, 0]                          # (M,) per-triangle plate stiffness
        sections = [BeamSection.circular(float(r)) for r in radii]
        wb = np.zeros(n)
        ws = 0.0
        md = 0.0
        mt = 0.0
        worst_beam_buck = 0.0
        worst_panel_buck = 0.0
        for loads in load_arrays:
            res = solve_beam_shell_laminate(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
                fixed_nodes=model.fixed_nodes, loads=loads,
            )
            wb = np.maximum(wb, von_mises_per_element(res, sections))
            skin_s = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=C_arg)
            ws = max(ws, float(membrane_von_mises(skin_s).max()))
            tip = model.tip_nodes
            md = max(md, float(np.linalg.norm(res.displacements[tip, :3], axis=1).max()))
            mt = max(mt, float(np.degrees(np.abs(res.displacements[tip, 5]).max())))
            if config.buckling_safety_factor is not None:
                bu = beam_euler_utilization(res.axial_force, radii, Lb, E=model.E_beam,
                                            K=config.euler_K, safety_factor=config.buckling_safety_factor)
                pu = panel_buckling_utilization(skin_s, Atri, D11=D11, t=t_tri,
                                                kc=config.panel_kc, safety_factor=config.buckling_safety_factor)
                worst_beam_buck = max(worst_beam_buck, float(bu.max()))
                worst_panel_buck = max(worst_panel_buck, float(pu.max()))
        out = (wb, ws, md, mt, worst_beam_buck, worst_panel_buck)
        cache[key] = out
        return out

    m_ref = beam_mass(model, np.full(n, config.r_max), rho=rho) + skin_mass(model, config.t_max, rho=rho)

    def mass(x):
        m = rho * np.sum(np.pi * x[:n] ** 2 * Lb) + rho * np.sum(x[n:n + B] * band_area)
        return m / m_ref

    def mass_grad(x):
        g = np.zeros(nx)
        g[:n] = rho * 2.0 * np.pi * x[:n] * Lb / m_ref
        g[n:n + B] = rho * band_area / m_ref
        return g

    def beam_con(x):
        wb, _, _, _, _, _ = evaluate(x)
        return 1.0 - wb / config.sigma_allow_Pa

    def skin_con(x):
        _, ws, _, _, _, _ = evaluate(x)
        return np.array([1.0 - ws / config.sigma_allow_Pa])

    def defl_con(x):
        _, _, d, _, _, _ = evaluate(x)
        return np.array([1.0 - d / config.tip_defl_max_m])

    def twist_con(x):
        _, _, _, t, _, _ = evaluate(x)
        return np.array([1.0 - t / config.tip_twist_max_deg])

    def frac_con(x):
        return np.array([1.0 - (x[fi0] + x[fi45])])

    bounds = (
        [(config.r_min, config.r_max)] * n
        + [(config.t_min, config.t_max)] * B
        + [(0.0, 1.0), (0.0, 1.0)]
    )
    constraints = [
        {"type": "ineq", "fun": beam_con},
        {"type": "ineq", "fun": skin_con},
        {"type": "ineq", "fun": defl_con},
        {"type": "ineq", "fun": twist_con},
        {"type": "ineq", "fun": frac_con},
    ]
    if config.buckling_safety_factor is not None:
        def beam_buck_con(x):
            *_, bbu, _ = evaluate(x)
            return np.array([1.0 - bbu])
        def panel_buck_con(x):
            *_, pbu = evaluate(x)
            return np.array([1.0 - pbu])
        constraints += [{"type": "ineq", "fun": beam_buck_con}, {"type": "ineq", "fun": panel_buck_con}]
    res = minimize(
        mass, x0, jac=mass_grad, method="SLSQP", bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol},
    )

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = np.clip(res.x, lo, hi)
    s = x[fi0] + x[fi45]
    if s > 1.0:
        x[fi0] /= s
        x[fi45] /= s
    wb, ws, d, t, bbu, pbu = evaluate(x)
    radii = x[:n]
    t_bands = x[n:n + B].copy()
    t_tri = t_bands[band_of_tri]
    f0 = float(x[fi0])
    f45 = float(x[fi45])
    f90 = max(0.0, 1.0 - f0 - f45)
    bm = beam_mass(model, radii, rho=rho)
    sm = float(rho * np.sum(t_tri * Atri))
    t_skin_mean = float(np.sum(t_tri * Atri) / A_skin_area)
    return LaminateSizingResult(
        radii=radii, t_skin=t_skin_mean, t_bands=t_bands, f0=f0, f45=f45, f90=f90,
        mass_kg=bm + sm, beam_mass_kg=bm, skin_mass_kg=sm,
        converged=bool(res.success), n_iter=int(res.nit),
        max_beam_vm_Pa=float(wb.max()), max_skin_vm_Pa=float(ws),
        tip_defl_m=float(d), tip_twist_deg=float(t),
        max_beam_buckling_util=float(bbu),
        max_panel_buckling_util=float(pbu),
    )
```

IMPORTANT: keep all existing imports at the top of the file (`laminate_stiffness`,
`laminate_stiffness_offset`, `BeamSection`, `von_mises_per_element`,
`recover_membrane_stress_C`, `membrane_von_mises`, `beam_euler_utilization`,
`panel_buckling_utilization`, `skin_datum_angles`, `minimize`). Only the `from
.shell_sizing import ...` line changes (3a).

- [ ] **Step 4: Run the new sizer tests**

Run: `uv run pytest tests/beams/test_banded_skin.py -k "band_one or banded_feasible" -v`
Expected: PASS (2 tests). Then the whole file: `uv run pytest tests/beams/test_banded_skin.py -v`

- [ ] **Step 5: Run the FULL suite — backward-compat gate**

Run: `uv run pytest -q`
Expected: ALL pass (prior 84 + new banded tests). The `n_skin_bands=1` path builds
per-triangle stiffness with all-equal thickness, which assembles the identical global
stiffness as the prior single-laminate path, so existing laminate-sizing tests and
`examples/32` semantics are preserved.

Run: `grep -rn "LaminateSizingResult(" src/ examples/ tests/`
Expected: only the single construction in `src/wing_design/beams/laminate_sizing.py`.
(If any other site constructs it positionally, STOP and report — the new `t_bands`
field would break it. As of `main` there are none.)

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/laminate_sizing.py tests/beams/test_banded_skin.py
git commit -m "$(cat <<'EOF'
Banded skin: per-spanwise-band thickness in the CLT sizer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Export the band helpers

**Files:**
- Modify: `src/wing_design/beams/__init__.py`
- Test: `tests/beams/test_banded_skin.py` (append)

- [ ] **Step 1: Append the failing test**

```python
def test_band_helpers_exported():
    import wing_design.beams as b
    assert hasattr(b, "skin_band_map")
    assert hasattr(b, "skin_band_areas")
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/beams/test_banded_skin.py -k exported -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `src/wing_design/beams/__init__.py`, add `skin_band_areas` and `skin_band_map` to the
`from .shell_sizing import (...)` block (alphabetical-ish, after `beam_mass`):
```python
from .shell_sizing import (
    BeamShellSizingConfig,
    BeamShellSizingResult,
    beam_lengths,
    beam_mass,
    size_beam_shell,
    skin_areas,
    skin_band_areas,
    skin_band_map,
    skin_mass,
)
```
and add to `__all__`:
```python
    "skin_band_map",
    "skin_band_areas",
```

- [ ] **Step 4: Run to verify pass + full suite**

Run: `uv run pytest tests/beams/test_banded_skin.py -k exported -v`  → PASS
Run: `uv run pytest -q`  → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/__init__.py tests/beams/test_banded_skin.py
git commit -m "$(cat <<'EOF'
Banded skin: export band-map helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Uniform-vs-banded comparison example

**Files:**
- Create: `examples/33_banded_skin.py`

**Context:** Mirror `examples/32_final_design.py`'s setup; size uniform (B=1) and banded
(B=4) under identical constraints and print the comparison. Examples are run manually
(not pytest).

- [ ] **Step 1: Create `examples/33_banded_skin.py`**

```python
"""Per-spanwise-band skin thickness: does tapering the skin beat a uniform skin?

Sizes the span-datum CLT design (as in examples/32) twice under identical constraints:
uniform skin (n_skin_bands=1) vs 4 spanwise thickness bands. Prints the per-band
thicknesses and a mass comparison. FEA-only (no CAD).
"""
from __future__ import annotations

import numpy as np

from wing_design import default_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell_laminate,
    skin_band_map,
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

    def cfg(nb):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim,
            ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=sf_buck, n_skin_bands=nb)

    # which band is the tip end vs keel end? (level 0 = tip per default_z_levels)
    bm4 = skin_band_map(model, 4)
    tri_z = model.nodes[model.shell_tris].mean(axis=1)[:, 2]
    band_mean_z = [float(tri_z[bm4 == k].mean()) for k in range(4)]

    print("sizing uniform skin (1 band)...")
    uni = size_beam_shell_laminate(model, load_arrays, cfg(1), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=200)
    print("sizing banded skin (4 bands)...")
    ban = size_beam_shell_laminate(model, load_arrays, cfg(4), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=200)

    def show(tag, r):
        print(f"\n  [{tag}] converged={r.converged} iters={r.n_iter}")
        print(f"    TOTAL MASS = {r.mass_kg:6.2f} kg  (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f})")
        if r.t_bands.shape[0] == 1:
            print(f"    skin thickness (uniform) = {r.t_skin*1e3:.2f} mm")
        else:
            bands = "  ".join(f"{t*1e3:.2f}" for t in r.t_bands)
            print(f"    skin t by band [{bands}] mm (mean {r.t_skin*1e3:.2f}); band z-mean {['%.1f'%z for z in band_mean_z]}")
        print(f"    tip defl {r.tip_defl_m*1e3:.1f}/{defl_lim*1e3:.0f} mm | twist {r.tip_twist_deg:.2f}/{twist_lim:.0f} deg")
        print(f"    buckling util: beam {r.max_beam_buckling_util:.2f} | panel {r.max_panel_buckling_util:.2f}")

    show("uniform", uni)
    show("banded", ban)

    dmass = ban.mass_kg - uni.mass_kg
    print(f"\n  Δmass {dmass:+.2f} kg ({100*dmass/uni.mass_kg:+.1f}%)")
    verdict = ("lighter" if dmass < -0.05 else "heavier" if dmass > 0.05 else "≈ equal")
    print(f"  Verdict: banded skin is {verdict}. (band z-mean low = keel/root, high = tip)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example end-to-end**

Run: `uv run python examples/33_banded_skin.py`
Expected: completes (a few minutes — two full sizings), prints both `[uniform]` and
`[banded]` blocks, the per-band thicknesses, and the `Δmass` + `Verdict`. If both report
`converged=False` that is still a completing run — report it, don't treat as a crash.

- [ ] **Step 3: CAPTURE THE FULL OUTPUT** verbatim into your report (Task 6 needs the numbers).

- [ ] **Step 4: Commit**

```bash
git add examples/33_banded_skin.py
git commit -m "$(cat <<'EOF'
Banded skin: uniform-vs-banded comparison example

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Record the finding in the living plan

**Files:**
- Modify: `docs/plan.md`

**Context:** Add a status note under the Phase-E / `### Phase F` area (place it right
after the combined-final-design note, before `### Phase F`) and a Decisions-log row.
Use the ACTUAL numbers from the Task 5 run — do not invent.

- [ ] **Step 1: Add the status note** (find the paragraph ending "...per-tri-local +
buckling reached 26.06 kg)..." — the combined-final-design note — and add AFTER it):

```markdown
**Per-band skin thickness done (2026-06-07).** The CLT sizer's single uniform skin
thickness is now an opt-in set of B contiguous spanwise thickness bands
(`LaminateSizingConfig.n_skin_bands`, default 1 = unchanged; `beams.skin_band_map`).
`examples/33_banded_skin.py` compares uniform vs. 4 bands at equal constraints (span
datum + buckling SF 1.5, n_beams=16, n_levels=8): uniform [Mu] kg vs banded [Mb] kg
([±X]%); band thicknesses [t0 t1 t2 t3] mm (tip→keel: [...]). [One-sentence verdict on
the taper and the saving, consistent with the skin/buckling-dominated picture.]
```

- [ ] **Step 2: Add the Decisions-log row** (after the `Combined final design` row):

```markdown
| Per-band skin thickness | Skin thickness split into B contiguous spanwise bands (opt-in `n_skin_bands`, default 1 = uniform/unchanged; `beams.skin_band_map`), each a thickness DV in the SLSQP laminate loop; per-triangle thickness drives CLT stiffness, panel buckling, and mass. Uniform [Mu] kg → 4-band [Mb] kg ([±X]%), taper [t0..t3] mm tip→keel. [Verdict]. Per-ply Tsai-Wu + per-band layup deferred. |
```

- [ ] **Step 3: Commit**

```bash
git add docs/plan.md
git commit -m "$(cat <<'EOF'
Banded skin: record finding in plan

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification (after all tasks)

- [ ] `uv run pytest -q` — expect prior 84 + new `test_banded_skin.py`, all green.
- [ ] `grep -rn "LaminateSizingResult(" src/ examples/ tests/` shows only the one site in `laminate_sizing.py`.
- [ ] Spot-confirm `examples/32_final_design.py` still imports/reads `r.t_skin` cleanly (it does; `t_skin` retained).
- [ ] Dispatch a final code review over the whole diff, then use superpowers:finishing-a-development-branch.

## Notes / risks for the implementer

- **Backward compatibility (HARD):** `n_skin_bands=1` must reproduce the current design.
  The per-triangle-with-equal-thickness assembly is mathematically identical to the old
  single-laminate path; the full suite passing (84) is the gate.
- **`t_skin` is now the area-weighted mean thickness** (equals the single value at B=1),
  retained so `examples/32` and any other `r.t_skin` reader keep working.
- **Layup stays global** (one `f0/f45/f90`); only thickness is banded. Do not add
  per-band layup — that is explicitly out of scope.
- **Band ordering** is by level index (band 0 = tip end). The sizer is orientation-
  agnostic; only the example labels ends via node z.
