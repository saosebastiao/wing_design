# Per-band Skin Layup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Opt-in per-band skin layup — each `n_skin_bands` band gets its own `(f0,f45,f90)` — composing with von-Mises and Tsai-Wu. Mass win is INDIRECT (per-band fibre mix lets thickness bands shrink).

**Architecture:** Generalize the Tsai-Wu batch to accept per-triangle `(M,)` fractions (Task 1), then add `per_band_layup` to the sizer: the 2 global layup DVs become `2*L` group DVs (`L=B` when on, else 1), the fraction-simplex becomes `L` constraints, and the result reports `f*_bands` (Task 2). Default off = byte-identical to today.

**Tech Stack:** numpy, scipy SLSQP, pytest.

**Domain notes:**
- Current sizer DV layout: `x = [radii(n), t_band(B), f0, f45]`, `nx = n+B+2`, `fi0=n+B`, `fi45=n+B+1`. This plan generalizes the layup portion to `L` groups.
- Mass is fraction-independent (objective/grad depend only on radii + thickness) — DO NOT change `mass`/`mass_grad`.
- `band_of_tri = skin_band_map(model, B)` maps each triangle to its band; `f0_band_vec[band_of_tri]` gives per-triangle fractions.
- `laminate_min_strength_ratio_batch(ply, eps, *, f0, f45, f90, offset_deg)` exists and is vectorized over M triangles.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1: Tsai-Wu batch accepts per-triangle fractions

**Files:**
- Modify: `src/wing_design/materials/failure.py`
- Test: `tests/materials/test_failure.py` (append)

**Context:** Generalize `laminate_min_strength_ratio_batch` so `f0/f45/f90` may be scalars OR `(M,)` arrays, gating each orientation element-wise. Scalar inputs must reduce to today's behavior exactly (existing equivalence tests stay green).

- [ ] **Step 1: Append failing test to `tests/materials/test_failure.py`:**

```python
def test_batch_array_fractions_match_per_element():
    rng = np.random.default_rng(7)
    M = 16
    eps = rng.normal(scale=1e-3, size=(M, 3))
    offs = rng.uniform(-90.0, 90.0, size=M)
    f0 = rng.uniform(0.1, 0.6, size=M)
    f45 = rng.uniform(0.1, 0.6, size=M) * (1.0 - f0)
    f90 = 1.0 - f0 - f45
    from wing_design.materials.failure import laminate_min_strength_ratio_batch, laminate_min_strength_ratio
    batch = laminate_min_strength_ratio_batch(PLY, eps, f0=f0, f45=f45, f90=f90, offset_deg=offs)
    per = np.array([
        laminate_min_strength_ratio(PLY, eps[i], f0=float(f0[i]), f45=float(f45[i]), f90=float(f90[i]),
                                    offset_deg=float(offs[i]))
        for i in range(M)
    ])
    assert batch.shape == (M,)
    assert np.allclose(batch, per, rtol=1e-9, atol=0.0)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/materials/test_failure.py -k array_fractions -v`
Expected: FAIL (the current batch can't take array fractions correctly — likely a shape/gating error or mismatch).

- [ ] **Step 3: Replace `laminate_min_strength_ratio_batch`** in `src/wing_design/materials/failure.py` with the per-element-gated version (keep `_strength_ratio_array`, `laminate_min_strength_ratio` delegate, and all scalar helpers unchanged):

```python
def laminate_min_strength_ratio_batch(
    ply: UDPly, eps_local, *, f0, f45, f90, offset_deg,
) -> np.ndarray:
    """(M,) min Tsai-Wu strength ratio per triangle over the present ply orientations.

    ``eps_local`` is (M,3) element-local engineering strain. ``offset_deg`` and the
    fractions ``f0/f45/f90`` may each be a scalar (broadcast) or a length-M array (the
    per-band-layup case). Each of the 0/+45/-45/90 orientations is gated element-wise by
    its fraction (>1e-9 to count as present); the min is over present orientations.
    Vectorized equivalent of ``laminate_min_strength_ratio`` per row.
    """
    eps = np.asarray(eps_local, dtype=float)
    if eps.ndim != 2 or eps.shape[1] != 3:
        raise ValueError(f"eps_local must be (M,3), got {eps.shape}")
    M = eps.shape[0]
    off = np.broadcast_to(np.asarray(offset_deg, dtype=float), (M,))
    f0a = np.broadcast_to(np.asarray(f0, dtype=float), (M,))
    f45a = np.broadcast_to(np.asarray(f45, dtype=float), (M,))
    f90a = np.broadcast_to(np.asarray(f90, dtype=float), (M,))
    ex, ey, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    Q = reduced_stiffness_Q(ply)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    coeffs = tsai_wu_coefficients(ply)
    eps_tol = 1.0e-9

    orientations = ((0.0, f0a), (45.0, f45a), (-45.0, f45a), (90.0, f90a))
    ratios = np.full((len(orientations), M), _SAFE_R)
    for k, (base, frac) in enumerate(orientations):
        th = np.radians(base + off)
        c, s = np.cos(th), np.sin(th)
        e1 = ex * c * c + ey * s * s + gxy * s * c
        e2 = ex * s * s + ey * c * c - gxy * s * c
        g12 = -2.0 * ex * s * c + 2.0 * ey * s * c + gxy * (c * c - s * s)
        s1 = Q11 * e1 + Q12 * e2
        s2 = Q12 * e1 + Q22 * e2
        t12 = Q66 * g12
        r = _strength_ratio_array(s1, s2, t12, coeffs)
        ratios[k] = np.where(frac > eps_tol, r, _SAFE_R)
    return ratios.min(axis=0)
```

- [ ] **Step 4: Run failure tests**

Run: `uv run pytest tests/materials/test_failure.py -v`
Expected: PASS — the new array-fraction test AND all existing batch/scalar tests
(`test_batch_matches_scalar`, `test_batch_skips_absent_orientations`,
`test_batch_handles_zero_and_shear_rows`, and the scalar calibration tests).

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/materials/failure.py tests/materials/test_failure.py
git commit -m "$(cat <<'EOF'
Per-band layup: Tsai-Wu batch accepts per-triangle fractions

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: per_band_layup in the sizer

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py`
- Test: `tests/beams/test_per_band_layup.py` (new)

**Context:** Add `per_band_layup: bool = False` and `f0_bands/f45_bands/f90_bands` to the
result, and generalize `size_beam_shell_laminate`'s layup DVs to `L` groups. Read the
whole function first. `per_band_layup=False` must reproduce today's results.

- [ ] **Step 1: Write the failing tests** — create `tests/beams/test_per_band_layup.py`
(note: use `small_scenario` — `default_scenario` no longer exists):

```python
import numpy as np

from wing_design.scenario import small_scenario
from wing_design.beams import (
    LaminateSizingConfig, build_beam_frame, build_beam_shell_model,
    project_panels_to_beam_nodes, size_beam_shell_laminate,
)
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.materials.unidir import T700_EPOXY


def _small(n_beams=8, n_levels=5):
    P = small_scenario()
    spec = P.geometry
    model = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    airplane = build_airplane(spec)
    env = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    return P, spec, model, loads


def test_global_layup_default_bands_broadcast():
    P, spec, model, loads = _small()
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0, n_skin_bands=3)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=15)
    assert r.f0_bands.shape == (3,)
    assert np.allclose(r.f0_bands, r.f0_bands[0])  # global layup -> all bands equal
    assert np.isclose(r.f0, r.f0_bands[0])


def test_per_band_layup_feasible():
    P, spec, model, loads = _small()
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        buckling_safety_factor=1.5, n_skin_bands=3, per_band_layup=True)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=40)
    assert r.f0_bands.shape == (3,) and r.f45_bands.shape == (3,) and r.f90_bands.shape == (3,)
    assert np.all(r.f0_bands >= -1e-9) and np.all(r.f45_bands >= -1e-9) and np.all(r.f90_bands >= -1e-9)
    assert np.allclose(r.f0_bands + r.f45_bands + r.f90_bands, 1.0, atol=1e-6)


def test_per_band_layup_with_tsai_wu():
    P, spec, model, loads = _small()
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        ply_angle_datum=(0.0, 0.0, 1.0), skin_failure="tsai_wu", tsai_wu_safety_factor=2.0,
        n_skin_bands=3, per_band_layup=True)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=40)
    assert r.min_skin_strength_ratio is not None
    assert r.min_skin_strength_ratio >= 2.0 - 0.05
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/beams/test_per_band_layup.py -v`
Expected: FAIL with `AttributeError: ... 'f0_bands'` (or a config TypeError on `per_band_layup`).

- [ ] **Step 3a: Add config field** in `LaminateSizingConfig` (after `tsai_wu_safety_factor`):
```python
    per_band_layup: bool = False
```

- [ ] **Step 3b: Add result fields** in `LaminateSizingResult` (after `f90: float`):
```python
    f0_bands: np.ndarray
    f45_bands: np.ndarray
    f90_bands: np.ndarray
```

- [ ] **Step 3c: Replace the body of `size_beam_shell_laminate`** (from `n = ...` through the `return`) with the L-group version below. It preserves all existing logic and reduces exactly to today's behavior when `per_band_layup=False` (L=1).

```python
    if not load_arrays:
        raise ValueError("load_arrays is empty")
    n = model.beam_elements.shape[0]
    B = config.n_skin_bands
    L = B if config.per_band_layup else 1
    Lb = beam_lengths(model)
    Atri = skin_areas(model)
    A_skin_area = float(Atri.sum())
    band_of_tri = skin_band_map(model, B)
    band_area = skin_band_areas(model, band_of_tri, B)
    nx = n + B + 2 * L
    f0_lo = n + B
    f45_lo = n + B + L
    x0 = np.concatenate([
        np.full(n, config.r_max), np.full(B, config.t_max),
        np.full(L, 1.0 / 3.0), np.full(L, 1.0 / 3.0),
    ])
    datum_offsets_deg = (
        np.degrees(skin_datum_angles(model, config.ply_angle_datum))
        if config.ply_angle_datum is not None else None
    )

    def band_fracs(x):
        """Per-band (length B) f0/f45/f90 from the L layup groups."""
        f0_grp = x[f0_lo:f0_lo + L]
        f45_grp = x[f45_lo:f45_lo + L]
        if config.per_band_layup:
            f0b = np.asarray(f0_grp, dtype=float)
            f45b = np.asarray(f45_grp, dtype=float)
        else:
            f0b = np.full(B, float(f0_grp[0]))
            f45b = np.full(B, float(f45_grp[0]))
        f90b = np.maximum(0.0, 1.0 - f0b - f45b)
        return f0b, f45b, f90b

    cache: dict = {}

    def evaluate(x):
        key = np.asarray(x, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        radii = x[:n]
        t_band_vec = x[n:n + B]
        t_tri = t_band_vec[band_of_tri]
        f0b, f45b, f90b = band_fracs(x)
        f0_tri = f0b[band_of_tri]
        f45_tri = f45b[band_of_tri]
        f90_tri = f90b[band_of_tri]
        if datum_offsets_deg is None:
            band_mats = [laminate_stiffness(ply, f0=float(f0b[b]), f45=float(f45b[b]),
                                            f90=float(f90b[b]), thickness=float(t_band_vec[b]))
                         for b in range(B)]
            A_band = np.stack([m[0] for m in band_mats])
            D_band = np.stack([m[1] for m in band_mats])
            C_band = np.stack([m[2] for m in band_mats])
            A_arg = A_band[band_of_tri]
            D_arg = D_band[band_of_tri]
            C_arg = C_band[band_of_tri]
        else:
            mats = [laminate_stiffness_offset(ply, f0=float(f0_tri[e]), f45=float(f45_tri[e]),
                                              f90=float(f90_tri[e]), thickness=float(t_tri[e]),
                                              offset_deg=float(o))
                    for e, o in enumerate(datum_offsets_deg)]
            A_arg = np.stack([m[0] for m in mats])
            D_arg = np.stack([m[1] for m in mats])
            C_arg = np.stack([m[2] for m in mats])
        D11 = D_arg[:, 0, 0]
        sections = [BeamSection.circular(float(r)) for r in radii]
        wb = np.zeros(n)
        ws = 0.0
        md = 0.0
        mt = 0.0
        worst_beam_buck = 0.0
        worst_panel_buck = 0.0
        worst_R = np.inf
        for loads in load_arrays:
            res = solve_beam_shell_laminate(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
                fixed_nodes=model.fixed_nodes, loads=loads,
            )
            wb = np.maximum(wb, von_mises_per_element(res, sections))
            skin_s = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=C_arg)
            ws = max(ws, float(membrane_von_mises(skin_s).max()))
            if config.skin_failure == "tsai_wu":
                eps = recover_membrane_strain(model.nodes, model.shell_tris, res.displacements)
                offs = datum_offsets_deg if datum_offsets_deg is not None else np.zeros(eps.shape[0])
                R = laminate_min_strength_ratio_batch(
                    ply, eps, f0=f0_tri, f45=f45_tri, f90=f90_tri, offset_deg=offs)
                worst_R = min(worst_R, float(R.min()))
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
        out = (wb, ws, md, mt, worst_beam_buck, worst_panel_buck, worst_R)
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

    def frac_con(x):
        f0_grp = x[f0_lo:f0_lo + L]
        f45_grp = x[f45_lo:f45_lo + L]
        return 1.0 - (np.asarray(f0_grp, dtype=float) + np.asarray(f45_grp, dtype=float))

    bounds = (
        [(config.r_min, config.r_max)] * n
        + [(config.t_min, config.t_max)] * B
        + [(0.0, 1.0)] * (2 * L)
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
            return np.array([1.0 - evaluate(x)[4]])
        def panel_buck_con(x):
            return np.array([1.0 - evaluate(x)[5]])
        constraints += [{"type": "ineq", "fun": beam_buck_con}, {"type": "ineq", "fun": panel_buck_con}]
    res = minimize(
        mass, x0, jac=mass_grad, method="SLSQP", bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol},
    )

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = np.clip(res.x, lo, hi)
    # Per-group simplex re-projection: if f0_g + f45_g > 1, rescale that group.
    fg = x[f0_lo:f0_lo + L].copy()
    hg = x[f45_lo:f45_lo + L].copy()
    scale = np.where(fg + hg > 1.0, fg + hg, 1.0)
    x[f0_lo:f0_lo + L] = fg / scale
    x[f45_lo:f45_lo + L] = hg / scale
    wb, ws, d, t, bbu, pbu, worst_R = evaluate(x)
    radii = x[:n]
    t_bands = x[n:n + B].copy()
    t_tri = t_bands[band_of_tri]
    f0b, f45b, f90b = band_fracs(x)
    f0_tri = f0b[band_of_tri]
    f45_tri = f45b[band_of_tri]
    f90_tri = f90b[band_of_tri]
    bm = beam_mass(model, radii, rho=rho)
    sm = float(rho * np.sum(t_tri * Atri))
    t_skin_mean = float(np.sum(t_tri * Atri) / A_skin_area)
    f0_mean = float(np.sum(f0_tri * Atri) / A_skin_area)
    f45_mean = float(np.sum(f45_tri * Atri) / A_skin_area)
    f90_mean = float(np.sum(f90_tri * Atri) / A_skin_area)
    return LaminateSizingResult(
        radii=radii, t_skin=t_skin_mean, t_bands=t_bands,
        f0=f0_mean, f45=f45_mean, f90=f90_mean,
        f0_bands=f0b, f45_bands=f45b, f90_bands=f90b,
        mass_kg=bm + sm, beam_mass_kg=bm, skin_mass_kg=sm,
        converged=bool(res.success), n_iter=int(res.nit),
        max_beam_vm_Pa=float(wb.max()), max_skin_vm_Pa=float(ws),
        tip_defl_m=float(d), tip_twist_deg=float(t),
        max_beam_buckling_util=float(bbu),
        max_panel_buckling_util=float(pbu),
        min_skin_strength_ratio=(float(worst_R) if config.skin_failure == "tsai_wu" else None),
    )
```

- [ ] **Step 4: Run the new tests**

Run: `uv run pytest tests/beams/test_per_band_layup.py -v`  → PASS (3 tests).

- [ ] **Step 5: FULL suite (backward-compat gate) + construction-site check**

Run: `uv run pytest -q`  → ALL pass (~8 min). The `per_band_layup=False` path is L=1,
identical layout/values to before; the Tsai-Wu call now passes constant per-triangle
arrays (equivalent to the old scalars).
Run: `grep -rn "LaminateSizingResult(" src/ examples/ tests/`  → only the one site in
`laminate_sizing.py`.

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/laminate_sizing.py tests/beams/test_per_band_layup.py
git commit -m "$(cat <<'EOF'
Per-band layup: opt-in per-band (f0,f45,f90) in the CLT sizer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Global-vs-per-band-layup example

**Files:**
- Create: `examples/35_per_band_layup.py`

**Context:** Medium wingsail, B=4 thickness bands, span datum, buckling SF 1.5. Size with
global layup vs per-band layup (both banded thickness) and compare. Not run by pytest;
gate with py_compile + the orchestrator runs it timed in the background.

- [ ] **Step 1: Create `examples/35_per_band_layup.py`:**

```python
"""Per-band skin layup vs a single global layup on the medium (22 m) wingsail.

Both runs use 4 spanwise thickness bands, the span ply-angle datum, and buckling
(SF 1.5). Per-band layup gives each band its own (f0,f45,f90); since mass is
fraction-independent, any mass win is INDIRECT -- a band tuned to its local load can
meet the buckling/twist/stress constraints at a thinner thickness. Prints the per-band
layup + thickness taper and the mass delta. FEA-only.
"""
from __future__ import annotations

from wing_design import medium_scenario
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
    P = medium_scenario()
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

    def cfg(per_band):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim,
            ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=sf_buck,
            n_skin_bands=4, per_band_layup=per_band)

    print("sizing global layup (banded thickness)...")
    g = size_beam_shell_laminate(model, load_arrays, cfg(False), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300)
    print("sizing per-band layup (banded thickness)...")
    pb = size_beam_shell_laminate(model, load_arrays, cfg(True), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=600)

    def show(tag, r):
        print(f"\n  [{tag}] converged={r.converged} iters={r.n_iter}")
        print(f"    TOTAL MASS = {r.mass_kg:6.2f} kg  (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f})")
        tb = "  ".join(f"{t*1e3:.2f}" for t in r.t_bands)
        print(f"    t by band [{tb}] mm (mean {r.t_skin*1e3:.2f})")
        for b in range(r.f0_bands.shape[0]):
            print(f"    band {b}: layup {r.f0_bands[b]*100:.0f}/{r.f45_bands[b]*100:.0f}/{r.f90_bands[b]*100:.0f} span/±45/chord")
        print(f"    tip defl {r.tip_defl_m*1e3:.1f}/{defl_lim*1e3:.0f} mm | twist {r.tip_twist_deg:.2f}/{twist_lim:.0f} deg")
        print(f"    buckling util: beam {r.max_beam_buckling_util:.2f} | panel {r.max_panel_buckling_util:.2f}")

    show("global layup", g)
    show("per-band layup", pb)

    dmass = pb.mass_kg - g.mass_kg
    print(f"\n  Δmass {dmass:+.2f} kg ({100*dmass/g.mass_kg:+.1f}%)")
    verdict = ("lighter" if dmass < -0.05 else "heavier" if dmass > 0.05 else "≈ equal")
    print(f"  Verdict: per-band layup is {verdict} (indirect: per-band fibre mix -> thinner bands).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile-check (do NOT run — slow)**

Run: `uv run python -m py_compile examples/35_per_band_layup.py && echo ok`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add examples/35_per_band_layup.py
git commit -m "$(cat <<'EOF'
Per-band layup: global-vs-per-band-layup comparison example

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

(The orchestrator will run this example timed in the background to capture the headline
numbers for Task 4 — the implementer does NOT run it.)

---

## Task 4: Record the finding (orchestrator fills numbers after the timed run)

**Files:**
- Modify: `docs/plan.md`

**Context:** After `examples/35_per_band_layup.py` runs (timed, background), add a status
note after the Tsai-Wu note and a Decisions-log row, using the ACTUAL numbers and the
measured wall-clock.

- [ ] **Step 1: Add status note** (after the "Per-ply Tsai-Wu skin failure done" note):

```markdown
**Per-band skin layup done (2026-06-08).** Opt-in `LaminateSizingConfig.per_band_layup`
gives each of the `n_skin_bands` spanwise bands its own `(f0,f45,f90)` (default off =
global layup). Since skin mass is fraction-independent, the win is indirect — a band's
fibre mix can meet the buckling/twist/stress constraints at a thinner thickness.
`examples/35_per_band_layup.py` (medium wingsail, 4 bands, datum + buckling): global
layup [Mg] kg vs per-band [Mpb] kg ([±X]%); per-band fibre taper [summarize bands].
[One-sentence verdict.] (Run wall-clock [T].)
```

- [ ] **Step 2: Add Decisions-log row** (after the "Per-ply Tsai-Wu skin failure" row):

```markdown
| Per-band skin layup | Opt-in `per_band_layup` — each n_skin_bands band gets its own (f0,f45,f90) DVs; L-group design vector + L fraction constraints; Tsai-Wu batch generalized to per-triangle fractions. Mass is fraction-independent so the win is indirect (per-band mix -> thinner bands). Global [Mg] kg vs per-band [Mpb] kg ([±X]%). [Verdict]. Datum stays global; independent layup-band-count deferred. |
```

- [ ] **Step 3: Commit**

```bash
git add docs/plan.md
git commit -m "$(cat <<'EOF'
Per-band layup: record finding in plan

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] `uv run pytest -q` — all green.
- [ ] `grep -rn "LaminateSizingResult(" src/ examples/ tests/` — only one site.
- [ ] Dispatch a final code review over the diff, then use superpowers:finishing-a-development-branch.

## Notes / risks

- **Backward compat (HARD):** `per_band_layup=False` → L=1 → identical DV layout/values
  to today; the Tsai-Wu call passes constant per-triangle arrays (equivalent to old
  scalars). Full suite green is the gate.
- **Mass is fraction-independent** — do not "fix" `mass`/`mass_grad` to include layup;
  the win is via thinner thickness bands, not the layup DVs directly.
- **Per-group simplex** both in `frac_con` (L constraints) and the post-solve reprojection.
- The example uses higher `maxiter` for the per-band run (3× the layup DVs).
