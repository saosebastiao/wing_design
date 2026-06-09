# Beam Symmetry + Monotonic Taper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `size_beam_shell_laminate` enforce, by default, mirror-symmetric beam radii (mirror-paired beams share one radius DV; auto-detected, graceful fallback) and monotonic non-increasing radius from keel to tip. Then a full sized run exporting STL meshes.

**Architecture:** A `beam_radius_groups(model)` maps each beam-element to a radius DV group (mirror beams → same group when placement is symmetric, else one group per element). The sizer's radii block becomes the G group-radii; `radii_full = r_group[group_of_element]` is used everywhere; mass-grad scatters per-element gradients into the G groups; monotonic taper is added as cheap algebraic constraints. `result.radii` is expanded back to full length n.

**Tech Stack:** numpy, scipy SLSQP, pytest; build123d for the export example.

**Domain notes:**
- Node id = `b*n_levels + k`; beam-element index = `b*(n_levels-1) + k` (beam-major). `result.radii` length `n = n_beams*(n_levels-1)`.
- Mirror across chord plane: y → −y; beam `b` ↔ `(n_beams-b)%n_beams`. Self-mirror: `b==0` (TE) and `b==n_beams//2` (LE, even n_beams).
- `default_z_levels` runs tip(high z)→keel(low z): level 0 = tip.
- Current sizer DV layout: `x = [radii(n), t_band(B), f0_grp(L), f45_grp(L)]`, `nx=n+B+2L`, `f0_lo=n+B`, `f45_lo=n+B+L`, `L = B if per_band_layup else 1`. This plan changes the radii block from `n` per-element to `G` group radii.
- `wb = np.zeros(n)` in `evaluate` is an ELEMENT count (von_mises_per_element returns n) — it STAYS `n`. Only DV-index `n` becomes `G`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1: `beam_radius_groups` helper

**Files:**
- Modify: `src/wing_design/beams/shell_sizing.py`
- Modify: `src/wing_design/beams/__init__.py`
- Test: `tests/beams/test_beam_symmetry.py` (new)

- [ ] **Step 1: Write failing tests** — create `tests/beams/test_beam_symmetry.py`:

```python
import numpy as np

from wing_design.scenario import small_scenario
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_radius_groups


def _model(n_beams=8, n_levels=5, arc_fractions=None):
    P = small_scenario()
    return build_beam_shell_model(P.geometry, n_beams=n_beams, n_levels=n_levels,
                                  beam_radius=0.02, arc_fractions=arc_fractions)


def test_symmetric_default_groups_reduced():
    m = _model(n_beams=8, n_levels=5)
    g, ng = beam_radius_groups(m)
    n = m.beam_elements.shape[0]
    assert g.shape == (n,)
    # even n_beams: U = n_beams//2 + 1 unique beams
    assert ng == (8 // 2 + 1) * (5 - 1)
    # mirror beams b and 8-b share group ids per level
    seg = 5 - 1
    for b in range(1, 4):
        mb = (8 - b) % 8
        assert np.array_equal(g[b * seg:(b + 1) * seg], g[mb * seg:(mb + 1) * seg])
    # chord-line beams 0 and 4 are NOT shared with any other beam
    g0 = set(g[0:seg]); g4 = set(g[4 * seg:5 * seg])
    others = set(g) - g0 - g4
    assert g0.isdisjoint(others) and g4.isdisjoint(others)


def test_asymmetric_placement_falls_back():
    nb, nl = 8, 5
    # deliberately asymmetric arc placement (not mirror-symmetric)
    af = np.linspace(0.0, 1.0, nb, endpoint=False) ** 1.5
    m = _model(n_beams=nb, n_levels=nl, arc_fractions=af)
    g, ng = beam_radius_groups(m)
    n = m.beam_elements.shape[0]
    assert ng == n
    assert np.array_equal(g, np.arange(n))
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/beams/test_beam_symmetry.py -k "groups or fall" -v`
Expected: FAIL with `ImportError: cannot import name 'beam_radius_groups'`.

- [ ] **Step 3: Implement** — add to `src/wing_design/beams/shell_sizing.py` (after `skin_band_areas`):

```python
def beam_radius_groups(model: BeamShellModel) -> tuple[np.ndarray, int]:
    """Map each beam-element to a radius design-variable group, exploiting chord symmetry.

    Beams are ordered TE(0) -> LE(n_beams//2) around a chord-symmetric section, so beam
    ``b`` mirrors ``(n_beams - b) % n_beams``. When the beam node positions are actually
    mirror-symmetric across the chord plane (y -> -y) -- the default even arc-spacing --
    mirror-paired beams SHARE radii: element ``(b,k)`` maps to group
    ``rank(rep(b))*(n_levels-1) + k`` where ``rep(b) = min(b, mirror(b))``. Returns
    ``(group_of_element (n,), n_groups)``. If placement is NOT mirror-symmetric (e.g.
    non-uniform arc_fractions) it falls back to one group per element
    (``arange(n), n``), so callers can stay symmetry-on by default safely.
    """
    nb = model.n_beams
    nl = model.n_levels
    seg = nl - 1
    n = model.beam_elements.shape[0]
    nodes = model.nodes
    scale = float(np.abs(nodes).max()) if nodes.size else 1.0
    tol = 1.0e-6 * max(1.0, scale)

    symmetric = True
    for b in range(nb):
        mb = (nb - b) % nb
        for k in range(nl):
            pb = nodes[b * nl + k]
            pm = nodes[mb * nl + k]
            if (abs(pb[0] - pm[0]) > tol or abs(pb[2] - pm[2]) > tol
                    or abs(pb[1] + pm[1]) > tol):
                symmetric = False
                break
        if not symmetric:
            break

    if not symmetric:
        return np.arange(n, dtype=int), n

    reps = sorted({min(b, (nb - b) % nb) for b in range(nb)})
    rank = {r: i for i, r in enumerate(reps)}
    group_of_element = np.empty(n, dtype=int)
    for b in range(nb):
        ui = rank[min(b, (nb - b) % nb)]
        for k in range(seg):
            group_of_element[b * seg + k] = ui * seg + k
    return group_of_element, len(reps) * seg
```

- [ ] **Step 4: Export** — add `beam_radius_groups` to `beams/__init__.py` (`from .shell_sizing import (...)` + `__all__`).

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/beams/test_beam_symmetry.py -k "groups or fall" -v`  → PASS (2).

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/shell_sizing.py src/wing_design/beams/__init__.py tests/beams/test_beam_symmetry.py
git commit -m "$(cat <<'EOF'
Beam symmetry: beam_radius_groups (mirror grouping + auto-detect fallback)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Symmetric radii + monotonic taper in the sizer (always-on)

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py`
- Test: `tests/beams/test_beam_symmetry.py` (append); update `tests/beams/test_multistart.py`

**Context:** Reparameterize the sizer radii to group DVs and add monotonic taper
constraints, by default. Read the whole `size_beam_shell_laminate` first.

- [ ] **Step 1: Append failing tests to `tests/beams/test_beam_symmetry.py`:**

```python
from wing_design.scenario import small_scenario as _ss
from wing_design.beams import (
    LaminateSizingConfig, build_beam_frame, build_beam_shell_model as _bm,
    project_panels_to_beam_nodes, size_beam_shell_laminate,
)
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.materials.unidir import T700_EPOXY


def _sized(n_beams=8, n_levels=5, **cfgkw):
    P = _ss()
    spec = P.geometry
    model = _bm(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    airplane = build_airplane(spec)
    env = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0, **cfgkw)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=25)
    return model, r


def test_sizer_radii_are_mirror_symmetric():
    nb, nl = 8, 5
    model, r = _sized(nb, nl, buckling_safety_factor=1.5)
    assert r.radii.shape == (nb * (nl - 1),)
    R = r.radii.reshape(nb, nl - 1)
    for b in range(1, nb // 2):
        assert np.allclose(R[b], R[(nb - b) % nb], rtol=1e-9, atol=1e-12)


def test_sizer_radii_taper_monotonic_keel_to_tip():
    nb, nl = 8, 5
    model, r = _sized(nb, nl, buckling_safety_factor=1.5)
    R = r.radii.reshape(nb, nl - 1)
    # default_z_levels is tip->keel, so segment index increases toward the keel;
    # radius must be NON-DECREASING with segment index (>= tip ... <= keel).
    for b in range(nb):
        assert np.all(np.diff(R[b]) >= -1e-9)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/beams/test_beam_symmetry.py -k "symmetric or taper" -v`
Expected: FAIL (radii not symmetric / not monotonic under the current sizer).

- [ ] **Step 3: Implement** — edits to `src/wing_design/beams/laminate_sizing.py`:

(3a) Update the import to include the grouping helper:
```python
from .shell_sizing import (
    beam_lengths, beam_mass, beam_radius_groups, skin_areas, skin_band_areas,
    skin_band_map, skin_mass,
)
```
(adjust to match the existing `from .shell_sizing import ...` — add `beam_radius_groups`.)

(3b) Update `laminate_design_bounds` to use the group count G for the radii block:
```python
def laminate_design_bounds(model: BeamShellModel, config: LaminateSizingConfig):
    """(lo, hi) bounds for [r_group(G), t_band(B), f0_grp(L), f45_grp(L)]."""
    _, G = beam_radius_groups(model)
    B = config.n_skin_bands
    L = B if config.per_band_layup else 1
    lo = np.concatenate([np.full(G, config.r_min), np.full(B, config.t_min), np.zeros(2 * L)])
    hi = np.concatenate([np.full(G, config.r_max), np.full(B, config.t_max), np.ones(2 * L)])
    return lo, hi
```

(3c) Replace the body of `size_beam_shell_laminate` (from `if not load_arrays:` through the
final `return`) with the version below. Changes vs current: compute
`group_of_element, G`; radii block is G group-radii expanded via `group_of_element`;
`nx`, `f0_lo`, `f45_lo` use G; `mass`/`mass_grad` operate on group radii (gradient
scattered with `np.add.at`); a precomputed monotonic taper constraint is added; the
result expands `radii` back to full length n.

```python
    if not load_arrays:
        raise ValueError("load_arrays is empty")
    n = model.beam_elements.shape[0]
    B = config.n_skin_bands
    L = B if config.per_band_layup else 1
    group_of_element, G = beam_radius_groups(model)
    Lb = beam_lengths(model)
    Atri = skin_areas(model)
    A_skin_area = float(Atri.sum())
    band_of_tri = skin_band_map(model, B)
    band_area = skin_band_areas(model, band_of_tri, B)
    nx = G + B + 2 * L
    f0_lo = G + B
    f45_lo = G + B + L
    if x0 is None:
        x0 = np.concatenate([
            np.full(G, config.r_max), np.full(B, config.t_max),
            np.full(L, 1.0 / 3.0), np.full(L, 1.0 / 3.0),
        ])
    else:
        lo_b, hi_b = laminate_design_bounds(model, config)
        x0 = np.clip(np.asarray(x0, dtype=float), lo_b, hi_b).copy()
        if x0.shape != (nx,):
            raise ValueError(f"x0 must have length nx={nx}, got {x0.shape}")
        fg = x0[f0_lo:f0_lo + L]
        hg = x0[f45_lo:f45_lo + L]
        scale = np.where(fg + hg > 1.0, fg + hg, 1.0)
        x0[f0_lo:f0_lo + L] = fg / scale
        x0[f45_lo:f45_lo + L] = hg / scale
    datum_offsets_deg = (
        np.degrees(skin_datum_angles(model, config.ply_angle_datum))
        if config.ply_angle_datum is not None else None
    )

    # Monotonic taper: per radius-unit, radius non-increasing keel->tip. Group DV index
    # = unit*(n_levels-1) + segment; order consecutive segments by their midpoint z.
    seg = model.n_levels - 1
    U = G // seg
    level_z = model.nodes[:model.n_levels, 2]
    seg_z = 0.5 * (level_z[:-1] + level_z[1:])
    mono_lo_list: list[int] = []
    mono_hi_list: list[int] = []
    for u in range(U):
        for k in range(seg - 1):
            ia = u * seg + k
            ib = u * seg + (k + 1)
            if seg_z[k] >= seg_z[k + 1]:   # segment k is tip-ward (higher z)
                mono_lo_list.append(ib); mono_hi_list.append(ia)
            else:
                mono_lo_list.append(ia); mono_hi_list.append(ib)
    mono_lo = np.asarray(mono_lo_list, dtype=int)
    mono_hi = np.asarray(mono_hi_list, dtype=int)

    def band_fracs(x):
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
        radii = x[:G][group_of_element]
        t_band_vec = x[G:G + B]
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
        radii = x[:G][group_of_element]
        m = rho * np.sum(np.pi * radii ** 2 * Lb) + rho * np.sum(x[G:G + B] * band_area)
        return m / m_ref

    def mass_grad(x):
        g = np.zeros(nx)
        radii = x[:G][group_of_element]
        per_elem = rho * 2.0 * np.pi * radii * Lb / m_ref
        np.add.at(g, group_of_element, per_elem)   # scatter element grads into the G groups
        g[G:G + B] = rho * band_area / m_ref
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

    def monotonic_con(x):
        return x[mono_lo] - x[mono_hi]   # >= 0 : keel-side radius >= tip-side radius

    lo_b, hi_b = laminate_design_bounds(model, config)
    bounds = [(float(lo_b[i]), float(hi_b[i])) for i in range(nx)]
    constraints = [
        {"type": "ineq", "fun": beam_con},
        {"type": "ineq", "fun": skin_con},
        {"type": "ineq", "fun": defl_con},
        {"type": "ineq", "fun": twist_con},
        {"type": "ineq", "fun": frac_con},
    ]
    if mono_lo.size > 0:
        constraints.append({"type": "ineq", "fun": monotonic_con})
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
    fg = x[f0_lo:f0_lo + L].copy()
    hg = x[f45_lo:f45_lo + L].copy()
    scale = np.where(fg + hg > 1.0, fg + hg, 1.0)
    x[f0_lo:f0_lo + L] = fg / scale
    x[f45_lo:f45_lo + L] = hg / scale
    wb, ws, d, t, bbu, pbu, worst_R = evaluate(x)
    radii = x[:G][group_of_element]
    t_bands = x[G:G + B].copy()
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

- [ ] **Step 4: Update the multistart tests for the new bounds length.** In
`tests/beams/test_multistart.py`, two tests assume the radii block is `n`:
  - `test_design_bounds_shape`: replace the expected length. With symmetry the radii block
    is `G = beam_radius_groups(model)[1]`, not `n`. Change the assertion to compute G:
    ```python
    from wing_design.beams.shell_sizing import beam_radius_groups
    _, G = beam_radius_groups(model)
    assert lo.shape == hi.shape == (G + 3 + 2 * 3,)
    assert np.all(lo[G:] == 0.0)  # ... adjust the fraction-bounds slice to start at G+3
    ```
    (Update the fraction-bound checks to index from `G` not `n`.)
  - `test_x0_roundtrip_backward_compat`: the explicit `default` vector must use `G` radii:
    ```python
    _, G = beam_radius_groups(model)
    default = np.concatenate([np.full(G, cfg.r_max), np.full(1, cfg.t_max), [1/3], [1/3]])
    ```
Keep all other multistart tests as-is (the stubbed orchestration tests don't depend on G).

- [ ] **Step 5: Run the symmetry + multistart tests**

Run: `uv run pytest tests/beams/test_beam_symmetry.py tests/beams/test_multistart.py -v`
Expected: PASS (symmetry, taper, bounds, roundtrip, stubbed orchestration).

- [ ] **Step 6: FULL suite**

Run: `uv run pytest -q`  → all green (~8 min). Fix any test that asserts SPECIFIC radius
values (shape/feasibility ones hold; symmetric/monotonic changes exact radii). Do NOT
weaken feasibility assertions.
Run: `grep -rn "LaminateSizingResult(" src/ examples/ tests/`  → only the one site.

- [ ] **Step 7: Commit**

```bash
git add src/wing_design/beams/laminate_sizing.py tests/beams/test_beam_symmetry.py tests/beams/test_multistart.py
git commit -m "$(cat <<'EOF'
Beam symmetry + monotonic taper: shared mirror radius DVs + keel->tip taper (default)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Sized mesh-export example

**Files:**
- Create: `examples/37_sized_export.py`

**Context:** Size the medium wingsail (new default constraints), build sized geometry, and
export STL meshes. Compile-checked only here; the orchestrator runs it (Task 4).

- [ ] **Step 1: Create `examples/37_sized_export.py`:**

```python
"""Size the medium (22 m) wingsail and export the resulting structure as STL meshes.

Sizes with the default mirror-symmetric + monotonic-taper beam constraints (span datum +
buckling), densifies the sized beam radii onto a finer geometry resolution, lofts the
sized form beams (lens sections, circular fallback) and the OML skin, and writes STL
meshes (skin / beams / assembly) to exports/. Prints the sized headline.
"""
from __future__ import annotations

from pathlib import Path

from build123d import Compound, export_stl

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.geometry import build_wing_solid, medium_wingsail
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    build_sized_circular_beams,
    build_sized_lens_beams,
    project_panels_to_beam_nodes,
    resample_segment_radii,
    size_beam_shell_laminate,
)
from wing_design.materials.unidir import T700_EPOXY


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8
    n_geom = 40  # geometry densification levels

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

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=1.5, n_skin_bands=4)

    print("sizing (symmetric + monotonic beams, span datum + buckling)...")
    r = size_beam_shell_laminate(model, load_arrays, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300)
    print(f"  converged={r.converged} iters={r.n_iter} mass={r.mass_kg:.2f} kg "
          f"(beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f})")
    print(f"  beam radii {r.radii.min()*1e3:.1f}-{r.radii.max()*1e3:.1f} mm | "
          f"t_bands {[round(t*1e3,2) for t in r.t_bands]} mm")
    print(f"  twist {r.tip_twist_deg:.2f} deg | buck {r.max_beam_buckling_util:.2f}/{r.max_panel_buckling_util:.2f}")

    dense = resample_segment_radii(r.radii, n_beams=n_beams, n_from=n_levels, n_to=n_geom)
    try:
        beams = build_sized_lens_beams(spec, dense, n_beams=n_beams, n_levels=n_geom)
        kind = "lens"
    except Exception as exc:  # build123d lofting can fail on the lens faces
        print(f"  lens loft failed ({exc}); falling back to circular sections")
        beams = build_sized_circular_beams(spec, dense, n_beams=n_beams, n_levels=n_geom)
        kind = "circular"
    skin = build_wing_solid(medium_wingsail)

    out = Path(__file__).resolve().parent.parent / "exports"
    out.mkdir(exist_ok=True)
    beam_compound = Compound(label="beams", children=list(beams))
    assembly = Compound(label="wingsail_sized", children=[skin, beam_compound])
    export_stl(skin, str(out / "wingsail_sized_skin.stl"))
    export_stl(beam_compound, str(out / "wingsail_sized_beams.stl"))
    export_stl(assembly, str(out / "wingsail_sized_assembly.stl"))
    print(f"  wrote STL ({kind} beams) to {out}/wingsail_sized_*.stl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile-check ONLY (do NOT run — the orchestrator runs it)**

Run: `uv run python -m py_compile examples/37_sized_export.py && echo ok`

- [ ] **Step 3: Commit**

```bash
git add examples/37_sized_export.py
git commit -m "$(cat <<'EOF'
Beam symmetry: sized-geometry STL mesh export example

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Record finding (orchestrator runs the full export + fills numbers)

**Files:**
- Modify: `docs/plan.md`

**Context:** After Tasks 1-3, the ORCHESTRATOR runs `examples/37_sized_export.py` at full
settings (timed, background) to produce the headline + STL artifacts. Then record:

- [ ] **Step 1: Status note** (after the multi-start note, before `### Phase F`):

```markdown
**Beam symmetry + monotonic taper done (2026-06-08) — now the sizer default.**
`size_beam_shell_laminate` enforces mirror-symmetric beam radii (mirror-paired beams
share one radius DV via `beams.beam_radius_groups`, auto-detecting symmetric placement
with graceful fallback) and monotonic non-increasing radius keel→tip (algebraic
constraints). This ~halves the radius DVs (16 beams → 9 unique groups) for a faster, more
physical solve. NOTE: this changes the sizer default — earlier headlines assumed
independent per-element radii and are historical. Full medium-wingsail run (symmetric +
monotonic, span datum + buckling, `examples/37_sized_export.py`): [mass] kg in [iters]
iters, wall-clock [T] (vs the prior ~2 h class); STL meshes of the sized skin/beams/
assembly exported to `exports/`.
```

- [ ] **Step 2: Decisions-log row** (after the multi-start row):

```markdown
| Beam symmetry + monotonic taper | Sizer default: mirror-paired beams share one radius DV (`beam_radius_groups`, auto-detect + fallback), radius monotonic non-increasing keel→tip (algebraic constraints). ~Halves radius DVs → faster. Changes the default (prior headlines historical). `result.radii` still full length n. Full run: [mass] kg, [T] wall-clock; sized STL meshes exported. |
```

- [ ] **Step 3: Commit** (orchestrator, after the run).

---

## Final verification

- [ ] `uv run pytest -q` — all green.
- [ ] `uv run python -m py_compile examples/37_sized_export.py` — compiles.
- [ ] Orchestrator: run `examples/37_sized_export.py` timed (background); confirm STL files in `exports/`; surface them.
- [ ] Dispatch a final code review over the diff, then use superpowers:finishing-a-development-branch.

## Notes / risks

- **`wb = np.zeros(n)` stays element-count** — only DV-index `n` becomes `G`.
- **`mass_grad` uses `np.add.at`** to scatter per-element gradients into the G groups
  (chain rule for shared radii). Verify gradient correctness implicitly via convergence
  (and the existing feasibility tests).
- **Always-on default change** — update only exact-value test assertions; never weaken
  feasibility. `result.radii` stays length n so geometry/stock-catalog consumers are
  unaffected.
- **Symmetry tolerance**: `1e-6 * scale`. Default even spacing is symmetric to float
  precision; deliberate non-uniform spacing differs by ~cm, so the classification is
  robust. If the default unexpectedly falls back (n_groups == n), loosen the tol and
  re-check.
- The export example's STL files are large; they go to the gitignored `exports/`.
