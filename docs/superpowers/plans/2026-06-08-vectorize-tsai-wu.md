# Vectorize the Tsai-Wu Inner Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the per-triangle Python loop evaluating the per-ply Tsai-Wu strength ratio with a vectorized numpy batch producing identical results, cutting the `skin_failure="tsai_wu"` sizing path from ~2 h to minutes.

**Architecture:** Add `laminate_min_strength_ratio_batch` (operates on `(M,3)` strain + `(M,)` offset) to `materials/failure.py`; the scalar `laminate_min_strength_ratio` delegates to it (one source of truth). The sizer swaps its `for e in range(M)` loop for one batch call. No criterion/constraint/value changes.

**Tech Stack:** numpy, pytest, `uv run pytest`.

**Domain notes:**
- Strain/stress vectors are `[xx, yy, xy]`, engineering shear. `reduced_stiffness_Q(ply)` → material-axis `Q` `[[Q11,Q12,0],[Q12,Q22,0],[0,0,Q66]]`.
- `_SAFE_R = 1.0e9` is the no-stress sentinel.
- Present orientations: 0 if `f0>1e-9`; +45 AND -45 if `f45>1e-9`; 90 if `f90>1e-9`. Layup fractions are scalars → same orientation set for all triangles.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1: Vectorized batch in materials/failure.py (scalar delegates)

**Files:**
- Modify: `src/wing_design/materials/failure.py`
- Modify: `src/wing_design/materials/__init__.py`
- Test: `tests/materials/test_failure.py` (append)

- [ ] **Step 1: Append failing tests to `tests/materials/test_failure.py`:**

```python
def test_batch_matches_scalar():
    rng = np.random.default_rng(0)
    M = 20
    eps = rng.normal(scale=1e-3, size=(M, 3))
    offs = rng.uniform(-90.0, 90.0, size=M)
    from wing_design.materials.failure import laminate_min_strength_ratio_batch
    batch = laminate_min_strength_ratio_batch(PLY, eps, f0=0.34, f45=0.33, f90=0.33, offset_deg=offs)
    scalar = np.array([
        laminate_min_strength_ratio(PLY, eps[i], f0=0.34, f45=0.33, f90=0.33, offset_deg=float(offs[i]))
        for i in range(M)
    ])
    assert batch.shape == (M,)
    assert np.allclose(batch, scalar, rtol=1e-9, atol=0.0)


def test_batch_skips_absent_orientations():
    rng = np.random.default_rng(1)
    M = 12
    eps = rng.normal(scale=1e-3, size=(M, 3))
    offs = np.zeros(M)
    from wing_design.materials.failure import laminate_min_strength_ratio_batch
    batch = laminate_min_strength_ratio_batch(PLY, eps, f0=1.0, f45=0.0, f90=0.0, offset_deg=offs)
    scalar = np.array([
        laminate_min_strength_ratio(PLY, eps[i], f0=1.0, f45=0.0, f90=0.0, offset_deg=0.0)
        for i in range(M)
    ])
    assert np.allclose(batch, scalar, rtol=1e-9)


def test_batch_handles_zero_and_shear_rows():
    from wing_design.materials.failure import laminate_min_strength_ratio_batch
    eps = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1e-3]])
    batch = laminate_min_strength_ratio_batch(PLY, eps, f0=1.0, f45=0.0, f90=0.0, offset_deg=np.zeros(2))
    assert batch[0] >= 1.0e8                       # zero strain -> safe
    assert np.isfinite(batch[1]) and batch[1] > 0  # shear row -> finite positive R
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/materials/test_failure.py -k batch -v`
Expected: FAIL with `ImportError: cannot import name 'laminate_min_strength_ratio_batch'`

- [ ] **Step 3: Implement in `src/wing_design/materials/failure.py`**

Add a vectorized strength-ratio helper and the batch function; reimplement the scalar
`laminate_min_strength_ratio` to delegate. Insert the helper after
`tsai_wu_strength_ratio`, and REPLACE the existing `laminate_min_strength_ratio` body
with the delegate:

```python
def _strength_ratio_array(s1, s2, t12, coeffs) -> np.ndarray:
    """Vectorized Tsai-Wu strength ratio over material-axis stress arrays (each (M,))."""
    F1, F2, F11, F22, F66, F12 = coeffs
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    t12 = np.asarray(t12, dtype=float)
    a = F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t12 ** 2 + 2.0 * F12 * s1 * s2
    b = F1 * s1 + F2 * s2
    # Quadratic branch a > tol: R = (-b + sqrt(b^2 + 4a)) / (2a). Compute on safe
    # denominators so the unused branch never emits warnings/NaNs.
    a_safe = np.where(a > 1.0e-30, a, 1.0)
    R_quad = (-b + np.sqrt(b * b + 4.0 * a_safe)) / (2.0 * a_safe)
    # Linear branch a ~ 0: R = 1/b if b > tol else _SAFE_R.
    b_safe = np.where(b > 1.0e-30, b, 1.0)
    R_lin = np.where(b > 1.0e-30, 1.0 / b_safe, _SAFE_R)
    R = np.where(a > 1.0e-30, R_quad, R_lin)
    return np.minimum(R, _SAFE_R)


def laminate_min_strength_ratio_batch(
    ply: UDPly, eps_local, *, f0: float, f45: float, f90: float, offset_deg,
) -> np.ndarray:
    """(M,) min Tsai-Wu strength ratio per triangle over the present ply orientations.

    ``eps_local`` is (M,3) element-local engineering strain; ``offset_deg`` is (M,) (or a
    scalar, broadcast). Present orientations (0/+/-45/90) are gated by the layup
    fractions; ratios are computed for each across all triangles, then min'd element-wise.
    Vectorized equivalent of ``laminate_min_strength_ratio`` per row.
    """
    eps = np.asarray(eps_local, dtype=float)
    if eps.ndim != 2 or eps.shape[1] != 3:
        raise ValueError(f"eps_local must be (M,3), got {eps.shape}")
    M = eps.shape[0]
    off = np.broadcast_to(np.asarray(offset_deg, dtype=float), (M,))
    ex, ey, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    Q = reduced_stiffness_Q(ply)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    coeffs = tsai_wu_coefficients(ply)

    eps_tol = 1.0e-9
    bases: list[float] = []
    if f0 > eps_tol:
        bases.append(0.0)
    if f45 > eps_tol:
        bases.append(45.0)
        bases.append(-45.0)
    if f90 > eps_tol:
        bases.append(90.0)
    if not bases:
        return np.full(M, _SAFE_R)

    ratios = np.empty((len(bases), M))
    for k, base in enumerate(bases):
        th = np.radians(base + off)
        c, s = np.cos(th), np.sin(th)
        e1 = ex * c * c + ey * s * s + gxy * s * c
        e2 = ex * s * s + ey * c * c - gxy * s * c
        g12 = -2.0 * ex * s * c + 2.0 * ey * s * c + gxy * (c * c - s * s)
        s1 = Q11 * e1 + Q12 * e2
        s2 = Q12 * e1 + Q22 * e2
        t12 = Q66 * g12
        ratios[k] = _strength_ratio_array(s1, s2, t12, coeffs)
    return ratios.min(axis=0)


def laminate_min_strength_ratio(
    ply: UDPly, eps_local, *, f0: float, f45: float, f90: float, offset_deg: float,
) -> float:
    """Min Tsai-Wu strength ratio over the present ply orientations (scalar; see batch)."""
    out = laminate_min_strength_ratio_batch(
        ply, np.asarray(eps_local, dtype=float).reshape(1, 3),
        f0=f0, f45=f45, f90=f90, offset_deg=np.asarray([offset_deg], dtype=float),
    )
    return float(out[0])
```
(Leave `tsai_wu_coefficients`, `tsai_wu_index`, `tsai_wu_strength_ratio`,
`_strain_to_ply_axes`, `ply_strength_ratio` as they are — their scalar behavior is
unchanged. Remove the now-unused old `laminate_min_strength_ratio` body, replaced above.)

- [ ] **Step 4: Export from `materials/__init__.py`**

Add `laminate_min_strength_ratio_batch` to the `from .failure import (...)` block and to
`__all__`.

- [ ] **Step 5: Run failure tests + verify scalar still calibrated**

Run: `uv run pytest tests/materials/test_failure.py -v`
Expected: PASS (all — the new batch tests AND the existing scalar calibration tests,
since the scalar now delegates).

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/materials/failure.py src/wing_design/materials/__init__.py tests/materials/test_failure.py
git commit -m "$(cat <<'EOF'
Vectorize Tsai-Wu: batched laminate_min_strength_ratio; scalar delegates

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Swap the sizer's per-triangle loop for the batch call

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py`
- Test: `tests/beams/test_tsai_wu_sizing.py` (existing — must still pass)

**Context:** Replace the Python `for e in range(M)` Tsai-Wu loop inside
`size_beam_shell_laminate.evaluate` with a single batched call. Result values must be
identical; the existing `test_tsai_wu_feasible_and_active` is the gate.

- [ ] **Step 1: Extend the failure import** in `laminate_sizing.py`:
```python
from ..materials.failure import laminate_min_strength_ratio, laminate_min_strength_ratio_batch
```
(or replace the existing `from ..materials.failure import laminate_min_strength_ratio`
line; the scalar import may be dropped if no longer used after the swap — check.)

- [ ] **Step 2: Replace the loop** inside `evaluate` (the `if config.skin_failure ==
"tsai_wu":` block within `for loads in load_arrays:`):

Replace:
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
with:
```python
            if config.skin_failure == "tsai_wu":
                eps = recover_membrane_strain(model.nodes, model.shell_tris, res.displacements)
                offs = datum_offsets_deg if datum_offsets_deg is not None else np.zeros(eps.shape[0])
                R = laminate_min_strength_ratio_batch(
                    ply, eps, f0=f0, f45=f45, f90=f90, offset_deg=offs)
                worst_R = min(worst_R, float(R.min()))
```
(If the exact surrounding text differs, read the function and adapt — preserve
`worst_R`'s init `np.inf` and its later use.)

- [ ] **Step 3: Run the sizer test (correctness gate)**

Run: `uv run pytest tests/beams/test_tsai_wu_sizing.py -v`
Expected: PASS (3 tests), including `test_tsai_wu_feasible_and_active` — same result as
before, just faster.

- [ ] **Step 4: Quick speed sanity (optional, fast)**

Run a single small-problem timed check to confirm the batch path is fast (do NOT run the
2 h example):
```bash
time uv run pytest tests/beams/test_tsai_wu_sizing.py::test_tsai_wu_feasible_and_active -q
```
Record the wall-clock in the commit/report (expectation: well under a minute for the
small problem).

- [ ] **Step 5: Full suite green**

Run: `uv run pytest -q`  → all pass (~7-8 min). If the summary line is swallowed, exit 0
+ no F/E markers = pass.

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/laminate_sizing.py
git commit -m "$(cat <<'EOF'
Vectorize Tsai-Wu: sizer uses batched strength-ratio (no per-triangle loop)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] `uv run pytest -q` — all green.
- [ ] Confirm no behavior change: the Tsai-Wu sizer test result is unchanged.
- [ ] (Optional) re-run `examples/34_tsai_wu_skin.py` ONLY if a fresh measured headline
  is wanted — wrap with `time` and run in the background; not required for correctness.
- [ ] Dispatch a final code review over the diff, then use
  superpowers:finishing-a-development-branch.

## Notes / risks

- **Equivalence is the gate.** `test_batch_matches_scalar` (random strains + offsets) is
  the core check; the scalar now delegates to the batch so the calibration tests also
  exercise the new path.
- **No-warning quadratic:** compute both branches on safe denominators and select with
  `np.where` so the inactive branch never produces RuntimeWarning/NaN.
- **Don't change** the criterion, the `R/SF - 1` constraint, or `worst_R` semantics.
