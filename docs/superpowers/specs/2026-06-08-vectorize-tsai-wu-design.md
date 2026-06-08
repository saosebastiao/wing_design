# Vectorize the Tsai-Wu inner loop: design

**Date:** 2026-06-08
**Status:** Approved (brainstorming → ready for implementation plan)

## Goal

Replace the per-triangle Python loop that evaluates the per-ply Tsai-Wu strength ratio
with vectorized numpy, producing **identical** results, so the `skin_failure="tsai_wu"`
sizing path runs in minutes instead of the measured ~2 h 11 m.

## Motivation

`examples/34_tsai_wu_skin.py` measured ~2 h 11 m wall-clock. The cost is the sizer's
inner loop: for every SLSQP `evaluate` (called ~115×/iteration by the finite-difference
Jacobian, over ~120 iterations) the code loops over all M skin triangles in Python,
each calling `laminate_min_strength_ratio`, which itself loops over up to 4 ply
orientations doing scalar strain-transform + matmul + quadratic solve. Vectorizing the
triangle/orientation work into numpy array ops removes the dominant overhead with no
change to the numbers.

## Approach

Ply stress per orientation depends only on the element-local membrane strain and the
ply angle; the layup fractions are scalars, so the **set** of present orientations
(0 / ±45 / 90 gated by f0/f45/f90 > 0) is identical for every triangle. So we compute
the strength ratio for each present orientation across all triangles at once, then take
the element-wise minimum — no per-triangle masking.

### 1. Batched failure helper — `materials/failure.py`

```
laminate_min_strength_ratio_batch(ply, eps_local, *, f0, f45, f90, offset_deg) -> np.ndarray
```
- `eps_local`: `(M, 3)` element-local engineering strain `[exx, eyy, gxy]` per triangle.
- `offset_deg`: `(M,)` per-triangle datum offset in degrees (a scalar is broadcast).
- Returns `(M,)` minimum Tsai-Wu strength ratio over the present orientations.
- Internals (all vectorized over M):
  - For each present orientation `base ∈ {0, +45, -45, 90}` (gated by `f0 > eps`,
    `f45 > eps` for both ±45, `f90 > eps`), form `th = radians(base + offset_deg)` `(M,)`,
    transform strain to ply axes:
    `e1 = ex c² + ey s² + gxy s c`, `e2 = ex s² + ey c² - gxy s c`,
    `g12 = -2 ex s c + 2 ey s c + gxy (c² - s²)` (c, s = cos/sin of `th`).
  - Ply stress `s1 = Q11 e1 + Q12 e2`, `s2 = Q12 e1 + Q22 e2`, `t12 = Q66 g12`
    (`Q` from `reduced_stiffness_Q(ply)`).
  - Strength ratio via a vectorized quadratic solve (see §2).
  - Stack the present-orientation ratios `(K, M)` and return `min(axis=0)` `(M,)`.
  - If NO orientation is present (degenerate), return `full(M, _SAFE_R)`.

### 2. Vectorized strength ratio

A vectorized form of `tsai_wu_strength_ratio` over arrays `s1, s2, t12` `(M,)`:
`a = F11 s1² + F22 s2² + F66 t12² + 2 F12 s1 s2`, `b = F1 s1 + F2 s2`. Then
`R = where(a > tol, (-b + sqrt(b² + 4a)) / (2a), where(b > tol, 1/b, _SAFE_R))`,
clipped to `_SAFE_R`. Guard the `sqrt`/division so no warnings or NaNs leak from the
inactive branch (e.g. evaluate `2a` and `b` with `np.maximum(..., tiny)` denominators
inside the masked branches, or compute both branches on safe inputs and select with
`np.where`). The result must equal the scalar `tsai_wu_strength_ratio` element-wise.

This may be factored as a small private helper (e.g. `_strength_ratio_array(s1, s2, t12,
coeffs)`) shared by both the scalar and batch paths, or inlined in the batch — either is
fine as long as the scalar function's behavior is unchanged.

### 3. Scalar delegates to batch

Reimplement `laminate_min_strength_ratio(ply, eps_local, *, f0, f45, f90, offset_deg)`
(scalar, returns `float`) to call the batch with `eps_local` reshaped to `(1, 3)` and
`offset_deg` as a length-1 array, returning `float(result[0])`. This keeps one source of
truth; its public signature and returned values are unchanged (existing unit tests must
still pass). `ply_strength_ratio` and `tsai_wu_strength_ratio`/`tsai_wu_index` keep their
current scalar signatures (calibration tests depend on them); they may share the §2
helper but their behavior is unchanged.

### 4. Sizer swap — `beams/laminate_sizing.py`

In `size_beam_shell_laminate.evaluate`, replace:
```python
if config.skin_failure == "tsai_wu":
    eps = recover_membrane_strain(...)
    offs = datum_offsets_deg if datum_offsets_deg is not None else np.zeros(eps.shape[0])
    for e in range(eps.shape[0]):
        R_e = laminate_min_strength_ratio(ply, eps[e], f0=f0, f45=f45, f90=f90, offset_deg=float(offs[e]))
        if R_e < worst_R:
            worst_R = R_e
```
with a single batched call:
```python
if config.skin_failure == "tsai_wu":
    eps = recover_membrane_strain(...)
    offs = datum_offsets_deg if datum_offsets_deg is not None else np.zeros(eps.shape[0])
    R = laminate_min_strength_ratio_batch(ply, eps, f0=f0, f45=f45, f90=f90, offset_deg=offs)
    worst_R = min(worst_R, float(R.min()))
```
Import `laminate_min_strength_ratio_batch` (extend the existing
`from ..materials.failure import ...`). No other sizer logic changes; `worst_R` semantics
are identical.

## Correctness invariant (the headline gate)

The vectorized path must equal the current per-triangle scalar path to floating-point
tolerance. Verified by: an equivalence test (batch vs per-triangle scalar over a random
`(M,3)` strain field + `(M,)` offsets), the unchanged `materials/test_failure.py`
calibration tests (scalar now delegates), and the unchanged
`test_tsai_wu_feasible_and_active` sizer test.

## Testing

`tests/materials/test_failure.py` (append):
- `test_batch_matches_scalar`: for a random `(M,3)` `eps` and random `(M,)` `offset_deg`,
  `laminate_min_strength_ratio_batch(...)` equals `[laminate_min_strength_ratio(ply,
  eps[i], ..., offset_deg=offset[i]) for i]` (allclose, rtol 1e-9).
- `test_batch_scalar_strength_ratio_array`: the vectorized strength ratio over an array
  of stresses matches the scalar `tsai_wu_strength_ratio` element-wise (including a
  zero-stress row → `_SAFE_R` and a pure-shear row).
- `test_batch_skips_absent_orientations`: with `f45=0`, the batch result equals the
  scalar (which skips ±45) — i.e. absent orientations don't lower the min.
- Existing scalar calibration tests still pass (scalar delegates to batch).

The sizer behavior is covered by the existing `tests/beams/test_tsai_wu_sizing.py`
(`test_tsai_wu_feasible_and_active`) — it must still pass (same result, faster).

## Components / file map

| File | Responsibility | Change |
|---|---|---|
| `materials/failure.py` | add `laminate_min_strength_ratio_batch`; vectorized strength-ratio helper; scalar `laminate_min_strength_ratio` delegates | modify |
| `materials/__init__.py` | export `laminate_min_strength_ratio_batch` | modify |
| `beams/laminate_sizing.py` | swap per-triangle loop for the batch call | modify |
| `tests/materials/test_failure.py` | equivalence + vectorized tests | modify |

## Success criteria

- Vectorized batch produces results identical (to fp tolerance) to the per-triangle
  scalar path; all existing failure + sizer tests still pass.
- The `skin_failure="tsai_wu"` sizing path is dramatically faster (target: minutes, not
  hours) — confirmed by a quick timed micro-run of one `evaluate` or the small-problem
  sizer test, NOT by re-running the 2 h full example.
- No change to the Tsai-Wu criterion, constraint, or any reported value.

## Out of scope (deferred)

- Per-band layup (separate follow-up) and multi-start (separate follow-up).
- Further FEA-solve speedups (the FEA solve per `evaluate` is unchanged here).
- Analytic Jacobian for the Tsai-Wu constraint (still finite-difference).
