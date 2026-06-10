# Analytic-Jacobian Caching Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Precompute the design-dependent `∂K/∂x` element matrices once per design point (a `SensCache`) and reuse them across all constraints/adjoints, so the analytic Jacobian stops rebuilding them ~n× per gradient. EXACT — gradients bit-for-bit unchanged; all FD tests pass.

**CPU note:** a long sizing run is in the background. Run ONLY the fast targeted tests during dev (`tests/beams/test_sensitivity.py`, and the small `tests/beams/test_analytic_sizing.py`). DO NOT run the full suite or any sizing example here — those are deferred to the orchestrator once the background run frees the CPU.

**Domain facts (current `beams/sensitivity.py`):**
- `lambdaT_dK_x(factored, ds, lam) -> (nx,)`: loops beam elements building `dkg_e = T_eᵀ·dkloc_dr(E,G,r_e,L_e)·T_e` and triangles building `dke_dAD(...)` for thickness + `dQeff_df`/`dAD_df` for f0/f45, contracting `lam[dofs] @ dk @ u[dofs]` into `out[dv_index]`. `factored.u`, `factored.transforms`, `factored.beam_elements`; `ds` has `G,B,L, group_of_element, band_of_tri, layup_group_of_band, radii_full, beam_lengths, t_tri, Qeff_tri, offset_tri, ply, drilling_factor, model` (model.nodes, shell_tris, E_beam, G_beam).
- `grad_tip_defl(factored, ds)`, `grad_tip_twist(factored, ds)`, `grad_beam_vm(factored, ds, sigma_allow)`, `grad_beam_buckling(factored, ds, *, euler_K, safety_factor)`, `grad_skin_vm(factored, ds, sigma_allow)`, `grad_panel_buckling(factored, ds, *, panel_kc, safety_factor, areas)`, `grad_skin_tsai_wu(factored, ds, *, safety_factor)` — each calls `lambdaT_dK_x(factored, ds, lam)` internally.
- The `∂kg_e`/`∂ke_t` matrices depend only on the DESIGN (radii/thickness/layup/geometry via `transforms`), NOT on `lam` or `u` or the load case.
- `default_scenario` removed → `small_scenario`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1: SensCache + cached contraction + grad cache kwarg

**Files:** Modify `src/wing_design/beams/sensitivity.py`; Test `tests/beams/test_sensitivity.py` (append).

- [ ] **Step 1: Failing equivalence test** — append to `tests/beams/test_sensitivity.py` (reuse the existing small-case helpers that build a `factored` solve + `DesignSens` `ds`; mirror how the Task-3 adjoint tests construct them):
```python
def test_cached_lambdaT_dK_x_matches_uncached():
    factored, ds = _adjoint_small_case()   # reuse the helper used by the defl/twist grad tests
    rng = np.random.default_rng(0)
    lam = rng.normal(size=factored.ndof)
    from wing_design.beams.sensitivity import prepare_sensitivity, lambdaT_dK_x, lambdaT_dK_x_cached
    cache = prepare_sensitivity(ds, factored)
    a = lambdaT_dK_x(factored, ds, lam)
    b = lambdaT_dK_x_cached(cache, lam, factored.u)
    assert np.allclose(a, b, atol=1e-12, rtol=0.0)
```
(If no shared small-case helper exists, factor one out of an existing grad test, or inline the construction used there. Use a model with a datum + n_skin_bands≥2 so thickness/layup DVs are exercised.)

- [ ] **Step 2:** Run → FAIL (`prepare_sensitivity`/`lambdaT_dK_x_cached` missing).

- [ ] **Step 3: Implement** in `sensitivity.py`:
```python
@dataclass(frozen=True)
class SensCache:
    nx: int
    # beam (radius) contributions
    beam_dofs: list          # list of (12,) int dof-index arrays
    beam_dkg: list           # list of (12,12) global ∂kg/∂r
    beam_dv: np.ndarray      # (n_beam,) radius-group DV index per element
    # triangle contributions
    tri_dofs: list           # list of (18,) int dof-index arrays
    tri_dke_t: list          # (18,18) ∂ke/∂t
    tri_dke_f0: list         # (18,18) ∂ke/∂f0
    tri_dke_f45: list        # (18,18) ∂ke/∂f45
    tri_dv_t: np.ndarray     # (n_tri,) thickness DV index = G + band
    tri_dv_f0: np.ndarray    # (n_tri,) f0 DV index = G+B + layup_group
    tri_dv_f45: np.ndarray   # (n_tri,) f45 DV index = G+B+L + layup_group


def prepare_sensitivity(ds, factored) -> SensCache:
    """Precompute design-dependent ∂K element matrices once (reused across all adjoints/LCs)."""
    G, B, L = ds.G, ds.B, ds.L
    nx = G + B + 2 * L
    be = factored.beam_elements
    beam_dofs, beam_dkg = [], []
    for e in range(be.shape[0]):
        i, j = int(be[e, 0]), int(be[e, 1])
        dk = dkloc_dr(ds.model.E_beam, ds.model.G_beam, float(ds.radii_full[e]), float(ds.beam_lengths[e]))
        T = factored.transforms[e]
        beam_dkg.append(T.T @ dk @ T)
        beam_dofs.append(np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6])
    beam_dv = np.asarray(ds.group_of_element, dtype=int)

    tris = ds.model.shell_tris
    tri_dofs, tri_dke_t, tri_dke_f0, tri_dke_f45 = [], [], [], []
    tri_dv_t = np.empty(tris.shape[0], dtype=int)
    tri_dv_f0 = np.empty(tris.shape[0], dtype=int)
    tri_dv_f45 = np.empty(tris.shape[0], dtype=int)
    for t in range(tris.shape[0]):
        n0, n1, n2 = int(tris[t, 0]), int(tris[t, 1]), int(tris[t, 2])
        p1, p2, p3 = ds.model.nodes[n0], ds.model.nodes[n1], ds.model.nodes[n2]
        tri_dofs.append(np.r_[6 * n0:6 * n0 + 6, 6 * n1:6 * n1 + 6, 6 * n2:6 * n2 + 6])
        tt = float(ds.t_tri[t]); off = float(ds.offset_tri[t]); Qe = ds.Qeff_tri[t]
        dA, dD = dAD_dt(Qe, tt)
        tri_dke_t.append(dke_dAD(p1, p2, p3, dA, dD, ds.drilling_factor))
        dA0, dD0 = dAD_df(dQeff_df(ds.ply, which="f0", offset_deg=off), tt)
        tri_dke_f0.append(dke_dAD(p1, p2, p3, dA0, dD0, ds.drilling_factor))
        dA4, dD4 = dAD_df(dQeff_df(ds.ply, which="f45", offset_deg=off), tt)
        tri_dke_f45.append(dke_dAD(p1, p2, p3, dA4, dD4, ds.drilling_factor))
        b = int(ds.band_of_tri[t]); lg = int(ds.layup_group_of_band[b])
        tri_dv_t[t] = G + b; tri_dv_f0[t] = G + B + lg; tri_dv_f45[t] = G + B + L + lg
    return SensCache(nx, beam_dofs, beam_dkg, beam_dv, tri_dofs, tri_dke_t,
                     tri_dke_f0, tri_dke_f45, tri_dv_t, tri_dv_f0, tri_dv_f45)


def lambdaT_dK_x_cached(cache: SensCache, lam, u) -> np.ndarray:
    """Cheap contraction λᵀ(∂K/∂x)u using cached ∂K matrices (identical to lambdaT_dK_x)."""
    out = np.zeros(cache.nx)
    for e in range(len(cache.beam_dofs)):
        d = cache.beam_dofs[e]
        out[cache.beam_dv[e]] += lam[d] @ cache.beam_dkg[e] @ u[d]
    for t in range(len(cache.tri_dofs)):
        d = cache.tri_dofs[t]; le = lam[d]; ue = u[d]
        out[cache.tri_dv_t[t]] += le @ cache.tri_dke_t[t] @ ue
        out[cache.tri_dv_f0[t]] += le @ cache.tri_dke_f0[t] @ ue
        out[cache.tri_dv_f45[t]] += le @ cache.tri_dke_f45[t] @ ue
    return out
```
Then update every `grad_*` to accept `cache: "SensCache | None" = None` and use the cached path: at the point each currently calls `lambdaT_dK_x(factored, ds, lam)`, replace with:
```python
        _cache = cache if cache is not None else prepare_sensitivity(ds, factored)
        dkx = lambdaT_dK_x_cached(_cache, lam, factored.u)
```
and use `dkx` exactly where `lambdaT_dK_x(...)`'s result was used (sign/explicit-term handling unchanged). Keep `lambdaT_dK_x` as-is (reference + the equivalence test uses it).

- [ ] **Step 4:** Run the FAST targeted tests only:
`uv run pytest tests/beams/test_sensitivity.py -q` → all pass (the equivalence test + ALL existing FD-validation grad tests, now exercising the cached path via lazy build, must pass UNCHANGED — this proves exactness).
DO NOT run the full suite (background run is using the CPU).

- [ ] **Step 5: Commit**
```bash
git add src/wing_design/beams/sensitivity.py tests/beams/test_sensitivity.py
git commit -m "$(cat <<'EOF'
Analytic jac: SensCache (precompute dK/dx once) + cached contraction; exact

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Build the cache once per design point in the sizer

**Files:** Modify `src/wing_design/beams/laminate_sizing.py`; Test `tests/beams/test_analytic_sizing.py` (already exists).

**Context:** In `evaluate_jac`'s analytic block, build ONE `SensCache` per design point (the `∂K` matrices are design-only — independent of load case and constraint) and thread it into every constraint-gradient call so `beam_con`'s ~n rows and all other constraints reuse it.

- [ ] **Step 1:** Read the analytic block (`evaluate_jac`/`_binding_grad`/`beam_con_jac` and the per-constraint `jac` closures). It builds per-lc `FactoredBeamShell`s `facs` and a `DesignSens` `ds`. After `ds` is built (and the factored solves), add:
```python
        from .sensitivity import prepare_sensitivity
        sens_cache = prepare_sensitivity(ds, facs[0])   # design-only; reused across LCs + constraints
```
(use whatever the local names are — `ds` and the list of per-lc factored solves).

- [ ] **Step 2:** Thread `cache=sens_cache` into every `grad_*(...)` call inside `_binding_grad` and the per-constraint jac closures (`grad_beam_vm(..., cache=sens_cache)`, `grad_tip_defl(..., cache=sens_cache)`, etc.). The binding-lc selection still passes the binding lc's `FactoredBeamShell` for `u`/`λ`; only the `∂K` cache is shared.

- [ ] **Step 3: Verify exactness** — `uv run pytest tests/beams/test_analytic_sizing.py -q` → passes (analytic still reaches the same feasible optimum as FD; results unchanged because gradients are identical). Also re-run `tests/beams/test_sensitivity.py -q`. DO NOT run the full suite here.

- [ ] **Step 4: Commit**
```bash
git add src/wing_design/beams/laminate_sizing.py
git commit -m "$(cat <<'EOF'
Analytic jac: build one SensCache per design point in evaluate_jac (reuse across constraints/LCs)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 (ORCHESTRATOR, after the background run frees the CPU): full suite + re-benchmark + finding

**Files:** Modify `docs/plan.md`.

- [ ] Full suite: `uv run pytest -q` → all green (exact → unchanged results).
- [ ] Re-run `examples/38_analytic_speedup.py` (timed) → compare FD vs analytic wall-clock now (was 489 s vs 309 s = 1.58×). Record the new ratio.
- [ ] `docs/plan.md`: status note + Decisions-log row — caching optimization, exact (gradients/results unchanged), measured speedup [old 1.58× → new X×]. Note remaining levers if any (aggregate beam_con, mixed direct/adjoint).
- [ ] Commit; final review; superpowers:finishing-a-development-branch.

## Notes / risks
- **Exactness is the gate** — `lambdaT_dK_x_cached` must equal `lambdaT_dK_x` to 1e-12, and ALL existing FD/grad/analytic-sizing tests must pass unchanged. If any differs, the cache assembly has a bug (dof map / dv-index / transform).
- The cache is **design-only** — safe to build from any one load case's `factored` (transforms are geometry-only) and reuse for all LCs and constraints.
- Keep `lambdaT_dK_x` (uncached) as the reference; do not delete it.
- Do NOT run the full suite / sizing examples during Tasks 1–2 (CPU shared with the background deflection run); the orchestrator runs those in Task 3.
