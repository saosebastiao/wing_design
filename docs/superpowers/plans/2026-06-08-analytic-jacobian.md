# Analytic (Adjoint) Constraint Jacobian Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Analytic adjoint gradients for the laminate-sizer objective + all six FEA-dependent constraints, gated by `use_analytic_jacobian` (default False). Eliminates SLSQP's ~n_DV FEA solves/iteration.

**Architecture:** A factorization-returning solve exposes one `splu(K_ff)` reused for the nominal `u` and all adjoint solves. A `beams/sensitivity.py` module assembles `∂K/∂x` (reusing element routines with derivative coefficients) and computes each constraint's gradient via the adjoint identity `dg/dxᵢ = ∂g/∂xᵢ|explicit − λᵀ(∂K/∂xᵢ)u`, `Kλ = ∂g/∂u`. The sizer supplies `jac=` to every constraint when the flag is on.

**Method note (TDD for gradients):** every analytic gradient is implemented to pass a **central-finite-difference validation test** — the FD check is the executable spec. Derive the closed form, implement it, and make `analytic ≈ central_diff` to tight tolerance. If unsure of a term, the FD test will catch it.

**Domain facts:**
- Linear FEA `K(x)u=f`, `f` fixed; clamped keel; solve free DOFs `K_ff u_f = f_f`; `K` symmetric.
- DV layout `x = [r_group(G), t_band(B), f0_grp(L), f45_grp(L)]`; `radii_full = r_group[group_of_element]`.
- `BeamSection.circular(r)`: `A=πr²`, `Iy=Iz=πr⁴/4`, `J=πr⁴/2`, `r=r`.
- `von_mises_per_element`: `σn=|N|/A + Mres·r/Iz`, `τ=|T|·r/J`, `vM=√(σn²+3τ²)`.
- `beam_euler_utilization`: `comp=max(0,−axial)`, `Pcr=π²E·(πr⁴/4)/(K·L)²`, `util=comp·SF/Pcr`.
- `panel_buckling_utilization`: `σcr=kc·π²·D11/(b²·t)`, `b=√area`, compressive principal stress drives it, `util=comp·SF/σcr`.
- Skin membrane stress `= C·Bm·u_local` (`C=Qeff`); `recover_membrane_stress_C`, `recover_membrane_strain` exist.
- Element stiffness is LINEAR in `(A,I,J)` (beam) and `(A_skin,D_skin)` (shell) → derivatives reuse the same routines with derivative coefficients.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**Hard rule:** `use_analytic_jacobian=False` (default) keeps today's FD path byte-identical. The full suite green under the default is the gate at every task.

---

## Task 1: Factorization-returning sensitivity solve

**Files:** Modify `src/wing_design/structural/beam_shell.py`; Test `tests/beams/test_sensitivity.py` (new).

**Context:** Refactor the shared assembly out of `solve_beam_shell_laminate` and add a sibling that returns the `splu` factorization + per-element data, without changing the existing function's behavior.

- [ ] **Step 1: Failing test** — create `tests/beams/test_sensitivity.py`:
```python
import numpy as np
from wing_design.scenario import small_scenario
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.structural.frame import BeamSection
from wing_design.materials.unidir import T700_EPOXY, laminate_stiffness
from wing_design.structural.beam_shell import solve_beam_shell_laminate, solve_beam_shell_laminate_factored


def _case(n_beams=6, n_levels=4):
    P = small_scenario()
    m = build_beam_shell_model(P.geometry, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    secs = [BeamSection.circular(0.01)] * m.beam_elements.shape[0]
    A, D, _ = laminate_stiffness(T700_EPOXY, f0=0.34, f45=0.33, f90=0.33, thickness=0.0015)
    loads = np.zeros((m.nodes.shape[0], 6)); loads[m.tip_nodes, 0] = 100.0
    return m, secs, A, D, loads


def test_factored_solve_matches_spsolve():
    m, secs, A, D, loads = _case()
    base = solve_beam_shell_laminate(m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D, fixed_nodes=m.fixed_nodes, loads=loads)
    fac = solve_beam_shell_laminate_factored(m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D, fixed_nodes=m.fixed_nodes, loads=loads)
    assert np.allclose(base.displacements, fac.result.displacements, atol=1e-12)
    # the factorization solves K_ff x = rhs
    free = fac.free
    rhs = np.random.default_rng(0).normal(size=free.shape[0])
    x = fac.lu.solve(rhs)
    assert np.allclose(fac.K_ff @ x, rhs, atol=1e-8)
```

- [ ] **Step 2:** Run → FAIL (`solve_beam_shell_laminate_factored` missing).

- [ ] **Step 3: Implement** in `beam_shell.py`: extract an `_assemble_beam_shell_laminate(...)` helper returning `(K_csr, free, fixed_dofs, beam_data, tri_data, transforms, klocals)` where `beam_data`/`tri_data` carry per-element dof maps + the inputs needed later. Refactor `solve_beam_shell_laminate` to use it (behavior unchanged — keep its existing tests green). Add:
```python
from dataclasses import dataclass
import scipy.sparse.linalg as spla

@dataclass
class FactoredBeamShell:
    result: FrameResult
    lu: object                 # splu of K_ff
    K_ff: object               # csc K_ff (for adjoint matvecs / tests)
    free: np.ndarray
    ndof: int
    u: np.ndarray              # full ndof displacement vector
    # per-element data for sensitivity assembly:
    beam_elements: np.ndarray
    transforms: list
    klocals: list
    # skin: store nodes/tris so sensitivity can rebuild dke; plus the A/D used.

def solve_beam_shell_laminate_factored(...same signature...) -> FactoredBeamShell:
    # assemble (shared helper), K_ff = K[free][:,free].tocsc(), lu = spla.splu(K_ff)
    # u_free = lu.solve(f_free); recover FrameResult exactly as solve_beam_shell_laminate
    # return FactoredBeamShell(...)
```
Keep `solve_beam_shell_laminate` returning the same `FrameResult` (now via the helper).

- [ ] **Step 4:** Run the test + existing `tests/beams/test_laminate_sizing.py` → green.
- [ ] **Step 5:** Commit (`Analytic jac: factorization-returning beam-shell solve`).

---

## Task 2: FD-validation helper + `∂K/∂x` assemblers

**Files:** Create `src/wing_design/beams/sensitivity.py`; Test `tests/beams/test_sensitivity.py` (append).

**Context:** Add the three `∂K/∂xᵢ` assemblers (global sparse, or a function returning the affected element local `∂kₑ` + dof maps) and validate each by matrix central-difference: `(K(x+h eᵢ)−K(x−h eᵢ))/2h ≈ ∂K/∂xᵢ`.

- [ ] **Step 1: Failing tests** — append FD checks: for a small model, perturb one radius group / one thickness band / one layup fraction by `±h`, reassemble `K_ff` (via the Task-1 assembly), and compare the central difference to the analytic `∂K_ff/∂xᵢ` from `sensitivity.py` (Frobenius rel-error < 1e-6).
```python
def test_dK_dradius_matches_fd(): ...
def test_dK_dthickness_matches_fd(): ...
def test_dK_dlayup_matches_fd(): ...
```
(Write them to build K via the same assembly path at `x±h` and compare to the analytic assembler.)

- [ ] **Step 2:** Run → FAIL (assemblers missing).

- [ ] **Step 3: Implement** `beams/sensitivity.py`:
- `dK_dradius_group(...)`: for each beam element in the group, `∂kloc/∂r = local_beam_stiffness(E,G, BeamSection(A=2πr, Iy=Iz=πr³, J=2πr³, r=r), L)` (derivative section); `∂kg = Tᵀ ∂kloc T`; scatter into a global sparse `∂K` (or return list of `(dofs, ∂kg)`).
- `dK_dthickness_band(...)`: for each triangle in the band, `∂ke = tri_element_stiffness_laminate(p1,p2,p3, A=Qeff, D=(t²/4)Qeff, drilling_factor=...)` (since `A=tQeff, D=(t³/12)Qeff`). NOTE drilling uses `A66`; passing `A=∂A/∂t=Qeff` makes the drilling term's derivative consistent (it is linear in A66). Validate via FD.
- `dK_dlayup(...)`: `∂Qeff/∂f0 = Qbar(0)−Qbar(90)`, `∂Qeff/∂f45 = ½(Qbar(45)+Qbar(−45))−Qbar(90)` (use `transformed_Qbar` with the triangle's datum offset if `ply_angle_datum` set — mirror how the sizer builds laminates); `∂A=t·∂Qeff`, `∂D=(t³/12)·∂Qeff`; `∂ke = tri_element_stiffness_laminate(A=∂A, D=∂D)`.
- A `central_diff(f, x, i, h)` helper for the validation tests.

- [ ] **Step 4:** Run → the 3 FD checks pass. **Step 5:** Commit.

---

## Task 3: Adjoint engine + tip deflection/twist gradients (simplest)

**Files:** Modify `beams/sensitivity.py`; Test append.

**Context:** Implement the adjoint solve and the two pure-`u` constraint gradients first (no explicit `x` term) to validate the engine end-to-end.

- [ ] **Step 1: Failing test** — for a small fully-built sizer problem at a fixed `x`, compare the analytic `d(defl)/dx` and `d(twist)/dx` rows to central FD of the *constraint values* (re-running the sizer's constraint at `x±h eᵢ`), rel-error < 1e-5.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement** `adjoint_gradient(factored, dg_du, dKdx_terms, u)`:
  - `λ_free = factored.lu.solve(dg_du[free])`; `λ` zero elsewhere.
  - `dg/dxᵢ = explicit_i − Σ_{affected e} λ_eᵀ (∂kₑ/∂xᵢ) u_e` (assemble per DV from the Task-2 assemblers).
  - `defl`: active tip node `n*` = argmax‖u_trans‖; `g = ‖u[n*,:3]‖`; `∂g/∂u = û` at those 3 DOFs; explicit term 0. (Constraint is `1−g/lim` → divide row by `−lim`.)
  - `twist`: active tip node argmax|u_twist|; `∂g/∂u = sign` at that twist DOF; `1−g/lim`.
- [ ] **Step 4:** FD checks pass. **Step 5:** Commit.

---

## Task 4: Beam von Mises + Euler buckling gradients

**Files:** `beams/sensitivity.py`; Test append.

- [ ] **Step 1: Failing test** — analytic `d(beam_con)/dx` and `d(beam_buck_con)/dx` rows vs central FD (rel-error < 1e-4; buckling has a `max(0,·)` kink — test at a point with a clearly compressive active element).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** Active element `e*` = argmax. Internal forces `floc = kloc·T·u_e`; `axial=floc[6]`, `Mres = bending_moment[e*]`, `tors=floc[9]`.
  - **beam vM** (`von_mises_per_element`): `vM=√(σn²+3τ²)`, `σn=|N|/A+Mres·r/Iz`, `τ=|T|·r/J`. `∂vM/∂u` via `∂(N,Mres,T)/∂u_e = (kloc·T) rows [6], [bending], [9]` (note `Mres=max(√(My²+Mz²) at each end)` — differentiate the active end's resultant). `∂vM/∂r` explicit: chain through `A(r),r,Iz(r),J(r)`. Constraint `1−vM/σ_allow`.
  - **Euler util** = `comp·SF/Pcr`, `comp=max(0,−axial)`, `Pcr=π²E(πr⁴/4)/(KL)²`. `∂util/∂u` via `∂axial/∂u_e` (= `−` row when compressive); `∂util/∂r` via `Pcr∝r⁴` (`∂util/∂r=−4util/r`) plus `comp` has no r-dependence. Constraint `1−util`.
- [ ] **Step 4:** FD checks pass. **Step 5:** Commit.

---

## Task 5: Skin von Mises + panel buckling gradients

**Files:** `beams/sensitivity.py`; Test append.

- [ ] **Step 1: Failing test** — analytic `d(skin_con[vM])/dx` and `d(panel_buck_con)/dx` rows vs central FD.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** Active triangle `t*`. Membrane stress `s = C·Bm·u_local` (`C=Qeff` for that triangle; `Bm`, local frame from the shell routines). 
  - **skin vM**: `vM(s)` (membrane_von_mises); `∂vM/∂u` via `∂vM/∂s · C · Bm · R_local` mapped to the triangle's translational DOFs; explicit `∂vM/∂(t,f)` via `∂C/∂(t,f)` (= `∂Qeff/∂(t,f)`; note `Qeff` is t-independent, so vM's explicit `t`-derivative is 0 — vM depends on `t` only through `u`; layup `f` enters `C` explicitly). Constraint `1−vM/σ_allow`.
  - **panel util** = `comp·SF/σcr`, `σcr=kc·π²·D11/(b²·t)`. `comp` = most-compressive principal of `s` (function of `u`,`f`); `D11=D[0,0]=(t³/12)Qeff[0,0]` (depends `t`,`f`); explicit `∂util/∂t` via `σcr∝D11/t∝t³/t=t²` ⇒ `util∝1/t²` from σcr plus `comp` t-dependence is via u only → combine; `∂util/∂f` via `D11(f)` and `comp(f via C)`. Derive carefully; FD test is the gate.
- [ ] **Step 4:** FD checks pass. **Step 5:** Commit.

---

## Task 6: Tsai-Wu skin gradient

**Files:** `beams/sensitivity.py`, possibly `materials/failure.py` (add a strength-ratio stress-derivative helper); Test append.

- [ ] **Step 1: Failing test** — analytic `d(skin_con[tsai_wu])/dx` vs central FD on a datum+tsai_wu small problem (active triangle + active ply).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** Constraint `min_R/SF − 1`. Active (triangle, ply orientation). `R = strength_ratio(σ123)`, `σ123 = Q·ε123`, `ε123 = T_ε(θ)·ε_local`, `ε_local = Bm·u_local`. Chain: `∂R/∂σ123` (add a vectorized derivative in `materials.failure`: `∂R/∂σ` from `aR²+bR−1=0` ⇒ `∂R/∂σ = −(R²·∂a/∂σ + R·∂b/∂σ)/(2aR+b)`), `∂σ123/∂u` via `Q·T_ε·Bm·R_local`; explicit `∂` is 0 (R depends on layup only through which orientations are present, not continuously — fractions don't enter R given the strain). Constraint row `= (1/SF)·dR/dx`.
- [ ] **Step 4:** FD check passes. **Step 5:** Commit.

---

## Task 7: Sizer wiring + `use_analytic_jacobian` flag

**Files:** `beams/laminate_sizing.py`, `beams/__init__.py`; Test `tests/beams/test_analytic_sizing.py` (new).

- [ ] **Step 1: Failing tests** — `tests/beams/test_analytic_sizing.py`:
  - `test_default_is_fd_unchanged`: `use_analytic_jacobian` defaults False; a small sizing equals the current result (sanity: same mass/feasible).
  - `test_analytic_matches_fd_optimum`: same small problem with `use_analytic_jacobian=True` reaches the same feasible optimum (allclose mass within a small tol; both feasible).
- [ ] **Step 2:** Run → FAIL (flag missing).
- [ ] **Step 3: Implement.** Add `use_analytic_jacobian: bool = False` to `LaminateSizingConfig`. When True: in the sizer, after the cached factored solve per `x`, build an `evaluate_jac(x)` that returns the gradient rows for objective + each constraint (calling `beams.sensitivity`), and attach `jac=` to each constraint dict and keep `jac=mass_grad` on the objective. `frac_con`/`monotonic_con` get constant analytic Jacobians (their rows are ±1 patterns). When False: unchanged (no `jac` on constraints). The factored solve replaces the plain solve in `evaluate` only when the flag is on (or always — it returns the same `FrameResult`; cheaper to always use it but keep FD path identical to avoid result drift — use the factored solve only when the flag is on to guarantee byte-identical FD default).
- [ ] **Step 4:** Run the new tests + FULL suite (`uv run pytest -q`) → green (default FD path unchanged; analytic path matches). Grep `LaminateSizingResult(` → one site.
- [ ] **Step 5:** Commit.

---

## Task 8: Speedup example + finding

**Files:** Create `examples/38_analytic_speedup.py`; Modify `docs/plan.md`.

- [ ] **Step 1:** Create `examples/38_analytic_speedup.py`: a SMALL problem (n_beams=8, n_levels=5) sized twice — `use_analytic_jacobian=False` vs `True` — each wrapped in `time`, printing mass/iters/feasibility and wall-clock for both, asserting the masses match. Compile-check only (orchestrator runs it timed).
- [ ] **Step 2:** Orchestrator runs it timed (background); record actual wall-clocks.
- [ ] **Step 3:** `docs/plan.md`: status note + Decisions-log row — adjoint analytic Jacobian, `use_analytic_jacobian` flag (default off), FD-validated, measured speedup [FD t] → [analytic t] on the small problem, same optimum.
- [ ] **Step 4:** Commit.

---

## Final verification
- [ ] `uv run pytest -q` — all green (default FD path unchanged; sensitivity FD-validation suite green; analytic==FD optimum).
- [ ] All gradient FD-validation tests pass to tolerance — the core correctness gate.
- [ ] Orchestrator: timed analytic-vs-FD run recorded.
- [ ] Final code review (focus: gradient correctness already FD-gated; check the FD path is byte-identical when flag off, and no factorization/leak issues), then superpowers:finishing-a-development-branch.

## Notes / risks
- **FD validation is the safety net** — no gradient ships without a passing central-difference check. Use central differences with a per-quantity step (`h` ~ 1e-6·scale) and compare relative error.
- **Default flag off** — the FD path must stay byte-identical; the factored solve is used only when the flag is on (avoids any result drift in the default).
- **Non-smooth max/min constraints** — gradients are the active member's (subgradient). Test at points with a clear unique active member to keep FD checks clean.
- **Adjoint reuse** — one `splu(K_ff)` per design point serves the nominal solve + all ~6 adjoint solves; never refactorize per DV.
- **Symmetry groups** — `∂K/∂r_group` sums over ALL elements in the mirror group (the shared DV); the FD check (perturbing the group DV) covers this automatically.
- This is the largest increment; keep tasks small and FD-gated. If a gradient can't be made to match FD, fall back to leaving `use_analytic_jacobian=False` (FD) for that path and flag it — never ship an unvalidated gradient.
