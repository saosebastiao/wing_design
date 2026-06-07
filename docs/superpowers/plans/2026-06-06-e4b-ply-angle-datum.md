# E.4b — Consistent Ply-Angle Datum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the CLT skin layup a *manufacturable* prescription by measuring ply angles against a consistent global datum (the **span axis**) instead of each skin triangle's arbitrary local frame. Then re-run the CLT co-sizing so the optimal `(f0, f45, f90)` is a coherent layup ("0° = spanwise"), closing the E.4 limitation.

**Scope note:** Datum = span axis (+Z geom), projected into each triangle's plane. Each triangle's laminate is built with every ply angle offset by that triangle's local-frame angle to the datum, so a uniform global layup maps to per-triangle-rotated stiffness. Backward-compatible: the existing per-triangle-local behavior stays the default; the datum is opt-in via config. A balanced laminate viewed off-datum legitimately has A16/A26 ≠ 0 (the shell element already handles full 3×3 A). Stress for the skin-vm constraint uses the per-triangle (rotated) Qeff.

**Architecture:** A geometry helper `skin_datum_angles(model, datum_dir)` precomputes the per-triangle datum offset δ_e (constant — geometry only). `laminate_stiffness_offset(ply, ..., offset_deg)` builds a laminate with all ply angles shifted by δ_e (reusing `transformed_Qbar`). `solve_beam_shell_laminate` and `recover_membrane_stress_C` are generalized to accept per-triangle stiffness ((M,3,3)) as well as a single (3,3) matrix (backward-compatible). `size_beam_shell_laminate` gains an opt-in `ply_angle_datum` config: when set, each evaluate builds per-triangle (A_e, D_e, Qeff_e) from the design layup + δ_e.

**Tech Stack:** numpy, the E.4 CLT stack (`materials.laminate_stiffness`/`transformed_Qbar`, `structural.solve_beam_shell_laminate`, `recover_membrane_stress_C`, `beams.laminate_sizing`).

---

## Background facts (verified)

- `structural/shell.py::_triangle_local_frame(p1,p2,p3) -> (R, local, area)`, R columns = (e1,e2,e3) global; e1 along p2−p1, e3 = normal.
- The E.4 limitation: ply angles are relative to each triangle's e1; tiling frames are ~50/50 spanwise/chordwise (mean |cos∠(local-x,span)|=0.49), so a uniform layup is not a coherent global fibre direction.
- `materials/unidir.py::reduced_stiffness_Q(ply)`, `transformed_Qbar(Q, angle_deg)`, `laminate_stiffness(ply, *, f0, f45, f90, thickness) -> (A,D,Qeff)` (A=t·Qeff, D=t³/12·Qeff; balanced → A16=A26=0 in the laminate's own frame).
- `structural/beam_shell.py::solve_beam_shell_laminate(nodes, beam_elements, beam_sections, shell_tris, *, E_beam, G_beam, A_skin, D_skin, fixed_nodes, loads, drilling_factor=1e-4)` — loops triangles calling `tri_element_stiffness_laminate(p, A=A_skin, D=D_skin)` (same A,D for every triangle).
- `structural/shell.py::recover_membrane_stress_C(nodes, triangles, displacements, *, C)` — `σ = C·Bm·u` per triangle (same C for all). `membrane_von_mises`.
- `beams/laminate_sizing.py::size_beam_shell_laminate(model, load_arrays, config, *, ply, rho, ...)` — design `[radii, t, f0, f45]`; evaluate builds `A,D,Qeff = laminate_stiffness(ply,f0,f45,f90,t)`, solves, recovers skin stress with `C=Qeff`. `LaminateSizingConfig` is a frozen dataclass (now also has buckling fields). `BeamShellModel.shell_tris` is (M,3); `model.nodes` (N,3).

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/materials/unidir.py` (modify) | `laminate_stiffness_offset(ply, *, f0,f45,f90,thickness, offset_deg)`. |
| `src/wing_design/beams/shell_model.py` (modify) | `skin_datum_angles(model, datum_dir=(0,0,1)) -> (M,)`. |
| `src/wing_design/structural/beam_shell.py` (modify) | `solve_beam_shell_laminate` accepts per-triangle A/D ((M,3,3)) or single (3,3). |
| `src/wing_design/structural/shell.py` (modify) | `recover_membrane_stress_C` accepts per-triangle C ((M,3,3)) or single (3,3). |
| `src/wing_design/beams/laminate_sizing.py` (modify) | `ply_angle_datum` config; datum-aware per-triangle laminate in evaluate. |
| `src/wing_design/{materials,beams}/__init__.py` (modify) | Export `laminate_stiffness_offset`, `skin_datum_angles`. |
| `examples/28_ply_datum.py` (create) | Re-size CLT with span datum; report the now-meaningful layup vs the per-tri-local E.4 result. |
| Tests: `tests/materials/test_clt.py` (+), `tests/beams/test_shell_model.py` (+), `tests/structural/test_beam_shell.py`+`test_membrane_stress.py` (+), `tests/beams/test_laminate_sizing.py` (+). |
| `docs/plan.md` (modify) | Mark E.4b done. |

---

## Task 1: Datum angles + offset laminate

**Files:**
- Modify: `src/wing_design/beams/shell_model.py`, `src/wing_design/materials/unidir.py`, both `__init__.py`
- Test: `tests/beams/test_shell_model.py`, `tests/materials/test_clt.py`

- [ ] **Step 1: Write failing tests.**

Append to `tests/materials/test_clt.py`:
```python
def test_laminate_offset_zero_matches_base():
    from wing_design.materials.unidir import laminate_stiffness, laminate_stiffness_offset
    a0 = laminate_stiffness(T700_EPOXY, f0=0.5, f45=0.3, f90=0.2, thickness=0.003)
    a1 = laminate_stiffness_offset(T700_EPOXY, f0=0.5, f45=0.3, f90=0.2, thickness=0.003, offset_deg=0.0)
    for m0, m1 in zip(a0, a1):
        assert np.allclose(m0, m1, rtol=1e-12)


def test_laminate_offset_90_swaps_axial():
    # offsetting an all-0° laminate by 90° makes it behave like an all-90° one:
    # A11(offset 90 of f0=1) == A22(f0=1, no offset).
    from wing_design.materials.unidir import laminate_stiffness, laminate_stiffness_offset
    A_base, _, _ = laminate_stiffness(T700_EPOXY, f0=1.0, f45=0.0, f90=0.0, thickness=0.003)
    A_off, _, _ = laminate_stiffness_offset(T700_EPOXY, f0=1.0, f45=0.0, f90=0.0, thickness=0.003, offset_deg=90.0)
    assert abs(A_off[0, 0] - A_base[1, 1]) < 1e-3 * A_base[0, 0]
```

Append to `tests/beams/test_shell_model.py`:
```python
def test_skin_datum_angles_shape_and_range():
    from wing_design.beams.shell_model import skin_datum_angles
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    ang = skin_datum_angles(model, datum_dir=(0.0, 0.0, 1.0))
    assert ang.shape == (model.shell_tris.shape[0],)
    assert np.all(np.abs(ang) <= np.pi + 1e-9)   # radians
```

- [ ] **Step 2: Run, expect ImportError.**

- [ ] **Step 3: Implement.**

In `materials/unidir.py` (after `laminate_stiffness`):
```python
def laminate_stiffness_offset(
    ply: UDPly, *, f0: float, f45: float, f90: float, thickness: float, offset_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like `laminate_stiffness` but with every ply angle shifted by `offset_deg`.

    Used to express a laminate defined against a global datum in an element's local
    frame whose x-axis sits `offset_deg` from that datum. With offset_deg=0 this is
    identical to `laminate_stiffness`. (Off-datum, a balanced laminate legitimately
    gains A16/A26 ≠ 0 — the shell element handles the full 3×3.)
    """
    Q = reduced_stiffness_Q(ply)
    o = offset_deg
    Qeff = (
        f0 * transformed_Qbar(Q, o)
        + 0.5 * f45 * (transformed_Qbar(Q, 45.0 + o) + transformed_Qbar(Q, -45.0 + o))
        + f90 * transformed_Qbar(Q, 90.0 + o)
    )
    A = thickness * Qeff
    D = (thickness**3 / 12.0) * Qeff
    return A, D, Qeff
```

In `beams/shell_model.py` (add import `from ..structural.shell import _triangle_local_frame` at top):
```python
def skin_datum_angles(model: BeamShellModel, datum_dir=(0.0, 0.0, 1.0)) -> np.ndarray:
    """(n_tris,) angle [rad] from each skin triangle's local x-axis to ``datum_dir``,
    measured in the triangle's plane.

    `datum_dir` is a global direction (default span = +Z). For each triangle it is
    projected onto the triangle plane and expressed in local (e1,e2); the returned
    angle δ is `atan2(d·e2, d·e1)`. A laminate defined against this datum is built
    per-triangle via `materials.laminate_stiffness_offset(..., offset_deg=degrees(δ))`.
    If the datum is normal to a triangle (no in-plane component) δ defaults to 0.
    """
    d = np.asarray(datum_dir, dtype=float)
    d = d / np.linalg.norm(d)
    out = np.zeros(model.shell_tris.shape[0])
    for e in range(model.shell_tris.shape[0]):
        a, b, c = (int(v) for v in model.shell_tris[e])
        R, _, _ = _triangle_local_frame(model.nodes[a], model.nodes[b], model.nodes[c])
        e1, e2 = R[:, 0], R[:, 1]
        dx, dy = float(d @ e1), float(d @ e2)
        out[e] = 0.0 if (dx == 0.0 and dy == 0.0) else float(np.arctan2(dy, dx))
    return out
```

- [ ] **Step 4:** Export `laminate_stiffness_offset` (materials/__init__) and `skin_datum_angles` (beams/__init__). Run the new tests + full suite.

- [ ] **Step 5: Commit** — `feat(materials,beams): offset laminate + per-triangle ply-angle datum` + trailer.

---

## Task 2: Per-triangle stiffness in solver + stress recovery

**Files:**
- Modify: `src/wing_design/structural/beam_shell.py`, `src/wing_design/structural/shell.py`
- Test: `tests/structural/test_beam_shell.py`, `tests/structural/test_membrane_stress.py`

- [ ] **Step 1: Write failing tests.**

Append to `tests/structural/test_beam_shell.py`:
```python
def test_laminate_solver_accepts_per_triangle_stiffness():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 0.2, 0], [1, 0.2, 0]], dtype=float)
    beam_elems = np.array([[0, 1], [2, 3]], dtype=int)
    secs = [BeamSection.circular(0.01)] * 2
    loads = np.zeros((4, 6)); loads[1, 2] = 100.0; loads[3, 2] = 100.0
    fixed = np.array([0, 2])
    tris = np.array([[0, 1, 3], [0, 3, 2]], dtype=int)
    E, nu, t = 70e9, 0.3, 0.003
    Dm = (E / (1 - nu**2)) * np.array([[1, nu, 0.0], [nu, 1, 0.0], [0, 0, (1 - nu) / 2]])
    A, D = t * Dm, (t**3 / 12.0) * Dm
    single = solve_beam_shell_laminate(nodes, beam_elems, secs, tris, E_beam=E, G_beam=E / 2.6,
                                       A_skin=A, D_skin=D, fixed_nodes=fixed, loads=loads)
    # per-triangle arrays of the SAME matrices must give identical results
    A_tris = np.repeat(A[None], len(tris), axis=0)
    D_tris = np.repeat(D[None], len(tris), axis=0)
    multi = solve_beam_shell_laminate(nodes, beam_elems, secs, tris, E_beam=E, G_beam=E / 2.6,
                                      A_skin=A_tris, D_skin=D_tris, fixed_nodes=fixed, loads=loads)
    assert np.allclose(single.displacements, multi.displacements, rtol=1e-12)
```

Append to `tests/structural/test_membrane_stress.py`:
```python
def test_recover_membrane_stress_C_per_triangle():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    disp = np.zeros((4, 6)); disp[:, 0] = 1e-3 * nodes[:, 0]
    Dm = (E / (1 - NU**2)) * np.array([[1, NU, 0.0], [NU, 1, 0.0], [0, 0, (1 - NU) / 2]])
    s_single = recover_membrane_stress_C(nodes, tris, disp, C=Dm)
    s_multi = recover_membrane_stress_C(nodes, tris, disp, C=np.repeat(Dm[None], 2, axis=0))
    assert np.allclose(s_single, s_multi, rtol=1e-12)
```

- [ ] **Step 2: Run, expect failure** (per-tri array rejected / mis-shaped).

- [ ] **Step 3: Implement** (backward-compatible shape handling).

In `structural/shell.py::recover_membrane_stress_C`, at the top normalize C:
```python
    C = np.asarray(C, dtype=float)
    per_tri = C.ndim == 3
```
and inside the loop use `C[e]` when `per_tri` else `C`:
```python
        Ce = C[e] if per_tri else C
        out[e] = Ce @ Bm @ u_membrane
```
Update the docstring: "C may be a single (3,3) constitutive matrix (all triangles) or a (M,3,3) stack (one per triangle)."

In `structural/beam_shell.py::solve_beam_shell_laminate`, normalize A_skin/D_skin:
```python
    A_skin = np.asarray(A_skin, dtype=float)
    D_skin = np.asarray(D_skin, dtype=float)
    a_per_tri = A_skin.ndim == 3
    d_per_tri = D_skin.ndim == 3
```
and in the shell-triangle loop pass the per-triangle slice:
```python
        Ae = A_skin[t] if a_per_tri else A_skin
        De = D_skin[t] if d_per_tri else D_skin
        ke = tri_element_stiffness_laminate(nodes[n0], nodes[n1], nodes[n2], A=Ae, D=De, drilling_factor=drilling_factor)
```
(Adjust the loop index variable name to whatever the existing loop uses.) Update the docstring: "A_skin/D_skin: a single (3,3) (all triangles) or (M,3,3) per-triangle."

- [ ] **Step 4:** Run the new tests + full suite (existing single-matrix tests must still pass).

- [ ] **Step 5: Commit** — `feat(structural): per-triangle laminate stiffness + stress recovery` + trailer.

---

## Task 3: Datum-aware CLT sizing

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py`
- Test: `tests/beams/test_laminate_sizing.py`

- [ ] **Step 1: Write the failing test** (append):
```python
def test_clt_datum_sizing_runs_and_valid():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 200.0
    loads[model.tip_nodes, 0] = 200.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e8, tip_defl_max_m=1.0, tip_twist_max_deg=2.0,
        r_min=0.004, r_max=0.03, t_min=0.0005, t_max=0.01,
        ply_angle_datum=(0.0, 0.0, 1.0),
    )
    res = size_beam_shell_laminate(model, [loads], cfg, ply=T700_EPOXY, rho=1550.0, maxiter=60)
    assert abs(res.f0 + res.f45 + res.f90 - 1.0) < 1e-6
    assert res.max_beam_vm_Pa <= cfg.sigma_allow_Pa * 1.05
    assert res.max_skin_vm_Pa <= cfg.sigma_allow_Pa * 1.05
```

- [ ] **Step 2: Run, expect failure** (config field missing).

- [ ] **Step 3: Implement.**
- `LaminateSizingConfig`: append `ply_angle_datum: tuple[float, float, float] | None = None`.
- In `size_beam_shell_laminate`, near the top (after `Lb`/`Atri`): if `config.ply_angle_datum is not None`, precompute `datum_offsets_deg = np.degrees(skin_datum_angles(model, config.ply_angle_datum))` (constant). Import `skin_datum_angles` from `.shell_model` and `laminate_stiffness_offset` from `..materials.unidir`.
- In `evaluate(x)`, when datum is set, build per-triangle stiffness instead of a single laminate:
```python
        if config.ply_angle_datum is None:
            A, D, Qeff = laminate_stiffness(ply, f0=f0, f45=f45, f90=f90, thickness=t)
            A_arg, D_arg, C_arg = A, D, Qeff
            D11 = float(D[0, 0])
        else:
            mats = [laminate_stiffness_offset(ply, f0=f0, f45=f45, f90=f90, thickness=t, offset_deg=o)
                    for o in datum_offsets_deg]
            A_arg = np.stack([m[0] for m in mats])
            D_arg = np.stack([m[1] for m in mats])
            C_arg = np.stack([m[2] for m in mats])
            D11 = float(np.max([m[1][0, 0] for m in mats]))   # worst panel bending for buckling
```
Then call `solve_beam_shell_laminate(..., A_skin=A_arg, D_skin=D_arg)` and `recover_membrane_stress_C(..., C=C_arg)`. (The buckling block, if enabled, uses `D11` as above — for the per-tri case use the max D11; acceptable for the spike. Keep the existing single-laminate path byte-identical.)

- [ ] **Step 4:** Run the new test + existing laminate tests (datum default None ⇒ unchanged) + full suite.

- [ ] **Step 5: Commit** — `feat(beams): datum-aware (span) ply angles in CLT co-sizing` + trailer.

---

## Task 4: Example + plan update

**Files:**
- Create: `examples/28_ply_datum.py`
- Modify: `docs/plan.md`

- [ ] **Step 1: Create `examples/28_ply_datum.py`** — co-size the CLT model (n_beams=16, n_levels=8) twice under the same envelope/limits: per-triangle-local (E.4, `ply_angle_datum=None`) vs **span datum** (`ply_angle_datum=(0,0,1)`). Print for each: total mass + split, skin thickness, layup `(f0,f45,f90)`, stresses, tip defl/twist. Emphasize that with the datum the layup is now a coherent global prescription (0°=spanwise). Mirror `examples/26_clt_skin.py`. (FOREGROUND run, ~4–7 min.)

- [ ] **Step 2: Run** `uv run python examples/28_ply_datum.py`. Paste full output. Report the datum-consistent optimal layup and mass vs the per-tri-local result; interpret the layup physically (now meaningful: e.g. "X% spanwise / Y% ±45 / Z% chordwise").

- [ ] **Step 3: Update `docs/plan.md`** — under the E.4 limitation note, add an **E.4b done** entry with the ACTUAL datum-consistent layup + mass, stating the layup is now manufacturable-meaningful. Decisions-log row:
```markdown
| Phase-E.4b ply datum | Ply angles measured against the span axis (per-triangle offset via `skin_datum_angles` + `laminate_stiffness_offset`; solver/recovery generalized to per-triangle stiffness). The optimized layup is now a coherent global prescription (0°=spanwise), not per-arbitrary-edge. Opt-in via `LaminateSizingConfig.ply_angle_datum`. |
```

- [ ] **Step 4:** `uv run pytest` → green.

- [ ] **Step 5: Commit** — `feat(beams): E.4b span-datum ply-angle example; mark done` + trailer.

---

## Self-Review

- **Spec coverage:** span datum → `skin_datum_angles` (Task 1) + `ply_angle_datum` config (Task 3); per-triangle laminate → `laminate_stiffness_offset` (Task 1) + per-tri solver/recovery (Task 2); meaningful layup result → example (Task 4).
- **Placeholder scan:** none.
- **Backward compatibility:** `offset_deg=0` ≡ `laminate_stiffness` (test); per-tri arrays of a repeated single matrix ≡ single-matrix path (tests); `ply_angle_datum=None` ≡ the merged E.4 sizer (existing tests unchanged). All three are explicit anchors.
- **Type consistency:** `skin_datum_angles` returns radians; the sizer converts to degrees for `laminate_stiffness_offset`; solver/recovery accept (3,3) or (M,3,3); per-tri `C_arg=Qeff stack` feeds the skin-vm constraint consistently with the stiffness used in the solve.
- **Known limitations (intentional):** datum = span only (local-surface/principal-direction datum deferred — Phase-F-flavored); per-tri buckling D11 uses the max over triangles (conservative); off-datum balanced laminates carry A16/A26 (correct, handled by the full-3×3 element). The mass change from E.4b is expected to be small — the value is a *correct, manufacturable layout*, not mass.
