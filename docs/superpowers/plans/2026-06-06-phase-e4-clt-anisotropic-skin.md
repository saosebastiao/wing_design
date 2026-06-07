# Phase E.4 — CLT Anisotropic Skin (co-size layup) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the isotropic-equivalent skin with a Classical-Laminate-Theory (CLT) anisotropic skin, and co-size the laminate layup (ply-angle fractions) alongside beam radii and skin thickness — so the optimizer can put ±45° plies where the now-binding **tip-twist** constraint needs shear stiffness, cutting mass further than the isotropic skin (E.2: 27.5 kg, twist-governed).

**Scope note:** Phase E increment 4. Co-sizes a **single uniform symmetric-balanced laminate** for the whole skin, parameterized by area fractions `(f0, f45, f90)` (summing to 1) × total thickness `t`. Symmetric + balanced ⇒ B=0; bending uses the **smeared-laminate** approximation `D = (t²/12)·A` (a fractions-only design space doesn't fix stacking order, so D can't depend on order). Skin stress is the **laminate-average membrane von Mises** (`σ = Qeff·ε`); per-ply Tsai-Wu failure is a future refinement. MILP catalog (E.3) and panel buckling remain deferred.

**Architecture:** CLT machinery in `materials/unidir.py` produces `(A, D, Qeff)` from a ply + fractions + thickness. `shell.py` gains `tri_element_stiffness_laminate(p1,p2,p3,*,A,D,drilling_factor)` (membrane from A, bending from D, drilling from `A66`) — which reduces exactly to the isotropic element. `beam_shell.py` gains `solve_beam_shell_laminate(...)` (isotropic-reducible). A CLT-aware membrane stress recovery (`recover_membrane_stress_C`, taking a constitutive matrix) feeds the skin-stress constraint. `beams/laminate_sizing.py` runs the co-sizing NLP over `[beam_radii, t_skin, f0, f45]` (with `f90 = 1−f0−f45`).

**Tech Stack:** numpy, scipy.optimize SLSQP, the E.1/E.2 beam-shell stack, `materials.unidir`.

---

## Background facts (verified)

- `materials/unidir.py::UDPly` fields: `E1_Pa, E2_Pa, G12_Pa, nu12, Xt_Pa, ...`. `T700_EPOXY` instance. No CLT yet.
- `shell.py::_cst_membrane_K(local, area, E, nu, t)`: `K = t·area·Bmᵀ·Dm·Bm`, `Dm = E/(1−ν²)·[[1,ν,0],[ν,1,0],[0,0,(1−ν)/2]]`.
- `shell.py::_dkt_bending_K(local, area, E, nu, t)`: `K = Σ_gauss (1/3)·area·Bbᵀ·Db·Bb`, `Db = E·t³/(12(1−ν²))·[[1,ν,0],[ν,1,0],[0,0,(1−ν)/2]]`, Gauss pts `[(.5,0),(.5,.5),(0,.5)]`; `Bb = _dkt_B_matrix(xi,eta, x23,y23,x31,y31,x12,y12, P4..r6)`.
- `shell.py::_element_K(p1,p2,p3,*,E,nu,t,drilling_factor)`: builds Km (membrane DOFs `[0,1,6,7,12,13]`), Kb (bending DOFs `[2,3,4,8,9,10,14,15,16]`), drilling `drilling_factor·G·t·area` on DOFs `[5,11,17]` (`G=E/(2(1+ν))`); rotates to global with block-diag T (6 R-blocks). **Isotropic `A66 = t·Dm[2,2] = G·t`, so drilling from `A66` matches.**
- `shell.py::recover_membrane_stress(nodes,triangles,displacements,*,E,nu)`: `Dm@Bm@u_membrane` per tri (Pa). `membrane_von_mises(stress)`.
- `structural/beam_shell.py::solve_beam_shell(nodes, beam_elements, beam_sections, shell_tris, *, E_beam, G_beam, E_skin, nu_skin, t_skin, fixed_nodes, loads, drilling_factor=1e-4) -> FrameResult` (assembles beam elems via frame internals + skin tris via `tri_element_stiffness`).
- `beams/shell_model.py::BeamShellModel` (nodes, beam_elements, shell_tris, n_beams, n_levels, fixed_nodes, tip_nodes, section, E_beam, G_beam, E_skin, nu_skin, t_skin); `build_beam_shell_model(...)`.
- `beams/shell_sizing.py`: `beam_lengths`, `skin_areas`, `beam_mass`, `skin_mass`, `BeamShellSizingConfig/Result`, `size_beam_shell` (Phase E.2 — the isotropic co-sizing this mirrors). E.2 result: 27.5 kg, twist-governed.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/materials/unidir.py` (modify) | Add `reduced_stiffness_Q`, `transformed_Qbar`, `laminate_stiffness` (→ A, D, Qeff from ply+fractions+t). |
| `src/wing_design/materials/__init__.py` (modify) | Export the CLT functions (check it re-exports unidir; if not, import from `materials.unidir` is fine). |
| `src/wing_design/structural/shell.py` (modify) | Add `tri_element_stiffness_laminate(p1,p2,p3,*,A,D,drilling_factor)` and `recover_membrane_stress_C(nodes,triangles,displacements,*,C)`. |
| `src/wing_design/structural/__init__.py` (modify) | Export both. |
| `src/wing_design/structural/beam_shell.py` (modify) | Add `solve_beam_shell_laminate(..., A_skin, D_skin, ...)`. |
| `src/wing_design/structural/__init__.py` (modify) | Export `solve_beam_shell_laminate`. |
| `src/wing_design/beams/laminate_sizing.py` (create) | `LaminateSizingConfig/Result`, `size_beam_shell_laminate` (co-size radii + t + f0 + f45). |
| `src/wing_design/beams/__init__.py` (modify) | Export the laminate sizing API. |
| `examples/26_clt_skin.py` (create) | Co-size with CLT skin; compare mass/twist/layup vs E.2 isotropic skin. |
| Tests: `tests/materials/test_clt.py`, `tests/structural/test_laminate_element.py`, `tests/beams/test_laminate_sizing.py` (create). |
| `docs/plan.md` (modify) | Record E.4 done. |

---

## Task 1: CLT laminate stiffness

**Files:**
- Modify: `src/wing_design/materials/unidir.py`
- Test: `tests/materials/test_clt.py`

- [ ] **Step 1: Write the failing tests** (`tests/materials/test_clt.py`)

```python
import numpy as np

from wing_design.materials.unidir import T700_EPOXY, reduced_stiffness_Q, laminate_stiffness


def test_Q_symmetric_and_positive():
    Q = reduced_stiffness_Q(T700_EPOXY)
    assert Q.shape == (3, 3)
    assert np.allclose(Q, Q.T)
    assert Q[0, 0] > Q[1, 1] > 0          # E1 > E2
    assert Q[2, 2] > 0                      # shear


def test_balanced_laminate_has_no_extension_shear_coupling():
    # symmetric balanced → A16 = A26 = 0
    A, D, Qeff = laminate_stiffness(T700_EPOXY, f0=0.5, f45=0.25, f90=0.25, thickness=0.003)
    assert A.shape == (3, 3) and D.shape == (3, 3)
    assert abs(A[0, 2]) < 1e-3 * abs(A[0, 0])
    assert abs(A[1, 2]) < 1e-3 * abs(A[1, 1])
    # smeared bending: D == t^2/12 * A
    assert np.allclose(D, (0.003**2 / 12.0) * A, rtol=1e-9)


def test_pm45_maximizes_shear_stiffness():
    # A ±45-rich laminate has higher in-plane shear stiffness A66 than an all-0 one.
    _, _, q_all0 = laminate_stiffness(T700_EPOXY, f0=1.0, f45=0.0, f90=0.0, thickness=0.003)
    _, _, q_45 = laminate_stiffness(T700_EPOXY, f0=0.0, f45=1.0, f90=0.0, thickness=0.003)
    assert q_45[2, 2] > 2.0 * q_all0[2, 2]   # ±45 dramatically stiffer in shear
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/materials/test_clt.py -v` → ImportError.

- [ ] **Step 3: Implement** — add to `src/wing_design/materials/unidir.py` (np already imported):

```python
def reduced_stiffness_Q(ply: UDPly) -> np.ndarray:
    """Plane-stress reduced stiffness Q (3,3) in the ply material axes [Pa]."""
    nu21 = ply.nu12 * ply.E2_Pa / ply.E1_Pa
    denom = 1.0 - ply.nu12 * nu21
    Q11 = ply.E1_Pa / denom
    Q22 = ply.E2_Pa / denom
    Q12 = ply.nu12 * ply.E2_Pa / denom
    Q66 = ply.G12_Pa
    return np.array([[Q11, Q12, 0.0], [Q12, Q22, 0.0], [0.0, 0.0, Q66]])


def transformed_Qbar(Q: np.ndarray, angle_deg: float) -> np.ndarray:
    """Q rotated to laminate axes by ply `angle_deg` (3,3) [Pa]."""
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    c2, s2 = c * c, s * s
    c4, s4 = c2 * c2, s2 * s2
    s2c2 = s2 * c2
    Q11b = Q11 * c4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * s4
    Q22b = Q11 * s4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * c4
    Q12b = (Q11 + Q22 - 4.0 * Q66) * s2c2 + Q12 * (s4 + c4)
    Q66b = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * s2c2 + Q66 * (s4 + c4)
    Q16b = (Q11 - Q12 - 2.0 * Q66) * s * c * c2 + (Q12 - Q22 + 2.0 * Q66) * s * s2 * c
    Q26b = (Q11 - Q12 - 2.0 * Q66) * s * s2 * c + (Q12 - Q22 + 2.0 * Q66) * s * c * c2
    return np.array([[Q11b, Q12b, Q16b], [Q12b, Q22b, Q26b], [Q16b, Q26b, Q66b]])


def laminate_stiffness(
    ply: UDPly,
    *,
    f0: float,
    f45: float,
    f90: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric-balanced laminate (A, D, Qeff) from 0/±45/90 area fractions.

    `f45` is the total ±45 fraction (split equally +45/−45, so balanced → A16=A26=0).
    Qeff = f0·Qbar(0) + ½f45·(Qbar(45)+Qbar(−45)) + f90·Qbar(90)  (effective stiffness).
    A = thickness·Qeff (N/m). Smeared bending D = (thickness³/12)·Qeff (N·m): a
    fractions-only layup does not fix stacking order, so D uses the smeared model.
    """
    Q = reduced_stiffness_Q(ply)
    Qeff = (
        f0 * transformed_Qbar(Q, 0.0)
        + 0.5 * f45 * (transformed_Qbar(Q, 45.0) + transformed_Qbar(Q, -45.0))
        + f90 * transformed_Qbar(Q, 90.0)
    )
    A = thickness * Qeff
    D = (thickness**3 / 12.0) * Qeff
    return A, D, Qeff
```

- [ ] **Step 4:** Export `reduced_stiffness_Q`, `transformed_Qbar`, `laminate_stiffness` from `materials/__init__.py` (match its style; if it does `from .unidir import ...`, add them there + `__all__`).

- [ ] **Step 5:** `uv run pytest tests/materials/test_clt.py -v` → 3 PASS; `uv run pytest` → green.

- [ ] **Step 6: Commit** — `git add` the two src files + test; message `feat(materials): CLT laminate stiffness (Q, Qbar, A/D from fractions)` + trailer.

---

## Task 2: Laminate shell element + CLT stress recovery

**Files:**
- Modify: `src/wing_design/structural/shell.py`, `src/wing_design/structural/__init__.py`
- Test: `tests/structural/test_laminate_element.py`

- [ ] **Step 1: Write the failing tests** (`tests/structural/test_laminate_element.py`)

```python
import numpy as np

from wing_design.structural.shell import (
    tri_element_stiffness, tri_element_stiffness_laminate,
    recover_membrane_stress, recover_membrane_stress_C,
)

E, NU, T = 70e9, 0.3, 0.003


def _iso_A_D(E, nu, t):
    Dm = (E / (1 - nu**2)) * np.array([[1, nu, 0.0], [nu, 1, 0.0], [0, 0, (1 - nu) / 2]])
    return t * Dm, (t**3 / 12.0) * Dm


def test_laminate_element_reduces_to_isotropic():
    p1, p2, p3 = np.array([0, 0, 0.]), np.array([1, 0, 0.2]), np.array([0, 1, 0.1])
    A, D = _iso_A_D(E, NU, T)
    K_iso = tri_element_stiffness(p1, p2, p3, E=E, nu=NU, t=T)
    K_lam = tri_element_stiffness_laminate(p1, p2, p3, A=A, D=D)
    assert np.allclose(K_iso, K_lam, rtol=1e-9, atol=1e-3)


def test_recover_membrane_stress_C_matches_isotropic():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    tris = np.array([[0, 1, 2]], dtype=int)
    disp = np.zeros((3, 6)); disp[:, 0] = 1e-3 * nodes[:, 0]
    Dm = (E / (1 - NU**2)) * np.array([[1, NU, 0.0], [NU, 1, 0.0], [0, 0, (1 - NU) / 2]])
    s_iso = recover_membrane_stress(nodes, tris, disp, E=E, nu=NU)
    s_C = recover_membrane_stress_C(nodes, tris, disp, C=Dm)
    assert np.allclose(s_iso, s_C, rtol=1e-9)
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/structural/test_laminate_element.py -v` → ImportError.

- [ ] **Step 3: Implement** — add to `src/wing_design/structural/shell.py`:

```python
def tri_element_stiffness_laminate(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
    *, A: np.ndarray, D: np.ndarray, drilling_factor: float = 1.0e-4,
) -> np.ndarray:
    """18×18 global shell stiffness from laminate membrane A and bending D matrices.

    A (3,3) is the CLT membrane stiffness [N/m], D (3,3) the bending stiffness [N·m].
    Reduces exactly to `_element_K(E,nu,t)` when A = t·Dm, D = (t³/12)·Dm (isotropic),
    since the drilling penalty uses A66 (= G·t in the isotropic case).
    """
    R, local, area = _triangle_local_frame(p1, p2, p3)
    # Membrane (CST) with A
    x1, y1 = local[0]; x2, y2 = local[1]; x3, y3 = local[2]
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    two_A = 2.0 * area
    Bm = np.zeros((3, 6))
    for i in range(3):
        Bm[0, 2 * i + 0] = b[i] / two_A
        Bm[1, 2 * i + 1] = c[i] / two_A
        Bm[2, 2 * i + 0] = c[i] / two_A
        Bm[2, 2 * i + 1] = b[i] / two_A
    Km = area * (Bm.T @ A @ Bm)
    # Bending (DKT) with D
    x23, y23 = x2 - x3, y2 - y3
    x31, y31 = x3 - x1, y3 - y1
    x12, y12 = x1 - x2, y1 - y2
    l23, l31, l12 = x23**2 + y23**2, x31**2 + y31**2, x12**2 + y12**2
    P4, P5, P6 = -6.0 * x23 / l23, -6.0 * x31 / l31, -6.0 * x12 / l12
    t4, t5, t6 = -6.0 * y23 / l23, -6.0 * y31 / l31, -6.0 * y12 / l12
    q4, q5, q6 = 3.0 * x23 * y23 / l23, 3.0 * x31 * y31 / l31, 3.0 * x12 * y12 / l12
    r4, r5, r6 = 3.0 * y23**2 / l23, 3.0 * y31**2 / l31, 3.0 * y12**2 / l12
    Kb = np.zeros((9, 9))
    for xi, eta in [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]:
        Bb = _dkt_B_matrix(xi, eta, x23, y23, x31, y31, x12, y12,
                           P4, P5, P6, t4, t5, t6, q4, q5, q6, r4, r5, r6)
        Kb += (1.0 / 3.0) * area * (Bb.T @ D @ Bb)
    # Assemble into 18×18 local, then rotate (mirrors _element_K)
    Ke_local = np.zeros((18, 18))
    membrane_dofs = [0, 1, 6, 7, 12, 13]
    for a_, da in enumerate(membrane_dofs):
        for b_, db in enumerate(membrane_dofs):
            Ke_local[da, db] += Km[a_, b_]
    bending_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16]
    for a_, da in enumerate(bending_dofs):
        for b_, db in enumerate(bending_dofs):
            Ke_local[da, db] += Kb[a_, b_]
    drill_k = drilling_factor * float(A[2, 2]) * area
    for d in (5, 11, 17):
        Ke_local[d, d] += drill_k
    T = np.zeros((18, 18))
    for n in range(3):
        for k in range(2):
            base = 6 * n + 3 * k
            T[base:base + 3, base:base + 3] = R
    return T @ Ke_local @ T.T


def recover_membrane_stress_C(
    nodes: np.ndarray, triangles: np.ndarray, displacements: np.ndarray, *, C: np.ndarray,
) -> np.ndarray:
    """Per-triangle membrane stress (M,3) using a general constitutive matrix C (3,3).

    For a CLT laminate pass C = Qeff (= A/thickness); for isotropic, C = Dm. Stress
    is the laminate-average membrane stress σ = C·ε in the element-local frame [Pa].
    """
    M = triangles.shape[0]
    out = np.zeros((M, 3))
    for e in range(M):
        i, j, l = (int(v) for v in triangles[e])
        R, local, area = _triangle_local_frame(nodes[i], nodes[j], nodes[l])
        x1, y1 = local[0]; x2, y2 = local[1]; x3, y3 = local[2]
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        two_A = 2.0 * area
        Bm = np.zeros((3, 6))
        for k in range(3):
            Bm[0, 2 * k + 0] = b[k] / two_A
            Bm[1, 2 * k + 1] = c[k] / two_A
            Bm[2, 2 * k + 0] = c[k] / two_A
            Bm[2, 2 * k + 1] = b[k] / two_A
        u_m = np.zeros(6)
        for n_idx, gn in enumerate((i, j, l)):
            u_local = R.T @ displacements[gn, 0:3]
            u_m[2 * n_idx + 0] = u_local[0]
            u_m[2 * n_idx + 1] = u_local[1]
        out[e] = C @ Bm @ u_m
    return out
```

- [ ] **Step 4:** Export `tri_element_stiffness_laminate`, `recover_membrane_stress_C` from `structural/__init__.py`.

- [ ] **Step 5:** `uv run pytest tests/structural/test_laminate_element.py -v` → 2 PASS (the isotropic-reduction tests are the key anchors); `uv run pytest` → green.

- [ ] **Step 6: Commit** — message `feat(structural): anisotropic (CLT) shell element + general-C stress recovery` + trailer.

---

## Task 3: Laminate beam+shell solver

**Files:**
- Modify: `src/wing_design/structural/beam_shell.py`, `src/wing_design/structural/__init__.py`
- Test: `tests/structural/test_beam_shell.py`

- [ ] **Step 1: Write the failing test** (append to `tests/structural/test_beam_shell.py`)

```python
from wing_design.structural.beam_shell import solve_beam_shell_laminate


def test_laminate_solver_reduces_to_isotropic():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 0.2, 0], [1, 0.2, 0]], dtype=float)
    beam_elems = np.array([[0, 1], [2, 3]], dtype=int)
    secs = [BeamSection.circular(0.01)] * 2
    loads = np.zeros((4, 6)); loads[1, 2] = 100.0; loads[3, 2] = 100.0
    fixed = np.array([0, 2])
    tris = np.array([[0, 1, 3], [0, 3, 2]], dtype=int)
    E, nu, t = 70e9, 0.3, 0.003
    Dm = (E / (1 - nu**2)) * np.array([[1, nu, 0.0], [nu, 1, 0.0], [0, 0, (1 - nu) / 2]])
    A, D = t * Dm, (t**3 / 12.0) * Dm
    iso = solve_beam_shell(nodes, beam_elems, secs, tris, E_beam=E, G_beam=E / 2.6,
                           E_skin=E, nu_skin=nu, t_skin=t, fixed_nodes=fixed, loads=loads)
    lam = solve_beam_shell_laminate(nodes, beam_elems, secs, tris, E_beam=E, G_beam=E / 2.6,
                                    A_skin=A, D_skin=D, fixed_nodes=fixed, loads=loads)
    assert np.allclose(iso.displacements, lam.displacements, rtol=1e-9, atol=1e-12)
```

- [ ] **Step 2: Run to verify it fails** — ImportError.

- [ ] **Step 3: Implement** — add `solve_beam_shell_laminate` to `beam_shell.py`. It is identical to `solve_beam_shell` except the skin loop uses `tri_element_stiffness_laminate(nodes[n0],nodes[n1],nodes[n2], A=A_skin, D=D_skin, drilling_factor=drilling_factor)` instead of `tri_element_stiffness(..., E=E_skin, nu=nu_skin, t=t_skin)`. Signature:

```python
def solve_beam_shell_laminate(
    nodes, beam_elements, beam_sections, shell_tris, *,
    E_beam, G_beam, A_skin, D_skin, fixed_nodes, loads, drilling_factor=1.0e-4,
) -> FrameResult:
```

Add `from .shell import tri_element_stiffness_laminate` to the imports. Keep the same `beam_sections` length guard, beam assembly, BC elimination, solve, and beam-force recovery as `solve_beam_shell` (factor a shared helper if convenient, or duplicate the well-tested beam loop). Export `solve_beam_shell_laminate` from `structural/__init__.py`.

- [ ] **Step 4:** `uv run pytest tests/structural/test_beam_shell.py -v` → all pass (incl. the new reduction test); `uv run pytest` → green.

- [ ] **Step 5: Commit** — message `feat(structural): laminate (CLT) beam+shell solver` + trailer.

---

## Task 4: CLT co-sizing (layup as design variables)

**Files:**
- Create: `src/wing_design/beams/laminate_sizing.py`
- Test: `tests/beams/test_laminate_sizing.py`

- [ ] **Step 1: Write the failing tests** (`tests/beams/test_laminate_sizing.py`)

```python
import numpy as np

from wing_design.geometry import WingSpec
from wing_design.materials.unidir import T700_EPOXY
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.laminate_sizing import LaminateSizingConfig, size_beam_shell_laminate


def test_clt_cosizing_feasible_and_valid_fractions():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    n = model.beam_elements.shape[0]
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 200.0
    loads[model.tip_nodes, 0] = 200.0   # add a chordwise component → some twist
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e8, tip_defl_max_m=1.0, tip_twist_max_deg=2.0,
        r_min=0.004, r_max=0.03, t_min=0.0005, t_max=0.01,
    )
    res = size_beam_shell_laminate(model, [loads], cfg, ply=T700_EPOXY, rho=1550.0, maxiter=60)
    assert res.radii.shape == (n,)
    # fractions valid: each in [0,1], sum to 1
    assert res.f0 >= -1e-6 and res.f45 >= -1e-6 and res.f90 >= -1e-6
    assert abs(res.f0 + res.f45 + res.f90 - 1.0) < 1e-6
    assert cfg.t_min - 1e-9 <= res.t_skin <= cfg.t_max + 1e-9
    assert res.max_beam_vm_Pa <= cfg.sigma_allow_Pa * 1.05
    assert res.max_skin_vm_Pa <= cfg.sigma_allow_Pa * 1.05
```

- [ ] **Step 2: Run to verify it fails** — ImportError.

- [ ] **Step 3: Implement** `src/wing_design/beams/laminate_sizing.py`. Design vector `x = [radii (n), t_skin, f0, f45]`; `f90 = 1 − f0 − f45`. Each evaluate: `A, D, Qeff = laminate_stiffness(ply, f0, f45, f90, t_skin)`; `solve_beam_shell_laminate(..., A_skin=A, D_skin=D, ...)`; beam vm via `von_mises_per_element`; skin vm via `membrane_von_mises(recover_membrane_stress_C(nodes, tris, disp, C=Qeff))`; tip defl/twist from `model.tip_nodes`. Objective = total (beam + skin) mass `/ m_ref`; skin mass uses `skin_mass(model, t, rho)` (reuse from `shell_sizing`). Analytic gradient for radii & t (fractions affect only constraints → leave their objective-gradient entries 0; SLSQP FDs the constraints). Constraints (normalized ≥0): beam vm, skin vm, tip defl, tip twist, and `1 − (f0 + f45)` (so `f90 ≥ 0`). Bounds: radii `[r_min,r_max]`, t `[t_min,t_max]`, f0 `[0,1]`, f45 `[0,1]`. Clip x to bounds before the final evaluate. `x0`: radii at r_max, t at t_max, `f0=f45=1/3` (quasi-iso start).

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..materials.unidir import UDPly, laminate_stiffness
from ..structural.beam_shell import solve_beam_shell_laminate
from ..structural.frame import BeamSection, von_mises_per_element
from ..structural.shell import membrane_von_mises, recover_membrane_stress_C
from .shell_model import BeamShellModel
from .shell_sizing import beam_lengths, beam_mass, skin_areas, skin_mass


@dataclass(frozen=True)
class LaminateSizingConfig:
    sigma_allow_Pa: float
    tip_defl_max_m: float
    tip_twist_max_deg: float
    r_min: float = 0.004
    r_max: float = 0.04
    t_min: float = 0.0005
    t_max: float = 0.02


@dataclass(frozen=True)
class LaminateSizingResult:
    radii: np.ndarray
    t_skin: float
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


def size_beam_shell_laminate(model, load_arrays, config, *, ply, rho, maxiter=80, ftol=1e-4):
    if not load_arrays:
        raise ValueError("load_arrays is empty")
    n = model.beam_elements.shape[0]
    Lb = beam_lengths(model)
    A_skin_area = float(skin_areas(model).sum())
    x0 = np.concatenate([np.full(n, config.r_max), [config.t_max, 1.0 / 3.0, 1.0 / 3.0]])
    cache: dict = {}

    def evaluate(x):
        key = np.asarray(x, dtype=float).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        radii = x[:n]; t = float(x[n]); f0 = float(x[n + 1]); f45 = float(x[n + 2])
        f90 = max(0.0, 1.0 - f0 - f45)
        A, D, Qeff = laminate_stiffness(ply, f0=f0, f45=f45, f90=f90, thickness=t)
        sections = [BeamSection.circular(float(r)) for r in radii]
        wb = np.zeros(n); ws = 0.0; md = 0.0; mt = 0.0
        for loads in load_arrays:
            res = solve_beam_shell_laminate(
                model.nodes, model.beam_elements, sections, model.shell_tris,
                E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A, D_skin=D,
                fixed_nodes=model.fixed_nodes, loads=loads,
            )
            wb = np.maximum(wb, von_mises_per_element(res, sections))
            skin_s = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=Qeff)
            ws = max(ws, float(membrane_von_mises(skin_s).max()))
            tip = model.tip_nodes
            md = max(md, float(np.linalg.norm(res.displacements[tip, :3], axis=1).max()))
            mt = max(mt, float(np.degrees(np.abs(res.displacements[tip, 5]).max())))
        out = (wb, ws, md, mt)
        cache[key] = out
        return out

    m_ref = beam_mass(model, np.full(n, config.r_max), rho=rho) + skin_mass(model, config.t_max, rho=rho)

    def mass(x):
        return (rho * np.sum(np.pi * x[:n] ** 2 * Lb) + rho * x[n] * A_skin_area) / m_ref

    def mass_grad(x):
        g = np.zeros(n + 3)
        g[:n] = rho * 2.0 * np.pi * x[:n] * Lb / m_ref
        g[n] = rho * A_skin_area / m_ref
        return g  # fractions don't change mass (uniform density)

    def beam_con(x):
        wb, _, _, _ = evaluate(x); return 1.0 - wb / config.sigma_allow_Pa

    def skin_con(x):
        _, ws, _, _ = evaluate(x); return np.array([1.0 - ws / config.sigma_allow_Pa])

    def defl_con(x):
        _, _, d, _ = evaluate(x); return np.array([1.0 - d / config.tip_defl_max_m])

    def twist_con(x):
        _, _, _, t = evaluate(x); return np.array([1.0 - t / config.tip_twist_max_deg])

    def frac_con(x):
        return np.array([1.0 - (x[n + 1] + x[n + 2])])   # f90 = 1 - f0 - f45 >= 0

    bounds = [(config.r_min, config.r_max)] * n + [(config.t_min, config.t_max), (0.0, 1.0), (0.0, 1.0)]
    res = minimize(mass, x0, jac=mass_grad, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": beam_con},
                                {"type": "ineq", "fun": skin_con},
                                {"type": "ineq", "fun": defl_con},
                                {"type": "ineq", "fun": twist_con},
                                {"type": "ineq", "fun": frac_con}],
                   options={"maxiter": maxiter, "ftol": ftol})

    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    x = np.clip(res.x, lo, hi)
    wb, ws, d, t = evaluate(x)
    radii = x[:n]; t_skin = float(x[n]); f0 = float(x[n + 1]); f45 = float(x[n + 2])
    f90 = max(0.0, 1.0 - f0 - f45)
    bm = beam_mass(model, radii, rho=rho); sm = skin_mass(model, t_skin, rho=rho)
    return LaminateSizingResult(
        radii=radii, t_skin=t_skin, f0=f0, f45=f45, f90=f90,
        mass_kg=bm + sm, beam_mass_kg=bm, skin_mass_kg=sm,
        converged=bool(res.success), n_iter=int(res.nit),
        max_beam_vm_Pa=float(wb.max()), max_skin_vm_Pa=float(ws),
        tip_defl_m=float(d), tip_twist_deg=float(t),
    )
```

- [ ] **Step 4:** `uv run pytest tests/beams/test_laminate_sizing.py -v` → PASS (tiny problem). Then `uv run pytest`. **If infeasible or fractions invalid, STOP and report.**

- [ ] **Step 5: Commit** — message `feat(beams): CLT co-sizing (beam radii + skin thickness + layup fractions)` + trailer.

---

## Task 5: Export + example + plan update

**Files:**
- Modify: `src/wing_design/beams/__init__.py`
- Create: `examples/26_clt_skin.py`
- Modify: `docs/plan.md`

- [ ] **Step 1:** Export `LaminateSizingConfig`, `LaminateSizingResult`, `size_beam_shell_laminate` from `beams/__init__.py`.

- [ ] **Step 2: Create `examples/26_clt_skin.py`** — co-size with CLT skin at the working scenario (n_beams=16, n_levels=8), and compare to the E.2 isotropic co-sizing (`size_beam_shell`) under the same loads/limits. Print: E.2 isotropic mass + twist; E.4 CLT mass + chosen layup `(f0,f45,f90)` + skin thickness + beam radii range + beam/skin stress + tip defl/twist. Use `from wing_design.beams import size_beams, size_beam_shell, size_beam_shell_laminate, ...` and `from wing_design.materials.unidir import T700_EPOXY`. Mirror the structure of `examples/25_resize_with_skin.py`. Constraints: `sigma_allow=P.sigma_allow_Pa`, `tip_defl_max=0.02*span`, `tip_twist_max=5.0`. Expect the optimizer to choose a **±45-rich** layup (high f45) to satisfy the twist constraint at lower mass than isotropic.

- [ ] **Step 3: Run** — `uv run python examples/26_clt_skin.py` (FOREGROUND, ~3–6 min: two SLSQP sizings). Paste full output. Report: CLT total mass vs E.2 isotropic mass, the chosen `(f0,f45,f90)` layup, twist (should sit at/under the 5° limit), and whether CLT is lighter. **Honest reporting if not lighter** (e.g. if the smeared isotropic skin already had enough shear) — that is itself a finding.

- [ ] **Step 4: Update `docs/plan.md`** — Phase-E status note with ACTUAL numbers: E.4 done, CLT anisotropic skin co-sized (layup fractions as design vars); chosen layup; mass vs E.2; whether ±45-tailoring helped the twist-governed design. Decisions-log row:

```markdown
| Phase-E.4 CLT skin | Anisotropic skin via CLT (symmetric-balanced laminate, smeared D); laminate (A,D) feed `tri_element_stiffness_laminate` / `solve_beam_shell_laminate`; co-sizes beam radii + skin thickness + layup fractions (f0,f45,f90) minimizing mass under beam-vm, skin-vm, deflection, twist. Per-ply Tsai-Wu / per-band layup / MILP / buckling deferred. |
```

- [ ] **Step 5:** `uv run pytest` → green.

- [ ] **Step 6: Commit** — message `feat(beams): Phase-E.4 CLT anisotropic-skin co-sizing example; mark phase` + trailer.

---

## Self-Review

- **Spec coverage:** CLT anisotropic skin → Tasks 1–3 (laminate stiffness, laminate element, laminate solver); co-size layup → Task 4 (`f0,f45` design vars, `f90` derived, fraction constraint); symmetric-balanced fractions + smeared D → Task 1; demonstration vs isotropic → Task 5.
- **Correctness anchors:** Task 2 and Task 3 each assert the laminate path **reduces exactly to the isotropic path** when the laminate is built from `(E,nu,t)` — this guards the whole generalization. Task 1 checks balanced⇒A16=0 and ±45⇒high A66.
- **Placeholder scan:** none.
- **Type consistency:** `laminate_stiffness` returns `(A, D, Qeff)`; A,D feed `tri_element_stiffness_laminate`/`solve_beam_shell_laminate`; Qeff feeds `recover_membrane_stress_C` for the skin-vm constraint; `beam_mass`/`skin_mass`/`beam_lengths`/`skin_areas` reused from `shell_sizing`. Design-vector layout `[radii, t, f0, f45]` consistent across evaluate/objective/gradient/result.
- **Known limitations (intentional):** smeared bending D (fractions don't fix stacking); laminate-average membrane von Mises as the skin stress proxy (per-ply Tsai-Wu deferred); one uniform laminate for the whole skin (per-band deferred); fraction objective-gradient entries are 0 (fractions don't change mass at uniform density — they enter only through constraints, FD'd by SLSQP).
