# M.2 Ply Discretization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the continuous skin laminate of a sized design into an integer-ply, hoop-floored **as-built** laminate (round-and-repair to feasibility) and report the ideal→as-built mass delta on the 1021.6 kg headline.

**Architecture:** A small, self-contained post-processing module — it consumes a `LaminateSizingResult` + config, rounds each skin band to integer plies (0.25 mm/ply) with a mandatory ≥1 hoop (90°) ply, then re-verifies feasibility (closed-form constraints via a `maxiter=0` evaluate + the converged n≥24 buckling eigen) and repairs upward by adding plies until feasible. It does NOT touch the sizer.

**Tech Stack:** Python 3.13, NumPy. Run via `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python`. Tests: `uv run pytest` / the same prefix + `-m pytest`.

**Conventions (CLAUDE.md):** a result counts only if converged AND feasible; **verify buckling at n≥24** (converged cluster floor, the n=8 eigen is unconservatively overstiff); measure wall-clock.

---

## File structure

- `src/wing_design/beams/ply_discretization.py` — NEW. `PlySchedule` dataclass, `round_laminate` (pure), `discretized_carrier` (build a result-like carrier with rounded bands), `reverify` (closed-form + n≥24 eigen feasibility), `discretize_laminate` (round + repair). Self-contained; no sizer edits.
- `runs/asbuilt_m2.py` — NEW (tracked). Loads the headline, runs `discretize_laminate`, prints ideal vs as-built.
- `tests/beams/test_ply_discretization.py` — NEW. Rounding unit tests + repair/feasibility + integration.

---

## Task 1: `PlySchedule` + `round_laminate` (pure rounding, the core)

**Files:**
- Create: `src/wing_design/beams/ply_discretization.py`
- Test: `tests/beams/test_ply_discretization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/beams/test_ply_discretization.py
import numpy as np
from wing_design.beams.ply_discretization import round_laminate, PlySchedule, T_PLY


def test_round_laminate_sum_and_hoop_floor():
    # ~1.15 mm pure ±45 face (the headline): 1.15/0.25 = 4.6 -> 5 plies, f45=1 -> all 45,
    # but the hoop floor forces >=1 ply of 90, taken from the largest (45).
    s = round_laminate(1.15e-3, f0=0.0, f45=1.0, f90=0.0)
    assert isinstance(s, PlySchedule)
    assert s.n0 + s.n45 + s.n90 == s.n          # counts sum to total
    assert s.n90 >= 1                            # mandatory hoop floor
    assert s.n0 == 0                             # no 0 demanded
    assert s.n == 5 and s.n45 == 4 and s.n90 == 1
    assert np.isclose(s.t, 5 * T_PLY)
    assert np.isclose(s.f45, 4 / 5) and np.isclose(s.f90, 1 / 5)


def test_round_laminate_largest_remainder_preserves_total():
    # quasi-iso 0.25/0.50/0.25 on 8 plies (2.0 mm) -> 2/4/2 exactly
    s = round_laminate(2.0e-3, f0=0.25, f45=0.50, f90=0.25)
    assert (s.n0, s.n45, s.n90, s.n) == (2, 4, 2, 8)


def test_round_laminate_thin_band_bumps_to_host_hoop():
    # a band thinner than hoop_min+1 plies (e.g. 0.2 mm < 1 ply) bumps n up so it
    # can carry the hoop floor plus a structural ply.
    s = round_laminate(0.2e-3, f0=0.0, f45=1.0, f90=0.0)
    assert s.n >= 2 and s.n90 >= 1 and (s.n45 + s.n0) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_ply_discretization.py -k round_laminate -q`
Expected: FAIL — `cannot import name 'round_laminate'`.

- [ ] **Step 3: Implement the pure rounding**

```python
# src/wing_design/beams/ply_discretization.py
"""M.2 — skin-laminate ply discretization (round-and-repair to an as-built laminate).

Post-processing only: rounds each skin band's continuous (t, f0, f45, f90) to an
integer ply schedule at T_PLY per ply, enforces the manufacturing concept's
mandatory first-layer HOOP (chordwise / 90 deg) wind (n90 >= HOOP_MIN), then
re-verifies feasibility (closed-form constraints + the converged n>=24 buckling
eigen per CLAUDE.md) and repairs upward by adding plies until feasible. Does NOT
touch the sizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T_PLY = 0.25e-3      # m per cured ply (~0.25 mm)
HOOP_MIN = 1         # mandatory first-layer hoop (90 deg / chordwise) plies per band


@dataclass(frozen=True)
class PlySchedule:
    n0: int
    n45: int
    n90: int

    @property
    def n(self) -> int:
        return self.n0 + self.n45 + self.n90

    @property
    def t(self) -> float:
        return self.n * T_PLY

    @property
    def f0(self) -> float:
        return self.n0 / self.n

    @property
    def f45(self) -> float:
        return self.n45 / self.n

    @property
    def f90(self) -> float:
        return self.n90 / self.n


def _largest_remainder(fracs, n):
    """Round fractional counts (fracs*n) to integers summing exactly to n
    (largest-remainder / Hamilton method)."""
    raw = np.asarray(fracs, dtype=float) * n
    floor = np.floor(raw).astype(int)
    rem = n - int(floor.sum())
    order = np.argsort(-(raw - floor))      # largest fractional remainder first
    for i in range(rem):
        floor[order[i % len(order)]] += 1
    return [int(v) for v in floor]


def round_laminate(t_band, f0, f45, f90, *, t_ply=T_PLY, hoop_min=HOOP_MIN) -> PlySchedule:
    """Round one band's continuous laminate to an integer PlySchedule with the
    mandatory hoop floor (n90 >= hoop_min). Total plies are preserved when the
    floor is satisfied by re-tagging plies; a too-thin band is bumped up so it can
    host the hoop floor plus >=1 structural ply."""
    n = max(1, int(round(float(t_band) / t_ply)))
    if n < hoop_min + 1:                       # can't host hoop + a structural ply
        n = hoop_min + 1
    n0, n45, n90 = _largest_remainder([f0, f45, f90], n)
    if n90 < hoop_min:                         # enforce hoop floor by re-tagging
        need = hoop_min - n90
        # take the needed plies from the larger structural stack (45 preferred)
        if n45 >= n0:
            take = min(need, n45); n45 -= take; n90 += take; need -= take
            n0_take = min(need, n0); n0 -= n0_take; n90 += n0_take
        else:
            take = min(need, n0); n0 -= take; n90 += take; need -= take
            n45_take = min(need, n45); n45 -= n45_take; n90 += n45_take
    return PlySchedule(n0=n0, n45=n45, n90=n90)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_ply_discretization.py -k round_laminate -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/ply_discretization.py tests/beams/test_ply_discretization.py
git commit -m "feat(M.2): PlySchedule + round_laminate (integer plies + mandatory hoop floor)"
```

---

## Task 2: `discretized_carrier` — a result-like carrier with rounded bands

**Files:**
- Modify: `src/wing_design/beams/ply_discretization.py`
- Test: `tests/beams/test_ply_discretization.py`

The re-eval path (`design_vector_from_result`, `eigen_worst`) reads result attributes `radii, t_bands, f0_bands, f45_bands, f90_bands, r_tube, t_wall, t_hollow, t_core, brace_radius`. We build a minimal duck-typed carrier copying those, overriding the bands with rounded values.

- [ ] **Step 1: Write the failing test**

```python
# tests/beams/test_ply_discretization.py (append)
def test_discretized_carrier_overrides_bands_keeps_rest():
    from wing_design.beams.ply_discretization import discretized_carrier, round_laminate

    class R:  # stand-in for a LaminateSizingResult
        radii = np.array([0.02, 0.03]); t_bands = np.array([1.15e-3])
        f0_bands = np.array([0.0]); f45_bands = np.array([1.0]); f90_bands = np.array([0.0])
        r_tube = np.array([0.02]); t_wall = np.array([0.001])
        t_hollow = None; t_core = np.array([5.0e-3]); brace_radius = None
    scheds = [round_laminate(1.15e-3, 0.0, 1.0, 0.0)]
    c = discretized_carrier(R(), scheds)
    assert np.isclose(c.t_bands[0], scheds[0].t)            # rounded thickness
    assert np.isclose(c.f45_bands[0], scheds[0].f45) and np.isclose(c.f90_bands[0], scheds[0].f90)
    assert np.allclose(c.radii, R.radii) and np.allclose(c.t_core, R.t_core)  # rest unchanged
```

- [ ] **Step 2: Run** `... -k discretized_carrier -q` — Expected: FAIL (no `discretized_carrier`).

- [ ] **Step 3: Implement**

```python
# append to ply_discretization.py
class _Carrier:
    """Minimal duck-typed stand-in for LaminateSizingResult that design_vector_from_result
    and eigen_worst read."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def discretized_carrier(result, schedules):
    """Copy `result`'s design arrays but replace the per-band laminate with the
    rounded `schedules` (one PlySchedule per band)."""
    t = np.array([s.t for s in schedules], dtype=float)
    f0 = np.array([s.f0 for s in schedules], dtype=float)
    f45 = np.array([s.f45 for s in schedules], dtype=float)
    f90 = np.array([s.f90 for s in schedules], dtype=float)
    return _Carrier(
        radii=np.asarray(result.radii, dtype=float),
        t_bands=t, f0_bands=f0, f45_bands=f45, f90_bands=f90,
        r_tube=getattr(result, "r_tube", None), t_wall=getattr(result, "t_wall", None),
        t_hollow=getattr(result, "t_hollow", None), t_core=getattr(result, "t_core", None),
        brace_radius=getattr(result, "brace_radius", None))
```

- [ ] **Step 4: Run** `... -k discretized_carrier -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(M.2): discretized_carrier (rounded bands, rest of design unchanged)"
```

---

## Task 3: `reverify` — closed-form feasibility + converged n≥24 eigen

**Files:**
- Modify: `src/wing_design/beams/ply_discretization.py`
- Test: `tests/beams/test_ply_discretization.py` (a slow integration-ish test; mark accordingly)

`reverify(model, config, carrier, loads, presses, rho)` evaluates a carrier WITHOUT re-optimizing and returns feasibility. Reuse the proven patterns:
- closed-form constraints + mass: `x = design_vector_from_result(model, config, carrier); r = size_beam_shell_laminate(model, loads, config, ply=T700_EPOXY, rho=rho, maxiter=0, x0=x, panel_pressures=presses); feas_cf = laminate_result_is_feasible(r, config)` (the `runs/inspect_best.py` maxiter=0 pattern).
- converged eigen: resample `carrier` to n_levels∈{24,28,32} via `runs/mesh_converge_diag.py` helpers + `chain_rebuild.eigen_worst`; converged λ = min over those meshes of the worst λ1 (the cluster floor). `feas_eig = conv_lambda >= 1.5`.

- [ ] **Step 1: Write the failing test** (slow; verifies the ORIGINAL headline carrier re-verifies as feasible — sanity that the eval path works and reproduces a known-feasible design):

```python
# tests/beams/test_ply_discretization.py (append)
import pytest

@pytest.mark.sizing
def test_reverify_headline_is_feasible():
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path("runs")))
    from wing_design.beams.ply_discretization import reverify, _Carrier
    from wing_design import medium_scenario
    from wing_design.aero import build_airplane, sweep_envelope
    from wing_design.beams import build_beam_shell_model
    from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
    from wing_design.beams.laminate_sizing import LaminateSizingConfig
    from wing_design.materials.unidir import PVC_H80
    from chain_rebuild import DATUM
    import numpy as np, json
    P = medium_scenario(); spec = P.geometry
    model = build_beam_shell_model(spec, n_beams=16, n_levels=8, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m, core_tube=True, hollow_beams=True)
    env = sweep_envelope(build_airplane(spec), P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    active=[ar for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N)>=1.0]
    loads=[project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor) for ar in active]
    press=[panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor) for ar in active]
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02*spec.span,
        tip_twist_max_deg=5.0, ply_angle_datum=DATUM, buckling_safety_factor=1.5,
        use_analytic_jacobian=True, panel_width_mode="strip", accel_vectors=(
            (0.,0.,-9.81),(0.,9.81*np.sin(np.radians(30)),-9.81*np.cos(np.radians(30)))),
        core=PVC_H80, ks_rho=50.0, optimizer="ipopt", beam_buckling_model="foundation",
        panel_d_mode="datum_ortho")
    d = np.load("runs/multistart_v2_best.npz")
    r = _Carrier(**{k: d[k] for k in ("radii","t_bands","f0_bands","f45_bands","f90_bands",
                                      "r_tube","t_wall","t_hollow","t_core")}, brace_radius=None)
    rep = reverify(model, cfg, r, loads, press, P.rho_kgm3)
    assert rep["feasible_closedform"] and rep["conv_lambda"] >= 1.5 and rep["mass_kg"] > 0
```

- [ ] **Step 2: Run** `... -k reverify_headline -q` — Expected: FAIL (no `reverify`).

- [ ] **Step 3: Implement** `reverify` (study `runs/inspect_best.py` for the maxiter=0 eval and `runs/mesh_converge_diag.py` + `runs/survival_mesh_eigen.py` for the n≥24 resample+eigen; reuse their helpers — `resample_segment_radii`, `map_hollow_vector`, `resample_tube_segments`, `chain_rebuild.eigen_worst`). Return a dict `{feasible_closedform, conv_lambda, mass_kg, utils}`. Build the n≥24 verification by resampling the carrier to each n and taking the min worst-λ as the cluster floor (or track lowest-3 modes per the convention).

- [ ] **Step 4: Run** `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python -m pytest tests/beams/test_ply_discretization.py -k reverify_headline -q` — Expected: PASS (the original headline is feasible + n≥24 eigen ≥1.5, as established).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(M.2): reverify (closed-form + converged n>=24 eigen) on a carrier"
```

---

## Task 4: `discretize_laminate` — round all bands + repair to feasibility

**Files:**
- Modify: `src/wing_design/beams/ply_discretization.py`
- Test: `tests/beams/test_ply_discretization.py`

- [ ] **Step 1: Write the failing test** (a fast repair-logic test on a stubbed reverify so it doesn't need the FEA — inject a `reverify_fn` that reports infeasible until a band reaches a ply count, proving the loop adds plies and terminates):

```python
# tests/beams/test_ply_discretization.py (append)
def test_discretize_repair_loop_adds_plies_until_feasible():
    from wing_design.beams.ply_discretization import discretize_laminate, round_laminate

    class R:
        radii=np.array([0.02]); t_bands=np.array([1.0e-3])
        f0_bands=np.array([0.0]); f45_bands=np.array([1.0]); f90_bands=np.array([0.0])
        r_tube=None; t_wall=None; t_hollow=None; t_core=None; brace_radius=None
    calls = {"n": 0}
    def fake_reverify(scheds):                       # feasible once band0 has >=6 plies
        calls["n"] += 1
        ok = scheds[0].n >= 6
        return {"feasible_closedform": ok, "conv_lambda": 1.6 if ok else 1.0,
                "mass_kg": 100.0 + scheds[0].n, "utils": {"max_panel_buckling_util": 0.9 if ok else 1.3}}
    out = discretize_laminate(R(), reverify_fn=fake_reverify, max_repair=20)
    assert out["feasible"] and out["schedules"][0].n >= 6
    assert out["as_built_mass_kg"] == 100.0 + out["schedules"][0].n
    assert calls["n"] >= 2                            # rounded once, repaired up
```

- [ ] **Step 2: Run** `... -k repair_loop -q` — Expected: FAIL.

- [ ] **Step 3: Implement** `discretize_laminate`

```python
# append to ply_discretization.py
def discretize_laminate(result, *, reverify_fn, max_repair=40):
    """Round every band to integer plies (+ hoop floor), then repair upward (add
    one ply at a time to the band/orientation that relieves the worst-violated
    constraint) until `reverify_fn(schedules)` reports feasible. `reverify_fn`
    takes the list of PlySchedule and returns a dict with feasible_closedform,
    conv_lambda, mass_kg, utils."""
    t = np.atleast_1d(np.asarray(result.t_bands, dtype=float))
    f0 = np.atleast_1d(np.asarray(result.f0_bands, dtype=float))
    f45 = np.atleast_1d(np.asarray(result.f45_bands, dtype=float))
    f90 = np.atleast_1d(np.asarray(result.f90_bands, dtype=float))
    scheds = [round_laminate(t[b], f0[b], f45[b], f90[b]) for b in range(len(t))]
    rep = reverify_fn(scheds)
    for _ in range(max_repair):
        if rep["feasible_closedform"] and rep["conv_lambda"] >= 1.5:
            break
        # add one ply to the (single-band default) worst band; orientation by the
        # binding constraint: deflection/panel -> thickness via ±45 (the structural
        # stack), buckling -> ±45. 90 is already floored. Default: add a ±45 ply.
        b = 0  # single-band headline; generalize to argmax-util band when B>1
        s = scheds[b]
        scheds[b] = PlySchedule(n0=s.n0, n45=s.n45 + 1, n90=s.n90)
        rep = reverify_fn(scheds)
    feasible = rep["feasible_closedform"] and rep["conv_lambda"] >= 1.5
    return {"feasible": feasible, "schedules": scheds,
            "as_built_mass_kg": rep["mass_kg"], "conv_lambda": rep["conv_lambda"],
            "utils": rep.get("utils", {})}
```

- [ ] **Step 4: Run** `... -k repair_loop -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(M.2): discretize_laminate round-and-repair loop"
```

---

## Task 5: `runs/asbuilt_m2.py` — ideal→as-built on the headline + finding

**Files:**
- Create: `runs/asbuilt_m2.py`
- (record a finding per record-finding after the run)

- [ ] **Step 1: Write the run script** that builds the model/loads/cfg exactly as Task 3's test, loads `runs/multistart_v2_best.npz` into a `_Carrier`, builds a `reverify_fn` closure binding (model,cfg,loads,press,rho) → `reverify(...)`, calls `discretize_laminate`, and prints:
  - ideal mass (1021.6 kg from the npz meta) → as-built mass + % penalty,
  - per-band ply schedule (e.g. `n0/n45/n90 = 0/4/1`, t mm, layup fractions),
  - the binding constraint + n≥24 conv eigen of the as-built design,
  - and `M2_ASBUILT DONE`.

```python
# runs/asbuilt_m2.py — skeleton; fill the model/loads/cfg from Task 3's test verbatim
# reverify_fn = lambda scheds: reverify(model, cfg,
#     discretized_carrier(headline_carrier, scheds), loads, press, P.rho_kgm3)
# out = discretize_laminate(headline_carrier, reverify_fn=reverify_fn)
# print ideal vs out["as_built_mass_kg"], schedules, out["conv_lambda"]
```

- [ ] **Step 2: Run it** (measured wall-clock):

Run: `PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python runs/asbuilt_m2.py`
Expected: prints the ideal→as-built delta + ply schedule + `M2_ASBUILT DONE`, no error. (Each repair iteration is a `maxiter=0` eval + an n≥24 eigen — seconds each; a few iterations total.)

- [ ] **Step 3: Add an integration test** asserting the headline discretizes to a feasible, eigen-verified, **as-built mass ≥ ideal** design with a hoop ply present:

```python
# tests/beams/test_ply_discretization.py (append, @pytest.mark.sizing)
@pytest.mark.sizing
def test_headline_discretizes_to_feasible_asbuilt():
    # (reuse Task 3's model/loads/cfg/carrier setup, or import from a shared helper)
    # reverify_fn bound to the real reverify; out = discretize_laminate(carrier, reverify_fn=...)
    # assert out["feasible"]; assert out["as_built_mass_kg"] >= 1021.6 * 0.999
    # assert out["schedules"][0].n90 >= 1   # hoop present
    ...
```
Fill it in from Task 3's setup (do NOT leave the `...` — paste the concrete setup).

- [ ] **Step 4: Commit**

```bash
git add runs/asbuilt_m2.py tests/beams/test_ply_discretization.py
git commit -m "feat(M.2): as-built ply-discretization run + integration test"
```

- [ ] **Step 5: Record the finding** (record-finding skill): ideal 1021.6 kg → as-built integer-ply mass, the per-band ply schedule, the hoop-floor vs rounding penalty split, the binding constraint + n≥24 eigen, measured wall-clock. First Phase-M "as-built" finding.

---

## Self-review

**Spec coverage:** ply model + 0.25 mm + largest-remainder rounding (T1) ✓; mandatory hoop floor n90≥1 (T1) ✓; discretized design build (T2) ✓; re-verify closed-form + n≥24 eigen (T3) ✓; round-and-repair loop (T4) ✓; ideal→as-built report + finding (T5) ✓; faces-only scope, no gradient/FD (no FD task) ✓; non-goals (core catalog, beams, re-optimize) untouched ✓.

**Placeholder scan:** Task 5's run-script body and the integration test are skeletons with explicit "fill from Task 3's setup verbatim" — the implementer must paste the concrete setup (flagged, not left vague). The repair heuristic adds a ±45 ply (the structural stack) per the spec; the B>1 band-selection generalization is noted as a single-band default (the headline is single-band).

**Type/name consistency:** `PlySchedule(n0,n45,n90)` with `.n/.t/.f0/.f45/.f90`; `round_laminate`, `discretized_carrier`, `_Carrier`, `reverify` (returns `{feasible_closedform, conv_lambda, mass_kg, utils}`), `discretize_laminate(result, *, reverify_fn, max_repair)` returning `{feasible, schedules, as_built_mass_kg, conv_lambda, utils}` — used identically across tasks. `T_PLY=0.25e-3`, `HOOP_MIN=1`.
