# P.3 Sandwich (cored) skin: design

**Date:** 2026-06-10
**Status:** Approved (plan.md P.3; backlog P#5). Prestress (P#2) measured null, so
the "partially redundant — measure which wins" question resolves to: sandwich is
the remaining panel-D lever.

## Why now (measured)

Running best 1924.6 kg has skin = 1246 kg (65%) with panel-buck +259 kg/SF and
beam-buck +326 kg/SF binding. σcr ∝ D11; a low-density core (ρ_c/ρ_f ≈ 5%)
multiplies D at small mass cost. Build-compatible per the manufacturing concept:
wind inner skin → place core → wind outer skin.

## Model

1. **Constitutive (the key identity).** Faces = the existing CLT laminate of
   TOTAL thickness t split symmetrically (f = t/2 each) around a core of
   thickness c:
   - Membrane: **A unchanged** (= t·Qeff; the core carries no membrane load);
     stress recovery and all strength checks unchanged (face stress = N/t).
   - Bending: D_sandwich = Qeff·(t³/48 + (t/4)(c + t/2)²) =
     **D_monolithic · φ(t,c)**, with φ = 1/4 + 3(c/t + 1/2)².
     **φ(t,0) = 1 exactly** — monolithic is continuously recoverable at c = 0,
     so the lever can only help. The scalar φ multiplies D per triangle and flows
     into BOTH the DKT bending stiffness (global FEA) and the panel-buckling σcr.
2. **Core material:** `materials.CoreMaterial(E_c, G_c, rho)`; default
   PVC-foam-H80-class (E 80 MPa, G 27 MPa, ρ 80 kg/m³). Config:
   `LaminateSizingConfig.core: CoreMaterial | None = None` (None = unchanged).
3. **DVs:** `("t_core", B)` per skin band, bounds [0, 60 mm], default start 5 mm.
4. **New failure checks (the price of core), both with the smooth activation
   ramp s(c) = c/(c + 1 mm)** so they vanish continuously at c = 0:
   - **Face wrinkling** (Hoff): σ_wr = 0.5·(E_f·E_c·G_c)^(1/3) with
     E_f = Qeff11/1 (local frame, conservative); util = SF·σ_comp,principal/σ_wr·s(c).
   - **Shear crimping:** σ_crimp = G_c·(c + t)/t; util = SF·σ_comp/σ_crimp·s(c).
   Both scalar max-over-(tris × combos) constraints (specs "core_wrinkle",
   "core_crimp"), shadow key on the shared buckling SF.
5. **Mass:** + ρ_c · c_band · band_area; gradient trivial.
6. **Sensitivities:** D's t-derivative changes (∂D/∂t = Qeff·(t²/16 +
   (c+t/2)²/4 + t(c+t/2)/4)) and gains ∂D/∂c = Qeff·t(c+t/2)/2; SensCache tri
   entries gain dke_c when core is active; `grad_panel_buckling` σcr becomes
   sandwich-aware (D11·φ); wrinkle/crimp gradients follow the panel pattern
   (implicit via membrane stress + explicit t, c terms). All FD-validated; the
   core-off path must stay bit-identical (ulp discipline: φ never multiplies
   when `core is None`).

## Measurement

`examples/53_sandwich_skin.py`: running-best config (tube + hollow, 16 beams)
± core, warm-started from the 1924.6 kg optimum with c = 0 start, eigen-verified.
Caveat recorded: the eigen solve's flat facets see the stiffened D in Kσ/K, but
inter-beam curvature remains unmodeled (P#6); wrinkling at the eigen level
unresolved (sub-mesh).

## Non-goals

Core ply/thickness discretization (M#2), core taper/drops at panel edges,
core shear in the global FEA (faces-only membrane shear assumption), honeycomb
anisotropy.
