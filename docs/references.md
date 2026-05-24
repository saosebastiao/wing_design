# References

## Prior art (read)

### Jiang, Tang, Seidel, Chen, Wonka — *Computational Design of Lightweight Trusses* (arXiv 2019)

`docs/prior_art/computational_design_of_lightweight_trusses.pdf`

- **Input**: external forces, supporting joints, design region, material props (multi-load OK).
- **Output**: pin-jointed truss with optimized joint positions, topology, and per-bar cross-section areas.
- **Pipeline**:
  1. Initialize with grid of intermediate joints densely connected (Ground Structure Method).
  2. *Coarse phase*: interleave geometric optimization (ALP) with local topology ops (delete sub-threshold bars, merge close joints, fix narrow triangles / T-junctions).
  3. *Refinement phase*: Michell-inspired subdivision adds joints/bars along orthogonal tension/compression directions; interleave with ALP.
- **ALP (key contribution)**:
  - Algorithm a: solve force densities w_i = s_i / l_i via LP with fixed joints.
  - Algorithm b: solve joint displacements u_j and Δw via linearized LP.
  - Alternate, with line search.
- **Extensions used here**: multi-load formulation (§6.1); discrete cross-section catalog via Jiang et al. 2017 as post-process (§6.3); external-stability check via Kassimali (§6.2).
- **Limits**: pin-jointed (axial-only), bending/buckling not in the objective — handled by external FEA loop.

### Arora, Jacobson, Langlois, Huang, Mueller, Matusik, Shamir, Singh, Levin — *Volumetric Michell Trusses for Parametric Design & Fabrication* (SCF 2019)

`docs/prior_art/volumetric_michell_truss_scf19_arora_et_al.pdf` (+ supplemental)

- **Input**: arbitrary 3D solid domain Ω + static loads/supports on ∂Ω.
- **Output**: three families of stress-aligned curves inside Ω that form a globally-parametric Michell-like truss. Open-source MATLAB at <https://github.com/rarora7777/VolumetricTruss>.
- **Pipeline**:
  1. Linear elastic FEA on a tet mesh → per-tet Cauchy stress σ. Smoothed to a continuous field via Loubignac iterations.
  2. Fit a stress-aligned SO(3) frame field R(x) per vertex by minimizing a Rayleigh-quotient-induced norm + Laplacian smoothness (rotation-vector parametrization, matrix exponential, fmincon).
  3. Solve a Poisson-like quadratic system for a global parametrization φ: Ω → ℝ³ whose gradients align with R.
  4. Trace integer isocurves of φ → three curve families = the truss centerlines; intersections = nodes.
- **Why it matters here**: the output curves are *exactly* the shapes we want for unidirectional CFRP beams — straight portions are pultrudable, gently curved portions are RTM-able, intersections are candidate joint locations.
- **Validation**: mechanically beats both stress-naive grid trusses and boundary-aligned hex meshes under FEA + bench loading. Stress-aligned bridge sustained ~93 kg ABS-printed at 140 g.
- **Limits**: no fabrication constraints, no cross-section sizing — both flagged as future work, both directly addressed by Jiang et al.

## How we combine the two

| Stage                                  | Source                | Notes                                              |
| -------------------------------------- | --------------------- | -------------------------------------------------- |
| 3D stress field on wing solid          | Arora §4.1            | Use linear FEA + Loubignac smoothing.              |
| Stress-aligned frame field             | Arora §4.2            | Reimplement in Python.                             |
| Global parametrization                 | Arora §4.3            | Quadratic system, easy in scipy.                   |
| Curve extraction → initial truss       | Arora §4.4            | Integer isocurve tracing.                          |
| Joint relocation + topology cleanup    | Jiang §4.3, §5        | ALP on the extracted truss.                        |
| Discrete cross-section sizing          | Jiang §6.3            | Constrain to pultrusion / RTM stock catalog.       |
| Stability / bracing                    | Jiang §6.2            | Kassimali count; add bracing where needed.        |
| Skin contribution + fiber orientation  | Custom (CLT + winding) | Homogenize back into FEA for next iteration.       |

## AeroSandbox 4.2.9 — what we get for free

Installed at `.venv/lib/python3.12/site-packages/aerosandbox/`. Highlights:

- **Geometry**: `Airfoil` (NACA 4-digit built in, plus UIUC database + Kulfan CST); `Wing` (`WingXSec` per station — chord/twist/airfoil); `Airplane`; STEP/STL export.
- **Aerodynamics 3D**: `AeroBuildup` (fast, differentiable, subsonic+transonic, includes fuselage effects and per-section data via `compute_section_aerodynamics`); `VortexLatticeMethod` (per-panel `forces_geometry`); `LiftingLine` / `NonlinearLiftingLine` (nonlinear, 2D-polar viscous); `AVL` and `XFoil` wrappers.
- **Atmosphere / OperatingPoint**: ISA + smooth differentiable atmosphere; full attitude + rotation rates.
- **Optimization**: `Opti` is a thin layer over CasADi + IPOPT. `aerosandbox.numpy` is a drop-in NumPy replacement that traces for AD — every ASB analysis is differentiable and composable into one `Opti` problem.
- **Structures**: `TubeSparBendingStructure` (tapered cantilever, isotropic, distributed + point loads, with stress / buckling outputs); `buckling.column_buckling_critical_load`, `thin_walled_tube_crippling_buckling_critical_load`.
- **Most useful reference example**: `aerodynamics/aero_3D/test_aero_3D/test_vlm/test_airplane_optimization.py` (tapered wing + VLM + L/D opt in one `Opti`).

**Gaps we still have to fill ourselves**:

| Need                                                | Severity | Notes                                                                 |
| --------------------------------------------------- | -------- | --------------------------------------------------------------------- |
| Panel → FEA-mesh pressure projection                | Medium   | Manual: per-panel force × interpolation onto build123d surface mesh.  |
| Composite (CLT, ply schedule, anisotropy)           | High     | ASB beams are isotropic-equivalent only.                              |
| 3D volumetric FEA → Cauchy stress tensor field      | Critical | Need external (sfepy / dolfinx) or roll our own linear-tet solver.    |
| Stress-aligned frame field + parametrization        | Critical | Port / reimplement Arora.                                             |
| Truss ALP + manufacturability constraints           | High     | Build on top of `cvxpy` or ASB `Opti`.                                |
| Filament-winding path planning                      | High     | Custom; no off-the-shelf option for non-axisymmetric parts.           |

## To research as the project progresses

- IsoTruss design literature (BYU group) — for joint-reinforcement winding patterns.
- Variable-stiffness composites / tow-steered laminates — for the skin winding path planner.
- 3D anisotropic FEA in Python: sfepy, FEniCSx (dolfinx), PyNiteFEA — decision deferred to plan question 2.
