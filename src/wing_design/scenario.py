"""Centralized design parameters for the wingsail project.

Every parameter the project uses lives in `DesignParameters`. Examples
construct one instance (via `default_scenario()` for the current 5 m demo
wingsail) and pass field values into each module — no more hardcoded
constants scattered across `examples/*.py`. To change the working scenario,
edit `default_scenario()` here; every example picks up the change.

This file is scoped to **Phase 4-5 only** (structural FEA + principal-
direction extraction): the geometry, materials, mesh, aero solver, and
the frame-field knobs. Phase 6 sizing / topology parameters are not
included on this branch.

All dataclasses are frozen so a scenario can't be accidentally mutated
during a run. Use `dataclasses.replace(params, target_element_size_m=…)`
to derive a modified scenario without touching the original.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .aero.cases import DESIGN_CASES, LoadCase
from .geometry.wing import WingSpec
from .materials.unidir import T700_EPOXY, UDPly


# ---------------------------------------------------------------------------
# Subsystem parameter groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialParameters:
    """Isotropic-equivalent material model for the spike pipeline.

    The UD ply material itself is stored in `DesignParameters.material`
    (a `UDPly`). This subdataclass holds the dimensionless knockdown +
    Poisson + safety-factor parameters that turn the UD ply into the
    isotropic-equivalent (E, ν, σ_allow) the Phase-4 FEA actually uses.
    """
    skin_E_knockdown: float = 0.5
    nu_isotropic: float = 0.32
    sigma_allow_safety_factor: float = 2.0


@dataclass(frozen=True)
class MeshParameters:
    """gmsh tet-mesh + shell-extraction resolution."""
    target_element_size_m: float = 0.05


@dataclass(frozen=True)
class AeroParameters:
    """LiftingLine + AeroBuildup solver knobs."""
    spanwise_resolution: int = 16


@dataclass(frozen=True)
class Phase5FrameFieldParameters:
    """Phase 5a-5b: principal-frame parametrization on the volumetric tet σ field."""
    n_levels: int = 24
    sigma_floor_fraction: float = 0.05
    sigma_augment_fraction: float = 1.0    # Phase 5d spar augmentation hack — obsolete with shell


@dataclass(frozen=True)
class SkinParameters:
    """Skin shell thickness used by the Phase 4b shell FEA."""
    t_baseline_m: float = 0.003


@dataclass(frozen=True)
class GenerativeParameters:
    """Knobs for the constraint-based generative truss stack (Milestone 1).

    Tractability levers keep the precomputed candidate menu and the CP-SAT
    model small; the structural/manufacturability bounds feed the constraints
    and the coverage proxy. See
    docs/superpowers/specs/2026-06-02-constraint-based-generation-design.md.
    """
    # Beam-count bounds (a mirror pair counts as its two physical beams)
    n_beams_min: int = 1  # M2: the gate guarantees adequacy; floor no longer governs
    n_beams_max: int = 40
    # Manufacturability
    box_max_height_m: float = 1.5
    beam_min_radius_m: float = 0.3
    cross_section_area_max_m2: float = 5.0e-3
    n_area_buckets: int = 6
    # Coverage proxy
    coverage_safety_factor: float = 2.0
    # Candidate-menu tractability levers
    max_node_count: int = 400
    max_library_size: int = 200
    poisson_disk_radius_m: float = 0.15
    n_z_layers: int = 12
    # Frame discretization for the gate: subdivide beam segments to this length so
    # the inboard span has nodes to receive distributed load (M2). None = no split.
    frame_max_element_length_m: float = 0.3
    # Gate limits (used by Plan 1B; defined here so the scenario is complete)
    tip_deflection_limit_m: float = 0.25
    # CP-SAT solve controls
    solver_max_time_s: float = 30.0
    top_n_designs: int = 8


# ---------------------------------------------------------------------------
# Umbrella container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignParameters:
    """Single source of truth for one wingsail design scenario.

    Pass an instance into the pipeline and it propagates through every
    phase. Use `default_scenario()` to get the project's current working
    scenario (5 m demo wingsail); use `dataclasses.replace(params, …)`
    to derive variants without mutating it.
    """

    geometry: WingSpec = field(default_factory=WingSpec)
    material: UDPly = T700_EPOXY
    material_iso: MaterialParameters = field(default_factory=MaterialParameters)
    load_cases: tuple[LoadCase, ...] = DESIGN_CASES
    mesh: MeshParameters = field(default_factory=MeshParameters)
    aero: AeroParameters = field(default_factory=AeroParameters)
    frame_field: Phase5FrameFieldParameters = field(default_factory=Phase5FrameFieldParameters)
    skin_sizing: SkinParameters = field(default_factory=SkinParameters)
    generative: GenerativeParameters = field(default_factory=GenerativeParameters)

    # Convenience properties derived from material + scaling factors

    @property
    def E_iso_Pa(self) -> float:
        """Isotropic-equivalent Young's modulus = E1 × knockdown."""
        return self.material.isotropic_equivalent_modulus(
            knockdown=self.material_iso.skin_E_knockdown,
        )

    @property
    def nu_iso(self) -> float:
        return self.material_iso.nu_isotropic

    @property
    def G_iso_Pa(self) -> float:
        return self.E_iso_Pa / (2.0 * (1.0 + self.material_iso.nu_isotropic))

    @property
    def sigma_allow_Pa(self) -> float:
        """Tensile allowable = Xt / safety_factor."""
        return self.material.allowable_tensile_stress(
            safety_factor=self.material_iso.sigma_allow_safety_factor,
        )

    @property
    def rho_kgm3(self) -> float:
        return self.material.rho_kgm3


# ---------------------------------------------------------------------------
# Default scenario
# ---------------------------------------------------------------------------


def default_scenario() -> DesignParameters:
    """Return the project's working scenario — the 5 m demo wingsail.

    Equivalent to `DesignParameters()` today, but kept as an explicit
    function so future scenarios (FPV drone wing, 100 m turbine blade)
    can be added as siblings:

        def fpv_drone_scenario() -> DesignParameters: ...
        def turbine_blade_scenario() -> DesignParameters: ...
    """
    return DesignParameters()
