"""Centralized design parameters for the wingsail project.

Every parameter the project uses lives in `DesignParameters`. Examples
construct one instance (via `small_scenario()`/`medium_scenario()`/
`large_scenario()` for the demo wingsails) and pass field values into each
module — no more hardcoded constants scattered across `examples/*.py`. To
change a working scenario, edit the relevant builder here; every example
picks up the change.

This file is scoped to **Phase 4-5 only** (structural FEA + principal-
direction extraction): the geometry, materials, mesh, aero solver, and
the frame-field knobs. Phase 6 sizing / topology parameters are not
included on this branch.

All dataclasses are frozen so a scenario can't be accidentally mutated
during a run. Use `dataclasses.replace(params, target_element_size_m=…)`
to derive a modified scenario without touching the original.
"""
from __future__ import annotations

from dataclasses import dataclass

from .aero.cases import DESIGN_CASES, LoadCase
from .geometry.wing import WingSpec
from .geometry.wingsails import small_wingsail, medium_wingsail, large_wingsail
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
    skin_E_knockdown: float
    nu_isotropic: float
    sigma_allow_safety_factor: float


@dataclass(frozen=True)
class MeshParameters:
    """gmsh tet-mesh + shell-extraction resolution."""
    target_element_size_m: float


@dataclass(frozen=True)
class AeroParameters:
    """LiftingLine + AeroBuildup solver knobs."""
    spanwise_resolution: int


@dataclass(frozen=True)
class Phase5FrameFieldParameters:
    """Phase 5a-5b: principal-frame parametrization on the volumetric tet σ field."""
    n_levels: int
    sigma_floor_fraction: float
    sigma_augment_fraction: float    # Phase 5d spar augmentation hack — obsolete with shell


@dataclass(frozen=True)
class SkinParameters:
    """Skin shell thickness used by the Phase 4b shell FEA."""
    t_baseline_m: float


# ---------------------------------------------------------------------------
# Umbrella container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignParameters:
    """Single source of truth for one wingsail design scenario.

    Pass an instance into the pipeline and it propagates through every
    phase. Use `small_scenario()`/`medium_scenario()`/`large_scenario()` to
    get a working scenario; use `dataclasses.replace(params, …)` to derive
    variants without mutating it.
    """

    geometry: WingSpec
    material: UDPly
    material_iso: MaterialParameters
    load_cases: tuple[LoadCase, ...]
    mesh: MeshParameters
    aero: AeroParameters
    frame_field: Phase5FrameFieldParameters
    skin_sizing: SkinParameters

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


# ---------------------------------------------------------------------------
# Named shared parameter sets (resolution / material knobs, size-independent)
# ---------------------------------------------------------------------------

STANDARD_MATERIAL_ISO = MaterialParameters(
    skin_E_knockdown=0.5, nu_isotropic=0.32, sigma_allow_safety_factor=2.0)
STANDARD_MESH = MeshParameters(target_element_size_m=0.05)
STANDARD_AERO = AeroParameters(spanwise_resolution=16)
STANDARD_FRAME_FIELD = Phase5FrameFieldParameters(
    n_levels=24, sigma_floor_fraction=0.05, sigma_augment_fraction=1.0)
STANDARD_SKIN = SkinParameters(t_baseline_m=0.003)


def _scenario(geometry: WingSpec) -> DesignParameters:
    return DesignParameters(
        geometry=geometry,
        material=T700_EPOXY,
        material_iso=STANDARD_MATERIAL_ISO,
        load_cases=DESIGN_CASES,
        mesh=STANDARD_MESH,
        aero=STANDARD_AERO,
        frame_field=STANDARD_FRAME_FIELD,
        skin_sizing=STANDARD_SKIN,
    )


def small_scenario() -> DesignParameters:
    """Project working scenario at the 5 m demo wingsail (used by the test suite)."""
    return _scenario(small_wingsail)


def medium_scenario() -> DesignParameters:
    """22 m wingsail scenario (used by the examples)."""
    return _scenario(medium_wingsail)


def large_scenario() -> DesignParameters:
    """45 m wingsail scenario."""
    return _scenario(large_wingsail)
