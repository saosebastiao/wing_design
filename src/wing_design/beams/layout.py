"""Stress-driven beam layout helpers (Phase F): place beams where the skin is loaded.

`stress_weighted_targets` turns per-perimeter-segment stress weights into N arc-fraction
targets at equal cumulative weight (clustering beams where stress is high); uniform
weights give exactly even spacing. `cross_section_stress_weights` derives those weights
from a baseline beam-shell solve's skin membrane stress.
"""
from __future__ import annotations

import numpy as np

from ..structural.shell import membrane_von_mises, recover_membrane_stress
from .shell_model import BeamShellModel, solve_beam_shell_model


def stress_weighted_targets(weights: np.ndarray, n: int) -> np.ndarray:
    """(n,) arc fractions in [0,1) placing N beams at equal cumulative ``weights``.

    `weights[i]` is the stress intensity (density) of perimeter segment i of m equal
    arc-length segments. Beams are placed where the cumulative weight reaches
    (b/n)*total, so they cluster where weights are high. Uniform weights -> even
    spacing (b/n). Returns ascending fractions.
    """
    w = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    m = w.shape[0]
    seg = 1.0 / m
    cum = np.concatenate([[0.0], np.cumsum(w * seg)])
    boundaries = np.concatenate([np.arange(m) / m, [1.0]])
    total = cum[-1]
    target_w = (np.arange(n) / n) * total
    return np.interp(target_w, cum, boundaries)


def chord_symmetrize_weights(weights: np.ndarray) -> np.ndarray:
    """Make per-perimeter-segment weights mirror-symmetric across the chord (max-of-mirror).

    Segment ``b`` (between beams b and b+1) mirrors segment ``(n-1-b) % n`` across the
    chord plane (the section is traversed TE->upper->LE->lower->TE). Returns
    ``w_sym[b] = max(weights[b], weights[(n-1-b) % n])`` so stress-weighted placement
    (``stress_weighted_targets``) yields a chord-symmetric beam layout that keeps the
    radius-symmetry grouping (``beam_radius_groups``) active. Both-tack-conservative.
    """
    w = np.asarray(weights, dtype=float)
    n = w.shape[0]
    mirror = w[(n - 1 - np.arange(n)) % n]
    return np.maximum(w, mirror)


def cross_section_stress_weights(model: BeamShellModel, load_arrays: list[np.ndarray]) -> np.ndarray:
    """(n_beams,) per-perimeter-segment skin stress intensity from a baseline solve.

    For each perimeter segment (the skin panel between beams b and b+1), the peak
    membrane von Mises over its triangles, all z-levels, and all load cases.
    """
    nb = model.n_beams
    w = np.zeros(nb)
    for loads in load_arrays:
        res = solve_beam_shell_model(model, loads)
        vm = membrane_von_mises(
            recover_membrane_stress(model.nodes, model.shell_tris, res.displacements,
                                    E=model.E_skin, nu=model.nu_skin)
        )
        for e in range(model.shell_tris.shape[0]):
            b = (e // 2) % nb
            w[b] = max(w[b], float(vm[e]))
    return w
