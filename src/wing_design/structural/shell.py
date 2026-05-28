"""Flat-shell triangle FEA: CST membrane + DKT bending + drilling stabilization.

This is the architectural-refactor FEA — instead of meshing the wing as a
solid block and solving 3-D continuum elasticity, we mesh the **OML surface**
with triangles and treat the wing as a stressed skin (a structural shell)
that's bound together by an internal frame (added in a later session).

Each shell triangle has 18 DOFs (6/node × 3 nodes):

  * (u, v, w): translations in the local element frame  → membrane behavior
  * (θ_x, θ_y): in-plane bending rotations              → out-of-plane bending
  * θ_z: drilling rotation                              → no physical stiffness
    but required to make the global K positive-definite when adjacent
    triangles are non-coplanar; penalised with a small stiffness.

Element stiffness is the sum of three parts:

  K_e = K_membrane(CST) + K_bending(DKT) + K_drilling(penalty)

The membrane part is the standard Constant-Strain Triangle (CST): linear
basis, constant in-plane strain per element, 6 DOF (u, v) acting in 6×6.
The bending part is the Discrete Kirchhoff Triangle (Batoz, Bathe & Ho 1980),
which gives linearly-varying curvature per element and passes the patch
test in the thin-plate limit. The drilling penalty is a 3×3 spring stiffness
along the θ_z direction at each node.

All three parts are assembled in the element's local frame, then rotated to
the global frame via a 6×6 block-diagonal rotation matrix per node (3D
rotation applied independently to the translation block and the rotation
block).

Output: per-element membrane stress (σ_xx, σ_yy, σ_xy) and bending moment
(M_xx, M_yy, M_xy) in the element local frame, plus nodal displacements +
rotations in the global frame.

Reference:
  Batoz J.L., Bathe K.J., Ho L.W. (1980). "A study of three-node triangular
  plate bending elements." Int. J. Numer. Methods Eng. 15(12), 1771-1812.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Mesh container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellMesh:
    """Triangular shell mesh used by `solve_shell_elastic`.

    `nodes` are 3-D global coordinates. `triangles[k] = (i, j, l)` indexes
    nodes for the k-th element. `dirichlet_nodes` are clamped (all 6 DOFs).
    `loaded_tris` is a boolean mask flagging the triangles that receive
    panel loads — typically the wing OML above the spar attachment.
    """

    nodes: np.ndarray             # (N, 3)
    triangles: np.ndarray         # (M, 3) int
    dirichlet_nodes: np.ndarray   # (P,) int
    loaded_tris: np.ndarray       # (M,) bool

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_tris(self) -> int:
        return int(self.triangles.shape[0])


def shell_mesh_from_tet_mesh(tet_mesh) -> ShellMesh:
    """Build a `ShellMesh` from a `structural.mesh.TetMesh` of the wing.

    Uses the tet mesh's full boundary (every surface triangle) as the shell
    domain. Nodes that aren't on the surface are dropped and the triangle
    array is re-indexed. `dirichlet_nodes` carry over (spar-base clamp);
    `loaded_tris` flags the wing-OML subset (above z = 0, ignoring the
    spar's circular base face).
    """
    surf_tris = tet_mesh.surface_tris
    if surf_tris.size == 0:
        raise ValueError("Tet mesh has no surface triangles to make a shell from.")
    used_nodes = np.unique(surf_tris.ravel())
    remap = -np.ones(tet_mesh.n_nodes, dtype=np.int64)
    remap[used_nodes] = np.arange(used_nodes.shape[0])
    shell_nodes = tet_mesh.nodes[used_nodes]
    shell_tris = remap[surf_tris]
    # Carry over the Dirichlet clamp at the spar-base disk.
    shell_dirichlet = remap[tet_mesh.dirichlet_nodes]
    shell_dirichlet = shell_dirichlet[shell_dirichlet >= 0]
    # Loaded tris = those in tet_mesh.oml_tris (panel-loaded wing skin)
    # Match by sorted node-triple to identify which surface tris are in oml_tris.
    surf_keys = np.sort(surf_tris, axis=1)
    oml_keys = np.sort(tet_mesh.oml_tris, axis=1) if tet_mesh.oml_tris.size else np.empty((0, 3), int)
    surf_set = {tuple(k) for k in surf_keys}
    oml_set = {tuple(k) for k in oml_keys}
    loaded = np.array([tuple(k) in oml_set for k in surf_keys], dtype=bool)
    return ShellMesh(
        nodes=shell_nodes,
        triangles=shell_tris,
        dirichlet_nodes=shell_dirichlet.astype(np.int64),
        loaded_tris=loaded,
    )


# ---------------------------------------------------------------------------
# Element geometry helpers
# ---------------------------------------------------------------------------


def _triangle_local_frame(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (R_3x3, local_coords_3x2, area) for a triangle in 3-D.

    The local frame has e1 along (p2 − p1), e3 along the outward normal
    (right-hand rule from e1 toward p3), e2 = e3 × e1. `local_coords` holds
    the three vertices expressed in (e1, e2) — vertex 1 is the origin.
    """
    v12 = p2 - p1
    v13 = p3 - p1
    e1 = v12 / np.linalg.norm(v12)
    n = np.cross(v12, v13)
    n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        raise ValueError("Degenerate triangle (collinear vertices).")
    e3 = n / n_norm
    e2 = np.cross(e3, e1)
    R = np.column_stack([e1, e2, e3])    # global = R @ local
    # Local 2-D coordinates of the three vertices
    x1 = np.array([0.0, 0.0])
    x2 = np.array([float(v12 @ e1), float(v12 @ e2)])
    x3 = np.array([float(v13 @ e1), float(v13 @ e2)])
    local = np.stack([x1, x2, x3])
    area = 0.5 * float(n_norm)
    return R, local, area


# ---------------------------------------------------------------------------
# CST membrane element
# ---------------------------------------------------------------------------


def _cst_membrane_K(local_coords: np.ndarray, area: float, E: float, nu: float, t: float) -> np.ndarray:
    """6×6 CST membrane stiffness in element-local frame (3 nodes × (u, v))."""
    x1, y1 = local_coords[0]
    x2, y2 = local_coords[1]
    x3, y3 = local_coords[2]
    # Shape-function gradients in local coords (constant for CST)
    # ∂N_i/∂x = b_i / (2A),  ∂N_i/∂y = c_i / (2A)
    b = np.array([y2 - y3, y3 - y1, y1 - y2])     # (3,)
    c = np.array([x3 - x2, x1 - x3, x2 - x1])     # (3,)
    two_A = 2.0 * area
    # Strain-displacement matrix B_m (3×6): rows = (ε_xx, ε_yy, γ_xy)
    Bm = np.zeros((3, 6))
    for i in range(3):
        Bm[0, 2 * i + 0] = b[i] / two_A
        Bm[1, 2 * i + 1] = c[i] / two_A
        Bm[2, 2 * i + 0] = c[i] / two_A
        Bm[2, 2 * i + 1] = b[i] / two_A
    # Plane-stress elasticity matrix (3×3)
    factor = E / (1.0 - nu * nu)
    Dm = factor * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])
    K = t * area * Bm.T @ Dm @ Bm
    return K


# ---------------------------------------------------------------------------
# DKT bending element  (Batoz, Bathe, Ho 1980)
# ---------------------------------------------------------------------------


def _dkt_bending_K(local_coords: np.ndarray, area: float, E: float, nu: float, t: float) -> np.ndarray:
    """9×9 DKT bending stiffness in element-local frame (3 nodes × (w, θ_x, θ_y)).

    The Hermite-edge rotations are condensed out before returning; we
    evaluate the curvature operator B_b at three Gauss points (mid-edges)
    and integrate analytically — for linearly-varying B on a triangle the
    3-point mid-edge rule is exact.
    """
    x1, y1 = local_coords[0]
    x2, y2 = local_coords[1]
    x3, y3 = local_coords[2]
    # Edge vectors (vertex k → vertex k+1, cyclic)
    x23, y23 = x2 - x3, y2 - y3
    x31, y31 = x3 - x1, y3 - y1
    x12, y12 = x1 - x2, y1 - y2
    # Edge squared lengths
    l23_sq = x23 * x23 + y23 * y23
    l31_sq = x31 * x31 + y31 * y31
    l12_sq = x12 * x12 + y12 * y12
    # DKT auxiliary constants (Batoz et al. eq 22-26)
    P4, P5, P6 = -6.0 * x23 / l23_sq, -6.0 * x31 / l31_sq, -6.0 * x12 / l12_sq
    t4, t5, t6 = -6.0 * y23 / l23_sq, -6.0 * y31 / l31_sq, -6.0 * y12 / l12_sq
    q4 = 3.0 * x23 * y23 / l23_sq
    q5 = 3.0 * x31 * y31 / l31_sq
    q6 = 3.0 * x12 * y12 / l12_sq
    r4 = 3.0 * y23 * y23 / l23_sq
    r5 = 3.0 * y31 * y31 / l31_sq
    r6 = 3.0 * y12 * y12 / l12_sq
    # The B-matrix at a Gauss point (ξ, η). Batoz et al. derive it via
    # Hermite cubic edge interpolation of β_x, β_y. The β functions and their
    # derivatives at the three mid-edge points are tabulated below.
    # We use the three-point mid-edge quadrature (η, ξ pairs):
    gauss = [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
    K = np.zeros((9, 9))
    for xi, eta in gauss:
        Bb = _dkt_B_matrix(
            xi, eta,
            x23, y23, x31, y31, x12, y12,
            P4, P5, P6, t4, t5, t6, q4, q5, q6, r4, r5, r6,
        )
        # Bending elasticity (plate)
        factor = E * t ** 3 / (12.0 * (1.0 - nu * nu))
        Db = factor * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])
        K += (1.0 / 3.0) * area * Bb.T @ Db @ Bb       # weight = 1/3 per Gauss point
    return K


def _dkt_B_matrix(
    xi: float, eta: float,
    x23: float, y23: float, x31: float, y31: float, x12: float, y12: float,
    P4: float, P5: float, P6: float,
    t4: float, t5: float, t6: float,
    q4: float, q5: float, q6: float,
    r4: float, r5: float, r6: float,
) -> np.ndarray:
    """Curvature B-matrix (3×9) at a Gauss point (xi, eta) on the reference tri.

    Reference frame: ξ ∈ [0, 1], η ∈ [0, 1−ξ], so vertices are at (0,0),
    (1,0), (0,1). Returns rows for (κ_xx, κ_yy, 2·κ_xy) versus the 9 DOFs
    (w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3).
    """
    # First-derivative shape functions for the Hermite rotation field β_x, β_y.
    # Following Batoz et al., we form Hx and Hy — 9-vectors — that interpolate
    # β_x and β_y from the 9 DOFs. Their derivatives w.r.t. ξ and η give the
    # curvature B matrix.
    zeta = 1.0 - xi - eta
    # β_x = Hx ⋅ d   where d = (w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3)
    # We need ∂β_x/∂x and ∂β_x/∂y — chain through the Jacobian.
    # For ease we compute ∂Hx/∂ξ and ∂Hx/∂η, then map to global derivatives.
    Hx_xi, Hx_eta, Hy_xi, Hy_eta = _dkt_shape_derivs(
        xi, eta,
        P4, P5, P6, t4, t5, t6, q4, q5, q6, r4, r5, r6,
    )
    # Jacobian of the (x, y) mapping in element-local frame.
    # x(ξ, η) = ξ x1 + η x2 + ζ x3   (with linear-tri shape functions on x, y)
    # wait — we use ξ at vertex 2, η at vertex 3 here; vertex 1 is the origin.
    # Re-derive: x(ξ, η) = (1−ξ−η) x1 + ξ x2 + η x3 ⇒ ∂x/∂ξ = x2 − x1, etc.
    # We have x1 = (0, 0), so ∂x/∂ξ = x2, ∂x/∂η = x3. Same for y.
    # Use x23, y23, x31, y31, x12, y12 with the convention they were defined.
    # x2 − x1 = (x12 negated) actually x12 = x1 − x2, so x2 − x1 = −x12.
    dx_dxi = -x12
    dy_dxi = -y12
    dx_deta = x31           # x31 = x3 − x1; vertex 1 is origin → x3 directly
    dy_deta = y31
    detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
    inv = (1.0 / detJ) * np.array([[dy_deta, -dy_dxi], [-dx_deta, dx_dxi]])
    # ∂Hx/∂x = inv[0,0] Hx_xi + inv[0,1] Hx_eta
    # ∂Hx/∂y = inv[1,0] Hx_xi + inv[1,1] Hx_eta
    Hx_x = inv[0, 0] * Hx_xi + inv[0, 1] * Hx_eta
    Hx_y = inv[1, 0] * Hx_xi + inv[1, 1] * Hx_eta
    Hy_x = inv[0, 0] * Hy_xi + inv[0, 1] * Hy_eta
    Hy_y = inv[1, 0] * Hy_xi + inv[1, 1] * Hy_eta
    # Curvature operator (κ_xx = ∂β_x/∂x, κ_yy = ∂β_y/∂y, 2κ_xy = ∂β_x/∂y + ∂β_y/∂x)
    B = np.zeros((3, 9))
    B[0, :] = Hx_x
    B[1, :] = Hy_y
    B[2, :] = Hx_y + Hy_x
    return B


def _dkt_shape_derivs(
    xi: float, eta: float,
    P4: float, P5: float, P6: float,
    t4: float, t5: float, t6: float,
    q4: float, q5: float, q6: float,
    r4: float, r5: float, r6: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derivatives of DKT shape functions Hx, Hy w.r.t. (ξ, η).

    Returns (∂Hx/∂ξ, ∂Hx/∂η, ∂Hy/∂ξ, ∂Hy/∂η), each a 9-vector indexed by
    DOF in the order (w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3).
    """
    # Pre-compute partials of edge Hermite functions (see Batoz et al.)
    zeta = 1.0 - xi - eta
    # Hx (β_x) derivative w.r.t. ξ — 9-vector
    Hx_xi = np.zeros(9)
    Hx_xi[0] = P6 * (1.0 - 2.0 * xi) + (P5 - P6) * eta
    Hx_xi[1] = q6 * (1.0 - 2.0 * xi) - (q5 + q6) * eta
    Hx_xi[2] = -4.0 + 6.0 * (xi + eta) + r6 * (1.0 - 2.0 * xi) - eta * (r5 + r6)
    Hx_xi[3] = -P6 * (1.0 - 2.0 * xi) + (P4 + P6) * eta
    Hx_xi[4] = q6 * (1.0 - 2.0 * xi) - (q6 - q4) * eta
    Hx_xi[5] = -2.0 + 6.0 * xi + r6 * (1.0 - 2.0 * xi) + eta * (r4 - r6)
    Hx_xi[6] = -eta * (P5 + P4)
    Hx_xi[7] = eta * (q4 - q5)
    Hx_xi[8] = -eta * (r5 - r4)

    # Hx derivative w.r.t. η
    Hx_eta = np.zeros(9)
    Hx_eta[0] = -P5 * (1.0 - 2.0 * eta) - (P6 - P5) * xi
    Hx_eta[1] = q5 * (1.0 - 2.0 * eta) - (q5 + q6) * xi
    Hx_eta[2] = -4.0 + 6.0 * (xi + eta) + r5 * (1.0 - 2.0 * eta) - xi * (r5 + r6)
    Hx_eta[3] = xi * (P4 + P6)
    Hx_eta[4] = xi * (q4 - q6)
    Hx_eta[5] = -xi * (r6 - r4)
    Hx_eta[6] = P5 * (1.0 - 2.0 * eta) - xi * (P4 + P5)
    Hx_eta[7] = q5 * (1.0 - 2.0 * eta) + xi * (q4 - q5)
    Hx_eta[8] = -2.0 + 6.0 * eta + r5 * (1.0 - 2.0 * eta) + xi * (r4 - r5)

    # Hy is the mirror with t / q-derived terms (signs as in Batoz et al.)
    Hy_xi = np.zeros(9)
    Hy_xi[0] = t6 * (1.0 - 2.0 * xi) + eta * (t5 - t6)
    Hy_xi[1] = 1.0 + r6 * (1.0 - 2.0 * xi) - eta * (r5 + r6)
    Hy_xi[2] = -q6 * (1.0 - 2.0 * xi) + eta * (q5 + q6)
    Hy_xi[3] = -t6 * (1.0 - 2.0 * xi) + eta * (t4 + t6)
    Hy_xi[4] = -1.0 + r6 * (1.0 - 2.0 * xi) + eta * (r4 - r6)
    Hy_xi[5] = -q6 * (1.0 - 2.0 * xi) - eta * (q4 - q6)
    Hy_xi[6] = -eta * (t4 + t5)
    Hy_xi[7] = eta * (r4 - r5)
    Hy_xi[8] = -eta * (q4 - q5)

    Hy_eta = np.zeros(9)
    Hy_eta[0] = -t5 * (1.0 - 2.0 * eta) - xi * (t6 - t5)
    Hy_eta[1] = 1.0 + r5 * (1.0 - 2.0 * eta) - xi * (r5 + r6)
    Hy_eta[2] = -q5 * (1.0 - 2.0 * eta) + xi * (q5 + q6)
    Hy_eta[3] = xi * (t4 + t6)
    Hy_eta[4] = xi * (r4 - r6)
    Hy_eta[5] = -xi * (q4 - q6)
    Hy_eta[6] = t5 * (1.0 - 2.0 * eta) - xi * (t4 + t5)
    Hy_eta[7] = -1.0 + r5 * (1.0 - 2.0 * eta) + xi * (r4 - r5)
    Hy_eta[8] = -q5 * (1.0 - 2.0 * eta) - xi * (q4 - q5)

    return Hx_xi, Hx_eta, Hy_xi, Hy_eta


# ---------------------------------------------------------------------------
# Element K (18×18 in global frame)
# ---------------------------------------------------------------------------


def _element_K(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
    *, E: float, nu: float, t: float,
    drilling_factor: float = 1.0e-4,
) -> np.ndarray:
    """Full shell element stiffness, 18×18 in global frame.

    Local DOFs per node (in order): u, v, w, θ_x, θ_y, θ_z.
    `drilling_factor` is the dimensionless penalty on θ_z relative to
    the membrane shear modulus × volume (typically 1e-3 to 1e-6).
    """
    R, local, area = _triangle_local_frame(p1, p2, p3)
    Km = _cst_membrane_K(local, area, E, nu, t)      # 6×6 on (u, v)
    Kb = _dkt_bending_K(local, area, E, nu, t)       # 9×9 on (w, θ_x, θ_y)

    Ke_local = np.zeros((18, 18))
    # Membrane (u, v) → DOFs 0, 1, 6, 7, 12, 13
    membrane_dofs = [0, 1, 6, 7, 12, 13]
    for a, da in enumerate(membrane_dofs):
        for b, db in enumerate(membrane_dofs):
            Ke_local[da, db] += Km[a, b]
    # Bending (w, θ_x, θ_y) → DOFs 2, 3, 4, 8, 9, 10, 14, 15, 16
    bending_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16]
    for a, da in enumerate(bending_dofs):
        for b, db in enumerate(bending_dofs):
            Ke_local[da, db] += Kb[a, b]
    # Drilling (θ_z) → DOFs 5, 11, 17. Penalty proportional to membrane shear.
    G = E / (2.0 * (1.0 + nu))
    drill_k = drilling_factor * G * t * area
    for d in (5, 11, 17):
        Ke_local[d, d] += drill_k

    # Rotate to global frame. The 18×18 transformation is block-diagonal with
    # six 3×3 R blocks (translation block + rotation block per node).
    T = np.zeros((18, 18))
    for n in range(3):
        for k in range(2):  # 2 blocks per node: translation (DOFs 0-2), rotation (DOFs 3-5)
            base = 6 * n + 3 * k
            T[base:base + 3, base:base + 3] = R
    return T @ Ke_local @ T.T


# ---------------------------------------------------------------------------
# Result + solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellFEAResult:
    displacements: np.ndarray         # (N, 6) — global frame: u, v, w, θ_x, θ_y, θ_z
    membrane_stress: np.ndarray       # (M, 3) — σ_xx, σ_yy, σ_xy in element-local frame
    bending_moment: np.ndarray        # (M, 3) — M_xx, M_yy, M_xy in element-local frame
    element_local_frames: np.ndarray  # (M, 3, 3) — R per element (global = R @ local)
    element_areas: np.ndarray         # (M,)

    @property
    def n_nodes(self) -> int:
        return int(self.displacements.shape[0])

    def membrane_von_mises(self) -> np.ndarray:
        sxx = self.membrane_stress[:, 0]
        syy = self.membrane_stress[:, 1]
        sxy = self.membrane_stress[:, 2]
        return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)

    def membrane_principal_2d(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-element 2-D principal stresses (σ_1 ≥ σ_2) and angles θ (rad)
        of the max-stress direction relative to the element-local x axis."""
        sxx = self.membrane_stress[:, 0]
        syy = self.membrane_stress[:, 1]
        sxy = self.membrane_stress[:, 2]
        mean = 0.5 * (sxx + syy)
        radius = np.sqrt(0.25 * (sxx - syy) ** 2 + sxy ** 2)
        sigma1 = mean + radius
        sigma2 = mean - radius
        theta = 0.5 * np.arctan2(2.0 * sxy, (sxx - syy + 1.0e-30))
        return np.stack([sigma1, sigma2], axis=1), theta

    def membrane_principal_dirs_3d(self) -> np.ndarray:
        """Per-element 3-D unit vectors for the two principal directions.

        Returns an array of shape (M, 2, 3): [e, k, :] is the global-frame
        direction of principal stress k ∈ {0 = max, 1 = min} at element e.
        These are the stress-line tracing directions for Phase 5.
        """
        _, theta = self.membrane_principal_2d()
        c, s = np.cos(theta), np.sin(theta)
        # Local 2-D principal directions (in element frame's first 2 axes)
        v1_local = np.stack([c, s, np.zeros_like(c)], axis=1)            # (M, 3)
        v2_local = np.stack([-s, c, np.zeros_like(c)], axis=1)
        # Rotate to global: v_global = R @ v_local
        R = self.element_local_frames                                     # (M, 3, 3)
        v1_global = np.einsum("eij,ej->ei", R, v1_local)
        v2_global = np.einsum("eij,ej->ei", R, v2_local)
        return np.stack([v1_global, v2_global], axis=1)                  # (M, 2, 3)


def solve_shell_elastic(
    mesh: ShellMesh,
    *,
    E: float,
    nu: float,
    thickness_m: float,
    tri_force_vectors: np.ndarray,
    drilling_factor: float = 1.0e-4,
) -> ShellFEAResult:
    """Solve the linear shell K u = f problem.

    Parameters
    ----------
    mesh : ShellMesh
        Triangular shell mesh with Dirichlet-clamped nodes already tagged.
    E, nu : float
        Isotropic-equivalent skin material.
    thickness_m : float
        Uniform shell thickness (m).
    tri_force_vectors : (M, 3) float
        Total force per triangle in the global frame. Distributed equally
        to the three vertices (consistent for constant traction).
    drilling_factor : float
        Penalty multiplier on θ_z DOFs (dimensionless).
    """
    if tri_force_vectors.shape != (mesh.n_tris, 3):
        raise ValueError(
            f"tri_force_vectors must be ({mesh.n_tris}, 3), got {tri_force_vectors.shape}"
        )
    N = mesh.n_nodes
    M = mesh.n_tris
    dof_per_node = 6

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    element_Rs = np.empty((M, 3, 3))
    element_As = np.empty(M)

    for e in range(M):
        i, j, l = (int(v) for v in mesh.triangles[e])
        p1, p2, p3 = mesh.nodes[i], mesh.nodes[j], mesh.nodes[l]
        Ke = _element_K(p1, p2, p3, E=E, nu=nu, t=thickness_m, drilling_factor=drilling_factor)
        R, _, area = _triangle_local_frame(p1, p2, p3)
        element_Rs[e] = R
        element_As[e] = area
        dofs = np.array([
            6 * i + 0, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5,
            6 * j + 0, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5,
            6 * l + 0, 6 * l + 1, 6 * l + 2, 6 * l + 3, 6 * l + 4, 6 * l + 5,
        ], dtype=np.int64)
        rr = np.repeat(dofs, 18)
        cc = np.tile(dofs, 18)
        rows.append(rr)
        cols.append(cc)
        vals.append(Ke.ravel())

    K = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dof_per_node * N, dof_per_node * N),
    ).tocsr()

    # Build the load vector: split each triangle's force equally to its three nodes.
    f = np.zeros(dof_per_node * N)
    third = tri_force_vectors / 3.0
    for e in range(M):
        i, j, l = (int(v) for v in mesh.triangles[e])
        F3 = third[e]
        for n in (i, j, l):
            f[6 * n + 0] += F3[0]
            f[6 * n + 1] += F3[1]
            f[6 * n + 2] += F3[2]

    # Dirichlet: clamp all 6 DOFs of every dirichlet node.
    free = np.ones(dof_per_node * N, dtype=bool)
    for n in mesh.dirichlet_nodes:
        base = dof_per_node * int(n)
        free[base:base + dof_per_node] = False

    u = np.zeros(dof_per_node * N)
    Kff = K[free][:, free]
    u[free] = spla.spsolve(Kff.tocsc(), f[free])
    displacements = u.reshape(N, dof_per_node)

    # Stress recovery: rebuild local B matrices once per element and project
    # (u_local) onto the membrane and bending DOFs to get σ_m and M_b.
    membrane_stress = np.zeros((M, 3))
    bending_moment = np.zeros((M, 3))
    for e in range(M):
        i, j, l = (int(v) for v in mesh.triangles[e])
        R = element_Rs[e]
        # Build local frame
        p1, p2, p3 = mesh.nodes[i], mesh.nodes[j], mesh.nodes[l]
        _, local, area = _triangle_local_frame(p1, p2, p3)
        # Membrane B
        x1, y1 = local[0]; x2, y2 = local[1]; x3, y3 = local[2]
        b_grad = np.array([y2 - y3, y3 - y1, y1 - y2])
        c_grad = np.array([x3 - x2, x1 - x3, x2 - x1])
        two_A = 2.0 * area
        Bm = np.zeros((3, 6))
        for k in range(3):
            Bm[0, 2 * k + 0] = b_grad[k] / two_A
            Bm[1, 2 * k + 1] = c_grad[k] / two_A
            Bm[2, 2 * k + 0] = c_grad[k] / two_A
            Bm[2, 2 * k + 1] = b_grad[k] / two_A
        factor = E / (1.0 - nu * nu)
        Dm = factor * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])
        # Rotate nodal displacements to local frame; extract (u, v) for each node
        u_local = np.zeros(18)
        for n_idx, gn in enumerate((i, j, l)):
            u_global_trans = displacements[gn, 0:3]
            u_global_rot = displacements[gn, 3:6]
            u_local[6 * n_idx + 0:6 * n_idx + 3] = R.T @ u_global_trans
            u_local[6 * n_idx + 3:6 * n_idx + 6] = R.T @ u_global_rot
        u_membrane = u_local[[0, 1, 6, 7, 12, 13]]      # (u, v) at 3 nodes
        membrane_stress[e] = (Dm @ Bm @ u_membrane) * thickness_m   # stress resultant per unit width
        # NOTE: the line above gives membrane force per unit width (N_xx, etc.)
        # in N/m. To get stress in Pa, divide by thickness:
        membrane_stress[e] /= thickness_m
        # Bending moments — evaluate at element centroid (xi=eta=1/3).
        # Reuse the DKT B matrix.
        u_bending = u_local[[2, 3, 4, 8, 9, 10, 14, 15, 16]]
        # Recompute DKT constants
        x23, y23 = x2 - x3, y2 - y3
        x31, y31 = x3 - x1, y3 - y1
        x12, y12 = x1 - x2, y1 - y2
        l23_sq = x23 * x23 + y23 * y23
        l31_sq = x31 * x31 + y31 * y31
        l12_sq = x12 * x12 + y12 * y12
        P4, P5, P6 = -6.0 * x23 / l23_sq, -6.0 * x31 / l31_sq, -6.0 * x12 / l12_sq
        t4, t5, t6 = -6.0 * y23 / l23_sq, -6.0 * y31 / l31_sq, -6.0 * y12 / l12_sq
        q4 = 3.0 * x23 * y23 / l23_sq
        q5 = 3.0 * x31 * y31 / l31_sq
        q6 = 3.0 * x12 * y12 / l12_sq
        r4 = 3.0 * y23 * y23 / l23_sq
        r5 = 3.0 * y31 * y31 / l31_sq
        r6 = 3.0 * y12 * y12 / l12_sq
        Bb_centroid = _dkt_B_matrix(
            1.0 / 3.0, 1.0 / 3.0,
            x23, y23, x31, y31, x12, y12,
            P4, P5, P6, t4, t5, t6, q4, q5, q6, r4, r5, r6,
        )
        Db = (E * thickness_m ** 3 / (12.0 * (1.0 - nu * nu))) * np.array([
            [1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0],
        ])
        bending_moment[e] = Db @ Bb_centroid @ u_bending

    return ShellFEAResult(
        displacements=displacements,
        membrane_stress=membrane_stress,
        bending_moment=bending_moment,
        element_local_frames=element_Rs,
        element_areas=element_As,
    )
