"""@njit helpers + frozen Li-6 D2 constants for the live diagonalizer model.

The sympy-built constant matrices (``build_constant_matrices``), the dipole
coupling tensor and the diabatic order map (``build_order_map``) are constructed
ONCE here at import time in the Python layer via
``src.diagonalizer_setup.get_li6_setup`` and bound as module-level frozen
globals. Numba bakes these read-only arrays into the compiled @njit code as
compile-time constants, so the live jitclass needs NO array state, stays
genuinely zero-arg, and is thread-safe under prange (no shared mutable state,
Pitfall 1).

sympy is pulled in transitively by ``diagonalizer_setup`` at THIS import only;
it never crosses the JIT boundary.

CACHE CAVEAT: because the constants are frozen into the compiled njit code at
first compile, changing the Li-6 D2 physics constants later requires clearing
Numba's on-disk cache — the old values stay baked into the cached kernels.
"""

from __future__ import annotations

import numpy as np
from numba import njit

import src.diagonalizer_setup as ds

H_PLANCK = ds.H_PLANCK

# Build the fixed Li-6 D2 setup once: constant matrices, coupling tensor,
# diabatic order map and the cycling-transition strength normalization.
_SETUP = ds.get_li6_setup()

# Module-level frozen constants read directly by the @njit helpers below.
_H_HFS_G = np.ascontiguousarray(_SETUP["Hg_hfs"], dtype=np.float64)   # (6, 6)
_H_ZEE_G = np.ascontiguousarray(_SETUP["Hg_zee"], dtype=np.float64)   # (6, 6)
_H_HFS_E = np.ascontiguousarray(_SETUP["He_hfs"], dtype=np.float64)   # (12, 12)
_H_ZEE_E = np.ascontiguousarray(_SETUP["He_zee"], dtype=np.float64)   # (12, 12)
_COUPLING = np.ascontiguousarray(_SETUP["coupling"], dtype=np.float64)  # (3,12,6)
_ORDER_MAP_G = np.ascontiguousarray(_SETUP["order_g"], dtype=np.int32)  # (n, 6)
_ORDER_MAP_E = np.ascontiguousarray(_SETUP["order_e"], dtype=np.int32)  # (n, 12)
_B_AXIS = np.ascontiguousarray(_SETUP["b_axis_g"], dtype=np.float64)    # (n,)
_SCALE = float(_SETUP["scale"])

# B=0 line-center reference (single scalar, Hz). Anchors the cycling transition
# GS5(4) -> ES11(10) to the Lithium18LevelInteraction zero-field position so both models
# add their shift onto the SAME base transition_frequency the kernel uses 
_CYCLING_GROUND_OFFSET = -76.75e6  # Hz (calibrated to full-18 B=0 cycling line)
_eg0 = np.linalg.eigh(_H_HFS_G)[0]  # B=0 eigenvalues, ascending
_ee0 = np.linalg.eigh(_H_HFS_E)[0]
_LINE_CENTER = float(
    (_ee0[_ORDER_MAP_E[0, 11]] - _eg0[_ORDER_MAP_G[0, 5]]) / H_PLANCK
    - _CYCLING_GROUND_OFFSET
)


@njit
def _nearest_node(B):
    """Nearest node on the (non-uniform) order-map grid for the diabatic reorder.

    The grid is log-dense below ~1 G (2e-4 T), where the excited-state HFS and
    Zeeman energies cross over and the eigenstate ordering is most volatile, so
    the nearest node closely matches the actual field and its permutation applies
    without skipping a level crossing. |B| is clamped to the grid endpoints.
    Binary search (the grid is sorted, non-uniform).
    """
    n = _B_AXIS.shape[0]
    if B <= _B_AXIS[0]:
        return 0
    if B >= _B_AXIS[n - 1]:
        return n - 1
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _B_AXIS[mid] <= B:
            lo = mid
        else:
            hi = mid
    if (B - _B_AXIS[lo]) <= (_B_AXIS[hi] - B):
        return lo
    return hi


@njit
def _bracket(b_axis, b):
    """Bracketing (lo, hi, frac) for linear interp of a sorted ``b_axis`` at |B|.

    ``b_axis`` is strictly increasing but NOT necessarily uniform (the table is
    log-dense below ~1 G, validated at load time), so the interval is found by
    binary search; out-of-range |B| clamps to an endpoint (lo == hi, frac 0).
    Callers read the two bracketing table scalars directly. NOTE: the table
    model's per-substep cost is dominated NOT by this helper but by the repeated
    jitclass array-attribute reads (self.b_axis/pos_table/strength_table) in the
    caller — a numba refcount overhead the fitted model avoids by using
    module-level data. Fixing that needs a batched-lookup API, not a change here.
    """
    n = b_axis.shape[0]
    if b <= b_axis[0]:
        return 0, 0, 0.0
    if b >= b_axis[n - 1]:
        return n - 1, n - 1, 0.0
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if b_axis[mid] <= b:
            lo = mid
        else:
            hi = mid
    return lo, hi, (b - b_axis[lo]) / (b_axis[hi] - b_axis[lo])


@njit
def _summe(pol, vg, ve):
    """Coupling contraction sum_a sum_b coupling[pol, a, b] * ve[a] * vg[b]."""
    s = 0.0
    for a in range(ve.shape[0]):
        tmp = 0.0
        for b in range(vg.shape[0]):
            tmp += _COUPLING[pol, a, b] * vg[b]
        s += ve[a] * tmp
    return s


@njit
def shift(gs, es, B):
    """Field-induced transition-frequency shift in Hz (added to f0).

    Diagonalizes H(|B|) = H_hfs + |B|*H_Zeeman for both manifolds, applies the
    precomputed diabatic reorder at the nearest B-node, and returns the labeled
    line position minus the B=0 line center.
    """
    eg = np.linalg.eigh(_H_HFS_G + B * _H_ZEE_G)[0]
    ee = np.linalg.eigh(_H_HFS_E + B * _H_ZEE_E)[0]
    node = _nearest_node(B)
    gi = _ORDER_MAP_G[node, gs]
    ei = _ORDER_MAP_E[node, es]
    return (ee[ei] - eg[gi]) / H_PLANCK - _LINE_CENTER


@njit
def strength(pol, gs, es, B):
    """Normalized |CG|^2 transition strength (cycling GS5->ES11 sigma+ == 0.25).

    Uses the reordered eigenvectors of H(|B|) and the frozen coupling tensor;
    the global scale anchors the cycling transition to 0.25.
    """
    vg = np.linalg.eigh(_H_HFS_G + B * _H_ZEE_G)[1]
    ve = np.linalg.eigh(_H_HFS_E + B * _H_ZEE_E)[1]
    node = _nearest_node(B)
    gi = _ORDER_MAP_G[node, gs]
    ei = _ORDER_MAP_E[node, es]
    s = _summe(pol, vg[:, gi], ve[:, ei])
    return s * s * _SCALE
