"""Python-layer setup for the Li-6 D2 diagonalizer model (phase 04).

Faithful port of the out-of-repo udem_code ``SetMatrix``/``Linien`` core,
generalized in ``(I, J, g_J, A_hfs, B_hfs)`` per manifold (D-04). The sympy
Wigner-symbol math lives ONLY in this module and is B-independent:

    H(B) = H_hfs + B * H_Zeeman

so the symbolic construction runs once in the Python layer and never
crosses the Numba JIT boundary (D-01 key insight).

This module
  * builds the constant matrices ``H_hfs`` / ``H_Zeeman`` and the B-independent
    dipole-coupling tensor for both manifolds,
  * solves the hyperfine+Zeeman Hamiltonian at any |B| (``np.linalg.eigh``),
  * reproduces the udem line positions and transition strengths (``Linien``),
  * computes an overlap-based diabatic GS0-5 / ES0-11 order map over 0-1 T
    (D-06/D-07), replacing udem's fragile linear-extrapolation line tracker,
  * emits the dev-local index <-> |F, mF> label artifact (D-07).

sympy is offline/dev-only and MUST NOT be imported by any @njit module.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import scipy.constants as scc
from scipy.optimize import linear_sum_assignment
from sympy.physics.wigner import wigner_3j, wigner_6j

H_PLANCK = scc.h  # Planck constant [J*s]; matches the reference's h_bar*2*pi
BOHR_MAGNETON = scc.physical_constants["Bohr magneton"][0]

# Li-6 D2 reference constants (udem_code __main__). Freqs in Hz, B in Tesla.
LI6_D2 = {
    "I": 1.0,
    "ground": {"J": 0.5, "g_J": 2.002, "A": 150e6, "B_hfs": 0.0},
    "excited": {"J": 1.5, "g_J": 1.335, "A": -1.15e6, "B_hfs": -0.1e6},
}

# Canonical GS0-5 / ES0-11 labels (SimpleEighteenLevelInteraction ordering,
# src/interactions.py:654-661). F values follow from the |F, mF> assignment:
# ground gf -0.6667 -> F=1/2, +0.6667 -> F=3/2; excited by |F,mF| consistency.
GROUND_MF = np.array([0.5, 0.5, -0.5, -0.5, -1.5, 1.5])
GROUND_F = np.array([0.5, 1.5, 0.5, 1.5, 1.5, 1.5])
EXCITED_MF = np.array(
    [1.5, 1.5, -0.5, -0.5, -0.5, -1.5, -1.5, 0.5, 0.5, 0.5, -2.5, 2.5]
)
EXCITED_F = np.array(
    [2.5, 1.5, 2.5, 0.5, 1.5, 2.5, 1.5, 0.5, 2.5, 1.5, 2.5, 2.5]
)

# SimpleEighteenLevelInteraction Lande gF values (low-B check reference).
GROUND_GF = np.array([-0.6667, 0.6667, -0.6667, 0.6667, 0.6667, 0.6667])
EXCITED_GF = np.array(
    [0.8004, 0.9783, 0.8004, 1.1111, 0.9783, 0.8004,
     0.9783, 1.1111, 0.8004, 0.9783, 0.8004, 0.8004]
)

_LABEL_ARTIFACT = "interaction_tables/li6_d2_state_labels.csv"


# --------------------------------------------------------------------------- #
# Low-level helpers (port of the udem Delta / Minus / MI / MJ / Index)
# --------------------------------------------------------------------------- #
def _minus(x: float) -> float:
    """(-1)^x for integer x (udem ``Minus``)."""
    return -1.0 if int(round(x)) % 2 else 1.0


def _basis(I: float, J: float):
    """(mi, mj) per basis index, mj outer / mi inner (udem ``Index``)."""
    mi_list, mj_list = [], []
    for mj in np.arange(-J, J + 1):
        for mi in np.arange(-I, I + 1):
            mi_list.append(mi)
            mj_list.append(mj)
    return np.asarray(mi_list, dtype=float), np.asarray(mj_list, dtype=float)


def _ground_index(mi: float, mj: float, I: float, J: float):
    """0-based basis index for (mi, mj) or None if out of the manifold."""
    if mj < -J - 1e-9 or mj > J + 1e-9:
        return None
    idx = int(round((mj + J) * (2 * I + 1) + mi + I))
    dim = int(round((2 * J + 1) * (2 * I + 1)))
    return idx if 0 <= idx < dim else None


# --------------------------------------------------------------------------- #
# SetMatrix port + B factorization (D-04, D-01)
# --------------------------------------------------------------------------- #
def set_matrix(I, J, A, B_hfs, B, g_J):
    """Interaction Hamiltonian matrix (general port of udem ``SetMatrix``).

    Returns a real-symmetric ``NMAX x NMAX`` matrix in the |mI, mJ> basis, with
    ``NMAX = (2J+1)(2I+1)``. Energies in Joules. A, B_hfs given in Hz.
    """
    dim = int(round((2 * J + 1) * (2 * I + 1)))
    H = np.zeros((dim, dim))
    A_e = A * H_PLANCK  # magnetic-dipole constant, freq -> energy
    B_e = B_hfs * H_PLANCK  # electric-quadrupole constant
    has_dipole = I > 0 and J > 0
    has_quad = I > 0.5 and J > 0.5
    sj = (
        float(wigner_6j(1, J, J, J, 1, 2) * wigner_6j(1, I, I, I, 1, 2))
        if has_quad
        else 0.0
    )
    dip_pref = np.sqrt(J * (J + 1) * (2 * J + 1) * I * (I + 1) * (2 * I + 1))
    quad_pref = (
        15.0
        / 2.0
        * ((2 * J + 1) * (J + 1) * (2 * I + 1) * (I + 1))
        / ((2 * J - 1) * (2 * I - 1))
        if has_quad
        else 0.0
    )
    mi_arr, mj_arr = _basis(I, J)
    for m in range(dim):
        mi1, mj1 = mi_arr[m], mj_arr[m]
        for n in range(dim):
            mi2, mj2 = mi_arr[n], mj_arr[n]
            val = 0.0
            if abs(mi1 - mi2) < 1e-9 and abs(mj1 - mj2) < 1e-9:
                val += B * BOHR_MAGNETON * g_J * mj1  # Zeeman (linear in B)
            if has_dipole:
                val += (
                    A_e
                    * _minus(mj2 + mi1 + J + I)
                    * dip_pref
                    * float(wigner_3j(J, 1, J, mj2, mj1 - mj2, -mj1))
                    * float(wigner_3j(I, 1, I, mi2, mj2 - mj1, -mi1))
                )
            if has_quad:
                val += (
                    B_e
                    * _minus(mj2 + mi1 - J - I)
                    * quad_pref
                    * float(wigner_3j(J, 2, J, mj2, mj1 - mj2, -mj1))
                    * float(wigner_3j(I, 2, I, mi2, mj2 - mj1, -mi1))
                    * sj
                )
            H[m, n] = val
    return H


def build_constant_matrices(I, J, g_J, A, B_hfs):
    """Return (H_hfs, H_Zeeman) so that H(B) = H_hfs + B * H_Zeeman.

    The Zeeman term is exactly linear in B, so H_Zeeman is recovered as the
    per-Tesla coefficient matrix ``SetMatrix(..., B=1) - SetMatrix(..., B=0)``.
    """
    H_hfs = set_matrix(I, J, A, B_hfs, 0.0, g_J)
    H_zeeman = set_matrix(I, J, A, B_hfs, 1.0, g_J) - H_hfs
    return H_hfs, H_zeeman


def build_coupling_tensor(I, J_ground, J_excited):
    """B-independent dipole-coupling tensor (port of udem ``Linien`` factors).

    ``coupling[q, l_excited, l_ground]`` = (-1)^(Ja-mj) * wigner_3j(Ja,1,Jg;
    -mj, q-1, mj-q+1) for q in {0: sigma-, 1: pi, 2: sigma+}, using the shared
    |mI, mJ> basis map between manifolds; 0 where no valid ground index exists.
    """
    da = int(round((2 * J_excited + 1) * (2 * I + 1)))
    dg = int(round((2 * J_ground + 1) * (2 * I + 1)))
    coupling = np.zeros((3, da, dg))
    mi_e, mj_e = _basis(I, J_excited)
    for q in range(3):
        for le in range(da):
            mi, mj = mi_e[le], mj_e[le]
            lg = _ground_index(mi, mj - (q - 1), I, J_ground)
            if lg is None:
                continue
            coupling[q, le, lg] = _minus(J_excited - mj) * float(
                wigner_3j(J_excited, 1, J_ground, -mj, q - 1, mj - (q - 1))
            )
    return coupling


# --------------------------------------------------------------------------- #
# Diagonalization, positions, strengths (port of udem ``Linien`` body)
# --------------------------------------------------------------------------- #
def solve_at_field(H_hfs, H_zeeman, B):
    """Eigenvalues (ascending) and orthonormal eigenvectors of H(B)."""
    H = H_hfs + B * H_zeeman
    return np.linalg.eigh(H)  # real symmetric -> real, sorted


def line_positions(e_ground, e_excited):
    """pos[gs, es] = (E_excited[es] - E_ground[gs]) / h  in Hz."""
    return (e_excited[None, :] - e_ground[:, None]) / H_PLANCK


def transition_strengths(coupling, v_ground, v_excited, hauf=1.0):
    """Raw udem intensities intensity[q, gs, es] = Summe^2 * hauf.

    Summe = v_excited[:, es] . coupling[q] . v_ground[:, gs].
    """
    n_g = v_ground.shape[1]
    n_e = v_excited.shape[1]
    intensity = np.empty((3, n_g, n_e))
    for q in range(3):
        m = v_excited.T @ coupling[q] @ v_ground  # (es, gs)
        intensity[q] = (m.T ** 2) * hauf  # (gs, es)
    return intensity


# --------------------------------------------------------------------------- #
# Diabatic ordering (D-07) — replaces udem's fragile line tracker
# --------------------------------------------------------------------------- #
def build_fsquared(I, J):
    """F^2 operator in the |mI, mJ> basis: I(I+1)+J(J+1)+2 I.J."""
    mi_arr, mj_arr = _basis(I, J)
    dim = len(mi_arr)
    f2 = np.zeros((dim, dim))
    for a in range(dim):
        mi, mj = mi_arr[a], mj_arr[a]
        f2[a, a] += I * (I + 1) + J * (J + 1) + 2 * mi * mj
        for b in range(dim):
            mi2, mj2 = mi_arr[b], mj_arr[b]
            if abs(mi2 - (mi + 1)) < 1e-9 and abs(mj2 - (mj - 1)) < 1e-9:
                f2[b, a] += np.sqrt(I * (I + 1) - mi * (mi + 1)) * np.sqrt(
                    J * (J + 1) - mj * (mj - 1)
                )
            if abs(mi2 - (mi - 1)) < 1e-9 and abs(mj2 - (mj + 1)) < 1e-9:
                f2[b, a] += np.sqrt(I * (I + 1) - mi * (mi - 1)) * np.sqrt(
                    J * (J + 1) - mj * (mj + 1)
                )
    return f2


def _state_mf(evecs, I, J):
    """<mI + mJ> per eigenvector (a good quantum number for all B)."""
    mi_arr, mj_arr = _basis(I, J)
    return (evecs ** 2).T @ (mi_arr + mj_arr)


def _state_f(evecs, fsq):
    """F per eigenvector from <F^2> = F(F+1)."""
    f2 = np.einsum("ik,ij,jk->k", evecs, fsq, evecs)
    return (-1.0 + np.sqrt(1.0 + 4.0 * np.maximum(f2, 0.0))) / 2.0


def _label_states(evecs, I, J, fsq, canonical_mf, canonical_f):
    """Map canonical label -> eigenvector index by (mF, F) character (low B).

    mF is weighted heavily so states match primarily by mF; F breaks ties
    within a shared-mF group. Valid where the Zeeman splitting lifts the mF
    degeneracy (any small B > 0).
    """
    mf = _state_mf(evecs, I, J)
    fval = _state_f(evecs, fsq)
    cost = 100.0 * np.abs(
        mf[None, :] - canonical_mf[:, None]
    ) + np.abs(fval[None, :] - canonical_f[:, None])
    row, col = linear_sum_assignment(cost)
    perm = np.empty(len(canonical_mf), dtype=np.int32)
    perm[row] = col
    return perm


def _track(v_prev, v_now, order_prev):
    """Continue diabatic labels by maximizing |<v_prev | v_now>| overlap."""
    labeled_prev = v_prev[:, order_prev]
    overlap = np.abs(labeled_prev.T @ v_now)
    row, col = linear_sum_assignment(-overlap)
    order_now = np.empty(len(order_prev), dtype=np.int32)
    order_now[row] = col
    return order_now


def build_order_map(
    H_hfs, H_zeeman, I, J, canonical_mf, canonical_f,
    b_max=1.0, b_min=1e-6, b_knee=2e-4, n_low=80, n_high=200,
):
    """Per-node diabatic permutation over b_min..b_max T (D-06/D-07).

    The grid is log-spaced from ``b_min`` to ``b_knee`` (dense through the
    <~1 G / 2e-4 T regime, where the excited-state HFS ~1 MHz and the Zeeman
    energy cross over and the eigenstate mixing is most volatile) then linear
    from ``b_knee`` to ``b_max``. Labels are seeded by (mF, F) character at
    ``b_min`` -- deep in the F-dominated regime for BOTH manifolds, where the
    assignment is unambiguous -- then continued outward by eigenvector overlap.
    The fine low-field steps keep both the overlap tracking and the runtime
    nearest-node reorder correct down to ~b_min; the previous uniform 5 mT grid
    seeded in the excited Paschen-Back regime and mis-assigned excited states
    below ~0.3 mT (even the cycling line read 0).

    Returns (b_axis, order) with ``order[node, label] = eigenvector index``.
    """
    fsq = build_fsquared(I, J)
    b_low = np.logspace(
        np.log10(b_min), np.log10(b_knee), n_low, endpoint=False
    )
    b_high = np.linspace(b_knee, b_max, n_high)
    b_axis = np.concatenate([b_low, b_high])
    n_nodes = b_axis.shape[0]
    n_labels = len(canonical_mf)
    evecs_all = [solve_at_field(H_hfs, H_zeeman, B)[1] for B in b_axis]
    order = np.zeros((n_nodes, n_labels), dtype=np.int32)
    order[0] = _label_states(
        evecs_all[0], I, J, fsq, canonical_mf, canonical_f
    )
    for k in range(1, n_nodes):
        order[k] = _track(evecs_all[k - 1], evecs_all[k], order[k - 1])
    return b_axis, order


# --------------------------------------------------------------------------- #
# Li-6 D2 convenience layer (matrices, order maps, normalized strengths)
# --------------------------------------------------------------------------- #
_LI6_CACHE = None


def li6_constant_matrices():
    """(Hg_hfs, Hg_zee, He_hfs, He_zee, coupling) for the Li-6 D2 line."""
    I = LI6_D2["I"]
    g, e = LI6_D2["ground"], LI6_D2["excited"]
    hg, zg = build_constant_matrices(I, g["J"], g["g_J"], g["A"], g["B_hfs"])
    he, ze = build_constant_matrices(I, e["J"], e["g_J"], e["A"], e["B_hfs"])
    coupling = build_coupling_tensor(I, g["J"], e["J"])
    return hg, zg, he, ze, coupling


def _labeled_vectors(H_hfs, H_zee, B, I, J, fsq, canonical_mf, canonical_f):
    _, v = solve_at_field(H_hfs, H_zee, B)
    perm = _label_states(v, I, J, fsq, canonical_mf, canonical_f)
    return v[:, perm]


def get_li6_setup():
    """Build (and cache) the full Li-6 D2 setup: matrices, order maps, scale.

    The strength scale anchors the cycling transition GS5 -> ES11 (sigma+) to
    |CG|^2 = 0.25 (Pitfall 3) and is applied globally.
    """
    global _LI6_CACHE
    if _LI6_CACHE is not None:
        return _LI6_CACHE
    I = LI6_D2["I"]
    jg, je = LI6_D2["ground"]["J"], LI6_D2["excited"]["J"]
    hg, zg, he, ze, coupling = li6_constant_matrices()
    fsq_g, fsq_e = build_fsquared(I, jg), build_fsquared(I, je)
    b_axis_g, order_g = build_order_map(hg, zg, I, jg, GROUND_MF, GROUND_F)
    b_axis_e, order_e = build_order_map(he, ze, I, je, EXCITED_MF, EXCITED_F)
    raw_cycling = _raw_li6_strength(
        5, 11, 2, 1e-3, hg, zg, he, ze, coupling, I, jg, je, fsq_g, fsq_e
    )
    _LI6_CACHE = {
        "I": I,
        "Jg": jg,
        "Je": je,
        "Hg_hfs": hg,
        "Hg_zee": zg,
        "He_hfs": he,
        "He_zee": ze,
        "coupling": coupling,
        "fsq_g": fsq_g,
        "fsq_e": fsq_e,
        "b_axis_g": b_axis_g,
        "order_g": order_g,
        "b_axis_e": b_axis_e,
        "order_e": order_e,
        "scale": 0.25 / raw_cycling,
    }
    return _LI6_CACHE


def _raw_li6_strength(gs, es, pol, B, hg, zg, he, ze, coupling, I, jg, je,
                      fsq_g, fsq_e):
    v_g = _labeled_vectors(hg, zg, B, I, jg, fsq_g, GROUND_MF, GROUND_F)
    v_e = _labeled_vectors(he, ze, B, I, je, fsq_e, EXCITED_MF, EXCITED_F)
    summe = v_e[:, es] @ coupling[pol] @ v_g[:, gs]
    return summe ** 2


def li6_d2_strength(gs, es, pol, B):
    """Diabatically-labeled Li-6 D2 transition strength (|CG|^2 normalized)."""
    s = get_li6_setup()
    raw = _raw_li6_strength(
        gs, es, pol, B, s["Hg_hfs"], s["Hg_zee"], s["He_hfs"], s["He_zee"],
        s["coupling"], s["I"], s["Jg"], s["Je"], s["fsq_g"], s["fsq_e"]
    )
    return raw * s["scale"]


# Field at which the linear-Zeeman slope is read: deep in the g_F*mF regime
# (Zeeman ~0.14 MHz << hyperfine ~225 MHz), before F=1/2 / F=3/2 mixing.
_SLOPE_FIELD = 1e-5


def li6_d2_zeeman_slope(manifold, label):
    """dE/dB / (mu_B) for a labeled state = g_F * mF at low B (units: mu_B)."""
    s = get_li6_setup()
    if manifold == "ground":
        v = _labeled_vectors(
            s["Hg_hfs"], s["Hg_zee"], _SLOPE_FIELD, s["I"], s["Jg"],
            s["fsq_g"], GROUND_MF, GROUND_F
        )
        return float(v[:, label] @ s["Hg_zee"] @ v[:, label]) / BOHR_MAGNETON
    v = _labeled_vectors(
        s["He_hfs"], s["He_zee"], _SLOPE_FIELD, s["I"], s["Je"],
        s["fsq_e"], EXCITED_MF, EXCITED_F
    )
    return float(v[:, label] @ s["He_zee"] @ v[:, label]) / BOHR_MAGNETON


# --------------------------------------------------------------------------- #
# Precomputed |B|-table generator (D-12) — offline, no runtime diagonalization
# --------------------------------------------------------------------------- #
def li6_d2_constants():
    """Fixed Li-6 D2 setup (matrices, order maps, cycling-strength scale).

    Thin alias for ``get_li6_setup`` used by ``generate_table`` and callers that
    want the frozen constants without importing the cache dict directly.
    """
    return get_li6_setup()


def generate_table(constants, b_min=0.0, b_max=1.0, n_nodes=200,
                   out_path="interaction_tables/li6_d2_table.npz"):
    """Sweep |B| and write a precomputed diagonalizer table as an NPZ (D-12).

    For each of ``n_nodes`` uniform nodes over ``[b_min, b_max]`` (default
    0-1 T, D-06) the live diagonalizer is evaluated: line positions minus the
    B=0 center (Hz) go into ``pos_table[nb, 6, 12]`` and cycling-normalized
    (=0.25) strengths into ``strength_table[nb, 6, 12, 3]``. The live @njit
    helpers (``diw.shift`` / ``diw.strength``) are reused so the table
    reproduces the live model at its nodes to machine precision; ``diw`` is
    imported lazily to avoid the ``diagonalizer_setup <- diagonalizer_wrappers``
    import cycle. Written with ``np.savez`` into the gitignored
    ``interaction_tables/`` dir.
    """
    # Lazy import: diagonalizer_wrappers imports this module at its top level.
    import src.interaction_wrappers.diagonalizer_wrappers as diw

    ng = int(constants["Hg_hfs"].shape[0])   # 6 ground states
    ne = int(constants["He_hfs"].shape[0])    # 12 excited states
    b_axis = np.linspace(b_min, b_max, n_nodes)
    pos_table = np.empty((n_nodes, ng, ne), dtype=np.float64)
    strength_table = np.empty((n_nodes, ng, ne, 3), dtype=np.float64)
    for k, B in enumerate(b_axis):
        for gs in range(ng):
            for es in range(ne):
                pos_table[k, gs, es] = diw.shift(gs, es, B)
                for pol in range(3):
                    strength_table[k, gs, es, pol] = diw.strength(pol, gs, es, B)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        b_axis=b_axis,
        pos_table=pos_table,
        strength_table=strength_table,
    )
    return str(out)


# --------------------------------------------------------------------------- #
# D-07 label artifact (dev-local, regenerable, gitignored by *.csv)
# --------------------------------------------------------------------------- #
def _fmt(x):
    return f"{float(x):.1f}"


def _manifold_rows(tag, H_hfs, H_zee, I, J, order, b_axis, mf, fvals):
    mi_arr, mj_arr = _basis(I, J)
    _, v_hi = solve_at_field(H_hfs, H_zee, b_axis[-1])
    labeled_hi = v_hi[:, order[-1]]  # dominant |mI, mJ> at Paschen-Back
    rows = []
    for lbl in range(len(mf)):
        dom = int(np.argmax(labeled_hi[:, lbl] ** 2))
        rows.append(
            [tag, lbl, _fmt(fvals[lbl]), _fmt(mf[lbl]),
             _fmt(mi_arr[dom]), _fmt(mj_arr[dom])]
        )
    return rows


def emit_label_artifact(path=_LABEL_ARTIFACT):
    """Write the dev-local index <-> |F, mF> / dominant |mI, mJ> table (D-07).

    Regenerable from the fixed Li-6 D2 constants; gitignored by the repo-wide
    ``*.csv`` rule. No downstream code depends on the file existing on disk —
    the order map is returned in memory by ``build_order_map``.
    """
    s = get_li6_setup()
    rows = _manifold_rows(
        "GS", s["Hg_hfs"], s["Hg_zee"], s["I"], s["Jg"], s["order_g"],
        s["b_axis_g"], GROUND_MF, GROUND_F
    )
    rows += _manifold_rows(
        "ES", s["He_hfs"], s["He_zee"], s["I"], s["Je"], s["order_e"],
        s["b_axis_e"], EXCITED_MF, EXCITED_F
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["manifold", "index", "F", "mF", "mI", "mJ"])
        writer.writerows(rows)
    return str(out)


if __name__ == "__main__":
    print("Li-6 D2 diagonalizer setup — emitting label artifact")
    print("wrote", emit_label_artifact())
