"""@njit helpers for the simplified 18-level Li-6 interaction model."""

import scipy.constants as scc
from numba import njit


@njit
def _calculate_transition_frequency_shift(
    magnetic_field_strength: float,
    mu_B: float,
    ground_mf: float,
    excited_mf: float,
    ground_gf: float,
    excited_gf: float,
    excited_b0: float,
):

    E_excited = excited_gf * mu_B * excited_mf * magnetic_field_strength
    E_ground = ground_gf * mu_B * ground_mf * magnetic_field_strength
    zeeman = (E_excited - E_ground) / scc.h

    # Ground-manifold B=0 offset (gF sign marks the manifold: gF>0 -> F=3/2,
    # gF<0 -> F=1/2). F=3/2 is the shared cycling anchor -76.75 MHz (same base
    # line as full-18/diagonalizer so JSON detunings stay consistent); F=1/2
    # sits one literature ground HFS splitting above it (228.205 MHz, Li-6
    # 2S1/2 A=152.1368 MHz). excited_b0 adds the resolved excited-HFS shift so
    # each transition matches the diagonalizer at B=0.
    if ground_gf > 0:  # F = 3/2 ground manifold
        return zeeman - 76.75e6 + excited_b0
    elif ground_gf < 0:  # F = 1/2 ground manifold
        return zeeman + 151.4552e6 + excited_b0


# _calculate_excitation_rate and _calculate_saturation_parameter live in
# src/interaction_wrappers/common.py — they are identical across all models.
