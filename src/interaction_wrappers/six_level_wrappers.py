"""@njit helpers for the 6-level Li-6 interaction model."""

import numpy as np
import scipy.constants as scc
from numba import njit


@njit
def _calculate_transition_frequency_shift(
    ground_state: int,
    excited_state: int,
    magnetic_field_strength: float,
    mu_B: float,
    ground_mJ: float,
    excited_mJ: float,
):

    # Landé g-factors: S1/2 (g = 2) and P3/2 (g = 4/3)
    g_s = 2.0
    g_p = 4.0 / 3.0

    E_excited = (
        g_p * mu_B * excited_mJ[excited_state] * magnetic_field_strength
    )
    E_ground = g_s * mu_B * ground_mJ[ground_state] * magnetic_field_strength

    # The -76.6 MHz offset compensates the hyperfine COG: in the 18-level model
    # the hyperfine structure is explicit, here it is not. The standard trap /
    # repump frequencies refer to the F=1/2 and F=3/2 substates, so we shift
    # the J-only transition frequency by the COG so that the same JSON detuning
    # values stay consistent across the 6- and 18-level interactions at B = 0.
    return (E_excited - E_ground) / scc.h - 76.6e6


@njit
def _is_transition_allowed(
    polarization: int,
    ground_state: int,
    excited_state: int,
    allowed_transitions: np.ndarray,
):

    for i in range(allowed_transitions.shape[0]):
        if (
            allowed_transitions[i, 0] == ground_state
            and allowed_transitions[i, 1] == excited_state
            and allowed_transitions[i, 2] == polarization
        ):
            return True
    return False


# _calculate_excitation_rate and _calculate_saturation_parameter live in
# src/interaction_wrappers/common.py — they are identical across all models.
