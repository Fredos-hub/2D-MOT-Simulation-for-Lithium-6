#########################################################################################
#
#
#                               Wrapper Functions
#
#
##########################################################################################

from numba import njit
import numpy as np
import scipy.constants as scc

@njit
def _calculate_transition_frequency_shift(magnetic_field_strength: float,
                                          mu_B: float,
                                          ground_mf: float,
                                          excited_mf: float,
                                          ground_gf: float,
                                          excited_gf: float):

    E_excited = excited_gf * mu_B * excited_mf * magnetic_field_strength
    E_ground = ground_gf * mu_B * ground_mf * magnetic_field_strength

    # Account for HFS shifts by adding or substracting the zero field shift.
    if ground_gf > 0:    # Relates to F = 1/2 ground state
        return (E_excited - E_ground)/scc.h - 76.7e6
    elif ground_gf < 0:  # Relates to F = 3/2 ground state
        return (E_excited - E_ground)/scc.h + 151.3e6



# _calculate_excitation_rate and _calculate_saturation_parameter live in
# src/interaction_wrappers/common.py — they are identical across all models.