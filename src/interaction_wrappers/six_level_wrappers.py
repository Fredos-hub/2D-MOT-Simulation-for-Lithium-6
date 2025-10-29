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
import math

@njit
def _calculate_transition_frequency_shift(
                                          ground_state: int, 
                                          excited_state: int, 
                                          magnetic_field_strength: float,
 
                                          mu_B: float, 
                                          ground_mJ: float, 
                                          excited_mJ: float):


    # Landé g-factors: S1/2 (g = 2) and P3/2 (g = 4/3)
    g_s = 2.0
    g_p = 4.0 / 3.0

    E_excited = g_p * mu_B * excited_mJ[excited_state] * magnetic_field_strength
    E_ground = g_s * mu_B * ground_mJ[ground_state] * magnetic_field_strength
    #print("frequency_shift: allowed transition")

    return (E_excited - E_ground)/scc.h -77e6




@njit
def _is_transition_allowed(polarization: int, 
                           ground_state: int, 
                           excited_state: int, 
                           allowed_transitions: np.ndarray):
    
    for i in range(allowed_transitions.shape[0]):
        if (allowed_transitions[i, 0] == ground_state and
            allowed_transitions[i, 1] == excited_state and
            allowed_transitions[i, 2] == polarization):
            return True
    return False


@njit
def _calculate_excitation_rate(saturation_parameters,
                   total_saturation_parameter,
                   natural_linewidth,
                   excitation_rates):
    n_lasers = excitation_rates.shape[0]
    n_excited = excitation_rates.shape[1]
    for j in range(n_lasers):
        for ex in range(n_excited):
            for pol in range(3):
                sat = saturation_parameters[j, ex, pol]
                excitation_rates[j, ex, pol] = (0.5* sat * natural_linewidth) / (1.0 + total_saturation_parameter)


@njit
def _calculate_saturation_parameter(effective_transition_frequency: float, 
                                    doppler_shift: float, 
                                    laser_beam_frequency: float, 
                                    detuning: float, 
                                    transition_strength: float, 
                                    laser_intensity: float, 
                                    natural_linewidth: float):



        effective_detuning = (effective_transition_frequency - (laser_beam_frequency - doppler_shift/(2*scc.pi) + detuning/(2*scc.pi)))*2*scc.pi

        # Calculate Rabi frequencies (with a scaling factor from literature).

        rabi_frequency = 2*scc.pi * 1e6 * 11.925*4.37* math.sqrt(transition_strength * 0.001*laser_intensity)

        #  Compute saturation parameters using squared effective detunings.
        saturation_parameter = 0.5 * rabi_frequency**2 / (effective_detuning**2 + 0.25 * natural_linewidth**2)
        return saturation_parameter