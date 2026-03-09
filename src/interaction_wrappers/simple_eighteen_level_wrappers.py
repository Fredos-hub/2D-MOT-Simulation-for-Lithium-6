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
def _calculate_transition_frequency_shift(magnetic_field_strength: float,
                                          mu_B: float, 
                                          ground_mF: float, 
                                          excited_mF: float,
                                          g_ground: float,
                                          g_excited: float):


    # Landé g-factors:
    g_ground = g_ground
    g_excited = g_excited



    E_excited = g_excited * mu_B * excited_mF * magnetic_field_strength
    E_ground = g_ground * mu_B * ground_mF * magnetic_field_strength

    # Account for HFS shifts by adding or substracting the zero field shift.
    if g_ground >0 : # Relates to F = 1/2 ground state
        return (E_excited - E_ground)/scc.h -76.7e6
    elif g_ground < 0: # Relates to F = 3/2 ground state
        return (E_excited - E_ground)/scc.h + 151.3e6



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
def _calculate_saturation_parameter(effective_transition_frequency: float, # in Hz
                                    doppler_shift: float, # in rad/s
                                    laser_beam_frequency: float, # in Hz
                                    detuning: float, # in rad/s
                                    transition_strength: float, 
                                    laser_intensity: float, # in W
                                    natural_linewidth: float):# in rad/s



        laser_beam_frequency_rad = laser_beam_frequency*2*scc.pi

        effective_transition_frequency_rad = effective_transition_frequency*2*scc.pi
        
        
        effective_detuning =  laser_beam_frequency_rad - doppler_shift + detuning - effective_transition_frequency_rad 

        # Calculate Rabi frequencies (with a scaling factor from literature).

        rabi_frequency = 2*scc.pi * 1e6 * 11.925*4.37* transition_strength * math.sqrt(0.001*laser_intensity) # 11.925 is the reduced D2-line matrix element for Li6 (Gehm 2003)

        #  Compute saturation parameters using squared effective detunings.
        saturation_parameter = 0.5 * rabi_frequency**2 / (effective_detuning**2 + 0.25 * natural_linewidth**2)
        return saturation_parameter