"""Shared @njit helpers used by every interaction model.

The saturation-parameter and excitation-rate formulas were duplicated verbatim
in the 6/18/simple-18/4-level wrappers; this module is the single source of
truth.
"""

import math

from numba import njit


@njit
def _calculate_excitation_rate(
    saturation_parameters,
    total_saturation_parameter,
    natural_linewidth,
    excitation_rates,
):
    n_lasers = excitation_rates.shape[0]
    n_excited = excitation_rates.shape[1]
    for j in range(n_lasers):
        for ex in range(n_excited):
            for pol in range(3):
                sat = saturation_parameters[j, ex, pol]
                excitation_rates[j, ex, pol] = (
                    0.5 * sat * natural_linewidth
                ) / (1.0 + total_saturation_parameter)


@njit
def _calculate_saturation_parameter(
    effective_transition_frequency: float,  # in Hz
    doppler_shift: float,  # in rad/s
    laser_beam_frequency: float,  # in Hz
    detuning: float,  # in rad/s
    transition_strength: float,  # |CG|^2 (dimensionless)
    laser_intensity: float,  # in W/m^2 (after Gaussian profile)
    natural_linewidth: float,
):  # in rad/s

    laser_beam_frequency_rad = laser_beam_frequency * 2 * math.pi
    effective_transition_frequency_rad = (
        effective_transition_frequency * 2 * math.pi
    )

    effective_detuning = (
        laser_beam_frequency_rad
        - doppler_shift
        + detuning
        - effective_transition_frequency_rad
    )

    # Ω ∝ |CG| · √I and the lookup tables hold transition_strength = |CG|²,
    # so it sits inside the sqrt alongside the intensity.
    # 11.925 is the reduced D2-line matrix element for Li-6 (Gehm 2003).
    rabi_frequency = (
        2
        * math.pi
        * 1e6
        * 11.925
        * 4.37
        * math.sqrt(transition_strength * 0.001 * laser_intensity)
    )

    saturation_parameter = (
        0.5
        * rabi_frequency**2
        / (effective_detuning**2 + 0.25 * natural_linewidth**2)
    )
    return saturation_parameter
