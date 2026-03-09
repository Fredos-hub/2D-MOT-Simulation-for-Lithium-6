#####################################################################################
#Contains classes to create objects for the several parts of the experimental setup.#
#####################################################################################

import numpy as np
import scipy.constants as scc
from util.simulation_typing import ECSAtoms
import src.distributions as distributions
from numba.experimental import jitclass
from numba import int32, float64, njit, prange
import util.geometry as geometry


# Define the ECS-like LaserComponent class
laser_component_spec = [
    ("n_lasers", int32),
    ('beam_waists', float64[:]),
    ('origins', float64[:, :]),  # Array of shape (n, 3)
    ('normalized_directions', float64[:, :]),  # Array of shape (n, 3)
    ('beam_powers', float64[:]),
    ('beam_frequencies', float64[:]),
    ('detunings', float64[:]),
    ('handedness', int32[:]), # +1 for right handed and -1 for left handed
    ('wave_vectors', float64[:, :]),  # Array of shape (n, 3)
    ('beam_wavelengths', float64[:]),
    ('initial_intensities', float64[:]),
    ('refractive_indices', float64[:])
]

@jitclass(laser_component_spec)
class LaserComponent:
    def __init__(self, n_lasers):
        # Initialize arrays to store laser properties for n lasers
        self.n_lasers = n_lasers
        self.beam_waists = np.zeros(n_lasers, dtype=np.float64)
        self.origins = np.zeros((n_lasers, 3), dtype=np.float64)
        self.normalized_directions = np.zeros((n_lasers, 3), dtype=np.float64)
        self.beam_powers = np.zeros(n_lasers, dtype=np.float64)
        self.beam_frequencies = np.zeros(n_lasers, dtype=np.float64)
        self.detunings = np.zeros(n_lasers, dtype=np.float64)
        self.handedness = np.zeros(n_lasers, dtype=np.int32)
        self.wave_vectors = np.zeros((n_lasers, 3), dtype=np.float64)
        self.beam_wavelengths = np.zeros(n_lasers, dtype=np.float64)
        self.initial_intensities = np.zeros(n_lasers, dtype=np.float64)
        self.refractive_indices = np.ones(n_lasers, dtype = np.float64)

        
    def add_laser(self, index, waist, origin, direction, beam_power, beam_frequency, detuning,handedness):
        """
        Add or update a laser at a specific index in the component.
        """
        self.beam_waists[index] = waist
        self.origins[index, :] = origin
        self.normalized_directions[index, :] = direction / np.linalg.norm(direction)
        self.beam_powers[index] = beam_power
        self.beam_frequencies[index] = beam_frequency
        self.detunings[index] = detuning
        self.handedness[index] = handedness

        # Calculate derived properties
        self.beam_wavelengths[index] = scc.c / beam_frequency
        k = 2 * scc.pi / self.beam_wavelengths[index]
        self.wave_vectors[index, :] = k * self.normalized_directions[index, :]
        self.initial_intensities[index] = 2 * beam_power / (np.pi * waist**2)



if __name__ == '__main__':
    pass