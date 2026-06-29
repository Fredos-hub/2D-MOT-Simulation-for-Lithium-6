"""Jitclass container holding the per-beam laser properties (LaserComponent)."""

import math

import numpy as np
import scipy.constants as scc
from numba import float64, int32
from numba.experimental import jitclass

# Define the ECS-like LaserComponent class
laser_component_spec = [
    ("n_lasers", int32),
    ("beam_waists", float64[:]),
    ("origins", float64[:, :]),  # Array of shape (n, 3)
    ("normalized_directions", float64[:, :]),  # Array of shape (n, 3)
    ("beam_powers", float64[:]),
    ("beam_frequencies", float64[:]),
    ("detunings", float64[:]),
    ("handedness", int32[:]),  # +1 for right handed and -1 for left handed
    ("wave_vectors", float64[:, :]),  # Array of shape (n, 3)
    ("beam_wavelengths", float64[:]),
    ("initial_intensities", float64[:]),
    ("refractive_indices", float64[:]),
]


@jitclass(laser_component_spec)
class LaserComponent:
    """ECS-style container holding the properties of all laser beams.

    Each attribute is an array indexed by laser, so the full set of beams
    crosses the Numba boundary as a single jitclass instance.

    Attributes
    ----------
    n_lasers : int
        Number of laser beams.
    beam_waists : ndarray (n,), float64
        1/e^2 beam waists in m.
    origins : ndarray (n, 3), float64
        Beam origin positions in m.
    normalized_directions : ndarray (n, 3), float64
        Unit propagation directions.
    beam_powers : ndarray (n,), float64
        Beam powers in W.
    beam_frequencies : ndarray (n,), float64
        Beam frequencies in Hz.
    detunings : ndarray (n,), float64
        Detunings in rad/s.
    handedness : ndarray (n,), int32
        +1 right-handed, -1 left-handed, 0 linear.
    wave_vectors : ndarray (n, 3), float64
        Wave vectors k in rad/m.
    beam_wavelengths : ndarray (n,), float64
        Beam wavelengths in m.
    initial_intensities : ndarray (n,), float64
        Peak (on-axis) intensities in W/m^2.
    refractive_indices : ndarray (n,), float64
        Per-beam refractive indices (default 1).
    """

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
        self.refractive_indices = np.ones(n_lasers, dtype=np.float64)

    def add_laser(
        self,
        index,
        waist,
        origin,
        direction,
        beam_power,
        beam_frequency,
        detuning,
        handedness,
    ):
        """Add or update a laser at a specific index in the component."""
        self.beam_waists[index] = waist
        self.origins[index, :] = origin
        self.normalized_directions[index, :] = direction / np.linalg.norm(
            direction
        )
        self.beam_powers[index] = beam_power
        self.beam_frequencies[index] = beam_frequency
        self.detunings[index] = detuning
        self.handedness[index] = handedness

        # Calculate derived properties
        self.beam_wavelengths[index] = scc.c / beam_frequency
        k = 2.0 * math.pi / self.beam_wavelengths[index]
        self.wave_vectors[index, :] = k * self.normalized_directions[index, :]
        self.initial_intensities[index] = (
            2.0 * beam_power / (math.pi * waist**2)
        )


if __name__ == "__main__":
    pass
