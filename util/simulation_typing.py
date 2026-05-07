import numpy as np
import scipy.constants as scc


# ---------------------------------------------------------------------------
# Dummy classes for type hints and IDE tooltips only.
# The real implementations are jitclasses in src/ — these dummies are never
# instantiated in a simulation run.
# ---------------------------------------------------------------------------


class ECSAtoms:
    """
    Dummy interface matching the Li6 jitclass in src/atoms.py.

    Attributes
    ----------
    n : int
    mass_u : float                   atomic mass in u
    mass : float                     atomic mass in kg
    natural_linewidth : float        rad/s
    transition_frequency : float     Hz (D2 line COG)
    saturation_intensity : float     W/m²
    velocities : ndarray (n, 3)      float64
    positions : ndarray (n, 3)       float64
    magnetic_field_vectors : ndarray (n, 3)  float64
    magnetic_field_strength : ndarray (n,)   float64
    max_step_lengths : ndarray (n,)  float64
    subjective_time : ndarray (n,)   float64
    time_overshoot : ndarray (n,)    float64  – pending event time carried across steps
    status : ndarray (n,)            int32    – -1 inactive, 1 alive, 0 dead
    location_tags : ndarray (n,)     int32
    groundstates : ndarray (n,)      int32
    atom_ids : ndarray (n,)          int32
    """

    def __init__(self, n: int = 1000) -> None:
        self.n = n
        self.mass_u = 6.015
        self.mass = self.mass_u * scc.physical_constants["atomic mass constant"][0]
        self.natural_linewidth = 2 * np.pi * 5.87e6
        self.transition_frequency = 446799648.889e6
        self.saturation_intensity = (
            np.pi * scc.h * scc.c * self.natural_linewidth
        ) / (3.0 * (scc.c / self.transition_frequency) ** 3)

        self.velocities = np.zeros((n, 3), dtype=np.float64)
        self.positions = np.zeros((n, 3), dtype=np.float64)
        self.magnetic_field_vectors = np.zeros((n, 3), dtype=np.float64)
        self.magnetic_field_strength = np.zeros(n, dtype=np.float64)
        self.max_step_lengths = np.zeros(n, dtype=np.float64)
        self.subjective_time = np.zeros(n, dtype=np.float64)
        self.time_overshoot = np.zeros(n, dtype=np.float64)
        self.status = np.full(n, -1, dtype=np.int32)
        self.location_tags = np.zeros(n, dtype=np.int32)
        self.groundstates = np.zeros(n, dtype=np.int32)
        self.atom_ids = np.arange(n, dtype=np.int32)

    def set_starting_conditions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        groundstates: np.ndarray,
        starting_times: np.ndarray,
    ) -> None:
        """
        Sets positions, velocities, groundstates and time_overshoot.
        Does NOT reset status — atoms start as inactive (-1) until activated by Simulation.step().
        """
        self.positions[:] = positions
        self.velocities[:] = velocities
        self.groundstates[:] = groundstates
        self.time_overshoot[:] = starting_times


# ---------------------------------------------------------------------------


class MagneticField:
    """
    Dummy interface matching the magnetic-field jitclasses in src/magnetic_field.py.

    All methods receive a single integer atom_id (not an array) when called from
    inside the @njit per-atom while-loop.
    """

    def calculate_magnetic_field(self, simulation_atoms: ECSAtoms, atom_id: int) -> None:
        raise NotImplementedError

    def calculate_max_step_length(self, simulation_atoms: ECSAtoms, atom_id: int) -> None:
        raise NotImplementedError

    def calculate_mean_free_path(self, mean_excitation_time: float, atom_velocity: np.ndarray) -> float:
        raise NotImplementedError

    def calculate_max_time_step(self, max_step_length: float, atom_velocity: np.ndarray) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------


class ECSLasers:
    """
    Dummy interface matching the LaserComponent jitclass in src/lasers.py.

    Attributes
    ----------
    n_lasers : int
    beam_waists : ndarray (n,)             float64  m
    origins : ndarray (n, 3)               float64  m
    normalized_directions : ndarray (n, 3) float64
    beam_powers : ndarray (n,)             float64  W
    beam_frequencies : ndarray (n,)        float64  Hz
    detunings : ndarray (n,)               float64  rad/s
    handedness : ndarray (n,)              int32    +1 σ⁺, -1 σ⁻, 0 linear
    wave_vectors : ndarray (n, 3)          float64  rad/m
    beam_wavelengths : ndarray (n,)        float64  m
    initial_intensities : ndarray (n,)     float64  W/m²
    refractive_indices : ndarray (n,)      float64
    """

    def __init__(self, n_lasers: int) -> None:
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
        index: int,
        waist: float,
        origin: np.ndarray,
        direction: np.ndarray,
        beam_power: float,
        beam_frequency: float,
        detuning: float,
        handedness: int,
    ) -> None:
        self.beam_waists[index] = waist
        self.origins[index] = origin
        norm = np.linalg.norm(direction)
        self.normalized_directions[index] = direction / norm
        self.beam_powers[index] = beam_power
        self.beam_frequencies[index] = beam_frequency
        self.detunings[index] = detuning
        self.handedness[index] = handedness
        self.beam_wavelengths[index] = scc.c / beam_frequency
        k = 2 * np.pi / self.beam_wavelengths[index]
        self.wave_vectors[index] = k * self.normalized_directions[index]
        self.initial_intensities[index] = 2 * beam_power / (np.pi * waist ** 2)


# ---------------------------------------------------------------------------


class LightAtomInteraction:
    """
    Dummy interface matching the interaction jitclasses in src/interactions.py.

    All four concrete classes (Lithium6LevelInteraction, Lithium18LevelInteraction,
    Lithium4LevelInteraction, SimpleEighteenLevelInteraction) expose this interface.

    Attributes
    ----------
    number_of_ground_states : int
    number_of_excited_states : int
    """

    def __init__(self) -> None:
        self.number_of_ground_states = 2
        self.number_of_excited_states = 4

    def calculate_saturation_parameter(
        self,
        polarization: int,
        magnetic_field_strength: float,
        ground_state: int,
        excited_state: int,
        laser_intensity: float,
        natural_linewidth: float,
        saturation_intensity: float,
        effective_transition_frequency: float,
        doppler_shift: float,
        laser_beam_frequency: float,
        detuning: float,
    ) -> float:
        """
        Returns the saturation parameter s for one (laser, ground, excited, pol) combination.
        Called once per (laser × excited_state × polarization) per while-loop iteration.
        """
        return 0.0

    def calculate_rate(
        self,
        saturation_parameters: np.ndarray,
        total_saturation_parameter: float,
        natural_linewidth: float,
        excitation_rates: np.ndarray,
    ) -> None:
        """
        Fills excitation_rates in-place from saturation_parameters.
        Shape of both arrays: (n_lasers, n_excited_states, 3).
        """
        pass

    def calculate_transition_frequency_shift(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ) -> float:
        """Returns the Zeeman frequency shift (rad/s) for the given transition."""
        return 0.0

    def calculate_branching_ratio(
        self,
        ground_state: int,
        excited_state: int,
        polarization: int,
        magnetic_field_strength: float,
    ) -> float:
        """
        Returns the spontaneous-emission branching weight into
        |excited_state⟩ → |ground_state⟩ for the given polarization channel.
        """
        return 0.0
