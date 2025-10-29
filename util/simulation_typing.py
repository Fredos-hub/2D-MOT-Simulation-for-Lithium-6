import numpy as np
import scipy.constants as scc


# General class for ECS atoms
# This is a dummy implementation for type hints and signatures.

class ECSAtoms:
    """
    A class representing atoms in an ECS-based simulation.

    Note: In the actual simulation, this class is JIT-compiled using Numba for performance.
    This dummy implementation is intended for providing type hints, signatures, and in-editor tooltips.

    Attributes
    ----------
    n : int
        Number of atoms in the simulation.
    mass : float
        Mass of each atom.
    natural_linewidth : float
        Natural linewidth of the atomic transition.
    transition_frequency : float
        Transition frequency of the atom.
    saturation_intensity : float
        Saturation intensity of the atomic transition.
    velocities : np.ndarray
        Velocities of the atoms (n x 3).
    positions : np.ndarray
        Positions of the atoms (n x 3).
    magnetic_field_vectors : np.ndarray
        Magnetic field vectors acting on each atom (n x 3).
    magnetic_field_strength : np.ndarray
        Magnetic field strengths for each atom (n).
    max_step_lengths : np.ndarray
        Maximum step lengths for each atom.
    subjective_time : np.ndarray
        Subjective clock for each atom.
    status : np.ndarray
        Status flag for each atom (e.g., 1 for alive, 0 for dead).
    location_tags : np.ndarray
        Tags indicating the atoms' positions within the experimental setup.
    groundstates : np.ndarray
        Ground state properties for each atom.
    atom_ids : np.ndarray
        Unique identifier for each atom.
    """

    def __init__(self,
                 n: int = 1000
               ) -> None:
        """
        Initializes the AtomECS instance with default or provided atomic properties.

        Note: In the actual simulation, this class is JIT-compiled using Numba for performance.

        Parameters
        ----------
        n : int, optional
            Total number of atoms in the simulation (default is 1000).

        """
        self.n = n
        self.mass_u = 0
        self.mass = 0
        self.natural_linewidth = 0
        self.transition_frequency = 0
        self.saturation_intensity = (np.pi * scc.h * scc.c * self.natural_linewidth) / (3.0 * (scc.c/self.transition_frequency)**3)

        self.velocities = np.zeros((n, 3), dtype=np.float64)
        self.positions = np.zeros((n, 3), dtype=np.float64)
        self.magnetic_field_vectors = np.zeros((n, 3), dtype=np.float64)
        self.magnetic_field_strength = np.zeros(n, dtype=np.float64)
        self.max_step_lengths = np.zeros(n, dtype=np.float64)
        self.subjective_time = np.zeros(n, dtype=np.float64)
        self.status = np.ones(n, dtype=np.int32)
        self.location_tags = np.zeros(n, dtype=np.int32)
        self.groundstates = np.ones(n, dtype=np.int32)
        self.atom_ids = np.arange(n, dtype=np.int32)

    def set_starting_conditions(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """
        Sets the initial positions and velocities for the atoms.

        Parameters
        ----------
        positions : np.ndarray
            A (n x 3) array of initial positions for the atoms.
        velocities : np.ndarray
            A (n x 3) array of initial velocities for the atoms.

        Raises
        ------
        ValueError
            If the shapes of the provided positions or velocities do not match (n, 3).
        """
        if positions.shape != (self.n, 3):
            raise ValueError(f"Positions must have shape ({self.n}, 3). Received shape: {positions.shape}")
        if velocities.shape != (self.n, 3):
            raise ValueError(f"Velocities must have shape ({self.n}, 3). Received shape: {velocities.shape}")

        self.positions[:] = positions
        self.velocities[:] = velocities


###############################

# Dummy implementation of the magnetic field class for type hints and signatures.
# This is a placeholder and does not represent the actual implementation.


class MagneticField:
    """
    A dummy interface for magnetic field calculations in the simulation.

    Each magnetic field implementation (e.g. Zeeman, quadrupole) should provide the following methods:

      - calculate_magnetic_field(simulation_atoms, atom_ids)
      - calculate_max_step_length(simulation_atoms, atom_ids)
      - calculate_mean_free_path(mean_excitation_time, atom_velocity)
      - calculate_max_time_step(max_step_length, atom_velocity)

    Note: In the actual simulation, these classes are JIT-compiled with Numba for performance.
    This dummy implementation is intended solely to support type hints and IDE tooltips.
    """

    def calculate_magnetic_field(self, simulation_atoms, atom_ids) -> None:
        """
        Updates the magnetic field strength for the given atoms.

        Parameters
        ----------
        simulation_atoms : AtomECS
            An instance containing the atomic simulation data.
        atom_ids : np.ndarray
            An array of atom indices for which to calculate the field.
        """
        raise NotImplementedError("calculate_magnetic_field must be implemented by the specific field class.")

    def calculate_max_step_length(self, simulation_atoms, atom_ids) -> None:
        """
        Computes the maximum step length for the given atoms based on their positions.

        Parameters
        ----------
        simulation_atoms : AtomECS
            An instance containing the atomic simulation data.
        atom_ids : np.ndarray
            An array of atom indices for which to compute the step lengths.
        """
        raise NotImplementedError("calculate_max_step_length must be implemented by the specific field class.")

    def calculate_mean_free_path(self, mean_excitation_time: float, atom_velocity: np.ndarray) -> float:
        """
        Calculates the mean free path for an atom based on its excitation time and velocity.

        Parameters
        ----------
        mean_excitation_time : float
            The mean excitation time of the atom.
        atom_velocity : np.ndarray
            The velocity vector of the atom.

        Returns
        -------
        float
            The computed mean free path.
        """
        raise NotImplementedError("calculate_mean_free_path must be implemented by the specific field class.")

    def calculate_max_time_step(self, max_step_length: float, atom_velocity: np.ndarray) -> float:
        """
        Calculates the maximum time step given a maximum step length and the atom's velocity.

        Parameters
        ----------
        max_step_length : float
            The maximum step length calculated for the atom.
        atom_velocity : np.ndarray
            The velocity vector of the atom.

        Returns
        -------
        float
            The computed maximum time step.
        """
        raise NotImplementedError("calculate_max_time_step must be implemented by the specific field class.")
    




#############################
# Dummy implementation of the laser class for type hints and signatures.
# This is a placeholder and does not represent the actual implementation.

class ECSLasers:
    """
    A class representing lasers in an ECS-based simulation.

    Note: In the actual simulation, this class is JIT-compiled using Numba for performance.
    This dummy implementation is intended solely for providing type hints, signatures, and in-editor tooltips.

    Attributes
    ----------
    n_lasers : int
        Number of lasers in the simulation.
    beam_waists : np.ndarray
        Beam waist sizes for each laser.
    origins : np.ndarray
        Origins of the laser beams (shape: (n_lasers, 3)).
    normalized_directions : np.ndarray
        Normalized direction vectors for the lasers (shape: (n_lasers, 3)).
    beam_powers : np.ndarray
        Power of each laser beam.
    beam_frequencies : np.ndarray
        Frequency of each laser beam.
    detunings : np.ndarray
        Detuning values for each laser.
    polarizations : np.ndarray
        Polarization settings for each laser.
    wave_vectors : np.ndarray
        Wave vectors for each laser (shape: (n_lasers, 3)).
    beam_wavelengths : np.ndarray
        Wavelengths of the lasers.
    initial_intensities : np.ndarray
        Initial intensities computed for each laser.
    refractive_indices : np.ndarray
        Refractive indices for each laser beam.
    """

    def __init__(self, n_lasers: int):
        """
        Initializes the ECSLasers instance with the specified number of lasers.

        Parameters
        ----------
        n_lasers : int
            The number of lasers.
        """
        self.n_lasers = n_lasers
        self.beam_waists = np.zeros(n_lasers, dtype=np.float64)
        self.origins = np.zeros((n_lasers, 3), dtype=np.float64)
        self.normalized_directions = np.zeros((n_lasers, 3), dtype=np.float64)
        self.beam_powers = np.zeros(n_lasers, dtype=np.float64)
        self.beam_frequencies = np.zeros(n_lasers, dtype=np.float64)
        self.detunings = np.zeros(n_lasers, dtype=np.float64)
        self.polarizations = np.zeros(n_lasers, dtype=np.int32)
        self.wave_vectors = np.zeros((n_lasers, 3), dtype=np.float64)
        self.beam_wavelengths = np.zeros(n_lasers, dtype=np.float64)
        self.initial_intensities = np.zeros(n_lasers, dtype=np.float64)
        self.refractive_indices = np.ones(n_lasers, dtype=np.float64)

    def add_laser(self, index: int, waist: float, origin: np.ndarray, direction: np.ndarray,
                  beam_power: float, beam_frequency: float, detuning: float, polarization: int) -> None:
        """
        Add or update a laser at a specific index in the ECSLasers component.

        Parameters
        ----------
        index : int
            Index at which to add or update the laser.
        waist : float
            Beam waist of the laser.
        origin : np.ndarray
            Origin of the laser beam (array of shape (3,)).
        direction : np.ndarray
            Direction vector of the laser beam (will be normalized).
        beam_power : float
            Power of the laser beam.
        beam_frequency : float
            Frequency of the laser beam.
        detuning : float
            Detuning value for the laser.
        polarization : int
            Polarization setting of the laser.
        """
        self.beam_waists[index] = waist
        self.origins[index, :] = origin

        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Direction vector cannot be zero.")
        self.normalized_directions[index, :] = direction / norm

        self.beam_powers[index] = beam_power
        self.beam_frequencies[index] = beam_frequency
        self.detunings[index] = detuning
        self.polarizations[index] = polarization

        # Derived properties:
        # For the derived calculations, we assume the presence of physical constants.
        # In the actual implementation, these would be obtained from a constants module (e.g., scc).
        c = 299792458.0  # Speed of light in m/s
        pi = np.pi

        # Calculate beam wavelength: lambda = c / frequency
        wavelength = c / beam_frequency
        self.beam_wavelengths[index] = wavelength

        # Calculate the wave vector: k = 2*pi / lambda * (normalized direction)
        self.wave_vectors[index, :] = self.normalized_directions[index, :] * (2 * pi / wavelength)

        # Calculate initial intensity: I = 2*beam_power / (pi * waist^2)
        self.initial_intensities[index] = 2 * beam_power / (pi * waist**2)


######################
# Dummy implementation of the interaction class for type hints and signatures.
# This is a placeholder and does not represent the actual implementation.

class LightAtomInteraction:
    """
    A dummy interaction class that implements the same interface as more complex 
    interaction objects (e.g., Lithium6LevelInteraction, Lithium18LevelInteraction).
    
    This class is used solely for providing type hints and a unified interface when 
    running in Numba's nopython mode. All methods return default dummy values.
    
    Attributes
    ----------
    number_of_ground_states : int
        Number of ground states (static).
    number_of_excited_states : int
        Number of excited states (static).
    mu_B : float
        Bohr magneton constant.
    allowed_transitions : np.ndarray
        Allowed transitions matrix. Each row is of the form:
        (ground_state, excited_state, polarization) where polarization:
            0 -> σ⁻, 1 -> π, 2 -> σ⁺.
    ground_mJ : np.ndarray
        Magnetic quantum numbers for the ground states.
    excited_mJ : np.ndarray
        Magnetic quantum numbers for the excited states.
    """

    def __init__(self):
        # For example, assume a 6-level system.
        self.number_of_ground_states = 2
        self.number_of_excited_states = 4
        # Retrieve the Bohr magneton from your constants module (here a dummy value is used)
        self.mu_B = 9.274009994e-24  
        
        # Dummy m_J values that mimic a typical 6-level atom.
        self.ground_mJ = np.array([-0.5, 0.5], dtype=np.float64)
        self.excited_mJ = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
        
        # Dummy allowed transitions matrix.
        self.allowed_transitions = np.array([
            [0, 0, 0],  # Ground state 0 (-1/2) -> Excited state 0 (-3/2), σ⁻
            [1, 1, 0],  # Ground state 1 (+1/2) -> Excited state 1 (-1/2), σ⁻
            [0, 1, 1],  # Ground state 0 (-1/2) -> Excited state 1 (-1/2), π
            [1, 2, 1],  # Ground state 1 (+1/2) -> Excited state 2 (+1/2), π
            [0, 2, 2],  # Ground state 0 (-1/2) -> Excited state 2 (+1/2), σ⁺
            [1, 3, 2]   # Ground state 1 (+1/2) -> Excited state 3 (+3/2), σ⁺
        ], dtype=np.int32)

    def calculate_rate(self, 
                       polarization: int,
                       effective_magnetic_field_strength: float, 
                       ground_state: int, 
                       excited_state: int,
                       laser_intensity: float, 
                       natural_linewidth: float,
                       saturation_intensity: float,
                       effective_transition_frequency: float, 
                       doppler_shift: float, 
                       laser_beam_frequency: float, 
                       detuning: float) -> float:
        """
        Dummy method to compute the (de-)excitation rate (Hz) for the specified transition.
        
        Parameters
        ----------
        polarization : int
            Polarization index (0 for σ⁻, 1 for π, 2 for σ⁺).
        effective_magnetic_field_strength : float
            Effective magnetic field strength (Tesla). May be negative.
        ground_state : int
            Ground state index.
        excited_state : int
            Excited state index.
        laser_intensity : float
            Laser intensity.
        natural_linewidth : float
            Natural linewidth in rad/s.
        saturation_intensity : float
            Saturation intensity (W/m²).
        effective_transition_frequency : float
            Effective transition frequency (Hz).
        doppler_shift : float
            Doppler shift (Hz).
        laser_beam_frequency : float
            Laser beam frequency (Hz).
        detuning : float
            Detuning in rad/s.
            
        Returns
        -------
        float
            Dummy transition rate in Hz (always 0.0).
        """
        return 0.0

    def calculate_transition_frequency_shift(self, 
                                             polarization: int, 
                                             ground_state: int, 
                                             excited_state: int, 
                                             effective_magnetic_field_strength: float) -> float:
        """
        Dummy method to compute the Zeeman frequency shift (in Joules) for a given transition.
        
        Parameters
        ----------
        polarization : int
            Polarization index (0 for σ⁻, 1 for π, 2 for σ⁺).
        ground_state : int
            Ground state index.
        excited_state : int
            Excited state index.
        effective_magnetic_field_strength : float
            Effective magnetic field strength (Tesla).
            
        Returns
        -------
        float
            Dummy energy shift (always 0.0).
        """
        return 0.0

    def calculate_saturation_intensity(self, 
                                       effective_transition_frequency: float, 
                                       natural_linewidth: float) -> float:
        """
        Dummy method to compute the saturation intensity for a given transition.
        
        Parameters
        ----------
        effective_transition_frequency : float
            Effective transition frequency (Hz).
        natural_linewidth : float
            Natural linewidth (rad/s).
            
        Returns
        -------
        float
            Dummy saturation intensity (always 0.0).
        """
        return 0.0
    

    def calculate_branching_ratio(self, ground_state: int,
                                        excited_state: int,
                                        polarization: int,
                                        magnetic_field_strength: float):
        """
        Dummy method to compute the branching ratio for a given transition.
        Parameters
        ----------
        ground_state : int
            Ground state index.
        excited_state : int
            Excited state index.
        polarization : int
            Polarization index (0 for σ⁻, 1 for π, 2 for σ⁺).
        magnetic_field_strength : float
            Magnetic field strength (Tesla).
        Returns
        -------
        float
            Dummy branching ratio (always 0.0).
        """
        return 0.0