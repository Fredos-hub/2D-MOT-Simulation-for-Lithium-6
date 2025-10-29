import numpy as np
from numba import float64, int32
from numba.experimental import jitclass
import scipy.constants as scc



ATOMIC_MASS = scc.physical_constants['atomic mass constant'][0]  # kg
# Define the structure (specification) for the JIT-compiled ECSAtoms class
atom_spec = [
    # Constants: Properties that are uniform for all atoms
    ('n', int32),                   # Number of atoms in the simulation
    ('mass', float64),              # Mass of each atom
    ('mass_u', float64),            # Mass of each atom in atomic mass units (u)
    ('natural_linewidth', float64), # Natural linewidth of the atomic transition
    ('transition_frequency', float64),  # Transition frequency of the atom
    ('saturation_intensity', float64),  # Saturation intensity of the atomic transition

    # Per-atom attributes stored as arrays (each of length n, or shape (n, 3) etc.)
    ('velocities', float64[:, :]),       # Velocities of the atoms (n x 3)
    ('positions', float64[:, :]),          # Positions of the atoms (n x 3)
    ('magnetic_field_vectors', float64[:, :]),  # Magnetic field vectors (n x 3)
    ('magnetic_field_strength', float64[:]),      # Magnetic field strengths for each atom (n)
    ('max_step_lengths', float64[:]),          # Maximum step length for the Atom. Dependent on Magnetic Field at position.
    ('subjective_time', float64[:]),       # Subjective clock of each atom (n)
    ('time_overshoot', float64[:]),
    ('status', int32[:]),                  # Status (alive=1, dead=0) for each atom (n)
    ('location_tags', int32[:]),           # Tags indicating position in the experimental setup (n)
    ('groundstates', int32[:]),       # Ground state properties for each atom (n x 2)
    ('atom_ids', int32[:])                 # Unique IDs for atoms (n)
]


@jitclass(atom_spec)
class Li6:
    """
    A JIT-compiled class representing atoms in an ECS-based simulation.

    The attributes are categorized into two groups:
      1. Constants: Uniform properties for all atoms (e.g., mass, transition frequency).
      2. Per-atom arrays: Properties stored as arrays, with one value (or row) per atom 
         (e.g., positions, velocities, status).

    Attributes
    ----------
    n : int
        Number of atoms in the simulation.
    mass : float
        Mass of each atom.
    natural_linewidth : float
        Natural linewidth of the atomic transition.
    transition_frequency : float
        Frequency of the atomic transition.
    saturation_intensity : float
        Saturation intensity for a simplified 2-level system.
    velocities : np.ndarray (n x 3)
        Velocities of the atoms.
    positions : np.ndarray (n x 3)
        Positions of the atoms.
    magnetic_field_vectors : np.ndarray (n x 3)
        Magnetic field vectors acting on each atom.
    magnetic_field_strength : np.ndarray (n)
        Magnitude of the magnetic field for each atom.
    max_step_lengths : np.ndarray (n)
        Maximum step lengths for each atom, dependent on the magnetic field at their position.
    subjective_time : np.ndarray (n)
        Subjective time (or internal clock) for each atom.
    status : np.ndarray (n)
        Status flag for each atom (1 for alive, 0 for dead).
    location_tags : np.ndarray (n)
        Tags indicating the atoms' positions within the experimental setup.
    groundstates : np.ndarray (n x 2)
        Ground state properties for each atom.
    atom_ids : np.ndarray (n)
        Unique identifier for each atom.
    """

    def __init__(self, 
                 n: int = 1000) -> None:
        """
        Initializes the ECSAtoms instance with default or provided atomic properties.

        Parameters
        ----------
        n : int, optional
            Total number of atoms in the simulation (default is 1000).

        """
        # --- Constants (Uniform for all atoms) ---
        self.n = n
        self.mass_u = 6.015
        self.mass = self.mass_u * ATOMIC_MASS
        self.natural_linewidth =  2 * np.pi * 5.87e6
        self.transition_frequency =   446799648.889e6 # in Hz. D2 line COG from Li et al. (2020).
        self.saturation_intensity = (np.pi * scc.h * scc.c * self.natural_linewidth) / (3.0 * (scc.c/self.transition_frequency)**3)
        # --- Per-atom properties initialized as arrays ---
        # Initialize magnetic field properties
        self.magnetic_field_strength = np.zeros(self.n, dtype=np.float64)
        self.magnetic_field_vectors = np.zeros((self.n, 3), dtype=np.float64)

        # Initialize maximum step length
        self.max_step_lengths = np.zeros(self.n, dtype=np.float64) 

        # Initialize positions and velocities (each with shape (n, 3))
        self.positions = np.zeros((self.n, 3), dtype=np.float64)
        self.velocities = np.zeros((self.n, 3), dtype=np.float64)
        
        # Each atom has its own subjective clock (initialized to zero)
        self.subjective_time = np.zeros(self.n, dtype=np.float64)
        self.time_overshoot = np.zeros(self.n, dtype=np.float64)
        # Initialize status: -1 indicates "inactive" for every atom initially.
        self.status = np.full_like(self.max_step_lengths, -1,  dtype=np.int32)
        
        # Initialize location tags (could be used to label regions in the setup)
        self.location_tags = np.zeros(self.n, dtype=np.int32)
        
        # Initialize ground state properties (for example, two-level ground state system)
        self.groundstates = np.full_like(self.location_tags,0, dtype=np.int32)
        
        # Assign unique IDs to each atom for tracking purposes.
        self.atom_ids = np.arange(0, self.n, dtype=np.int32)

    def set_starting_conditions(self, positions: np.ndarray, velocities: np.ndarray, groundstates: np.ndarray, starting_times: np.ndarray) -> None:
        """
        Sets the initial conditions for the atoms, including positions and velocities.

        The function performs validation to ensure that the provided arrays have the correct
        shapes and that the velocities are not all zero.

        Parameters
        ----------
        positions : np.ndarray
            A (n x 3) array of initial positions for the atoms.
        velocities : np.ndarray
            A (n x 3) array of initial velocities for the atoms. Must not be (0,0,0) for all atoms.

        Raises
        ------
        ValueError
            If the shapes of the provided positions or velocities do not match (n, 3),
            or if all velocity vectors are (0, 0, 0).
        """
        # Validate positions shape
        if positions.shape != (self.n, 3):
            raise ValueError(f"Positions must have shape ({self.n}, 3). Received shape: {positions.shape}")

        # Validate velocities: ensure no atom has a (0,0,0) velocity across all atoms
        #if np.all(velocities == 0):
        #    raise ValueError("Velocity must not be (0,0,0) for all atoms.")
        if velocities.shape != (self.n, 3):
            raise ValueError(f"Velocities must have shape ({self.n}, 3). Received shape: {velocities.shape}")
        if groundstates.shape != (self.n,):
            raise ValueError(f"Groundstates must have shape ({self.n}). Received shape: {groundstates.shape}")
        # If validations pass, update the positions and velocities of the atoms.
        self.positions[:] = positions
        self.velocities[:] = velocities
        self.groundstates = groundstates
        self.time_overshoot = starting_times




@jitclass(atom_spec)
class Sr88:
    """
    A JIT-compiled class representing atoms in an ECS-based simulation.

    The attributes are categorized into two groups:
      1. Constants: Uniform properties for all atoms (e.g., mass, transition frequency).
      2. Per-atom arrays: Properties stored as arrays, with one value (or row) per atom 
         (e.g., positions, velocities, status).

    Attributes
    ----------
    n : int
        Number of atoms in the simulation.
    mass : float
        Mass of each atom.
    natural_linewidth : float
        Natural linewidth of the atomic transition.
    transition_frequency : float
        Frequency of the atomic transition.
    saturation_intensity : float
        Saturation intensity for a simplified 2-level system.
    velocities : np.ndarray (n x 3)
        Velocities of the atoms.
    positions : np.ndarray (n x 3)
        Positions of the atoms.
    magnetic_field_vectors : np.ndarray (n x 3)
        Magnetic field vectors acting on each atom.
    magnetic_field_strength : np.ndarray (n)
        Magnitude of the magnetic field for each atom.
    max_step_lengths : np.ndarray (n)
        Maximum step lengths for each atom, dependent on the magnetic field at their position.
    subjective_time : np.ndarray (n)
        Subjective time (or internal clock) for each atom.
    status : np.ndarray (n)
        Status flag for each atom (1 for alive, 0 for dead).
    location_tags : np.ndarray (n)
        Tags indicating the atoms' positions within the experimental setup.
    groundstates : np.ndarray (n x 2)
        Ground state properties for each atom.
    atom_ids : np.ndarray (n)
        Unique identifier for each atom.
    """

    def __init__(self, 
                 n: int = 1000) -> None:
        """
        Initializes the ECSAtoms instance with default or provided atomic properties.

        Parameters
        ----------
        n : int, optional
            Total number of atoms in the simulation (default is 1000).

        """
        # --- Constants (Uniform for all atoms) ---
        self.n = n
        self.mass_u = 87.9056
        self.mass = self.mass_u * ATOMIC_MASS
        self.natural_linewidth  = 2 * np.pi * 32e6
        self.transition_frequency  = scc.c/461e-9
        self.saturation_intensity = (np.pi * scc.h * scc.c * self.natural_linewidth) / (3.0 * (scc.c/self.transition_frequency)**3)
        # --- Per-atom properties initialized as arrays ---
        # Initialize magnetic field properties
        self.magnetic_field_strength = np.zeros(self.n, dtype=np.float64)
        self.magnetic_field_vectors = np.zeros((self.n, 3), dtype=np.float64)

        # Initialize maximum step length
        self.max_step_lengths = np.zeros(self.n, dtype=np.float64) 

        # Initialize positions and velocities (each with shape (n, 3))
        self.positions = np.zeros((self.n, 3), dtype=np.float64)
        self.velocities = np.zeros((self.n, 3), dtype=np.float64)
        
        # Each atom has its own subjective clock (initialized to zero)
        self.subjective_time = np.zeros(self.n, dtype=np.float64)
        
        # Initialize status: 1 indicates "alive" for every atom initially.
        self.status = np.ones(self.n, dtype=np.int32)
        
        # Initialize location tags (could be used to label regions in the setup)
        self.location_tags = np.zeros(self.n, dtype=np.int32)
        
        # Initialize ground state properties (for example, two-level ground state system)
        self.groundstates = np.ones((self.n), dtype=np.int32)
        
        # Assign unique IDs to each atom for tracking purposes.
        self.atom_ids = np.arange(0, self.n, dtype=np.int32)

    def set_starting_conditions(self, positions: np.ndarray, velocities: np.ndarray, groundstates: np.ndarray) -> None:
        """
        Sets the initial conditions for the atoms, including positions and velocities.

        The function performs validation to ensure that the provided arrays have the correct
        shapes and that the velocities are not all zero.

        Parameters
        ----------
        positions : np.ndarray
            A (n x 3) array of initial positions for the atoms.
        velocities : np.ndarray
            A (n x 3) array of initial velocities for the atoms. Must not be (0,0,0) for all atoms.

        Raises
        ------
        ValueError
            If the shapes of the provided positions or velocities do not match (n, 3),
            or if all velocity vectors are (0, 0, 0).
        """
        # Validate positions shape
        if positions.shape != (self.n, 3):
            raise ValueError(f"Positions must have shape ({self.n}, 3). Received shape: {positions.shape}")

        # Validate velocities: ensure no atom has a (0,0,0) velocity across all atoms
        if np.all(velocities == 0):
            raise ValueError("Velocity must not be (0,0,0) for all atoms.")
        if velocities.shape != (self.n, 3):
            raise ValueError(f"Velocities must have shape ({self.n}, 3). Received shape: {velocities.shape}")
        if groundstates.shape != (self.n, 3):
            raise ValueError(f"Groundstates must have shape ({self.n}, 3). Received shape: {groundstates.shape}")
        # If validations pass, update the positions and velocities of the atoms.
        self.positions[:] = positions
        self.velocities[:] = velocities
        self.groundstates = groundstates


if __name__ == '__main__':
    pass
