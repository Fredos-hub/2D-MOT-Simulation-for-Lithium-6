###################################################################################################################
#Class to handle calculation of magnetic field. Either by interpolation/grid or by calculation for ideal quadropol#
###################################################################################################################

import numpy as np
from matplotlib import pyplot as plt
from numba import njit, float64, int32
from numba.experimental import jitclass
from util.simulation_typing import ECSAtoms
import math

##########################################
# Ideal Quadrupole Magnetic Field Class  #
##########################################

quadropole_spec = [('gradient', float64),
                   ('offset', float64[:])]

@jitclass(quadropole_spec)
class IdealQuadropoleField:
    """
    Represents an ideal quadrupole magnetic field with a linear gradient.
    
    In the x–y plane the field is defined as:
    
        B = gradient * (y, x, 0)
    
    (Note: In the calculation function below the roles of x and y are interchanged.
    Here we follow the formulas in the wrapper functions.)
    """
    def __init__(self, gradient: float, offset: np.ndarray) -> None:
        self.gradient = gradient
        self.offset = offset  
    def calculate_magnetic_field(self, simulation_atoms: ECSAtoms, atom_id: np.ndarray) -> None:
        """
        Updates the magnetic field vector of the given atoms using the ideal quadrupole formula.
        """
        calculate_ideal_quadropole_field(self.gradient, self.offset, simulation_atoms, atom_id)

    def field_at_positions(self, positions: np.ndarray) -> np.ndarray[np.ndarray]:
        """
        Computes the magnetic field (and its norm) at arbitrary positions.
        Used for Plotting.
        Parameters
        ----------
        positions : np.ndarray
            Array of positions with shape (..., 3).
        
        Returns
        -------
        B : np.ndarray
            Magnetic field vector at the given positions.
        norm : np.ndarray
            Magnitude of the magnetic field.
            
        Uses the same formulas as in the wrapper functions:
            B_x = gradient * y
            B_y = gradient * x
            B_z = 0
        """
        x = positions[..., 0]
        y = positions[..., 1]
        B_x = self.gradient * y *0.6
        B_y = self.gradient * x
        B_z = np.zeros_like(x)
        B = np.stack((B_x, B_y, B_z), axis=-1)
        norm = np.sqrt(B_x**2 + B_y**2 + B_z**2)
        return B, norm


    def calculate_max_step_length(self, simulation_atoms: ECSAtoms, atom_id: np.ndarray) -> None:
        # Extract the field strengths for these atom IDs
        B = simulation_atoms.magnetic_field_strength[atom_id]


        if (B >= 0.5*0.0025) & (B < 1e-1):
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        if (B > 0) & (B < 0.5*0.0025):    
            simulation_atoms.max_step_lengths[atom_id] = 1e-6


        return

    def calculate_mean_free_path(self, mean_excitation_time, atom_velocity):

        mean_free_path = mean_excitation_time * np.sqrt(atom_velocity[0]**2 + atom_velocity[1]**2)

        return mean_free_path

    def calculate_max_time_step(self, max_step_length, atom_velocity):

        max_time_step = max_step_length/(np.sqrt(atom_velocity[0]**2 + atom_velocity[1]**2)+ 1e-12)

        return max_time_step


##########################################
# Zeeman Slower Magnetic Field Class     #
##########################################

zeeman_spec = [('slower_length', float64), 
               ('B_0', float64),
               ('B_bias', float64),
               ('delta_B', float64)]

@jitclass(zeeman_spec)
class ZeemanField:
    """
    Models the magnetic field profile of an ideal Zeeman slower.
    
    The field strength is given by:
    
        B(y) = B_0 * sqrt(1 - (y / slower_length)) + B_bias
    
    where the field is assumed to point along the y-direction.
    """
    def __init__(self, slower_length: float, B_0: float, B_bias: float, delta_B: float) -> None:

        self.slower_length = slower_length
        self.B_0 = B_0
        self.B_bias = B_bias
        self.delta_B = delta_B

    # Method to calculate the magnetic field strength for selected atoms.

    def calculate_magnetic_field(self, simulation_atoms: ECSAtoms, atom_ids: np.ndarray) -> None:
        """
        Updates the magnetic field strength for the given atoms.
        """
        calculate_zeeman_field(self.B_0, self.slower_length, simulation_atoms, atom_ids, self.B_bias)



    def calculate_max_step_length(self, simulation_atoms: ECSAtoms, atom_ids: np.ndarray) -> None:
        """
        Computes the maximum step length for the given atoms based on their positions.
        """

        calculate_zeeman_max_step_length(simulation_atoms = simulation_atoms,
                                         atom_ids = atom_ids,
                                         slower_length = self.slower_length,
                                         delta_B = self.delta_B)
    
    def calculate_mean_free_path(self, mean_excitation_time, atom_velocity):

        return calculate_zeeman_mean_free_path(mean_excitation_time = mean_excitation_time, atom_velocity=atom_velocity)
    
    def calculate_max_time_step(self, max_step_length, atom_velocity):



        max_time_step = max_step_length/(atom_velocity[1] + 1e-12)



        return max_time_step

    def field_at_positions(self, positions: np.ndarray) -> np.ndarray[np.ndarray]:
        """
        Computes the magnetic field vector (assumed along y) and its magnitude at arbitrary positions.
        Used for Plotting.
        Parameters
        ----------
        positions : np.ndarray
            Array of positions with shape (..., 3).
        
        Returns
        -------
        B : np.ndarray
            Magnetic field vector (with nonzero component only along y).
        norm : np.ndarray
            Magnitude of the magnetic field.
            
        Uses the formula:
            B(y) = B_0 * sqrt(1 - (y / slower_length))
        """

        y = positions[..., 1]
        field_strength = self.B_0 * np.sqrt(1.0 - (y / self.slower_length)) + self.B_bias
        B = np.zeros_like(positions)
        B[..., 1] = field_strength  # Field along y direction
        norm = np.abs(field_strength)
        return B, norm

#####################################
#       Dipole-Bar Magnetic Field       #
#####################################
dipole_spec = [
    ("n_dipoles",             int32),
    ("positions",             float64[:, :]),   # (n_dipoles,3)
    ("dimensions",            float64[:, :]),   # (n_dipoles,3)
    ("volumes",               float64[:]),      # (n_dipoles,)
    ("dipole_moment_vectors", float64[:, :]),   # (n_dipoles,3)
    ("mu0_over_4pi",          float64)
]

@jitclass(dipole_spec)
class DipoleBarMagneticField:

    def __init__(self, n_dipoles):
        self.n_dipoles             = n_dipoles
        self.positions             = np.zeros((n_dipoles,3), dtype=np.float64)
        self.dimensions            = np.zeros((n_dipoles,3), dtype=np.float64)
        self.volumes               = np.zeros(n_dipoles,   dtype=np.float64)
        self.dipole_moment_vectors = np.zeros((n_dipoles,3), dtype=np.float64)

        # μ₀/(4π) in SI
        self.mu0_over_4pi = 1e-7

    def add_dipole(self, idx, position, dimension, orientation, magnetization):
        """
        Call this once per dipole to set its geometry & magnetization.
        """
        self.positions[idx]  = position
        self.dimensions[idx] = dimension

        # volume
        vol = dimension[0] * dimension[1] * dimension[2]
        self.volumes[idx] = vol

        # pre‑scaled dipole moment vector = M∙V ∙ orientation_unit
        norm_orient = np.linalg.norm(orientation)
        orient_unit = orientation / norm_orient
        self.dipole_moment_vectors[idx] = orient_unit * (magnetization * vol)


    def calculate_magnetic_field(self, simulation_atoms, atom_id):
        """
        Updates each atom’s B⃗ and |B| by summing bar-dipole contributions.
        """
        calculate_bar_dipole_field(
            self.n_dipoles,
            self.positions,
            self.dipole_moment_vectors,
            self.mu0_over_4pi,
            simulation_atoms,
            atom_id
        )

    def calculate_max_step_length(self, simulation_atoms, atom_id: np.ndarray) -> None:
        B = simulation_atoms.magnetic_field_strength[atom_id]

        if (B >= 0.5*0.01) & (B < 0.5*0.02):
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        elif (B>=0.0025*0.5) & (B< 0.5*0.01):
            simulation_atoms.max_step_lengths[atom_id] = 1e-5
        elif (B > 0) & (B < 0.5*0.0025):    
            simulation_atoms.max_step_lengths[atom_id] = 1e-6
        return
    
    def calculate_mean_free_path(self, mean_excitation_time, atom_velocity):

        mean_free_path = mean_excitation_time * np.sqrt(atom_velocity[0]**2 + atom_velocity[1]**2 + atom_velocity[2]**2)

        return mean_free_path

    def calculate_max_time_step(self, max_step_length, atom_velocity):

        max_time_step = max_step_length/(np.sqrt(atom_velocity[0]**2 + atom_velocity[1]**2+atom_velocity[2]**2)+  1e-12)

        return max_time_step 




# -------------------------------
# NUMBA SPEC
# -------------------------------
elliptical_spec = [
    ('g_x',     float64),     # gradient along principal x' (units: T/m)
    ('g_y',     float64),
    ('g_z',     float64),     # gradient along principal y' (units: T/m)
    ('offset',  float64[:]),  # length-3 array [x0, y0, z0]
    ('theta',   float64),     # tilt angle in radians (CCW)
]
# -------------------------------
# JITCLASS: EllipticalMagneticField
# -------------------------------
@jitclass(elliptical_spec)
class EllipticalMagneticField:
    """
    Elliptical, tilted, offset 'quadrupole-like' linear magnetic field.

    Principal-frame definition (x', y'):
        B'_x = g_x * y'
        B'_y = g_y * x'
        B'_z = 0

    Coordinates are relative to 'offset' and rotated by angle 'theta':
        [x'; y'] = R(theta)^T * ([x; y] - [x0; y0])

    Finally, the field is rotated back to lab frame:
        [B_x; B_y] = R(theta) * [B'_x; B'_y]
        B_z = 0  (can be extended with bias if desired)

    Parameters
    ----------
    g_x, g_y : float
        Linear gradients (T/m) controlling ellipticity (g_x != g_y).
    delta_B : float
        Kept for API compatibility; currently not used (set a bias in kernel if needed).
    offset : np.ndarray
        (3,) array defining where the field is zero: e.g., [-1.3e-3, 1.4e-3, 0.0]
    theta : float
        Tilt angle in radians (CCW). Negative leans "left".
    """
    def __init__(self, g_x: float, g_y: float, g_z: float, offset: np.ndarray, theta: float) -> None:
        self.g_x = g_x
        self.g_y = g_y
        self.g_z = g_z
        self.offset = offset
        self.theta = theta/180 * np.pi

    def calculate_magnetic_field(self, simulation_atoms, atom_id: np.ndarray) -> None:
        """
        Updates simulation_atoms.magnetic_field_vectors (and strength if present),
        following the elliptical, tilted model.
        """
        calculate_elliptical_field(self.g_x, self.g_y, self.g_z, self.offset, self.theta,
                                   simulation_atoms, atom_id)

    def field_at_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Computes the magnetic field (and its norm) at arbitrary positions.
        Uses the same formulas as in the kernel.

        Parameters
        ----------
        positions : array_like (..., 3)

        Returns
        -------
        B : (..., 3) array
        norm : (...) array
        """
        pos = np.asarray(positions)
        R  = _rot2(self.theta)
        RT = R.T

        # broadcast offset
        dx = pos[..., 0] - self.offset[0]
        dy = pos[..., 1] - self.offset[1]
        dz = pos[..., 2] - self.offset[3]
        # to principal frame
        xprime = RT[0,0]*dx + RT[0,1]*dy
        yprime = RT[1,0]*dx + RT[1,1]*dy

        # field in principal frame
        Bx_p = self.g_x * yprime
        By_p = self.g_y * xprime
        Bz_p = self.g_z * dz

        # back to lab frame
        Bx = R[0,0]*Bx_p + R[0,1]*By_p
        By = R[1,0]*Bx_p + R[1,1]*By_p
        Bz = Bz_p

        B = np.stack((Bx, By, Bz), axis=-1)
        norm = np.sqrt(Bx*Bx + By*By + Bz*Bz)
        return B, norm

    # Keep the remaining API identical to IdealQuadropoleField
    def calculate_max_step_length(self, simulation_atoms, atom_id: np.ndarray) -> None:
        B = simulation_atoms.magnetic_field_strength[atom_id]

        if (B >= 0.5*0.01):
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        elif (B>=0.0025*0.5) & (B< 0.5*0.01):
            simulation_atoms.max_step_lengths[atom_id] = 1e-5
        elif (B > 0) & (B < 0.5*0.0025):    
            simulation_atoms.max_step_lengths[atom_id] = 1e-6
        return

    def calculate_mean_free_path(self, mean_excitation_time, atom_velocity):
        return mean_excitation_time * np.sqrt(atom_velocity[0]**2 + atom_velocity[1]**2)

    def calculate_max_time_step(self, max_step_length, atom_velocity):
        return max_step_length / (np.sqrt(atom_velocity[0]**2 + atom_velocity[1]**2) + 1e-12)

@njit
def calculate_zeeman_field(B_0: float, slower_length: float, simulation_atoms: ECSAtoms, atom_ids: np.ndarray, B_bias: float) -> None:
    """
    Computes the Zeeman magnetic field strength for selected atoms.
    """
    y = simulation_atoms.positions[atom_ids, 1]
    field_strength = np.where(y < slower_length,
                               B_0 * np.sqrt(1.0 - y / slower_length) + B_bias,
                               0.0)
    simulation_atoms.magnetic_field_strength[atom_ids] = field_strength
    simulation_atoms.magnetic_field_vectors[atom_ids, 1] = field_strength



@njit
def calculate_zeeman_max_step_length(simulation_atoms: ECSAtoms, atom_ids: np.ndarray, slower_length: float, delta_B: float) -> None:
    """
    Calculates the maximum step size in the y-direction such that the magnetic field
    changes by 0.1% at a given position.

    The magnetic field profile is given by:
        B(y) = B_0 * sqrt(1 - y / slower_length)
    and its absolute derivative is:
        |dB/dy| = B_0 / (2 * slower_length) * (1 - y / slower_length)^(-1/2)

    For a relative change of 0.1% in the magnetic field:
        0.001 * B(y) = |dB/dy| * Δy
    Solving for Δy, we obtain:
        Δy = 2 * 0.001 * slower_length * (1 - y/slower_length)
    
    This Δy is then used as the maximum allowed step size in the y-direction.

    Parameters
    ----------
    simulation_atoms : ECSAtoms
        The simulation atoms object.
    atom_ids : np.ndarray
        Array of atom indices for which to calculate the maximum step length.
    slower_length : float
        The total length of the slower (or characteristic length scale).
    """
    # Hardcoded fractional change (0.1%)
    delta_fraction = delta_B

    # Get the y-positions of the specified atoms
    y = simulation_atoms.positions[atom_ids, 1]
    
    # Calculate the maximum step size in y such that the magnetic field changes by 0.1%
    delta_y_max = 2 * delta_fraction * slower_length * (1.0 - y / slower_length)
    
    # Set the calculated maximum step sizes for these atoms
    simulation_atoms.max_step_lengths[atom_ids] = delta_y_max
    return

@njit
def calculate_zeeman_mean_free_path(mean_excitation_time: float, atom_velocity: np.ndarray) -> float:

    return mean_excitation_time * atom_velocity[1]

@njit
def calculate_ideal_quadropole_field(gradient: float, offset , simulation_atoms: ECSAtoms, atom_id: np.ndarray) -> None:
    """
    Computes the ideal quadrupole magnetic field for selected atoms.
    """
    x = simulation_atoms.positions[atom_id, 0] - offset[0]
    y = simulation_atoms.positions[atom_id, 1] - offset[1]
    z = simulation_atoms.positions[atom_id, 2] - offset[2]
    B_x = gradient * y
    B_y = gradient * x
    B_z = 0
    simulation_atoms.magnetic_field_vectors[atom_id, 0] = B_x
    simulation_atoms.magnetic_field_vectors[atom_id, 1] = B_y
    simulation_atoms.magnetic_field_vectors[atom_id, 2] = B_z
    
    simulation_atoms.magnetic_field_strength[atom_id] = math.sqrt(B_x**2 + B_y**2 + B_z**2)



@njit
def calculate_bar_dipole_field(
        n_dipoles,
        positions,
        dipole_moment_vectors,
        mu0_over_4pi,
        atoms: ECSAtoms,
        atom_id
    ):

    rx, ry, rz = atoms.positions[atom_id]

    Bx = 0.0
    By = 0.0
    Bz = 0.0

    for d in range(n_dipoles):
        px, py, pz = positions[d]
        dx = rx - px
        dy = ry - py
        dz = rz - pz

        r2 = dx*dx + dy*dy + dz*dz
        if r2 < 1e-24:
            continue

        inv_r = 1.0 / math.sqrt(r2)
        inv_r3 = inv_r * inv_r * inv_r

        ux = dx * inv_r
        uy = dy * inv_r
        uz = dz * inv_r

        mx, my, mz = dipole_moment_vectors[d]

        m_dot_u = mx*ux + my*uy + mz*uz
        factor  = mu0_over_4pi * inv_r3

        Bx += factor * (3.0*ux*m_dot_u - mx)
        By += factor * (3.0*uy*m_dot_u - my)
        Bz += factor * (3.0*uz*m_dot_u - mz)

    atoms.magnetic_field_vectors[atom_id, 0]  = Bx
    atoms.magnetic_field_vectors[atom_id, 1]  = By
    atoms.magnetic_field_vectors[atom_id, 2]  = Bz
    atoms.magnetic_field_strength[atom_id]    = math.sqrt(Bx*Bx + By*By + Bz*Bz)



# -------------------------------
# HELPER: rotation matrix for theta
# -------------------------------
@njit(cache=True)
def _rot2(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.empty((2,2), dtype=np.float64)
    R[0,0] = c;  R[0,1] = -s
    R[1,0] = s;  R[1,1] =  c
    return R

# -------------------------------
# CORE KERNEL: compute B at given atom ids, in-place on simulation_atoms
# Expects simulation_atoms to expose:
#   positions: (N,3)
#   magnetic_field_vectors: (N,3)
#   magnetic_field_strength: (N,)   (if present; optional but common)
# -------------------------------

@njit
def calculate_elliptical_field(g_x: float,
                               g_y: float,
                               g_z: float,
                               offset: np.ndarray,
                               theta: float,
                               simulation_atoms,
                               atom_id) -> None:
    """
    Expects atom_ids as a 1D array of integer indices. Theta in radians.
    Principal-frame (diagonal) form: Bx' = g_x * x', By' = g_y * y', Bz' = g_z * z'.
    """
    c = math.cos(theta)
    s = math.sin(theta)

    # For each requested atom, compute field and assign
 

    i = atom_id
    x = simulation_atoms.positions[i, 0] - offset[0]
    y = simulation_atoms.positions[i, 1] - offset[1]
    z = simulation_atoms.positions[i, 2] - offset[2]

    # principal-frame coordinates (R^T * (r - offset))
    xprime =  c * x + s * y
    yprime = -s * x + c * y

    # principal-frame (diagonal) field
    Bx_p = g_x * xprime
    By_p = g_y * yprime
    Bz_p = g_z * z

    # rotate back: B = R * B'
    B_x = c * Bx_p - s * By_p
    B_y = s * Bx_p + c * By_p
    B_z = Bz_p

    simulation_atoms.magnetic_field_vectors[i, 0] = B_x
    simulation_atoms.magnetic_field_vectors[i, 1] = B_y
    simulation_atoms.magnetic_field_vectors[i, 2] = B_z

    # store strength
    simulation_atoms.magnetic_field_strength[i] = math.sqrt(B_x*B_x + B_y*B_y + B_z*B_z)




##########################################
# Plotting Functions                     #
##########################################

def plot_magnetic_field_vectors(field: IdealQuadropoleField) -> None:
    """
    Plots the quadrupole magnetic field with vector directions and color-coded magnitudes.
    """
    data_points = np.linspace(-5, 5, 20)
    grid_x, grid_y = np.meshgrid(data_points, data_points)
    positions = np.stack((grid_x, grid_y, np.zeros_like(grid_x)), axis=-1)
    B, norm = field.field_at_positions(positions)
    B_x = B[..., 0]
    B_y = B[..., 1]
    
    plt.figure()
    quiver = plt.quiver(
        grid_x, grid_y, B_x, B_y, norm, 
        cmap="plasma", scale=50, pivot='middle'
    )
    plt.colorbar(quiver, label="Magnetic Field Magnitude")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Quadrupole Magnetic Field Vectors")
    plt.axis("equal")
    plt.show()

def plot_magnetic_field_streamplot(field: IdealQuadropoleField) -> None:
    """
    Plots the quadrupole magnetic field lines as a streamplot.
    """
    data_points = np.linspace(-0.05, 0.05, 50)
    grid_x, grid_y = np.meshgrid(data_points, data_points)
    positions = np.stack((grid_x, grid_y, np.zeros_like(grid_x)), axis=-1)
    B, _ = field.field_at_positions(positions)
    B_x = B[..., 0]
    B_y = B[..., 1]

    plt.figure()
    plt.streamplot(grid_x, grid_y, B_x, B_y, color="black", density=1.2)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Quadrupole Magnetic Field Lines")
    plt.axis("equal")
    plt.show()

def plot_zeeman_field_vs_y(zeeman_field: ZeemanField) -> None:
    """
    Plots the Zeeman slower magnetic field strength as a function of y.
    """
    # Choose y values from 0 to slower_length (adjust as needed)
    y_values = np.linspace(0, zeeman_field.slower_length, 500)
    positions = np.zeros((len(y_values), 3))
    positions[:, 1] = y_values  # set y-coordinate; x and z remain zero
    _, norm = zeeman_field.field_at_positions(positions)
    
    plt.figure()
    plt.plot(y_values, norm, label="B(y)")
    plt.xlabel("y (m)")
    plt.ylabel("Magnetic Field Strength (T)")
    plt.title("Zeeman Slower Magnetic Field vs. y")
    plt.legend()
    plt.grid(True)
    plt.show()


##########################################
# Main Block                             #
##########################################

if __name__ == '__main__':
    ## Example for the quadrupole field
    #B_Field = IdealQuadropoleField(0.5)
    #plot_magnetic_field_vectors(B_Field)
    #plot_magnetic_field_streamplot(B_Field)
    #
    ## Example for the Zeeman field
    ## (using arbitrary parameters for demonstration)
    zeeman = ZeemanField(slower_length=1, B_0=0.079)
    plot_zeeman_field_vs_y(zeeman)
